//! The query model — a filter-aggregate-group plan and its row-level evaluation (the **map** side).
//!
//! A [`Query`] is the BigQuery-shaped core: `SELECT <aggregates> FROM <dataset> WHERE <predicate>
//! GROUP BY <keys>`. It is split into two halves that meet at a [`Partial`] (see [`crate::combine`]):
//!
//! - **map** (here): a host fetches a shard, filters rows with the [`Predicate`], buckets the
//!   survivors by their `GROUP BY` key, and folds each [`Agg`] into a per-group [`Accum`]. The
//!   result is a [`Partial`] — small (one entry per group), associative, and mergeable.
//! - **reduce** ([`crate::combine`]): the coordinator merges every shard's [`Partial`] into one,
//!   then **finalises** each group's accumulators into output values.
//!
//! The map step is **pure** and deterministic: same shard + same query => same partial, no I/O. That
//! is what makes the engine testable (property tests below) and what makes a dropped shard safely
//! retryable on another host — re-running a map is free of side effects.

use crate::combine::{Accum, GroupResult, Partial};
use crate::dataset::Row;
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A scalar comparison operator used in a [`Predicate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    /// Apply this operator to the ordering of `left` against `right`. Non-comparable values (e.g. a
    /// number vs a string, or a missing field) compare as "not ordered", so only `Ne` is true and
    /// every ordered operator is false — a row with the wrong/absent field simply fails the filter
    /// rather than erroring. This keeps schema-on-read queries panic-free.
    pub fn eval(&self, left: &serde_json::Value, right: &serde_json::Value) -> bool {
        use std::cmp::Ordering;
        let ord = json_cmp(left, right);
        match (self, ord) {
            (CmpOp::Eq, Some(Ordering::Equal)) => true,
            (CmpOp::Eq, _) => false,
            (CmpOp::Ne, Some(Ordering::Equal)) => false,
            (CmpOp::Ne, _) => true, // unequal or incomparable => "not equal"
            (CmpOp::Lt, Some(Ordering::Less)) => true,
            (CmpOp::Le, Some(Ordering::Less | Ordering::Equal)) => true,
            (CmpOp::Gt, Some(Ordering::Greater)) => true,
            (CmpOp::Ge, Some(Ordering::Greater | Ordering::Equal)) => true,
            _ => false,
        }
    }
}

/// A boolean predicate over a row. `And`/`Or`/`Not` compose; `Cmp` tests one field against a
/// literal; `True` matches everything (the default `WHERE`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum Predicate {
    /// Matches every row (the absent / default `WHERE`).
    #[default]
    True,
    Cmp { field: String, op: CmpOp, value: serde_json::Value },
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
}

impl Predicate {
    /// Evaluate the predicate against a row. A `Cmp` on an absent field evaluates its operator
    /// against JSON `null` (so `field = null` can match an explicit null but ordered comparisons on
    /// a missing field are false). Pure and total — never panics.
    pub fn eval(&self, row: &Row) -> bool {
        match self {
            Predicate::True => true,
            Predicate::Cmp { field, op, value } => {
                let null = serde_json::Value::Null;
                let lhs = row.get(field).unwrap_or(&null);
                op.eval(lhs, value)
            }
            Predicate::And(a, b) => a.eval(row) && b.eval(row),
            Predicate::Or(a, b) => a.eval(row) || b.eval(row),
            Predicate::Not(a) => !a.eval(row),
        }
    }
}

/// An aggregate function over a column (or `*` for `Count`). Each maps to an [`Accum`] that folds
/// many values associatively, so a query's aggregates can be computed shard-by-shard and merged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Agg {
    /// `COUNT(*)` — number of rows in the group (ignores the field).
    Count,
    /// `SUM(field)` — sum of numeric values (non-numeric values are skipped).
    Sum(String),
    /// `MIN(field)` — minimum numeric value.
    Min(String),
    /// `MAX(field)` — maximum numeric value.
    Max(String),
    /// `AVG(field)` — mean of numeric values (carried as sum+count, finalised at the end).
    Avg(String),
}

impl Agg {
    /// The output column name this aggregate produces (`count`, `sum_x`, `avg_y`, ...).
    pub fn output_name(&self) -> String {
        match self {
            Agg::Count => "count".to_string(),
            Agg::Sum(f) => format!("sum_{f}"),
            Agg::Min(f) => format!("min_{f}"),
            Agg::Max(f) => format!("max_{f}"),
            Agg::Avg(f) => format!("avg_{f}"),
        }
    }

    /// A fresh zero-accumulator for this aggregate's identity element. `Sum` starts in the lossless
    /// integer lane ([`Accum::IntSum`]) and promotes to compensated float on the first fractional
    /// value.
    pub fn zero(&self) -> Accum {
        match self {
            Agg::Count => Accum::Count(0),
            Agg::Sum(_) => Accum::IntSum(0),
            Agg::Min(_) => Accum::Min(None),
            Agg::Max(_) => Accum::Max(None),
            Agg::Avg(_) => Accum::Avg { sum: crate::combine::KahanSum::zero(), count: 0 },
        }
    }

    /// Fold one row's contribution into `acc`. Reads the row's field (none for `Count`); non-numeric
    /// or missing values are skipped for numeric aggregates so a single bad cell never corrupts the
    /// result. `SUM` stays in the exact integer lane while every value is integral and only then falls
    /// back to compensated float. Pure.
    pub fn fold(&self, acc: &mut Accum, row: &Row) {
        match self {
            Agg::Count => {
                if let Accum::Count(n) = acc {
                    *n = n.saturating_add(1);
                }
            }
            Agg::Sum(f) => {
                let Some(cell) = row.get(f) else { return };
                let Some(num) = as_number(cell) else { return };
                fold_sum(acc, num);
            }
            Agg::Min(f) => {
                if let (Accum::Min(m), Some(v)) = (acc, row.get(f).and_then(as_f64)) {
                    *m = Some(m.map_or(v, |c| c.min(v)));
                }
            }
            Agg::Max(f) => {
                if let (Accum::Max(m), Some(v)) = (acc, row.get(f).and_then(as_f64)) {
                    *m = Some(m.map_or(v, |c| c.max(v)));
                }
            }
            Agg::Avg(f) => {
                if let (Accum::Avg { sum, count }, Some(v)) = (acc, row.get(f).and_then(as_f64)) {
                    sum.add(v);
                    *count = count.saturating_add(1);
                }
            }
        }
    }
}

/// A parsed numeric cell: an exact integer, or a float (for fractional / very-large-magnitude values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    /// An exact `i128` (the integer/money lane).
    Int(i128),
    /// A floating-point value (anything not exactly representable as `i128`).
    Float(f64),
}

/// Fold one numeric value into a SUM accumulator, keeping the lossless integer lane while possible
/// and otherwise promoting to compensated float.
fn fold_sum(acc: &mut Accum, num: Num) {
    match (&mut *acc, num) {
        (Accum::IntSum(s), Num::Int(v)) => match s.checked_add(v) {
            Some(t) => *s = t,
            None => {
                let mut k = crate::combine::KahanSum::zero();
                k.add(*s as f64);
                k.add(v as f64);
                *acc = Accum::Sum(k);
            }
        },
        (Accum::IntSum(s), Num::Float(v)) => {
            let mut k = crate::combine::KahanSum::zero();
            k.add(*s as f64);
            k.add(v);
            *acc = Accum::Sum(k);
        }
        (Accum::Sum(k), Num::Int(v)) => k.add(v as f64),
        (Accum::Sum(k), Num::Float(v)) => k.add(v),
        _ => {}
    }
}

/// Sort direction for an [`OrderKey`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderDir {
    /// Ascending (`ASC`, the default).
    Asc,
    /// Descending (`DESC`).
    Desc,
}

/// One `ORDER BY` key: the name of an output column (a group key column or an aggregate output name
/// such as `count` / `sum_amount`) and the direction to sort it.
///
/// Ordering is applied **after** the reduce/finalise step, over the [`GroupResult`] rows — it is a
/// pure post-processing shape on the (already small) result set, not a distributed concern. A key
/// that names neither a group column nor an aggregate output sorts every row equal (a no-op rather
/// than an error), so a typo never panics; callers can validate against the schema if they wish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderKey {
    /// The output column to sort by (group key column or aggregate output name).
    pub column: String,
    /// Ascending or descending.
    pub dir: OrderDir,
}

impl OrderKey {
    /// An ascending order key on `column`.
    pub fn asc(column: impl Into<String>) -> OrderKey {
        OrderKey { column: column.into(), dir: OrderDir::Asc }
    }
    /// A descending order key on `column`.
    pub fn desc(column: impl Into<String>) -> OrderKey {
        OrderKey { column: column.into(), dir: OrderDir::Desc }
    }
}

/// A full query: aggregates over a dataset, filtered and grouped, then ordered and limited. Built
/// with the fluent methods or parsed from SQL ([`crate::sql::parse`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// Dataset (table) name the query reads.
    pub dataset: String,
    /// Aggregates to compute (the `SELECT` list, after any group keys).
    pub aggregates: Vec<Agg>,
    /// Row filter (`WHERE`). Defaults to [`Predicate::True`].
    #[serde(default)]
    pub predicate: Predicate,
    /// Group-by key columns (empty = single global group).
    #[serde(default)]
    pub group_by: Vec<String>,
    /// `ORDER BY` keys, applied left-to-right to the finalised result rows (empty = keep the engine's
    /// natural by-group-key order).
    #[serde(default)]
    pub order_by: Vec<OrderKey>,
    /// `LIMIT` — keep at most this many result rows after ordering (`None` = unlimited).
    #[serde(default)]
    pub limit: Option<usize>,
    /// **Projection** columns for a non-aggregate `SELECT a, b FROM t WHERE …` query. When `Some`,
    /// the query returns raw (filtered, projected) rows instead of aggregates — the single most
    /// common BigQuery shape. `Some(vec![])` means `SELECT *` (all columns). Mutually exclusive with
    /// `aggregates`/`group_by`: a query is either an aggregate query or a projection query.
    #[serde(default)]
    pub projection: Option<Vec<String>>,
}

impl Query {
    /// Start a query over `dataset` with no aggregates, no filter, no grouping.
    pub fn new(dataset: impl Into<String>) -> Query {
        Query {
            dataset: dataset.into(),
            aggregates: Vec::new(),
            predicate: Predicate::True,
            group_by: Vec::new(),
            order_by: Vec::new(),
            limit: None,
            projection: None,
        }
    }

    /// Make this a **projection** query selecting `columns` (empty = `SELECT *`). Clears aggregates
    /// and grouping — a projection and an aggregate query are distinct shapes.
    pub fn project(mut self, columns: Vec<String>) -> Query {
        self.aggregates.clear();
        self.group_by.clear();
        self.projection = Some(columns);
        self
    }

    /// True if this is a projection (raw-row) query rather than an aggregate query.
    pub fn is_projection(&self) -> bool {
        self.projection.is_some()
    }

    /// **Project-map**: filter a shard's rows by the predicate and keep only the projected columns
    /// (all columns when the projection list is empty). Pure and deterministic — the projection
    /// analogue of [`Query::map_shard`], with predicate + column pruning applied during the scan so a
    /// host ships only the rows and columns the query asked for.
    pub fn project_shard(&self, rows: &[Row]) -> Vec<Row> {
        let cols = self.projection.as_deref().unwrap_or(&[]);
        rows.iter()
            .filter(|r| self.predicate.eval(r))
            .map(|r| {
                if cols.is_empty() {
                    r.clone()
                } else {
                    cols.iter()
                        .filter_map(|c| r.get(c).map(|v| (c.clone(), v.clone())))
                        .collect()
                }
            })
            .collect()
    }

    /// Add an aggregate to the `SELECT` list (builder).
    pub fn agg(mut self, a: Agg) -> Query {
        self.aggregates.push(a);
        self
    }

    /// Set the `WHERE` predicate (builder).
    pub fn filter(mut self, p: Predicate) -> Query {
        self.predicate = p;
        self
    }

    /// Add a `GROUP BY` key column (builder).
    pub fn group(mut self, key: impl Into<String>) -> Query {
        self.group_by.push(key.into());
        self
    }

    /// Add an `ORDER BY` key (builder). Keys apply left-to-right (the first is the primary sort).
    pub fn order(mut self, key: OrderKey) -> Query {
        self.order_by.push(key);
        self
    }

    /// Set the `LIMIT` (builder): keep at most `n` result rows after ordering.
    pub fn limit(mut self, n: usize) -> Query {
        self.limit = Some(n);
        self
    }

    /// Apply this query's `ORDER BY` and `LIMIT` to a finalised result set, returning the shaped
    /// rows. Pure: ordering is a stable multi-key sort (see [`crate::order::order_rows`]) followed by
    /// a `LIMIT` truncation. With neither clause this is the identity (the engine's natural order is
    /// preserved). Called by the engine after the reduce, so it works identically for local and
    /// distributed runs.
    pub fn shape(&self, mut rows: Vec<GroupResult>) -> Vec<GroupResult> {
        crate::order::order_rows(&mut rows, &self.group_by, &self.order_by);
        crate::order::apply_limit(&mut rows, self.limit);
        rows
    }

    /// Validate the query is runnable. It must name a dataset and be exactly one of:
    /// - an **aggregate** query (at least one of COUNT/SUM/MIN/MAX/AVG), or
    /// - a **projection** query (`SELECT a, b …` / `SELECT *`).
    ///
    /// A projection query may not also carry aggregates or `GROUP BY`; an aggregate query may not
    /// carry a projection. These shapes are mutually exclusive.
    pub fn validate(&self) -> Result<()> {
        if self.dataset.trim().is_empty() {
            bail!("query has no dataset");
        }
        if self.is_projection() {
            if !self.aggregates.is_empty() {
                bail!("a projection query cannot also select aggregates");
            }
            if !self.group_by.is_empty() {
                bail!("a projection query cannot use GROUP BY (it returns raw rows)");
            }
            return Ok(());
        }
        if self.aggregates.is_empty() {
            bail!("query selects no aggregates (need at least one of COUNT/SUM/MIN/MAX/AVG) or a projection");
        }
        Ok(())
    }

    /// **Map**: run this query over one shard's rows, producing a [`Partial`] (per-group folded
    /// accumulators). Pure and deterministic — the unit of work fanned to a host, and safe to retry
    /// on a different host because it has no side effects.
    pub fn map_shard(&self, rows: &[Row]) -> Partial {
        let mut groups: BTreeMap<Vec<String>, Vec<Accum>> = BTreeMap::new();
        for row in rows {
            if !self.predicate.eval(row) {
                continue;
            }
            let key = self.group_key(row);
            let accs = groups
                .entry(key)
                .or_insert_with(|| self.aggregates.iter().map(|a| a.zero()).collect());
            for (agg, acc) in self.aggregates.iter().zip(accs.iter_mut()) {
                agg.fold(acc, row);
            }
        }
        Partial { aggregates: self.aggregates.clone(), groups }
    }

    /// The group key for a row: the string form of each `GROUP BY` column's value (missing => empty
    /// string). With no `GROUP BY` this is the empty vector (one global group).
    fn group_key(&self, row: &Row) -> Vec<String> {
        self.group_by
            .iter()
            .map(|k| match row.get(k) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            })
            .collect()
    }
}

/// Total order over JSON values for comparison: numbers by value, strings lexically, bools, null.
/// Returns `None` for cross-type or non-orderable comparisons (arrays/objects) so the caller can
/// treat them as "not ordered".
pub fn json_cmp(a: &serde_json::Value, b: &serde_json::Value) -> Option<std::cmp::Ordering> {
    use serde_json::Value::*;
    match (a, b) {
        (Null, Null) => Some(std::cmp::Ordering::Equal),
        (Bool(x), Bool(y)) => Some(x.cmp(y)),
        (Number(_), Number(_)) => {
            let (x, y) = (as_f64(a)?, as_f64(b)?);
            x.partial_cmp(&y)
        }
        (String(x), String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Coerce a JSON value to `f64` if it is a number (or a numeric string). Returns `None` otherwise.
pub fn as_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Coerce a JSON value to a [`Num`]: an exact integer when the value is an integral number (or a
/// numeric string that parses as `i128`), otherwise a float. Returns `None` for non-numeric cells.
/// This is what keeps SUM in the lossless integer lane for counts and money base units.
pub fn as_number(v: &serde_json::Value) -> Option<Num> {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(Num::Int(i as i128))
            } else if let Some(u) = n.as_u64() {
                Some(Num::Int(u as i128))
            } else {
                n.as_f64().map(Num::Float)
            }
        }
        serde_json::Value::String(s) => {
            let t = s.trim();
            if let Ok(i) = t.parse::<i128>() {
                Some(Num::Int(i))
            } else {
                t.parse::<f64>().ok().map(Num::Float)
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    fn row(pairs: &[(&str, serde_json::Value)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn as_number_classifies_int_vs_float() {
        assert_eq!(as_number(&json!(42)), Some(Num::Int(42)));
        assert_eq!(as_number(&json!(-7)), Some(Num::Int(-7)));
        assert_eq!(as_number(&json!("100")), Some(Num::Int(100)));
        assert_eq!(as_number(&json!(1.5)), Some(Num::Float(1.5)));
        assert_eq!(as_number(&json!("2.5")), Some(Num::Float(2.5)));
        assert_eq!(as_number(&json!("nope")), None);
        assert_eq!(as_number(&json!(true)), None);
    }

    // The integer SUM lane is exact: summing many large integers via map_shard equals the plain i128
    // sum, with no floating-point loss.
    proptest! {
        #[test]
        fn integer_sum_is_exact(values in prop::collection::vec(-1_000_000_000_000i64..1_000_000_000_000, 0..50)) {
            let q = Query::new("t").agg(Agg::Sum("v".into()));
            let rows: Vec<Row> = values.iter().map(|&v| row(&[("v", json!(v))])).collect();
            let out = q.map_shard(&rows).finalize();
            let expected: i128 = values.iter().map(|&v| v as i128).sum();
            if values.is_empty() {
                // No rows -> no groups at all (a global aggregate over zero rows is the empty set).
                prop_assert!(out.is_empty());
            } else {
                prop_assert_eq!(out[0].values["sum_v"].as_i64().map(|x| x as i128), Some(expected));
            }
        }
    }

    #[test]
    fn cmp_ops_numeric() {
        assert!(CmpOp::Eq.eval(&json!(3), &json!(3)));
        assert!(CmpOp::Lt.eval(&json!(2), &json!(3)));
        assert!(CmpOp::Ge.eval(&json!(3), &json!(3)));
        assert!(!CmpOp::Gt.eval(&json!(3), &json!(3)));
    }

    #[test]
    fn cmp_cross_type_is_unordered() {
        // number vs string: only Ne is true.
        assert!(CmpOp::Ne.eval(&json!(3), &json!("3")));
        assert!(!CmpOp::Eq.eval(&json!(3), &json!("3")));
        assert!(!CmpOp::Lt.eval(&json!(3), &json!("3")));
    }

    #[test]
    fn predicate_missing_field_filters_out() {
        let p = Predicate::Cmp { field: "x".into(), op: CmpOp::Gt, value: json!(0) };
        assert!(!p.eval(&row(&[("y", json!(5))])));
    }

    #[test]
    fn predicate_and_or_not() {
        let r = row(&[("a", json!(5)), ("b", json!("hi"))]);
        let a = Predicate::Cmp { field: "a".into(), op: CmpOp::Gt, value: json!(0) };
        let b = Predicate::Cmp { field: "b".into(), op: CmpOp::Eq, value: json!("no") };
        assert!(Predicate::And(Box::new(a.clone()), Box::new(Predicate::True)).eval(&r));
        assert!(Predicate::Or(Box::new(a.clone()), Box::new(b.clone())).eval(&r));
        assert!(Predicate::Not(Box::new(b)).eval(&r));
    }

    #[test]
    fn map_shard_global_count_sum() {
        let rows = vec![
            row(&[("v", json!(10))]),
            row(&[("v", json!(20))]),
            row(&[("v", json!(30))]),
        ];
        let q = Query::new("t").agg(Agg::Count).agg(Agg::Sum("v".into()));
        let p = q.map_shard(&rows);
        let out = p.finalize();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].values["count"], json!(3));
        // All-integer SUM finalises to an exact JSON integer (the money/int lane).
        assert_eq!(out[0].values["sum_v"], json!(60));
    }

    #[test]
    fn sum_stays_integer_lane_then_promotes_on_fraction() {
        // Integers only -> exact integer result.
        let q = Query::new("t").agg(Agg::Sum("v".into()));
        let rows = vec![row(&[("v", json!(1_000_000_000_000i64))]); 3];
        let out = q.map_shard(&rows).finalize();
        assert_eq!(out[0].values["sum_v"], json!(3_000_000_000_000i64));
        // A fractional value promotes the lane to float.
        let rows2 = vec![row(&[("v", json!(1))]), row(&[("v", json!(0.5))])];
        let out2 = q.map_shard(&rows2).finalize();
        assert_eq!(out2[0].values["sum_v"], json!(1.5));
    }

    #[test]
    fn map_shard_group_by() {
        let rows = vec![
            row(&[("g", json!("a")), ("v", json!(1))]),
            row(&[("g", json!("b")), ("v", json!(2))]),
            row(&[("g", json!("a")), ("v", json!(3))]),
        ];
        let q = Query::new("t").agg(Agg::Sum("v".into())).group("g");
        let out = q.map_shard(&rows).finalize();
        // Two groups: a -> 4, b -> 2.
        let mut by_key: BTreeMap<String, f64> = BTreeMap::new();
        for g in out {
            by_key.insert(g.key[0].clone(), g.values["sum_v"].as_f64().unwrap());
        }
        assert_eq!(by_key["a"], 4.0);
        assert_eq!(by_key["b"], 2.0);
    }

    #[test]
    fn avg_and_minmax() {
        let rows = vec![
            row(&[("v", json!(2))]),
            row(&[("v", json!(4))]),
            row(&[("v", json!(6))]),
        ];
        let q = Query::new("t")
            .agg(Agg::Avg("v".into()))
            .agg(Agg::Min("v".into()))
            .agg(Agg::Max("v".into()));
        let out = q.map_shard(&rows).finalize();
        assert_eq!(out[0].values["avg_v"], json!(4.0));
        assert_eq!(out[0].values["min_v"], json!(2.0));
        assert_eq!(out[0].values["max_v"], json!(6.0));
    }

    #[test]
    fn non_numeric_values_skipped() {
        let rows = vec![
            row(&[("v", json!(10))]),
            row(&[("v", json!("oops"))]),
            row(&[("v", json!(20))]),
        ];
        let q = Query::new("t").agg(Agg::Sum("v".into())).agg(Agg::Count);
        let out = q.map_shard(&rows).finalize();
        // sum skips the non-numeric, count includes every (filtered) row.
        assert_eq!(out[0].values["sum_v"], json!(30));
        assert_eq!(out[0].values["count"], json!(3));
    }

    #[test]
    fn validate_requires_aggregate() {
        assert!(Query::new("t").validate().is_err());
        assert!(Query::new("t").agg(Agg::Count).validate().is_ok());
        assert!(Query::new("").agg(Agg::Count).validate().is_err());
    }

    #[test]
    fn projection_query_validates_and_maps() {
        let q = Query::new("t").project(vec!["a".into(), "b".into()]);
        assert!(q.is_projection());
        assert!(q.validate().is_ok());
        let rows = vec![
            row(&[("a", json!(1)), ("b", json!("x")), ("c", json!(true))]),
            row(&[("a", json!(2)), ("b", json!("y"))]),
        ];
        let out = q.project_shard(&rows);
        assert_eq!(out.len(), 2);
        // Only projected columns survive; absent columns are simply not present.
        assert_eq!(out[0].get("a"), Some(&json!(1)));
        assert_eq!(out[0].get("b"), Some(&json!("x")));
        assert!(!out[0].contains_key("c"), "c was not projected");
    }

    #[test]
    fn projection_star_keeps_all_columns_and_applies_where() {
        let q = Query::new("t")
            .project(vec![]) // SELECT *
            .filter(Predicate::Cmp { field: "a".into(), op: CmpOp::Gt, value: json!(1) });
        let rows = vec![
            row(&[("a", json!(1)), ("b", json!("x"))]),
            row(&[("a", json!(5)), ("b", json!("y"))]),
        ];
        let out = q.project_shard(&rows);
        assert_eq!(out.len(), 1, "WHERE a > 1 keeps one row");
        assert_eq!(out[0].get("a"), Some(&json!(5)));
        assert_eq!(out[0].get("b"), Some(&json!("y")), "SELECT * keeps all columns");
    }

    #[test]
    fn projection_and_aggregate_are_mutually_exclusive() {
        // project() clears aggregates; adding an aggregate after still validates as long as
        // projection is unset. A manually-constructed inconsistent query is rejected by validate().
        let mut q = Query::new("t").agg(Agg::Count);
        q.projection = Some(vec!["a".into()]);
        assert!(q.validate().is_err(), "projection + aggregate must be rejected");
    }

    #[test]
    fn agg_output_names() {
        assert_eq!(Agg::Count.output_name(), "count");
        assert_eq!(Agg::Sum("x".into()).output_name(), "sum_x");
        assert_eq!(Agg::Avg("y".into()).output_name(), "avg_y");
    }

    #[test]
    fn builder_order_and_limit() {
        let q = Query::new("t")
            .agg(Agg::Count)
            .group("g")
            .order(OrderKey::desc("count"))
            .order(OrderKey::asc("g"))
            .limit(7);
        assert_eq!(q.order_by, vec![OrderKey::desc("count"), OrderKey::asc("g")]);
        assert_eq!(q.limit, Some(7));
    }

    #[test]
    fn shape_orders_then_limits() {
        // shape() must order before truncating: top-2 by count desc.
        let q = Query::new("t").agg(Agg::Count).group("g").order(OrderKey::desc("count")).limit(2);
        let make = |k: &str, c: i64| crate::combine::GroupResult {
            key: vec![k.into()],
            values: [("count".to_string(), json!(c))].into_iter().collect(),
        };
        let out = q.shape(vec![make("a", 3), make("b", 9), make("c", 1)]);
        let keys: Vec<&str> = out.iter().map(|g| g.key[0].as_str()).collect();
        assert_eq!(keys, vec!["b", "a"]);
    }

    #[test]
    fn query_serde_roundtrip_with_order_limit() {
        let q = Query::new("t").agg(Agg::Count).group("g").order(OrderKey::asc("g")).limit(5);
        let json = serde_json::to_string(&q).unwrap();
        let back: Query = serde_json::from_str(&json).unwrap();
        assert_eq!(back, q);
        // Older payloads without order_by/limit still deserialize (serde default).
        let legacy = r#"{"dataset":"t","aggregates":["Count"],"group_by":["g"]}"#;
        let q2: Query = serde_json::from_str(legacy).unwrap();
        assert!(q2.order_by.is_empty());
        assert_eq!(q2.limit, None);
    }
}
