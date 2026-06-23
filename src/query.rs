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

use crate::combine::{Accum, Partial};
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

    /// A fresh zero-accumulator for this aggregate's identity element.
    pub fn zero(&self) -> Accum {
        match self {
            Agg::Count => Accum::Count(0),
            Agg::Sum(_) => Accum::Sum(0.0),
            Agg::Min(_) => Accum::Min(None),
            Agg::Max(_) => Accum::Max(None),
            Agg::Avg(_) => Accum::Avg { sum: 0.0, count: 0 },
        }
    }

    /// Fold one row's contribution into `acc`. Reads the row's field (none for `Count`); non-numeric
    /// or missing values are skipped for numeric aggregates so a single bad cell never corrupts the
    /// result. Pure.
    pub fn fold(&self, acc: &mut Accum, row: &Row) {
        match self {
            Agg::Count => {
                if let Accum::Count(n) = acc {
                    *n += 1;
                }
            }
            Agg::Sum(f) => {
                if let (Accum::Sum(s), Some(v)) = (acc, row.get(f).and_then(as_f64)) {
                    *s += v;
                }
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
                    *sum += v;
                    *count += 1;
                }
            }
        }
    }
}

/// A full query: aggregates over a dataset, filtered and grouped. Built with the fluent methods or
/// parsed from SQL ([`crate::sql::parse`]).
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
}

impl Query {
    /// Start a query over `dataset` with no aggregates, no filter, no grouping.
    pub fn new(dataset: impl Into<String>) -> Query {
        Query {
            dataset: dataset.into(),
            aggregates: Vec::new(),
            predicate: Predicate::True,
            group_by: Vec::new(),
        }
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

    /// Validate the query is runnable: it must request at least one aggregate. (A projection-only
    /// query is a different, future code path — this engine is aggregate-first.)
    pub fn validate(&self) -> Result<()> {
        if self.dataset.trim().is_empty() {
            bail!("query has no dataset");
        }
        if self.aggregates.is_empty() {
            bail!("query selects no aggregates (need at least one of COUNT/SUM/MIN/MAX/AVG)");
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(pairs: &[(&str, serde_json::Value)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
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
        assert_eq!(out[0].values["sum_v"], json!(60.0));
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
        assert_eq!(out[0].values["sum_v"], json!(30.0));
        assert_eq!(out[0].values["count"], json!(3));
    }

    #[test]
    fn validate_requires_aggregate() {
        assert!(Query::new("t").validate().is_err());
        assert!(Query::new("t").agg(Agg::Count).validate().is_ok());
        assert!(Query::new("").agg(Agg::Count).validate().is_err());
    }

    #[test]
    fn agg_output_names() {
        assert_eq!(Agg::Count.output_name(), "count");
        assert_eq!(Agg::Sum("x".into()).output_name(), "sum_x");
        assert_eq!(Agg::Avg("y".into()).output_name(), "avg_y");
    }
}
