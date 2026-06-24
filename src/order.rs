//! Result shaping — `ORDER BY` and `LIMIT` over the finalised result rows.
//!
//! Ordering and limiting are **post-reduce** concerns: by the time the coordinator has a
//! [`GroupResult`] set, the distributed map-reduce is done and the answer is small (one row per
//! group). So unlike the aggregate algebra, ordering is *not* a monoid and does *not* need to be
//! distributable — it is a plain, pure, deterministic sort applied once at the end, identically for
//! the coordinator-local and mesh-distributed engines.
//!
//! ## What a key can name
//!
//! An [`OrderKey`] column names either a **group-key column** (matched positionally against the
//! query's `GROUP BY` list) or an **aggregate output column** (`count`, `sum_amount`, ...). The two
//! namespaces are resolved by [`row_cell`]: a group column reads the corresponding entry of
//! [`GroupResult::key`]; otherwise the name is looked up in [`GroupResult::values`].
//!
//! ## Total, type-aware ordering
//!
//! Cells are compared with [`cell_cmp`], which orders numbers numerically, strings lexically, and
//! gives a fixed cross-type ordering (null < bool < number < string) so the sort is **total and
//! deterministic** for heterogeneous schema-on-read data — there is no panic and no
//! "incomparable" gap. The sort is **stable**, so rows equal on all keys keep their incoming
//! (group-key) order, and a key naming a non-existent column is a no-op (every row compares equal on
//! it) rather than an error.

use crate::combine::GroupResult;
use crate::query::{OrderDir, OrderKey, json_cmp};
use std::cmp::Ordering;

/// Resolve an order-by `column` to a row's cell value. If `column` is the name of one of the
/// `group_by` columns, the matching entry of the row's group key is returned (as a JSON string);
/// otherwise the name is looked up among the aggregate output `values`. Returns [`None`] when the
/// column matches neither (so the caller treats every row as equal on that key — a no-op).
pub fn row_cell<'a>(
    row: &'a GroupResult,
    group_by: &[String],
    column: &str,
) -> Option<std::borrow::Cow<'a, serde_json::Value>> {
    if let Some(idx) = group_by.iter().position(|g| g == column) {
        // Group-key columns are carried as strings; present them as JSON strings for comparison.
        return row
            .key
            .get(idx)
            .map(|s| std::borrow::Cow::Owned(serde_json::Value::String(s.clone())));
    }
    row.values.get(column).map(std::borrow::Cow::Borrowed)
}

/// A **total** order over two JSON cells for sorting. Numbers compare numerically and strings
/// lexically (via [`json_cmp`]); across types a fixed rank (null < bool < number < string < other)
/// breaks the tie, so the comparison is never "undefined" and the resulting sort is deterministic
/// even on ragged schema-on-read rows. Two `null`s (or two missing cells) are equal.
pub fn cell_cmp(a: &serde_json::Value, b: &serde_json::Value) -> Ordering {
    if let Some(ord) = json_cmp(a, b) {
        return ord;
    }
    // Cross-type or non-orderable: fall back to a stable type-rank ordering.
    type_rank(a).cmp(&type_rank(b))
}

/// A fixed ordering rank per JSON type so cross-type comparisons are total and deterministic.
fn type_rank(v: &serde_json::Value) -> u8 {
    use serde_json::Value::*;
    match v {
        Null => 0,
        Bool(_) => 1,
        Number(_) => 2,
        String(_) => 3,
        Array(_) => 4,
        Object(_) => 5,
    }
}

/// Sort `rows` in place by `order_by` (left-to-right key priority), resolving each key against the
/// `group_by` columns and aggregate outputs. The sort is **stable**, so an empty `order_by` leaves
/// the input order untouched and rows tied on all keys keep their relative order. Pure.
pub fn order_rows(rows: &mut [GroupResult], group_by: &[String], order_by: &[OrderKey]) {
    if order_by.is_empty() {
        return;
    }
    rows.sort_by(|a, b| cmp_by_keys(a, b, group_by, order_by));
}

/// Compare two rows by the ordered key list: the first key that distinguishes them decides; ties
/// fall through to the next key; all-equal yields [`Ordering::Equal`] (and the stable sort then
/// preserves input order).
fn cmp_by_keys(
    a: &GroupResult,
    b: &GroupResult,
    group_by: &[String],
    order_by: &[OrderKey],
) -> Ordering {
    let null = serde_json::Value::Null;
    for OrderKey { column, dir } in order_by {
        let av = row_cell(a, group_by, column);
        let bv = row_cell(b, group_by, column);
        let av = av.as_deref().unwrap_or(&null);
        let bv = bv.as_deref().unwrap_or(&null);
        let mut ord = cell_cmp(av, bv);
        if *dir == OrderDir::Desc {
            ord = ord.reverse();
        }
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Truncate `rows` to at most `limit` entries (no-op when `limit` is [`None`]). Applied **after**
/// ordering so a `LIMIT` always returns the top-N of the requested order.
pub fn apply_limit(rows: &mut Vec<GroupResult>, limit: Option<usize>) {
    if let Some(n) = limit {
        rows.truncate(n);
    }
}

/// Sort raw projected rows in place by `order_by` (left-to-right key priority). Unlike
/// [`order_rows`], a key here names a **row column** directly (projection queries have no group keys
/// or aggregate outputs). A key naming an absent column compares every row equal on it (a stable
/// no-op), and cells are compared with the same total, type-aware [`cell_cmp`]. Pure.
pub fn order_raw_rows(rows: &mut [crate::dataset::Row], order_by: &[OrderKey]) {
    if order_by.is_empty() {
        return;
    }
    let null = serde_json::Value::Null;
    rows.sort_by(|a, b| {
        for OrderKey { column, dir } in order_by {
            let av = a.get(column).unwrap_or(&null);
            let bv = b.get(column).unwrap_or(&null);
            let mut ord = cell_cmp(av, bv);
            if *dir == OrderDir::Desc {
                ord = ord.reverse();
            }
            if ord != Ordering::Equal {
                return ord;
            }
        }
        Ordering::Equal
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Agg, Query};
    use serde_json::json;
    use std::collections::BTreeMap;

    /// Build a result row with a single group key and a set of named aggregate values.
    fn gr(key: &str, vals: &[(&str, serde_json::Value)]) -> GroupResult {
        let values: BTreeMap<String, serde_json::Value> =
            vals.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        GroupResult { key: vec![key.to_string()], values }
    }

    #[test]
    fn order_by_aggregate_desc() {
        let mut rows = vec![
            gr("a", &[("count", json!(3))]),
            gr("b", &[("count", json!(10))]),
            gr("c", &[("count", json!(1))]),
        ];
        order_rows(&mut rows, &["g".into()], &[OrderKey::desc("count")]);
        let keys: Vec<&str> = rows.iter().map(|r| r.key[0].as_str()).collect();
        assert_eq!(keys, vec!["b", "a", "c"]);
    }

    #[test]
    fn order_by_aggregate_asc() {
        let mut rows = vec![
            gr("a", &[("sum_v", json!(3.0))]),
            gr("b", &[("sum_v", json!(10.0))]),
            gr("c", &[("sum_v", json!(1.0))]),
        ];
        order_rows(&mut rows, &["g".into()], &[OrderKey::asc("sum_v")]);
        let keys: Vec<&str> = rows.iter().map(|r| r.key[0].as_str()).collect();
        assert_eq!(keys, vec!["c", "a", "b"]);
    }

    #[test]
    fn order_by_group_key_column() {
        let mut rows =
            vec![gr("z", &[("count", json!(1))]), gr("a", &[("count", json!(1))]), gr("m", &[("count", json!(1))])];
        order_rows(&mut rows, &["region".into()], &[OrderKey::asc("region")]);
        let keys: Vec<&str> = rows.iter().map(|r| r.key[0].as_str()).collect();
        assert_eq!(keys, vec!["a", "m", "z"]);
    }

    #[test]
    fn multi_key_tiebreak() {
        // Primary key ties (all count=5), secondary key (sum_v) breaks the tie ascending.
        let mut rows = vec![
            gr("a", &[("count", json!(5)), ("sum_v", json!(30.0))]),
            gr("b", &[("count", json!(5)), ("sum_v", json!(10.0))]),
            gr("c", &[("count", json!(2)), ("sum_v", json!(99.0))]),
        ];
        order_rows(
            &mut rows,
            &["g".into()],
            &[OrderKey::desc("count"), OrderKey::asc("sum_v")],
        );
        let keys: Vec<&str> = rows.iter().map(|r| r.key[0].as_str()).collect();
        // count desc => a,b before c; within count=5, sum_v asc => b before a.
        assert_eq!(keys, vec!["b", "a", "c"]);
    }

    #[test]
    fn limit_truncates_after_order() {
        let mut rows = vec![
            gr("a", &[("count", json!(3))]),
            gr("b", &[("count", json!(10))]),
            gr("c", &[("count", json!(1))]),
        ];
        order_rows(&mut rows, &["g".into()], &[OrderKey::desc("count")]);
        apply_limit(&mut rows, Some(2));
        let keys: Vec<&str> = rows.iter().map(|r| r.key[0].as_str()).collect();
        assert_eq!(keys, vec!["b", "a"]); // top-2 by count desc
    }

    #[test]
    fn limit_none_and_zero() {
        let mut rows = vec![gr("a", &[("count", json!(1))]), gr("b", &[("count", json!(2))])];
        let mut copy = rows.clone();
        apply_limit(&mut copy, None);
        assert_eq!(copy.len(), 2);
        apply_limit(&mut rows, Some(0));
        assert!(rows.is_empty());
    }

    #[test]
    fn empty_order_by_is_stable_noop() {
        let rows = vec![gr("z", &[("count", json!(1))]), gr("a", &[("count", json!(2))])];
        let mut shaped = rows.clone();
        order_rows(&mut shaped, &["g".into()], &[]);
        assert_eq!(shaped, rows, "no ORDER BY keys must preserve input order");
    }

    #[test]
    fn unknown_column_is_noop_not_panic() {
        // A key naming neither a group column nor an aggregate output: every row equal => stable.
        let rows = vec![gr("z", &[("count", json!(1))]), gr("a", &[("count", json!(2))])];
        let mut shaped = rows.clone();
        order_rows(&mut shaped, &["g".into()], &[OrderKey::asc("nope")]);
        assert_eq!(shaped, rows);
    }

    #[test]
    fn cell_cmp_is_total_across_types() {
        // null < bool < number < string is the fixed cross-type ranking.
        assert_eq!(cell_cmp(&json!(null), &json!(true)), Ordering::Less);
        assert_eq!(cell_cmp(&json!(true), &json!(1)), Ordering::Less);
        assert_eq!(cell_cmp(&json!(1), &json!("a")), Ordering::Less);
        // Same type compares by value.
        assert_eq!(cell_cmp(&json!(2), &json!(1)), Ordering::Greater);
        assert_eq!(cell_cmp(&json!("a"), &json!("b")), Ordering::Less);
        assert_eq!(cell_cmp(&json!(null), &json!(null)), Ordering::Equal);
    }

    #[test]
    fn query_shape_orders_and_limits() {
        // End-to-end through Query::shape: order by count desc, limit 1.
        let q = Query::new("t").agg(Agg::Count).group("g").order(OrderKey::desc("count")).limit(1);
        let rows = vec![
            gr("a", &[("count", json!(3))]),
            gr("b", &[("count", json!(10))]),
            gr("c", &[("count", json!(1))]),
        ];
        let out = q.shape(rows);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].key[0], "b");
    }

    #[test]
    fn query_shape_identity_without_clauses() {
        let q = Query::new("t").agg(Agg::Count).group("g");
        let rows = vec![gr("a", &[("count", json!(3))]), gr("b", &[("count", json!(10))])];
        assert_eq!(q.shape(rows.clone()), rows);
    }

    // Property: order then limit-N equals taking the first N of the fully ordered list. A limit must
    // never change *which* rows are the top-N versus a full sort (no reordering leak).
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn limit_is_prefix_of_full_order(
            vals in prop::collection::vec(-1000i64..1000, 0..40),
            n in 0usize..50,
            desc in any::<bool>(),
        ) {
            let key = if desc { OrderKey::desc("count") } else { OrderKey::asc("count") };
            let rows: Vec<GroupResult> = vals.iter().enumerate()
                .map(|(i, &v)| gr(&format!("k{i:03}"), &[("count", json!(v))]))
                .collect();

            let mut full = rows.clone();
            order_rows(&mut full, &["g".into()], std::slice::from_ref(&key));
            let expected_prefix: Vec<GroupResult> = full.iter().take(n).cloned().collect();

            let mut limited = rows.clone();
            order_rows(&mut limited, &["g".into()], std::slice::from_ref(&key));
            apply_limit(&mut limited, Some(n));

            prop_assert_eq!(limited, expected_prefix);
        }
    }

    // Property: ordering is a permutation — shaping never drops or duplicates rows (absent a LIMIT),
    // and the output is sorted (each adjacent pair is in non-decreasing order under the key).
    proptest! {
        #[test]
        fn order_is_sorted_permutation(vals in prop::collection::vec(-1000i64..1000, 0..40)) {
            let rows: Vec<GroupResult> = vals.iter().enumerate()
                .map(|(i, &v)| gr(&format!("k{i:03}"), &[("count", json!(v))]))
                .collect();
            let mut sorted = rows.clone();
            order_rows(&mut sorted, &["g".into()], &[OrderKey::asc("count")]);

            // Same multiset (permutation): equal length and equal sorted key sets.
            prop_assert_eq!(sorted.len(), rows.len());
            let mut a: Vec<String> = rows.iter().map(|r| r.key[0].clone()).collect();
            let mut b: Vec<String> = sorted.iter().map(|r| r.key[0].clone()).collect();
            a.sort(); b.sort();
            prop_assert_eq!(a, b);

            // Sorted ascending by count.
            for w in sorted.windows(2) {
                let x = w[0].values["count"].as_i64().unwrap();
                let y = w[1].values["count"].as_i64().unwrap();
                prop_assert!(x <= y, "not ascending: {} then {}", x, y);
            }
        }
    }
}
