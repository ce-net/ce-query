//! Equi-joins across two datasets — a hash-join over `serde_json` rows.
//!
//! A join takes a **left** and a **right** row stream and a pair of key columns, and emits one
//! combined row for every `(l, r)` pair whose `l[left_key] == r[right_key]` (an inner equi-join).
//! Columns are merged into a single output row; on a name clash the right side's value is written
//! under a prefixed name (`right.<col>`) so no data is silently lost.
//!
//! ## Why a hash-join, and how it distributes
//!
//! The in-memory [`hash_join`] is the reference semantics: build a multimap on the smaller side keyed
//! by the join value, then probe it with the other side — `O(n + m)` instead of the `O(n·m)`
//! nested-loop. It is **pure and total**: a missing/null key never panics, and rows with no match are
//! simply absent (inner join).
//!
//! The distribution story is **co-partitioning** ([`partition_by_key`]): hash each row by its join
//! key into the same number of buckets on both sides, so bucket `i` of the left can only match bucket
//! `i` of the right. Each bucket pair is then an independent local [`hash_join`] — the map-reduce
//! shape for joins (a host that owns left+right bucket `i` joins them with no cross-host shuffle).
//! [`distributed_join`] composes these: partition both sides identically, join bucket-wise, and
//! concatenate — and a property test proves it equals the single in-memory [`hash_join`].
//!
//! The result of a join is itself a [`Row`] stream, so it feeds straight back into the aggregate
//! engine: `JOIN then GROUP BY` is `distributed_join(...)` followed by [`Query::map_shard`] over the
//! joined rows.
//!
//! [`Query::map_shard`]: crate::query::Query::map_shard

use crate::dataset::Row;
use std::collections::HashMap;

/// The two key columns of an equi-join: which left column equals which right column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinKeys {
    /// Column on the left side whose value must match `right`.
    pub left: String,
    /// Column on the right side whose value must match `left`.
    pub right: String,
}

impl JoinKeys {
    /// Join on the same column name on both sides (the common case).
    pub fn on(column: impl Into<String>) -> JoinKeys {
        let c = column.into();
        JoinKeys { left: c.clone(), right: c }
    }

    /// Join on differently-named columns.
    pub fn new(left: impl Into<String>, right: impl Into<String>) -> JoinKeys {
        JoinKeys { left: left.into(), right: right.into() }
    }
}

/// The canonical string form of a join-key cell, used both for hashing rows into buckets and for the
/// equality test. Strings join on their content; other JSON scalars on their compact serialization;
/// a missing key yields [`None`] so the row participates in **no** match (NULL never equi-joins,
/// matching SQL). Keeping the key as a `String` makes the hash map cheap and the bucketing
/// deterministic.
pub fn key_str(row: &Row, column: &str) -> Option<String> {
    match row.get(column) {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(v) => Some(v.to_string()),
    }
}

/// Merge a left and right row into one output row. Left columns are copied as-is; each right column
/// is added under its own name, unless that name already exists (a clash), in which case it is stored
/// under `right.<col>` so no value is lost and the join is non-destructive.
pub fn merge_rows(left: &Row, right: &Row) -> Row {
    let mut out = left.clone();
    for (k, v) in right {
        if out.contains_key(k) {
            out.insert(format!("right.{k}"), v.clone());
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

/// In-memory inner equi-join: emit a merged row for every left/right pair sharing a join key. Builds
/// the hash side on `right` (probed by `left`). Order is deterministic: left rows in input order, and
/// for each left row its matches in right input order. Rows with a missing/null key never match.
pub fn hash_join(left: &[Row], right: &[Row], keys: &JoinKeys) -> Vec<Row> {
    // Build: key -> indices into `right` (a multimap; duplicate keys all match).
    let mut index: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in right.iter().enumerate() {
        if let Some(k) = key_str(r, &keys.right) {
            index.entry(k).or_default().push(i);
        }
    }
    let mut out = Vec::new();
    for l in left {
        let Some(k) = key_str(l, &keys.left) else { continue };
        if let Some(matches) = index.get(&k) {
            for &ri in matches {
                out.push(merge_rows(l, &right[ri]));
            }
        }
    }
    out
}

/// Partition rows into `n` buckets by the hash of their join-key value, so that two rows with equal
/// keys (on either side) always land in the same bucket index. Rows with a missing/null key are
/// dropped (they can never match in an inner join). `n` is clamped to at least 1.
///
/// This is the co-partitioning primitive: partition both join sides with the *same* `n` and join
/// bucket-wise — a row in left bucket `i` can only match a row in right bucket `i`.
pub fn partition_by_key(rows: &[Row], column: &str, n: usize) -> Vec<Vec<Row>> {
    let n = n.max(1);
    let mut buckets: Vec<Vec<Row>> = vec![Vec::new(); n];
    for row in rows {
        if let Some(k) = key_str(row, column) {
            let b = (bucket_hash(&k) as usize) % n;
            buckets[b].push(row.clone());
        }
    }
    buckets
}

/// A stable 64-bit hash of a key string for bucketing — sha256-derived so it is independent of the
/// process-randomised `std` hasher (a partition must be reproducible across hosts and runs).
fn bucket_hash(key: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(key.as_bytes());
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(buf)
}

/// Distributed-shaped equi-join: co-partition both sides into `partitions` buckets by their join key,
/// join each bucket pair locally with [`hash_join`], and concatenate. With `partitions == 1` this is
/// exactly [`hash_join`]; with more buckets it is the same answer computed as independent per-bucket
/// joins — the join analogue of map-reduce, validated equal to [`hash_join`] by a property test.
pub fn distributed_join(
    left: &[Row],
    right: &[Row],
    keys: &JoinKeys,
    partitions: usize,
) -> Vec<Row> {
    let n = partitions.max(1);
    let lb = partition_by_key(left, &keys.left, n);
    let rb = partition_by_key(right, &keys.right, n);
    let mut out = Vec::new();
    for i in 0..n {
        out.extend(hash_join(&lb[i], &rb[i], keys));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Agg, Query};
    use proptest::prelude::*;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn row(pairs: &[(&str, serde_json::Value)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    /// Stable signature of a joined row set (sorted) for order-independent equality checks.
    fn sig(rows: &[Row]) -> Vec<String> {
        let mut s: Vec<String> = rows.iter().map(|r| serde_json::to_string(r).unwrap()).collect();
        s.sort();
        s
    }

    #[test]
    fn inner_join_basic() {
        let users = vec![
            row(&[("uid", json!(1)), ("name", json!("alice"))]),
            row(&[("uid", json!(2)), ("name", json!("bob"))]),
        ];
        let orders = vec![
            row(&[("uid", json!(1)), ("amount", json!(10))]),
            row(&[("uid", json!(1)), ("amount", json!(20))]),
            row(&[("uid", json!(3)), ("amount", json!(99))]), // no matching user
        ];
        let out = hash_join(&users, &orders, &JoinKeys::on("uid"));
        // alice matches two orders; bob matches none; order uid=3 has no user.
        assert_eq!(out.len(), 2);
        for r in &out {
            assert_eq!(r["name"], json!("alice"));
        }
    }

    #[test]
    fn join_on_different_column_names() {
        let left = vec![row(&[("id", json!(1)), ("a", json!("x"))])];
        let right = vec![row(&[("user_id", json!(1)), ("b", json!("y"))])];
        let out = hash_join(&left, &right, &JoinKeys::new("id", "user_id"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["a"], json!("x"));
        assert_eq!(out[0]["b"], json!("y"));
    }

    #[test]
    fn column_clash_is_prefixed_not_lost() {
        // Both sides have a `val` column; the right one must survive under `right.val`.
        let left = vec![row(&[("k", json!(1)), ("val", json!("L"))])];
        let right = vec![row(&[("k", json!(1)), ("val", json!("R"))])];
        let out = hash_join(&left, &right, &JoinKeys::on("k"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["val"], json!("L"));
        assert_eq!(out[0]["right.val"], json!("R"));
    }

    #[test]
    fn null_and_missing_key_never_matches() {
        let left = vec![
            row(&[("k", json!(null)), ("a", json!(1))]),
            row(&[("a", json!(2))]), // missing k entirely
        ];
        let right = vec![row(&[("k", json!(null)), ("b", json!(3))])];
        let out = hash_join(&left, &right, &JoinKeys::on("k"));
        assert!(out.is_empty(), "null/missing keys must not join (SQL NULL semantics)");
    }

    #[test]
    fn many_to_many_cartesian_within_key() {
        // 2 left x 2 right on the same key => 4 combined rows.
        let left = vec![row(&[("k", json!(1)), ("l", json!("a"))]), row(&[("k", json!(1)), ("l", json!("b"))])];
        let right = vec![row(&[("k", json!(1)), ("r", json!("x"))]), row(&[("k", json!(1)), ("r", json!("y"))])];
        let out = hash_join(&left, &right, &JoinKeys::on("k"));
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn string_keys_join() {
        let left = vec![row(&[("city", json!("NYC")), ("pop", json!(8))])];
        let right = vec![row(&[("city", json!("NYC")), ("zip", json!("10001"))])];
        let out = hash_join(&left, &right, &JoinKeys::on("city"));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["zip"], json!("10001"));
    }

    #[test]
    fn partition_is_co_located() {
        // Equal keys on both sides must hash to the same bucket index.
        let n = 5;
        let left: Vec<Row> = (0..50).map(|i| row(&[("k", json!(i)), ("s", json!("L"))])).collect();
        let right: Vec<Row> = (0..50).map(|i| row(&[("k", json!(i)), ("s", json!("R"))])).collect();
        let lb = partition_by_key(&left, "k", n);
        let rb = partition_by_key(&right, "k", n);
        // For each key, find its left bucket and right bucket; they must be equal.
        for i in 0..50 {
            let key = i.to_string();
            let lbi = lb.iter().position(|b| b.iter().any(|r| key_str(r, "k").as_deref() == Some(&key)));
            let rbi = rb.iter().position(|b| b.iter().any(|r| key_str(r, "k").as_deref() == Some(&key)));
            assert_eq!(lbi, rbi, "key {key} landed in different buckets");
        }
    }

    #[test]
    fn partition_drops_null_keys() {
        let rows = vec![row(&[("k", json!(null)), ("a", json!(1))]), row(&[("k", json!(5))])];
        let buckets = partition_by_key(&rows, "k", 3);
        let total: usize = buckets.iter().map(|b| b.len()).sum();
        assert_eq!(total, 1, "null/missing-key rows are dropped from partitions");
    }

    #[test]
    fn distributed_join_equals_hash_join_single_partition() {
        let left = vec![row(&[("k", json!(1)), ("a", json!("p"))]), row(&[("k", json!(2)), ("a", json!("q"))])];
        let right = vec![row(&[("k", json!(1)), ("b", json!("x"))]), row(&[("k", json!(2)), ("b", json!("y"))])];
        let one = hash_join(&left, &right, &JoinKeys::on("k"));
        let dist = distributed_join(&left, &right, &JoinKeys::on("k"), 1);
        assert_eq!(sig(&one), sig(&dist));
    }

    #[test]
    fn join_feeds_aggregate_engine() {
        // JOIN then GROUP BY: total order amount per user name.
        let users = vec![
            row(&[("uid", json!(1)), ("name", json!("alice"))]),
            row(&[("uid", json!(2)), ("name", json!("bob"))]),
        ];
        let orders = vec![
            row(&[("uid", json!(1)), ("amount", json!(10))]),
            row(&[("uid", json!(1)), ("amount", json!(5))]),
            row(&[("uid", json!(2)), ("amount", json!(7))]),
        ];
        let joined = distributed_join(&users, &orders, &JoinKeys::on("uid"), 3);
        let q = Query::new("j").agg(Agg::Sum("amount".into())).group("name");
        let out = q.map_shard(&joined).finalize();
        let mut by: BTreeMap<String, f64> = BTreeMap::new();
        for g in out {
            by.insert(g.key[0].clone(), g.values["sum_amount"].as_f64().unwrap());
        }
        assert_eq!(by["alice"], 15.0);
        assert_eq!(by["bob"], 7.0);
    }

    // Property: the co-partitioned distributed join equals the single-pass hash join for any number
    // of partitions — co-partitioning never changes the answer, only how the work is split.
    proptest! {
        #[test]
        fn distributed_equals_hashjoin(
            lkeys in prop::collection::vec(0i64..8, 0..20),
            rkeys in prop::collection::vec(0i64..8, 0..20),
            parts in 1usize..7,
        ) {
            let left: Vec<Row> = lkeys.iter().enumerate()
                .map(|(i, &k)| row(&[("k", json!(k)), ("li", json!(i as i64))]))
                .collect();
            let right: Vec<Row> = rkeys.iter().enumerate()
                .map(|(i, &k)| row(&[("k", json!(k)), ("ri", json!(i as i64))]))
                .collect();
            let one = hash_join(&left, &right, &JoinKeys::on("k"));
            let dist = distributed_join(&left, &right, &JoinKeys::on("k"), parts);
            prop_assert_eq!(sig(&one), sig(&dist));
        }
    }

    // Property: an inner equi-join produces exactly sum over keys of (count_left(k) * count_right(k))
    // rows — the textbook cardinality, which guards against dropped or duplicated matches.
    proptest! {
        #[test]
        fn join_cardinality(
            lkeys in prop::collection::vec(0i64..5, 0..25),
            rkeys in prop::collection::vec(0i64..5, 0..25),
        ) {
            let left: Vec<Row> = lkeys.iter().map(|&k| row(&[("k", json!(k))])).collect();
            let right: Vec<Row> = rkeys.iter().map(|&k| row(&[("k", json!(k))])).collect();
            let out = hash_join(&left, &right, &JoinKeys::on("k"));

            let mut lc: HashMap<i64, usize> = HashMap::new();
            for &k in &lkeys { *lc.entry(k).or_insert(0) += 1; }
            let mut rc: HashMap<i64, usize> = HashMap::new();
            for &k in &rkeys { *rc.entry(k).or_insert(0) += 1; }
            let expected: usize = lc.iter().map(|(k, &c)| c * rc.get(k).copied().unwrap_or(0)).sum();

            prop_assert_eq!(out.len(), expected);
        }
    }
}
