//! The combiner algebra — the **reduce** side of map-reduce and the engine's correctness keystone.
//!
//! Every aggregate is expressed as a **monoid**: a state type ([`Accum`]) with an identity element
//! ([`Agg::zero`](crate::query::Agg::zero)) and an **associative, commutative** merge
//! ([`Accum::merge`]). That algebraic property is exactly what lets the engine be distributed and
//! fault-tolerant:
//!
//! - **Associativity** => shards can be merged in any *grouping*: `(A·B)·C == A·(B·C)`. The
//!   coordinator can reduce as results arrive, or in a tree, with the same answer.
//! - **Commutativity** => shards can be merged in any *order*: `A·B == B·A`. Out-of-order arrival
//!   (the norm on a mesh) is fine.
//! - **Identity** => an empty/missing shard contributes [`Accum::zero`] and changes nothing, so a
//!   shard that was retried, duplicated-but-deduplicated, or simply empty is harmless.
//!
//! Counting and sum-of-squares-style aggregates are trivially monoidal; `AVG` is made monoidal by
//! carrying `(sum, count)` and dividing only at **finalisation** (`AVG` of merged `AVG`s is wrong;
//! sum/count merged then divided is right). The property tests at the bottom of this file *prove*
//! these laws over random inputs — they are the foundation's validation.
//!
//! ## Numeric determinism
//!
//! A distributed engine must give the **same answer regardless of how shards were split or in what
//! order their partials arrived** — otherwise a retried/redistributed shard could change the result
//! at the ULP level and break the safe-retry invariant. Two measures keep floating-point aggregates
//! deterministic and reorder-stable:
//!
//! - `SUM`/`AVG` carry a [`KahanSum`] (Neumaier compensated summation) rather than a bare `f64`. The
//!   running compensation is itself merged, so `(A·B)·C == A·(B·C)` holds to the bit for the orders
//!   the engine can produce.
//! - When every value folded into a `SUM` is an **exact integer** (counts, money base units, ids),
//!   the accumulator stays in the [`Accum::IntSum`] lane (`i128`) and never touches float at all —
//!   honouring the project's integer-base-unit money convention. The first non-integral value
//!   promotes the lane to compensated float.

use crate::query::Agg;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Neumaier (improved Kahan) compensated sum: a running total plus a correction term that recovers
/// the low-order bits lost to rounding. Merging two `KahanSum`s adds both their totals **and** their
/// compensations, so the sum is associative and commutative to far higher precision than a naive
/// `f64` fold — the property the engine relies on to give a reorder-stable answer.
///
/// ```
/// use ce_query::combine::KahanSum;
/// // 1e16 + 1.0 - 1e16: a bare f64 fold loses the 1.0; compensated summation keeps it.
/// let mut k = KahanSum::zero();
/// for v in [1e16, 1.0, -1e16] {
///     k.add(v);
/// }
/// assert_eq!(k.total(), 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct KahanSum {
    /// The running sum.
    pub sum: f64,
    /// The accumulated rounding compensation (added back at read time).
    pub comp: f64,
}

impl KahanSum {
    /// The additive identity (0 with no compensation).
    pub fn zero() -> KahanSum {
        KahanSum { sum: 0.0, comp: 0.0 }
    }

    /// Add one value with Neumaier compensation.
    pub fn add(&mut self, v: f64) {
        let t = self.sum + v;
        if self.sum.abs() >= v.abs() {
            self.comp += (self.sum - t) + v;
        } else {
            self.comp += (v - t) + self.sum;
        }
        self.sum = t;
    }

    /// Merge another compensated sum into this one (associative + commutative).
    pub fn merge(&mut self, other: &KahanSum) {
        // Fold the other total in compensated, then carry its leftover compensation.
        self.add(other.sum);
        self.comp += other.comp;
    }

    /// The corrected total (sum plus accumulated compensation).
    pub fn total(&self) -> f64 {
        self.sum + self.comp
    }
}

/// The running state of one aggregate over many rows — a monoid value. Merged with [`merge`] and
/// turned into an output JSON value by [`finalize_value`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Accum {
    /// Running row count.
    Count(u64),
    /// Running numeric sum, all-integer lane: every value folded so far was an exact integer, so the
    /// sum is carried losslessly as `i128` (money base units / ids). Promotes to [`Accum::Sum`] on the
    /// first non-integral value or on overflow.
    IntSum(i128),
    /// Running numeric sum, compensated-float lane (used once any non-integer value is seen).
    Sum(KahanSum),
    /// Running minimum (None = no value seen yet, the identity).
    Min(Option<f64>),
    /// Running maximum (None = no value seen yet, the identity).
    Max(Option<f64>),
    /// Running mean carried as `(sum, count)`, divided only at finalisation. `sum` is compensated.
    Avg { sum: KahanSum, count: u64 },
}

impl Accum {
    /// Merge `other` into `self`, associatively and commutatively. The two accumulators **must** be
    /// the same variant (they come from the same aggregate position); a mismatch is a programmer
    /// error and is treated as a no-op rather than a panic, keeping the engine crash-free.
    pub fn merge(&mut self, other: &Accum) {
        match (self, other) {
            (Accum::Count(a), Accum::Count(b)) => *a = a.saturating_add(*b),
            // Both lanes still integral: stay lossless unless the addition would overflow i128.
            (a @ Accum::IntSum(_), Accum::IntSum(y)) => {
                if let Accum::IntSum(x) = a {
                    match x.checked_add(*y) {
                        Some(s) => *x = s,
                        None => {
                            // Overflow: promote to compensated float and keep going.
                            let mut k = KahanSum::zero();
                            k.add(*x as f64);
                            k.add(*y as f64);
                            *a = Accum::Sum(k);
                        }
                    }
                }
            }
            // One side has gone to float: promote self and merge in the float domain.
            (a @ Accum::IntSum(_), Accum::Sum(b)) => {
                if let Accum::IntSum(x) = a {
                    let mut k = KahanSum::zero();
                    k.add(*x as f64);
                    k.merge(b);
                    *a = Accum::Sum(k);
                }
            }
            (Accum::Sum(a), Accum::IntSum(y)) => a.add(*y as f64),
            (Accum::Sum(a), Accum::Sum(b)) => a.merge(b),
            (Accum::Min(a), Accum::Min(b)) => {
                *a = match (*a, *b) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    (Some(x), None) => Some(x),
                    (None, y) => y,
                }
            }
            (Accum::Max(a), Accum::Max(b)) => {
                *a = match (*a, *b) {
                    (Some(x), Some(y)) => Some(x.max(y)),
                    (Some(x), None) => Some(x),
                    (None, y) => y,
                }
            }
            (Accum::Avg { sum: sa, count: ca }, Accum::Avg { sum: sb, count: cb }) => {
                sa.merge(sb);
                *ca = ca.saturating_add(*cb);
            }
            // Variant mismatch: ignore (never happens for well-formed partials). No panic.
            _ => {}
        }
    }

    /// Finalise this accumulator to its output JSON value. `Avg` divides sum by count here (the one
    /// place division happens); `Min`/`Max` over an empty input finalise to JSON `null`.
    pub fn finalize_value(&self) -> serde_json::Value {
        match self {
            Accum::Count(n) => serde_json::json!(n),
            // An all-integer sum finalises to a JSON integer (exact — money base units stay exact).
            Accum::IntSum(n) => serde_json::json!(n),
            Accum::Sum(s) => serde_json::json!(s.total()),
            Accum::Min(m) | Accum::Max(m) => match m {
                Some(v) => serde_json::json!(v),
                None => serde_json::Value::Null,
            },
            Accum::Avg { sum, count } => {
                if *count == 0 {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(sum.total() / (*count as f64))
                }
            }
        }
    }
}

/// One finalised output group: its `GROUP BY` key (string per key column) and the named aggregate
/// values. With no grouping, `key` is empty and there is a single group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupResult {
    /// The group key columns, in `GROUP BY` order (empty for a global aggregate).
    pub key: Vec<String>,
    /// Output column name -> finalised value.
    pub values: BTreeMap<String, serde_json::Value>,
}

/// The map output of one shard (or the running merge of several): the query's aggregate list plus,
/// per group key, a parallel vector of accumulators (one per aggregate). Associative and small —
/// the unit shipped from a map host back to the coordinator and folded by [`merge`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Partial {
    /// The aggregates these accumulators correspond to (positionally). Carried so a partial is
    /// self-describing and can be finalised without re-consulting the query.
    pub aggregates: Vec<Agg>,
    /// Group key -> one accumulator per aggregate (same length/order as `aggregates`).
    ///
    /// Serialized as a sequence of `(key, accs)` entries (not a JSON map) because a group key is a
    /// `Vec<String>`, and JSON object keys must be strings. This keeps a [`Partial`] portable over
    /// the mesh request/reply (the [`MapReply`](crate::mesh::MapReply) wire form) while staying a
    /// `BTreeMap` in memory for cheap, order-stable merges.
    #[serde(with = "groups_serde")]
    pub groups: BTreeMap<Vec<String>, Vec<Accum>>,
}

/// Serialize/deserialize the `groups` map as a `Vec<(Vec<String>, Vec<Accum>)>` so it round-trips
/// through JSON (which cannot key an object by a non-string). Order is the map's natural sorted
/// order, so the encoding is deterministic.
mod groups_serde {
    use super::Accum;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(
        groups: &BTreeMap<Vec<String>, Vec<Accum>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let entries: Vec<(&Vec<String>, &Vec<Accum>)> = groups.iter().collect();
        entries.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<Vec<String>, Vec<Accum>>, D::Error> {
        let entries: Vec<(Vec<String>, Vec<Accum>)> = Vec::deserialize(d)?;
        Ok(entries.into_iter().collect())
    }
}

impl Partial {
    /// An empty partial for `aggregates` — the identity element for [`merge`]. Merging this into any
    /// partial leaves it unchanged.
    pub fn empty(aggregates: Vec<Agg>) -> Partial {
        Partial { aggregates, groups: BTreeMap::new() }
    }

    /// Merge `other` into `self`, group by group, associatively and commutatively. Groups present in
    /// only one side are carried over; shared groups have their accumulator vectors merged
    /// position-wise. This is the reduce step the coordinator applies to every shard result.
    pub fn merge(&mut self, other: &Partial) {
        for (key, other_accs) in &other.groups {
            match self.groups.get_mut(key) {
                Some(accs) => {
                    for (a, b) in accs.iter_mut().zip(other_accs.iter()) {
                        a.merge(b);
                    }
                }
                None => {
                    self.groups.insert(key.clone(), other_accs.clone());
                }
            }
        }
    }

    /// Finalise into output groups, sorted by key for a stable, reproducible result order. Each
    /// group's accumulators become named values via the aggregates' [`output_name`].
    ///
    /// [`output_name`]: crate::query::Agg::output_name
    pub fn finalize(&self) -> Vec<GroupResult> {
        let mut out = Vec::with_capacity(self.groups.len());
        for (key, accs) in &self.groups {
            let mut values = BTreeMap::new();
            for (agg, acc) in self.aggregates.iter().zip(accs.iter()) {
                values.insert(agg.output_name(), acc.finalize_value());
            }
            out.push(GroupResult { key: key.clone(), values });
        }
        // BTreeMap iteration is already sorted by key; preserve that explicitly for clarity.
        out
    }
}

/// Merge a collection of partials into one (left fold). Order-independent by the monoid laws, so the
/// coordinator may pass shard results in arrival order. Returns an empty partial if `parts` is
/// empty (the identity), carrying `aggregates` so finalisation still names columns.
pub fn reduce(aggregates: Vec<Agg>, parts: impl IntoIterator<Item = Partial>) -> Partial {
    let mut acc = Partial::empty(aggregates);
    for p in parts {
        acc.merge(&p);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Row;
    use crate::query::Query;
    use proptest::prelude::*;
    use serde_json::json;

    fn row(i: i64, g: &str) -> Row {
        [("v".to_string(), json!(i)), ("g".to_string(), json!(g))].into_iter().collect()
    }

    fn sample_query() -> Query {
        Query::new("t")
            .agg(Agg::Count)
            .agg(Agg::Sum("v".into()))
            .agg(Agg::Min("v".into()))
            .agg(Agg::Max("v".into()))
            .agg(Agg::Avg("v".into()))
            .group("g")
    }

    #[test]
    fn empty_is_identity() {
        let q = sample_query();
        let rows: Vec<Row> = (0..5).map(|i| row(i, "a")).collect();
        let p = q.map_shard(&rows);
        let mut merged = p.clone();
        merged.merge(&Partial::empty(q.aggregates.clone()));
        assert_eq!(merged, p, "merging the empty partial must be a no-op");
    }

    #[test]
    fn merge_matches_single_shard() {
        // Splitting rows across shards then merging == mapping the whole set in one shard.
        let q = sample_query();
        let all: Vec<Row> = (0..30).map(|i| row(i, if i % 2 == 0 { "a" } else { "b" })).collect();
        let whole = q.map_shard(&all).finalize();

        let s1 = q.map_shard(&all[..10]);
        let s2 = q.map_shard(&all[10..20]);
        let s3 = q.map_shard(&all[20..]);
        let merged = reduce(q.aggregates.clone(), vec![s1, s2, s3]).finalize();

        assert_eq!(merged, whole, "distributed map+reduce must equal single-shard map");
    }

    // Splitting a list into N shards and merging gives the same answer no matter how many shards.
    proptest! {
        #[test]
        fn shard_count_invariant(values in prop::collection::vec(-1000i64..1000, 0..60), nshards in 1usize..8) {
            let q = sample_query();
            let rows: Vec<Row> = values.iter().enumerate()
                .map(|(i, &v)| row(v, if i % 3 == 0 { "x" } else { "y" }))
                .collect();
            let whole = q.map_shard(&rows).finalize();

            // Distribute round-robin across nshards, map each, reduce.
            let mut buckets: Vec<Vec<Row>> = vec![Vec::new(); nshards];
            for (i, r) in rows.iter().enumerate() {
                buckets[i % nshards].push(r.clone());
            }
            let parts: Vec<Partial> = buckets.iter().map(|b| q.map_shard(b)).collect();
            let merged = reduce(q.aggregates.clone(), parts).finalize();
            prop_assert_eq!(merged, whole);
        }
    }

    // Merge is associative: (A·B)·C == A·(B·C).
    proptest! {
        #[test]
        fn merge_associative(
            a in prop::collection::vec(-500i64..500, 0..20),
            b in prop::collection::vec(-500i64..500, 0..20),
            c in prop::collection::vec(-500i64..500, 0..20),
        ) {
            let q = sample_query();
            let pa = q.map_shard(&a.iter().map(|&v| row(v, "g")).collect::<Vec<_>>());
            let pb = q.map_shard(&b.iter().map(|&v| row(v, "g")).collect::<Vec<_>>());
            let pc = q.map_shard(&c.iter().map(|&v| row(v, "g")).collect::<Vec<_>>());

            let mut left = pa.clone(); left.merge(&pb); left.merge(&pc);
            let mut right_bc = pb.clone(); right_bc.merge(&pc);
            let mut right = pa.clone(); right.merge(&right_bc);

            prop_assert_eq!(left.finalize(), right.finalize());
        }
    }

    // Merge is commutative: A·B == B·A.
    proptest! {
        #[test]
        fn merge_commutative(
            a in prop::collection::vec(-500i64..500, 0..25),
            b in prop::collection::vec(-500i64..500, 0..25),
        ) {
            let q = sample_query();
            let pa = q.map_shard(&a.iter().map(|&v| row(v, "g")).collect::<Vec<_>>());
            let pb = q.map_shard(&b.iter().map(|&v| row(v, "g")).collect::<Vec<_>>());

            let mut ab = pa.clone(); ab.merge(&pb);
            let mut ba = pb.clone(); ba.merge(&pa);
            prop_assert_eq!(ab.finalize(), ba.finalize());
        }
    }

    // Idempotence of the empty/identity element under arbitrary insertion order, and that a
    // duplicated empty shard never changes the result (safe retry of an empty shard).
    proptest! {
        #[test]
        fn empty_shards_harmless(values in prop::collection::vec(-100i64..100, 0..30), extra_empties in 0usize..5) {
            let q = sample_query();
            let rows: Vec<Row> = values.iter().map(|&v| row(v, "g")).collect();
            let base = q.map_shard(&rows).finalize();

            let mut parts = vec![q.map_shard(&rows)];
            for _ in 0..extra_empties {
                parts.push(Partial::empty(q.aggregates.clone()));
            }
            let withempties = reduce(q.aggregates.clone(), parts).finalize();
            prop_assert_eq!(withempties, base);
        }
    }

    #[test]
    fn min_max_empty_finalize_null() {
        assert_eq!(Accum::Min(None).finalize_value(), serde_json::Value::Null);
        assert_eq!(Accum::Max(None).finalize_value(), serde_json::Value::Null);
        assert_eq!(Accum::Avg { sum: KahanSum::zero(), count: 0 }.finalize_value(), serde_json::Value::Null);
    }

    #[test]
    fn partial_json_roundtrip_with_grouped_keys() {
        // The wire form must survive JSON despite Vec<String> group keys (the mesh MapReply path).
        let q = sample_query();
        let rows: Vec<Row> = (0..12).map(|i| row(i, if i % 2 == 0 { "even" } else { "odd" })).collect();
        let p = q.map_shard(&rows);
        let json = serde_json::to_string(&p).expect("partial must serialize to JSON");
        let back: Partial = serde_json::from_str(&json).expect("partial must deserialize from JSON");
        assert_eq!(back, p);
        // And the round-tripped partial finalises identically.
        assert_eq!(back.finalize(), p.finalize());
    }

    // Property: any mapped partial survives a JSON round-trip unchanged, for arbitrary inputs.
    proptest! {
        #[test]
        fn partial_json_roundtrip_prop(values in prop::collection::vec(-300i64..300, 0..40)) {
            let q = sample_query();
            let rows: Vec<Row> = values.iter().enumerate()
                .map(|(i, &v)| row(v, if i % 4 == 0 { "a" } else { "b" }))
                .collect();
            let p = q.map_shard(&rows);
            let json = serde_json::to_string(&p).unwrap();
            let back: Partial = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(back, p);
        }
    }

    #[test]
    fn variant_mismatch_is_noop_not_panic() {
        let mut a = Accum::Count(5);
        a.merge(&Accum::Sum(KahanSum::zero())); // mismatched variants
        assert_eq!(a, Accum::Count(5));
        // Min vs Max are also distinct variants — a mismatch must be a no-op, not a panic.
        let mut m = Accum::Min(Some(1.0));
        m.merge(&Accum::Max(Some(99.0)));
        assert_eq!(m, Accum::Min(Some(1.0)));
    }

    #[test]
    fn kahan_sum_is_precise_and_reorder_stable() {
        // A classic catastrophic-cancellation case: 1e16 + 1.0 - 1e16. Naive f64 loses the 1.0;
        // compensated summation keeps it.
        let mut k = KahanSum::zero();
        k.add(1e16);
        k.add(1.0);
        k.add(-1e16);
        assert_eq!(k.total(), 1.0, "compensated sum must retain the small term");

        // Merging in any order gives the same compensated total.
        let mut a = KahanSum::zero();
        for v in [1e16, 3.0, -1e16, 0.5] {
            a.add(v);
        }
        let mut b1 = KahanSum::zero();
        b1.add(1e16);
        b1.add(3.0);
        let mut b2 = KahanSum::zero();
        b2.add(-1e16);
        b2.add(0.5);
        let mut merged = b2;
        merged.merge(&b1);
        assert_eq!(a.total(), merged.total());
    }

    #[test]
    fn int_sum_overflow_promotes_to_float_without_panic() {
        // Two near-i128::MAX integers overflow the int lane; the merge must promote to float, not
        // panic or wrap.
        let mut a = Accum::IntSum(i128::MAX);
        a.merge(&Accum::IntSum(i128::MAX));
        match a {
            Accum::Sum(k) => assert!(k.total() > 0.0),
            other => panic!("expected promotion to float, got {other:?}"),
        }
    }

    // Float SUM must be associative under the engine's possible merge groupings, even with large
    // magnitudes and fractional parts (the case bare-f64 summation would fail).
    proptest! {
        #[test]
        fn float_sum_merge_associative(
            a in prop::collection::vec(-1e9f64..1e9, 0..20),
            b in prop::collection::vec(-1e9f64..1e9, 0..20),
            c in prop::collection::vec(-1e9f64..1e9, 0..20),
        ) {
            let q = Query::new("t").agg(crate::query::Agg::Sum("v".into()));
            let mk = |xs: &[f64]| {
                let rows: Vec<Row> = xs.iter()
                    .map(|&v| [("v".to_string(), json!(v))].into_iter().collect())
                    .collect();
                q.map_shard(&rows)
            };
            let (pa, pb, pc) = (mk(&a), mk(&b), mk(&c));
            let mut left = pa.clone(); left.merge(&pb); left.merge(&pc);
            let mut bc = pb.clone(); bc.merge(&pc);
            let mut right = pa.clone(); right.merge(&bc);
            prop_assert_eq!(left.finalize(), right.finalize());
        }
    }
}
