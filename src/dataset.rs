//! Datasets — a named, sharded collection of content-addressed blobs.
//!
//! A **dataset** is a logical table. Its rows live in **shards**: each shard is one blob (a CID),
//! holding a chunk of newline-delimited JSON records (NDJSON). Sharding is how the query engine
//! parallelises — every shard can be mapped on a different host, independently, because each shard
//! is self-contained content-addressed bytes.
//!
//! The dataset itself is just **metadata**: a name, an optional column schema, and the ordered list
//! of shard CIDs (plus per-shard row/byte counts for planning). This metadata is small and lives in
//! a local JSON catalog (the single-writer owner's view); a published dataset could equally be
//! stored as its own blob and shared by CID. Nothing here moves bytes — shards are fetched lazily
//! at query time via [`ce_rs::CeClient::get_object`].
//!
//! ## Why NDJSON
//!
//! NDJSON keeps the foundation honest and testable without a columnar dependency: a shard is a
//! `Vec<Row>` where `Row = serde_json::Map`. The engine's algebra (filter, project, aggregate) is
//! defined over `serde_json::Value`, so the columnar (Arrow/Parquet) upgrade later is a shard
//! *encoding* swap, not a query-semantics change. Real BigQuery-scale would store columnar blobs;
//! the planner, combiner, and SQL layers are encoding-agnostic by design.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One record in a dataset: an ordered string-keyed JSON object. Ordered (`BTreeMap`) so a row's
/// serialization is deterministic — its bytes (and therefore a shard's CID) are stable across runs.
pub type Row = BTreeMap<String, serde_json::Value>;

/// Per-column min/max statistics for a shard, used for **partition pruning**: if a query's `WHERE`
/// range cannot overlap a column's `[min, max]` in a shard, the planner can skip that shard entirely
/// without fetching it — BigQuery's core efficiency story (scan only the partitions that can match).
/// Stats are recorded over numeric columns only (the comparable lane); a column with any
/// non-numeric value in the shard is omitted (no stat = the shard is never pruned on it, which is
/// safe — pruning only ever *removes* provably-non-matching shards).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ShardStats {
    /// Column name -> (min, max) of its numeric values within the shard.
    #[serde(default)]
    pub numeric: BTreeMap<String, (f64, f64)>,
}

/// A single shard's descriptor: the blob CID plus counts used by the planner to balance work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shard {
    /// Content id (hex sha256 manifest CID) of the blob holding this shard's NDJSON rows.
    pub cid: String,
    /// Number of rows in the shard (0 if unknown — the engine still processes it).
    #[serde(default)]
    pub rows: u64,
    /// Encoded byte length of the shard (0 if unknown).
    #[serde(default)]
    pub bytes: u64,
    /// Optional per-column min/max stats for partition pruning (`None` = no stats, never pruned).
    #[serde(default)]
    pub stats: Option<ShardStats>,
}

impl Shard {
    /// A shard descriptor with the given CID and counts and no stats.
    pub fn new(cid: impl Into<String>, rows: u64, bytes: u64) -> Shard {
        Shard { cid: cid.into(), rows, bytes, stats: None }
    }
}

/// A dataset: name + schema + ordered shard list. Pure metadata; rows live in the referenced blobs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dataset {
    /// Logical table name (unique within a catalog).
    pub name: String,
    /// Declared column names, if any. Informational — NDJSON rows may carry extra/missing keys; the
    /// engine reads whatever a row actually has. An empty schema means "schema-on-read".
    #[serde(default)]
    pub schema: Vec<String>,
    /// Ordered shards. Order is preserved for reproducibility but the engine does not rely on it.
    #[serde(default)]
    pub shards: Vec<Shard>,
}

impl Dataset {
    /// Create an empty, unsharded dataset with the given name and (optional) schema.
    pub fn new(name: impl Into<String>, schema: Vec<String>) -> Dataset {
        Dataset { name: name.into(), schema, shards: Vec::new() }
    }

    /// Total declared row count across all shards.
    pub fn total_rows(&self) -> u64 {
        self.shards.iter().map(|s| s.rows).sum()
    }

    /// Total declared byte size across all shards.
    pub fn total_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.bytes).sum()
    }

    /// Append a shard (by CID + counts) with no stats. Returns the new shard index.
    pub fn add_shard(&mut self, cid: impl Into<String>, rows: u64, bytes: u64) -> usize {
        self.shards.push(Shard { cid: cid.into(), rows, bytes, stats: None });
        self.shards.len() - 1
    }

    /// Append a shard with computed min/max stats. Returns the new shard index.
    pub fn add_shard_with_stats(
        &mut self,
        cid: impl Into<String>,
        rows: u64,
        bytes: u64,
        stats: ShardStats,
    ) -> usize {
        self.shards.push(Shard { cid: cid.into(), rows, bytes, stats: Some(stats) });
        self.shards.len() - 1
    }
}

/// Compute per-column numeric min/max statistics for a shard's rows. A column is recorded only if
/// **every** value present for it across the rows is numeric (mixed-type columns are skipped so a
/// pruning decision is never made on an incomparable column). Used to populate [`Shard::stats`] at
/// registration time for partition pruning.
pub fn compute_stats(rows: &[Row]) -> ShardStats {
    use crate::query::as_f64;
    let mut numeric: BTreeMap<String, (f64, f64)> = BTreeMap::new();
    let mut disqualified: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for row in rows {
        for (col, val) in row {
            if disqualified.contains(col) {
                continue;
            }
            match as_f64(val) {
                Some(v) => {
                    let e = numeric.entry(col.clone()).or_insert((v, v));
                    e.0 = e.0.min(v);
                    e.1 = e.1.max(v);
                }
                None => {
                    // A non-numeric value in this column disqualifies it from numeric stats.
                    numeric.remove(col);
                    disqualified.insert(col.clone());
                }
            }
        }
    }
    ShardStats { numeric }
}

/// The maximum length of a dataset name, and the allowed charset. Names key the catalog map and the
/// capability `path_prefix` scope, so they are restricted to a safe, filesystem/identifier-friendly
/// set: ASCII alphanumerics plus `_ - . :`. This rejects whitespace, path separators, and control
/// characters that could confuse scope matching or catalog persistence.
pub const MAX_DATASET_NAME_LEN: usize = 128;

/// Validate a dataset name: non-empty, within [`MAX_DATASET_NAME_LEN`], and drawn only from
/// `[A-Za-z0-9_.:-]`. Returns the trimmed name on success or a clear error.
pub fn validate_dataset_name(name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        bail!("dataset name is empty");
    }
    if name.len() > MAX_DATASET_NAME_LEN {
        bail!("dataset name is {} chars, exceeds the {MAX_DATASET_NAME_LEN}-char limit", name.len());
    }
    if let Some(bad) = name.chars().find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))) {
        bail!("dataset name contains an invalid character `{bad}` (allowed: A-Z a-z 0-9 _ - . :)");
    }
    Ok(name.to_string())
}

/// Encode a slice of rows as NDJSON bytes — one compact JSON object per line. This is the canonical
/// shard wire format; [`decode_shard`] is its exact inverse. Empty input yields empty bytes.
pub fn encode_shard(rows: &[Row]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for row in rows {
        let line = serde_json::to_vec(row).context("serializing row to JSON")?;
        out.extend_from_slice(&line);
        out.push(b'\n');
    }
    Ok(out)
}

/// Decode NDJSON shard bytes back into rows. Blank lines are skipped (tolerant of a trailing
/// newline). A line that is not a JSON **object** is a hard error — a shard must hold records, and
/// silently dropping malformed input would corrupt aggregates. Never panics on bad input.
pub fn decode_shard(bytes: &[u8]) -> Result<Vec<Row>> {
    let text = std::str::from_utf8(bytes).context("shard is not valid UTF-8")?;
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .with_context(|| format!("malformed JSON on shard line {}", i + 1))?;
        match value {
            serde_json::Value::Object(map) => rows.push(map.into_iter().collect()),
            other => bail!("shard line {} is not a JSON object: {other}", i + 1),
        }
    }
    Ok(rows)
}

/// Split a flat row stream into `shard_count` shards by round-robin, each encoded to NDJSON bytes.
/// Returns one `(Vec<Row>, encoded_bytes)` per shard. Round-robin (not contiguous) spreads any
/// row-ordering skew evenly so shards are similar in size. `shard_count` must be positive; a count
/// larger than the row count simply yields some empty shards (still valid).
pub fn shard_rows(rows: &[Row], shard_count: usize) -> Result<Vec<(Vec<Row>, Vec<u8>)>> {
    if shard_count == 0 {
        bail!("shard_count must be positive");
    }
    let mut buckets: Vec<Vec<Row>> = vec![Vec::new(); shard_count];
    for (i, row) in rows.iter().enumerate() {
        buckets[i % shard_count].push(row.clone());
    }
    let mut out = Vec::with_capacity(shard_count);
    for bucket in buckets {
        let bytes = encode_shard(&bucket)?;
        out.push((bucket, bytes));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(pairs: &[(&str, serde_json::Value)]) -> Row {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn shard_roundtrip() {
        let rows = vec![
            row(&[("a", json!(1)), ("b", json!("x"))]),
            row(&[("a", json!(2)), ("b", json!("y"))]),
        ];
        let bytes = encode_shard(&rows).unwrap();
        let back = decode_shard(&bytes).unwrap();
        assert_eq!(back, rows);
    }

    #[test]
    fn empty_shard_roundtrips_to_empty() {
        let bytes = encode_shard(&[]).unwrap();
        assert!(bytes.is_empty());
        assert!(decode_shard(&bytes).unwrap().is_empty());
    }

    #[test]
    fn decode_skips_blank_lines_and_trailing_newline() {
        let bytes = b"{\"a\":1}\n\n{\"a\":2}\n".to_vec();
        let rows = decode_shard(&bytes).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn decode_rejects_malformed_json() {
        let err = decode_shard(b"{not json}\n").unwrap_err();
        assert!(err.to_string().contains("malformed JSON"), "{err}");
    }

    #[test]
    fn decode_rejects_non_object_line() {
        let err = decode_shard(b"[1,2,3]\n").unwrap_err();
        assert!(err.to_string().contains("not a JSON object"), "{err}");
    }

    #[test]
    fn decode_rejects_non_utf8() {
        let err = decode_shard(&[0xff, 0xfe, 0x00]).unwrap_err();
        assert!(err.to_string().contains("not valid UTF-8"), "{err}");
    }

    #[test]
    fn shard_rows_round_robin_balances() {
        let rows: Vec<Row> = (0..10).map(|i| row(&[("i", json!(i))])).collect();
        let shards = shard_rows(&rows, 3).unwrap();
        assert_eq!(shards.len(), 3);
        // 10 rows over 3 shards => 4,3,3.
        let counts: Vec<usize> = shards.iter().map(|(r, _)| r.len()).collect();
        assert_eq!(counts, vec![4, 3, 3]);
        // Every encoded shard decodes back to its rows.
        for (r, b) in &shards {
            assert_eq!(&decode_shard(b).unwrap(), r);
        }
        // Union of shards == original (as a multiset of row count).
        let total: usize = shards.iter().map(|(r, _)| r.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn shard_rows_more_shards_than_rows_yields_empties() {
        let rows = vec![row(&[("i", json!(0))])];
        let shards = shard_rows(&rows, 4).unwrap();
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].0.len(), 1);
        assert!(shards[1].0.is_empty());
    }

    #[test]
    fn shard_rows_zero_count_errors() {
        assert!(shard_rows(&[], 0).is_err());
    }

    #[test]
    fn dataset_name_validation() {
        assert_eq!(validate_dataset_name("sales").unwrap(), "sales");
        assert_eq!(validate_dataset_name("  logs_2026.q1 ").unwrap(), "logs_2026.q1");
        assert!(validate_dataset_name("").is_err());
        assert!(validate_dataset_name("   ").is_err());
        assert!(validate_dataset_name("bad name").is_err()); // whitespace
        assert!(validate_dataset_name("../etc/passwd").is_err()); // path traversal
        assert!(validate_dataset_name("a/b").is_err()); // separator
        assert!(validate_dataset_name(&"x".repeat(MAX_DATASET_NAME_LEN + 1)).is_err());
    }

    #[test]
    fn compute_stats_records_numeric_min_max() {
        let rows = vec![
            row(&[("v", json!(5)), ("t", json!("a"))]),
            row(&[("v", json!(1)), ("t", json!("b"))]),
            row(&[("v", json!(9)), ("t", json!("c"))]),
        ];
        let s = compute_stats(&rows);
        assert_eq!(s.numeric.get("v"), Some(&(1.0, 9.0)));
        // `t` is a string column -> no numeric stat.
        assert!(!s.numeric.contains_key("t"));
    }

    #[test]
    fn compute_stats_disqualifies_mixed_type_column() {
        // A column that is numeric in one row and a string in another is omitted (can't prune on it).
        let rows = vec![
            row(&[("v", json!(5))]),
            row(&[("v", json!("oops"))]),
            row(&[("v", json!(9))]),
        ];
        let s = compute_stats(&rows);
        assert!(!s.numeric.contains_key("v"), "mixed-type column must be disqualified");
    }

    #[test]
    fn add_shard_with_stats_roundtrips() {
        let mut d = Dataset::new("t", vec![]);
        let stats = ShardStats { numeric: [("v".to_string(), (0.0, 10.0))].into_iter().collect() };
        d.add_shard_with_stats("cid", 5, 50, stats.clone());
        assert_eq!(d.shards[0].stats, Some(stats));
    }

    #[test]
    fn dataset_totals() {
        let mut d = Dataset::new("t", vec!["a".into()]);
        d.add_shard("cid1", 100, 1000);
        d.add_shard("cid2", 50, 500);
        assert_eq!(d.shards.len(), 2);
        assert_eq!(d.total_rows(), 150);
        assert_eq!(d.total_bytes(), 1500);
    }
}
