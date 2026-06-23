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

/// A single shard's descriptor: the blob CID plus counts used by the planner to balance work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shard {
    /// Content id (hex sha256 manifest CID) of the blob holding this shard's NDJSON rows.
    pub cid: String,
    /// Number of rows in the shard (0 if unknown — the engine still processes it).
    #[serde(default)]
    pub rows: u64,
    /// Encoded byte length of the shard (0 if unknown).
    #[serde(default)]
    pub bytes: u64,
}

/// A dataset: name + schema + ordered shard list. Pure metadata; rows live in the referenced blobs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Append a shard (by CID + counts). Returns the new shard index.
    pub fn add_shard(&mut self, cid: impl Into<String>, rows: u64, bytes: u64) -> usize {
        self.shards.push(Shard { cid: cid.into(), rows, bytes });
        self.shards.len() - 1
    }
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
    fn dataset_totals() {
        let mut d = Dataset::new("t", vec!["a".into()]);
        d.add_shard("cid1", 100, 1000);
        d.add_shard("cid2", 50, 500);
        assert_eq!(d.shards.len(), 2);
        assert_eq!(d.total_rows(), 150);
        assert_eq!(d.total_bytes(), 1500);
    }
}
