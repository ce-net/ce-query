//! End-to-end integration tests for `ce-query` that drive the **public** API (lib surface) the way
//! the CLI does — parse SQL, shard rows, plan, and run map-reduce / projection — but against an
//! in-memory [`MapHost`] so no running CE node is required. These tests fail if any public seam
//! regresses (parser -> query -> engine -> reduce -> shape), complementing the unit tests inside each
//! module.

use async_trait::async_trait;
use ce_query::dataset::{compute_stats, encode_shard, shard_rows};
use ce_query::engine::{MapError, MapHost, RunConfig, run, run_projection};
use ce_query::{Dataset, Query, Row, Shard, sql};
use serde_json::json;
use std::collections::HashMap;

/// A fully in-memory host: shards live in a `cid -> rows` map; `map`/`project` just run the pure
/// query logic. This is exactly what a real host does after fetching the shard blob, minus the
/// network — so a passing test here proves the whole compute path end-to-end.
struct MemHost {
    store: HashMap<String, Vec<Row>>,
}

#[async_trait]
impl MapHost for MemHost {
    async fn map(
        &self,
        _host: &str,
        shard: &Shard,
        query: &Query,
    ) -> Result<ce_query::Partial, MapError> {
        let rows = self.store.get(&shard.cid).ok_or_else(|| MapError::MissingBlob(shard.cid.clone()))?;
        Ok(query.map_shard(rows))
    }
    async fn project(&self, _host: &str, shard: &Shard, query: &Query) -> Result<Vec<Row>, MapError> {
        let rows = self.store.get(&shard.cid).ok_or_else(|| MapError::MissingBlob(shard.cid.clone()))?;
        Ok(query.project_shard(rows))
    }
}

/// Build a sharded in-memory dataset (with stats) from a flat row list.
fn build(rows: &[Row], n: usize) -> (Dataset, MemHost) {
    let mut ds = Dataset::new("sales", vec![]);
    let mut store = HashMap::new();
    for (rs, bytes) in shard_rows(rows, n).unwrap() {
        if rs.is_empty() {
            continue;
        }
        let cid = ce_rs::cid(&bytes);
        ds.add_shard_with_stats(cid.clone(), rs.len() as u64, bytes.len() as u64, compute_stats(&rs));
        store.insert(cid, rs);
    }
    (ds, MemHost { store })
}

fn sales_rows() -> Vec<Row> {
    // 6 rows across two regions with integer amounts.
    [
        ("EU", 100),
        ("US", 200),
        ("EU", 50),
        ("US", 75),
        ("EU", 25),
        ("US", 25),
    ]
    .into_iter()
    .map(|(r, a)| {
        [("region".to_string(), json!(r)), ("amount".to_string(), json!(a))]
            .into_iter()
            .collect()
    })
    .collect()
}

fn hosts(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("h{i}")).collect()
}

#[tokio::test]
async fn sql_aggregate_group_by_end_to_end() {
    let (ds, host) = build(&sales_rows(), 3);
    let q = sql::parse("SELECT SUM(amount), COUNT(*) FROM sales GROUP BY region ORDER BY sum_amount DESC")
        .unwrap();
    let report = run(&q, &ds.shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
    // EU = 100+50+25 = 175 (3 rows); US = 200+75+25 = 300 (3 rows). Ordered by sum desc -> US first.
    assert_eq!(report.results.len(), 2);
    assert_eq!(report.results[0].key[0], "US");
    assert_eq!(report.results[0].values["sum_amount"], json!(300)); // exact integer lane
    assert_eq!(report.results[0].values["count"], json!(3));
    assert_eq!(report.results[1].key[0], "EU");
    assert_eq!(report.results[1].values["sum_amount"], json!(175));
}

#[tokio::test]
async fn sql_projection_with_where_and_limit_end_to_end() {
    let (ds, host) = build(&sales_rows(), 4);
    let q = sql::parse("SELECT region, amount FROM sales WHERE amount >= 75 ORDER BY amount DESC LIMIT 2")
        .unwrap();
    let report = run_projection(&q, &ds.shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
    // amounts >= 75 are 200, 100, 75 -> top 2 by amount desc = 200, 100.
    let amounts: Vec<i64> = report.rows.iter().map(|r| r["amount"].as_i64().unwrap()).collect();
    assert_eq!(amounts, vec![200, 100]);
    // Only projected columns survive.
    assert!(report.rows[0].contains_key("region"));
    assert!(report.rows[0].contains_key("amount"));
    assert_eq!(report.rows[0].len(), 2);
    assert!(report.truncated, "3 matched, limit 2 -> truncated");
}

#[tokio::test]
async fn sharding_count_does_not_change_the_answer() {
    let rows = sales_rows();
    let q = sql::parse("SELECT SUM(amount) FROM sales").unwrap();
    let mut last: Option<serde_json::Value> = None;
    for n in 1..=6 {
        let (ds, host) = build(&rows, n);
        let report = run(&q, &ds.shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
        let v = report.results[0].values["sum_amount"].clone();
        if let Some(prev) = &last {
            assert_eq!(&v, prev, "answer changed with shard count {n}");
        }
        last = Some(v);
    }
    assert_eq!(last, Some(json!(475))); // 100+200+50+75+25+25
}

#[tokio::test]
async fn encode_decode_shard_is_the_wire_contract() {
    // The bytes a host fetches must round-trip to the same rows the coordinator sharded.
    let rows = sales_rows();
    let bytes = encode_shard(&rows).unwrap();
    let back = ce_query::dataset::decode_shard(&bytes).unwrap();
    assert_eq!(back, rows);
}
