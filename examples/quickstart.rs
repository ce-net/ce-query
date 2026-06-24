//! A runnable, node-free quickstart for `ce-query`: load the sample NDJSON dataset
//! (`examples/sales.ndjson`), shard it in memory, and run both an aggregate and a projection query
//! against an in-process [`MapHost`] — no CE node required.
//!
//! Run it with:
//!
//! ```bash
//! cargo run --example quickstart
//! ```
//!
//! This demonstrates the exact compute path the CLI/mesh use (shard -> plan -> map -> reduce/shape),
//! minus the network. To run the same queries over a real mesh, register the dataset with
//! `ce-query dataset add` and use `ce-query run` (see the README).

use async_trait::async_trait;
use ce_query::dataset::{compute_stats, decode_shard, shard_rows};
use ce_query::engine::{MapError, MapHost, RunConfig, run, run_projection};
use ce_query::{Dataset, Partial, Query, Row, Shard, sql};
use std::collections::HashMap;

/// An in-memory host: shards live in a `cid -> rows` map and the query logic runs in-process.
struct MemHost {
    store: HashMap<String, Vec<Row>>,
}

#[async_trait]
impl MapHost for MemHost {
    async fn map(&self, _h: &str, shard: &Shard, q: &Query) -> Result<Partial, MapError> {
        let rows = self.store.get(&shard.cid).ok_or_else(|| MapError::MissingBlob(shard.cid.clone()))?;
        Ok(q.map_shard(rows))
    }
    async fn project(&self, _h: &str, shard: &Shard, q: &Query) -> Result<Vec<Row>, MapError> {
        let rows = self.store.get(&shard.cid).ok_or_else(|| MapError::MissingBlob(shard.cid.clone()))?;
        Ok(q.project_shard(rows))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load and shard the bundled sample dataset (4 shards), populating per-shard stats for pruning.
    let raw = include_bytes!("sales.ndjson");
    let rows = decode_shard(raw)?;
    let mut ds = Dataset::new("sales", vec![]);
    let mut store = HashMap::new();
    for (rs, bytes) in shard_rows(&rows, 4)? {
        if rs.is_empty() {
            continue;
        }
        let cid = ce_rs::cid(&bytes);
        ds.add_shard_with_stats(cid.clone(), rs.len() as u64, bytes.len() as u64, compute_stats(&rs));
        store.insert(cid, rs);
    }
    let host = MemHost { store };
    let hosts: Vec<String> = (0..3).map(|i| format!("host{i}")).collect();
    let cfg = RunConfig::default();

    println!("dataset `sales`: {} rows in {} shards\n", ds.total_rows(), ds.shards.len());

    // 1) Aggregate: total + average revenue per region, biggest first.
    let q = sql::parse(
        "SELECT SUM(amount), AVG(amount), COUNT(*) FROM sales GROUP BY region ORDER BY sum_amount DESC",
    )?;
    let report = run(&q, &ds.shards, &hosts, &host, &cfg).await?;
    println!("revenue by region (desc):");
    for g in &report.results {
        println!(
            "  {:<5} sum={:<6} avg={:<8} count={}",
            g.key[0],
            g.values["sum_amount"],
            g.values["avg_amount"],
            g.values["count"]
        );
    }

    // 2) Projection: the high-value orders, newest-style top-N.
    let p = sql::parse("SELECT region, product, amount FROM sales WHERE amount >= 200 ORDER BY amount DESC LIMIT 3")?;
    let proj = run_projection(&p, &ds.shards, &hosts, &host, &cfg).await?;
    println!("\ntop {} orders >= 200:", proj.rows.len());
    for r in &proj.rows {
        println!("  {} {} ${}", r["region"], r["product"], r["amount"]);
    }

    Ok(())
}
