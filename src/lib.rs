//! # ce-query — distributed query / map-reduce over CE blob datasets
//!
//! `ce-query` is a **BigQuery-shaped analytics engine built on CE primitives** (the SDK/app tier,
//! alongside `ce-storage`, `ce-pin`, `ce-coord`, `swarm`) — **not** a node feature. It turns CE's
//! flat content-addressed blob layer into queryable **datasets**: register a dataset by sharding its
//! rows into blobs, then run a `SELECT … WHERE … GROUP BY …` map-reduce that fans map tasks across
//! atlas hosts (compute goes to the data) and reduces partial aggregates into a final answer.
//!
//! ## What it composes (it reinvents nothing)
//!
//! | Concern | CE primitive used |
//! |---|---|
//! | Dataset storage (shards) | `ce-rs` `put_object`/`get_object` over the node `/blobs` store — content-addressed, hash-verified |
//! | Durability / availability | `ce-pin` (pin shard CIDs across N hosts) — orthogonal, reuse as-is |
//! | Host selection for map tasks | `ce-rs` atlas (`/atlas`) + `find_service` for the `ce-query` agent |
//! | Fan-out map / collect reduce | `ce-rs` mesh `request`/`reply` to each assigned host (`MeshMapHost`) |
//! | Authorization | `ce-cap` signed, attenuating chains scoped to a dataset (`query:read`), offline-verifiable |
//! | Dataset catalog (`name -> shards`) | a local JSON map — the ce-coord `RMap` shape kept local for the owner |
//!
//! ## The architecture in one paragraph
//!
//! A [`Query`] splits into a pure **map** ([`Query::map_shard`] → a [`Partial`]) and a pure
//! **reduce** ([`combine::reduce`] of [`Partial`]s, then [`Partial::finalize`]). Every aggregate is a
//! **monoid** ([`combine::Accum`]) so the reduce is associative and commutative — which is exactly
//! what makes the engine distributable and fault-tolerant: shards map independently on different
//! hosts ([`plan`] assigns them by rendezvous hash), partials merge in any order, and a dropped host
//! is handled by redistributing its shard to the next-best candidate ([`engine::run`]). The map/host
//! seam is the [`engine::MapHost`] trait, with a coordinator-local ([`mesh::LocalMapHost`]) and a
//! true-distributed ([`mesh::MeshMapHost`]) implementation.
//!
//! ```no_run
//! use ce_query::{Query, query::Agg, sql};
//! # async fn demo() -> anyhow::Result<()> {
//! // Builder API …
//! let q = Query::new("sales").agg(Agg::Sum("amount".into())).agg(Agg::Count).group("region");
//! // … or the SQL-ish front end (equivalent):
//! let q2 = sql::parse("SELECT SUM(amount), COUNT(*) FROM sales GROUP BY region")?;
//! assert_eq!(q, q2);
//! # Ok(()) }
//! ```

pub mod catalog;
pub mod caps;
pub mod combine;
pub mod dataset;
pub mod engine;
pub mod join;
pub mod mesh;
pub mod order;
pub mod plan;
pub mod query;
pub mod sql;

pub use catalog::Catalog;
pub use combine::{Accum, GroupResult, Partial};
pub use dataset::{Dataset, Row, Shard};
pub use engine::{MapError, MapHost, RunConfig, RunReport, run};
pub use join::{JoinKeys, distributed_join, hash_join};
pub use query::{Agg, CmpOp, OrderDir, OrderKey, Predicate, Query};

use anyhow::{Context, Result};
use ce_rs::CeClient;

/// Register a dataset from a flat row stream: shard the rows into `shard_count` blobs, upload each to
/// the CE blob store, and return a [`Dataset`] with the resulting shard CIDs + counts. The dataset
/// metadata is returned (not persisted) so the caller can place it in a [`Catalog`].
///
/// This is the write path behind `ce-query dataset add`. Each shard is an NDJSON object stored via
/// [`ce_rs::CeClient::put_object`]; its CID is the manifest hash, so the same rows registered twice
/// dedup to the same shard CIDs (content addressing).
pub async fn register_dataset(
    client: &CeClient,
    name: impl Into<String>,
    schema: Vec<String>,
    rows: &[Row],
    shard_count: usize,
) -> Result<Dataset> {
    let name = name.into();
    let mut ds = Dataset::new(name.clone(), schema);
    let parts = dataset::shard_rows(rows, shard_count).context("sharding rows")?;
    for (shard_rows, bytes) in parts {
        // Skip wholly-empty shards: they carry no rows, so there is nothing to store or query.
        if shard_rows.is_empty() {
            continue;
        }
        let cid = client
            .put_object(&bytes)
            .await
            .with_context(|| format!("uploading a shard of dataset `{name}`"))?;
        ds.add_shard(cid, shard_rows.len() as u64, bytes.len() as u64);
    }
    Ok(ds)
}

/// Discover candidate map hosts from the atlas. When `service_only` is true, returns only nodes that
/// advertise the `ce-query` map agent (the distributed mode); otherwise returns every peer in the
/// atlas (suitable as fetch providers for the coordinator-local mode). Sorted and de-duplicated.
pub async fn discover_hosts(client: &CeClient, service_only: bool) -> Result<Vec<String>> {
    let mut hosts: Vec<String> = if service_only {
        client.find_service(mesh::QUERY_SERVICE).await.unwrap_or_default()
    } else {
        client.atlas().await.context("reading atlas")?.into_iter().map(|e| e.node_id).collect()
    };
    hosts.sort();
    hosts.dedup();
    Ok(hosts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// register_dataset's pure sharding/counting behaviour is exercised here without a node by
    /// re-deriving the shard CIDs the same way the SDK would (ce_rs::cid over encoded bytes). This
    /// asserts the metadata the function would produce is self-consistent.
    #[test]
    fn dataset_sharding_is_consistent() {
        let rows: Vec<Row> = (0..10)
            .map(|i| [("v".to_string(), json!(i))].into_iter().collect())
            .collect();
        let parts = dataset::shard_rows(&rows, 3).unwrap();
        let mut total_rows = 0u64;
        for (rs, bytes) in &parts {
            total_rows += rs.len() as u64;
            // Non-empty shards have a stable 64-hex CID.
            if !rs.is_empty() {
                assert_eq!(ce_rs::cid(bytes).len(), 64);
            }
        }
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn builder_equals_sql() {
        let q = Query::new("t").agg(Agg::Count).group("g");
        let q2 = sql::parse("SELECT COUNT(*) FROM t GROUP BY g").unwrap();
        assert_eq!(q, q2);
    }
}
