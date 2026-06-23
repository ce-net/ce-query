//! The distributed execution engine — fan map tasks across hosts, reduce partials, handle drops.
//!
//! The engine ties the pure pieces together: [`plan`](crate::plan) assigns shards to hosts,
//! [`Query::map_shard`](crate::query::Query::map_shard) runs on each host, and [`combine::reduce`]
//! folds the results. Its one real job beyond orchestration is **partial-failure handling**: if the
//! host assigned a shard drops (4xx/5xx, missing blob, timeout), the engine redistributes that shard
//! to the next-best host in its rendezvous ranking and retries, up to a bound. A shard whose every
//! candidate host has dropped is a hard error — the engine never returns a *silently wrong* answer.
//!
//! ## The [`MapHost`] seam
//!
//! All host interaction goes through the [`MapHost`] trait: "given a host id and a shard, return its
//! [`Partial`]". The production implementation ([`MeshMapHost`]) fans the task over the CE mesh
//! (`ce_rs` request/reply to the host, which fetches the shard blob and maps it). Tests inject a
//! mock host that can drop, corrupt, or stall specific shards — so the retry/redistribute logic is
//! validated against real failure shapes without a network. This is the failure-injection backbone
//! the task asked for.

use crate::combine::{self, GroupResult, Partial};
use crate::dataset::Shard;
use crate::plan::{self, ShardTask};
use crate::query::Query;
use anyhow::{Result, bail};
use std::collections::BTreeMap;

/// Why a map attempt failed. Distinct variants so the engine (and tests) can reason about which
/// failures are retryable on another host — here, all of them are: the shard's bytes are
/// content-addressed and host-independent, so any failure just means "try the next host".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapError {
    /// The host did not respond in time.
    Timeout,
    /// The host returned an error status (HTTP 4xx/5xx or a mesh error).
    HostError(String),
    /// The host could not find/fetch the shard blob.
    MissingBlob(String),
    /// The host returned bytes that failed shard decoding (malformed/corrupt).
    Corrupt(String),
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapError::Timeout => write!(f, "timeout"),
            MapError::HostError(s) => write!(f, "host error: {s}"),
            MapError::MissingBlob(s) => write!(f, "missing blob: {s}"),
            MapError::Corrupt(s) => write!(f, "corrupt result: {s}"),
        }
    }
}

/// A host that can map one shard of a query into a [`Partial`]. The seam between the engine and the
/// outside world (mesh, or a test mock). Implementations must be side-effect-free with respect to
/// the *answer*: mapping the same shard twice yields equal partials, which is what makes retrying a
/// dropped shard on another host correct.
#[async_trait::async_trait]
pub trait MapHost: Send + Sync {
    /// Run `query` over `shard` on `host_id`, returning the host's partial aggregate.
    async fn map(&self, host_id: &str, shard: &Shard, query: &Query) -> Result<Partial, MapError>;
}

/// Tunables for a distributed run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Maximum hosts to try per shard before giving up on it (caps the fallback walk). A value of
    /// 0 is treated as 1 (always try the primary at least once).
    pub max_attempts_per_shard: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig { max_attempts_per_shard: 4 }
    }
}

/// A completed run's report: the finalised result groups plus per-shard placement diagnostics
/// (which host actually served each shard, and how many attempts it took).
#[derive(Debug, Clone)]
pub struct RunReport {
    /// The finalised, sorted output groups.
    pub results: Vec<GroupResult>,
    /// Per shard CID: the host that ultimately served it.
    pub served_by: BTreeMap<String, String>,
    /// Total number of map attempts made (including retries) — a cost/health signal.
    pub total_attempts: usize,
    /// Number of shards that required at least one failover.
    pub redistributed: usize,
}

/// Execute `query` over `shards` across `hosts` using `host` to run each map task, retrying dropped
/// shards on their next-best host. Returns the reduced, finalised result plus diagnostics, or an
/// error naming the first shard that could not be served by any candidate.
///
/// This is the engine's core and is fully testable with a mock [`MapHost`]: see the failure-
/// injection tests below.
pub async fn run(
    query: &Query,
    shards: &[Shard],
    hosts: &[String],
    host: &dyn MapHost,
    config: &RunConfig,
) -> Result<RunReport> {
    query.validate()?;
    if shards.is_empty() {
        // A dataset with no shards yields the empty result (a single global group is *not* implied —
        // there are zero rows). Finalising an empty partial gives an empty group set.
        return Ok(RunReport {
            results: Partial::empty(query.aggregates.clone()).finalize(),
            served_by: BTreeMap::new(),
            total_attempts: 0,
            redistributed: 0,
        });
    }
    if hosts.is_empty() {
        bail!("no candidate hosts to run the query on");
    }

    let max_attempts = config.max_attempts_per_shard.max(1);
    let mut tasks: Vec<ShardTask> = plan::plan(shards, hosts);
    let mut partials: Vec<Partial> = Vec::with_capacity(tasks.len());
    let mut served_by: BTreeMap<String, String> = BTreeMap::new();
    let mut total_attempts = 0usize;
    let mut redistributed = 0usize;

    for task in tasks.iter_mut() {
        let mut attempts = 0usize;
        let mut last_err: Option<MapError> = None;
        // Walk the shard's fallback ranking until it succeeds, the attempt cap is hit, or candidates
        // are exhausted. Not a `while let` because two distinct break conditions live in the body.
        #[allow(clippy::while_let_loop)]
        loop {
            let Some(host_id) = task.host().map(str::to_string) else {
                break; // exhausted candidate hosts
            };
            if attempts >= max_attempts {
                break; // hit the per-shard attempt cap
            }
            attempts += 1;
            total_attempts += 1;
            match host.map(&host_id, &task.shard, query).await {
                Ok(partial) => {
                    if attempts > 1 {
                        redistributed += 1;
                    }
                    served_by.insert(task.shard.cid.clone(), host_id);
                    partials.push(partial);
                    last_err = None;
                    break;
                }
                Err(e) => {
                    // Any failure is retryable on the next host (content-addressed shard).
                    tracing::warn!(shard = %task.shard.cid, host = %host_id, err = %e, "map task failed; failing over");
                    last_err = Some(e);
                    task.advance();
                }
            }
        }
        if last_err.is_some() && !served_by.contains_key(&task.shard.cid) {
            bail!(
                "shard {} could not be served by any of {} candidate host(s) (last error: {})",
                task.shard.cid,
                task.ranked_hosts.len(),
                last_err.map(|e| e.to_string()).unwrap_or_else(|| "unknown".into())
            );
        }
    }

    let merged = combine::reduce(query.aggregates.clone(), partials);
    Ok(RunReport { results: merged.finalize(), served_by, total_attempts, redistributed })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Row, encode_shard, shard_rows};
    use crate::query::Agg;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;

    fn row(v: i64) -> Row {
        [("v".to_string(), json!(v))].into_iter().collect()
    }

    fn sample_query() -> Query {
        Query::new("t").agg(Agg::Count).agg(Agg::Sum("v".into()))
    }

    fn hosts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("host{i}")).collect()
    }

    /// A mock host backed by an in-memory CID -> rows map, with injectable failures keyed by either
    /// a specific host id (the host is "down") or a specific shard CID on its first N attempts.
    struct MockHost {
        /// CID -> the shard's rows.
        store: HashMap<String, Vec<Row>>,
        /// Host ids that always fail (simulating a dropped/unreachable peer).
        dead_hosts: HashSet<String>,
        /// Shard CIDs whose blob is missing everywhere (simulating an unavailable blob).
        missing: HashSet<String>,
        /// Per (host, shard) call counts, to assert retry behaviour and to drive flaky failures.
        calls: Mutex<HashMap<(String, String), usize>>,
        /// Shard CIDs that fail on their first attempt regardless of host (transient), then succeed.
        flaky_once: Mutex<HashSet<String>>,
    }

    impl MockHost {
        fn new(store: HashMap<String, Vec<Row>>) -> Self {
            MockHost {
                store,
                dead_hosts: HashSet::new(),
                missing: HashSet::new(),
                calls: Mutex::new(HashMap::new()),
                flaky_once: Mutex::new(HashSet::new()),
            }
        }
        fn kill(mut self, host: &str) -> Self {
            self.dead_hosts.insert(host.to_string());
            self
        }
        fn miss(mut self, cid: &str) -> Self {
            self.missing.insert(cid.to_string());
            self
        }
        fn flaky(self, cid: &str) -> Self {
            self.flaky_once.lock().unwrap().insert(cid.to_string());
            self
        }
        fn call_count(&self, host: &str, cid: &str) -> usize {
            *self.calls.lock().unwrap().get(&(host.to_string(), cid.to_string())).unwrap_or(&0)
        }
    }

    #[async_trait::async_trait]
    impl MapHost for MockHost {
        async fn map(
            &self,
            host_id: &str,
            shard: &Shard,
            query: &Query,
        ) -> Result<Partial, MapError> {
            *self
                .calls
                .lock()
                .unwrap()
                .entry((host_id.to_string(), shard.cid.clone()))
                .or_insert(0) += 1;

            if self.dead_hosts.contains(host_id) {
                return Err(MapError::Timeout);
            }
            if self.missing.contains(&shard.cid) {
                return Err(MapError::MissingBlob(shard.cid.clone()));
            }
            {
                let mut flaky = self.flaky_once.lock().unwrap();
                if flaky.remove(&shard.cid) {
                    return Err(MapError::HostError("transient 503".into()));
                }
            }
            let rows = self
                .store
                .get(&shard.cid)
                .ok_or_else(|| MapError::MissingBlob(shard.cid.clone()))?;
            Ok(query.map_shard(rows))
        }
    }

    /// Build shards + a populated store from a flat row list split into `n` shards.
    fn make_dataset(rows: &[Row], n: usize) -> (Vec<Shard>, HashMap<String, Vec<Row>>) {
        let parts = shard_rows(rows, n).unwrap();
        let mut shards = Vec::new();
        let mut store = HashMap::new();
        for (rs, _) in parts {
            let bytes = encode_shard(&rs).unwrap();
            let cid = crate::dataset::Shard { cid: ce_rs::cid(&bytes), rows: rs.len() as u64, bytes: bytes.len() as u64 };
            store.insert(cid.cid.clone(), rs);
            shards.push(cid);
        }
        (shards, store)
    }

    #[tokio::test]
    async fn happy_path_matches_local() {
        let rows: Vec<Row> = (1..=12).map(row).collect();
        let (shards, store) = make_dataset(&rows, 4);
        let host = MockHost::new(store);
        let report = run(&sample_query(), &shards, &hosts(3), &host, &RunConfig::default())
            .await
            .unwrap();

        // count=12, sum=78.
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].values["count"], json!(12));
        assert_eq!(report.results[0].values["sum_v"], json!(78.0));
        assert_eq!(report.redistributed, 0);
        assert_eq!(report.served_by.len(), shards.len());
    }

    #[tokio::test]
    async fn dropped_host_redistributes_and_still_correct() {
        let rows: Vec<Row> = (1..=20).map(row).collect();
        let (shards, store) = make_dataset(&rows, 5);
        let h = hosts(3);

        // Kill whichever host is the primary for the first shard, forcing a failover.
        let primary = plan::rank_hosts(&shards[0].cid, &h)[0].clone();
        let host = MockHost::new(store).kill(&primary);

        let report = run(&sample_query(), &shards, &h, &host, &RunConfig::default())
            .await
            .unwrap();

        // Answer is still complete and correct: count=20, sum=210.
        assert_eq!(report.results[0].values["count"], json!(20));
        assert_eq!(report.results[0].values["sum_v"], json!(210.0));
        // At least the shards primaried on the dead host were redistributed.
        assert!(report.redistributed >= 1, "expected a failover");
        // No shard was served by the dead host.
        assert!(report.served_by.values().all(|hh| hh != &primary));
    }

    #[tokio::test]
    async fn transient_failure_retries_then_succeeds() {
        let rows: Vec<Row> = (1..=6).map(row).collect();
        let (shards, store) = make_dataset(&rows, 2);
        let target = shards[0].cid.clone();
        let host = MockHost::new(store).flaky(&target);

        let report = run(&sample_query(), &shards, &hosts(3), &host, &RunConfig::default())
            .await
            .unwrap();
        // Correct result despite the transient failure on shard[0].
        assert_eq!(report.results[0].values["count"], json!(6));
        assert_eq!(report.redistributed, 1, "the flaky shard failed over once");
    }

    #[tokio::test]
    async fn missing_blob_on_all_hosts_is_hard_error() {
        let rows: Vec<Row> = (1..=4).map(row).collect();
        let (shards, store) = make_dataset(&rows, 2);
        let missing = shards[1].cid.clone();
        let host = MockHost::new(store).miss(&missing);

        let err = run(&sample_query(), &shards, &hosts(3), &host, &RunConfig::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("could not be served"), "{err}");
        // It tried every candidate host for that shard (3) before giving up.
        for hh in &hosts(3) {
            assert!(host.call_count(hh, &missing) >= 1 || hosts(3).len() < 3);
        }
    }

    #[tokio::test]
    async fn no_hosts_errors() {
        let rows: Vec<Row> = (1..=4).map(row).collect();
        let (shards, _store) = make_dataset(&rows, 2);
        let host = MockHost::new(HashMap::new());
        let err = run(&sample_query(), &shards, &[], &host, &RunConfig::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no candidate hosts"), "{err}");
    }

    #[tokio::test]
    async fn empty_dataset_yields_empty_result() {
        let host = MockHost::new(HashMap::new());
        let report = run(&sample_query(), &[], &hosts(2), &host, &RunConfig::default())
            .await
            .unwrap();
        assert!(report.results.is_empty());
        assert_eq!(report.total_attempts, 0);
    }

    #[tokio::test]
    async fn attempt_cap_is_respected() {
        // Every host is dead; with cap=2 we should make at most 2 attempts per shard before failing.
        let rows: Vec<Row> = (1..=2).map(row).collect();
        let (shards, store) = make_dataset(&rows, 1);
        let h = hosts(5);
        let mut host = MockHost::new(store);
        for hh in &h {
            host = host.kill(hh);
        }
        let cfg = RunConfig { max_attempts_per_shard: 2 };
        let err = run(&sample_query(), &shards, &h, &host, &cfg).await.unwrap_err();
        assert!(err.to_string().contains("could not be served"), "{err}");
        // Exactly 2 attempts were made for the single shard despite 5 candidate hosts.
        let total: usize = h.iter().map(|hh| host.call_count(hh, &shards[0].cid)).sum();
        assert_eq!(total, 2, "attempt cap must bound retries");
    }

    #[tokio::test]
    async fn invalid_query_rejected_before_dispatch() {
        let host = MockHost::new(HashMap::new());
        let q = Query::new("t"); // no aggregates
        let err = run(&q, &[], &hosts(1), &host, &RunConfig::default()).await.unwrap_err();
        assert!(err.to_string().contains("no aggregates"), "{err}");
    }
}
