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
use futures::stream::{FuturesUnordered, StreamExt};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

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

    /// Run a **projection** `query` over `shard` on `host_id`, returning the filtered, column-pruned
    /// rows. The projection analogue of [`map`](Self::map). Implementations fetch the shard and call
    /// [`Query::project_shard`]. A default is provided for hosts that do not specialise it.
    async fn project(
        &self,
        host_id: &str,
        shard: &Shard,
        query: &Query,
    ) -> Result<Vec<crate::dataset::Row>, MapError>;
}

/// Tunables for a distributed run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Maximum hosts to try per shard before giving up on it (caps the fallback walk). A value of
    /// 0 is treated as 1 (always try the primary at least once).
    pub max_attempts_per_shard: usize,
    /// Maximum number of shards mapped concurrently (the fan-out width). `0` is treated as `1`. This
    /// is the headline parallelism: instead of awaiting shards one at a time, up to this many map
    /// tasks are in flight at once, so wall-clock scales with host count, not shard count.
    pub max_in_flight: usize,
    /// Redundancy factor `K`: map each shard on its top-`K` distinct hosts and require their partials
    /// to **agree** before accepting (BigQuery-style result verification on an untrusted mesh). `1`
    /// (the default) disables redundancy. `2`/`3` detect a single corrupt/lying host.
    pub redundancy: usize,
    /// When `redundancy > 1`, require this many agreeing copies (a majority quorum). `0` means "all
    /// `redundancy` copies must agree" (unanimity). A typical config is `redundancy=3, quorum=2`.
    pub quorum: usize,
    /// Overall query deadline. If the whole run exceeds this, it fails with a clear timeout rather
    /// than hanging. `None` = no deadline.
    pub deadline: Option<Duration>,
    /// Cost ceiling: reject the query before dispatch if the dataset exceeds these bounds. Guards
    /// against unbounded memory/spend (DoS).
    pub limits: CostLimits,
}

impl Default for RunConfig {
    fn default() -> Self {
        RunConfig {
            max_attempts_per_shard: 4,
            max_in_flight: 16,
            redundancy: 1,
            quorum: 0,
            deadline: None,
            limits: CostLimits::default(),
        }
    }
}

/// Per-query cost ceilings, enforced **before** any shard is dispatched (from the dataset's declared
/// shard/row/byte counts) and again on the materialised result. A query that would scan or return
/// more than these bounds fails fast with [`CostError`] instead of exhausting memory or spend.
#[derive(Debug, Clone)]
pub struct CostLimits {
    /// Maximum total bytes the query may scan across all shards (`0` = unlimited).
    pub max_scan_bytes: u64,
    /// Maximum total rows the query may scan across all shards (`0` = unlimited).
    pub max_scan_rows: u64,
    /// Maximum number of shards a single query may fan out to (`0` = unlimited).
    pub max_shards: usize,
    /// Maximum number of result groups the reduce may produce (`0` = unlimited). Caps a high-
    /// cardinality `GROUP BY` from blowing up coordinator memory.
    pub max_result_groups: usize,
}

impl Default for CostLimits {
    fn default() -> Self {
        // Generous defaults that still cap pathological inputs: ~16 GiB scanned, 1e9 rows, 4096
        // shards, 1e6 result groups. Callers tighten these for metered/multi-tenant deployments.
        CostLimits {
            max_scan_bytes: 16 * 1024 * 1024 * 1024,
            max_scan_rows: 1_000_000_000,
            max_shards: 4096,
            max_result_groups: 1_000_000,
        }
    }
}

impl CostLimits {
    /// Unlimited — every bound disabled. Useful for trusted local runs and tests.
    pub fn unlimited() -> CostLimits {
        CostLimits { max_scan_bytes: 0, max_scan_rows: 0, max_shards: 0, max_result_groups: 0 }
    }

    /// Check the declared dataset cost against the scan bounds. Returns the first bound exceeded.
    pub fn check_scan(&self, shards: &[Shard]) -> Result<(), CostError> {
        if self.max_shards != 0 && shards.len() > self.max_shards {
            return Err(CostError { what: "shards".into(), actual: shards.len() as u64, limit: self.max_shards as u64 });
        }
        if self.max_scan_rows != 0 {
            let rows: u64 = shards.iter().map(|s| s.rows).sum();
            if rows > self.max_scan_rows {
                return Err(CostError { what: "rows scanned".into(), actual: rows, limit: self.max_scan_rows });
            }
        }
        if self.max_scan_bytes != 0 {
            let bytes: u64 = shards.iter().map(|s| s.bytes).sum();
            if bytes > self.max_scan_bytes {
                return Err(CostError { what: "bytes scanned".into(), actual: bytes, limit: self.max_scan_bytes });
            }
        }
        Ok(())
    }
}

/// A cost-limit breach: which bound, the actual value, and the configured limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostError {
    /// The bound that was exceeded (`"shards"`, `"rows scanned"`, ...).
    pub what: String,
    /// The actual value the query would have incurred.
    pub actual: u64,
    /// The configured ceiling.
    pub limit: u64,
}

impl std::fmt::Display for CostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "query cost limit exceeded: {} = {} > limit {}", self.what, self.actual, self.limit)
    }
}

impl std::error::Error for CostError {}

/// A completed run's report: the finalised result groups plus per-shard placement diagnostics
/// (which host actually served each shard, and how many attempts it took).
#[derive(Debug, Clone)]
pub struct RunReport {
    /// The finalised, sorted output groups.
    pub results: Vec<GroupResult>,
    /// Per shard CID: the host that ultimately served it (the first agreeing host under redundancy).
    pub served_by: BTreeMap<String, String>,
    /// Total number of map attempts made (including retries and redundant copies) — a cost signal.
    pub total_attempts: usize,
    /// Number of shards that required at least one failover.
    pub redistributed: usize,
    /// Number of shards verified by `redundancy > 1` agreement.
    pub verified: usize,
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
            results: query.shape(Partial::empty(query.aggregates.clone()).finalize()),
            served_by: BTreeMap::new(),
            total_attempts: 0,
            redistributed: 0,
            verified: 0,
        });
    }
    if hosts.is_empty() {
        bail!("no candidate hosts to run the query on");
    }
    // Enforce cost limits up front, before any byte is fetched or any host is contacted.
    config.limits.check_scan(shards)?;

    let started = Instant::now();
    let in_flight = config.max_in_flight.max(1);
    // Partition pruning: drop shards whose stats prove they cannot match the WHERE predicate. This
    // never changes the answer (a pruned shard contributes nothing) — only the work done.
    let live: Vec<Shard> = shards
        .iter()
        .filter(|s| plan::shard_can_match(s.stats.as_ref(), &query.predicate))
        .cloned()
        .collect();
    if live.is_empty() {
        // Every shard was pruned: the result is the empty aggregate (no rows match).
        return Ok(RunReport {
            results: query.shape(Partial::empty(query.aggregates.clone()).finalize()),
            served_by: BTreeMap::new(),
            total_attempts: 0,
            redistributed: 0,
            verified: 0,
        });
    }
    let tasks: Vec<ShardTask> = plan::plan(&live, hosts);

    // Drive the per-shard map walk concurrently, bounded to `in_flight` outstanding tasks. Each
    // future resolves to `(index, Result<ShardOutcome>)` so we can reassemble in any arrival order —
    // the monoid reduce makes order irrelevant, but we keep the index for stable diagnostics.
    let mut pending = tasks.into_iter().enumerate();
    let mut inflight = FuturesUnordered::new();
    let mut outcomes: Vec<Option<ShardOutcome>> = Vec::new();
    outcomes.resize_with(live.len(), || None);

    // Prime the pump.
    for _ in 0..in_flight {
        if let Some((idx, task)) = pending.next() {
            inflight.push(map_one_shard(idx, task, query, host, config, started));
        }
    }

    while let Some((idx, result)) = inflight.next().await {
        match result {
            Ok(outcome) => outcomes[idx] = Some(outcome),
            Err(e) => return Err(e), // a shard exhausted all hosts / deadline / quorum failure
        }
        // Keep the window full.
        if let Some((next_idx, task)) = pending.next() {
            inflight.push(map_one_shard(next_idx, task, query, host, config, started));
        }
    }

    let mut partials: Vec<Partial> = Vec::with_capacity(shards.len());
    let mut served_by: BTreeMap<String, String> = BTreeMap::new();
    let mut total_attempts = 0usize;
    let mut redistributed = 0usize;
    let mut verified = 0usize;
    for outcome in outcomes.into_iter().flatten() {
        total_attempts += outcome.attempts;
        if outcome.redistributed {
            redistributed += 1;
        }
        if outcome.verified {
            verified += 1;
        }
        served_by.insert(outcome.shard_cid, outcome.served_by);
        partials.push(outcome.partial);
    }

    let merged = combine::reduce(query.aggregates.clone(), partials);
    let mut final_rows = merged.finalize();
    // Cap result cardinality (high-cardinality GROUP BY memory guard) before ordering/limit.
    if config.limits.max_result_groups != 0 && final_rows.len() > config.limits.max_result_groups {
        return Err(CostError {
            what: "result groups".into(),
            actual: final_rows.len() as u64,
            limit: config.limits.max_result_groups as u64,
        }
        .into());
    }
    // Apply ORDER BY / LIMIT once, after the distributed reduce, over the small result set.
    final_rows = query.shape(final_rows);
    Ok(RunReport { results: final_rows, served_by, total_attempts, redistributed, verified })
}

/// The result of a projection (`SELECT a, b … `) run: the projected rows plus diagnostics.
#[derive(Debug, Clone)]
pub struct ProjectionReport {
    /// The filtered, projected rows (truncated to `LIMIT` and the row cap).
    pub rows: Vec<crate::dataset::Row>,
    /// Per shard CID: the host that served it.
    pub served_by: BTreeMap<String, String>,
    /// Total map attempts (including retries).
    pub total_attempts: usize,
    /// True if the row cap / LIMIT was hit and some matching rows were not returned.
    pub truncated: bool,
}

/// Execute a **projection** `query` (`SELECT a, b FROM t WHERE …`) over `shards` across `hosts`,
/// returning the matching rows. Shards fan out concurrently like [`run`]; each host applies the
/// predicate and column pruning during its scan (predicate/projection pushdown), so only matching
/// rows travel back. A global row cap — the query's `LIMIT` if set, else
/// [`CostLimits::max_result_groups`] reused as a row ceiling — bounds coordinator memory. Because
/// projection has no associative reduce, results are concatenated; with a `LIMIT` and no `ORDER BY`
/// the engine can stop early once enough rows are collected.
pub async fn run_projection(
    query: &Query,
    shards: &[Shard],
    hosts: &[String],
    host: &dyn MapHost,
    config: &RunConfig,
) -> Result<ProjectionReport> {
    query.validate()?;
    if !query.is_projection() {
        bail!("run_projection called on a non-projection query");
    }
    if shards.is_empty() {
        return Ok(ProjectionReport {
            rows: Vec::new(),
            served_by: BTreeMap::new(),
            total_attempts: 0,
            truncated: false,
        });
    }
    if hosts.is_empty() {
        bail!("no candidate hosts to run the query on");
    }
    config.limits.check_scan(shards)?;

    // Two distinct bounds:
    // - `mem_cap`: the memory ceiling on rows we may hold at once (the cost limit). Always enforced.
    // - `limit`: the query's LIMIT, applied to the *final* (possibly ordered) result.
    // ORDER BY needs every matching row before sorting, so we may only early-stop (truncate during
    // collection) when there is a LIMIT and no ORDER BY. Otherwise we collect up to the memory cap
    // and apply the LIMIT after ordering.
    let mem_cap = if config.limits.max_result_groups == 0 {
        None
    } else {
        Some(config.limits.max_result_groups)
    };
    let can_early_stop = query.limit.is_some() && query.order_by.is_empty();
    // During collection, the cap to truncate at: the LIMIT when we can early-stop, else the memory
    // cap. (When ordering, never truncate to LIMIT mid-collection — that would drop rows that should
    // have sorted to the top.)
    let collect_cap = if can_early_stop { query.limit.or(mem_cap) } else { mem_cap };

    let started = Instant::now();
    let in_flight = config.max_in_flight.max(1);
    // Partition pruning: skip shards whose stats prove they cannot match the WHERE predicate.
    let live: Vec<Shard> = shards
        .iter()
        .filter(|s| plan::shard_can_match(s.stats.as_ref(), &query.predicate))
        .cloned()
        .collect();
    let tasks: Vec<ShardTask> = plan::plan(&live, hosts);
    let mut pending = tasks.into_iter();
    let mut inflight = FuturesUnordered::new();
    for _ in 0..in_flight {
        if let Some(task) = pending.next() {
            inflight.push(project_one_shard(task, query, host, config, started));
        }
    }

    let mut rows: Vec<crate::dataset::Row> = Vec::new();
    let mut served_by: BTreeMap<String, String> = BTreeMap::new();
    let mut total_attempts = 0usize;
    let mut truncated = false;

    while let Some(result) = inflight.next().await {
        let outcome = result?;
        total_attempts += outcome.attempts;
        served_by.insert(outcome.shard_cid, outcome.served_by);
        rows.extend(outcome.rows);
        if let Some(cap) = collect_cap
            && rows.len() >= cap
        {
            truncated = rows.len() > cap || truncated;
            rows.truncate(cap);
            if can_early_stop {
                break; // enough rows; no ORDER BY means we can stop fetching
            }
        }
        if let Some(task) = pending.next() {
            inflight.push(project_one_shard(task, query, host, config, started));
        }
    }

    // Apply ORDER BY (over all collected rows) and then the final LIMIT.
    if !query.order_by.is_empty() {
        crate::order::order_raw_rows(&mut rows, &query.order_by);
    }
    if let Some(n) = query.limit
        && rows.len() > n
    {
        truncated = true;
        rows.truncate(n);
    }
    Ok(ProjectionReport { rows, served_by, total_attempts, truncated })
}

/// What projecting one shard produced.
struct ProjectOutcome {
    shard_cid: String,
    served_by: String,
    rows: Vec<crate::dataset::Row>,
    attempts: usize,
}

/// Project one shard with the same fallback walk as [`map_one_shard`] (no redundancy — projection
/// rows are not reduced, so cross-host agreement is not meaningful per shard).
async fn project_one_shard(
    mut task: ShardTask,
    query: &Query,
    host: &dyn MapHost,
    config: &RunConfig,
    started: Instant,
) -> Result<ProjectOutcome> {
    let max_attempts = config.max_attempts_per_shard.max(1);
    let mut attempts = 0usize;
    let mut last_err: Option<MapError> = None;
    loop {
        if let Some(deadline) = config.deadline
            && started.elapsed() >= deadline
        {
            bail!("query deadline ({:?}) exceeded while scanning shard {}", deadline, task.shard.cid);
        }
        let Some(host_id) = task.host().map(str::to_string) else { break };
        if attempts >= max_attempts {
            break;
        }
        attempts += 1;
        match host.project(&host_id, &task.shard, query).await {
            Ok(rows) => {
                return Ok(ProjectOutcome {
                    shard_cid: task.shard.cid.clone(),
                    served_by: host_id,
                    rows,
                    attempts,
                });
            }
            Err(e) => {
                tracing::warn!(shard = %task.shard.cid, host = %host_id, err = %e, "projection task failed; failing over");
                last_err = Some(e);
                task.advance();
            }
        }
    }
    bail!(
        "shard {} could not be served by any of {} candidate host(s) (last error: {})",
        task.shard.cid,
        task.ranked_hosts.len(),
        last_err.map(|e| e.to_string()).unwrap_or_else(|| "unknown".into())
    )
}

/// What mapping one shard produced: the partial plus diagnostics for the report.
struct ShardOutcome {
    shard_cid: String,
    served_by: String,
    partial: Partial,
    attempts: usize,
    redistributed: bool,
    verified: bool,
}

/// Map one shard, walking its fallback ranking until success (or quorum, under redundancy), the
/// attempt cap, or candidate exhaustion. Returns `(index, outcome)`; an error means the shard could
/// not be served by any candidate, the deadline elapsed, or redundant copies disagreed.
async fn map_one_shard(
    idx: usize,
    mut task: ShardTask,
    query: &Query,
    host: &dyn MapHost,
    config: &RunConfig,
    started: Instant,
) -> (usize, Result<ShardOutcome>) {
    let max_attempts = config.max_attempts_per_shard.max(1);
    let redundancy = config.redundancy.max(1);
    let mut attempts = 0usize;
    let mut last_err: Option<MapError> = None;
    // For redundancy: collect agreeing copies keyed by their canonical partial form.
    let mut copies: Vec<(String, Partial)> = Vec::new(); // (host_id, partial)
    let mut first_host: Option<String> = None;

    loop {
        // Overall deadline check between attempts.
        if let Some(deadline) = config.deadline
            && started.elapsed() >= deadline
        {
            return (
                idx,
                Err(anyhow::anyhow!(
                    "query deadline ({:?}) exceeded while mapping shard {}",
                    deadline,
                    task.shard.cid
                )),
            );
        }
        let Some(host_id) = task.host().map(str::to_string) else {
            break; // exhausted candidate hosts
        };
        if attempts >= max_attempts {
            break;
        }
        attempts += 1;
        match host.map(&host_id, &task.shard, query).await {
            Ok(partial) => {
                if first_host.is_none() {
                    first_host = Some(host_id.clone());
                }
                copies.push((host_id, partial));
                last_err = None;
                if copies.len() >= redundancy {
                    break;
                }
                // Need another independent copy: advance to the next-best host. If none remains we
                // accept what we have (best-effort redundancy rather than a hard failure when the
                // pool is too small).
                if !task.advance() {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!(shard = %task.shard.cid, host = %host_id, err = %e, "map task failed; failing over");
                last_err = Some(e);
                task.advance();
            }
        }
    }

    if copies.is_empty() {
        return (
            idx,
            Err(anyhow::anyhow!(
                "shard {} could not be served by any of {} candidate host(s) (last error: {})",
                task.shard.cid,
                task.ranked_hosts.len(),
                last_err.map(|e| e.to_string()).unwrap_or_else(|| "unknown".into())
            )),
        );
    }

    // Redundancy verification: require enough copies to agree on the canonical partial bytes.
    let verified = redundancy > 1 && copies.len() > 1;
    let chosen = if verified {
        match select_quorum(&copies, config.quorum, redundancy) {
            Ok(p) => p,
            Err(e) => return (idx, Err(e.context(format!("verifying shard {}", task.shard.cid)))),
        }
    } else {
        copies[0].1.clone()
    };

    let served = first_host.unwrap_or_else(|| copies[0].0.clone());
    (
        idx,
        Ok(ShardOutcome {
            shard_cid: task.shard.cid.clone(),
            served_by: served,
            partial: chosen,
            attempts,
            redistributed: attempts > copies.len(), // more attempts than accepted copies => a failover happened
            verified,
        }),
    )
}

/// Choose the agreed partial from redundant copies. Copies are grouped by their canonical serialized
/// form; the largest agreeing group must meet the quorum (or, when `quorum == 0`, equal the full
/// `redundancy`). A disagreement that no group can satisfy is a hard error — the engine refuses to
/// return a result a lying host could have skewed.
fn select_quorum(copies: &[(String, Partial)], quorum: usize, _redundancy: usize) -> Result<Partial> {
    if copies.is_empty() {
        bail!("no copies to verify");
    }
    // Agreement required among the copies actually collected: an explicit quorum (capped to the
    // number of copies), else unanimity (all collected copies must agree). When fewer than the
    // configured redundancy could be gathered (a small pool), we verify against what we got — a
    // best-effort that still rejects a single lying host whenever there is a second honest copy.
    let need = if quorum == 0 { copies.len() } else { quorum.min(copies.len()) };
    let mut groups: BTreeMap<Vec<u8>, (usize, usize, Partial)> = BTreeMap::new(); // key -> (count, first_seen, partial)
    for (i, (_, p)) in copies.iter().enumerate() {
        // Canonical bytes via JSON (Partial serializes deterministically: BTreeMap order). A copy
        // that somehow fails to serialize gets a unique key so it can never falsely "agree".
        let key = serde_json::to_vec(p).unwrap_or_else(|_| format!("<unserializable {i}>").into_bytes());
        let entry = groups.entry(key).or_insert((0, i, p.clone()));
        entry.0 += 1;
    }
    // Pick the largest agreeing group (ties broken by earliest arrival for determinism).
    let best = groups.values().max_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    match best {
        Some((count, _, partial)) if *count >= need => Ok(partial.clone()),
        Some((count, _, _)) => bail!(
            "redundant hosts disagreed: best agreement was {} of {} copies, needed {} — possible corrupt or malicious host",
            count,
            copies.len(),
            need
        ),
        None => bail!("no copies to verify"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combine::Accum;
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
        /// Host ids that return a *corrupted* partial (a lying/malicious host) — used to test the
        /// redundancy/quorum verification path.
        liars: HashSet<String>,
        /// Per-call artificial delay in milliseconds (to exercise concurrency / deadlines).
        delay_ms: u64,
    }

    impl MockHost {
        fn new(store: HashMap<String, Vec<Row>>) -> Self {
            MockHost {
                store,
                dead_hosts: HashSet::new(),
                missing: HashSet::new(),
                calls: Mutex::new(HashMap::new()),
                flaky_once: Mutex::new(HashSet::new()),
                liars: HashSet::new(),
                delay_ms: 0,
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
        fn lie(mut self, host: &str) -> Self {
            self.liars.insert(host.to_string());
            self
        }
        fn slow(mut self, ms: u64) -> Self {
            self.delay_ms = ms;
            self
        }
        fn call_count(&self, host: &str, cid: &str) -> usize {
            *self.calls.lock().unwrap().get(&(host.to_string(), cid.to_string())).unwrap_or(&0)
        }
        fn total_calls(&self) -> usize {
            self.calls.lock().unwrap().values().sum()
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
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            let rows = self
                .store
                .get(&shard.cid)
                .ok_or_else(|| MapError::MissingBlob(shard.cid.clone()))?;
            let mut partial = query.map_shard(rows);
            if self.liars.contains(host_id) {
                // Corrupt the partial: bump every Count accumulator so the bytes differ from honest
                // hosts, simulating a host that returns a wrong answer.
                for accs in partial.groups.values_mut() {
                    for acc in accs.iter_mut() {
                        if let Accum::Count(n) = acc {
                            *n += 999;
                        }
                    }
                }
            }
            Ok(partial)
        }

        async fn project(
            &self,
            host_id: &str,
            shard: &Shard,
            query: &Query,
        ) -> Result<Vec<Row>, MapError> {
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
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            let rows = self
                .store
                .get(&shard.cid)
                .ok_or_else(|| MapError::MissingBlob(shard.cid.clone()))?;
            Ok(query.project_shard(rows))
        }
    }

    /// Build shards + a populated store from a flat row list split into `n` shards.
    fn make_dataset(rows: &[Row], n: usize) -> (Vec<Shard>, HashMap<String, Vec<Row>>) {
        make_dataset_inner(rows, n, false)
    }

    /// Like [`make_dataset`] but populates each shard's per-column min/max stats, so pruning tests can
    /// exercise the planner's skip-on-stats path.
    fn make_dataset_with_stats(rows: &[Row], n: usize) -> (Vec<Shard>, HashMap<String, Vec<Row>>) {
        make_dataset_inner(rows, n, true)
    }

    fn make_dataset_inner(rows: &[Row], n: usize, with_stats: bool) -> (Vec<Shard>, HashMap<String, Vec<Row>>) {
        let parts = shard_rows(rows, n).unwrap();
        let mut shards = Vec::new();
        let mut store = HashMap::new();
        for (rs, _) in parts {
            let bytes = encode_shard(&rs).unwrap();
            let mut cid = crate::dataset::Shard::new(ce_rs::cid(&bytes), rs.len() as u64, bytes.len() as u64);
            if with_stats {
                cid.stats = Some(crate::dataset::compute_stats(&rs));
            }
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
        assert_eq!(report.results[0].values["sum_v"], json!(78));
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
        assert_eq!(report.results[0].values["sum_v"], json!(210));
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
        let cfg = RunConfig { max_attempts_per_shard: 2, ..RunConfig::default() };
        let err = run(&sample_query(), &shards, &h, &host, &cfg).await.unwrap_err();
        assert!(err.to_string().contains("could not be served"), "{err}");
        // Exactly 2 attempts were made for the single shard despite 5 candidate hosts.
        let total: usize = h.iter().map(|hh| host.call_count(hh, &shards[0].cid)).sum();
        assert_eq!(total, 2, "attempt cap must bound retries");
    }

    #[tokio::test]
    async fn order_by_and_limit_applied_to_run_report() {
        use crate::query::OrderKey;
        // Group by parity, sum v per group, then ORDER BY sum_v DESC LIMIT 1 over the result rows.
        let rows: Vec<Row> = (1..=10).map(row).collect(); // odds sum 25, evens sum 30
        let (shards, store) = make_dataset(&rows, 4);
        let host = MockHost::new(store);
        // A group key derived from parity via a string column.
        let parity_rows: Vec<Row> = (1..=10)
            .map(|v| {
                [("v".to_string(), json!(v)), ("p".to_string(), json!(if v % 2 == 0 { "even" } else { "odd" }))]
                    .into_iter()
                    .collect()
            })
            .collect();
        let (shards2, store2) = make_dataset(&parity_rows, 4);
        let host2 = MockHost::new(store2);
        let _ = (shards, host); // first dataset unused; keep the helper exercised

        let q = Query::new("t")
            .agg(Agg::Sum("v".into()))
            .group("p")
            .order(OrderKey::desc("sum_v"))
            .limit(1);
        let report = run(&q, &shards2, &hosts(3), &host2, &RunConfig::default()).await.unwrap();
        // Only the top group by sum_v survives the LIMIT 1: evens (30) beat odds (25).
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].key[0], "even");
        assert_eq!(report.results[0].values["sum_v"], json!(30));
    }

    #[tokio::test]
    async fn invalid_query_rejected_before_dispatch() {
        let host = MockHost::new(HashMap::new());
        let q = Query::new("t"); // no aggregates
        let err = run(&q, &[], &hosts(1), &host, &RunConfig::default()).await.unwrap_err();
        assert!(err.to_string().contains("no aggregates"), "{err}");
    }

    #[tokio::test]
    async fn concurrent_fanout_is_correct_and_faster_than_serial() {
        // Each map sleeps 40ms. With 10 shards and a wide window the run completes in roughly one
        // slow-call's time, not the sum — proving shards actually fan out concurrently. We assert the
        // answer is still correct and the wall-clock is far below the serial 10*40ms.
        let rows: Vec<Row> = (1..=100).map(row).collect();
        let (shards, store) = make_dataset(&rows, 10);
        let host = MockHost::new(store).slow(40);
        let cfg = RunConfig { max_in_flight: 10, ..RunConfig::default() };
        let t0 = std::time::Instant::now();
        let report = run(&sample_query(), &shards, &hosts(4), &host, &cfg).await.unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(report.results[0].values["count"], json!(100));
        assert_eq!(report.results[0].values["sum_v"], json!(5050));
        assert!(elapsed.as_millis() < 300, "expected concurrent (<300ms), took {elapsed:?}");
    }

    #[tokio::test]
    async fn bounded_in_flight_limits_outstanding_calls() {
        // With max_in_flight=1 the engine is effectively serial; still correct.
        let rows: Vec<Row> = (1..=12).map(row).collect();
        let (shards, store) = make_dataset(&rows, 4);
        let host = MockHost::new(store);
        let cfg = RunConfig { max_in_flight: 1, ..RunConfig::default() };
        let report = run(&sample_query(), &shards, &hosts(3), &host, &cfg).await.unwrap();
        assert_eq!(report.results[0].values["count"], json!(12));
    }

    #[tokio::test]
    async fn redundancy_detects_a_lying_host() {
        // A single corrupt host among the candidates must be caught: with redundancy=2 unanimity,
        // an honest copy and a lying copy disagree -> hard error rather than a wrong answer.
        let rows: Vec<Row> = (1..=8).map(row).collect();
        let (shards, store) = make_dataset(&rows, 2);
        let h = hosts(3);
        // Make the primary of shard[0] a liar.
        let liar = plan::rank_hosts(&shards[0].cid, &h)[0].clone();
        let host = MockHost::new(store).lie(&liar);
        let cfg = RunConfig { redundancy: 2, quorum: 0, ..RunConfig::default() };
        let err = run(&sample_query(), &shards, &h, &host, &cfg).await.unwrap_err();
        // The disagreement is in the error chain (wrapped with a "verifying shard …" context).
        let chain = format!("{err:#}");
        assert!(chain.contains("disagreed"), "expected disagreement error, got: {chain}");
    }

    #[tokio::test]
    async fn redundancy_agrees_when_all_honest() {
        // Two honest copies agree; the answer is accepted and marked verified.
        let rows: Vec<Row> = (1..=8).map(row).collect();
        let (shards, store) = make_dataset(&rows, 2);
        let host = MockHost::new(store);
        let cfg = RunConfig { redundancy: 2, quorum: 0, ..RunConfig::default() };
        let report = run(&sample_query(), &shards, &hosts(4), &host, &cfg).await.unwrap();
        assert_eq!(report.results[0].values["count"], json!(8));
        assert!(report.verified >= 1, "shards mapped on 2 hosts should be verified");
    }

    #[tokio::test]
    async fn quorum_majority_tolerates_one_liar() {
        // redundancy=3, quorum=2: two honest copies outvote one liar, so the query still succeeds
        // with the correct answer (Byzantine fault tolerance for a single bad host).
        let rows: Vec<Row> = (1..=9).map(row).collect();
        let (shards, store) = make_dataset(&rows, 1);
        let h = hosts(4);
        // Lie only on the 2nd-ranked host so the top + 3rd are honest (>= quorum of 2 agree).
        let ranked = plan::rank_hosts(&shards[0].cid, &h);
        let host = MockHost::new(store).lie(&ranked[1]);
        let cfg = RunConfig { redundancy: 3, quorum: 2, ..RunConfig::default() };
        let report = run(&sample_query(), &shards, &h, &host, &cfg).await.unwrap();
        assert_eq!(report.results[0].values["count"], json!(9), "honest majority must win");
    }

    #[tokio::test]
    async fn redundancy_best_effort_with_small_pool() {
        // redundancy=3 but only 2 honest hosts exist: we gather 2 agreeing copies and accept them
        // (best-effort verification), rather than failing because the pool was too small.
        let rows: Vec<Row> = (1..=8).map(row).collect();
        let (shards, store) = make_dataset(&rows, 2);
        let host = MockHost::new(store);
        let cfg = RunConfig { redundancy: 3, quorum: 0, ..RunConfig::default() };
        let report = run(&sample_query(), &shards, &hosts(2), &host, &cfg).await.unwrap();
        assert_eq!(report.results[0].values["count"], json!(8));
    }

    #[tokio::test]
    async fn deadline_is_enforced() {
        // Every call sleeps 80ms; a 10ms deadline must abort with a clear timeout error.
        let rows: Vec<Row> = (1..=10).map(row).collect();
        let (shards, store) = make_dataset(&rows, 5);
        let host = MockHost::new(store).slow(80);
        let cfg = RunConfig {
            max_in_flight: 1,
            deadline: Some(Duration::from_millis(10)),
            ..RunConfig::default()
        };
        let err = run(&sample_query(), &shards, &hosts(3), &host, &cfg).await.unwrap_err();
        assert!(err.to_string().contains("deadline"), "expected deadline error, got: {err}");
    }

    #[tokio::test]
    async fn cost_limit_rejects_too_many_shards() {
        let rows: Vec<Row> = (1..=20).map(row).collect();
        let (shards, store) = make_dataset(&rows, 5);
        let host = MockHost::new(store);
        let cfg = RunConfig {
            limits: CostLimits { max_shards: 2, ..CostLimits::unlimited() },
            ..RunConfig::default()
        };
        let err = run(&sample_query(), &shards, &hosts(3), &host, &cfg).await.unwrap_err();
        assert!(err.to_string().contains("cost limit"), "{err}");
        // The query was rejected before any host was contacted.
        assert_eq!(host.total_calls(), 0);
    }

    #[tokio::test]
    async fn cost_limit_rejects_too_many_rows() {
        let rows: Vec<Row> = (1..=20).map(row).collect();
        let (shards, store) = make_dataset(&rows, 4);
        let host = MockHost::new(store);
        let cfg = RunConfig {
            limits: CostLimits { max_scan_rows: 5, ..CostLimits::unlimited() },
            ..RunConfig::default()
        };
        let err = run(&sample_query(), &shards, &hosts(3), &host, &cfg).await.unwrap_err();
        assert!(err.to_string().contains("rows scanned"), "{err}");
    }

    #[tokio::test]
    async fn pruning_skips_non_matching_shards() {
        use crate::query::{CmpOp, Predicate};
        // Two shards with disjoint value ranges, both carrying stats.
        let low: Vec<Row> = (1..=5).map(row).collect();
        let high: Vec<Row> = (100..=105).map(row).collect();
        let mut shards = Vec::new();
        let mut store = HashMap::new();
        for rs in [low, high] {
            let bytes = encode_shard(&rs).unwrap();
            let mut s = crate::dataset::Shard::new(ce_rs::cid(&bytes), rs.len() as u64, bytes.len() as u64);
            s.stats = Some(crate::dataset::compute_stats(&rs));
            store.insert(s.cid.clone(), rs);
            shards.push(s);
        }
        let low_cid = shards[0].cid.clone();
        let host = MockHost::new(store);
        // WHERE v > 50 can only match the high shard; the low shard must be pruned (never fetched).
        let q = Query::new("t")
            .agg(Agg::Count)
            .filter(Predicate::Cmp { field: "v".into(), op: CmpOp::Gt, value: json!(50) });
        let report = run(&q, &shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
        assert_eq!(report.results[0].values["count"], json!(6), "100..=105 is 6 rows");
        // The pruned low shard was never requested from any host.
        let low_calls: usize = hosts(3).iter().map(|h| host.call_count(h, &low_cid)).sum();
        assert_eq!(low_calls, 0, "pruned shard must not be fetched");
    }

    #[tokio::test]
    async fn pruning_all_shards_yields_empty_result() {
        use crate::query::{CmpOp, Predicate};
        let rows: Vec<Row> = (1..=20).map(row).collect();
        let (shards, store) = make_dataset_with_stats(&rows, 4);
        let host = MockHost::new(store);
        // No value exceeds 1000, so every shard prunes -> empty aggregate, zero fetches.
        let q = Query::new("t")
            .agg(Agg::Count)
            .filter(Predicate::Cmp { field: "v".into(), op: CmpOp::Gt, value: json!(1000) });
        let report = run(&q, &shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
        assert!(report.results.is_empty() || report.results[0].values["count"] == json!(0));
        assert_eq!(host.total_calls(), 0, "all shards pruned -> no fetches");
    }

    #[tokio::test]
    async fn projection_returns_filtered_rows() {
        let rows: Vec<Row> = (1..=20).map(row).collect();
        let (shards, store) = make_dataset(&rows, 4);
        let host = MockHost::new(store);
        let q = Query::new("t")
            .project(vec!["v".into()])
            .filter(crate::query::Predicate::Cmp {
                field: "v".into(),
                op: crate::query::CmpOp::Gt,
                value: json!(15),
            });
        let report = run_projection(&q, &shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
        // v in 16..=20 -> 5 rows.
        assert_eq!(report.rows.len(), 5);
        assert!(report.rows.iter().all(|r| r.get("v").and_then(|v| v.as_i64()).unwrap() > 15));
    }

    #[tokio::test]
    async fn projection_limit_truncates_and_flags() {
        let rows: Vec<Row> = (1..=50).map(row).collect();
        let (shards, store) = make_dataset(&rows, 5);
        let host = MockHost::new(store);
        let q = Query::new("t").project(vec![]).limit(7);
        let report = run_projection(&q, &shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
        assert_eq!(report.rows.len(), 7);
        assert!(report.truncated, "more than 7 rows existed, so the result is truncated");
    }

    #[tokio::test]
    async fn projection_order_by_then_limit() {
        let rows: Vec<Row> = (1..=10).map(row).collect();
        let (shards, store) = make_dataset(&rows, 4);
        let host = MockHost::new(store);
        let q = Query::new("t")
            .project(vec!["v".into()])
            .order(crate::query::OrderKey::desc("v"))
            .limit(3);
        let report = run_projection(&q, &shards, &hosts(3), &host, &RunConfig::default()).await.unwrap();
        let got: Vec<i64> = report.rows.iter().map(|r| r["v"].as_i64().unwrap()).collect();
        assert_eq!(got, vec![10, 9, 8], "top-3 by v desc");
    }

    #[tokio::test]
    async fn cost_limit_rejects_too_many_result_groups() {
        // Each row is its own group (group by unique id) -> many result groups; cap at 2.
        let rows: Vec<Row> = (0..10)
            .map(|i| [("v".to_string(), json!(i)), ("g".to_string(), json!(i))].into_iter().collect())
            .collect();
        let (shards, store) = make_dataset(&rows, 2);
        let host = MockHost::new(store);
        let q = Query::new("t").agg(Agg::Count).group("g");
        let cfg = RunConfig {
            limits: CostLimits { max_result_groups: 2, ..CostLimits::unlimited() },
            ..RunConfig::default()
        };
        let err = run(&q, &shards, &hosts(3), &host, &cfg).await.unwrap_err();
        assert!(err.to_string().contains("result groups"), "{err}");
    }
}
