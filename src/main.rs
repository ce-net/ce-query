//! `ce-query` — CLI for the distributed query / map-reduce engine over CE blob datasets.
//!
//! Subcommands:
//! - `dataset add <name> --from <file.ndjson> [--shards N]` — shard a local NDJSON file into blobs
//!   and register the dataset in the local catalog.
//! - `dataset ls` / `dataset show <name>` / `dataset rm <name>` — inspect the catalog.
//! - `run "<sql>"` — parse a SQL-ish query, plan the shard assignment, run map-reduce over the mesh
//!   (coordinator-local fetch by default; `--distributed` to fan to `ce-query` host agents), print
//!   the result groups as JSON.
//! - `plan "<sql>"` — show the shard-to-host assignment without executing (planning visibility).
//! - `grant <dataset> [--for <node>] [--expires <secs>]` — mint a `ce-cap` query token for a dataset.
//! - `serve` — run a host-side map agent: accept `ce-query/map/1` requests, fetch + map locally.

use anyhow::{Context, Result, bail};
use ce_query::caps::{self, Scope};
use ce_query::catalog::{Catalog, default_catalog_path};
use ce_query::engine::{CostLimits, RunConfig, run};
use ce_query::mesh::{LocalMapHost, MapRequest, MeshMapHost, serve_map};
use ce_query::{discover_hosts, plan, register_dataset, sql};
use ce_rs::CeClient;
use clap::{Parser, Subcommand};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, Semaphore};

#[derive(Parser)]
#[command(
    name = "ce-query",
    about = "Distributed query / map-reduce over CE content-addressed blob datasets",
    version
)]
struct Cli {
    /// Override the catalog path (default: <CE data dir>/query/catalog.json).
    #[arg(long, global = true)]
    catalog: Option<PathBuf>,
    /// CE node HTTP API base URL.
    #[arg(long, global = true, default_value = ce_rs::DEFAULT_BASE_URL)]
    node: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dataset catalog operations.
    Dataset {
        #[command(subcommand)]
        op: DatasetCmd,
    },
    /// Run a SQL-ish query over a registered dataset (map-reduce across the mesh).
    Run {
        /// The query, e.g. "SELECT SUM(amount), COUNT(*) FROM sales GROUP BY region".
        sql: String,
        /// Fan map tasks to remote `ce-query` host agents instead of fetching shards locally.
        #[arg(long)]
        distributed: bool,
        /// Capability token (hex `ce-cap` chain) authorizing the query, forwarded to hosts.
        #[arg(long)]
        grant: Option<String>,
        /// Max hosts to try per shard before failing it.
        #[arg(long, default_value_t = 4)]
        max_attempts: usize,
        /// Maximum shards mapped concurrently (fan-out width).
        #[arg(long, default_value_t = 16)]
        concurrency: usize,
        /// Redundancy factor K: map each shard on K hosts and require agreement (result verification).
        #[arg(long, default_value_t = 1)]
        redundancy: usize,
        /// Agreeing copies required under redundancy (0 = unanimity).
        #[arg(long, default_value_t = 0)]
        quorum: usize,
        /// Overall query deadline in milliseconds (0 = none).
        #[arg(long, default_value_t = 0)]
        deadline_ms: u64,
        /// Maximum total bytes the query may scan (0 = the built-in default ceiling).
        #[arg(long, default_value_t = 0)]
        max_scan_bytes: u64,
        /// Maximum total rows the query may scan (0 = the built-in default ceiling).
        #[arg(long, default_value_t = 0)]
        max_scan_rows: u64,
    },
    /// Show the shard-to-host assignment for a query without executing it.
    Plan {
        /// The query string (only the FROM dataset is used for planning).
        sql: String,
    },
    /// Mint a `ce-cap` query token scoped to a dataset.
    Grant {
        /// Dataset name to scope the token to (omit for all datasets).
        dataset: Option<String>,
        /// Audience node id (hex) the token is issued to (default: a bearer token to self).
        #[arg(long)]
        r#for: Option<String>,
        /// Expiry in seconds from now (0 = never).
        #[arg(long, default_value_t = 0)]
        expires: u64,
        /// Unique nonce for on-chain revocability.
        #[arg(long, default_value_t = 1)]
        nonce: u64,
    },
    /// Run a host-side map agent (accept and serve map requests over the mesh).
    Serve {
        /// Require a valid `query:read` capability on every request (open host if omitted).
        #[arg(long)]
        require_grant: bool,
        /// Poll interval in milliseconds for the inbox.
        #[arg(long, default_value_t = 500)]
        poll_ms: u64,
        /// Seconds between refreshes of the on-chain revocation set (so revocations take effect on a
        /// long-running agent without a restart).
        #[arg(long, default_value_t = 30)]
        revocation_refresh_secs: u64,
        /// Additional accepted capability root node ids (hex), beyond this host's own key. Repeatable.
        #[arg(long = "root")]
        roots: Vec<String>,
        /// Maximum map requests handled concurrently (bounds head-of-line blocking and memory).
        #[arg(long, default_value_t = 16)]
        max_concurrent: usize,
    },
}

#[derive(Subcommand)]
enum DatasetCmd {
    /// Shard a local NDJSON file into blobs and register it.
    Add {
        /// Dataset name.
        name: String,
        /// Local NDJSON file (one JSON object per line).
        #[arg(long)]
        from: PathBuf,
        /// Number of shards to split the rows into.
        #[arg(long, default_value_t = 4)]
        shards: usize,
    },
    /// List registered datasets.
    Ls,
    /// Show a dataset's shards and counts.
    Show {
        /// Dataset name.
        name: String,
    },
    /// Remove a dataset from the catalog (shards' blobs are left content-addressed).
    Rm {
        /// Dataset name.
        name: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let catalog_path = cli.catalog.clone().unwrap_or_else(default_catalog_path);

    match cli.cmd {
        Cmd::Dataset { op } => dataset_cmd(op, &catalog_path, &cli.node).await,
        Cmd::Run {
            sql,
            distributed,
            grant,
            max_attempts,
            concurrency,
            redundancy,
            quorum,
            deadline_ms,
            max_scan_bytes,
            max_scan_rows,
        } => {
            let opts = RunOpts {
                distributed,
                grant,
                max_attempts,
                concurrency,
                redundancy,
                quorum,
                deadline_ms,
                max_scan_bytes,
                max_scan_rows,
            };
            run_cmd(&sql, opts, &catalog_path, &cli.node).await
        }
        Cmd::Plan { sql } => plan_cmd(&sql, &catalog_path),
        Cmd::Grant { dataset, r#for, expires, nonce } => {
            grant_cmd(dataset, r#for, expires, nonce)
        }
        Cmd::Serve { require_grant, poll_ms, revocation_refresh_secs, roots, max_concurrent } => {
            serve_cmd(
                require_grant,
                poll_ms,
                revocation_refresh_secs,
                roots,
                max_concurrent,
                &cli.node,
            )
            .await
        }
    }
}

/// Parsed `run` options, threaded into [`run_cmd`].
struct RunOpts {
    distributed: bool,
    grant: Option<String>,
    max_attempts: usize,
    concurrency: usize,
    redundancy: usize,
    quorum: usize,
    deadline_ms: u64,
    max_scan_bytes: u64,
    max_scan_rows: u64,
}

async fn dataset_cmd(op: DatasetCmd, catalog_path: &Path, node: &str) -> Result<()> {
    match op {
        DatasetCmd::Add { name, from, shards } => {
            let bytes = std::fs::read(&from).with_context(|| format!("reading {from:?}"))?;
            let rows = ce_query::dataset::decode_shard(&bytes)
                .with_context(|| format!("parsing NDJSON {from:?}"))?;
            if rows.is_empty() {
                bail!("input {from:?} has no rows");
            }
            // Upload shards (network I/O) before taking the catalog lock, so the lock is held only
            // for the brief load-insert-save window.
            let client = CeClient::new(node.to_string());
            let ds = register_dataset(&client, &name, Vec::new(), &rows, shards).await?;
            println!(
                "registered dataset `{}`: {} rows in {} shard(s)",
                ds.name,
                ds.total_rows(),
                ds.shards.len()
            );
            for (i, s) in ds.shards.iter().enumerate() {
                println!("  shard[{i}] {} ({} rows, {} bytes)", s.cid, s.rows, s.bytes);
            }
            Catalog::mutate(catalog_path, |c| {
                c.put(ds);
                Ok(())
            })?;
        }
        DatasetCmd::Ls => {
            let catalog = Catalog::load(catalog_path)?;
            if catalog.datasets.is_empty() {
                println!("(no datasets registered)");
            }
            for (name, d) in &catalog.datasets {
                println!("{name}\t{} rows\t{} shards", d.total_rows(), d.shards.len());
            }
        }
        DatasetCmd::Show { name } => {
            let catalog = Catalog::load(catalog_path)?;
            let d = catalog.require(&name)?;
            println!("dataset `{}`", d.name);
            println!("  rows:   {}", d.total_rows());
            println!("  bytes:  {}", d.total_bytes());
            println!("  shards: {}", d.shards.len());
            for (i, s) in d.shards.iter().enumerate() {
                println!("    [{i}] {} ({} rows, {} bytes)", s.cid, s.rows, s.bytes);
            }
        }
        DatasetCmd::Rm { name } => {
            Catalog::mutate(catalog_path, |c| c.remove(&name).map(|_| ()))?;
            println!("removed dataset `{name}`");
        }
    }
    Ok(())
}

async fn run_cmd(sql_str: &str, opts: RunOpts, catalog_path: &Path, node: &str) -> Result<()> {
    let query = sql::parse(sql_str)?;
    let catalog = Catalog::load(catalog_path)?;
    let ds = catalog.require(&query.dataset)?.clone();

    let client = CeClient::new(node.to_string());
    let hosts = discover_hosts(&client, opts.distributed).await.unwrap_or_default();
    if hosts.is_empty() {
        bail!(
            "no candidate hosts found (atlas empty{}). Is the node running and mining?",
            if opts.distributed { " for the ce-query service" } else { "" }
        );
    }

    let mut limits = CostLimits::default();
    if opts.max_scan_bytes != 0 {
        limits.max_scan_bytes = opts.max_scan_bytes;
    }
    if opts.max_scan_rows != 0 {
        limits.max_scan_rows = opts.max_scan_rows;
    }
    let config = RunConfig {
        max_attempts_per_shard: opts.max_attempts,
        max_in_flight: opts.concurrency,
        redundancy: opts.redundancy,
        quorum: opts.quorum,
        deadline: if opts.deadline_ms == 0 {
            None
        } else {
            Some(std::time::Duration::from_millis(opts.deadline_ms))
        },
        limits,
    };

    // Build the chosen host implementation once, then dispatch projection vs aggregate.
    enum Host {
        Mesh(MeshMapHost),
        Local(LocalMapHost),
    }
    let host = if opts.distributed {
        Host::Mesh(MeshMapHost::new(client.clone(), opts.grant, 30_000))
    } else {
        Host::Local(LocalMapHost::new(client.clone()))
    };
    let map_host: &dyn ce_query::MapHost = match &host {
        Host::Mesh(h) => h,
        Host::Local(h) => h,
    };

    if query.is_projection() {
        let report = ce_query::run_projection(&query, &ds.shards, &hosts, map_host, &config).await?;
        let out: Vec<serde_json::Value> =
            report.rows.iter().map(|r| serde_json::Value::Object(r.clone().into_iter().collect())).collect();
        println!("{}", serde_json::to_string_pretty(&out)?);
        eprintln!(
            "[{} shard(s), {} attempt(s), {} row(s){}]",
            ds.shards.len(),
            report.total_attempts,
            report.rows.len(),
            if report.truncated { ", truncated" } else { "" }
        );
        return Ok(());
    }

    let report = run(&query, &ds.shards, &hosts, map_host, &config).await?;

    // Emit results as a JSON array (one object per group), plus a diagnostics line on stderr.
    let out: Vec<serde_json::Value> = report
        .results
        .iter()
        .map(|g| {
            let mut obj = serde_json::Map::new();
            for (i, key) in query.group_by.iter().enumerate() {
                obj.insert(key.clone(), serde_json::Value::String(g.key.get(i).cloned().unwrap_or_default()));
            }
            for (k, v) in &g.values {
                obj.insert(k.clone(), v.clone());
            }
            serde_json::Value::Object(obj)
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&out)?);
    eprintln!(
        "[{} shard(s), {} attempt(s), {} redistributed, {} verified]",
        ds.shards.len(),
        report.total_attempts,
        report.redistributed,
        report.verified
    );
    Ok(())
}

fn plan_cmd(sql_str: &str, catalog_path: &Path) -> Result<()> {
    let query = sql::parse(sql_str)?;
    let catalog = Catalog::load(catalog_path)?;
    let ds = catalog.require(&query.dataset)?;
    // Plan against the dataset's shards using placeholder host ids so users can see the assignment
    // shape offline; a live `run` uses the real atlas hosts.
    let hosts: Vec<String> = (0..4).map(|i| format!("host{i:02}")).collect();
    let tasks = plan::plan(&ds.shards, &hosts);
    println!("query plan for `{}` ({} shards over {} sample hosts):", query.dataset, ds.shards.len(), hosts.len());
    for (i, t) in tasks.iter().enumerate() {
        println!(
            "  shard[{i}] {} -> {} (fallbacks: {})",
            t.shard.cid,
            t.host().unwrap_or("<none>"),
            t.ranked_hosts.iter().skip(1).cloned().collect::<Vec<_>>().join(", ")
        );
    }
    for (host, count) in plan::load_summary(&tasks) {
        println!("  {host}: {count} primary shard(s)");
    }
    Ok(())
}

fn grant_cmd(dataset: Option<String>, audience: Option<String>, expires: u64, nonce: u64) -> Result<()> {
    let identity = load_identity()?;
    let aud = match audience {
        Some(hex_id) => parse_node_id(&hex_id)?,
        None => identity.node_id(),
    };
    let scope = match dataset {
        Some(d) => Scope::dataset(d),
        None => Scope::all(),
    };
    let not_after = if expires == 0 { 0 } else { now_secs() + expires };
    let token = caps::mint(&identity, aud, &scope, not_after, nonce)?;
    println!("{token}");
    Ok(())
}

async fn serve_cmd(
    require_grant: bool,
    poll_ms: u64,
    revocation_refresh_secs: u64,
    roots_hex: Vec<String>,
    max_concurrent: usize,
    node: &str,
) -> Result<()> {
    let client = CeClient::new(node.to_string());
    let identity = load_identity()?;
    let self_id = identity.node_id();

    // Parse the accepted capability roots (beyond this host's own key).
    let mut accepted_roots = Vec::new();
    for r in &roots_hex {
        accepted_roots.push(parse_node_id(r).with_context(|| format!("parsing --root `{r}`"))?);
    }
    let accepted_roots = Arc::new(accepted_roots);

    // The on-chain revocation set, refreshed periodically so a revocation takes effect on a long-
    // running agent without a restart. Keyed by (issuer NodeId, nonce).
    let revoked: Arc<Mutex<HashSet<(ce_identity::NodeId, u64)>>> = Arc::new(Mutex::new(HashSet::new()));
    let last_refresh = Arc::new(AtomicU64::new(0));
    refresh_revocations(&client, &revoked).await;
    last_refresh.store(now_secs(), Ordering::Relaxed);

    // Advertise the service and subscribe to the map topic so requests reach our inbox.
    client.advertise_service(ce_query::mesh::QUERY_SERVICE).await.ok();
    client.subscribe(ce_query::mesh::MAP_TOPIC).await.ok();
    tracing::info!(topic = %ce_query::mesh::MAP_TOPIC, require_grant, "ce-query map agent serving");

    let client = Arc::new(client);
    let self_id = Arc::new(self_id);
    let limiter = Arc::new(Semaphore::new(max_concurrent.max(1)));

    loop {
        // Refresh the revocation set on its interval (cheap, off the hot path).
        if revocation_refresh_secs != 0
            && now_secs().saturating_sub(last_refresh.load(Ordering::Relaxed)) >= revocation_refresh_secs
        {
            refresh_revocations(&client, &revoked).await;
            last_refresh.store(now_secs(), Ordering::Relaxed);
        }

        let messages = client.messages().await.unwrap_or_default();
        for msg in messages {
            if msg.topic != ce_query::mesh::MAP_TOPIC {
                continue;
            }
            let Some(token) = msg.reply_token else { continue };
            let payload = match msg.payload() {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(err = %e, "bad request payload");
                    continue;
                }
            };
            // Bound request body size before deserialization (DoS guard).
            if payload.len() > ce_query::mesh::MAX_MAP_PAYLOAD_BYTES {
                tracing::warn!(bytes = payload.len(), "oversized map request rejected");
                let reply = ce_query::mesh::MapReply::Err("request too large".into());
                if let Ok(bytes) = serde_json::to_vec(&reply) {
                    let _ = client.reply(token, &bytes).await;
                }
                continue;
            }
            let req: MapRequest = match serde_json::from_slice(&payload) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(err = %e, "malformed map request");
                    continue;
                }
            };
            let requester = parse_node_id(&msg.from).ok();

            // Handle each request concurrently (bounded), so one slow shard fetch does not block the
            // whole inbox (head-of-line blocking).
            let permit = match Arc::clone(&limiter).acquire_owned().await {
                Ok(p) => p,
                Err(_) => continue,
            };
            let client = Arc::clone(&client);
            let self_id = Arc::clone(&self_id);
            let roots = Arc::clone(&accepted_roots);
            let revoked = Arc::clone(&revoked);
            tokio::spawn(async move {
                let _permit = permit; // held for the duration of the request
                let revoked_set = revoked.lock().await.clone();
                let reply = serve_map(&client, &req, |r| {
                    if !require_grant {
                        return Ok(());
                    }
                    let req_id = requester.ok_or_else(|| "unknown requester".to_string())?;
                    let grant = r.grant.as_deref().ok_or_else(|| "grant required".to_string())?;
                    caps::verify(
                        &self_id,
                        &roots,
                        &[],
                        now_secs(),
                        &req_id,
                        &r.query.dataset,
                        grant,
                        &|issuer: &ce_identity::NodeId, nonce: u64| {
                            revoked_set.contains(&(*issuer, nonce))
                        },
                    )
                })
                .await;
                // Surface a serialization failure as a host error rather than replying with empty
                // bytes (which the coordinator would mis-decode as Corrupt).
                let reply_bytes = match serde_json::to_vec(&reply) {
                    Ok(b) => b,
                    Err(e) => serde_json::to_vec(&ce_query::mesh::MapReply::Err(format!(
                        "encoding reply: {e}"
                    )))
                    .unwrap_or_else(|_| b"{\"Err\":\"encode failure\"}".to_vec()),
                };
                if let Err(e) = client.reply(token, &reply_bytes).await {
                    tracing::warn!(err = %e, "failed to send reply");
                }
            });
        }
        tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
    }
}

/// Refresh the in-memory revocation set from the node's on-chain revocation list. Issuer ids that do
/// not parse as 64-hex are skipped (malformed entries can never match a real chain).
async fn refresh_revocations(
    client: &CeClient,
    revoked: &Mutex<HashSet<(ce_identity::NodeId, u64)>>,
) {
    match client.revoked().await {
        Ok(list) => {
            let mut set = HashSet::with_capacity(list.len());
            for (issuer_hex, nonce) in list {
                if let Ok(id) = parse_node_id(&issuer_hex) {
                    set.insert((id, nonce));
                }
            }
            *revoked.lock().await = set;
        }
        Err(e) => tracing::warn!(err = %e, "could not refresh revocation set; keeping previous"),
    }
}

// ----- small helpers -----

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Load the CE identity from the standard data dir (for minting/verifying capabilities).
fn load_identity() -> Result<ce_identity::Identity> {
    let dir = directories::ProjectDirs::from("", "", "ce")
        .map(|d| d.data_dir().join("identity"))
        .context("resolving CE data dir")?;
    ce_identity::Identity::load_or_generate(&dir).context("loading CE identity")
}

/// Parse a 64-hex node id into a [`ce_identity::NodeId`].
fn parse_node_id(hex_id: &str) -> Result<ce_identity::NodeId> {
    let bytes = hex::decode(hex_id.trim()).context("node id is not valid hex")?;
    let arr: [u8; 32] = bytes.as_slice().try_into().context("node id must be 32 bytes (64 hex chars)")?;
    Ok(arr)
}
