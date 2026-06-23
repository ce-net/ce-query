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
use ce_query::engine::{RunConfig, run};
use ce_query::mesh::{LocalMapHost, MapRequest, MeshMapHost, serve_map};
use ce_query::{discover_hosts, plan, register_dataset, sql};
use ce_rs::CeClient;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
        Cmd::Run { sql, distributed, grant, max_attempts } => {
            run_cmd(&sql, distributed, grant, max_attempts, &catalog_path, &cli.node).await
        }
        Cmd::Plan { sql } => plan_cmd(&sql, &catalog_path),
        Cmd::Grant { dataset, r#for, expires, nonce } => {
            grant_cmd(dataset, r#for, expires, nonce)
        }
        Cmd::Serve { require_grant, poll_ms } => serve_cmd(require_grant, poll_ms, &cli.node).await,
    }
}

async fn dataset_cmd(op: DatasetCmd, catalog_path: &Path, node: &str) -> Result<()> {
    let mut catalog = Catalog::load(catalog_path)?;
    match op {
        DatasetCmd::Add { name, from, shards } => {
            let bytes = std::fs::read(&from).with_context(|| format!("reading {from:?}"))?;
            let rows = ce_query::dataset::decode_shard(&bytes)
                .with_context(|| format!("parsing NDJSON {from:?}"))?;
            if rows.is_empty() {
                bail!("input {from:?} has no rows");
            }
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
            catalog.put(ds);
            catalog.save(catalog_path)?;
        }
        DatasetCmd::Ls => {
            if catalog.datasets.is_empty() {
                println!("(no datasets registered)");
            }
            for name in catalog.names() {
                let d = catalog.get(&name).expect("listed name exists");
                println!("{name}\t{} rows\t{} shards", d.total_rows(), d.shards.len());
            }
        }
        DatasetCmd::Show { name } => {
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
            catalog.remove(&name)?;
            catalog.save(catalog_path)?;
            println!("removed dataset `{name}`");
        }
    }
    Ok(())
}

async fn run_cmd(
    sql_str: &str,
    distributed: bool,
    grant: Option<String>,
    max_attempts: usize,
    catalog_path: &Path,
    node: &str,
) -> Result<()> {
    let query = sql::parse(sql_str)?;
    let catalog = Catalog::load(catalog_path)?;
    let ds = catalog.require(&query.dataset)?.clone();

    let client = CeClient::new(node.to_string());
    let hosts = discover_hosts(&client, distributed).await.unwrap_or_default();
    if hosts.is_empty() {
        bail!(
            "no candidate hosts found (atlas empty{}). Is the node running and mining?",
            if distributed { " for the ce-query service" } else { "" }
        );
    }

    let config = RunConfig { max_attempts_per_shard: max_attempts };
    let report = if distributed {
        let host = MeshMapHost::new(client.clone(), grant, 30_000);
        run(&query, &ds.shards, &hosts, &host, &config).await?
    } else {
        let host = LocalMapHost::new(client.clone());
        run(&query, &ds.shards, &hosts, &host, &config).await?
    };

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
        "[{} shard(s), {} attempt(s), {} redistributed]",
        ds.shards.len(),
        report.total_attempts,
        report.redistributed
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

async fn serve_cmd(require_grant: bool, poll_ms: u64, node: &str) -> Result<()> {
    let client = CeClient::new(node.to_string());
    let identity = load_identity()?;
    let self_id = identity.node_id();

    // Advertise the service and subscribe to the map topic so requests reach our inbox.
    client.advertise_service(ce_query::mesh::QUERY_SERVICE).await.ok();
    client.subscribe(ce_query::mesh::MAP_TOPIC).await.ok();
    tracing::info!(topic = %ce_query::mesh::MAP_TOPIC, "ce-query map agent serving");

    loop {
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
            let req: MapRequest = match serde_json::from_slice(&payload) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(err = %e, "malformed map request");
                    continue;
                }
            };
            let requester = parse_node_id(&msg.from).ok();
            let reply = serve_map(&client, &req, |r| {
                if !require_grant {
                    return Ok(());
                }
                let req_id = requester.ok_or_else(|| "unknown requester".to_string())?;
                let grant = r.grant.as_deref().ok_or_else(|| "grant required".to_string())?;
                caps::verify(
                    &self_id,
                    &[],
                    &[],
                    now_secs(),
                    &req_id,
                    &r.query.dataset,
                    grant,
                    &|_, _| false,
                )
            })
            .await;
            let reply_bytes = serde_json::to_vec(&reply).unwrap_or_default();
            if let Err(e) = client.reply(token, &reply_bytes).await {
                tracing::warn!(err = %e, "failed to send reply");
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(poll_ms)).await;
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
