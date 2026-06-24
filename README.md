# ce-query

**Distributed query / map-reduce (BigQuery-shaped) over CE content-addressed blob datasets.**

ce-query is an **application built on CE primitives** (the SDK/app tier, alongside `ce-storage`,
`ce-pin`, `ce-coord`, `swarm`, `rdev`) — **not** a node feature. It turns CE's flat,
content-addressed blob layer into queryable **datasets**: register a dataset by sharding its rows
into blobs, then run a `SELECT … WHERE … GROUP BY …` map-reduce that fans map tasks across atlas
hosts (compute goes to the data) and reduces the partial aggregates into a final answer.

> Pronounced like the rest of CE ("Sea"). Rows in, aggregates out, no per-TB-scanned tax, and the
> data never leaves the mesh.

## What it composes (it reinvents nothing)

| Concern | CE primitive used |
|---|---|
| Dataset storage (shards) | `ce-rs` `put_object`/`get_object` over the node `/blobs` store — 1 MiB chunks, content-addressed, hash-verified on read |
| Durability / availability | `ce-pin` (pin shard CIDs across N hosts) — orthogonal, reuse as-is |
| Host selection for map tasks | `ce-rs` atlas (`/atlas`) + `find_service` for the `ce-query` map agent |
| Fan-out map / collect reduce | `ce-rs` mesh `request`/`reply` to each assigned host (`MeshMapHost`) |
| Authorization | `ce-cap` signed, attenuating chains scoped to a dataset (`query:read`), offline-verifiable |
| Dataset catalog (`name -> shards`) | a local JSON map — the ce-coord `RMap` shape kept local for the owner |

No new node endpoints, no allowlists, no stored ip:port. Authorization between coordinator and host
is a `ce-cap` chain. Everything routes over the SDK.

## The architecture in one paragraph

A query splits into a pure **map** (`Query::map_shard` → a `Partial`) and a pure **reduce**
(`combine::reduce` of `Partial`s, then `Partial::finalize`). Every aggregate is a **monoid**
(`combine::Accum`: identity + associative, commutative merge) — which is exactly what makes the
engine distributable and fault-tolerant:

- **shards map independently** on different hosts (the planner assigns them by rendezvous hash);
- **partials merge in any order** (associativity + commutativity), so out-of-order mesh arrival is
  fine and the coordinator can reduce as results stream in;
- **a dropped host is handled** by redistributing its shard to the next-best candidate in its
  rendezvous ranking and retrying — a shard whose every candidate dropped is a hard error, so the
  engine **never returns a silently-wrong answer**.

The map/host seam is the `engine::MapHost` trait, with two implementations:

- `mesh::LocalMapHost` — the coordinator fetches each shard blob itself and maps it in-process. No
  remote agent needed; the zero-dependency default the CLI uses out of the box.
- `mesh::MeshMapHost` — the true distributed mode: send each map task to the assigned host over the
  mesh; a `ce-query serve` agent on that host fetches the shard locally and maps it. This is the
  BigQuery shape (compute goes to the data).

## Supported query surface

A builder API and an equivalent SQL-ish front end:

```text
SELECT <agg> [, <agg>]* | <col> [, <col>]* | *   FROM <dataset> [WHERE <pred>]
       [GROUP BY <col> [, <col>]*] [ORDER BY <col> [ASC|DESC] [, ...]] [LIMIT <n>]
<agg>   := COUNT(*) | SUM(col) | MIN(col) | MAX(col) | AVG(col)
<pred>  := <term> [ (AND|OR) <term> ]*
<term>  := [NOT] col <op> <literal>
<op>    := = | != | <> | < | <= | > | >=
<literal> := number | 'string' | "string" | true | false | null
```

Two query shapes, mutually exclusive:

- **Aggregate** — `SELECT COUNT(*), SUM(amount) FROM t GROUP BY region` — the map-reduce path.
- **Projection** — `SELECT region, amount FROM t WHERE amount >= 100 ORDER BY amount DESC LIMIT 10`
  or `SELECT *` — raw filtered/column-pruned rows, the most common BigQuery shape. Predicate and
  column pruning are applied **during the shard scan** (pushdown), so a host ships back only the rows
  and columns the query asked for. With a `LIMIT` and no `ORDER BY` the engine stops fetching shards
  once enough rows are collected.

`AVG` is carried as `(sum, count)` and divided only at finalisation, so it stays a monoid (averaging
averages would be wrong; merging sums and counts then dividing is right). Non-numeric / missing cells
are skipped for numeric aggregates, so one bad value never corrupts a result.

### Numeric correctness (money-safe)

`SUM` runs in a **lossless integer lane** (`i128`) while every value folded is an exact integer —
counts, ids, and money base units stay bit-exact and finalise to a JSON integer, honouring CE's
integer-base-unit money convention. The first fractional value promotes the lane to **Neumaier
compensated summation**, so even `SUM`/`AVG` over many large-magnitude fractional values is
associative and commutative to far higher precision than a bare `f64` fold — the property the engine
relies on so that a retried/redistributed shard never changes the answer at the ULP level.

### Result verification (redundancy / quorum)

On an open mesh a host could return a wrong partial. Pass `--redundancy K` (and optionally
`--quorum Q`) to map each shard on its top-`K` hosts and require their partials to **agree** before
accepting: `K=2` unanimity detects any single lying host (hard error on disagreement), while
`K=3 --quorum 2` is Byzantine-tolerant — two honest copies outvote one liar and the query still
returns the correct answer. This is BigQuery's "verification via redundancy", made real.

### Partition pruning, cost limits, deadlines, concurrency

- **Partition pruning** — each shard records per-column numeric min/max stats at registration; a
  `WHERE` range that cannot overlap a shard's range skips that shard entirely (never fetched).
- **Cost limits** — `--max-scan-bytes` / `--max-scan-rows` (and a result-group cap) reject a query
  *before* dispatch if it would scan/return more than the budget, bounding memory and spend (DoS
  guard). Map request/reply payloads are size-capped on both ends.
- **Deadlines** — `--deadline-ms` bounds the whole run.
- **Concurrency** — shards fan out concurrently (`--concurrency`, default 16), so wall-clock scales
  with host count, not shard count. Out-of-order partial arrival is fine (monoid reduce).

`ORDER BY` keys name an output column — a group-key column or an aggregate output name (`count`,
`sum_amount`, …) — and `LIMIT` keeps the top-N. Both are applied **once, after the distributed
reduce**, over the (already small) result rows, so they cost nothing on the map side and behave
identically for the local and mesh-distributed engines.

**Equi-joins across two datasets** live in the `join` module (a separate entry point, not the SQL
front end): an inner hash-join on a key pair, plus a **co-partitioned** `distributed_join` that hashes
both sides into the same buckets so each bucket pair joins independently (the map-reduce shape for
joins). A joined row stream feeds straight back into the aggregate engine, so `JOIN … GROUP BY …` is
`distributed_join(...)` followed by `map_shard`.

```rust
use ce_query::{hash_join, distributed_join, JoinKeys};
let joined = distributed_join(&users, &orders, &JoinKeys::on("uid"), 8); // co-partitioned
```

Subqueries and `HAVING` remain out of scope — this is the aggregate-first map-reduce core.

```rust
use ce_query::{Query, query::Agg, sql};
// Builder …
let q = Query::new("sales").agg(Agg::Sum("amount".into())).agg(Agg::Count).group("region");
// … or SQL-ish (equivalent):
let q2 = sql::parse("SELECT SUM(amount), COUNT(*) FROM sales GROUP BY region")?;
assert_eq!(q, q2);
```

## CLI

```bash
cd ~/ce-net/ce-query
cargo build --release            # binary at ../.cargo-shared/release/ce-query

# Register a dataset: shard a local NDJSON file into blobs and record it in the catalog.
ce-query dataset add sales --from sales.ndjson --shards 8

# Inspect the catalog.
ce-query dataset ls
ce-query dataset show sales

# See the shard-to-host assignment for a query without running it.
ce-query plan "SELECT SUM(amount), COUNT(*) FROM sales GROUP BY region"

# Run it (coordinator-local fetch by default).
ce-query run "SELECT SUM(amount), COUNT(*) FROM sales GROUP BY region"

# A projection query (raw rows): top-10 biggest orders.
ce-query run "SELECT region, amount FROM sales WHERE amount >= 100 ORDER BY amount DESC LIMIT 10"

# Run it distributed across ce-query host agents (compute goes to the data), verified on 3 hosts.
ce-query run "SELECT AVG(latency_ms) FROM logs WHERE status >= 500" \
    --distributed --grant <token> --redundancy 3 --quorum 2 --deadline-ms 30000

# Cost-bounded run (reject if it would scan more than the budget).
ce-query run "SELECT COUNT(*) FROM events" --max-scan-rows 1000000 --max-scan-bytes 536870912

# Host side: run a map agent that serves map requests over the mesh, honouring on-chain revocation.
ce-query serve --require-grant --revocation-refresh-secs 30 --max-concurrent 16

# Mint a ce-cap query token scoped to one dataset, valid for an hour.
ce-query grant sales --expires 3600
```

The `run` command prints the result as a JSON array (one object per group, or one per row for a
projection) and a one-line diagnostics summary on stderr
(`[N shard(s), M attempt(s), K redistributed, V verified]`).

### Runnable example (no node required)

```bash
cargo run --example quickstart   # shards examples/sales.ndjson in memory and runs two queries
```

## Authorization (`ce-cap`)

`ce-query grant <dataset>` mints a signed, attenuating capability whose ability is `query:read` and
whose `path_prefix` caveat names the dataset. A `ce-query serve --require-grant` host verifies the
presented chain **offline in microseconds** (`ce_cap::authorize`, no policy server), then checks the
dataset scope (exact match — a scope named `sales` never admits a sibling `sales-secret`). A holder
can attenuate (narrow to one dataset, add an expiry) and re-delegate. See `src/caps.rs`.

The serving agent consults the node's **on-chain revocation set** and refreshes it on an interval
(`--revocation-refresh-secs`), so a revoked-but-unexpired token stops being honoured on a
long-running host without a restart. Additional capability roots beyond the host's own key are
configured with `--root <node-id>`. Inbound requests are handled concurrently (bounded by
`--max-concurrent`) so one slow shard fetch never blocks the whole inbox, and request/reply payloads
are size-capped to close the OOM vector.

## Library layout

| Module | Responsibility |
|---|---|
| `dataset` | Dataset model, NDJSON shard encode/decode, round-robin sharding, per-shard min/max **stats**, name validation |
| `query` | Query model + pure **map** (`map_shard`) and **project** (`project_shard`), predicate/aggregate eval, integer/compensated numeric lanes, `ORDER BY`/`LIMIT` shaping |
| `combine` | The monoid algebra: `Accum` (int + Neumaier-compensated float), `Partial`, associative+commutative `merge`, `reduce`, finalisation, `KahanSum` |
| `order` | `ORDER BY` (stable, total, type-aware multi-key sort) + `LIMIT` over finalised rows and over raw projected rows |
| `join` | Inner equi-join across two datasets: in-memory `hash_join` + co-partitioned `distributed_join` |
| `plan` | Rendezvous-hash shard→host assignment, deterministic failover ranking, **partition pruning** (`shard_can_match`) |
| `engine` | The distributed executor: concurrent fan-out, retry/redistribute on drop, **quorum verification**, cost limits, deadlines, reduce; plus `run_projection` |
| `mesh` | `MapHost` implementations over `ce-rs` (`LocalMapHost`, `MeshMapHost`) + the on-host `serve_map`; payload size bounds |
| `caps` | `ce-cap` dataset-scoped `query:read` tokens (mint / inspect / verify) |
| `catalog` | Local JSON `name -> Dataset` catalog: schema-versioned, atomic fsync save, advisory file lock + `mutate` for concurrent-safe writes |

## Testing

The pure planning/combining core is the foundation, so it is validated from the start:

- **Unit tests on every public function** (happy + error path) across all modules.
- **Property tests** (`proptest`) for the algebra that the distribution correctness rests on:
  - the combiner `merge` is **associative** and **commutative**;
  - **shard-count invariance** — splitting rows into any number of shards and reducing gives the same
    answer as mapping them whole;
  - the empty/identity partial is **harmless** under arbitrary insertion order (safe retry);
  - a `Partial` survives a **JSON round-trip** unchanged (the mesh wire form);
  - rendezvous placement is **balanced**, and dropping one host **re-homes only its shards**;
  - **`ORDER BY` + `LIMIT`** — ordering is a sorted permutation, and `LIMIT n` is exactly the first
    `n` of the full order (no reordering leak);
  - **joins** — the co-partitioned `distributed_join` equals the single-pass `hash_join` for any
    partition count, and an equi-join emits the textbook `Σ countL(k)·countR(k)` cardinality.
- **Failure injection** in `engine` via a mock `MapHost`: dropped peers, transient 5xx, missing
  blobs, attempt caps, no-hosts, empty datasets, an invalid query, **redundancy disagreement**,
  **quorum majority over a liar**, **deadline expiry**, and **cost-limit breaches** — every path
  returns a clear error or a correct result, and **never panics**.
- **Concurrency / robustness**: bounded concurrent fan-out correctness and a wall-clock check that
  shards really parallelise; catalog atomic save, schema migration, advisory-lock contention; and
  partition-pruning soundness (a pruned shard is never fetched, all-pruned yields the empty result).
- **Integration** (`tests/end_to_end.rs`): drives the public API the way the CLI does — SQL →
  query → engine → reduce/shape — for both aggregate and projection queries, against an in-memory
  host (no node).

```bash
# Heavy compilation runs on the shared Hetzner box, not the laptop:
#   tools/remote-test.sh ce-query --clippy
# 151 lib tests + 4 integration tests + 2 doctests, zero clippy warnings.
```

## ce-query vs BigQuery — what is and isn't here

| Capability | ce-query | Notes |
|---|---|---|
| Aggregate map-reduce (COUNT/SUM/MIN/MAX/AVG, GROUP BY) | ✅ | Monoid algebra, distributed + fault-tolerant |
| Projection / `SELECT *` / `WHERE` / `ORDER BY` / `LIMIT` | ✅ | With predicate + column pushdown |
| Concurrent fan-out, deadlines, cost limits | ✅ | DoS-bounded |
| Result verification via redundancy / quorum | ✅ | Single-liar detection / Byzantine majority |
| Partition pruning (skip-on-stats) | ✅ | Numeric min/max per shard |
| Equi-joins | ✅ in the `join` module | in-memory + co-partitioned `distributed_join` |
| Money-safe numeric (integer lane + compensated float) | ✅ | exact integers; reorder-stable floats |
| Capability-gated, offline-verifiable authorization | ✅ | `ce-cap` `query:read`, with on-chain revocation in `serve` |
| `JOIN … ON` in the SQL front end | ❌ deferred | use the `join` module API directly |
| `HAVING`, `DISTINCT`, `COUNT(DISTINCT)`, subqueries, expressions in SELECT/WHERE | ❌ deferred | aggregate-first core |
| Columnar storage (Arrow/Parquet) | ❌ deferred | NDJSON shards; encoding-agnostic by design |
| Streaming reduce / pagination of huge result sets | ❌ deferred | results buffered, but cardinality-capped |

### Deferred / known caveats

- The SQL parser does not yet reach the `join` module (joins are a library API only) or support
  `HAVING`/`DISTINCT`/subqueries.
- Storage is NDJSON, not columnar; the planner/combiner are encoding-agnostic so a Parquet shard
  encoding is a future swap, not a query-semantics change.
- Result sets are materialised in memory (bounded by the result-group cap); there is no streaming
  reduce or result pagination yet.
- `AVG`/`SUM` over fractional values use Neumaier-compensated `f64`: reorder-stable and far more
  precise than a naive fold, but still floating-point. Integer/money sums are exact (`i128` lane).

## License

MIT.
