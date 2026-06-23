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
SELECT <agg> [, <agg>]* FROM <dataset> [WHERE <pred>]
       [GROUP BY <col> [, <col>]*] [ORDER BY <col> [ASC|DESC] [, ...]] [LIMIT <n>]
<agg>   := COUNT(*) | SUM(col) | MIN(col) | MAX(col) | AVG(col)
<pred>  := <term> [ (AND|OR) <term> ]*
<term>  := [NOT] col <op> <literal>
<op>    := = | != | <> | < | <= | > | >=
<literal> := number | 'string' | "string" | true | false | null
```

`AVG` is carried as `(sum, count)` and divided only at finalisation, so it stays a monoid (averaging
averages would be wrong; merging sums and counts then dividing is right). Non-numeric / missing cells
are skipped for numeric aggregates, so one bad value never corrupts a result.

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

# Run it distributed across ce-query host agents (compute goes to the data).
ce-query run "SELECT AVG(latency_ms) FROM logs WHERE status >= 500" --distributed --grant <token>

# Host side: run a map agent that serves map requests over the mesh.
ce-query serve --require-grant

# Mint a ce-cap query token scoped to one dataset, valid for an hour.
ce-query grant sales --expires 3600
```

The `run` command prints the result as a JSON array (one object per group) and a one-line
diagnostics summary on stderr (`[N shard(s), M attempt(s), K redistributed]`).

## Authorization (`ce-cap`)

`ce-query grant <dataset>` mints a signed, attenuating capability whose ability is `query:read` and
whose `path_prefix` caveat names the dataset. A `ce-query serve --require-grant` host verifies the
presented chain **offline in microseconds** (`ce_cap::authorize`, no policy server), then checks the
dataset scope. A holder can attenuate (narrow to one dataset, add an expiry) and re-delegate. See
`src/caps.rs`.

## Library layout

| Module | Responsibility |
|---|---|
| `dataset` | Dataset model, NDJSON shard encode/decode, round-robin sharding |
| `query` | The query model + the pure **map** (`map_shard`), predicate/aggregate evaluation, `ORDER BY`/`LIMIT` shaping |
| `combine` | The monoid algebra: `Accum`, `Partial`, associative+commutative `merge`, `reduce`, finalisation |
| `order` | `ORDER BY` (stable, total, type-aware multi-key sort) + `LIMIT` over the finalised rows |
| `join` | Inner equi-join across two datasets: in-memory `hash_join` + co-partitioned `distributed_join` |
| `plan` | Rendezvous-hash shard→host assignment + deterministic failover ranking |
| `engine` | The distributed executor: fan map tasks, retry/redistribute on drop, reduce |
| `mesh` | `MapHost` implementations over `ce-rs` (`LocalMapHost`, `MeshMapHost`) + the on-host `serve_map` |
| `caps` | `ce-cap` dataset-scoped `query:read` tokens (mint / inspect / verify) |
| `catalog` | Local JSON `name -> Dataset` catalog (atomic save) |

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
  blobs, attempt caps, no-hosts, empty datasets, and an invalid query — every path returns a clear
  error or a correct redistributed result, and **never panics**.

```bash
cargo test          # 109 unit/integration tests + 1 doctest, all green
cargo clippy --all-targets   # zero warnings
```

## License

MIT.
