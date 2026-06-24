# Changelog

All notable changes to `ce-query` are documented here.

## [Unreleased]

### Added

- **Projection queries**: `SELECT a, b FROM t WHERE … ORDER BY … LIMIT n` and `SELECT *` return raw
  filtered, column-pruned rows (`Query::project_shard`, `engine::run_projection`). Predicate and
  column pruning are applied during the shard scan (pushdown); with a `LIMIT` and no `ORDER BY` the
  engine stops fetching shards early once enough rows are collected.
- **Concurrent shard fan-out**: the engine maps shards concurrently with a bounded in-flight window
  (`RunConfig::max_in_flight`), so wall-clock scales with host count, not shard count.
- **Result verification via redundancy / quorum** (`RunConfig::redundancy` / `quorum`): map each
  shard on its top-`K` hosts and require their partials to agree. `K=2` detects any single lying
  host; `K=3, quorum=2` tolerates one liar (Byzantine majority).
- **Cost limits** (`CostLimits`): reject a query before dispatch if it would scan more than a
  byte/row/shard budget, and cap result-group cardinality. Bounds memory and spend (DoS guard).
- **Query deadline** (`RunConfig::deadline`): bound the whole run.
- **Partition pruning** (`plan::shard_can_match` + per-shard `ShardStats` min/max): skip shards a
  `WHERE` range provably cannot match (never fetched). Stats are computed at registration.
- **Money-safe numerics**: `SUM` runs in a lossless `i128` integer lane while inputs are integral and
  promotes to Neumaier-compensated (`KahanSum`) float otherwise; `AVG` carries a compensated sum.
  Aggregates are reorder-stable so retried/redistributed shards never change the answer.
- **Catalog hardening**: schema `version` field with forward migration; atomic fsync save with a
  per-pid temp file; an advisory lock + `Catalog::mutate` for concurrent-safe load-modify-save.
- **Serving agent hardening**: consults the on-chain revocation set (refreshed on an interval),
  accepts additional capability roots (`--root`), handles requests concurrently (bounded), bounds
  inbound payload size, validates the query before doing work, and surfaces encode failures as host
  errors instead of replying with empty bytes.
- **Map payload bounds**: `MAX_MAP_PAYLOAD_BYTES` / `MAX_PROJECTION_ROWS` enforced on both ends.
- **Dataset name validation**: charset/length-checked names (rejects whitespace, path separators).
- `examples/` (a runnable, node-free `quickstart` over a bundled `sales.ndjson`) and an
  `tests/end_to_end.rs` integration suite driving the public API the way the CLI does.

### Changed

- `Shard` carries an optional `stats` field; `Dataset`/`Shard`/`ShardTask` no longer derive `Eq`
  (stats hold `f64`). Use `Shard::new` / `Dataset::add_shard_with_stats`.
- `RunReport` gained a `verified` count; the `run` CLI prints `… , V verified`.

### Notes / deferred

- `JOIN … ON` is not yet wired into the SQL front end (use the `join` module API directly).
- `HAVING`, `DISTINCT`, subqueries, and SELECT/WHERE expressions remain out of scope.
- Storage is NDJSON (not columnar); results are materialised in memory (cardinality-capped, no
  streaming reduce / pagination yet).
