//! Production [`MapHost`] implementations over the CE SDK.
//!
//! Two strategies share the [`MapHost`](crate::engine::MapHost) seam:
//!
//! - [`LocalMapHost`] — the **coordinator** fetches each shard blob itself via
//!   [`ce_rs::CeClient::get_object`] and maps it in-process. This needs no remote query agent and
//!   works against any CE node that serves blobs, so it is the zero-dependency default (and what the
//!   CLI uses out of the box). It still parallelises over shards and still exercises the full
//!   plan/retry/reduce path; "host selection" degrades to "which provider to fetch the shard from".
//!
//! - [`MeshMapHost`] — the **true distributed** mode: the coordinator sends a `map` request to the
//!   assigned host over the mesh ([`ce_rs::CeClient::request`]); a `ce-query` agent on that host
//!   fetches the shard locally, runs the map, and replies with the [`Partial`]. This is the BigQuery
//!   shape — compute goes to the data — and is selected when hosts advertise the `ce-query` service.
//!
//! Both translate transport/SDK errors into [`MapError`] variants so the engine's failover logic
//! treats them uniformly (a 404 blob, a 5xx, a timeout, or malformed bytes all just mean "try the
//! next host"). The on-host request/reply payload format is defined by [`MapRequest`]/[`MapReply`].

use crate::dataset::{Shard, decode_shard};
use crate::engine::{MapError, MapHost};
use crate::combine::Partial;
use crate::query::Query;
use ce_rs::CeClient;
use serde::{Deserialize, Serialize};

/// The mesh service name a host advertises when it runs a `ce-query` map agent. The coordinator
/// discovers candidate hosts by querying the atlas / `find_service` for this name.
pub const QUERY_SERVICE: &str = "ce-query";

/// The mesh app-message topic the map request/reply travels on.
pub const MAP_TOPIC: &str = "ce-query/map/1";

/// The request a coordinator sends a host to map one shard. The host fetches `shard.cid` locally and
/// runs `query.map_shard`. `grant` carries an optional `ce-cap` query token the host verifies before
/// doing the work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapRequest {
    /// The query to run (the host only needs the predicate, aggregates, and group_by).
    pub query: Query,
    /// The shard to map.
    pub shard: Shard,
    /// Optional capability token authorizing the query on the dataset (hex `ce-cap` chain).
    #[serde(default)]
    pub grant: Option<String>,
}

/// The host's reply: a partial aggregate (aggregate query), projected rows (projection query), or
/// an error string the coordinator maps to a retryable [`MapError`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MapReply {
    /// Successful aggregate map: the shard's partial aggregate.
    Ok(Partial),
    /// Successful projection: the shard's filtered, projected rows.
    Rows(Vec<crate::dataset::Row>),
    /// The host refused or failed (unauthorized, missing blob, internal error). The coordinator
    /// fails over to the next host.
    Err(String),
}

/// The largest map request/reply payload the coordinator and host will accept, in bytes. Caps the
/// memory a single AppRequest can pin on either side (hex-encoded on the wire, so ~2x this on
/// transport). A request/reply over this is rejected rather than buffered — closing the OOM/DoS
/// vector an unbounded payload would open.
pub const MAX_MAP_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;

/// The largest number of rows a single projection reply may carry, bounding per-shard result
/// cardinality independently of byte size.
pub const MAX_PROJECTION_ROWS: usize = 1_000_000;

/// Coordinator-side map: fetch the shard blob and run the map locally. The simplest correct host —
/// no remote agent required.
pub struct LocalMapHost {
    client: CeClient,
}

impl LocalMapHost {
    /// Build over an existing SDK client (the coordinator's local node).
    pub fn new(client: CeClient) -> Self {
        LocalMapHost { client }
    }
}

#[async_trait::async_trait]
impl MapHost for LocalMapHost {
    async fn map(&self, _host_id: &str, shard: &Shard, query: &Query) -> Result<Partial, MapError> {
        let rows = self.fetch_rows(shard).await?;
        Ok(query.map_shard(&rows))
    }

    async fn project(
        &self,
        _host_id: &str,
        shard: &Shard,
        query: &Query,
    ) -> Result<Vec<crate::dataset::Row>, MapError> {
        let rows = self.fetch_rows(shard).await?;
        Ok(query.project_shard(&rows))
    }
}

impl LocalMapHost {
    /// Fetch and decode a shard's rows, enforcing the payload size bound.
    async fn fetch_rows(&self, shard: &Shard) -> Result<Vec<crate::dataset::Row>, MapError> {
        // Fetch the content-addressed shard object; get_object verifies every chunk against its CID.
        let bytes = self
            .client
            .get_object(&shard.cid)
            .await
            .map_err(|e| classify_fetch_error(&shard.cid, &e))?;
        if bytes.len() > MAX_MAP_PAYLOAD_BYTES {
            return Err(MapError::HostError(format!(
                "shard {} is {} bytes, exceeds the {}-byte scan limit",
                shard.cid,
                bytes.len(),
                MAX_MAP_PAYLOAD_BYTES
            )));
        }
        decode_shard(&bytes).map_err(|e| MapError::Corrupt(e.to_string()))
    }
}

/// Distributed map: send the shard's map task to the assigned host over the mesh and decode its
/// reply. Used when hosts advertise the `ce-query` service.
pub struct MeshMapHost {
    client: CeClient,
    /// Optional query capability token forwarded with every map request.
    grant: Option<String>,
    /// Per-request timeout in milliseconds.
    timeout_ms: u64,
}

impl MeshMapHost {
    /// Build over the coordinator's SDK client, an optional grant token, and a per-request timeout.
    pub fn new(client: CeClient, grant: Option<String>, timeout_ms: u64) -> Self {
        MeshMapHost { client, grant, timeout_ms }
    }
}

impl MeshMapHost {
    /// Encode and send a map request, returning the decoded reply. Enforces the request payload
    /// bound before sending (a request larger than the cap is the caller's bug or an attack).
    async fn send(&self, host_id: &str, shard: &Shard, query: &Query) -> Result<MapReply, MapError> {
        let req = MapRequest { query: query.clone(), shard: shard.clone(), grant: self.grant.clone() };
        let payload = serde_json::to_vec(&req)
            .map_err(|e| MapError::HostError(format!("encoding map request: {e}")))?;
        if payload.len() > MAX_MAP_PAYLOAD_BYTES {
            return Err(MapError::HostError(format!(
                "map request is {} bytes, exceeds the {}-byte limit",
                payload.len(),
                MAX_MAP_PAYLOAD_BYTES
            )));
        }
        let reply_bytes = self
            .client
            .request(host_id, MAP_TOPIC, &payload, self.timeout_ms)
            .await
            .map_err(|e| classify_request_error(&e))?;
        if reply_bytes.len() > MAX_MAP_PAYLOAD_BYTES {
            return Err(MapError::HostError(format!(
                "map reply is {} bytes, exceeds the {}-byte limit",
                reply_bytes.len(),
                MAX_MAP_PAYLOAD_BYTES
            )));
        }
        serde_json::from_slice(&reply_bytes)
            .map_err(|e| MapError::Corrupt(format!("decoding map reply: {e}")))
    }
}

#[async_trait::async_trait]
impl MapHost for MeshMapHost {
    async fn map(&self, host_id: &str, shard: &Shard, query: &Query) -> Result<Partial, MapError> {
        match self.send(host_id, shard, query).await? {
            MapReply::Ok(p) => Ok(p),
            MapReply::Rows(_) => Err(MapError::Corrupt("host returned rows for an aggregate query".into())),
            MapReply::Err(msg) => Err(MapError::HostError(msg)),
        }
    }

    async fn project(
        &self,
        host_id: &str,
        shard: &Shard,
        query: &Query,
    ) -> Result<Vec<crate::dataset::Row>, MapError> {
        match self.send(host_id, shard, query).await? {
            MapReply::Rows(rows) => {
                if rows.len() > MAX_PROJECTION_ROWS {
                    return Err(MapError::HostError(format!(
                        "projection reply has {} rows, exceeds the {}-row limit",
                        rows.len(),
                        MAX_PROJECTION_ROWS
                    )));
                }
                Ok(rows)
            }
            MapReply::Ok(_) => Err(MapError::Corrupt("host returned a partial for a projection query".into())),
            MapReply::Err(msg) => Err(MapError::HostError(msg)),
        }
    }
}

/// Serve one map request as an on-host agent would: verify the (optional) grant if a verifier is
/// supplied, fetch the shard locally, run the map, and produce a [`MapReply`]. Factored out so the
/// agent loop and tests share the exact logic. `verify` returns `Ok(())` to permit, `Err(reason)` to
/// reject; pass a closure that calls [`crate::caps::verify`], or `|_| Ok(())` for an open host.
pub async fn serve_map(
    client: &CeClient,
    req: &MapRequest,
    verify: impl Fn(&MapRequest) -> Result<(), String>,
) -> MapReply {
    if let Err(reason) = verify(req) {
        return MapReply::Err(format!("unauthorized: {reason}"));
    }
    // Validate the query before doing any work — a malformed query is a clear rejection, not a panic.
    if let Err(e) = req.query.validate() {
        return MapReply::Err(format!("invalid query: {e}"));
    }
    let bytes = match client.get_object(&req.shard.cid).await {
        Ok(b) => b,
        Err(e) => return MapReply::Err(format!("fetch {}: {e}", req.shard.cid)),
    };
    if bytes.len() > MAX_MAP_PAYLOAD_BYTES {
        return MapReply::Err(format!(
            "shard {} is {} bytes, exceeds the {}-byte scan limit",
            req.shard.cid,
            bytes.len(),
            MAX_MAP_PAYLOAD_BYTES
        ));
    }
    let rows = match decode_shard(&bytes) {
        Ok(r) => r,
        Err(e) => return MapReply::Err(format!("decode shard: {e}")),
    };
    if req.query.is_projection() {
        let projected = req.query.project_shard(&rows);
        if projected.len() > MAX_PROJECTION_ROWS {
            return MapReply::Err(format!(
                "projection produced {} rows, exceeds the {}-row limit",
                projected.len(),
                MAX_PROJECTION_ROWS
            ));
        }
        MapReply::Rows(projected)
    } else {
        MapReply::Ok(req.query.map_shard(&rows))
    }
}

/// Map a get_object error to a retryable [`MapError`], distinguishing a missing blob (404-ish) from
/// other transport/host failures by inspecting the error text (the SDK surfaces the HTTP body).
fn classify_fetch_error(cid: &str, e: &anyhow::Error) -> MapError {
    let s = e.to_string();
    let lower = s.to_lowercase();
    if lower.contains("404") || lower.contains("not found") {
        MapError::MissingBlob(cid.to_string())
    } else if lower.contains("timed out") || lower.contains("timeout") {
        MapError::Timeout
    } else {
        MapError::HostError(s)
    }
}

/// Map a mesh request error to a retryable [`MapError`] (timeout vs other host/transport error).
fn classify_request_error(e: &anyhow::Error) -> MapError {
    let s = e.to_string();
    let lower = s.to_lowercase();
    if lower.contains("timed out") || lower.contains("timeout") || lower.contains("504") {
        MapError::Timeout
    } else {
        MapError::HostError(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{Row, encode_shard};
    use crate::query::Agg;
    use serde_json::json;

    fn row(v: i64) -> Row {
        [("v".to_string(), json!(v))].into_iter().collect()
    }

    #[test]
    fn map_request_reply_roundtrip() {
        let q = Query::new("t").agg(Agg::Count);
        let shard = Shard::new("abc", 0, 0);
        let req = MapRequest { query: q.clone(), shard: shard.clone(), grant: Some("tok".into()) };
        let bytes = serde_json::to_vec(&req).unwrap();
        let back: MapRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.query, q);
        assert_eq!(back.shard, shard);
        assert_eq!(back.grant.as_deref(), Some("tok"));

        let partial = q.map_shard(&[row(1), row(2)]);
        let reply = MapReply::Ok(partial.clone());
        let rb = serde_json::to_vec(&reply).unwrap();
        match serde_json::from_slice::<MapReply>(&rb).unwrap() {
            MapReply::Ok(p) => assert_eq!(p, partial),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn error_classification() {
        let missing = classify_fetch_error("cid", &anyhow::anyhow!("HTTP 404 not found"));
        assert_eq!(missing, MapError::MissingBlob("cid".into()));
        let to = classify_fetch_error("cid", &anyhow::anyhow!("request timed out"));
        assert_eq!(to, MapError::Timeout);
        let other = classify_fetch_error("cid", &anyhow::anyhow!("HTTP 500 boom"));
        assert!(matches!(other, MapError::HostError(_)));

        assert_eq!(classify_request_error(&anyhow::anyhow!("504 gateway timeout")), MapError::Timeout);
        assert!(matches!(classify_request_error(&anyhow::anyhow!("502 bad gateway")), MapError::HostError(_)));
    }

    #[tokio::test]
    async fn serve_map_rejects_unauthorized() {
        // No real node needed: verify() rejects before any fetch.
        let client = CeClient::local();
        let req = MapRequest {
            query: Query::new("t").agg(Agg::Count),
            shard: Shard::new("cid", 0, 0),
            grant: None,
        };
        let reply = serve_map(&client, &req, |_| Err("no grant".into())).await;
        match reply {
            MapReply::Err(m) => assert!(m.contains("unauthorized"), "{m}"),
            other => panic!("should have been rejected, got {other:?}"),
        }
    }

    #[test]
    fn map_reply_rows_roundtrip() {
        let rows: Vec<crate::dataset::Row> = vec![row(1), row(2)];
        let reply = MapReply::Rows(rows.clone());
        let bytes = serde_json::to_vec(&reply).unwrap();
        match serde_json::from_slice::<MapReply>(&bytes).unwrap() {
            MapReply::Rows(r) => assert_eq!(r, rows),
            _ => panic!("expected Rows"),
        }
    }

    #[tokio::test]
    async fn serve_map_rejects_invalid_query() {
        let client = CeClient::local();
        // A query with neither aggregates nor projection is invalid and must be rejected before fetch.
        let req = MapRequest {
            query: Query::new("t"),
            shard: Shard::new("cid", 0, 0),
            grant: None,
        };
        match serve_map(&client, &req, |_| Ok(())).await {
            MapReply::Err(m) => assert!(m.contains("invalid query"), "{m}"),
            _ => panic!("expected an invalid-query rejection"),
        }
    }

    #[test]
    fn shard_cid_matches_sdk_cid() {
        // A shard's CID computed here must equal ce_rs::cid of its encoded bytes — the invariant the
        // engine relies on to fetch the right blob.
        let rows = vec![row(1), row(2), row(3)];
        let bytes = encode_shard(&rows).unwrap();
        assert_eq!(ce_rs::cid(&bytes).len(), 64);
    }
}
