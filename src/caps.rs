//! Dataset-scoped query authorization — a `ce-cap` capability granting `query:read` on a dataset,
//! encoded as a portable token. The map-task analog of ce-storage's presigned links.
//!
//! When the coordinator fans a map task to a host, the host must decide whether the requester is
//! allowed to query that dataset *before* fetching shards and burning CPU. The answer is a signed,
//! attenuating `ce-cap` chain: the dataset owner mints a capability whose ability is `query:read`,
//! whose resource is the owning node, and whose `path_prefix` caveat names the dataset. The host
//! verifies it offline in microseconds via [`ce_cap::authorize`] — no policy server, no shared
//! secret — and a holder can attenuate it (narrow to one dataset, add an expiry) and re-delegate.
//!
//! Ability used by this app (opaque to `ce-cap`):
//! - `query:read` — run map-reduce queries over the scoped dataset(s).
//!
//! The dataset scope lives in the `path_prefix` caveat. A capability scoped to `"sales"` covers
//! exactly the dataset named `sales`; an empty scope covers every dataset under the owner. The host
//! enforces the dataset match ([`scope_allows`]) on top of the `ce-cap` chain check.

use anyhow::{Context, Result};
use ce_cap::{Caveats, Resource, SignedCapability};
use ce_identity::{Identity, NodeId};

/// Ability string: run queries over the scoped dataset(s).
pub const ABILITY_QUERY: &str = "query:read";

/// A parsed query scope: which dataset a capability covers (empty = every dataset under the owner).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scope {
    /// Dataset name the capability is scoped to (empty string = all datasets).
    pub dataset: String,
}

impl Scope {
    /// A scope covering exactly one dataset.
    pub fn dataset(name: impl Into<String>) -> Scope {
        Scope { dataset: name.into() }
    }

    /// A scope covering every dataset under the owner.
    pub fn all() -> Scope {
        Scope { dataset: String::new() }
    }

    /// Encode as the `path_prefix` caveat string.
    pub fn to_caveat(&self) -> String {
        self.dataset.clone()
    }

    /// Parse a `path_prefix` caveat string back into a scope.
    pub fn from_caveat(s: &str) -> Scope {
        Scope { dataset: s.to_string() }
    }
}

/// Does this scope permit querying `dataset`? True iff the scope is empty (all) or names **exactly**
/// `dataset`. This is the app caveat enforcement that `ce-cap` defers to the action.
///
/// EXACT match is deliberate and is the chosen, pinned semantics for this app. A dataset scope is a
/// flat name with no hierarchical sub-structure, so there is no "scope `sales` covers `sales/q1`"
/// relationship to honor. This is intentionally *stricter* than `ce-cap`'s `path_prefix` caveat,
/// which narrows by raw prefix during attenuation: a chain may legitimately attenuate `path_prefix`
/// from `""` (all) to `"sales"`, but it can never attenuate to a wider set, so the leaf scope this
/// function sees is always at least as tight as every ancestor. Because we then require an exact
/// equality here, a scope named `dataset` can never admit a sibling like `dataset-secret` (which a
/// boundary-unaware `starts_with` would wrongly allow). The empty scope is the only widening, and it
/// is explicit (all datasets under the owner). See the `scope_*` regression tests below, which pin
/// that prefix-style and sibling names do NOT widen the scope.
pub fn scope_allows(scope: &Scope, dataset: &str) -> bool {
    scope.dataset.is_empty() || scope.dataset == dataset
}

/// Mint a query capability: a single self-issued capability granting `query:read` on `scope`, valid
/// until `not_after` (unix seconds, 0 = no expiry), as a portable hex token.
///
/// `owner` is the dataset-owning identity (the chain root — a node always accepts its own key as a
/// root). `audience` is the holder the token is issued to (pass the owner's own node id for a bearer
/// token, or a specific node id to bind it). `nonce` should be unique per issued token for on-chain
/// revocability.
pub fn mint(
    owner: &Identity,
    audience: NodeId,
    scope: &Scope,
    not_after: u64,
    nonce: u64,
) -> Result<String> {
    let caveats = Caveats {
        not_after,
        path_prefix: Some(scope.to_caveat()),
        ..Default::default()
    };
    let cap = SignedCapability::issue(
        owner,
        audience,
        vec![ABILITY_QUERY.to_string()],
        Resource::Node(owner.node_id()),
        caveats,
        nonce,
        None,
    );
    Ok(ce_cap::encode_chain(&[cap]))
}

/// Decode a token into its leaf abilities and dataset scope for inspection. Does not verify the
/// signature/expiry — call [`verify`] for that.
pub fn inspect(token: &str) -> Result<(Vec<String>, Scope)> {
    let chain = ce_cap::decode_chain(token).context("decoding query capability")?;
    let leaf = chain.last().context("empty capability chain")?;
    let scope = leaf
        .cap
        .caveats
        .path_prefix
        .as_deref()
        .map(Scope::from_caveat)
        .unwrap_or_else(Scope::all);
    Ok((leaf.cap.abilities.clone(), scope))
}

/// Verify a presented query token against a serving host's identity for `dataset`.
///
/// Runs the full `ce-cap` chain check (signature, attenuation, temporal caveats, revocation) rooted
/// at `self_id` (or `accepted_roots`), then enforces the app-level dataset scope caveat. The
/// requester is the leaf audience. `now` is unix seconds; `is_revoked` consults the on-chain set.
#[allow(clippy::too_many_arguments)]
pub fn verify(
    self_id: &NodeId,
    accepted_roots: &[NodeId],
    self_tags: &[String],
    now: u64,
    requester: &NodeId,
    dataset: &str,
    token: &str,
    is_revoked: &dyn Fn(&NodeId, u64) -> bool,
) -> Result<(), String> {
    let chain = ce_cap::decode_chain(token).map_err(|e| e.to_string())?;
    ce_cap::authorize(
        self_id,
        accepted_roots,
        self_tags,
        now,
        requester,
        ABILITY_QUERY,
        &chain,
        is_revoked,
    )?;
    let leaf = chain.last().ok_or_else(|| "empty chain".to_string())?;
    let scope = leaf
        .cap
        .caveats
        .path_prefix
        .as_deref()
        .map(Scope::from_caveat)
        .unwrap_or_else(Scope::all);
    if !scope_allows(&scope, dataset) {
        return Err(format!("token scope `{}` does not cover dataset `{dataset}`", scope.dataset));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ce_identity::Identity;

    fn ident(seed: &str) -> Identity {
        let dir = std::env::temp_dir().join(format!("ce-query-cap-{}-{}", seed, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let id = Identity::load_or_generate(&dir).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        id
    }

    fn never_revoked(_: &NodeId, _: u64) -> bool {
        false
    }

    #[test]
    fn scope_allows_logic() {
        assert!(scope_allows(&Scope::all(), "anything"));
        assert!(scope_allows(&Scope::dataset("sales"), "sales"));
        assert!(!scope_allows(&Scope::dataset("sales"), "events"));
    }

    /// REGRESSION (review Theme A — prefix-match confusion): the dataset scope is matched by EXACT
    /// equality, never by prefix. A scope named `dataset` must NOT admit a sibling whose name merely
    /// shares that prefix (`dataset-secret`, `datasets`), which a boundary-unaware `starts_with`
    /// check would wrongly allow. This pins the chosen semantics: prefix-style scopes do not widen.
    #[test]
    fn scope_exact_match_does_not_widen_to_siblings() {
        let scope = Scope::dataset("dataset");
        // The exact name is allowed.
        assert!(scope_allows(&scope, "dataset"));
        // Sibling names sharing the prefix must all be rejected.
        for sibling in ["dataset-secret", "datasets", "dataset/secret", "dataset.bak", "datasetX"] {
            assert!(
                !scope_allows(&scope, sibling),
                "exact-match scope `dataset` must NOT admit sibling `{sibling}`"
            );
        }
        // A shorter name that the scope is a prefix-extension of must also be rejected.
        assert!(!scope_allows(&Scope::dataset("dataset-secret"), "dataset"));
    }

    /// REGRESSION end-to-end: a minted, signed token scoped to `dataset` must be rejected by
    /// [`verify`] when used against the sibling dataset `dataset-secret`. Guards the full
    /// chain-check + app-scope path, not just the bare [`scope_allows`] predicate.
    #[test]
    fn verify_rejects_sibling_prefix_dataset() {
        let owner = ident("owner-sibling");
        let token = mint(&owner, owner.node_id(), &Scope::dataset("dataset"), 0, 7).unwrap();
        for sibling in ["dataset-secret", "datasets", "dataset/secret"] {
            let r = verify(
                &owner.node_id(),
                &[],
                &[],
                1000,
                &owner.node_id(),
                sibling,
                &token,
                &never_revoked,
            );
            assert!(
                r.is_err(),
                "scope `dataset` must reject sibling `{sibling}`, got {r:?}"
            );
        }
        // ...while the exact dataset still verifies.
        let ok = verify(
            &owner.node_id(),
            &[],
            &[],
            1000,
            &owner.node_id(),
            "dataset",
            &token,
            &never_revoked,
        );
        assert!(ok.is_ok(), "exact dataset must still verify: {ok:?}");
    }

    #[test]
    fn scope_caveat_roundtrip() {
        let s = Scope::dataset("sales");
        assert_eq!(Scope::from_caveat(&s.to_caveat()), s);
    }

    #[test]
    fn mint_and_verify() {
        let owner = ident("owner");
        let token = mint(&owner, owner.node_id(), &Scope::dataset("sales"), 0, 1).unwrap();
        let r = verify(
            &owner.node_id(),
            &[],
            &[],
            1000,
            &owner.node_id(),
            "sales",
            &token,
            &never_revoked,
        );
        assert!(r.is_ok(), "valid token should verify: {r:?}");
    }

    #[test]
    fn verify_rejects_other_dataset() {
        let owner = ident("owner2");
        let token = mint(&owner, owner.node_id(), &Scope::dataset("sales"), 0, 2).unwrap();
        let r = verify(
            &owner.node_id(),
            &[],
            &[],
            1000,
            &owner.node_id(),
            "events",
            &token,
            &never_revoked,
        );
        assert!(r.is_err(), "out-of-scope dataset must be rejected");
    }

    #[test]
    fn verify_rejects_expired() {
        let owner = ident("owner3");
        let token = mint(&owner, owner.node_id(), &Scope::all(), 500, 3).unwrap();
        let r = verify(
            &owner.node_id(),
            &[],
            &[],
            1000, // now > not_after
            &owner.node_id(),
            "any",
            &token,
            &never_revoked,
        );
        assert!(r.is_err(), "expired token must be rejected");
    }

    #[test]
    fn all_scope_covers_every_dataset() {
        let owner = ident("owner4");
        let token = mint(&owner, owner.node_id(), &Scope::all(), 0, 4).unwrap();
        for ds in ["a", "b", "c"] {
            let r = verify(
                &owner.node_id(),
                &[],
                &[],
                10,
                &owner.node_id(),
                ds,
                &token,
                &never_revoked,
            );
            assert!(r.is_ok(), "all-scope must cover {ds}: {r:?}");
        }
    }

    #[test]
    fn inspect_reports_ability_and_scope() {
        let owner = ident("owner5");
        let token = mint(&owner, owner.node_id(), &Scope::dataset("logs"), 0, 5).unwrap();
        let (abilities, scope) = inspect(&token).unwrap();
        assert_eq!(abilities, vec![ABILITY_QUERY.to_string()]);
        assert_eq!(scope, Scope::dataset("logs"));
    }

    #[test]
    fn inspect_rejects_garbage() {
        assert!(inspect("not-hex-token!!").is_err());
    }
}
