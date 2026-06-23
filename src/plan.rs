//! Query planning — assign each shard to a map host, with stable, balanced placement and
//! deterministic failover.
//!
//! Given a dataset's shards and a set of candidate hosts (from the atlas), the planner decides
//! **which host maps which shard**. The chosen scheme is **rendezvous hashing** (a.k.a. highest
//! random weight, HRW): for a `(shard, host)` pair, compute `score = sha256(shard_cid || host_id)`,
//! and assign each shard to the host with the highest score. HRW gives three properties the engine
//! relies on:
//!
//! 1. **Determinism** — same shards + same host set => same plan, so the plan is reproducible and a
//!    coordinator restart re-derives identical assignments (matches CE's chain-archive sharding).
//! 2. **Minimal disruption on membership change** — adding/removing one host only re-homes the
//!    shards that hashed to it, not the whole dataset.
//! 3. **Built-in failover ranking** — the *ordered* list of hosts by descending score is a ready
//!    fallback chain: if a shard's first host drops mid-query, retry on its **next** host with zero
//!    extra coordination ([`fallback_host`]).
//!
//! This module is pure (no network): it turns a host list into an assignment. The engine
//! ([`crate::engine`]) executes it and, on failure, calls back here for the next candidate.

use crate::dataset::Shard;
use sha2::{Digest, Sha256};

/// A planned unit of work: one shard, the ranked host candidates (best-first), and the currently
/// selected host index into that ranking. The engine starts at `attempt = 0` (the top-ranked host)
/// and advances on failure via [`crate::plan::ShardTask::advance`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardTask {
    /// The shard to map.
    pub shard: Shard,
    /// Host node ids ranked best-first by rendezvous score. Length == number of candidate hosts.
    pub ranked_hosts: Vec<String>,
    /// Which host in `ranked_hosts` is currently assigned (0 = primary).
    pub attempt: usize,
}

impl ShardTask {
    /// The host currently assigned to this shard, or `None` if every candidate has been exhausted
    /// (all ranked hosts tried and dropped).
    pub fn host(&self) -> Option<&str> {
        self.ranked_hosts.get(self.attempt).map(String::as_str)
    }

    /// Advance to the next-best host after a failure. Returns `true` if a further candidate exists
    /// (the task can be retried), `false` if exhausted. This is the redistribute-on-drop primitive.
    pub fn advance(&mut self) -> bool {
        if self.attempt + 1 < self.ranked_hosts.len() {
            self.attempt += 1;
            true
        } else {
            // Mark as exhausted by moving past the end.
            self.attempt = self.ranked_hosts.len();
            false
        }
    }

    /// True if all candidate hosts for this shard have been tried and dropped.
    pub fn exhausted(&self) -> bool {
        self.attempt >= self.ranked_hosts.len()
    }
}

/// Rendezvous (HRW) score of a `(shard, host)` pair: the first 8 bytes of `sha256(cid || host)` as a
/// big-endian `u64`. Higher wins. Deterministic and uniformly distributed.
pub fn rendezvous_score(shard_cid: &str, host_id: &str) -> u64 {
    let mut h = Sha256::new();
    h.update(shard_cid.as_bytes());
    h.update(b"|");
    h.update(host_id.as_bytes());
    let digest = h.finalize();
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(buf)
}

/// Rank `hosts` for a shard by descending rendezvous score (ties broken by host id for stability).
/// Returns host ids best-first. An empty host list yields an empty ranking.
pub fn rank_hosts(shard_cid: &str, hosts: &[String]) -> Vec<String> {
    let mut scored: Vec<(u64, &String)> =
        hosts.iter().map(|h| (rendezvous_score(shard_cid, h), h)).collect();
    // Sort by score descending; on a score tie, by host id ascending for determinism.
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().map(|(_, h)| h.clone()).collect()
}

/// Build the plan: one [`ShardTask`] per shard, each with its hosts ranked best-first. Pure and
/// deterministic. With no hosts, every task is born exhausted (the engine reports an error rather
/// than silently dropping data).
pub fn plan(shards: &[Shard], hosts: &[String]) -> Vec<ShardTask> {
    shards
        .iter()
        .map(|s| ShardTask { shard: s.clone(), ranked_hosts: rank_hosts(&s.cid, hosts), attempt: 0 })
        .collect()
}

/// The fallback host for a shard at a given attempt depth: the `attempt`-th ranked host, or `None`
/// if beyond the candidate list. Convenience over re-ranking.
pub fn fallback_host(shard_cid: &str, hosts: &[String], attempt: usize) -> Option<String> {
    rank_hosts(shard_cid, hosts).into_iter().nth(attempt)
}

/// Summarise a plan's host load: host id -> number of primary (attempt-0) shards assigned. Used by
/// the CLI's `plan` command and to assert balance in tests.
pub fn load_summary(tasks: &[ShardTask]) -> std::collections::BTreeMap<String, usize> {
    let mut m = std::collections::BTreeMap::new();
    for t in tasks {
        if let Some(h) = t.host() {
            *m.entry(h.to_string()).or_insert(0) += 1;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn shard(cid: &str) -> Shard {
        Shard { cid: cid.to_string(), rows: 0, bytes: 0 }
    }

    fn hosts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("host{i:02}")).collect()
    }

    #[test]
    fn ranking_is_deterministic() {
        let h = hosts(5);
        let a = rank_hosts("cidA", &h);
        let b = rank_hosts("cidA", &h);
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        // All hosts present, no duplicates.
        let set: std::collections::HashSet<_> = a.iter().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn plan_assigns_every_shard_a_primary() {
        let shards: Vec<Shard> = (0..20).map(|i| shard(&format!("cid{i}"))).collect();
        let h = hosts(4);
        let tasks = plan(&shards, &h);
        assert_eq!(tasks.len(), 20);
        assert!(tasks.iter().all(|t| t.host().is_some()));
    }

    #[test]
    fn no_hosts_means_exhausted_tasks() {
        let tasks = plan(&[shard("c0"), shard("c1")], &[]);
        assert!(tasks.iter().all(|t| t.exhausted()));
        assert!(tasks.iter().all(|t| t.host().is_none()));
    }

    #[test]
    fn next_walks_the_fallback_chain_then_exhausts() {
        let h = hosts(3);
        let mut t = plan(&[shard("cid")], &h).remove(0);
        let primary = t.host().unwrap().to_string();
        assert_eq!(t.attempt, 0);

        assert!(t.advance()); // -> second
        assert_ne!(t.host().unwrap(), primary);
        assert!(t.advance()); // -> third
        assert!(!t.advance()); // exhausted (only 3 hosts)
        assert!(t.exhausted());
        assert!(t.host().is_none());
    }

    #[test]
    fn fallback_host_matches_ranking() {
        let h = hosts(4);
        let ranked = rank_hosts("cidZ", &h);
        for (i, want) in ranked.iter().enumerate() {
            assert_eq!(fallback_host("cidZ", &h, i).as_ref(), Some(want));
        }
        assert_eq!(fallback_host("cidZ", &h, 4), None);
    }

    #[test]
    fn removing_a_host_only_rehomes_its_shards() {
        // HRW property: dropping one host re-homes only shards whose primary was that host.
        let shards: Vec<Shard> = (0..200).map(|i| shard(&format!("cid{i}"))).collect();
        let full = hosts(6);
        let dropped = &full[3]; // remove host03
        let reduced: Vec<String> = full.iter().filter(|h| *h != dropped).cloned().collect();

        let before = plan(&shards, &full);
        let after = plan(&shards, &reduced);

        for (b, a) in before.iter().zip(after.iter()) {
            let before_host = b.host().unwrap();
            let after_host = a.host().unwrap();
            if before_host != dropped {
                // A shard not homed on the dropped host keeps its primary.
                assert_eq!(before_host, after_host, "shard {} moved unexpectedly", b.shard.cid);
            }
        }
    }

    // Rendezvous placement is reasonably balanced: with many shards and a few hosts, every host
    // gets a non-trivial share (no host is starved or monopolised).
    proptest! {
        #[test]
        fn balanced_assignment(nshards in 100usize..300, nhosts in 2usize..8) {
            let shards: Vec<Shard> = (0..nshards).map(|i| shard(&format!("s{i}"))).collect();
            let h = hosts(nhosts);
            let tasks = plan(&shards, &h);
            let load = load_summary(&tasks);
            // Every host should receive at least one shard for these sizes, and none should hold
            // more than ~3x the fair share (loose bound; HRW is uniform in expectation).
            let fair = nshards as f64 / nhosts as f64;
            prop_assert_eq!(load.len(), nhosts);
            for (_h, count) in load {
                prop_assert!(count as f64 <= fair * 3.0 + 5.0, "host overloaded: {} vs fair {}", count, fair);
            }
        }
    }
}
