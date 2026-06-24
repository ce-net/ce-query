//! The dataset catalog — a small, JSON-persisted map of `name -> Dataset`, the single-writer
//! owner's view of which datasets exist and where their shards live.
//!
//! This mirrors ce-storage's bucket index: a local sorted map kept on disk. A published catalog
//! could equally be stored as a blob and shared by CID (it is just serde-serialisable), but the
//! local file is the zero-config default. The catalog holds **metadata only** — never row bytes;
//! shards live in the CE blob store and are fetched by CID at query time.
//!
//! All mutations are explicit and the file is rewritten atomically (write-temp-then-rename) so a
//! crash mid-write cannot corrupt the catalog. Loading a missing catalog yields an empty one.

use crate::dataset::Dataset;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The current on-disk catalog schema version. Bumped when the persisted shape changes so [`load`]
/// can migrate older files forward rather than failing.
///
/// [`load`]: Catalog::load
pub const CATALOG_SCHEMA_VERSION: u32 = 1;

/// The on-disk catalog: dataset name -> dataset metadata, plus a schema version for forward
/// migration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Catalog {
    /// On-disk schema version (see [`CATALOG_SCHEMA_VERSION`]). Defaults to 0 for pre-versioned
    /// files so they are recognised and upgraded on the next save.
    #[serde(default)]
    pub version: u32,
    /// All datasets known to this owner, keyed by name (sorted for stable serialization).
    #[serde(default)]
    pub datasets: BTreeMap<String, Dataset>,
}

impl Default for Catalog {
    fn default() -> Self {
        Catalog { version: CATALOG_SCHEMA_VERSION, datasets: BTreeMap::new() }
    }
}

impl Catalog {
    /// An empty catalog.
    pub fn new() -> Catalog {
        Catalog::default()
    }

    /// Load the catalog from `path`, or return an empty catalog if the file does not exist. A
    /// present-but-corrupt file is a hard error (so we never silently lose datasets). A file written
    /// by an older schema version is migrated forward in memory (and persisted on the next save).
    pub fn load(path: &Path) -> Result<Catalog> {
        match std::fs::read(path) {
            Ok(bytes) => {
                let mut c: Catalog = serde_json::from_slice(&bytes)
                    .with_context(|| format!("parsing catalog {path:?}"))?;
                c.migrate()?;
                Ok(c)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Catalog::new()),
            Err(e) => Err(e).with_context(|| format!("reading catalog {path:?}")),
        }
    }

    /// Migrate a loaded catalog forward to [`CATALOG_SCHEMA_VERSION`]. Newer-than-known versions are
    /// a hard error (a downgrade must not silently drop fields it does not understand).
    fn migrate(&mut self) -> Result<()> {
        if self.version > CATALOG_SCHEMA_VERSION {
            bail!(
                "catalog schema version {} is newer than this build supports ({CATALOG_SCHEMA_VERSION}); upgrade ce-query",
                self.version
            );
        }
        // v0 (pre-versioned) -> v1: nothing structural changed; just stamp the version.
        self.version = CATALOG_SCHEMA_VERSION;
        Ok(())
    }

    /// Persist the catalog to `path` atomically (write a sibling temp file, fsync it, then rename over
    /// the target). Creates the parent directory if missing. The temp file name includes the process
    /// id so two concurrent writers never collide on the same temp path. Stamps the current schema
    /// version before writing.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating catalog dir {parent:?}"))?;
        }
        let mut to_write = self.clone();
        to_write.version = CATALOG_SCHEMA_VERSION;
        let bytes = serde_json::to_vec_pretty(&to_write).context("serializing catalog")?;
        let tmp = path.with_extension(format!("json.tmp.{}", std::process::id()));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)
                .with_context(|| format!("writing temp catalog {tmp:?}"))?;
            f.write_all(&bytes).with_context(|| format!("writing temp catalog {tmp:?}"))?;
            f.sync_all().with_context(|| format!("fsync temp catalog {tmp:?}"))?;
        }
        std::fs::rename(&tmp, path).with_context(|| format!("renaming catalog into place {path:?}"))?;
        Ok(())
    }

    /// Atomically load, mutate, and save the catalog under an advisory lock, so two concurrent
    /// `ce-query` processes cannot clobber each other's writes (the lost-update race). The closure
    /// receives the freshly-loaded catalog and may return a value to pass back to the caller. The
    /// lock is a best-effort exclusive lock file next to the catalog, released on drop.
    pub fn mutate<T>(path: &Path, f: impl FnOnce(&mut Catalog) -> Result<T>) -> Result<T> {
        let _lock = CatalogLock::acquire(path, Duration::from_secs(10))?;
        let mut catalog = Catalog::load(path)?;
        let out = f(&mut catalog)?;
        catalog.save(path)?;
        Ok(out)
    }

    /// Insert or replace a dataset. Returns the previous dataset of the same name, if any.
    pub fn put(&mut self, dataset: Dataset) -> Option<Dataset> {
        self.datasets.insert(dataset.name.clone(), dataset)
    }

    /// Get a dataset by name.
    pub fn get(&self, name: &str) -> Option<&Dataset> {
        self.datasets.get(name)
    }

    /// Get a dataset by name or a clear "not found" error.
    pub fn require(&self, name: &str) -> Result<&Dataset> {
        self.datasets.get(name).ok_or_else(|| anyhow::anyhow!("no such dataset `{name}`"))
    }

    /// Remove a dataset by name. Errors if it does not exist.
    pub fn remove(&mut self, name: &str) -> Result<Dataset> {
        match self.datasets.remove(name) {
            Some(d) => Ok(d),
            None => bail!("no such dataset `{name}`"),
        }
    }

    /// Dataset names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }
}

/// A best-effort exclusive advisory lock for the catalog, implemented as an `O_EXCL` lock file
/// beside it. Acquisition spins (with a short sleep) until the lock is free or the timeout elapses;
/// the lock file is removed on drop. A stale lock older than 60s is reclaimed (a crashed writer must
/// not deadlock the catalog forever).
struct CatalogLock {
    path: PathBuf,
}

impl CatalogLock {
    fn acquire(catalog_path: &Path, timeout: Duration) -> Result<CatalogLock> {
        let lock_path = catalog_path.with_extension("lock");
        if let Some(parent) = lock_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let started = Instant::now();
        loop {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock_path) {
                Ok(mut f) => {
                    use std::io::Write;
                    let _ = write!(f, "{}", std::process::id());
                    return Ok(CatalogLock { path: lock_path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Reclaim a stale lock from a crashed writer.
                    if let Ok(meta) = std::fs::metadata(&lock_path)
                        && let Ok(modified) = meta.modified()
                        && modified.elapsed().map(|d| d > Duration::from_secs(60)).unwrap_or(false)
                    {
                        let _ = std::fs::remove_file(&lock_path);
                        continue;
                    }
                    if started.elapsed() >= timeout {
                        bail!("timed out acquiring catalog lock at {lock_path:?} (another ce-query process holds it)");
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(e) => return Err(e).with_context(|| format!("opening catalog lock {lock_path:?}")),
            }
        }
    }
}

impl Drop for CatalogLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Default catalog path: `<CE data dir>/query/catalog.json`. Falls back to a relative path when the
/// platform data dir cannot be resolved (e.g. in a minimal CI container).
pub fn default_catalog_path() -> PathBuf {
    match directories::ProjectDirs::from("", "", "ce") {
        Some(dirs) => dirs.data_dir().join("query").join("catalog.json"),
        None => PathBuf::from("ce-query-catalog.json"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ce-query-cat-{}-{}.json", name, std::process::id()))
    }

    #[test]
    fn load_missing_is_empty() {
        let p = tmp_path("missing");
        let _ = std::fs::remove_file(&p);
        let c = Catalog::load(&p).unwrap();
        assert!(c.datasets.is_empty());
    }

    #[test]
    fn put_get_remove_roundtrip_on_disk() {
        let p = tmp_path("roundtrip");
        let _ = std::fs::remove_file(&p);

        let mut c = Catalog::new();
        let mut d = Dataset::new("sales", vec!["amount".into()]);
        d.add_shard("cid1", 10, 100);
        c.put(d.clone());
        c.save(&p).unwrap();

        let loaded = Catalog::load(&p).unwrap();
        assert_eq!(loaded.get("sales"), Some(&d));
        assert_eq!(loaded.names(), vec!["sales".to_string()]);

        let mut loaded = loaded;
        let removed = loaded.remove("sales").unwrap();
        assert_eq!(removed, d);
        assert!(loaded.remove("sales").is_err());

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn require_errors_clearly() {
        let c = Catalog::new();
        let err = c.require("nope").unwrap_err();
        assert!(err.to_string().contains("no such dataset"), "{err}");
    }

    #[test]
    fn corrupt_catalog_is_error_not_empty() {
        let p = tmp_path("corrupt");
        std::fs::write(&p, b"{ this is not json").unwrap();
        assert!(Catalog::load(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn put_returns_previous() {
        let mut c = Catalog::new();
        assert!(c.put(Dataset::new("t", vec![])).is_none());
        let prev = c.put(Dataset::new("t", vec!["x".into()]));
        assert!(prev.is_some());
        assert!(prev.unwrap().schema.is_empty());
    }

    #[test]
    fn save_stamps_schema_version() {
        let p = tmp_path("version");
        let _ = std::fs::remove_file(&p);
        Catalog::new().save(&p).unwrap();
        let raw = std::fs::read_to_string(&p).unwrap();
        assert!(raw.contains("\"version\""), "saved catalog must carry a version: {raw}");
        let loaded = Catalog::load(&p).unwrap();
        assert_eq!(loaded.version, CATALOG_SCHEMA_VERSION);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn pre_versioned_file_migrates_forward() {
        // A legacy file with no `version` field loads as version 0 and migrates to the current one.
        let p = tmp_path("legacy");
        std::fs::write(&p, br#"{"datasets":{}}"#).unwrap();
        let loaded = Catalog::load(&p).unwrap();
        assert_eq!(loaded.version, CATALOG_SCHEMA_VERSION);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn newer_schema_version_is_rejected() {
        let p = tmp_path("future");
        std::fs::write(&p, br#"{"version":999999,"datasets":{}}"#).unwrap();
        let err = Catalog::load(&p).unwrap_err();
        assert!(err.to_string().contains("newer than this build"), "{err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn mutate_is_atomic_load_modify_save() {
        let p = tmp_path("mutate");
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(p.with_extension("lock"));
        Catalog::mutate(&p, |c| {
            c.put(Dataset::new("a", vec![]));
            Ok(())
        })
        .unwrap();
        Catalog::mutate(&p, |c| {
            c.put(Dataset::new("b", vec![]));
            Ok(())
        })
        .unwrap();
        let loaded = Catalog::load(&p).unwrap();
        assert_eq!(loaded.names(), vec!["a".to_string(), "b".to_string()]);
        // The lock file must be released (removed) after each mutate.
        assert!(!p.with_extension("lock").exists());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn lock_blocks_a_second_holder_until_released() {
        let p = tmp_path("locktwice");
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(p.with_extension("lock"));
        let lock = CatalogLock::acquire(&p, Duration::from_millis(100)).unwrap();
        // A second acquire with a short timeout must fail while the first is held.
        let second = CatalogLock::acquire(&p, Duration::from_millis(50));
        assert!(second.is_err(), "second lock must not be granted while held");
        drop(lock);
        // After release, acquisition succeeds.
        assert!(CatalogLock::acquire(&p, Duration::from_millis(100)).is_ok());
        let _ = std::fs::remove_file(p.with_extension("lock"));
        let _ = std::fs::remove_file(&p);
    }
}
