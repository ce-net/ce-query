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

/// The on-disk catalog: dataset name -> dataset metadata.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Catalog {
    /// All datasets known to this owner, keyed by name (sorted for stable serialization).
    #[serde(default)]
    pub datasets: BTreeMap<String, Dataset>,
}

impl Catalog {
    /// An empty catalog.
    pub fn new() -> Catalog {
        Catalog::default()
    }

    /// Load the catalog from `path`, or return an empty catalog if the file does not exist. A
    /// present-but-corrupt file is a hard error (so we never silently lose datasets).
    pub fn load(path: &Path) -> Result<Catalog> {
        match std::fs::read(path) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes).with_context(|| format!("parsing catalog {path:?}"))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Catalog::new()),
            Err(e) => Err(e).with_context(|| format!("reading catalog {path:?}")),
        }
    }

    /// Persist the catalog to `path` atomically (write a sibling temp file, then rename over the
    /// target). Creates the parent directory if missing.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating catalog dir {parent:?}"))?;
        }
        let bytes = serde_json::to_vec_pretty(self).context("serializing catalog")?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &bytes).with_context(|| format!("writing temp catalog {tmp:?}"))?;
        std::fs::rename(&tmp, path).with_context(|| format!("renaming catalog into place {path:?}"))?;
        Ok(())
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
}
