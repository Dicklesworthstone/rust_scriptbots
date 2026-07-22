//! Curated gallery manifest and verification logic (`bd-16g.8.3`).

use serde::{Deserialize, Serialize};

/// A single entry in the curated gallery manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GalleryWorld {
    /// Unique identifier for the world entry.
    pub id: String,
    /// Human-readable title of the scenario.
    pub title: String,
    /// Brief story caption (<= 200 characters).
    pub story: String,
    /// Encoded permalink string (`sbw1...`).
    pub permalink: String,
    /// Target simulation tick horizon for verification.
    pub horizon_ticks: u64,
    /// Author who submitted the entry.
    pub added_by: String,
    /// Date added (YYYY-MM-DD).
    pub added_at: String,
    /// Build identity version when blessed.
    pub blessed_on_build: u64,
}

/// Manifest containing all curated gallery worlds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GalleryManifest {
    /// List of world entries.
    #[serde(rename = "world")]
    pub worlds: Vec<GalleryWorld>,
}

impl GalleryManifest {
    /// Parse gallery manifest from TOML string.
    pub fn parse_toml(content: &str) -> Result<Self, String> {
        toml::from_str(content).map_err(|e| format!("failed to parse gallery manifest: {e}"))
    }

    /// Validate all structural and invariant bounds of the manifest.
    pub fn validate(&self) -> Result<(), String> {
        let mut ids = std::collections::HashSet::new();
        for w in &self.worlds {
            if w.id.is_empty() {
                return Err("world id cannot be empty".into());
            }
            if !ids.insert(&w.id) {
                return Err(format!("duplicate world id: {}", w.id));
            }
            if w.story.len() > 200 {
                return Err(format!("story for {} exceeds 200 chars limit", w.id));
            }
            if w.horizon_ticks > 50_000 {
                return Err(format!("horizon_ticks for {} exceeds 50000 limit", w.id));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_validate_manifest() {
        let toml_data = r#"
[[world]]
id = "test-1"
title = "Test World One"
story = "A short test description."
permalink = "sbw1.test"
horizon_ticks = 1000
added_by = "test"
added_at = "2026-07-22"
blessed_on_build = 1
"#;
        let manifest = GalleryManifest::parse_toml(toml_data).expect("parse manifest");
        manifest.validate().expect("validate manifest");
        assert_eq!(manifest.worlds.len(), 1);
        assert_eq!(manifest.worlds[0].id, "test-1");
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let toml_data = r#"
[[world]]
id = "dupe"
title = "One"
story = "Story"
permalink = "sbw1.1"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1

[[world]]
id = "dupe"
title = "Two"
story = "Story"
permalink = "sbw1.2"
horizon_ticks = 100
added_by = "b"
added_at = "2026-07-22"
blessed_on_build = 1
"#;
        let manifest = GalleryManifest::parse_toml(toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_story_over_length_rejected() {
        let long_story = "a".repeat(201);
        let toml_data = format!(
            r#"
[[world]]
id = "long"
title = "Title"
story = "{long_story}"
permalink = "sbw1.1"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }
}
