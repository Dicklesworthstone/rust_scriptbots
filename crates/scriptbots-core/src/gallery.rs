//! Curated gallery manifest and verification logic (`bd-16g.8.3`).

use serde::{Deserialize, Serialize};

/// A semantic event expected to occur during a gallery run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpectedEvent {
    /// Simulation tick on which the event is expected.
    pub tick: u64,
    /// Type/kind of narrative event (e.g. `PopulationCrash`, `Extinction`).
    pub kind: String,
    /// Optional metric name associated with the event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
}

/// Details about a timeline divergence during gallery verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DivergenceDetails {
    /// Index of the event where divergence occurred.
    pub index: usize,
    /// Expected tick if available.
    pub expected_tick: Option<u64>,
    /// Actual tick if available.
    pub actual_tick: Option<u64>,
    /// Expected event kind if available.
    pub expected_kind: Option<String>,
    /// Actual event kind if available.
    pub actual_kind: Option<String>,
    /// Description of the divergence reason.
    pub reason: String,
}

/// Report summarizing gallery verification results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Gallery world ID.
    pub world_id: String,
    /// Simulation tick horizon.
    pub horizon_ticks: u64,
    /// Number of expected events.
    pub events_expected: usize,
    /// Number of actual events produced.
    pub events_actual: usize,
    /// Whether verification passed without divergence.
    pub passed: bool,
    /// Details of divergence if verification failed.
    pub divergence: Option<DivergenceDetails>,
}

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
    /// Semantic timeline of expected narrative events.
    #[serde(default)]
    pub expected_timeline: Vec<ExpectedEvent>,
    /// Author who submitted the entry.
    pub added_by: String,
    /// Date added (YYYY-MM-DD).
    pub added_at: String,
    /// Build identity version when blessed.
    pub blessed_on_build: u64,
}

impl GalleryWorld {
    /// Compare expected timeline with actual narrative events from a run up to `horizon_ticks`.
    #[must_use]
    pub fn verify_timeline(
        &self,
        actual_events: &[crate::narrative::EventRecord],
    ) -> VerificationReport {
        // Filter actual events to those occurring on or before horizon_ticks
        let filtered_actual: Vec<_> = actual_events
            .iter()
            .filter(|e| e.tick.0 <= self.horizon_ticks)
            .collect();

        if self.expected_timeline.len() != filtered_actual.len() {
            let idx = self.expected_timeline.len().min(filtered_actual.len());
            let exp_event = self.expected_timeline.get(idx);
            let act_event = filtered_actual.get(idx);

            let div = DivergenceDetails {
                index: idx,
                expected_tick: exp_event.map(|e| e.tick),
                actual_tick: act_event.map(|e| e.tick.0),
                expected_kind: exp_event.map(|e| e.kind.clone()),
                actual_kind: act_event.map(|e| format!("{:?}", e.kind)),
                reason: format!(
                    "event count mismatch: expected {} events, got {}",
                    self.expected_timeline.len(),
                    filtered_actual.len()
                ),
            };

            return VerificationReport {
                world_id: self.id.clone(),
                horizon_ticks: self.horizon_ticks,
                events_expected: self.expected_timeline.len(),
                events_actual: filtered_actual.len(),
                passed: false,
                divergence: Some(div),
            };
        }

        for (idx, (exp, act)) in self
            .expected_timeline
            .iter()
            .zip(filtered_actual.iter())
            .enumerate()
        {
            let act_kind_str = format!("{:?}", act.kind);
            let kind_matches = exp.kind == act_kind_str
                || exp.kind == act.metric
                || exp.metric.as_ref() == Some(&act.metric);
            let tick_matches = exp.tick == act.tick.0;

            if !tick_matches || !kind_matches {
                let div = DivergenceDetails {
                    index: idx,
                    expected_tick: Some(exp.tick),
                    actual_tick: Some(act.tick.0),
                    expected_kind: Some(exp.kind.clone()),
                    actual_kind: Some(act_kind_str),
                    reason: format!(
                        "divergence at index {idx}: expected (tick={}, kind='{}'), actual (tick={}, kind='{:?}')",
                        exp.tick, exp.kind, act.tick.0, act.kind
                    ),
                };

                return VerificationReport {
                    world_id: self.id.clone(),
                    horizon_ticks: self.horizon_ticks,
                    events_expected: self.expected_timeline.len(),
                    events_actual: filtered_actual.len(),
                    passed: false,
                    divergence: Some(div),
                };
            }
        }

        VerificationReport {
            world_id: self.id.clone(),
            horizon_ticks: self.horizon_ticks,
            events_expected: self.expected_timeline.len(),
            events_actual: filtered_actual.len(),
            passed: true,
            divergence: None,
        }
    }
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

    #[test]
    fn test_verify_timeline_success() {
        let world = GalleryWorld {
            id: "world-1".into(),
            title: "World 1".into(),
            story: "Story".into(),
            permalink: "sbw1.1".into(),
            horizon_ticks: 500,
            expected_timeline: vec![ExpectedEvent {
                tick: 100,
                kind: "PopulationCrash".into(),
                metric: None,
            }],
            added_by: "test".into(),
            added_at: "2026-07-22".into(),
            blessed_on_build: 1,
        };

        let actual = vec![crate::narrative::EventRecord {
            schema_version: 1,
            tick: crate::Tick(100),
            kind: crate::narrative::EventKind::PopulationCrash,
            severity: 0.8,
            magnitude: 0.5,
            window: (80, 100),
            metric: "population".into(),
            before: 100.0,
            after: 50.0,
            score: 0.8,
            subject: None,
            human_text: "population fell 50%".into(),
        }];

        let report = world.verify_timeline(&actual);
        assert!(report.passed, "verification should pass: {report:?}");
        assert_eq!(report.events_expected, 1);
        assert_eq!(report.events_actual, 1);
    }

    #[test]
    fn test_verify_timeline_tick_divergence() {
        let world = GalleryWorld {
            id: "world-1".into(),
            title: "World 1".into(),
            story: "Story".into(),
            permalink: "sbw1.1".into(),
            horizon_ticks: 500,
            expected_timeline: vec![ExpectedEvent {
                tick: 100,
                kind: "PopulationCrash".into(),
                metric: None,
            }],
            added_by: "test".into(),
            added_at: "2026-07-22".into(),
            blessed_on_build: 1,
        };

        // Actual event at tick 101 instead of 100
        let actual = vec![crate::narrative::EventRecord {
            schema_version: 1,
            tick: crate::Tick(101),
            kind: crate::narrative::EventKind::PopulationCrash,
            severity: 0.8,
            magnitude: 0.5,
            window: (80, 101),
            metric: "population".into(),
            before: 100.0,
            after: 50.0,
            score: 0.8,
            subject: None,
            human_text: "population fell 50%".into(),
        }];

        let report = world.verify_timeline(&actual);
        assert!(!report.passed, "divergent tick must fail verification");
        assert!(report.divergence.is_some());
        let div = report.divergence.unwrap();
        assert_eq!(div.expected_tick, Some(100));
        assert_eq!(div.actual_tick, Some(101));
    }
}
