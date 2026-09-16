//! Curated gallery manifest and verification logic (`bd-16g.8.3`).

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Write as _;

use crate::narrative::EventRecord;
use crate::permalink::{BuildLink, Permalink, config_digest_with_diff};
use crate::{ScriptBotsConfig, WorldState};

/// A semantic event expected to occur during a gallery run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Permalink of the world that diverged.
    pub permalink: String,
    /// Description of the divergence reason.
    pub reason: String,
}

/// Report summarizing gallery verification results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

impl VerificationReport {
    /// Format a human-readable one-line summary of the verification result.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.passed {
            format!(
                "PASS {}: {}/{} events matched up to horizon {}",
                self.world_id, self.events_expected, self.events_actual, self.horizon_ticks
            )
        } else if let Some(div) = &self.divergence {
            format!(
                "FAIL {}: divergence at index {} ({}) [permalink: {}]",
                self.world_id, div.index, div.reason, div.permalink
            )
        } else {
            format!(
                "FAIL {}: verification failed (unknown reason)",
                self.world_id
            )
        }
    }
}

/// A single entry in the curated gallery manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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

/// Check if the given event kind string corresponds to a valid narrative event kind.
#[must_use]
pub fn is_known_event_kind(kind: &str) -> bool {
    matches!(
        kind,
        "PopulationCrash"
            | "population_crash"
            | "PopulationBoom"
            | "population_boom"
            | "DietShift"
            | "diet_shift"
            | "Extinction"
            | "extinction"
            | "EnergyCollapse"
            | "energy_collapse"
            | "EnergyRecovery"
            | "energy_recovery"
            | "CombatSurge"
            | "combat_surge"
            | "RegimeChange"
            | "regime_change"
            | "PredatorEmergence"
            | "predator_emergence"
            | "AltruismOnset"
            | "altruism_onset"
            | "SpeciationHint"
            | "speciation_hint"
            | "FloorEngaged"
            | "floor_engaged"
            | "ResourceCollapse"
            | "resource_collapse"
    )
}

/// Validate date format `YYYY-MM-DD`.
#[must_use]
pub fn is_valid_date(date: &str) -> bool {
    if date.len() != 10 {
        return false;
    }
    let b = date.as_bytes();
    if b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    if !b[0..4].iter().all(u8::is_ascii_digit)
        || !b[5..7].iter().all(u8::is_ascii_digit)
        || !b[8..10].iter().all(u8::is_ascii_digit)
    {
        return false;
    }
    let Ok(year) = date[0..4].parse::<u32>() else {
        return false;
    };
    let Ok(month) = date[5..7].parse::<u32>() else {
        return false;
    };
    let Ok(day) = date[8..10].parse::<u32>() else {
        return false;
    };
    year >= 2000 && (1..=12).contains(&month) && (1..=31).contains(&day)
}

/// Validate identifier format (1..=64 ascii alphanumeric, '-', or '_').
#[must_use]
pub fn is_valid_world_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

impl GalleryWorld {
    /// Validate individual world constraints.
    pub fn validate(&self) -> Result<(), String> {
        if !is_valid_world_id(&self.id) {
            return Err(format!(
                "world '{}' has invalid id (must be 1..=64 ascii alphanumeric, '-' or '_')",
                self.id
            ));
        }
        if self.title.trim().is_empty() {
            return Err(format!("world '{}' has empty title", self.id));
        }
        if self.title.len() > 100 {
            return Err(format!("world '{}' title exceeds 100 chars limit", self.id));
        }
        if self.story.trim().is_empty() {
            return Err(format!("world '{}' has empty story", self.id));
        }
        if self.story.len() > 200 {
            return Err(format!("story for '{}' exceeds 200 chars limit", self.id));
        }
        if self.added_by.trim().is_empty() {
            return Err(format!("world '{}' has empty added_by", self.id));
        }
        if !is_valid_date(&self.added_at) {
            return Err(format!(
                "world '{}' has invalid added_at date '{}' (expected YYYY-MM-DD)",
                self.id, self.added_at
            ));
        }
        if self.blessed_on_build == 0 {
            return Err(format!(
                "blessed_on_build must be greater than 0 for world '{}'",
                self.id
            ));
        }
        if self.horizon_ticks == 0 {
            return Err(format!(
                "horizon_ticks must be greater than 0 for world '{}'",
                self.id
            ));
        }
        if self.horizon_ticks > 50_000 {
            return Err(format!(
                "horizon_ticks for '{}' exceeds 50000 limit",
                self.id
            ));
        }
        let permalink = Permalink::from_url_string(&self.permalink)
            .map_err(|e| format!("invalid permalink for '{}': {e}", self.id))?;
        permalink
            .validate_knobs()
            .map_err(|e| format!("invalid knobs in permalink for '{}': {e}", self.id))?;

        if self.expected_timeline.is_empty() {
            return Err(format!(
                "expected_timeline cannot be empty for curated world '{}' (vacuous timeline prevented)",
                self.id
            ));
        }

        let mut prev_tick = 0u64;
        for (idx, event) in self.expected_timeline.iter().enumerate() {
            if !is_known_event_kind(&event.kind) {
                return Err(format!(
                    "unknown event kind '{}' at index {idx} in timeline of '{}'",
                    event.kind, self.id
                ));
            }
            if event.tick > self.horizon_ticks {
                return Err(format!(
                    "event tick {} at index {idx} exceeds horizon_ticks {} in '{}'",
                    event.tick, self.horizon_ticks, self.id
                ));
            }
            if idx > 0 && event.tick < prev_tick {
                return Err(format!(
                    "unsorted timeline in '{}': tick {} at index {idx} < previous tick {prev_tick}",
                    self.id, event.tick
                ));
            }
            prev_tick = event.tick;
        }

        Ok(())
    }

    /// Compare expected timeline with actual narrative events from a run up to `horizon_ticks`.
    #[must_use]
    pub fn verify_timeline(&self, actual_events: &[EventRecord]) -> VerificationReport {
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
                permalink: self.permalink.clone(),
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
            let act_kind_debug = format!("{:?}", act.kind);
            let act_kind_str = act.kind.as_str();
            let kind_matches = exp.kind == act_kind_debug
                || exp.kind == act_kind_str
                || exp.kind == act.metric
                || exp.metric.as_ref() == Some(&act.metric);
            let tick_matches = exp.tick == act.tick.0;

            if !tick_matches || !kind_matches {
                let div = DivergenceDetails {
                    index: idx,
                    expected_tick: Some(exp.tick),
                    actual_tick: Some(act.tick.0),
                    expected_kind: Some(exp.kind.clone()),
                    actual_kind: Some(act_kind_debug),
                    permalink: self.permalink.clone(),
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
#[serde(deny_unknown_fields)]
pub struct GalleryManifest {
    /// List of world entries.
    #[serde(rename = "world", default)]
    pub worlds: Vec<GalleryWorld>,
}

impl GalleryManifest {
    /// Parse gallery manifest from TOML string.
    pub fn parse_toml(content: &str) -> Result<Self, String> {
        toml::from_str(content).map_err(|e| format!("failed to parse gallery manifest: {e}"))
    }

    /// Validate all structural and invariant bounds of the manifest.
    pub fn validate(&self) -> Result<(), String> {
        if self.worlds.is_empty() {
            return Err("gallery manifest must contain at least one world".into());
        }
        let mut ids = HashSet::with_capacity(self.worlds.len());
        for w in &self.worlds {
            if !ids.insert(&w.id) {
                return Err(format!("duplicate world id: {}", w.id));
            }
            w.validate()?;
        }
        Ok(())
    }
}

/// Apply a series of knob diff assignments to a `ScriptBotsConfig`.
pub fn apply_knob_diff(
    config: &mut ScriptBotsConfig,
    diff: &[(String, f64)],
) -> Result<(), String> {
    let mut val =
        serde_json::to_value(&*config).map_err(|e| format!("failed to serialize config: {e}"))?;
    for (path, num) in diff {
        set_json_path(&mut val, path, *num)?;
    }
    *config = serde_json::from_value(val)
        .map_err(|e| format!("failed to deserialize modified config: {e}"))?;
    Ok(())
}

fn set_json_path(root: &mut serde_json::Value, path: &str, value: f64) -> Result<(), String> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = root;
    for &part in &parts[..parts.len() - 1] {
        match current {
            serde_json::Value::Object(map) => {
                current = map
                    .entry(part)
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
            }
            _ => return Err(format!("path segment '{part}' is not an object")),
        }
    }
    let last = *parts.last().expect("path non-empty");
    match current {
        serde_json::Value::Object(map) => {
            if let Some(existing) = map.get(last) {
                if existing.is_u64() {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let u = value as u64;
                    map.insert(last.to_string(), serde_json::Value::Number(u.into()));
                    return Ok(());
                } else if existing.is_i64() {
                    #[allow(clippy::cast_possible_truncation)]
                    let i = value as i64;
                    map.insert(last.to_string(), serde_json::Value::Number(i.into()));
                    return Ok(());
                }
            }
            let num = serde_json::Number::from_f64(value)
                .ok_or_else(|| format!("non-finite knob value for {path}"))?;
            map.insert(last.to_string(), serde_json::Value::Number(num));
            Ok(())
        }
        _ => Err(format!("cannot set knob '{path}' on non-object")),
    }
}

/// Canonical build identity for curated gallery worlds.
#[must_use]
pub const fn canonical_build_link() -> BuildLink {
    BuildLink {
        toolchain_digest: 0x1602_0901_0000_0001,
        lockfile_digest: 0x1602_0901_0000_0002,
        core_digest: 0x1602_0901_0000_0003,
    }
}

/// Base configuration associated with a scenario id.
#[must_use]
pub fn base_config_for_scenario(scenario_id: &str) -> ScriptBotsConfig {
    match scenario_id {
        "meadow" => ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            food_max: 0.5,
            food_growth_rate: 0.01,
            food_respawn_interval: 25,
            food_respawn_amount: 0.5,
            population_minimum: 24,
            population_spawn_interval: 600,
            ..ScriptBotsConfig::default()
        },
        _ => ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            ..ScriptBotsConfig::default()
        },
    }
}

/// Create a valid, canonical gallery permalink URL string (`sbw1...`).
pub fn create_gallery_permalink(
    scenario_id: &str,
    seed: u64,
    knob_diff: Vec<(String, f64)>,
) -> Result<String, String> {
    let base_config = base_config_for_scenario(scenario_id);
    let scenario_config =
        serde_json::to_value(&base_config).map_err(|e| format!("serialize base config: {e}"))?;
    let config_digest = config_digest_with_diff(&scenario_config, &knob_diff);

    let permalink = Permalink {
        scenario_id: scenario_id.to_string(),
        seed,
        knob_diff,
        config_digest,
        build: canonical_build_link(),
    };
    permalink
        .validate_knobs()
        .map_err(|e| format!("invalid knobs: {e}"))?;
    Ok(permalink.to_url_string())
}

/// Reconstruct a `ScriptBotsConfig` from a `Permalink`.
///
/// This is the shared composition path used across native and browser targets.
/// It validates knobs against the registry, resolves the scenario's base configuration,
/// verifies the embedded configuration digest, applies the inline knob diff, sets the root RNG seed,
/// and validates the resulting configuration invariants before allocation.
pub fn reconstruct_config_from_permalink(link: &Permalink) -> Result<ScriptBotsConfig, String> {
    link.validate_knobs()
        .map_err(|e| format!("validate knobs: {e}"))?;

    let mut config = base_config_for_scenario(&link.scenario_id);
    let scenario_json =
        serde_json::to_value(&config).map_err(|e| format!("serialize base config: {e}"))?;
    link.verify_config_digest(&scenario_json)
        .map_err(|e| format!("config digest verification failed: {e}"))?;

    apply_knob_diff(&mut config, &link.knob_diff)?;
    config.rng_seed = Some(link.seed);
    config
        .validate()
        .map_err(|e| format!("validate composed config: {e}"))?;

    Ok(config)
}

/// Run a simulation from a permalink up to `horizon_ticks` and return all narrative events.
pub fn run_world_from_permalink(
    permalink_str: &str,
    horizon_ticks: u64,
) -> Result<Vec<EventRecord>, String> {
    let link =
        Permalink::from_url_string(permalink_str).map_err(|e| format!("decode permalink: {e}"))?;
    let config = reconstruct_config_from_permalink(&link)?;
    let mut world = WorldState::new(config).map_err(|e| format!("construct world: {e}"))?;

    for _ in 0..horizon_ticks {
        world.step().map_err(|e| format!("step world: {e}"))?;
    }

    Ok(world.narrative_events().iter().cloned().collect())
}

/// Run a simulation from a permalink up to `horizon_ticks` with pinned thread count.
pub fn run_world_hermetic(
    permalink_str: &str,
    horizon_ticks: u64,
    thread_count: usize,
) -> Result<Vec<EventRecord>, String> {
    if thread_count == 0 {
        return Err("thread_count must be positive".into());
    }
    #[cfg(feature = "parallel")]
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|e| {
                format!("failed to create Rayon thread pool with {thread_count} threads: {e}")
            })?;
        pool.install(|| run_world_from_permalink(permalink_str, horizon_ticks))
    }
    #[cfg(not(feature = "parallel"))]
    {
        if thread_count > 1 {
            return Err(format!(
                "parallel feature not enabled, requested {thread_count} threads"
            ));
        }
        run_world_from_permalink(permalink_str, horizon_ticks)
    }
}

/// Verify all worlds in the manifest against their permalinks, running each world headless
/// to its horizon and verifying the actual narrative events match the expected timeline.
pub fn verify_manifest(manifest: &GalleryManifest) -> Result<Vec<VerificationReport>, String> {
    manifest.validate()?;
    let mut reports = Vec::with_capacity(manifest.worlds.len());

    for world in &manifest.worlds {
        let actual_events = run_world_from_permalink(&world.permalink, world.horizon_ticks)
            .map_err(|e| format!("failed to run world '{}': {e}", world.id))?;
        let report = world.verify_timeline(&actual_events);
        reports.push(report);
    }

    Ok(reports)
}

/// Bless a targeted entry in the manifest with actual narrative events and build id.
/// Rewrites only the targeted entry in stable order and is byte-idempotent on subsequent runs.
pub fn bless_entry(
    manifest_content: &str,
    world_id: &str,
    actual_events: &[EventRecord],
    build_id: u64,
) -> Result<String, String> {
    if build_id == 0 {
        return Err("blessed_on_build must be greater than 0".into());
    }
    let mut manifest = GalleryManifest::parse_toml(manifest_content)?;
    let target = manifest
        .worlds
        .iter_mut()
        .find(|w| w.id == world_id)
        .ok_or_else(|| format!("world '{world_id}' not found in manifest"))?;

    let filtered_events: Vec<ExpectedEvent> = actual_events
        .iter()
        .filter(|e| e.tick.0 <= target.horizon_ticks)
        .map(|e| ExpectedEvent {
            tick: e.tick.0,
            kind: format!("{:?}", e.kind),
            metric: if e.metric.is_empty() {
                None
            } else {
                Some(e.metric.clone())
            },
        })
        .collect();

    target.expected_timeline = filtered_events;
    target.blessed_on_build = build_id;

    format_manifest_stable(&manifest)
}

/// Format a `GalleryManifest` in deterministic, stable TOML.
pub fn format_manifest_stable(manifest: &GalleryManifest) -> Result<String, String> {
    let mut out = String::with_capacity(1024);
    out.push_str("# Curated Gallery Manifest (bd-16g.8.3)\n");
    out.push_str("# An in-repo, versioned manifest of worlds worth seeing.\n\n");
    for (i, w) in manifest.worlds.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str("[[world]]\n");
        let _ = writeln!(out, "id = \"{}\"", w.id);
        let _ = writeln!(out, "title = \"{}\"", w.title);
        let _ = writeln!(out, "story = \"{}\"", w.story);
        let _ = writeln!(out, "permalink = \"{}\"", w.permalink);
        let _ = writeln!(out, "horizon_ticks = {}", w.horizon_ticks);
        let _ = writeln!(out, "added_by = \"{}\"", w.added_by);
        let _ = writeln!(out, "added_at = \"{}\"", w.added_at);
        let _ = writeln!(out, "blessed_on_build = {}", w.blessed_on_build);

        for event in &w.expected_timeline {
            out.push_str("\n[[world.expected_timeline]]\n");
            let _ = writeln!(out, "tick = {}", event.tick);
            let _ = writeln!(out, "kind = \"{}\"", event.kind);
            if let Some(metric) = &event.metric {
                let _ = writeln!(out, "metric = \"{metric}\"");
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_validate_manifest() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "test-1"
title = "Test World One"
story = "A short test description."
permalink = "{permalink}"
horizon_ticks = 1000
added_by = "test"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 100
kind = "PopulationCrash"
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse manifest");
        manifest.validate().expect("validate manifest");
        assert_eq!(manifest.worlds.len(), 1);
        assert_eq!(manifest.worlds[0].id, "test-1");
    }

    #[test]
    fn test_deny_unknown_fields_on_world() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "test-1"
title = "Test World One"
story = "A short test description."
permalink = "{permalink}"
horizon_ticks = 1000
added_by = "test"
added_at = "2026-07-22"
blessed_on_build = 1
bogus_field = "unexpected"

[[world.expected_timeline]]
tick = 100
kind = "PopulationCrash"
"#
        );
        assert!(
            GalleryManifest::parse_toml(&toml_data).is_err(),
            "unknown field must be rejected"
        );
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "dupe"
title = "One"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"

[[world]]
id = "dupe"
title = "Two"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "b"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_story_over_length_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let long_story = "a".repeat(201);
        let toml_data = format!(
            r#"
[[world]]
id = "long"
title = "Title"
story = "{long_story}"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_horizon_cap_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "over-horizon"
title = "Title"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 50001
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_blessed_on_build_zero_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "zero-build"
title = "Title"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 0

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_empty_vacuous_timeline_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "vacuous"
title = "Title"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(
            manifest.validate().is_err(),
            "empty expected timeline must be rejected"
        );
    }

    #[test]
    fn test_invalid_date_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "bad-date"
title = "Title"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "a"
added_at = "2026/07/22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_unknown_event_kind_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "bad-kind"
title = "Title"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "AlienInvasion"
"#
        );
        let manifest = GalleryManifest::parse_toml(&toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_unsorted_timeline_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let toml_data = format!(
            r#"
[[world]]
id = "unsorted"
title = "Title"
story = "Story"
permalink = "{permalink}"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"

[[world.expected_timeline]]
tick = 30
kind = "PopulationBoom"
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

        let actual = vec![EventRecord {
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
        assert!(report.summary().starts_with("PASS"));
    }

    #[test]
    fn test_verify_timeline_tick_divergence() {
        let world = GalleryWorld {
            id: "world-1".into(),
            title: "World 1".into(),
            story: "Story".into(),
            permalink: "sbw1.testlink".into(),
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

        // Actual event at tick 101 instead of 100 (one-tick shift)
        let actual = vec![EventRecord {
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
        let div = report.divergence.as_ref().unwrap();
        assert_eq!(div.expected_tick, Some(100));
        assert_eq!(div.actual_tick, Some(101));
        assert_eq!(div.permalink, "sbw1.testlink");
        assert!(report.summary().contains("divergence at index 0"));
    }

    #[test]
    fn test_bless_entry_idempotence_and_minimality() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let initial_manifest = format!(
            r#"# Curated Gallery Manifest (bd-16g.8.3)
# An in-repo, versioned manifest of worlds worth seeing.

[[world]]
id = "world-quiet-01"
title = "Quiet Oasis"
story = "A peaceful ecosystem."
permalink = "{permalink}"
horizon_ticks = 300
added_by = "test"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "PopulationCrash"

[[world]]
id = "world-other-02"
title = "Other World"
story = "Untouched world."
permalink = "{permalink}"
horizon_ticks = 300
added_by = "test"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 60
kind = "PopulationBoom"
"#
        );

        let actual = vec![EventRecord {
            schema_version: 1,
            tick: crate::Tick(90),
            kind: crate::narrative::EventKind::PopulationBoom,
            severity: 0.9,
            magnitude: 0.8,
            window: (60, 90),
            metric: "population".into(),
            before: 10.0,
            after: 30.0,
            score: 0.9,
            subject: None,
            human_text: "boom".into(),
        }];

        let blessed_once = bless_entry(&initial_manifest, "world-quiet-01", &actual, 42)
            .expect("bless world-quiet-01");
        // Verify only world-quiet-01 was modified
        assert!(blessed_once.contains("blessed_on_build = 42"));
        assert!(blessed_once.contains("tick = 90"));
        assert!(blessed_once.contains("Other World"));
        assert!(blessed_once.contains("tick = 60"));

        let blessed_twice = bless_entry(&blessed_once, "world-quiet-01", &actual, 42)
            .expect("re-bless world-quiet-01");
        assert_eq!(
            blessed_once, blessed_twice,
            "blessing must be byte-idempotent"
        );
    }

    #[test]
    fn test_invalid_permalink_rejected() {
        let toml_data = r#"
[[world]]
id = "bad-link"
title = "Title"
story = "Story"
permalink = "sbw1.not-a-valid-payload"
horizon_ticks = 100
added_by = "a"
added_at = "2026-07-22"
blessed_on_build = 1

[[world.expected_timeline]]
tick = 50
kind = "CombatSurge"
"#;
        let manifest = GalleryManifest::parse_toml(toml_data).expect("parse");
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn test_removed_kind_fails_verification() {
        let world = GalleryWorld {
            id: "world-1".into(),
            title: "World 1".into(),
            story: "Story".into(),
            permalink: "sbw1.1".into(),
            horizon_ticks: 500,
            expected_timeline: vec![
                ExpectedEvent {
                    tick: 50,
                    kind: "CombatSurge".into(),
                    metric: None,
                },
                ExpectedEvent {
                    tick: 100,
                    kind: "PopulationCrash".into(),
                    metric: None,
                },
            ],
            added_by: "test".into(),
            added_at: "2026-07-22".into(),
            blessed_on_build: 1,
        };

        // Only one event occurred (second event removed/missing)
        let actual = vec![EventRecord {
            schema_version: 1,
            tick: crate::Tick(50),
            kind: crate::narrative::EventKind::CombatSurge,
            severity: 0.8,
            magnitude: 0.5,
            window: (30, 50),
            metric: "spike_hits".into(),
            before: 10.0,
            after: 50.0,
            score: 0.8,
            subject: None,
            human_text: "surge".into(),
        }];

        let report = world.verify_timeline(&actual);
        assert!(!report.passed, "removed event must fail verification");
        assert_eq!(report.events_expected, 2);
        assert_eq!(report.events_actual, 1);
        assert!(report.divergence.is_some());
    }

    #[test]
    fn test_curated_worlds_verified_end_to_end() {
        let manifest_content = include_str!("../../../gallery/manifest.toml");
        let manifest =
            GalleryManifest::parse_toml(manifest_content).expect("parse in-repo gallery manifest");
        manifest.validate().expect("validate manifest");

        let reports = verify_manifest(&manifest).expect("verify all curated worlds");
        assert_eq!(reports.len(), 3);
        for report in reports {
            assert!(
                report.passed,
                "curated world {} failed verification: {}",
                report.world_id,
                report.summary()
            );
        }
    }

    #[test]
    fn test_wrong_thread_count_rejected() {
        let permalink = create_gallery_permalink("meadow", 42, vec![]).expect("permalink");
        let result = run_world_hermetic(&permalink, 10, 0);
        assert!(result.is_err(), "thread count 0 must be rejected");
    }
}
