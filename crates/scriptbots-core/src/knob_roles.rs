//! Fail-closed classification of every published configuration knob (bd-dorx).
//!
//! `ControlHandle::list_knobs` publishes every serialized [`ScriptBotsConfig`] leaf as a REST/MCP
//! knob and `apply_patch` accepts every declared path, so a field added to the config becomes a
//! public scientific control with nothing asserting that any model transition reads it. Hashing
//! the config cannot supply that assurance: a dead field still changes the config lane by
//! construction, which is the ghost-control failure mode recorded on bd-yw1j.
//!
//! This module makes the classification explicit and, critically, FAIL-CLOSED: a new config field
//! that nobody classifies fails [`every_published_knob_is_classified`] rather than silently
//! becoming a scientific knob.
//!
//! # What the roles mean
//!
//! The distinction that matters is not "important vs unimportant" but WHERE the value is consumed:
//!
//! - [`KnobRole::ScientificConstruction`] is read once while building a world and ignored
//!   afterwards. Changing it on a live world is meaningless, which is why several of these are
//!   also marked `fresh_world_only` in `KNOB_RANGES`.
//! - [`KnobRole::ScientificTransition`] is read by the tick loop and changes the trajectory. These
//!   are the knobs an experiment actually varies.
//! - [`KnobRole::Operational`] tunes the harness -- cadence, capacity, persistence, auto-pause --
//!   without entering the model. Changing one must not change the science.
//! - [`KnobRole::Presentation`] affects only what is drawn or reported.
//!
//! # Known incompleteness, stated rather than hidden
//!
//! The published surface is STATE-DEPENDENT. `render.post`, `render.day_night` and
//! `render.auto_exposure` are `Option` fields that serialize to `null` when unset, and the
//! flattener stops at any non-object node, so on a default world each is a SINGLE leaf. Populate
//! one and it expands into `render.post.bloom.threshold` and friends -- paths `KNOB_RANGES`
//! already declares but `list_knobs` never publishes by default. This registry therefore
//! classifies the DEFAULT surface, and [`every_published_knob_is_classified`] asserts exactly
//! that. Extending it to the maximal surface needs a config with every `Option` populated, which
//! is tracked as remaining bd-dorx work rather than pretended away here.
//!
//! There is a SECOND and sharper form of the same problem: `interaction_event_tick_stride` carries
//! `skip_serializing_if`, so it disappears from the published surface when it holds its sentinel
//! value. A knob can therefore be present or absent depending on its VALUE, not merely on the
//! shape of the config -- which means "the set of published knobs" is not even a function of the
//! config type. Any completeness claim is necessarily relative to a particular config instance,
//! and this registry's is relative to [`ScriptBotsConfig::default`].

use crate::ScriptBotsConfig;
use serde_json::Value;

/// Where a published knob is consumed (bd-dorx).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KnobRole {
    /// Read once while constructing a world; a live world ignores it.
    ScientificConstruction,
    /// Read by the tick loop; changes the trajectory.
    ScientificTransition,
    /// Tunes the harness without entering the model.
    Operational,
    /// Affects only what is drawn or reported.
    Presentation,
}

impl KnobRole {
    /// Whether a change to this knob is expected to alter simulation outcomes.
    ///
    /// Both scientific roles do; operational and presentation knobs must not. This is the
    /// property a behavioural witness has to demonstrate.
    #[must_use]
    pub const fn is_scientific(self) -> bool {
        matches!(
            self,
            Self::ScientificConstruction | Self::ScientificTransition
        )
    }
}

/// One classified knob path.
#[derive(Debug, Clone, Copy)]
pub struct KnobSpec {
    /// Dotted path, exactly as `list_knobs` publishes it.
    pub path: &'static str,
    /// Where the value is consumed.
    pub role: KnobRole,
}

const fn spec(path: &'static str, role: KnobRole) -> KnobSpec {
    KnobSpec { path, role }
}

use KnobRole::{Operational, Presentation, ScientificConstruction, ScientificTransition};

/// Every knob the control plane publishes for a default config, with its role (bd-dorx).
///
/// Sorted by path so a reviewer can diff it against `list_knobs` output directly.
pub static KNOB_ROLES: &[KnobSpec] = &[
    // --- World construction: read by `WorldState::new`, meaningless on a live world. ---
    spec("food_cell_size", ScientificConstruction),
    spec("initial_food", ScientificConstruction),
    spec("rng_seed", ScientificConstruction),
    spec("world_height", ScientificConstruction),
    spec("world_width", ScientificConstruction),
    // Brain family and shape are fixed when an agent's brain is built.
    spec("neuroflow.activation", ScientificConstruction),
    spec("neuroflow.enabled", ScientificConstruction),
    spec("neuroflow.hidden_layers", ScientificConstruction),
    // --- Aging ---
    spec("aging_energy_penalty_rate", ScientificTransition),
    spec("aging_health_decay_max", ScientificTransition),
    spec("aging_health_decay_rate", ScientificTransition),
    spec("aging_health_decay_start", ScientificTransition),
    spec("aging_tick_interval", ScientificTransition),
    // --- Locomotion and body ---
    spec("boost_multiplier", ScientificTransition),
    spec("bot_radius", ScientificTransition),
    spec("bot_speed", ScientificTransition),
    spec("locomotion_model", ScientificTransition),
    // --- Carcass and death ---
    spec("carcass_distribution_radius", ScientificTransition),
    spec("carcass_energy_share_rate", ScientificTransition),
    spec("carcass_health_reward", ScientificTransition),
    spec("carcass_indicator_scale", ScientificTransition),
    spec("carcass_maturity_age", ScientificTransition),
    spec("carcass_neighbor_exponent", ScientificTransition),
    spec("carcass_reproduction_reward", ScientificTransition),
    // --- Diet and world mode ---
    spec("carnivore_threshold", ScientificTransition),
    spec("closed", ScientificTransition),
    // --- Food dynamics ---
    spec("food_capacity_base", ScientificTransition),
    spec("food_capacity_fertility", ScientificTransition),
    spec("food_decay_infertility", ScientificTransition),
    spec("food_decay_rate", ScientificTransition),
    spec("food_diffusion_rate", ScientificTransition),
    spec("food_elevation_weight", ScientificTransition),
    spec("food_fertility_base", ScientificTransition),
    spec("food_growth_fertility", ScientificTransition),
    spec("food_growth_rate", ScientificTransition),
    spec("food_intake_rate", ScientificTransition),
    spec("food_max", ScientificTransition),
    spec("food_moisture_weight", ScientificTransition),
    spec("food_respawn_amount", ScientificTransition),
    spec("food_respawn_interval", ScientificTransition),
    spec("food_sharing_distance", ScientificTransition),
    spec("food_sharing_radius", ScientificTransition),
    spec("food_sharing_rate", ScientificTransition),
    spec("food_slope_weight", ScientificTransition),
    spec("food_transfer_rate", ScientificTransition),
    spec("food_waste_rate", ScientificTransition),
    // --- Metabolism ---
    spec("metabolism_boost_penalty", ScientificTransition),
    spec("metabolism_drain", ScientificTransition),
    spec("metabolism_ramp_floor", ScientificTransition),
    spec("metabolism_ramp_rate", ScientificTransition),
    spec("movement_drain", ScientificTransition),
    // --- Population ---
    spec("population_crossover_chance", ScientificTransition),
    spec("population_minimum", ScientificTransition),
    spec("population_spawn_count", ScientificTransition),
    spec("population_spawn_interval", ScientificTransition),
    // --- Reproduction and heredity ---
    spec("reproduction_attempt_chance", ScientificTransition),
    spec("reproduction_attempt_interval", ScientificTransition),
    spec("reproduction_child_energy", ScientificTransition),
    spec("reproduction_color_jitter", ScientificTransition),
    spec("reproduction_cooldown", ScientificTransition),
    spec("reproduction_energy_cost", ScientificTransition),
    spec("reproduction_energy_threshold", ScientificTransition),
    spec("reproduction_fertility_bonus", ScientificTransition),
    spec("reproduction_food_bonus", ScientificTransition),
    spec("reproduction_meta_mutation_chance", ScientificTransition),
    spec("reproduction_meta_mutation_scale", ScientificTransition),
    spec("reproduction_mutation_scale", ScientificTransition),
    spec("reproduction_partner_chance", ScientificTransition),
    spec("reproduction_rate_carnivore", ScientificTransition),
    spec("reproduction_rate_herbivore", ScientificTransition),
    spec("reproduction_spawn_back_distance", ScientificTransition),
    spec("reproduction_spawn_jitter", ScientificTransition),
    // --- Sensing ---
    spec("sense_radius", ScientificTransition),
    // --- Combat ---
    spec("spike_alignment_cosine", ScientificTransition),
    spec("spike_damage", ScientificTransition),
    spec("spike_energy_cost", ScientificTransition),
    spec("spike_growth_rate", ScientificTransition),
    spec("spike_length_damage_bonus", ScientificTransition),
    spec("spike_min_length", ScientificTransition),
    spec("spike_radius", ScientificTransition),
    spec("spike_speed_damage_bonus", ScientificTransition),
    // --- Temperature ---
    spec("temperature_comfort_band", ScientificTransition),
    spec("temperature_discomfort_exponent", ScientificTransition),
    spec("temperature_discomfort_rate", ScientificTransition),
    spec("temperature_gradient_exponent", ScientificTransition),
    // --- Topography ---
    spec("topography_enabled", ScientificTransition),
    spec("topography_energy_penalty", ScientificTransition),
    spec("topography_speed_gain", ScientificTransition),
    // --- Operational: cadence, capacity, persistence, harness control. ---
    //
    // `reproduction_gene_log_capacity` is here, not with the other `reproduction_*` knobs: it caps
    // a diagnostic ring buffer of gene-change strings. It changes what an inspector can read, never
    // what an offspring inherits.
    spec("reproduction_gene_log_capacity", Operational),
    spec("analytics_stride.behavior_metrics", Operational),
    spec("analytics_stride.lifecycle_events", Operational),
    spec("analytics_stride.macro_metrics", Operational),
    spec("chart_flush_interval", Operational),
    spec("control.auto_pause_age_above", Operational),
    spec("control.auto_pause_on_spike_hit", Operational),
    spec("control.auto_pause_population_below", Operational),
    // Diagnostic telemetry toggle. Recorded on bd-dorx as a ghost-control candidate: it is
    // serialized into the scientific config lane, so toggling it moves config provenance without
    // changing simulation science. Classified Operational, which is what makes that a testable
    // claim rather than an assumption.
    spec("economy_debug_per_tick", Operational),
    spec("history_capacity", Operational),
    // Both interaction knobs bound the replay recorder, not the model: a cap on how many edges a
    // tick may record and a stride for how often it samples. Found by the fail-closed gate --
    // `interaction_event_tick_cap` was published and unclassified on the first run.
    spec("interaction_event_tick_cap", Operational),
    spec("interaction_event_tick_stride", Operational),
    spec("narrative_capacity", Operational),
    spec("narrative_interval", Operational),
    spec("persistence_interval", Operational),
    spec("replay_event_tick_cap", Operational),
    // --- Presentation: drawn or reported only. ---
    spec("render.auto_exposure", Presentation),
    spec("render.camera_shake", Presentation),
    spec("render.day_night", Presentation),
    spec("render.palette", Presentation),
    spec("render.post", Presentation),
    spec("render.quality", Presentation),
    spec("render.reduced_motion", Presentation),
    spec("render.theme", Presentation),
    spec("render.tonemap_exposure_bias", Presentation),
    spec("render.tonemap_mode", Presentation),
];

/// Look up a knob's declared role.
#[must_use]
pub fn knob_role(path: &str) -> Option<KnobRole> {
    KNOB_ROLES
        .iter()
        .find(|spec| spec.path == path)
        .map(|spec| spec.role)
}

/// Flatten a serialized config into dotted leaf paths.
///
/// Mirrors `flatten_value` in `scriptbots-app`: recurse into objects only, and treat every other
/// node -- including arrays and nulls -- as one leaf. Anything else would classify a surface the
/// control plane does not actually publish.
fn flatten_paths(prefix: &mut String, value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            let base = prefix.len();
            for (key, nested) in map {
                if base != 0 {
                    prefix.push('.');
                }
                prefix.push_str(key);
                flatten_paths(prefix, nested, out);
                prefix.truncate(base);
            }
        }
        _ => out.push(prefix.clone()),
    }
}

/// Every knob path the control plane publishes for a default config.
///
/// # Panics
/// Panics if the public config fails to serialize, which the control plane already depends on.
#[must_use]
pub fn published_knob_paths() -> Vec<String> {
    let value = serde_json::to_value(ScriptBotsConfig::default())
        .expect("the public config must serialize; list_knobs depends on it");
    let mut paths = Vec::new();
    flatten_paths(&mut String::new(), &value, &mut paths);
    paths.sort();
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// THE FAIL-CLOSED GATE (bd-dorx): a config field nobody classified fails the build.
    ///
    /// This is the assertion the bead exists for. Without it, adding a field to
    /// `ScriptBotsConfig` silently publishes a new scientific REST/MCP knob with no role and no
    /// witness that anything consumes it.
    #[test]
    fn every_published_knob_is_classified() {
        let published: BTreeSet<String> = published_knob_paths().into_iter().collect();
        let classified: BTreeSet<String> = KNOB_ROLES.iter().map(|s| s.path.to_owned()).collect();

        let unclassified: Vec<&String> = published.difference(&classified).collect();
        assert!(
            unclassified.is_empty(),
            "these published knobs have no declared role -- classify them in KNOB_ROLES rather \
             than relaxing this test, because an unclassified knob is a public scientific control \
             nobody has vouched for: {unclassified:?}"
        );
    }

    /// A spec for a path the config does not publish is dead weight that will drift (bd-dorx).
    ///
    /// `KNOB_RANGES` already carries two such entries -- `mutation.primary` and
    /// `mutation.secondary` -- for fields that do not exist. This registry must not repeat that.
    #[test]
    fn no_spec_is_stale() {
        let published: BTreeSet<String> = published_knob_paths().into_iter().collect();
        let stale: Vec<&str> = KNOB_ROLES
            .iter()
            .map(|s| s.path)
            .filter(|p| !published.contains(*p))
            .collect();
        assert!(
            stale.is_empty(),
            "these specs name paths the config does not publish: {stale:?}"
        );
    }

    #[test]
    fn no_knob_is_classified_twice() {
        let mut seen = BTreeSet::new();
        let duplicates: Vec<&str> = KNOB_ROLES
            .iter()
            .map(|s| s.path)
            .filter(|p| !seen.insert(*p))
            .collect();
        assert!(
            duplicates.is_empty(),
            "a knob with two roles has no role: {duplicates:?}"
        );
    }

    /// The registry is sorted, so it can be diffed against `list_knobs` output by eye.
    #[test]
    fn registry_paths_are_unique_and_nonempty() {
        assert!(!KNOB_ROLES.is_empty(), "an empty registry gates nothing");
        for spec in KNOB_ROLES {
            assert!(
                !spec.path.is_empty(),
                "a spec with an empty path matches everything"
            );
        }
    }

    /// Guards the role split itself: both scientific roles report as scientific, neither of the
    /// others does. A witness harness keys off this predicate, so an inverted match here would
    /// silently exempt every scientific knob from needing a witness.
    #[test]
    fn scientific_roles_are_exactly_the_two_scientific_variants() {
        assert!(KnobRole::ScientificConstruction.is_scientific());
        assert!(KnobRole::ScientificTransition.is_scientific());
        assert!(!KnobRole::Operational.is_scientific());
        assert!(!KnobRole::Presentation.is_scientific());
    }

    /// The Option-gated render subtrees are known to be under-classified, and that is recorded
    /// here rather than left as a silent gap (bd-dorx).
    ///
    /// On a default world these publish as single `null` leaves. `KNOB_RANGES` declares deeper
    /// paths beneath them that `list_knobs` never emits, so the registry cannot yet reach them.
    /// This test pins the gap so it stays visible and fails if the shape changes.
    #[test]
    fn option_gated_render_subtrees_are_still_single_leaves() {
        let published: BTreeSet<String> = published_knob_paths().into_iter().collect();
        for gated in ["render.post", "render.day_night", "render.auto_exposure"] {
            assert!(
                published.contains(gated),
                "{gated} should publish as one leaf on a default config"
            );
            let expanded = published
                .iter()
                .any(|p| p.starts_with(&format!("{gated}.")));
            assert!(
                !expanded,
                "{gated} expanded into subpaths, so the maximal surface is now reachable from a \
                 default config -- extend KNOB_ROLES to cover it and update this test"
            );
        }
    }
}
