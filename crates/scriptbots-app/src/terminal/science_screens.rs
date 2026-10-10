//! Science screens, data models, query states, and renderers for FrankenTUI Evolution Lab (bd-2z0.6.6).
//!
//! Provides routed science screens:
//! - **Lineages**: founders, branches, reproductive success, clades, and lineage events.
//! - **Brain Arena**: family cohorts, activation inspection, sensor attributions, effective outputs.
//! - **Experiments**: variants, seed cohorts, progress, confidence intervals, paired comparisons.
//! - **Replay**: checkpoints, scrubber, first-divergence tracking, export bundles.
//! - **Environment**: terrain distribution, hydrology currents, fertility, hazards, regrowth.
//! - **Diagnostics**: queue depths, storage lag, stage timings, build provenance, digest neutrality.
//!
//! Every screen implements a bounded, cancellable query state machine (`Empty`, `Loading`,
//! `Error`, `Data`), supports keyboard navigation, preserves simulation digest neutrality,
//! and renders to deterministic testable frame buffers.

use super::frankentui_shell::SHELL_FOOTER_ROWS;
use ftui::render::cell::Cell;
use ftui::render::drawing::Draw;
use ftui::render::frame::Frame;
use serde::{Deserialize, Serialize};

/// Bounded, cancellable query state for a routed science screen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScreenDataState<T> {
    /// No data loaded yet.
    Empty { message: String },
    /// Bounded asynchronous query currently in flight.
    Loading { query_id: u64, operation: String },
    /// Query or data retrieval failed.
    Error { query_id: u64, error: String },
    /// Populated data view ready for rendering.
    Data(T),
}

impl<T> Default for ScreenDataState<T> {
    fn default() -> Self {
        Self::Empty {
            message: "No data loaded yet".into(),
        }
    }
}

impl<T> ScreenDataState<T> {
    pub fn empty(message: impl Into<String>) -> Self {
        Self::Empty {
            message: message.into(),
        }
    }

    pub fn loading(query_id: u64, operation: impl Into<String>) -> Self {
        Self::Loading {
            query_id,
            operation: operation.into(),
        }
    }

    pub fn error(query_id: u64, error: impl Into<String>) -> Self {
        Self::Error {
            query_id,
            error: error.into(),
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty { .. })
    }

    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Loading { .. })
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    pub fn is_data(&self) -> bool {
        matches!(self, Self::Data(_))
    }

    pub fn data(&self) -> Option<&T> {
        match self {
            Self::Data(d) => Some(d),
            _ => None,
        }
    }

    pub fn data_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Data(d) => Some(d),
            _ => None,
        }
    }

    /// Cancel in-flight query if query_id matches, transitioning back to Empty.
    pub fn cancel_if_matching(&mut self, target_query_id: u64, fallback_empty_msg: &str) -> bool {
        if let Self::Loading { query_id, .. } = self
            && *query_id == target_query_id
        {
            *self = Self::Empty {
                message: format!("Query {target_query_id} cancelled. {fallback_empty_msg}"),
            };
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// 1. Lineages Screen Data Types
// ---------------------------------------------------------------------------

/// Individual clade or species entry in the phylogeny view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CladeEntry {
    pub species_id: u64,
    pub name: String,
    pub founder_id: u64,
    pub population: usize,
    pub peak_population: usize,
    pub origin_tick: u64,
    pub extinction_tick: Option<u64>,
    pub mean_diet: f32,
    pub fitness_score: f32,
}

/// A significant speciation, radiation, or extinction event in the timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageEventEntry {
    pub tick: u64,
    pub event_kind: String,
    pub description: String,
}

/// Descendant counts and contribution observed in the canonical ancestry graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FounderLineageRow {
    pub founder_uid: u64,
    /// Descendants with no death in the retained birth/death stream. Migration
    /// is not observed by this graph, so this is not a local population census.
    pub descendants_without_recorded_death: usize,
    pub total_descendants: usize,
    pub contribution_share: f32,
}

/// View data for the Lineages science screen.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineagesViewData {
    pub total_founders: Option<usize>,
    pub active_clades: Option<usize>,
    pub extinct_clades: Option<usize>,
    pub clades: Vec<CladeEntry>,
    pub founder_rows: Vec<FounderLineageRow>,
    pub observed_tick: Option<u64>,
    pub recent_events: Vec<LineageEventEntry>,
    pub selected_index: usize,
}

// ---------------------------------------------------------------------------
// 2. Brain Arena Screen Data Types
// ---------------------------------------------------------------------------

/// Cohort summary for a specific brain family (e.g. MLP, DWRAON, NeuroFlow).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainFamilyCohort {
    pub family: String,
    pub agent_count: usize,
    pub mean_fitness: Option<f32>,
    pub mean_eval_nanos: Option<u64>,
}

/// Inspected brain layer and sensor attribution telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainInspectionSummary {
    pub agent_id: u64,
    pub family: String,
    pub layer_shapes: Vec<String>,
    pub top_sensor_attributions: Vec<(String, f32)>,
    pub effective_outputs: Vec<(String, f32)>,
}

/// View data for the Brain Arena science screen.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainArenaViewData {
    pub family_breakdown: Vec<BrainFamilyCohort>,
    pub selected_brain: Option<BrainInspectionSummary>,
    pub total_evaluations: Option<u64>,
    pub selected_family_index: usize,
}

// ---------------------------------------------------------------------------
// 3. Experiments Screen Data Types
// ---------------------------------------------------------------------------

/// Individual arm or variant in a matched-seed experiment batch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentArmView {
    pub variant_id: String,
    pub brain_family: String,
    pub seed: u64,
    pub current_tick: Option<u64>,
    pub target_ticks: u64,
    pub mean_fitness: Option<f32>,
    pub uncertainty_ci_95: Option<(f32, f32)>,
    pub digest: String,
}

/// Pairwise statistical comparison between baseline and challenger arms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentComparisonSummary {
    pub baseline_arm: String,
    pub challenger_arm: String,
    pub effect_size_hedges_g: f32,
    pub p_value_estimate: f32,
    pub verdict: String,
}

/// View data for the Experiments science screen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentsViewData {
    pub experiment_id: String,
    pub batch_status: String,
    pub active_arms: Vec<ExperimentArmView>,
    pub selected_arm_index: usize,
    pub comparison_summary: Option<ExperimentComparisonSummary>,
}

impl Default for ExperimentsViewData {
    fn default() -> Self {
        Self {
            experiment_id: String::new(),
            batch_status: "Idle".into(),
            active_arms: Vec::new(),
            selected_arm_index: 0,
            comparison_summary: None,
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Replay Screen Data Types
// ---------------------------------------------------------------------------

/// Checkpoint metadata entry for replay, divergence checking, and branching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointViewEntry {
    pub checkpoint_id: String,
    pub tick: u64,
    pub agent_count: Option<usize>,
    /// Scientific state digest, when recorded separately from the artifact checksum.
    pub digest: Option<String>,
    pub file_size_bytes: u64,
}

/// First divergence diagnostics when replaying or validating determinism.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayDivergenceInfo {
    pub divergence_tick: u64,
    pub expected_digest: String,
    pub observed_digest: String,
    pub falsified_subsystem: String,
}

/// View data for the Replay & Checkpoint science screen.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayViewData {
    pub checkpoints: Vec<CheckpointViewEntry>,
    /// Latest live host tick observed independently of any replay execution.
    pub observed_live_tick: Option<u64>,
    pub first_divergence: Option<ReplayDivergenceInfo>,
    pub selected_checkpoint_index: usize,
    pub export_receipt: Option<String>,
}

// ---------------------------------------------------------------------------
// 5. Environment Screen Data Types
// ---------------------------------------------------------------------------

/// View data for the Environment & Biome science screen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentViewData {
    pub world_size: (u32, u32),
    pub terrain_distribution: Vec<(String, f32)>,
    pub mean_fertility: Option<f32>,
    pub mean_temperature: Option<f32>,
    pub total_food_energy: f32,
    pub current_flow_rate: Option<f32>,
    /// Reported timed intervention labels, rather than a census of environmental hazards.
    pub reported_timed_interventions: Vec<String>,
    pub regrowth_rate: f32,
}

impl Default for EnvironmentViewData {
    fn default() -> Self {
        Self {
            world_size: (100, 100),
            terrain_distribution: Vec::new(),
            mean_fertility: None,
            mean_temperature: None,
            total_food_energy: 0.0,
            current_flow_rate: None,
            reported_timed_interventions: Vec::new(),
            regrowth_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Diagnostics Screen Data Types
// ---------------------------------------------------------------------------

/// View data for the System Diagnostics & Health science screen.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticsViewData {
    pub host_queue_depth: usize,
    pub host_queue_capacity: Option<usize>,
    /// Latest analytics snapshot committed tick; this is not the durable storage watermark.
    pub analytics_committed_tick: Option<u64>,
    pub storage_uncommitted_records: Option<usize>,
    pub snapshot_timings_ms: Vec<(String, f32)>,
    pub sim_tick_hz: Option<f32>,
    pub render_fps: Option<f32>,
    pub build_commit: String,
    pub digest_neutrality_verified: Option<bool>,
}

impl Default for DiagnosticsViewData {
    fn default() -> Self {
        Self {
            host_queue_depth: 0,
            host_queue_capacity: None,
            analytics_committed_tick: None,
            storage_uncommitted_records: None,
            snapshot_timings_ms: Vec::new(),
            sim_tick_hz: None,
            render_fps: None,
            build_commit: "unknown".into(),
            digest_neutrality_verified: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Unified Science Screens Container
// ---------------------------------------------------------------------------

/// Composite state of all routed science screens.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScienceScreensState {
    pub lineages: ScreenDataState<LineagesViewData>,
    pub brain_arena: ScreenDataState<BrainArenaViewData>,
    pub experiments: ScreenDataState<ExperimentsViewData>,
    pub replay: ScreenDataState<ReplayViewData>,
    pub environment: ScreenDataState<EnvironmentViewData>,
    pub diagnostics: ScreenDataState<DiagnosticsViewData>,
    pub active_query_counter: u64,
    pub export_history: Vec<String>,
    pub branch_history: Vec<String>,
}

impl Default for ScienceScreensState {
    fn default() -> Self {
        Self {
            lineages: ScreenDataState::empty("No active lineages recorded yet"),
            brain_arena: ScreenDataState::empty("No brain arena evaluations recorded yet"),
            experiments: ScreenDataState::empty("No experiment batches configured or running"),
            replay: ScreenDataState::empty("No simulation checkpoints available for replay"),
            environment: ScreenDataState::empty("No environment layer telemetry available"),
            diagnostics: ScreenDataState::empty("Diagnostics telemetry awaiting host connection"),
            active_query_counter: 0,
            export_history: Vec::new(),
            branch_history: Vec::new(),
        }
    }
}

impl ScienceScreensState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate next monotonic query identifier for a cancellable query.
    pub fn next_query_id(&mut self) -> u64 {
        self.active_query_counter += 1;
        self.active_query_counter
    }

    /// Cancel an active query across all screens if the ID matches.
    pub fn cancel_query(&mut self, query_id: u64) -> bool {
        let mut cancelled = false;
        cancelled |= self
            .lineages
            .cancel_if_matching(query_id, "Lineages query cancelled.");
        cancelled |= self
            .brain_arena
            .cancel_if_matching(query_id, "Brain Arena query cancelled.");
        cancelled |= self
            .experiments
            .cancel_if_matching(query_id, "Experiments query cancelled.");
        cancelled |= self
            .replay
            .cancel_if_matching(query_id, "Replay query cancelled.");
        cancelled |= self
            .environment
            .cancel_if_matching(query_id, "Environment query cancelled.");
        cancelled |= self
            .diagnostics
            .cancel_if_matching(query_id, "Diagnostics query cancelled.");
        cancelled
    }
}

// ---------------------------------------------------------------------------
// Screen Renderers
// ---------------------------------------------------------------------------

const SCIENCE_TABLE_ROWS: usize = 5;

fn first_visible_table_index(selected_index: usize, visible_rows: usize) -> usize {
    selected_index.saturating_sub(visible_rows.saturating_sub(1))
}

fn observation_text<T>(value: Option<T>, format_value: impl FnOnce(T) -> String) -> String {
    value.map_or_else(|| "unavailable".into(), format_value)
}

fn print_text(frame: &mut Frame, x: u16, y: u32, text: &str) {
    let Ok(y) = u16::try_from(y) else {
        return;
    };
    let max_w = frame.width();
    if y >= frame.height() || x >= max_w {
        return;
    }
    frame.print_text_clipped(x, y, text, Cell::default(), max_w);
}

fn render_state_banner(
    frame: &mut Frame,
    y: u32,
    empty_msg: Option<&str>,
    loading_op: Option<(u64, &str)>,
    err: Option<(u64, &str)>,
) {
    if let Some(msg) = empty_msg {
        print_text(frame, 4, y, &format!("  [EMPTY] {msg}"));
        print_text(
            frame,
            4,
            y + 2,
            "  Press [Space] to resume sim or query host",
        );
    } else if let Some((qid, op)) = loading_op {
        print_text(frame, 4, y, &format!("  [LOADING] Query #{qid}: {op}..."));
        print_text(frame, 4, y + 2, "  Press [Esc] to cancel in-flight query");
    } else if let Some((qid, error)) = err {
        print_text(
            frame,
            4,
            y,
            &format!("  [ERROR] Query #{qid} failed: {error}"),
        );
        print_text(frame, 4, y + 2, "  Press [Esc] to dismiss or retry action");
    }
}

/// Render Lineages science screen (founders, clades, events).
pub fn render_lineages_screen(
    frame: &mut Frame,
    state: &ScreenDataState<LineagesViewData>,
    y_start: u16,
) {
    let y_start = u32::from(y_start);
    print_text(frame, 2, y_start, "LINEAGES & PHYLOGENY LAB");
    match state {
        ScreenDataState::Empty { message } => {
            render_state_banner(frame, y_start + 2, Some(message), None, None);
        }
        ScreenDataState::Loading {
            query_id,
            operation,
        } => {
            render_state_banner(frame, y_start + 2, None, Some((*query_id, operation)), None);
        }
        ScreenDataState::Error { query_id, error } => {
            render_state_banner(frame, y_start + 2, None, None, Some((*query_id, error)));
        }
        ScreenDataState::Data(data) => {
            let summary = format!(
                "  Total Founders: {:<5} Active Clades: {:<5} Extinct Clades: {:<5}",
                observation_text(data.total_founders, |value| value.to_string()),
                observation_text(data.active_clades, |value| value.to_string()),
                observation_text(data.extinct_clades, |value| value.to_string())
            );
            print_text(frame, 2, y_start + 2, &summary);

            if !data.founder_rows.is_empty() {
                let observed_tick = observation_text(data.observed_tick, |value| value.to_string());
                print_text(
                    frame,
                    2,
                    y_start + 4,
                    "  FOUNDER DESCENDANTS (recorded births/deaths)",
                );
                print_text(
                    frame,
                    4,
                    y_start + 5,
                    "  Founder UID      No Recorded Death   Total Descendants  Contribution Share",
                );
                print_text(
                    frame,
                    4,
                    y_start + 6,
                    &format!(
                        "  canonical arrivals/deaths through tick {observed_tick}; migration unobserved"
                    ),
                );
                let content_end = u32::from(frame.height().saturating_sub(SHELL_FOOTER_ROWS));
                let rows = (y_start.saturating_add(7)..content_end).take(SCIENCE_TABLE_ROWS);
                let visible_rows = rows.clone().count();
                let founders = data
                    .founder_rows
                    .iter()
                    .enumerate()
                    .skip(first_visible_table_index(data.selected_index, visible_rows));
                for (row, (idx, founder)) in rows.zip(founders) {
                    let cursor = if idx == data.selected_index { ">" } else { " " };
                    let line = format!(
                        " {} {:<16} {:<19} {:<18} {:.3}",
                        cursor,
                        founder.founder_uid,
                        founder.descendants_without_recorded_death,
                        founder.total_descendants,
                        founder.contribution_share
                    );
                    print_text(frame, 4, row, &line);
                }
            } else {
                print_text(
                    frame,
                    2,
                    y_start + 4,
                    "  CLADE TABLE (Deterministic Phylogeny Sort):",
                );
                print_text(
                    frame,
                    4,
                    y_start + 5,
                    "  ID   Name             Founder  Pop    Peak   Origin  Diet   Fitness",
                );
                print_text(
                    frame,
                    4,
                    y_start + 6,
                    "  -------------------------------------------------------------------",
                );

                if data.clades.is_empty() {
                    print_text(frame, 6, y_start + 7, "(No clades segmented yet)");
                } else {
                    let content_end = u32::from(frame.height().saturating_sub(SHELL_FOOTER_ROWS));
                    let rows = (y_start.saturating_add(7)..content_end).take(SCIENCE_TABLE_ROWS);
                    let visible_rows = rows.clone().count();
                    let clades = data
                        .clades
                        .iter()
                        .enumerate()
                        .skip(first_visible_table_index(data.selected_index, visible_rows));
                    for (row, (idx, clade)) in rows.zip(clades) {
                        let cursor = if idx == data.selected_index { ">" } else { " " };
                        let diet_str = if clade.mean_diet < 0.35 {
                            "Plant"
                        } else if clade.mean_diet > 0.65 {
                            "Pred "
                        } else {
                            "Omni "
                        };
                        let line = format!(
                            " {} {:<4} {:<16} {:<8} {:<6} {:<6} {:<7} {:<6} {:.2}",
                            cursor,
                            clade.species_id,
                            clade.name,
                            clade.founder_id,
                            clade.population,
                            clade.peak_population,
                            clade.origin_tick,
                            diet_str,
                            clade.fitness_score
                        );
                        print_text(frame, 4, row, &line);
                    }
                }
            }

            let events_y = y_start + 13;
            print_text(frame, 2, events_y, "  RECENT PHYLOGENY EVENTS:");
            if data.recent_events.is_empty() {
                print_text(frame, 4, events_y + 1, "  (No lineage events reported)");
            } else {
                let rows = (events_y.saturating_add(1)..u32::from(frame.height())).take(3);
                for (row, ev) in rows.zip(&data.recent_events) {
                    let line = format!(
                        "  Tick {:<6} [{:<10}] {}",
                        ev.tick, ev.event_kind, ev.description
                    );
                    print_text(frame, 4, row, &line);
                }
            }
        }
    }
}

/// Render Brain Arena science screen (cohorts, layer activations, sensor attributions).
pub fn render_brain_arena_screen(
    frame: &mut Frame,
    state: &ScreenDataState<BrainArenaViewData>,
    y_start: u16,
) {
    let y_start = u32::from(y_start);
    print_text(frame, 2, y_start, "BRAIN ARENA & EVALUATOR COHORTS");
    match state {
        ScreenDataState::Empty { message } => {
            render_state_banner(frame, y_start + 2, Some(message), None, None);
        }
        ScreenDataState::Loading {
            query_id,
            operation,
        } => {
            render_state_banner(frame, y_start + 2, None, Some((*query_id, operation)), None);
        }
        ScreenDataState::Error { query_id, error } => {
            render_state_banner(frame, y_start + 2, None, None, Some((*query_id, error)));
        }
        ScreenDataState::Data(data) => {
            let header = format!(
                "  Total Brain Evaluations: {:<10} Cohort Families: {}",
                observation_text(data.total_evaluations, |value| value.to_string()),
                data.family_breakdown.len()
            );
            print_text(frame, 2, y_start + 2, &header);

            print_text(frame, 2, y_start + 4, "  FAMILY COHORT PERFORMANCE:");
            print_text(
                frame,
                4,
                y_start + 5,
                "  Family         Count   Mean Fitness  Eval Latency",
            );
            print_text(
                frame,
                4,
                y_start + 6,
                "  --------------------------------------------------",
            );

            if data.family_breakdown.is_empty() {
                print_text(frame, 6, y_start + 7, "(No family cohorts active)");
            } else {
                let content_end = u32::from(frame.height().saturating_sub(SHELL_FOOTER_ROWS));
                let rows = (y_start.saturating_add(7)..content_end).take(SCIENCE_TABLE_ROWS);
                let visible_rows = rows.clone().count();
                let cohorts =
                    data.family_breakdown
                        .iter()
                        .enumerate()
                        .skip(first_visible_table_index(
                            data.selected_family_index,
                            visible_rows,
                        ));
                for (row, (idx, cohort)) in rows.zip(cohorts) {
                    let cursor = if idx == data.selected_family_index {
                        ">"
                    } else {
                        " "
                    };
                    let line = format!(
                        " {} {:<14} {:<7} {:<12} {}",
                        cursor,
                        cohort.family,
                        cohort.agent_count,
                        observation_text(cohort.mean_fitness, |value| format!("{value:.3}")),
                        observation_text(cohort.mean_eval_nanos, |value| format!("{value} ns"))
                    );
                    print_text(frame, 4, row, &line);
                }
            }

            let insp_y = y_start + 12;
            print_text(
                frame,
                2,
                insp_y,
                "  SELECTED BRAIN ACTIVATION & TOP-K ATTRIBUTIONS:",
            );
            if let Some(brain) = &data.selected_brain {
                let info = format!(
                    "  Agent: {:<6} Family: {:<10} Architecture: {}",
                    brain.agent_id,
                    brain.family,
                    brain.layer_shapes.join(" -> ")
                );
                print_text(frame, 4, insp_y + 1, &info);

                let mut attr_strs = Vec::new();
                for (name, val) in &brain.top_sensor_attributions {
                    attr_strs.push(format!("{name}:{val:+.2}"));
                }
                print_text(
                    frame,
                    4,
                    insp_y + 2,
                    &format!("  Sensors: {}", attr_strs.join(" | ")),
                );

                let mut out_strs = Vec::new();
                for (name, val) in &brain.effective_outputs {
                    out_strs.push(format!("{name}:{val:+.2}"));
                }
                print_text(
                    frame,
                    4,
                    insp_y + 3,
                    &format!("  Outputs: {}", out_strs.join(" | ")),
                );
            } else {
                print_text(
                    frame,
                    4,
                    insp_y + 1,
                    "  (No single agent selected for deep brain probe)",
                );
            }
        }
    }
}

/// Render Experiments science screen (variants, confidence intervals, comparison).
pub fn render_experiments_screen(
    frame: &mut Frame,
    state: &ScreenDataState<ExperimentsViewData>,
    y_start: u16,
) {
    let y_start = u32::from(y_start);
    print_text(
        frame,
        2,
        y_start,
        "MATCHED-SEED EXPERIMENTS & INTERVENTIONS",
    );
    match state {
        ScreenDataState::Empty { message } => {
            render_state_banner(frame, y_start + 2, Some(message), None, None);
        }
        ScreenDataState::Loading {
            query_id,
            operation,
        } => {
            render_state_banner(frame, y_start + 2, None, Some((*query_id, operation)), None);
        }
        ScreenDataState::Error { query_id, error } => {
            render_state_banner(frame, y_start + 2, None, None, Some((*query_id, error)));
        }
        ScreenDataState::Data(data) => {
            let status = format!(
                "  Experiment ID: {:<16} Batch Status: {}",
                data.experiment_id, data.batch_status
            );
            print_text(frame, 2, y_start + 2, &status);

            print_text(frame, 2, y_start + 4, "  ACTIVE EXPERIMENTAL ARMS:");
            print_text(
                frame,
                4,
                y_start + 5,
                "  Arm Variant    Family     Seed    Ticks/Target  Fitness  95% Conf Interval",
            );
            print_text(
                frame,
                4,
                y_start + 6,
                "  --------------------------------------------------------------------------",
            );

            if data.active_arms.is_empty() {
                print_text(frame, 6, y_start + 7, "(No experiment arms defined)");
            } else {
                let content_end = u32::from(frame.height().saturating_sub(SHELL_FOOTER_ROWS));
                let rows = (y_start.saturating_add(7)..content_end).take(SCIENCE_TABLE_ROWS);
                let visible_rows = rows.clone().count();
                let arms = data
                    .active_arms
                    .iter()
                    .enumerate()
                    .skip(first_visible_table_index(
                        data.selected_arm_index,
                        visible_rows,
                    ));
                for (row, (idx, arm)) in rows.zip(arms) {
                    let cursor = if idx == data.selected_arm_index {
                        ">"
                    } else {
                        " "
                    };
                    let line = format!(
                        " {} {:<14} {:<10} {:<7} {:>5}/{:<5}   {:<11} {}",
                        cursor,
                        arm.variant_id,
                        arm.brain_family,
                        arm.seed,
                        observation_text(arm.current_tick, |value| value.to_string()),
                        arm.target_ticks,
                        observation_text(arm.mean_fitness, |value| format!("{value:.2}")),
                        observation_text(arm.uncertainty_ci_95, |(low, high)| {
                            format!("[{low:.2}, {high:.2}]")
                        })
                    );
                    print_text(frame, 4, row, &line);
                }
            }

            let comp_y = y_start + 12;
            print_text(frame, 2, comp_y, "  STATISTICAL EFFECT SIZE & VERDICT:");
            if let Some(comp) = &data.comparison_summary {
                let comp_line = format!(
                    "  Baseline: {:<12} Challenger: {:<12} Hedges' g: {:+.3} (p ~= {:.3})",
                    comp.baseline_arm,
                    comp.challenger_arm,
                    comp.effect_size_hedges_g,
                    comp.p_value_estimate
                );
                print_text(frame, 4, comp_y + 1, &comp_line);
                print_text(
                    frame,
                    4,
                    comp_y + 2,
                    &format!("  Verdict: {}", comp.verdict),
                );
            } else {
                print_text(
                    frame,
                    4,
                    comp_y + 1,
                    "  (Select two arms and press [C] to run paired comparison)",
                );
            }
        }
    }
}

/// Render Replay science screen (checkpoints, divergence, export).
pub fn render_replay_screen(
    frame: &mut Frame,
    state: &ScreenDataState<ReplayViewData>,
    y_start: u16,
) {
    let y_start = u32::from(y_start);
    print_text(
        frame,
        2,
        y_start,
        "DETERMINISTIC REPLAY & FIRST-DIVERGENCE TRACKER",
    );
    match state {
        ScreenDataState::Empty { message } => {
            render_state_banner(frame, y_start + 2, Some(message), None, None);
        }
        ScreenDataState::Loading {
            query_id,
            operation,
        } => {
            render_state_banner(frame, y_start + 2, None, Some((*query_id, operation)), None);
        }
        ScreenDataState::Error { query_id, error } => {
            render_state_banner(frame, y_start + 2, None, None, Some((*query_id, error)));
        }
        ScreenDataState::Data(data) => {
            let live_observation = format!(
                "  Observed live tick: {:<11} Checkpoints Available: {}",
                observation_text(data.observed_live_tick, |value| value.to_string()),
                data.checkpoints.len()
            );
            print_text(frame, 2, y_start + 2, &live_observation);

            print_text(frame, 2, y_start + 4, "  CHECKPOINT PERSISTENCE CATALOG:");
            print_text(
                frame,
                4,
                y_start + 5,
                "  Checkpoint ID       Tick    Agents  Size(KB)  State Digest",
            );
            print_text(
                frame,
                4,
                y_start + 6,
                "  -------------------------------------------------------------",
            );

            if data.checkpoints.is_empty() {
                print_text(
                    frame,
                    6,
                    y_start + 7,
                    "(No checkpoints recorded. Press [K] to create one)",
                );
            } else {
                let content_end = u32::from(frame.height().saturating_sub(SHELL_FOOTER_ROWS));
                let rows = (y_start.saturating_add(7)..content_end).take(SCIENCE_TABLE_ROWS);
                let visible_rows = rows.clone().count();
                let checkpoints =
                    data.checkpoints
                        .iter()
                        .enumerate()
                        .skip(first_visible_table_index(
                            data.selected_checkpoint_index,
                            visible_rows,
                        ));
                for (row, (idx, cp)) in rows.zip(checkpoints) {
                    let cursor = if idx == data.selected_checkpoint_index {
                        ">"
                    } else {
                        " "
                    };
                    let kb = cp.file_size_bytes / 1024;
                    let line = format!(
                        " {} {:<19} {:<7} {:<7} {:<8}  {:.12}",
                        cursor,
                        cp.checkpoint_id,
                        cp.tick,
                        observation_text(cp.agent_count, |value| value.to_string()),
                        kb,
                        cp.digest.as_deref().unwrap_or("unavailable")
                    );
                    print_text(frame, 4, row, &line);
                }
            }

            let div_y = y_start + 12;
            print_text(
                frame,
                2,
                div_y,
                "  DETERMINISM VERIFICATION & FIRST DIVERGENCE:",
            );
            if let Some(div) = &data.first_divergence {
                print_text(
                    frame,
                    4,
                    div_y + 1,
                    &format!(
                        "  ✗ DIVERGENCE DETECTED at Tick {}: Subsystem '{}'",
                        div.divergence_tick, div.falsified_subsystem
                    ),
                );
                print_text(
                    frame,
                    4,
                    div_y + 2,
                    &format!(
                        "    Expected: {}  Observed: {}",
                        div.expected_digest, div.observed_digest
                    ),
                );
            } else {
                print_text(
                    frame,
                    4,
                    div_y + 1,
                    "  No replay comparison recorded; divergence status unavailable",
                );
            }

            if let Some(receipt) = &data.export_receipt {
                print_text(frame, 2, div_y + 3, &format!("  Export: {receipt}"));
            }
        }
    }
}

/// Render Environment science screen (terrain, hydrology, hazards).
pub fn render_environment_screen(
    frame: &mut Frame,
    state: &ScreenDataState<EnvironmentViewData>,
    y_start: u16,
) {
    let y_start = u32::from(y_start);
    print_text(frame, 2, y_start, "ENVIRONMENT, HYDROLOGY & BIOME LAYERS");
    match state {
        ScreenDataState::Empty { message } => {
            render_state_banner(frame, y_start + 2, Some(message), None, None);
        }
        ScreenDataState::Loading {
            query_id,
            operation,
        } => {
            render_state_banner(frame, y_start + 2, None, Some((*query_id, operation)), None);
        }
        ScreenDataState::Error { query_id, error } => {
            render_state_banner(frame, y_start + 2, None, None, Some((*query_id, error)));
        }
        ScreenDataState::Data(data) => {
            let dim = format!(
                "  World Grid: {}x{}  Food Energy Pool: {:.1}  Regrowth: {:.2}/tick",
                data.world_size.0, data.world_size.1, data.total_food_energy, data.regrowth_rate
            );
            print_text(frame, 2, y_start + 2, &dim);

            print_text(frame, 2, y_start + 4, "  TERRAIN COMPOSITION:");
            let rows =
                (y_start.saturating_add(5)..u32::from(frame.height())).take(SCIENCE_TABLE_ROWS);
            for (row, (t_name, pct)) in rows.zip(&data.terrain_distribution) {
                let bar_len = (*pct * 20.0).clamp(0.0, 20.0) as usize;
                let bar = "#".repeat(bar_len);
                let line = format!("    {:<12} [{:<20}] {:>5.1}%", t_name, bar, pct * 100.0);
                print_text(frame, 2, row, &line);
            }

            let hydro_y = y_start + 10;
            print_text(frame, 2, hydro_y, "  HYDROLOGY & CLIMATE TELEMETRY:");
            let metrics = format!(
                "    Mean Fertility: {}  Temperature: {}  Current Flow: {}",
                observation_text(data.mean_fertility, |value| format!("{value:.2}")),
                observation_text(data.mean_temperature, |value| format!("{value:.1}")),
                observation_text(data.current_flow_rate, |value| format!("{value:.2}"))
            );
            print_text(frame, 2, hydro_y + 1, &metrics);

            print_text(frame, 2, hydro_y + 3, "  REPORTED TIMED INTERVENTIONS:");
            if data.reported_timed_interventions.is_empty() {
                print_text(
                    frame,
                    4,
                    hydro_y + 4,
                    "  (No timed interventions reported in current snapshot)",
                );
            } else {
                let rows = hydro_y.saturating_add(4)..u32::from(frame.height());
                for (row, intervention) in rows.zip(&data.reported_timed_interventions) {
                    print_text(frame, 4, row, &format!("  • {intervention}"));
                }
            }
        }
    }
}

/// Render Diagnostics science screen (queues, lag, timings, digest neutrality).
pub fn render_diagnostics_screen(
    frame: &mut Frame,
    state: &ScreenDataState<DiagnosticsViewData>,
    y_start: u16,
) {
    let y_start = u32::from(y_start);
    print_text(frame, 2, y_start, "SYSTEM DIAGNOSTICS & PIPELINE HEALTH");
    match state {
        ScreenDataState::Empty { message } => {
            render_state_banner(frame, y_start + 2, Some(message), None, None);
        }
        ScreenDataState::Loading {
            query_id,
            operation,
        } => {
            render_state_banner(frame, y_start + 2, None, Some((*query_id, operation)), None);
        }
        ScreenDataState::Error { query_id, error } => {
            render_state_banner(frame, y_start + 2, None, None, Some((*query_id, error)));
        }
        ScreenDataState::Data(data) => {
            let perf = format!(
                "  Simulation Rate: {}  Render Refresh: {}  Build: {}",
                observation_text(data.sim_tick_hz, |value| format!("{value:.1} TPS")),
                observation_text(data.render_fps, |value| format!("{value:.1} FPS")),
                data.build_commit
            );
            print_text(frame, 2, y_start + 2, &perf);

            print_text(frame, 2, y_start + 4, "  HOST & STORAGE PIPELINE HEALTH:");
            let q_ratio = data
                .host_queue_capacity
                .filter(|capacity| *capacity > 0)
                .map(|capacity| (data.host_queue_depth as f32 / capacity as f32) * 100.0);
            let queue_str = format!(
                "    Admitted Command Queue: {:>2}/{} ({} capacity)",
                data.host_queue_depth,
                observation_text(data.host_queue_capacity, |value| value.to_string()),
                observation_text(q_ratio, |value| format!("{value:.0}%"))
            );
            print_text(frame, 2, y_start + 5, &queue_str);

            let storage_str = format!(
                "    Analytics Snapshot: Committed Tick {:<7} Uncommitted: {}",
                observation_text(data.analytics_committed_tick, |value| value.to_string()),
                observation_text(data.storage_uncommitted_records, |value| value.to_string())
            );
            print_text(frame, 2, y_start + 6, &storage_str);

            let neutrality_str = match data.digest_neutrality_verified {
                Some(true) => {
                    "    Introspection Neutrality: ✓ CONFIRMED (Compared science digests match)"
                }
                Some(false) => {
                    "    Introspection Neutrality: ✗ FAILED (Compared science digests differ)"
                }
                None => "    Introspection Neutrality: not measured",
            };
            print_text(frame, 2, y_start + 7, neutrality_str);

            let timing_y = y_start + 9;
            print_text(frame, 2, timing_y, "  SIMULATION STAGE LATENCIES (ms):");
            let rows = timing_y.saturating_add(1)..u32::from(frame.height());
            for (row, (stage, ms)) in rows.zip(&data.snapshot_timings_ms) {
                let line = format!("    {:<20} {:>6.2} ms", stage, ms);
                print_text(frame, 2, row, &line);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::frankentui_shell::{FrankenTuiSimulatorHarness, ShellMessage, ShellRoute};
    use ftui::render::grapheme_pool::GraphemePool;

    fn render(route: ShellRoute, message: ShellMessage) -> String {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();
        harness.dispatch(message);
        harness.dispatch(ShellMessage::Navigate(route));
        harness.capture_frame_str(180, 40)
    }

    fn row<'a>(frame: &'a str, label: &str) -> &'a str {
        frame
            .lines()
            .find(|line| line.contains(label))
            .expect("expected rendered observation row")
    }

    #[test]
    fn long_text_stops_at_the_frame_edge_without_wrapping_indices() {
        let mut pool = GraphemePool::new();
        let mut frame = Frame::new(8, 2, &mut pool);
        for y in 0..frame.height() {
            for x in 0..frame.width() {
                frame.buffer.set_raw(x, y, Cell::from_char('-'));
            }
        }
        let long_text = "x".repeat(usize::from(u16::MAX) + 2);
        print_text(&mut frame, 2, 0, &long_text);
        for x in 0..frame.width() {
            let expected = if x < 2 { '-' } else { 'x' };
            assert_eq!(
                frame
                    .buffer
                    .get(x, 0)
                    .expect("row zero cell")
                    .content
                    .as_char(),
                Some(expected)
            );
            assert_eq!(
                frame
                    .buffer
                    .get(x, 1)
                    .expect("row one cell")
                    .content
                    .as_char(),
                Some('-')
            );
        }
        let before = frame.buffer.clone();
        print_text(&mut frame, u16::MAX, 0, "out of bounds");
        print_text(&mut frame, 0, u32::from(u16::MAX), "out of bounds");
        print_text(&mut frame, 0, u32::MAX, "out of bounds");
        assert!(frame.buffer.content_eq(&before));
    }

    #[test]
    fn offscreen_science_sections_do_not_wrap_vertical_offsets() {
        let mut pool = GraphemePool::new();
        let mut frame = Frame::new(8, 2, &mut pool);
        frame.buffer.set_raw(0, 0, Cell::from_char('S'));
        let before = frame.buffer.clone();
        render_lineages_screen(
            &mut frame,
            &ScreenDataState::Data(LineagesViewData::default()),
            u16::MAX,
        );
        render_brain_arena_screen(
            &mut frame,
            &ScreenDataState::Data(BrainArenaViewData::default()),
            u16::MAX,
        );
        render_experiments_screen(
            &mut frame,
            &ScreenDataState::Data(ExperimentsViewData::default()),
            u16::MAX,
        );
        render_replay_screen(
            &mut frame,
            &ScreenDataState::Data(ReplayViewData::default()),
            u16::MAX,
        );
        render_environment_screen(
            &mut frame,
            &ScreenDataState::Data(EnvironmentViewData::default()),
            u16::MAX,
        );
        render_diagnostics_screen(
            &mut frame,
            &ScreenDataState::Data(DiagnosticsViewData::default()),
            u16::MAX,
        );
        assert!(frame.buffer.content_eq(&before));
    }

    #[test]
    fn text_keeps_combining_graphemes_and_clips_whole_wide_characters() {
        let mut pool = GraphemePool::new();
        let mut frame = Frame::new(7, 3, &mut pool);
        print_text(&mut frame, 0, 0, "e\u{301}中Z");
        let combining = frame
            .buffer
            .get(0, 0)
            .expect("combining grapheme cell")
            .content
            .grapheme_id()
            .expect("combining sequence stays in one grapheme");
        assert_eq!(frame.pool.get(combining), Some("e\u{301}"));
        let wide = frame
            .buffer
            .get(1, 0)
            .expect("wide grapheme cell")
            .content
            .grapheme_id()
            .expect("wide character keeps its display width");
        assert_eq!(frame.pool.get(wide), Some("中"));
        assert!(frame.buffer.get(2, 0).expect("wide tail").is_continuation());
        assert_eq!(
            frame
                .buffer
                .get(3, 0)
                .expect("following character")
                .content
                .as_char(),
            Some('Z')
        );

        frame.buffer.set_raw(6, 1, Cell::from_char('-'));
        print_text(&mut frame, 6, 1, "中Z");
        assert_eq!(
            frame
                .buffer
                .get(6, 1)
                .expect("last column")
                .content
                .as_char(),
            Some('-'),
            "a two-cell character cannot be clipped into a single cell"
        );
        print_text(&mut frame, 5, 2, "中Z");
        let wide = frame
            .buffer
            .get(5, 2)
            .expect("fitting wide character")
            .content
            .grapheme_id()
            .expect("wide character remains whole");
        assert_eq!(frame.pool.get(wide), Some("中"));
        assert!(
            frame
                .buffer
                .get(6, 2)
                .expect("last column tail")
                .is_continuation()
        );
    }

    #[test]
    fn oversized_cohort_list_cannot_wrap_into_headers_or_inspection() {
        let cohort = BrainFamilyCohort {
            family: "regular-cohort".into(),
            agent_count: 1,
            mean_fitness: None,
            mean_eval_nanos: None,
        };
        let mut cohorts = vec![cohort; usize::from(u16::MAX) + 2];
        cohorts.last_mut().expect("last cohort").family = "late-wrapped-cohort".into();
        let mut data = BrainArenaViewData {
            family_breakdown: cohorts,
            ..BrainArenaViewData::default()
        };
        let frame = render(
            ShellRoute::BrainArena,
            ShellMessage::SetBrainArenaData(ScreenDataState::Data(data.clone())),
        );
        assert!(frame.contains("BRAIN ARENA & EVALUATOR COHORTS"));
        assert!(frame.contains("SELECTED BRAIN ACTIVATION & TOP-K ATTRIBUTIONS"));
        assert!(frame.contains("regular-cohort"));
        assert!(!frame.contains("late-wrapped-cohort"));

        data.selected_family_index = data.family_breakdown.len() - 1;
        let frame = render(
            ShellRoute::BrainArena,
            ShellMessage::SetBrainArenaData(ScreenDataState::Data(data)),
        );
        assert!(
            row(&frame, "late-wrapped-cohort")
                .trim_start()
                .starts_with('>')
        );
        assert!(frame.contains("BRAIN ARENA & EVALUATOR COHORTS"));
        assert!(frame.contains("SELECTED BRAIN ACTIVATION & TOP-K ATTRIBUTIONS"));
    }

    #[test]
    fn lineage_founder_observations_do_not_invent_species_or_clade_counts() {
        let data = LineagesViewData {
            total_founders: Some(1),
            founder_rows: vec![FounderLineageRow {
                founder_uid: 42,
                descendants_without_recorded_death: 2,
                total_descendants: 3,
                contribution_share: 0.375,
            }],
            observed_tick: Some(12),
            ..LineagesViewData::default()
        };
        let frame = render(
            ShellRoute::Lineages,
            ShellMessage::SetLineagesData(ScreenDataState::Data(data.clone())),
        );
        assert!(frame.contains("Total Founders: 1"));
        assert_eq!(
            row(&frame, "Active Clades").matches("unavailable").count(),
            2
        );
        assert!(frame.contains("canonical arrivals/deaths through tick 12"));
        assert!(frame.contains("migration unobserved"));
        assert!(frame.contains("No Recorded Death"));
        assert!(!frame.contains("Living Descendants"));
        assert!(frame.contains("Founder UID"));
        assert_eq!(
            row(&frame, "42").split_whitespace().collect::<Vec<_>>(),
            [">", "42", "2", "3", "0.375"]
        );
        assert!(!frame.contains("CLADE TABLE"));
        assert!(!frame.contains("Diet"));

        let encoded = serde_json::to_value(&data).expect("serialize founder observations");
        assert_eq!(encoded["total_founders"], 1);
        assert!(encoded["active_clades"].is_null());
        assert!(encoded["extinct_clades"].is_null());
        assert_eq!(encoded["founder_rows"][0]["founder_uid"], 42);
        assert_eq!(
            encoded["founder_rows"][0]["descendants_without_recorded_death"],
            2
        );
        assert_eq!(encoded["observed_tick"], 12);
        assert_eq!(
            serde_json::from_value::<LineagesViewData>(encoded)
                .expect("deserialize founder observations"),
            data
        );

        let unbounded = LineagesViewData {
            observed_tick: None,
            ..data
        };
        let frame = render(
            ShellRoute::Lineages,
            ShellMessage::SetLineagesData(ScreenDataState::Data(unbounded)),
        );
        assert!(frame.contains("canonical arrivals/deaths through tick unavailable"));
        assert!(!frame.contains("through tick 0)"));
    }

    #[test]
    fn brain_arena_distinguishes_measured_zero_from_missing_observations() {
        let data = BrainArenaViewData {
            family_breakdown: vec![
                BrainFamilyCohort {
                    family: "measured".into(),
                    agent_count: 0,
                    mean_fitness: Some(0.0),
                    mean_eval_nanos: Some(0),
                },
                BrainFamilyCohort {
                    family: "missing".into(),
                    agent_count: 7,
                    mean_fitness: None,
                    mean_eval_nanos: None,
                },
            ],
            total_evaluations: Some(0),
            ..BrainArenaViewData::default()
        };
        let frame = render(
            ShellRoute::BrainArena,
            ShellMessage::SetBrainArenaData(ScreenDataState::Data(data.clone())),
        );
        assert!(row(&frame, "measured").contains("0.000"));
        assert!(row(&frame, "measured").contains("0 ns"));
        assert!(!row(&frame, "measured").contains("unavailable"));
        assert_eq!(row(&frame, "missing").matches("unavailable").count(), 2);
        assert!(!row(&frame, "missing").contains("0 ns"));
        assert!(row(&frame, "Total Brain Evaluations").contains("0"));

        let encoded = serde_json::to_value(&data).expect("serialize brain measurements");
        assert_eq!(encoded["family_breakdown"][0]["mean_fitness"], 0.0);
        assert_eq!(encoded["family_breakdown"][0]["mean_eval_nanos"], 0);
        assert!(encoded["family_breakdown"][1]["mean_fitness"].is_null());
        assert!(encoded["family_breakdown"][1]["mean_eval_nanos"].is_null());
        assert_eq!(
            serde_json::from_value::<BrainArenaViewData>(encoded)
                .expect("deserialize brain measurements"),
            data
        );

        let frame = render(
            ShellRoute::BrainArena,
            ShellMessage::SetBrainArenaData(ScreenDataState::Data(BrainArenaViewData::default())),
        );
        assert!(row(&frame, "Total Brain Evaluations").contains("unavailable"));
    }

    #[test]
    fn experiment_progress_fitness_and_interval_require_observations() {
        let data = ExperimentsViewData {
            active_arms: vec![
                ExperimentArmView {
                    variant_id: "measured-arm".into(),
                    brain_family: "MLP".into(),
                    seed: 7,
                    current_tick: Some(0),
                    target_ticks: 100,
                    mean_fitness: Some(0.0),
                    uncertainty_ci_95: Some((0.0, 0.0)),
                    digest: String::new(),
                },
                ExperimentArmView {
                    variant_id: "missing-arm".into(),
                    brain_family: "DWRAON".into(),
                    seed: 7,
                    current_tick: None,
                    target_ticks: 100,
                    mean_fitness: None,
                    uncertainty_ci_95: None,
                    digest: String::new(),
                },
            ],
            ..ExperimentsViewData::default()
        };
        let frame = render(
            ShellRoute::Experiments,
            ShellMessage::SetExperimentsData(ScreenDataState::Data(data.clone())),
        );
        assert!(row(&frame, "measured-arm").contains("0/100"));
        assert!(row(&frame, "measured-arm").contains("[0.00, 0.00]"));
        assert!(!row(&frame, "measured-arm").contains("unavailable"));
        assert_eq!(row(&frame, "missing-arm").matches("unavailable").count(), 3);
        assert!(!row(&frame, "missing-arm").contains("0/100"));
        assert!(!row(&frame, "missing-arm").contains("[0.00, 0.00]"));

        let encoded = serde_json::to_value(&data).expect("serialize experiment measurements");
        assert_eq!(encoded["active_arms"][0]["current_tick"], 0);
        assert!(encoded["active_arms"][1]["current_tick"].is_null());
        assert_eq!(encoded["active_arms"][0]["mean_fitness"], 0.0);
        assert!(encoded["active_arms"][1]["mean_fitness"].is_null());
        assert_eq!(encoded["active_arms"][0]["uncertainty_ci_95"][0], 0.0);
        assert!(encoded["active_arms"][1]["uncertainty_ci_95"].is_null());
        assert_eq!(
            serde_json::from_value::<ExperimentsViewData>(encoded)
                .expect("deserialize experiment measurements"),
            data
        );
    }

    #[test]
    fn checkpoint_catalog_does_not_invent_population_digest_or_replay_verification() {
        let data = ReplayViewData {
            checkpoints: vec![
                CheckpointViewEntry {
                    checkpoint_id: "measured-cp".into(),
                    tick: 0,
                    agent_count: Some(0),
                    digest: Some("science12345678".into()),
                    file_size_bytes: 1024,
                },
                CheckpointViewEntry {
                    checkpoint_id: "missing-cp".into(),
                    tick: 10,
                    agent_count: None,
                    digest: None,
                    file_size_bytes: 2048,
                },
            ],
            ..ReplayViewData::default()
        };
        let frame = render(
            ShellRoute::Replay,
            ShellMessage::SetReplayData(ScreenDataState::Data(data.clone())),
        );
        assert!(row(&frame, "measured-cp").contains("science12345"));
        assert!(!row(&frame, "measured-cp").contains("unavailable"));
        assert_eq!(row(&frame, "missing-cp").matches("unavailable").count(), 2);
        assert!(frame.contains("No replay comparison recorded; divergence status unavailable"));
        assert!(!frame.contains("Determinism verified"));
        assert!(frame.contains("Observed live tick: unavailable"));
        assert!(!frame.contains("Replay Scrubber"));

        let encoded = serde_json::to_value(&data).expect("serialize checkpoint observations");
        assert_eq!(encoded["checkpoints"][0]["agent_count"], 0);
        assert!(encoded["checkpoints"][1]["agent_count"].is_null());
        assert_eq!(encoded["checkpoints"][0]["digest"], "science12345678");
        assert!(encoded["checkpoints"][1]["digest"].is_null());
        assert!(encoded["observed_live_tick"].is_null());
        assert_eq!(
            serde_json::from_value::<ReplayViewData>(encoded)
                .expect("deserialize checkpoint observations"),
            data
        );

        let observed_zero = ReplayViewData {
            observed_live_tick: Some(0),
            ..data
        };
        let frame = render(
            ShellRoute::Replay,
            ShellMessage::SetReplayData(ScreenDataState::Data(observed_zero.clone())),
        );
        assert!(frame.contains("Observed live tick: 0"));
        assert!(!row(&frame, "Observed live tick").contains("unavailable"));
        assert!(frame.contains("No replay comparison recorded; divergence status unavailable"));
        let encoded =
            serde_json::to_value(&observed_zero).expect("serialize live tick observation");
        assert_eq!(encoded["observed_live_tick"], 0);
        assert!(encoded.get("current_replay_tick").is_none());
        assert_eq!(
            serde_json::from_value::<ReplayViewData>(encoded)
                .expect("deserialize live tick observation"),
            observed_zero
        );

        let divergence = ReplayViewData {
            first_divergence: Some(ReplayDivergenceInfo {
                divergence_tick: 12,
                expected_digest: "expected-science".into(),
                observed_digest: "observed-science".into(),
                falsified_subsystem: "genome".into(),
            }),
            ..ReplayViewData::default()
        };
        let frame = render(
            ShellRoute::Replay,
            ShellMessage::SetReplayData(ScreenDataState::Data(divergence)),
        );
        assert!(frame.contains("DIVERGENCE DETECTED at Tick 12"));
        assert!(frame.contains("expected-science"));
        assert!(frame.contains("observed-science"));
        assert!(!frame.contains("No replay comparison recorded"));
    }

    #[test]
    fn environment_missing_telemetry_does_not_establish_climate_units_or_wrap_policy() {
        let missing = EnvironmentViewData::default();
        let frame = render(
            ShellRoute::Environment,
            ShellMessage::SetEnvironmentData(ScreenDataState::Data(missing.clone())),
        );
        assert_eq!(
            row(&frame, "Mean Fertility").matches("unavailable").count(),
            3
        );
        assert!(frame.contains("No timed interventions reported in current snapshot"));
        for unsupported in ["Torus", "benign climate", "°C", "m/s"] {
            assert!(!frame.contains(unsupported));
        }
        let measured = EnvironmentViewData {
            mean_fertility: Some(0.0),
            mean_temperature: Some(0.0),
            current_flow_rate: Some(0.0),
            ..missing.clone()
        };
        let frame = render(
            ShellRoute::Environment,
            ShellMessage::SetEnvironmentData(ScreenDataState::Data(measured.clone())),
        );
        assert!(frame.contains("Mean Fertility: 0.00"));
        assert!(frame.contains("Temperature: 0.0"));
        assert!(frame.contains("Current Flow: 0.00"));
        assert!(!row(&frame, "Mean Fertility").contains("unavailable"));

        let absent =
            serde_json::to_value(&missing).expect("serialize absent environment telemetry");
        let observed = serde_json::to_value(&measured).expect("serialize environment measurements");
        for field in ["mean_fertility", "mean_temperature", "current_flow_rate"] {
            assert!(absent[field].is_null());
            assert_eq!(observed[field], 0.0);
        }
        assert_eq!(
            serde_json::from_value::<EnvironmentViewData>(observed)
                .expect("deserialize environment measurements"),
            measured
        );
    }

    #[test]
    fn diagnostics_missing_measurements_remain_unavailable_and_neutrality_unmeasured() {
        let missing = DiagnosticsViewData::default();
        let frame = render(
            ShellRoute::Diagnostics,
            ShellMessage::SetDiagnosticsData(ScreenDataState::Data(missing.clone())),
        );
        assert_eq!(
            row(&frame, "Simulation Rate")
                .matches("unavailable")
                .count(),
            2
        );
        assert_eq!(
            row(&frame, "Admitted Command Queue")
                .matches("unavailable")
                .count(),
            2
        );
        assert_eq!(
            row(&frame, "Analytics Snapshot")
                .matches("unavailable")
                .count(),
            2
        );
        assert!(frame.contains("Introspection Neutrality: not measured"));
        for fabricated in ["0.0 TPS", "0.0 FPS", "CONFIRMED", "FAILED"] {
            assert!(!frame.contains(fabricated));
        }

        let measured = DiagnosticsViewData {
            host_queue_capacity: Some(64),
            analytics_committed_tick: Some(0),
            storage_uncommitted_records: Some(0),
            sim_tick_hz: Some(0.0),
            render_fps: Some(0.0),
            digest_neutrality_verified: Some(true),
            ..missing.clone()
        };
        let frame = render(
            ShellRoute::Diagnostics,
            ShellMessage::SetDiagnosticsData(ScreenDataState::Data(measured.clone())),
        );
        assert!(frame.contains("Simulation Rate: 0.0 TPS"));
        assert!(frame.contains("Render Refresh: 0.0 FPS"));
        assert!(frame.contains("Uncommitted: 0"));
        assert!(frame.contains("0% capacity"));
        assert!(frame.contains("Introspection Neutrality: ✓ CONFIRMED"));
        assert!(!frame.contains("unavailable"));

        let absent = serde_json::to_value(&missing).expect("serialize absent diagnostics");
        let observed = serde_json::to_value(&measured).expect("serialize measured diagnostics");
        for field in [
            "host_queue_capacity",
            "analytics_committed_tick",
            "storage_uncommitted_records",
            "sim_tick_hz",
            "render_fps",
            "digest_neutrality_verified",
        ] {
            assert!(absent[field].is_null());
        }
        assert_eq!(observed["analytics_committed_tick"], 0);
        assert_eq!(observed["storage_uncommitted_records"], 0);
        assert_eq!(observed["sim_tick_hz"], 0.0);
        assert_eq!(observed["render_fps"], 0.0);
        assert_eq!(observed["digest_neutrality_verified"], true);
        assert_eq!(
            serde_json::from_value::<DiagnosticsViewData>(observed)
                .expect("deserialize measured diagnostics"),
            measured
        );
    }

    #[test]
    fn diagnostics_zero_capacity_and_failed_comparison_do_not_report_success() {
        let data = DiagnosticsViewData {
            host_queue_capacity: Some(0),
            digest_neutrality_verified: Some(false),
            ..DiagnosticsViewData::default()
        };
        let frame = render(
            ShellRoute::Diagnostics,
            ShellMessage::SetDiagnosticsData(ScreenDataState::Data(data)),
        );
        assert!(row(&frame, "Admitted Command Queue").contains("0/0 (unavailable capacity)"));
        assert!(!row(&frame, "Admitted Command Queue").contains("0%"));
        assert!(frame.contains("Introspection Neutrality: ✗ FAILED"));
        assert!(!frame.contains("CONFIRMED"));
        assert!(!frame.contains("Introspection mutated simulation state"));
    }
}
