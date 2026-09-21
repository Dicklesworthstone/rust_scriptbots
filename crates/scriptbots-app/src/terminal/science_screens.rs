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

use ftui::render::cell::Cell;
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

/// View data for the Lineages science screen.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineagesViewData {
    pub total_founders: usize,
    pub active_clades: usize,
    pub extinct_clades: usize,
    pub clades: Vec<CladeEntry>,
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
    pub mean_fitness: f32,
    pub mean_eval_nanos: u64,
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
    pub total_evaluations: u64,
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
    pub current_tick: u64,
    pub target_ticks: u64,
    pub mean_fitness: f32,
    pub uncertainty_ci_95: (f32, f32),
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
    pub agent_count: usize,
    pub digest: String,
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
    pub current_replay_tick: u64,
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
    pub mean_fertility: f32,
    pub mean_temperature: f32,
    pub total_food_energy: f32,
    pub current_flow_rate: f32,
    pub active_hazards: Vec<String>,
    pub regrowth_rate: f32,
}

impl Default for EnvironmentViewData {
    fn default() -> Self {
        Self {
            world_size: (100, 100),
            terrain_distribution: Vec::new(),
            mean_fertility: 0.0,
            mean_temperature: 0.0,
            total_food_energy: 0.0,
            current_flow_rate: 0.0,
            active_hazards: Vec::new(),
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
    pub host_queue_capacity: usize,
    pub storage_watermark_tick: u64,
    pub storage_uncommitted_records: usize,
    pub snapshot_timings_ms: Vec<(String, f32)>,
    pub sim_tick_hz: f32,
    pub render_fps: f32,
    pub build_commit: String,
    pub digest_neutrality_verified: bool,
}

impl Default for DiagnosticsViewData {
    fn default() -> Self {
        Self {
            host_queue_depth: 0,
            host_queue_capacity: 64,
            storage_watermark_tick: 0,
            storage_uncommitted_records: 0,
            snapshot_timings_ms: Vec::new(),
            sim_tick_hz: 0.0,
            render_fps: 0.0,
            build_commit: "unknown".into(),
            digest_neutrality_verified: true,
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

fn print_text(frame: &mut Frame, x: u16, y: u16, text: &str) {
    if y >= frame.height() {
        return;
    }
    let max_w = frame.width();
    for (i, ch) in text.chars().enumerate() {
        let col = x + i as u16;
        if col < max_w {
            frame.buffer.set_raw(col, y, Cell::from_char(ch));
        }
    }
}

fn render_state_banner(
    frame: &mut Frame,
    y: u16,
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
                data.total_founders, data.active_clades, data.extinct_clades
            );
            print_text(frame, 2, y_start + 2, &summary);

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
                for (idx, clade) in data.clades.iter().take(5).enumerate() {
                    let row = y_start + 7 + idx as u16;
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

            let events_y = y_start + 13;
            print_text(frame, 2, events_y, "  RECENT PHYLOGENY EVENTS:");
            if data.recent_events.is_empty() {
                print_text(frame, 4, events_y + 1, "  (No speciation events recorded)");
            } else {
                for (idx, ev) in data.recent_events.iter().take(3).enumerate() {
                    let line = format!(
                        "  Tick {:<6} [{:<10}] {}",
                        ev.tick, ev.event_kind, ev.description
                    );
                    print_text(frame, 4, events_y + 1 + idx as u16, &line);
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
                data.total_evaluations,
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
                for (idx, cohort) in data.family_breakdown.iter().enumerate() {
                    let row = y_start + 7 + idx as u16;
                    let cursor = if idx == data.selected_family_index {
                        ">"
                    } else {
                        " "
                    };
                    let line = format!(
                        " {} {:<14} {:<7} {:.3}         {} ns",
                        cursor,
                        cohort.family,
                        cohort.agent_count,
                        cohort.mean_fitness,
                        cohort.mean_eval_nanos
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
                for (idx, arm) in data.active_arms.iter().enumerate() {
                    let row = y_start + 7 + idx as u16;
                    let cursor = if idx == data.selected_arm_index {
                        ">"
                    } else {
                        " "
                    };
                    let line = format!(
                        " {} {:<14} {:<10} {:<7} {:>5}/{:<5}   {:.2}     [{:.2}, {:.2}]",
                        cursor,
                        arm.variant_id,
                        arm.brain_family,
                        arm.seed,
                        arm.current_tick,
                        arm.target_ticks,
                        arm.mean_fitness,
                        arm.uncertainty_ci_95.0,
                        arm.uncertainty_ci_95.1
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
            let scrubber = format!(
                "  Replay Scrubber: Tick {:<8} Checkpoints Available: {}",
                data.current_replay_tick,
                data.checkpoints.len()
            );
            print_text(frame, 2, y_start + 2, &scrubber);

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
                for (idx, cp) in data.checkpoints.iter().enumerate() {
                    let row = y_start + 7 + idx as u16;
                    let cursor = if idx == data.selected_checkpoint_index {
                        ">"
                    } else {
                        " "
                    };
                    let kb = cp.file_size_bytes / 1024;
                    let line = format!(
                        " {} {:<19} {:<7} {:<7} {:<8}  {:.12}",
                        cursor, cp.checkpoint_id, cp.tick, cp.agent_count, kb, cp.digest
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
                    "  ✓ Determinism verified: zero state divergences detected across replay stream",
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
                "  World Grid: {}x{} Torus  Food Energy Pool: {:.1}  Regrowth: {:.2}/tick",
                data.world_size.0, data.world_size.1, data.total_food_energy, data.regrowth_rate
            );
            print_text(frame, 2, y_start + 2, &dim);

            print_text(frame, 2, y_start + 4, "  TERRAIN COMPOSITION:");
            for (idx, (t_name, pct)) in data.terrain_distribution.iter().enumerate() {
                let bar_len = (*pct * 20.0).clamp(0.0, 20.0) as usize;
                let bar = "#".repeat(bar_len);
                let line = format!("    {:<12} [{:<20}] {:>5.1}%", t_name, bar, pct * 100.0);
                print_text(frame, 2, y_start + 5 + idx as u16, &line);
            }

            let hydro_y = y_start + 10;
            print_text(frame, 2, hydro_y, "  HYDROLOGY & CLIMATE TELEMETRY:");
            let metrics = format!(
                "    Mean Fertility: {:.2}  Temperature: {:.1}°C  Current Velocity: {:.2} m/s",
                data.mean_fertility, data.mean_temperature, data.current_flow_rate
            );
            print_text(frame, 2, hydro_y + 1, &metrics);

            print_text(frame, 2, hydro_y + 3, "  ACTIVE ENVIRONMENTAL HAZARDS:");
            if data.active_hazards.is_empty() {
                print_text(
                    frame,
                    4,
                    hydro_y + 4,
                    "  (No hazards active - benign climate)",
                );
            } else {
                for (idx, hazard) in data.active_hazards.iter().enumerate() {
                    print_text(frame, 4, hydro_y + 4 + idx as u16, &format!("  • {hazard}"));
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
                "  Simulation Rate: {:>5.1} TPS  Render Refresh: {:>5.1} FPS  Build: {}",
                data.sim_tick_hz, data.render_fps, data.build_commit
            );
            print_text(frame, 2, y_start + 2, &perf);

            print_text(frame, 2, y_start + 4, "  HOST & STORAGE PIPELINE HEALTH:");
            let q_ratio = if data.host_queue_capacity > 0 {
                (data.host_queue_depth as f32 / data.host_queue_capacity as f32) * 100.0
            } else {
                0.0
            };
            let queue_str = format!(
                "    Host Ingress Queue: {:>2}/{} ({:.0}% capacity)",
                data.host_queue_depth, data.host_queue_capacity, q_ratio
            );
            print_text(frame, 2, y_start + 5, &queue_str);

            let storage_str = format!(
                "    FrankenSQLite Journal: Watermark Tick {:<7} Uncommitted: {}",
                data.storage_watermark_tick, data.storage_uncommitted_records
            );
            print_text(frame, 2, y_start + 6, &storage_str);

            let neutrality_str = if data.digest_neutrality_verified {
                "    Introspection Neutrality: ✓ CONFIRMED (Zero science digest corruption)"
            } else {
                "    Introspection Neutrality: ✗ FAILED (Introspection mutated simulation state!)"
            };
            print_text(frame, 2, y_start + 7, neutrality_str);

            let timing_y = y_start + 9;
            print_text(frame, 2, timing_y, "  SIMULATION STAGE LATENCIES (ms):");
            for (idx, (stage, ms)) in data.snapshot_timings_ms.iter().enumerate() {
                let line = format!("    {:<20} {:>6.2} ms", stage, ms);
                print_text(frame, 2, timing_y + 1 + idx as u16, &line);
            }
        }
    }
}
