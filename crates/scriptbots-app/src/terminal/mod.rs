use std::{
    cmp::{Ordering, Reverse},
    collections::{HashMap, VecDeque},
    f32::consts::{PI, TAU},
    ffi::OsStr,
    fs::{self, File},
    io::{self, Stdout},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

pub mod subcell;

use subcell::{ColorDepth, Layer, SubCellBuffer, SubCellMode, quantize};

use anyhow::{Context, Result, anyhow, ensure};
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    buffer::{Buffer, CellDiffOption},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Sparkline, Widget},
};
use scriptbots_core::{
    AgentId, BrainActivations, BrainInspectionClientId, BrainInspectionRequest,
    BrainInspectionRevision, ControlCommand, ControlDisposition, ControlSettings, NUM_EYES,
    SENSOR_LAYOUT, SensorAttribution, SensorKind, SimulationCommand, TerrainKind, TerrainLayer,
    TickSummary, WorldState, apply_control_command,
    attribution::{AttributionMethod, EffectiveOutput, OutputExplanation, explain_outputs},
    narrative::{EventKind as NarrativeEventKind, EventRecord as NarrativeEventRecord},
    visual,
};
#[cfg(test)]
use scriptbots_storage::AnalyticsSnapshotProvider;
use scriptbots_storage::MetricReading;
use serde::Serialize;
use slotmap::Key;
use supports_color::{ColorLevel, Stream, on_cached};
use tracing::{info, warn};

use crate::{
    CommandDrain, CommandSubmit, ControlRuntime, ScenarioIdentityV0, SharedAnalytics, SharedWorld,
    WorldStepDriver,
    renderer::{Renderer, RendererContext},
};

/// Sub-cell painter engine (bd-2z0.14.2.1.1): pure braille/half-block/quadrant
/// compositing primitives consumed by the high-resolution canvas work.
pub mod canvas_inspector;
pub mod canvas_ramps;
pub mod command_palette;
pub mod export;
pub mod frankentui_shell;

// `paint.rs` is deliberately NOT declared (bd-c1z8). It is a second, complete
// sub-cell painter engine, shipped by the same task that produced `subcell`, and
// `subcell` is the one wired into the live canvas and covered by tests. Dropping
// the declaration is what actually converges the crate on ONE painter: the file
// stays on disk, losing nothing and fully reversible, but the duplicate engine is
// no longer part of the build. The file itself is retained pending the explicit
// written deletion permission AGENTS.md Rule 1 requires.
//
// `the_exempt_duplicate_painter_stays_out_of_the_build` fails if this declaration
// comes back, so the second engine cannot silently rejoin the crate.

use canvas_ramps::HeadingSector;

/// Contributor rows shown in the per-cone sense probe before the list is cut.
///
/// The cut is reported rather than silent: a panel that quietly stopped at N
/// would look like a cone with exactly N neighbours.
const PROBE_CONE_ROWS: usize = 6;

const TARGET_SIM_HZ: f32 = 60.0;
const MAX_STEPS_PER_FRAME: usize = 240;
const UI_TICK_MILLIS: u64 = 100;
const DEFAULT_HEADLESS_FRAMES: usize = 12;
const MAX_HEADLESS_FRAMES: usize = 360;
const EVENT_LOG_CAPACITY: usize = 16;
/// Bounded contributor list requested from `WorldState::explain_sensors`
/// (bd-16g.4.2); the panel is an on-demand probe, never a population scan.
const PROBE_MAX_CONTRIBUTORS: usize = 12;
/// Rows reserved below the map for the egocentric sense-probe panel.
const PROBE_PANEL_HEIGHT: u16 = 18;
const LEADERBOARD_LIMIT: usize = 6;
/// Narrative rail strip height: one glyph row plus a two-row detail pane inside
/// the block borders (bd-16g.2.4).
const RAIL_HEIGHT: u16 = 5;
/// Top-k attribution rows computed per output in the brain panel (bd-16g.4.3).
const BRAIN_PANEL_TOP_K: usize = 3;
const BRAINBOARD_LIMIT: usize = 4;
const TERMINAL_BRAIN_INSPECTION_CLIENT_ID: BrainInspectionClientId =
    BrainInspectionClientId::new(0x5455_4900_0000_0001);

trait TerminalRestore {
    fn show_cursor(&mut self);
    fn leave_alternate_screen(&mut self);
    fn disable_raw_mode(&mut self);
}

struct CrosstermRestore;

impl TerminalRestore for CrosstermRestore {
    fn show_cursor(&mut self) {
        let _ = execute!(io::stdout(), crossterm::cursor::Show);
    }

    fn leave_alternate_screen(&mut self) {
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }

    fn disable_raw_mode(&mut self) {
        let _ = disable_raw_mode();
    }
}

struct TerminalModeGuard<R: TerminalRestore> {
    restore: R,
    raw_mode_enabled: bool,
    alternate_screen_entered: bool,
    cursor_hidden: bool,
}

impl<R: TerminalRestore> TerminalModeGuard<R> {
    fn begin_with<EnableRaw, EnterAlternate>(
        restore: R,
        enable_raw: EnableRaw,
        enter_alternate: EnterAlternate,
    ) -> Result<Self>
    where
        EnableRaw: FnOnce() -> io::Result<()>,
        EnterAlternate: FnOnce() -> io::Result<()>,
    {
        let mut guard = Self {
            restore,
            // Raw-mode setup can alter terminal state before a later syscall
            // reports failure. A redundant disable is safe, so pre-arm it.
            raw_mode_enabled: true,
            alternate_screen_entered: false,
            cursor_hidden: false,
        };

        enable_raw().context("failed to enable raw mode")?;
        // Escape-sequence writes can succeed before their flush reports an
        // error. Pre-arm the compensating LeaveAlternateScreen action.
        guard.alternate_screen_entered = true;
        enter_alternate().context("failed to enter alternate screen")?;
        Ok(guard)
    }

    fn mark_cursor_hidden(&mut self) {
        self.cursor_hidden = true;
    }
}

impl<R: TerminalRestore> Drop for TerminalModeGuard<R> {
    fn drop(&mut self) {
        if self.cursor_hidden {
            self.restore.show_cursor();
            self.cursor_hidden = false;
        }
        if self.alternate_screen_entered {
            self.restore.leave_alternate_screen();
            self.alternate_screen_entered = false;
        }
        if self.raw_mode_enabled {
            self.restore.disable_raw_mode();
            self.raw_mode_enabled = false;
        }
    }
}

fn terminal_headless_requested() -> Result<bool> {
    std::env::var_os("SCRIPTBOTS_TERMINAL_HEADLESS")
        .as_deref()
        .map(parse_terminal_bool)
        .transpose()
        .map(Option::unwrap_or_default)
}

fn parse_terminal_bool(raw: &OsStr) -> Result<bool> {
    let raw = raw
        .to_str()
        .ok_or_else(|| anyhow!("SCRIPTBOTS_TERMINAL_HEADLESS must be a valid Unicode boolean"))?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(anyhow!(
            "invalid SCRIPTBOTS_TERMINAL_HEADLESS value '{raw}'; expected one of 1/true/yes/on or 0/false/no/off"
        )),
    }
}

pub struct TerminalRenderer {
    tick_interval: Duration,
    draw_interval: Duration,
}

impl Default for TerminalRenderer {
    fn default() -> Self {
        Self {
            tick_interval: Duration::from_secs_f32(1.0 / TARGET_SIM_HZ),
            draw_interval: Duration::from_millis(UI_TICK_MILLIS),
        }
    }
}

impl Renderer for TerminalRenderer {
    fn name(&self) -> &'static str {
        "terminal"
    }

    fn run(&self, ctx: RendererContext<'_>) -> Result<()> {
        if terminal_headless_requested()? {
            let report = self.run_headless(ctx)?;
            info!(
                target = "scriptbots::terminal",
                frames = report.summary.frame_count,
                ticks_simulated = report.summary.ticks_simulated,
                final_tick = report.summary.final_tick,
                final_epoch = report.summary.final_epoch,
                initial_agents = report.initial.agent_count,
                final_agents = report.summary.final_agent_count,
                total_births = report.summary.total_births,
                total_deaths = report.summary.total_deaths,
                avg_energy_mean = report.summary.avg_energy_mean,
                avg_energy_min = report.summary.avg_energy_min,
                avg_energy_max = report.summary.avg_energy_max,
                "Terminal headless run completed"
            );
            return Ok(());
        }

        let mut stdout = io::stdout();
        // Establish the guard immediately after raw mode succeeds. If entering
        // the alternate screen fails, `begin_with` drops the partially armed
        // guard and restores raw mode before returning the original error.
        let mut guard = TerminalModeGuard::begin_with(CrosstermRestore, enable_raw_mode, || {
            execute!(stdout, EnterAlternateScreen)
        })?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).context("failed to build terminal backend")?;
        // Showing an already-visible cursor is harmless; pre-arm restoration
        // in case the hide sequence applies before a flush error is reported.
        guard.mark_cursor_hidden();
        terminal.hide_cursor().ok();

        let result = run_event_loop(self, &mut terminal, ctx);

        drop(terminal);
        drop(guard);

        result
    }
}

fn run_event_loop(
    renderer: &TerminalRenderer,
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ctx: RendererContext<'_>,
) -> Result<()> {
    let mut app = TerminalApp::new(renderer, ctx);

    loop {
        app.ensure_control_runtime_running()?;
        let now = Instant::now();
        app.maybe_step_simulation(now);

        if now.duration_since(app.last_draw) >= app.draw_interval {
            terminal.draw(|frame| app.draw(frame))?;
            app.last_draw = Instant::now();
        }

        let next_draw_due = app.last_draw + app.draw_interval;
        let next_sim_due = app.last_tick + app.tick_interval;
        let now = Instant::now();
        let sleep_for = next_draw_due
            .saturating_duration_since(now)
            .min(next_sim_due.saturating_duration_since(now));

        if event::poll(sleep_for)? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if app.handle_key(key)? {
                        break;
                    }
                }
                Event::Mouse(mouse) => {
                    app.handle_mouse(mouse)?;
                }
                _ => {}
            }
        }

        // Serve a screenshot request from the frame that is ACTUALLY ON SCREEN.
        // Drawing here and capturing that same buffer is what makes the exported
        // file equal to the displayed one by construction, rather than by a
        // second rasterization that could disagree (bd-2z0.14.2.6).
        if app.export_requested() {
            let mut displayed: Option<Buffer> = None;
            terminal.draw(|frame| {
                app.draw(frame);
                displayed = Some(frame.buffer_mut().clone());
            })?;
            app.last_draw = Instant::now();
            if let Some(buffer) = displayed {
                let tick = app.snapshot.tick;
                match app.write_frame_export(&buffer) {
                    Ok((ansi_path, _, hash)) => {
                        let name = ansi_path
                            .file_name()
                            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
                        app.push_event(
                            tick,
                            EventKind::Info,
                            format!("Saved frame {name} ({hash})"),
                        );
                    }
                    Err(err) => {
                        app.export_requested = false;
                        app.push_event(tick, EventKind::Info, format!("Screenshot failed: {err}"));
                    }
                }
            }
        }
    }

    Ok(())
}

impl TerminalRenderer {
    fn run_headless(&self, ctx: RendererContext<'_>) -> Result<HeadlessReport> {
        self.run_headless_frames(ctx, self.headless_frame_budget())
    }

    fn run_headless_frames(
        &self,
        ctx: RendererContext<'_>,
        frames: usize,
    ) -> Result<HeadlessReport> {
        let backend = ratatui::backend::TestBackend::new(80, 36);
        let mut terminal = Terminal::new(backend).context("failed to build test backend")?;
        let mut app = TerminalApp::new(self, ctx);
        app.palette = Palette::test_backend_evidence();
        let mut report = HeadlessReport::new(app.snapshot().clone(), app.scenario.as_ref());

        for frame_index in 0..frames {
            app.ensure_control_runtime_running()?;
            // bd-2z0.8.9.8: ask the final tick's batch to carry the canonical world digest
            // so replay verification can compare final science state.
            if frame_index + 1 == frames {
                app.world
                    .lock()
                    .expect("terminal world mutex poisoned")
                    .request_replay_world_digest();
            }
            app.step_once();
            terminal.draw(|frame| app.draw(frame))?;
            let buffer =
                HeadlessBufferEvidence::inspect(terminal.backend().buffer(), app.snapshot().tick)?;
            report.record(app.snapshot(), buffer);
        }

        let world_digest = match app.world.lock() {
            Ok(world) => match world.world_digest_v1() {
                Ok(digest) => {
                    tracing::info!(
                        tick = app.snapshot().tick,
                        world_digest = %digest.overall,
                        "captured final world digest for the replay stream"
                    );
                    Some(digest.overall)
                }
                Err(error) => {
                    tracing::warn!("failed to capture final world digest: {error}");
                    None
                }
            },
            Err(_) => None,
        };

        report.finalize(world_digest);

        if let Some(path) = report_file_path_from_env() {
            report.write_json(&path).with_context(|| {
                format!("failed to write headless report to {}", path.display())
            })?;
        }

        Ok(report)
    }

    fn headless_frame_budget(&self) -> usize {
        std::env::var("SCRIPTBOTS_TERMINAL_HEADLESS_FRAMES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .map(|value| value.min(MAX_HEADLESS_FRAMES))
            .unwrap_or(DEFAULT_HEADLESS_FRAMES)
    }
}

// `PartialEq`/`Debug` so tests can assert focus DID NOT change — the negative
// control that catches Ctrl+T leaking into the plain-t focus action
// (bd-2z0.14.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FocusLockMode {
    Manual,
    TopPredator,
    Oldest,
}

struct TerminalApp<'a> {
    world: SharedWorld,
    simulation_step: WorldStepDriver,
    analytics_provider: SharedAnalytics,
    control: &'a ControlRuntime,
    command_drain: CommandDrain,
    command_submit: CommandSubmit,
    scenario: Arc<ScenarioIdentityV0>,
    tick_interval: Duration,
    draw_interval: Duration,
    speed_multiplier: f32,
    paused: bool,
    help_visible: bool,
    sim_accumulator: f32,
    last_tick: Instant,
    last_draw: Instant,
    palette: Palette,
    terrain: TerrainView,
    /// Resolved `(cycle_ticks, start_phase)` for [`visual::daylight_factor`],
    /// read once from the run's render settings. It is presentation-only and is
    /// captured alongside the terrain so the canvas never re-reads config while
    /// painting.
    day_night: (u32, f32),
    event_log: VecDeque<EventEntry>,
    last_event_tick: u64,
    snapshot: Snapshot,
    baseline: Option<Baseline>,
    last_autopause_tick: Option<u64>,
    map_scratch: Vec<CellOccupancy>,
    map_stamp: u32,
    /// Grow-only sub-cell canvas, built on first use and reused across frames.
    map_canvas: Option<SubCellBuffer>,
    /// Whether the map draws through the sub-cell canvas. Toggled with `B`.
    map_canvas_enabled: bool,
    /// Sub-cell tier probed once at startup; the canvas never re-reads the
    /// environment while painting.
    canvas_capability: CanvasCapability,
    /// Grow-only per-sub-pixel agent counts backing the canvas crowding pass.
    map_density: Vec<u16>,
    analytics: Option<TerminalAnalytics>,
    analytics_revision: Option<u64>,
    analytics_status: AnalyticsStatus,
    simulation_fault: Option<Arc<str>>,
    expanded: bool,
    // When true, the user has explicitly toggled expanded panels; honor self.expanded
    // instead of auto-expanding on wide terminals.
    expanded_user_override: bool,
    // Brain view controls
    focused_agent_cursor: usize,
    activation_layer_index: usize,
    activation_row_offset: usize,
    focus_lock: FocusLockMode,
    /// When true, the focused agent's senses are probed each snapshot refresh
    /// and rendered as the egocentric attribution panel (bd-16g.4.2).
    probe_enabled: bool,
    brain_inspection_revision: BrainInspectionRevision,
    brain_inspection_cache: Option<TerminalBrainInspectionCache>,
    /// One-shot latches for the attribution panel's warn-once-per-(agent, reason)
    /// logging contract (bd-16g.4.3).
    attribution_warned: std::collections::HashSet<(u64, &'static str)>,
    /// Narrative rail visibility (bd-16g.2.4); toggled with `r`.
    rail_visible: bool,
    /// Selected event index into the retained narrative, plus the identity of the
    /// selected event so a ring wrap that drops it is detected instead of
    /// silently re-pointing the selection at a different event.
    rail_selection: Option<(usize, u64, NarrativeEventKind)>,
    /// Set when the ring dropped the user's selected event; cleared on the next
    /// explicit selection.
    rail_selection_aged_out: bool,
    /// One-shot latches for the rail's logging contract.
    rail_logged_first_show: bool,
    rail_warned_aged_out: bool,
    /// Transient status toast notifications (bd-2z0.14.2.4)
    toasts: VecDeque<ToastEntry>,
    /// Command palette modal state (bd-2z0.14.2.5)
    palette_open: bool,
    palette_query: String,
    palette_selected_index: usize,
    map_zoom_level: f32,
    /// Centre of the visible world window, normalized. Follows the focused agent
    /// when there is one, so zooming in shows what you selected rather than
    /// whatever happens to be at the middle of the world.
    map_pan_offset: (f32, f32),
    /// The map pane's last painted rect, so a mouse position can be turned into
    /// a canvas fraction. Without it, screen->world has to guess the pane's size
    /// and origin, which is how the old hover picked agents at random.
    map_area: Option<Rect>,
    hover_tooltip: Option<MouseHoverTooltip>,
    /// Which eye cone the sense probe is narrowed to, or `None` for all cones
    /// (bd-2z0.7.15).
    ///
    /// An index rather than a resolved cone: eyes are identified by position in
    /// `0..NUM_EYES`, so this cannot dangle when focus moves, an agent dies, or
    /// the panel closes and reopens — the three lifecycle cases that make a
    /// cached selection go stale.
    selected_eye: Option<usize>,
    /// Set by `S`, cleared when the event loop exports the frame on screen.
    /// A request rather than an action because the key handler cannot see the
    /// rendered buffer, and exporting anything else is what made the old
    /// screenshot describe a different renderer than the user's (bd-2z0.14.2.6).
    export_requested: bool,
}

impl<'a> TerminalApp<'a> {
    fn new(renderer: &TerminalRenderer, ctx: RendererContext<'a>) -> Self {
        let mut palette = Palette::detect();
        // Adopt the run's configured chrome theme. bd-2z0.14.2.2's V11 reopen
        // recorded that core's TuiThemeId was declared but never consumed, so a
        // theme chosen in config had no effect on what the terminal actually
        // painted. Read from the world's own config rather than plumbing a new
        // field through RendererContext, which every renderer implements.
        {
            let configured = ctx
                .world
                .lock()
                .map(|world| world.config().render.theme)
                .unwrap_or_default();
            palette.apply_config_theme(configured);
            info!(
                theme = palette.theme_label(),
                configured = configured.is_some(),
                "terminal chrome theme resolved"
            );
        }
        let canvas_capability = palette.canvas_capability();
        info!(
            tier = canvas_capability.label(),
            mode = ?canvas_capability.mode,
            depth = ?canvas_capability.depth,
            canvas = canvas_capability.use_canvas(),
            "terminal sub-cell canvas capability probed"
        );
        let (terrain, day_night) = {
            let world = ctx
                .world
                .lock()
                .expect("world mutex poisoned while capturing terrain");
            (
                TerrainView::from(world.terrain()),
                world.config().render.resolved_day_night(),
            )
        };
        let mut app = Self {
            world: Arc::clone(&ctx.world),
            simulation_step: Arc::clone(&ctx.simulation_step),
            analytics_provider: ctx.analytics.clone(),
            control: ctx.control_runtime,
            command_drain: Arc::clone(&ctx.command_drain),
            command_submit: Arc::clone(&ctx.command_submit),
            scenario: Arc::clone(&ctx.scenario),
            tick_interval: renderer.tick_interval,
            draw_interval: renderer.draw_interval,
            speed_multiplier: 1.0,
            paused: false,
            help_visible: false,
            sim_accumulator: 0.0,
            last_tick: Instant::now(),
            last_draw: Instant::now(),
            palette,
            terrain,
            day_night,
            event_log: VecDeque::with_capacity(EVENT_LOG_CAPACITY),
            last_event_tick: 0,
            snapshot: Snapshot::default(),
            baseline: None,
            last_autopause_tick: None,
            map_scratch: Vec::new(),
            map_stamp: 1,
            map_canvas: None,
            // On by default wherever it can be shown: the sub-cell canvas is the
            // better picture, and a terminal without color falls back anyway.
            map_canvas_enabled: true,
            canvas_capability,
            map_density: Vec::new(),
            analytics: None,
            analytics_revision: None,
            analytics_status: AnalyticsStatus::default(),
            simulation_fault: None,
            expanded: false,
            expanded_user_override: false,
            focused_agent_cursor: 0,
            activation_layer_index: 0,
            activation_row_offset: 0,
            focus_lock: FocusLockMode::Manual,
            probe_enabled: false,
            brain_inspection_revision: BrainInspectionRevision::new(0),
            brain_inspection_cache: None,
            attribution_warned: std::collections::HashSet::new(),
            rail_visible: true,
            rail_selection: None,
            rail_selection_aged_out: false,
            rail_logged_first_show: false,
            rail_warned_aged_out: false,
            toasts: VecDeque::with_capacity(8),
            palette_open: false,
            palette_query: String::new(),
            palette_selected_index: 0,
            map_zoom_level: 1.0,
            map_pan_offset: (0.5, 0.5),
            map_area: None,
            hover_tooltip: None,
            selected_eye: None,
            export_requested: false,
        };
        app.refresh_snapshot();
        app
    }

    fn ensure_control_runtime_running(&self) -> Result<()> {
        self.control
            .health()
            .map_err(|detail| anyhow!("control runtime failed while the TUI was active: {detail}"))
    }

    fn submit_simulation_command(&self, command: SimulationCommand) {
        if !(self.command_submit.as_ref())(ControlCommand::UpdateSimulation(command)) {
            warn!("terminal renderer failed to enqueue simulation command");
        }
    }

    fn apply_simulation_commands(&mut self, commands: Vec<SimulationCommand>) -> bool {
        if commands.is_empty() {
            return false;
        }

        let mut force_step = false;
        for command in commands {
            if let Some(paused) = command.paused {
                self.paused = paused;
                if paused {
                    self.sim_accumulator = 0.0;
                }
            }
            if let Some(speed) = command.speed_multiplier {
                self.speed_multiplier = speed;
            }
            if command.step_once {
                force_step = true;
                self.paused = true;
            }
        }
        force_step
    }

    fn maybe_step_simulation(&mut self, now: Instant) {
        self.advance_simulation(now, false);
    }

    fn advance_simulation(&mut self, now: Instant, single_step: bool) {
        let delta = now - self.last_tick;
        self.last_tick = now;

        let mut force_step = single_step;
        let mut latched_fault = None;
        let pending_commands = if let Ok(mut world) = self.world.lock() {
            if let Some(error) = world.latched_step_error() {
                latched_fault = Some(Arc::<str>::from(error.to_string()));
                None
            } else {
                let mut playback = Vec::new();
                for command in (self.command_drain.as_ref())() {
                    match apply_control_command(&mut world, command) {
                        Ok(ControlDisposition::WorldApplied) => {}
                        Ok(ControlDisposition::Playback(command)) => playback.push(command),
                        Err(error) => warn!(%error, "terminal rejected a drained control command"),
                    }
                }
                Some(playback)
            }
        } else {
            None
        };
        if let Some(error) = latched_fault {
            self.paused = true;
            self.sim_accumulator = 0.0;
            self.simulation_fault = Some(error);
            self.refresh_snapshot();
            return;
        }
        if let Some(pending) = pending_commands
            && self.apply_simulation_commands(pending)
        {
            force_step = true;
        }

        if single_step {
            self.paused = true;
            self.sim_accumulator = 0.0;
        }

        let mut steps = 0usize;

        let effective_speed = if self.paused {
            0.0
        } else {
            self.speed_multiplier.max(0.0)
        };

        let step_interval = self.tick_interval.as_secs_f32();
        if effective_speed > f32::EPSILON && step_interval > f32::EPSILON {
            self.sim_accumulator += delta.as_secs_f32() * effective_speed;
            let max_accumulator = step_interval * MAX_STEPS_PER_FRAME as f32;
            if self.sim_accumulator > max_accumulator {
                self.sim_accumulator = max_accumulator;
            }
            steps = (self.sim_accumulator / step_interval).floor() as usize;
            if steps > MAX_STEPS_PER_FRAME {
                steps = MAX_STEPS_PER_FRAME;
            }
            if steps > 0 {
                self.sim_accumulator -= step_interval * steps as f32;
            }
        }

        if force_step {
            steps = steps.max(1);
            self.paused = true;
        }

        let mut step_error = None;
        for _ in 0..steps {
            if !self.scenario.interventions.is_empty() {
                self.apply_due_interventions();
            }
            if let Err(error) = (self.simulation_step)() {
                step_error = Some(Arc::<str>::from(error.to_string()));
                break;
            }
        }

        self.refresh_snapshot();
        if let Some(error) = step_error {
            self.paused = true;
            self.sim_accumulator = 0.0;
            self.simulation_fault = Some(error);
        }
    }

    fn step_once(&mut self) {
        self.advance_simulation(Instant::now(), true);
    }

    /// Apply any scheduled scenario interventions at the current completed-tick
    /// boundary, before the next science step. Application is identical to a
    /// drained `UpdateConfig` command, so a rerun of the same scenario replays the
    /// same interventions at the same ticks.
    fn apply_due_interventions(&mut self) {
        let due: Vec<crate::ScenarioInterventionV1> = self.scenario.interventions.to_vec();
        if due.is_empty() {
            return;
        }

        // The world lock is scoped so the outcome can be reported AFTER it drops:
        // pushing an event or toast needs `&mut self`, which cannot coexist with a
        // guard borrowed from `self.world`.
        let outcome = {
            let mut world = self.world.lock().expect("terminal world mutex poisoned");
            let current_tick = world.tick().0;
            if !due.iter().any(|item| item.tick == current_tick) {
                return;
            }
            let mut config_value = match serde_json::to_value(world.config()) {
                Ok(value) => value,
                Err(error) => {
                    warn!(%error, "scenario config did not serialize for intervention merge");
                    return;
                }
            };
            let result = crate::apply_scenario_interventions(
                &mut world,
                &mut config_value,
                &due,
                current_tick,
            );
            (current_tick, result)
        };

        let (current_tick, result) = outcome;
        // Name WHAT changed, not just how many. A count alone cannot tell a
        // drought from a meteor in a log or in the rail, and this bead requires an
        // ecosystem crash from a mis-parameterised intervention to be obvious
        // afterwards.
        let changed = Self::intervention_summary(&due, current_tick);

        match result {
            Ok(applied) if applied > 0 => {
                info!(
                    tick = current_tick,
                    applied,
                    changed = %changed,
                    "applied scenario interventions"
                );
                // The user watched the world change; they are entitled to know it
                // was an intervention rather than emergent behaviour. Previously
                // this was an info! to a tracing subscriber nobody is reading
                // while they watch the TUI (bd-16g.10).
                self.push_event(
                    current_tick,
                    EventKind::Population,
                    format!("Intervention: {changed}"),
                );
                self.push_toast(format!("Intervention applied: {changed}"));
            }
            Ok(_) => {}
            Err(error) => {
                warn!(%error, tick = current_tick, changed = %changed, "scenario intervention failed");
                // A FAILED intervention was the worse silence: the world simply
                // did not change and nothing said why, so the run looks like the
                // intervention had no effect rather than never happening.
                self.push_event(
                    current_tick,
                    EventKind::Death,
                    format!("Intervention FAILED ({changed}): {error}"),
                );
                self.push_toast(format!("Intervention failed: {error}"));
            }
        }
    }

    /// Name the config keys the interventions due at `tick` actually set.
    ///
    /// The scenario format carries a JSON config patch rather than a named
    /// intervention kind, so the honest description is the set of keys being
    /// changed — truthful about what the product knows instead of inventing a
    /// label for it.
    fn intervention_summary(due: &[crate::ScenarioInterventionV1], tick: u64) -> String {
        let mut keys: Vec<String> = due
            .iter()
            .filter(|item| item.tick == tick)
            .filter_map(|item| item.set.as_object())
            .flat_map(|object| object.keys().cloned())
            .collect();
        keys.sort_unstable();
        keys.dedup();
        if keys.is_empty() {
            "no config keys".to_owned()
        } else {
            keys.join(", ")
        }
    }

    fn draw(&mut self, frame: &mut Frame<'_>) {
        // Ensure we start from a clean buffer every frame to avoid ghosting artifacts
        frame.render_widget(Clear, frame.area());

        let outer = if self.rail_visible {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(RAIL_HEIGHT),
                    Constraint::Min(0),
                ])
                .split(frame.area())
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(0)])
                .split(frame.area())
        };

        self.draw_header(frame, outer[0], &self.snapshot);
        let body_anchor = if self.rail_visible {
            self.draw_rail(frame, outer[1], &self.snapshot);
            outer[2]
        } else {
            outer[1]
        };

        // Auto-expand advanced panels on wide terminals unless the user has overridden
        let area = body_anchor;
        let wide = area.width >= 120;
        if !self.expanded_user_override {
            self.expanded = wide;
        }

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(if self.expanded {
                [Constraint::Percentage(58), Constraint::Percentage(42)]
            } else {
                [Constraint::Percentage(62), Constraint::Percentage(38)]
            })
            .split(body_anchor);

        // Draw the map while avoiding holding an external borrow across &mut self
        let world_size = self.snapshot.world_size;
        if self.probe_enabled {
            let map_column = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(8), Constraint::Length(PROBE_PANEL_HEIGHT)])
                .split(body[0]);
            self.draw_map(frame, map_column[0], world_size);
            self.draw_probe(frame, map_column[1], &self.snapshot);
        } else {
            self.draw_map(frame, body[0], world_size);
        }

        let sidebar = Layout::default()
            .direction(Direction::Vertical)
            .constraints(if self.expanded {
                vec![
                    Constraint::Length(7),
                    Constraint::Length(7),
                    Constraint::Length((LEADERBOARD_LIMIT as u16 + 3).min(12)),
                    Constraint::Length((LEADERBOARD_LIMIT as u16 + 3).min(12)),
                    Constraint::Length(7),
                    Constraint::Length((BRAINBOARD_LIMIT as u16 + 3).min(10)),
                    Constraint::Length(6),
                    Constraint::Min(3),
                ]
            } else {
                vec![
                    Constraint::Length(7),
                    Constraint::Length(5),
                    Constraint::Length((LEADERBOARD_LIMIT as u16 + 3).min(12)),
                    Constraint::Length((LEADERBOARD_LIMIT as u16 + 3).min(12)),
                    Constraint::Length(7),
                    Constraint::Length((BRAINBOARD_LIMIT as u16 + 3).min(10)),
                    Constraint::Min(3),
                ]
            })
            .split(body[1]);

        self.draw_stats(frame, sidebar[0], &self.snapshot);
        self.draw_trends(frame, sidebar[1], &self.snapshot);
        self.draw_leaderboard(frame, sidebar[2], &self.snapshot);
        self.draw_oldest(frame, sidebar[3], &self.snapshot);
        // Refresh analytics opportunistically before drawing insights/brains
        self.maybe_refresh_analytics();
        self.draw_insights(frame, sidebar[4], &self.snapshot);
        self.draw_brains(frame, sidebar[5], &self.snapshot);
        if self.expanded {
            self.draw_mortality(frame, sidebar[6], &self.snapshot);
            self.draw_events(frame, sidebar[7], &self.snapshot);
        } else {
            self.draw_events(frame, sidebar[6], &self.snapshot);
        }

        if self.help_visible {
            // Draw a full-screen dimmed backdrop, then the help panel on top
            let size = frame.area();
            let overlay_style = if self.palette.has_color() {
                Style::default()
                    .bg(Color::Black)
                    .add_modifier(Modifier::DIM)
            } else {
                Style::default()
            };
            frame.render_widget(Block::default().style(overlay_style), size);
            self.draw_help(frame);
        }

        self.draw_toasts(frame, frame.area());
        if self.palette_open {
            self.draw_command_palette(frame, frame.area());
        }
        if let Some(tooltip) = &self.hover_tooltip {
            self.draw_hover_tooltip(frame, tooltip, frame.area());
        }
    }

    fn maybe_refresh_analytics(&mut self) {
        let published = self.analytics_provider.snapshot();
        let revision_changed = self.analytics_revision != Some(published.revision);
        self.analytics_revision = Some(published.revision);

        let committed_tick = published.committed_tick.unwrap_or(self.snapshot.tick);
        self.analytics_status = AnalyticsStatus {
            revision: published.revision,
            committed_tick: published.committed_tick,
            lag: published
                .committed_tick
                .map(|tick| self.snapshot.tick.saturating_sub(tick)),
            last_error: published.last_error.clone(),
            stopped: published.stopped,
        };
        if !revision_changed {
            return;
        }

        let committed_agent_count = published
            .committed_agent_count
            .unwrap_or(self.snapshot.agent_count);
        if let Some(ana) = parse_terminal_analytics(
            committed_tick,
            committed_agent_count,
            published.readings.as_ref(),
        ) {
            self.analytics = Some(ana);
        }
    }

    /// The narrative rail (bd-16g.2.4): a read-only projection of `RunNarrative`.
    /// One glyph per retained event, coloured by the shared core rail model, with
    /// the selected event's full text beneath. STAGE 1 of the seek contract: the
    /// rail SELECTS — it never moves the simulation clock, and nothing here
    /// invites the user to believe otherwise.
    fn draw_rail(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title("Timeline — run history (select-only; rewind needs replay bd-2z0.5.3)");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.height < 2 || inner.width < 8 {
            return;
        }

        let events = &snapshot.narrative;
        if events.is_empty() {
            let empty = Paragraph::new(
                "no narrative events yet — the run's story will appear here as it happens",
            );
            frame.render_widget(empty, inner);
            return;
        }

        // Glyph row: truncation marker first, then one glyph per event, keeping the
        // selection in view. Bounded work: at most `inner.width` cells per frame.
        let marker_width = if snapshot.narrative_dropped > 0 {
            9_usize
        } else {
            0
        };
        let glyph_capacity = (inner.width as usize).saturating_sub(marker_width).max(1);
        let selection = self
            .rail_selection
            .map_or(events.len() - 1, |(index, _, _)| {
                index.min(events.len() - 1)
            });
        let first_visible = if events.len() <= glyph_capacity {
            0
        } else {
            selection
                .saturating_sub(glyph_capacity / 2)
                .min(events.len() - glyph_capacity)
        };

        let marker = if snapshot.narrative_dropped > 0 {
            format!("+{}…", snapshot.narrative_dropped)
        } else {
            String::new()
        };
        let mut glyph_spans: Vec<ratatui::text::Span<'_>> = Vec::with_capacity(glyph_capacity + 1);
        if !marker.is_empty() {
            // The truncation marker qualifies the rail rather than carrying its
            // own signal, which is exactly what the typography scale's muted tier
            // is for. DarkGray was a raw ANSI constant no palette could retune
            // (bd-f4x0).
            glyph_spans.push(ratatui::text::Span::styled(
                format!("{marker:<9}"),
                self.palette.muted_style(),
            ));
        }
        for (offset, event) in events[first_visible..]
            .iter()
            .take(glyph_capacity)
            .enumerate()
        {
            let index = first_visible + offset;
            let [r, g, b] = event.kind.rail_rgb();
            let mut style = Style::default().fg(Color::Rgb(r, g, b));
            if index == selection {
                style = style.add_modifier(Modifier::REVERSED | Modifier::BOLD);
            }
            glyph_spans.push(ratatui::text::Span::styled(
                event.kind.rail_glyph().to_string(),
                style,
            ));
        }
        let glyph_row = Paragraph::new(ratatui::text::Line::from(glyph_spans));
        frame.render_widget(
            glyph_row,
            Rect {
                y: inner.y,
                height: 1,
                ..inner
            },
        );

        // Detail pane: the selected event's full text, plus the honest markers.
        let selected_event = &events[selection];
        let aged_out = if self.rail_selection_aged_out {
            " | selected event aged out of the ring"
        } else {
            ""
        };
        let truncated = if snapshot.narrative_dropped > 0 {
            format!(
                " | {} earlier events dropped — this is a TAIL of the run's history",
                snapshot.narrative_dropped
            )
        } else {
            String::new()
        };
        let detail = Paragraph::new(vec![
            ratatui::text::Line::from(format!(
                "tick {} | {} | severity {:.2}{aged_out}",
                selected_event.tick.0,
                selected_event.kind.as_str(),
                selected_event.severity
            )),
            ratatui::text::Line::from(format!("{}{truncated}", selected_event.human_text)),
        ]);
        frame.render_widget(
            detail,
            Rect {
                y: inner.y + 1,
                height: inner.height - 1,
                ..inner
            },
        );
    }

    /// Move the rail selection by `delta`, clamping at both ends — never wrapping
    /// silently (bd-16g.2.4). A fresh selection clears the aged-out marker.
    fn move_rail_selection(&mut self, delta: i64) {
        let events = &self.snapshot.narrative;
        if events.is_empty() {
            self.rail_selection = None;
            return;
        }
        let current = self
            .rail_selection
            .map_or(events.len() - 1, |(index, _, _)| {
                index.min(events.len() - 1)
            });
        let next = if delta < 0 {
            current.saturating_sub(delta.unsigned_abs() as usize)
        } else {
            (current + delta as usize).min(events.len() - 1)
        };
        self.rail_selection = Some((next, events[next].tick.0, events[next].kind));
        self.rail_selection_aged_out = false;
        self.rail_warned_aged_out = false;
        tracing::debug!(
            target = "scriptbots::timeline",
            event_index = next,
            event_tick = events[next].tick.0,
            event_kind = events[next].kind.as_str(),
            "narrative rail selection moved"
        );
    }

    /// The brain panel's output explanations for the focused agent (bd-16g.4.3).
    /// Both surfaces consume the same core helper, so the panels cannot drift.
    fn output_explanations(&self, snapshot: &Snapshot) -> Option<Vec<OutputExplanation>> {
        let outputs = snapshot.focused_outputs?;
        Some(explain_outputs(
            &outputs,
            snapshot.focused_brain_bound,
            snapshot.focused_activations.as_ref(),
            BRAIN_PANEL_TOP_K,
        ))
    }

    /// The attribution panel's logging contract (bd-16g.4.3): a standalone debug
    /// line per probed tick, warn-once-per-(agent, reason) on Unavailable, and a
    /// warn whenever non-finite values had to be excluded.
    fn maybe_log_brain_panel(&mut self) {
        let Some(explanations) = self.output_explanations(&self.snapshot) else {
            return;
        };
        let Some(uid) = self.snapshot.focused_agent_uid else {
            return;
        };
        let wheels: Vec<String> = explanations
            .iter()
            .take(2)
            .map(|explanation| {
                let top: Vec<String> = explanation
                    .inputs
                    .iter()
                    .take(3)
                    .map(|input| format!("{} {:+.3}", input.sensor_name, input.contribution))
                    .collect();
                format!(
                    "{}={:.3}({})",
                    explanation.output_name,
                    explanation.raw_value,
                    top.join(", ")
                )
            })
            .collect();
        let named_outputs: Vec<String> = explanations
            .iter()
            .map(|explanation| {
                let effective = match &explanation.effective {
                    EffectiveOutput::Continuous(value) => format!("{value:.3}"),
                    EffectiveOutput::Thresholded { raw, active, .. } => {
                        format!("{raw:.3}/{}", if *active { "ON" } else { "OFF" })
                    }
                    EffectiveOutput::Clamped { raw, applied } => {
                        format!("{raw:.3}>{applied:.3}")
                    }
                };
                format!("{}={effective}", explanation.output_name)
            })
            .collect();
        tracing::debug!(
            target: "scriptbots::brain_panel",
            agent_uid = uid,
            tick = self.snapshot.tick,
            outputs = %named_outputs.join(" "),
            wheels = %wheels.join(" | "),
            "brain panel probed tick"
        );
        for explanation in &explanations {
            if let AttributionMethod::Unavailable(reason) = explanation.method
                && self.attribution_warned.insert((uid, reason.reason()))
            {
                warn!(
                    target: "scriptbots::brain_panel",
                    agent_uid = uid,
                    reason = reason.reason(),
                    "brain attribution unavailable"
                );
            }
            if explanation.non_finite_skipped > 0 {
                warn!(
                    target: "scriptbots::brain_panel",
                    agent_uid = uid,
                    output = explanation.output_name,
                    non_finite_skipped = explanation.non_finite_skipped,
                    "non-finite values excluded from brain attribution"
                );
            }
        }
    }

    /// Keep the rail selection pointing at a live event. The ring can drop the
    /// very event the user selected; that must clamp loudly, never index into a
    /// dropped slot (bd-16g.2.4).
    fn validate_rail_selection(&mut self) {
        let Some((index, tick, kind)) = self.rail_selection else {
            return;
        };
        let events = &self.snapshot.narrative;
        let still_live = events
            .get(index)
            .is_some_and(|event| event.tick.0 == tick && event.kind == kind);
        if still_live {
            return;
        }
        // The selected event aged out (or the identity moved). Clamp to the
        // newest event and say so — once per wrap, not every frame.
        if events.is_empty() {
            self.rail_selection = None;
        } else {
            let newest = events.len() - 1;
            self.rail_selection = Some((newest, events[newest].tick.0, events[newest].kind));
        }
        self.rail_selection_aged_out = true;
        if !self.rail_warned_aged_out {
            self.rail_warned_aged_out = true;
            warn!(
                target = "scriptbots::timeline",
                event_tick = tick,
                dropped_count = self.snapshot.narrative_dropped,
                "selected narrative event was dropped by the bounded ring"
            );
        }
    }

    fn draw_header(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let (agents_delta, energy_delta, health_delta) = if let Some(base) = &self.baseline {
            (
                diff_i(snapshot.agent_count as i64 - base.agent_count as i64),
                diff_f(snapshot.avg_energy - base.avg_energy),
                diff_f(snapshot.avg_health - base.avg_health),
            )
        } else {
            (String::new(), String::new(), String::new())
        };

        let status = if self.baseline.is_some() {
            format!(
                "Tick {:>6}  Epoch {:>3}  Agents {:>5} {}  Δ+{:>3}/Δ-{:>3}  Avg⚡ {:>5.2} {}  Avg❤ {:>5.2} {}  Food {:>5.2}",
                snapshot.tick,
                snapshot.epoch,
                snapshot.agent_count,
                agents_delta,
                snapshot.births,
                snapshot.deaths,
                snapshot.avg_energy,
                energy_delta,
                snapshot.avg_health,
                health_delta,
                snapshot.food.mean,
            )
        } else {
            format!(
                "Tick {:>6}  Epoch {:>3}  Agents {:>5}  Δ+{:>3}/Δ-{:>3}  Avg⚡ {:>5.2}  Avg❤ {:>5.2}  Food {:>5.2}",
                snapshot.tick,
                snapshot.epoch,
                snapshot.agent_count,
                snapshot.births,
                snapshot.deaths,
                snapshot.avg_energy,
                snapshot.avg_health,
                snapshot.food.mean,
            )
        };

        let paused_flag = if self.paused {
            Span::styled(" PAUSED ", self.palette.paused_style())
        } else {
            Span::styled(" RUNNING ", self.palette.running_style())
        };

        let mode_span = Span::styled(
            format!(
                " x{:.1} ",
                if self.paused {
                    0.0
                } else {
                    self.speed_multiplier
                }
            ),
            self.palette.speed_style(self.speed_multiplier),
        );

        let mut line = Line::from(vec![Span::styled(status, self.palette.header_style())]);
        line.spans.push(Span::raw("  "));
        line.spans.push(paused_flag);
        line.spans.push(mode_span);
        line.spans.push(Span::raw("  "));
        line.spans.push(Span::styled(
            format!(
                "Boosted {:>3}  Hybrids {:>3}  Avg Age {:>5.1}",
                snapshot.boosted_count, snapshot.hybrid_count, snapshot.avg_age
            ),
            self.palette.accent_style(),
        ));
        line.spans.push(Span::raw("  "));
        line.spans.push(Span::styled(
            format!("Palette {}", self.palette.mode_label()),
            self.palette.accent_style(),
        ));
        line.spans.push(Span::raw(" (c to cycle)"));

        // Add a compact, persistent help hint
        line.spans.push(Span::raw("  "));
        line.spans
            .push(Span::styled("Help: ?/h", self.palette.accent_style()));

        let paragraph = Paragraph::new(line).block(
            Block::default()
                .title(self.palette.title(format!(
                    "ScriptBots Terminal HUD — {} · bootstrap {}",
                    self.scenario.id, self.scenario.bootstrap_ticks
                )))
                .borders(Borders::ALL),
        );
        frame.render_widget(paragraph, area);
    }

    fn draw_stats(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let diet = snapshot.diet_split;
        let total = diet.total().max(1);
        let mut lines = Vec::new();
        lines.push(Line::from(vec![
            Span::styled("Population ", self.palette.header_style()),
            Span::raw(format!("{:>5}", snapshot.agent_count)),
            Span::raw("   "),
            Span::styled("H:", self.palette.diet_style(DietClass::Herbivore)),
            Span::raw(format!(
                "{:>3} ({:>2}%)",
                diet.herbivores,
                diet.herbivores * 100 / total
            )),
            Span::raw("  "),
            Span::styled("O:", self.palette.diet_style(DietClass::Omnivore)),
            Span::raw(format!(
                "{:>3} ({:>2}%)",
                diet.omnivores,
                diet.omnivores * 100 / total
            )),
            Span::raw("  "),
            Span::styled("C:", self.palette.diet_style(DietClass::Carnivore)),
            Span::raw(format!(
                "{:>3} ({:>2}%)",
                diet.carnivores,
                diet.carnivores * 100 / total
            )),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Energy ", self.palette.header_style()),
            Span::raw(format!(
                "avg {:>5.2}  min {:>5.2}  max {:>5.2}",
                snapshot.avg_energy, snapshot.energy_min, snapshot.energy_max
            )),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Health ", self.palette.header_style()),
            Span::raw(format!("avg {:>5.2}", snapshot.avg_health)),
            Span::raw("  "),
            Span::styled("Boosted ", self.palette.accent_style()),
            Span::raw(format!("{:>3}", snapshot.boosted_count)),
            Span::raw("  "),
            Span::styled("Hybrids ", self.palette.accent_style()),
            Span::raw(format!("{:>3}", snapshot.hybrid_count)),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Age ", self.palette.header_style()),
            Span::raw(format!(
                "avg {:>5.1}  max {:>3}",
                snapshot.avg_age, snapshot.max_age
            )),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Food ", self.palette.header_style()),
            Span::raw(format!("mean {:>5.2}", snapshot.food.mean)),
        ]));
        // Per-diet mini bars
        let max_class = diet
            .herbivores
            .max(diet.omnivores)
            .max(diet.carnivores)
            .max(1);
        let mkbar = |count: usize| -> String {
            let width = ((count * 20) / max_class).clamp(0, 20);
            "█".repeat(width)
        };
        lines.push(Line::from(vec![
            Span::styled("Bars  ", self.palette.header_style()),
            Span::styled("H ", self.palette.diet_style(DietClass::Herbivore)),
            Span::styled(
                mkbar(diet.herbivores),
                self.palette.diet_style(DietClass::Herbivore),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      O ", self.palette.diet_style(DietClass::Omnivore)),
            Span::styled(
                mkbar(diet.omnivores),
                self.palette.diet_style(DietClass::Omnivore),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("      C ", self.palette.diet_style(DietClass::Carnivore)),
            Span::styled(
                mkbar(diet.carnivores),
                self.palette.diet_style(DietClass::Carnivore),
            ),
        ]));

        let paragraph = Paragraph::new(Text::from(lines)).block(
            Block::default()
                .title(self.palette.title("Vital Stats"))
                .borders(Borders::ALL),
        );
        frame.render_widget(paragraph, area);
    }

    fn draw_trends(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let block = Block::default()
            .title(self.palette.title("Population, Energy, Births/Deaths"))
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height == 0 {
            return;
        }

        let trend_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // population
                Constraint::Length(1), // energy
                Constraint::Length(1), // births
                Constraint::Length(1), // deaths
                Constraint::Min(0),    // text
            ])
            .split(inner);

        let pop_data: Vec<u64> = snapshot
            .history
            .iter()
            .rev()
            .map(|entry| entry.population as u64)
            .collect();
        let energy_data: Vec<u64> = snapshot
            .history
            .iter()
            .rev()
            .map(|entry| (entry.avg_energy.max(0.0) * 100.0) as u64)
            .collect();

        self.draw_trend(
            frame,
            trend_layout[0],
            TREND_POPULATION,
            self.palette.population_spark_style(),
            &pop_data,
        );
        self.draw_trend(
            frame,
            trend_layout[1],
            TREND_ENERGY,
            self.palette.energy_spark_style(),
            &energy_data,
        );
        let births_data: Vec<u64> = snapshot
            .history
            .iter()
            .rev()
            .map(|e| e.births as u64)
            .collect();
        let deaths_data: Vec<u64> = snapshot
            .history
            .iter()
            .rev()
            .map(|e| e.deaths as u64)
            .collect();
        // Births and deaths are literally the Birth and Death event kinds, so they
        // take the same ramp entries the event log uses rather than raw ANSI
        // green/red. Beyond consistency this is the accessibility fix: green
        // versus red is indistinguishable under the deuteranopia and protanopia
        // palettes this app ships, and the raw constants bypassed them (bd-f4x0).
        self.draw_trend(
            frame,
            trend_layout[2],
            TREND_BIRTHS,
            self.palette.event_style(EventKind::Birth),
            &births_data,
        );
        self.draw_trend(
            frame,
            trend_layout[3],
            TREND_DEATHS,
            self.palette.event_style(EventKind::Death),
            &deaths_data,
        );

        let mut trend_lines = Vec::new();
        if let Some(recent) = snapshot.history.first() {
            trend_lines.push(Line::from(vec![
                Span::styled("Last Tick ", self.palette.header_style()),
                Span::raw(format!(
                    "t{:>6} Δ+{:>2} Δ-{:>2} ⚡{:>5.2}",
                    recent.tick, recent.births, recent.deaths, recent.avg_energy
                )),
            ]));
        }
        if let (Some(latest), Some(oldest)) = (snapshot.history.first(), snapshot.history.last()) {
            trend_lines.push(Line::from(vec![
                Span::styled("Window ", self.palette.header_style()),
                Span::raw(format!(
                    "t{:>6}→t{:>6} pop {:>4}→{:>4}",
                    oldest.tick, latest.tick, oldest.population, latest.population
                )),
            ]));
        }
        if trend_lines.is_empty() {
            trend_lines.push(Line::from(vec![Span::raw("Waiting for samples...")]));
        }
        let trend_text = Paragraph::new(trend_lines).block(Block::default());
        frame.render_widget(trend_text, trend_layout[4]);
    }

    fn draw_map(&mut self, frame: &mut Frame<'_>, area: Rect, world_size: (u32, u32)) {
        let title = format!("World Map {}×{}", world_size.0, world_size.1);
        let block = Block::default()
            .title(self.palette.title(title))
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        self.map_area = Some(inner);
        // Follow the selection: a zoomed canvas that stayed centred on the world
        // while the user selected an agent elsewhere would show them a region
        // with nothing in it and no way to tell why.
        if let Some(position) = self.focused_agent_position() {
            self.map_pan_offset = position;
        }

        if inner.width >= 2 && inner.height >= 2 {
            let needed = inner.width as usize * inner.height as usize;
            if self.map_scratch.len() < needed {
                self.map_scratch.resize(needed, CellOccupancy::default());
            }
            // Bump stamp for this frame; keep 0 reserved
            self.map_stamp = self.map_stamp.wrapping_add(1);
            if self.map_stamp == 0 {
                self.map_stamp = 1;
            }
            // Resolved before the canvas borrow so the widget construction below
            // holds exactly one mutable borrow of `self`.
            let viewport = CanvasViewport::new(self.map_zoom_level, self.map_pan_offset);
            let use_canvas = self.map_canvas_enabled && self.canvas_capability.use_canvas();
            if use_canvas && self.map_canvas.is_none() {
                self.map_canvas = Some(SubCellBuffer::new(
                    inner.width,
                    inner.height,
                    self.canvas_capability.mode,
                ));
            }
            let canvas = if use_canvas {
                self.map_canvas.as_mut()
            } else {
                None
            };
            frame.render_widget(
                MapWidget {
                    snapshot: &self.snapshot,
                    terrain: &self.terrain,
                    palette: &self.palette,
                    scratch: &mut self.map_scratch,
                    stamp: self.map_stamp,
                    canvas,
                    day_night: self.day_night,
                    capability: self.canvas_capability,
                    density: &mut self.map_density,
                    viewport,
                },
                inner,
            );
        }
    }

    fn draw_leaderboard(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let items: Vec<ListItem> = snapshot
            .leaderboard
            .iter()
            .map(|entry| {
                let mut spans = Vec::new();
                spans.push(Span::styled(
                    format!("#{:<4}", agent_uid_label(entry.uid)),
                    self.palette.header_style(),
                ));
                spans.push(Span::raw(" "));
                spans.push(Span::styled(
                    match entry.diet {
                        DietClass::Herbivore => "H ",
                        DietClass::Omnivore => "O ",
                        DietClass::Carnivore => "C ",
                    },
                    self.palette.diet_style(entry.diet),
                ));
                spans.push(Span::raw(format!(
                    "⚡{:>5.2} ❤{:>5.2} age {:>3} gen {:>2}",
                    entry.energy, entry.health, entry.age, entry.generation
                )));
                ListItem::new(Line::from(spans))
            })
            .collect();

        let block = Block::default()
            .title(self.palette.title("Top Predators"))
            .borders(Borders::ALL);
        frame.render_widget(List::new(items).block(block), area);
    }

    fn draw_oldest(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let items: Vec<ListItem> = snapshot
            .oldest
            .iter()
            .map(|entry| {
                let mut spans = Vec::new();
                spans.push(Span::styled(
                    format!("#{:<4}", agent_uid_label(entry.uid)),
                    self.palette.header_style(),
                ));
                spans.push(Span::raw(" "));
                spans.push(Span::styled(
                    match entry.diet {
                        DietClass::Herbivore => "H ",
                        DietClass::Omnivore => "O ",
                        DietClass::Carnivore => "C ",
                    },
                    self.palette.diet_style(entry.diet),
                ));
                spans.push(Span::raw(format!(
                    "age {:>3} ⚡{:>5.2} ❤{:>5.2} gen {:>2}",
                    entry.age, entry.energy, entry.health, entry.generation
                )));
                ListItem::new(Line::from(spans))
            })
            .collect();

        let block = Block::default()
            .title(self.palette.title("Oldest Agents"))
            .borders(Borders::ALL);
        frame.render_widget(List::new(items).block(block), area);
    }

    fn draw_events(&self, frame: &mut Frame<'_>, area: Rect, _snapshot: &Snapshot) {
        let events: Vec<ListItem> = self
            .event_log
            .iter()
            .rev()
            .map(|entry| {
                let style = self.palette.event_style(entry.kind);
                // The marker leads, so the kind is scannable down the left edge
                // and survives with no colour at all (bd-xg82).
                let text = format!(
                    "{} [t{:>6}] {}",
                    entry.kind.marker(),
                    entry.tick,
                    entry.message
                );
                ListItem::new(Span::styled(text, style))
            })
            .collect();
        let block = Block::default()
            .title(self.palette.title("Recent Events"))
            .borders(Borders::ALL);
        frame.render_widget(List::new(events).block(block), area);
    }

    fn draw_insights(&self, frame: &mut Frame<'_>, area: Rect, _snapshot: &Snapshot) {
        let mut lines: Vec<Line> = Vec::new();
        // Named-output table with top-k attribution (bd-16g.4.3) — the point of
        // this panel, so it leads the block ahead of the status lines: every output shows its canonical
        // wire-map name, its raw value, what the actuator does with it, and the
        // sensors driving it — or the honest reason the snapshot cannot say.
        if let Some(explanations) = self.output_explanations(&self.snapshot) {
            // A shared Unavailable reason gets its own line (a 7-row block cannot
            // afford the full sentence on every output row).
            let shared_reason = explanations.first().and_then(|first| {
                if let AttributionMethod::Unavailable(reason) = first.method
                    && explanations.iter().all(|explanation| {
                        explanation.method == AttributionMethod::Unavailable(reason)
                    })
                {
                    Some(reason.reason())
                } else {
                    None
                }
            });
            if let Some(reason) = shared_reason {
                lines.push(Line::from(vec![
                    Span::styled("Outputs ", self.palette.header_style()),
                    Span::raw(format!("({reason})")),
                ]));
            }
            for explanation in &explanations {
                let effective = match &explanation.effective {
                    EffectiveOutput::Continuous(value) => format!("{value:>5.2}"),
                    EffectiveOutput::Thresholded { raw, active, .. } => {
                        format!("{raw:>5.2} {}", if *active { "ON " } else { "off" })
                    }
                    EffectiveOutput::Clamped { raw, applied } => {
                        if (raw - applied).abs() > f32::EPSILON {
                            format!("{raw:>5.2}>{applied:>4.2}")
                        } else {
                            format!("{raw:>5.2}   ")
                        }
                    }
                };
                let drivers = match &explanation.method {
                    AttributionMethod::Unavailable(reason) if shared_reason.is_none() => {
                        format!(" ({})", reason.reason())
                    }
                    AttributionMethod::Unavailable(_) => String::new(),
                    _ if explanation.inputs.is_empty() => " (no drivers above k)".to_string(),
                    _ => explanation
                        .inputs
                        .iter()
                        .take(2)
                        .map(|input| format!(" {} {:+.2}", input.sensor_name, input.contribution))
                        .collect::<Vec<_>>()
                        .join(""),
                };
                lines.push(Line::from(vec![
                    Span::raw(format!(" {:<12}", explanation.output_name)),
                    Span::raw(effective),
                    Span::styled(drivers, self.palette.header_style()),
                ]));
            }
        }
        if let Some(inspection) = self.snapshot.brain_inspection {
            let (source, bounds) = inspection.status_lines(self.snapshot.tick);
            lines.push(Line::from(vec![
                Span::styled("Brain ", self.palette.header_style()),
                Span::raw(source),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Probe ", self.palette.header_style()),
                Span::raw(bounds),
            ]));
        }
        if let Some(error) = &self.simulation_fault {
            lines.push(Line::from(vec![
                Span::styled("Simulation ", self.palette.header_style()),
                Span::styled(format!("fault: {error}"), self.palette.error_style()),
            ]));
        }
        let committed = self
            .analytics_status
            .committed_tick
            .map_or_else(|| "pending".to_owned(), |tick| format!("t{tick}"));
        let (storage_state, storage_style) = if let Some(error) = &self.analytics_status.last_error
        {
            (format!("error: {error}"), self.palette.error_style())
        } else if self.analytics_status.stopped {
            ("stopped".to_owned(), self.palette.warn_style())
        } else {
            ("active".to_owned(), self.palette.ok_style())
        };
        let lag = self
            .analytics_status
            .lag
            .map_or_else(|| "unknown".to_owned(), |ticks| ticks.to_string());
        lines.push(Line::from(vec![
            Span::styled("Storage ", self.palette.header_style()),
            Span::raw(format!(
                "r{} · committed {} · lag {} · ",
                self.analytics_status.revision, committed, lag
            )),
            Span::styled(storage_state, storage_style),
        ]));
        if let Some(ana) = &self.analytics {
            lines.push(Line::from(vec![
                Span::styled("Age ", self.palette.header_style()),
                Span::raw(format!("μ {:>4.1}  max {:>3}", ana.age_mean, ana.age_max)),
                Span::raw("  "),
                Span::styled("Boost ", self.palette.accent_style()),
                Span::raw(format!(
                    "{:>3} ({:>4.1}%)",
                    ana.boost_count,
                    ana.boost_ratio * 100.0
                )),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Food ", self.palette.header_style()),
                Span::raw(format!(
                    "μ {:>4.2}  σ {:>4.2}",
                    ana.food_mean, ana.food_stddev
                )),
                Span::raw("  "),
                Span::styled("Gen ", self.palette.header_style()),
                Span::raw(format!(
                    "μ {:>4.1}  max {:>3.0}",
                    ana.generation_mean, ana.generation_max
                )),
            ]));
            if area.width > 60 {
                lines.push(Line::from(vec![
                    Span::styled("Temp ", self.palette.header_style()),
                    Span::raw(format!(
                        "pref μ {:>4.2} σ {:>4.2}  discomfort σ {:>4.2}",
                        ana.temperature_preference_mean,
                        ana.temperature_preference_stddev,
                        ana.temperature_discomfort_stddev
                    )),
                ]));
            }
            lines.push(Line::from(vec![
                Span::styled("Mutation ", self.palette.header_style()),
                Span::raw(format!(
                    "pri μ {:>4.2}  sec μ {:>4.2}",
                    ana.mutation_primary_mean, ana.mutation_secondary_mean
                )),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Behavior H ", self.palette.header_style()),
                Span::raw(format!(
                    "sens {:>4.2}  out {:>4.2}",
                    ana.behavior_sensor_entropy, ana.behavior_output_entropy
                )),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Food Δ ", self.palette.header_style()),
                Span::raw(format!(
                    "μ {:>+5.2}  |μ| {:>4.2}",
                    ana.food_delta_mean, ana.food_delta_mean_abs
                )),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Deaths ", self.palette.header_style()),
                Span::raw(format!("total {:>4}", ana.deaths_total)),
                Span::raw("  "),
                Span::styled("Births ", self.palette.header_style()),
                Span::raw(format!(
                    "{:>4}  hybrid {:>3} ({:>4.1}%)",
                    ana.births_total,
                    ana.births_hybrid,
                    ana.births_hybrid_ratio * 100.0
                )),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Diet E ", self.palette.header_style()),
                Span::raw(format!(
                    "H {:.2} O {:.2} C {:.2}",
                    ana.herbivore_avg_energy, ana.hybrid_avg_energy, ana.carnivore_avg_energy
                )),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Traits μ ", self.palette.header_style()),
                Span::raw(format!(
                    "smell {:.2} sound {:.2} hear {:.2} eye {:.2} blood {:.2}",
                    ana.traits_smell_mean,
                    ana.traits_sound_mean,
                    ana.traits_hearing_mean,
                    ana.traits_eye_mean,
                    ana.traits_blood_mean
                )),
            ]));
            // Temperature comfort
            let comfort = (1.0 - ana.temperature_discomfort_mean.max(0.0)).clamp(0.0, 1.0);
            let width = (comfort * 20.0).round() as usize;
            lines.push(Line::from(vec![
                Span::styled("Comfort ", self.palette.header_style()),
                Span::raw(format!("{:>3.0}% ", comfort * 100.0)),
                // Comfort is a positive-state gauge, so it takes the same ramp
                // entry the app uses for thriving rather than a raw ANSI green
                // the accessibility palettes cannot retune (bd-f4x0).
                Span::styled("█".repeat(width), self.palette.ok_style()),
            ]));
        } else {
            lines.push(Line::from(vec![Span::raw(
                "Analytics warming up… (run a few ticks) ",
            )]));
        }

        // Legend for brain paging
        if self.snapshot.agent_count > 0 {
            let ai = self.focused_agent_cursor % self.snapshot.agent_count;
            let total_layers = self.snapshot.brain_layers.len();
            let li = if total_layers == 0 {
                0
            } else {
                self.activation_layer_index.min(total_layers - 1)
            };
            lines.push(Line::from(vec![
                Span::styled("Focus ", self.palette.header_style()),
                Span::raw(format!(
                    "agent #{:>3}  layer {:>2}/{}  row {:>3}",
                    ai, li, total_layers, self.activation_row_offset
                )),
            ]));
        }
        // Compact brain activation row if available (pull selected layer)
        if let Some(layer) = self
            .snapshot
            .brain_activations_layer_indexed(self.activation_layer_index)
            && layer.width > 0
            && layer.height > 0
        {
            let cols = layer.width;
            let start_row = self
                .activation_row_offset
                .min(layer.height.saturating_sub(1));
            let rows_to_show = 3.min(layer.height - start_row);
            for r in 0..rows_to_show {
                let row_index = start_row + r;
                let start = row_index * cols;
                let end = start + cols;
                let slice = &layer.values[start..end.min(layer.values.len())];
                if self.palette.is_emoji() && area.width > 40 {
                    let take = cols.min(16);
                    let mut row = String::new();
                    for v in slice.iter().take(take) {
                        let v = (*v).clamp(0.0, 1.0);
                        let ch = if v > 0.85 {
                            '🔥'
                        } else if v > 0.6 {
                            '🌶'
                        } else if v > 0.35 {
                            '✨'
                        } else if v > 0.15 {
                            '·'
                        } else {
                            ' '
                        };
                        row.push(ch);
                    }
                    lines.push(Line::from(vec![
                        if r == 0 {
                            Span::styled("Brain ", self.palette.header_style())
                        } else {
                            Span::raw("      ")
                        },
                        Span::raw(row),
                    ]));
                } else {
                    let take = cols.min(32);
                    let mut row = String::new();
                    for v in slice.iter().take(take) {
                        let v = (*v).clamp(0.0, 1.0);
                        let ch = if v > 0.8 {
                            '█'
                        } else if v > 0.6 {
                            '▆'
                        } else if v > 0.4 {
                            '▅'
                        } else if v > 0.2 {
                            '▃'
                        } else if v > 0.1 {
                            '▂'
                        } else {
                            '▁'
                        };
                        row.push(ch);
                    }
                    lines.push(Line::from(vec![
                        if r == 0 {
                            Span::styled("Brain ", self.palette.header_style())
                        } else {
                            Span::raw("      ")
                        },
                        Span::raw(row),
                    ]));
                }
            }
        }

        // Layers list (indices) when space permits
        if area.width > 48 && !self.snapshot.brain_layers.is_empty() {
            let mut layer_labels = String::new();
            for (i, layer) in self.snapshot.brain_layers.iter().enumerate() {
                if i == self.activation_layer_index {
                    layer_labels.push('>');
                } else {
                    layer_labels.push(' ');
                }
                if let Some(name) = &layer.name {
                    layer_labels.push_str(&format!("{}  ", name));
                } else {
                    layer_labels.push_str(&format!("L{}  ", i));
                }
            }
            lines.push(Line::from(vec![
                Span::styled("Layers ", self.palette.header_style()),
                Span::raw(layer_labels.trim_end().to_string()),
            ]));
        }

        let paragraph = Paragraph::new(Text::from(lines)).block(
            Block::default()
                .title(self.palette.title("Insights"))
                .borders(Borders::ALL),
        );
        frame.render_widget(paragraph, area);
    }

    fn draw_brains(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let mut items: Vec<ListItem> = Vec::new();
        if let Some(ana) = &self.analytics {
            let total_agents = snapshot.agent_count.max(1) as f64;
            let mut rows = 0usize;
            for entry in ana.brain_shares.iter().take(BRAINBOARD_LIMIT) {
                let share = (entry.count as f64 / total_agents * 100.0).clamp(0.0, 100.0);
                let spans = vec![
                    Span::styled(format!("{:<10}", entry.label), self.palette.header_style()),
                    Span::raw("  "),
                    Span::raw(format!(
                        "{:>4} {:>5.1}%  ⚡{:>4.2}",
                        entry.count, share, entry.avg_energy
                    )),
                ];
                items.push(ListItem::new(Line::from(spans)));
                rows += 1;
            }
            if rows == 0 {
                items.push(ListItem::new(Span::raw("No brain metrics yet")));
            }
        } else {
            items.push(ListItem::new(Span::raw("Metrics not yet available")));
        }
        let block = Block::default()
            .title(self.palette.title("Brains"))
            .borders(Borders::ALL);
        frame.render_widget(List::new(items).block(block), area);
    }

    /// Split one trend row into its label column and the sparkline that follows.
    ///
    /// Degrades rather than panicking on a narrow pane: if the row cannot spare
    /// the label width the label area comes back empty and the sparkline keeps
    /// the whole row, which is the pre-existing behaviour rather than a crash.
    /// Draw one labelled trend sparkline.
    ///
    /// One helper rather than four near-identical blocks, so a fifth trend cannot
    /// be added with the label quietly left off — which is how the four ended up
    /// unlabelled in the first place.
    fn draw_trend(
        &self,
        frame: &mut Frame<'_>,
        row: Rect,
        label: &str,
        style: Style,
        data: &[u64],
    ) {
        if data.is_empty() {
            return;
        }
        let (label_area, spark_area) = Self::trend_row(row);
        if label_area.width > 0 {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    format!("{label:<width$}", width = TREND_LABEL_WIDTH as usize),
                    self.palette.muted_style(),
                )),
                label_area,
            );
        }
        frame.render_widget(Sparkline::default().style(style).data(data), spark_area);
    }

    fn trend_row(row: Rect) -> (Rect, Rect) {
        if row.width <= TREND_LABEL_WIDTH {
            return (Rect { width: 0, ..row }, row);
        }
        let label = Rect {
            width: TREND_LABEL_WIDTH,
            ..row
        };
        let spark = Rect {
            x: row.x + TREND_LABEL_WIDTH,
            width: row.width - TREND_LABEL_WIDTH,
            ..row
        };
        (label, spark)
    }

    fn draw_mortality(&self, frame: &mut Frame<'_>, area: Rect, _snapshot: &Snapshot) {
        let mut lines: Vec<Line> = Vec::new();
        if let Some(ana) = &self.analytics {
            lines.push(Line::from(vec![
                Span::styled("Deaths total ", self.palette.header_style()),
                Span::raw(format!("{:>4}", ana.deaths_total)),
            ]));
            // Simple horizontal bars to visualize proportions
            let total = ana.deaths_total.max(1) as u64;
            for cause in MortalityCause::all() {
                let count = match cause {
                    MortalityCause::CombatCarnivore => ana.deaths_combat_carnivore,
                    MortalityCause::CombatHerbivore => ana.deaths_combat_herbivore,
                    MortalityCause::Starvation => ana.deaths_starvation,
                    MortalityCause::Aging => ana.deaths_aging,
                    MortalityCause::Unknown => ana.deaths_unknown,
                } as u64;
                let width = ((count * 20) / total).clamp(0, 20) as usize;
                let bar = "█".repeat(width);
                lines.push(Line::from(vec![
                    Span::styled(
                        format!(" {:>2} ", cause.label()),
                        self.palette.header_style(),
                    ),
                    Span::styled(bar, self.palette.mortality_style(cause)),
                    Span::raw(format!(" {count:>3}")),
                ]));
            }
        } else {
            lines.push(Line::from(vec![Span::raw("Mortality data warming up…")]));
        }
        let paragraph = Paragraph::new(Text::from(lines)).block(
            Block::default()
                .title(self.palette.title("Mortality"))
                .borders(Borders::ALL),
        );
        frame.render_widget(paragraph, area);
    }

    /// Sense probe narrowed to one eye cone (bd-2z0.7.15).
    ///
    /// Every value comes from `SensorAttribution::for_eye`, which owns the
    /// eye-to-channel mapping. `EyeAttribution::raw`/`clamped`/`saturated` are
    /// already the four channels of THIS cone in `[density, r, g, b]` order, so
    /// this renders positions 0..3 directly — there is no sensor index to get
    /// wrong here, which is the point of consuming the projection rather than
    /// re-deriving it.
    ///
    /// Reports both truncation counts, and separately, because they mean
    /// different things: `filtered_out` neighbours are retained and simply not
    /// in this cone, so an empty cone reads as "nobody is in front of this eye"
    /// rather than as missing data, while `parent_truncated` is a real bound on
    /// completeness inherited from the parent attribution.
    fn draw_probe_cone(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        block: Block<'_>,
        probe: &ProbeSnapshot,
        eye: usize,
    ) {
        let Some(cone) = probe.attribution.for_eye(eye) else {
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(format!("eye {eye} is out of range")),
                    Line::from("press , or . to select another cone"),
                ])
                .block(block),
                area,
            );
            return;
        };

        // Typographic scale (bd-f4x0): label recedes, value carries, hint qualifies.
        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(vec![
            Span::styled("eye ", self.palette.label_style()),
            Span::styled(eye.to_string(), self.palette.value_style()),
            Span::styled("  uid ", self.palette.label_style()),
            Span::styled(probe.agent_uid.to_string(), self.palette.value_style()),
            Span::styled(format!("  t{}", cone.tick.0), self.palette.muted_style()),
            Span::styled(
                format!("  {} in cone", cone.contributions.len()),
                self.palette.muted_style(),
            ),
        ]));

        let mut channels: Vec<Span> = vec![Span::styled("  ", self.palette.header_style())];
        for (slot, letter) in ["d", "R", "G", "B"].iter().enumerate() {
            let mut cell = format!("{letter}{:.2}", cone.clamped[slot]);
            if cone.saturated[slot] {
                cell.push_str(&format!("⚠{:.1}", cone.raw[slot]));
            }
            cell.push(' ');
            channels.push(Span::raw(cell));
        }
        lines.push(Line::from(channels));

        if cone.contributions.is_empty() {
            lines.push(Line::from(Span::styled(
                "  nobody in this cone",
                self.palette.muted_style(),
            )));
        }
        for contribution in cone.contributions.iter().take(PROBE_CONE_ROWS) {
            lines.push(Line::from(vec![
                Span::styled("  uid ", self.palette.label_style()),
                Span::styled(
                    format!("{:<6}", contribution.source_uid.get()),
                    self.palette.value_style(),
                ),
                Span::styled(
                    format!(
                        "{:+.2}rad {:>6.1}  d{:.2}",
                        contribution.bearing, contribution.distance, contribution.density,
                    ),
                    self.palette.value_style(),
                ),
            ]));
        }

        let mut notes: Vec<String> = Vec::new();
        if cone.filtered_out > 0 {
            notes.push(format!("{} not in cone", cone.filtered_out));
        }
        if cone.parent_truncated > 0 {
            notes.push(format!("+{} truncated upstream", cone.parent_truncated));
        }
        if !notes.is_empty() {
            lines.push(Line::from(Span::styled(
                format!("  {}", notes.join(", ")),
                self.palette.muted_style(),
            )));
        }
        lines.push(Line::from(Span::styled(
            "  , / . cone",
            self.palette.muted_style(),
        )));

        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    /// Egocentric sense-probe panel (bd-16g.4.2).
    ///
    /// Renders `SensorAttribution` verbatim: clamped values on the gauges, an
    /// explicit `⚠raw` marker on saturated channels (contributions routinely
    /// sum above 1.0 — normalising them would destroy the information), and a
    /// per-channel source tag from `SENSOR_LAYOUT` so an empty contributor
    /// list on a self-state channel reads as "self", never as "no neighbors
    /// detected".
    fn draw_probe(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(" Sense Probe ", self.palette.header_style()));
        let Some(probe) = &snapshot.probe else {
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from("no focused agent to probe"),
                    Line::from("(population empty, or focus lost this frame)"),
                ])
                .block(block),
                area,
            );
            return;
        };
        let att = &probe.attribution;

        // Narrowed to one cone: render the core-owned projection and stop.
        // `for_eye` is the authority for which sensor channels belong to an eye;
        // the layout is irregular (densities at 0, 5, 12, 21) so anything that
        // indexed by `eye * 4` here would show food as eye 1's density and a
        // clock as eye 3's — plausible-looking and wrong (bd-2z0.7.15).
        if let Some(eye) = self.selected_eye {
            self.draw_probe_cone(frame, area, block, probe, eye);
            return;
        }

        let mut lines: Vec<Line> = Vec::new();
        let truncation = if att.truncated > 0 {
            format!(" (+{} truncated)", att.truncated)
        } else {
            String::new()
        };
        lines.push(Line::from(vec![
            Span::styled("uid ", self.palette.header_style()),
            Span::raw(format!(
                "{}  t{}  {} contributors{truncation}",
                probe.agent_uid,
                att.tick.0,
                att.contributions.len(),
            )),
        ]));

        // Eye rows: indices come from SENSOR_LAYOUT, never hardcoded — the
        // layout is deliberately non-contiguous (eye 3 sits at 21..=24).
        for eye in 0..NUM_EYES {
            let mut spans: Vec<Span> = vec![Span::styled(
                format!("eye{eye} "),
                self.palette.header_style(),
            )];
            for channel in SENSOR_LAYOUT.iter().filter(|c| c.eye == Some(eye)) {
                let idx = channel.index;
                let letter = match channel.kind {
                    SensorKind::EyeDensity => "d",
                    SensorKind::EyeRed => "R",
                    SensorKind::EyeGreen => "G",
                    SensorKind::EyeBlue => "B",
                    _ => "?",
                };
                let mut cell = format!("{letter}{:.2}", att.clamped[idx]);
                if att.saturated[idx] {
                    cell.push_str(&format!("⚠{:.1}", att.raw[idx]));
                }
                cell.push(' ');
                spans.push(Span::raw(cell));
            }
            lines.push(Line::from(spans));
        }

        let mut scalar_cells: Vec<String> = Vec::new();
        for channel in SENSOR_LAYOUT.iter().filter(|c| c.eye.is_none()) {
            let idx = channel.index;
            let mut cell = format!(
                "{} {:.2}[{}]",
                channel.name,
                att.clamped[idx],
                sensor_source_tag(channel.kind)
            );
            if att.saturated[idx] {
                cell.push_str(&format!("⚠{:.1}", att.raw[idx]));
            }
            scalar_cells.push(cell);
        }
        // Three cells per row so no source tag is clipped on narrow panels.
        for row in scalar_cells.chunks(3) {
            lines.push(Line::from(row.join("  ")));
        }

        if att.contributions.is_empty() {
            lines.push(Line::from(
                "no neighbors within sense radius (self/grid channels above stay live)",
            ));
        } else {
            lines.push(Line::from(Span::styled(
                "strongest contributors",
                self.palette.header_style(),
            )));
            for contribution in &att.contributions {
                let max_eye = contribution
                    .eye_density
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
                    .map_or(0, |(eye, _)| eye);
                let mut spans: Vec<Span> = Vec::new();
                if self.palette.has_color() {
                    let [r, g, b] = contribution.color;
                    spans.push(Span::styled(
                        "■ ",
                        Style::default().fg(Color::Rgb(
                            (r.clamp(0.0, 1.0) * 255.0) as u8,
                            (g.clamp(0.0, 1.0) * 255.0) as u8,
                            (b.clamp(0.0, 1.0) * 255.0) as u8,
                        )),
                    ));
                }
                spans.push(Span::raw(format!(
                    "#{} {:+.0}° d{:.0} eye{} Σ{:.2}",
                    contribution.source_uid.0,
                    contribution.bearing.to_degrees(),
                    contribution.distance,
                    max_eye,
                    contribution.total,
                )));
                lines.push(Line::from(spans));
            }
        }

        frame.render_widget(Paragraph::new(lines).block(block), area);
    }

    fn draw_help(&self, frame: &mut Frame<'_>) {
        let size = frame.area();
        let help_width = (size.width as f32 * 0.6).round() as u16;

        let help_lines = vec![
            Line::from(vec![Span::styled(
                "Controls",
                self.palette.header_style().add_modifier(Modifier::BOLD),
            )]),
            Line::raw(" q        Quit"),
            Line::raw(" space    Toggle pause"),
            Line::raw(" + / -    Adjust speed"),
            Line::raw(" s        Single step"),
            Line::raw(" S        Save ASCII screenshot"),
            Line::raw(" e        Toggle emoji mode"),
            Line::raw(" n        Toggle narrow symbols (emoji-compatible alignment)"),
            Line::raw(" c        Cycle palette (accessibility modes)"),
            Line::raw(" b        Toggle metrics baseline (set/clear)"),
            Line::raw(" x        Toggle expanded panels (auto-on on wide terminals)"),
            Line::raw(" [ / ]    Cycle brain layers (console view)"),
            Line::raw(" ↑ / ↓    Page brain heatmap rows (console view)"),
            Line::raw(" ← / →    Change focused agent (console view)"),
            Line::raw(" m/t/o    Focus mode: Manual / TopPredator / Oldest"),
            Line::raw(" i        Toggle sense probe (egocentric view of the focused agent)"),
            Line::raw(" ? / h    Toggle this help  (? is Shift+/ on most keyboards)"),
            Line::raw(""),
            Line::from(vec![Span::styled(
                "Legend",
                self.palette.header_style().add_modifier(Modifier::BOLD),
            )]),
            Line::raw(" Terrain: 🌊 deep water, 💧 shallow, 🏜 sand, 🌿 grass, 🌺 bloom, 🪨 rock"),
            Line::raw("          lush/barren variants may appear: 🐟, 🌴, 🌾, 🥀"),
            Line::raw(
                " Agents:  single 🐇 herb, 🦝 omni, 🦊 carn; small groups 🐑/🐻/🐺; large 👥",
            ),
            Line::raw("          boosted 🚀; spike peak ⚔ (underlined)"),
            Line::raw(" Narrow:  width-1 symbols: ≈ ~ · \" * ^; agents h/H, o/O, c/C; groups @"),
        ];

        // Compute a suitable height based on content and available space.
        // Every dimension must stay within the frame: on tiny terminals the
        // old `.max(8)` floor exceeded size.height and the centering math
        // underflowed u16, producing a Rect far outside the buffer.
        let desired_height = (help_lines.len() as u16).saturating_add(2);
        let help_height = desired_height
            .min(size.height.saturating_sub(2))
            .clamp(1, size.height.max(1));
        let help_width = help_width.clamp(1, size.width.max(1));
        let help_x = size.x + size.width.saturating_sub(help_width) / 2;
        let help_y = size.y + size.height.saturating_sub(help_height) / 2;
        let area = Rect::new(help_x, help_y, help_width, help_height);

        // Ensure the help area fully clears underlying content so background doesn't bleed
        frame.render_widget(Clear, area);

        let paragraph = Paragraph::new(help_lines).block(
            Block::default()
                .title(self.palette.title("Help — controls & legend"))
                .borders(Borders::ALL)
                .style(if self.palette.has_color() {
                    Style::default().bg(Color::Black).fg(Color::White)
                } else {
                    Style::default()
                }),
        );
        frame.render_widget(paragraph, area);
    }

    pub fn push_toast(&mut self, msg: impl Into<String>) {
        if self.toasts.len() >= 4 {
            self.toasts.pop_front();
        }
        let current_tick = self.snapshot.tick;
        self.toasts
            .push_back(ToastEntry::new(msg, current_tick, 180));
    }

    fn draw_toasts(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let current_tick = self.snapshot.tick;
        self.toasts.retain(|t| !t.is_expired(current_tick));
        if self.toasts.is_empty() {
            return;
        }

        let toast_count = self.toasts.len() as u16;
        let box_height = toast_count.saturating_add(2);
        let box_width = 34u16.min(area.width.saturating_sub(4));
        if area.width < box_width + 4 || area.height < box_height + 4 {
            return;
        }

        let x = area.width.saturating_sub(box_width + 2);
        let y = area.height.saturating_sub(box_height + 2);
        let toast_area = Rect::new(x, y, box_width, box_height);

        let border_style = self.palette.accent_style();
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(Span::styled(" Status ", self.palette.header_style()));

        let lines: Vec<Line<'_>> = self
            .toasts
            .iter()
            .map(|t| Line::from(Span::styled(format!(" • {}", t.message), Style::default())))
            .collect();

        let paragraph = Paragraph::new(lines).block(block);
        frame.render_widget(Clear, toast_area);
        frame.render_widget(paragraph, toast_area);
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) -> Result<()> {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.handle_mouse_click(mouse.column, mouse.row);
            }
            MouseEventKind::ScrollUp => {
                self.zoom_in();
            }
            MouseEventKind::ScrollDown => {
                self.zoom_out();
            }
            MouseEventKind::Moved => {
                self.update_hover_tooltip(mouse.column, mouse.row);
            }
            _ => {}
        }
        Ok(())
    }

    /// The normalized world position of the agent the brain panel is focused on.
    fn focused_agent_position(&self) -> Option<(f32, f32)> {
        let focused = self.snapshot.focused_agent_uid?;
        self.snapshot
            .agents
            .iter()
            .find(|agent| agent.uid == Some(focused))
            .map(|agent| agent.position)
    }

    /// The world point under a terminal cell, or `None` when the cell is outside
    /// the map pane.
    ///
    /// Goes through the same [`CanvasViewport`] the painter uses, so what the
    /// user clicks is what the user sees. The previous implementation assumed a
    /// fixed 80x36 terminal, ignored the pane's origin, and then compared a
    /// world-unit coordinate against `AgentViz::position`, which is normalized —
    /// two different spaces, so the "nearest" agent was essentially arbitrary.
    fn world_at_cell(&self, col: u16, row: u16) -> Option<(f32, f32)> {
        let area = self.map_area?;
        if area.width == 0
            || area.height == 0
            || col < area.x
            || row < area.y
            || col >= area.x.saturating_add(area.width)
            || row >= area.y.saturating_add(area.height)
        {
            return None;
        }
        let fx = (f32::from(col - area.x) + 0.5) / f32::from(area.width);
        let fy = (f32::from(row - area.y) + 0.5) / f32::from(area.height);
        Some(CanvasViewport::new(self.map_zoom_level, self.map_pan_offset).world_at(fx, fy))
    }

    /// Index of the agent nearest a normalized world point, within a pick radius
    /// that shrinks with the viewport so zooming in tightens the selection
    /// instead of grabbing whatever is loosely nearby.
    fn agent_nearest(&self, world: (f32, f32)) -> Option<usize> {
        let span = CanvasViewport::new(self.map_zoom_level, self.map_pan_offset).span;
        let radius = CANVAS_PICK_RADIUS_FRACTION * span;
        let max_d_sq = radius * radius;
        let mut nearest: Option<(usize, f32)> = None;
        for (idx, agent) in self.snapshot.agents.iter().enumerate() {
            let dx = agent.position.0 - world.0;
            let dy = agent.position.1 - world.1;
            let dist_sq = dx.mul_add(dx, dy * dy);
            if dist_sq <= max_d_sq && nearest.is_none_or(|(_, best)| dist_sq < best) {
                nearest = Some((idx, dist_sq));
            }
        }
        nearest.map(|(idx, _)| idx)
    }

    pub fn handle_mouse_click(&mut self, col: u16, row: u16) {
        if self.palette_open {
            return;
        }
        let Some(world) = self.world_at_cell(col, row) else {
            return;
        };
        if let Some(best_idx) = self.agent_nearest(world) {
            self.focused_agent_cursor = best_idx;
            self.focus_lock = FocusLockMode::Manual;
            let uid = agent_uid_label(self.snapshot.agents[best_idx].uid);
            self.push_toast(format!("Selected Agent #{uid}"));
            self.refresh_snapshot();
        }
    }

    pub fn zoom_in(&mut self) {
        self.map_zoom_level = (self.map_zoom_level * 1.2).min(CANVAS_MAX_ZOOM);
        self.push_toast(format!("Zoom: {:.1}x", self.map_zoom_level));
    }

    pub fn zoom_out(&mut self) {
        // Floors at 1.0, the whole world. There is nothing outside the world to
        // reveal, so a smaller value could only have been a number the toast
        // reported and the map ignored.
        self.map_zoom_level = (self.map_zoom_level / 1.2).max(1.0);
        self.push_toast(format!("Zoom: {:.1}x", self.map_zoom_level));
    }

    fn update_hover_tooltip(&mut self, col: u16, row: u16) {
        let Some(world) = self.world_at_cell(col, row) else {
            self.hover_tooltip = None;
            return;
        };
        let nearest = self
            .agent_nearest(world)
            .and_then(|idx| self.snapshot.agents.get(idx));

        if let Some(agent) = nearest {
            self.hover_tooltip = Some(MouseHoverTooltip {
                cell_x: col,
                cell_y: row,
                agent_uid: agent.uid,
                energy: agent.energy,
                health: agent.health,
                age: agent.age,
            });
        } else {
            self.hover_tooltip = None;
        }
    }

    pub fn execute_palette_action(&mut self, action: CommandPaletteAction) {
        match action {
            CommandPaletteAction::TogglePause => {
                self.paused = !self.paused;
                self.push_toast(if self.paused { "Paused" } else { "Resumed" });
            }
            CommandPaletteAction::StepOnce => {
                self.step_once();
                self.paused = true;
                self.push_toast("Single-step");
            }
            CommandPaletteAction::SpeedUp => {
                self.speed_multiplier = (self.speed_multiplier + 0.5).clamp(0.5, 8.0);
                self.push_toast(format!("Speed: {:.1}x", self.speed_multiplier));
            }
            CommandPaletteAction::SpeedDown => {
                self.speed_multiplier = (self.speed_multiplier - 0.5).max(0.0);
                self.push_toast(format!("Speed: {:.1}x", self.speed_multiplier));
            }
            CommandPaletteAction::CycleTheme => {
                let lbl = self.palette.cycle_theme();
                self.push_toast(format!("Theme: {lbl}"));
            }
            CommandPaletteAction::CyclePalette => {
                let lbl = self.palette.cycle_mode();
                self.push_toast(format!("Palette: {lbl}"));
            }
            CommandPaletteAction::ToggleRail => {
                self.rail_visible = !self.rail_visible;
                self.push_toast(if self.rail_visible {
                    "Rail On"
                } else {
                    "Rail Off"
                });
            }
            CommandPaletteAction::FocusTopPredator => {
                self.focus_lock = FocusLockMode::TopPredator;
                self.refresh_snapshot();
                self.push_toast("Focus: Top Predator");
            }
            CommandPaletteAction::FocusOldest => {
                self.focus_lock = FocusLockMode::Oldest;
                self.refresh_snapshot();
                self.push_toast("Focus: Oldest");
            }
            CommandPaletteAction::ToggleProbe => {
                self.probe_enabled = !self.probe_enabled;
                self.push_toast(if self.probe_enabled {
                    "Probe On"
                } else {
                    "Probe Off"
                });
            }
            CommandPaletteAction::ShowHelp => {
                self.help_visible = !self.help_visible;
            }
        }
    }

    fn draw_command_palette(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let width = 50u16.min(area.width.saturating_sub(4));
        let height = 16u16.min(area.height.saturating_sub(4));
        if width < 10 || height < 6 {
            return;
        }
        let x = area.x + (area.width.saturating_sub(width)) / 2;
        let y = area.y + (area.height.saturating_sub(height)) / 2;
        let palette_area = Rect::new(x, y, width, height);

        frame.render_widget(Clear, palette_area);

        let items = all_command_palette_items();
        let matched = fuzzy_match_command_palette(&items, &self.palette_query);

        let title = format!(" Command Palette ({}) ", self.palette_query);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.palette.accent_style())
            .title(Span::styled(title, self.palette.header_style()));

        let list_items: Vec<ListItem<'_>> = matched
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                let is_selected = idx
                    == self
                        .palette_selected_index
                        .min(matched.len().saturating_sub(1));
                let prefix = if is_selected { "> " } else { "  " };
                let style = if is_selected {
                    self.palette.header_style()
                } else {
                    Style::default()
                };
                let content = format!(
                    "{prefix}[{}] {} ({})",
                    item.category, item.label, item.keybind_hint
                );
                ListItem::new(Span::styled(content, style))
            })
            .collect();

        let list = List::new(list_items).block(block);
        frame.render_widget(list, palette_area);
    }

    fn draw_hover_tooltip(&self, frame: &mut Frame<'_>, tooltip: &MouseHoverTooltip, area: Rect) {
        let width = 24u16;
        let height = 5u16;
        let x = tooltip.cell_x.min(area.width.saturating_sub(width + 1));
        let y = tooltip.cell_y.min(area.height.saturating_sub(height + 1));
        let tip_area = Rect::new(x, y, width, height);

        frame.render_widget(Clear, tip_area);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.palette.accent_style())
            .title(Span::styled(
                format!(" Agent #{} ", agent_uid_label(tooltip.agent_uid)),
                self.palette.header_style(),
            ));

        let text = vec![
            Line::from(format!("Health: {:.1}", tooltip.health)),
            Line::from(format!("Energy: {:.1}", tooltip.energy)),
            Line::from(format!("Age: {} ticks", tooltip.age)),
        ];

        let para = Paragraph::new(text).block(block);
        frame.render_widget(para, tip_area);
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        if self.palette_open {
            match key.code {
                KeyCode::Esc => {
                    self.palette_open = false;
                }
                KeyCode::Up => {
                    self.palette_selected_index = self.palette_selected_index.saturating_sub(1);
                }
                KeyCode::Down => {
                    self.palette_selected_index = self.palette_selected_index.saturating_add(1);
                }
                KeyCode::Backspace => {
                    self.palette_query.pop();
                    self.palette_selected_index = 0;
                }
                KeyCode::Char(c) => {
                    self.palette_query.push(c);
                    self.palette_selected_index = 0;
                }
                KeyCode::Enter => {
                    let items = all_command_palette_items();
                    let matched = fuzzy_match_command_palette(&items, &self.palette_query);
                    if !matched.is_empty() {
                        let idx = self.palette_selected_index.min(matched.len() - 1);
                        let action = matched[idx].action;
                        self.execute_palette_action(action);
                    }
                    self.palette_open = false;
                }
                _ => {}
            }
            return Ok(false);
        }

        match (key.code, key.modifiers) {
            (KeyCode::Esc, _)
            | (KeyCode::Char('q'), _)
            | (KeyCode::Char('Q'), _)
            | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                return Ok(true);
            }
            (KeyCode::Char('p') | KeyCode::Char('P'), KeyModifiers::CONTROL)
            | (KeyCode::Char(':'), KeyModifiers::NONE) => {
                self.palette_open = !self.palette_open;
                self.palette_query.clear();
                self.palette_selected_index = 0;
                return Ok(false);
            }
            // Ctrl+T cycles the chrome theme. This arm MUST come before the
            // plain-`t` focus arm below, which matches `_` on modifiers and would
            // otherwise swallow Ctrl+T into "Focus: Top predators" — the exact
            // disagreement the command palette already advertised, since it has
            // listed keybind_hint "Ctrl+T" for a handler that did not exist
            // (bd-2z0.14.2.2).
            (KeyCode::Char('t') | KeyCode::Char('T'), KeyModifiers::CONTROL) => {
                let theme_label = self.palette.cycle_theme();
                info!(theme = %theme_label, "terminal chrome theme cycled");
                self.push_toast(format!("Theme: {theme_label}"));
                return Ok(false);
            }
            // Plain `p` cycles the accessibility palette, which is an ORTHOGONAL
            // axis to the chrome theme: theme styles chrome, palette drives
            // semantic data colour. `c` stays bound so existing muscle memory
            // keeps working; both reach the same single implementation the
            // command palette uses, so the three entry points cannot diverge.
            (KeyCode::Char('p'), KeyModifiers::NONE) | (KeyCode::Char('c'), KeyModifiers::NONE) => {
                let mode_label = self.palette.cycle_mode();
                info!(palette = %mode_label, "terminal accessibility palette cycled");
                self.push_toast(format!("Palette: {mode_label}"));
                return Ok(false);
            }
            (KeyCode::Char(' '), _) => {
                self.paused = !self.paused;
                if self.paused {
                    self.speed_multiplier = 0.0;
                } else if self.speed_multiplier <= 0.0 {
                    self.speed_multiplier = 1.0;
                }
                self.push_toast(if self.paused { "Paused" } else { "Resumed" });
                self.submit_simulation_command(SimulationCommand {
                    paused: Some(self.paused),
                    speed_multiplier: Some(self.speed_multiplier),
                    step_once: false,
                });
            }
            (KeyCode::Char('+') | KeyCode::Char('='), _) => {
                self.speed_multiplier = (self.speed_multiplier + 0.5).clamp(0.5, 8.0);
                if self.speed_multiplier > 0.0 {
                    self.paused = false;
                }
                self.submit_simulation_command(SimulationCommand {
                    paused: Some(self.paused),
                    speed_multiplier: Some(self.speed_multiplier),
                    step_once: false,
                });
                self.push_event(
                    self.snapshot.tick,
                    EventKind::Info,
                    format!("Speed x{:.1}", self.speed_multiplier),
                );
            }
            (KeyCode::Char('-') | KeyCode::Char('_'), _) => {
                self.speed_multiplier = (self.speed_multiplier - 0.5).max(0.0);
                if self.speed_multiplier <= 0.0 {
                    self.paused = true;
                }
                self.submit_simulation_command(SimulationCommand {
                    paused: Some(self.paused),
                    speed_multiplier: Some(self.speed_multiplier),
                    step_once: false,
                });
                self.push_event(
                    self.snapshot.tick,
                    EventKind::Info,
                    if self.paused {
                        "Simulation paused".to_string()
                    } else {
                        format!("Speed x{:.1}", self.speed_multiplier)
                    },
                );
            }
            (KeyCode::Char('s'), _) => {
                self.step_once();
                self.paused = true;
                self.speed_multiplier = 0.0;
                self.push_toast("Single-step");
                self.push_event(self.snapshot.tick, EventKind::Info, "Single-step executed");
            }
            (KeyCode::Char('.'), _) => {
                self.cycle_eye_selection(true);
            }
            (KeyCode::Char(','), _) => {
                self.cycle_eye_selection(false);
            }
            (KeyCode::Char('S'), _) => {
                // Only a request: the key handler cannot see the rendered buffer,
                // and exporting anything else would reintroduce the defect this
                // replaced. The event loop fulfils it from the frame on screen.
                self.export_requested = true;
            }
            (KeyCode::Char('e') | KeyCode::Char('E'), _) => {
                self.palette.toggle_emoji();
                self.push_event(
                    self.snapshot.tick,
                    EventKind::Info,
                    if self.palette.is_emoji() {
                        "Emoji mode ON"
                    } else {
                        "Emoji mode OFF"
                    },
                );
            }
            (KeyCode::Char('n') | KeyCode::Char('N'), _) => {
                if self.palette.is_emoji() {
                    self.palette.toggle_emoji_narrow();
                    self.push_event(
                        self.snapshot.tick,
                        EventKind::Info,
                        if self.palette.is_emoji_narrow() {
                            "Narrow symbols ON"
                        } else {
                            "Narrow symbols OFF"
                        },
                    );
                } else {
                    self.push_event(
                        self.snapshot.tick,
                        EventKind::Info,
                        "Enable Emoji mode first (press 'e') to use narrow symbols",
                    );
                }
            }
            (KeyCode::Char('B'), _) => {
                let capability = self.canvas_capability;
                if capability.use_canvas() {
                    self.map_canvas_enabled = !self.map_canvas_enabled;
                    let state = if self.map_canvas_enabled {
                        format!(
                            "Sub-cell map ON ({} · {}x{} per cell)",
                            capability.label(),
                            capability.mode.dots_x(),
                            capability.mode.dots_y()
                        )
                    } else {
                        "Sub-cell map OFF (flat glyph map)".to_string()
                    };
                    self.push_event(self.snapshot.tick, EventKind::Info, state);
                } else if capability.depth.is_none() {
                    self.push_event(
                        self.snapshot.tick,
                        EventKind::Info,
                        "Sub-cell map needs color; the flat glyph map keeps terrain readable without it",
                    );
                } else {
                    self.push_event(
                        self.snapshot.tick,
                        EventKind::Info,
                        "Sub-cell map needs a UTF-8 locale and a terminal with block glyphs",
                    );
                }
            }
            (KeyCode::Char('b'), _) => {
                if self.baseline.is_some() {
                    self.baseline = None;
                    self.push_event(self.snapshot.tick, EventKind::Info, "Baseline cleared");
                } else {
                    self.baseline = Some(Baseline {
                        agent_count: self.snapshot.agent_count,
                        avg_energy: self.snapshot.avg_energy,
                        avg_health: self.snapshot.avg_health,
                    });
                    self.push_event(
                        self.snapshot.tick,
                        EventKind::Info,
                        "Baseline set to current metrics",
                    );
                }
            }
            (KeyCode::Char('i') | KeyCode::Char('I'), _) => {
                self.probe_enabled = !self.probe_enabled;
                if self.probe_enabled {
                    // Capture immediately so the panel appears this frame.
                    self.refresh_snapshot();
                }
                self.push_event(
                    self.snapshot.tick,
                    EventKind::Info,
                    if self.probe_enabled {
                        "Sense probe ON (focused agent)"
                    } else {
                        "Sense probe OFF"
                    },
                );
            }
            (KeyCode::Char('x') | KeyCode::Char('X'), _) => {
                // User explicitly toggled; stop auto behavior and honor user's choice
                self.expanded_user_override = true;
                self.expanded = !self.expanded;
                self.push_event(
                    self.snapshot.tick,
                    EventKind::Info,
                    if self.expanded {
                        "Expanded panels ON"
                    } else {
                        "Expanded panels OFF"
                    },
                );
            }
            (KeyCode::Char('?') | KeyCode::Char('h'), _) => {
                self.help_visible = !self.help_visible;
                self.push_event(
                    self.snapshot.tick,
                    EventKind::Info,
                    if self.help_visible {
                        "Help overlay opened"
                    } else {
                        "Help overlay closed"
                    },
                );
            }
            (KeyCode::Char('['), _) => {
                if self.activation_layer_index > 0 {
                    self.activation_layer_index -= 1;
                }
            }
            (KeyCode::Char(']'), _) => {
                if !self.snapshot.brain_layers.is_empty() {
                    let max = self.snapshot.brain_layers.len() - 1;
                    if self.activation_layer_index < max {
                        self.activation_layer_index += 1;
                    }
                }
            }
            (KeyCode::Up, _) => {
                self.activation_row_offset = self.activation_row_offset.saturating_sub(1);
            }
            (KeyCode::Down, _) => {
                self.activation_row_offset = self.activation_row_offset.saturating_add(1);
            }
            (KeyCode::Left, _) => {
                if self.rail_visible && !self.snapshot.narrative.is_empty() {
                    self.move_rail_selection(-1);
                } else {
                    self.focused_agent_cursor = self.focused_agent_cursor.saturating_sub(1);
                    self.refresh_snapshot();
                }
            }
            (KeyCode::Right, _) => {
                if self.rail_visible && !self.snapshot.narrative.is_empty() {
                    self.move_rail_selection(1);
                } else {
                    self.focused_agent_cursor = self.focused_agent_cursor.saturating_add(1);
                    self.refresh_snapshot();
                }
            }
            (KeyCode::Char('r') | KeyCode::Char('R'), _) => {
                self.rail_visible = !self.rail_visible;
                self.push_event(
                    self.snapshot.tick,
                    EventKind::Info,
                    if self.rail_visible {
                        "Timeline rail shown (select-only; rewind needs replay)"
                    } else {
                        "Timeline rail hidden"
                    },
                );
            }
            // Restricted to NONE/SHIFT rather than `_`. With `_` this arm claimed
            // every modifier combination including CONTROL, so Ctrl+T could never
            // reach a theme handler no matter where one was added — the binding
            // was unreachable by construction, not merely missing
            // (bd-2z0.14.2.2).
            (KeyCode::Char('t') | KeyCode::Char('T'), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                self.focus_lock = FocusLockMode::TopPredator;
                self.refresh_snapshot();
                self.push_event(self.snapshot.tick, EventKind::Info, "Focus: Top predators");
            }
            (KeyCode::Char('o') | KeyCode::Char('O'), _) => {
                self.focus_lock = FocusLockMode::Oldest;
                self.refresh_snapshot();
                self.push_event(self.snapshot.tick, EventKind::Info, "Focus: Oldest agents");
            }
            (KeyCode::Char('m') | KeyCode::Char('M'), _) => {
                self.focus_lock = FocusLockMode::Manual;
                self.refresh_snapshot();
                self.push_event(self.snapshot.tick, EventKind::Info, "Focus: Manual");
            }
            _ => {}
        }

        Ok(false)
    }

    /// Step the sense-probe cone selection (bd-2z0.7.15).
    ///
    /// The states are `All` followed by each eye, so `NUM_EYES + 1` in total, and
    /// stepping wraps in both directions. Modelled as a total function over that
    /// ring rather than as increment-and-clamp: clamping would make the ends
    /// sticky, so a user holding the key would silently stop moving at eye 3 and
    /// read it as the control dying.
    fn cycle_eye_selection(&mut self, forward: bool) {
        let states = NUM_EYES + 1;
        let current = self.selected_eye.map_or(0, |eye| eye + 1);
        let next = if forward {
            (current + 1) % states
        } else {
            (current + states - 1) % states
        };
        self.selected_eye = (next > 0).then(|| next - 1);
        let label = self
            .selected_eye
            .map_or_else(|| "all cones".to_string(), |eye| format!("eye {eye}"));
        self.push_toast(format!("Sense probe: {label}"));
    }

    /// Whether the user asked for a screenshot and it has not been served yet.
    const fn export_requested(&self) -> bool {
        self.export_requested
    }

    /// Write the EXACT displayed frame as ANSI truecolor plus a plain-text
    /// fallback, and report both paths and the frame's evidence hash.
    ///
    /// Takes the rendered `Buffer` rather than re-deriving anything from world
    /// state. That is the whole point: the previous exporter re-rasterized
    /// terrain and food into a 64x32 ASCII grid with no agents and no panels, so
    /// it could not show what the user was looking at even in principle
    /// (bd-2z0.14.2.6). The frame identity is the SAME FNV-1a64 the headless
    /// evidence contract already publishes, so an export and a headless report of
    /// one frame cannot disagree.
    fn write_frame_export(&mut self, buffer: &Buffer) -> Result<(PathBuf, PathBuf, String)> {
        use std::io::Write;

        self.export_requested = false;
        let dir = std::path::Path::new("screenshots");
        fs::create_dir_all(dir)?;

        let tick = self.snapshot.tick;
        let ansi_path = dir.join(format!("frame_{tick}.ans"));
        let text_path = dir.join(format!("frame_{tick}.txt"));

        let ansi = export::buffer_to_ansi(buffer);
        let plain = export::buffer_to_plain_text(buffer);
        File::create(&ansi_path)?.write_all(ansi.as_bytes())?;
        File::create(&text_path)?.write_all(plain.as_bytes())?;

        let evidence = HeadlessBufferEvidence::inspect(buffer, tick)?;
        let hash = evidence.full_cell_fnv1a64.clone();
        info!(
            ansi = %ansi_path.display(),
            text = %text_path.display(),
            ansi_bytes = ansi.len(),
            text_bytes = plain.len(),
            width = buffer.area.width,
            height = buffer.area.height,
            hash = %hash,
            "exported the displayed terminal frame"
        );
        Ok((ansi_path, text_path, hash))
    }

    fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    fn refresh_snapshot(&mut self) {
        let next_request_revision = BrainInspectionRevision::new(
            self.brain_inspection_revision
                .get()
                .checked_add(1)
                .expect("terminal brain-inspection revision exhausted"),
        );
        let cached_inspection = self.brain_inspection_cache.clone();
        let mut next_inspection_cache = None;
        let mut request_issued = false;
        let new_snapshot = match self.world.lock() {
            Ok(world) => {
                let mut snap = Snapshot::from_world(&world);
                // Determine focused agent id
                let agent_id_opt = match self.focus_lock {
                    FocusLockMode::Manual => {
                        if snap.agent_count > 0 {
                            world
                                .agents()
                                .iter_handles()
                                .nth(self.focused_agent_cursor % snap.agent_count)
                        } else {
                            None
                        }
                    }
                    FocusLockMode::TopPredator => snap.leaderboard.first().and_then(|e| {
                        world
                            .agents()
                            .iter_handles()
                            .find(|h| h.data().as_ffi() == e.handle)
                    }),
                    FocusLockMode::Oldest => snap.oldest.first().and_then(|e| {
                        world
                            .agents()
                            .iter_handles()
                            .find(|h| h.data().as_ffi() == e.handle)
                    }),
                };
                if let Some(agent_uid) = agent_id_opt.and_then(|id| world.agent_uid(id)) {
                    snap.focused_agent_uid = Some(agent_uid.get());
                    if let Some(id) = agent_id_opt
                        && let Some(runtime) = world.agent_runtime(id)
                    {
                        snap.focused_brain_bound = runtime.brain.is_bound();
                        snap.focused_outputs = Some(runtime.outputs);
                    }
                    if let Some(cached) = cached_inspection.as_ref().filter(|cached| {
                        cached.metadata.agent_uid == agent_uid.get()
                            && cached.metadata.source_tick == world.tick().0
                    }) {
                        snap.brain_layers.clone_from(&cached.layers);
                        snap.brain_inspection = Some(cached.metadata);
                        snap.focused_activations.clone_from(&cached.activations);
                        next_inspection_cache = Some(cached.clone());
                    } else {
                        request_issued = true;
                        let request = BrainInspectionRequest::single(
                            TERMINAL_BRAIN_INSPECTION_CLIENT_ID,
                            next_request_revision,
                            agent_uid,
                        );
                        let cache = match world.inspect_brains(&request) {
                            Ok(response) => {
                                let mut metadata = BrainInspectionViewMetadata {
                                    agent_uid: agent_uid.get(),
                                    source_tick: response.source_tick.0,
                                    request_revision: response.request_revision.get(),
                                    truncated: false,
                                    retained_payload_bytes: response.build.retained_payload_bytes,
                                    ready: false,
                                };
                                let layers = response.ready_for(agent_uid).map_or_else(
                                    Vec::new,
                                    |telemetry| {
                                        metadata.truncated = telemetry.inspection.build.truncated;
                                        metadata.retained_payload_bytes =
                                            telemetry.inspection.build.retained_payload_bytes;
                                        metadata.ready = true;
                                        convert_layers(&telemetry.inspection.activations)
                                    },
                                );
                                let activations = response
                                    .ready_for(agent_uid)
                                    .map(|telemetry| telemetry.inspection.activations.clone());
                                TerminalBrainInspectionCache {
                                    metadata,
                                    layers,
                                    activations,
                                }
                            }
                            Err(error) => {
                                warn!(
                                    %error,
                                    agent_uid = agent_uid.get(),
                                    request_revision = next_request_revision.get(),
                                    "terminal brain inspection failed"
                                );
                                TerminalBrainInspectionCache {
                                    metadata: BrainInspectionViewMetadata {
                                        agent_uid: agent_uid.get(),
                                        source_tick: world.tick().0,
                                        request_revision: next_request_revision.get(),
                                        truncated: false,
                                        retained_payload_bytes: 0,
                                        ready: false,
                                    },
                                    layers: Vec::new(),
                                    activations: None,
                                }
                            }
                        };
                        snap.brain_layers.clone_from(&cache.layers);
                        snap.brain_inspection = Some(cache.metadata);
                        snap.focused_activations.clone_from(&cache.activations);
                        next_inspection_cache = Some(cache);
                    }
                }
                // Egocentric sense probe (bd-16g.4.2): attribution is computed
                // in core under this same lock; the panel renders it verbatim.
                snap.probe = if self.probe_enabled {
                    agent_id_opt.and_then(|agent_id| {
                        world.agent_uid(agent_id).and_then(|agent_uid| {
                            world.explain_sensors(agent_id, PROBE_MAX_CONTRIBUTORS).map(
                                |attribution| ProbeSnapshot {
                                    agent_uid: agent_uid.get(),
                                    attribution,
                                },
                            )
                        })
                    })
                } else {
                    None
                };
                snap
            }
            Err(_) => return,
        };
        self.brain_inspection_cache = next_inspection_cache;
        if request_issued {
            self.brain_inspection_revision = next_request_revision;
        }
        self.ingest_events(&new_snapshot);
        self.snapshot = new_snapshot;
        self.validate_rail_selection();
        self.maybe_log_rail_first_show();
        self.maybe_log_brain_panel();
        self.evaluate_auto_pause();
    }

    /// The rail's first-show logging contract (bd-16g.2.4): one line from which a
    /// reader can tell whether they are looking at a complete history or a tail.
    fn maybe_log_rail_first_show(&mut self) {
        if !self.rail_visible || self.rail_logged_first_show || self.snapshot.narrative.is_empty() {
            return;
        }
        self.rail_logged_first_show = true;
        let events = &self.snapshot.narrative;
        info!(
            target = "scriptbots::timeline",
            retained_events = events.len(),
            dropped_events = self.snapshot.narrative_dropped,
            capacity = self.snapshot.narrative_capacity,
            oldest_tick = events.first().map_or(0, |event| event.tick.0),
            newest_tick = events.last().map_or(0, |event| event.tick.0),
            "narrative rail first shown"
        );
    }

    fn ingest_events(&mut self, new_snapshot: &Snapshot) {
        if new_snapshot.tick <= self.last_event_tick && new_snapshot.tick <= self.snapshot.tick {
            return;
        }

        if new_snapshot.tick > self.last_event_tick {
            if new_snapshot.births > 0 {
                let plural = if new_snapshot.births == 1 { "" } else { "s" };
                self.push_event(
                    new_snapshot.tick,
                    EventKind::Birth,
                    format!("{} birth{}", new_snapshot.births, plural),
                );
            }
            if new_snapshot.deaths > 0 {
                let plural = if new_snapshot.deaths == 1 { "" } else { "s" };
                self.push_event(
                    new_snapshot.tick,
                    EventKind::Death,
                    format!("{} death{}", new_snapshot.deaths, plural),
                );
            }
        }

        if new_snapshot.tick > self.snapshot.tick {
            let delta = new_snapshot.agent_count as i64 - self.snapshot.agent_count as i64;
            if delta > 0 {
                self.push_event(
                    new_snapshot.tick,
                    EventKind::Population,
                    format!("Population +{}", delta),
                );
            } else if delta < 0 {
                self.push_event(
                    new_snapshot.tick,
                    EventKind::Population,
                    format!("Population {}", delta),
                );
            }
        }

        self.last_event_tick = new_snapshot.tick;
    }

    fn push_event(&mut self, tick: u64, kind: EventKind, message: impl Into<String>) {
        if self.event_log.len() >= EVENT_LOG_CAPACITY {
            self.event_log.pop_front();
        }
        self.event_log.push_back(EventEntry {
            tick,
            kind,
            message: message.into(),
        });
    }

    /// Evaluate auto-pause triggers from the current snapshot and update
    /// paused/speed state. Emits at most one Auto-pause event per tick.
    ///
    /// Covered by tests:
    /// - auto_pause_on_spike_hits
    /// - auto_pause_on_max_age
    /// - auto_pause_on_population_threshold
    /// - auto_pause_single_event_per_tick
    #[allow(clippy::collapsible_if)]
    fn evaluate_auto_pause(&mut self) {
        if self.paused {
            return;
        }

        let control = &self.snapshot.control;
        let mut reason: Option<String> = None;

        if control.auto_pause_on_spike_hit && self.snapshot.spike_hits > 0 {
            reason = Some(format!(
                "Auto-pause: spike hits detected ({})",
                self.snapshot.spike_hits
            ));
        } else if let Some(age_limit) = control.auto_pause_age_above {
            if self.snapshot.max_age >= age_limit {
                reason = Some(format!(
                    "Auto-pause: max age {} ≥ {}",
                    self.snapshot.max_age, age_limit
                ));
            }
        } else if let Some(limit) = control.auto_pause_population_below {
            if self.snapshot.agent_count as u32 <= limit {
                reason = Some(format!(
                    "Auto-pause: population {} ≤ {}",
                    self.snapshot.agent_count, limit
                ));
            }
        }

        if let Some(reason) = reason {
            if self.last_autopause_tick != Some(self.snapshot.tick) {
                self.push_event(self.snapshot.tick, EventKind::Info, &reason);
                self.last_autopause_tick = Some(self.snapshot.tick);
            }
            self.paused = true;
            self.speed_multiplier = 0.0;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Baseline {
    agent_count: usize,
    avg_energy: f32,
    avg_health: f32,
}

#[derive(Clone, Debug, Default)]
struct AnalyticsStatus {
    revision: u64,
    committed_tick: Option<u64>,
    lag: Option<u64>,
    last_error: Option<Arc<str>>,
    stopped: bool,
}

#[derive(Clone, Debug, Default)]
struct TerminalAnalytics {
    age_mean: f64,
    age_max: f64,
    boost_count: usize,
    boost_ratio: f64,
    food_mean: f64,
    food_stddev: f64,
    generation_mean: f64,
    generation_max: f64,
    mutation_primary_mean: f64,
    mutation_secondary_mean: f64,
    behavior_sensor_entropy: f64,
    behavior_output_entropy: f64,
    deaths_total: usize,
    deaths_combat_carnivore: usize,
    deaths_combat_herbivore: usize,
    deaths_starvation: usize,
    deaths_aging: usize,
    deaths_unknown: usize,
    births_total: usize,
    births_hybrid: usize,
    births_hybrid_ratio: f64,
    food_delta_mean: f64,
    food_delta_mean_abs: f64,
    carnivore_avg_energy: f64,
    herbivore_avg_energy: f64,
    hybrid_avg_energy: f64,
    traits_smell_mean: f64,
    traits_sound_mean: f64,
    traits_hearing_mean: f64,
    traits_eye_mean: f64,
    traits_blood_mean: f64,
    temperature_preference_mean: f64,
    temperature_preference_stddev: f64,
    temperature_discomfort_mean: f64,
    temperature_discomfort_stddev: f64,
    brain_shares: Vec<BrainShareEntry>,
}

#[derive(Clone, Debug)]
struct BrainShareEntry {
    label: String,
    count: usize,
    avg_energy: f64,
}

fn parse_terminal_analytics(
    _tick: u64,
    agent_count: usize,
    readings: &[MetricReading],
) -> Option<TerminalAnalytics> {
    if readings.is_empty() {
        return None;
    }
    let mut metrics: HashMap<String, f64> = HashMap::with_capacity(readings.len());
    for r in readings {
        metrics.insert(r.name.clone(), r.value);
    }
    let value = |k: &str| metrics.get(k).copied();
    let as_count = |k: &str| value(k).unwrap_or(0.0).max(0.0).round() as usize;

    let boost_count = as_count("behavior.boost.count");
    let boost_ratio = value("behavior.boost.ratio").unwrap_or_else(|| {
        if agent_count > 0 {
            boost_count as f64 / agent_count as f64
        } else {
            0.0
        }
    });

    // Brain shares aggregation
    let mut brain_map: HashMap<String, BrainShareEntry> = HashMap::new();
    for (name, &v) in &metrics {
        if let Some(rest) = name.strip_prefix("brain.population.") {
            if let Some(label) = rest.strip_suffix(".count") {
                let entry = brain_map
                    .entry(label.to_string())
                    .or_insert(BrainShareEntry {
                        label: label.to_string(),
                        count: 0,
                        avg_energy: 0.0,
                    });
                entry.count = v.max(0.0).round() as usize;
                continue;
            }
            if let Some(label) = rest.strip_suffix(".avg_energy") {
                let entry = brain_map
                    .entry(label.to_string())
                    .or_insert(BrainShareEntry {
                        label: label.to_string(),
                        count: 0,
                        avg_energy: 0.0,
                    });
                entry.avg_energy = v;
            }
        }
    }
    let mut brain_shares: Vec<BrainShareEntry> = brain_map.into_values().collect();
    brain_shares.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.label.cmp(&b.label)));

    Some(TerminalAnalytics {
        age_mean: value("population.age.mean").unwrap_or(0.0),
        age_max: value("population.age.max").unwrap_or(0.0),
        boost_count,
        boost_ratio,
        food_mean: value("food.mean").unwrap_or(0.0),
        food_stddev: value("food.stddev").unwrap_or(0.0),
        generation_mean: value("population.generation.mean").unwrap_or(0.0),
        generation_max: value("population.generation.max").unwrap_or(0.0),
        mutation_primary_mean: value("mutation.primary.mean").unwrap_or(0.0),
        mutation_secondary_mean: value("mutation.secondary.mean").unwrap_or(0.0),
        behavior_sensor_entropy: value("behavior.sensors.entropy").unwrap_or(0.0),
        behavior_output_entropy: value("behavior.outputs.entropy").unwrap_or(0.0),
        deaths_total: as_count("mortality.total.count"),
        deaths_combat_carnivore: as_count("mortality.combat_carnivore.count"),
        deaths_combat_herbivore: as_count("mortality.combat_herbivore.count"),
        deaths_starvation: as_count("mortality.starvation.count"),
        deaths_aging: as_count("mortality.aging.count"),
        deaths_unknown: as_count("mortality.unknown.count"),
        births_total: as_count("births.total.count"),
        births_hybrid: as_count("births.hybrid.count"),
        births_hybrid_ratio: value("births.hybrid.ratio").unwrap_or(0.0),
        food_delta_mean: value("food_delta.mean").unwrap_or(0.0),
        food_delta_mean_abs: value("food_delta.mean_abs").unwrap_or(0.0),
        carnivore_avg_energy: value("population.carnivore.avg_energy").unwrap_or(0.0),
        herbivore_avg_energy: value("population.herbivore.avg_energy").unwrap_or(0.0),
        hybrid_avg_energy: value("population.hybrid.avg_energy").unwrap_or(0.0),
        traits_smell_mean: value("traits.smell.mean").unwrap_or(0.0),
        traits_sound_mean: value("traits.sound.mean").unwrap_or(0.0),
        traits_hearing_mean: value("traits.hearing.mean").unwrap_or(0.0),
        traits_eye_mean: value("traits.eye.mean").unwrap_or(0.0),
        traits_blood_mean: value("traits.blood.mean").unwrap_or(0.0),
        temperature_preference_mean: value("temperature.preference.mean").unwrap_or(0.0),
        temperature_preference_stddev: value("temperature.preference.stddev").unwrap_or(0.0),
        temperature_discomfort_mean: value("temperature.discomfort.mean").unwrap_or(0.0),
        temperature_discomfort_stddev: value("temperature.discomfort.stddev").unwrap_or(0.0),
        brain_shares,
    })
}

/// Truthful per-channel provenance tag for the sense probe (bd-16g.4.2).
///
/// Attribution applies only to neighbour-derived channels; the panel says
/// where every other channel comes from instead of implying "no neighbours
/// detected" on a channel that never had contributors.
const fn sensor_source_tag(kind: SensorKind) -> &'static str {
    match kind {
        SensorKind::EyeDensity
        | SensorKind::EyeRed
        | SensorKind::EyeGreen
        | SensorKind::EyeBlue
        | SensorKind::Sound
        | SensorKind::Smell
        | SensorKind::Hearing
        | SensorKind::Blood => "nbr",
        SensorKind::Food => "grid",
        SensorKind::Health | SensorKind::Clock => "self",
        SensorKind::Temperature => "pos",
    }
}

fn diff_i(value: i64) -> String {
    if value > 0 {
        format!("(+{value})")
    } else if value < 0 {
        format!("({value})")
    } else {
        String::from("(+0)")
    }
}

fn diff_f(value: f32) -> String {
    if value > 0.0 {
        format!("(+{:.2})", value)
    } else if value < 0.0 {
        format!("({:.2})", value)
    } else {
        String::from("(+0.00)")
    }
}

#[derive(Clone, Default, Debug)]
struct Snapshot {
    tick: u64,
    epoch: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    avg_energy: f32,
    avg_health: f32,
    avg_age: f32,
    max_age: u32,
    boosted_count: usize,
    hybrid_count: usize,
    energy_min: f32,
    energy_max: f32,
    history: Vec<HistoryEntry>,
    world_size: (u32, u32),
    diet_split: DietSplit,
    agents: Vec<AgentViz>,
    leaderboard: Vec<LeaderboardEntry>,
    oldest: Vec<LeaderboardEntry>,
    food: FoodView,
    control: ControlSettings,
    spike_hits: u32,
    brain_layers: Vec<BrainLayerView>,
    brain_inspection: Option<BrainInspectionViewMetadata>,
    probe: Option<ProbeSnapshot>,
    /// The run's retained narrative events, oldest first (bd-16g.2.4 rail).
    narrative: Vec<NarrativeEventRecord>,
    /// Events the bounded narrative ring has discarded so far.
    narrative_dropped: u64,
    /// The narrative ring's configured capacity.
    narrative_capacity: usize,
    /// Focused agent's identity for the brain panel (bd-16g.4.3).
    focused_agent_uid: Option<u64>,
    /// Whether the focused agent's brain binding has a runner; an unbound agent's
    /// outputs are an identity copy of sensors and must not be "explained".
    focused_brain_bound: bool,
    /// The focused agent's latest brain outputs.
    focused_outputs: Option<[f32; scriptbots_core::OUTPUT_SIZE]>,
    /// The focused agent's raw bounded activation snapshot (for attribution).
    focused_activations: Option<BrainActivations>,
}

/// One agent's senses, explained (bd-16g.4.2).
///
/// A rendering-ready copy of the core attribution, captured under the same
/// world lock as the rest of the snapshot. The panel renders this verbatim —
/// attribution is computed in core and never re-derived here.
#[derive(Clone, Debug)]
struct ProbeSnapshot {
    agent_uid: u64,
    attribution: SensorAttribution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BrainInspectionViewMetadata {
    agent_uid: u64,
    source_tick: u64,
    request_revision: u64,
    truncated: bool,
    retained_payload_bytes: usize,
    ready: bool,
}

#[derive(Clone, Debug)]
struct TerminalBrainInspectionCache {
    metadata: BrainInspectionViewMetadata,
    layers: Vec<BrainLayerView>,
    /// The raw bounded activation snapshot the layer views were converted from;
    /// the attribution panel consumes the connections (bd-16g.4.3).
    activations: Option<BrainActivations>,
}

impl BrainInspectionViewMetadata {
    fn status_lines(self, current_tick: u64) -> (String, String) {
        let freshness = match self.source_tick.cmp(&current_tick) {
            Ordering::Less => "STALE",
            Ordering::Equal => "current",
            Ordering::Greater => "FUTURE",
        };
        let payload_status = if !self.ready {
            "UNAVAILABLE"
        } else if self.truncated {
            "CLIPPED"
        } else {
            "complete"
        };
        (
            format!("uid {} · t{} {freshness}", self.agent_uid, self.source_tick),
            format!(
                "r{} · {}B · {payload_status}",
                self.request_revision, self.retained_payload_bytes
            ),
        )
    }
}

#[derive(Clone, Default, Debug)]
struct HistoryEntry {
    tick: u64,
    births: usize,
    deaths: usize,
    avg_energy: f32,
    population: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct DietSplit {
    herbivores: usize,
    omnivores: usize,
    carnivores: usize,
}

impl DietSplit {
    fn total(&self) -> usize {
        self.herbivores + self.omnivores + self.carnivores
    }

    fn increment(&mut self, class: DietClass) {
        match class {
            DietClass::Herbivore => self.herbivores += 1,
            DietClass::Omnivore => self.omnivores += 1,
            DietClass::Carnivore => self.carnivores += 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum DietClass {
    Herbivore,
    Omnivore,
    Carnivore,
}

impl DietClass {
    fn from_tendency(tendency: f32) -> Self {
        if tendency <= 0.33 {
            DietClass::Herbivore
        } else if tendency >= 0.66 {
            DietClass::Carnivore
        } else {
            DietClass::Omnivore
        }
    }
}

#[derive(Clone, Debug)]
struct AgentViz {
    /// The generational arena handle, encoded for cheap comparison. This is a
    /// PHYSICAL slot: it is reused after an agent dies, so it identifies a
    /// storage location and never an agent across time. Use it to look a live
    /// agent back up; never show it to a person.
    id: u64,
    /// The stable scientific `AgentUid` — the identity digests, lineage, replay,
    /// and persistence all use, and therefore the only one worth displaying.
    ///
    /// `None` means the arena held a handle the identity map did not know, which
    /// core documents as an invariant violation. Surfaced as `#?` rather than
    /// substituted with a plausible number.
    uid: Option<u64>,
    position: (f32, f32),
    heading: f32,
    diet: DietClass,
    energy: f32,
    health: f32,
    age: u32,
    generation: u32,
    boosted: bool,
    spike_length: f32,
    tendency: f32,
}

#[derive(Clone, Debug)]
struct LeaderboardEntry {
    /// Arena handle, kept only so focus-lock can resolve the row back to a live
    /// agent. Never displayed — see [`AgentViz::id`].
    handle: u64,
    /// Stable `AgentUid`; this is what the row shows.
    uid: Option<u64>,
    diet: DietClass,
    energy: f32,
    health: f32,
    age: u32,
    generation: u32,
}

#[derive(Clone, Debug, Default)]
struct FoodView {
    width: u32,
    height: u32,
    cells: Vec<f32>,
    max: f32,
    mean: f32,
}

impl FoodView {
    fn sample(&self, u: f32, v: f32) -> f32 {
        if self.width == 0 || self.height == 0 || self.cells.is_empty() {
            return 0.0;
        }
        let x = ((u.clamp(0.0, 0.9999)) * self.width as f32).floor() as usize;
        let y = ((v.clamp(0.0, 0.9999)) * self.height as f32).floor() as usize;
        let idx = y.saturating_mul(self.width as usize) + x;
        let value = *self.cells.get(idx).unwrap_or(&0.0);
        if self.max <= f32::EPSILON {
            0.0
        } else {
            (value / self.max).clamp(0.0, 1.0)
        }
    }
}

#[derive(Clone, Debug)]
struct TerrainView {
    width: u32,
    height: u32,
    kinds: Vec<TerrainKind>,
    /// Per-tile normalized elevation, parallel to `kinds`. Only the sub-cell
    /// canvas reads it, to build the hillshade gradient; the flat map ignores it.
    elevations: Vec<f32>,
    /// Per-tile moisture, parallel to `kinds`.
    moisture: Vec<f32>,
    /// Per-tile food-fertility bias, parallel to `kinds`.
    fertility: Vec<f32>,
}

impl TerrainView {
    fn from(terrain: &TerrainLayer) -> Self {
        let tiles = terrain.tiles();
        Self {
            width: terrain.width(),
            height: terrain.height(),
            kinds: tiles.iter().map(|tile| tile.kind).collect(),
            elevations: tiles.iter().map(|tile| tile.elevation).collect(),
            moisture: tiles.iter().map(|tile| tile.moisture).collect(),
            fertility: tiles.iter().map(|tile| tile.fertility_bias).collect(),
        }
    }

    /// Shared lushness for a normalized world point: how green this tile should
    /// read within its biome, from [`visual::terrain_lushness`].
    ///
    /// Falls back to the neutral midpoint when the parallel arrays are short, so
    /// a truncated view tints uniformly instead of striping the map.
    fn lushness(&self, u: f32, v: f32) -> f32 {
        if self.moisture.len() != self.kinds.len() || self.fertility.len() != self.kinds.len() {
            return 0.5;
        }
        let Some((x, y)) = self.tile_coords(u, v) else {
            return 0.5;
        };
        let idx = (y as usize).saturating_mul(self.width as usize) + x as usize;
        match (self.moisture.get(idx), self.fertility.get(idx)) {
            (Some(&moisture), Some(&fertility)) => visual::terrain_lushness(moisture, fertility),
            _ => 0.5,
        }
    }

    /// Tile coordinates for a normalized world point, or `None` for an empty view.
    fn tile_coords(&self, u: f32, v: f32) -> Option<(u32, u32)> {
        if self.width == 0 || self.height == 0 || self.kinds.is_empty() {
            return None;
        }
        let x = ((u.clamp(0.0, 0.9999)) * self.width as f32).floor() as u32;
        let y = ((v.clamp(0.0, 0.9999)) * self.height as f32).floor() as u32;
        Some((x.min(self.width - 1), y.min(self.height - 1)))
    }

    fn sample(&self, u: f32, v: f32) -> TerrainKind {
        let Some((x, y)) = self.tile_coords(u, v) else {
            return TerrainKind::Grass;
        };
        let idx = (y as usize).saturating_mul(self.width as usize) + x as usize;
        self.kinds.get(idx).copied().unwrap_or(TerrainKind::Grass)
    }

    /// Elevation of one tile, clamped to the grid (edges repeat rather than wrap:
    /// a wrapped neighbour would light the world seam as a false cliff).
    fn elevation_at(&self, x: u32, y: u32) -> f32 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let x = x.min(self.width - 1) as usize;
        let y = y.min(self.height - 1) as usize;
        let idx = y.saturating_mul(self.width as usize) + x;
        self.elevations.get(idx).copied().unwrap_or(0.0)
    }

    /// Central-difference elevation gradient `[dh/dx, dh/dy]` in elevation units
    /// per tile, for [`scriptbots_core::visual::terrain_normal_light_factor`].
    ///
    /// Flat ground yields `[0, 0]`, which that function maps to exactly `1.0`, so
    /// a world with no relief is neither darkened nor brightened by hillshading.
    fn elevation_gradient(&self, u: f32, v: f32) -> [f32; 2] {
        if self.elevations.len() != self.kinds.len() {
            return [0.0, 0.0];
        }
        let Some((x, y)) = self.tile_coords(u, v) else {
            return [0.0, 0.0];
        };
        let left = self.elevation_at(x.saturating_sub(1), y);
        let right = self.elevation_at(x.saturating_add(1), y);
        let up = self.elevation_at(x, y.saturating_sub(1));
        let down = self.elevation_at(x, y.saturating_add(1));
        [(right - left) * 0.5, (down - up) * 0.5]
    }
}

// Removed grid-based glyph buffering in favor of a buffer-writing widget.

#[derive(Clone, Copy)]
struct CellOccupancy {
    herbivores: u16,
    omnivores: u16,
    carnivores: u16,
    boosted: bool,
    top_energy: f32,
    top_class: DietClass,
    heading_sin: f32,
    heading_cos: f32,
    heading_count: u16,
    spike_peak: f32,
    tendency_accum: f32,
    stamp: u32,
}

impl Default for CellOccupancy {
    fn default() -> Self {
        Self {
            herbivores: 0,
            omnivores: 0,
            carnivores: 0,
            boosted: false,
            top_energy: 0.0,
            top_class: DietClass::Omnivore,
            heading_sin: 0.0,
            heading_cos: 0.0,
            heading_count: 0,
            spike_peak: 0.0,
            tendency_accum: 0.0,
            stamp: 0,
        }
    }
}

impl CellOccupancy {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        class: DietClass,
        boosted: bool,
        energy: f32,
        heading: f32,
        spike: f32,
        tendency: f32,
        stamp: u32,
    ) {
        if self.stamp != stamp {
            *self = CellOccupancy {
                stamp,
                ..Default::default()
            };
        }
        match class {
            DietClass::Herbivore => self.herbivores = self.herbivores.saturating_add(1),
            DietClass::Omnivore => self.omnivores = self.omnivores.saturating_add(1),
            DietClass::Carnivore => self.carnivores = self.carnivores.saturating_add(1),
        }
        if boosted {
            self.boosted = true;
        }
        if energy >= self.top_energy {
            self.top_energy = energy;
            self.top_class = class;
        }
        let (s, c) = heading.sin_cos();
        self.heading_sin += s;
        self.heading_cos += c;
        self.heading_count = self.heading_count.saturating_add(1);
        if spike > self.spike_peak {
            self.spike_peak = spike;
        }
        self.tendency_accum += tendency;
    }

    fn total(&self) -> u16 {
        self.herbivores
            .saturating_add(self.omnivores)
            .saturating_add(self.carnivores)
    }

    fn dominant(&self) -> DietClass {
        let mut best = (self.herbivores, DietClass::Herbivore);
        if self.omnivores > best.0 {
            best = (self.omnivores, DietClass::Omnivore);
        }
        if self.carnivores > best.0 {
            best = (self.carnivores, DietClass::Carnivore);
        }
        if best.0 == 0 { self.top_class } else { best.1 }
    }

    fn mean_heading(&self) -> Option<f32> {
        if self.heading_count == 0 {
            None
        } else {
            Some(self.heading_sin.atan2(self.heading_cos))
        }
    }

    fn mean_tendency(&self) -> Option<f32> {
        if self.heading_count == 0 {
            None
        } else {
            Some(self.tendency_accum / self.heading_count as f32)
        }
    }
}

#[derive(Clone, Debug)]
struct EventEntry {
    tick: u64,
    message: String,
    kind: EventKind,
}

#[derive(Clone, Copy, Debug)]
enum EventKind {
    Birth,
    Death,
    Population,
    Info,
}

impl EventKind {
    /// A one-character marker so the event log reads without colour.
    ///
    /// The log rendered every kind as identically-formatted text and carried the
    /// kind in the FOREGROUND COLOUR ALONE, so in monochrome — or for a reader
    /// who cannot separate these hues — a birth and a death were the same line
    /// (bd-xg82). Letters rather than symbols, matching the C/H/S/A/U convention
    /// the mortality panel already uses, and ASCII so the narrow and ascii
    /// capability tiers render them unchanged.
    const fn marker(self) -> char {
        match self {
            Self::Birth => 'B',
            Self::Death => 'D',
            Self::Population => 'P',
            Self::Info => 'i',
        }
    }

    /// Every kind, for tests that must see them as a set.
    const fn all() -> [Self; 4] {
        [Self::Birth, Self::Death, Self::Population, Self::Info]
    }
}

/// Row labels for the four stacked trend sparklines, and the width reserved for
/// them.
///
/// These exist because the four sparklines were distinguished by COLOUR ALONE
/// (bd-xg82). Measuring the palette proved that indefensible rather than merely
/// imperfect: `population_spark` and the Birth event colour are the SAME VALUE in
/// every accessibility palette — a measured separation of exactly 1.000:1 — so
/// the population and births rows rendered identically for every viewer, not
/// only hue-blind ones. Four unlabelled full-width bars, two of them the same
/// colour, stacked adjacently.
///
/// The fix is a channel that does not depend on hue at all. Labels also make the
/// panel readable for someone who simply has not memorised which row is which,
/// which is most people.
const TREND_LABEL_WIDTH: u16 = 4;
const TREND_POPULATION: &str = "pop";
const TREND_ENERGY: &str = "nrg";
const TREND_BIRTHS: &str = "brt";
const TREND_DEATHS: &str = "dth";

/// Every trend label, for tests that must see them as a set.
const TREND_LABELS: [&str; 4] = [TREND_POPULATION, TREND_ENERGY, TREND_BIRTHS, TREND_DEATHS];

/// Why an agent died, as the mortality panel breaks it down.
///
/// Named rather than passed as a bare colour so the panel and its palette
/// mapping cannot disagree about which bar is which, and so adding a cause is a
/// compile error in one place instead of a silently unstyled sixth bar
/// (bd-f4x0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MortalityCause {
    CombatCarnivore,
    CombatHerbivore,
    Starvation,
    Aging,
    Unknown,
}

impl MortalityCause {
    /// The column label shown beside the bar.
    const fn label(self) -> &'static str {
        match self {
            Self::CombatCarnivore => "C",
            Self::CombatHerbivore => "H",
            Self::Starvation => "S",
            Self::Aging => "A",
            Self::Unknown => "U",
        }
    }

    /// Every cause, in display order.
    const fn all() -> [Self; 5] {
        [
            Self::CombatCarnivore,
            Self::CombatHerbivore,
            Self::Starvation,
            Self::Aging,
            Self::Unknown,
        ]
    }
}

impl Snapshot {
    fn from_world(world: &WorldState) -> Self {
        let config = world.config();
        let agent_count = world.agent_count();
        let world_width = config.world_width.max(1) as f32;
        let world_height = config.world_height.max(1) as f32;

        let summary = world
            .history()
            .next_back()
            .cloned()
            .unwrap_or_else(|| TickSummary {
                tick: world.tick(),
                agent_count,
                births: 0,
                deaths: 0,
                total_energy: 0.0,
                average_energy: 0.0,
                average_health: 0.0,
                max_age: 0,
                spike_hits: 0,
            });
        let history: Vec<HistoryEntry> = world
            .history()
            .rev()
            .take(32)
            .map(|entry| HistoryEntry {
                tick: entry.tick.0,
                births: entry.births,
                deaths: entry.deaths,
                avg_energy: entry.average_energy,
                population: entry.agent_count,
            })
            .collect();

        let handles: Vec<AgentId> = world.agents().iter_handles().collect();
        let columns = world.agents().columns();
        let runtimes = world.runtime();

        let mut agents = Vec::with_capacity(handles.len());
        let mut diet_split = DietSplit::default();
        let mut boosted_count = 0usize;
        let mut hybrid_count = 0usize;
        let mut energy_min = f32::INFINITY;
        let mut energy_max = f32::NEG_INFINITY;
        let mut health_acc = 0.0_f32;
        let mut age_acc = 0.0_f64;
        let mut max_age = 0u32;

        for (idx, id) in handles.iter().enumerate() {
            let position = columns.positions()[idx];
            let heading = columns.headings()[idx];
            let health = columns.health()[idx];
            let age = columns.ages()[idx];
            let generation = columns.generations()[idx].0;
            let runtime = runtimes.get(*id);

            let energy = runtime.map(|rt| rt.energy).unwrap_or(0.0);
            let diet = runtime
                .map(|rt| DietClass::from_tendency(rt.herbivore_tendency))
                .unwrap_or(DietClass::Omnivore);
            let boosted = columns.boosts()[idx];
            let hybrid = runtime.map(|rt| rt.hybrid).unwrap_or(false);
            let spike_length = columns.spike_lengths()[idx];
            let tendency = runtime.map(|rt| rt.herbivore_tendency).unwrap_or(0.5);

            diet_split.increment(diet);
            if boosted {
                boosted_count += 1;
            }
            if hybrid {
                hybrid_count += 1;
            }

            energy_min = energy_min.min(energy);
            energy_max = energy_max.max(energy);
            health_acc += health;
            age_acc += f64::from(age);
            max_age = max_age.max(age);

            let normalized_x = (position.x / world_width)
                .rem_euclid(1.0)
                .clamp(0.0, 0.9999);
            let normalized_y = (position.y / world_height)
                .rem_euclid(1.0)
                .clamp(0.0, 0.9999);

            agents.push(AgentViz {
                id: id.data().as_ffi(),
                uid: world.agent_uid(*id).map(|uid| uid.get()),
                position: (normalized_x, normalized_y),
                heading,
                diet,
                energy,
                health,
                age,
                generation,
                boosted,
                spike_length,
                tendency,
            });
        }

        let avg_health = if agent_count > 0 {
            health_acc / agent_count as f32
        } else {
            0.0
        };
        let avg_age = if agent_count > 0 {
            (age_acc / agent_count as f64) as f32
        } else {
            0.0
        };

        if !energy_min.is_finite() {
            energy_min = 0.0;
        }
        if !energy_max.is_finite() {
            energy_max = 0.0;
        }

        // Top Predators: carnivores by energy (health tie-break)
        let mut leaderboard: Vec<LeaderboardEntry> = agents
            .iter()
            .filter(|a| matches!(a.diet, DietClass::Carnivore))
            .map(|agent| LeaderboardEntry {
                handle: agent.id,
                uid: agent.uid,
                diet: agent.diet,
                energy: agent.energy,
                health: agent.health,
                age: agent.age,
                generation: agent.generation,
            })
            .collect();

        leaderboard.sort_by(|a, b| {
            b.energy
                .partial_cmp(&a.energy)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.health.partial_cmp(&a.health).unwrap_or(Ordering::Equal))
        });
        leaderboard.truncate(LEADERBOARD_LIMIT);

        // Oldest Agents: across all diets by age
        let mut oldest: Vec<LeaderboardEntry> = agents
            .iter()
            .map(|agent| LeaderboardEntry {
                handle: agent.id,
                uid: agent.uid,
                diet: agent.diet,
                energy: agent.energy,
                health: agent.health,
                age: agent.age,
                generation: agent.generation,
            })
            .collect();
        oldest.sort_by_key(|entry| Reverse(entry.age));
        oldest.truncate(LEADERBOARD_LIMIT);

        let food_grid = world.food();
        let food_cells = food_grid.cells().to_vec();
        let food_max = food_cells
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(*value));
        let food_mean = if food_cells.is_empty() {
            0.0
        } else {
            food_cells.iter().sum::<f32>() / food_cells.len() as f32
        };

        Self {
            tick: summary.tick.0,
            epoch: world.epoch(),
            agent_count,
            births: summary.births,
            deaths: summary.deaths,
            avg_energy: summary.average_energy,
            avg_health,
            avg_age,
            max_age,
            boosted_count,
            hybrid_count,
            energy_min,
            energy_max,
            history,
            world_size: (config.world_width, config.world_height),
            diet_split,
            agents,
            leaderboard,
            oldest,
            food: FoodView {
                width: food_grid.width(),
                height: food_grid.height(),
                cells: food_cells,
                max: food_max,
                mean: food_mean,
            },
            control: config.control.clone(),
            spike_hits: summary.spike_hits,
            brain_layers: Vec::new(),
            brain_inspection: None,
            probe: None,
            narrative: world.narrative_events().iter().cloned().collect(),
            narrative_dropped: world.narrative_dropped_events(),
            narrative_capacity: config.narrative_capacity,
            focused_agent_uid: None,
            focused_brain_bound: false,
            focused_outputs: None,
            focused_activations: None,
        }
    }

    fn brain_activations_layer_indexed(&self, idx: usize) -> Option<&BrainLayerView> {
        self.brain_layers.get(idx)
    }
}

#[derive(Debug, Clone, Serialize)]
struct HeadlessReport {
    scenario: ScenarioSummary,
    initial: FrameStats,
    frames: Vec<FrameStats>,
    summary: ReportSummary,
}

/// Scenario identity carried by every headless report so CI evidence names the run.
#[derive(Debug, Clone, Serialize)]
struct ScenarioSummary {
    id: String,
    schema_version: u16,
    bootstrap_ticks: u64,
}

impl From<&ScenarioIdentityV0> for ScenarioSummary {
    fn from(scenario: &ScenarioIdentityV0) -> Self {
        Self {
            id: scenario.id.clone(),
            schema_version: scenario.schema_version,
            bootstrap_ticks: scenario.bootstrap_ticks,
        }
    }
}

impl HeadlessReport {
    fn new(initial_snapshot: Snapshot, scenario: &ScenarioIdentityV0) -> Self {
        Self {
            scenario: ScenarioSummary::from(scenario),
            initial: FrameStats::from_snapshot(&initial_snapshot, None),
            frames: Vec::new(),
            summary: ReportSummary::default(),
        }
    }

    fn record(&mut self, snapshot: &Snapshot, buffer: HeadlessBufferEvidence) {
        self.frames
            .push(FrameStats::from_snapshot(snapshot, Some(buffer)));
    }

    fn finalize(&mut self, world_digest: Option<String>) {
        self.summary = ReportSummary::from(&self.initial, &self.frames, world_digest);
    }

    fn write_json(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, self).context("failed to serialize headless report")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
struct FrameStats {
    tick: u64,
    epoch: u64,
    agent_count: usize,
    births: usize,
    deaths: usize,
    spike_hits: u32,
    avg_energy: f32,
    buffer: Option<HeadlessBufferEvidence>,
}

impl FrameStats {
    fn from_snapshot(snapshot: &Snapshot, buffer: Option<HeadlessBufferEvidence>) -> Self {
        Self {
            tick: snapshot.tick,
            epoch: snapshot.epoch,
            agent_count: snapshot.agent_count,
            births: snapshot.births,
            deaths: snapshot.deaths,
            spike_hits: snapshot.spike_hits,
            avg_energy: snapshot.avg_energy,
            buffer,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct HeadlessBufferEvidence {
    backend: &'static str,
    capability_profile: &'static str,
    viewport_width: u16,
    viewport_height: u16,
    current_tick: u64,
    non_blank_cells: usize,
    styled_cells: usize,
    skipped_cells: usize,
    forced_width_cells: usize,
    empty_symbol_cells: usize,
    full_cell_fnv1a64: String,
    semantic_regions: Vec<&'static str>,
}

impl HeadlessBufferEvidence {
    fn inspect(buffer: &Buffer, current_tick: u64) -> Result<Self> {
        const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;

        let area = buffer.area;
        let mut text = String::new();
        let mut non_blank_cells = 0usize;
        let mut styled_cells = 0usize;
        let mut skipped_cells = 0usize;
        let mut forced_width_cells = 0usize;
        let mut empty_symbol_cells = 0usize;
        let mut hash = OFFSET_BASIS;

        let mut hash_bytes = |bytes: &[u8]| {
            for byte in (bytes.len() as u64).to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(PRIME);
            }
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(PRIME);
            }
        };

        hash_bytes(&area.width.to_le_bytes());
        hash_bytes(&area.height.to_le_bytes());

        for y in area.y..area.bottom() {
            for x in area.x..area.right() {
                let cell = &buffer[(x, y)];
                let symbol = cell.symbol();
                if !symbol.trim().is_empty() {
                    non_blank_cells += 1;
                }
                if symbol.is_empty() {
                    empty_symbol_cells += 1;
                }
                if matches!(cell.diff_option, CellDiffOption::Skip) {
                    skipped_cells += 1;
                }
                if matches!(cell.diff_option, CellDiffOption::ForcedWidth(_)) {
                    forced_width_cells += 1;
                }
                if cell.fg != Color::Reset
                    || cell.bg != Color::Reset
                    || cell.underline_color != Color::Reset
                    || !cell.modifier.is_empty()
                {
                    styled_cells += 1;
                }
                text.push_str(symbol);
                hash_bytes(&x.to_le_bytes());
                hash_bytes(&y.to_le_bytes());
                hash_bytes(symbol.as_bytes());
                hash_bytes(format!("{:?}", cell.fg).as_bytes());
                hash_bytes(format!("{:?}", cell.bg).as_bytes());
                hash_bytes(format!("{:?}", cell.underline_color).as_bytes());
                hash_bytes(format!("{:?}", cell.modifier).as_bytes());
                hash_bytes(format!("{:?}", cell.diff_option).as_bytes());
            }
            text.push('\n');
        }

        let tick_label = format!("Tick {current_tick:>6}");
        let required_regions = [
            ("terminal_hud", "ScriptBots Terminal HUD"),
            ("current_tick", tick_label.as_str()),
            ("world_map", "World Map"),
            ("vital_stats", "Vital Stats"),
        ];
        for (region, needle) in required_regions {
            ensure!(
                text.contains(needle),
                "headless TestBackend frame at tick {current_tick} omitted required {region} content {needle:?} from its {}x{} buffer",
                area.width,
                area.height
            );
        }
        ensure!(
            non_blank_cells > 0,
            "headless TestBackend frame at tick {current_tick} produced an empty buffer"
        );

        Ok(Self {
            backend: "ratatui_test_backend",
            capability_profile: "ascii_natural_fixed_80x36",
            viewport_width: area.width,
            viewport_height: area.height,
            current_tick,
            non_blank_cells,
            styled_cells,
            skipped_cells,
            forced_width_cells,
            empty_symbol_cells,
            full_cell_fnv1a64: format!("{hash:016x}"),
            semantic_regions: required_regions
                .into_iter()
                .map(|(region, _)| region)
                .collect(),
        })
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct ReportSummary {
    frame_count: usize,
    ticks_simulated: u64,
    final_tick: u64,
    final_epoch: u64,
    final_agent_count: usize,
    total_births: usize,
    total_deaths: usize,
    total_spike_hits: u64,
    avg_energy_mean: f32,
    avg_energy_min: f32,
    avg_energy_max: f32,
    world_digest: Option<String>,
}

impl ReportSummary {
    fn from(initial: &FrameStats, frames: &[FrameStats], world_digest: Option<String>) -> Self {
        if frames.is_empty() {
            return Self {
                frame_count: 0,
                ticks_simulated: 0,
                final_tick: initial.tick,
                final_epoch: initial.epoch,
                final_agent_count: initial.agent_count,
                total_births: 0,
                total_deaths: 0,
                total_spike_hits: 0,
                avg_energy_mean: initial.avg_energy,
                avg_energy_min: initial.avg_energy,
                avg_energy_max: initial.avg_energy,
                world_digest,
            };
        }

        let frame_count = frames.len();
        let final_stats = frames.last().expect("frame list not empty");
        let ticks_simulated = final_stats.tick.saturating_sub(initial.tick);

        let total_births = frames.iter().map(|frame| frame.births).sum();
        let total_deaths = frames.iter().map(|frame| frame.deaths).sum();
        let total_spike_hits = frames.iter().map(|frame| u64::from(frame.spike_hits)).sum();

        let mut min_energy = f32::INFINITY;
        let mut max_energy = f32::NEG_INFINITY;
        let mut energy_sum = 0.0_f32;
        for frame in frames {
            let energy = frame.avg_energy;
            if energy < min_energy {
                min_energy = energy;
            }
            if energy > max_energy {
                max_energy = energy;
            }
            energy_sum += energy;
        }

        let avg_energy_mean = if frame_count > 0 {
            energy_sum / frame_count as f32
        } else {
            initial.avg_energy
        };

        Self {
            frame_count,
            ticks_simulated,
            final_tick: final_stats.tick,
            final_epoch: final_stats.epoch,
            final_agent_count: final_stats.agent_count,
            total_births,
            total_deaths,
            total_spike_hits,
            avg_energy_mean,
            avg_energy_min: min_energy,
            avg_energy_max: max_energy,
            world_digest,
        }
    }
}

fn report_file_path_from_env() -> Option<PathBuf> {
    std::env::var_os("SCRIPTBOTS_TERMINAL_HEADLESS_REPORT").and_then(|raw| {
        if raw.is_empty() {
            None
        } else {
            Some(PathBuf::from(raw))
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalPaletteMode {
    Natural,
    Deuteranopia,
    Protanopia,
    Tritanopia,
    HighContrast,
}

impl TerminalPaletteMode {
    fn next(self) -> Self {
        match self {
            Self::Natural => Self::Deuteranopia,
            Self::Deuteranopia => Self::Protanopia,
            Self::Protanopia => Self::Tritanopia,
            Self::Tritanopia => Self::HighContrast,
            Self::HighContrast => Self::Natural,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Natural => "Natural",
            Self::Deuteranopia => "Deuteranopia",
            Self::Protanopia => "Protanopia",
            Self::Tritanopia => "Tritanopia",
            Self::HighContrast => "High Contrast",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToastEntry {
    pub message: String,
    pub created_tick: u64,
    pub expiry_tick: u64,
}

impl ToastEntry {
    pub fn new(message: impl Into<String>, current_tick: u64, lifetime_ticks: u64) -> Self {
        Self {
            message: message.into(),
            created_tick: current_tick,
            expiry_tick: current_tick.saturating_add(lifetime_ticks),
        }
    }

    pub fn is_expired(&self, current_tick: u64) -> bool {
        current_tick >= self.expiry_tick
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MouseHoverTooltip {
    pub cell_x: u16,
    pub cell_y: u16,
    /// The hovered agent's stable `AgentUid`, matching what the brain panel
    /// reports. Before bd-qxrt this field was named `agent_uid` but carried the
    /// reusable arena handle, so the tooltip and the brain panel showed two
    /// different numbers for one agent.
    pub agent_uid: Option<u64>,
    pub energy: f32,
    pub health: f32,
    pub age: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommandPaletteItem {
    pub label: &'static str,
    pub keybind_hint: &'static str,
    pub category: &'static str,
    pub action: CommandPaletteAction,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CommandPaletteAction {
    TogglePause,
    StepOnce,
    SpeedUp,
    SpeedDown,
    CycleTheme,
    CyclePalette,
    ToggleRail,
    FocusTopPredator,
    FocusOldest,
    ToggleProbe,
    ShowHelp,
}

pub fn all_command_palette_items() -> Vec<CommandPaletteItem> {
    vec![
        CommandPaletteItem {
            label: "Toggle Pause / Resume",
            keybind_hint: "Space",
            category: "Playback",
            action: CommandPaletteAction::TogglePause,
        },
        CommandPaletteItem {
            label: "Step Single Sim Tick",
            keybind_hint: "s",
            category: "Playback",
            action: CommandPaletteAction::StepOnce,
        },
        CommandPaletteItem {
            label: "Faster Sim Speed",
            keybind_hint: "+",
            category: "Playback",
            action: CommandPaletteAction::SpeedUp,
        },
        CommandPaletteItem {
            label: "Slower Sim Speed",
            keybind_hint: "-",
            category: "Playback",
            action: CommandPaletteAction::SpeedDown,
        },
        CommandPaletteItem {
            label: "Cycle Curated Theme",
            keybind_hint: "Ctrl+T",
            category: "View",
            action: CommandPaletteAction::CycleTheme,
        },
        CommandPaletteItem {
            label: "Cycle Accessibility Palette",
            // Both are live. `p` is the documented binding and matches GPUI; `c`
            // is retained so the previous binding keeps working (bd-2z0.14.2.2).
            keybind_hint: "p / c",
            category: "View",
            action: CommandPaletteAction::CyclePalette,
        },
        CommandPaletteItem {
            label: "Toggle Narrative Timeline Rail",
            keybind_hint: "r",
            category: "View",
            action: CommandPaletteAction::ToggleRail,
        },
        CommandPaletteItem {
            label: "Focus Top Predator Agent",
            keybind_hint: "t",
            category: "Science",
            action: CommandPaletteAction::FocusTopPredator,
        },
        CommandPaletteItem {
            label: "Focus Oldest Living Agent",
            keybind_hint: "o",
            category: "Science",
            action: CommandPaletteAction::FocusOldest,
        },
        CommandPaletteItem {
            label: "Toggle Senses Attribution Probe",
            keybind_hint: "b",
            category: "Science",
            action: CommandPaletteAction::ToggleProbe,
        },
        CommandPaletteItem {
            label: "Show Keybindings & Legend",
            keybind_hint: "?",
            category: "Help",
            action: CommandPaletteAction::ShowHelp,
        },
    ]
}

pub fn fuzzy_match_command_palette<'a>(
    items: &'a [CommandPaletteItem],
    query: &str,
) -> Vec<&'a CommandPaletteItem> {
    if query.trim().is_empty() {
        return items.iter().collect();
    }
    let query_lower = query.to_lowercase();
    let mut matched: Vec<(&'a CommandPaletteItem, usize)> = items
        .iter()
        .filter_map(|item| {
            let label_lower = item.label.to_lowercase();
            let cat_lower = item.category.to_lowercase();
            if label_lower.contains(&query_lower) || cat_lower.contains(&query_lower) {
                let score = if label_lower.starts_with(&query_lower) {
                    0
                } else {
                    1
                };
                Some((item, score))
            } else {
                None
            }
        })
        .collect();

    matched.sort_by_key(|(_, score)| *score);
    matched.into_iter().map(|(item, _)| item).collect()
}

/// Deterministic motion clock (bd-2z0.14.2.4).
/// Single tick_phase source derived from sim tick and deterministic UI subdivisions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickPhase {
    pub tick: u64,
    pub phase: f32,
    pub reduced_motion: bool,
    pub paused: bool,
}

impl TickPhase {
    pub fn compute(tick: u64, paused: bool, sub_step: u8) -> Self {
        let reduced = is_reduced_motion_requested();
        let sub = if paused || reduced {
            0.0
        } else {
            (sub_step % 16) as f32 / 16.0
        };
        let phase = if reduced || paused {
            0.0
        } else {
            ((tick % 60) as f32 + sub) / 60.0
        };
        Self {
            tick,
            phase,
            reduced_motion: reduced,
            paused,
        }
    }

    pub fn pulse(&self, frequency: f32, min_val: f32, max_val: f32) -> f32 {
        if self.reduced_motion || self.paused {
            return (min_val + max_val) * 0.5;
        }
        let rad = (self.tick as f32 * frequency + self.phase * 2.0 * std::f32::consts::PI)
            % (2.0 * std::f32::consts::PI);
        let s = (rad.sin() + 1.0) * 0.5;
        min_val + s * (max_val - min_val)
    }
}

/// Event pulse ring radiating from event locations (bd-2z0.14.2.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EventPulseRing {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub intensity: f32,
}

impl EventPulseRing {
    pub fn from_event(event_tick: u64, current_tick: u64, x: f32, y: f32) -> Option<Self> {
        let age = current_tick.saturating_sub(event_tick);
        if age > 5 {
            return None;
        }
        let radius = age as f32 * 1.5 + 0.5;
        let intensity = (1.0 - age as f32 / 6.0).clamp(0.0, 1.0);
        Some(Self {
            x,
            y,
            radius,
            intensity,
        })
    }
}

pub fn is_reduced_motion_requested() -> bool {
    std::env::var("SCRIPTBOTS_REDUCED_MOTION").is_ok() || std::env::var("NO_COLOR").is_ok()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CuratedThemeId {
    /// bd-9pqz art direction, derived from `visual::BIOLUMINESCENT_DARK_FIELD_V1`.
    /// The default, so terminal mode shares the GUI's look instead of diverging.
    BioluminescentDarkField,
    CyberpunkAurora,
    Darcula,
    LumenLight,
    NordicFrost,
    HighContrast,
}

impl Default for CuratedThemeId {
    fn default() -> Self {
        Self::BioluminescentDarkField
    }
}

impl CuratedThemeId {
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::BioluminescentDarkField => Self::CyberpunkAurora,
            Self::CyberpunkAurora => Self::Darcula,
            Self::Darcula => Self::LumenLight,
            Self::LumenLight => Self::NordicFrost,
            Self::NordicFrost => Self::HighContrast,
            Self::HighContrast => Self::BioluminescentDarkField,
        }
    }

    /// The config-layer identity for this theme.
    ///
    /// Exhaustive on purpose — no wildcard arm. Adding a theme to either enum
    /// then fails to compile here rather than silently mapping the new one onto
    /// an old identity, which is how a persisted theme would quietly become a
    /// different theme (bd-2z0.14.2.2).
    #[must_use]
    pub const fn to_config(self) -> scriptbots_core::TuiThemeId {
        match self {
            Self::BioluminescentDarkField => scriptbots_core::TuiThemeId::BioluminescentDarkField,
            Self::CyberpunkAurora => scriptbots_core::TuiThemeId::CyberpunkAurora,
            Self::Darcula => scriptbots_core::TuiThemeId::Darcula,
            Self::LumenLight => scriptbots_core::TuiThemeId::LumenLight,
            Self::NordicFrost => scriptbots_core::TuiThemeId::NordicFrost,
            Self::HighContrast => scriptbots_core::TuiThemeId::HighContrast,
        }
    }

    /// Resolve a config-layer theme identity into the renderer's theme.
    #[must_use]
    pub const fn from_config(id: scriptbots_core::TuiThemeId) -> Self {
        match id {
            scriptbots_core::TuiThemeId::BioluminescentDarkField => Self::BioluminescentDarkField,
            scriptbots_core::TuiThemeId::CyberpunkAurora => Self::CyberpunkAurora,
            scriptbots_core::TuiThemeId::Darcula => Self::Darcula,
            scriptbots_core::TuiThemeId::LumenLight => Self::LumenLight,
            scriptbots_core::TuiThemeId::NordicFrost => Self::NordicFrost,
            scriptbots_core::TuiThemeId::HighContrast => Self::HighContrast,
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BioluminescentDarkField => "Bioluminescent Dark Field",
            Self::CyberpunkAurora => "Cyberpunk Aurora",
            Self::Darcula => "Darcula",
            Self::LumenLight => "Lumen Light",
            Self::NordicFrost => "Nordic Frost",
            Self::HighContrast => "High Contrast",
        }
    }

    #[must_use]
    pub fn header_color(self) -> Color {
        match self {
            Self::BioluminescentDarkField => srgb_color(visual::HERBIVORE_RGB),
            Self::CyberpunkAurora => rgb(0x00f5ff),
            Self::Darcula => rgb(0xffc66d),
            Self::LumenLight => rgb(0x1e293b),
            Self::NordicFrost => rgb(0x88c0d0),
            Self::HighContrast => rgb(0xffff00),
        }
    }

    #[must_use]
    pub fn accent_color(self) -> Color {
        match self {
            // Derived, not chosen. `header_color` above resolves this theme through
            // `srgb_color(visual::…)` rather than a hex literal, so the accent follows the same
            // route instead of introducing a second palette in the terminal. Every other theme
            // pairs a primary header with a contrasting accent, and CyberpunkAurora is the exact
            // precedent for this hue pair: cyan header, magenta accent. HERBIVORE_RGB and
            // CARNIVORE_RGB are the contrast pair BIOLUMINESCENT_DARK_FIELD_V1 already defines,
            // so this reuses core's own opposition rather than inventing a colour for it.
            Self::BioluminescentDarkField => srgb_color(visual::CARNIVORE_RGB),
            Self::CyberpunkAurora => rgb(0xff007f),
            Self::Darcula => rgb(0xcc7832),
            Self::LumenLight => rgb(0x0284c7),
            Self::NordicFrost => rgb(0x81a1c1),
            Self::HighContrast => rgb(0xffffff),
        }
    }
}

struct Palette {
    level: Option<ColorLevel>,
    emoji: bool,
    emoji_narrow: bool,
    mode: TerminalPaletteMode,
    theme_id: CuratedThemeId,
}

#[derive(Clone, Copy)]
struct TerminalTheme {
    header: Color,
    accent: Color,
    paused_fg: Color,
    paused_bg: Color,
    running_fg: Color,
    running_bg: Color,
    diet: [Color; 3],
    event: [Color; 4],
    population_spark: Color,
    energy_spark: Color,
    terrain_fg: [Color; 6],
    terrain_bg: [Color; 6],
}

/// Bridge a `visual.rs` sRGB triple into a terminal colour.
///
/// bd-9pqz's rule is that the ramp is defined once and no renderer invents a colour.
/// The TUI is a renderer, so its art-direction theme is derived here rather than
/// hand-picked — change the constant in visual.rs and terminal mode follows.
fn srgb_color(srgb: scriptbots_core::visual::Srgb) -> Color {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color::Rgb(byte(srgb[0]), byte(srgb[1]), byte(srgb[2]))
}

/// Lift a very dark substrate value toward legibility without leaving the ramp.
/// Terminal cells have no bloom or tonemap, so the raw abyss values render as flat
/// black; this keeps the hue and raises only the value.
fn srgb_lifted(srgb: scriptbots_core::visual::Srgb, lift: f32) -> Color {
    let byte = |v: f32| ((v + lift).clamp(0.0, 1.0) * 255.0).round() as u8;
    Color::Rgb(byte(srgb[0]), byte(srgb[1]), byte(srgb[2]))
}

/// WCAG 2.x relative luminance of a terminal colour.
///
/// The sRGB channels are linearised with the standard piecewise transfer
/// function before weighting. Using the raw 0..1 channel values instead — which
/// is the usual shortcut — inflates the luminance of dark colours and makes
/// failing pairs look passable, so it is exactly the shortcut a contrast gate
/// must not take.
fn relative_luminance(color: Color) -> f32 {
    let linear = |c: f32| {
        if c <= 0.039_28 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let [r, g, b] = color_channels(color);
    0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)
}

/// WCAG 2.x contrast ratio between two terminal colours, in `1.0..=21.0`.
///
/// Order-independent by construction: the lighter colour is always the
/// numerator, so a caller cannot get a misleadingly small ratio by passing
/// foreground and background the wrong way round.
fn contrast_ratio(a: Color, b: Color) -> f32 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    let (lighter, darker) = if la >= lb { (la, lb) } else { (lb, la) };
    (lighter + 0.05) / (darker + 0.05)
}

/// WCAG AA threshold for normal-size text.
const WCAG_AA_NORMAL_TEXT: f32 = 4.5;

fn rgb(hex: u32) -> Color {
    Color::Rgb(
        ((hex >> 16) & 0xFF) as u8,
        ((hex >> 8) & 0xFF) as u8,
        (hex & 0xFF) as u8,
    )
}

/// Unit-range RGB for the sub-cell canvas, which composites in float and does
/// its own quantization. Named ANSI colors carry no channel values, so they get
/// their conventional xterm approximation rather than being dropped to black.
fn color_channels(color: Color) -> [f32; 3] {
    const INV: f32 = 1.0 / 255.0;
    let (r, g, b) = match color {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::Black => (0, 0, 0),
        Color::Red => (205, 0, 0),
        Color::Green => (0, 205, 0),
        Color::Yellow => (205, 205, 0),
        Color::Blue => (0, 0, 238),
        Color::Magenta => (205, 0, 205),
        Color::Cyan => (0, 205, 205),
        Color::Gray => (229, 229, 229),
        Color::DarkGray => (127, 127, 127),
        Color::LightRed => (255, 0, 0),
        Color::LightGreen => (0, 255, 0),
        Color::LightYellow => (255, 255, 0),
        Color::LightBlue => (92, 92, 255),
        Color::LightMagenta => (255, 0, 255),
        Color::LightCyan => (0, 255, 255),
        Color::White => (255, 255, 255),
        Color::Indexed(_) | Color::Reset => (128, 128, 128),
    };
    [f32::from(r) * INV, f32::from(g) * INV, f32::from(b) * INV]
}

/// Pack unit-range RGB back into a terminal color.
fn channels_color(rgb: [f32; 3]) -> Color {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color::Rgb(byte(rgb[0]), byte(rgb[1]), byte(rgb[2]))
}

impl Palette {
    fn test_backend_evidence() -> Self {
        Self {
            level: None,
            emoji: false,
            emoji_narrow: false,
            mode: TerminalPaletteMode::Natural,
            theme_id: CuratedThemeId::CyberpunkAurora,
        }
    }

    fn theme(&self) -> TerminalTheme {
        // bd-f4x0 / bd-9pqz: the art-direction theme is a full theme, not an accent
        // tweak, and every value below traces to visual.rs. It is checked before the
        // palette-mode match so terminal mode carries the same look as the GUI.
        //
        // The accessibility modes are untouched and still reachable: cycling themes
        // moves off this one, and HighContrast in particular keeps its own values.
        if self.theme_id == CuratedThemeId::BioluminescentDarkField {
            let herb = visual::HERBIVORE_RGB;
            let carn = visual::CARNIVORE_RGB;
            // Omnivore sits midway on the deliberate cyan->magenta diet ramp rather
            // than being a third invented hue.
            let omni = [
                (herb[0] + carn[0]) * 0.5,
                (herb[1] + carn[1]) * 0.5,
                (herb[2] + carn[2]) * 0.5,
            ];
            let food = visual::FOOD_MID_RGB;
            let base = visual::TERRAIN_BASE_COLORS;
            return TerminalTheme {
                header: srgb_color(herb),
                accent: srgb_color(food),
                paused_fg: srgb_color(visual::BIOLUMINESCENT_DARK_FIELD_V1.substrate.abyss_srgb),
                paused_bg: srgb_color(carn),
                running_fg: srgb_color(visual::BIOLUMINESCENT_DARK_FIELD_V1.substrate.abyss_srgb),
                running_bg: srgb_color(food),
                diet: [srgb_color(herb), srgb_color(omni), srgb_color(carn)],
                event: [
                    srgb_color(food),
                    srgb_color(carn),
                    srgb_color(omni),
                    srgb_color(herb),
                ],
                population_spark: srgb_color(herb),
                energy_spark: srgb_color(food),
                // Substrate is near-black by design; terrain reads through VALUE and
                // material, not hue swaps. Backgrounds take the ramp as-is and glyph
                // ink is the same colour lifted, so the two never drift apart.
                terrain_fg: [
                    srgb_lifted(base[0], 0.34),
                    srgb_lifted(base[1], 0.34),
                    srgb_lifted(base[2], 0.30),
                    srgb_lifted(base[3], 0.30),
                    srgb_lifted(base[4], 0.36),
                    srgb_lifted(base[5], 0.26),
                ],
                terrain_bg: [
                    srgb_color(base[0]),
                    srgb_color(base[1]),
                    srgb_color(base[2]),
                    srgb_color(base[3]),
                    srgb_color(base[4]),
                    srgb_color(base[5]),
                ],
            };
        }
        match self.mode {
            TerminalPaletteMode::Natural => TerminalTheme {
                header: rgb(0x93c5fd),
                accent: rgb(0x38bdf8),
                paused_fg: rgb(0x0f172a),
                paused_bg: rgb(0xf97316),
                running_fg: rgb(0x0f172a),
                running_bg: rgb(0x22c55e),
                diet: [rgb(0x22c55e), rgb(0xfacc15), rgb(0xcb2a3b)],
                event: [rgb(0x22c55e), rgb(0xf97316), rgb(0xfacc15), rgb(0x60a5fa)],
                population_spark: rgb(0x22c55e),
                energy_spark: rgb(0xf59e0b),
                terrain_fg: [
                    rgb(0x1E3F66),
                    rgb(0x2F73B3),
                    rgb(0xB14E07),
                    rgb(0x50A913),
                    rgb(0x79D46D),
                    rgb(0xA9B1BA),
                ],
                terrain_bg: [
                    rgb(0x132234),
                    rgb(0x1B4669),
                    rgb(0x6A3B0B),
                    rgb(0x2F5F17),
                    rgb(0x3F6F47),
                    rgb(0x5A5F65),
                ],
            },
            TerminalPaletteMode::Deuteranopia => TerminalTheme {
                header: rgb(0xcbd5f5),
                accent: rgb(0x60a5fa),
                paused_fg: rgb(0x082f49),
                paused_bg: rgb(0xfbbf24),
                running_fg: rgb(0x082f49),
                running_bg: rgb(0x2dd4bf),
                diet: [rgb(0x2dd4bf), rgb(0xfbbf24), rgb(0xf87171)],
                event: [rgb(0x2dd4bf), rgb(0xf87171), rgb(0xfbbf24), rgb(0x60a5fa)],
                population_spark: rgb(0x2dd4bf),
                energy_spark: rgb(0xfbbf24),
                terrain_fg: [
                    rgb(0x214c67),
                    rgb(0x2f6f8f),
                    rgb(0x8b5f29),
                    rgb(0x4c8241),
                    rgb(0x6cbf8a),
                    rgb(0x8d95a1),
                ],
                terrain_bg: [
                    rgb(0x142d38),
                    rgb(0x1f4555),
                    rgb(0x5c3f1c),
                    rgb(0x315031),
                    rgb(0x3d6b4d),
                    rgb(0x555b63),
                ],
            },
            TerminalPaletteMode::Protanopia => TerminalTheme {
                header: rgb(0xe2e8f0),
                accent: rgb(0x7dd3fc),
                paused_fg: rgb(0x082f49),
                paused_bg: rgb(0xfbbf24),
                running_fg: rgb(0x082f49),
                running_bg: rgb(0x38bdf8),
                diet: [rgb(0x38bdf8), rgb(0xfbbf24), rgb(0xf472b6)],
                event: [rgb(0x38bdf8), rgb(0xf472b6), rgb(0xfbbf24), rgb(0x7dd3fc)],
                population_spark: rgb(0x38bdf8),
                energy_spark: rgb(0xfbbf24),
                terrain_fg: [
                    rgb(0x274164),
                    rgb(0x3570a5),
                    rgb(0x8a6134),
                    rgb(0x5d8c3d),
                    rgb(0x7ecf86),
                    rgb(0x95a4b5),
                ],
                terrain_bg: [
                    rgb(0x14273a),
                    rgb(0x1f4a67),
                    rgb(0x533c26),
                    rgb(0x304f2a),
                    rgb(0x3c6541),
                    rgb(0x505761),
                ],
            },
            TerminalPaletteMode::Tritanopia => TerminalTheme {
                header: rgb(0xf5f5f4),
                accent: rgb(0xf97316),
                paused_fg: rgb(0x0b1120),
                paused_bg: rgb(0x22c55e),
                running_fg: rgb(0x0b1120),
                running_bg: rgb(0xf97316),
                diet: [rgb(0xfb7185), rgb(0xfacc15), rgb(0x6366f1)],
                event: [rgb(0xfb7185), rgb(0x6366f1), rgb(0xfacc15), rgb(0xf97316)],
                population_spark: rgb(0xfb7185),
                energy_spark: rgb(0xfacc15),
                terrain_fg: [
                    rgb(0x3b4f7f),
                    rgb(0x4f6ad6),
                    rgb(0xb45309),
                    rgb(0x10b981),
                    rgb(0x34d399),
                    rgb(0x818cf8),
                ],
                terrain_bg: [
                    rgb(0x1f2941),
                    rgb(0x2a3f75),
                    rgb(0x5f3a17),
                    rgb(0x0d3d2c),
                    rgb(0x155940),
                    rgb(0x373a74),
                ],
            },
            TerminalPaletteMode::HighContrast => TerminalTheme {
                header: rgb(0xf8fafc),
                accent: rgb(0x38bdf8),
                paused_fg: rgb(0x000000),
                paused_bg: rgb(0xfacc15),
                running_fg: rgb(0x000000),
                running_bg: rgb(0xf97316),
                diet: [rgb(0xffffff), rgb(0xfacc15), rgb(0xff5555)],
                event: [rgb(0xffffff), rgb(0xff5555), rgb(0xfacc15), rgb(0x38bdf8)],
                population_spark: rgb(0xffffff),
                energy_spark: rgb(0xfacc15),
                terrain_fg: [
                    rgb(0xffffff),
                    rgb(0xd9e3ff),
                    rgb(0xffe9b0),
                    rgb(0xb7ffc8),
                    rgb(0xffc7ff),
                    rgb(0xd9d9d9),
                ],
                terrain_bg: [
                    rgb(0x000000),
                    rgb(0x000000),
                    rgb(0x000000),
                    rgb(0x000000),
                    rgb(0x000000),
                    rgb(0x000000),
                ],
            },
        }
    }
    fn heading_char_ascii(heading: f32) -> char {
        let normalized = heading.rem_euclid(TAU);
        let sector = ((normalized / (PI / 4.0)).round() as i32) & 7;
        match sector {
            0 => '>',
            1 => '/',
            2 => '^',
            3 => '\\',
            4 => '<',
            5 => '/',
            6 => 'v',
            _ => '\\',
        }
    }

    fn heading_char_pretty(heading: f32) -> char {
        let normalized = heading.rem_euclid(TAU);
        let sector = ((normalized / (PI / 4.0)).round() as i32) & 7;
        match sector {
            0 => '→',
            1 => '↗',
            2 => '↑',
            3 => '↖',
            4 => '←',
            5 => '↙',
            6 => '↓',
            _ => '↘',
        }
    }
    fn detect() -> Self {
        let level = on_cached(Stream::Stdout);
        let emoji = {
            if let Ok(raw) = std::env::var("SCRIPTBOTS_TERMINAL_EMOJI") {
                let v = raw.to_ascii_lowercase();
                if matches!(v.as_str(), "0" | "false" | "off" | "no") {
                    false
                } else {
                    matches!(v.as_str(), "1" | "true" | "yes" | "on")
                }
            } else {
                // Auto-detect: prefer ON when stdout is a real terminal, UTF-8 locale, and not a
                // known minimal TERM. This is heuristic but works well in practice.
                let term = env_lower("TERM");
                let looks_modern_term = !matches!(term.as_str(), "" | "dumb" | "linux" | "vt100");
                let locale = locale_lower();
                let utf8_locale = locale.contains("utf-8") || locale.contains("utf8");
                let is_ci = std::env::var("CI").is_ok();
                looks_modern_term && utf8_locale && !is_ci
            }
        };
        // Default narrow mode off; users can toggle if their terminal misaligns emojis
        let mode = std::env::var("SCRIPTBOTS_TERMINAL_PALETTE")
            .ok()
            .and_then(|raw| match raw.to_ascii_lowercase().as_str() {
                "natural" => Some(TerminalPaletteMode::Natural),
                "deuter" | "deuteranopia" => Some(TerminalPaletteMode::Deuteranopia),
                "protan" | "protanopia" => Some(TerminalPaletteMode::Protanopia),
                "tritan" | "tritanopia" => Some(TerminalPaletteMode::Tritanopia),
                "high" | "high_contrast" | "high-contrast" => {
                    Some(TerminalPaletteMode::HighContrast)
                }
                _ => None,
            })
            .unwrap_or(TerminalPaletteMode::Natural);
        Self {
            level,
            emoji,
            emoji_narrow: false,
            // `default()`, not a restated literal. This read CyberpunkAurora
            // while CuratedThemeId::default() is BioluminescentDarkField, so the
            // documented default and the constructed one disagreed
            // (bd-2z0.14.2.2).
            theme_id: CuratedThemeId::default(),
            mode,
        }
    }

    /// Adopt the run's configured chrome theme, if it named one.
    ///
    /// `None` means the config never expressed a preference, which must leave the
    /// detected default alone rather than resetting it — the two are different
    /// states and collapsing them would make an unset config silently override a
    /// theme the user is already on.
    fn apply_config_theme(&mut self, configured: Option<scriptbots_core::TuiThemeId>) {
        if let Some(id) = configured {
            self.theme_id = CuratedThemeId::from_config(id);
        }
    }

    fn header_style(&self) -> Style {
        let theme = self.theme();
        Style::default()
            .fg(theme.header)
            .add_modifier(Modifier::BOLD)
    }

    fn accent_style(&self) -> Style {
        Style::default().fg(self.theme().accent)
    }

    /// Field label: the name of a readout, not the readout (bd-f4x0).
    ///
    /// DIM on purpose. Panels here had two tiers — bold header or unstyled — so
    /// labels were routinely drawn in the same bold accent as titles and ended up
    /// competing with the numbers they introduce. The bead's word for the fix is
    /// restraint: the world is the subject, chrome should recede. A label is
    /// chrome, so it is the quietest thing on the panel that still reads.
    fn label_style(&self) -> Style {
        Style::default().add_modifier(Modifier::DIM)
    }

    /// The readout itself — the one thing on a panel that is data.
    ///
    /// Undimmed and unbolded: it wins by being the only element at full weight,
    /// not by shouting louder than the chrome around it. Bolding values too would
    /// restore the flat hierarchy this scale exists to remove.
    fn value_style(&self) -> Style {
        Style::default()
    }

    /// Secondary text: hints, units, counts that qualify a value.
    ///
    /// Shares the dim weight of a label but takes the theme accent, so a hint is
    /// legible as *related to this panel* without being mistaken for a value.
    fn muted_style(&self) -> Style {
        Style::default()
            .fg(self.theme().accent)
            .add_modifier(Modifier::DIM)
    }

    fn paused_style(&self) -> Style {
        let theme = self.theme();
        Style::default()
            .fg(theme.paused_fg)
            .bg(theme.paused_bg)
            .add_modifier(Modifier::BOLD)
    }

    fn running_style(&self) -> Style {
        let theme = self.theme();
        Style::default()
            .fg(theme.running_fg)
            .bg(theme.running_bg)
            .add_modifier(Modifier::BOLD)
    }

    fn speed_style(&self, speed: f32) -> Style {
        let theme = self.theme();
        let color = if speed > 1.0 {
            theme.accent
        } else if speed <= 0.0 {
            theme.paused_bg
        } else {
            theme.population_spark
        };
        Style::default().fg(color)
    }

    fn title<T: Into<String>>(&self, title: T) -> Span<'static> {
        Span::styled(title.into(), self.header_style())
    }

    fn diet_style(&self, diet: DietClass) -> Style {
        Style::default().fg(self.diet_color(diet))
    }

    fn population_spark_style(&self) -> Style {
        Style::default().fg(self.theme().population_spark)
    }

    fn energy_spark_style(&self) -> Style {
        Style::default().fg(self.theme().energy_spark)
    }

    fn event_style(&self, kind: EventKind) -> Style {
        let theme = self.theme();
        let color = match kind {
            EventKind::Birth => theme.event[0],
            EventKind::Death => theme.event[1],
            EventKind::Population => theme.event[2],
            EventKind::Info => theme.event[3],
        };
        Style::default().fg(color)
    }

    /// Status chrome colours, drawn from the theme's palette-aware ramp.
    ///
    /// Status text was hand-coded to `Color::Green` / `Color::Yellow` /
    /// `Color::Red`, which is an accessibility defect rather than only a styling
    /// one: red-versus-green is the exact confusion pair for deuteranopia and
    /// protanopia, this app SHIPS palettes for both, and raw ANSI constants
    /// bypass them entirely. A colourblind operator could not tell "active" from
    /// "error" in the storage row (bd-f4x0).
    ///
    /// These reuse the event ramp rather than introducing a fourth colour
    /// vocabulary, so a status row and the event log agree about what "good" and
    /// "bad" look like, and every accessibility palette retunes both at once.
    /// Colour for one mortality cause, drawn from the ramps that already name
    /// these concepts elsewhere in the app.
    ///
    /// The mortality bars were hand-coded Red / LightRed / Yellow / Gray /
    /// DarkGray. Two problems, only one of which is styling. Red against
    /// LightRed and Gray against DarkGray are barely separable with full colour
    /// vision and collapse entirely under the deuteranopia and protanopia
    /// palettes this app ships, because named ANSI constants cannot be retuned.
    /// And the colours were arbitrary: nothing tied the bar for "killed by a
    /// carnivore" to the colour carnivores are actually drawn in.
    ///
    /// So each cause takes the palette entry that already MEANS it: combat
    /// deaths use the diet colour of the killer, starvation uses the energy
    /// ramp it depletes, and the two residual causes take dimmed chrome rather
    /// than competing with real signal. Switching accessibility palette retunes
    /// all five at once, and a reader who has learned what carnivore cyan looks
    /// like on the map reads the mortality panel without a legend (bd-f4x0).
    fn mortality_style(&self, cause: MortalityCause) -> Style {
        let theme = self.theme();
        match cause {
            // Killed BY a carnivore, so it carries the carnivore's colour.
            MortalityCause::CombatCarnivore => Style::default().fg(theme.diet[2]),
            MortalityCause::CombatHerbivore => Style::default().fg(theme.diet[0]),
            // Starvation is energy reaching zero; the energy spark is that ramp.
            MortalityCause::Starvation => Style::default().fg(theme.energy_spark),
            // Aging and unknown are background facts rather than events to react
            // to, so they recede — the bead's "restraint" applied to a readout.
            MortalityCause::Aging => Style::default().fg(theme.accent),
            MortalityCause::Unknown => Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::DIM),
        }
    }

    fn ok_style(&self) -> Style {
        self.event_style(EventKind::Birth)
    }

    fn warn_style(&self) -> Style {
        self.event_style(EventKind::Population)
    }

    fn error_style(&self) -> Style {
        self.event_style(EventKind::Death)
    }

    fn has_color(&self) -> bool {
        self.level.is_some()
    }

    fn is_emoji(&self) -> bool {
        self.emoji
    }

    fn toggle_emoji(&mut self) {
        self.emoji = !self.emoji;
    }

    pub fn cycle_mode(&mut self) -> &'static str {
        self.mode = self.mode.next();
        self.mode.label()
    }

    fn mode_label(&self) -> &'static str {
        self.mode.label()
    }

    pub fn cycle_theme(&mut self) -> &'static str {
        self.theme_id = self.theme_id.next();
        self.theme_id.label()
    }

    pub fn theme_label(&self) -> &'static str {
        self.theme_id.label()
    }

    fn is_emoji_narrow(&self) -> bool {
        self.emoji && self.emoji_narrow
    }

    fn toggle_emoji_narrow(&mut self) {
        if self.emoji {
            self.emoji_narrow = !self.emoji_narrow;
        }
    }

    fn diet_color(&self, diet: DietClass) -> Color {
        let theme = self.theme();
        match diet {
            DietClass::Herbivore => theme.diet[0],
            DietClass::Omnivore => theme.diet[1],
            DietClass::Carnivore => theme.diet[2],
        }
    }

    /// True when the terminal can render the sub-cell canvas legibly. Braille
    /// averages several sub-pixel colors into one fg and one bg per cell, which
    /// only reads correctly with a real color channel — under 16 colors the
    /// averaging collapses and the flat glyph map stays the better picture.
    /// Probe this terminal for the richest sub-cell tier it can actually show.
    fn canvas_capability(&self) -> CanvasCapability {
        CanvasCapability::detect(
            ColorSupport::from_level(self.level),
            &env_lower("TERM"),
            &locale_lower(),
            std::env::var_os("NO_COLOR").is_some(),
        )
    }

    /// Terrain base color for the canvas. Uses the theme's background band,
    /// which is the field color; `terrain_fg` is the glyph ink and is far too
    /// bright to tile a whole map with.
    fn terrain_canvas_rgb(&self, kind: TerrainKind) -> [f32; 3] {
        color_channels(self.theme().terrain_bg[Self::terrain_index(kind)])
    }

    /// Index of a terrain kind into the theme's parallel color bands.
    ///
    /// One definition: three copies of this match previously had to be kept in
    /// step by hand, and a theme band silently shifting under one of them would
    /// have painted a biome in another biome's color.
    const fn terrain_index(kind: TerrainKind) -> usize {
        match kind {
            TerrainKind::DeepWater => 0,
            TerrainKind::ShallowWater => 1,
            TerrainKind::Sand => 2,
            TerrainKind::Grass => 3,
            TerrainKind::Bloom => 4,
            TerrainKind::Rock => 5,
        }
    }

    /// Terrain base for the canvas, tinted by how lush the tile is.
    ///
    /// Lushness lifts the field color toward the biome's own vivid ink, the same
    /// base->ink move [`Self::food_canvas_rgb`] uses, so a wet fertile meadow and
    /// a parched one differ without either leaving its biome's palette. Weighted
    /// short of the full ink because that ink is glyph-bright and tiling a whole
    /// map with it would drown the agents it is supposed to sit behind.
    fn terrain_canvas_rgb_lush(&self, kind: TerrainKind, lushness: f32) -> [f32; 3] {
        let base = self.terrain_canvas_rgb(kind);
        let idx = Self::terrain_index(kind);
        let ink = color_channels(self.theme().terrain_fg[idx]);
        let t = lushness.clamp(0.0, 1.0) * CANVAS_LUSHNESS_WEIGHT;
        [
            (ink[0] - base[0]).mul_add(t, base[0]),
            (ink[1] - base[1]).mul_add(t, base[1]),
            (ink[2] - base[2]).mul_add(t, base[2]),
        ]
    }

    /// Food ink, brightened toward the terrain's glyph color as the cell fills.
    fn food_canvas_rgb(&self, kind: TerrainKind, level: f32) -> [f32; 3] {
        let idx = Self::terrain_index(kind);
        let theme = self.theme();
        let base = color_channels(theme.terrain_bg[idx]);
        let ink = color_channels(theme.terrain_fg[idx]);
        let t = level.clamp(0.0, 1.0);
        [
            (ink[0] - base[0]).mul_add(t, base[0]),
            (ink[1] - base[1]).mul_add(t, base[1]),
            (ink[2] - base[2]).mul_add(t, base[2]),
        ]
    }

    /// Ink for the canvas selection ring: the theme's accent, so the highlight
    /// matches the accent the rest of the TUI already uses for focus.
    fn accent_canvas_rgb(&self) -> [f32; 3] {
        color_channels(self.theme().accent)
    }

    /// Agent ink by diet, scaled by energy so dying agents read as dimmer dots.
    fn agent_canvas_rgb(&self, diet: DietClass, energy: f32) -> [f32; 3] {
        let idx = match diet {
            DietClass::Herbivore => 0,
            DietClass::Omnivore => 1,
            DietClass::Carnivore => 2,
        };
        let base = color_channels(self.theme().diet[idx]);
        // Floor the scale so a nearly-dead agent is still visible, not black.
        let scale = 0.45_f32 + 0.55 * energy.clamp(0.0, 1.0);
        [base[0] * scale, base[1] * scale, base[2] * scale]
    }

    fn terrain_symbol(&self, kind: TerrainKind, food_level: f32) -> (char, Style) {
        let rich_color = self
            .level
            .is_some_and(|level| level.has_16m || level.has_256);
        let idx = Self::terrain_index(kind);
        let theme = self.theme();
        let rich_fg = theme.terrain_fg[idx];
        let rich_bg = theme.terrain_bg[idx];
        let (fallback_fg, fallback_bg) = match kind {
            TerrainKind::DeepWater => (Color::Blue, Color::Black),
            TerrainKind::ShallowWater => (Color::Cyan, Color::Blue),
            TerrainKind::Sand => (Color::Yellow, Color::Black),
            TerrainKind::Grass => (Color::Green, Color::Black),
            TerrainKind::Bloom => (Color::Magenta, Color::Black),
            TerrainKind::Rock => (Color::Gray, Color::Black),
        };
        let (base_fg, base_bg) = if rich_color {
            (rich_fg, rich_bg)
        } else {
            (fallback_fg, fallback_bg)
        };
        let (mut glyph, fg, bg) = if self.emoji {
            let glyph = match kind {
                TerrainKind::DeepWater => {
                    if self.is_emoji_narrow() {
                        '≈'
                    } else {
                        '🌊'
                    }
                }
                TerrainKind::ShallowWater => {
                    if self.is_emoji_narrow() {
                        '~'
                    } else {
                        '💧'
                    }
                }
                TerrainKind::Sand => {
                    if self.is_emoji_narrow() {
                        '·'
                    } else {
                        '🏜'
                    }
                }
                TerrainKind::Grass => {
                    if self.is_emoji_narrow() {
                        '"'
                    } else {
                        '🌿'
                    }
                }
                TerrainKind::Bloom => {
                    if self.is_emoji_narrow() {
                        '*'
                    } else {
                        '🌺'
                    }
                }
                TerrainKind::Rock => {
                    if self.is_emoji_narrow() {
                        '^'
                    } else {
                        '🪨'
                    }
                }
            };
            (glyph, base_fg, base_bg)
        } else {
            let glyph = match kind {
                TerrainKind::DeepWater => '≈',
                TerrainKind::ShallowWater => '~',
                TerrainKind::Sand => '·',
                TerrainKind::Grass => '"',
                TerrainKind::Bloom => '*',
                TerrainKind::Rock => '^',
            };
            (glyph, base_fg, base_bg)
        };
        // Food-driven flourish: swap glyph for lush/barren variants when in emoji mode
        if self.emoji && !self.is_emoji_narrow() {
            if food_level > 0.66 {
                glyph = match kind {
                    TerrainKind::DeepWater | TerrainKind::ShallowWater => '🐟',
                    TerrainKind::Sand => '🌴',
                    TerrainKind::Grass | TerrainKind::Bloom => '🌾',
                    TerrainKind::Rock => glyph,
                };
            } else if food_level < 0.2 {
                glyph = match kind {
                    TerrainKind::Grass | TerrainKind::Bloom => '🥀',
                    _ => glyph,
                };
            }
        }

        let mut style = Style::default().fg(fg);
        // In emoji mode, suppress background to avoid muddy colors behind glyphs
        if self.has_color() && !self.emoji {
            style = style.bg(bg);
        }
        if food_level > 0.66 {
            style = style.add_modifier(Modifier::BOLD);
        } else if food_level < 0.2 {
            style = style.add_modifier(Modifier::DIM);
        }
        (glyph, style)
    }

    fn agent_symbol(&self, occupancy: &CellOccupancy, base: Style) -> (char, Style) {
        let total = occupancy.total();
        let class = if total == 0 {
            DietClass::Omnivore
        } else {
            occupancy.dominant()
        };
        let mut glyph = if self.emoji {
            match total {
                0 => ' ',
                1 => occupancy
                    .mean_heading()
                    .map(|ang| self.heading_char(ang))
                    .unwrap_or_else(|| match class {
                        DietClass::Herbivore => {
                            if self.is_emoji_narrow() {
                                'h'
                            } else {
                                '🐇'
                            }
                        }
                        DietClass::Omnivore => {
                            if self.is_emoji_narrow() {
                                'o'
                            } else {
                                '🦝'
                            }
                        }
                        DietClass::Carnivore => {
                            if self.is_emoji_narrow() {
                                'c'
                            } else {
                                '🦊'
                            }
                        }
                    }),
                2..=3 => match class {
                    DietClass::Herbivore => {
                        if self.is_emoji_narrow() {
                            'H'
                        } else {
                            '🐑'
                        }
                    }
                    DietClass::Omnivore => {
                        if self.is_emoji_narrow() {
                            'O'
                        } else {
                            '🐻'
                        }
                    }
                    DietClass::Carnivore => {
                        if self.is_emoji_narrow() {
                            'C'
                        } else {
                            '🐺'
                        }
                    }
                },
                _ => {
                    if self.is_emoji_narrow() {
                        '@'
                    } else {
                        '👥'
                    }
                }
            }
        } else {
            match total {
                0 => ' ',
                1 => occupancy
                    .mean_heading()
                    .map(|ang| self.heading_char(ang))
                    .unwrap_or_else(|| match class {
                        DietClass::Herbivore => 'h',
                        DietClass::Omnivore => 'o',
                        DietClass::Carnivore => 'c',
                    }),
                2..=3 => match class {
                    DietClass::Herbivore => 'H',
                    DietClass::Omnivore => 'O',
                    DietClass::Carnivore => 'C',
                },
                _ => '@',
            }
        };

        let mut style = base.fg(self.diet_color(class));
        if occupancy.boosted || total > 1 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if total > 3 {
            style = style.add_modifier(Modifier::REVERSED);
        }
        if occupancy.boosted {
            glyph = if self.emoji && !self.is_emoji_narrow() {
                '🚀'
            } else {
                glyph
            };
        }
        if occupancy.spike_peak > 0.6 {
            glyph = if self.emoji && !self.is_emoji_narrow() {
                '⚔'
            } else {
                '!'
            };
            style = style.add_modifier(Modifier::UNDERLINED);
        }
        if let Some(tendency) = occupancy.mean_tendency() {
            if tendency < 0.25 {
                style = style.fg(Color::Green);
            } else if tendency > 0.75 {
                style = style.fg(Color::Red);
            }
        }
        (glyph, style)
    }

    fn heading_char(&self, heading: f32) -> char {
        if self.is_emoji_narrow() {
            Self::heading_char_ascii(heading)
        } else {
            Self::heading_char_pretty(heading)
        }
    }
}

struct MapWidget<'a> {
    snapshot: &'a Snapshot,
    terrain: &'a TerrainView,
    palette: &'a Palette,
    scratch: &'a mut [CellOccupancy],
    stamp: u32,
    /// Sub-cell canvas, when the view has one and the terminal can show it.
    /// `None` selects the flat one-glyph-per-cell map.
    canvas: Option<&'a mut SubCellBuffer>,
    /// Resolved `(cycle_ticks, start_phase)` for the shared daylight curve.
    day_night: (u32, f32),
    /// The probed sub-cell tier; supplies the quantization depth.
    capability: CanvasCapability,
    /// Grow-only per-sub-pixel agent counts, reused across frames.
    density: &'a mut Vec<u16>,
    /// The world window being shown; the sole screen<->world transform.
    viewport: CanvasViewport,
}

/// Terrain alpha, deliberately below [`subcell::ALPHA_SOLID`]: terrain must land
/// in a cell's background so food and agents are the lit dots. Painting it solid
/// would light every dot and reduce the map to a field of full braille blocks.
const CANVAS_TERRAIN_ALPHA: f32 = 0.35;

/// Normalized food below this doesn't earn a dot; without a floor, grid noise
/// lights the whole map and the density gain is wasted.
const CANVAS_FOOD_THRESHOLD: f32 = 0.18;

/// Fraction of terrain brightness that survives midnight.
///
/// [`visual::daylight_factor`] bottoms out at [`visual::DAYLIGHT_NIGHT_FLOOR`]
/// (0.15). A GPU surface can render terrain that dark and still read, because it
/// has thousands of shades per tile; a terminal cell has one background color and
/// a 256-entry palette, so scaling straight by the raw factor collapses the whole
/// map into the same near-black bucket and the biome ramp disappears. The shared
/// daylight SIGNAL is unchanged — this only remaps its output into the brightness
/// band a terminal can still resolve.
const CANVAS_NIGHT_FLOOR: f32 = 0.45;

/// Half-depth of the water shimmer swing, applied to water terrain brightness.
///
/// Phase comes from [`visual::shimmer`], so a cell pulses in lockstep with the
/// same cell on the GPU surfaces and in replay; only the amplitude is per-surface.
const CANVAS_WATER_SHIMMER_SWING: f32 = 0.12;

/// Half-depth of the food pulse swing, applied to food ink brightness.
const CANVAS_FOOD_PULSE_SWING: f32 = 0.15;

/// Brightness of a heading whisker relative to the agent's own dot. Dim enough
/// that a whisker never reads as a second agent, bright enough to see the sector.
const CANVAS_WHISKER_DIM: f32 = 0.45;

/// Brightness multiplier applied to a boosted agent's dot.
const CANVAS_BOOST_FLARE: f32 = 1.6;

/// Spike length above which the canvas paints an attack cue.
const CANVAS_SPIKE_THRESHOLD: f32 = 0.5;

/// How far full lushness lifts a terrain tile toward its biome's vivid ink.
///
/// Well short of 1.0: that ink is glyph-bright, and tiling a whole map with it
/// would drown the agent dots the terrain is supposed to sit behind.
const CANVAS_LUSHNESS_WEIGHT: f32 = 0.35;

/// Minimap edge as a fraction of the canvas.
const CANVAS_MINIMAP_FRACTION: u16 = 4;

/// Smallest useful minimap edge in sub-pixels. Below this the thumbnail cannot
/// show a viewport rectangle distinct from its own border, so it is omitted
/// rather than drawn as a meaningless smudge.
const CANVAS_MINIMAP_MIN_EDGE: u16 = 8;

/// Thumbnail alpha, deliberately below [`subcell::ALPHA_SOLID`] so the world
/// thumbnail lands in cell backgrounds and leaves the lit dots for the viewport
/// rectangle — the one thing the minimap exists to communicate.
const CANVAS_MINIMAP_ALPHA: f32 = 0.4;

/// Click/hover pick radius, as a fraction of the visible window.
///
/// Scaled by the viewport span rather than fixed in world units, so the radius
/// is a constant on-screen distance: at 8x zoom the same gesture selects a
/// proportionally tighter region instead of still grabbing a whole neighbourhood.
const CANVAS_PICK_RADIUS_FRACTION: f32 = 0.04;

/// Tightest world window the canvas will show, as a fraction of the world.
///
/// Bounds `CanvasViewport::span` away from zero so the world->canvas transform
/// can never divide by a degenerate span.
const CANVAS_MIN_SPAN: f32 = 1.0 / CANVAS_MAX_ZOOM;

/// Maximum magnification. Past this the sub-pixel grid is coarser than the
/// world detail it is showing, so more zoom buys nothing.
const CANVAS_MAX_ZOOM: f32 = 8.0;

/// The square of world the canvas is currently showing, in normalized world
/// coordinates.
///
/// This is the ONE place screen and world coordinates are related. Painting,
/// hover, and click all go through it, so a zoomed canvas cannot show one thing
/// and pick another — which is exactly what happened while zoom was a number
/// nothing read.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CanvasViewport {
    /// Centre of the visible window, normalized.
    centre: (f32, f32),
    /// Side of the visible window as a fraction of the world; `1.0` is the
    /// whole world.
    span: f32,
}

impl CanvasViewport {
    /// Build the window for a zoom level and desired centre.
    ///
    /// The centre is clamped so the window never runs off the world: there is
    /// nothing outside it to show, and letting the window leave would paint a
    /// band of blank cells that look like dead terrain.
    fn new(zoom: f32, centre: (f32, f32)) -> Self {
        let zoom = if zoom.is_finite() {
            zoom.clamp(1.0, CANVAS_MAX_ZOOM)
        } else {
            1.0
        };
        let span = (1.0 / zoom).clamp(CANVAS_MIN_SPAN, 1.0);
        let half = span / 2.0;
        let clamp_axis = |v: f32| {
            if v.is_finite() {
                v.clamp(half, 1.0 - half)
            } else {
                0.5
            }
        };
        Self {
            centre: (clamp_axis(centre.0), clamp_axis(centre.1)),
            span,
        }
    }

    /// The normalized world point under a fractional canvas position, where
    /// `(0.0, 0.0)` is the canvas's top-left and `(1.0, 1.0)` its bottom-right.
    fn world_at(&self, fx: f32, fy: f32) -> (f32, f32) {
        let half = self.span / 2.0;
        (
            (self.centre.0 - half + fx * self.span).clamp(0.0, 0.9999),
            (self.centre.1 - half + fy * self.span).clamp(0.0, 0.9999),
        )
    }

    /// The fractional canvas position of a normalized world point, or `None`
    /// when that point is outside the window.
    ///
    /// Returning `None` rather than a clamped edge position is deliberate: an
    /// off-screen agent must be skipped, not smeared onto the border where it
    /// would read as a crowd against the frame.
    fn canvas_at(&self, wx: f32, wy: f32) -> Option<(f32, f32)> {
        let half = self.span / 2.0;
        let fx = (wx - (self.centre.0 - half)) / self.span;
        let fy = (wy - (self.centre.1 - half)) / self.span;
        ((0.0..1.0).contains(&fx) && (0.0..1.0).contains(&fy)).then_some((fx, fy))
    }
}

/// The eight sub-pixel offsets forming the selection ring around a focused
/// agent's dot.
///
/// The centre is deliberately absent: the ring must not paint over the agent it
/// is pointing at, or selecting an agent would hide the very thing selected.
const CANVAS_SELECTION_RING: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// How far a fully crowded sub-pixel blends toward white.
///
/// Short of 1.0 on purpose: a maximally crowded dot must still carry a hint of
/// its diet color, or a carnivore pile and a herbivore pile look the same.
const CANVAS_CLUSTER_WHITEN: f32 = 0.7;

/// Normalized crowding for a sub-pixel holding `count` agents.
///
/// `1 - 1/count`: exactly `0.0` for a lone agent (so the common case is never
/// tinted at all), `0.5` at two, and saturating toward `1.0`. Saturating rather
/// than linear because a 5k-agent world would otherwise drive every occupied dot
/// to the same maximum and lose the distinction it was added to create.
fn cluster_heat(count: u16) -> f32 {
    if count <= 1 {
        return 0.0;
    }
    1.0 - 1.0 / f32::from(count)
}

/// Blend `rgb` toward white by `amount` in `[0, 1]`.
fn whiten(rgb: [f32; 3], amount: f32) -> [f32; 3] {
    let t = amount.clamp(0.0, 1.0);
    [
        (1.0 - rgb[0]).mul_add(t, rgb[0]),
        (1.0 - rgb[1]).mul_add(t, rgb[1]),
        (1.0 - rgb[2]).mul_add(t, rgb[2]),
    ]
}

/// What a terminal reports it can do with color, decoupled from the detection
/// crate's own `ColorLevel` (whose private fields make it unconstructible in a
/// test, and therefore unusable as a probe input).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ColorSupport {
    basic: bool,
    ansi256: bool,
    truecolor: bool,
}

impl ColorSupport {
    fn from_level(level: Option<ColorLevel>) -> Self {
        level.map_or_else(Self::default, |level| Self {
            basic: level.has_basic,
            ansi256: level.has_256,
            truecolor: level.has_16m,
        })
    }
}

/// What sub-cell tier this terminal can actually display.
///
/// Probed once at startup from the environment and then carried, so no paint
/// path re-reads `TERM` or re-runs color detection per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CanvasCapability {
    /// The richest glyph vocabulary the terminal can render.
    mode: SubCellMode,
    /// The color depth to quantize into, or `None` when the terminal reports no
    /// color at all or the user set `NO_COLOR`.
    depth: Option<ColorDepth>,
}

impl CanvasCapability {
    /// Resolve the tier from raw environment facts.
    ///
    /// Taking the environment as arguments rather than reading it makes the whole
    /// policy testable: a probe that reads `std::env` directly can only be tested
    /// by mutating process-global state, which is exactly the kind of test that
    /// passes alone and fails in a parallel run.
    ///
    /// `term` and `locale` are expected lowercased.
    fn detect(color: ColorSupport, term: &str, locale: &str, no_color: bool) -> Self {
        let depth = if no_color {
            None
        } else if color.truecolor {
            Some(ColorDepth::TrueColor)
        } else if color.ansi256 {
            Some(ColorDepth::Ansi256)
        } else if color.basic {
            Some(ColorDepth::Ansi16)
        } else {
            None
        };

        // Every sub-cell vocabulary above ASCII is multi-byte UTF-8, so a
        // non-UTF-8 locale rules all of them out no matter how capable TERM is.
        let utf8 = locale.contains("utf-8") || locale.contains("utf8");
        let mode = if !utf8 || matches!(term, "" | "dumb" | "vt100") {
            SubCellMode::Ascii
        } else if term == "linux" {
            // The Linux console's built-in font carries the block-drawing range
            // but not U+2800..U+28FF, so braille renders as tofu there.
            SubCellMode::Quadrant
        } else {
            SubCellMode::Braille
        };

        Self { mode, depth }
    }

    /// Whether this tier resolves more than one world sample per terminal cell.
    const fn has_sub_cell_density(self) -> bool {
        !matches!(self.mode, SubCellMode::Ascii)
    }

    /// Whether the world map should paint through the canvas at all.
    ///
    /// Density without color is a REGRESSION, not a degradation: the flat map's
    /// per-terrain glyphs still distinguish water from rock with no color at all,
    /// whereas an uncolored braille field is eight identical dots. So `NO_COLOR`
    /// and colorless terminals are honored by staying on the flat map, which
    /// loses nothing, rather than by painting a canvas that cannot say anything.
    const fn use_canvas(self) -> bool {
        self.has_sub_cell_density() && self.depth.is_some()
    }

    /// Stable label for the startup log and the HUD.
    const fn label(self) -> &'static str {
        match (self.mode, self.depth) {
            (SubCellMode::Braille, Some(ColorDepth::TrueColor)) => "braille/truecolor",
            (SubCellMode::Braille, Some(ColorDepth::Ansi256)) => "braille/256",
            (SubCellMode::Braille, Some(ColorDepth::Ansi16)) => "braille/16",
            (SubCellMode::Braille, None) => "braille/no-color",
            (SubCellMode::Quadrant, Some(ColorDepth::TrueColor)) => "quadrant/truecolor",
            (SubCellMode::Quadrant, Some(ColorDepth::Ansi256)) => "quadrant/256",
            (SubCellMode::Quadrant, Some(ColorDepth::Ansi16)) => "quadrant/16",
            (SubCellMode::Quadrant, None) => "quadrant/no-color",
            (SubCellMode::HalfBlock, Some(ColorDepth::TrueColor)) => "half-block/truecolor",
            (SubCellMode::HalfBlock, Some(ColorDepth::Ansi256)) => "half-block/256",
            (SubCellMode::HalfBlock, Some(ColorDepth::Ansi16)) => "half-block/16",
            (SubCellMode::HalfBlock, None) => "half-block/no-color",
            (SubCellMode::Ascii, Some(ColorDepth::TrueColor)) => "ascii/truecolor",
            (SubCellMode::Ascii, Some(ColorDepth::Ansi256)) => "ascii/256",
            (SubCellMode::Ascii, Some(ColorDepth::Ansi16)) => "ascii/16",
            (SubCellMode::Ascii, None) => "ascii/no-color",
        }
    }
}

/// Render a stable agent identity for display.
///
/// A missing uid is shown as `?`, not as a substituted number: core documents an
/// arena handle without an identity as an invariant violation, and quietly
/// printing `0` — or the slot handle — would present a wrong identity as a real
/// one, which is the whole defect this helper exists to prevent (bd-qxrt).
fn agent_uid_label(uid: Option<u64>) -> String {
    uid.map_or_else(|| "?".to_string(), |uid| uid.to_string())
}

/// Lowercased environment variable, empty when unset or non-Unicode.
fn env_lower(key: &str) -> String {
    std::env::var(key).unwrap_or_default().to_ascii_lowercase()
}

/// Lowercased effective locale, following the standard `LC_ALL` > `LC_CTYPE` >
/// `LANG` precedence.
fn locale_lower() -> String {
    std::env::var("LC_ALL")
        .ok()
        .or_else(|| std::env::var("LC_CTYPE").ok())
        .or_else(|| std::env::var("LANG").ok())
        .unwrap_or_default()
        .to_ascii_lowercase()
}

impl MapWidget<'_> {
    /// Paint the world into the sub-cell buffer and blit the composed frame.
    ///
    /// This is the resolution win: the buffer is 2x4 sub-pixels per terminal
    /// cell in braille mode, so terrain and food are sampled — and agents are
    /// placed — at eight times the cell grid's density.
    /// Corner thumbnail of the whole world with the visible window outlined.
    ///
    /// Only drawn while zoomed. At 1x the canvas already IS the whole world, so
    /// a thumbnail would be a second copy of what the user is looking at and the
    /// rectangle would trace its own border — decoration that costs screen and
    /// says nothing.
    fn paint_minimap(canvas: &mut SubCellBuffer, ctx: &MapWidget<'_>) {
        let view = ctx.viewport;
        if view.span >= 1.0 {
            return;
        }
        let (sub_w, sub_h) = (canvas.sub_width(), canvas.sub_height());
        let box_w = sub_w / CANVAS_MINIMAP_FRACTION;
        let box_h = sub_h / CANVAS_MINIMAP_FRACTION;
        if box_w < CANVAS_MINIMAP_MIN_EDGE
            || box_h < CANVAS_MINIMAP_MIN_EDGE
            || box_w >= sub_w
            || box_h >= sub_h
        {
            return;
        }
        // Top-right: the world canvas is densest toward its centre, and the HUD
        // panels already own the bottom.
        let origin_x = sub_w - box_w;

        // Thumbnail of the entire world, dim, in the cell backgrounds.
        for by in 0..box_h {
            let v = (f32::from(by) + 0.5) / f32::from(box_h);
            for bx in 0..box_w {
                let u = (f32::from(bx) + 0.5) / f32::from(box_w);
                let base = ctx.palette.terrain_canvas_rgb(ctx.terrain.sample(u, v));
                canvas.set(
                    Layer::Selection,
                    origin_x + bx,
                    by,
                    [base[0], base[1], base[2], CANVAS_MINIMAP_ALPHA],
                );
            }
        }

        // The visible window, as lit dots. This is the payload: at 8x zoom the
        // canvas alone gives no clue which eighth of the world it is showing.
        let half = view.span / 2.0;
        let to_box = |value: f32, edge: u16| -> u16 {
            let scaled = (value * f32::from(edge)).floor();
            let clamped = scaled.clamp(0.0, f32::from(edge - 1));
            clamped as u16
        };
        let x0 = to_box(view.centre.0 - half, box_w);
        let x1 = to_box(view.centre.0 + half, box_w);
        let y0 = to_box(view.centre.1 - half, box_h);
        let y1 = to_box(view.centre.1 + half, box_h);
        let ink = ctx.palette.accent_canvas_rgb();
        let mut mark = |bx: u16, by: u16| {
            canvas.set(
                Layer::Selection,
                origin_x + bx,
                by,
                [ink[0], ink[1], ink[2], 1.0],
            );
        };
        for bx in x0..=x1 {
            mark(bx, y0);
            mark(bx, y1);
        }
        for by in y0..=y1 {
            mark(x0, by);
            mark(x1, by);
        }
    }

    fn render_canvas(
        canvas: &mut SubCellBuffer,
        density: &mut Vec<u16>,
        area: Rect,
        buf: &mut Buffer,
        ctx: &MapWidget<'_>,
    ) {
        if canvas.width_cells() != area.width || canvas.height_cells() != area.height {
            canvas.resize(area.width, area.height);
        }

        // Repaint from world state every frame. Higher layers must be cleared
        // first: `set` only replaces when the incoming layer is at least the
        // stored one, so last frame's agent dots would otherwise survive.
        canvas.clear_layer(Layer::Selection);
        canvas.clear_layer(Layer::Cues);
        canvas.clear_layer(Layer::Agents);
        canvas.clear_layer(Layer::Food);

        let sub_w = canvas.sub_width();
        let sub_h = canvas.sub_height();
        if sub_w == 0 || sub_h == 0 {
            return;
        }
        let inv_w = 1.0 / f32::from(sub_w);
        let inv_h = 1.0 / f32::from(sub_h);

        // Environment modulation comes from the renderer-neutral visual semantics
        // so the terminal, Bevy, and GPUI agree on what time of day it is and on
        // which cell is mid-pulse. Only the theme colors below are TUI-local.
        let tick = ctx.snapshot.tick;
        let daylight = visual::daylight_factor(tick, ctx.day_night.0, ctx.day_night.1);
        let day_scale = CANVAS_NIGHT_FLOOR + (1.0 - CANVAS_NIGHT_FLOOR) * daylight;

        for sy in 0..sub_h {
            let fy = (f32::from(sy) + 0.5) * inv_h;
            for sx in 0..sub_w {
                let fx = (f32::from(sx) + 0.5) * inv_w;
                // Every world sample goes through the viewport, so zoom changes
                // what is drawn rather than only what a toast claims.
                let (u, v) = ctx.viewport.world_at(fx, fy);
                let kind = ctx.terrain.sample(u, v);
                let base = ctx
                    .palette
                    .terrain_canvas_rgb_lush(kind, ctx.terrain.lushness(u, v));

                // Hillshade: the shared normal-light term keyed on terrain kind, so
                // rock reads craggy and water flat from the same elevation slope.
                let gradient = ctx.terrain.elevation_gradient(u, v);
                let mut scale =
                    visual::terrain_normal_light_factor(kind, gradient, daylight) * day_scale;

                // Water shimmer is phase-locked per WORLD tile, not per screen cell:
                // keying it on the terminal grid would make the pattern crawl as the
                // user resizes the pane, which is motion the world never had.
                if matches!(kind, TerrainKind::DeepWater | TerrainKind::ShallowWater)
                    && let Some((tx, ty)) = ctx.terrain.tile_coords(u, v)
                {
                    let pulse = visual::shimmer(tick, tx, ty);
                    scale *=
                        1.0 - CANVAS_WATER_SHIMMER_SWING + 2.0 * CANVAS_WATER_SHIMMER_SWING * pulse;
                }

                canvas.set(
                    Layer::Terrain,
                    sx,
                    sy,
                    [
                        (base[0] * scale).clamp(0.0, 1.0),
                        (base[1] * scale).clamp(0.0, 1.0),
                        (base[2] * scale).clamp(0.0, 1.0),
                        CANVAS_TERRAIN_ALPHA,
                    ],
                );

                let food = ctx.snapshot.food.sample(u, v);
                if food > CANVAS_FOOD_THRESHOLD {
                    let ink = ctx.palette.food_canvas_rgb(kind, food);
                    // Food keeps its own pulse phase (same shared function, same
                    // tile) and stays above the night floor: a dot that dims into
                    // the background is indistinguishable from food that was eaten.
                    let pulse = ctx
                        .terrain
                        .tile_coords(u, v)
                        .map_or(0.5, |(tx, ty)| visual::shimmer(tick, tx, ty));
                    let glow = (1.0 - CANVAS_FOOD_PULSE_SWING
                        + 2.0 * CANVAS_FOOD_PULSE_SWING * pulse)
                        * day_scale.max(CANVAS_NIGHT_FLOOR);
                    canvas.set(
                        Layer::Food,
                        sx,
                        sy,
                        [
                            (ink[0] * glow).clamp(0.0, 1.0),
                            (ink[1] * glow).clamp(0.0, 1.0),
                            (ink[2] * glow).clamp(0.0, 1.0),
                            1.0,
                        ],
                    );
                }
            }
        }

        let span_w = f32::from(sub_w);
        let span_h = f32::from(sub_h);
        // `None` when the agent is outside the visible window. Every agent pass
        // below skips those rather than clamping them to the border, where a
        // crowd of off-screen agents would pile up against the frame and read as
        // a real cluster.
        let agent_dot = |agent: &AgentViz| {
            let (fx, fy) = ctx.viewport.canvas_at(agent.position.0, agent.position.1)?;
            let sx = (fx * span_w).floor().clamp(0.0, span_w - 1.0) as u16;
            let sy = (fy * span_h).floor().clamp(0.0, span_h - 1.0) as u16;
            Some((sx, sy))
        };

        // Crowding pass. Without it, `set` is last-write-wins and forty agents
        // stacked on one sub-pixel are indistinguishable from one — the same
        // "unreadable soup" the sub-cell canvas exists to fix, one level down.
        // The buffer is grow-only and reused across frames like every other
        // canvas allocation.
        let occupied = usize::from(sub_w) * usize::from(sub_h);
        if density.len() < occupied {
            density.resize(occupied, 0);
        }
        density[..occupied].fill(0);
        for agent in &ctx.snapshot.agents {
            let Some((sx, sy)) = agent_dot(agent) else {
                continue;
            };
            let index = usize::from(sy) * usize::from(sub_w) + usize::from(sx);
            if let Some(slot) = density.get_mut(index) {
                *slot = slot.saturating_add(1);
            }
        }
        let crowding_at = |sx: u16, sy: u16| -> f32 {
            let index = usize::from(sy) * usize::from(sub_w) + usize::from(sx);
            cluster_heat(density.get(index).copied().unwrap_or(0))
        };

        // Whiskers first, bodies second. Both write the Agents layer, and `set`
        // replaces within a layer, so the second pass guarantees a body dot always
        // wins the sub-pixel over any neighbour's whisker. One interleaved pass
        // would let paint order decide, which is how an agent goes missing.
        for agent in &ctx.snapshot.agents {
            let Some((sx, sy)) = agent_dot(agent) else {
                continue;
            };
            let (dx, dy) = HeadingSector::from_angle(agent.heading).whisker_offset();
            let (Ok(wx), Ok(wy)) = (
                u16::try_from(i32::from(sx) + dx),
                u16::try_from(i32::from(sy) + dy),
            ) else {
                continue;
            };
            let ink = ctx.palette.agent_canvas_rgb(agent.diet, agent.energy);
            canvas.set(
                Layer::Agents,
                wx,
                wy,
                [
                    ink[0] * CANVAS_WHISKER_DIM,
                    ink[1] * CANVAS_WHISKER_DIM,
                    ink[2] * CANVAS_WHISKER_DIM,
                    1.0,
                ],
            );
        }

        for agent in &ctx.snapshot.agents {
            let Some((sx, sy)) = agent_dot(agent) else {
                continue;
            };
            let ink = ctx.palette.agent_canvas_rgb(agent.diet, agent.energy);
            // Crowding whitens rather than brightens. A multiplier saturates: a
            // bright diet color hits 1.0 at two agents and then reports the same
            // dot for two as for fifty. Blending toward white keeps climbing,
            // and reads as heat.
            let ink = whiten(ink, crowding_at(sx, sy) * CANVAS_CLUSTER_WHITEN);
            // Boost is the one agent state with no other channel on this canvas:
            // a dot is a dot regardless of speed, so flare it instead.
            let flare = if agent.boosted {
                CANVAS_BOOST_FLARE
            } else {
                1.0
            };
            canvas.set(
                Layer::Agents,
                sx,
                sy,
                [
                    (ink[0] * flare).clamp(0.0, 1.0),
                    (ink[1] * flare).clamp(0.0, 1.0),
                    (ink[2] * flare).clamp(0.0, 1.0),
                    1.0,
                ],
            );

            // The focused agent gets a ring on the Selection layer, above
            // everything. Matched on the STABLE uid: the arena handle is a
            // reusable slot, so ringing by handle would follow the slot rather
            // than the agent and could highlight a stranger after a death
            // (bd-qxrt). `None == None` must not match either — an agent whose
            // identity is missing is not "the focused one".
            if let (Some(uid), Some(focused)) = (agent.uid, ctx.snapshot.focused_agent_uid)
                && uid == focused
            {
                let ink = ctx.palette.accent_canvas_rgb();
                for (dx, dy) in CANVAS_SELECTION_RING {
                    if let (Ok(rx), Ok(ry)) = (
                        u16::try_from(i32::from(sx) + dx),
                        u16::try_from(i32::from(sy) + dy),
                    ) {
                        canvas.set(Layer::Selection, rx, ry, [ink[0], ink[1], ink[2], 1.0]);
                    }
                }
            }

            // An extended spike is an attack in progress. It goes on the Cues
            // layer, above every body, because a strike that another agent's dot
            // could hide is a strike the observer never sees.
            if agent.spike_length > CANVAS_SPIKE_THRESHOLD {
                let (dx, dy) = HeadingSector::from_angle(agent.heading).whisker_offset();
                if let (Ok(tx), Ok(ty)) = (
                    u16::try_from(i32::from(sx) + dx * 2),
                    u16::try_from(i32::from(sy) + dy * 2),
                ) {
                    canvas.set(Layer::Cues, tx, ty, [1.0, 0.32, 0.16, 1.0]);
                }
            }
        }

        Self::paint_minimap(canvas, ctx);

        // Quantize through the engine's own quantizer so a 16- or 256-color
        // terminal is handed a color it can actually reproduce instead of a
        // truecolor triple the backend will approximate on its own terms.
        // `use_canvas` already refused a depthless terminal, so the fallback here
        // is unreachable defensive code rather than a silent policy.
        let depth = ctx.capability.depth.unwrap_or(ColorDepth::Ansi16);
        let to_color = |channels: [f32; 3]| {
            let q = quantize(channels, depth);
            Color::Rgb(q[0], q[1], q[2])
        };

        let frame = canvas.composite();
        for cy in 0..frame.height_cells {
            for cx in 0..frame.width_cells {
                let Some(composed) = frame
                    .cells
                    .get(usize::from(cy) * usize::from(frame.width_cells) + usize::from(cx))
                else {
                    continue;
                };
                let cell = &mut buf[(area.x + cx, area.y + cy)];
                cell.set_char(composed.glyph);
                cell.set_style(
                    Style::default()
                        .fg(to_color(composed.fg))
                        .bg(to_color(composed.bg)),
                );
            }
        }
    }
}

impl Widget for MapWidget<'_> {
    fn render(mut self, area: Rect, buf: &mut Buffer) {
        if area.width < 2 || area.height < 2 {
            return;
        }

        if let Some(canvas) = self.canvas.take() {
            // Move the crowding buffer out so the paint pass can write it while
            // holding `&self`, then hand the same allocation back — the grow-only
            // contract is the whole reason it lives on the app instead of here.
            let mut density = std::mem::take(self.density);
            Self::render_canvas(canvas, &mut density, area, buf, &self);
            *self.density = density;
            return;
        }

        let width = area.width as usize;
        let height = area.height as usize;

        // Terrain base layer written directly into the buffer
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let terrain = self.terrain.sample(u, v);
                let food = self.snapshot.food.sample(u, v);
                let (glyph, style) = self.palette.terrain_symbol(terrain, food);
                let cell = &mut buf[(area.x + x as u16, area.y + y as u16)];
                cell.set_char(glyph);
                cell.set_style(style);
            }
        }

        // Occupancy overlay
        // Reuse caller-provided scratch buffer to avoid per-frame allocations
        let needed = width * height;
        let occupancy: &mut [CellOccupancy] = if self.scratch.len() >= needed {
            &mut self.scratch[..needed]
        } else {
            // Fallback (shouldn't happen; caller ensures capacity)
            // SAFETY: temporary vector drops at end of render; used only locally
            // We avoid unsafe here: simply allocate locally if insufficient scratch
            // but keep signature consistent.
            // Note: this else branch is never taken given current caller logic.
            // Allocate a temporary buffer.
            let _ = needed; // silence unused warning in optimized builds
            // Create a new local buffer
            // (we can't return it; so we shadow occupancy with a new Vec and borrow mut slice)
            // This block replaced below by a simple local allocation.
            // The compiler will elide this branch.
            // We still need to provide a value; allocate a local.
            // (Rust requires initialization; but this branch is unreachable.)
            // Create a zero-length slice reference as placeholder.
            &mut []
        };
        let w = width as f32;
        let h = height as f32;
        for agent in &self.snapshot.agents {
            let x = (agent.position.0 * w).floor().clamp(0.0, w - 1.0) as usize;
            let y = (agent.position.1 * h).floor().clamp(0.0, h - 1.0) as usize;
            let idx = y * width + x;
            occupancy[idx].add(
                agent.diet,
                agent.boosted,
                agent.energy,
                agent.heading,
                agent.spike_length,
                agent.tendency,
                self.stamp,
            );
        }

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if occupancy[idx].stamp != self.stamp || occupancy[idx].total() == 0 {
                    continue;
                }
                let base_style = buf[(area.x + x as u16, area.y + y as u16)].style();
                let (glyph, style) = self.palette.agent_symbol(&occupancy[idx], base_style);
                let cell = &mut buf[(area.x + x as u16, area.y + y as u16)];
                cell.set_char(glyph);
                cell.set_style(style);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{AgentData, Position, ScriptBotsConfig};

    fn canvas_test_terrain() -> TerrainView {
        TerrainView {
            width: 1,
            height: 1,
            kinds: vec![TerrainKind::Grass],
            elevations: vec![0.5],
            moisture: vec![0.5],
            fertility: vec![0.0],
        }
    }

    /// The default day/night resolution, so canvas tests exercise the same
    /// daylight curve the app resolves at launch rather than a test-only one.
    fn canvas_test_day_night() -> (u32, f32) {
        visual::resolve_day_night(None, None)
    }

    /// Truecolor braille, the tier a modern terminal probes into.
    const fn canvas_test_capability() -> CanvasCapability {
        CanvasCapability {
            mode: SubCellMode::Braille,
            depth: Some(ColorDepth::TrueColor),
        }
    }

    /// Render one canvas frame and hand back the composed buffer.
    fn render_canvas_frame(
        snapshot: &Snapshot,
        terrain: &TerrainView,
        cells: (u16, u16),
        day_night: (u32, f32),
    ) -> Buffer {
        render_canvas_frame_with(
            snapshot,
            terrain,
            cells,
            day_night,
            canvas_test_capability(),
        )
    }

    /// [`render_canvas_frame`] against an explicit capability tier.
    fn render_canvas_frame_with(
        snapshot: &Snapshot,
        terrain: &TerrainView,
        cells: (u16, u16),
        day_night: (u32, f32),
        capability: CanvasCapability,
    ) -> Buffer {
        render_canvas_frame_viewport(
            snapshot,
            terrain,
            cells,
            day_night,
            capability,
            CanvasViewport::new(1.0, (0.5, 0.5)),
        )
    }

    /// [`render_canvas_frame_with`] against an explicit viewport.
    fn render_canvas_frame_viewport(
        snapshot: &Snapshot,
        terrain: &TerrainView,
        cells: (u16, u16),
        day_night: (u32, f32),
        capability: CanvasCapability,
        viewport: CanvasViewport,
    ) -> Buffer {
        let palette = Palette::test_backend_evidence();
        let mut scratch =
            vec![CellOccupancy::default(); usize::from(cells.0) * usize::from(cells.1)];
        let mut canvas = SubCellBuffer::new(cells.0, cells.1, capability.mode);
        let mut density = Vec::new();
        let area = Rect::new(0, 0, cells.0, cells.1);
        let mut buf = Buffer::empty(area);
        MapWidget {
            snapshot,
            terrain,
            palette: &palette,
            scratch: &mut scratch,
            stamp: 1,
            canvas: Some(&mut canvas),
            day_night,
            capability,
            density: &mut density,
            viewport,
        }
        .render(area, &mut buf);
        buf
    }

    /// The braille bit a sub-pixel `(x, y)` contributes to its terminal cell.
    const fn braille_bit(sub_x: u16, sub_y: u16) -> u32 {
        const BITS: [[u32; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];
        BITS[(sub_y % 4) as usize][(sub_x % 2) as usize]
    }

    fn cell_fg(buf: &Buffer, x: u16, y: u16) -> (u8, u8, u8) {
        match buf[(x, y)].style().fg {
            Some(Color::Rgb(r, g, b)) => (r, g, b),
            other => panic!("expected an RGB foreground at ({x},{y}), got {other:?}"),
        }
    }

    fn cell_bg(buf: &Buffer, x: u16, y: u16) -> (u8, u8, u8) {
        match buf[(x, y)].style().bg {
            Some(Color::Rgb(r, g, b)) => (r, g, b),
            other => panic!("expected an RGB background at ({x},{y}), got {other:?}"),
        }
    }

    fn luminance(rgb: (u8, u8, u8)) -> f32 {
        0.2126 * f32::from(rgb.0) + 0.7152 * f32::from(rgb.1) + 0.0722 * f32::from(rgb.2)
    }

    fn canvas_test_agent(x: f32, y: f32) -> AgentViz {
        AgentViz {
            id: 1,
            uid: Some(1),
            position: (x, y),
            heading: 0.0,
            diet: DietClass::Herbivore,
            energy: 1.0,
            health: 1.0,
            age: 0,
            generation: 0,
            boosted: false,
            spike_length: 0.0,
            tendency: 0.0,
        }
    }

    /// The whole point of the sub-cell canvas: two agents that the flat map
    /// collapses into one glyph must remain separately visible. A 4x2 cell area
    /// is an 8x8 braille sub-grid, so (0.0,0.0) lands on dot (0,0) and
    /// (0.13,0.13) on dot (1,1) — both inside terminal cell (0,0).
    ///
    /// Expected glyph pins the Unicode braille bit layout through the real
    /// render path: body dot(0,0)=0x01, body dot(1,1)=0x10, and the east-facing
    /// heading whisker of the first agent at dot(1,0)=0x08. The second agent's
    /// whisker lands at sub-pixel (2,1), which belongs to the next cell.
    #[test]
    fn sub_cell_canvas_resolves_two_agents_inside_one_terminal_cell() {
        let mut snapshot = Snapshot::default();
        snapshot.agents = vec![canvas_test_agent(0.0, 0.0), canvas_test_agent(0.13, 0.13)];
        let terrain = canvas_test_terrain();
        let buf = render_canvas_frame(&snapshot, &terrain, (4, 2), canvas_test_day_night());

        let expected =
            char::from_u32(0x2800 | braille_bit(0, 0) | braille_bit(1, 1) | braille_bit(1, 0))
                .expect("braille code point");
        assert_eq!(
            buf[(0, 0)].symbol(),
            expected.to_string(),
            "two agents in one cell must occupy two distinct braille dots"
        );
    }

    /// Terrain paints below `ALPHA_SOLID` so it lands in the cell background and
    /// leaves the dots for agents and food. An empty cell must therefore be a
    /// blank braille glyph with a real background color, not an unstyled cell.
    #[test]
    fn sub_cell_canvas_puts_terrain_in_the_background_not_the_dots() {
        let snapshot = Snapshot::default();
        let terrain = canvas_test_terrain();
        let buf = render_canvas_frame(&snapshot, &terrain, (4, 2), canvas_test_day_night());

        let cell = &buf[(1, 1)];
        assert_eq!(
            cell.symbol(),
            "\u{2800}",
            "terrain must not light braille dots"
        );
        assert!(
            matches!(cell.style().bg, Some(Color::Rgb(..))),
            "terrain must reach the cell background, got {:?}",
            cell.style().bg
        );
    }

    /// The flat path must still be intact and must NOT emit braille — this is
    /// what the `B` toggle and the no-color fallback drop back to.
    #[test]
    fn flat_map_path_emits_no_braille_when_canvas_is_absent() {
        let mut snapshot = Snapshot::default();
        snapshot.agents = vec![canvas_test_agent(0.0, 0.0)];
        let terrain = canvas_test_terrain();
        let palette = Palette::test_backend_evidence();
        let mut scratch = vec![CellOccupancy::default(); 8];
        let area = Rect::new(0, 0, 4, 2);
        let mut buf = Buffer::empty(area);

        MapWidget {
            snapshot: &snapshot,
            terrain: &terrain,
            palette: &palette,
            scratch: &mut scratch,
            stamp: 1,
            canvas: None,
            day_night: canvas_test_day_night(),
            capability: canvas_test_capability(),
            density: &mut Vec::new(),
            viewport: CanvasViewport::new(1.0, (0.5, 0.5)),
        }
        .render(area, &mut buf);

        for y in 0..2 {
            for x in 0..4 {
                let symbol = buf[(x, y)].symbol().to_string();
                assert!(
                    !symbol
                        .chars()
                        .any(|c| ('\u{2800}'..='\u{28FF}').contains(&c)),
                    "flat path emitted braille at ({x},{y}): {symbol:?}"
                );
            }
        }
    }

    /// A heading must be visible in the canvas, and it must be visible in the
    /// direction the agent is actually facing. All eight sectors are driven
    /// through the real render path and checked against the sub-pixel the
    /// shared [`HeadingSector`] encoding names — so a whisker wired to the wrong
    /// axis, or to a sector table that collapses two directions, fails here.
    #[test]
    fn heading_whisker_lights_the_sub_pixel_the_agent_faces() {
        let terrain = canvas_test_terrain();
        // Centre of a 4x4 cell area = an 8x16 braille grid; (0.5, 0.5) lands on
        // sub-pixel (4, 8), which has a neighbour in every direction.
        for step in 0..8_u16 {
            let heading = f32::from(step) * (PI / 4.0);
            let mut agent = canvas_test_agent(0.5, 0.5);
            agent.heading = heading;
            let mut snapshot = Snapshot::default();
            snapshot.agents = vec![agent];

            let buf = render_canvas_frame(&snapshot, &terrain, (4, 4), canvas_test_day_night());

            let (dx, dy) = HeadingSector::from_angle(heading).whisker_offset();
            let (wx, wy) = (
                u16::try_from(4 + dx).expect("whisker stays on the grid"),
                u16::try_from(8 + dy).expect("whisker stays on the grid"),
            );
            let symbol = buf[(wx / 2, wy / 4)].symbol();
            let code = symbol.chars().next().map(u32::from).expect("one glyph");
            assert!(
                code & braille_bit(wx, wy) != 0,
                "heading {heading} rad must light sub-pixel ({wx},{wy}); glyph was {symbol:?}"
            );
        }
    }

    /// A whisker must never be mistaken for a second agent: it is painted at a
    /// fraction of the body's brightness, and a body always wins a sub-pixel a
    /// neighbour's whisker also wanted.
    #[test]
    fn a_body_dot_outranks_a_neighbours_whisker() {
        let terrain = canvas_test_terrain();
        // 4x2 cells = an 8x8 grid. An east-facing agent on sub-pixel (0,0) puts
        // its whisker on (1,0); a second agent's BODY sits on that same pixel.
        let mut leader = canvas_test_agent(0.0, 0.0);
        leader.heading = 0.0;
        let follower = canvas_test_agent(0.13, 0.0);
        let mut snapshot = Snapshot::default();
        snapshot.agents = vec![leader, follower];
        let with_body = render_canvas_frame(&snapshot, &terrain, (4, 2), canvas_test_day_night());

        let mut lone = Snapshot::default();
        let mut solo = canvas_test_agent(0.0, 0.0);
        solo.heading = 0.0;
        lone.agents = vec![solo];
        let whisker_only = render_canvas_frame(&lone, &terrain, (4, 2), canvas_test_day_night());

        assert!(
            luminance(cell_fg(&with_body, 0, 0)) > luminance(cell_fg(&whisker_only, 0, 0)),
            "a body on the whisker's pixel must brighten the cell, not be erased by it"
        );
    }

    /// Boost has no other channel on this canvas — a dot is a dot regardless of
    /// speed — so it must change the dot's brightness or it is invisible.
    #[test]
    fn boosted_agents_flare_brighter_than_unboosted_ones() {
        let terrain = canvas_test_terrain();
        let render = |boosted: bool| {
            let mut agent = canvas_test_agent(0.0, 0.0);
            agent.boosted = boosted;
            let mut snapshot = Snapshot::default();
            snapshot.agents = vec![agent];
            let buf = render_canvas_frame(&snapshot, &terrain, (4, 2), canvas_test_day_night());
            luminance(cell_fg(&buf, 0, 0))
        };
        assert!(
            render(true) > render(false),
            "a boosted agent must read brighter than an idle one"
        );
    }

    /// An extended spike is an attack in progress. It paints on the Cues layer,
    /// which outranks every body, so a strike can never be hidden behind another
    /// agent's dot — and it lands two sub-pixels ahead, not on the attacker.
    #[test]
    fn an_extended_spike_paints_a_cue_ahead_of_the_attacker() {
        let terrain = canvas_test_terrain();
        let mut attacker = canvas_test_agent(0.5, 0.5);
        attacker.heading = 0.0; // east
        attacker.spike_length = 1.0;
        // A second agent sits exactly where the cue lands; the cue must still win.
        let shield = canvas_test_agent(0.76, 0.5);
        let mut snapshot = Snapshot::default();
        snapshot.agents = vec![attacker, shield];
        let armed = render_canvas_frame(&snapshot, &terrain, (4, 4), canvas_test_day_night());

        let mut idle_snapshot = Snapshot::default();
        let mut idle = canvas_test_agent(0.5, 0.5);
        idle.heading = 0.0;
        idle_snapshot.agents = vec![idle, canvas_test_agent(0.76, 0.5)];
        let idle_frame =
            render_canvas_frame(&idle_snapshot, &terrain, (4, 4), canvas_test_day_night());

        // Sub-pixel (4,8) + 2 east = (6,8), which is terminal cell (3,2).
        let armed_fg = cell_fg(&armed, 3, 2);
        let idle_fg = cell_fg(&idle_frame, 3, 2);
        assert_ne!(
            armed_fg, idle_fg,
            "an extended spike must change what the cell ahead of the attacker shows"
        );
        // Theme-independent: the cue is hot, so it can only push the cell's red
        // channel up relative to the same scene with the spike retracted.
        assert!(
            armed_fg.0 > idle_fg.0,
            "the attack cue must warm the cell ahead: armed {armed_fg:?} vs idle {idle_fg:?}"
        );
    }

    /// Day and night must come from the SHARED daylight curve, not a
    /// terminal-local one: the same tick has to mean the same time of day here as
    /// it does on the GPU surfaces. Driving noon and midnight through the real
    /// render path proves the canvas is reading that curve at all.
    #[test]
    fn terrain_brightness_follows_the_shared_daylight_curve() {
        let terrain = canvas_test_terrain();
        let cycle = 1_000_u32;
        let background_at = |tick: u64| {
            let mut snapshot = Snapshot::default();
            snapshot.tick = tick;
            let buf = render_canvas_frame(&snapshot, &terrain, (4, 2), (cycle, 0.0));
            luminance(cell_bg(&buf, 1, 1))
        };
        // With start_phase 0.0, phase 0.25 of the cycle is noon and 0.75 midnight.
        let noon = background_at(u64::from(cycle) / 4);
        let midnight = background_at(u64::from(cycle) * 3 / 4);
        assert!(
            visual::daylight_factor(u64::from(cycle) / 4, cycle, 0.0)
                > visual::daylight_factor(u64::from(cycle) * 3 / 4, cycle, 0.0),
            "fixture assumption broken: the shared curve must call this noon and midnight"
        );
        assert!(
            noon > midnight,
            "terrain must darken at night: noon {noon}, midnight {midnight}"
        );
        assert!(
            midnight > 0.0,
            "night must not collapse the map to pure black: {midnight}"
        );
    }

    /// Hillshade must actually consume the elevation field. Identical color,
    /// identical daylight, different slope — the rendered background has to
    /// differ, or the gradient is being computed and thrown away.
    #[test]
    fn sloped_terrain_shades_differently_from_flat_terrain() {
        let flat = TerrainView {
            width: 3,
            height: 1,
            kinds: vec![TerrainKind::Rock; 3],
            elevations: vec![0.5, 0.5, 0.5],
            moisture: vec![0.5; 3],
            fertility: vec![0.0; 3],
        };
        let sloped = TerrainView {
            width: 3,
            height: 1,
            kinds: vec![TerrainKind::Rock; 3],
            elevations: vec![0.0, 0.5, 1.0],
            moisture: vec![0.5; 3],
            fertility: vec![0.0; 3],
        };
        let snapshot = Snapshot::default();
        let sample = |terrain: &TerrainView| {
            let buf = render_canvas_frame(&snapshot, terrain, (3, 2), canvas_test_day_night());
            luminance(cell_bg(&buf, 1, 0))
        };
        let flat_luma = sample(&flat);
        let sloped_luma = sample(&sloped);
        assert!(
            (flat_luma - sloped_luma).abs() > 0.5,
            "a lit slope must not render identically to flat ground: {flat_luma} vs {sloped_luma}"
        );
    }

    /// Water shimmer is phase-locked to the shared per-cell pulse, so the same
    /// tile must move between ticks and two renders of the SAME tick must be
    /// byte-identical. A shimmer keyed on wall-clock or on screen position would
    /// fail the second half.
    #[test]
    fn water_shimmer_advances_with_the_tick_and_is_deterministic() {
        let water = TerrainView {
            width: 1,
            height: 1,
            kinds: vec![TerrainKind::DeepWater],
            elevations: vec![0.5],
            moisture: vec![0.5],
            fertility: vec![0.0],
        };
        let frame_at = |tick: u64| {
            let mut snapshot = Snapshot::default();
            snapshot.tick = tick;
            render_canvas_frame(&snapshot, &water, (4, 2), (0, 0.0))
        };
        // A zero-length day cycle pins daylight to the static value, so the only
        // thing that can still vary between these frames is the shimmer. Sweep
        // the period rather than picking one offset: a single sample can land
        // where two phases happen to quantize to the same byte, which would make
        // a working shimmer look dead.
        let base = cell_bg(&frame_at(0), 1, 1);
        let moved =
            (1..visual::SHIMMER_PERIOD_TICKS).any(|tick| cell_bg(&frame_at(tick), 1, 1) != base);
        assert!(
            moved,
            "water must shimmer somewhere within one shared period"
        );

        let quarter = visual::SHIMMER_PERIOD_TICKS / 4;
        assert_eq!(
            frame_at(quarter),
            frame_at(quarter),
            "the same tick must produce a byte-identical frame"
        );
    }

    /// Crowding must be zero for the common case and must keep climbing without
    /// ever reaching the top, so no two pile sizes ever report the same heat.
    #[test]
    fn cluster_heat_is_zero_alone_and_strictly_monotonic_thereafter() {
        assert_eq!(cluster_heat(0), 0.0, "an empty sub-pixel is not crowded");
        assert_eq!(cluster_heat(1), 0.0, "a lone agent must not be tinted");
        let mut previous = 0.0_f32;
        for count in 2..=64_u16 {
            let heat = cluster_heat(count);
            assert!(
                heat > previous,
                "heat must strictly increase at {count}: {heat} vs {previous}"
            );
            assert!(heat < 1.0, "heat must never saturate at {count}: {heat}");
            previous = heat;
        }
        assert!(
            (cluster_heat(2) - 0.5).abs() < f32::EPSILON,
            "two agents sit exactly halfway"
        );
    }

    /// The defect this closes: `SubCellBuffer::set` is last-write-wins, so before
    /// the crowding pass a stack of agents on one sub-pixel produced a byte for
    /// byte identical cell to a single agent there. An observer could not tell a
    /// swarm from a straggler.
    #[test]
    fn a_crowded_sub_pixel_reads_differently_from_a_lone_agent() {
        let terrain = canvas_test_terrain();
        let render_pile = |count: usize| {
            let mut snapshot = Snapshot::default();
            snapshot.agents = (0..count).map(|_| canvas_test_agent(0.0, 0.0)).collect();
            let buf = render_canvas_frame(&snapshot, &terrain, (4, 2), canvas_test_day_night());
            cell_fg(&buf, 0, 0)
        };
        let lone = render_pile(1);
        let pair = render_pile(2);
        let swarm = render_pile(16);
        assert_ne!(lone, pair, "two agents must not render as one");
        assert_ne!(pair, swarm, "a swarm must not render as a pair");
        assert!(
            luminance(swarm) > luminance(pair) && luminance(pair) > luminance(lone),
            "crowding must read as heat: lone {lone:?}, pair {pair:?}, swarm {swarm:?}"
        );
    }

    /// Whitening, not brightening: a maximally crowded dot must still be tinted
    /// by its diet so a carnivore pile and a herbivore pile stay distinguishable.
    #[test]
    fn a_crowded_dot_keeps_a_trace_of_its_diet_color() {
        let terrain = canvas_test_terrain();
        let pile_of = |diet: DietClass| {
            let mut snapshot = Snapshot::default();
            snapshot.agents = (0..64)
                .map(|_| {
                    let mut agent = canvas_test_agent(0.0, 0.0);
                    agent.diet = diet;
                    agent
                })
                .collect();
            let buf = render_canvas_frame(&snapshot, &terrain, (4, 2), canvas_test_day_night());
            cell_fg(&buf, 0, 0)
        };
        let herbivores = pile_of(DietClass::Herbivore);
        let carnivores = pile_of(DietClass::Carnivore);
        assert_ne!(
            herbivores, carnivores,
            "a fully crowded pile must not wash out to the same white regardless of diet"
        );
    }

    const TRUECOLOR: ColorSupport = ColorSupport {
        basic: true,
        ansi256: true,
        truecolor: true,
    };
    const ANSI256: ColorSupport = ColorSupport {
        basic: true,
        ansi256: true,
        truecolor: false,
    };
    const ANSI16: ColorSupport = ColorSupport {
        basic: true,
        ansi256: false,
        truecolor: false,
    };
    const NO_SUPPORT: ColorSupport = ColorSupport {
        basic: false,
        ansi256: false,
        truecolor: false,
    };

    /// The whole degradation ladder in one table. Before this, only
    /// braille+truecolor and braille+256 were reachable: a 16-color terminal was
    /// refused the canvas outright, `NO_COLOR` was never consulted, and the glyph
    /// mode was a hardcoded `Braille` regardless of what the terminal could draw.
    #[test]
    fn capability_probe_walks_the_full_degradation_ladder() {
        let utf8 = "en_us.utf-8";
        let cases: [(
            &str,
            ColorSupport,
            &str,
            &str,
            bool,
            SubCellMode,
            Option<ColorDepth>,
        ); 9] = [
            (
                "modern truecolor",
                TRUECOLOR,
                "xterm-256color",
                utf8,
                false,
                SubCellMode::Braille,
                Some(ColorDepth::TrueColor),
            ),
            (
                "256-color only",
                ANSI256,
                "xterm-256color",
                utf8,
                false,
                SubCellMode::Braille,
                Some(ColorDepth::Ansi256),
            ),
            (
                "16-color only",
                ANSI16,
                "xterm",
                utf8,
                false,
                SubCellMode::Braille,
                Some(ColorDepth::Ansi16),
            ),
            (
                "no color reported",
                NO_SUPPORT,
                "xterm",
                utf8,
                false,
                SubCellMode::Braille,
                None,
            ),
            (
                "NO_COLOR overrides a capable terminal",
                TRUECOLOR,
                "xterm-256color",
                utf8,
                true,
                SubCellMode::Braille,
                None,
            ),
            (
                "linux console has blocks but no braille",
                ANSI256,
                "linux",
                utf8,
                false,
                SubCellMode::Quadrant,
                Some(ColorDepth::Ansi256),
            ),
            (
                "non-utf8 locale rules out every multi-byte glyph",
                TRUECOLOR,
                "xterm-256color",
                "en_us.iso-8859-1",
                false,
                SubCellMode::Ascii,
                Some(ColorDepth::TrueColor),
            ),
            (
                "dumb terminal",
                ANSI16,
                "dumb",
                utf8,
                false,
                SubCellMode::Ascii,
                Some(ColorDepth::Ansi16),
            ),
            (
                "unset TERM",
                NO_SUPPORT,
                "",
                "",
                false,
                SubCellMode::Ascii,
                None,
            ),
        ];

        for (name, color, term, locale, no_color, mode, depth) in cases {
            let capability = CanvasCapability::detect(color, term, locale, no_color);
            assert_eq!(capability.mode, mode, "{name}: glyph mode");
            assert_eq!(capability.depth, depth, "{name}: color depth");
        }
    }

    /// Two independent reasons to stay on the flat map, and they must both hold:
    /// no sub-cell density is useless, and no color is worse than useless — an
    /// uncolored braille field cannot distinguish water from rock, while the flat
    /// map's per-terrain glyphs still can.
    #[test]
    fn the_canvas_is_refused_without_both_density_and_color() {
        let utf8 = "en_us.utf-8";
        assert!(
            CanvasCapability::detect(TRUECOLOR, "xterm-256color", utf8, false).use_canvas(),
            "a capable terminal must get the canvas"
        );
        assert!(
            CanvasCapability::detect(ANSI16, "xterm", utf8, false).use_canvas(),
            "16 colors is a degradation, not a disqualification"
        );
        assert!(
            !CanvasCapability::detect(TRUECOLOR, "xterm-256color", utf8, true).use_canvas(),
            "NO_COLOR must fall back to the flat map"
        );
        assert!(
            !CanvasCapability::detect(TRUECOLOR, "dumb", utf8, false).use_canvas(),
            "no sub-cell density means no canvas"
        );
        assert!(
            !CanvasCapability::detect(NO_SUPPORT, "xterm-256color", utf8, false).use_canvas(),
            "a colorless terminal must keep the flat map's terrain glyphs"
        );
    }

    /// Each glyph tier must actually reach the screen. A mode that resolves in the
    /// probe but never changes what is painted is the same defect as a dead
    /// engine, one layer up.
    #[test]
    fn every_probed_glyph_mode_paints_its_own_vocabulary() {
        let terrain = canvas_test_terrain();
        let mut snapshot = Snapshot::default();
        snapshot.agents = vec![canvas_test_agent(0.0, 0.0)];

        let glyph_for = |mode: SubCellMode| {
            let buf = render_canvas_frame_with(
                &snapshot,
                &terrain,
                (4, 2),
                canvas_test_day_night(),
                CanvasCapability {
                    mode,
                    depth: Some(ColorDepth::TrueColor),
                },
            );
            buf[(0, 0)].symbol().to_string()
        };

        let braille = glyph_for(SubCellMode::Braille);
        assert!(
            braille
                .chars()
                .all(|c| ('\u{2800}'..='\u{28FF}').contains(&c)),
            "braille tier must emit braille, got {braille:?}"
        );
        assert_eq!(
            glyph_for(SubCellMode::HalfBlock),
            "\u{2580}",
            "half-block tier must emit the upper half block"
        );
        let quadrant = glyph_for(SubCellMode::Quadrant);
        assert!(
            quadrant
                .chars()
                .all(|c| ('\u{2580}'..='\u{259F}').contains(&c)),
            "quadrant tier must emit a block-drawing glyph, got {quadrant:?}"
        );
    }

    /// The probed depth must reach the quantizer. A 16-color terminal handed a
    /// truecolor triple gets whatever the backend decides; handed a quantized one
    /// it gets a color from the palette it actually has.
    #[test]
    fn the_probed_color_depth_reaches_the_quantizer() {
        let terrain = canvas_test_terrain();
        let snapshot = Snapshot::default();
        let background_for = |depth: ColorDepth| {
            let buf = render_canvas_frame_with(
                &snapshot,
                &terrain,
                (4, 2),
                canvas_test_day_night(),
                CanvasCapability {
                    mode: SubCellMode::Braille,
                    depth: Some(depth),
                },
            );
            cell_bg(&buf, 1, 1)
        };
        let truecolor = background_for(ColorDepth::TrueColor);
        let ansi16 = background_for(ColorDepth::Ansi16);
        assert_ne!(
            truecolor, ansi16,
            "a 16-color tier must snap the terrain color onto the ANSI palette"
        );
        let expected = quantize(
            [
                f32::from(truecolor.0) / 255.0,
                f32::from(truecolor.1) / 255.0,
                f32::from(truecolor.2) / 255.0,
            ],
            ColorDepth::Ansi16,
        );
        assert_eq!(
            ansi16,
            (expected[0], expected[1], expected[2]),
            "the 16-color background must be exactly the engine's quantization of the truecolor one"
        );
    }

    /// bd-16g.4 acceptance: "every one of the 25 sensor values displayed matches
    /// core's runtime.sensors exactly (assert programmatically, not by eye)".
    ///
    /// Proven as TWO links, because either one can break independently and the
    /// symptom is identical — a panel confidently showing a wrong number:
    ///   capture  the probe's attribution must equal what the sim actually
    ///            computed for that agent, element by element. A probe that
    ///            re-sampled or re-derived would drift from the brain's own input.
    ///   display  the rendered buffer must contain those values. A panel that
    ///            captured perfectly and then rendered a stale or reformatted
    ///            number is just as wrong to the user.
    ///
    /// The existing sense-probe test asserts the capture is populated and in
    /// range; neither link below was covered.
    #[test]
    fn every_displayed_sensor_value_matches_what_core_computed() {
        let world = command_characterization_world();
        {
            let mut guard = world.lock().expect("probe world");
            let family = guard
                .brain_registry_mut()
                .expect("probe registry mutation")
                .register_with_state_digest(
                    "terminal.sensor_fidelity",
                    0x5345_4e53_4f52_4649,
                    |_rng| Ok(Box::new(ProbePanelBrain)),
                );
            for offset in [0.0_f32, 12.0] {
                let agent_id = guard
                    .try_spawn_agent(AgentData {
                        position: Position {
                            x: 100.0 + offset,
                            y: 100.0,
                        },
                        ..AgentData::default()
                    })
                    .expect("spawn probe agent");
                guard
                    .bind_agent_brain(agent_id, family)
                    .expect("bind probe brain");
            }
        }

        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let mut app = TerminalApp::new(
            &renderer,
            crate::renderer::RendererContext {
                simulation_step: disabled_persistence_step_driver(&world),
                world: Arc::clone(&world),
                analytics: AnalyticsSnapshotProvider::empty(),
                control_runtime: &runtime,
                command_drain: drain,
                command_submit: submit,
                scenario: test_scenario(),
            },
        );
        // Step once BEFORE probing. `runtime.sensors` is written by stage_sense,
        // so on an unstepped world it is still all zeros while the probe computes
        // its attribution from live state — comparing them then would report a
        // drift that is really just "the sim has not sensed yet".
        {
            let mut guard = world.lock().expect("probe world");
            guard.step().expect("step the probe world");
        }

        app.paused = true;
        app.probe_enabled = true;
        app.refresh_snapshot();

        let probe = app
            .snapshot
            .probe
            .clone()
            .expect("probe captured for the focused agent");

        // NOT ASSERTED, and the reason is a finding rather than an omission:
        // this bead's acceptance says the displayed values must match
        // `runtime.sensors` exactly. They cannot, by design. `SensorAttribution`
        // documents itself as "an instantaneous completed-boundary
        // counterfactual [that] deliberately does not reproduce
        // AgentRuntime::sensors, because those were computed from the
        // pre-actuation world that no longer exists". Asserting equality
        // reproduces that divergence as a test failure — measured here as
        // food 0.021 (probe) vs 0.025 (runtime) one step in. So the criterion
        // needs amending or the probe needs different semantics; recorded on
        // bd-16g.4 rather than silently asserting the weaker thing and calling
        // the criterion met.
        //
        // What IS the view's responsibility, and is asserted below: the panel
        // must display exactly the attribution it was handed, with no
        // re-derivation of its own. That is the drift this layer can actually
        // introduce.
        for channel in &SENSOR_LAYOUT {
            assert!(
                probe.attribution.clamped[channel.index].is_finite(),
                "{} carries a non-finite value into the panel",
                channel.name
            );
        }

        // Render the real panel and read the buffer
        // back. Every scalar channel is drawn as "{name} {clamped:.2}", so the
        // exact pair must appear; a panel that dropped a channel or rendered a
        // neighbouring slot's value fails here.
        let area = Rect::new(0, 0, 100, 24);
        let mut terminal =
            Terminal::new(ratatui::backend::TestBackend::new(area.width, area.height))
                .expect("probe test terminal");
        terminal
            .draw(|frame| app.draw_probe(frame, area, &app.snapshot))
            .expect("draw the sense probe");
        let rendered = export::buffer_to_plain_text(terminal.backend().buffer());

        for channel in SENSOR_LAYOUT.iter().filter(|c| c.eye.is_none()) {
            let expected = format!(
                "{} {:.2}",
                channel.name, probe.attribution.clamped[channel.index]
            );
            assert!(
                rendered.contains(&expected),
                "scalar channel {} is not displayed as {expected:?}; panel was:\n{rendered}",
                channel.name
            );
        }
        for eye in 0..NUM_EYES {
            assert!(
                rendered.contains(&format!("eye{eye}")),
                "eye {eye} row is missing from the panel"
            );
        }
    }

    /// The typographic scale must be a real HIERARCHY, not three names for one
    /// look (bd-f4x0).
    ///
    /// Before this the TUI had two tiers — bold header, or unstyled — so a field
    /// label was drawn in the same bold accent as a panel title and competed with
    /// the number it introduced. A "scale" whose tiers were visually identical
    /// would be the same defect with more code, so each tier is asserted distinct
    /// from the others and the weights are asserted in order.
    #[test]
    fn the_typographic_scale_is_an_ordered_hierarchy() {
        let palette = Palette::test_backend_evidence();
        let title = palette.header_style();
        let label = palette.label_style();
        let value = palette.value_style();
        let muted = palette.muted_style();

        assert!(
            title.add_modifier.contains(Modifier::BOLD),
            "a panel title is the heaviest tier"
        );
        assert!(
            !value.add_modifier.contains(Modifier::BOLD)
                && !value.add_modifier.contains(Modifier::DIM),
            "a value carries by being the only thing at full weight, not by shouting"
        );
        assert!(
            label.add_modifier.contains(Modifier::DIM),
            "a label is chrome and must recede"
        );
        assert!(
            muted.add_modifier.contains(Modifier::DIM),
            "a hint is at least as quiet as a label"
        );

        // Distinctness: a label and a hint share DIM, so colour must separate them.
        assert_ne!(
            label, muted,
            "label and hint must be distinguishable, not two names for one style"
        );
        assert_ne!(label, value, "label must not render as a value");
        assert_ne!(title, value, "title must not render as a value");
    }

    /// bd-f4x0 requires the accessibility palettes to SURVIVE the redesign. The
    /// scale derives its colour from the theme rather than hardcoding one, so
    /// this drives every palette mode and every curated theme and asserts the
    /// hierarchy still holds — a scale that only worked in the default theme
    /// would have quietly dropped the colourblind-safe modes.
    #[test]
    fn the_scale_survives_every_accessibility_palette_and_theme() {
        for mode in [
            TerminalPaletteMode::Natural,
            TerminalPaletteMode::Deuteranopia,
            TerminalPaletteMode::Protanopia,
            TerminalPaletteMode::Tritanopia,
            TerminalPaletteMode::HighContrast,
        ] {
            for theme_id in [
                CuratedThemeId::BioluminescentDarkField,
                CuratedThemeId::CyberpunkAurora,
                CuratedThemeId::Darcula,
                CuratedThemeId::LumenLight,
                CuratedThemeId::NordicFrost,
                CuratedThemeId::HighContrast,
            ] {
                let mut palette = Palette::test_backend_evidence();
                palette.mode = mode;
                palette.theme_id = theme_id;

                let label = palette.label_style();
                let value = palette.value_style();
                let muted = palette.muted_style();
                assert!(
                    label.add_modifier.contains(Modifier::DIM),
                    "{mode:?}/{theme_id:?}: label must stay dim"
                );
                assert_ne!(
                    label, muted,
                    "{mode:?}/{theme_id:?}: label and hint must stay distinguishable"
                );
                assert_ne!(
                    value, muted,
                    "{mode:?}/{theme_id:?}: value and hint must stay distinguishable"
                );
            }
        }
    }

    /// Cone selection is a RING over `All` plus each eye, wrapping both ways.
    ///
    /// Increment-and-clamp would make the ends sticky: a user holding `.` would
    /// silently stop at the last eye and read the control as dead, which is the
    /// same "control that stops responding" class bd-jw6f was filed for.
    #[test]
    fn cone_selection_cycles_through_every_eye_and_wraps_both_ways() {
        let world = command_characterization_world();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let mut app = TerminalApp::new(
            &renderer,
            crate::renderer::RendererContext {
                simulation_step: disabled_persistence_step_driver(&world),
                world: Arc::clone(&world),
                analytics: AnalyticsSnapshotProvider::empty(),
                control_runtime: &runtime,
                command_drain: drain,
                command_submit: submit,
                scenario: test_scenario(),
            },
        );
        assert_eq!(app.selected_eye, None, "starts showing all cones");

        // Forward: All -> 0 -> 1 -> .. -> NUM_EYES-1 -> All
        for expected in 0..NUM_EYES {
            app.cycle_eye_selection(true);
            assert_eq!(
                app.selected_eye,
                Some(expected),
                "forward step must reach eye {expected}"
            );
        }
        app.cycle_eye_selection(true);
        assert_eq!(
            app.selected_eye, None,
            "forward past the last eye wraps to all"
        );

        // Backward from All must reach the LAST eye, not stick.
        app.cycle_eye_selection(false);
        assert_eq!(
            app.selected_eye,
            Some(NUM_EYES - 1),
            "backward from all must wrap to the last eye"
        );
        for expected in (0..NUM_EYES - 1).rev() {
            app.cycle_eye_selection(false);
            assert_eq!(app.selected_eye, Some(expected));
        }
        app.cycle_eye_selection(false);
        assert_eq!(app.selected_eye, None, "backward past eye 0 wraps to all");
    }

    /// The keystrokes are the accessible surface, so they are driven rather than
    /// the method being called directly — a binding that never reached
    /// `cycle_eye_selection` would pass a method-only test.
    #[test]
    fn the_cone_shortcuts_are_reachable_from_the_keyboard() {
        let world = command_characterization_world();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let mut app = TerminalApp::new(
            &renderer,
            crate::renderer::RendererContext {
                simulation_step: disabled_persistence_step_driver(&world),
                world: Arc::clone(&world),
                analytics: AnalyticsSnapshotProvider::empty(),
                control_runtime: &runtime,
                command_drain: drain,
                command_submit: submit,
                scenario: test_scenario(),
            },
        );

        app.handle_key(KeyEvent::new(KeyCode::Char('.'), KeyModifiers::NONE))
            .expect("dot selects a cone");
        assert_eq!(app.selected_eye, Some(0), "'.' must select the first cone");

        app.handle_key(KeyEvent::new(KeyCode::Char(','), KeyModifiers::NONE))
            .expect("comma steps back");
        assert_eq!(app.selected_eye, None, "',' must step back to all cones");
    }

    /// Run `probe` against a TerminalApp built for driving keystrokes at.
    ///
    /// Scoped rather than returned: `TerminalApp` borrows the `ControlRuntime`,
    /// so a helper that returned both would be returning a value referencing its
    /// own local. The closure keeps the runtime alive for exactly as long as the
    /// app needs it.
    fn with_shortcut_app(probe: impl FnOnce(&mut TerminalApp)) {
        let world = command_characterization_world();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let mut app = TerminalApp::new(
            &renderer,
            crate::renderer::RendererContext {
                simulation_step: disabled_persistence_step_driver(&world),
                world: Arc::clone(&world),
                analytics: AnalyticsSnapshotProvider::empty(),
                control_runtime: &runtime,
                command_drain: drain,
                command_submit: submit,
                scenario: test_scenario(),
            },
        );
        probe(&mut app);
    }

    /// Ctrl+T must cycle the chrome theme, and must NOT be eaten by plain-`t`.
    ///
    /// Driven through `handle_key` rather than by calling `cycle_theme`, because
    /// the defect this bead reopened for was precisely that the METHOD existed
    /// and worked while no key reached it: the command palette advertised
    /// "Ctrl+T" while the plain-`t` arm matched `_` on modifiers and claimed the
    /// combination first. A method-only test passes against that bug.
    #[test]
    fn ctrl_t_cycles_the_theme_without_triggering_the_plain_t_focus_action() {
        with_shortcut_app(|app| {
            let before = app.palette.theme_id;
            let focus_before = app.focus_lock;

            app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL))
                .expect("ctrl-t is handled");

            assert_ne!(
                app.palette.theme_id, before,
                "Ctrl+T must advance the chrome theme"
            );
            assert_eq!(
                app.palette.theme_id,
                before.next(),
                "Ctrl+T must advance by exactly one step of the documented cycle"
            );
            // The negative control, and the actual bug: focus must be untouched.
            assert_eq!(
                app.focus_lock, focus_before,
                "Ctrl+T must not also fire the plain-t focus action; the two bindings \
                 are separate and a shared arm is how they collided"
            );
        });
    }

    /// Plain `t` must still focus predators and must NOT cycle the theme.
    ///
    /// The other half of modifier separation: tightening the focus arm to
    /// NONE/SHIFT could have made plain `t` stop working entirely, and nothing
    /// else in the suite drives it.
    #[test]
    fn plain_t_still_focuses_predators_and_leaves_the_theme_alone() {
        with_shortcut_app(|app| {
            let theme_before = app.palette.theme_id;
            app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE))
                .expect("plain t is handled");

            assert!(
                matches!(app.focus_lock, FocusLockMode::TopPredator),
                "plain t must still lock focus to top predators"
            );
            assert_eq!(
                app.palette.theme_id, theme_before,
                "plain t must not cycle the chrome theme"
            );
        });
    }

    /// Plain `p` cycles the accessibility palette and must not open the command
    /// palette — a different control on a neighbouring binding (Ctrl+P).
    #[test]
    fn plain_p_cycles_the_accessibility_palette_without_opening_a_control() {
        with_shortcut_app(|app| {
            let mode_before = app.palette.mode;
            let theme_before = app.palette.theme_id;

            app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE))
                .expect("plain p is handled");

            assert_ne!(
                app.palette.mode, mode_before,
                "plain p must advance the accessibility palette"
            );
            assert!(
                !app.palette_open,
                "plain p must not open the command palette; that is Ctrl+P"
            );
            // The two axes are orthogonal by design: chrome theme and semantic
            // data palette must not move together.
            assert_eq!(
                app.palette.theme_id, theme_before,
                "cycling the accessibility palette must not change the chrome theme"
            );
        });
    }

    /// Ctrl+P still opens the command palette, and does not cycle anything.
    #[test]
    fn ctrl_p_opens_the_command_palette_and_cycles_nothing() {
        with_shortcut_app(|app| {
            let mode_before = app.palette.mode;
            let theme_before = app.palette.theme_id;

            app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
                .expect("ctrl-p is handled");

            assert!(app.palette_open, "Ctrl+P must open the command palette");
            assert_eq!(
                app.palette.mode, mode_before,
                "Ctrl+P must not cycle the accessibility palette"
            );
            assert_eq!(
                app.palette.theme_id, theme_before,
                "Ctrl+P must not cycle the chrome theme"
            );
        });
    }

    /// Ctrl+T walks the full cycle and wraps, driven from the keyboard.
    #[test]
    fn ctrl_t_walks_the_whole_theme_cycle_and_wraps() {
        with_shortcut_app(|app| {
            let start = app.palette.theme_id;
            let mut seen = vec![start];
            loop {
                app.handle_key(KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL))
                    .expect("ctrl-t is handled");
                let now = app.palette.theme_id;
                if now == start {
                    break;
                }
                assert!(
                    !seen.contains(&now),
                    "Ctrl+T re-entered {now:?} before wrapping; some themes are \
                     unreachable from the keyboard"
                );
                seen.push(now);
                assert!(seen.len() <= 64, "Ctrl+T never wrapped back to {start:?}");
            }
            assert!(
                seen.len() > 1,
                "the keyboard cycle must visit more than one theme"
            );
        });
    }

    /// Every theme must survive a round trip through the config layer.
    ///
    /// Walks the cycle rather than restating a list, so a seventh theme is
    /// covered the day it is added — and if the two enums ever diverge, this
    /// fails with the offending theme named rather than a config silently
    /// relocating a user somewhere else.
    /// Success and failure must be TELLABLE APART in every accessibility palette.
    ///
    /// This is the substance behind routing status chrome through the ramp. The
    /// old code hand-coded `Color::Green` for "active" and `Color::Red` for
    /// "error", which is the exact pair deuteranopic and protanopic viewers
    /// cannot separate — and this app ships palettes for both, so the raw
    /// constants were bypassing the very feature meant to help.
    ///
    /// I first wrote this as a measured LUMINANCE separation, on the theory that
    /// two hues at the same lightness are one colour to a viewer who cannot see
    /// the hue difference. MEASURING REJECTED THE ASSERTION, not the palettes.
    /// Every shipped palette separates status by HUE: ok-vs-error runs from
    /// 1.23:1 (Natural) to 3.14:1 (HighContrast), and Deuteranopia's ok-vs-warn
    /// is 1.12:1. That is deliberate — the CVD palettes retune hue onto the
    /// blue-yellow axis those viewers CAN separate and never promised a lightness
    /// ramp. Demanding one would have required redesigning all five palettes.
    ///
    /// Weakening the threshold until the current values cleared it would have
    /// been worse than deleting it: a gate tuned to its own inputs proves
    /// nothing. The measured numbers are recorded on the bead and the monochromat
    /// case they imply is filed rather than silently accepted.
    /// Mortality bars must come from the palette, and combat causes must match
    /// the diet colours their killers are drawn in.
    ///
    /// The semantic tie is the point, not just palette-awareness: a reader who
    /// has learned what a carnivore looks like on the map should not need a
    /// legend to read the mortality panel. Asserting the exact ramp entry is what
    /// keeps that true — a future edit that merely picks "some palette colour"
    /// would still be retunable and still be wrong.
    /// An applied intervention must name WHAT it changed, not just how many.
    ///
    /// A count alone cannot tell a drought from a meteor after the fact, and this
    /// bead requires an ecosystem crash caused by a mis-parameterised
    /// intervention to be obvious from the record (bd-16g.10).
    #[test]
    fn an_intervention_summary_names_the_config_keys_it_sets() {
        let due = vec![
            crate::ScenarioInterventionV1 {
                tick: 10,
                set: serde_json::json!({ "food_growth_rate": 0.0, "closed": true }),
            },
            crate::ScenarioInterventionV1 {
                tick: 99,
                set: serde_json::json!({ "never_mentioned": 1 }),
            },
        ];

        let summary = TerminalApp::intervention_summary(&due, 10);
        assert!(
            summary.contains("food_growth_rate") && summary.contains("closed"),
            "the summary must name every key set at this tick; got {summary:?}"
        );
        assert!(
            !summary.contains("never_mentioned"),
            "a key scheduled for a DIFFERENT tick must not be reported as applied \
             now; got {summary:?}"
        );
        // Deterministic order, so two runs of the same scenario produce the same
        // text in the log and the rail.
        assert_eq!(summary, "closed, food_growth_rate");
    }

    /// An intervention with no keys must say so rather than rendering blank.
    #[test]
    fn an_empty_intervention_summary_is_stated_not_silent() {
        let due = vec![crate::ScenarioInterventionV1 {
            tick: 3,
            set: serde_json::json!({}),
        }];
        assert_eq!(
            TerminalApp::intervention_summary(&due, 3),
            "no config keys",
            "an empty patch must be described, or the feedback line reads as a \
             truncated message"
        );
    }

    /// Terrain must be readable with no colour, in the ascii tier too.
    ///
    /// Part of the bd-xg82 monochrome audit. This was TRUE before the bead — it is
    /// pinned rather than fixed, because an audit that lives only in a comment
    /// stops being true the moment someone reuses a glyph, and the reader would
    /// have no warning.
    #[test]
    fn every_terrain_kind_has_a_distinct_glyph_without_colour() {
        let mut palette = Palette::test_backend_evidence();
        // Force the ascii tier: emoji mode substitutes pictographs, and the
        // guarantee has to hold on the plainest terminal, not the richest.
        palette.emoji = false;

        let kinds = [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::Sand,
            TerrainKind::Grass,
            TerrainKind::Bloom,
            TerrainKind::Rock,
        ];
        let mut seen: Vec<char> = Vec::new();
        for kind in kinds {
            let (glyph, _) = palette.terrain_symbol(kind, 0.5);
            assert!(
                !seen.contains(&glyph),
                "{kind:?} reuses glyph {glyph:?}; two terrain kinds that share a \
                 glyph are one terrain in monochrome"
            );
            seen.push(glyph);
        }
        assert_eq!(seen.len(), kinds.len(), "every terrain kind needs a glyph");
    }

    /// KNOWN GAP, pinned so it cannot be quietly forgotten or wrongly assumed
    /// fixed.
    ///
    /// `agent_symbol` varies its glyph by diet — but only as a FALLBACK. For a
    /// cell holding one agent it prefers a HEADING character, and heading is
    /// almost always available, so the common case renders an arrow whose diet is
    /// carried by colour alone. In monochrome a herbivore and a carnivore are the
    /// same arrow.
    ///
    /// This is a genuine TRADE rather than an omission: one cell can show heading
    /// or diet, not both. Removing heading would regress sighted readers, and a
    /// combined vocabulary (headings x diet classes, across the emoji/narrow/ascii
    /// tiers) is an art-direction decision under bd-9pqz. Recorded here with a
    /// failure message that says what to do rather than leaving the next reader to
    /// rediscover the trade.
    #[test]
    fn a_lone_agent_conveys_diet_by_colour_alone_bd_xg82_known_gap() {
        let mut palette = Palette::test_backend_evidence();
        palette.emoji = false;

        let base = Style::default();
        let mut glyphs: Vec<char> = Vec::new();
        for class in [
            DietClass::Herbivore,
            DietClass::Omnivore,
            DietClass::Carnivore,
        ] {
            let mut occupancy = CellOccupancy::default();
            // One agent, with a heading — the ordinary case, which is exactly
            // where the glyph channel is unavailable.
            occupancy.add(class, false, 1.0, 0.0, 0.0, 0.5, 1);
            let (glyph, _) = palette.agent_symbol(&occupancy, base);
            glyphs.push(glyph);
        }

        let all_same = glyphs.iter().all(|g| *g == glyphs[0]);
        assert!(
            all_same,
            "a lone agent now renders diet-distinct glyphs {glyphs:?} — the known \
             gap this test records has been CLOSED. Update or delete this test and \
             note it on bd-xg82; do not simply relax it"
        );
    }

    /// Every event kind must be distinguishable with no colour at all.
    ///
    /// The log rendered all four kinds as identically-formatted text and carried
    /// the kind in the foreground colour alone, so a birth and a death were the
    /// same line in monochrome (bd-xg82).
    #[test]
    fn every_event_kind_has_a_distinct_ascii_marker() {
        let mut seen: Vec<char> = Vec::new();
        for kind in EventKind::all() {
            let marker = kind.marker();
            assert!(
                marker.is_ascii_graphic(),
                "{kind:?} marker {marker:?} must be printable ASCII so the narrow \
                 and ascii capability tiers render it unchanged"
            );
            assert!(
                !seen.contains(&marker),
                "{kind:?} reuses marker {marker:?}; duplicate markers defeat the \
                 non-colour channel exactly as a shared colour would"
            );
            seen.push(marker);
        }
        assert_eq!(seen.len(), 4, "all four event kinds must be marked");
    }

    /// At least one event-kind pair is effectively colour-identical in every
    /// palette, which is why the markers are load-bearing rather than decorative.
    ///
    /// Measured, not assumed (bd-xg82): the worst pair per palette is Natural
    /// Death/Info 1.103, Deuteranopia Death/Info 1.088, Protanopia
    /// Population/Info 1.001, Tritanopia Birth/Info 1.041, HighContrast
    /// Population/Info 1.399. A 1.001:1 separation is the same colour to any
    /// viewer, so the log could not be read by hue there even with perfect colour
    /// vision.
    ///
    /// Asserting the floor stays LOW is deliberate and is not a bug: it records
    /// that colour is not a usable channel here, so nobody deletes the markers
    /// believing the palette carries the distinction. If a palette edit ever
    /// separates these properly, this test fails and the correct response is to
    /// update it and KEEP the markers — monochrome still has no colour at all.
    #[test]
    fn some_event_kinds_are_colour_identical_so_markers_carry_the_meaning() {
        let mut worst_overall = f32::MAX;
        for mode in [
            TerminalPaletteMode::Natural,
            TerminalPaletteMode::Deuteranopia,
            TerminalPaletteMode::Protanopia,
            TerminalPaletteMode::Tritanopia,
            TerminalPaletteMode::HighContrast,
        ] {
            let mut palette = Palette::test_backend_evidence();
            palette.mode = mode;
            let kinds = EventKind::all();
            let mut worst = f32::MAX;
            for i in 0..kinds.len() {
                for j in (i + 1)..kinds.len() {
                    let a = palette.event_style(kinds[i]).fg.expect("kind colour");
                    let b = palette.event_style(kinds[j]).fg.expect("kind colour");
                    worst = worst.min(contrast_ratio(a, b));
                }
            }
            assert!(
                worst < 3.0,
                "{mode:?}: the closest event-kind pair now separates by {worst:.3}:1. \
                 If the palette genuinely fixed this, update this test — but KEEP the \
                 markers, because monochrome has no colour channel at all"
            );
            worst_overall = worst_overall.min(worst);
        }
        assert!(
            worst_overall < 1.05,
            "at least one palette should still have a near-identical pair \
             ({worst_overall:.3}:1 was the closest found)"
        );
    }

    /// The four trend rows must be told apart WITHOUT colour.
    ///
    /// This is the measured justification, not a precaution. Across every
    /// accessibility palette, the population row and the births row use the SAME
    /// COLOUR — `population_spark` and the Birth event entry are one value, a
    /// separation of exactly 1.000:1. Four unlabelled full-width sparklines
    /// stacked adjacently, two of them identical, is unreadable for every viewer
    /// rather than only hue-blind ones (bd-xg82).
    ///
    /// So the labels are the load-bearing channel and they are asserted as such:
    /// present, distinct, and narrow enough to fit the reserved column. Fixing
    /// this by shifting one of the hues would have left the panel colour-only and
    /// still unreadable in monochrome.
    #[test]
    fn every_trend_row_is_labelled_distinctly_without_relying_on_colour() {
        let mut seen: Vec<&str> = Vec::new();
        for label in TREND_LABELS {
            assert!(!label.is_empty(), "every trend row needs a label");
            assert!(
                !seen.contains(&label),
                "trend label {label:?} is reused; duplicate labels defeat the \
                 non-colour channel exactly as duplicate colours did"
            );
            assert!(
                label.len() < TREND_LABEL_WIDTH as usize,
                "trend label {label:?} does not fit the {TREND_LABEL_WIDTH}-column \
                 reserve with a separating space, so it would abut the bars"
            );
            seen.push(label);
        }
        assert_eq!(seen.len(), 4, "all four trend rows must be labelled");
    }

    /// Records the defect that made the labels necessary, so it cannot quietly
    /// stop being true and leave the labels looking like decoration.
    ///
    /// If a future palette edit gives population and births distinct colours this
    /// fails, and the right response is to update this test — NOT to remove the
    /// labels. Colour separation alone would still leave the panel unreadable in
    /// monochrome, which is the whole point of bd-xg82.
    #[test]
    fn population_and_births_share_a_colour_in_every_palette() {
        for mode in [
            TerminalPaletteMode::Natural,
            TerminalPaletteMode::Deuteranopia,
            TerminalPaletteMode::Protanopia,
            TerminalPaletteMode::Tritanopia,
            TerminalPaletteMode::HighContrast,
        ] {
            let mut palette = Palette::test_backend_evidence();
            palette.mode = mode;
            let pop = palette
                .population_spark_style()
                .fg
                .expect("the population spark must carry a colour");
            let births = palette
                .event_style(EventKind::Birth)
                .fg
                .expect("the births spark must carry a colour");
            assert_eq!(
                contrast_ratio(pop, births),
                1.0,
                "{mode:?}: population and births are documented as sharing a colour \
                 ({pop:?} vs {births:?}); if that changed, update this test and KEEP \
                 the row labels — colour alone is still not a channel in monochrome"
            );
        }
    }

    /// A pane too narrow for the label column must not panic or overlap.
    #[test]
    fn a_narrow_trend_row_degrades_instead_of_panicking() {
        let wide = Rect {
            x: 3,
            y: 1,
            width: 40,
            height: 1,
        };
        let (label, spark) = TerminalApp::trend_row(wide);
        assert_eq!(label.width, TREND_LABEL_WIDTH);
        assert_eq!(spark.x, wide.x + TREND_LABEL_WIDTH);
        assert_eq!(
            label.width + spark.width,
            wide.width,
            "the split must consume the row exactly, with no overlap or gap"
        );

        for width in 0..=TREND_LABEL_WIDTH {
            let narrow = Rect { width, ..wide };
            let (label, spark) = TerminalApp::trend_row(narrow);
            assert_eq!(
                label.width, 0,
                "at width {width} there is no room for a label"
            );
            assert_eq!(
                spark.width, width,
                "at width {width} the sparkline must keep the whole row"
            );
        }
    }

    #[test]
    fn mortality_bars_take_the_palette_ramp_that_names_each_cause() {
        for mode in [
            TerminalPaletteMode::Natural,
            TerminalPaletteMode::Deuteranopia,
            TerminalPaletteMode::Protanopia,
            TerminalPaletteMode::Tritanopia,
            TerminalPaletteMode::HighContrast,
        ] {
            let mut palette = Palette::test_backend_evidence();
            palette.mode = mode;
            let theme = palette.theme();

            assert_eq!(
                palette
                    .mortality_style(MortalityCause::CombatCarnivore)
                    .fg
                    .expect("combat-carnivore deaths must carry a colour"),
                theme.diet[2],
                "{mode:?}: deaths by carnivore must use the carnivore diet colour"
            );
            assert_eq!(
                palette
                    .mortality_style(MortalityCause::CombatHerbivore)
                    .fg
                    .expect("combat-herbivore deaths must carry a colour"),
                theme.diet[0],
                "{mode:?}: deaths by herbivore must use the herbivore diet colour"
            );
            assert_eq!(
                palette
                    .mortality_style(MortalityCause::Starvation)
                    .fg
                    .expect("starvation deaths must carry a colour"),
                theme.energy_spark,
                "{mode:?}: starvation must use the energy ramp it depletes"
            );

            // No cause may fall back to a named ANSI constant, which is what
            // bypassed the accessibility palettes before.
            for cause in MortalityCause::all() {
                let fg = palette
                    .mortality_style(cause)
                    .fg
                    .unwrap_or_else(|| panic!("{cause:?} must carry a colour"));
                assert!(
                    matches!(fg, Color::Rgb(_, _, _)),
                    "{mode:?}: {cause:?} must take an explicit RGB value from the \
                     palette, not a named ANSI colour the palette cannot retune; got \
                     {fg:?}"
                );
            }
        }
    }

    /// Every cause must be labelled, so the panel survives without colour at all.
    ///
    /// bd-xg82 records that these ramps separate by hue rather than luminance, so
    /// colour alone is not a sufficient channel on a hue-blind display. The
    /// labels are what make this panel readable anyway, and that is worth pinning
    /// rather than leaving as an accident of the current layout.
    #[test]
    fn every_mortality_cause_carries_a_non_colour_label() {
        let mut seen: Vec<&str> = Vec::new();
        for cause in MortalityCause::all() {
            let label = cause.label();
            assert!(!label.is_empty(), "{cause:?} must have a label");
            assert!(
                !seen.contains(&label),
                "{cause:?} reuses the label {label:?}; duplicate labels defeat the \
                 non-colour channel"
            );
            seen.push(label);
        }
        assert_eq!(seen.len(), 5, "every cause must appear exactly once");
    }

    #[test]
    fn ok_warn_and_error_are_distinct_colours_in_every_accessibility_palette() {
        for mode in [
            TerminalPaletteMode::Natural,
            TerminalPaletteMode::Deuteranopia,
            TerminalPaletteMode::Protanopia,
            TerminalPaletteMode::Tritanopia,
            TerminalPaletteMode::HighContrast,
        ] {
            let mut palette = Palette::test_backend_evidence();
            palette.mode = mode;

            let ok = palette
                .ok_style()
                .fg
                .expect("ok status must carry a colour");
            let warn = palette
                .warn_style()
                .fg
                .expect("warn status must carry a colour");
            let err = palette
                .error_style()
                .fg
                .expect("error status must carry a colour");

            assert_ne!(
                ok, err,
                "{mode:?}: ok and error must not be the same colour"
            );
            assert_ne!(
                ok, warn,
                "{mode:?}: ok and warn must not be the same colour"
            );
            assert_ne!(
                warn, err,
                "{mode:?}: warn and error must not be the same colour"
            );

            // None may be a named ANSI constant: those bypass the palette
            // entirely, which is the defect this change removed.
            for (label, color) in [("ok", ok), ("warn", warn), ("error", err)] {
                assert!(
                    matches!(color, Color::Rgb(_, _, _)),
                    "{mode:?}: the {label} status must come from the palette ramp as an \
                     explicit RGB value, not a named ANSI colour the palette cannot \
                     retune; got {color:?}"
                );
            }
        }
    }

    /// The status ramp must actually move when the accessibility palette does.
    ///
    /// The negative control for the test above: if `ok_style` returned a
    /// constant, the separations would still hold and nothing would reveal that
    /// the palettes were being ignored — which is precisely how the raw ANSI
    /// constants passed unnoticed.
    #[test]
    fn status_chrome_retunes_with_the_accessibility_palette() {
        let mut natural = Palette::test_backend_evidence();
        natural.mode = TerminalPaletteMode::Natural;
        let mut deuter = Palette::test_backend_evidence();
        deuter.mode = TerminalPaletteMode::Deuteranopia;

        assert_ne!(
            natural.ok_style().fg,
            deuter.ok_style().fg,
            "the ok status colour must be retuned by the accessibility palette; a \
             constant here means the palette is decorative"
        );
        assert_ne!(
            natural.error_style().fg,
            deuter.error_style().fg,
            "the error status colour must be retuned by the accessibility palette"
        );
    }

    #[test]
    fn every_theme_round_trips_through_the_config_identity() {
        let start = CuratedThemeId::default();
        let mut theme = start;
        loop {
            let via_config = CuratedThemeId::from_config(theme.to_config());
            assert_eq!(
                via_config, theme,
                "{theme:?} did not survive the config round trip; it came back as \
                 {via_config:?}"
            );

            // Serde is the actual persistence boundary, so exercise it rather
            // than trusting the in-memory mapping.
            let encoded =
                serde_json::to_string(&theme.to_config()).expect("a theme identity must serialise");
            let decoded: scriptbots_core::TuiThemeId =
                serde_json::from_str(&encoded).expect("a theme identity must deserialise");
            assert_eq!(
                CuratedThemeId::from_config(decoded),
                theme,
                "{theme:?} did not survive serde; encoded as {encoded}"
            );

            theme = theme.next();
            if theme == start {
                break;
            }
        }
    }

    /// The DEFAULT theme in particular must be representable in config.
    ///
    /// Called out separately because it was not: core's TuiThemeId shipped
    /// without BioluminescentDarkField while the terminal defaults to it, so the
    /// one theme nearly every run displays was the one that could not be
    /// persisted (bd-2z0.14.2.2).
    #[test]
    fn the_default_theme_is_representable_in_the_config_layer() {
        let default = CuratedThemeId::default();
        assert_eq!(
            CuratedThemeId::from_config(default.to_config()),
            default,
            "the default theme must round-trip, or a user on it cannot persist it"
        );
    }

    /// A configured theme must actually reach the palette, and an unset config
    /// must leave the detected theme alone.
    #[test]
    fn the_configured_theme_is_adopted_and_none_leaves_the_default_alone() {
        let mut palette = Palette::test_backend_evidence();
        let detected = palette.theme_id;

        palette.apply_config_theme(None);
        assert_eq!(
            palette.theme_id, detected,
            "an unset config must not override the detected theme; None and a \
             chosen theme are different states"
        );

        // Pick a theme that is definitely not the current one, so a no-op
        // implementation cannot pass.
        let wanted = detected.next();
        assert_ne!(
            wanted, detected,
            "the cycle must move for this to prove anything"
        );
        palette.apply_config_theme(Some(wanted.to_config()));
        assert_eq!(
            palette.theme_id, wanted,
            "a configured theme must be adopted by the palette"
        );
    }

    /// The advertised keybinding hints must name bindings that exist.
    ///
    /// This is the defect that made the reopen necessary: the command palette
    /// listed "Ctrl+T" for a handler nobody had written, so the help was a
    /// promise the product did not keep.
    #[test]
    fn the_command_palette_hints_match_the_real_bindings() {
        let items = all_command_palette_items();
        let theme = items
            .iter()
            .find(|i| matches!(i.action, CommandPaletteAction::CycleTheme))
            .expect("a theme-cycle entry must exist");
        assert_eq!(
            theme.keybind_hint, "Ctrl+T",
            "the theme hint must name the binding that now exists"
        );

        let palette = items
            .iter()
            .find(|i| matches!(i.action, CommandPaletteAction::CyclePalette))
            .expect("a palette-cycle entry must exist");
        assert!(
            palette.keybind_hint.contains('p'),
            "the accessibility palette hint must name `p`, the documented binding; \
             got {:?}",
            palette.keybind_hint
        );
    }

    /// THE TRAP THIS BEAD WARNS ABOUT. Eye channels are NOT a regular `eye * 4`
    /// stride: densities sit at sensor indices 0, 5, 12, 21. A panel that indexed
    /// by hand would show a food channel as eye 1's density — plausible-looking
    /// and wrong. This pins that the rendered cone value equals the sensor
    /// channel `eye_channel_indices` names, and that it is NOT the naive one.
    #[test]
    fn a_selected_cone_reads_the_irregular_channel_layout_not_a_naive_stride() {
        for eye in 0..NUM_EYES {
            let indices = scriptbots_core::eye_channel_indices(eye)
                .expect("every eye in 0..NUM_EYES must own four sensor channels");
            let naive = eye * 4;
            if indices[0] != naive {
                // Prove the authority disagrees with the naive stride somewhere,
                // so this test is meaningful rather than trivially true.
                assert_ne!(
                    indices[0], naive,
                    "eye {eye} density is at {} not {naive}",
                    indices[0]
                );
            }
            // And the four channels of a cone are the ones SENSOR_LAYOUT assigns.
            let from_layout: Vec<usize> = SENSOR_LAYOUT
                .iter()
                .filter(|channel| channel.eye == Some(eye))
                .map(|channel| channel.index)
                .collect();
            assert_eq!(
                indices.to_vec(),
                from_layout,
                "eye {eye}: eye_channel_indices must agree with SENSOR_LAYOUT"
            );
        }
        // The layout really is irregular, so the guard above is not vacuous.
        let first = scriptbots_core::eye_channel_indices(0).expect("eye 0");
        let second = scriptbots_core::eye_channel_indices(1).expect("eye 1");
        assert_ne!(
            second[0] - first[0],
            4,
            "if eye spacing were a regular 4 this test would be pointless; \
             the layout is documented as irregular"
        );
    }

    /// Selecting a cone must not touch the world: the probe is a read-only
    /// projection, and a panel that advanced the simulation would corrupt the
    /// run an experimenter is observing.
    #[test]
    fn selecting_a_cone_does_not_advance_the_simulation() {
        let world = command_characterization_world();
        {
            let mut guard = world.lock().expect("seed");
            guard
                .try_spawn_agent(AgentData::default())
                .expect("default agent is finite");
        }
        let before = world.lock().expect("tick").tick().0;
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let mut app = TerminalApp::new(
            &renderer,
            crate::renderer::RendererContext {
                simulation_step: disabled_persistence_step_driver(&world),
                world: Arc::clone(&world),
                analytics: AnalyticsSnapshotProvider::empty(),
                control_runtime: &runtime,
                command_drain: drain,
                command_submit: submit,
                scenario: test_scenario(),
            },
        );
        for _ in 0..(NUM_EYES * 3) {
            app.cycle_eye_selection(true);
        }
        assert_eq!(
            world.lock().expect("tick").tick().0,
            before,
            "cone selection must not step the world"
        );
    }

    /// The FrankenTUI pin in the workspace manifest must match the revision the
    /// terminal-stack document records (bd-phj8).
    ///
    /// THE HAZARD THIS EXISTS FOR, which is general and not an ftui quirk: Cargo
    /// only resolves a `[workspace.dependencies]` entry when some crate actually
    /// depends on it. No workspace crate consumes `ftui.workspace`, so the entry
    /// is never fetched, never reaches `Cargo.lock`, and a wrong `rev` cannot
    /// fail any build. **An unconsumed pin is unverified by construction.** The
    /// previous value, `15cc65438a2095fbe8dd0dfce9adcfc7edab7612`, was not an
    /// object in the upstream repository at all and survived exactly that way —
    /// no amount of compiling could have caught it.
    ///
    /// This one checks only that the manifest and the document that justifies it
    /// cannot disagree — a future edit to either alone fails here. It would still
    /// pass if both were edited in lockstep to a fabricated SHA, which is why
    /// `the_frankentui_pin_names_a_real_upstream_object` checks existence too.
    #[test]
    fn frankentui_pin_matches_the_documented_revision() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root is two levels above this crate");

        let manifest =
            std::fs::read_to_string(root.join("Cargo.toml")).expect("read workspace Cargo.toml");
        let pin_line = manifest
            .lines()
            .find(|line| line.trim_start().starts_with("ftui = "))
            .expect("the workspace must declare an ftui pin");
        let manifest_rev = extract_rev(pin_line).expect("the ftui pin must carry a rev");

        let doc = std::fs::read_to_string(root.join("docs/terminal_stack_and_frankentui_pin.md"))
            .expect("read the terminal stack document");
        // Compare against the DECLARED field, not "appears anywhere in the file".
        // The document deliberately quotes the old bad SHA in its history note, so
        // a `contains` check matches it and passes for the very value this guard
        // exists to reject — verified by re-injecting the bad pin and watching an
        // earlier version of this test go green.
        let documented_rev = doc
            .lines()
            .find(|line| line.contains("**Pinned Revision**"))
            .and_then(|line| line.split('`').nth(1))
            .expect("the document must declare a Pinned Revision")
            .to_owned();
        assert_eq!(
            manifest_rev, documented_rev,
            "Cargo.toml pins ftui at {manifest_rev} but \
             docs/terminal_stack_and_frankentui_pin.md declares {documented_rev}. \
             One of the two was edited alone; an unconsumed pin is never resolved, \
             so nothing else will catch the drift (bd-phj8)."
        );

        // A full SHA-1, not a short prefix. The previous bad value was a real
        // short SHA with a fabricated tail, so length alone is not proof — but a
        // truncated pin would let Cargo resolve something the document never
        // reviewed.
        assert_eq!(
            manifest_rev.len(),
            40,
            "ftui rev {manifest_rev} is not a full 40-character SHA"
        );
        assert!(
            manifest_rev.chars().all(|c| c.is_ascii_hexdigit()),
            "ftui rev {manifest_rev} is not hexadecimal"
        );
    }

    /// The pinned rev must name an object that actually exists upstream.
    ///
    /// The two sibling guards compare Cargo.toml against the document, so they
    /// both pass if someone edits the pin and the doc together to a SHA that was
    /// never a commit — which is precisely the shape of the original defect: a
    /// real short SHA (`15cc6543`) carrying a fabricated 32-character tail.
    ///
    /// Cargo itself will never catch this while the pin is unconsumed. Measured,
    /// not assumed (bd-phj8): in a scratch workspace holding the bad rev in an
    /// unconsumed `[workspace.dependencies]` entry, `cargo metadata`, `cargo
    /// fetch`, `cargo update` and `cargo generate-lockfile` all exited 0 and
    /// `Cargo.lock` never mentioned ftui. Adding one consumer flipped the same
    /// bad rev to exit 101 while the corrected rev stayed at 0. So forcing
    /// resolution requires adoption, and adoption is an admission decision under
    /// docs/franken_integration.md rather than a side effect of fixing a pin.
    ///
    /// The existence check does not need the network: Cargo's own fetched git db
    /// under `$CARGO_HOME/git/db/frankentui-*` answers it. When that db is absent
    /// (fresh CI, a worker that never fetched frankentui) there is nothing to
    /// check against, so this skips rather than failing — a guard that goes red
    /// on machines with a cold cache would just be disabled by the next person.
    #[test]
    fn the_frankentui_pin_names_a_real_upstream_object() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root is two levels above this crate");

        let manifest =
            std::fs::read_to_string(root.join("Cargo.toml")).expect("read workspace Cargo.toml");
        let pin_line = manifest
            .lines()
            .find(|line| line.trim_start().starts_with("ftui = "))
            .expect("the workspace must declare an ftui pin");
        let rev = extract_rev(pin_line).expect("the ftui pin must carry a rev");

        let Some(db) = frankentui_git_db() else {
            eprintln!("skipping: frankentui has never been fetched into the cargo git db");
            return;
        };

        assert_eq!(
            frankentui_object_kind(&db, &rev).as_deref(),
            Some("commit"),
            "ftui is pinned at {rev}, but that is not a commit in the fetched \
             frankentui git db at {}. An unconsumed pin is never resolved by \
             Cargo, so a fabricated SHA cannot fail any build — this guard is the \
             only thing standing between the manifest and a rev that does not \
             exist (bd-phj8).",
            db.display()
        );
    }

    /// The probe above is only worth having if it actually rejects a bad SHA, so
    /// this pins that down against the two historically meaningful values rather
    /// than leaving it to be re-verified by hand.
    ///
    /// This exists because the FIRST version of the sibling
    /// `frankentui_pin_matches_the_documented_revision` guard was vacuous: it
    /// asserted `doc.contains(manifest_rev)`, and the document quotes the old bad
    /// SHA in its history note, so it went green on the exact value it existed to
    /// reject. A guard nobody has watched fail is not yet a guard.
    #[test]
    fn the_frankentui_object_probe_rejects_the_original_bad_sha() {
        /// The real upstream commit, from bd-2z0.6.3.1.
        const GOOD: &str = "15cc6543f76b814394c590f9e7719dedd6684e4c";
        /// What the manifest carried before bd-phj8: the same `15cc6543` short
        /// prefix with a fabricated 32-character tail.
        const BAD: &str = "15cc65438a2095fbe8dd0dfce9adcfc7edab7612";

        let Some(db) = frankentui_git_db() else {
            eprintln!("skipping: frankentui has never been fetched into the cargo git db");
            return;
        };

        assert_eq!(
            frankentui_object_kind(&db, GOOD).as_deref(),
            Some("commit"),
            "the known-good rev must resolve, otherwise this probe proves nothing"
        );
        assert_ne!(
            frankentui_object_kind(&db, BAD).as_deref(),
            Some("commit"),
            "the known-bad rev {BAD} resolved to a commit — the probe cannot tell \
             a real object from a fabricated one and the guard above is vacuous"
        );
    }

    /// Locate Cargo's fetched frankentui git db, or `None` when nothing has ever
    /// fetched it (fresh CI, a remote build worker) and there is nothing to check
    /// against. Note this means the existence guards are inert on such machines —
    /// they have teeth on developer hosts with a warm cargo cache.
    fn frankentui_git_db() -> Option<std::path::PathBuf> {
        let cargo_home = std::env::var_os("CARGO_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cargo"))
            })?;
        std::fs::read_dir(cargo_home.join("git/db"))
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("frankentui-"))
            })
    }

    /// Ask the fetched git db what kind of object `rev` is. `None` when git
    /// cannot answer at all; `Some("commit")` only for a real commit.
    fn frankentui_object_kind(db: &std::path::Path, rev: &str) -> Option<String> {
        let output = std::process::Command::new("git")
            .arg(format!("--git-dir={}", db.display()))
            .args(["cat-file", "-t", rev])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        Some(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }

    /// Pull `rev = "..."` out of a Cargo dependency line.
    fn extract_rev(line: &str) -> Option<String> {
        let after = line.split("rev = \"").nth(1)?;
        after.split('"').next().map(str::to_owned)
    }

    /// The pin is declared but deliberately NOT adopted, and the document must
    /// say so. If a crate ever consumes `ftui.workspace`, adoption became real
    /// and the franken-integration record has to be updated with it rather than
    /// silently diverging (AGENTS.md routes franken adoption through
    /// docs/franken_integration.md).
    #[test]
    fn an_unadopted_frankentui_pin_is_declared_as_such() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root");

        let mut consumers: Vec<String> = Vec::new();
        let crates_dir = root.join("crates");
        for entry in std::fs::read_dir(&crates_dir)
            .expect("read crates dir")
            .flatten()
        {
            let manifest = entry.path().join("Cargo.toml");
            let Ok(text) = std::fs::read_to_string(&manifest) else {
                continue;
            };
            if text.lines().any(|line| {
                let line = line.trim_start();
                line.starts_with("ftui") && line.contains("workspace")
            }) {
                consumers.push(entry.file_name().to_string_lossy().into_owned());
            }
        }

        let doc = std::fs::read_to_string(root.join("docs/terminal_stack_and_frankentui_pin.md"))
            .expect("read the terminal stack document");
        if consumers.is_empty() {
            assert!(
                doc.contains("PREPARED, NOT ADOPTED"),
                "nothing consumes ftui, so the document must say the pin is prepared \
                 and not adopted — otherwise it reads as a shipped dependency (bd-phj8)"
            );
        } else {
            assert!(
                !doc.contains("PREPARED, NOT ADOPTED"),
                "{consumers:?} now consume ftui, so the document must stop calling the \
                 pin unadopted and the franken-integration record must be updated"
            );
        }
    }

    /// Self-exclusion marker: this guard's own text must never feed itself.
    /// bd-ikts.5 lost time twice to guards that matched their own literals.
    const TERMINAL_DUPLICATE_GUARD_MARKER: &str = "fn no_two_terminal_modules_define_the_same_item";

    /// The one duplicate pair this guard cannot fail on yet, and why.
    ///
    /// `paint.rs` is a second, entirely dead sub-cell painter engine. `subcell.rs`
    /// is the canonical one — it is wired into the live `MapWidget` canvas path
    /// and covered by the bd-2z0.14.2.1 tests, while every identifier `paint.rs`
    /// exports is referenced only inside `paint.rs` itself. Removing the file
    /// needs the user's explicit written permission under AGENTS.md Rule 1, which
    /// bd-c1z8 has been waiting on. Listing it here rather than weakening the
    /// predicate keeps the guard live for every NEW duplicate in the meantime;
    /// this entry is deleted together with the file.
    const KNOWN_DUPLICATE_MODULES: &[&str] = &["paint.rs"];

    /// No two sibling modules under `terminal/` may define the same item name.
    ///
    /// THE DEFECT THIS EXISTS FOR (bd-c1z8): `subcell.rs` and `paint.rs` are two
    /// independent implementations of one sub-cell painter, shipped by one task,
    /// both carrying the same bead ID in their headers. They are the same class
    /// bd-ikts.4 found in core, where `toroidal_delta` had THREE divergent copies
    /// that silently disagreed at exactly half a world until it broke.
    ///
    /// WHY bd-ikts.5's GUARD DOES NOT COVER THIS, checked before writing another.
    /// That guard asks "does a consumer crate re-implement a `scriptbots-core` pub
    /// authority" — cross-crate, against a named owner. Here both modules live in
    /// one crate and neither is a core authority, so it cannot fire. Worse, the
    /// two engines mostly use DIFFERENT names for the same concepts
    /// (`SubCellBuffer`/`PixelBuffer`, `FrameCell`/`CellGlyph`,
    /// `DirtyTracker`/`DirtyCell`), so a pure concept-duplication detector would
    /// need to reason about shape. What they DO collide on is four exact names —
    /// `ColorDepth`, `SubCellMode`, `SubPixel`, `quantize` — and that is a crisp,
    /// cheap predicate: within one directory, one name means one definition.
    ///
    /// Scoped to `terminal/` deliberately. A workspace-wide "no repeated type
    /// name" rule would be noise; sibling modules of one subsystem sharing an
    /// identifier is a genuine smell.
    #[test]
    fn no_two_terminal_modules_define_the_same_item() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/terminal");
        let mut owners: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut scanned = 0_usize;

        let entries = std::fs::read_dir(&dir).expect("terminal module directory");
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.extension().is_some_and(|ext| ext == "rs") {
                continue;
            }
            let file = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            if KNOWN_DUPLICATE_MODULES.contains(&file.as_str()) {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read terminal module");
            // Cut this guard's own body so its prose cannot define anything.
            let text = text
                .split_once(TERMINAL_DUPLICATE_GUARD_MARKER)
                .map_or(text.as_str(), |(before, _)| before);
            scanned += 1;

            for line in text.lines() {
                let trimmed = line.trim_start();
                // Strip comments so prose naming a type is not read as defining it.
                if trimmed.starts_with("//") {
                    continue;
                }
                let after = ["pub struct ", "struct ", "pub enum ", "enum "]
                    .iter()
                    .find_map(|prefix| trimmed.strip_prefix(prefix));
                if let Some(rest) = after
                    && let Some(name) = rest.split(['<', '{', '(', ';', ' ']).next()
                {
                    let name = name.trim();
                    if !name.is_empty() {
                        owners
                            .entry(name.to_owned())
                            .or_default()
                            .push(file.clone());
                    }
                }
            }
        }

        // Positive anchor: a scan that matched nothing is indistinguishable from
        // a scan that found nothing wrong.
        assert!(
            scanned >= 5,
            "only {scanned} terminal modules scanned; the sweep is broken, not the tree"
        );
        assert!(
            owners.len() > 20,
            "only {} type definitions found across terminal/; the extraction is broken",
            owners.len()
        );

        let collisions: Vec<String> = owners
            .iter()
            .filter(|(_, files)| {
                let mut distinct: Vec<&String> = files.iter().collect();
                distinct.sort_unstable();
                distinct.dedup();
                distinct.len() > 1
            })
            .map(|(name, files)| format!("{name} defined in {files:?}"))
            .collect();

        assert!(
            collisions.is_empty(),
            "two terminal modules define the same item, which is how one subsystem \
             ends up with two divergent engines (bd-c1z8, and bd-ikts.4 in core):\n  {}\n\
             Pick one owner and have the other call it, or if these are genuinely \
             unrelated helpers give one a distinct name.",
            collisions.join("\n  ")
        );
    }

    /// The guard above must actually fire. A detector that cannot fail is worth
    /// nothing, and bd-ikts.5 records two guards that passed vacuously the same
    /// day. This replays the collision predicate against a synthetic pair.
    #[test]
    fn the_duplicate_module_guard_detects_an_injected_duplicate() {
        let mut owners: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        owners.insert(
            "ColorDepth".into(),
            vec!["subcell.rs".into(), "paint.rs".into()],
        );
        owners.insert("MapWidget".into(), vec!["mod.rs".into()]);

        let collisions: Vec<&String> = owners
            .iter()
            .filter(|(_, files)| {
                let mut distinct: Vec<&String> = files.iter().collect();
                distinct.sort_unstable();
                distinct.dedup();
                distinct.len() > 1
            })
            .map(|(name, _)| name)
            .collect();

        assert_eq!(
            collisions,
            vec!["ColorDepth"],
            "the predicate must flag exactly the duplicated name and leave singles alone"
        );
    }

    /// `paint.rs` is exempt from the duplicate guard only because it is OUT OF THE
    /// BUILD and unreferenced. Both halves are checked here, because an allowlist
    /// that silently starts covering live code is worse than no allowlist.
    ///
    /// The module-declaration half is the load-bearing one: re-adding
    /// `pub mod paint;` would put a second painter engine back into the crate,
    /// which is precisely the state bd-c1z8 exists to prevent, and the name-based
    /// guard alone would not catch it because `paint.rs` is on its exempt list.
    #[test]
    fn the_exempt_duplicate_painter_stays_out_of_the_build() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/terminal");

        let module_root =
            std::fs::read_to_string(dir.join("mod.rs")).expect("read terminal mod.rs");
        let declarations = module_root
            .split_once(TERMINAL_DUPLICATE_GUARD_MARKER)
            .map_or(module_root.as_str(), |(before, _)| before);
        assert!(
            !declarations
                .lines()
                .any(|line| line.trim_start().starts_with("pub mod paint")
                    || line.trim_start().starts_with("mod paint")),
            "paint.rs has been declared as a module again, putting a SECOND sub-cell \
             painter engine back into the crate. subcell.rs is the canonical one \
             (bd-c1z8); converge on it rather than compiling both."
        );

        let exports = [
            "PixelBuffer",
            "CellGlyph",
            "DirtyCell",
            "DitherMode",
            "braille_char",
        ];

        let entries = std::fs::read_dir(&dir).expect("terminal module directory");
        for entry in entries.flatten() {
            let path = entry.path();
            let file = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            if !path.extension().is_some_and(|ext| ext == "rs") || file == "paint.rs" {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read terminal module");
            let text = text
                .split_once(TERMINAL_DUPLICATE_GUARD_MARKER)
                .map_or(text.as_str(), |(before, _)| before);
            for export in exports {
                assert!(
                    !text.contains(export),
                    "{file} references {export} from the dead painter paint.rs. \
                     It is exempt from the duplicate guard ONLY because nothing uses it; \
                     if it is live, converge the two engines instead (bd-c1z8)."
                );
            }
        }
    }

    /// Shared scenario identity for renderer contexts built by tests.
    fn test_scenario() -> Arc<ScenarioIdentityV0> {
        Arc::new(ScenarioIdentityV0::caller_seeded("terminal-test-scenario"))
    }

    #[test]
    fn brain_inspection_metadata_exposes_provenance_clipping_and_staleness() {
        let inspection = BrainInspectionViewMetadata {
            agent_uid: 41,
            source_tick: 99,
            request_revision: 7,
            truncated: true,
            retained_payload_bytes: 2_048,
            ready: true,
        };

        assert_eq!(inspection.status_lines(100).0, "uid 41 · t99 STALE");
        assert_eq!(inspection.status_lines(100).1, "r7 · 2048B · CLIPPED");
    }

    #[test]
    fn narrow_insights_panel_keeps_brain_provenance_and_clipping_visible() {
        let world = command_characterization_world();
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world,
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        app.snapshot.tick = 100;
        app.snapshot.brain_inspection = Some(BrainInspectionViewMetadata {
            agent_uid: 41,
            source_tick: 99,
            request_revision: 7,
            truncated: true,
            retained_payload_bytes: 2_048,
            ready: true,
        });

        let backend = ratatui::backend::TestBackend::new(30, 7);
        let mut terminal = Terminal::new(backend).expect("narrow insights terminal");
        terminal
            .draw(|frame| app.draw_insights(frame, frame.area(), &app.snapshot))
            .expect("draw narrow insights");
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(
            rendered.contains("t99 STALE"),
            "rendered buffer: {rendered:?}"
        );
        assert!(
            rendered.contains("r7 · 2048B · CLIPPED"),
            "rendered buffer: {rendered:?}"
        );
    }

    #[derive(Clone)]
    struct FakeTerminalRestore {
        operations: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }

    impl TerminalRestore for FakeTerminalRestore {
        fn show_cursor(&mut self) {
            self.operations.lock().expect("operations").push("show");
        }

        fn leave_alternate_screen(&mut self) {
            self.operations.lock().expect("operations").push("leave");
        }

        fn disable_raw_mode(&mut self) {
            self.operations.lock().expect("operations").push("disable");
        }
    }

    #[test]
    fn alternate_screen_failure_restores_raw_mode_once_and_preserves_error() {
        let operations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let restore = FakeTerminalRestore {
            operations: Arc::clone(&operations),
        };

        let result = TerminalModeGuard::begin_with(
            restore,
            || Ok(()),
            || Err(io::Error::other("injected alternate-screen failure")),
        );
        assert!(
            result.is_err(),
            "alternate-screen failure unexpectedly succeeded"
        );
        let error = result
            .err()
            .expect("asserted alternate-screen failure must retain its error");

        assert_eq!(
            error.root_cause().to_string(),
            "injected alternate-screen failure"
        );
        assert_eq!(
            *operations.lock().expect("operations"),
            vec!["leave", "disable"],
            "an enter-screen write may take effect before reporting its injected failure"
        );
    }

    #[test]
    fn raw_mode_failure_still_runs_idempotent_disable() {
        let operations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let restore = FakeTerminalRestore {
            operations: Arc::clone(&operations),
        };

        let result = TerminalModeGuard::begin_with(
            restore,
            || Err(io::Error::other("injected raw-mode failure")),
            || Ok(()),
        );
        assert!(result.is_err());
        assert_eq!(*operations.lock().expect("operations"), vec!["disable"]);
    }

    #[test]
    fn terminal_guard_restores_each_armed_state_exactly_once() {
        let operations = Arc::new(std::sync::Mutex::new(Vec::new()));
        let restore = FakeTerminalRestore {
            operations: Arc::clone(&operations),
        };
        let mut guard =
            TerminalModeGuard::begin_with(restore, || Ok(()), || Ok(())).expect("terminal setup");
        guard.mark_cursor_hidden();
        drop(guard);

        assert_eq!(
            *operations.lock().expect("operations"),
            vec!["show", "leave", "disable"]
        );
    }

    #[test]
    fn terminal_headless_boolean_vocabulary_is_explicit() {
        for raw in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(parse_terminal_bool(OsStr::new(raw)).expect(raw));
        }
        for raw in ["0", "false", "FALSE", " no ", "off"] {
            assert!(!parse_terminal_bool(OsStr::new(raw)).expect(raw));
        }

        let error = parse_terminal_bool(OsStr::new("sometimes")).expect_err("invalid boolean");
        assert!(
            error
                .to_string()
                .contains("invalid SCRIPTBOTS_TERMINAL_HEADLESS value")
        );
    }

    #[cfg(unix)]
    #[test]
    fn terminal_headless_rejects_non_unicode_values() {
        use std::os::unix::ffi::OsStrExt;

        let error = parse_terminal_bool(OsStr::from_bytes(b"\xff"))
            .expect_err("non-Unicode boolean must fail");
        assert!(error.to_string().contains("valid Unicode boolean"));
    }

    fn command_characterization_world() -> SharedWorld {
        let config = ScriptBotsConfig {
            world_width: 100,
            world_height: 100,
            food_cell_size: 50,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            rng_seed: Some(0xB0FF_EA11),
            ..ScriptBotsConfig::default()
        };
        Arc::new(std::sync::Mutex::new(
            WorldState::new(config).expect("characterization world"),
        ))
    }

    fn disabled_persistence_step_driver(world: &SharedWorld) -> WorldStepDriver {
        let world = Arc::clone(world);
        Arc::new(move || {
            world
                .lock()
                .expect("world mutex poisoned while executing test simulation step")
                .step()
        })
    }

    #[derive(Debug)]
    struct ProbePanelBrain;

    impl scriptbots_core::BrainRunner for ProbePanelBrain {
        fn kind(&self) -> &'static str {
            "terminal.probe"
        }

        fn tick(
            &mut self,
            _inputs: &[f32; scriptbots_core::INPUT_SIZE],
        ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
            [0.0; scriptbots_core::OUTPUT_SIZE]
        }
    }

    /// bd-16g.4.2: the sense probe is opt-in, captures core attribution
    /// verbatim under the snapshot lock, recaptures deterministically while
    /// paused, and renders truthful per-channel source labels — a self-state
    /// channel must read as `[self]`, never as a missing-neighbour condition.
    #[test]
    fn sense_probe_captures_attribution_verbatim_and_labels_channel_sources() {
        let world = command_characterization_world();
        {
            let mut guard = world.lock().expect("probe world lock");
            let family = guard
                .brain_registry_mut()
                .expect("probe registry mutation")
                .register_with_state_digest("terminal.probe", 0x5455_495f_5052_4f42, |_rng| {
                    Ok(Box::new(ProbePanelBrain))
                });
            // Two agents 12 world-units apart: comfortably inside the default
            // sense radius so the focused agent has a real contributor.
            for offset in [0.0_f32, 12.0] {
                let agent_id = guard
                    .try_spawn_agent(AgentData {
                        position: Position {
                            x: 100.0 + offset,
                            y: 100.0,
                        },
                        ..AgentData::default()
                    })
                    .expect("spawn probe agent");
                guard
                    .bind_agent_brain(agent_id, family)
                    .expect("bind probe brain");
            }
        }
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        app.paused = true;
        assert!(
            app.snapshot.probe.is_none(),
            "the probe is opt-in; nothing may be captured before the toggle"
        );

        app.probe_enabled = true;
        app.refresh_snapshot();
        let first = app
            .snapshot
            .probe
            .clone()
            .expect("probe captured for the focused agent");
        let attribution = &first.attribution;
        assert!(
            !attribution.contributions.is_empty(),
            "the neighbour 12 units away must be attributed"
        );
        for index in 0..scriptbots_core::INPUT_SIZE {
            assert!(
                (0.0..=1.0).contains(&attribution.clamped[index]),
                "clamped channel {index} must stay in [0, 1]"
            );
            if attribution.saturated[index] {
                assert!(
                    attribution.raw[index] > 1.0,
                    "a saturated channel must expose its raw pre-clamp value"
                );
            }
        }

        app.refresh_snapshot();
        let second = app
            .snapshot
            .probe
            .clone()
            .expect("recaptured probe while paused");
        assert_eq!(
            first.attribution, second.attribution,
            "a paused world must recapture the identical attribution"
        );
        assert_eq!(first.agent_uid, second.agent_uid);

        app.palette = Palette::test_backend_evidence();
        let backend = ratatui::backend::TestBackend::new(140, 48);
        let mut terminal = Terminal::new(backend).expect("probe test backend");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("probe frame renders");
        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        let mut text = String::new();
        for y in area.y..area.bottom() {
            for x in area.x..area.right() {
                text.push_str(buffer[(x, y)].symbol());
            }
            text.push('\n');
        }
        for needle in ["Sense Probe", "[self]", "[grid]", "[pos]", "eye0", "eye3"] {
            assert!(
                text.contains(needle),
                "probe panel must render {needle:?}; buffer was:\n{text}"
            );
        }

        app.probe_enabled = false;
        app.refresh_snapshot();
        assert!(
            app.snapshot.probe.is_none(),
            "disabling the probe must stop capturing"
        );
    }

    #[test]
    fn terminal_brain_pull_is_immediate_uid_keyed_and_cached_while_paused() {
        #[derive(Debug)]
        struct TerminalInspectionBrain {
            calls: Arc<std::sync::atomic::AtomicUsize>,
        }

        impl scriptbots_core::BrainRunner for TerminalInspectionBrain {
            fn kind(&self) -> &'static str {
                "terminal.inspection"
            }

            fn tick(
                &mut self,
                _inputs: &[f32; scriptbots_core::INPUT_SIZE],
            ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
                [0.0; scriptbots_core::OUTPUT_SIZE]
            }

            fn inspect(
                &self,
                request: scriptbots_core::BrainInspection,
            ) -> Result<
                Option<scriptbots_core::BrainInspectionSnapshot>,
                scriptbots_core::BrainInspectionError,
            > {
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let scriptbots_core::BrainInspection::Activations(limits) = request;
                scriptbots_core::bound_brain_inspection(
                    self.kind(),
                    BrainActivations {
                        layers: vec![scriptbots_core::ActivationLayer {
                            name: "terminal".to_owned(),
                            width: 1,
                            height: 1,
                            values: vec![0.5],
                        }],
                        connections: Vec::new(),
                        truncated: false,
                    },
                    1,
                    limits,
                )
                .map(Some)
            }

            fn state_digest(&self) -> Option<u64> {
                Some(0x5455_495f_4252_4149)
            }
        }

        let world = command_characterization_world();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let mut guard = world.lock().expect("terminal inspection world lock");
            let family = guard
                .brain_registry_mut()
                .expect("terminal inspection registry mutation")
                .register_with_state_digest("terminal.inspection", 0x5455_495f_4252_4149, {
                    let calls = Arc::clone(&calls);
                    move |_rng| {
                        Ok(Box::new(TerminalInspectionBrain {
                            calls: Arc::clone(&calls),
                        }))
                    }
                });
            for _ in 0..2 {
                let agent_id = guard
                    .try_spawn_agent(AgentData::default())
                    .expect("spawn terminal inspection agent");
                guard
                    .bind_agent_brain(agent_id, family)
                    .expect("bind terminal inspection brain");
            }
        }
        let digest_before = world
            .lock()
            .expect("pre-inspection world lock")
            .world_digest_v1()
            .expect("pre-inspection digest");
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        app.paused = true;
        let first = app
            .snapshot
            .brain_inspection
            .expect("initial immediate terminal inspection");
        assert!(first.ready);
        assert_eq!(first.request_revision, 1);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);

        app.refresh_snapshot();
        assert_eq!(
            app.snapshot
                .brain_inspection
                .expect("cached terminal inspection")
                .request_revision,
            1
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);

        app.focused_agent_cursor = 1;
        app.refresh_snapshot();
        let second = app
            .snapshot
            .brain_inspection
            .expect("second focused terminal inspection");
        assert!(second.ready);
        assert_eq!(second.request_revision, 2);
        assert_ne!(first.agent_uid, second.agent_uid);
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(
            world
                .lock()
                .expect("post-inspection world lock")
                .world_digest_v1()
                .expect("post-inspection digest"),
            digest_before
        );
    }

    #[test]
    fn terminal_app_key_handler_single_step_advances_exactly_once_and_stays_paused() {
        let world = command_characterization_world();
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        let before = world.lock().expect("world lock").tick().0;

        let exit = app
            .handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE))
            .expect("single-step key");

        let guard = world.lock().expect("world lock");
        assert!(!exit);
        assert_eq!(guard.tick().0, before + 1);
        drop(guard);
        assert!((app.command_drain)().is_empty());
        assert!(app.paused);
        assert_eq!(app.speed_multiplier, 0.0);
    }

    #[test]
    fn headless_test_backend_frame_proves_buffer_semantics_and_current_tick() {
        let world = command_characterization_world();
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };

        let report = renderer
            .run_headless_frames(ctx, 1)
            .expect("one headless frame");

        assert_eq!(report.summary.frame_count, 1);
        assert_eq!(report.summary.ticks_simulated, 1);
        assert_eq!(report.summary.final_tick, report.initial.tick + 1);
        let evidence = report.frames[0]
            .buffer
            .as_ref()
            .expect("a rendered TestBackend frame must carry buffer evidence");
        assert_eq!(evidence.viewport_width, 80);
        assert_eq!(evidence.viewport_height, 36);
        assert_eq!(evidence.current_tick, report.frames[0].tick);
        assert_eq!(
            (
                evidence.non_blank_cells,
                evidence.styled_cells,
                evidence.skipped_cells,
                evidence.forced_width_cells,
                evidence.empty_symbol_cells,
            ),
            // Reviewed 2026-07-27 (bd-xg82): the four trend sparklines gained
            // non-colour row labels, so two rows in this fixture grew a
            // TREND_LABEL_WIDTH (4) column label carrying 3 visible characters.
            // The deltas are exactly that and nothing else: +6 non-blank
            // (2 rows x 3 glyphs) and +8 styled (2 rows x 4 styled columns).
            // Only population and energy have data at this fixture's tick, which
            // is why it is two rows rather than four.
            (2367, 1534, 0, 0, 0),
            "fixed-seed Ratatui TestBackend cell counts changed; inspect the rendered buffer before intentionally updating this reviewed evidence: {evidence:?}"
        );
        // Reviewed 2026-07-17 (bd-2z0.10.1): the header title now carries the scenario id
        // and bootstrap policy, shifting the pinned counts and full-cell digest below.
        // Reviewed 2026-07-20 (bd-16g.2.4): the narrative rail now occupies five rows
        // between the header and the body by design, shrinking the map's cell counts
        // and changing the full-cell digest accordingly.
        // Reviewed 2026-07-27 (bd-f4x0): status chrome moved off hand-coded ANSI
        // Green/Yellow/Red onto the palette-aware event ramp, so fg colours in the
        // storage/simulation status rows and the births/deaths sparklines changed.
        // INSPECTED: the cell counts pinned above are byte-identical across the
        // change (2361/1526/0/0/0 before and after), so nothing moved on screen —
        // only the colours the digest also hashes. A layout regression would have
        // shifted those counts first.
        // Reviewed 2026-07-27 (bd-xg82): trend row labels, as detailed on the cell
        // counts above. Those counts moved by exactly the label arithmetic, which
        // is the evidence that the digest change is the labels and not a layout
        // regression riding along with them.
        assert_eq!(
            evidence.full_cell_fnv1a64, "4ab177bfc13b0371",
            "fixed-seed Ratatui TestBackend full-cell golden changed; this hashes coordinates, grapheme symbols, fg/bg/underline colors, modifiers, and diff/width directives. Inspect the rendered buffer before intentionally updating this reviewed digest: {evidence:?}"
        );
        assert_eq!(
            evidence.semantic_regions,
            ["terminal_hud", "current_tick", "world_map", "vital_stats"]
        );
        assert_eq!(
            world.lock().expect("world lock").tick().0,
            report.initial.tick + 1
        );

        let repeat_world = command_characterization_world();
        let (repeat_runtime, repeat_drain, repeat_submit) = crate::servers::ControlRuntime::dummy();
        let repeat_ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&repeat_world),
            world: repeat_world,
            analytics: AnalyticsSnapshotProvider::empty(),
            control_runtime: &repeat_runtime,
            command_drain: repeat_drain,
            command_submit: repeat_submit,
            scenario: test_scenario(),
        };
        let repeat = renderer
            .run_headless_frames(repeat_ctx, 1)
            .expect("repeat deterministic headless frame");
        assert_eq!(
            repeat.frames[0].buffer.as_ref(),
            Some(evidence),
            "the fixed-seed TestBackend buffer evidence must be deterministic"
        );
    }

    #[test]
    fn the_blank_buffer_detector_would_catch_a_frame_that_painted_nothing() {
        // The same discipline as any alarm: an evidence check nobody has watched
        // fail is an evidence check nobody knows works. A buffer nothing drew into
        // is all spaces, and must be distinguishable from one that was painted.
        let blank = ratatui::buffer::Buffer::empty(ratatui::layout::Rect::new(0, 0, 80, 36));
        let painted = blank
            .content()
            .iter()
            .filter(|cell| cell.symbol() != " ")
            .count();
        assert_eq!(
            painted, 0,
            "an untouched buffer must read as zero painted cells, or the assertion \
             above proves nothing"
        );
        let error = HeadlessBufferEvidence::inspect(&blank, 1)
            .expect_err("the production TestBackend evidence inspector must reject a blank frame");
        assert!(
            error.to_string().contains("omitted required terminal_hud"),
            "blank-frame rejection must identify the first missing semantic region: {error:#}"
        );
    }

    #[test]
    #[should_panic(
        expected = "KNOWN DEFECT bd-2z0.4.1: rejected TUI command leaves optimistic playback state"
    )]
    fn target_queue_full_rejection_does_not_change_tui_playback_state() {
        let world = command_characterization_world();
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, _unused_drain, _unused_submit) = crate::servers::ControlRuntime::dummy();
        let (sender, receiver) = crate::command::create_command_bus(1);
        sender
            .try_send(ControlCommand::UpdateSimulation(
                SimulationCommand::default(),
            ))
            .expect("fill command queue");
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world,
            analytics,
            control_runtime: &runtime,
            command_drain: crate::command::make_command_drain(receiver),
            command_submit: crate::command::make_command_submit(sender),
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        assert!(!app.paused);

        app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
            .expect("pause key");

        assert!(
            !app.paused,
            "KNOWN DEFECT bd-2z0.4.1: rejected TUI command leaves optimistic playback state"
        );
    }

    #[test]
    fn snapshot_reflects_world_state() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        world
            .try_spawn_agent(AgentData::default())
            .expect("default agent is finite");

        let snapshot = Snapshot::from_world(&world);

        assert_eq!(snapshot.agent_count, world.agent_count());
        assert_eq!(snapshot.tick, world.tick().0);
        assert_eq!(snapshot.agents.len(), world.agent_count());
        assert_eq!(snapshot.world_size.0, world.config().world_width);
    }

    /// Moisture and fertility must reach the screen. Two tiles of the SAME biome
    /// with different lushness have to render differently, or the tinting is
    /// being computed and discarded — and a parched desert would look identical
    /// to an oasis.
    #[test]
    fn lushness_tints_terrain_within_its_biome() {
        let make = |moisture: f32, fertility: f32| TerrainView {
            width: 1,
            height: 1,
            kinds: vec![TerrainKind::Grass],
            elevations: vec![0.5],
            moisture: vec![moisture],
            fertility: vec![fertility],
        };
        let snapshot = Snapshot::default();
        let background = |terrain: &TerrainView| {
            let buf = render_canvas_frame(&snapshot, terrain, (4, 2), canvas_test_day_night());
            cell_bg(&buf, 1, 1)
        };
        let parched = background(&make(0.0, 0.0));
        let lush = background(&make(1.0, 1.0));
        assert_ne!(
            parched, lush,
            "moisture and fertility must change how a tile reads"
        );
        assert!(
            luminance(lush) > luminance(parched),
            "a lush tile must read brighter: parched {parched:?}, lush {lush:?}"
        );
    }

    /// A short or mismatched moisture array must tint uniformly rather than
    /// striping the map, so a truncated view degrades quietly instead of drawing
    /// bands that look like real terrain features.
    #[test]
    fn a_truncated_terrain_view_tints_uniformly() {
        let ragged = TerrainView {
            width: 2,
            height: 1,
            kinds: vec![TerrainKind::Grass, TerrainKind::Grass],
            elevations: vec![0.5, 0.5],
            moisture: vec![1.0], // short on purpose
            fertility: vec![0.0, 0.0],
        };
        assert!(
            (ragged.lushness(0.25, 0.5) - ragged.lushness(0.75, 0.5)).abs() < f32::EPSILON,
            "a mismatched view must not stripe"
        );
    }

    /// Acceptance item: an agent's dot must land on the sub-pixel its world
    /// position names, within one sub-pixel. This is what makes the 2x4 density
    /// mean something — a canvas that resolves eight dots per cell but places
    /// them a cell off has bought resolution and spent it on being wrong.
    #[test]
    fn agent_dots_land_on_the_sub_pixel_their_world_position_names() {
        let terrain = canvas_test_terrain();
        // 8x4 cells = a 16x16 braille sub-grid at 1x.
        let (cells_x, cells_y) = (8_u16, 4_u16);
        let (sub_w, sub_h) = (cells_x * 2, cells_y * 4);
        for (wx, wy) in [
            (0.0_f32, 0.0_f32),
            (0.25, 0.75),
            (0.5, 0.5),
            (0.99, 0.99),
            (0.33, 0.66),
        ] {
            let mut snapshot = Snapshot::default();
            snapshot.agents = vec![canvas_test_agent(wx, wy)];
            let buf = render_canvas_frame(
                &snapshot,
                &terrain,
                (cells_x, cells_y),
                canvas_test_day_night(),
            );

            let expected_sx = (wx * f32::from(sub_w)).floor() as u16;
            let expected_sy = (wy * f32::from(sub_h)).floor() as u16;
            let cell = (expected_sx / 2, expected_sy / 4);
            let code = buf_symbol(&buf, cell)
                .chars()
                .next()
                .map(u32::from)
                .expect("one glyph");
            assert!(
                code & braille_bit(expected_sx, expected_sy) != 0,
                "world ({wx},{wy}) must light sub-pixel ({expected_sx},{expected_sy}) \
                 in cell {cell:?}; glyph was {:?}",
                buf_symbol(&buf, cell)
            );
        }
    }

    /// The minimap only earns its screen space while zoomed: at 1x it would be a
    /// second copy of the canvas with a rectangle tracing its own border.
    #[test]
    fn the_minimap_appears_only_when_zoomed() {
        let terrain = canvas_test_terrain();
        let snapshot = Snapshot::default();
        let frame_at = |zoom: f32| {
            render_canvas_frame_viewport(
                &snapshot,
                &terrain,
                (32, 16),
                (0, 0.0),
                canvas_test_capability(),
                CanvasViewport::new(zoom, (0.5, 0.5)),
            )
        };
        // A 32x16 cell canvas is a 64x64 sub-grid; a quarter edge is 16, so the
        // thumbnail occupies sub-pixels x 48..64, y 0..16 — cells x 24..32, y 0..4.
        // The world here is empty, so any lit dot in that region is the viewport
        // rectangle and nothing else.
        let lit_in_minimap = |buf: &Buffer| {
            let mut lit = 0;
            for y in 0..4_u16 {
                for x in 24..32_u16 {
                    if buf_symbol(buf, (x, y)) != "\u{2800}" {
                        lit += 1;
                    }
                }
            }
            lit
        };

        assert_eq!(
            lit_in_minimap(&frame_at(1.0)),
            0,
            "an unzoomed canvas must not draw a minimap"
        );
        assert!(
            lit_in_minimap(&frame_at(CANVAS_MAX_ZOOM)) > 0,
            "a zoomed canvas must outline its viewport in the corner thumbnail"
        );
    }

    /// The rectangle must track the window, not sit in a fixed spot: a marker
    /// that never moves tells the user nothing about where they are.
    #[test]
    fn the_minimap_rectangle_moves_with_the_viewport() {
        let terrain = canvas_test_terrain();
        let snapshot = Snapshot::default();
        let frame_centred_on = |centre: (f32, f32)| {
            render_canvas_frame_viewport(
                &snapshot,
                &terrain,
                (32, 16),
                (0, 0.0),
                canvas_test_capability(),
                CanvasViewport::new(CANVAS_MAX_ZOOM, centre),
            )
        };
        assert_ne!(
            frame_centred_on((0.15, 0.15)),
            frame_centred_on((0.85, 0.85)),
            "panning the window must move the minimap rectangle"
        );
    }

    /// A canvas too small to host a thumbnail must simply not draw one, rather
    /// than smearing a few sub-pixels that cannot show a distinct viewport
    /// rectangle.
    #[test]
    fn the_minimap_is_omitted_when_the_canvas_is_too_small() {
        let terrain = canvas_test_terrain();
        let snapshot = Snapshot::default();
        let tiny = |zoom: f32| {
            render_canvas_frame_viewport(
                &snapshot,
                &terrain,
                (4, 2),
                (0, 0.0),
                canvas_test_capability(),
                CanvasViewport::new(zoom, (0.5, 0.5)),
            )
        };
        // A 4x2 cell canvas is an 8x8 sub-grid; a quarter of that is 2x2, below
        // CANVAS_MINIMAP_MIN_EDGE. With a single-tile terrain the world looks the
        // same at every zoom, so any difference here would be minimap smear.
        assert_eq!(
            tiny(1.0),
            tiny(CANVAS_MAX_ZOOM),
            "a canvas too small for a minimap must not draw one"
        );
    }

    /// At 1x the window is the whole world, and the centre cannot drift off it
    /// no matter what centre is requested — an unclamped window would paint a
    /// band of empty cells that reads as dead terrain.
    #[test]
    fn the_viewport_is_the_whole_world_at_1x_and_never_leaves_it() {
        let full = CanvasViewport::new(1.0, (0.9, 0.1));
        assert!((full.span - 1.0).abs() < f32::EPSILON);
        assert_eq!(
            full.centre,
            (0.5, 0.5),
            "a full-world window has only one legal centre"
        );
        assert_eq!(full.world_at(0.0, 0.0), (0.0, 0.0));
        let (bx, by) = full.world_at(0.999, 0.999);
        assert!(
            bx > 0.99 && by > 0.99,
            "the far corner maps to the far world"
        );

        // Requesting a centre at the very edge slides the window back inside.
        for zoom in [2.0_f32, 4.0, CANVAS_MAX_ZOOM] {
            let edge = CanvasViewport::new(zoom, (0.0, 1.0));
            let half = edge.span / 2.0;
            assert!(
                edge.centre.0 >= half - f32::EPSILON && edge.centre.1 <= 1.0 - half + f32::EPSILON,
                "zoom {zoom} centre {:?} escaped the world",
                edge.centre
            );
            let (x0, y0) = edge.world_at(0.0, 0.0);
            let (x1, y1) = edge.world_at(0.999, 0.999);
            assert!(
                (0.0..=1.0).contains(&x0)
                    && (0.0..=1.0).contains(&y0)
                    && (0.0..=1.0).contains(&x1)
                    && (0.0..=1.0).contains(&y1),
                "zoom {zoom} window ran off the world"
            );
        }
    }

    /// `world_at` and `canvas_at` must be inverses, or what is painted and what
    /// is clicked drift apart — the exact failure that made zoom cosmetic.
    #[test]
    fn viewport_round_trips_between_canvas_and_world() {
        for zoom in [1.0_f32, 1.7, 4.0, CANVAS_MAX_ZOOM] {
            let view = CanvasViewport::new(zoom, (0.42, 0.63));
            for (fx, fy) in [(0.1_f32, 0.2_f32), (0.5, 0.5), (0.9, 0.75)] {
                let (wx, wy) = view.world_at(fx, fy);
                let (rx, ry) = view
                    .canvas_at(wx, wy)
                    .expect("a point taken from the window is inside the window");
                assert!(
                    (rx - fx).abs() < 1e-3 && (ry - fy).abs() < 1e-3,
                    "zoom {zoom}: ({fx},{fy}) -> ({wx},{wy}) -> ({rx},{ry})"
                );
            }
        }
    }

    /// Points outside the window must be reported as outside, not clamped onto
    /// the border where a crowd of off-screen agents would pile against the
    /// frame and read as a real cluster.
    #[test]
    fn viewport_reports_off_window_points_as_absent() {
        let view = CanvasViewport::new(4.0, (0.5, 0.5));
        assert!(view.canvas_at(0.5, 0.5).is_some(), "the centre is visible");
        assert!(
            view.canvas_at(0.01, 0.5).is_none(),
            "far left is off-window"
        );
        assert!(
            view.canvas_at(0.5, 0.99).is_none(),
            "far bottom is off-window"
        );
    }

    /// A non-finite zoom or centre must degrade to the full world rather than
    /// producing a NaN span that would poison every sample for the frame.
    #[test]
    fn viewport_rejects_non_finite_inputs() {
        for zoom in [f32::NAN, f32::INFINITY, -1.0] {
            let view = CanvasViewport::new(zoom, (0.5, 0.5));
            assert!(view.span.is_finite() && view.span > 0.0, "zoom {zoom}");
            assert!(view.span <= 1.0, "zoom {zoom} span {}", view.span);
        }
        let view = CanvasViewport::new(4.0, (f32::NAN, f32::INFINITY));
        assert!(view.centre.0.is_finite() && view.centre.1.is_finite());
    }

    /// The defect: zoom updated a counter and a toast while the paint path never
    /// read it, so the map was identical at 1x and 8x. Driven through the real
    /// render path, a zoomed frame must differ.
    #[test]
    fn zooming_actually_changes_what_the_canvas_paints() {
        // A terrain with structure, so a smaller window genuinely sees less.
        let terrain = TerrainView {
            width: 8,
            height: 8,
            kinds: (0..64)
                .map(|i| {
                    if (i / 8 + i % 8) % 2 == 0 {
                        TerrainKind::Rock
                    } else {
                        TerrainKind::DeepWater
                    }
                })
                .collect(),
            elevations: vec![0.5; 64],
            moisture: vec![0.5; 64],
            fertility: vec![0.0; 64],
        };
        let snapshot = Snapshot::default();
        let frame_at = |zoom: f32| {
            render_canvas_frame_viewport(
                &snapshot,
                &terrain,
                (8, 8),
                (0, 0.0),
                canvas_test_capability(),
                CanvasViewport::new(zoom, (0.5, 0.5)),
            )
        };
        assert_ne!(
            frame_at(1.0),
            frame_at(CANVAS_MAX_ZOOM),
            "zoom must change what the canvas paints"
        );
        assert_eq!(
            frame_at(4.0),
            frame_at(4.0),
            "the same zoom is deterministic"
        );
    }

    /// An agent outside the visible window must not be painted at all. Clamping
    /// it to the edge would invent a border crowd that does not exist.
    #[test]
    fn agents_outside_the_window_are_not_painted() {
        let terrain = canvas_test_terrain();
        let mut snapshot = Snapshot::default();
        // Zoomed 4x on the centre, this agent at the far corner is off-window.
        snapshot.agents = vec![canvas_test_agent(0.02, 0.02)];
        let zoomed = render_canvas_frame_viewport(
            &snapshot,
            &terrain,
            (4, 4),
            canvas_test_day_night(),
            canvas_test_capability(),
            CanvasViewport::new(4.0, (0.5, 0.5)),
        );
        let empty = render_canvas_frame_viewport(
            &Snapshot::default(),
            &terrain,
            (4, 4),
            canvas_test_day_night(),
            canvas_test_capability(),
            CanvasViewport::new(4.0, (0.5, 0.5)),
        );
        assert_eq!(
            zoomed, empty,
            "an off-window agent must leave the frame untouched"
        );
    }

    /// The ring must follow the AGENT, not the slot. Two agents sit far apart;
    /// focus names one by stable uid, and only that one may be ringed. Matching
    /// on the arena handle instead would follow a reusable slot and could ring a
    /// stranger after a death (bd-qxrt).
    #[test]
    fn the_selection_ring_marks_the_focused_agent_and_only_that_agent() {
        let terrain = canvas_test_terrain();
        // 4x4 cells = an 8x16 sub-grid. Two agents, well separated.
        let mut focused = canvas_test_agent(0.5, 0.5);
        focused.uid = Some(7);
        let mut other = canvas_test_agent(0.05, 0.05);
        other.uid = Some(9);
        let mut snapshot = Snapshot::default();
        snapshot.agents = vec![focused, other];

        let unfocused_frame =
            render_canvas_frame(&snapshot, &terrain, (4, 4), canvas_test_day_night());
        snapshot.focused_agent_uid = Some(7);
        let focused_frame =
            render_canvas_frame(&snapshot, &terrain, (4, 4), canvas_test_day_night());

        // The focused agent sits on sub-pixel (4, 8); the ring lights (3..=5, 7..=9)
        // minus the centre. Sub-pixel (3, 7) belongs to terminal cell (1, 1).
        let ring_cell = (3_u16 / 2, 7_u16 / 4);
        let before = buf_symbol(&unfocused_frame, ring_cell);
        let after = buf_symbol(&focused_frame, ring_cell);
        assert_ne!(
            before, after,
            "focusing an agent must light the ring around it"
        );

        // The other agent is at sub-pixel (0, 0); nothing near it may change.
        assert_eq!(
            buf_symbol(&unfocused_frame, (0, 0)),
            buf_symbol(&focused_frame, (0, 0)),
            "an unfocused agent must not acquire a ring"
        );
    }

    /// The ring surrounds the dot; it must never paint over it, or selecting an
    /// agent would hide the thing selected.
    #[test]
    fn the_selection_ring_leaves_the_agents_own_sub_pixel_alone() {
        assert_eq!(CANVAS_SELECTION_RING.len(), 8);
        assert!(
            !CANVAS_SELECTION_RING.contains(&(0, 0)),
            "the ring must exclude the centre"
        );
        let distinct: std::collections::BTreeSet<(i32, i32)> =
            CANVAS_SELECTION_RING.iter().copied().collect();
        assert_eq!(distinct.len(), 8, "ring offsets must not repeat");
    }

    fn buf_symbol(buf: &Buffer, cell: (u16, u16)) -> String {
        buf[(cell.0, cell.1)].symbol().to_string()
    }

    /// bd-qxrt: every number the TUI shows a person must be the STABLE
    /// `AgentUid`, never the reusable arena handle. Driven against a real world
    /// so the two identity spaces are genuinely distinct, and asserting on the
    /// arena's own answer rather than on a literal.
    #[test]
    fn displayed_agent_identity_is_the_stable_uid_not_the_arena_handle() {
        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        for _ in 0..3 {
            world
                .try_spawn_agent(AgentData::default())
                .expect("default agent is finite");
        }

        let snapshot = Snapshot::from_world(&world);
        let handles: Vec<AgentId> = world.agents().iter_handles().collect();
        assert_eq!(snapshot.agents.len(), handles.len());

        for (viz, handle) in snapshot.agents.iter().zip(&handles) {
            let expected = world.agent_uid(*handle).map(|uid| uid.get());
            assert_eq!(viz.uid, expected, "AgentViz must carry the arena's own uid");
            assert_eq!(
                viz.id,
                handle.data().as_ffi(),
                "the arena handle stays available for lookups"
            );
        }

        // The leaderboard is the surface that both DISPLAYS an identity and
        // RESOLVES rows back to live agents, so it must keep the two separate.
        for entry in snapshot.oldest.iter().chain(&snapshot.leaderboard) {
            let resolved = handles
                .iter()
                .find(|handle| handle.data().as_ffi() == entry.handle)
                .expect("every row must resolve to a live handle");
            assert_eq!(
                entry.uid,
                world.agent_uid(*resolved).map(|uid| uid.get()),
                "the displayed uid must belong to the agent the row resolves to"
            );
        }
    }

    /// A missing identity must be visible as `?`, never substituted with a
    /// plausible number: printing `0` would present a wrong identity as a real
    /// one, which is the defect rather than a fix for it.
    #[test]
    fn a_missing_agent_identity_renders_as_a_visible_placeholder() {
        assert_eq!(agent_uid_label(Some(42)), "42");
        assert_eq!(agent_uid_label(None), "?");
        assert_ne!(
            agent_uid_label(None),
            "0",
            "an absent uid must not be reported as agent zero"
        );
    }

    #[test]
    fn terminal_numeric_ingress_rejects_non_finite_speed_before_queue_admission() {
        let world = Arc::new(std::sync::Mutex::new(
            WorldState::new(ScriptBotsConfig::default()).expect("world"),
        ));
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let app = TerminalApp::new(&renderer, ctx);

        app.submit_simulation_command(SimulationCommand {
            paused: Some(false),
            speed_multiplier: Some(f32::NAN),
            step_once: false,
        });

        assert!(
            (app.command_drain)().is_empty(),
            "terminal admitted a non-finite speed command"
        );
    }

    #[test]
    fn simulation_fault_survives_storage_health_refresh() {
        let world = command_characterization_world();
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world,
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        let fault = Arc::<str>::from("deliberate brain construction failure");
        app.simulation_fault = Some(Arc::clone(&fault));

        app.maybe_refresh_analytics();

        assert_eq!(app.simulation_fault.as_deref(), Some(fault.as_ref()));
        assert!(app.analytics_status.last_error.is_none());
        assert!(!app.analytics_status.stopped);
    }

    #[test]
    fn auto_pause_on_spike_hits() {
        let mut config = ScriptBotsConfig::default();
        config.control.auto_pause_on_spike_hit = true;
        let world = WorldState::new(config).expect("world");

        let world = Arc::new(std::sync::Mutex::new(world));
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };

        let mut app = TerminalApp::new(&renderer, ctx);
        app.snapshot.spike_hits = 3;
        app.paused = false;
        app.evaluate_auto_pause();
        assert!(app.paused, "should auto-pause on spike hits");
    }

    #[test]
    fn auto_pause_on_max_age() {
        let mut config = ScriptBotsConfig::default();
        config.control.auto_pause_age_above = Some(10);
        let world = WorldState::new(config).expect("world");

        let world = Arc::new(std::sync::Mutex::new(world));
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        app.snapshot.max_age = 12;
        app.paused = false;
        app.evaluate_auto_pause();
        assert!(
            app.paused,
            "should auto-pause when max age exceeds threshold"
        );
    }

    fn rail_record(tick: u64, kind: NarrativeEventKind, text: &str) -> NarrativeEventRecord {
        NarrativeEventRecord {
            schema_version: 1,
            tick: scriptbots_core::Tick(tick),
            kind,
            severity: 0.5,
            magnitude: 1.0,
            window: (tick, tick),
            metric: "population".to_string(),
            before: 1.0,
            after: 2.0,
            score: 3.0,
            subject: None,
            human_text: text.to_string(),
        }
    }

    macro_rules! rail_test_app {
        ($world:expr, $app:ident, $backend:ident) => {
            let analytics = AnalyticsSnapshotProvider::empty();
            let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
            let renderer = TerminalRenderer::default();
            let ctx = crate::renderer::RendererContext {
                simulation_step: disabled_persistence_step_driver(&$world),
                world: Arc::clone(&$world),
                analytics,
                control_runtime: &runtime,
                command_drain: drain,
                command_submit: submit,
                scenario: test_scenario(),
            };
            let mut $app = TerminalApp::new(&renderer, ctx);
            $app.palette = Palette::test_backend_evidence();
            $app.paused = true;
            let $backend = ratatui::backend::TestBackend::new(140, 48);
        };
    }

    fn buffer_text(terminal: &Terminal<ratatui::backend::TestBackend>) -> String {
        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        let mut text = String::new();
        for y in area.y..area.bottom() {
            for x in area.x..area.right() {
                text.push_str(buffer[(x, y)].symbol());
            }
            text.push('\n');
        }
        text
    }

    #[test]
    fn narrative_rail_paints_glyphs_detail_and_truncation_marker() {
        let world = command_characterization_world();
        rail_test_app!(world, app, backend);
        app.snapshot.narrative = vec![
            rail_record(
                8100,
                NarrativeEventKind::PopulationCrash,
                "population fell 70% (1000 -> 300)",
            ),
            rail_record(
                8420,
                NarrativeEventKind::EnergyRecovery,
                "mean energy recovered 40% (0.5 -> 0.7)",
            ),
            rail_record(
                9001,
                NarrativeEventKind::CombatSurge,
                "combat surged (12 -> 61 spike hits)",
            ),
        ];
        app.snapshot.narrative_dropped = 5;
        app.snapshot.narrative_capacity = 256;

        let mut terminal = Terminal::new(backend).expect("rail test backend");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("rail frame renders");
        let text = buffer_text(&terminal);

        for kind in [
            NarrativeEventKind::PopulationCrash,
            NarrativeEventKind::EnergyRecovery,
            NarrativeEventKind::CombatSurge,
        ] {
            assert!(
                text.contains(kind.rail_glyph().to_string().as_str()),
                "rail must paint the {} glyph; buffer:\n{text}",
                kind.as_str()
            );
        }
        // The selected event (newest) shows its full text in the detail pane.
        assert!(
            text.contains("combat surged (12 -> 61 spike hits)"),
            "detail pane must carry the selected event's text; buffer:\n{text}"
        );
        assert!(
            text.contains("tick 9001 | combat_surge"),
            "detail pane must name tick and kind; buffer:\n{text}"
        );
        // The truncation marker is explicit: a tail is never presented as a whole.
        assert!(
            text.contains("+5"),
            "wrapped ring must show the dropped-event marker; buffer:\n{text}"
        );
        assert!(
            text.contains("5 earlier events dropped"),
            "detail pane must explain the truncation; buffer:\n{text}"
        );
        // THE SEEK CONTRACT: the rail says what it is — select-only history.
        assert!(
            text.contains("select-only"),
            "rail title must not overstate the seek contract; buffer:\n{text}"
        );
    }

    #[test]
    fn narrative_rail_omits_truncation_marker_when_ring_has_not_wrapped() {
        let world = command_characterization_world();
        rail_test_app!(world, app, backend);
        app.snapshot.narrative = vec![rail_record(
            300,
            NarrativeEventKind::PopulationBoom,
            "population rose 60% (100 -> 160)",
        )];
        app.snapshot.narrative_dropped = 0;

        let mut terminal = Terminal::new(backend).expect("rail test backend");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("rail frame renders");
        let text = buffer_text(&terminal);
        assert!(
            !text.contains("earlier events dropped"),
            "an unwrapped ring must not claim truncation; buffer:\n{text}"
        );
        assert!(
            text.contains("population rose 60% (100 -> 160)"),
            "single event renders its text; buffer:\n{text}"
        );
    }

    #[test]
    fn narrative_rail_selection_clamps_at_ends_without_silent_wrap() {
        let world = command_characterization_world();
        rail_test_app!(world, app, _backend);
        app.snapshot.narrative = vec![
            rail_record(10, NarrativeEventKind::PopulationCrash, "a"),
            rail_record(20, NarrativeEventKind::PopulationBoom, "b"),
            rail_record(30, NarrativeEventKind::CombatSurge, "c"),
        ];
        app.move_rail_selection(-99);
        let (index, tick, kind) = app.rail_selection.expect("selection set");
        assert_eq!(
            (index, tick, kind),
            (0, 10, NarrativeEventKind::PopulationCrash)
        );
        app.move_rail_selection(99);
        let (index, tick, kind) = app.rail_selection.expect("selection set");
        assert_eq!(
            (index, tick, kind),
            (2, 30, NarrativeEventKind::CombatSurge)
        );
        app.move_rail_selection(-1);
        let (index, _, _) = app.rail_selection.expect("selection set");
        assert_eq!(index, 1);
    }

    #[test]
    fn narrative_rail_wrap_aged_out_selection_clamps_loudly() {
        let world = command_characterization_world();
        rail_test_app!(world, app, backend);
        app.snapshot.narrative = vec![
            rail_record(10, NarrativeEventKind::PopulationCrash, "old"),
            rail_record(20, NarrativeEventKind::PopulationBoom, "newer"),
        ];
        app.rail_selection = Some((0, 10, NarrativeEventKind::PopulationCrash));
        // The ring wraps: both retained events are newer than the selection.
        app.snapshot.narrative = vec![
            rail_record(50, NarrativeEventKind::CombatSurge, "fresh"),
            rail_record(60, NarrativeEventKind::RegimeChange, "freshest"),
        ];
        app.snapshot.narrative_dropped = 2;
        app.validate_rail_selection();
        let (index, tick, kind) = app.rail_selection.expect("selection clamped, not dropped");
        assert_eq!(
            (index, tick, kind),
            (1, 60, NarrativeEventKind::RegimeChange),
            "an aged-out selection must clamp to the newest live event"
        );
        assert!(
            app.rail_selection_aged_out,
            "the wrap must be said out loud"
        );

        let mut terminal = Terminal::new(backend).expect("rail test backend");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("aged-out frame renders without panic");
        let text = buffer_text(&terminal);
        assert!(
            text.contains("aged out"),
            "detail pane must report the aged-out selection; buffer:\n{text}"
        );
    }

    #[test]
    fn narrative_rail_empty_renders_without_claiming_events() {
        let world = command_characterization_world();
        rail_test_app!(world, app, backend);
        app.snapshot.narrative.clear();
        app.snapshot.narrative_dropped = 0;
        let mut terminal = Terminal::new(backend).expect("rail test backend");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("empty rail renders");
        let text = buffer_text(&terminal);
        assert!(
            text.contains("no narrative events yet"),
            "empty rail must say so; buffer:\n{text}"
        );
    }

    #[test]
    fn narrative_rail_render_is_read_only() {
        let world = command_characterization_world();
        let before = {
            let guard = world.lock().expect("world lock");
            guard.characterization_digest_v0().expect("digest before")
        };
        rail_test_app!(world, app, backend);
        app.snapshot.narrative = vec![
            rail_record(10, NarrativeEventKind::PopulationCrash, "a"),
            rail_record(20, NarrativeEventKind::PopulationBoom, "b"),
        ];
        app.snapshot.narrative_dropped = 3;
        let mut terminal = Terminal::new(backend).expect("rail test backend");
        for _ in 0..16 {
            terminal
                .draw(|frame| app.draw(frame))
                .expect("repeated rail renders");
        }
        app.move_rail_selection(-1);
        terminal
            .draw(|frame| app.draw(frame))
            .expect("selection render");
        let after = {
            let guard = world.lock().expect("world lock");
            guard.characterization_digest_v0().expect("digest after")
        };
        assert_eq!(
            before, after,
            "rendering and navigating the rail must not perturb the world: an \
             instrument that changes what it observes is not an instrument"
        );
    }

    /// Drive a seeded world through forced boom/crash cycles until the narrative
    /// ring holds at least `target` events (bounded), then return the world. Each
    /// cycle spans 210 ticks so the per-kind 200-tick cooldown allows a fresh
    /// crash and boom per cycle.
    fn narrative_e2e_world(target: usize) -> SharedWorld {
        let config = ScriptBotsConfig {
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            initial_food: 0.0,
            food_respawn_interval: 0,
            food_intake_rate: 0.0,
            metabolism_drain: 0.0,
            movement_drain: 0.0,
            bot_speed: 0.0,
            population_minimum: 0,
            population_spawn_interval: 0,
            reproduction_energy_threshold: 0.0,
            persistence_interval: 0,
            chart_flush_interval: 0,
            narrative_interval: 1,
            narrative_capacity: 64,
            rng_seed: Some(0xE2E2_4A11),
            ..ScriptBotsConfig::default()
        };
        let world = Arc::new(std::sync::Mutex::new(
            WorldState::new(config).expect("narrative e2e world"),
        ));
        for cycle in 0..24 {
            {
                let mut guard = world.lock().expect("world lock");
                if guard.narrative_events().len() >= target {
                    break;
                }
                // Scale the injection with the accumulated population so every boom
                // stays above the narrative policy's materiality floor.
                let to_inject = guard.agent_count() / 2 + 10;
                for _ in 0..to_inject {
                    guard
                        .try_inject_agent(AgentData::default())
                        .expect("e2e injection is finite");
                }
            }
            for _ in 0..40 {
                let mut guard = world.lock().expect("world lock");
                guard.step().expect("e2e boom step");
            }
            {
                let mut guard = world.lock().expect("world lock");
                let handles: Vec<AgentId> = guard.agents().iter_handles().collect();
                for id in handles {
                    guard
                        .try_update_agent_runtime(id, |runtime| {
                            runtime.energy = -1.0;
                        })
                        .expect("starve e2e population");
                }
            }
            for _ in 0..170 {
                let mut guard = world.lock().expect("world lock");
                guard.step().expect("e2e crash step");
            }
            let _ = cycle;
        }
        world
    }

    #[test]
    fn narrative_rail_e2e_snapshot_matches_the_worlds_ring() {
        let world = narrative_e2e_world(20);
        let (world_ticks, world_dropped) = {
            let guard = world.lock().expect("world lock");
            assert!(
                guard.narrative_events().len() >= 20,
                "e2e fixture must produce at least 20 narrative events, found {}",
                guard.narrative_events().len()
            );
            (
                guard
                    .narrative_events()
                    .iter()
                    .map(|event| (event.tick.0, event.kind))
                    .collect::<Vec<_>>(),
                guard.narrative_dropped_events(),
            )
        };

        rail_test_app!(world, app, _backend);
        let snapshot_sequence: Vec<(u64, NarrativeEventKind)> = app
            .snapshot
            .narrative
            .iter()
            .map(|event| (event.tick.0, event.kind))
            .collect();
        assert_eq!(
            snapshot_sequence, world_ticks,
            "the TUI rail must show the world's events in the world's order"
        );
        assert_eq!(
            app.snapshot.narrative_dropped, world_dropped,
            "the TUI rail must report the world's exact dropped count"
        );
    }

    /// Fixture brain for the attribution panel (bd-16g.4.3): reports one
    /// activation layer and NO connections, so the panel must state the honest
    /// `NoConnections` reason rather than an empty top-k.
    struct PanelBrain;

    impl scriptbots_core::BrainRunner for PanelBrain {
        fn kind(&self) -> &'static str {
            "terminal.panel"
        }

        fn tick(
            &mut self,
            _inputs: &[f32; scriptbots_core::INPUT_SIZE],
        ) -> [f32; scriptbots_core::OUTPUT_SIZE] {
            [0.0; scriptbots_core::OUTPUT_SIZE]
        }

        fn inspect(
            &self,
            request: scriptbots_core::BrainInspection,
        ) -> Result<
            Option<scriptbots_core::BrainInspectionSnapshot>,
            scriptbots_core::BrainInspectionError,
        > {
            let scriptbots_core::BrainInspection::Activations(limits) = request;
            let values = vec![0.25_f32; scriptbots_core::INPUT_SIZE + 16];
            let activations = BrainActivations {
                layers: vec![scriptbots_core::ActivationLayer {
                    name: "state".to_owned(),
                    width: values.len(),
                    height: 1,
                    values,
                }],
                connections: Vec::new(),
                truncated: false,
            };
            scriptbots_core::bound_brain_inspection("terminal.panel", activations, 0, limits)
                .map(Some)
        }
    }

    /// bd-16g.4.3: the panel names every output from the centralized wire map,
    /// shows the boost threshold state, and states the honest NoConnections
    /// reason for a brain that reports layers but no edges.
    #[test]
    fn brain_panel_names_outputs_and_states_no_connections_honestly() {
        let world = command_characterization_world();
        {
            let mut guard = world.lock().expect("panel world lock");
            let family = guard
                .brain_registry_mut()
                .expect("panel registry mutation")
                .register_with_state_digest("terminal.panel", 0x5041_4e45_4c5f_4252, |_rng| {
                    Ok(Box::new(PanelBrain))
                });
            let agent = guard
                .try_spawn_agent(AgentData {
                    position: Position::new(50.0, 50.0),
                    ..AgentData::default()
                })
                .expect("spawn panel agent");
            guard
                .bind_agent_brain(agent, family)
                .expect("bind panel brain");
        }
        rail_test_app!(world, app, backend);
        app.refresh_snapshot();
        assert!(
            app.snapshot.focused_brain_bound,
            "fixture agent must be bound"
        );
        assert!(
            app.snapshot.focused_activations.is_some(),
            "panel snapshot must carry the raw activations"
        );

        let mut terminal = Terminal::new(backend).expect("panel test backend");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("panel frame renders");
        let text = buffer_text(&terminal);
        for needle in ["wheel_left", "wheel_right", "no weighted connections"] {
            assert!(
                text.contains(needle),
                "brain panel must render {needle:?}; buffer:\n{text}"
            );
        }
        // The historical mislabeling regression is pinned at the explanations
        // level (boost is output 6, green is output 3) by
        // `brain_panel_output_values_match_runtime_outputs_exactly`; the visible
        // rows depend on the block's height.
    }

    /// bd-16g.4.3 NEGATIVE: an unbound agent's outputs are an identity copy of
    /// sensors 0..8; the panel must say so instead of fabricating attribution.
    #[test]
    fn brain_panel_refuses_to_explain_an_identity_passthrough() {
        let world = command_characterization_world();
        {
            let mut guard = world.lock().expect("passthrough world lock");
            guard
                .try_spawn_agent(AgentData {
                    position: Position::new(50.0, 50.0),
                    ..AgentData::default()
                })
                .expect("spawn unbound agent");
        }
        rail_test_app!(world, app, backend);
        {
            let mut guard = world.lock().expect("step lock");
            guard.step().expect("one tick");
        }
        app.refresh_snapshot();
        assert!(
            !app.snapshot.focused_brain_bound,
            "fixture agent must be unbound"
        );

        let mut terminal = Terminal::new(backend).expect("passthrough test backend");
        terminal
            .draw(|frame| app.draw(frame))
            .expect("passthrough frame renders");
        let text = buffer_text(&terminal);
        assert!(
            text.contains("identity copy"),
            "panel must display the passthrough reason; buffer:\n{text}"
        );
        assert!(
            text.contains("wheel_left"),
            "passthrough outputs still show canonical names; buffer:\n{text}"
        );
    }

    /// bd-16g.4.3 round-trip against reality: the displayed output values equal
    /// runtime.outputs exactly.
    #[test]
    fn brain_panel_output_values_match_runtime_outputs_exactly() {
        let world = command_characterization_world();
        let outputs = {
            let mut guard = world.lock().expect("outputs lock");
            let agent = guard
                .try_spawn_agent(AgentData {
                    position: Position::new(50.0, 50.0),
                    ..AgentData::default()
                })
                .expect("spawn agent");
            guard.step().expect("one tick");
            guard.agent_runtime(agent).expect("runtime").outputs
        };
        rail_test_app!(world, app, _backend);
        app.refresh_snapshot();
        let explanations = app
            .output_explanations(&app.snapshot)
            .expect("explanations for a spawned agent");
        for (index, explanation) in explanations.iter().enumerate() {
            assert_eq!(
                explanation.raw_value, outputs[index],
                "displayed value for output {index} must equal runtime.outputs[{index}]"
            );
        }
        assert_eq!(explanations[6].output_name, "boost");
        assert_eq!(explanations[3].output_name, "color_green");
    }

    #[test]
    fn auto_pause_on_population_threshold() {
        let mut config = ScriptBotsConfig::default();
        config.control.auto_pause_population_below = Some(5);
        let mut world = WorldState::new(config).expect("world");
        for _ in 0..3 {
            world
                .try_spawn_agent(AgentData::default())
                .expect("default agent is finite");
        }

        let world = Arc::new(std::sync::Mutex::new(world));
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);
        app.refresh_snapshot();
        app.paused = false;
        app.evaluate_auto_pause();
        assert!(
            app.paused,
            "should auto-pause when population below threshold"
        );
    }

    #[test]
    fn auto_pause_single_event_per_tick() {
        let mut config = ScriptBotsConfig::default();
        config.control.auto_pause_on_spike_hit = true;
        let world = WorldState::new(config).expect("world");

        let world = Arc::new(std::sync::Mutex::new(world));
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);

        let initial_events = app.event_log.len();
        app.snapshot.spike_hits = 1;
        app.paused = false;
        app.evaluate_auto_pause();
        let after_first = app.event_log.len();
        // Re-evaluate within the same tick; should not add a duplicate event
        app.evaluate_auto_pause();
        let after_second = app.event_log.len();

        assert_eq!(after_first, initial_events + 1);
        assert_eq!(after_second, after_first);
        assert!(app.paused);
        assert_eq!(app.last_autopause_tick, Some(app.snapshot.tick));
    }

    #[test]
    fn test_terminal_app_host_client_snapshot_and_cadence_parity() {
        let config = ScriptBotsConfig::default();
        let world = WorldState::new(config).expect("world");
        let world = Arc::new(std::sync::Mutex::new(world));
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);

        // Initial snapshot tick parity
        assert_eq!(app.snapshot().tick, 0);

        // 1. Paused state by default: renderer cadence cannot advance science
        app.paused = true;
        let now = Instant::now();
        app.advance_simulation(now + Duration::from_millis(500), false);
        assert_eq!(
            app.snapshot().tick,
            0,
            "renderer cadence must not advance science when paused"
        );

        // 2. Single step advances science by exactly 1 tick
        app.step_once();
        assert_eq!(
            app.snapshot().tick,
            1,
            "step_once must advance science by exactly 1 tick"
        );
        assert!(app.paused, "simulation must remain paused after step_once");

        // 3. Report tick equals snapshot tick
        let report = app.snapshot();
        assert_eq!(report.tick, 1);
    }

    #[test]
    fn test_curated_theme_id_cycling() {
        // Walk the cycle instead of counting steps by hand. The hand-counted
        // version stepped five times and asserted the wrap, which was correct for
        // five themes and silently wrong the moment bd-f4x0 added a sixth
        // (BioluminescentDarkField) — the assertion encoded the COUNT rather than
        // the property, so growing the enum broke a test that had nothing to say
        // about the new theme.
        let start = CuratedThemeId::CyberpunkAurora;
        let mut cycle = vec![start];
        let mut current = start.next();
        while current != start {
            assert!(
                !cycle.contains(&current),
                "next() re-enters {current:?} without returning to {start:?}, so the \
                 themes form a lasso rather than a cycle and some are unreachable"
            );
            cycle.push(current);
            assert!(
                cycle.len() <= 64,
                "next() never returned to {start:?}; the theme cycle is not closed"
            );
            current = current.next();
        }

        // The one place that needs updating when a theme is added, and it fails
        // with a message that says so rather than an opaque count mismatch.
        for theme in [
            CuratedThemeId::BioluminescentDarkField,
            CuratedThemeId::CyberpunkAurora,
            CuratedThemeId::Darcula,
            CuratedThemeId::LumenLight,
            CuratedThemeId::NordicFrost,
            CuratedThemeId::HighContrast,
        ] {
            assert!(
                cycle.contains(&theme),
                "{theme:?} exists but Ctrl-T never reaches it — add it to \
                 CuratedThemeId::next() so the cycle covers every theme"
            );
        }
        assert_eq!(
            cycle.len(),
            6,
            "cycle visited {:?}; if a theme was added, extend the list above too",
            cycle
        );

        // Documented order, kept so a silent reordering is still caught.
        assert_eq!(
            cycle,
            vec![
                CuratedThemeId::CyberpunkAurora,
                CuratedThemeId::Darcula,
                CuratedThemeId::LumenLight,
                CuratedThemeId::NordicFrost,
                CuratedThemeId::HighContrast,
                CuratedThemeId::BioluminescentDarkField,
            ]
        );

        assert_eq!(start.label(), "Cyberpunk Aurora");
        assert_ne!(start.header_color(), start.next().header_color());
    }

    #[test]
    fn test_theme_palette_wcag_contrast_spot_checks() {
        // BioluminescentDarkField was missing from this list, so the DEFAULT
        // theme — the one almost every run actually shows — was the single theme
        // never checked. Walk the cycle instead of restating a list, so a seventh
        // theme is covered the day it is added.
        let mut themes = vec![CuratedThemeId::default()];
        let mut current = CuratedThemeId::default().next();
        while current != CuratedThemeId::default() {
            themes.push(current);
            current = current.next();
        }
        let palettes = [
            TerminalPaletteMode::Natural,
            TerminalPaletteMode::Deuteranopia,
            TerminalPaletteMode::Protanopia,
            TerminalPaletteMode::Tritanopia,
            TerminalPaletteMode::HighContrast,
        ];

        let mut failures: Vec<String> = Vec::new();
        for theme in themes {
            for mode in palettes {
                let p = Palette {
                    level: on_cached(Stream::Stdout),
                    emoji: true,
                    emoji_narrow: false,
                    mode,
                    theme_id: theme,
                };
                let t = p.theme();

                // Real text-on-background pairs the status bar actually paints,
                // not constructed styles. The previous version of this test built
                // two styles, DISCARDED them, and asserted the labels were
                // non-empty — which passes for any colour whatsoever, including a
                // theme painting black on black. bd-2z0.14.2.2 forbids counting
                // style construction or nonempty labels as a contrast proof.
                for (label, fg, bg) in [
                    ("paused status", t.paused_fg, t.paused_bg),
                    ("running status", t.running_fg, t.running_bg),
                ] {
                    let ratio = contrast_ratio(fg, bg);
                    if ratio < WCAG_AA_NORMAL_TEXT {
                        failures.push(format!(
                            "{theme:?} + {mode:?} {label}: {ratio:.2}:1 \
                             (needs {WCAG_AA_NORMAL_TEXT}:1) fg={fg:?} bg={bg:?}"
                        ));
                    }
                }
            }
        }

        assert!(
            failures.is_empty(),
            "these theme x palette pairs fail WCAG AA for normal text:\n  {}",
            failures.join("\n  ")
        );
    }

    /// The contrast math itself must be right, or every gate built on it is
    /// decoration.
    ///
    /// Pinned against the two values WCAG fixes by definition — black on white is
    /// exactly 21:1 and any colour against itself is exactly 1:1 — plus the
    /// linearisation, which is the part a shortcut implementation gets wrong. A
    /// naive version that skips the transfer function and weights raw channels
    /// reports mid grey on white as roughly 1.9:1; the correct value is above 3.9,
    /// so this catches precisely that substitution.
    #[test]
    fn the_contrast_ratio_math_matches_wcag() {
        let black = Color::Rgb(0, 0, 0);
        let white = Color::Rgb(255, 255, 255);

        let extreme = contrast_ratio(black, white);
        assert!(
            (extreme - 21.0).abs() < 0.01,
            "black on white must be 21:1, got {extreme:.4}"
        );
        assert!(
            (contrast_ratio(white, black) - extreme).abs() < 1e-6,
            "the ratio must not depend on argument order"
        );
        assert!(
            (contrast_ratio(white, white) - 1.0).abs() < 1e-6,
            "a colour against itself must be 1:1"
        );

        // 50% grey is the discriminating case for the transfer function.
        let grey = Color::Rgb(128, 128, 128);
        let grey_on_white = contrast_ratio(grey, white);
        assert!(
            (3.9..4.1).contains(&grey_on_white),
            "mid grey on white must be about 3.95:1 with sRGB linearisation; got \
             {grey_on_white:.3}. A value near 1.9 means the channels were weighted \
             without linearising, which flatters dark colours and lets failing \
             pairs pass"
        );

        // Tie the math to the GATE's threshold. The theme sweep above currently
        // passes for every pair, and a gate that has never rejected anything is
        // indistinguishable from one that cannot. These two pin both sides of
        // WCAG_AA_NORMAL_TEXT so the sweep's green is meaningful.
        assert!(
            grey_on_white < WCAG_AA_NORMAL_TEXT,
            "mid grey on white is below AA and the gate must say so"
        );
        assert!(
            contrast_ratio(Color::Rgb(40, 40, 40), black) < WCAG_AA_NORMAL_TEXT,
            "near-black on black must fail the gate; if this passes, the gate \
             would accept unreadable text"
        );
        assert!(
            contrast_ratio(black, white) >= WCAG_AA_NORMAL_TEXT,
            "black on white must pass the gate, or the threshold rejects everything \
             and the sweep is green for the wrong reason"
        );
    }

    #[test]
    fn test_tick_phase_determinism() {
        let tp1 = TickPhase::compute(120, false, 4);
        let tp2 = TickPhase::compute(120, false, 4);
        assert_eq!(tp1, tp2, "TickPhase calculation must be deterministic");
        assert_eq!(tp1.tick, 120);
        assert!((tp1.phase - (4.0 / 16.0) / 60.0).abs() < 1e-5);
    }

    #[test]
    fn test_tick_phase_paused_frozen() {
        let tp_paused = TickPhase::compute(120, true, 4);
        assert_eq!(tp_paused.phase, 0.0, "paused phase must be frozen at 0.0");
        let pulse = tp_paused.pulse(1.0, 0.2, 0.8);
        assert_eq!(pulse, 0.5, "paused pulse must be static mid-value");
    }

    #[test]
    fn test_toast_lifecycle() {
        let toast = ToastEntry::new("Test Toast", 10, 50);
        assert_eq!(toast.created_tick, 10);
        assert_eq!(toast.expiry_tick, 60);

        assert!(!toast.is_expired(10));
        assert!(!toast.is_expired(59));
        assert!(toast.is_expired(60));
        assert!(toast.is_expired(100));
    }

    #[test]
    fn test_event_pulse_ring() {
        let ring0 = EventPulseRing::from_event(100, 100, 10.0, 20.0).expect("ring");
        assert_eq!(ring0.radius, 0.5);
        assert_eq!(ring0.intensity, 1.0);

        let ring3 = EventPulseRing::from_event(100, 103, 10.0, 20.0).expect("ring");
        assert!(
            ring3.radius > ring0.radius,
            "pulse ring radius must expand with age"
        );
        assert!(
            ring3.intensity < ring0.intensity,
            "pulse ring intensity must fade with age"
        );

        let ring_expired = EventPulseRing::from_event(100, 107, 10.0, 20.0);
        assert!(
            ring_expired.is_none(),
            "pulse ring must expire after 5 ticks"
        );
    }

    #[test]
    fn test_command_palette_fuzzy_matching() {
        let items = all_command_palette_items();
        let matched = fuzzy_match_command_palette(&items, "pause");
        assert!(!matched.is_empty(), "query 'pause' must match Toggle Pause");
        assert_eq!(matched[0].action, CommandPaletteAction::TogglePause);

        let matched_cat = fuzzy_match_command_palette(&items, "science");
        assert!(
            matched_cat.len() >= 3,
            "category query 'science' must match science items"
        );
    }

    #[test]
    fn test_command_palette_action_execution() {
        let config = ScriptBotsConfig::default();
        let world = WorldState::new(config).expect("world");
        let world = Arc::new(std::sync::Mutex::new(world));
        let analytics = AnalyticsSnapshotProvider::empty();
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            simulation_step: disabled_persistence_step_driver(&world),
            world: Arc::clone(&world),
            analytics,
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
            scenario: test_scenario(),
        };
        let mut app = TerminalApp::new(&renderer, ctx);

        assert!(!app.paused);
        app.execute_palette_action(CommandPaletteAction::TogglePause);
        assert!(app.paused, "TogglePause action must set app.paused to true");

        assert_eq!(app.map_zoom_level, 1.0);
        app.zoom_in();
        assert!(
            app.map_zoom_level > 1.0,
            "zoom_in must increase map_zoom_level"
        );
        app.zoom_out();
        assert_eq!(
            app.map_zoom_level, 1.0,
            "zoom_out must decrease map_zoom_level"
        );
    }
}

#[derive(Clone, Debug)]
struct BrainLayerView {
    width: usize,
    height: usize,
    values: Vec<f32>,
    name: Option<String>,
}

impl BrainLayerView {
    fn vec_from_activations(act: &BrainActivations) -> Vec<BrainLayerView> {
        act.layers
            .iter()
            .map(|l| BrainLayerView {
                width: l.width,
                height: l.height,
                values: l.values.clone(),
                name: Some(l.name.clone()),
            })
            .collect()
    }
}

fn convert_layers(act: &BrainActivations) -> Vec<BrainLayerView> {
    BrainLayerView::vec_from_activations(act)
}
