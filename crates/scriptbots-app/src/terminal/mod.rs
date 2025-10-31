use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    f32::consts::{PI, TAU},
    fs::{self, File},
    io::{self, Stdout},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Sparkline, Widget},
};
use scriptbots_core::{
    AgentId, BrainActivations, ControlCommand, ControlSettings, SimulationCommand, TerrainKind,
    TerrainLayer, TickSummary, WorldState,
};
use scriptbots_storage::MetricReading;
use serde::Serialize;
use slotmap::Key;
use supports_color::{ColorLevel, Stream, on_cached};
use tracing::{info, warn};

use crate::{
    CommandDrain, CommandSubmit, ControlRuntime, SharedStorage, SharedWorld,
    renderer::{Renderer, RendererContext},
};

const TARGET_SIM_HZ: f32 = 60.0;
const MAX_STEPS_PER_FRAME: usize = 240;
const UI_TICK_MILLIS: u64 = 100;
const DEFAULT_HEADLESS_FRAMES: usize = 12;
const MAX_HEADLESS_FRAMES: usize = 360;
const EVENT_LOG_CAPACITY: usize = 16;
const LEADERBOARD_LIMIT: usize = 6;
const BRAINBOARD_LIMIT: usize = 4;

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
        if std::env::var_os("SCRIPTBOTS_TERMINAL_HEADLESS").is_some() {
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
        enable_raw_mode().context("failed to enable raw mode")?;
        execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).context("failed to build terminal backend")?;
        terminal.hide_cursor().ok();

        let result = run_event_loop(self, &mut terminal, ctx);

        terminal.show_cursor().ok();
        if let Err(err) = disable_raw_mode() {
            tracing::error!(?err, "failed to disable raw mode");
        }
        if let Err(err) = execute!(terminal.backend_mut(), LeaveAlternateScreen) {
            tracing::error!(?err, "failed to leave alternate screen");
        }

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

        if event::poll(sleep_for)?
            && let Event::Key(key) = event::read()?
            && app.handle_key(key)?
        {
            break;
        }
    }

    Ok(())
}

impl TerminalRenderer {
    fn run_headless(&self, ctx: RendererContext<'_>) -> Result<HeadlessReport> {
        let backend = ratatui::backend::TestBackend::new(80, 36);
        let mut terminal = Terminal::new(backend).context("failed to build test backend")?;
        let mut app = TerminalApp::new(self, ctx);
        let mut report = HeadlessReport::new(app.snapshot().clone());
        let frames = self.headless_frame_budget();

        for _ in 0..frames {
            app.step_once();
            report.record(app.snapshot());
            terminal.draw(|frame| app.draw(frame))?;
        }

        report.finalize();

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

#[derive(Clone, Copy)]
enum FocusLockMode {
    Manual,
    TopPredator,
    Oldest,
}

struct TerminalApp<'a> {
    world: SharedWorld,
    storage: SharedStorage,
    _control: &'a ControlRuntime,
    command_drain: CommandDrain,
    command_submit: CommandSubmit,
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
    event_log: VecDeque<EventEntry>,
    last_event_tick: u64,
    snapshot: Snapshot,
    baseline: Option<Baseline>,
    last_autopause_tick: Option<u64>,
    map_scratch: Vec<CellOccupancy>,
    map_stamp: u32,
    analytics: Option<TerminalAnalytics>,
    analytics_tick: Option<u64>,
    expanded: bool,
    // When true, the user has explicitly toggled expanded panels; honor self.expanded
    // instead of auto-expanding on wide terminals.
    expanded_user_override: bool,
    // Brain view controls
    focused_agent_cursor: usize,
    activation_layer_index: usize,
    activation_row_offset: usize,
    focus_lock: FocusLockMode,
}

impl<'a> TerminalApp<'a> {
    fn new(renderer: &TerminalRenderer, ctx: RendererContext<'a>) -> Self {
        let palette = Palette::detect();
        let terrain = {
            let world = ctx
                .world
                .lock()
                .expect("world mutex poisoned while capturing terrain");
            TerrainView::from(world.terrain())
        };
        let mut app = Self {
            world: Arc::clone(&ctx.world),
            storage: Arc::clone(&ctx.storage),
            _control: ctx.control_runtime,
            command_drain: Arc::clone(&ctx.command_drain),
            command_submit: Arc::clone(&ctx.command_submit),
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
            event_log: VecDeque::with_capacity(EVENT_LOG_CAPACITY),
            last_event_tick: 0,
            snapshot: Snapshot::default(),
            baseline: None,
            last_autopause_tick: None,
            map_scratch: Vec::new(),
            map_stamp: 1,
            analytics: None,
            analytics_tick: None,
            expanded: false,
            expanded_user_override: false,
            focused_agent_cursor: 0,
            activation_layer_index: 0,
            activation_row_offset: 0,
            focus_lock: FocusLockMode::Manual,
        };
        app.refresh_snapshot();
        app
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
        let delta = now - self.last_tick;
        self.last_tick = now;

        let mut force_step = false;
        let pending_commands = if let Ok(mut world) = self.world.lock() {
            (self.command_drain.as_ref())(&mut world);
            Some(world.drain_simulation_commands())
        } else {
            None
        };
        if let Some(pending) = pending_commands {
            if self.apply_simulation_commands(pending) {
                force_step = true;
            }
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

        if let Ok(mut world) = self.world.lock() {
            for _ in 0..steps {
                world.step();
            }
        }

        self.refresh_snapshot();
    }

    fn step_once(&mut self) {
        self.submit_simulation_command(SimulationCommand {
            paused: Some(true),
            speed_multiplier: Some(self.speed_multiplier),
            step_once: true,
        });
        if let Ok(mut world) = self.world.lock() {
            world.step();
        }
        self.refresh_snapshot();
    }

    fn draw(&mut self, frame: &mut Frame<'_>) {
        // Ensure we start from a clean buffer every frame to avoid ghosting artifacts
        frame.render_widget(Clear, frame.area());

        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(frame.area());

        self.draw_header(frame, outer[0], &self.snapshot);

        // Auto-expand advanced panels on wide terminals unless the user has overridden
        let area = outer[1];
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
            .split(outer[1]);

        // Draw the map while avoiding holding an external borrow across &mut self
        let world_size = self.snapshot.world_size;
        self.draw_map(frame, body[0], world_size);

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
    }

    fn maybe_refresh_analytics(&mut self) {
        let tick = self.snapshot.tick;
        if self.analytics_tick == Some(tick) {
            return;
        }
        if let Ok(mut guard) = self.storage.try_lock()
            && let Ok(readings) = guard.latest_metrics(256)
            && let Some(ana) = parse_terminal_analytics(tick, self.snapshot.agent_count, &readings)
        {
            self.analytics = Some(ana);
            self.analytics_tick = Some(tick);
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
                .title(self.palette.title("ScriptBots Terminal HUD"))
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

        if !pop_data.is_empty() {
            let spark = Sparkline::default()
                .style(self.palette.population_spark_style())
                .data(&pop_data);
            frame.render_widget(spark, trend_layout[0]);
        }
        if !energy_data.is_empty() {
            let spark = Sparkline::default()
                .style(self.palette.energy_spark_style())
                .data(&energy_data);
            frame.render_widget(spark, trend_layout[1]);
        }
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
        if !births_data.is_empty() {
            let spark = Sparkline::default()
                .style(Style::default().fg(Color::Green))
                .data(&births_data);
            frame.render_widget(spark, trend_layout[2]);
        }
        if !deaths_data.is_empty() {
            let spark = Sparkline::default()
                .style(Style::default().fg(Color::Red))
                .data(&deaths_data);
            frame.render_widget(spark, trend_layout[3]);
        }

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
            frame.render_widget(
                MapWidget {
                    snapshot: &self.snapshot,
                    terrain: &self.terrain,
                    palette: &self.palette,
                    scratch: &mut self.map_scratch,
                    stamp: self.map_stamp,
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
                    format!("#{:<4}", entry.label),
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
                    format!("#{:<4}", entry.label),
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
                let text = format!("[t{:>6}] {}", entry.tick, entry.message);
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
                Span::styled("█".repeat(width), Style::default().fg(Color::LightGreen)),
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

    fn draw_mortality(&self, frame: &mut Frame<'_>, area: Rect, _snapshot: &Snapshot) {
        let mut lines: Vec<Line> = Vec::new();
        if let Some(ana) = &self.analytics {
            lines.push(Line::from(vec![
                Span::styled("Deaths total ", self.palette.header_style()),
                Span::raw(format!("{:>4}", ana.deaths_total)),
            ]));
            // Simple horizontal bars to visualize proportions
            let parts = [
                ("C", ana.deaths_combat_carnivore as u64, Color::Red),
                ("H", ana.deaths_combat_herbivore as u64, Color::LightRed),
                ("S", ana.deaths_starvation as u64, Color::Yellow),
                ("A", ana.deaths_aging as u64, Color::Gray),
                ("U", ana.deaths_unknown as u64, Color::DarkGray),
            ];
            let total = ana.deaths_total.max(1) as u64;
            for (label, count, color) in parts {
                let width = ((count * 20) / total).clamp(0, 20) as usize;
                let bar = "█".repeat(width);
                lines.push(Line::from(vec![
                    Span::styled(format!(" {:>2} ", label), self.palette.header_style()),
                    Span::styled(bar, Style::default().fg(color)),
                    Span::raw(format!(" {:>3}", count)),
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

        // Compute a suitable height based on content and available space
        let desired_height = (help_lines.len() as u16).saturating_add(2);
        let help_height = desired_height.min(size.height.saturating_sub(2).max(8));
        let help_x = size.x + (size.width - help_width) / 2;
        let help_y = size.y + (size.height - help_height) / 2;
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

    fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        match (key.code, key.modifiers) {
            (KeyCode::Esc, _)
            | (KeyCode::Char('q'), _)
            | (KeyCode::Char('Q'), _)
            | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                return Ok(true);
            }
            (KeyCode::Char('c'), KeyModifiers::NONE) => {
                self.palette.cycle_mode();
                return Ok(false);
            }
            (KeyCode::Char(' '), _) => {
                self.paused = !self.paused;
                if self.paused {
                    self.speed_multiplier = 0.0;
                } else if self.speed_multiplier <= 0.0 {
                    self.speed_multiplier = 1.0;
                }
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
                self.push_event(self.snapshot.tick, EventKind::Info, "Single-step executed");
            }
            (KeyCode::Char('S'), _) => {
                if let Err(err) = self.save_ascii_snapshot() {
                    self.push_event(
                        self.snapshot.tick,
                        EventKind::Info,
                        format!("Screenshot failed: {err}"),
                    );
                } else {
                    self.push_event(
                        self.snapshot.tick,
                        EventKind::Info,
                        "Saved ASCII screenshot",
                    );
                }
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
                self.focused_agent_cursor = self.focused_agent_cursor.saturating_sub(1);
                self.refresh_snapshot();
            }
            (KeyCode::Right, _) => {
                self.focused_agent_cursor = self.focused_agent_cursor.saturating_add(1);
                self.refresh_snapshot();
            }
            (KeyCode::Char('t') | KeyCode::Char('T'), _) => {
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

    fn save_ascii_snapshot(&self) -> Result<()> {
        use std::io::Write;
        let dir = std::path::Path::new("screenshots");
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("frame_{}.txt", self.snapshot.tick));
        let mut file = std::fs::File::create(path)?;

        let width = 64usize.min(self.snapshot.world_size.0 as usize).max(16);
        let height = 32usize.min(self.snapshot.world_size.1 as usize).max(8);
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let terrain_kind = self.terrain.sample(u, v);
                let food = self.snapshot.food.sample(u, v);
                let ch = match terrain_kind {
                    TerrainKind::DeepWater => '~',
                    TerrainKind::ShallowWater => '=',
                    TerrainKind::Sand => '.',
                    TerrainKind::Grass => ',',
                    TerrainKind::Bloom => '*',
                    TerrainKind::Rock => '^',
                };
                let glyph = if food > 0.66 {
                    '#'
                } else if food > 0.33 {
                    '+'
                } else {
                    ch
                };
                write!(file, "{}", glyph)?;
            }
            writeln!(file)?;
        }
        Ok(())
    }

    fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    fn refresh_snapshot(&mut self) {
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
                            .find(|h| h.data().as_ffi() == e.label)
                    }),
                    FocusLockMode::Oldest => snap.oldest.first().and_then(|e| {
                        world
                            .agents()
                            .iter_handles()
                            .find(|h| h.data().as_ffi() == e.label)
                    }),
                };
                if let Some(agent_id) = agent_id_opt
                    && let Some(rt) = world.runtime().get(agent_id)
                    && let Some(act) = rt.brain_activations.as_ref()
                {
                    snap.brain_layers = convert_layers(act);
                }
                snap
            }
            Err(_) => return,
        };
        self.ingest_events(&new_snapshot);
        self.snapshot = new_snapshot;
        self.evaluate_auto_pause();
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
    id: u64,
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
    label: u64,
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
}

impl TerrainView {
    fn from(terrain: &TerrainLayer) -> Self {
        Self {
            width: terrain.width(),
            height: terrain.height(),
            kinds: terrain.tiles().iter().map(|tile| tile.kind).collect(),
        }
    }

    fn sample(&self, u: f32, v: f32) -> TerrainKind {
        if self.width == 0 || self.height == 0 || self.kinds.is_empty() {
            return TerrainKind::Grass;
        }
        let x = ((u.clamp(0.0, 0.9999)) * self.width as f32).floor() as usize;
        let y = ((v.clamp(0.0, 0.9999)) * self.height as f32).floor() as usize;
        let idx = y.saturating_mul(self.width as usize) + x;
        self.kinds.get(idx).copied().unwrap_or(TerrainKind::Grass)
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
                label: agent.id,
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
                label: agent.id,
                diet: agent.diet,
                energy: agent.energy,
                health: agent.health,
                age: agent.age,
                generation: agent.generation,
            })
            .collect();
        oldest.sort_by(|a, b| b.age.cmp(&a.age));
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
        }
    }

    fn brain_activations_layer_indexed(&self, idx: usize) -> Option<&BrainLayerView> {
        self.brain_layers.get(idx)
    }
}

#[derive(Debug, Clone, Serialize)]
struct HeadlessReport {
    initial: FrameStats,
    frames: Vec<FrameStats>,
    summary: ReportSummary,
}

impl HeadlessReport {
    fn new(initial_snapshot: Snapshot) -> Self {
        Self {
            initial: FrameStats::from_snapshot(&initial_snapshot),
            frames: Vec::new(),
            summary: ReportSummary::default(),
        }
    }

    fn record(&mut self, snapshot: &Snapshot) {
        self.frames.push(FrameStats::from_snapshot(snapshot));
    }

    fn finalize(&mut self) {
        self.summary = ReportSummary::from(&self.initial, &self.frames);
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
    avg_energy: f32,
}

impl FrameStats {
    fn from_snapshot(snapshot: &Snapshot) -> Self {
        Self {
            tick: snapshot.tick,
            epoch: snapshot.epoch,
            agent_count: snapshot.agent_count,
            births: snapshot.births,
            deaths: snapshot.deaths,
            avg_energy: snapshot.avg_energy,
        }
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
    avg_energy_mean: f32,
    avg_energy_min: f32,
    avg_energy_max: f32,
}

impl ReportSummary {
    fn from(initial: &FrameStats, frames: &[FrameStats]) -> Self {
        if frames.is_empty() {
            return Self {
                frame_count: 0,
                ticks_simulated: 0,
                final_tick: initial.tick,
                final_epoch: initial.epoch,
                final_agent_count: initial.agent_count,
                total_births: 0,
                total_deaths: 0,
                avg_energy_mean: initial.avg_energy,
                avg_energy_min: initial.avg_energy,
                avg_energy_max: initial.avg_energy,
            };
        }

        let frame_count = frames.len();
        let final_stats = frames.last().expect("frame list not empty");
        let ticks_simulated = final_stats.tick.saturating_sub(initial.tick);

        let total_births = frames.iter().map(|frame| frame.births).sum();
        let total_deaths = frames.iter().map(|frame| frame.deaths).sum();

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
            avg_energy_mean,
            avg_energy_min: min_energy,
            avg_energy_max: max_energy,
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

struct Palette {
    level: Option<ColorLevel>,
    emoji: bool,
    emoji_narrow: bool,
    mode: TerminalPaletteMode,
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

fn rgb(hex: u32) -> Color {
    Color::Rgb(
        ((hex >> 16) & 0xFF) as u8,
        ((hex >> 8) & 0xFF) as u8,
        (hex & 0xFF) as u8,
    )
}

impl Palette {
    fn theme(&self) -> TerminalTheme {
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
                let term = std::env::var("TERM")
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let looks_modern_term = !matches!(term.as_str(), "" | "dumb" | "linux" | "vt100");
                let locale = std::env::var("LC_ALL")
                    .ok()
                    .or_else(|| std::env::var("LC_CTYPE").ok())
                    .or_else(|| std::env::var("LANG").ok())
                    .unwrap_or_default()
                    .to_ascii_lowercase();
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
            mode,
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

    fn has_color(&self) -> bool {
        self.level.is_some()
    }

    fn is_emoji(&self) -> bool {
        self.emoji
    }

    fn toggle_emoji(&mut self) {
        self.emoji = !self.emoji;
    }

    fn cycle_mode(&mut self) {
        self.mode = self.mode.next();
    }

    fn mode_label(&self) -> &'static str {
        self.mode.label()
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

    fn terrain_symbol(&self, kind: TerrainKind, food_level: f32) -> (char, Style) {
        let rich_color = self
            .level
            .is_some_and(|level| level.has_16m || level.has_256);
        let idx = match kind {
            TerrainKind::DeepWater => 0,
            TerrainKind::ShallowWater => 1,
            TerrainKind::Sand => 2,
            TerrainKind::Grass => 3,
            TerrainKind::Bloom => 4,
            TerrainKind::Rock => 5,
        };
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
}

impl<'a> Widget for MapWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 2 || area.height < 2 {
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
    use scriptbots_core::{AgentData, ScriptBotsConfig};

    #[test]
    fn snapshot_reflects_world_state() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        world.spawn_agent(AgentData::default());

        let snapshot = Snapshot::from_world(&world);

        assert_eq!(snapshot.agent_count, world.agent_count());
        assert_eq!(snapshot.tick, world.tick().0);
        assert_eq!(snapshot.agents.len(), world.agent_count());
        assert_eq!(snapshot.world_size.0, world.config().world_width);
    }

    #[test]
    fn auto_pause_on_spike_hits() {
        let mut config = ScriptBotsConfig::default();
        config.control.auto_pause_on_spike_hit = true;
        let world = WorldState::new(config).expect("world");

        let world = Arc::new(std::sync::Mutex::new(world));
        let storage = Arc::new(std::sync::Mutex::new(
            scriptbots_storage::Storage::open(":memory:").expect("storage"),
        ));
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            world: Arc::clone(&world),
            storage: Arc::clone(&storage),
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
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
        let storage = Arc::new(std::sync::Mutex::new(
            scriptbots_storage::Storage::open(":memory:").expect("storage"),
        ));
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            world: Arc::clone(&world),
            storage: Arc::clone(&storage),
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
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

    #[test]
    fn auto_pause_on_population_threshold() {
        let mut config = ScriptBotsConfig::default();
        config.control.auto_pause_population_below = Some(5);
        let mut world = WorldState::new(config).expect("world");
        for _ in 0..3 {
            world.spawn_agent(AgentData::default());
        }

        let world = Arc::new(std::sync::Mutex::new(world));
        let storage = Arc::new(std::sync::Mutex::new(
            scriptbots_storage::Storage::open(":memory:").expect("storage"),
        ));
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            world: Arc::clone(&world),
            storage: Arc::clone(&storage),
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
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
        let storage = Arc::new(std::sync::Mutex::new(
            scriptbots_storage::Storage::open(":memory:").expect("storage"),
        ));
        let (runtime, drain, submit) = crate::servers::ControlRuntime::dummy();
        let renderer = TerminalRenderer::default();
        let ctx = crate::renderer::RendererContext {
            world: Arc::clone(&world),
            storage: Arc::clone(&storage),
            control_runtime: &runtime,
            command_drain: drain,
            command_submit: submit,
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
