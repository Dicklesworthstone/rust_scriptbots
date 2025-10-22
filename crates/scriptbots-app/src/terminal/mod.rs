use std::{
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
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
};
#[cfg(test)]
use scriptbots_core::AgentData;
use scriptbots_core::{AgentColumns, TickSummary, WorldState};
use serde::Serialize;
use supports_color::{ColorLevel, Stream, on_cached};
use tracing::info;

use crate::{
    CommandDrain, CommandSubmit, ControlRuntime, SharedStorage, SharedWorld,
    renderer::{Renderer, RendererContext},
};

const TARGET_SIM_HZ: f32 = 60.0;
const MAX_STEPS_PER_FRAME: usize = 240;
const UI_TICK_MILLIS: u64 = 100;
const MAP_WIDTH: usize = 48;
const MAP_HEIGHT: usize = 24;
const DEFAULT_HEADLESS_FRAMES: usize = 12;
const MAX_HEADLESS_FRAMES: usize = 360;

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
            app.last_draw = now;
        }

        let timeout = renderer
            .draw_interval
            .saturating_sub(now.duration_since(app.last_event_check));
        let event_ready = event::poll(timeout).unwrap_or(false);
        if event_ready
            && let Event::Key(key) = event::read()?
            && app.handle_key(key)?
        {
            break;
        }
        if event_ready {
            app.last_event_check = Instant::now();
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

struct TerminalApp<'a> {
    world: SharedWorld,
    _storage: SharedStorage,
    _control: &'a ControlRuntime,
    command_drain: CommandDrain,
    _command_submit: CommandSubmit,
    tick_interval: Duration,
    draw_interval: Duration,
    speed_multiplier: f32,
    paused: bool,
    help_visible: bool,
    sim_accumulator: f32,
    last_tick: Instant,
    last_draw: Instant,
    last_event_check: Instant,
    palette: Palette,
    snapshot: Snapshot,
}

impl<'a> TerminalApp<'a> {
    fn new(renderer: &TerminalRenderer, ctx: RendererContext<'a>) -> Self {
        let palette = Palette::detect();
        let mut app = Self {
            world: Arc::clone(&ctx.world),
            _storage: Arc::clone(&ctx.storage),
            _control: ctx.control_runtime,
            command_drain: Arc::clone(&ctx.command_drain),
            _command_submit: Arc::clone(&ctx.command_submit),
            tick_interval: renderer.tick_interval,
            draw_interval: renderer.draw_interval,
            speed_multiplier: 1.0,
            paused: false,
            help_visible: false,
            sim_accumulator: 0.0,
            last_tick: Instant::now(),
            last_draw: Instant::now(),
            last_event_check: Instant::now(),
            palette,
            snapshot: Snapshot::default(),
        };
        app.refresh_snapshot();
        app
    }

    fn maybe_step_simulation(&mut self, now: Instant) {
        let delta = now - self.last_tick;
        self.last_tick = now;

        let effective_speed = if self.paused {
            0.0
        } else {
            self.speed_multiplier.max(0.0)
        };

        if effective_speed <= f32::EPSILON {
            return;
        }

        let step_interval = self.tick_interval.as_secs_f32();
        if step_interval <= f32::EPSILON {
            return;
        }

        self.sim_accumulator += delta.as_secs_f32() * effective_speed;
        let max_accumulator = step_interval * MAX_STEPS_PER_FRAME as f32;
        if self.sim_accumulator > max_accumulator {
            self.sim_accumulator = max_accumulator;
        }

        let mut steps = (self.sim_accumulator / step_interval).floor() as usize;
        if steps == 0 {
            return;
        }
        if steps > MAX_STEPS_PER_FRAME {
            steps = MAX_STEPS_PER_FRAME;
        }
        self.sim_accumulator -= step_interval * steps as f32;

        if let Ok(mut world) = self.world.lock() {
            (self.command_drain.as_ref())(&mut world);
            for _ in 0..steps {
                world.step();
            }
        }

        self.refresh_snapshot();
    }

    fn step_once(&mut self) {
        if let Ok(mut world) = self.world.lock() {
            (self.command_drain.as_ref())(&mut world);
            world.step();
        }
        self.refresh_snapshot();
    }

    fn draw(&mut self, frame: &mut Frame<'_>) {
        self.refresh_snapshot();
        let snapshot = self.snapshot.clone();

        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(7),
                Constraint::Min(0),
            ])
            .split(frame.area());

        self.draw_header(frame, layout[0], &snapshot);
        self.draw_history(frame, layout[1], &snapshot);
        self.draw_map(frame, layout[2], &snapshot);

        if self.help_visible {
            self.draw_help(frame);
        }
    }

    fn draw_header(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let status = format!(
            "Tick {:>6}  Epoch {:>3}  Agents {:>5}  Births {:>4}  Deaths {:>4}  Avg ⚡ {:>6.2}",
            snapshot.tick,
            snapshot.epoch,
            snapshot.agent_count,
            snapshot.births,
            snapshot.deaths,
            snapshot.avg_energy,
        );

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

        let paragraph = Paragraph::new(line).block(
            Block::default()
                .title(self.palette.title("ScriptBots Terminal HUD"))
                .borders(Borders::ALL),
        );
        frame.render_widget(paragraph, area);
    }

    fn draw_history(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let history_lines: Vec<Line> = snapshot
            .history
            .iter()
            .map(|entry| {
                Line::from(vec![
                    Span::raw(format!("t{:>6} ", entry.tick)),
                    Span::styled(
                        format!("Δ+{:>3} Δ-{:>3} ", entry.births, entry.deaths),
                        self.palette.accent_style(),
                    ),
                    Span::raw(format!("⚡ {:>6.2}", entry.avg_energy)),
                ])
            })
            .collect();

        let paragraph = Paragraph::new(Text::from(history_lines)).block(
            Block::default()
                .title(self.palette.title("Recent History"))
                .borders(Borders::ALL),
        );
        frame.render_widget(paragraph, area);
    }

    fn draw_map(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let map_lines: Vec<Line> = snapshot
            .map_rows
            .iter()
            .map(|row| Line::raw(row.clone()))
            .collect();

        let title = format!(
            "World Map {}×{}",
            snapshot.world_size.0, snapshot.world_size.1
        );
        let block = Block::default()
            .title(self.palette.title(title))
            .borders(Borders::ALL);

        let paragraph = Paragraph::new(Text::from(map_lines)).block(block);
        frame.render_widget(paragraph, area);
    }

    fn draw_help(&self, frame: &mut Frame<'_>) {
        let size = frame.area();
        let help_width = (size.width as f32 * 0.6).round() as u16;
        let help_height = 10;
        let help_x = size.x + (size.width - help_width) / 2;
        let help_y = size.y + (size.height - help_height) / 2;
        let area = Rect::new(help_x, help_y, help_width, help_height);

        let help_lines = vec![
            Line::from(vec![Span::styled(
                "Controls",
                self.palette.header_style().add_modifier(Modifier::BOLD),
            )]),
            Line::raw(" q      Quit"),
            Line::raw(" space  Toggle pause"),
            Line::raw(" + / - Adjust speed"),
            Line::raw(" s      Single step"),
            Line::raw(" ?      Toggle this help"),
        ];

        let paragraph = Paragraph::new(help_lines).block(
            Block::default()
                .title(self.palette.title("Help"))
                .borders(Borders::ALL)
                .style(Style::default().bg(Color::Black).fg(Color::White)),
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
            (KeyCode::Char(' '), _) => {
                self.paused = !self.paused;
                if self.paused {
                    self.speed_multiplier = 0.0;
                } else if self.speed_multiplier <= 0.0 {
                    self.speed_multiplier = 1.0;
                }
            }
            (KeyCode::Char('+') | KeyCode::Char('='), _) => {
                self.speed_multiplier = (self.speed_multiplier + 0.5).clamp(0.5, 8.0);
                if self.speed_multiplier > 0.0 {
                    self.paused = false;
                }
            }
            (KeyCode::Char('-') | KeyCode::Char('_'), _) => {
                self.speed_multiplier = (self.speed_multiplier - 0.5).max(0.0);
                if self.speed_multiplier <= 0.0 {
                    self.paused = true;
                }
            }
            (KeyCode::Char('s'), _) => {
                if let Ok(mut world) = self.world.lock() {
                    world.step();
                }
                self.paused = true;
                self.speed_multiplier = 0.0;
                self.refresh_snapshot();
            }
            (KeyCode::Char('?') | KeyCode::Char('h'), _) => {
                self.help_visible = !self.help_visible;
            }
            _ => {}
        }

        Ok(false)
    }

    fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    fn refresh_snapshot(&mut self) {
        if let Ok(world) = self.world.lock() {
            self.snapshot = Snapshot::from_world(&world, &self.palette);
        }
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
    history: Vec<HistoryEntry>,
    world_size: (u32, u32),
    map_rows: Vec<String>,
}

#[derive(Clone, Default, Debug)]
struct HistoryEntry {
    tick: u64,
    births: usize,
    deaths: usize,
    avg_energy: f32,
}

impl Snapshot {
    fn from_world(world: &WorldState, palette: &Palette) -> Self {
        let config = world.config();
        let agent_count = world.agent_count();

        let summaries: Vec<TickSummary> = world.history().cloned().collect();
        let summary = summaries.last().cloned().unwrap_or_else(|| TickSummary {
            tick: world.tick(),
            agent_count,
            births: 0,
            deaths: 0,
            total_energy: 0.0,
            average_energy: 0.0,
            average_health: 0.0,
        });

        let history = summaries
            .iter()
            .rev()
            .take(8)
            .map(|entry| HistoryEntry {
                tick: entry.tick.0,
                births: entry.births,
                deaths: entry.deaths,
                avg_energy: entry.average_energy,
            })
            .collect();

        let map_rows = build_map(
            world.agents().columns(),
            config.world_width,
            config.world_height,
            palette,
        );

        Self {
            tick: summary.tick.0,
            epoch: world.epoch(),
            agent_count,
            births: summary.births,
            deaths: summary.deaths,
            avg_energy: summary.average_energy,
            history,
            world_size: (config.world_width, config.world_height),
            map_rows,
        }
    }
}

fn build_map(
    columns: &AgentColumns,
    world_width: u32,
    world_height: u32,
    palette: &Palette,
) -> Vec<String> {
    let mut cells = vec![0usize; MAP_WIDTH * MAP_HEIGHT];
    let width = world_width.max(1) as f32;
    let height = world_height.max(1) as f32;

    for position in columns.positions() {
        let x = (position.x / width).clamp(0.0, 0.9999) * MAP_WIDTH as f32;
        let y = (position.y / height).clamp(0.0, 0.9999) * MAP_HEIGHT as f32;
        let xi = x.floor() as usize;
        let yi = y.floor() as usize;
        cells[yi * MAP_WIDTH + xi] += 1;
    }

    let mut rows = Vec::with_capacity(MAP_HEIGHT);
    for y in 0..MAP_HEIGHT {
        let mut row = String::with_capacity(MAP_WIDTH * 2);
        for x in 0..MAP_WIDTH {
            let count = cells[y * MAP_WIDTH + x];
            row.push_str(palette.map_symbol(count));
        }
        rows.push(row);
    }
    rows
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

struct Palette {
    level: Option<ColorLevel>,
}

impl Palette {
    fn detect() -> Self {
        Self {
            level: on_cached(Stream::Stdout),
        }
    }

    fn header_style(&self) -> Style {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    }

    fn accent_style(&self) -> Style {
        Style::default().fg(Color::LightMagenta)
    }

    fn paused_style(&self) -> Style {
        Style::default()
            .fg(Color::Black)
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD)
    }

    fn running_style(&self) -> Style {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Green)
            .add_modifier(Modifier::BOLD)
    }

    fn speed_style(&self, speed: f32) -> Style {
        let color = if speed > 1.0 {
            Color::Yellow
        } else if speed <= 0.0 {
            Color::DarkGray
        } else {
            Color::LightCyan
        };
        Style::default().fg(color)
    }

    fn title<T: Into<String>>(&self, title: T) -> Span<'static> {
        Span::styled(title.into(), self.header_style())
    }

    fn map_symbol(&self, count: usize) -> &'static str {
        if count == 0 {
            return "·";
        }
        match self.level {
            Some(level) if level.has_16m || level.has_256 => {
                if count <= 2 {
                    "🟢"
                } else if count <= 5 {
                    "🟠"
                } else {
                    "🔴"
                }
            }
            Some(level) if level.has_basic => {
                if count <= 2 {
                    "*"
                } else if count <= 5 {
                    "+"
                } else {
                    "#"
                }
            }
            _ => "#",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::ScriptBotsConfig;

    #[test]
    fn map_symbol_without_color_support_falls_back() {
        let palette = Palette { level: None };
        assert_eq!(palette.map_symbol(0), "·");
        assert_eq!(palette.map_symbol(1), "#");
        assert_eq!(palette.map_symbol(8), "#");
    }

    #[test]
    fn snapshot_reflects_world_state() {
        let config = ScriptBotsConfig::default();
        let mut world = WorldState::new(config).expect("world");
        world.spawn_agent(AgentData::default());

        let palette = Palette { level: None };
        let snapshot = Snapshot::from_world(&world, &palette);

        assert_eq!(snapshot.agent_count, world.agent_count());
        assert_eq!(snapshot.tick, world.tick().0);
        assert_eq!(snapshot.map_rows.len(), MAP_HEIGHT);
        assert!(snapshot.map_rows.iter().all(|row| !row.is_empty()));
    }
}
