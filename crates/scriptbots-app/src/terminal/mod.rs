use std::{
    cmp::Ordering,
    collections::VecDeque,
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
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline},
};
use scriptbots_core::{
    AgentId, ControlSettings, TerrainKind, TerrainLayer, TickSummary, WorldState,
};
use serde::Serialize;
use slotmap::Key;
use supports_color::{ColorLevel, Stream, on_cached};
use tracing::info;

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
    terrain: TerrainView,
    event_log: VecDeque<EventEntry>,
    last_event_tick: u64,
    snapshot: Snapshot,
    baseline: Option<Baseline>,
    last_autopause_tick: Option<u64>,
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
            terrain,
            event_log: VecDeque::with_capacity(EVENT_LOG_CAPACITY),
            last_event_tick: 0,
            snapshot: Snapshot::default(),
            baseline: None,
            last_autopause_tick: None,
        };
        app.refresh_snapshot();
        app
    }

    fn maybe_step_simulation(&mut self, now: Instant) {
        let delta = now - self.last_tick;
        self.last_tick = now;

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

        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(0)])
            .split(frame.area());

        self.draw_header(frame, outer[0], &snapshot);

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(outer[1]);

        self.draw_map(frame, body[0], &snapshot);

        let sidebar = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(5),
                Constraint::Length((LEADERBOARD_LIMIT as u16 + 3).min(12)),
                Constraint::Length((LEADERBOARD_LIMIT as u16 + 3).min(12)),
                Constraint::Min(3),
            ])
            .split(body[1]);

        self.draw_stats(frame, sidebar[0], &snapshot);
        self.draw_trends(frame, sidebar[1], &snapshot);
        self.draw_leaderboard(frame, sidebar[2], &snapshot);
        self.draw_oldest(frame, sidebar[3], &snapshot);
        self.draw_events(frame, sidebar[4], &snapshot);

        if self.help_visible {
            self.draw_help(frame);
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

        let paragraph = Paragraph::new(Text::from(lines)).block(
            Block::default()
                .title(self.palette.title("Vital Stats"))
                .borders(Borders::ALL),
        );
        frame.render_widget(paragraph, area);
    }

    fn draw_trends(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let block = Block::default()
            .title(self.palette.title("Population & Energy Trends"))
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height == 0 {
            return;
        }

        let trend_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
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
        frame.render_widget(trend_text, trend_layout[2]);
    }

    fn draw_map(&self, frame: &mut Frame<'_>, area: Rect, snapshot: &Snapshot) {
        let title = format!(
            "World Map {}×{}",
            snapshot.world_size.0, snapshot.world_size.1
        );
        let block = Block::default()
            .title(self.palette.title(title))
            .borders(Borders::ALL);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.width < 2 || inner.height < 2 {
            return;
        }

        let width = inner.width as usize;
        let height = inner.height as usize;
        let mut grid = vec![CellGlyph::default(); width * height];
        let mut occupancy = vec![CellOccupancy::default(); width * height];

        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32;
                let v = (y as f32 + 0.5) / height as f32;
                let terrain = self.terrain.sample(u, v);
                let food_level = snapshot.food.sample(u, v);
                let (glyph, style) = self.palette.terrain_symbol(terrain, food_level);
                grid[y * width + x] = CellGlyph { ch: glyph, style };
            }
        }

        for agent in &snapshot.agents {
            let x = (agent.position.0 * width as f32)
                .floor()
                .clamp(0.0, (width - 1) as f32) as usize;
            let y = (agent.position.1 * height as f32)
                .floor()
                .clamp(0.0, (height - 1) as f32) as usize;
            let idx = y * width + x;
            let cell = &mut occupancy[idx];
            cell.add(
                agent.diet,
                agent.boosted,
                agent.energy,
                agent.heading,
                agent.spike_length,
                agent.tendency,
            );
            let base = grid[idx].style;
            let (glyph, style) = self.palette.agent_symbol(cell, base);
            grid[idx].ch = glyph;
            grid[idx].style = style;
        }

        let mut lines = Vec::with_capacity(height);
        for y in 0..height {
            let mut spans = Vec::with_capacity(width);
            for x in 0..width {
                let cell = &grid[y * width + x];
                spans.push(Span::styled(cell.ch.to_string(), cell.style));
            }
            lines.push(Line::from(spans));
        }

        let paragraph = Paragraph::new(Text::from(lines));
        frame.render_widget(paragraph, inner);
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
            Line::raw(" S      Save ASCII screenshot"),
            Line::raw(" b      Toggle metrics baseline (set/clear)"),
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
            _ => {}
        }

        Ok(false)
    }

    fn save_ascii_snapshot(&self) -> Result<()> {
        use std::io::Write;
        let dir = std::path::Path::new("screenshots");
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
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
            Ok(world) => Snapshot::from_world(&world),
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

#[derive(Clone, Debug)]
struct CellGlyph {
    ch: char,
    style: Style,
}

impl Default for CellGlyph {
    fn default() -> Self {
        Self {
            ch: ' ',
            style: Style::default(),
        }
    }
}

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
        }
    }
}

impl CellOccupancy {
    fn add(
        &mut self,
        class: DietClass,
        boosted: bool,
        energy: f32,
        heading: f32,
        spike: f32,
        tendency: f32,
    ) {
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

        let summaries: Vec<TickSummary> = world.history().cloned().collect();
        let summary = summaries.last().cloned().unwrap_or_else(|| TickSummary {
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

        let history = summaries
            .iter()
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
        }
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

    fn diet_style(&self, diet: DietClass) -> Style {
        Style::default().fg(self.diet_color(diet))
    }

    fn population_spark_style(&self) -> Style {
        Style::default().fg(Color::Green)
    }

    fn energy_spark_style(&self) -> Style {
        Style::default().fg(Color::Yellow)
    }

    fn event_style(&self, kind: EventKind) -> Style {
        let color = match kind {
            EventKind::Birth => Color::Green,
            EventKind::Death => Color::Red,
            EventKind::Population => Color::Yellow,
            EventKind::Info => Color::Cyan,
        };
        Style::default().fg(color)
    }

    fn has_color(&self) -> bool {
        self.level.is_some()
    }

    fn diet_color(&self, diet: DietClass) -> Color {
        match diet {
            DietClass::Herbivore => Color::Green,
            DietClass::Omnivore => Color::Yellow,
            DietClass::Carnivore => Color::Red,
        }
    }

    fn terrain_symbol(&self, kind: TerrainKind, food_level: f32) -> (char, Style) {
        let rich_color = self
            .level
            .is_some_and(|level| level.has_16m || level.has_256);
        let (glyph, fg, bg) = match kind {
            TerrainKind::DeepWater => ('≈', Color::Cyan, Color::Blue),
            TerrainKind::ShallowWater => (
                '~',
                Color::Cyan,
                if rich_color {
                    Color::Rgb(0, 80, 160)
                } else {
                    Color::Blue
                },
            ),
            TerrainKind::Sand => (
                '·',
                Color::Yellow,
                if rich_color {
                    Color::Rgb(160, 120, 50)
                } else {
                    Color::Yellow
                },
            ),
            TerrainKind::Grass => (
                '"',
                Color::LightGreen,
                if rich_color {
                    Color::Rgb(30, 90, 30)
                } else {
                    Color::Green
                },
            ),
            TerrainKind::Bloom => (
                '*',
                Color::Magenta,
                if rich_color {
                    Color::Rgb(100, 30, 100)
                } else {
                    Color::Magenta
                },
            ),
            TerrainKind::Rock => (
                '^',
                Color::Gray,
                if rich_color {
                    Color::Rgb(70, 70, 70)
                } else {
                    Color::DarkGray
                },
            ),
        };
        let mut style = Style::default().fg(fg);
        if self.has_color() {
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
        let mut glyph = match total {
            0 => ' ',
            1 => occupancy
                .mean_heading()
                .map(Self::heading_char)
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
        };

        let mut style = base.fg(self.diet_color(class));
        if occupancy.boosted || total > 1 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if total > 3 {
            style = style.add_modifier(Modifier::REVERSED);
        }
        if occupancy.spike_peak > 0.6 {
            glyph = '!';
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

    fn heading_char(heading: f32) -> char {
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
