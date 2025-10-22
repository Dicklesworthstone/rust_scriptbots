use std::env;
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Frame, Terminal,
    backend::{CrosstermBackend, TestBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
};
use scriptbots_core::{AgentColumns, TickSummary, WorldState};
use supports_color::{ColorLevel, Stream, on_cached};

use crate::{
    CommandDrain, CommandSubmit, ControlRuntime, SharedStorage, SharedWorld,
    renderer::{Renderer, RendererContext},
};

const TARGET_SIM_HZ: f32 = 60.0;
const MAX_STEPS_PER_FRAME: usize = 240;
const UI_TICK_MILLIS: u64 = 100;
const MAP_WIDTH: usize = 48;
const MAP_HEIGHT: usize = 24;

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
        if env::var_os("SCRIPTBOTS_TERMINAL_HEADLESS").is_some() {
            return self.run_headless(ctx);
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
        if event::poll(timeout).unwrap_or(false) {
            if let Event::Key(key) = event::read()? {
                if app.handle_key(key)? {
                    break;
                }
            }
            app.last_event_check = Instant::now();
        }
    }

    Ok(())
}

impl TerminalRenderer {
    fn run_headless(&self, ctx: RendererContext<'_>) -> Result<()> {
        let backend = TestBackend::new(80, 36);
        let mut terminal = Terminal::new(backend).context("failed to build test backend")?;
        let mut app = TerminalApp::new(self, ctx);

        for _ in 0..5 {
            app.step_once();
            terminal.draw(|frame| app.draw(frame))?;
        }

        Ok(())
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
            (KeyCode::Char('q'), _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
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

    fn refresh_snapshot(&mut self) {
        if let Ok(world) = self.world.lock() {
            self.snapshot = Snapshot::from_world(&world, &self.palette);
        }
    }
}

#[derive(Clone, Default)]
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

#[derive(Clone, Default)]
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
