use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use crossterm::{
    cursor::{Hide, Show},
    event::{self, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use owo_colors::OwoColorize;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Row, Table},
};
use reqwest::Client;
use scriptbots_app::{
    ConfigPatchRequest, ConfigSnapshot, KnobApplyRequest, KnobEntry, KnobKind, KnobUpdate,
};
use serde::de::DeserializeOwned;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(
    name = "scriptbots-control",
    version,
    about = "Interact with ScriptBots runtime controls via REST"
)]
struct Cli {
    /// Base URL for the running ScriptBots REST control API.
    #[arg(
        long,
        env = "SCRIPTBOTS_CONTROL_URL",
        default_value = "http://127.0.0.1:8080"
    )]
    base_url: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List all available configuration knobs and their current values.
    List,
    /// Fetch the complete configuration snapshot as JSON.
    Get,
    /// Update a configuration knob by path.
    Set {
        /// Dot-delimited knob path (e.g. world_width or neuroflow.enabled).
        path: String,
        /// New value expressed as JSON (unquoted literal, or a raw string fallback).
        value: String,
    },
    /// Apply a JSON patch object to the configuration.
    Patch {
        /// Optional file to read the JSON patch from; if omitted, read from the CLI arguments.
        #[arg(short, long)]
        file: Option<PathBuf>,
        /// Inline JSON (joined with spaces) when --file is not supplied.
        json: Vec<String>,
    },
    /// Launch an updating TUI dashboard to monitor knobs.
    Watch {
        /// Refresh interval in milliseconds.
        #[arg(long, default_value_t = 500)]
        interval_ms: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = Client::builder()
        .build()
        .context("failed to build HTTP client")?;

    match cli.command {
        Command::List => list_command(&client, &cli.base_url).await?,
        Command::Get => get_command(&client, &cli.base_url).await?,
        Command::Set { path, value } => set_command(&client, &cli.base_url, path, value).await?,
        Command::Patch { file, json } => patch_command(&client, &cli.base_url, file, json).await?,
        Command::Watch { interval_ms } => {
            run_watch(
                client.clone(),
                cli.base_url.clone(),
                Duration::from_millis(interval_ms),
            )
            .await?
        }
    }

    Ok(())
}

async fn list_command(client: &Client, base_url: &str) -> Result<()> {
    let knobs: Vec<KnobEntry> = fetch_knobs(client, base_url).await?;
    if knobs.is_empty() {
        println!("{}", "No knobs reported by the control API".yellow());
        return Ok(());
    }

    let mut knobs = knobs;
    knobs.sort_by(|a, b| a.path.cmp(&b.path));

    println!(
        "{:<54} {:<10} {}",
        "PATH".bold().cyan(),
        "TYPE".bold().cyan(),
        "VALUE".bold().cyan()
    );
    println!("{}", "-".repeat(96).dimmed());

    for entry in knobs {
        println!(
            "{:<54} {:<10} {}",
            entry.path.bold(),
            knob_kind_label(entry.kind).italic().blue(),
            value_to_string(&entry.value, 32)
        );
    }

    Ok(())
}

async fn get_command(client: &Client, base_url: &str) -> Result<()> {
    let snapshot: ConfigSnapshot = fetch_config(client, base_url).await?;
    let pretty = serde_json::to_string_pretty(&snapshot.config)
        .context("failed to format configuration JSON")?;
    println!("{} {}", "tick".green().bold(), snapshot.tick);
    println!("{}", pretty.cyan());
    Ok(())
}

async fn set_command(
    client: &Client,
    base_url: &str,
    path: String,
    raw_value: String,
) -> Result<()> {
    let value = parse_value(&raw_value)?;
    let request = KnobApplyRequest {
        updates: vec![KnobUpdate {
            path: path.clone(),
            value: value.clone(),
        }],
    };
    let snapshot: ConfigSnapshot = apply_updates(client, base_url, &request).await?;
    let resolved = snapshot
        .config
        .pointer(&json_pointer(&path))
        .cloned()
        .unwrap_or(Value::Null);
    println!(
        "{} {} => {}",
        "updated".green().bold(),
        path,
        value_to_string(&resolved, 48)
    );
    Ok(())
}

async fn patch_command(
    client: &Client,
    base_url: &str,
    file: Option<PathBuf>,
    inline: Vec<String>,
) -> Result<()> {
    let patch_value = if let Some(path) = file {
        let data = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read patch file {}", path.display()))?;
        serde_json::from_str::<Value>(&data)
            .with_context(|| format!("patch file {} did not contain valid JSON", path.display()))?
    } else {
        let joined = inline.join(" ");
        if joined.trim().is_empty() {
            bail!("provide either --file or inline JSON patch arguments");
        }
        serde_json::from_str::<Value>(&joined)
            .context("inline patch arguments were not valid JSON")?
    };

    if !patch_value.is_object() {
        bail!("patch payload must be a JSON object");
    }

    let body = ConfigPatchRequest { patch: patch_value };
    let snapshot: ConfigSnapshot = apply_patch(client, base_url, &body).await?;
    println!("{}", "Configuration patch applied".green().bold());
    println!("tick {}", snapshot.tick);
    Ok(())
}

async fn run_watch(client: Client, base_url: String, interval: Duration) -> Result<()> {
    tokio::task::spawn_blocking(move || watch_blocking(client, base_url, interval)).await??;
    Ok(())
}

fn watch_blocking(client: Client, base_url: String, interval: Duration) -> Result<()> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, Hide).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;
    let _cleanup = TerminalCleanup;

    let handle = tokio::runtime::Handle::current();
    let mut previous: HashMap<String, Value> = HashMap::new();
    let mut rows: Vec<WatchRow> = Vec::new();
    let mut last_error: Option<String> = None;
    let mut last_refresh = Instant::now() - interval;

    loop {
        if last_refresh.elapsed() >= interval {
            match handle.block_on(fetch_knobs(&client, &base_url)) {
                Ok(entries) => {
                    rows = build_rows(&entries, &previous);
                    previous = entries
                        .into_iter()
                        .map(|entry| (entry.path, entry.value))
                        .collect();
                    last_error = None;
                }
                Err(err) => {
                    last_error = Some(err.to_string());
                }
            }
            last_refresh = Instant::now();
        }

        terminal
            .draw(|frame| draw_watch(frame, &rows, last_error.as_deref(), interval))
            .context("failed to draw watch UI")?;

        if event::poll(Duration::from_millis(100)).context("failed to poll terminal events")? {
            if let Event::Key(key) = event::read().context("failed to read terminal event")? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => break,
                    KeyCode::Char('r') => {
                        // Force refresh on demand.
                        last_refresh = Instant::now() - interval;
                    }
                    _ => {}
                }
            }
        }
    }

    terminal.show_cursor().ok();
    Ok(())
}

fn build_rows(entries: &[KnobEntry], previous: &HashMap<String, Value>) -> Vec<WatchRow> {
    let mut rows = Vec::with_capacity(entries.len());
    for entry in entries {
        let changed = previous
            .get(&entry.path)
            .map(|value| value != &entry.value)
            .unwrap_or(true);
        rows.push(WatchRow {
            path: entry.path.clone(),
            kind: entry.kind,
            value: entry.value.clone(),
            changed,
        });
    }
    rows.sort_by(|a, b| a.path.cmp(&b.path));
    rows
}

fn draw_watch(
    frame: &mut ratatui::Frame<CrosstermBackend<std::io::Stdout>>,
    rows: &[WatchRow],
    error: Option<&str>,
    interval: Duration,
) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(1)])
        .split(frame.size());

    let mut header_lines = vec![
        Line::from(vec![
            Span::styled(
                "ScriptBots Control Dashboard ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("(press 'q' to quit, 'r' to refresh)"),
        ]),
        Line::from(Span::styled(
            format!("Refresh interval: {} ms", interval.as_millis()),
            Style::default().fg(Color::Gray),
        )),
    ];

    if let Some(message) = error {
        header_lines.push(Line::from(Span::styled(
            format!("Last error: {message}"),
            Style::default().fg(Color::Red),
        )));
    }

    let header = Paragraph::new(header_lines).block(Block::default().borders(Borders::ALL));
    frame.render_widget(header, layout[0]);

    if rows.is_empty() {
        let empty = Paragraph::new("Waiting for knob data...")
            .block(Block::default().borders(Borders::ALL));
        frame.render_widget(empty, layout[1]);
        return;
    }

    let table_rows: Vec<Row> = rows
        .iter()
        .map(|row| {
            let style = if row.changed {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                Span::styled(&row.path, style),
                Span::styled(
                    knob_kind_label(row.kind),
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::ITALIC),
                ),
                Span::styled(value_to_string(&row.value, 40), style),
            ])
        })
        .collect();

    let table = Table::new(table_rows)
        .header(Row::new(vec![
            Span::styled(
                "PATH",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "TYPE",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "VALUE",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]))
        .block(Block::default().borders(Borders::ALL))
        .widths(&[
            Constraint::Percentage(45),
            Constraint::Length(12),
            Constraint::Percentage(43),
        ])
        .column_spacing(2);

    frame.render_widget(table, layout[1]);
}

struct TerminalCleanup;

impl Drop for TerminalCleanup {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, Show);
    }
}

#[derive(Clone)]
struct WatchRow {
    path: String,
    kind: KnobKind,
    value: Value,
    changed: bool,
}

async fn fetch_knobs(client: &Client, base_url: &str) -> Result<Vec<KnobEntry>> {
    let url = join_url(base_url, "/api/knobs");
    let response = client
        .get(url)
        .send()
        .await
        .context("failed to fetch knob list")?;
    parse_response(response).await
}

async fn fetch_config(client: &Client, base_url: &str) -> Result<ConfigSnapshot> {
    let url = join_url(base_url, "/api/config");
    let response = client
        .get(url)
        .send()
        .await
        .context("failed to fetch configuration")?;
    parse_response(response).await
}

async fn apply_updates(
    client: &Client,
    base_url: &str,
    request: &KnobApplyRequest,
) -> Result<ConfigSnapshot> {
    let url = join_url(base_url, "/api/knobs/apply");
    let response = client
        .post(url)
        .json(request)
        .send()
        .await
        .context("failed to apply knob updates")?;
    parse_response(response).await
}

async fn apply_patch(
    client: &Client,
    base_url: &str,
    request: &ConfigPatchRequest,
) -> Result<ConfigSnapshot> {
    let url = join_url(base_url, "/api/config");
    let response = client
        .patch(url)
        .json(request)
        .send()
        .await
        .context("failed to apply configuration patch")?;
    parse_response(response).await
}

async fn parse_response<T>(response: reqwest::Response) -> Result<T>
where
    T: DeserializeOwned,
{
    let status = response.status();
    if status.is_success() {
        response
            .json::<T>()
            .await
            .context("failed to deserialize control API response")
    } else {
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<unavailable>".to_string());
        bail!("control API request failed ({status}): {body}");
    }
}

fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    if path.starts_with('/') {
        format!("{base}{path}")
    } else {
        format!("{base}/{path}")
    }
}

fn parse_value(raw: &str) -> Result<Value> {
    match serde_json::from_str::<Value>(raw) {
        Ok(value) => Ok(value),
        Err(_) => Ok(Value::String(raw.to_string())),
    }
}

fn knob_kind_label(kind: KnobKind) -> &'static str {
    match kind {
        KnobKind::Number => "number",
        KnobKind::Integer => "integer",
        KnobKind::Boolean => "bool",
        KnobKind::String => "string",
        KnobKind::Array => "array",
        KnobKind::Object => "object",
        KnobKind::Null => "null",
    }
}

fn value_to_string(value: &Value, max_len: usize) -> String {
    let raw = match value {
        Value::String(s) => format!("\"{}\"", s),
        other => serde_json::to_string(other).unwrap_or_else(|_| "<error>".to_string()),
    };
    if raw.len() > max_len {
        format!("{}…", &raw[..max_len.saturating_sub(1)])
    } else {
        raw
    }
}

fn json_pointer(path: &str) -> String {
    let mut pointer = String::from("/");
    pointer.push_str(&path.replace('.', "/"));
    pointer
}
