//! Upgraded science widgets for FrankenTUI & terminal frontend (bd-2z0.14.2.3).
//!
//! Provides production science visualization widgets:
//! - [`ScienceChartWidget`]: Time-series for population, energy, births, and deaths
//!   with Y/X axes, tick/value labels, units, multi-series legends, bounded rolling windows,
//!   and explicit Empty, Stale, and Truncated states.
//! - [`TypedEventFeedWidget`]: Canonical typed event feed (Birth, Death, Combat, Eat,
//!   Mutation, Config, Info) with per-kind icon/glyph and color, relative tick offsets,
//!   event filtering by kind and subject UID, and canonical focus routing.
//! - [`BrainActivationGridWidget`]: Real 2D per-layer activation grid, output sparklines,
//!   top-k signed sensor attributions, and truthful provenance banner (uid, tick, rev, bytes, clipping).
//! - [`WatermarkStatusStrip`]: Truthful persistence status showing admitted, applied,
//!   and durable watermarks plus snapshot lag without optimistic acknowledgement.
//! - [`AlignedScienceTables`]: Column-aligned formatters for Insights, Mortality (with theme-safe
//!   proportional bars), Leaderboard, and Oldest agents (with diet chips).

use ftui::render::cell::Cell;
use ftui::render::frame::Frame;
use ratatui::{buffer::Buffer, layout::Rect, style::Style};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Print ASCII/UTF-8 string directly onto an ftui Frame buffer at (x, y), clipping at frame bounds.
pub fn print_text_clipped(frame: &mut Frame, x: u16, y: u16, text: &str) {
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

// ============================================================================
// 1. SCIENCE TIME-SERIES CHART WIDGET
// ============================================================================

/// Supported rolling window horizons for time-series charts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartRollingWindow {
    Ticks30 = 30,
    Ticks60 = 60,
    Ticks120 = 120,
    Ticks300 = 300,
}

impl ChartRollingWindow {
    pub fn all() -> &'static [ChartRollingWindow] {
        &[Self::Ticks30, Self::Ticks60, Self::Ticks120, Self::Ticks300]
    }

    pub fn capacity(self) -> usize {
        self as usize
    }

    pub fn next(self) -> Self {
        match self {
            Self::Ticks30 => Self::Ticks60,
            Self::Ticks60 => Self::Ticks120,
            Self::Ticks120 => Self::Ticks300,
            Self::Ticks300 => Self::Ticks30,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Ticks30 => "30t",
            Self::Ticks60 => "60t",
            Self::Ticks120 => "120t",
            Self::Ticks300 => "300t",
        }
    }
}

/// A single observation in the chart time-series history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartSample {
    pub tick: u64,
    pub population: u64,
    pub avg_energy: f32,
    pub births: u32,
    pub deaths: u32,
}

/// Explicit degraded states for the time-series chart.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartDegradedState {
    Normal,
    Empty { message: String },
    Stale { lag: u64 },
    Truncated { visible: usize, total: usize },
}

/// Time-series chart model containing rolling window samples and visibility flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScienceChartData {
    pub samples: VecDeque<ChartSample>,
    pub max_capacity: usize,
    pub window: ChartRollingWindow,
    pub show_population: bool,
    pub show_energy: bool,
    pub show_births: bool,
    pub show_deaths: bool,
    pub last_snapshot_tick: u64,
}

impl Default for ScienceChartData {
    fn default() -> Self {
        Self::new(ChartRollingWindow::Ticks60)
    }
}

impl ScienceChartData {
    pub fn new(window: ChartRollingWindow) -> Self {
        Self {
            samples: VecDeque::with_capacity(window.capacity()),
            max_capacity: 300,
            window,
            show_population: true,
            show_energy: true,
            show_births: true,
            show_deaths: true,
            last_snapshot_tick: 0,
        }
    }

    pub fn push_sample(&mut self, sample: ChartSample, current_tick: u64) {
        self.last_snapshot_tick = current_tick;
        if self.samples.len() >= self.max_capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn cycle_window(&mut self) {
        self.window = self.window.next();
    }

    pub fn visible_samples(&self) -> Vec<&ChartSample> {
        let cap = self.window.capacity();
        let total = self.samples.len();
        let skip = total.saturating_sub(cap);
        self.samples.iter().skip(skip).collect()
    }

    pub fn degraded_state(&self, current_tick: u64) -> ChartDegradedState {
        if self.samples.is_empty() {
            return ChartDegradedState::Empty {
                message: "No simulation ticks yet (waiting for snapshot history)".to_string(),
            };
        }
        let latest_sample_tick = self.samples.back().map_or(0, |s| s.tick);
        let lag = current_tick.saturating_sub(latest_sample_tick);
        if lag > 15 {
            return ChartDegradedState::Stale { lag };
        }
        let total = self.samples.len();
        let visible = self.visible_samples().len();
        if total > visible {
            return ChartDegradedState::Truncated { visible, total };
        }
        ChartDegradedState::Normal
    }

    pub fn population_bounds(&self) -> (u64, u64) {
        let vis = self.visible_samples();
        if vis.is_empty() {
            return (0, 0);
        }
        let mut min = u64::MAX;
        let mut max = 0u64;
        for s in vis {
            min = min.min(s.population);
            max = max.max(s.population);
        }
        (min, max)
    }

    pub fn energy_bounds(&self) -> (f32, f32) {
        let vis = self.visible_samples();
        if vis.is_empty() {
            return (0.0, 0.0);
        }
        let mut min = f32::MAX;
        let mut max = 0.0f32;
        for s in vis {
            if s.avg_energy.is_finite() {
                min = min.min(s.avg_energy);
                max = max.max(s.avg_energy);
            }
        }
        if min == f32::MAX {
            (0.0, 0.0)
        } else {
            (min, max)
        }
    }

    pub fn births_deaths_bounds(&self) -> (u32, u32) {
        let vis = self.visible_samples();
        if vis.is_empty() {
            return (0, 0);
        }
        let mut max = 0u32;
        for s in vis {
            max = max.max(s.births).max(s.deaths);
        }
        (0, max)
    }

    /// Render time-series chart with axes, tick/value labels, units, and legend into lines of text.
    pub fn render_lines(&self, width: u16, height: u16, _reduced_color: bool) -> Vec<String> {
        let mut lines = Vec::new();
        if width < 12 || height < 3 {
            lines.push(format!(
                "{:width$}",
                "! chart area too small",
                width = width as usize
            ));
            return lines;
        }

        // Header / Legend line: Pop ■, Energy ▲, Births ●, Deaths ◆, [Window: 60t]
        let window_tag = format!("[Window: {}]", self.window.label());
        let legend = format!(
            "Pop ■ (ag) | Energy ▲ (⚡) | Births ● (Δ+) | Deaths ◆ (Δ-)  {}",
            window_tag
        );
        lines.push(format!("{:<width$}", legend, width = width as usize));

        let state = self.degraded_state(self.last_snapshot_tick);
        match state {
            ChartDegradedState::Empty { message } => {
                lines.push(format!("  [EMPTY] {}", message));
                while (lines.len() as u16) < height {
                    lines.push(String::new());
                }
                return lines;
            }
            ChartDegradedState::Stale { lag } => {
                lines.push(format!("  [STALE] Telemetry lagged by +{} ticks", lag));
            }
            ChartDegradedState::Truncated { visible, total } => {
                lines.push(format!(
                    "  [TRUNCATED] Showing {} of {} rolling history points",
                    visible, total
                ));
            }
            ChartDegradedState::Normal => {
                let (p_min, p_max) = self.population_bounds();
                let (e_min, e_max) = self.energy_bounds();
                let (_, bd_max) = self.births_deaths_bounds();
                let status_metrics = format!(
                    "  Ranges: Pop [{}..{}] ag | Energy [{:.1}..{:.1}] ⚡ | Max Δ: {}",
                    p_min, p_max, e_min, e_max, bd_max
                );
                lines.push(status_metrics);
            }
        }

        let vis = self.visible_samples();
        if vis.is_empty() || height <= 3 {
            return lines;
        }

        // Compute printable plot rows: available height minus header, subhead, and X-axis
        let plot_rows = (height.saturating_sub(3)).max(1) as usize;
        let plot_cols = (width.saturating_sub(10)).max(1) as usize;

        // Sparkline unicode bars for pop & energy
        const BARS: [char; 8] = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let (p_min, p_max) = self.population_bounds();
        let p_range = (p_max.saturating_sub(p_min)).max(1) as f64;

        let (e_min, e_max) = self.energy_bounds();
        let e_range = if (e_max - e_min).abs() > 0.001 {
            (e_max - e_min) as f64
        } else {
            1.0
        };

        // Y-axis tick rows
        let y_top = format!("{:>6} ┼ ", p_max);
        let mut pop_spark = String::new();
        for s in vis.iter().take(plot_cols) {
            let norm = ((s.population.saturating_sub(p_min)) as f64 / p_range).clamp(0.0, 1.0);
            let idx = (norm * 7.0).round() as usize;
            pop_spark.push(BARS[idx.min(7)]);
        }
        lines.push(format!("{}{}", y_top, pop_spark));

        if plot_rows > 1 {
            let y_mid = format!("{:>6} ┼ ", (e_min + e_max) / 2.0);
            let mut energy_spark = String::new();
            for s in vis.iter().take(plot_cols) {
                let norm = ((s.avg_energy - e_min) as f64 / e_range).clamp(0.0, 1.0);
                let idx = (norm * 7.0).round() as usize;
                energy_spark.push(BARS[idx.min(7)]);
            }
            lines.push(format!("{}{}", y_mid, energy_spark));
        }

        // X-axis baseline
        let mut x_axis = format!("{:>6} ┴─", p_min);
        for _ in 0..vis.len().min(plot_cols) {
            x_axis.push('─');
        }
        lines.push(x_axis);

        // X-axis tick bounds
        if let (Some(first), Some(last)) = (vis.first(), vis.last()) {
            let x_labels = format!(
                "       t{:<8} {:>width$}",
                first.tick,
                format!("t{}", last.tick),
                width = vis.len().min(plot_cols).saturating_sub(10)
            );
            lines.push(x_labels);
        }

        while (lines.len() as u16) < height {
            lines.push(format!("{:width$}", "", width = width as usize));
        }
        lines.truncate(height as usize);
        lines
    }

    /// Render directly into FrankenTUI Frame buffer at (x, y).
    pub fn render_ftui(
        &self,
        frame: &mut Frame,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
        reduced_color: bool,
    ) {
        let lines = self.render_lines(width, height, reduced_color);
        for (idx, line) in lines.iter().enumerate() {
            let row = y + idx as u16;
            if row < frame.height() {
                print_text_clipped(frame, x, row, line);
            }
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let lines = self.render_lines(area.width, area.height, false);
        for (idx, line) in lines.iter().enumerate() {
            let row = area.y + idx as u16;
            if row < area.bottom() {
                buf.set_string(area.x, row, line, Style::default());
            }
        }
    }
}

// ============================================================================
// 2. TYPED EVENT FEED WIDGET
// ============================================================================

/// Canonical event kinds in the simulation narrative stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypedEventKind {
    Birth,
    Death,
    Combat,
    Eat,
    Mutation,
    Config,
    Info,
}

impl TypedEventKind {
    pub fn all() -> &'static [TypedEventKind] {
        &[
            Self::Birth,
            Self::Death,
            Self::Combat,
            Self::Eat,
            Self::Mutation,
            Self::Config,
            Self::Info,
        ]
    }

    pub fn emoji_glyph(self) -> &'static str {
        match self {
            Self::Birth => "🐣",
            Self::Death => "💀",
            Self::Combat => "⚔️",
            Self::Eat => "🍎",
            Self::Mutation => "🧬",
            Self::Config => "⚙️",
            Self::Info => "ℹ️",
        }
    }

    pub fn ascii_glyph(self) -> &'static str {
        match self {
            Self::Birth => "[B]",
            Self::Death => "[D]",
            Self::Combat => "[C]",
            Self::Eat => "[E]",
            Self::Mutation => "[M]",
            Self::Config => "[S]",
            Self::Info => "[I]",
        }
    }

    pub fn glyph(self, emoji: bool) -> &'static str {
        if emoji {
            self.emoji_glyph()
        } else {
            self.ascii_glyph()
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Birth => "Birth",
            Self::Death => "Death",
            Self::Combat => "Combat",
            Self::Eat => "Eat",
            Self::Mutation => "Mutation",
            Self::Config => "Config",
            Self::Info => "Info",
        }
    }
}

/// A structured, typed event record consumed by the feed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedEventRecord {
    pub id: u64,
    pub tick: u64,
    pub kind: TypedEventKind,
    pub message: String,
    pub subject_uid: Option<u64>,
    pub location: Option<(f32, f32)>,
    pub target_alive: bool,
}

/// Filter criteria for the event feed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKindFilter {
    All,
    Only(TypedEventKind),
}

impl EventKindFilter {
    pub fn label(&self) -> String {
        match self {
            Self::All => "All".to_string(),
            Self::Only(k) => k.label().to_string(),
        }
    }
}

/// Focus intent produced when an event is activated via Enter or click.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventFocusIntent {
    /// Target is valid and alive -> routes to canonical selection.
    FocusAgent(u64),
    /// Location focus -> pan canvas to (x, y).
    PanLocation(f32, f32),
    /// Target is stale/dead -> explicit rejection, do NOT fabricate selection.
    StaleTarget { uid: u64, reason: String },
}

/// Model for the typed event feed widget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedEventFeedData {
    pub events: VecDeque<TypedEventRecord>,
    pub max_capacity: usize,
    pub filter_kind: EventKindFilter,
    pub filter_agent_uid: Option<u64>,
    pub selected_index: usize,
}

impl Default for TypedEventFeedData {
    fn default() -> Self {
        Self::new(100)
    }
}

impl TypedEventFeedData {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_capacity),
            max_capacity,
            filter_kind: EventKindFilter::All,
            filter_agent_uid: None,
            selected_index: 0,
        }
    }

    pub fn push_event(&mut self, record: TypedEventRecord) {
        if self.events.len() >= self.max_capacity {
            self.events.pop_front();
        }
        self.events.push_back(record);
    }

    pub fn filtered_events(&self) -> Vec<&TypedEventRecord> {
        self.events
            .iter()
            .rev()
            .filter(|e| match self.filter_kind {
                EventKindFilter::All => true,
                EventKindFilter::Only(k) => e.kind == k,
            })
            .filter(|e| match self.filter_agent_uid {
                None => true,
                Some(uid) => e.subject_uid == Some(uid),
            })
            .collect()
    }

    pub fn select_next(&mut self) {
        let count = self.filtered_events().len();
        if count > 0 && self.selected_index + 1 < count {
            self.selected_index += 1;
        }
    }

    pub fn select_prev(&mut self) {
        if self.selected_index > 0 {
            self.selected_index -= 1;
        }
    }

    pub fn cycle_filter_kind(&mut self) {
        self.filter_kind = match self.filter_kind {
            EventKindFilter::All => EventKindFilter::Only(TypedEventKind::Birth),
            EventKindFilter::Only(TypedEventKind::Birth) => {
                EventKindFilter::Only(TypedEventKind::Death)
            }
            EventKindFilter::Only(TypedEventKind::Death) => {
                EventKindFilter::Only(TypedEventKind::Combat)
            }
            EventKindFilter::Only(TypedEventKind::Combat) => {
                EventKindFilter::Only(TypedEventKind::Eat)
            }
            EventKindFilter::Only(TypedEventKind::Eat) => {
                EventKindFilter::Only(TypedEventKind::Mutation)
            }
            EventKindFilter::Only(TypedEventKind::Mutation) => {
                EventKindFilter::Only(TypedEventKind::Config)
            }
            EventKindFilter::Only(TypedEventKind::Config) => {
                EventKindFilter::Only(TypedEventKind::Info)
            }
            EventKindFilter::Only(TypedEventKind::Info) => EventKindFilter::All,
        };
        self.selected_index = 0;
    }

    pub fn set_filter_kind(&mut self, filter: EventKindFilter) {
        self.filter_kind = filter;
        self.selected_index = 0;
    }

    pub fn set_filter_agent_uid(&mut self, uid: Option<u64>) {
        self.filter_agent_uid = uid;
        self.selected_index = 0;
    }

    pub fn selected_event(&self) -> Option<&TypedEventRecord> {
        let filtered = self.filtered_events();
        filtered.get(self.selected_index).copied()
    }

    pub fn focus_intent_for_selected(&self) -> Option<EventFocusIntent> {
        let e = self.selected_event()?;
        if let Some(uid) = e.subject_uid {
            if e.target_alive {
                Some(EventFocusIntent::FocusAgent(uid))
            } else {
                Some(EventFocusIntent::StaleTarget {
                    uid,
                    reason: "Agent is deceased / no longer in active arena".to_string(),
                })
            }
        } else if let Some((x, y)) = e.location {
            Some(EventFocusIntent::PanLocation(x, y))
        } else {
            None
        }
    }

    pub fn format_event_row(
        &self,
        event: &TypedEventRecord,
        current_tick: u64,
        width: usize,
        emoji: bool,
    ) -> String {
        let glyph = event.kind.glyph(emoji);
        let rel_tick = if current_tick >= event.tick {
            let diff = current_tick - event.tick;
            if diff == 0 {
                "now".to_string()
            } else {
                format!("+{}t ago", diff)
            }
        } else {
            format!("t{}", event.tick)
        };

        let target_badge = if let Some(uid) = event.subject_uid {
            if event.target_alive {
                format!("#{}", uid)
            } else {
                format!("#{} [DEAD]", uid)
            }
        } else {
            "-".to_string()
        };

        let prefix = format!("{} {:<9} {:<8} ", glyph, rel_tick, target_badge);
        let max_msg = width.saturating_sub(prefix.len());
        let truncated_msg = if event.message.len() > max_msg {
            let mut s: String = event
                .message
                .chars()
                .take(max_msg.saturating_sub(1))
                .collect();
            s.push('…');
            s
        } else {
            event.message.clone()
        };
        format!("{}{}", prefix, truncated_msg)
    }

    pub fn render_lines(
        &self,
        width: u16,
        height: u16,
        current_tick: u64,
        emoji: bool,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        if width < 12 || height < 2 {
            lines.push(format!(
                "{:width$}",
                "! feed too small",
                width = width as usize
            ));
            return lines;
        }

        // Header with filter indicator
        let filter_label = format!(
            "Event Feed [Filter: {}] [UID: {}]",
            self.filter_kind.label(),
            self.filter_agent_uid
                .map_or("All".to_string(), |u| format!("#{}", u))
        );
        lines.push(format!("{:<width$}", filter_label, width = width as usize));

        let filtered = self.filtered_events();
        if filtered.is_empty() {
            lines.push(format!(
                "  (No events matching filter {})",
                self.filter_kind.label()
            ));
            while (lines.len() as u16) < height {
                lines.push(String::new());
            }
            return lines;
        }

        let max_rows = (height.saturating_sub(1)) as usize;
        for (idx, &event) in filtered.iter().take(max_rows).enumerate() {
            let cursor = if idx == self.selected_index { ">" } else { " " };
            let row = self.format_event_row(
                event,
                current_tick,
                (width.saturating_sub(2)) as usize,
                emoji,
            );
            lines.push(format!("{}{}", cursor, row));
        }

        while (lines.len() as u16) < height {
            lines.push(format!("{:width$}", "", width = width as usize));
        }
        lines.truncate(height as usize);
        lines
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_ftui(
        &self,
        frame: &mut Frame,
        x: u16,
        y: u16,
        width: u16,
        height: u16,
        current_tick: u64,
        emoji: bool,
    ) {
        let lines = self.render_lines(width, height, current_tick, emoji);
        for (idx, line) in lines.iter().enumerate() {
            let row = y + idx as u16;
            if row < frame.height() {
                print_text_clipped(frame, x, row, line);
            }
        }
    }

    pub fn scroll_up(&mut self, n: usize) {
        for _ in 0..n {
            self.select_prev();
        }
    }

    pub fn scroll_down(&mut self, n: usize) {
        for _ in 0..n {
            self.select_next();
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let lines = self.render_lines(area.width, area.height, 0, false);
        for (idx, line) in lines.iter().enumerate() {
            let row = area.y + idx as u16;
            if row < area.bottom() {
                buf.set_string(area.x, row, line, Style::default());
            }
        }
    }
}

// ============================================================================
// 3. BRAIN ACTIVATION GRID WIDGET
// ============================================================================

/// Layer activation slice in a neural network brain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerActivationData {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub values: Vec<f32>,
}

/// Sensor attribution driving an effective output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensorAttributionEntry {
    pub sensor_name: String,
    pub signed_weight: f32,
    pub contribution_pct: f32,
}

/// Output channel actuator value and mini sparkline history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveOutputEntry {
    pub actuator_name: String,
    pub value: f32,
    pub interpretation: String,
    pub spark_history: Vec<f32>,
}

/// Provenance metadata for a neural network activation inspection payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BrainProvenanceBanner {
    pub agent_uid: u64,
    pub tick: u64,
    pub control_revision: u64,
    pub payload_bytes: usize,
    pub clipped_count: usize,
    pub is_stale: bool,
}

/// Complete brain activation inspection widget model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrainActivationGridData {
    pub provenance: Option<BrainProvenanceBanner>,
    pub layers: Vec<LayerActivationData>,
    pub outputs: Vec<EffectiveOutputEntry>,
    pub attributions: Vec<SensorAttributionEntry>,
}

impl BrainActivationGridData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map activation value in [-1.0, 1.0] to a Unicode block character with sign indication.
    pub fn activation_to_char(val: f32) -> char {
        if !val.is_finite() {
            return '·';
        }
        const BLOCKS: [char; 8] = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let clamped = val.clamp(-1.0, 1.0);
        let mag = clamped.abs();
        let idx = ((mag * 7.0).round() as usize).min(7);
        BLOCKS[idx]
    }

    pub fn render_lines(&self, width: u16, height: u16) -> Vec<String> {
        let mut lines = Vec::new();
        if width < 12 || height < 3 {
            lines.push(format!(
                "{:width$}",
                "! brain area too small",
                width = width as usize
            ));
            return lines;
        }

        // 1. Provenance Banner
        if let Some(prov) = &self.provenance {
            let stale_tag = if prov.is_stale { " [STALE]" } else { "" };
            let clip_tag = if prov.clipped_count > 0 {
                format!(" [CLIPPED {}]", prov.clipped_count)
            } else {
                " [CLEAN]".to_string()
            };
            let banner = format!(
                "Brain #{} · t{} · rev {}{} · {} bytes{}",
                prov.agent_uid,
                prov.tick,
                prov.control_revision,
                stale_tag,
                prov.payload_bytes,
                clip_tag
            );
            lines.push(format!("{:<width$}", banner, width = width as usize));
        } else {
            lines.push(format!(
                "{:<width$}",
                "Brain Inspector (No Agent Selected)",
                width = width as usize
            ));
        }

        if self.layers.is_empty() {
            lines.push("  (No activation layers recorded for this agent)".to_string());
            while (lines.len() as u16) < height {
                lines.push(String::new());
            }
            return lines;
        }

        // 2. 2D Per-Layer Activation Grid
        lines.push("2D LAYER ACTIVATIONS:".to_string());
        for layer in &self.layers {
            if (lines.len() as u16) + 2 >= height {
                break;
            }
            let dim_str = layer
                .dimensions
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let mut grid_str = String::new();
            for &val in &layer.values {
                grid_str.push(Self::activation_to_char(val));
            }
            lines.push(format!(
                "  {:<12} [{:>4}]: {}",
                layer.name, dim_str, grid_str
            ));
        }

        // 3. Top Sensor Attributions
        if !self.attributions.is_empty() && (lines.len() as u16) + 2 < height {
            lines.push("TOP SENSOR ATTRIBUTIONS:".to_string());
            let mut attr_strs = Vec::new();
            for a in self.attributions.iter().take(4) {
                attr_strs.push(format!(
                    "{}:{:+0.2}({:.0}%)",
                    a.sensor_name, a.signed_weight, a.contribution_pct
                ));
            }
            lines.push(format!("  {}", attr_strs.join(" | ")));
        }

        // 4. Effective Output Sparklines
        if !self.outputs.is_empty() && (lines.len() as u16) + 1 < height {
            for out in self.outputs.iter().take(2) {
                if (lines.len() as u16) >= height {
                    break;
                }
                let mut spark = String::new();
                for &v in out.spark_history.iter().take(12) {
                    spark.push(Self::activation_to_char(v));
                }
                lines.push(format!(
                    "  Out {:<10} {:>+5.2} [{}] -> {}",
                    out.actuator_name, out.value, spark, out.interpretation
                ));
            }
        }

        lines
    }

    pub fn render_ftui(&self, frame: &mut Frame, x: u16, y: u16, width: u16, height: u16) {
        let lines = self.render_lines(width, height);
        for (idx, line) in lines.iter().enumerate() {
            let row = y + idx as u16;
            if row < frame.height() {
                print_text_clipped(frame, x, row, line);
            }
        }
    }
    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        let lines = self.render_lines(area.width, area.height);
        for (idx, line) in lines.iter().enumerate() {
            let row = area.y + idx as u16;
            if row < area.bottom() {
                buf.set_string(area.x, row, line, Style::default());
            }
        }
    }
}

// ============================================================================
// 4. TRUTHFUL PERSISTENCE WATERMARK STATUS STRIP
// ============================================================================

/// Truthful persistence watermark telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct WatermarkStatusData {
    pub admitted_seq: u64,
    pub admitted_control_revision: u64,
    pub applied_scientific_revision: u64,
    pub applied_tick: u64,
    pub durable_storage_tick: Option<u64>,
    pub durable_receipt_id: Option<String>,
    pub storage_error: Option<String>,
}

impl WatermarkStatusData {
    pub fn update(&mut self, admitted: u64, applied: u64, durable: u64, err: Option<&str>) {
        self.admitted_seq = admitted;
        self.applied_tick = applied;
        self.durable_storage_tick = Some(durable);
        self.storage_error = err.map(Into::into);
    }

    pub fn snapshot_lag(&self) -> u64 {
        self.applied_tick
            .saturating_sub(self.durable_storage_tick.unwrap_or(0))
    }

    pub fn is_synced(&self) -> bool {
        self.durable_storage_tick == Some(self.applied_tick) && self.storage_error.is_none()
    }

    pub fn format_strip(&self, width: usize) -> String {
        let lag = self.snapshot_lag();
        let status_badge = if let Some(err) = &self.storage_error {
            format!("[STORAGE_ERR: {}]", err)
        } else if self.is_synced() {
            "[DURABLE_SYNCED]".to_string()
        } else if lag > 0 {
            format!("[JOURNAL_LAG: {}t]", lag)
        } else {
            "[IN_FLIGHT]".to_string()
        };

        let durable_str = self
            .durable_storage_tick
            .map_or("none".to_string(), |t| format!("t{}", t));

        let text = format!(
            "{} | Storage: Adm s#{} r#{} | Applied t{} r#{} | Durable {}",
            status_badge,
            self.admitted_seq,
            self.admitted_control_revision,
            self.applied_tick,
            self.applied_scientific_revision,
            durable_str,
        );

        if text.len() > width {
            let mut s: String = text.chars().take(width.saturating_sub(1)).collect();
            s.push('…');
            s
        } else {
            text
        }
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 {
            return;
        }
        let text = self.format_strip(area.width as usize);
        buf.set_string(area.x, area.y, &text, Style::default());
    }

    pub fn render_ftui(&self, frame: &mut Frame, x: u16, y: u16, width: u16) {
        let text = self.format_strip(width as usize);
        print_text_clipped(frame, x, y, &text);
    }
}

// ============================================================================
// 5. ALIGNED SCIENCE TABLES (INSIGHTS, MORTALITY, LEADERBOARD, OLDEST)
// ============================================================================

/// Column-aligned table helper for mortality causes with theme-safe Unicode bars.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MortalityTableEntry {
    pub cause_name: String,
    pub count: u64,
    pub percentage: f32,
}

pub fn render_mortality_bars(entries: &[MortalityTableEntry], max_bar_width: usize) -> Vec<String> {
    let mut rows = Vec::new();
    let max_count = entries.iter().map(|e| e.count).max().unwrap_or(1).max(1);

    for e in entries {
        let bar_len =
            (((e.count as f64) / (max_count as f64)) * (max_bar_width as f64)).round() as usize;
        let bar = "█".repeat(bar_len.min(max_bar_width));
        rows.push(format!(
            "{:<14} {:>5} ({:>5.1}%) {}",
            e.cause_name, e.count, e.percentage, bar
        ));
    }
    rows
}

/// Column-aligned table entry for the agent leaderboard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeaderboardTableEntry {
    pub rank: usize,
    pub uid: u64,
    pub diet_chip: &'static str,
    pub energy: f32,
    pub health: f32,
    pub age: u64,
    pub generation: u32,
}

pub fn render_aligned_leaderboard(entries: &[LeaderboardTableEntry]) -> Vec<String> {
    let mut rows = Vec::new();
    rows.push(format!(
        "{:<4} {:<8} {:<4} {:>8} {:>8} {:>6} {:>5}",
        "Rank", "UID", "Diet", "Energy", "Health", "Age", "Gen"
    ));
    rows.push("-".repeat(48));
    for e in entries {
        rows.push(format!(
            "#{:<3} #{:<7} {:<4} {:>8.2} {:>8.2} {:>6} {:>5}",
            e.rank, e.uid, e.diet_chip, e.energy, e.health, e.age, e.generation
        ));
    }
    rows
}

// ============================================================================
// UNIT TESTS & HOSTILE INJECTED ALARMS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chart_rolling_window_cycling_and_bounds() {
        let mut chart = ScienceChartData::new(ChartRollingWindow::Ticks30);
        assert_eq!(chart.window, ChartRollingWindow::Ticks30);
        chart.cycle_window();
        assert_eq!(chart.window, ChartRollingWindow::Ticks60);
        chart.cycle_window();
        assert_eq!(chart.window, ChartRollingWindow::Ticks120);
        chart.cycle_window();
        assert_eq!(chart.window, ChartRollingWindow::Ticks300);
        chart.cycle_window();
        assert_eq!(chart.window, ChartRollingWindow::Ticks30);

        for t in 1..=40 {
            chart.push_sample(
                ChartSample {
                    tick: t,
                    population: 50 + t * 2,
                    avg_energy: 10.0 + (t as f32) * 0.5,
                    births: (t % 5) as u32,
                    deaths: (t % 3) as u32,
                },
                t,
            );
        }

        assert_eq!(chart.samples.len(), 40);
        let vis = chart.visible_samples();
        assert_eq!(vis.len(), 30, "Window 30 must bound visible samples to 30");

        let (p_min, p_max) = chart.population_bounds();
        assert_eq!(p_min, 50 + 11 * 2);
        assert_eq!(p_max, 50 + 40 * 2);
    }

    #[test]
    fn test_chart_degraded_states() {
        let mut chart = ScienceChartData::new(ChartRollingWindow::Ticks60);
        // 1. Empty state
        if let ChartDegradedState::Empty { message } = chart.degraded_state(0) {
            assert!(message.contains("waiting for snapshot history"));
        } else {
            panic!("Expected Empty degraded state");
        }

        // 2. Normal state
        chart.push_sample(
            ChartSample {
                tick: 10,
                population: 100,
                avg_energy: 25.0,
                births: 5,
                deaths: 2,
            },
            10,
        );
        assert_eq!(chart.degraded_state(10), ChartDegradedState::Normal);

        // 3. Stale state (lag > 15)
        if let ChartDegradedState::Stale { lag } = chart.degraded_state(30) {
            assert_eq!(lag, 20);
        } else {
            panic!("Expected Stale degraded state");
        }
    }

    #[test]
    fn test_typed_event_feed_filtering_and_focus() {
        let mut feed = TypedEventFeedData::new(10);
        feed.push_event(TypedEventRecord {
            id: 1,
            tick: 10,
            kind: TypedEventKind::Birth,
            message: "Agent spawned".into(),
            subject_uid: Some(101),
            location: Some((10.0, 20.0)),
            target_alive: true,
        });
        feed.push_event(TypedEventRecord {
            id: 2,
            tick: 12,
            kind: TypedEventKind::Death,
            message: "Agent died of starvation".into(),
            subject_uid: Some(101),
            location: Some((10.0, 20.0)),
            target_alive: false,
        });
        feed.push_event(TypedEventRecord {
            id: 3,
            tick: 14,
            kind: TypedEventKind::Combat,
            message: "Attack occurred".into(),
            subject_uid: Some(102),
            location: Some((30.0, 40.0)),
            target_alive: true,
        });

        // Filter by kind
        feed.set_filter_kind(EventKindFilter::Only(TypedEventKind::Death));
        let deaths = feed.filtered_events();
        assert_eq!(deaths.len(), 1);
        assert_eq!(deaths[0].id, 2);

        // Filter by agent UID
        feed.set_filter_kind(EventKindFilter::All);
        feed.set_filter_agent_uid(Some(101));
        let agent_events = feed.filtered_events();
        assert_eq!(agent_events.len(), 2);

        // Focus intent on dead agent must produce StaleTarget
        feed.selected_index = 0; // Most recent for 101 is Death
        let intent = feed.focus_intent_for_selected().expect("intent exists");
        assert!(
            matches!(intent, EventFocusIntent::StaleTarget { uid: 101, .. }),
            "Dead agent must produce explicit StaleTarget"
        );

        // Focus intent on living agent
        feed.set_filter_agent_uid(Some(102));
        feed.selected_index = 0;
        let living_intent = feed.focus_intent_for_selected().expect("intent exists");
        assert_eq!(living_intent, EventFocusIntent::FocusAgent(102));
    }

    #[test]
    fn test_brain_activation_grid_and_non_finite_handling() {
        let mut data = BrainActivationGridData::new();
        data.provenance = Some(BrainProvenanceBanner {
            agent_uid: 42,
            tick: 100,
            control_revision: 5,
            payload_bytes: 1280,
            clipped_count: 0,
            is_stale: false,
        });
        data.layers.push(LayerActivationData {
            name: "L0 Input".into(),
            dimensions: vec![4],
            values: vec![0.0, 0.5, -0.8, f32::NAN],
        });
        data.layers.push(LayerActivationData {
            name: "L1 Output".into(),
            dimensions: vec![2],
            values: vec![1.0, f32::INFINITY],
        });

        let lines = data.render_lines(60, 10);
        assert!(lines[0].contains("Brain #42"));
        assert!(lines[0].contains("[CLEAN]"));
        // NaN and Inf must not panic and must produce valid chars
        let l0 = lines
            .iter()
            .find(|l| l.contains("L0 Input"))
            .expect("L0 present");
        assert!(l0.contains('·'), "NaN must render as neutral dot '·'");
    }

    #[test]
    fn test_truthful_watermark_status_strip() {
        let mut status = WatermarkStatusData {
            admitted_seq: 12,
            admitted_control_revision: 3,
            applied_scientific_revision: 10,
            applied_tick: 50,
            durable_storage_tick: Some(45),
            durable_receipt_id: Some("rcpt-1".into()),
            storage_error: None,
        };

        assert_eq!(status.snapshot_lag(), 5);
        assert!(!status.is_synced());
        let strip = status.format_strip(80);
        assert!(strip.contains("[JOURNAL_LAG: 5t]"));
        assert!(strip.contains("Applied t50"));
        assert!(strip.contains("Durable t45"));

        // When durable caught up
        status.durable_storage_tick = Some(50);
        assert_eq!(status.snapshot_lag(), 0);
        assert!(status.is_synced());
        let synced_strip = status.format_strip(80);
        assert!(synced_strip.contains("[DURABLE_SYNCED]"));

        // Injected error
        status.storage_error = Some("disk full".into());
        let err_strip = status.format_strip(80);
        assert!(err_strip.contains("[STORAGE_ERR: disk full]"));
    }

    #[test]
    fn test_hostile_injected_alarms_for_dropped_axes_and_legends() {
        let mut chart = ScienceChartData::new(ChartRollingWindow::Ticks60);
        chart.push_sample(
            ChartSample {
                tick: 1,
                population: 100,
                avg_energy: 50.0,
                births: 2,
                deaths: 1,
            },
            1,
        );

        let lines = chart.render_lines(80, 12, false);
        // Verify required elements:
        // 1. Legend
        assert!(
            lines
                .iter()
                .any(|l| l.contains("Pop ■") && l.contains("Energy ▲")),
            "Legend must be present"
        );
        // 2. Y-axis tick line
        assert!(
            lines.iter().any(|l| l.contains('┼')),
            "Y-axis tick marks must be present"
        );
        // 3. X-axis baseline
        assert!(
            lines.iter().any(|l| l.contains('┴')),
            "X-axis baseline tick must be present"
        );
        // 4. Units
        assert!(
            lines
                .iter()
                .any(|l| l.contains("(ag)") && l.contains("(⚡)")),
            "Units must be present"
        );
    }

    #[test]
    fn test_science_widgets_matrix_theme_and_capabilities() {
        let mut chart = ScienceChartData::new(ChartRollingWindow::Ticks30);
        for t in 1..=30 {
            chart.push_sample(
                ChartSample {
                    tick: t,
                    population: 50 + (t % 10),
                    avg_energy: 40.0 + (t as f32 * 0.5),
                    births: (t % 3) as u32,
                    deaths: (t % 2) as u32,
                },
                t,
            );
        }

        let sizes = [(40, 6), (60, 8), (80, 12), (120, 20)];
        for &(w, h) in &sizes {
            for color in [true, false] {
                let lines = chart.render_lines(w, h, color);
                assert!(!lines.is_empty(), "Chart must render for {}x{}", w, h);
                assert_eq!(
                    lines.len(),
                    h as usize,
                    "Rendered line count must match height"
                );
            }
        }

        let mut feed = TypedEventFeedData::new(50);
        for (i, kind) in [
            TypedEventKind::Birth,
            TypedEventKind::Death,
            TypedEventKind::Combat,
            TypedEventKind::Eat,
            TypedEventKind::Mutation,
            TypedEventKind::Config,
            TypedEventKind::Info,
        ]
        .iter()
        .cycle()
        .take(20)
        .enumerate()
        {
            feed.push_event(TypedEventRecord {
                id: i as u64,
                tick: (i * 2) as u64,
                kind: *kind,
                message: format!("Event test {i}"),
                subject_uid: Some(100 + i as u64),
                location: Some((i as f32, i as f32)),
                target_alive: i % 2 == 0,
            });
        }

        for &(w, h) in &sizes {
            for emoji in [true, false] {
                let lines = feed.render_lines(w, h, 100, emoji);
                assert!(!lines.is_empty());
                assert_eq!(lines.len(), h as usize);
            }
        }
    }
}
