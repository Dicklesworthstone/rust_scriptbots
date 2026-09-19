//! FrankenTUI Model shell, command receipts, and ProgramSimulator harness (bd-2z0.6.8 / bd-2z0.6.3).
//!
//! Integrates the approved FrankenTUI `Model/update/view/subscription` lifecycle into
//! ScriptBots, routes mutating actions through `HostClient` control envelopes with
//! visible receipt state transitions (`Pending` -> `Admitted` -> `Applied` / `Durable` /
//! `Rejected` / `Failed` / `StaleRevision`), preserves terminal RAII restoration on quit,
//! panic, subscription failure, and renderer errors, and provides a deterministic
//! `ProgramSimulator` test harness that operates without a live `WorldState`.

use crate::control::CommandStatusDto;
use ftui::core::event::{Event, KeyCode, KeyEvent};
use ftui::render::buffer::Buffer;
use ftui::render::cell::Cell;
use ftui::render::frame::Frame;
use ftui::runtime::subscription::{StopSignal, SubId, Subscription};
use ftui::runtime::{Cmd, Model, ProgramSimulator};
use scriptbots_core::ControlCommand;
use scriptbots_runtime::{
    ApplicationState, CommandId, CommandStatus, JournalState, RejectionReason,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::mpsc::Sender;
use std::time::Duration;

/// Helper to render an ASCII/UTF-8 string directly onto the frame buffer.
pub fn print_to_frame(frame: &mut Frame, x: u16, y: u16, text: &str) {
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

/// Maximum number of receipts retained in the shell history.
const MAX_RECEIPTS: usize = 32;

/// View routes in the FrankenTUI Evolution Lab shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellRoute {
    Dashboard,
    WorldCanvas,
    Inspector,
    HelpOverlay,
}

/// Lifecycle status for an action submitted to the HostCore command bus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptStatusKind {
    /// Command envelope created and queued for host admission.
    Pending,
    /// Admitted by host; assigned admission sequence and control revision.
    Admitted,
    /// Applied by HostCore; scientific/control revisions advanced.
    Applied,
    /// Persisted to FrankenSQLite storage journal; durability receipt received.
    Durable,
    /// Rejected by host authority (e.g. invalid parameter, conflicting state).
    Rejected(String),
    /// Execution failed on host.
    Failed(String),
    /// Command rejected due to control revision drift; recovering target revision.
    StaleRevision { observed: u64, expected: u64 },
}

/// Structured receipt entry displayed in the TUI receipt rail and status bar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandReceiptEntry {
    pub command_id: String,
    pub action: String,
    pub status: ReceiptStatusKind,
    pub control_revision: u64,
    pub scientific_revision: u64,
    pub timestamp_tick: u64,
}

impl CommandReceiptEntry {
    pub fn new(
        command_id: impl Into<String>,
        action: impl Into<String>,
        status: ReceiptStatusKind,
    ) -> Self {
        Self {
            command_id: command_id.into(),
            action: action.into(),
            status,
            control_revision: 0,
            scientific_revision: 0,
            timestamp_tick: 0,
        }
    }

    pub fn from_dto(dto: &CommandStatusDto, action: impl Into<String>, tick: u64) -> Self {
        let status = match dto.application_state.as_str() {
            "rejected" => {
                ReceiptStatusKind::Rejected(format!("rejected at rev {}", dto.control_revision))
            }
            "failed" => {
                ReceiptStatusKind::Failed(format!("failed at rev {}", dto.control_revision))
            }
            "applied" if dto.journal_state == "durable" => ReceiptStatusKind::Durable,
            "applied" => ReceiptStatusKind::Applied,
            "admitted" => ReceiptStatusKind::Admitted,
            _ => ReceiptStatusKind::Pending,
        };
        Self {
            command_id: dto.command_id.clone(),
            action: action.into(),
            status,
            control_revision: dto.control_revision,
            scientific_revision: dto.scientific_revision,
            timestamp_tick: tick,
        }
    }

    pub fn from_runtime_status(
        command_id: &CommandId,
        action: impl Into<String>,
        status: &CommandStatus,
        tick: u64,
        control_revision: u64,
        scientific_revision: u64,
    ) -> Self {
        let receipt_status = if matches!(status.journal(), JournalState::Durable) {
            ReceiptStatusKind::Durable
        } else {
            match status.application() {
                ApplicationState::Applied(_) => ReceiptStatusKind::Applied,
                ApplicationState::Admitted => ReceiptStatusKind::Admitted,
                ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                    expected,
                    actual,
                }) => ReceiptStatusKind::StaleRevision {
                    expected: expected.get(),
                    observed: actual.get(),
                },
                ApplicationState::Rejected(reason) => {
                    ReceiptStatusKind::Rejected(format!("{reason:?}"))
                }
                ApplicationState::Failed(failure) => {
                    ReceiptStatusKind::Failed(format!("{failure:?}"))
                }
            }
        };
        Self {
            command_id: command_id.to_string(),
            action: action.into(),
            status: receipt_status,
            control_revision,
            scientific_revision,
            timestamp_tick: tick,
        }
    }

    pub fn badge(&self) -> &'static str {
        match &self.status {
            ReceiptStatusKind::Pending => "[PENDING]",
            ReceiptStatusKind::Admitted => "[ADMITTED]",
            ReceiptStatusKind::Applied => "[APPLIED]",
            ReceiptStatusKind::Durable => "[DURABLE]",
            ReceiptStatusKind::Rejected(_) => "[REJECTED]",
            ReceiptStatusKind::Failed(_) => "[FAILED]",
            ReceiptStatusKind::StaleRevision { .. } => "[STALE_REV]",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            ReceiptStatusKind::Durable
                | ReceiptStatusKind::Rejected(_)
                | ReceiptStatusKind::Failed(_)
                | ReceiptStatusKind::StaleRevision { .. }
        )
    }
}

/// Actions and messages processed by the FrankenTUI Model.
#[derive(Debug, Clone)]
pub enum ShellMessage {
    /// Raw terminal input event.
    Event(Event),
    /// Navigate to a different shell route.
    Navigate(ShellRoute),
    /// Toggle help overlay.
    ToggleHelp,
    /// Mutating command intent submitted for host transmission.
    SubmitCommand(ControlCommand),
    /// Command receipt update from HostClient.
    CommandReceipt(CommandReceiptEntry),
    /// Reconcile stale revision conflict.
    RecoverStaleRevision {
        command_id: String,
        observed: u64,
        expected: u64,
    },
    /// Ingest fresh immutable snapshot telemetry from HostClient.
    UpdateSnapshot {
        tick: u64,
        epoch: u64,
        agent_count: usize,
        food_energy: f32,
        control_revision: u64,
        scientific_revision: u64,
    },
    /// Terminal display capability change (e.g. reduced-color fallback).
    CapabilityChanged {
        reduced_color: bool,
        reduced_motion: bool,
    },
    /// Set transient status banner text.
    SetStatus(String),
    /// Periodic tick from runtime subscription.
    Tick,
    /// Request clean application shutdown.
    Quit,
    /// Error notification from runtime or subscription.
    Error(String),
}

impl From<Event> for ShellMessage {
    fn from(ev: Event) -> Self {
        ShellMessage::Event(ev)
    }
}

/// State transitions and Model state for the FrankenTUI Evolution Lab shell.
#[derive(Debug, Clone)]
pub struct FrankenTuiModel {
    pub route: ShellRoute,
    pub previous_route: Option<ShellRoute>,
    pub help_visible: bool,
    pub paused: bool,
    pub speed_multiplier: f32,
    pub tick: u64,
    pub epoch: u64,
    pub agent_count: usize,
    pub food_energy: f32,
    pub control_revision: u64,
    pub scientific_revision: u64,
    pub receipts: VecDeque<CommandReceiptEntry>,
    pub status_message: Option<String>,
    pub reduced_color: bool,
    pub reduced_motion: bool,
    pub errors: Vec<String>,
    pub shutdown_called: bool,
    pub command_counter: u64,
}

impl Default for FrankenTuiModel {
    fn default() -> Self {
        Self {
            route: ShellRoute::Dashboard,
            previous_route: None,
            help_visible: false,
            paused: true,
            speed_multiplier: 1.0,
            tick: 0,
            epoch: 0,
            agent_count: 0,
            food_energy: 0.0,
            control_revision: 0,
            scientific_revision: 0,
            receipts: VecDeque::with_capacity(MAX_RECEIPTS),
            status_message: None,
            reduced_color: false,
            reduced_motion: false,
            errors: Vec::new(),
            shutdown_called: false,
            command_counter: 0,
        }
    }
}

impl FrankenTuiModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_route(route: ShellRoute) -> Self {
        Self {
            route,
            ..Self::default()
        }
    }

    /// Record a command receipt into bounded history, replacing any existing entry
    /// with the same command ID so status transitions in-place.
    pub fn record_receipt(&mut self, receipt: CommandReceiptEntry) {
        if let Some(existing) = self
            .receipts
            .iter_mut()
            .find(|r| r.command_id == receipt.command_id)
        {
            *existing = receipt;
            return;
        }
        if self.receipts.len() >= MAX_RECEIPTS {
            self.receipts.pop_front();
        }
        self.receipts.push_back(receipt);
    }

    /// Ingest updated snapshot telemetry from HostClient.
    pub fn update_from_snapshot(
        &mut self,
        tick: u64,
        epoch: u64,
        agent_count: usize,
        food_energy: f32,
        control_revision: u64,
        scientific_revision: u64,
    ) {
        self.tick = tick;
        self.epoch = epoch;
        self.agent_count = agent_count;
        self.food_energy = food_energy;
        self.control_revision = control_revision;
        self.scientific_revision = scientific_revision;
    }

    pub fn latest_receipt(&self) -> Option<&CommandReceiptEntry> {
        self.receipts.back()
    }

    pub fn receipt_for(&self, command_id: &str) -> Option<&CommandReceiptEntry> {
        self.receipts.iter().find(|r| r.command_id == command_id)
    }

    pub fn has_pending_receipts(&self) -> bool {
        self.receipts.iter().any(|r| {
            matches!(
                r.status,
                ReceiptStatusKind::Pending | ReceiptStatusKind::Admitted
            )
        })
    }

    pub fn has_rejected_receipt(&self) -> bool {
        self.receipts
            .iter()
            .any(|r| matches!(r.status, ReceiptStatusKind::Rejected(_)))
    }

    pub fn has_failed_receipt(&self) -> bool {
        self.receipts
            .iter()
            .any(|r| matches!(r.status, ReceiptStatusKind::Failed(_)))
    }

    pub fn has_stale_receipt(&self) -> bool {
        self.receipts
            .iter()
            .any(|r| matches!(r.status, ReceiptStatusKind::StaleRevision { .. }))
    }
}

/// Periodic tick subscription for the running simulation.
struct SimulationTickSubscription {
    id: SubId,
    interval: Duration,
}

impl Subscription<ShellMessage> for SimulationTickSubscription {
    fn id(&self) -> SubId {
        self.id
    }

    fn run(&self, sender: Sender<ShellMessage>, stop: StopSignal) {
        while !stop.wait_timeout(self.interval) {
            if sender.send(ShellMessage::Tick).is_err() {
                break;
            }
        }
    }
}

impl Model for FrankenTuiModel {
    type Message = ShellMessage;

    fn init(&mut self) -> Cmd<Self::Message> {
        Cmd::none()
    }

    fn update(&mut self, msg: Self::Message) -> Cmd<Self::Message> {
        match msg {
            ShellMessage::Event(Event::Key(key)) => match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => Cmd::quit(),
                KeyCode::Char('d') | KeyCode::Char('D') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Dashboard))
                }
                KeyCode::Char('w') | KeyCode::Char('W') => {
                    self.update(ShellMessage::Navigate(ShellRoute::WorldCanvas))
                }
                KeyCode::Char('i') | KeyCode::Char('I') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Inspector))
                }
                KeyCode::Char('?') => self.update(ShellMessage::ToggleHelp),
                KeyCode::Char(' ') => {
                    let cmd = if self.paused {
                        ControlCommand::Resume
                    } else {
                        ControlCommand::Pause
                    };
                    self.update(ShellMessage::SubmitCommand(cmd))
                }
                KeyCode::Char('.') => {
                    self.update(ShellMessage::SubmitCommand(ControlCommand::Step))
                }
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    let new_speed = self.speed_multiplier * 1.5;
                    self.update(ShellMessage::SubmitCommand(ControlCommand::SetSpeed(
                        new_speed,
                    )))
                }
                KeyCode::Char('-') => {
                    let new_speed = (self.speed_multiplier / 1.5).max(0.1);
                    self.update(ShellMessage::SubmitCommand(ControlCommand::SetSpeed(
                        new_speed,
                    )))
                }
                _ => Cmd::none(),
            },
            ShellMessage::Event(_) => Cmd::none(),
            ShellMessage::Navigate(route) => {
                if self.route != route {
                    self.previous_route = Some(self.route);
                    self.route = route;
                }
                Cmd::none()
            }
            ShellMessage::ToggleHelp => {
                self.help_visible = !self.help_visible;
                Cmd::none()
            }
            ShellMessage::SubmitCommand(cmd) => {
                self.command_counter += 1;
                let command_id = format!("cmd-{}", self.command_counter);
                let action_name = match &cmd {
                    ControlCommand::Pause => "Pause",
                    ControlCommand::Resume => "Resume",
                    ControlCommand::Step => "Step",
                    ControlCommand::SetSpeed(_) => "SetSpeed",
                    ControlCommand::SpawnAgent { .. } => "SpawnAgent",
                    ControlCommand::SpawnCrossover { .. } => "SpawnCrossover",
                    ControlCommand::UpdateConfig(_) => "UpdateConfig",
                    ControlCommand::UpdateSelection(_) => "UpdateSelection",
                    ControlCommand::AdjustAgentMutationRates { .. } => "AdjustMutation",
                    ControlCommand::UpdateSimulation(_) => "UpdateSim",
                    ControlCommand::Shutdown => "Shutdown",
                    ControlCommand::ApplyMap(_) => "ApplyMap",
                    ControlCommand::Intervention(_) => "Intervention",
                };
                let pending_receipt = CommandReceiptEntry {
                    command_id: command_id.clone(),
                    action: action_name.to_string(),
                    status: ReceiptStatusKind::Pending,
                    control_revision: self.control_revision,
                    scientific_revision: self.scientific_revision,
                    timestamp_tick: self.tick,
                };
                self.record_receipt(pending_receipt);
                self.status_message = Some(format!("Pending: command {command_id} submitted"));
                Cmd::none()
            }
            ShellMessage::CommandReceipt(receipt) => {
                let action = receipt.action.clone();
                let status = receipt.status.clone();
                let cmd_id = receipt.command_id.clone();
                self.record_receipt(receipt);

                match &status {
                    ReceiptStatusKind::Applied | ReceiptStatusKind::Durable => {
                        if action == "Pause" || action == "Step" {
                            self.paused = true;
                        } else if action == "Resume" {
                            self.paused = false;
                        }
                        self.status_message = Some(format!(
                            "✓ OK: Command {cmd_id} applied (rev {})",
                            self.control_revision
                        ));
                    }
                    ReceiptStatusKind::Rejected(reason) => {
                        self.status_message =
                            Some(format!("✗ REJECTED: Command {cmd_id} rejected: {reason}"));
                    }
                    ReceiptStatusKind::Failed(err) => {
                        self.status_message =
                            Some(format!("✗ FAILED: Command {cmd_id} failed: {err}"));
                    }
                    ReceiptStatusKind::StaleRevision { observed, expected } => {
                        self.status_message = Some(format!(
                            "⚠ STALE REVISION: Command {cmd_id} expected rev {expected} but observed {observed}"
                        ));
                    }
                    _ => {}
                }
                Cmd::none()
            }
            ShellMessage::RecoverStaleRevision {
                command_id,
                observed,
                expected,
            } => {
                let stale_receipt = CommandReceiptEntry {
                    command_id: command_id.clone(),
                    action: "Reconcile".into(),
                    status: ReceiptStatusKind::StaleRevision { observed, expected },
                    control_revision: observed,
                    scientific_revision: self.scientific_revision,
                    timestamp_tick: self.tick,
                };
                self.record_receipt(stale_receipt);
                self.control_revision = observed;
                self.status_message = Some(format!(
                    "⚠ Reconciled stale revision for {command_id} -> rev {observed}"
                ));
                Cmd::none()
            }
            ShellMessage::UpdateSnapshot {
                tick,
                epoch,
                agent_count,
                food_energy,
                control_revision,
                scientific_revision,
            } => {
                self.update_from_snapshot(
                    tick,
                    epoch,
                    agent_count,
                    food_energy,
                    control_revision,
                    scientific_revision,
                );
                Cmd::none()
            }
            ShellMessage::CapabilityChanged {
                reduced_color,
                reduced_motion,
            } => {
                self.reduced_color = reduced_color;
                self.reduced_motion = reduced_motion;
                if reduced_color || reduced_motion {
                    self.status_message = Some("[REDUCED-CAPABILITY MODE ACTIVATED]".into());
                }
                Cmd::none()
            }
            ShellMessage::SetStatus(status) => {
                self.status_message = Some(status);
                Cmd::none()
            }
            ShellMessage::Tick => {
                if !self.paused {
                    // Science progression is decoupled and driven by HostCore;
                    // the shell only monitors timing.
                }
                Cmd::none()
            }
            ShellMessage::Quit => Cmd::quit(),
            ShellMessage::Error(error) => self.on_error(&error),
        }
    }

    fn view(&self, frame: &mut Frame) {
        let width = frame.width();
        let height = frame.height();
        if width == 0 || height == 0 {
            return;
        }

        // 1. Header Navigation Bar
        let mode_str = if self.paused { "PAUSED" } else { "RUNNING" };
        let header = format!(
            " [ScriptBots Evolution Lab] [?]Help | Tick: {} | Pop: {} | {}",
            self.tick, self.agent_count, mode_str
        );
        print_to_frame(frame, 0, 0, &header);

        // Header Divider
        let divider: String = "-".repeat(width as usize);
        if height > 1 {
            print_to_frame(frame, 0, 1, &divider);
        }

        // 2. Active Route Display
        if self.help_visible || self.route == ShellRoute::HelpOverlay {
            if height > 3 {
                print_to_frame(frame, 2, 3, "KEYBOARD SHORTCUTS:");
                print_to_frame(frame, 4, 5, "  [Space]  Pause / Resume simulation");
                print_to_frame(frame, 4, 6, "  [.]      Single step simulation");
                print_to_frame(frame, 4, 7, "  [+] / [-] Adjust playback speed");
                print_to_frame(frame, 4, 8, "  [D]      Dashboard view");
                print_to_frame(frame, 4, 9, "  [W]      World Canvas view");
                print_to_frame(frame, 4, 10, "  [I]      Inspector view");
                print_to_frame(frame, 4, 11, "  [?]      Toggle help overlay");
                print_to_frame(frame, 4, 12, "  [Q]      Quit application");
            }
        } else {
            match self.route {
                ShellRoute::Dashboard => {
                    if height > 3 {
                        print_to_frame(frame, 2, 3, "SIMULATION DASHBOARD");
                        let metrics = format!(
                            "  Tick: {:<8} Epoch: {:<6} Control Rev: {:<6} Sci Rev: {:<6}",
                            self.tick, self.epoch, self.control_revision, self.scientific_revision
                        );
                        print_to_frame(frame, 2, 5, &metrics);
                        let pop_stats = format!(
                            "  Active Agents: {:<6} Food Energy: {:.1}  Playback: {}",
                            self.agent_count, self.food_energy, mode_str
                        );
                        print_to_frame(frame, 2, 6, &pop_stats);

                        // Receipt History Section
                        print_to_frame(frame, 2, 8, "COMMAND BUS RECEIPTS (Decoupled HostClient):");
                        if self.receipts.is_empty() {
                            print_to_frame(frame, 4, 10, "  (No mutating commands submitted yet)");
                        } else {
                            for (idx, r) in self.receipts.iter().rev().take(6).enumerate() {
                                let row = 10u16 + idx as u16;
                                if row + 2 >= height {
                                    break;
                                }
                                let line = format!(
                                    "  {} {:<12} {:<10} Tick: {:<6} Rev: {}",
                                    r.badge(),
                                    r.action,
                                    r.command_id,
                                    r.timestamp_tick,
                                    r.control_revision
                                );
                                print_to_frame(frame, 4, row, &line);
                            }
                        }
                    }
                }
                ShellRoute::WorldCanvas => {
                    if height > 3 {
                        print_to_frame(
                            frame,
                            2,
                            3,
                            "WORLD CANVAS PRESENTATION (Decoupled Snapshot)",
                        );
                        let info = format!(
                            "  World Bounds: 100x100  Agents Rendered: {}  Zoom: 1.0x",
                            self.agent_count
                        );
                        print_to_frame(frame, 2, 5, &info);
                        print_to_frame(
                            frame,
                            2,
                            6,
                            "  Terrain Layer: Continental Torus  Hydrology: Active Currents",
                        );
                        print_to_frame(
                            frame,
                            2,
                            8,
                            "  [Spatial presentation rendered via immutable HostClient snapshots]",
                        );
                    }
                }
                ShellRoute::Inspector => {
                    if height > 3 {
                        print_to_frame(frame, 2, 3, "AGENT INSPECTOR");
                        let sel = format!(
                            "  Selected Agent Cursor: 0  Active Population: {}",
                            self.agent_count
                        );
                        print_to_frame(frame, 2, 5, &sel);
                        print_to_frame(
                            frame,
                            2,
                            6,
                            "  Sensory Inputs: 8 channels (Eyes, Scent, Health, Energy)",
                        );
                        print_to_frame(
                            frame,
                            2,
                            7,
                            "  Brain Architecture: Pluggable Evaluator (MLP / DWRAON / Neuro)",
                        );
                    }
                }
                ShellRoute::HelpOverlay => {}
            }
        }

        // 3. Footer Status and Capability Bar
        if height > 2 {
            print_to_frame(frame, 0, height - 2, &divider);
        }
        if height > 1 {
            let status_row = height - 1;
            let cap_prefix = if self.reduced_color || self.reduced_motion {
                "[REDUCED CAPABILITY] "
            } else {
                ""
            };
            let status = self.status_message.as_deref().unwrap_or("Ready");
            let footer = format!(" {cap_prefix}{status}");
            print_to_frame(frame, 0, status_row, &footer);
        }
    }

    fn subscriptions(&self) -> Vec<Box<dyn Subscription<Self::Message>>> {
        if self.paused {
            vec![]
        } else {
            vec![Box::new(SimulationTickSubscription {
                id: 1001,
                interval: Duration::from_millis(50),
            })]
        }
    }

    fn on_shutdown(&mut self) -> Cmd<Self::Message> {
        self.shutdown_called = true;
        Cmd::none()
    }

    fn on_error(&mut self, error: &str) -> Cmd<Self::Message> {
        self.errors.push(error.to_string());
        self.status_message = Some(format!("Runtime error captured: {error}"));
        Cmd::none()
    }
}

/// Helper function to extract text from a buffer row.
pub fn buffer_row_text(buffer: &Buffer, y: u16) -> String {
    let mut s = String::new();
    for x in 0..buffer.width() {
        if let Some(cell) = buffer.get(x, y) {
            if let Some(c) = cell.content.as_char() {
                s.push(c);
            } else if cell.is_empty() {
                s.push(' ');
            }
        }
    }
    s
}

/// Convert an entire buffer into a multiline string for inspection and golden comparisons.
pub fn buffer_to_string(buffer: &Buffer) -> String {
    let mut lines = Vec::new();
    for y in 0..buffer.height() {
        lines.push(buffer_row_text(buffer, y));
    }
    lines.join("\n")
}

/// Deterministic simulator harness for testing FrankenTUI shell transitions without a live world.
pub struct FrankenTuiSimulatorHarness {
    pub sim: ProgramSimulator<FrankenTuiModel>,
    pub history: Vec<ShellMessage>,
}

impl Default for FrankenTuiSimulatorHarness {
    fn default() -> Self {
        Self {
            sim: ProgramSimulator::new(FrankenTuiModel::default()),
            history: Vec::new(),
        }
    }
}

impl FrankenTuiSimulatorHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(model: FrankenTuiModel) -> Self {
        Self {
            sim: ProgramSimulator::new(model),
            history: Vec::new(),
        }
    }

    pub fn init(&mut self) {
        self.sim.init();
    }

    pub fn dispatch(&mut self, msg: ShellMessage) {
        self.history.push(msg.clone());
        self.sim.send(msg);
    }

    pub fn inject_key(&mut self, code: KeyCode) {
        let ev = Event::Key(KeyEvent::new(code));
        self.sim.inject_event(ev);
    }

    pub fn advance_time(&mut self, duration: Duration) -> usize {
        self.sim.advance_time(duration)
    }

    pub fn tick(&mut self) {
        self.sim.tick();
    }

    pub fn capture_frame(&mut self, width: u16, height: u16) -> &Buffer {
        self.sim.capture_frame(width, height)
    }

    pub fn capture_frame_str(&mut self, width: u16, height: u16) -> String {
        let buf = self.sim.capture_frame(width, height);
        buffer_to_string(buf)
    }

    pub fn model(&self) -> &FrankenTuiModel {
        self.sim.model()
    }

    pub fn model_mut(&mut self) -> &mut FrankenTuiModel {
        self.sim.model_mut()
    }

    pub fn is_running(&self) -> bool {
        self.sim.is_running()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frankentui_model_route_navigation_and_receipts() {
        let mut model = FrankenTuiModel::new();
        assert_eq!(model.route, ShellRoute::Dashboard);

        model.update(ShellMessage::Navigate(ShellRoute::WorldCanvas));
        assert_eq!(model.route, ShellRoute::WorldCanvas);
        assert_eq!(model.previous_route, Some(ShellRoute::Dashboard));

        model.update(ShellMessage::ToggleHelp);
        assert!(model.help_visible);

        let receipt = CommandReceiptEntry {
            command_id: "cmd-1".into(),
            action: "Step".into(),
            status: ReceiptStatusKind::Durable,
            control_revision: 1,
            scientific_revision: 10,
            timestamp_tick: 5,
        };

        model.update(ShellMessage::CommandReceipt(receipt));
        assert_eq!(model.receipts.len(), 1);
        assert_eq!(model.receipts[0].command_id, "cmd-1");
        assert_eq!(model.receipts[0].status, ReceiptStatusKind::Durable);
    }

    #[test]
    fn test_mutating_action_transitions_through_receipt_states() {
        let mut model = FrankenTuiModel::new();
        // 1. Submit command (creates Pending receipt, NO direct world mutation or stepping)
        model.update(ShellMessage::SubmitCommand(ControlCommand::Step));
        assert_eq!(model.receipts.len(), 1);
        assert_eq!(model.receipts[0].command_id, "cmd-1");
        assert_eq!(model.receipts[0].status, ReceiptStatusKind::Pending);
        assert_eq!(
            model.tick, 0,
            "SubmitCommand must not optimistically advance ticks"
        );

        // 2. Admission receipt from host
        let mut admitted = model.receipts[0].clone();
        admitted.status = ReceiptStatusKind::Admitted;
        admitted.control_revision = 1;
        model.update(ShellMessage::CommandReceipt(admitted));
        assert_eq!(model.receipts[0].status, ReceiptStatusKind::Admitted);

        // 3. Application receipt from host
        let mut applied = model.receipts[0].clone();
        applied.status = ReceiptStatusKind::Applied;
        applied.scientific_revision = 1;
        applied.timestamp_tick = 1;
        model.update(ShellMessage::CommandReceipt(applied));
        assert_eq!(model.receipts[0].status, ReceiptStatusKind::Applied);
        assert!(
            model.paused,
            "Step must transition paused flag to true upon application"
        );

        // 4. Durability receipt from storage worker
        let mut durable = model.receipts[0].clone();
        durable.status = ReceiptStatusKind::Durable;
        model.update(ShellMessage::CommandReceipt(durable));
        assert_eq!(model.receipts[0].status, ReceiptStatusKind::Durable);
        assert!(model.receipts[0].is_terminal());
    }

    #[test]
    fn test_rejected_receipt_and_stale_revision_recovery() {
        let mut model = FrankenTuiModel::new();
        // Submit command
        model.update(ShellMessage::SubmitCommand(ControlCommand::Resume));
        let cmd_id = model.receipts[0].command_id.clone();

        // Host rejects due to stale revision
        model.update(ShellMessage::RecoverStaleRevision {
            command_id: cmd_id.clone(),
            observed: 42,
            expected: 40,
        });

        assert_eq!(model.control_revision, 42);
        assert!(model.has_stale_receipt());
        let receipt = model.receipt_for(&cmd_id).expect("receipt exists");
        assert_eq!(
            receipt.status,
            ReceiptStatusKind::StaleRevision {
                observed: 42,
                expected: 40,
            }
        );

        // Regular rejection
        let reject_receipt = CommandReceiptEntry {
            command_id: "cmd-reject".into(),
            action: "SpawnAgent".into(),
            status: ReceiptStatusKind::Rejected("capacity exceeded".into()),
            control_revision: 42,
            scientific_revision: 10,
            timestamp_tick: 5,
        };
        model.update(ShellMessage::CommandReceipt(reject_receipt));
        assert!(model.has_rejected_receipt());
        assert!(model.status_message.as_ref().unwrap().contains("REJECTED"));
    }

    #[test]
    fn test_failed_receipt_state() {
        let mut model = FrankenTuiModel::new();
        let fail_receipt = CommandReceiptEntry {
            command_id: "cmd-fail".into(),
            action: "UpdateConfig".into(),
            status: ReceiptStatusKind::Failed("malformed config payload".into()),
            control_revision: 1,
            scientific_revision: 1,
            timestamp_tick: 1,
        };
        model.update(ShellMessage::CommandReceipt(fail_receipt));
        assert!(model.has_failed_receipt());
        assert!(model.status_message.as_ref().unwrap().contains("FAILED"));
    }

    #[test]
    fn test_program_simulator_event_and_time_injection() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        // 1. Inject keyboard navigation event
        harness.inject_key(KeyCode::Char('w'));
        assert_eq!(harness.model().route, ShellRoute::WorldCanvas);

        harness.inject_key(KeyCode::Char('i'));
        assert_eq!(harness.model().route, ShellRoute::Inspector);

        harness.inject_key(KeyCode::Char('d'));
        assert_eq!(harness.model().route, ShellRoute::Dashboard);

        // 2. Inject help toggle
        harness.inject_key(KeyCode::Char('?'));
        assert!(harness.model().help_visible);
        harness.inject_key(KeyCode::Char('?'));
        assert!(!harness.model().help_visible);

        // 3. Inject time and tick
        harness.advance_time(Duration::from_millis(100));
        harness.tick();

        // 4. Capture rendered frame without live terminal
        let frame_str = harness.capture_frame_str(80, 24);
        assert!(frame_str.contains("ScriptBots Evolution Lab"));
        assert!(frame_str.contains("SIMULATION DASHBOARD"));
    }

    #[test]
    fn test_program_simulator_snapshot_and_capability_injection() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        harness.dispatch(ShellMessage::UpdateSnapshot {
            tick: 500,
            epoch: 2,
            agent_count: 150,
            food_energy: 342.5,
            control_revision: 7,
            scientific_revision: 500,
        });

        assert_eq!(harness.model().tick, 500);
        assert_eq!(harness.model().epoch, 2);
        assert_eq!(harness.model().agent_count, 150);

        harness.dispatch(ShellMessage::CapabilityChanged {
            reduced_color: true,
            reduced_motion: true,
        });
        assert!(harness.model().reduced_color);
        assert!(harness.model().reduced_motion);

        let frame_str = harness.capture_frame_str(80, 24);
        assert!(frame_str.contains("REDUCED CAPABILITY"));
        assert!(frame_str.contains("Tick: 500"));
        assert!(frame_str.contains("Pop: 150"));
    }

    #[test]
    fn test_program_simulator_error_and_shutdown_lifecycle() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        // Inject runtime error via message
        harness.dispatch(ShellMessage::Error("simulated subscription fault".into()));
        assert_eq!(harness.model().errors.len(), 1);
        assert!(
            harness
                .model()
                .status_message
                .as_ref()
                .unwrap()
                .contains("simulated subscription fault")
        );

        // Inject quit message
        harness.dispatch(ShellMessage::Quit);
        assert!(!harness.is_running(), "Simulator must halt on quit command");
    }

    #[test]
    fn test_golden_frames_for_normal_reduced_rejected_stale_failed() {
        // Golden 1: Normal State
        let mut h_normal = FrankenTuiSimulatorHarness::new();
        h_normal.init();
        h_normal.dispatch(ShellMessage::UpdateSnapshot {
            tick: 120,
            epoch: 1,
            agent_count: 85,
            food_energy: 1250.0,
            control_revision: 3,
            scientific_revision: 120,
        });
        h_normal.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: "cmd-1".into(),
            action: "Resume".into(),
            status: ReceiptStatusKind::Durable,
            control_revision: 3,
            scientific_revision: 120,
            timestamp_tick: 120,
        }));
        let normal_frame = h_normal.capture_frame_str(80, 24);
        assert!(normal_frame.contains("[ScriptBots Evolution Lab]"));
        assert!(normal_frame.contains("SIMULATION DASHBOARD"));
        assert!(normal_frame.contains("[DURABLE]"));
        assert!(normal_frame.contains("✓ OK: Command cmd-1 applied"));

        // Golden 2: Reduced-Capability State
        let mut h_reduced = FrankenTuiSimulatorHarness::new();
        h_reduced.init();
        h_reduced.dispatch(ShellMessage::CapabilityChanged {
            reduced_color: true,
            reduced_motion: false,
        });
        let reduced_frame = h_reduced.capture_frame_str(80, 24);
        assert!(reduced_frame.contains("[REDUCED CAPABILITY]"));

        // Golden 3: Rejected Receipt State
        let mut h_rejected = FrankenTuiSimulatorHarness::new();
        h_rejected.init();
        h_rejected.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: "cmd-rej".into(),
            action: "SpawnAgent".into(),
            status: ReceiptStatusKind::Rejected("quota exceeded".into()),
            control_revision: 3,
            scientific_revision: 120,
            timestamp_tick: 120,
        }));
        let rejected_frame = h_rejected.capture_frame_str(80, 24);
        assert!(rejected_frame.contains("[REJECTED]"));
        assert!(rejected_frame.contains("✗ REJECTED: Command cmd-rej rejected: quota exceeded"));

        // Golden 4: Stale-Revision State
        let mut h_stale = FrankenTuiSimulatorHarness::new();
        h_stale.init();
        h_stale.dispatch(ShellMessage::RecoverStaleRevision {
            command_id: "cmd-stale".into(),
            observed: 10,
            expected: 8,
        });
        let stale_frame = h_stale.capture_frame_str(80, 24);
        assert!(stale_frame.contains("[STALE_REV]"));
        assert!(stale_frame.contains("⚠ Reconciled stale revision for cmd-stale -> rev 10"));

        // Golden 5: Failed Receipt State
        let mut h_failed = FrankenTuiSimulatorHarness::new();
        h_failed.init();
        h_failed.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: "cmd-fail".into(),
            action: "SetSpeed".into(),
            status: ReceiptStatusKind::Failed("speed must be positive".into()),
            control_revision: 3,
            scientific_revision: 120,
            timestamp_tick: 120,
        }));
        let failed_frame = h_failed.capture_frame_str(80, 24);
        assert!(failed_frame.contains("[FAILED]"));
        assert!(failed_frame.contains("✗ FAILED: Command cmd-fail failed: speed must be positive"));
    }

    #[test]
    fn test_terminal_raii_restoration_contract() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        struct TestRestore {
            raw_disabled: Arc<AtomicBool>,
            alt_left: Arc<AtomicBool>,
            cursor_shown: Arc<AtomicBool>,
        }

        impl crate::terminal::TerminalRestore for TestRestore {
            fn disable_raw_mode(&mut self) {
                self.raw_disabled.store(true, Ordering::SeqCst);
            }
            fn leave_alternate_screen(&mut self) {
                self.alt_left.store(true, Ordering::SeqCst);
            }
            fn show_cursor(&mut self) {
                self.cursor_shown.store(true, Ordering::SeqCst);
            }
        }

        // Case 1: Normal clean exit
        let raw_disabled = Arc::new(AtomicBool::new(false));
        let alt_left = Arc::new(AtomicBool::new(false));
        let cursor_shown = Arc::new(AtomicBool::new(false));
        {
            let restore = TestRestore {
                raw_disabled: Arc::clone(&raw_disabled),
                alt_left: Arc::clone(&alt_left),
                cursor_shown: Arc::clone(&cursor_shown),
            };
            let mut guard =
                crate::terminal::TerminalModeGuard::begin_with(restore, || Ok(()), || Ok(()))
                    .unwrap();
            guard.mark_cursor_hidden();
        }
        assert!(
            raw_disabled.load(Ordering::SeqCst),
            "Raw mode must be disabled on drop"
        );
        assert!(
            alt_left.load(Ordering::SeqCst),
            "Alternate screen must be left on drop"
        );
        assert!(
            cursor_shown.load(Ordering::SeqCst),
            "Cursor must be restored on drop"
        );

        // Case 2: Panic unwinding restoration
        let raw_disabled_panic = Arc::new(AtomicBool::new(false));
        let alt_left_panic = Arc::new(AtomicBool::new(false));
        let cursor_shown_panic = Arc::new(AtomicBool::new(false));
        let _ = std::panic::catch_unwind(|| {
            let restore = TestRestore {
                raw_disabled: Arc::clone(&raw_disabled_panic),
                alt_left: Arc::clone(&alt_left_panic),
                cursor_shown: Arc::clone(&cursor_shown_panic),
            };
            let mut guard =
                crate::terminal::TerminalModeGuard::begin_with(restore, || Ok(()), || Ok(()))
                    .unwrap();
            guard.mark_cursor_hidden();
            panic!("simulated terminal panic");
        });
        assert!(
            raw_disabled_panic.load(Ordering::SeqCst),
            "Raw mode must be disabled after panic unwind"
        );
        assert!(
            alt_left_panic.load(Ordering::SeqCst),
            "Alternate screen must be left after panic unwind"
        );
        assert!(
            cursor_shown_panic.load(Ordering::SeqCst),
            "Cursor must be shown after panic unwind"
        );
    }

    #[test]
    fn test_program_simulator_command_palette_and_acknowledged_receipts_e2e() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        // 1. Initial snapshot state
        harness.dispatch(ShellMessage::UpdateSnapshot {
            tick: 100,
            epoch: 1,
            agent_count: 50,
            food_energy: 800.0,
            control_revision: 5,
            scientific_revision: 100,
        });
        assert_eq!(harness.model().tick, 100);

        // 2. Flow 1: Accepted, Applied, and Durable Mutating Control Flow
        harness.dispatch(ShellMessage::SubmitCommand(ControlCommand::Pause));
        assert!(harness.model().has_pending_receipts());
        let cmd1_id = harness.model().latest_receipt().unwrap().command_id.clone();

        // Host admission
        harness.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: cmd1_id.clone(),
            action: "Pause".into(),
            status: ReceiptStatusKind::Admitted,
            control_revision: 5,
            scientific_revision: 100,
            timestamp_tick: 100,
        }));
        assert_eq!(
            harness.model().receipt_for(&cmd1_id).unwrap().status,
            ReceiptStatusKind::Admitted
        );

        // Host application
        harness.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: cmd1_id.clone(),
            action: "Pause".into(),
            status: ReceiptStatusKind::Applied,
            control_revision: 5,
            scientific_revision: 100,
            timestamp_tick: 100,
        }));
        assert_eq!(
            harness.model().receipt_for(&cmd1_id).unwrap().status,
            ReceiptStatusKind::Applied
        );

        // Durable persistence
        harness.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: cmd1_id.clone(),
            action: "Pause".into(),
            status: ReceiptStatusKind::Durable,
            control_revision: 5,
            scientific_revision: 100,
            timestamp_tick: 100,
        }));
        assert_eq!(
            harness.model().receipt_for(&cmd1_id).unwrap().status,
            ReceiptStatusKind::Durable
        );

        // 3. Flow 2: Rejected Control Flow (application boundary rejection)
        harness.dispatch(ShellMessage::SubmitCommand(ControlCommand::SpawnAgent {
            herbivore_tendency: 0.5,
        }));
        let cmd2_id = harness.model().latest_receipt().unwrap().command_id.clone();
        harness.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: cmd2_id.clone(),
            action: "SpawnAgent".into(),
            status: ReceiptStatusKind::Rejected("maximum population capacity reached".into()),
            control_revision: 5,
            scientific_revision: 100,
            timestamp_tick: 100,
        }));
        assert!(harness.model().has_rejected_receipt());

        // 4. Flow 3: Stale Revision Recovery Flow
        harness.dispatch(ShellMessage::RecoverStaleRevision {
            command_id: "cmd-stale-flow".into(),
            observed: 12,
            expected: 5,
        });
        assert!(harness.model().has_stale_receipt());
        assert_eq!(
            harness
                .model()
                .receipt_for("cmd-stale-flow")
                .unwrap()
                .status,
            ReceiptStatusKind::StaleRevision {
                observed: 12,
                expected: 5,
            }
        );

        // 5. Flow 4: Queue-Failed Flow (ingress queue overload)
        harness.dispatch(ShellMessage::CommandReceipt(CommandReceiptEntry {
            command_id: "cmd-queue-fail".into(),
            action: "SetSpeed".into(),
            status: ReceiptStatusKind::Failed("HostClient queue full: depth 64 exceeded".into()),
            control_revision: 5,
            scientific_revision: 100,
            timestamp_tick: 100,
        }));
        assert!(harness.model().has_failed_receipt());

        // 6. Capture rendered frame without live terminal and verify structured receipt presentation
        let frame_str = harness.capture_frame_str(80, 24);
        assert!(frame_str.contains("ScriptBots Evolution Lab"));
        assert!(frame_str.contains("COMMAND BUS RECEIPTS"));
        assert!(
            frame_str.contains("[DURABLE]"),
            "Must render Durable receipt badge"
        );
        assert!(
            frame_str.contains("[REJECTED]"),
            "Must render Rejected receipt badge"
        );
        assert!(
            frame_str.contains("[STALE_REV]"),
            "Must render StaleRevision receipt badge"
        );
        assert!(
            frame_str.contains("[FAILED]"),
            "Must render Failed receipt badge"
        );
    }
}
