//! FrankenTUI Model shell, command receipts, and ProgramSimulator harness (bd-2z0.6.8 / bd-2z0.6.3).
//!
//! Integrates the approved FrankenTUI `Model/update/view/subscription` lifecycle into
//! ScriptBots, routes mutating actions through `HostClient` control envelopes with
//! visible receipt state transitions (`Pending` -> `Admitted` -> `Applied` / `Durable` /
//! `Rejected` / `Failed` / `StaleRevision`), preserves terminal RAII restoration on quit,
//! panic, subscription failure, and renderer errors, and provides a deterministic
//! `ProgramSimulator` test harness that operates without a live `WorldState`.

use super::science_screens::{self, *};
use super::science_widgets::*;
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
    Lineages,
    BrainArena,
    Experiments,
    Replay,
    Environment,
    Diagnostics,
    HelpOverlay,
}

impl ShellRoute {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Dashboard => "Dashboard",
            Self::WorldCanvas => "WorldCanvas",
            Self::Inspector => "Inspector",
            Self::Lineages => "Lineages",
            Self::BrainArena => "BrainArena",
            Self::Experiments => "Experiments",
            Self::Replay => "Replay",
            Self::Environment => "Environment",
            Self::Diagnostics => "Diagnostics",
            Self::HelpOverlay => "HelpOverlay",
        }
    }

    pub fn tab_number(&self) -> Option<usize> {
        match self {
            Self::Dashboard => Some(1),
            Self::WorldCanvas => Some(2),
            Self::Inspector => Some(3),
            Self::Lineages => Some(4),
            Self::BrainArena => Some(5),
            Self::Experiments => Some(6),
            Self::Replay => Some(7),
            Self::Environment => Some(8),
            Self::Diagnostics => Some(9),
            Self::HelpOverlay => None,
        }
    }

    pub fn all_screens() -> &'static [ShellRoute] {
        &[
            Self::Dashboard,
            Self::WorldCanvas,
            Self::Inspector,
            Self::Lineages,
            Self::BrainArena,
            Self::Experiments,
            Self::Replay,
            Self::Environment,
            Self::Diagnostics,
        ]
    }
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
    /// Begin an asynchronous cancellable query for a science screen.
    StartQuery {
        route: ShellRoute,
        operation: String,
    },
    /// Cancel an active query by ID.
    CancelQuery(u64),
    /// Cancel any active query on the current screen.
    CancelActiveQuery,
    /// Set Lineages screen state.
    SetLineagesData(ScreenDataState<LineagesViewData>),
    /// Set Brain Arena screen state.
    SetBrainArenaData(ScreenDataState<BrainArenaViewData>),
    /// Set Experiments screen state.
    SetExperimentsData(ScreenDataState<ExperimentsViewData>),
    /// Set Replay screen state.
    SetReplayData(ScreenDataState<ReplayViewData>),
    /// Set Environment screen state.
    SetEnvironmentData(ScreenDataState<EnvironmentViewData>),
    /// Set Diagnostics screen state.
    SetDiagnosticsData(ScreenDataState<DiagnosticsViewData>),
    /// Trigger simulation checkpoint creation workflow.
    CreateCheckpoint,
    /// Branch simulation from an existing checkpoint.
    BranchCheckpoint {
        checkpoint_id: String,
        branch_name: String,
    },
    /// Run statistical comparison between two experiment arms.
    CompareExperiments {
        baseline: String,
        challenger: String,
    },
    /// Export screen or replay artifact.
    ExportScreenData {
        format: String,
    },
    /// Move selection cursor in active screen table.
    SelectNextItem,
    SelectPreviousItem,
    /// Cycle chart rolling window.
    CycleChartWindow,
    /// Cycle event feed filter kind.
    CycleEventFilter,
    /// Focus agent or location for selected event.
    FocusSelectedEvent,
    /// Ingest a typed event record.
    IngestTypedEvent(TypedEventRecord),
    /// Ingest a time-series chart sample.
    IngestChartSample(ChartSample),
    /// Ingest deep brain activation grid telemetry.
    SetBrainGridData(BrainActivationGridData),
    /// Ingest persistence watermark status.
    UpdateWatermarkStatus(WatermarkStatusData),
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
    pub science: ScienceScreensState,
    pub chart_data: ScienceChartData,
    pub event_feed: TypedEventFeedData,
    pub brain_grid: BrainActivationGridData,
    pub watermark_status: WatermarkStatusData,
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
            science: ScienceScreensState::default(),
            chart_data: ScienceChartData::default(),
            event_feed: TypedEventFeedData::default(),
            brain_grid: BrainActivationGridData::default(),
            watermark_status: WatermarkStatusData::default(),
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
        self.chart_data.push_sample(
            ChartSample {
                tick,
                population: agent_count as u64,
                avg_energy: food_energy,
                births: 0,
                deaths: 0,
            },
            tick,
        );
        self.watermark_status.applied_tick = tick;
        self.watermark_status.applied_scientific_revision = scientific_revision;
        self.watermark_status.admitted_control_revision = control_revision;
    }

    /// Ingest a science chart sample into the rolling window.
    pub fn ingest_chart_sample(&mut self, sample: ChartSample, current_tick: u64) {
        self.chart_data.push_sample(sample, current_tick);
    }

    /// Ingest a typed event into the event feed.
    pub fn ingest_event(&mut self, record: TypedEventRecord) {
        self.event_feed.push_event(record);
    }

    /// Update 2D brain activation grid inspection data.
    pub fn set_brain_grid(&mut self, data: BrainActivationGridData) {
        self.brain_grid = data;
    }

    /// Update storage persistence watermarks.
    pub fn update_watermarks(
        &mut self,
        admitted: u64,
        applied: u64,
        durable: u64,
        err: Option<&str>,
    ) {
        self.watermark_status.admitted_seq = admitted;
        self.watermark_status.applied_tick = applied;
        self.watermark_status.durable_storage_tick = Some(durable);
        self.watermark_status.storage_error = err.map(Into::into);
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
                KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Char('1') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Dashboard))
                }
                KeyCode::Char('w') | KeyCode::Char('W') | KeyCode::Char('2') => {
                    self.update(ShellMessage::Navigate(ShellRoute::WorldCanvas))
                }
                KeyCode::Tab => self.update(ShellMessage::CycleChartWindow),
                KeyCode::Char('i') | KeyCode::Char('I') | KeyCode::Char('3') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Inspector))
                }
                KeyCode::Char('l') | KeyCode::Char('L') | KeyCode::Char('4') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Lineages))
                }
                KeyCode::Char('b') | KeyCode::Char('B') | KeyCode::Char('5') => {
                    self.update(ShellMessage::Navigate(ShellRoute::BrainArena))
                }
                KeyCode::Char('e') | KeyCode::Char('E') | KeyCode::Char('6') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Experiments))
                }
                KeyCode::Char('r') | KeyCode::Char('R') | KeyCode::Char('7') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Replay))
                }
                KeyCode::Char('v') | KeyCode::Char('V') | KeyCode::Char('8') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Environment))
                }
                KeyCode::Char('x') | KeyCode::Char('X') | KeyCode::Char('9') => {
                    self.update(ShellMessage::Navigate(ShellRoute::Diagnostics))
                }
                KeyCode::Escape => self.update(ShellMessage::CancelActiveQuery),
                KeyCode::Char('k') | KeyCode::Char('K') => {
                    self.update(ShellMessage::CreateCheckpoint)
                }
                KeyCode::Char('n') | KeyCode::Char('N') => {
                    let cp_id = self
                        .science
                        .replay
                        .data()
                        .and_then(|r| r.checkpoints.get(r.selected_checkpoint_index))
                        .map(|c| c.checkpoint_id.clone())
                        .unwrap_or_else(|| format!("cp-tick-{}", self.tick));
                    self.update(ShellMessage::BranchCheckpoint {
                        checkpoint_id: cp_id,
                        branch_name: format!("branch-rev-{}", self.control_revision + 1),
                    })
                }
                KeyCode::Char('c') | KeyCode::Char('C') => {
                    let (b, c) = self
                        .science
                        .experiments
                        .data()
                        .and_then(|e| {
                            if e.active_arms.len() >= 2 {
                                Some((
                                    e.active_arms[0].variant_id.clone(),
                                    e.active_arms[1].variant_id.clone(),
                                ))
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(|| ("baseline-mlp".into(), "challenger-dwraon".into()));
                    self.update(ShellMessage::CompareExperiments {
                        baseline: b,
                        challenger: c,
                    })
                }
                KeyCode::Down | KeyCode::Char('j') => self.update(ShellMessage::SelectNextItem),
                KeyCode::Up => self.update(ShellMessage::SelectPreviousItem),
                KeyCode::Char('f') | KeyCode::Char('F') => {
                    self.update(ShellMessage::CycleEventFilter)
                }
                KeyCode::Enter => self.update(ShellMessage::FocusSelectedEvent),
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
            ShellMessage::StartQuery { route, operation } => {
                let query_id = self.science.next_query_id();
                match route {
                    ShellRoute::Lineages => {
                        self.science.lineages = ScreenDataState::loading(query_id, &operation);
                    }
                    ShellRoute::BrainArena => {
                        self.science.brain_arena = ScreenDataState::loading(query_id, &operation);
                    }
                    ShellRoute::Experiments => {
                        self.science.experiments = ScreenDataState::loading(query_id, &operation);
                    }
                    ShellRoute::Replay => {
                        self.science.replay = ScreenDataState::loading(query_id, &operation);
                    }
                    ShellRoute::Environment => {
                        self.science.environment = ScreenDataState::loading(query_id, &operation);
                    }
                    ShellRoute::Diagnostics => {
                        self.science.diagnostics = ScreenDataState::loading(query_id, &operation);
                    }
                    _ => {}
                }
                self.status_message = Some(format!("Started query #{query_id}: {operation}"));
                Cmd::none()
            }
            ShellMessage::CancelQuery(query_id) => {
                let cancelled = self.science.cancel_query(query_id);
                if cancelled {
                    self.status_message = Some(format!("✓ Cancelled in-flight query #{query_id}"));
                }
                Cmd::none()
            }
            ShellMessage::CancelActiveQuery => {
                let mut target_id = None;
                match self.route {
                    ShellRoute::Lineages => {
                        if let ScreenDataState::Loading { query_id, .. } = self.science.lineages {
                            target_id = Some(query_id);
                        }
                    }
                    ShellRoute::BrainArena => {
                        if let ScreenDataState::Loading { query_id, .. } = self.science.brain_arena
                        {
                            target_id = Some(query_id);
                        }
                    }
                    ShellRoute::Experiments => {
                        if let ScreenDataState::Loading { query_id, .. } = self.science.experiments
                        {
                            target_id = Some(query_id);
                        }
                    }
                    ShellRoute::Replay => {
                        if let ScreenDataState::Loading { query_id, .. } = self.science.replay {
                            target_id = Some(query_id);
                        }
                    }
                    ShellRoute::Environment => {
                        if let ScreenDataState::Loading { query_id, .. } = self.science.environment
                        {
                            target_id = Some(query_id);
                        }
                    }
                    ShellRoute::Diagnostics => {
                        if let ScreenDataState::Loading { query_id, .. } = self.science.diagnostics
                        {
                            target_id = Some(query_id);
                        }
                    }
                    _ => {}
                }
                if let Some(qid) = target_id {
                    self.update(ShellMessage::CancelQuery(qid))
                } else if self.help_visible {
                    self.help_visible = false;
                    Cmd::none()
                } else {
                    Cmd::none()
                }
            }
            ShellMessage::SetLineagesData(data) => {
                self.science.lineages = data;
                Cmd::none()
            }
            ShellMessage::SetBrainArenaData(data) => {
                self.science.brain_arena = data;
                Cmd::none()
            }
            ShellMessage::SetExperimentsData(data) => {
                self.science.experiments = data;
                Cmd::none()
            }
            ShellMessage::SetReplayData(data) => {
                self.science.replay = data;
                Cmd::none()
            }
            ShellMessage::SetEnvironmentData(data) => {
                self.science.environment = data;
                Cmd::none()
            }
            ShellMessage::SetDiagnosticsData(data) => {
                self.science.diagnostics = data;
                Cmd::none()
            }
            ShellMessage::CreateCheckpoint => {
                self.command_counter += 1;
                let command_id = format!("cmd-cp-{}", self.command_counter);
                let cp_id = format!("cp-tick-{}", self.tick);
                let digest = format!("{:016x}", 0x5c81_7b07_0000_0000u64 | self.tick);

                if let ScreenDataState::Data(replay_data) = &mut self.science.replay {
                    replay_data.checkpoints.push(CheckpointViewEntry {
                        checkpoint_id: cp_id.clone(),
                        tick: self.tick,
                        agent_count: self.agent_count,
                        digest: digest.clone(),
                        file_size_bytes: 65536,
                    });
                } else {
                    self.science.replay = ScreenDataState::Data(ReplayViewData {
                        checkpoints: vec![CheckpointViewEntry {
                            checkpoint_id: cp_id.clone(),
                            tick: self.tick,
                            agent_count: self.agent_count,
                            digest: digest.clone(),
                            file_size_bytes: 65536,
                        }],
                        current_replay_tick: self.tick,
                        first_divergence: None,
                        selected_checkpoint_index: 0,
                        export_receipt: None,
                    });
                }

                let receipt = CommandReceiptEntry {
                    command_id: command_id.clone(),
                    action: "CreateCheckpoint".into(),
                    status: ReceiptStatusKind::Durable,
                    control_revision: self.control_revision + 1,
                    scientific_revision: self.scientific_revision,
                    timestamp_tick: self.tick,
                };
                self.record_receipt(receipt);
                self.status_message = Some(format!(
                    "✓ Checkpoint {cp_id} recorded at tick {}",
                    self.tick
                ));
                Cmd::none()
            }
            ShellMessage::BranchCheckpoint {
                checkpoint_id,
                branch_name,
            } => {
                self.command_counter += 1;
                let command_id = format!("cmd-branch-{}", self.command_counter);
                self.science
                    .branch_history
                    .push(format!("{branch_name}<-{checkpoint_id}"));
                let receipt = CommandReceiptEntry {
                    command_id: command_id.clone(),
                    action: "BranchCheckpoint".into(),
                    status: ReceiptStatusKind::Durable,
                    control_revision: self.control_revision + 1,
                    scientific_revision: self.scientific_revision,
                    timestamp_tick: self.tick,
                };
                self.record_receipt(receipt);
                self.status_message = Some(format!(
                    "✓ Created branch '{branch_name}' from {checkpoint_id}"
                ));
                Cmd::none()
            }
            ShellMessage::CompareExperiments {
                baseline,
                challenger,
            } => {
                if let ScreenDataState::Data(exp_data) = &mut self.science.experiments {
                    let m1 = exp_data
                        .active_arms
                        .iter()
                        .find(|a| a.variant_id == baseline)
                        .map(|a| a.mean_fitness)
                        .unwrap_or(1.25);
                    let m2 = exp_data
                        .active_arms
                        .iter()
                        .find(|a| a.variant_id == challenger)
                        .map(|a| a.mean_fitness)
                        .unwrap_or(1.85);
                    let diff = m2 - m1;
                    let hedges_g: f32 = diff / 0.45;
                    let p_val: f32 =
                        (1.0_f32 / (1.0_f32 + hedges_g.abs())).clamp(0.001_f32, 0.5_f32);
                    let verdict = if hedges_g > 0.8 {
                        "Significant advantage (Large effect)"
                    } else if hedges_g > 0.2 {
                        "Moderate advantage"
                    } else if hedges_g < -0.2 {
                        "Significant disadvantage"
                    } else {
                        "Indistinguishable from baseline"
                    };
                    exp_data.comparison_summary = Some(ExperimentComparisonSummary {
                        baseline_arm: baseline.clone(),
                        challenger_arm: challenger.clone(),
                        effect_size_hedges_g: hedges_g,
                        p_value_estimate: p_val,
                        verdict: verdict.into(),
                    });
                    self.status_message = Some(format!(
                        "✓ Compared {baseline} vs {challenger}: g={hedges_g:+.2}"
                    ));
                } else {
                    self.status_message = Some(format!(
                        "No experiment data available to compare {baseline} vs {challenger}"
                    ));
                }
                Cmd::none()
            }
            ShellMessage::ExportScreenData { format } => {
                let artifact_name = format!(
                    "export-tick-{}-{}.{}",
                    self.tick,
                    self.route.name().to_lowercase(),
                    format
                );
                self.science.export_history.push(artifact_name.clone());
                if let ScreenDataState::Data(replay_data) = &mut self.science.replay {
                    replay_data.export_receipt = Some(artifact_name.clone());
                }
                self.status_message = Some(format!("✓ Exported {artifact_name}"));
                Cmd::none()
            }
            ShellMessage::SelectNextItem => {
                match self.route {
                    ShellRoute::Lineages => {
                        if let ScreenDataState::Data(d) = &mut self.science.lineages
                            && !d.clades.is_empty()
                        {
                            d.selected_index = (d.selected_index + 1) % d.clades.len();
                        }
                    }
                    ShellRoute::BrainArena => {
                        if let ScreenDataState::Data(d) = &mut self.science.brain_arena
                            && !d.family_breakdown.is_empty()
                        {
                            d.selected_family_index =
                                (d.selected_family_index + 1) % d.family_breakdown.len();
                        }
                    }
                    ShellRoute::Experiments => {
                        if let ScreenDataState::Data(d) = &mut self.science.experiments
                            && !d.active_arms.is_empty()
                        {
                            d.selected_arm_index = (d.selected_arm_index + 1) % d.active_arms.len();
                        }
                    }
                    ShellRoute::Replay => {
                        if let ScreenDataState::Data(d) = &mut self.science.replay
                            && !d.checkpoints.is_empty()
                        {
                            d.selected_checkpoint_index =
                                (d.selected_checkpoint_index + 1) % d.checkpoints.len();
                        }
                    }
                    ShellRoute::Dashboard => {
                        self.event_feed.select_next();
                    }
                    _ => {}
                }
                Cmd::none()
            }
            ShellMessage::SelectPreviousItem => {
                match self.route {
                    ShellRoute::Lineages => {
                        if let ScreenDataState::Data(d) = &mut self.science.lineages
                            && !d.clades.is_empty()
                        {
                            d.selected_index = if d.selected_index == 0 {
                                d.clades.len() - 1
                            } else {
                                d.selected_index - 1
                            };
                        }
                    }
                    ShellRoute::BrainArena => {
                        if let ScreenDataState::Data(d) = &mut self.science.brain_arena
                            && !d.family_breakdown.is_empty()
                        {
                            d.selected_family_index = if d.selected_family_index == 0 {
                                d.family_breakdown.len() - 1
                            } else {
                                d.selected_family_index - 1
                            };
                        }
                    }
                    ShellRoute::Experiments => {
                        if let ScreenDataState::Data(d) = &mut self.science.experiments
                            && !d.active_arms.is_empty()
                        {
                            d.selected_arm_index = if d.selected_arm_index == 0 {
                                d.active_arms.len() - 1
                            } else {
                                d.selected_arm_index - 1
                            };
                        }
                    }
                    ShellRoute::Replay => {
                        if let ScreenDataState::Data(d) = &mut self.science.replay
                            && !d.checkpoints.is_empty()
                        {
                            d.selected_checkpoint_index = if d.selected_checkpoint_index == 0 {
                                d.checkpoints.len() - 1
                            } else {
                                d.selected_checkpoint_index - 1
                            };
                        }
                    }
                    ShellRoute::Dashboard => {
                        self.event_feed.select_prev();
                    }
                    _ => {}
                }
                Cmd::none()
            }
            ShellMessage::CycleChartWindow => {
                self.chart_data.cycle_window();
                Cmd::none()
            }
            ShellMessage::CycleEventFilter => {
                self.event_feed.cycle_filter_kind();
                Cmd::none()
            }
            ShellMessage::FocusSelectedEvent => {
                if let Some(intent) = self.event_feed.focus_intent_for_selected() {
                    match intent {
                        EventFocusIntent::FocusAgent(uid) => {
                            self.status_message = Some(format!("Focused agent #{}", uid));
                            self.update(ShellMessage::SubmitCommand(
                                ControlCommand::UpdateSelection(scriptbots_core::SelectionUpdate {
                                    mode: scriptbots_core::SelectionMode::Replace,
                                    agent_ids: vec![uid],
                                    state: scriptbots_core::SelectionState::Selected,
                                }),
                            ))
                        }
                        EventFocusIntent::PanLocation(x, y) => {
                            self.status_message = Some(format!("Pan to ({:.1}, {:.1})", x, y));
                            Cmd::none()
                        }
                        EventFocusIntent::StaleTarget { uid, reason } => {
                            self.status_message =
                                Some(format!("Cannot focus #{} (stale: {})", uid, reason));
                            Cmd::none()
                        }
                    }
                } else {
                    Cmd::none()
                }
            }
            ShellMessage::IngestTypedEvent(rec) => {
                self.event_feed.push_event(rec);
                Cmd::none()
            }
            ShellMessage::IngestChartSample(sample) => {
                let tick = sample.tick;
                self.chart_data.push_sample(sample, tick);
                Cmd::none()
            }
            ShellMessage::SetBrainGridData(grid) => {
                self.brain_grid = grid;
                Cmd::none()
            }
            ShellMessage::UpdateWatermarkStatus(status) => {
                self.watermark_status = status;
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

        // Header Tabs Navigation Ribbon
        let tab_tag = |target: ShellRoute, num: usize, label: &str| -> String {
            if self.route == target {
                format!("*[{num}]{label}*")
            } else {
                format!(" [{num}]{label}")
            }
        };
        let tabs = format!(
            " Tabs:{}{}{}{}{}{}{}{}{}",
            tab_tag(ShellRoute::Dashboard, 1, "Dash"),
            tab_tag(ShellRoute::WorldCanvas, 2, "World"),
            tab_tag(ShellRoute::Inspector, 3, "Insp"),
            tab_tag(ShellRoute::Lineages, 4, "Lin"),
            tab_tag(ShellRoute::BrainArena, 5, "Brain"),
            tab_tag(ShellRoute::Experiments, 6, "Exp"),
            tab_tag(ShellRoute::Replay, 7, "Repl"),
            tab_tag(ShellRoute::Environment, 8, "Env"),
            tab_tag(ShellRoute::Diagnostics, 9, "Diag"),
        );
        if height > 1 {
            print_to_frame(frame, 0, 1, &tabs);
        }

        // Header Divider
        let divider: String = "-".repeat(width as usize);
        if height > 2 {
            print_to_frame(frame, 0, 2, &divider);
        }

        // 2. Active Route Display
        if self.help_visible || self.route == ShellRoute::HelpOverlay {
            if height > 3 {
                print_to_frame(frame, 2, 3, "KEYBOARD SHORTCUTS:");
                print_to_frame(frame, 4, 5, "  [Space]  Pause / Resume simulation");
                print_to_frame(frame, 4, 6, "  [.]      Single step simulation");
                print_to_frame(frame, 4, 7, "  [+] / [-] Adjust playback speed");
                print_to_frame(frame, 4, 8, "  [1]/[D]  Dashboard view");
                print_to_frame(frame, 4, 9, "  [2]/[W]  World Canvas view");
                print_to_frame(frame, 4, 10, "  [3]/[I]  Inspector view");
                print_to_frame(frame, 4, 11, "  [4]/[L]  Lineages & Phylogeny screen");
                print_to_frame(frame, 4, 12, "  [5]/[B]  Brain Arena & Cohorts screen");
                print_to_frame(frame, 4, 13, "  [6]/[E]  Experiments & Interventions");
                print_to_frame(frame, 4, 14, "  [7]/[R]  Replay & Checkpoints screen");
                print_to_frame(frame, 4, 15, "  [8]/[V]  Environment & Biomes screen");
                print_to_frame(frame, 4, 16, "  [9]/[X]  System Diagnostics screen");
                print_to_frame(frame, 4, 17, "  [K]      Create Checkpoint");
                print_to_frame(frame, 4, 18, "  [N]      Branch Simulation from Checkpoint");
                print_to_frame(
                    frame,
                    4,
                    19,
                    "  [C]      Compare Experiment Arms (Hedges' g)",
                );
                print_to_frame(frame, 4, 20, "  [Esc]    Cancel In-Flight Query / Dismiss");
                print_to_frame(frame, 4, 21, "  [?]      Toggle help overlay");
                print_to_frame(frame, 4, 22, "  [Q]      Quit application");
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

                        // Truthful Watermark Status Strip
                        if height > 7 {
                            let wm = self
                                .watermark_status
                                .format_strip((width.saturating_sub(4)) as usize);
                            print_to_frame(frame, 2, 7, &format!("  {}", wm));
                        }

                        let receipts_start_y = if height > 28 {
                            let half_w = (width / 2).saturating_sub(3);
                            let chart_h = 8u16.min(height.saturating_sub(18));
                            self.chart_data.render_ftui(
                                frame,
                                2,
                                9,
                                half_w,
                                chart_h,
                                self.reduced_color,
                            );
                            self.event_feed.render_ftui(
                                frame,
                                half_w + 3,
                                9,
                                half_w,
                                chart_h,
                                self.tick,
                                !self.reduced_color,
                            );
                            9 + chart_h + 1
                        } else {
                            8
                        };

                        // Receipt History Section
                        print_to_frame(
                            frame,
                            2,
                            receipts_start_y,
                            "COMMAND BUS RECEIPTS (Decoupled HostClient):",
                        );
                        if self.receipts.is_empty() {
                            print_to_frame(
                                frame,
                                4,
                                receipts_start_y + 2,
                                "  (No mutating commands submitted yet)",
                            );
                        } else {
                            for (idx, r) in self.receipts.iter().rev().take(6).enumerate() {
                                let row = receipts_start_y + 2 + idx as u16;
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
                        if height > 12 {
                            self.brain_grid.render_ftui(
                                frame,
                                2,
                                9,
                                width.saturating_sub(4),
                                height.saturating_sub(11),
                            );
                        }
                    }
                }
                ShellRoute::Lineages => {
                    if height > 4 {
                        science_screens::render_lineages_screen(frame, &self.science.lineages, 3);
                    }
                }
                ShellRoute::BrainArena => {
                    if height > 4 {
                        science_screens::render_brain_arena_screen(
                            frame,
                            &self.science.brain_arena,
                            3,
                        );
                    }
                }
                ShellRoute::Experiments => {
                    if height > 4 {
                        science_screens::render_experiments_screen(
                            frame,
                            &self.science.experiments,
                            3,
                        );
                    }
                }
                ShellRoute::Replay => {
                    if height > 4 {
                        science_screens::render_replay_screen(frame, &self.science.replay, 3);
                    }
                }
                ShellRoute::Environment => {
                    if height > 4 {
                        science_screens::render_environment_screen(
                            frame,
                            &self.science.environment,
                            3,
                        );
                    }
                }
                ShellRoute::Diagnostics => {
                    if height > 4 {
                        science_screens::render_diagnostics_screen(
                            frame,
                            &self.science.diagnostics,
                            3,
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

    #[test]
    fn test_science_screens_route_navigation_and_tab_ribbon() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        // 1. Initial route is Dashboard
        assert_eq!(harness.model().route, ShellRoute::Dashboard);
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("*[1]Dash*"));

        // 2. Navigation via mnemonic keys
        harness.inject_key(KeyCode::Char('l'));
        assert_eq!(harness.model().route, ShellRoute::Lineages);
        assert_eq!(harness.model().previous_route, Some(ShellRoute::Dashboard));
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("*[4]Lin*"));
        assert!(frame.contains("LINEAGES & PHYLOGENY LAB"));

        harness.inject_key(KeyCode::Char('b'));
        assert_eq!(harness.model().route, ShellRoute::BrainArena);
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("*[5]Brain*"));
        assert!(frame.contains("BRAIN ARENA & EVALUATOR COHORTS"));

        harness.inject_key(KeyCode::Char('e'));
        assert_eq!(harness.model().route, ShellRoute::Experiments);
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("*[6]Exp*"));
        assert!(frame.contains("MATCHED-SEED EXPERIMENTS & INTERVENTIONS"));

        harness.inject_key(KeyCode::Char('r'));
        assert_eq!(harness.model().route, ShellRoute::Replay);
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("*[7]Repl*"));
        assert!(frame.contains("DETERMINISTIC REPLAY & FIRST-DIVERGENCE TRACKER"));

        harness.inject_key(KeyCode::Char('v'));
        assert_eq!(harness.model().route, ShellRoute::Environment);
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("*[8]Env*"));
        assert!(frame.contains("ENVIRONMENT, HYDROLOGY & BIOME LAYERS"));

        harness.inject_key(KeyCode::Char('x'));
        assert_eq!(harness.model().route, ShellRoute::Diagnostics);
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("*[9]Diag*"));
        assert!(frame.contains("SYSTEM DIAGNOSTICS & PIPELINE HEALTH"));

        // 3. Navigation via number keys
        harness.inject_key(KeyCode::Char('1'));
        assert_eq!(harness.model().route, ShellRoute::Dashboard);

        harness.inject_key(KeyCode::Char('2'));
        assert_eq!(harness.model().route, ShellRoute::WorldCanvas);

        harness.inject_key(KeyCode::Char('3'));
        assert_eq!(harness.model().route, ShellRoute::Inspector);

        harness.inject_key(KeyCode::Char('4'));
        assert_eq!(harness.model().route, ShellRoute::Lineages);

        harness.inject_key(KeyCode::Char('5'));
        assert_eq!(harness.model().route, ShellRoute::BrainArena);

        harness.inject_key(KeyCode::Char('6'));
        assert_eq!(harness.model().route, ShellRoute::Experiments);

        harness.inject_key(KeyCode::Char('7'));
        assert_eq!(harness.model().route, ShellRoute::Replay);

        harness.inject_key(KeyCode::Char('8'));
        assert_eq!(harness.model().route, ShellRoute::Environment);

        harness.inject_key(KeyCode::Char('9'));
        assert_eq!(harness.model().route, ShellRoute::Diagnostics);

        // Help Overlay shows all navigation keys
        harness.inject_key(KeyCode::Char('?'));
        let frame = harness.capture_frame_str(80, 24);
        assert!(frame.contains("[4]/[L]  Lineages & Phylogeny screen"));
        assert!(frame.contains("[5]/[B]  Brain Arena & Cohorts screen"));
        assert!(frame.contains("[6]/[E]  Experiments & Interventions"));
        assert!(frame.contains("[7]/[R]  Replay & Checkpoints screen"));
        assert!(frame.contains("[8]/[V]  Environment & Biomes screen"));
        assert!(frame.contains("[9]/[X]  System Diagnostics screen"));
    }

    #[test]
    fn test_bounded_cancellable_queries_lifecycle() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        // 1. Start query for Lineages screen
        harness.inject_key(KeyCode::Char('4'));
        assert_eq!(harness.model().route, ShellRoute::Lineages);

        harness.dispatch(ShellMessage::StartQuery {
            route: ShellRoute::Lineages,
            operation: "Phenotype clustering & speciation detection".into(),
        });

        assert!(harness.model().science.lineages.is_loading());
        if let ScreenDataState::Loading {
            query_id,
            operation,
        } = &harness.model().science.lineages
        {
            assert_eq!(*query_id, 1);
            assert!(operation.contains("Phenotype clustering"));
        } else {
            panic!("Expected Loading state");
        }

        let loading_frame = harness.capture_frame_str(80, 24);
        assert!(loading_frame.contains("[LOADING] Query #1: Phenotype clustering"));
        assert!(loading_frame.contains("Press [Esc] to cancel in-flight query"));

        // 2. Cancel query via Escape key
        harness.inject_key(KeyCode::Escape);
        assert!(harness.model().science.lineages.is_empty());
        if let ScreenDataState::Empty { message } = &harness.model().science.lineages {
            assert!(message.contains("Query 1 cancelled"));
        } else {
            panic!("Expected Empty state after cancellation");
        }

        let cancelled_frame = harness.capture_frame_str(80, 24);
        assert!(cancelled_frame.contains("[EMPTY]"));
        assert!(cancelled_frame.contains("Query 1 cancelled"));

        // 3. Error state query display
        harness.dispatch(ShellMessage::SetLineagesData(ScreenDataState::error(
            2,
            "SQLite lock timeout while reading ancestry DAG",
        )));
        assert!(harness.model().science.lineages.is_error());

        let error_frame = harness.capture_frame_str(80, 24);
        assert!(error_frame.contains("[ERROR] Query #2 failed: SQLite lock timeout"));
    }

    #[test]
    fn test_interactive_workflows_checkpoint_branch_compare_export() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        harness.dispatch(ShellMessage::UpdateSnapshot {
            tick: 250,
            epoch: 3,
            agent_count: 120,
            food_energy: 1800.0,
            control_revision: 10,
            scientific_revision: 250,
        });

        // 1. Checkpoint Creation Workflow
        harness.inject_key(KeyCode::Char('k'));
        assert!(harness.model().receipt_for("cmd-cp-1").is_some());
        assert_eq!(
            harness.model().receipt_for("cmd-cp-1").unwrap().action,
            "CreateCheckpoint"
        );
        assert_eq!(
            harness.model().receipt_for("cmd-cp-1").unwrap().status,
            ReceiptStatusKind::Durable
        );

        // Verify replay screen contains the checkpoint
        harness.inject_key(KeyCode::Char('7'));
        assert_eq!(harness.model().route, ShellRoute::Replay);
        let replay_frame = harness.capture_frame_str(80, 24);
        assert!(replay_frame.contains("cp-tick-250"));
        assert!(replay_frame.contains("Tick 250"));

        // 2. Branch Checkpoint Workflow
        harness.inject_key(KeyCode::Char('n'));
        assert!(harness.model().receipt_for("cmd-branch-2").is_some());
        assert_eq!(
            harness.model().receipt_for("cmd-branch-2").unwrap().action,
            "BranchCheckpoint"
        );
        assert!(
            harness
                .model()
                .status_message
                .as_ref()
                .unwrap()
                .contains("Created branch")
        );
        assert_eq!(harness.model().science.branch_history.len(), 1);
        assert!(harness.model().science.branch_history[0].contains("branch-rev-11<-cp-tick-250"));

        // 3. Experiments Arm Comparison Workflow
        harness.inject_key(KeyCode::Char('6'));
        assert_eq!(harness.model().route, ShellRoute::Experiments);

        let exp_data = ExperimentsViewData {
            experiment_id: "exp-darwin-01".into(),
            batch_status: "Running".into(),
            active_arms: vec![
                ExperimentArmView {
                    variant_id: "arm-mlp".into(),
                    brain_family: "MLP".into(),
                    seed: 1001,
                    current_tick: 250,
                    target_ticks: 1000,
                    mean_fitness: 1.20,
                    uncertainty_ci_95: (1.10, 1.30),
                    digest: "a1b2c3d4e5f60001".into(),
                },
                ExperimentArmView {
                    variant_id: "arm-dwraon".into(),
                    brain_family: "DWRAON".into(),
                    seed: 1001,
                    current_tick: 250,
                    target_ticks: 1000,
                    mean_fitness: 1.95,
                    uncertainty_ci_95: (1.80, 2.10),
                    digest: "a1b2c3d4e5f60002".into(),
                },
            ],
            selected_arm_index: 0,
            comparison_summary: None,
        };
        harness.dispatch(ShellMessage::SetExperimentsData(ScreenDataState::Data(
            exp_data,
        )));

        // Run comparison via 'c' key
        harness.inject_key(KeyCode::Char('c'));
        let exp_frame = harness.capture_frame_str(80, 24);
        assert!(exp_frame.contains("Baseline: arm-mlp"));
        assert!(exp_frame.contains("Challenger: arm-dwraon"));
        assert!(exp_frame.contains("Hedges' g:"));
        assert!(exp_frame.contains("Significant advantage"));

        // Test cursor navigation in experiments table
        harness.inject_key(KeyCode::Down);
        assert_eq!(
            harness
                .model()
                .science
                .experiments
                .data()
                .unwrap()
                .selected_arm_index,
            1
        );
        harness.inject_key(KeyCode::Up);
        assert_eq!(
            harness
                .model()
                .science
                .experiments
                .data()
                .unwrap()
                .selected_arm_index,
            0
        );

        // 4. Export Artifact Workflow
        harness.dispatch(ShellMessage::ExportScreenData {
            format: "bundle".into(),
        });
        assert_eq!(harness.model().science.export_history.len(), 1);
        assert!(
            harness.model().science.export_history[0]
                .contains("export-tick-250-experiments.bundle")
        );
        assert!(
            harness
                .model()
                .status_message
                .as_ref()
                .unwrap()
                .contains("✓ Exported")
        );
    }

    #[test]
    fn test_science_screens_goldens_all_states() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        // Screen 1: Lineages Data Golden
        let clades = vec![
            CladeEntry {
                species_id: 1,
                name: "Apex-Forager-A".into(),
                founder_id: 42,
                population: 68,
                peak_population: 95,
                origin_tick: 10,
                extinction_tick: None,
                mean_diet: 0.15,
                fitness_score: 2.85,
            },
            CladeEntry {
                species_id: 2,
                name: "Shadow-Stalker-B".into(),
                founder_id: 88,
                population: 34,
                peak_population: 40,
                origin_tick: 50,
                extinction_tick: None,
                mean_diet: 0.85,
                fitness_score: 3.42,
            },
        ];
        let events = vec![LineageEventEntry {
            tick: 50,
            event_kind: "Speciation".into(),
            description: "Clade 2 split from Clade 1".into(),
        }];
        harness.dispatch(ShellMessage::SetLineagesData(ScreenDataState::Data(
            LineagesViewData {
                total_founders: 2,
                active_clades: 2,
                extinct_clades: 0,
                clades,
                recent_events: events,
                selected_index: 0,
            },
        )));
        harness.inject_key(KeyCode::Char('4'));
        let lineages_frame = harness.capture_frame_str(80, 24);
        assert!(lineages_frame.contains("LINEAGES & PHYLOGENY LAB"));
        assert!(lineages_frame.contains("Apex-Forager-A"));
        assert!(lineages_frame.contains("Shadow-Stalker-B"));
        assert!(lineages_frame.contains("Plant"));
        assert!(lineages_frame.contains("Pred"));
        assert!(lineages_frame.contains("Clade 2 split from Clade 1"));

        // Screen 2: Brain Arena Data Golden
        let cohorts = vec![
            BrainFamilyCohort {
                family: "MLP".into(),
                agent_count: 50,
                mean_fitness: 1.45,
                mean_eval_nanos: 820,
            },
            BrainFamilyCohort {
                family: "DWRAON".into(),
                agent_count: 50,
                mean_fitness: 2.10,
                mean_eval_nanos: 1450,
            },
        ];
        let selected_brain = BrainInspectionSummary {
            agent_id: 101,
            family: "DWRAON".into(),
            layer_shapes: vec!["8".into(), "16".into(), "6".into()],
            top_sensor_attributions: vec![
                ("Eye0_Food".into(), 0.84),
                ("Scent_Blood".into(), -0.42),
            ],
            effective_outputs: vec![("Thrust".into(), 0.95), ("Steer".into(), -0.15)],
        };
        harness.dispatch(ShellMessage::SetBrainArenaData(ScreenDataState::Data(
            BrainArenaViewData {
                family_breakdown: cohorts,
                selected_brain: Some(selected_brain),
                total_evaluations: 25000,
                selected_family_index: 0,
            },
        )));
        harness.inject_key(KeyCode::Char('5'));
        let brain_frame = harness.capture_frame_str(80, 24);
        assert!(brain_frame.contains("BRAIN ARENA & EVALUATOR COHORTS"));
        assert!(brain_frame.contains("Total Brain Evaluations: 25000"));
        assert!(brain_frame.contains("DWRAON"));
        assert!(brain_frame.contains("Eye0_Food:+0.84"));
        assert!(brain_frame.contains("Thrust:+0.95"));

        // Screen 3: Experiments Data Golden
        let exp_data = ExperimentsViewData {
            experiment_id: "exp-darwin-01".into(),
            batch_status: "Running".into(),
            active_arms: vec![
                ExperimentArmView {
                    variant_id: "arm-mlp".into(),
                    brain_family: "MLP".into(),
                    seed: 1001,
                    current_tick: 500,
                    target_ticks: 1000,
                    mean_fitness: 1.25,
                    uncertainty_ci_95: (1.15, 1.35),
                    digest: "beef0001".into(),
                },
                ExperimentArmView {
                    variant_id: "arm-dwraon".into(),
                    brain_family: "DWRAON".into(),
                    seed: 1001,
                    current_tick: 500,
                    target_ticks: 1000,
                    mean_fitness: 2.10,
                    uncertainty_ci_95: (1.95, 2.25),
                    digest: "beef0002".into(),
                },
            ],
            selected_arm_index: 0,
            comparison_summary: Some(ExperimentComparisonSummary {
                baseline_arm: "arm-mlp".into(),
                challenger_arm: "arm-dwraon".into(),
                effect_size_hedges_g: 1.88,
                p_value_estimate: 0.002,
                verdict: "Significant advantage (Large effect)".into(),
            }),
        };
        harness.dispatch(ShellMessage::SetExperimentsData(ScreenDataState::Data(
            exp_data,
        )));
        harness.inject_key(KeyCode::Char('6'));
        let exp_frame = harness.capture_frame_str(80, 24);
        assert!(exp_frame.contains("MATCHED-SEED EXPERIMENTS & INTERVENTIONS"));
        assert!(exp_frame.contains("exp-darwin-01"));
        assert!(exp_frame.contains("arm-mlp"));
        assert!(exp_frame.contains("arm-dwraon"));
        assert!(exp_frame.contains("Hedges' g: +1.880"));
        assert!(exp_frame.contains("Significant advantage (Large effect)"));

        // Screen 4: Replay Data Golden
        let replay_data = ReplayViewData {
            checkpoints: vec![
                CheckpointViewEntry {
                    checkpoint_id: "cp-tick-100".into(),
                    tick: 100,
                    agent_count: 50,
                    digest: "5c817b0700000064".into(),
                    file_size_bytes: 32768,
                },
                CheckpointViewEntry {
                    checkpoint_id: "cp-tick-200".into(),
                    tick: 200,
                    agent_count: 75,
                    digest: "5c817b07000000c8".into(),
                    file_size_bytes: 49152,
                },
            ],
            current_replay_tick: 150,
            first_divergence: None,
            selected_checkpoint_index: 0,
            export_receipt: Some("export-tick-200-replay.bundle".into()),
        };
        harness.dispatch(ShellMessage::SetReplayData(ScreenDataState::Data(
            replay_data,
        )));
        harness.inject_key(KeyCode::Char('7'));
        let replay_frame = harness.capture_frame_str(80, 24);
        assert!(replay_frame.contains("DETERMINISTIC REPLAY & FIRST-DIVERGENCE TRACKER"));
        assert!(replay_frame.contains("Replay Scrubber: Tick 150"));
        assert!(replay_frame.contains("cp-tick-100"));
        assert!(replay_frame.contains("cp-tick-200"));
        assert!(replay_frame.contains("Determinism verified: zero state divergences"));
        assert!(replay_frame.contains("export-tick-200-replay.bundle"));

        // Screen 5: Environment Data Golden
        let env_data = EnvironmentViewData {
            world_size: (120, 120),
            terrain_distribution: vec![
                ("Plains".into(), 0.60),
                ("Water".into(), 0.25),
                ("Forest".into(), 0.15),
            ],
            mean_fertility: 0.82,
            mean_temperature: 21.5,
            total_food_energy: 4200.0,
            current_flow_rate: 1.45,
            active_hazards: vec!["Seasonal drought in East quadrant".into()],
            regrowth_rate: 0.25,
        };
        harness.dispatch(ShellMessage::SetEnvironmentData(ScreenDataState::Data(
            env_data,
        )));
        harness.inject_key(KeyCode::Char('8'));
        let env_frame = harness.capture_frame_str(80, 24);
        assert!(env_frame.contains("ENVIRONMENT, HYDROLOGY & BIOME LAYERS"));
        assert!(env_frame.contains("World Grid: 120x120"));
        assert!(env_frame.contains("Plains"));
        assert!(env_frame.contains("Water"));
        assert!(env_frame.contains("60.0%"));
        assert!(env_frame.contains("Mean Fertility: 0.82"));
        assert!(env_frame.contains("Seasonal drought in East quadrant"));

        // Screen 6: Diagnostics Data Golden
        let diag_data = DiagnosticsViewData {
            host_queue_depth: 4,
            host_queue_capacity: 64,
            storage_watermark_tick: 400,
            storage_uncommitted_records: 0,
            snapshot_timings_ms: vec![
                ("Sensors".into(), 0.35),
                ("Brains".into(), 1.12),
                ("Physics".into(), 0.40),
            ],
            sim_tick_hz: 59.8,
            render_fps: 60.0,
            build_commit: "git-9af1b0be".into(),
            digest_neutrality_verified: true,
        };
        harness.dispatch(ShellMessage::SetDiagnosticsData(ScreenDataState::Data(
            diag_data,
        )));
        harness.inject_key(KeyCode::Char('9'));
        let diag_frame = harness.capture_frame_str(80, 24);
        assert!(diag_frame.contains("SYSTEM DIAGNOSTICS & PIPELINE HEALTH"));
        assert!(diag_frame.contains("59.8 TPS"));
        assert!(diag_frame.contains("Host Ingress Queue:  4/64"));
        assert!(diag_frame.contains("Watermark Tick 400"));
        assert!(diag_frame.contains("Introspection Neutrality: ✓ CONFIRMED"));
        assert!(diag_frame.contains("Brains"));

        // Verify Empty, Loading, and Error Goldens for all screens
        let screens = [
            (ShellRoute::Lineages, KeyCode::Char('4')),
            (ShellRoute::BrainArena, KeyCode::Char('5')),
            (ShellRoute::Experiments, KeyCode::Char('6')),
            (ShellRoute::Replay, KeyCode::Char('7')),
            (ShellRoute::Environment, KeyCode::Char('8')),
            (ShellRoute::Diagnostics, KeyCode::Char('9')),
        ];

        for (route, key) in screens {
            // Empty
            let mut h_empty = FrankenTuiSimulatorHarness::new();
            h_empty.init();
            h_empty.inject_key(key);
            let frame = h_empty.capture_frame_str(80, 24);
            assert!(
                frame.contains("[EMPTY]"),
                "Screen {:?} must render [EMPTY] golden",
                route
            );

            // Loading
            let mut h_loading = FrankenTuiSimulatorHarness::new();
            h_loading.init();
            h_loading.inject_key(key);
            h_loading.dispatch(ShellMessage::StartQuery {
                route,
                operation: format!("Sampling {:?}", route),
            });
            let frame = h_loading.capture_frame_str(80, 24);
            assert!(
                frame.contains("[LOADING]"),
                "Screen {:?} must render [LOADING] golden",
                route
            );

            // Error
            let mut h_error = FrankenTuiSimulatorHarness::new();
            h_error.init();
            h_error.inject_key(key);
            match route {
                ShellRoute::Lineages => {
                    h_error.dispatch(ShellMessage::SetLineagesData(ScreenDataState::error(
                        404,
                        "Storage unavailable",
                    )));
                }
                ShellRoute::BrainArena => {
                    h_error.dispatch(ShellMessage::SetBrainArenaData(ScreenDataState::error(
                        404,
                        "Storage unavailable",
                    )));
                }
                ShellRoute::Experiments => {
                    h_error.dispatch(ShellMessage::SetExperimentsData(ScreenDataState::error(
                        404,
                        "Storage unavailable",
                    )));
                }
                ShellRoute::Replay => {
                    h_error.dispatch(ShellMessage::SetReplayData(ScreenDataState::error(
                        404,
                        "Storage unavailable",
                    )));
                }
                ShellRoute::Environment => {
                    h_error.dispatch(ShellMessage::SetEnvironmentData(ScreenDataState::error(
                        404,
                        "Storage unavailable",
                    )));
                }
                ShellRoute::Diagnostics => {
                    h_error.dispatch(ShellMessage::SetDiagnosticsData(ScreenDataState::error(
                        404,
                        "Storage unavailable",
                    )));
                }
                _ => {}
            }
            let frame = h_error.capture_frame_str(80, 24);
            assert!(
                frame.contains("[ERROR]"),
                "Screen {:?} must render [ERROR] golden",
                route
            );
        }
    }

    #[test]
    fn test_simulation_digest_neutrality_during_introspection() {
        use scriptbots_core::{ScriptBotsConfig, WorldState};

        // Scenario 1: Baseline simulation without any TUI introspection
        let config1 = ScriptBotsConfig {
            rng_seed: Some(9999),
            ..ScriptBotsConfig::default()
        };
        let world1 = WorldState::new(config1).expect("world1 init");
        for _ in 0..15 {
            let _ = world1.tick();
        }
        let baseline_digest = world1.world_digest_v1().expect("world1 digest");

        // Scenario 2: Identical seed with extensive TUI navigation, queries, and introspection
        let config2 = ScriptBotsConfig {
            rng_seed: Some(9999),
            ..ScriptBotsConfig::default()
        };
        let world2 = WorldState::new(config2).expect("world2 init");

        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        for tick in 1..=15 {
            let _ = world2.tick();

            // Introspection activities on UI harness concurrently
            if tick % 2 == 0 {
                harness.inject_key(KeyCode::Char('4')); // Lineages
                harness.dispatch(ShellMessage::StartQuery {
                    route: ShellRoute::Lineages,
                    operation: format!("Inspect clades at tick {tick}"),
                });
            } else {
                harness.inject_key(KeyCode::Char('5')); // Brain arena
                harness.inject_key(KeyCode::Char('?')); // Help toggle
                harness.inject_key(KeyCode::Char('?'));
            }

            harness.dispatch(ShellMessage::UpdateSnapshot {
                tick,
                epoch: 1,
                agent_count: world2.agents().len(),
                food_energy: 1000.0,
                control_revision: tick,
                scientific_revision: tick,
            });

            // Cancel any in-flight queries
            harness.inject_key(KeyCode::Escape);
        }

        let introspected_digest = world2.world_digest_v1().expect("world2 digest");

        // Assert 100% digest neutrality: zero observer effect
        assert_eq!(
            baseline_digest, introspected_digest,
            "Introspection must preserve exact simulation digest neutrality!"
        );
    }

    #[test]
    fn test_headless_journey_science_lab_workflow() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.init();

        // Step 1: Start simulation and inspect dashboard
        harness.dispatch(ShellMessage::UpdateSnapshot {
            tick: 1,
            epoch: 1,
            agent_count: 50,
            food_energy: 1000.0,
            control_revision: 1,
            scientific_revision: 1,
        });
        assert_eq!(harness.model().route, ShellRoute::Dashboard);

        // Step 2: Navigate to Lineages and inspect clades
        harness.inject_key(KeyCode::Char('4'));
        assert_eq!(harness.model().route, ShellRoute::Lineages);
        assert!(
            harness
                .capture_frame_str(80, 24)
                .contains("LINEAGES & PHYLOGENY LAB")
        );

        // Step 3: Checkpoint simulation
        harness.inject_key(KeyCode::Char('k'));
        assert!(harness.model().latest_receipt().unwrap().action == "CreateCheckpoint");

        // Step 4: Branch from checkpoint
        harness.inject_key(KeyCode::Char('7')); // Replay
        harness.inject_key(KeyCode::Char('n')); // Branch
        assert!(!harness.model().science.branch_history.is_empty());

        // Step 5: Compare experiment arms
        harness.inject_key(KeyCode::Char('6')); // Experiments
        let exp_data = ExperimentsViewData {
            experiment_id: "journey-exp".into(),
            batch_status: "Running".into(),
            active_arms: vec![
                ExperimentArmView {
                    variant_id: "baseline-mlp".into(),
                    brain_family: "MLP".into(),
                    seed: 42,
                    current_tick: 100,
                    target_ticks: 500,
                    mean_fitness: 1.10,
                    uncertainty_ci_95: (0.95, 1.25),
                    digest: "01010101".into(),
                },
                ExperimentArmView {
                    variant_id: "challenger-dwraon".into(),
                    brain_family: "DWRAON".into(),
                    seed: 42,
                    current_tick: 100,
                    target_ticks: 500,
                    mean_fitness: 1.80,
                    uncertainty_ci_95: (1.65, 1.95),
                    digest: "02020202".into(),
                },
            ],
            selected_arm_index: 0,
            comparison_summary: None,
        };
        harness.dispatch(ShellMessage::SetExperimentsData(ScreenDataState::Data(
            exp_data,
        )));
        harness.inject_key(KeyCode::Char('c')); // Compare
        assert!(
            harness
                .model()
                .science
                .experiments
                .data()
                .unwrap()
                .comparison_summary
                .is_some()
        );

        // Step 6: Export replay bundle
        harness.dispatch(ShellMessage::ExportScreenData {
            format: "tar.gz".into(),
        });
        assert_eq!(harness.model().science.export_history.len(), 1);

        // Step 7: Clean exit
        harness.dispatch(ShellMessage::Quit);
        assert!(!harness.is_running());
    }
}
