//! FrankenTUI Model shell, command receipts, and simulator harness (bd-2z0.6.3).

use crate::control::{CommandStatusDto, ControlError, ControlHandle};
use scriptbots_core::ControlCommand;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// View routes in the FrankenTUI Evolution Lab shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellRoute {
    Dashboard,
    WorldCanvas,
    Inspector,
    HelpOverlay,
}

/// Actions accepted by the FrankenTUI shell model.
#[derive(Debug, Clone)]
pub enum ShellMessage {
    Navigate(ShellRoute),
    ToggleHelp,
    SubmitCommand(ControlCommand),
    CommandReceipt(CommandStatusDto),
    UpdateSnapshot { tick: u64, agent_count: usize },
    SetStatus(String),
}

/// State transitions and Model state for the FrankenTUI shell.
#[derive(Debug, Clone)]
pub struct FrankenTuiModel {
    pub route: ShellRoute,
    pub previous_route: Option<ShellRoute>,
    pub help_visible: bool,
    pub paused: bool,
    pub speed_multiplier: f32,
    pub tick: u64,
    pub agent_count: usize,
    pub receipts: VecDeque<CommandStatusDto>,
    pub status_message: Option<String>,
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
            agent_count: 0,
            receipts: VecDeque::with_capacity(32),
            status_message: None,
        }
    }
}

impl FrankenTuiModel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, msg: ShellMessage) {
        match msg {
            ShellMessage::Navigate(route) => {
                if self.route != route {
                    self.previous_route = Some(self.route);
                    self.route = route;
                }
            }
            ShellMessage::ToggleHelp => {
                self.help_visible = !self.help_visible;
            }
            ShellMessage::SubmitCommand(cmd) => match cmd {
                ControlCommand::Pause => self.paused = true,
                ControlCommand::Resume => self.paused = false,
                ControlCommand::SetSpeed(s) => self.speed_multiplier = s,
                ControlCommand::Step => self.paused = true,
                _ => {}
            },
            ShellMessage::CommandReceipt(receipt) => {
                self.record_receipt(receipt);
            }
            ShellMessage::UpdateSnapshot { tick, agent_count } => {
                self.tick = tick;
                self.agent_count = agent_count;
            }
            ShellMessage::SetStatus(msg) => {
                self.status_message = Some(msg);
            }
        }
    }

    pub fn record_receipt(&mut self, receipt: CommandStatusDto) {
        if self.receipts.len() >= 32 {
            self.receipts.pop_front();
        }
        self.receipts.push_back(receipt);
    }
}

/// Deterministic simulator harness for testing FrankenTUI shell transitions without a live world.
#[derive(Debug, Default)]
pub struct FrankenTuiSimulatorHarness {
    pub model: FrankenTuiModel,
    pub history: Vec<ShellMessage>,
}

impl FrankenTuiSimulatorHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dispatch(&mut self, msg: ShellMessage) {
        self.history.push(msg.clone());
        self.model.update(msg);
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

        let receipt = CommandStatusDto {
            command_id: "cmd-1".into(),
            admission_sequence: Some(1),
            application_state: "applied".into(),
            journal_state: "durable".into(),
            control_revision: 1,
            scientific_revision: 10,
        };

        model.update(ShellMessage::CommandReceipt(receipt));
        assert_eq!(model.receipts.len(), 1);
        assert_eq!(model.receipts[0].command_id, "cmd-1");
    }

    #[test]
    fn test_simulator_harness_records_deterministic_history() {
        let mut harness = FrankenTuiSimulatorHarness::new();
        harness.dispatch(ShellMessage::UpdateSnapshot {
            tick: 100,
            agent_count: 50,
        });
        harness.dispatch(ShellMessage::Navigate(ShellRoute::Inspector));

        assert_eq!(harness.model.tick, 100);
        assert_eq!(harness.model.agent_count, 50);
        assert_eq!(harness.model.route, ShellRoute::Inspector);
        assert_eq!(harness.history.len(), 2);
    }
}
