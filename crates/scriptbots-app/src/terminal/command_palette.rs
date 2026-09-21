//! Searchable command palette, key binding registry, and acknowledged controls (bd-2z0.6.5).
//!
//! Provides the single live command registry driving the terminal command palette,
//! help overlay, keyboard shortcuts, mouse selection, and FrankenTUI Model shell.
//! Every mutating action routes through HostClient command envelopes with visible
//! receipt transitions (`Pending` -> `Admitted` -> `Applied` / `Durable` / `Rejected` /
//! `Failed` / `StaleRevision`). Stale revision conflicts explain recovery without
//! optimistic state mutation.

use crate::control::{CommandStatusDto, ControlError, ControlHandle};
use scriptbots_core::interventions::{self, InterventionCommand, InterventionSurface, Region};
use scriptbots_core::{ControlCommand, SimulationCommand};
use serde::{Deserialize, Serialize};

/// Action represented in the command palette and keybinding registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandPaletteAction {
    // Playback & Stepping Controls
    TogglePause,
    Pause,
    Resume,
    StepOnce,
    SpeedUp,
    SpeedDown,
    SetSpeed1x,
    SetSpeed2x,
    SetSpeed4x,
    SetSpeedMax,

    // Scenario & Config Interventions
    SpawnHerbivore,
    SpawnCarnivore,
    TriggerDrought,
    ResetWorld,
    ReloadConfig,

    // Checkpoint & Export
    CreateCheckpoint,
    BranchCheckpoint,
    CompareExperiments,
    ExportAsciiScreenshot,
    ExportData,

    // Screen Navigation
    NavigateDashboard,
    NavigateWorld,
    NavigateInspector,
    NavigateLineages,
    NavigateBrainArena,
    NavigateExperiments,
    NavigateReplay,
    NavigateEnvironment,
    NavigateDiagnostics,
    ToggleArchipelago,
    ToggleRail,

    // Diagnostics & View
    ToggleDiagnostics,
    ToggleProbe,
    FocusTopPredator,
    FocusOldest,
    CycleTheme,
    CyclePalette,
    ShowHelp,
    Quit,
}

impl CommandPaletteAction {
    /// Convert mutating actions to canonical [`ControlCommand`].
    /// Non-mutating UI/navigation actions return `None`.
    #[must_use]
    pub fn to_control_command(
        &self,
        current_speed: f32,
        current_paused: bool,
    ) -> Option<ControlCommand> {
        match self {
            Self::TogglePause => Some(ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(!current_paused),
                speed_multiplier: Some(if !current_paused {
                    0.0
                } else {
                    current_speed.max(1.0)
                }),
                step_once: false,
            })),
            Self::Pause => Some(ControlCommand::Pause),
            Self::Resume => Some(ControlCommand::Resume),
            Self::StepOnce => Some(ControlCommand::Step),
            Self::SpeedUp => {
                let speed = (current_speed + 0.5).clamp(0.5, 8.0);
                Some(ControlCommand::UpdateSimulation(SimulationCommand {
                    paused: Some(false),
                    speed_multiplier: Some(speed),
                    step_once: false,
                }))
            }
            Self::SpeedDown => {
                let speed = (current_speed - 0.5).max(0.0);
                Some(ControlCommand::UpdateSimulation(SimulationCommand {
                    paused: Some(speed == 0.0),
                    speed_multiplier: Some(speed),
                    step_once: false,
                }))
            }
            Self::SetSpeed1x => Some(ControlCommand::SetSpeed(1.0)),
            Self::SetSpeed2x => Some(ControlCommand::SetSpeed(2.0)),
            Self::SetSpeed4x => Some(ControlCommand::SetSpeed(4.0)),
            Self::SetSpeedMax => Some(ControlCommand::SetSpeed(8.0)),
            Self::SpawnHerbivore => Some(ControlCommand::SpawnAgent {
                herbivore_tendency: 1.0,
            }),
            Self::SpawnCarnivore => Some(ControlCommand::SpawnAgent {
                herbivore_tendency: 0.0,
            }),
            Self::TriggerDrought => {
                let region = Region::All;
                if let Ok(drought) = interventions::drought(region, 200, 0.2) {
                    Some(ControlCommand::Intervention(Box::new(
                        InterventionCommand::new(
                            drought,
                            InterventionSurface::Tui,
                            "tui_command_palette",
                        ),
                    )))
                } else {
                    None
                }
            }
            Self::ResetWorld => Some(ControlCommand::UpdateConfig(Box::default())),
            Self::ReloadConfig => Some(ControlCommand::UpdateConfig(Box::default())),
            Self::CreateCheckpoint | Self::BranchCheckpoint => {
                Some(ControlCommand::UpdateSimulation(SimulationCommand {
                    paused: Some(true),
                    speed_multiplier: None,
                    step_once: false,
                }))
            }
            Self::Quit => Some(ControlCommand::Shutdown),
            _ => None,
        }
    }

    /// Whether this action mutates world/host simulation state (routes via HostClient).
    #[must_use]
    pub const fn is_mutating(&self) -> bool {
        matches!(
            self,
            Self::TogglePause
                | Self::Pause
                | Self::Resume
                | Self::StepOnce
                | Self::SpeedUp
                | Self::SpeedDown
                | Self::SetSpeed1x
                | Self::SetSpeed2x
                | Self::SetSpeed4x
                | Self::SetSpeedMax
                | Self::SpawnHerbivore
                | Self::SpawnCarnivore
                | Self::TriggerDrought
                | Self::ResetWorld
                | Self::ReloadConfig
                | Self::CreateCheckpoint
                | Self::BranchCheckpoint
                | Self::Quit
        )
    }
}

/// An entry in the live command palette and keybinding registry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandPaletteItem {
    pub id: &'static str,
    pub label: &'static str,
    pub keybind_hint: &'static str,
    pub category: &'static str,
    pub action: CommandPaletteAction,
}

/// Backwards compatibility alias for `CommandPaletteItem`.
pub type CommandPaletteEntry = CommandPaletteItem;

/// Canonical registry of all available command palette items.
///
/// Drives palette search, help overlay, keyboard shortcuts, and mouse selection.
#[must_use]
pub fn all_command_palette_items() -> Vec<CommandPaletteItem> {
    vec![
        // Playback
        CommandPaletteItem {
            id: "playback.toggle_pause",
            label: "Toggle Pause / Resume",
            keybind_hint: "Space",
            category: "Playback",
            action: CommandPaletteAction::TogglePause,
        },
        CommandPaletteItem {
            id: "playback.step",
            label: "Step Single Sim Tick",
            keybind_hint: "s",
            category: "Playback",
            action: CommandPaletteAction::StepOnce,
        },
        CommandPaletteItem {
            id: "playback.speed_up",
            label: "Faster Sim Speed",
            keybind_hint: "+",
            category: "Playback",
            action: CommandPaletteAction::SpeedUp,
        },
        CommandPaletteItem {
            id: "playback.speed_down",
            label: "Slower Sim Speed",
            keybind_hint: "-",
            category: "Playback",
            action: CommandPaletteAction::SpeedDown,
        },
        CommandPaletteItem {
            id: "playback.speed_1x",
            label: "Set Speed 1.0x (Normal)",
            keybind_hint: "1",
            category: "Playback",
            action: CommandPaletteAction::SetSpeed1x,
        },
        CommandPaletteItem {
            id: "playback.speed_2x",
            label: "Set Speed 2.0x (Fast)",
            keybind_hint: "2",
            category: "Playback",
            action: CommandPaletteAction::SetSpeed2x,
        },
        CommandPaletteItem {
            id: "playback.speed_4x",
            label: "Set Speed 4.0x (Very Fast)",
            keybind_hint: "4",
            category: "Playback",
            action: CommandPaletteAction::SetSpeed4x,
        },
        CommandPaletteItem {
            id: "playback.speed_max",
            label: "Set Speed Maximum (8.0x)",
            keybind_hint: "8",
            category: "Playback",
            action: CommandPaletteAction::SetSpeedMax,
        },
        // Scenario & Interventions
        CommandPaletteItem {
            id: "scenario.spawn_herbivore",
            label: "Spawn Herbivore Agent (Plant Eater)",
            keybind_hint: "H",
            category: "Scenario",
            action: CommandPaletteAction::SpawnHerbivore,
        },
        CommandPaletteItem {
            id: "scenario.spawn_carnivore",
            label: "Spawn Carnivore Agent (Predator)",
            keybind_hint: "C",
            category: "Scenario",
            action: CommandPaletteAction::SpawnCarnivore,
        },
        CommandPaletteItem {
            id: "scenario.trigger_drought",
            label: "Trigger Drought Intervention (Food Suppression)",
            keybind_hint: "D",
            category: "Scenario",
            action: CommandPaletteAction::TriggerDrought,
        },
        CommandPaletteItem {
            id: "scenario.reload_config",
            label: "Reload Default Simulation Configuration",
            keybind_hint: "R",
            category: "Scenario",
            action: CommandPaletteAction::ReloadConfig,
        },
        CommandPaletteItem {
            id: "scenario.reset_world",
            label: "Reset World Simulation State",
            keybind_hint: "",
            category: "Scenario",
            action: CommandPaletteAction::ResetWorld,
        },
        // Checkpoint & Export
        CommandPaletteItem {
            id: "export.checkpoint",
            label: "Create Simulation Checkpoint (Snapshot)",
            keybind_hint: "K",
            category: "Export",
            action: CommandPaletteAction::CreateCheckpoint,
        },
        CommandPaletteItem {
            id: "export.branch_checkpoint",
            label: "Branch Simulation from Checkpoint",
            keybind_hint: "N",
            category: "Export",
            action: CommandPaletteAction::BranchCheckpoint,
        },
        CommandPaletteItem {
            id: "export.ascii_screenshot",
            label: "Save ASCII / ANSI Frame Export",
            keybind_hint: "Shift+S",
            category: "Export",
            action: CommandPaletteAction::ExportAsciiScreenshot,
        },
        CommandPaletteItem {
            id: "export.data_bundle",
            label: "Export Science Data / Replay Bundle",
            keybind_hint: "X",
            category: "Export",
            action: CommandPaletteAction::ExportData,
        },
        // Navigation
        CommandPaletteItem {
            id: "nav.dashboard",
            label: "Navigate to Dashboard Screen",
            keybind_hint: "1 / D",
            category: "Navigation",
            action: CommandPaletteAction::NavigateDashboard,
        },
        CommandPaletteItem {
            id: "nav.world_canvas",
            label: "Navigate to World Canvas View",
            keybind_hint: "2 / W",
            category: "Navigation",
            action: CommandPaletteAction::NavigateWorld,
        },
        CommandPaletteItem {
            id: "nav.inspector",
            label: "Navigate to Agent Brain Inspector",
            keybind_hint: "3 / I",
            category: "Navigation",
            action: CommandPaletteAction::NavigateInspector,
        },
        CommandPaletteItem {
            id: "nav.lineages",
            label: "Navigate to Lineages & Phylogeny Screen",
            keybind_hint: "4 / L",
            category: "Navigation",
            action: CommandPaletteAction::NavigateLineages,
        },
        CommandPaletteItem {
            id: "nav.brain_arena",
            label: "Navigate to Brain Arena & Cohorts Screen",
            keybind_hint: "5 / B",
            category: "Navigation",
            action: CommandPaletteAction::NavigateBrainArena,
        },
        CommandPaletteItem {
            id: "nav.experiments",
            label: "Navigate to Experiments & Interventions Screen",
            keybind_hint: "6 / E",
            category: "Navigation",
            action: CommandPaletteAction::NavigateExperiments,
        },
        CommandPaletteItem {
            id: "nav.replay",
            label: "Navigate to Replay & Checkpoints Screen",
            keybind_hint: "7 / R",
            category: "Navigation",
            action: CommandPaletteAction::NavigateReplay,
        },
        CommandPaletteItem {
            id: "nav.environment",
            label: "Navigate to Environment & Biomes Screen",
            keybind_hint: "8 / V",
            category: "Navigation",
            action: CommandPaletteAction::NavigateEnvironment,
        },
        CommandPaletteItem {
            id: "nav.diagnostics",
            label: "Navigate to System Diagnostics Screen",
            keybind_hint: "9 / X",
            category: "Navigation",
            action: CommandPaletteAction::NavigateDiagnostics,
        },
        CommandPaletteItem {
            id: "nav.toggle_archipelago",
            label: "Toggle Tiled Archipelago View",
            keybind_hint: "a",
            category: "View",
            action: CommandPaletteAction::ToggleArchipelago,
        },
        CommandPaletteItem {
            id: "nav.toggle_rail",
            label: "Toggle Narrative Timeline Rail",
            keybind_hint: "r",
            category: "View",
            action: CommandPaletteAction::ToggleRail,
        },
        // Science Workflows
        CommandPaletteItem {
            id: "science.compare_experiments",
            label: "Compare Experiment Arms (Hedges' g)",
            keybind_hint: "C",
            category: "Science",
            action: CommandPaletteAction::CompareExperiments,
        },
        // Diagnostics & Science
        CommandPaletteItem {
            id: "science.focus_top_predator",
            label: "Focus Top Predator Agent",
            keybind_hint: "t",
            category: "Science",
            action: CommandPaletteAction::FocusTopPredator,
        },
        CommandPaletteItem {
            id: "science.focus_oldest",
            label: "Focus Oldest Living Agent",
            keybind_hint: "o",
            category: "Science",
            action: CommandPaletteAction::FocusOldest,
        },
        CommandPaletteItem {
            id: "science.toggle_probe",
            label: "Toggle Senses Attribution Probe",
            keybind_hint: "b",
            category: "Science",
            action: CommandPaletteAction::ToggleProbe,
        },
        CommandPaletteItem {
            id: "diag.toggle_diagnostics",
            label: "Toggle System Diagnostics Overlay",
            keybind_hint: "F12",
            category: "Diagnostics",
            action: CommandPaletteAction::ToggleDiagnostics,
        },
        CommandPaletteItem {
            id: "view.cycle_theme",
            label: "Cycle Curated Theme",
            keybind_hint: "Ctrl+T",
            category: "View",
            action: CommandPaletteAction::CycleTheme,
        },
        CommandPaletteItem {
            id: "view.cycle_palette",
            label: "Cycle Accessibility Palette",
            keybind_hint: "p / c",
            category: "View",
            action: CommandPaletteAction::CyclePalette,
        },
        CommandPaletteItem {
            id: "system.show_help",
            label: "Show Keybindings & Legend",
            keybind_hint: "?",
            category: "System",
            action: CommandPaletteAction::ShowHelp,
        },
        CommandPaletteItem {
            id: "system.quit",
            label: "Shutdown and Exit Simulation",
            keybind_hint: "q / Esc",
            category: "System",
            action: CommandPaletteAction::Quit,
        },
    ]
}

/// Pure fuzzy match against command palette items.
///
/// Matches against label, id, category, and keybind hint with prefix boost.
#[must_use]
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
            let id_lower = item.id.to_lowercase();
            let cat_lower = item.category.to_lowercase();
            let hint_lower = item.keybind_hint.to_lowercase();

            let matched = label_lower.contains(&query_lower)
                || id_lower.contains(&query_lower)
                || cat_lower.contains(&query_lower)
                || hint_lower.contains(&query_lower);

            if matched {
                let score = if label_lower.starts_with(&query_lower)
                    || id_lower.starts_with(&query_lower)
                {
                    0
                } else if label_lower.contains(&query_lower) {
                    1
                } else if cat_lower.starts_with(&query_lower) {
                    2
                } else {
                    3
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

/// Searchable command palette UI state and navigation model.
#[derive(Debug, Clone)]
pub struct CommandPalette {
    pub query: String,
    pub selected_index: usize,
    pub visible: bool,
    pub items: Vec<CommandPaletteItem>,
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandPalette {
    #[must_use]
    pub fn new() -> Self {
        Self {
            query: String::new(),
            selected_index: 0,
            visible: false,
            items: all_command_palette_items(),
        }
    }

    #[must_use]
    pub fn default_registry() -> Self {
        Self::new()
    }

    pub fn open(&mut self) {
        self.visible = true;
        self.query.clear();
        self.selected_index = 0;
    }

    pub fn close(&mut self) {
        self.visible = false;
        self.query.clear();
        self.selected_index = 0;
    }

    pub fn toggle(&mut self) {
        if self.visible {
            self.close();
        } else {
            self.open();
        }
    }

    pub fn input_char(&mut self, c: char) {
        self.query.push(c);
        self.selected_index = 0;
    }

    pub fn backspace(&mut self) {
        self.query.pop();
        self.selected_index = 0;
    }

    pub fn select_next(&mut self) {
        let count = self.filtered_items().len();
        if count > 0 {
            self.selected_index = (self.selected_index + 1) % count;
        }
    }

    pub fn select_previous(&mut self) {
        let count = self.filtered_items().len();
        if count > 0 {
            if self.selected_index == 0 {
                self.selected_index = count - 1;
            } else {
                self.selected_index -= 1;
            }
        }
    }

    #[must_use]
    pub fn filtered_items(&self) -> Vec<&CommandPaletteItem> {
        fuzzy_match_command_palette(&self.items, &self.query)
    }

    #[must_use]
    pub fn filtered_entries(&self) -> Vec<&CommandPaletteEntry> {
        self.filtered_items()
    }

    #[must_use]
    pub fn selected_item(&self) -> Option<&CommandPaletteItem> {
        let matched = self.filtered_items();
        matched.get(self.selected_index).copied()
    }

    /// Execution via direct ControlHandle for CLI/server fallback tests.
    pub fn execute_selected(
        &self,
        handle: &ControlHandle,
    ) -> Option<Result<CommandStatusDto, ControlError>> {
        let item = self.selected_item()?;
        match item.action {
            CommandPaletteAction::Pause => Some(handle.pause(None)),
            CommandPaletteAction::Resume => Some(handle.resume(None)),
            CommandPaletteAction::StepOnce => Some(handle.step()),
            CommandPaletteAction::SetSpeed1x => Some(handle.set_speed(1.0, None)),
            CommandPaletteAction::SetSpeed2x => Some(handle.set_speed(2.0, None)),
            CommandPaletteAction::Quit => Some(handle.shutdown()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_palette_filtering_and_navigation() {
        let mut palette = CommandPalette::new();
        assert_eq!(
            palette.filtered_items().len(),
            all_command_palette_items().len()
        );

        palette.query = "pause".into();
        let filtered = palette.filtered_items();
        assert!(!filtered.is_empty());
        assert!(
            filtered
                .iter()
                .any(|i| i.action == CommandPaletteAction::TogglePause)
        );

        palette.query = "speed".into();
        let speed_matches_count = palette.filtered_items().len();
        assert!(speed_matches_count >= 4);

        palette.select_next();
        assert_eq!(palette.selected_index, 1);

        palette.select_previous();
        assert_eq!(palette.selected_index, 0);

        palette.select_previous();
        assert_eq!(palette.selected_index, speed_matches_count - 1);
    }

    #[test]
    fn test_all_items_have_valid_identifiers_and_labels() {
        let items = all_command_palette_items();
        assert!(
            items.len() >= 20,
            "Registry must contain comprehensive controls"
        );

        let mut ids = std::collections::HashSet::new();
        for item in &items {
            assert!(!item.id.is_empty(), "Item ID must not be empty");
            assert!(!item.label.is_empty(), "Item label must not be empty");
            assert!(!item.category.is_empty(), "Item category must not be empty");
            assert!(ids.insert(item.id), "Duplicate item id: {}", item.id);
        }
    }

    #[test]
    fn test_mutating_actions_produce_valid_control_commands() {
        let speed = 2.0;
        let paused = false;

        assert!(matches!(
            CommandPaletteAction::Pause.to_control_command(speed, paused),
            Some(ControlCommand::Pause)
        ));
        assert!(matches!(
            CommandPaletteAction::Resume.to_control_command(speed, paused),
            Some(ControlCommand::Resume)
        ));
        assert!(matches!(
            CommandPaletteAction::StepOnce.to_control_command(speed, paused),
            Some(ControlCommand::Step)
        ));
        assert!(matches!(
            CommandPaletteAction::SetSpeed1x.to_control_command(speed, paused),
            Some(ControlCommand::SetSpeed(s)) if (s - 1.0).abs() < f32::EPSILON
        ));
        assert!(matches!(
            CommandPaletteAction::SpawnHerbivore.to_control_command(speed, paused),
            Some(ControlCommand::SpawnAgent { herbivore_tendency }) if (herbivore_tendency - 1.0).abs() < f32::EPSILON
        ));
        assert!(matches!(
            CommandPaletteAction::SpawnCarnivore.to_control_command(speed, paused),
            Some(ControlCommand::SpawnAgent { herbivore_tendency }) if herbivore_tendency.abs() < f32::EPSILON
        ));
        assert!(matches!(
            CommandPaletteAction::TriggerDrought.to_control_command(speed, paused),
            Some(ControlCommand::Intervention(_))
        ));
        assert!(matches!(
            CommandPaletteAction::Quit.to_control_command(speed, paused),
            Some(ControlCommand::Shutdown)
        ));

        // Non-mutating actions must return None
        assert!(
            CommandPaletteAction::CycleTheme
                .to_control_command(speed, paused)
                .is_none()
        );
        assert!(
            CommandPaletteAction::NavigateDashboard
                .to_control_command(speed, paused)
                .is_none()
        );
        assert!(
            CommandPaletteAction::ShowHelp
                .to_control_command(speed, paused)
                .is_none()
        );
    }

    #[test]
    fn test_palette_open_close_toggle_lifecycle() {
        let mut palette = CommandPalette::new();
        assert!(!palette.visible);

        palette.open();
        assert!(palette.visible);
        palette.input_char('t');
        palette.input_char('e');
        palette.input_char('s');
        palette.input_char('t');
        assert_eq!(palette.query, "test");

        palette.backspace();
        assert_eq!(palette.query, "tes");

        palette.close();
        assert!(!palette.visible);
        assert_eq!(palette.query, "");

        palette.toggle();
        assert!(palette.visible);
        palette.toggle();
        assert!(!palette.visible);
    }
}
