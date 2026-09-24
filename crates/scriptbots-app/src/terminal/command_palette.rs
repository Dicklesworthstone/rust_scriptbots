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

    // Export
    ExportAsciiScreenshot,

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
    CycleChartWindow,
    CycleEventFilter,
    FocusEventSubject,
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
        // Export
        CommandPaletteItem {
            id: "export.ascii_screenshot",
            label: "Save ASCII / ANSI Frame Export",
            keybind_hint: "Shift+S",
            category: "Export",
            action: CommandPaletteAction::ExportAsciiScreenshot,
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
            id: "science.cycle_chart_window",
            label: "Cycle Time-Series Chart Window (30/60/120/300t)",
            keybind_hint: "w",
            category: "Science",
            action: CommandPaletteAction::CycleChartWindow,
        },
        CommandPaletteItem {
            id: "science.cycle_event_filter",
            label: "Cycle Event Feed Filter (All/Birth/Death/Combat/Eat/Mutation/Config/Info)",
            keybind_hint: "f",
            category: "Science",
            action: CommandPaletteAction::CycleEventFilter,
        },
        CommandPaletteItem {
            id: "science.focus_event_subject",
            label: "Focus Selected Event Agent or Pan Location",
            keybind_hint: "Enter",
            category: "Science",
            action: CommandPaletteAction::FocusEventSubject,
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
/// Scored match result containing character indices for highlighting and recency status.
#[derive(Debug, Clone)]
pub struct ScoredPaletteMatch<'a> {
    pub item: &'a CommandPaletteItem,
    pub score: i32,
    pub matched_label_indices: Vec<usize>,
    pub matched_cat_indices: Vec<usize>,
    pub is_recent: bool,
}

/// Rich scored fuzzy match returning character positions for UI highlighting and recency boosts.
#[must_use]
pub fn fuzzy_match_command_palette_rich<'a>(
    items: &'a [CommandPaletteItem],
    query: &str,
    recent_actions: &[CommandPaletteAction],
) -> Vec<ScoredPaletteMatch<'a>> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        let mut results: Vec<ScoredPaletteMatch<'a>> = items
            .iter()
            .map(|item| {
                let recent_pos = recent_actions.iter().position(|&a| a == item.action);
                let (score, is_recent) = if let Some(pos) = recent_pos {
                    (1000 - pos as i32, true)
                } else {
                    (0, false)
                };
                ScoredPaletteMatch {
                    item,
                    score,
                    matched_label_indices: Vec::new(),
                    matched_cat_indices: Vec::new(),
                    is_recent,
                }
            })
            .collect();
        // Recent pinned first, then original order
        results.sort_by_key(|a| std::cmp::Reverse(a.score));
        return results;
    }

    let query_lower = trimmed.to_lowercase();
    let mut matches: Vec<ScoredPaletteMatch<'a>> = items
        .iter()
        .filter_map(|item| {
            let label_lower = item.label.to_lowercase();
            let id_lower = item.id.to_lowercase();
            let cat_lower = item.category.to_lowercase();
            let hint_lower = item.keybind_hint.to_lowercase();

            let is_recent = recent_actions.contains(&item.action);

            let mut matched_label_indices = Vec::new();
            let mut matched_cat_indices = Vec::new();
            let mut score = 0i32;

            // 1. Label match
            if let Some(pos) = label_lower.find(&query_lower) {
                for i in pos..(pos + query_lower.len()) {
                    matched_label_indices.push(i);
                }
                if pos == 0 {
                    score += 500; // Prefix match
                } else if label_lower
                    .as_bytes()
                    .get(pos.saturating_sub(1))
                    .is_some_and(|&b| b == b' ' || b == b'-' || b == b'_')
                {
                    score += 300; // Word boundary match
                } else {
                    score += 150; // Substring match
                }
            } else {
                // Try subsequence match on label
                let mut label_chars = label_lower.char_indices();
                let mut matched_all = true;
                let mut indices = Vec::new();
                for qc in query_lower.chars() {
                    let mut found = false;
                    for (idx, lc) in label_chars.by_ref() {
                        if qc == lc {
                            indices.push(idx);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        matched_all = false;
                        break;
                    }
                }
                if matched_all && !indices.is_empty() {
                    score += 80;
                    matched_label_indices = indices;
                }
            }

            // 2. Category match
            if let Some(pos) = cat_lower.find(&query_lower) {
                for i in pos..(pos + query_lower.len()) {
                    matched_cat_indices.push(i);
                }
                score += 50;
            }

            // 3. ID and keybind match
            if id_lower.contains(&query_lower) {
                score += 40;
            }
            if hint_lower.contains(&query_lower) {
                score += 60;
            }

            if is_recent {
                score += 200; // Recency boost
            }

            if score > 0 {
                Some(ScoredPaletteMatch {
                    item,
                    score,
                    matched_label_indices,
                    matched_cat_indices,
                    is_recent,
                })
            } else {
                None
            }
        })
        .collect();

    matches.sort_by_key(|a| std::cmp::Reverse(a.score));
    matches
}

/// Matches against label, id, category, and keybind hint with prefix boost.
#[must_use]
pub fn fuzzy_match_command_palette<'a>(
    items: &'a [CommandPaletteItem],
    query: &str,
) -> Vec<&'a CommandPaletteItem> {
    fuzzy_match_command_palette_rich(items, query, &[])
        .into_iter()
        .map(|m| m.item)
        .collect()
}

/// Searchable command palette UI state and navigation model.
#[derive(Debug, Clone)]
pub struct CommandPalette {
    pub query: String,
    pub selected_index: usize,
    pub visible: bool,
    pub items: Vec<CommandPaletteItem>,
    pub recent_actions: Vec<CommandPaletteAction>,
    pub last_receipt_summary: Option<String>,
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
            recent_actions: Vec::new(),
            last_receipt_summary: None,
        }
    }

    /// Record that an action was executed to rank it in recent history.
    pub fn record_action(&mut self, action: CommandPaletteAction) {
        self.recent_actions.retain(|&a| a != action);
        self.recent_actions.insert(0, action);
        self.recent_actions.truncate(5);
    }

    /// Record the latest acknowledged command receipt summary for the footer.
    pub fn set_receipt_summary(&mut self, summary: impl Into<String>) {
        self.last_receipt_summary = Some(summary.into());
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

        // No palette action may replace the live scientific config wholesale: the former
        // Reset/Reload entries submitted `UpdateConfig(default)` and silently clobbered it.
        for item in all_command_palette_items() {
            assert!(
                !matches!(
                    item.action.to_control_command(speed, paused),
                    Some(ControlCommand::UpdateConfig(_))
                ),
                "palette item {} would overwrite the live configuration",
                item.id
            );
        }

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

    #[test]
    fn test_fuzzy_match_rich_ranking_and_highlighting() {
        let items = all_command_palette_items();

        // 1. Prefix vs substring ranking
        let matches = fuzzy_match_command_palette_rich(&items, "pause", &[]);
        assert!(!matches.is_empty());
        // Toggle Pause or Pause should be top
        assert!(
            matches[0].item.action == CommandPaletteAction::Pause
                || matches[0].item.action == CommandPaletteAction::TogglePause
        );
        // Matching characters must be identified
        assert!(!matches[0].matched_label_indices.is_empty());

        // 2. Character highlighting indices are accurate
        let query = "spawn";
        let spawn_matches = fuzzy_match_command_palette_rich(&items, query, &[]);
        assert!(!spawn_matches.is_empty());
        for m in &spawn_matches {
            if m.item.label.to_lowercase().contains(query) {
                assert_eq!(m.matched_label_indices.len(), query.len());
                let label_chars: Vec<char> = m.item.label.to_lowercase().chars().collect();
                let matched_str: String = m
                    .matched_label_indices
                    .iter()
                    .map(|&idx| label_chars[idx])
                    .collect();
                assert_eq!(matched_str, query);
            }
        }
    }

    #[test]
    fn test_recent_pinning_and_ordering() {
        let items = all_command_palette_items();
        let mut palette = CommandPalette::new();

        palette.record_action(CommandPaletteAction::TriggerDrought);
        palette.record_action(CommandPaletteAction::SpawnCarnivore);

        assert_eq!(palette.recent_actions.len(), 2);
        assert_eq!(
            palette.recent_actions[0],
            CommandPaletteAction::SpawnCarnivore
        );
        assert_eq!(
            palette.recent_actions[1],
            CommandPaletteAction::TriggerDrought
        );

        // Empty query matches all, with recent pinned at top
        let matches = fuzzy_match_command_palette_rich(&items, "", &palette.recent_actions);
        assert!(matches.len() >= 2);
        assert!(matches[0].is_recent);
        assert_eq!(matches[0].item.action, CommandPaletteAction::SpawnCarnivore);
        assert!(matches[1].is_recent);
        assert_eq!(matches[1].item.action, CommandPaletteAction::TriggerDrought);
        assert!(!matches[2].is_recent);
    }
}
