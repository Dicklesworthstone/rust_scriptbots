//! Searchable command palette, key binding registry, and acknowledged controls (bd-2z0.6.5).

use crate::control::{CommandStatusDto, ControlError, ControlHandle};
use scriptbots_core::ControlCommand;
use serde::{Deserialize, Serialize};

/// Command entry in the command palette registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandPaletteEntry {
    pub id: String,
    pub label: String,
    pub category: String,
    pub key_shortcut: Option<String>,
    pub command: ControlCommand,
}

/// Searchable command palette model.
#[derive(Debug, Clone)]
pub struct CommandPalette {
    pub query: String,
    pub selected_index: usize,
    pub visible: bool,
    pub entries: Vec<CommandPaletteEntry>,
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::default_registry()
    }
}

impl CommandPalette {
    pub fn default_registry() -> Self {
        let entries = vec![
            CommandPaletteEntry {
                id: "pause".into(),
                label: "Pause Simulation".into(),
                category: "Control".into(),
                key_shortcut: Some("Space / p".into()),
                command: ControlCommand::Pause,
            },
            CommandPaletteEntry {
                id: "resume".into(),
                label: "Resume Simulation".into(),
                category: "Control".into(),
                key_shortcut: Some("Space / r".into()),
                command: ControlCommand::Resume,
            },
            CommandPaletteEntry {
                id: "step".into(),
                label: "Step 1 Tick".into(),
                category: "Control".into(),
                key_shortcut: Some("Period .".into()),
                command: ControlCommand::Step,
            },
            CommandPaletteEntry {
                id: "speed_1x".into(),
                label: "Set Speed 1.0x".into(),
                category: "Control".into(),
                key_shortcut: Some("1".into()),
                command: ControlCommand::SetSpeed(1.0),
            },
            CommandPaletteEntry {
                id: "speed_2x".into(),
                label: "Set Speed 2.0x".into(),
                category: "Control".into(),
                key_shortcut: Some("2".into()),
                command: ControlCommand::SetSpeed(2.0),
            },
            CommandPaletteEntry {
                id: "shutdown".into(),
                label: "Shutdown Simulation".into(),
                category: "System".into(),
                key_shortcut: Some("q / Esc".into()),
                command: ControlCommand::Shutdown,
            },
        ];

        Self {
            query: String::new(),
            selected_index: 0,
            visible: false,
            entries,
        }
    }

    pub fn filtered_entries(&self) -> Vec<&CommandPaletteEntry> {
        if self.query.trim().is_empty() {
            self.entries.iter().collect()
        } else {
            let q = self.query.to_lowercase();
            self.entries
                .iter()
                .filter(|e| e.label.to_lowercase().contains(&q) || e.category.to_lowercase().contains(&q))
                .collect()
        }
    }

    pub fn select_next(&mut self) {
        let count = self.filtered_entries().len();
        if count > 0 {
            self.selected_index = (self.selected_index + 1) % count;
        }
    }

    pub fn select_previous(&mut self) {
        let count = self.filtered_entries().len();
        if count > 0 {
            if self.selected_index == 0 {
                self.selected_index = count - 1;
            } else {
                self.selected_index -= 1;
            }
        }
    }

    pub fn execute_selected(&self, handle: &ControlHandle) -> Option<Result<CommandStatusDto, ControlError>> {
        let filtered = self.filtered_entries();
        if let Some(entry) = filtered.get(self.selected_index) {
            match entry.command {
                ControlCommand::Pause => Some(handle.pause()),
                ControlCommand::Resume => Some(handle.resume()),
                ControlCommand::Step => Some(handle.step()),
                ControlCommand::SetSpeed(s) => Some(handle.set_speed(s)),
                ControlCommand::Shutdown => Some(handle.shutdown()),
                _ => None,
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_palette_filtering_and_navigation() {
        let mut palette = CommandPalette::default_registry();
        assert_eq!(palette.filtered_entries().len(), 6);

        palette.query = "pause".into();
        assert_eq!(palette.filtered_entries().len(), 1);
        assert_eq!(palette.filtered_entries()[0].id, "pause");

        palette.query = "speed".into();
        assert_eq!(palette.filtered_entries().len(), 2);

        palette.select_next();
        assert_eq!(palette.selected_index, 1);

        palette.select_previous();
        assert_eq!(palette.selected_index, 0);
    }
}
