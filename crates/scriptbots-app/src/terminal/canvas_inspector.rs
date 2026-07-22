//! Responsive layered world canvas and agent inspector (bd-2z0.6.4).

use serde::{Deserialize, Serialize};

/// Breakpoint density modes for responsive layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutBreakpoint {
    Compact,  // < 80 cols
    Standard, // 80..120 cols
    Wide,     // > 120 cols
}

impl LayoutBreakpoint {
    pub fn from_cols(cols: u16) -> Self {
        if cols < 80 {
            Self::Compact
        } else if cols <= 120 {
            Self::Standard
        } else {
            Self::Wide
        }
    }
}

/// Responsive canvas and inspector viewport state.
#[derive(Debug, Clone)]
pub struct ResponsiveCanvasInspector {
    pub breakpoint: LayoutBreakpoint,
    pub zoom_level: f32,
    pub pan_offset: (f32, f32),
    pub selected_agent_id: Option<u64>,
    pub inspector_expanded: bool,
}

impl Default for ResponsiveCanvasInspector {
    fn default() -> Self {
        Self {
            breakpoint: LayoutBreakpoint::Standard,
            zoom_level: 1.0,
            pan_offset: (0.0, 0.0),
            selected_agent_id: None,
            inspector_expanded: false,
        }
    }
}

impl ResponsiveCanvasInspector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resize(&mut self, cols: u16) {
        self.breakpoint = LayoutBreakpoint::from_cols(cols);
    }

    pub fn zoom_in(&mut self) {
        self.zoom_level = (self.zoom_level * 1.2).min(8.0);
    }

    pub fn zoom_out(&mut self) {
        self.zoom_level = (self.zoom_level / 1.2).max(0.25);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.pan_offset.0 += dx;
        self.pan_offset.1 += dy;
    }

    pub fn select_agent(&mut self, agent_id: u64) {
        self.selected_agent_id = Some(agent_id);
        self.inspector_expanded = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_breakpoints() {
        assert_eq!(LayoutBreakpoint::from_cols(70), LayoutBreakpoint::Compact);
        assert_eq!(LayoutBreakpoint::from_cols(100), LayoutBreakpoint::Standard);
        assert_eq!(LayoutBreakpoint::from_cols(140), LayoutBreakpoint::Wide);
    }

    #[test]
    fn test_canvas_zoom_and_pan() {
        let mut canvas = ResponsiveCanvasInspector::new();
        assert_eq!(canvas.zoom_level, 1.0);

        canvas.zoom_in();
        assert!(canvas.zoom_level > 1.0);

        canvas.zoom_out();
        assert!(canvas.zoom_level <= 1.0);

        canvas.pan(10.0, -5.0);
        assert_eq!(canvas.pan_offset, (10.0, -5.0));

        canvas.select_agent(42);
        assert_eq!(canvas.selected_agent_id, Some(42));
        assert!(canvas.inspector_expanded);
    }
}
