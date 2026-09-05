//! True-buffer export: serialize the EXACT rendered frame (bd-2z0.14.2.6).
//!
//! # What this replaces, and why it counted as a defect
//!
//! The `S` action used to call `save_ascii_snapshot`, which re-rasterized terrain
//! and food into a fixed 64x32 ASCII grid. It shared nothing with the rendered
//! frame: no agents, no panels, no colors, no sub-cell canvas, and it ignored
//! zoom and the viewport entirely. Two worlds that looked completely different on
//! screen could export identical files, and a frame with a broken widget exported
//! clean. It was an artifact that described a *different renderer* than the one
//! the user was looking at.
//!
//! Everything here serializes the real `Buffer` instead, so what you see is what
//! saves — including every panel, because the buffer is the whole screen.
//!
//! # No hash lives here, deliberately
//!
//! The headless evidence contract already owns an FNV-1a64 over every cell
//! (`HeadlessBufferEvidence`). Export reuses it rather than defining a second
//! one, so the exported frame and the headless report cannot disagree about what
//! a frame's identity is. A module that duplicated that hash would be the same
//! defect bd-c1z8 was filed for, one directory over.

use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};

/// Serialize the buffer as plain text, one line per row.
///
/// Deterministic and lossless in symbols: every cell contributes its grapheme
/// exactly once, so the row count is the buffer height and a row's visual width
/// is the buffer width. Trailing blanks are kept rather than trimmed — a trimmed
/// row would silently change the column a reader counts to, which defeats using
/// the export to locate a rendering fault.
#[must_use]
pub fn buffer_to_plain_text(buffer: &Buffer) -> String {
    let area = buffer.area;
    let mut out = String::with_capacity(usize::from(area.width) * usize::from(area.height));
    for y in area.y..area.bottom() {
        for x in area.x..area.right() {
            out.push_str(cell_symbol(buffer, x, y));
        }
        out.push('\n');
    }
    out
}

/// A cell's grapheme, normalizing the empty continuation cell of a wide glyph.
///
/// Ratatui stores a double-width glyph in its first cell and leaves the second
/// with an EMPTY symbol — not a space. Emitting that empty string verbatim would
/// shorten the line by one column and misalign everything after it, which is
/// exactly the misreading the emoji vocabulary is prone to.
fn cell_symbol(buffer: &Buffer, x: u16, y: u16) -> &str {
    let symbol = buffer[(x, y)].symbol();
    if symbol.is_empty() { "" } else { symbol }
}

/// Serialize the buffer as ANSI, preserving per-cell foreground, background, and
/// the modifiers a terminal can round-trip.
///
/// Style is emitted only when it CHANGES between adjacent cells, and every line
/// is terminated with a reset. That keeps the output small, keeps a line's colors
/// from bleeding into the next, and means a diff between two exports shows the
/// cells that actually differ rather than a wall of identical escapes.
#[must_use]
pub fn buffer_to_ansi(buffer: &Buffer) -> String {
    const RESET: &str = "\x1b[0m";
    let area = buffer.area;
    let mut out = String::new();
    for y in area.y..area.bottom() {
        let mut active: Option<(Color, Color, Modifier)> = None;
        for x in area.x..area.right() {
            let cell = &buffer[(x, y)];
            let style = (cell.fg, cell.bg, cell.modifier);
            if active != Some(style) {
                out.push_str(RESET);
                out.push_str(&sgr_for(cell.fg, cell.bg, cell.modifier));
                active = Some(style);
            }
            out.push_str(cell_symbol(buffer, x, y));
        }
        out.push_str(RESET);
        out.push('\n');
    }
    out
}

/// Build the SGR sequence for one cell's style.
fn sgr_for(fg: Color, bg: Color, modifier: Modifier) -> String {
    let mut codes: Vec<String> = Vec::new();
    if modifier.contains(Modifier::BOLD) {
        codes.push("1".into());
    }
    if modifier.contains(Modifier::DIM) {
        codes.push("2".into());
    }
    if modifier.contains(Modifier::ITALIC) {
        codes.push("3".into());
    }
    if modifier.contains(Modifier::UNDERLINED) {
        codes.push("4".into());
    }
    if modifier.contains(Modifier::REVERSED) {
        codes.push("7".into());
    }
    if let Some(code) = color_code(fg, true) {
        codes.push(code);
    }
    if let Some(code) = color_code(bg, false) {
        codes.push(code);
    }
    if codes.is_empty() {
        String::new()
    } else {
        format!("\x1b[{}m", codes.join(";"))
    }
}

/// SGR parameter for one color, or `None` for `Reset` (the terminal default).
///
/// `Rgb` is emitted as truecolor so a 24-bit canvas exports losslessly; indexed
/// and named colors are emitted in their own encodings rather than being
/// approximated into RGB, so a 256-color or 16-color frame exports as the tier it
/// was actually rendered at. The export documents the degradation instead of
/// hiding it.
fn color_code(color: Color, foreground: bool) -> Option<String> {
    let base = if foreground { 30 } else { 40 };
    let extended = if foreground { 38 } else { 48 };
    let bright_base = if foreground { 90 } else { 100 };
    let code = match color {
        Color::Reset => return None,
        Color::Black => (base).to_string(),
        Color::Red => (base + 1).to_string(),
        Color::Green => (base + 2).to_string(),
        Color::Yellow => (base + 3).to_string(),
        Color::Blue => (base + 4).to_string(),
        Color::Magenta => (base + 5).to_string(),
        Color::Cyan => (base + 6).to_string(),
        Color::Gray => (base + 7).to_string(),
        Color::DarkGray => (bright_base).to_string(),
        Color::LightRed => (bright_base + 1).to_string(),
        Color::LightGreen => (bright_base + 2).to_string(),
        Color::LightYellow => (bright_base + 3).to_string(),
        Color::LightBlue => (bright_base + 4).to_string(),
        Color::LightMagenta => (bright_base + 5).to_string(),
        Color::LightCyan => (bright_base + 6).to_string(),
        Color::White => (bright_base + 7).to_string(),
        Color::Rgb(r, g, b) => format!("{extended};2;{r};{g};{b}"),
        Color::Indexed(index) => format!("{extended};5;{index}"),
    };
    Some(code)
}

/// Strip ANSI SGR sequences, recovering the plain text an ANSI export carries.
///
/// Exists so a test can prove the two exports describe the SAME frame rather than
/// trusting that they do; a colored export whose glyphs drifted from the plain
/// one would be the original defect wearing a nicer format.
#[must_use]
pub fn strip_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch != '\x1b' {
            out.push(ch);
            continue;
        }
        // CSI ... m — consume through the terminating byte.
        if chars.next() != Some('[') {
            continue;
        }
        for tail in chars.by_ref() {
            if tail.is_ascii_alphabetic() {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::layout::Rect;
    use ratatui::style::Style;

    fn buffer_with(width: u16, height: u16) -> Buffer {
        Buffer::empty(Rect::new(0, 0, width, height))
    }

    /// The export must have one line per buffer row and one column per buffer
    /// column. The old 64x32 re-sample had neither, which is why it could not be
    /// used to locate a fault by coordinate.
    #[test]
    fn plain_text_preserves_the_buffer_geometry() {
        let mut buffer = buffer_with(6, 3);
        buffer[(0, 0)].set_char('a');
        buffer[(5, 2)].set_char('z');

        let text = buffer_to_plain_text(&buffer);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "one line per row");
        for (row, line) in lines.iter().enumerate() {
            assert_eq!(
                line.chars().count(),
                6,
                "row {row} must keep all six columns including trailing blanks: {line:?}"
            );
        }
        assert!(lines[0].starts_with('a'));
        assert!(lines[2].ends_with('z'));
    }

    /// Every cell reaches the export, including panels far from the map. The old
    /// exporter wrote only terrain and food, so a broken panel exported clean.
    #[test]
    fn plain_text_includes_every_region_of_the_frame() {
        let mut buffer = buffer_with(10, 4);
        buffer[(0, 0)].set_char('T'); // top-left
        buffer[(9, 0)].set_char('R'); // top-right, where the rail lives
        buffer[(0, 3)].set_char('B'); // bottom-left
        buffer[(9, 3)].set_char('C'); // bottom-right corner

        let text = buffer_to_plain_text(&buffer);
        for marker in ['T', 'R', 'B', 'C'] {
            assert!(
                text.contains(marker),
                "{marker} is in the buffer but missing from the export"
            );
        }
    }

    /// A double-width glyph occupies two cells, the second holding an EMPTY
    /// symbol. Emitting that verbatim keeps the line's visual width correct;
    /// emitting a space instead would widen the row by one column per emoji.
    #[test]
    fn wide_glyph_continuation_cells_do_not_widen_the_row() {
        let mut buffer = buffer_with(4, 1);
        buffer[(0, 0)].set_symbol("🌊");
        buffer[(1, 0)].set_symbol("");
        buffer[(2, 0)].set_char('x');
        buffer[(3, 0)].set_char('y');

        let text = buffer_to_plain_text(&buffer);
        let line = text.lines().next().expect("one row");
        assert_eq!(
            line, "🌊xy",
            "the continuation cell must contribute nothing"
        );
        assert_eq!(
            line.chars().count(),
            3,
            "one wide glyph plus two narrow ones is three graphemes across four columns"
        );
    }

    /// The two formats must describe the same frame. If they ever diverge, the
    /// pretty one is lying, which is the defect this module replaces.
    #[test]
    fn ansi_and_plain_exports_agree_on_the_glyphs() {
        let mut buffer = buffer_with(8, 3);
        for (index, ch) in "hello".chars().enumerate() {
            let cell = &mut buffer[(index as u16, 1)];
            cell.set_char(ch);
            cell.set_style(Style::default().fg(Color::Rgb(12, 34, 56)));
        }
        buffer[(0, 2)].set_style(Style::default().bg(Color::Indexed(200)));

        let plain = buffer_to_plain_text(&buffer);
        let recovered = strip_ansi(&buffer_to_ansi(&buffer));
        assert_eq!(
            recovered, plain,
            "stripping color from the ANSI export must yield the plain export"
        );
    }

    /// Truecolor must survive as truecolor, and an indexed frame must export as
    /// indexed. Approximating one into the other would make the golden set
    /// misreport which capability tier actually rendered.
    #[test]
    fn ansi_preserves_the_color_tier_each_cell_was_rendered_at() {
        let mut buffer = buffer_with(3, 1);
        buffer[(0, 0)].set_style(Style::default().fg(Color::Rgb(10, 20, 30)));
        buffer[(1, 0)].set_style(Style::default().fg(Color::Indexed(99)));
        buffer[(2, 0)].set_style(Style::default().fg(Color::Red));

        let ansi = buffer_to_ansi(&buffer);
        assert!(ansi.contains("38;2;10;20;30"), "truecolor stays truecolor");
        assert!(ansi.contains("38;5;99"), "indexed stays indexed");
        assert!(ansi.contains("\x1b[31m"), "a named color stays named");
    }

    /// Modifiers are part of what the user sees; an export that dropped them
    /// would show emphasis-free text for a frame that had emphasis.
    #[test]
    fn ansi_carries_modifiers() {
        let mut buffer = buffer_with(1, 1);
        buffer[(0, 0)].set_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED),
        );
        let ansi = buffer_to_ansi(&buffer);
        assert!(ansi.contains('1'), "bold");
        assert!(ansi.contains('4'), "underline");
    }

    /// Same buffer, same bytes — an export used as evidence must not vary run to
    /// run, or a golden comparison means nothing.
    #[test]
    fn exports_are_deterministic() {
        let mut buffer = buffer_with(5, 2);
        buffer[(2, 1)].set_char('q');
        buffer[(2, 1)].set_style(Style::default().fg(Color::Rgb(1, 2, 3)));

        assert_eq!(buffer_to_plain_text(&buffer), buffer_to_plain_text(&buffer));
        assert_eq!(buffer_to_ansi(&buffer), buffer_to_ansi(&buffer));
    }

    /// An empty frame must export as blank rows, not as nothing. A zero-byte file
    /// is indistinguishable from a failed write.
    #[test]
    fn an_empty_frame_still_exports_its_geometry() {
        let text = buffer_to_plain_text(&buffer_with(4, 2));
        assert_eq!(text.lines().count(), 2);
        assert!(text.lines().all(|line| line.chars().count() == 4));
    }
}
