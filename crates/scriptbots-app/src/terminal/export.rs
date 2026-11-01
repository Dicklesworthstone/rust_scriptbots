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

use ratatui::buffer::{Buffer, Cell, CellDiffOption};
use ratatui::style::{Color, Modifier};

/// Serialize the buffer as plain text, one line per row.
///
/// Each leading grapheme is emitted once; its covered cells are skipped, so wide
/// symbols do not shift later columns. Trailing blanks are kept rather than
/// trimmed, preserving the columns needed to locate a rendering fault.
#[must_use]
pub fn buffer_to_plain_text(buffer: &Buffer) -> String {
    let area = buffer.area;
    let mut out = String::with_capacity(usize::from(area.width) * usize::from(area.height));
    for y in area.y..area.bottom() {
        for cell in row_cells(buffer, y) {
            out.push_str(cell.symbol());
        }
        out.push('\n');
    }
    out
}

/// Visit grapheme-leading cells, skipping the columns covered by a wide glyph.
/// Ratatui resets those covered cells to spaces. Their symbols and styles must
/// not be emitted again: the leading grapheme already occupies their columns.
/// Use the same width policy as the live FrankenTUI presentation bridge.
fn row_cells(buffer: &Buffer, y: u16) -> impl Iterator<Item = &Cell> {
    let mut x = buffer.area.x;
    std::iter::from_fn(move || {
        if x >= buffer.area.right() {
            return None;
        }
        let cell = &buffer[(x, y)];
        let width = match cell.diff_option {
            CellDiffOption::ForcedWidth(width) => width.get(),
            _ => u16::try_from(ftui::core::text_width::display_width(cell.symbol()))
                .unwrap_or(u16::MAX),
        };
        x = x.saturating_add(width.max(1));
        Some(cell)
    })
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
        let mut active: Option<(Color, Color, Color, Modifier)> = None;
        for cell in row_cells(buffer, y) {
            let style = (cell.fg, cell.bg, cell.underline_color, cell.modifier);
            if active != Some(style) {
                out.push_str(RESET);
                out.push_str(&sgr_for(
                    cell.fg,
                    cell.bg,
                    cell.underline_color,
                    cell.modifier,
                ));
                active = Some(style);
            }
            out.push_str(cell.symbol());
        }
        out.push_str(RESET);
        out.push('\n');
    }
    out
}

/// Build the SGR sequence for one cell's style.
fn sgr_for(fg: Color, bg: Color, underline_color: Color, modifier: Modifier) -> String {
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
    // SGR's blink rates are mutually exclusive; rapid takes precedence when
    // both modifier bits are present.
    if modifier.contains(Modifier::RAPID_BLINK) {
        codes.push("6".into());
    } else if modifier.contains(Modifier::SLOW_BLINK) {
        codes.push("5".into());
    }
    if modifier.contains(Modifier::REVERSED) {
        codes.push("7".into());
    }
    if modifier.contains(Modifier::HIDDEN) {
        codes.push("8".into());
    }
    if modifier.contains(Modifier::CROSSED_OUT) {
        codes.push("9".into());
    }
    if let Some(code) = color_code(fg, true) {
        codes.push(code);
    }
    if let Some(code) = color_code(bg, false) {
        codes.push(code);
    }
    if let Some(code) = underline_color_code(underline_color) {
        codes.push(code);
    }
    if codes.is_empty() {
        String::new()
    } else {
        format!("\x1b[{}m", codes.join(";"))
    }
}

fn underline_color_code(color: Color) -> Option<String> {
    let index = match color {
        Color::Reset => return None,
        Color::Rgb(r, g, b) => return Some(format!("58;2;{r};{g};{b}")),
        Color::Indexed(index) => index,
        Color::Black => 0,
        Color::Red => 1,
        Color::Green => 2,
        Color::Yellow => 3,
        Color::Blue => 4,
        Color::Magenta => 5,
        Color::Cyan => 6,
        Color::Gray => 7,
        Color::DarkGray => 8,
        Color::LightRed => 9,
        Color::LightGreen => 10,
        Color::LightYellow => 11,
        Color::LightBlue => 12,
        Color::LightMagenta => 13,
        Color::LightCyan => 14,
        Color::White => 15,
    };
    Some(format!("58;5;{index}"))
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

    /// Explicit empty continuation symbols remain supported as well as the
    /// reset-to-space cells produced by Ratatui's actual string renderer.
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

    #[test]
    fn rendered_wide_glyphs_keep_their_displayed_columns_in_both_exports() {
        let mut buffer = Buffer::empty(Rect::new(3, 2, 10, 2));
        buffer.set_string(3, 2, "🌊xy界e\u{301} Z ", Style::default());
        buffer.set_string(3, 3, " A 界 B   ", Style::default());
        assert_eq!(buffer[(4, 2)].symbol(), " ");

        let expected = "🌊xy界e\u{301} Z \n A 界 B   \n";
        assert_eq!(buffer_to_plain_text(&buffer), expected);
        assert_eq!(strip_ansi(&buffer_to_ansi(&buffer)), expected);
    }

    #[test]
    fn ansi_preserves_all_displayed_modifiers_and_underline_color() {
        let mut buffer = buffer_with(1, 1);
        buffer[(0, 0)].set_char('X').set_style(
            Style::default()
                .add_modifier(Modifier::all())
                .underline_color(Color::Rgb(12, 34, 56)),
        );
        let ansi = buffer_to_ansi(&buffer);
        assert_eq!(ansi, "\x1b[0m\x1b[1;2;3;4;6;7;8;9;58;2;12;34;56mX\x1b[0m\n");
    }

    #[test]
    fn ansi_preserves_slow_blink_and_changes_in_underline_color() {
        let mut buffer = buffer_with(3, 1);
        for (x, color) in [Color::Red, Color::Indexed(200), Color::Reset]
            .into_iter()
            .enumerate()
        {
            buffer[(x as u16, 0)].set_char('X').set_style(
                Style::default()
                    .add_modifier(Modifier::UNDERLINED | Modifier::SLOW_BLINK)
                    .underline_color(color),
            );
        }
        assert_eq!(
            buffer_to_ansi(&buffer),
            "\x1b[0m\x1b[4;5;58;5;1mX\x1b[0m\x1b[4;5;58;5;200mX\x1b[0m\x1b[4;5mX\x1b[0m\n"
        );
    }

    #[test]
    fn wide_glyphs_clipped_by_the_renderer_keep_narrow_rows_blank() {
        let mut buffer = buffer_with(1, 2);
        buffer.set_string(0, 0, "🌊", Style::default());
        buffer.set_string(0, 1, "界", Style::default());
        assert_eq!(buffer_to_plain_text(&buffer), " \n \n");
        assert_eq!(strip_ansi(&buffer_to_ansi(&buffer)), " \n \n");
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
