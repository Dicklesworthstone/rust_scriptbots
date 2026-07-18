//! Sub-cell painter core for the `FrankenTUI` world canvas (bd-2z0.14.2.1.1).
//!
//! The legacy terminal map draws one character per terminal cell; a dense
//! world becomes unreadable soup. This module is the pure compositor engine
//! behind the plan §8.4 canvas: braille 2x4 sub-cell density (8x resolution),
//! half-block 1x2 and quadrant 2x2 fallbacks, ASCII 1x1 last resort, layered
//! compositing, exact dirty-cell tracking, grow-only buffers, and color-depth
//! quantization. It is deliberately renderer-agnostic: the later canvas bead
//! (bd-2z0.14.2.1) wires world data in; tests drive it directly.
//!
//! Everything is deterministic — same paint calls produce byte-identical
//! frames — and nothing allocates per frame after warmup (buffers grow,
//! never shrink).

use std::collections::BTreeSet;

/// Which sub-cell glyph encoding the painter uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubCellMode {
    /// Unicode braille: 2 wide x 4 tall dots per cell (8x cell density).
    #[default]
    Braille,
    /// Upper/lower half blocks: 1 x 2 pixels per cell.
    HalfBlock,
    /// Quadrant blocks: 2 x 2 pixels per cell.
    Quadrant,
    /// Plain 1 x 1 fallback for strict terminals.
    Ascii,
}

impl SubCellMode {
    /// Sub-pixel columns per terminal cell.
    #[must_use]
    pub const fn dots_x(self) -> u16 {
        match self {
            Self::Braille | Self::Quadrant => 2,
            Self::HalfBlock | Self::Ascii => 1,
        }
    }

    /// Sub-pixel rows per terminal cell.
    #[must_use]
    pub const fn dots_y(self) -> u16 {
        match self {
            Self::Braille => 4,
            Self::HalfBlock | Self::Quadrant => 2,
            Self::Ascii => 1,
        }
    }
}

/// Paint layers, lowest to highest precedence.
///
/// A higher layer wins a sub-pixel only when its alpha is at least
/// [`ALPHA_SOLID`]; otherwise the lower layer shows through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    Terrain = 0,
    Water = 1,
    Food = 2,
    Agents = 3,
    Cues = 4,
    Selection = 5,
}

/// Alpha at or above which a sub-pixel counts as "set" for glyph packing and
/// foreground color.
pub const ALPHA_SOLID: f32 = 0.5;

/// One sub-pixel: a color plus the highest layer that painted it.
#[derive(Debug, Clone, Copy, PartialEq)]
struct SubPixel {
    rgba: [f32; 4],
    layer: Layer,
}

impl Default for SubPixel {
    fn default() -> Self {
        Self {
            rgba: [0.0, 0.0, 0.0, 0.0],
            layer: Layer::Terrain,
        }
    }
}

/// One composed terminal cell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameCell {
    /// Foreground color (glyph ink), linear 0..1.
    pub fg: [f32; 3],
    /// Background color, linear 0..1.
    pub bg: [f32; 3],
    /// The composed glyph.
    pub glyph: char,
}

/// A composed frame of terminal cells, row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct CellFrame {
    /// Terminal cell columns.
    pub width_cells: u16,
    /// Terminal cell rows.
    pub height_cells: u16,
    /// Row-major cells, `width_cells * height_cells` long.
    pub cells: Vec<FrameCell>,
}

/// Exact per-cell dirty tracking: a frame rebuild can repaint only cells
/// whose contents changed.
#[derive(Debug, Default)]
pub struct DirtyTracker {
    all: bool,
    cells: BTreeSet<(u16, u16)>,
}

impl DirtyTracker {
    fn mark_all(&mut self) {
        self.all = true;
        self.cells.clear();
    }

    fn mark_cell(&mut self, x: u16, y: u16) {
        if !self.all {
            self.cells.insert((x, y));
        }
    }

    /// Drain the dirty set. Returns `None` when nothing is dirty; `Some(None)`
    /// when everything is dirty; `Some(Some(cells))` for an exact cell list.
    pub fn take(&mut self) -> Option<Option<Vec<(u16, u16)>>> {
        if self.all {
            self.all = false;
            return Some(None);
        }
        if self.cells.is_empty() {
            return None;
        }
        Some(Some(std::mem::take(&mut self.cells).into_iter().collect()))
    }
}

/// The sub-cell painter: a layered float buffer sampled down to terminal
/// cells on [`SubCellBuffer::composite`].
#[derive(Debug)]
pub struct SubCellBuffer {
    width_cells: u16,
    height_cells: u16,
    mode: SubCellMode,
    sub_w: u16,
    sub_h: u16,
    pixels: Vec<SubPixel>,
    dirty: DirtyTracker,
}

impl SubCellBuffer {
    /// Create a painter for `width_cells x height_cells` terminal cells.
    #[must_use]
    pub fn new(width_cells: u16, height_cells: u16, mode: SubCellMode) -> Self {
        let sub_w = width_cells.saturating_mul(mode.dots_x());
        let sub_h = height_cells.saturating_mul(mode.dots_y());
        let mut buffer = Self {
            width_cells,
            height_cells,
            mode,
            sub_w,
            sub_h,
            pixels: Vec::new(),
            dirty: DirtyTracker::default(),
        };
        buffer
            .pixels
            .resize(usize::from(sub_w) * usize::from(sub_h), SubPixel::default());
        buffer.dirty.mark_all();
        buffer
    }

    /// Terminal cell columns.
    #[must_use]
    pub const fn width_cells(&self) -> u16 {
        self.width_cells
    }

    /// Terminal cell rows.
    #[must_use]
    pub const fn height_cells(&self) -> u16 {
        self.height_cells
    }

    /// Active glyph mode.
    #[must_use]
    pub const fn mode(&self) -> SubCellMode {
        self.mode
    }

    /// Sub-pixel width (for callers mapping world coordinates).
    #[must_use]
    pub const fn sub_width(&self) -> u16 {
        self.sub_w
    }

    /// Sub-pixel height.
    #[must_use]
    pub const fn sub_height(&self) -> u16 {
        self.sub_h
    }

    /// Current backing capacity (for the grow-only contract test).
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.pixels.capacity()
    }

    /// Resize the logical grid. The backing allocation only ever grows: a
    /// smaller logical size keeps the capacity (and stale pixels beyond the
    /// new extent are simply never read).
    pub fn resize(&mut self, width_cells: u16, height_cells: u16) {
        self.width_cells = width_cells;
        self.height_cells = height_cells;
        self.sub_w = width_cells.saturating_mul(self.mode.dots_x());
        self.sub_h = height_cells.saturating_mul(self.mode.dots_y());
        let needed = usize::from(self.sub_w) * usize::from(self.sub_h);
        if self.pixels.len() < needed {
            self.pixels.resize(needed, SubPixel::default());
        }
        self.dirty.mark_all();
    }

    fn index(&self, sub_x: u16, sub_y: u16) -> Option<usize> {
        if sub_x < self.sub_w && sub_y < self.sub_h {
            Some(usize::from(sub_y) * usize::from(self.sub_w) + usize::from(sub_x))
        } else {
            None
        }
    }

    fn sanitize(rgba: [f32; 4]) -> [f32; 4] {
        let clean = |v: f32| {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        [
            clean(rgba[0]),
            clean(rgba[1]),
            clean(rgba[2]),
            clean(rgba[3]),
        ]
    }

    /// Paint one sub-pixel on `layer` (out-of-bounds writes are ignored, so
    /// world->cell mapping clamping bugs cannot panic the frame loop).
    ///
    /// Precedence: a SOLID write (alpha > 0) replaces the pixel when its
    /// layer is at least the stored layer, or the stored pixel is empty.
    /// A fully transparent write is a no-op (erasing is [`Self::clear_layer`]'s
    /// job); replacement is destructive, which is safe because canvases
    /// repaint every frame from world data.
    pub fn set(&mut self, layer: Layer, sub_x: u16, sub_y: u16, rgba: [f32; 4]) {
        let Some(index) = self.index(sub_x, sub_y) else {
            return;
        };
        let rgba = Self::sanitize(rgba);
        if rgba[3] == 0.0 {
            return;
        }
        let pixel = &mut self.pixels[index];
        if layer >= pixel.layer || pixel.rgba[3] == 0.0 {
            *pixel = SubPixel { rgba, layer };
        }
        let dots_x = self.mode.dots_x();
        let dots_y = self.mode.dots_y();
        self.dirty.mark_cell(sub_x / dots_x, sub_y / dots_y);
    }

    /// Fill a sub-pixel rectangle (clamped to the buffer).
    pub fn fill_rect(&mut self, layer: Layer, x: u16, y: u16, w: u16, h: u16, rgba: [f32; 4]) {
        let rgba = Self::sanitize(rgba);
        let x1 = x.saturating_add(w).min(self.sub_w);
        let y1 = y.saturating_add(h).min(self.sub_h);
        let mut yy = y.min(self.sub_h);
        while yy < y1 {
            let mut xx = x.min(self.sub_w);
            while xx < x1 {
                self.set(layer, xx, yy, rgba);
                xx += 1;
            }
            yy += 1;
        }
    }

    /// Reset one layer back to transparent (higher layers keep their pixels;
    /// lower layers show through again). O(buffer) but touches only pixels
    /// tagged with the layer.
    pub fn clear_layer(&mut self, layer: Layer) {
        let sub_w = usize::from(self.sub_w);
        for (index, pixel) in self.pixels.iter_mut().enumerate() {
            if pixel.layer == layer {
                *pixel = SubPixel::default();
                // Index is bounded by the buffer dimensions, so the casts fit.
                let sub_x = u16::try_from(index % sub_w).unwrap_or(u16::MAX);
                let sub_y = u16::try_from(index / sub_w).unwrap_or(u16::MAX);
                self.dirty
                    .mark_cell(sub_x / self.mode.dots_x(), sub_y / self.mode.dots_y());
            }
        }
    }

    /// Drain the dirty set (see [`DirtyTracker::take`]).
    pub fn take_dirty(&mut self) -> Option<Option<Vec<(u16, u16)>>> {
        self.dirty.take()
    }

    /// Compose the buffer into terminal cells.
    ///
    /// Per mode: braille packs set-dots into the Unicode braille bit layout
    /// with fg = alpha-weighted mean of set sub-pixels and bg = mean of
    /// unset; half-block is fg=top/bg=bottom `▀`; quadrant picks the dominant
    /// pair by a max-contrast split; ASCII maps the top layer's color over a
    /// caller-supplied glyph table (or a shade ramp by default).
    #[must_use]
    pub fn composite(&self) -> CellFrame {
        self.composite_with_ascii(&ASCII_SHADE_RAMP)
    }

    /// [`Self::composite`] with a custom ASCII vocabulary (used by the
    /// emoji/narrow/ascii vocabulary tiers: the caller maps the top layer to
    /// a glyph; `glyphs` are tried from least to most "solid").
    #[must_use]
    pub fn composite_with_ascii(&self, ascii_glyphs: &[char]) -> CellFrame {
        let mut cells =
            Vec::with_capacity(usize::from(self.width_cells) * usize::from(self.height_cells));
        for cy in 0..self.height_cells {
            for cx in 0..self.width_cells {
                cells.push(self.compose_cell(cx, cy, ascii_glyphs));
            }
        }
        CellFrame {
            width_cells: self.width_cells,
            height_cells: self.height_cells,
            cells,
        }
    }

    fn pixel_at(&self, sub_x: u16, sub_y: u16) -> SubPixel {
        self.index(sub_x, sub_y)
            .map_or_else(SubPixel::default, |index| self.pixels[index])
    }

    fn compose_cell(&self, cx: u16, cy: u16, ascii_glyphs: &[char]) -> FrameCell {
        match self.mode {
            SubCellMode::Braille => self.compose_braille(cx, cy),
            SubCellMode::HalfBlock => self.compose_half_block(cx, cy),
            SubCellMode::Quadrant => self.compose_quadrant(cx, cy),
            SubCellMode::Ascii => self.compose_ascii(cx, cy, ascii_glyphs),
        }
    }

    fn compose_braille(&self, cx: u16, cy: u16) -> FrameCell {
        // Braille bit layout: (0,0)=0x01,(0,1)=0x02,(0,2)=0x04,(1,0)=0x08,
        // (1,1)=0x10,(1,2)=0x20,(0,3)=0x40,(1,3)=0x80.
        const BRAILLE_BITS: [[u8; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];
        let base_x = cx * 2;
        let base_y = cy * 4;
        let mut bits = 0u8;
        let mut set_rgb = [0.0_f32; 3];
        let mut set_weight = 0.0_f32;
        let mut unset_rgb = [0.0_f32; 3];
        let mut unset_count = 0.0_f32;
        for dy in 0..4_u16 {
            for dx in 0..2_u16 {
                let pixel = self.pixel_at(base_x + dx, base_y + dy);
                if pixel.rgba[3] >= ALPHA_SOLID {
                    bits |= BRAILLE_BITS[dy as usize][dx as usize];
                    set_rgb[0] = pixel.rgba[0].mul_add(pixel.rgba[3], set_rgb[0]);
                    set_rgb[1] = pixel.rgba[1].mul_add(pixel.rgba[3], set_rgb[1]);
                    set_rgb[2] = pixel.rgba[2].mul_add(pixel.rgba[3], set_rgb[2]);
                    set_weight += pixel.rgba[3];
                } else {
                    unset_rgb[0] += pixel.rgba[0];
                    unset_rgb[1] += pixel.rgba[1];
                    unset_rgb[2] += pixel.rgba[2];
                    unset_count += 1.0;
                }
            }
        }
        let fg = if set_weight > 0.0 {
            [
                set_rgb[0] / set_weight,
                set_rgb[1] / set_weight,
                set_rgb[2] / set_weight,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };
        let bg = if unset_count > 0.0 {
            [
                unset_rgb[0] / unset_count,
                unset_rgb[1] / unset_count,
                unset_rgb[2] / unset_count,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };
        FrameCell {
            fg,
            bg,
            glyph: char::from_u32(0x2800 + u32::from(bits)).unwrap_or('\u{2800}'),
        }
    }

    fn compose_half_block(&self, cx: u16, cy: u16) -> FrameCell {
        let top = self.pixel_at(cx, cy * 2);
        let bottom = self.pixel_at(cx, cy * 2 + 1);
        FrameCell {
            fg: [top.rgba[0], top.rgba[1], top.rgba[2]],
            bg: [bottom.rgba[0], bottom.rgba[1], bottom.rgba[2]],
            glyph: '\u{2580}', // upper half block
        }
    }

    fn compose_quadrant(&self, cx: u16, cy: u16) -> FrameCell {
        // Pixels: a=(0,0) b=(1,0) c=(0,1) d=(1,1). Choose the quadrant glyph
        // whose lit pattern best matches the alpha mask, then fg = mean of
        // lit pixels, bg = mean of unlit.
        let a = self.pixel_at(cx * 2, cy * 2);
        let b = self.pixel_at(cx * 2 + 1, cy * 2);
        let c = self.pixel_at(cx * 2, cy * 2 + 1);
        let d = self.pixel_at(cx * 2 + 1, cy * 2 + 1);
        let pixels = [a, b, c, d];
        // Bit per pixel in the standard quadrant order (tl=1, tr=2, bl=4, br=8).
        let mut mask = 0u8;
        for (index, pixel) in pixels.iter().enumerate() {
            if pixel.rgba[3] >= ALPHA_SOLID {
                mask |= 1 << index;
            }
        }
        let glyph = quadrant_glyph(mask);
        let mut fg = [0.0_f32; 3];
        let mut fg_n = 0.0_f32;
        let mut bg = [0.0_f32; 3];
        let mut bg_n = 0.0_f32;
        for (index, pixel) in pixels.iter().enumerate() {
            if mask & (1 << index) != 0 {
                fg[0] += pixel.rgba[0];
                fg[1] += pixel.rgba[1];
                fg[2] += pixel.rgba[2];
                fg_n += 1.0;
            } else {
                bg[0] += pixel.rgba[0];
                bg[1] += pixel.rgba[1];
                bg[2] += pixel.rgba[2];
                bg_n += 1.0;
            }
        }
        let scale = |v: [f32; 3], n: f32| {
            if n > 0.0 {
                [v[0] / n, v[1] / n, v[2] / n]
            } else {
                [0.0, 0.0, 0.0]
            }
        };
        FrameCell {
            fg: scale(fg, fg_n),
            bg: scale(bg, bg_n),
            glyph,
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn compose_ascii(&self, cx: u16, cy: u16, ascii_glyphs: &[char]) -> FrameCell {
        let pixel = self.pixel_at(cx, cy);
        let alpha = pixel.rgba[3];
        let ramp: &[char] = if ascii_glyphs.is_empty() {
            &ASCII_SHADE_RAMP
        } else {
            ascii_glyphs
        };
        let last = ramp.len().saturating_sub(1);
        // alpha is sanitized into [0, 1] at paint time and ramps are far below
        // f32's exact-integer ceiling, so every cast here is exact and
        // sign-safe; the allows above exist to say exactly that.
        let index = (alpha * last as f32).round() as usize;
        let glyph = *ramp.get(index.min(last)).unwrap_or(&' ');
        FrameCell {
            fg: [pixel.rgba[0], pixel.rgba[1], pixel.rgba[2]],
            bg: [0.0, 0.0, 0.0],
            glyph,
        }
    }
}

/// Default ASCII density ramp (least to most solid).
pub const ASCII_SHADE_RAMP: [char; 5] = [' ', '\u{2591}', '\u{2592}', '\u{2593}', '\u{2588}'];

const fn quadrant_glyph(mask: u8) -> char {
    // Bits: tl=1, tr=2, bl=4, br=8.
    match mask {
        0b0001 => '\u{2598}', // quadrant upper left
        0b0010 => '\u{259D}', // upper right
        0b0011 => '\u{2580}', // upper half
        0b0100 => '\u{2596}', // lower left
        0b0101 => '\u{258C}', // left half
        0b0110 => '\u{259E}', // diagonal tr+bl
        0b0111 => '\u{259B}', // all but br
        0b1000 => '\u{2597}', // lower right
        0b1001 => '\u{259A}', // diagonal tl+br
        0b1010 => '\u{2590}', // right half
        0b1011 => '\u{259C}', // all but bl
        0b1100 => '\u{2584}', // lower half
        0b1101 => '\u{2599}', // all but tr
        0b1110 => '\u{259F}', // all but tl
        0b1111 => '\u{2588}', // full block
        _ => ' ',
    }
}

/// Color depth for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorDepth {
    /// 24-bit truecolor (no quantization).
    TrueColor,
    /// xterm 256-color palette (6x6x6 cube + 24 greys).
    Ansi256,
    /// The 16 standard ANSI colors.
    Ansi16,
}

/// Quantize an RGB triple for a terminal color depth.
///
/// `TrueColor` is the identity. `Ansi256` uses the standard xterm mapping
/// (color cube index 16 + 36r + 6g + b, greys 232..255 when the color is
/// near-neutral), with optional deterministic Bayer 2x2 dithering handled by
/// the caller via [`bayer_threshold`]. `Ansi16` picks the nearest of the 16
/// standard colors by weighted euclidean distance.
///
/// Every input channel is clamped into `[0, 1]` first, so the casts below
/// always fit and are annotated at the one helper that performs them.
#[must_use]
pub fn quantize(rgb: [f32; 3], depth: ColorDepth) -> [u8; 3] {
    let bytes = [to_byte(rgb[0]), to_byte(rgb[1]), to_byte(rgb[2])];
    match depth {
        ColorDepth::TrueColor => bytes,
        ColorDepth::Ansi256 => ansi256_bytes(rgb),
        ColorDepth::Ansi16 => ansi16_bytes(rgb),
    }
}

/// Channel float in `[0, 1]` to a display byte (round-to-nearest).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
const fn to_byte(v: f32) -> u8 {
    // Clamped before scaling, so the cast always fits.
    v.clamp(0.0, 1.0).mul_add(255.0, 0.5) as u8
}

/// xterm 6x6x6 cube level for one channel (0..5).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn cube_level(v: f32) -> u8 {
    let scaled = v.clamp(0.0, 1.0).mul_add(5.0, 0.5) as u8;
    scaled.min(5)
}

/// The byte value of a cube level (the xterm ramp is 0, 95, 135, 175, 215, 255).
fn cube_byte(level: u8) -> u8 {
    const RAMP: [u8; 6] = [0, 95, 135, 175, 215, 255];
    RAMP[level.min(5) as usize]
}

fn ansi256_bytes(rgb: [f32; 3]) -> [u8; 3] {
    let mean = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
    let spread = (rgb[0] - mean).abs() + (rgb[1] - mean).abs() + (rgb[2] - mean).abs();
    if spread < 0.06 {
        // Near-neutral: use the 24-entry grey ramp (232..=255, 8..238 step 10).
        let grey = grey_ramp_index(mean);
        let v = 8 + grey * 10;
        return [v, v, v];
    }
    [
        cube_byte(cube_level(rgb[0])),
        cube_byte(cube_level(rgb[1])),
        cube_byte(cube_level(rgb[2])),
    ]
}

/// Map a near-neutral mean into the xterm grey-ramp index (0..=23).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn grey_ramp_index(mean: f32) -> u8 {
    let grey = (mean.clamp(0.0, 1.0).mul_add(255.0, -8.0) / 10.0 + 0.5) as i32;
    grey.clamp(0, 23) as u8
}

/// The 16 standard ANSI colors as RGB bytes.
const ANSI16_PALETTE: [[u8; 3]; 16] = [
    [0, 0, 0],
    [205, 0, 0],
    [0, 205, 0],
    [205, 205, 0],
    [0, 0, 238],
    [205, 0, 205],
    [0, 205, 205],
    [229, 229, 229],
    [127, 127, 127],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [92, 92, 255],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 255],
];

fn ansi16_bytes(rgb: [f32; 3]) -> [u8; 3] {
    let bytes = [
        i32::from(to_byte(rgb[0])),
        i32::from(to_byte(rgb[1])),
        i32::from(to_byte(rgb[2])),
    ];
    let mut best = ANSI16_PALETTE[0];
    let mut best_dist = i64::MAX;
    for candidate in ANSI16_PALETTE {
        let dr = i64::from(bytes[0] - i32::from(candidate[0]));
        let dg = i64::from(bytes[1] - i32::from(candidate[1]));
        let db = i64::from(bytes[2] - i32::from(candidate[2]));
        // Weight green highest (perceptual sensitivity), then red, then blue.
        let dist = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
        if dist < best_dist {
            best_dist = dist;
            best = candidate;
        }
    }
    best
}

/// Deterministic Bayer 2x2 dithering threshold in `[0, 1)` for callers that
/// want ordered dithering before quantizing to [`ColorDepth::Ansi256`].
#[must_use]
pub const fn bayer_threshold(cell_x: u16, cell_y: u16) -> f32 {
    const MATRIX: [[u8; 2]; 2] = [[0, 2], [3, 1]];
    (MATRIX[(cell_y % 2) as usize][(cell_x % 2) as usize] as f32 + 0.5) / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: f32, g: f32, b: f32) -> [f32; 4] {
        [r, g, b, 1.0]
    }

    #[test]
    fn braille_bit_packing_golden_vectors() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::Braille);
        // Each of the 8 dot positions maps to its documented bit.
        let cases: [((u16, u16), u32); 8] = [
            ((0, 0), 0x2801),
            ((0, 1), 0x2802),
            ((0, 2), 0x2804),
            ((0, 3), 0x2840),
            ((1, 0), 0x2808),
            ((1, 1), 0x2810),
            ((1, 2), 0x2820),
            ((1, 3), 0x2880),
        ];
        for ((x, y), expected) in cases {
            let mut b = SubCellBuffer::new(1, 1, SubCellMode::Braille);
            b.set(Layer::Agents, x, y, solid(1.0, 0.0, 0.0));
            let frame = b.composite();
            let cell = frame.cells[0];
            assert_eq!(
                cell.glyph,
                char::from_u32(expected).unwrap(),
                "dot ({x},{y})"
            );
            assert!(cell.fg[0] > 0.9, "dot color is the painted red");
        }
        // Full cell: all eight dots set.
        for y in 0..4 {
            for x in 0..2 {
                buffer.set(Layer::Agents, x, y, solid(0.0, 1.0, 0.0));
            }
        }
        let cell = buffer.composite().cells[0];
        assert_eq!(cell.glyph, '\u{28FF}');
    }

    #[test]
    fn layer_precedence_and_transparency() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::Braille);
        buffer.set(Layer::Terrain, 0, 0, solid(0.0, 0.5, 0.0));
        buffer.set(Layer::Agents, 0, 0, solid(1.0, 0.0, 0.0));
        let frame = buffer.composite();
        assert!(frame.cells[0].fg[0] > 0.9, "agents overpaint terrain");
        // A transparent higher-layer write is a no-op: terrain stays.
        let mut b2 = SubCellBuffer::new(1, 1, SubCellMode::Braille);
        b2.set(Layer::Terrain, 0, 0, solid(0.0, 0.5, 0.0));
        b2.set(Layer::Agents, 0, 0, [0.0, 0.0, 0.0, 0.0]);
        let frame2 = b2.composite();
        assert!(
            frame2.cells[0].fg[1] > 0.4,
            "transparent agent write never erases terrain: {:?}",
            frame2.cells[0].fg
        );
        // A lower layer cannot overpaint a higher one.
        buffer.set(Layer::Terrain, 0, 0, solid(0.0, 0.0, 1.0));
        let frame3 = buffer.composite();
        assert!(
            frame3.cells[0].fg[0] > 0.9,
            "terrain cannot displace the agent pixel"
        );
    }

    #[test]
    fn braille_fg_bg_split_on_two_color_cell() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::Braille);
        // Left column red painted; right column left empty (transparent).
        for y in 0..4 {
            buffer.set(Layer::Agents, 0, y, solid(1.0, 0.0, 0.0));
        }
        let cell = buffer.composite().cells[0];
        assert!(
            cell.fg[0] > 0.9 && cell.fg[2] < 0.1,
            "fg is red: {:?}",
            cell.fg
        );
        assert_eq!(
            cell.bg,
            [0.0, 0.0, 0.0],
            "empty sub-pixels read as black bg"
        );
        assert_eq!(
            cell.glyph,
            char::from_u32(0x2800 | 0x47).unwrap(),
            "only the left-column dots are set"
        );
        // Painting both columns yields a full block whose fg blends both colors
        // (the density compositor truth: every painted dot is set).
        for y in 0..4 {
            buffer.set(Layer::Terrain, 1, y, solid(0.0, 0.0, 1.0));
        }
        let cell = buffer.composite().cells[0];
        assert_eq!(cell.glyph, '\u{28FF}');
        assert!(
            (cell.fg[0] - 0.5).abs() < 0.05 && (cell.fg[2] - 0.5).abs() < 0.05,
            "fg averages painted dots: {:?}",
            cell.fg
        );
    }

    #[test]
    fn half_block_maps_top_to_fg_bottom_to_bg() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::HalfBlock);
        buffer.set(Layer::Terrain, 0, 0, solid(1.0, 1.0, 0.0));
        buffer.set(Layer::Water, 0, 1, solid(0.0, 0.0, 1.0));
        let cell = buffer.composite().cells[0];
        assert_eq!(cell.glyph, '\u{2580}');
        assert!(cell.fg[0] > 0.9 && cell.fg[1] > 0.9, "top pixel is fg");
        assert!(cell.bg[2] > 0.9, "bottom pixel is bg");
    }

    #[test]
    fn quadrant_glyphs_match_masks() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::Quadrant);
        buffer.set(Layer::Agents, 0, 0, solid(1.0, 0.0, 0.0)); // tl only
        assert_eq!(buffer.composite().cells[0].glyph, '\u{2598}');
        buffer.set(Layer::Agents, 1, 1, solid(1.0, 0.0, 0.0)); // tl + br diagonal
        assert_eq!(buffer.composite().cells[0].glyph, '\u{259A}');
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            buffer.set(Layer::Agents, x, y, solid(1.0, 0.0, 0.0));
        }
        assert_eq!(buffer.composite().cells[0].glyph, '\u{2588}');
    }

    #[test]
    fn ascii_ramp_tracks_alpha() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::Ascii);
        buffer.set(Layer::Terrain, 0, 0, [0.5, 0.5, 0.5, 0.0]);
        assert_eq!(buffer.composite().cells[0].glyph, ' ');
        buffer.set(Layer::Terrain, 0, 0, [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(buffer.composite().cells[0].glyph, '\u{2588}');
    }

    #[test]
    fn dirty_tracking_is_exact_per_mode() {
        for mode in [
            SubCellMode::Braille,
            SubCellMode::HalfBlock,
            SubCellMode::Quadrant,
            SubCellMode::Ascii,
        ] {
            let mut buffer = SubCellBuffer::new(8, 8, mode);
            let first = buffer.take_dirty();
            assert!(matches!(first, Some(None)), "new buffer starts fully dirty");
            buffer.set(Layer::Agents, 3, 2, solid(1.0, 0.0, 0.0));
            let dirty = buffer
                .take_dirty()
                .expect("one write dirties")
                .expect("exact cells");
            let dots_x = mode.dots_x();
            let dots_y = mode.dots_y();
            assert_eq!(dirty, vec![(3 / dots_x, 2 / dots_y)], "mode {mode:?}");
            assert!(buffer.take_dirty().is_none(), "dirty drains once");
        }
    }

    #[test]
    fn fill_rect_dirties_exact_span() {
        let mut buffer = SubCellBuffer::new(8, 4, SubCellMode::Braille);
        let _ = buffer.take_dirty();
        buffer.fill_rect(Layer::Food, 0, 4, 16, 2, solid(0.0, 1.0, 0.0));
        let dirty = buffer.take_dirty().expect("fill dirties").expect("exact");
        // Sub rows 4..6 (two rows: 4 and 5) lie entirely inside terminal row 1;
        // sub cols 0..16 cover terminal cols 0..8. Exact span = 8 cells.
        assert_eq!(dirty.len(), 8, "exact span: {dirty:?}");
        assert!(dirty.iter().all(|&(_x, y)| y == 1));
        assert!(dirty.contains(&(0, 1)) && dirty.contains(&(7, 1)));
    }

    #[test]
    fn resize_is_grow_only() {
        let mut buffer = SubCellBuffer::new(16, 16, SubCellMode::Braille);
        let big_capacity = buffer.capacity();
        buffer.resize(4, 4);
        assert_eq!(buffer.capacity(), big_capacity, "shrink keeps capacity");
        assert_eq!(buffer.sub_width(), 8);
        buffer.resize(64, 64);
        assert!(buffer.capacity() >= big_capacity, "grow may reallocate up");
    }

    #[test]
    fn clear_layer_allows_clean_repaint() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::Braille);
        buffer.set(Layer::Terrain, 0, 0, solid(0.0, 0.5, 0.0));
        buffer.set(Layer::Agents, 0, 0, solid(1.0, 0.0, 0.0));
        buffer.clear_layer(Layer::Agents);
        // Replacement is destructive, so the cleared pixel is now empty; the
        // per-frame canvas repaint restores the terrain underneath.
        buffer.set(Layer::Terrain, 0, 0, solid(0.0, 0.5, 0.0));
        let cell = buffer.composite().cells[0];
        assert!(
            cell.fg[1] > 0.4,
            "terrain repaints cleanly after agent clear"
        );
        assert!(cell.fg[0] < 0.1, "agent red is gone");
    }

    #[test]
    fn quantization_depths() {
        // TrueColor identity.
        assert_eq!(
            quantize([0.5, 0.25, 0.75], ColorDepth::TrueColor),
            [128, 64, 191]
        );
        // Ansi256 cube mapping: pure channels land on cube ramp bytes.
        assert_eq!(quantize([1.0, 0.0, 0.0], ColorDepth::Ansi256), [255, 0, 0]);
        assert_eq!(quantize([0.5, 0.5, 0.5], ColorDepth::Ansi256).len(), 3);
        // Near-neutral maps onto the grey ramp (all channels equal).
        let grey = quantize([0.4, 0.4, 0.4], ColorDepth::Ansi256);
        assert_eq!(grey[0], grey[1]);
        assert_eq!(grey[1], grey[2]);
        // Ansi16 nearest: pure red -> bright or dark red family.
        let red = quantize([1.0, 0.05, 0.05], ColorDepth::Ansi16);
        assert!(red[0] >= 205 && red[1] <= 10, "red stays red: {red:?}");
        let white = quantize([1.0, 1.0, 1.0], ColorDepth::Ansi16);
        assert_eq!(white, [255, 255, 255]);
        let black = quantize([0.0, 0.0, 0.0], ColorDepth::Ansi16);
        assert_eq!(black, [0, 0, 0]);
    }

    #[test]
    fn ansi256_cube_is_monotonic_per_channel() {
        // In the chromatic region the cube ramp never decreases.
        let mut previous = 0u8;
        for i in 2..=20 {
            let v = i as f32 / 20.0;
            let q = quantize([v, 0.0, 0.0], ColorDepth::Ansi256)[0];
            assert!(q >= previous, "cube channel never decreases at {v}");
            previous = q;
        }
        // Near-black values quantize to black or the darkest grey (both are
        // perceptually black; the grey-ramp transition is documented).
        let near_black = quantize([0.02, 0.02, 0.02], ColorDepth::Ansi256)[0];
        assert!(near_black <= 8, "near-black stays near-black: {near_black}");
    }

    #[test]
    fn compositing_is_deterministic() {
        let paint = |buffer: &mut SubCellBuffer| {
            for y in 0..8_u16 {
                for x in 0..16_u16 {
                    let layer = match (x + y) % 4 {
                        0 => Layer::Terrain,
                        1 => Layer::Water,
                        2 => Layer::Food,
                        _ => Layer::Agents,
                    };
                    buffer.set(
                        layer,
                        x,
                        y,
                        [x as f32 / 16.0, y as f32 / 8.0, (x + y) as f32 / 24.0, 0.9],
                    );
                }
            }
        };
        let mut a = SubCellBuffer::new(8, 2, SubCellMode::Braille);
        let mut b = SubCellBuffer::new(8, 2, SubCellMode::Braille);
        paint(&mut a);
        paint(&mut b);
        assert_eq!(
            a.composite(),
            b.composite(),
            "identical paint, identical frame"
        );
    }

    #[test]
    fn out_of_bounds_writes_never_panic() {
        let mut buffer = SubCellBuffer::new(2, 2, SubCellMode::Braille);
        buffer.set(Layer::Agents, u16::MAX, u16::MAX, solid(1.0, 0.0, 0.0));
        buffer.fill_rect(Layer::Food, 3, 3, 100, 100, solid(0.0, 1.0, 0.0));
        let _ = buffer.composite();
    }

    #[test]
    fn nan_inputs_sanitize_to_zero() {
        let mut buffer = SubCellBuffer::new(1, 1, SubCellMode::Braille);
        buffer.set(
            Layer::Agents,
            0,
            0,
            [f32::NAN, f32::INFINITY, -1.0, f32::NAN],
        );
        let cell = buffer.composite().cells[0];
        assert_eq!(cell.fg, [0.0, 0.0, 0.0]);
        assert_eq!(cell.glyph, '\u{2800}');
    }
}
