//! Sub-cell painter core (bd-2z0.14.2.1.1).
//!
//! Pure terminal-graphics primitives for the high-resolution world canvas:
//! a grow-only float RGBA sub-pixel buffer, braille 2×4 dot packing (eight
//! sub-pixels per terminal cell), half-block and quadrant fallbacks, and
//! deterministic 24-bit → 256 → 16 color quantization with an optional
//! ordered-dither mode. No terminal I/O, no world types, no allocation on the
//! steady-state path — this module is the engine; the canvas migration
//! (bd-2z0.6.4) and mode selection own everything else.

/// One sub-pixel: linear RGB plus coverage/occupancy alpha in `[0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubPixel {
    /// Linear red in `[0, 1]`.
    pub r: f32,
    /// Linear green in `[0, 1]`.
    pub g: f32,
    /// Linear blue in `[0, 1]`.
    pub b: f32,
    /// Coverage: 0 = empty, 1 = fully lit. Braille packing thresholds this.
    pub a: f32,
}

impl SubPixel {
    /// Fully-specified sub-pixel.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

/// Sub-cell rendering mode: how many sub-pixels one terminal cell carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubCellMode {
    /// 2×4 braille dots: eight sub-pixels per cell, highest resolution.
    Braille,
    /// 1×2 via `▀`: upper sub-pixel is the foreground, lower the background.
    HalfBlock,
    /// 2×2 quadrant blocks.
    Quadrant,
}

impl SubCellMode {
    /// Sub-pixels per terminal cell as `(width, height)`.
    #[must_use]
    pub const fn cell_pixel_size(self) -> (usize, usize) {
        match self {
            Self::Braille => (2, 4),
            Self::HalfBlock => (1, 2),
            Self::Quadrant => (2, 2),
        }
    }
}

/// One composited terminal cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellGlyph {
    /// Character to draw.
    pub ch: char,
    /// Foreground color.
    pub fg: QuantizedColor,
    /// Background color.
    pub bg: QuantizedColor,
}

/// A cell invalidated since the previous composite, with its new content.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirtyCell {
    /// Terminal cell column.
    pub cell_x: u16,
    /// Terminal cell row.
    pub cell_y: u16,
    /// The cell's freshly composited content.
    pub glyph: CellGlyph,
}

/// Grow-only sub-pixel buffer.
///
/// `ensure_size` never shrinks and never reallocates when the requested area
/// already fits, so steady-state frames perform zero allocations; the
/// `grow_events` counter exists so tests can prove that claim instead of
/// assuming it. Every write marks its sub-pixel dirty, and
/// [`Self::composite_dirty`] emits exactly the terminal cells whose
/// sub-pixels changed — unchanged cells cost nothing.
#[derive(Debug, Default)]
pub struct PixelBuffer {
    width: usize,
    height: usize,
    pixels: Vec<SubPixel>,
    dirty: Vec<bool>,
    grow_events: usize,
}

impl PixelBuffer {
    /// Empty buffer; the first `ensure_size` performs the initial growth.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resize the logical area, growing the backing storage only when the
    /// requested area exceeds the current capacity. Contents are cleared to
    /// the default (empty) sub-pixel either way.
    pub fn ensure_size(&mut self, width: usize, height: usize) {
        let needed = width.saturating_mul(height);
        if needed > self.pixels.capacity() {
            self.pixels.reserve(needed - self.pixels.len());
            self.grow_events += 1;
        }
        self.pixels.clear();
        self.pixels.resize(needed, SubPixel::default());
        self.dirty.clear();
        // A resize invalidates the whole surface.
        self.dirty.resize(needed, true);
        self.width = width;
        self.height = height;
    }

    /// Logical width in sub-pixels.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Logical height in sub-pixels.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Times the backing storage actually grew. Steady-state rendering must
    /// keep this constant across frames.
    #[must_use]
    pub const fn grow_events(&self) -> usize {
        self.grow_events
    }

    /// Immutable sub-pixel access; `None` outside the logical area.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<&SubPixel> {
        (x < self.width && y < self.height).then(|| &self.pixels[y * self.width + x])
    }

    /// Mutable sub-pixel access; `None` outside the logical area.
    ///
    /// Marks the sub-pixel dirty eagerly: a caller holding `&mut` is assumed
    /// to write. The alternative — diffing on composite — would charge every
    /// unchanged cell for the comparison, which is the exact cost dirty
    /// tracking exists to remove.
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut SubPixel> {
        if x < self.width && y < self.height {
            let index = y * self.width + x;
            self.dirty[index] = true;
            Some(&mut self.pixels[index])
        } else {
            None
        }
    }

    /// Composite `source` over the sub-pixel at `(x, y)` using source-over
    /// alpha, the single blend rule every layer of the canvas uses. Layer
    /// precedence is therefore exactly paint order: terrain first, selection
    /// cues last.
    pub fn blend_over(&mut self, x: usize, y: usize, source: SubPixel) {
        if let Some(destination) = self.get_mut(x, y) {
            let src_a = source.a.clamp(0.0, 1.0);
            let inv = 1.0 - src_a;
            destination.r = source.r.mul_add(src_a, destination.r * inv);
            destination.g = source.g.mul_add(src_a, destination.g * inv);
            destination.b = source.b.mul_add(src_a, destination.b * inv);
            destination.a = src_a.mul_add(1.0, destination.a * inv).clamp(0.0, 1.0);
        }
    }
}

impl PixelBuffer {
    /// Invalidate the whole surface (first frame, palette change, mode
    /// change — anything that must repaint every cell).
    pub fn mark_all_dirty(&mut self) {
        self.dirty.clear();
        self.dirty.resize(self.pixels.len(), true);
    }

    /// Composite every terminal cell containing a dirty sub-pixel, append the
    /// results to `out` (cleared first), clear their dirtiness, and return
    /// how many cells were emitted. Unchanged cells are never visited beyond
    /// the dirty-bit scan — that is the whole contract.
    pub fn composite_dirty(
        &mut self,
        mode: SubCellMode,
        depth: ColorDepth,
        dither: DitherMode,
        out: &mut Vec<DirtyCell>,
    ) -> usize {
        out.clear();
        let (pixel_w, pixel_h) = mode.cell_pixel_size();
        if self.width == 0 || self.height == 0 {
            return 0;
        }
        let cells_x = self.width.div_ceil(pixel_w);
        let cells_y = self.height.div_ceil(pixel_h);
        for cell_y in 0..cells_y {
            for cell_x in 0..cells_x {
                let mut cell_dirty = false;
                'scan: for dy in 0..pixel_h {
                    for dx in 0..pixel_w {
                        let x = cell_x * pixel_w + dx;
                        let y = cell_y * pixel_h + dy;
                        if x < self.width && y < self.height && self.dirty[y * self.width + x] {
                            cell_dirty = true;
                            break 'scan;
                        }
                    }
                }
                if !cell_dirty {
                    continue;
                }
                for dy in 0..pixel_h {
                    for dx in 0..pixel_w {
                        let x = cell_x * pixel_w + dx;
                        let y = cell_y * pixel_h + dy;
                        if x < self.width && y < self.height {
                            self.dirty[y * self.width + x] = false;
                        }
                    }
                }
                let glyph = self.composite_cell(mode, depth, dither, cell_x, cell_y);
                out.push(DirtyCell {
                    cell_x: u16::try_from(cell_x).unwrap_or(u16::MAX),
                    cell_y: u16::try_from(cell_y).unwrap_or(u16::MAX),
                    glyph,
                });
            }
        }
        out.len()
    }

    fn sub_pixel_or_empty(&self, x: usize, y: usize) -> SubPixel {
        self.get(x, y).copied().unwrap_or_default()
    }

    fn composite_cell(
        &self,
        mode: SubCellMode,
        depth: ColorDepth,
        dither: DitherMode,
        cell_x: usize,
        cell_y: usize,
    ) -> CellGlyph {
        let (pixel_w, pixel_h) = mode.cell_pixel_size();
        let base_x = cell_x * pixel_w;
        let base_y = cell_y * pixel_h;

        if matches!(mode, SubCellMode::HalfBlock) {
            // Pure two-pixel color mapping: coverage does not gate it.
            let upper = self.sub_pixel_or_empty(base_x, base_y);
            let lower = self.sub_pixel_or_empty(base_x, base_y + 1);
            return CellGlyph {
                ch: HALF_BLOCK_CHAR,
                fg: quantize(upper, cell_x, cell_y, depth, dither),
                bg: quantize(lower, cell_x, cell_y, depth, dither),
            };
        }

        let mut lit_sum = [0.0_f32; 3];
        let mut lit_count = 0_u32;
        let mut unlit_sum = [0.0_f32; 3];
        let mut unlit_count = 0_u32;
        let mut lit = [[false; 2]; 4];
        for dy in 0..pixel_h {
            for dx in 0..pixel_w {
                let pixel = self.sub_pixel_or_empty(base_x + dx, base_y + dy);
                if pixel.a >= COVERAGE_THRESHOLD {
                    lit[dy][dx] = true;
                    lit_sum[0] += pixel.r;
                    lit_sum[1] += pixel.g;
                    lit_sum[2] += pixel.b;
                    lit_count += 1;
                } else {
                    unlit_sum[0] += pixel.r;
                    unlit_sum[1] += pixel.g;
                    unlit_sum[2] += pixel.b;
                    unlit_count += 1;
                }
            }
        }
        let average = |sum: [f32; 3], count: u32| -> SubPixel {
            if count == 0 {
                SubPixel::default()
            } else {
                let n = count as f32;
                SubPixel::new(sum[0] / n, sum[1] / n, sum[2] / n, 1.0)
            }
        };
        let fg = quantize(average(lit_sum, lit_count), cell_x, cell_y, depth, dither);
        let bg = quantize(
            average(unlit_sum, unlit_count),
            cell_x,
            cell_y,
            depth,
            dither,
        );
        let ch = match mode {
            SubCellMode::Braille => braille_char([lit[0], lit[1], lit[2], lit[3]]),
            SubCellMode::Quadrant => quadrant_char(lit[0][0], lit[0][1], lit[1][0], lit[1][1]),
            SubCellMode::HalfBlock => unreachable!("handled above"),
        };
        CellGlyph { ch, fg, bg }
    }
}

/// Coverage above which a sub-pixel lights its braille dot / block half.
pub const COVERAGE_THRESHOLD: f32 = 0.5;

/// Pack a 2×4 coverage block into its braille character.
///
/// `dots[y][x]` is row-major with `y` growing downward, matching the buffer.
/// Unicode braille bit layout (U+2800 base) is deliberately non-row-major:
/// dots 1-3 and 7 form the left column (bits 0x01, 0x02, 0x04, 0x40) and
/// dots 4-6 and 8 the right (0x08, 0x10, 0x20, 0x80). Re-deriving this from
/// intuition is the classic way to get mirrored output; the tests pin the
/// reference vectors.
#[must_use]
pub fn braille_char(dots: [[bool; 2]; 4]) -> char {
    const BITS: [[u32; 2]; 4] = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];
    let mut code = 0x2800_u32;
    for (row, bits) in dots.iter().zip(BITS.iter()) {
        for (lit, bit) in row.iter().zip(bits.iter()) {
            if *lit {
                code |= bit;
            }
        }
    }
    char::from_u32(code).expect("braille block is a valid unicode range")
}

/// Map a 2×2 coverage block to its quadrant character.
///
/// Index bit order: upper-left = 1, upper-right = 2, lower-left = 4,
/// lower-right = 8.
#[must_use]
pub const fn quadrant_char(
    upper_left: bool,
    upper_right: bool,
    lower_left: bool,
    lower_right: bool,
) -> char {
    const TABLE: [char; 16] = [
        ' ', '\u{2598}', '\u{259D}', '\u{2580}', // none, UL, UR, upper half
        '\u{2596}', '\u{258C}', '\u{259E}', '\u{259B}', // LL, left, anti-diag, no-LR
        '\u{2597}', '\u{259A}', '\u{2590}', '\u{259C}', // LR, diag, right, no-LL
        '\u{2584}', '\u{2599}', '\u{259F}', '\u{2588}', // lower half, no-UR, no-UL, full
    ];
    let index = (upper_left as usize)
        | ((upper_right as usize) << 1)
        | ((lower_left as usize) << 2)
        | ((lower_right as usize) << 3);
    TABLE[index]
}

/// Half-block pairing: one terminal cell shows two vertical sub-pixels via
/// `▀` with the upper color as foreground and the lower as background.
pub const HALF_BLOCK_CHAR: char = '\u{2580}';

/// A quantized terminal color at one of the supported depths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizedColor {
    /// Direct 24-bit color.
    True(u8, u8, u8),
    /// xterm-256 palette index.
    Indexed256(u8),
    /// Basic 16-color palette index (0..=15).
    Basic16(u8),
}

/// Target color depth for quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorDepth {
    /// Emit 24-bit color unchanged.
    TrueColor,
    /// Quantize to the xterm-256 palette (6×6×6 cube + gray ramp).
    Palette256,
    /// Quantize to the 16 basic ANSI colors.
    Basic16,
}

/// Optional ordered dithering applied before palette quantization.
///
/// Deterministic by construction: the 2×2 Bayer offset depends only on the
/// sub-pixel coordinate, never on state or randomness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitherMode {
    /// Straight nearest-color quantization.
    None,
    /// 2×2 Bayer ordered dithering.
    Ordered2x2,
}

const BAYER_2X2: [[f32; 2]; 2] = [[-0.375, 0.125], [0.375, -0.125]];

fn linear_to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// xterm-256 cube component levels.
const CUBE_LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];

fn nearest_cube_level(value: u8) -> (u8, u8) {
    let mut best_index = 0_u8;
    let mut best_distance = u16::MAX;
    for (index, level) in CUBE_LEVELS.iter().enumerate() {
        let distance = (i32::from(value) - i32::from(*level)).unsigned_abs() as u16;
        if distance < best_distance {
            best_distance = distance;
            best_index = index as u8;
        }
    }
    (best_index, CUBE_LEVELS[best_index as usize])
}

/// Quantize an sRGB byte triplet to the xterm-256 palette.
///
/// Chooses between the 6×6×6 color cube (indices 16..=231) and the 24-step
/// gray ramp (232..=255) by squared-error, exactly and deterministically.
#[must_use]
pub fn quantize_256(r: u8, g: u8, b: u8) -> u8 {
    let (ri, rl) = nearest_cube_level(r);
    let (gi, gl) = nearest_cube_level(g);
    let (bi, bl) = nearest_cube_level(b);
    let cube_index = 16 + 36 * u16::from(ri) + 6 * u16::from(gi) + u16::from(bi);
    let cube_error = squared_error(r, g, b, rl, gl, bl);

    // Gray ramp: 232 + n, level = 8 + 10n for n in 0..24.
    let gray = (u16::from(r) + u16::from(g) + u16::from(b)) / 3;
    let n = ((gray.saturating_sub(8)) + 5) / 10;
    let n = n.min(23);
    let gray_level = (8 + 10 * n) as u8;
    let gray_error = squared_error(r, g, b, gray_level, gray_level, gray_level);

    if gray_error < cube_error {
        (232 + n) as u8
    } else {
        cube_index as u8
    }
}

fn squared_error(r: u8, g: u8, b: u8, pr: u8, pg: u8, pb: u8) -> u32 {
    let dr = i32::from(r) - i32::from(pr);
    let dg = i32::from(g) - i32::from(pg);
    let db = i32::from(b) - i32::from(pb);
    (dr * dr + dg * dg + db * db) as u32
}

/// The standard 16-color palette used for `Basic16` quantization.
const BASIC_16: [(u8, u8, u8); 16] = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
];

/// Quantize an sRGB byte triplet to the basic 16-color palette by
/// squared-error, deterministically.
#[must_use]
pub fn quantize_16(r: u8, g: u8, b: u8) -> u8 {
    let mut best_index = 0_u8;
    let mut best_error = u32::MAX;
    for (index, (pr, pg, pb)) in BASIC_16.iter().enumerate() {
        let error = squared_error(r, g, b, *pr, *pg, *pb);
        if error < best_error {
            best_error = error;
            best_index = index as u8;
        }
    }
    best_index
}

/// Quantize one linear-RGB sub-pixel color for the requested depth, with the
/// requested dithering applied at the sub-pixel's coordinate.
#[must_use]
pub fn quantize(
    pixel: SubPixel,
    x: usize,
    y: usize,
    depth: ColorDepth,
    dither: DitherMode,
) -> QuantizedColor {
    let offset = match dither {
        DitherMode::None => 0.0,
        DitherMode::Ordered2x2 => BAYER_2X2[y % 2][x % 2] * (1.0 / 12.0),
    };
    let r = linear_to_byte(pixel.r + offset);
    let g = linear_to_byte(pixel.g + offset);
    let b = linear_to_byte(pixel.b + offset);
    match depth {
        ColorDepth::TrueColor => QuantizedColor::True(r, g, b),
        ColorDepth::Palette256 => QuantizedColor::Indexed256(quantize_256(r, g, b)),
        ColorDepth::Basic16 => QuantizedColor::Basic16(quantize_16(r, g, b)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braille_packing_matches_reference_vectors() {
        assert_eq!(braille_char([[false; 2]; 4]), '\u{2800}');
        assert_eq!(braille_char([[true; 2]; 4]), '\u{28FF}');
        // Single dots: unicode braille dot numbering, not row-major.
        assert_eq!(
            braille_char([[true, false], [false; 2], [false; 2], [false; 2]]),
            '\u{2801}' // dot 1 (upper-left)
        );
        assert_eq!(
            braille_char([[false, true], [false; 2], [false; 2], [false; 2]]),
            '\u{2808}' // dot 4 (upper-right)
        );
        assert_eq!(
            braille_char([[false; 2], [false; 2], [false; 2], [true, false]]),
            '\u{2840}' // dot 7 (lower-left)
        );
        assert_eq!(
            braille_char([[false; 2], [false; 2], [false; 2], [false, true]]),
            '\u{2880}' // dot 8 (lower-right)
        );
        // Left column fully lit: dots 1,2,3,7 = 0x47.
        assert_eq!(
            braille_char([[true, false], [true, false], [true, false], [true, false]]),
            '\u{2847}'
        );
    }

    #[test]
    fn quadrant_table_is_exhaustive_and_symmetric() {
        assert_eq!(quadrant_char(false, false, false, false), ' ');
        assert_eq!(quadrant_char(true, true, true, true), '\u{2588}');
        assert_eq!(quadrant_char(true, true, false, false), '\u{2580}');
        assert_eq!(quadrant_char(false, false, true, true), '\u{2584}');
        assert_eq!(quadrant_char(true, false, true, false), '\u{258C}');
        assert_eq!(quadrant_char(false, true, false, true), '\u{2590}');
        // All sixteen combinations produce distinct glyphs.
        let mut seen = std::collections::BTreeSet::new();
        for bits in 0_u8..16 {
            let glyph = quadrant_char(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
            assert!(seen.insert(glyph), "duplicate quadrant glyph {glyph:?}");
        }
    }

    #[test]
    fn quantization_is_deterministic_and_monotone_on_the_gray_axis() {
        for value in 0_u16..=255 {
            let byte = value as u8;
            assert_eq!(
                quantize_256(byte, byte, byte),
                quantize_256(byte, byte, byte),
                "256-quantization must be deterministic"
            );
        }
        // Gray-axis monotonicity: brighter input never maps to a darker
        // palette entry (comparing the represented gray levels).
        let represented = |index: u8| -> u16 {
            if (232..=255).contains(&index) {
                u16::from(8 + 10 * (u16::from(index) - 232) as u8)
            } else if (16..=231).contains(&index) {
                let cube = u16::from(index) - 16;
                let r = CUBE_LEVELS[(cube / 36) as usize];
                let g = CUBE_LEVELS[((cube / 6) % 6) as usize];
                let b = CUBE_LEVELS[(cube % 6) as usize];
                (u16::from(r) + u16::from(g) + u16::from(b)) / 3
            } else {
                let (r, g, b) = BASIC_16[index as usize];
                (u16::from(r) + u16::from(g) + u16::from(b)) / 3
            }
        };
        let mut previous = 0_u16;
        for value in 0_u16..=255 {
            let byte = value as u8;
            let level = represented(quantize_256(byte, byte, byte));
            assert!(
                level >= previous,
                "gray ramp went backwards at input {value}: {level} < {previous}"
            );
            previous = level;
        }

        // Extremes land exactly.
        assert_eq!(quantize_16(0, 0, 0), 0);
        assert_eq!(quantize_16(255, 255, 255), 15);
        assert_eq!(quantize_16(255, 0, 0), 9);
    }

    #[test]
    fn dithering_is_a_pure_function_of_coordinates() {
        let pixel = SubPixel::new(0.5, 0.5, 0.5, 1.0);
        for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 5)] {
            let first = quantize(pixel, x, y, ColorDepth::Palette256, DitherMode::Ordered2x2);
            let second = quantize(pixel, x, y, ColorDepth::Palette256, DitherMode::Ordered2x2);
            assert_eq!(first, second);
            // Tiling: the Bayer pattern repeats every 2 sub-pixels.
            let tiled = quantize(
                pixel,
                x + 2,
                y + 2,
                ColorDepth::Palette256,
                DitherMode::Ordered2x2,
            );
            assert_eq!(first, tiled);
        }
    }

    #[test]
    fn pixel_buffer_grows_once_and_reuses_capacity() {
        let mut buffer = PixelBuffer::new();
        buffer.ensure_size(80, 40);
        assert_eq!(buffer.grow_events(), 1);
        buffer.get_mut(3, 4).expect("in bounds").r = 1.0;

        // Same size and smaller sizes must not grow — and must clear.
        buffer.ensure_size(80, 40);
        assert_eq!(buffer.grow_events(), 1);
        assert_eq!(buffer.get(3, 4).expect("in bounds").r, 0.0);
        buffer.ensure_size(40, 20);
        assert_eq!(buffer.grow_events(), 1);
        assert_eq!(buffer.width(), 40);
        assert!(buffer.get(79, 39).is_none(), "logical bounds shrank");

        // Growing beyond capacity is exactly one more growth event.
        buffer.ensure_size(160, 80);
        assert_eq!(buffer.grow_events(), 2);
    }

    #[test]
    fn dirty_tracking_invalidates_exactly_the_touched_cells() {
        let mut buffer = PixelBuffer::new();
        let mut out = Vec::new();
        // 8x8 sub-pixels = 4x2 braille cells.
        buffer.ensure_size(8, 8);
        let full = buffer.composite_dirty(
            SubCellMode::Braille,
            ColorDepth::TrueColor,
            DitherMode::None,
            &mut out,
        );
        assert_eq!(full, 8, "a resize invalidates every cell");

        // No writes -> nothing dirty -> zero cost, zero cells.
        assert_eq!(
            buffer.composite_dirty(
                SubCellMode::Braille,
                ColorDepth::TrueColor,
                DitherMode::None,
                &mut out,
            ),
            0
        );

        // One sub-pixel write invalidates exactly its braille cell (1, 1).
        buffer.blend_over(3, 5, SubPixel::new(1.0, 1.0, 1.0, 1.0));
        let emitted = buffer.composite_dirty(
            SubCellMode::Braille,
            ColorDepth::TrueColor,
            DitherMode::None,
            &mut out,
        );
        assert_eq!(emitted, 1);
        assert_eq!((out[0].cell_x, out[0].cell_y), (1, 1));
        // Sub-pixel (3,5) inside cell (1,1) is local (1,1) => braille dot 5
        // (right column, second row): 0x2800 | 0x10.
        assert_eq!(out[0].glyph.ch, '\u{2810}');
        assert_eq!(out[0].glyph.fg, QuantizedColor::True(255, 255, 255));

        // And it composites clean afterwards.
        assert_eq!(
            buffer.composite_dirty(
                SubCellMode::Braille,
                ColorDepth::TrueColor,
                DitherMode::None,
                &mut out,
            ),
            0
        );
    }

    #[test]
    fn half_block_cells_map_upper_to_fg_and_lower_to_bg() {
        let mut buffer = PixelBuffer::new();
        buffer.ensure_size(1, 2);
        buffer.blend_over(0, 0, SubPixel::new(1.0, 0.0, 0.0, 1.0));
        buffer.blend_over(0, 1, SubPixel::new(0.0, 0.0, 1.0, 1.0));
        let mut out = Vec::new();
        let emitted = buffer.composite_dirty(
            SubCellMode::HalfBlock,
            ColorDepth::TrueColor,
            DitherMode::None,
            &mut out,
        );
        assert_eq!(emitted, 1);
        assert_eq!(out[0].glyph.ch, HALF_BLOCK_CHAR);
        assert_eq!(out[0].glyph.fg, QuantizedColor::True(255, 0, 0));
        assert_eq!(out[0].glyph.bg, QuantizedColor::True(0, 0, 255));
    }

    #[test]
    fn quadrant_cells_light_by_coverage_threshold() {
        let mut buffer = PixelBuffer::new();
        buffer.ensure_size(2, 2);
        buffer.blend_over(0, 0, SubPixel::new(0.0, 1.0, 0.0, 1.0));
        buffer.blend_over(1, 1, SubPixel::new(0.0, 1.0, 0.0, 0.4)); // below threshold
        let mut out = Vec::new();
        buffer.composite_dirty(
            SubCellMode::Quadrant,
            ColorDepth::TrueColor,
            DitherMode::None,
            &mut out,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].glyph.ch, '\u{2598}', "only the upper-left is lit");
    }

    #[test]
    fn blend_over_applies_source_over_in_paint_order() {
        let mut buffer = PixelBuffer::new();
        buffer.ensure_size(2, 2);
        // Opaque terrain, then translucent water over it.
        buffer.blend_over(0, 0, SubPixel::new(1.0, 0.0, 0.0, 1.0));
        buffer.blend_over(0, 0, SubPixel::new(0.0, 0.0, 1.0, 0.5));
        let pixel = *buffer.get(0, 0).expect("in bounds");
        assert!((pixel.r - 0.5).abs() < 1e-6);
        assert!((pixel.b - 0.5).abs() < 1e-6);
        assert!((pixel.a - 1.0).abs() < 1e-6);
        // A later fully-transparent layer changes nothing: precedence is
        // paint order, not layer identity.
        buffer.blend_over(0, 0, SubPixel::new(0.0, 1.0, 0.0, 0.0));
        assert_eq!(*buffer.get(0, 0).expect("in bounds"), pixel);
    }
}
