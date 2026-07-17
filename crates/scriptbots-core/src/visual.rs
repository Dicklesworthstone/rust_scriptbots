//! Renderer-neutral visual semantics (bd-2z0.14.3.2).
//!
//! One implementation of every "what should this look like" decision, consumed
//! identically by the Bevy 3D frontend, the FrankenTUI terminal canvas, the
//! GPUI frontend while it lives, and the wgpu capture lane. Frontends may
//! RENDER differently (PBR lighting vs braille cells); they must not DECIDE
//! differently. Before this module, GPUI, Bevy, and the terminal each
//! hand-rolled their own agent colors, terrain brightness, and accessibility
//! transforms — three divergent answers to one question.
//!
//! Everything here is pure and deterministic: same inputs produce the same
//! outputs on every platform, with no wall-clock reads, no allocation on
//! per-frame paths, and no iteration-order dependence. Animation phases derive
//! from the simulation tick, never from a system clock, so replays and
//! cross-renderer equivalence tests see identical frames.
//!
//! # Layering contract
//!
//! Semantic functions return colors in the *natural* palette. Frontends apply
//! [`apply_accessibility_palette`] as the FINAL stage before display so every
//! surface (3D PBR, terminal truecolor, ASCII fallback) transforms identically.

use crate::{AccessibilityPalette, BirthOrigin, DeathCause, TerrainKind};

// ---------------------------------------------------------------------------
// Accessibility palette transform (exact lift from the GPUI and Bevy
// implementations, which already used byte-identical matrices).
// ---------------------------------------------------------------------------

/// Deuteranopia simulation/correction matrix (green-blind).
const DEUTERANOPIA_MATRIX: [[f32; 3]; 3] =
    [[0.43, 0.72, -0.15], [0.34, 0.57, 0.09], [-0.02, 0.03, 0.97]];

/// Protanopia simulation/correction matrix (red-blind).
const PROTAOPIA_MATRIX: [[f32; 3]; 3] =
    [[0.20, 0.99, -0.19], [0.16, 0.79, 0.04], [0.01, -0.01, 1.00]];

/// Tritanopia simulation/correction matrix (blue-blind).
const TRITANOPIA_MATRIX: [[f32; 3]; 3] =
    [[0.95, 0.05, 0.00], [0.00, 0.43, 0.56], [0.00, 0.47, 0.53]];

const fn transform_palette(rgb: [f32; 3], matrix: &[[f32; 3]; 3]) -> [f32; 3] {
    [
        (rgb[0] * matrix[0][0] + rgb[1] * matrix[0][1] + rgb[2] * matrix[0][2]).clamp(0.0, 1.0),
        (rgb[0] * matrix[1][0] + rgb[1] * matrix[1][1] + rgb[2] * matrix[1][2]).clamp(0.0, 1.0),
        (rgb[0] * matrix[2][0] + rgb[1] * matrix[2][1] + rgb[2] * matrix[2][2]).clamp(0.0, 1.0),
    ]
}

/// Apply the accessibility palette to a semantic color.
///
/// This is the exact transform both legacy renderers already used (the GPUI
/// `apply_palette` and Bevy `apply_palette_rgb` agreed byte-for-byte); it now
/// lives in exactly one place. `Natural` is the identity. `HighContrast`
/// brightens light colors and darkens dark ones around the rec.709 luminance
/// midpoint. Input components are clamped into `[0, 1]` first so out-of-range
/// science-side colors cannot smear.
#[must_use]
pub fn apply_accessibility_palette(rgb: [f32; 3], palette: AccessibilityPalette) -> [f32; 3] {
    let clamped = [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ];
    match palette {
        AccessibilityPalette::Natural => clamped,
        AccessibilityPalette::Deuteranopia => transform_palette(clamped, &DEUTERANOPIA_MATRIX),
        AccessibilityPalette::Protanopia => transform_palette(clamped, &PROTAOPIA_MATRIX),
        AccessibilityPalette::Tritanopia => transform_palette(clamped, &TRITANOPIA_MATRIX),
        AccessibilityPalette::HighContrast => {
            let luminance = 0.2126 * clamped[0] + 0.7152 * clamped[1] + 0.0722 * clamped[2];
            if luminance > 0.5 {
                [
                    (clamped[0] + 0.15).min(1.0),
                    (clamped[1] + 0.15).min(1.0),
                    (clamped[2] + 0.15).min(1.0),
                ]
            } else {
                [
                    (clamped[0] * 0.6).clamp(0.0, 1.0),
                    (clamped[1] * 0.6).clamp(0.0, 1.0),
                    (clamped[2] * 0.6).clamp(0.0, 1.0),
                ]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Agent semantics (lifted from the Bevy frontend's current 3D visual language;
// the GPUI avatar painter encodes the same meanings as 2D paths).
// ---------------------------------------------------------------------------

/// Herbivore end of the diet stripe ramp.
pub const HERBIVORE_RGB: [f32; 3] = [0.18, 0.84, 0.36];
/// Carnivore end of the diet stripe ramp.
pub const CARNIVORE_RGB: [f32; 3] = [0.86, 0.22, 0.2];
/// Cold-preference temperature accent.
pub const TEMP_COLD_RGB: [f32; 3] = [0.2, 0.45, 1.0];
/// Warm-preference temperature accent.
pub const TEMP_WARM_RGB: [f32; 3] = [1.0, 0.52, 0.24];
/// Neutral wheel body color.
pub const WHEEL_BASE_RGB: [f32; 3] = [0.14, 0.16, 0.22];

/// Selection/hover state shared by every frontend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VisualSelection {
    #[default]
    None,
    Hovered,
    Selected,
}

impl VisualSelection {
    /// Emissive highlight ramp: 0.12 unselected, 0.28 hovered, 0.48 selected
    /// (the exact legacy values).
    #[must_use]
    pub const fn highlight(self) -> f32 {
        match self {
            Self::None => 0.12,
            Self::Hovered => 0.28,
            Self::Selected => 0.48,
        }
    }
}

/// Plain-data inputs the agent visual mapping reads.
///
/// Field meanings mirror the simulation's canonical semantics: `health` spans
/// `0..=2`, `herbivore_tendency` and `temperature_preference` span `[0, 1]`,
/// wheel outputs are the requested wheel efforts, and `sound`/`food_delta` are
/// per-tick activity channels. Every input is clamped defensively inside the
/// mapping; out-of-range values indicate an upstream bug and are logged by the
/// calling frontend, never silently trusted.
#[derive(Debug, Clone, Copy, Default)]
pub struct AgentVisualInput {
    /// Genome-inherited body color (lineage hue).
    pub genome_color: [f32; 3],
    /// Current health on the simulation's `0..=2` scale.
    pub health: f32,
    /// Diet axis: 0 = pure carnivore, 1 = pure herbivore.
    pub herbivore_tendency: f32,
    /// Temperature preference axis: 0 = cold-loving, 1 = heat-loving.
    pub temperature_preference: f32,
    /// Requested left/right wheel efforts (sign = direction).
    pub wheel_left: f32,
    pub wheel_right: f32,
    /// Whether the spike is currently extended.
    pub spike_extended: bool,
    /// Current spike length (world units, may be fractional).
    pub spike_length: f32,
    /// Whether the agent is boosting.
    pub boosting: bool,
    /// Requested sound output level this tick.
    pub sound_output: f32,
    /// Configured sound trait multiplier.
    pub sound_multiplier: f32,
    /// Measured sound level this tick.
    pub sound_level: f32,
    /// Net food intake delta this tick (positive = eating).
    pub food_delta: f32,
    /// Smell trait modifier.
    pub trait_smell: f32,
    /// Hearing trait modifier.
    pub trait_hearing: f32,
    /// Frontend selection state.
    pub selection: VisualSelection,
}

/// Renderer-neutral outputs of the agent visual mapping.
#[derive(Debug, Clone, Copy)]
pub struct AgentVisualParams {
    /// Body base color after the health factor (natural palette).
    pub body_color: [f32; 3],
    /// Body emissive color including the selection highlight ramp.
    pub body_emissive: [f32; 3],
    /// Diet/temperature stripe color.
    pub stripe_color: [f32; 3],
    /// Stripe emissive (45% of stripe color, the legacy ratio).
    pub stripe_emissive: [f32; 3],
    /// Wheel base colors after speed brightening.
    pub wheel_colors: [[f32; 3]; 2],
    /// Wheel emissive colors (speed-scaled, cool-tinted).
    pub wheel_emissives: [[f32; 3]; 2],
    /// Mouth activity in `[0, 1]` (eat/yell/sound composite).
    pub mouth_activity: f32,
    /// Mouth base color (red deepens with activity).
    pub mouth_color: [f32; 3],
    /// Nose tint from the smell trait.
    pub nose_color: [f32; 3],
    /// Spike readiness in `[0, 1]`: 1.0 when extended, else fractional growth.
    pub spike_readiness: f32,
}

const fn mix_vec3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

const fn clamp01(v: f32) -> f32 {
    // NaN input means an upstream science bug; display it as 0 rather than
    // smearing NaN through every downstream blend (f32::clamp propagates NaN).
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The diet stripe color for a tendency/preference pair.
///
/// Exact legacy formula: carnivore-red ↔ herbivore-green mixed by tendency,
/// then an 18% blend toward the temperature accent so climate preference
/// reads as a warm/cool tint on the same stripe.
#[must_use]
pub fn diet_stripe_color(herbivore_tendency: f32, temperature_preference: f32) -> [f32; 3] {
    let herbivore = clamp01(herbivore_tendency);
    let temp_pref = clamp01(temperature_preference);
    let mut stripe = mix_vec3(CARNIVORE_RGB, HERBIVORE_RGB, herbivore);
    let temp_accent = mix_vec3(TEMP_COLD_RGB, TEMP_WARM_RGB, temp_pref);
    stripe = mix_vec3(stripe, temp_accent, 0.18);
    stripe
}

/// The body health factor: `health/2` clamped into `[0.45, 1.0]`.
///
/// The floor keeps even dying agents visible; the divisor is the simulation's
/// canonical `0..=2` health scale (a legacy Bevy HUD path once normalized
/// against 100 — that bug class is impossible here because this is the only
/// implementation).
#[must_use]
pub fn health_factor(health: f32) -> f32 {
    if health.is_finite() {
        (health / 2.0).clamp(0.45, 1.0)
    } else {
        0.45
    }
}

/// Age desaturation/weathering factor in `[0.82, 1.0]`.
///
/// Multiplies body saturation so elders read as seasoned rather than freshly
/// spawned; the floor keeps ancient agents colorful enough to stay
/// identifiable. `reference_age` is the age at which the factor reaches its
/// floor (frontends pass the observed maximum or a scenario constant); age
/// beyond it simply stays at the floor. Pure and deterministic.
#[must_use]
pub fn age_factor(age_ticks: u64, reference_age: u64) -> f32 {
    if reference_age == 0 {
        return 1.0;
    }
    let t = (age_ticks as f32 / reference_age as f32).min(1.0);
    1.0 - 0.18 * t
}

/// Apply a saturation factor (e.g. [`age_factor`]) to a color, preserving hue
/// and luminance bias toward the rec.709 grey point.
#[must_use]
pub fn apply_saturation(rgb: [f32; 3], factor: f32) -> [f32; 3] {
    let f = clamp01(factor);
    let lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    [
        clamp01(lum + (rgb[0] - lum) * f),
        clamp01(lum + (rgb[1] - lum) * f),
        clamp01(lum + (rgb[2] - lum) * f),
    ]
}

/// Compute every per-agent visual parameter from plain inputs.
///
/// Pure, total, and allocation-free. Colors are in the natural palette;
/// frontends apply [`apply_accessibility_palette`] at display time.
#[must_use]
pub fn agent_visual_params(input: &AgentVisualInput) -> AgentVisualParams {
    let rgb = [
        clamp01(input.genome_color[0]),
        clamp01(input.genome_color[1]),
        clamp01(input.genome_color[2]),
    ];
    let hf = health_factor(input.health);
    let body_color = [rgb[0] * hf, rgb[1] * hf, rgb[2] * hf];

    let highlight = input.selection.highlight();
    let body_emissive = [
        (rgb[0] + highlight * 0.8).min(1.0),
        (rgb[1] + highlight * 0.6).min(1.0),
        (rgb[2] + highlight).min(1.0),
    ];

    let stripe = diet_stripe_color(input.herbivore_tendency, input.temperature_preference);
    let stripe_emissive = [stripe[0] * 0.45, stripe[1] * 0.45, stripe[2] * 0.45];

    let left_speed = clamp01(input.wheel_left.abs());
    let right_speed = clamp01(input.wheel_right.abs());
    let left_rgb = [
        WHEEL_BASE_RGB[0] * (0.65 + left_speed * 0.55),
        WHEEL_BASE_RGB[1] * (0.65 + left_speed * 0.55),
        WHEEL_BASE_RGB[2] * (0.65 + left_speed * 0.55),
    ];
    let right_rgb = [
        WHEEL_BASE_RGB[0] * (0.65 + right_speed * 0.55),
        WHEEL_BASE_RGB[1] * (0.65 + right_speed * 0.55),
        WHEEL_BASE_RGB[2] * (0.65 + right_speed * 0.55),
    ];
    let left_emissive = [
        left_rgb[0] * left_speed * 0.8,
        left_rgb[1] * left_speed * 0.7,
        left_rgb[2] * left_speed * 1.1,
    ];
    let right_emissive = [
        right_rgb[0] * right_speed * 0.8,
        right_rgb[1] * right_speed * 0.7,
        right_rgb[2] * right_speed * 1.1,
    ];

    let vocal_energy = clamp01(input.sound_output.abs() * input.sound_multiplier.max(0.1));
    let mouth_activity =
        clamp01(input.food_delta.abs() * 0.75 + vocal_energy * 0.9 + input.sound_level * 0.35);
    let mouth_color = [
        0.58 + mouth_activity * 0.3,
        0.1 + mouth_activity * 0.12,
        0.12 + mouth_activity * 0.08,
    ];

    let nose_color = mix_vec3(
        [0.94, 0.84, 0.66],
        [0.98, 0.92, 0.78],
        clamp01(input.trait_smell * 0.4),
    );

    let spike_readiness = if input.spike_extended {
        1.0
    } else {
        clamp01(input.spike_length)
    };

    AgentVisualParams {
        body_color,
        body_emissive,
        stripe_color: stripe,
        stripe_emissive,
        wheel_colors: [left_rgb, right_rgb],
        wheel_emissives: [left_emissive, right_emissive],
        mouth_activity,
        mouth_color,
        nose_color,
        spike_readiness,
    }
}

// ---------------------------------------------------------------------------
// Terrain semantics (base palette lifted from the Bevy chunk renderer, which
// itself inherited the GPUI canvas hues; brightness factors made explicit).
// ---------------------------------------------------------------------------

/// Base color per terrain kind (natural palette, sRGB bytes as floats).
pub const TERRAIN_BASE_COLORS: [[f32; 3]; 6] = [
    [0.117_647, 0.247_059, 0.400_000], // Deep water
    [0.184_314, 0.450_980, 0.701_961], // Shallow water
    [0.694_118, 0.305_882, 0.027_451], // Sand
    [0.313_725, 0.662_745, 0.074_510], // Grass
    [0.474_510, 0.831_373, 0.427_451], // Bloom
    [0.662_745, 0.694_118, 0.729_412], // Rock
];

/// Base color for a terrain kind.
#[must_use]
pub fn terrain_kind_base_color(kind: TerrainKind) -> [f32; 3] {
    let index = match kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    };
    TERRAIN_BASE_COLORS[index]
}

/// The daylight level used when the day/night cycle is off (the historical
/// constant both renderers hard-coded).
pub const DAYLIGHT_STATIC: f32 = 0.65;
/// Night floor of the day/night curve: scenes stay readable while emissives
/// carry the frame.
pub const DAYLIGHT_NIGHT_FLOOR: f32 = 0.15;

/// Shared day/night curve.
///
/// `cycle_ticks == 0` is the historical static lighting and returns
/// [`DAYLIGHT_STATIC`]. Otherwise the phase advances one full cycle per
/// `cycle_ticks` ticks (plus `start_phase` in `[0, 1)`), and the curve is a
/// cosine between [`DAYLIGHT_NIGHT_FLOOR`] at midnight and `1.0` at noon, so
/// the Bevy sun and the terminal tint can never disagree about time of day.
#[must_use]
pub fn daylight_factor(tick: u64, cycle_ticks: u32, start_phase: f32) -> f32 {
    if cycle_ticks == 0 {
        return DAYLIGHT_STATIC;
    }
    let phase = (tick % u64::from(cycle_ticks)) as f32 / cycle_ticks as f32 + start_phase;
    let phase = phase - phase.floor();
    // phase 0.0 = dawn, 0.25 = noon, 0.5 = dusk, 0.75 = midnight.
    let sun = ((phase - 0.25) * 2.0 * core::f32::consts::PI).cos() * 0.5 + 0.5;
    DAYLIGHT_NIGHT_FLOOR + (1.0 - DAYLIGHT_NIGHT_FLOOR) * sun
}

/// Inputs for terrain shading of one tile.
#[derive(Debug, Clone, Copy)]
pub struct TerrainShadeInput {
    /// Tile kind.
    pub kind: TerrainKind,
    /// Moisture/fertility channel in `[0, 1]`.
    pub moisture: f32,
    /// Elevation channel (normalized by the frontend against world range).
    pub elevation: f32,
    /// Local slope magnitude in `[0, 1]`.
    pub slope: f32,
    /// Small deterministic per-tile accent (flower/ore variation).
    pub accent: f32,
    /// Daylight level from [`daylight_factor`].
    pub daylight: f32,
}

/// Shaded terrain color for one tile (natural palette).
///
/// Exact legacy composition: per-kind brightness window, then a second-stage
/// moisture/accent/slope factor for the living biomes, clamped to `[0, 1]`.
/// `elevation`, `slope`, `accent`, and `daylight` are clamped defensively;
/// `moisture` likewise.
#[must_use]
pub fn terrain_shaded_color(input: &TerrainShadeInput) -> [f32; 3] {
    let moisture = clamp01(input.moisture);
    let elevation = clamp01(input.elevation);
    let slope = clamp01(input.slope);
    let accent = clamp01(input.accent);
    let daylight = clamp01(input.daylight);

    let base = terrain_kind_base_color(input.kind);
    let brightness = match input.kind {
        TerrainKind::DeepWater => (0.42 + daylight * 0.25 + moisture * 0.2).clamp(0.25, 1.05),
        TerrainKind::ShallowWater => (0.55 + daylight * 0.35 + moisture * 0.3).clamp(0.4, 1.25),
        TerrainKind::Sand => (0.72 + daylight * 0.18 + elevation * 0.35).clamp(0.45, 1.35),
        TerrainKind::Grass => (0.62 + daylight * 0.28 + moisture * 0.4).clamp(0.4, 1.35),
        TerrainKind::Bloom => (0.68 + daylight * 0.35 + moisture * 0.5).clamp(0.45, 1.45),
        TerrainKind::Rock => (0.60 + daylight * 0.22 + slope * 0.45).clamp(0.35, 1.25),
    };

    let mut rgb = [
        base[0] * brightness,
        base[1] * brightness,
        base[2] * brightness,
    ];

    let factor = match input.kind {
        TerrainKind::Bloom | TerrainKind::Grass => {
            (0.9 + moisture * 0.3 + accent * 0.05).clamp(0.6, 1.4)
        }
        TerrainKind::Sand => (0.9 + accent * 0.08).clamp(0.6, 1.3),
        TerrainKind::Rock => (0.85 + slope * 0.3).clamp(0.6, 1.2),
        TerrainKind::DeepWater | TerrainKind::ShallowWater => 1.0,
    };
    if factor != 1.0 {
        rgb = [rgb[0] * factor, rgb[1] * factor, rgb[2] * factor];
    }

    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

// ---------------------------------------------------------------------------
// Food semantics: density ramp + deterministic shimmer phase.
// ---------------------------------------------------------------------------

/// Food density ramp endpoints (natural palette).
pub const FOOD_SPARSE_RGB: [f32; 3] = [0.10, 0.35, 0.12];
/// Mid-density green.
pub const FOOD_MID_RGB: [f32; 3] = [0.35, 0.80, 0.25];
/// Dense, ready-to-eat gold.
pub const FOOD_DENSE_RGB: [f32; 3] = [0.95, 0.85, 0.30];

/// Food density color ramp: sparse dark green -> vibrant green -> ripe gold.
///
/// Two-segment ramp over `density` in `[0, 1]` (clamped). This is the
/// canonical food color answer; previously each frontend picked its own
/// greens (or encoded food only as brightness modifiers).
#[must_use]
pub fn food_density_color(density: f32) -> [f32; 3] {
    let d = clamp01(density);
    if d < 0.5 {
        mix_vec3(FOOD_SPARSE_RGB, FOOD_MID_RGB, d * 2.0)
    } else {
        mix_vec3(FOOD_MID_RGB, FOOD_DENSE_RGB, (d - 0.5) * 2.0)
    }
}

/// Shimmer period for food/water pulse animations, in ticks.
pub const SHIMMER_PERIOD_TICKS: u64 = 120;

/// Deterministic per-cell animation phase in `[0, 1)`.
///
/// FNV-1a 64 over the cell coordinates (the same diagnostic hash family the
/// core already uses for characterization), folded into a phase offset. The
/// TUI and the 3D water/food shaders call this same function so a given cell
/// pulses in lockstep on every surface and in every replay.
#[must_use]
pub fn cell_phase(cell_x: u32, cell_y: u32) -> f32 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in [
        (cell_x & 0xff) as u8,
        (cell_x >> 8) as u8,
        (cell_x >> 16) as u8,
        (cell_x >> 24) as u8,
        (cell_y & 0xff) as u8,
        (cell_y >> 8) as u8,
        (cell_y >> 16) as u8,
        (cell_y >> 24) as u8,
    ] {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (hash % 10_000) as f32 / 10_000.0
}

/// The shared shimmer value in `[0, 1]` for a cell at a tick.
///
/// `0.5 + 0.5 * sin(2*pi * (tick/PERIOD + phase))`: fully deterministic,
/// frozen when the simulation is paused (tick stops advancing), identical on
/// every renderer.
#[must_use]
pub fn shimmer(tick: u64, cell_x: u32, cell_y: u32) -> f32 {
    let t = (tick % SHIMMER_PERIOD_TICKS) as f32 / SHIMMER_PERIOD_TICKS as f32;
    let phase = t + cell_phase(cell_x, cell_y);
    let phase = phase - phase.floor();
    0.5 + 0.5 * (phase * 2.0 * core::f32::consts::PI).sin()
}

// ---------------------------------------------------------------------------
// Event -> visual cue table: the single art-direction answer for "what does
// that world event look like", consumed by Bevy particles (bd-2z0.14.1.7)
// and the TUI pulse layer (bd-2z0.14.2.4) alike.
// ---------------------------------------------------------------------------

/// Discrete visual events a frontend may be asked to portray.
///
/// Positions are world coordinates; intensities are caller-measured (e.g.
/// damage) and normalized by this table. The stream that PRODUCES these
/// events (today: tick deltas + persistence records; later: the typed
/// `EventRecord` stream from bd-16g.2.2) is a separate concern owned by the
/// frontend/projection beads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorldVisualEvent {
    /// An agent arrived. Color follows the origin.
    Birth { origin: BirthOrigin },
    /// An agent died. Effect follows the cause.
    Death { cause: DeathCause },
    /// A spike connected. `damage` is the dealt amount (pre-normalization).
    CombatHit { damage: f32 },
    /// An agent ate. `amount` is the intake delta.
    Eat { amount: f32 },
    /// A reproduction pulse (distinct from the child's Birth cue).
    Reproduce,
    /// A spike began extending (telegraph).
    SpikeExtend,
}

/// Cue families the two renderers know how to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualCueKind {
    /// Upward sparkle burst (births).
    Sparkle,
    /// Directional impact shards (combat deaths/hits).
    Shards,
    /// Sinking grey motes (starvation/aging).
    Wilt,
    /// Small arcing flecks toward the mouth (eating).
    Nibble,
    /// White-yellow spark cone along the spike vector.
    SparkCone,
    /// Expanding dual-tone ring (reproduction).
    PulseRing,
    /// Brief full-body flash (spike telegraph).
    Flash,
}

/// A resolved visual cue: what to draw, in which color, for how long.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualCue {
    /// Which effect family.
    pub kind: VisualCueKind,
    /// Base color (natural palette).
    pub color: [f32; 3],
    /// Secondary color for two-tone effects; equals `color` otherwise.
    pub accent_color: [f32; 3],
    /// Emissive intensity in `[0, 1]` (drives bloom and TUI brightness).
    pub intensity: f32,
    /// Effect radius in world units.
    pub radius: f32,
    /// Effect lifetime in ticks (deterministic; never wall-clock).
    pub duration_ticks: u32,
}

/// Maximum combat damage used to normalize hit intensity.
pub const COMBAT_DAMAGE_REFERENCE: f32 = 2.0;

/// Resolve a world event to its visual cue.
///
/// Pure table: same event, same cue, on every renderer. Returns the canonical
/// colors/durations from the art bible (bd-2z0.14.3.6); intensity is scaled
/// by caller-measured magnitudes where applicable.
#[must_use]
pub fn visual_cue_for_event(event: &WorldVisualEvent) -> VisualCue {
    match *event {
        WorldVisualEvent::Birth { origin } => {
            let (color, intensity) = match origin {
                BirthOrigin::Born => ([1.0, 0.9, 0.55], 0.9),
                BirthOrigin::Seeded => ([0.75, 0.85, 1.0], 0.6),
                BirthOrigin::Injected => ([0.8, 0.6, 1.0], 0.75),
            };
            VisualCue {
                kind: VisualCueKind::Sparkle,
                color,
                accent_color: color,
                intensity,
                radius: 6.0,
                duration_ticks: 24,
            }
        }
        WorldVisualEvent::Death { cause } => match cause {
            DeathCause::CombatCarnivore | DeathCause::CombatHerbivore => VisualCue {
                kind: VisualCueKind::Shards,
                color: [1.0, 0.35, 0.15],
                accent_color: [1.0, 0.8, 0.2],
                intensity: 1.0,
                radius: 8.0,
                duration_ticks: 20,
            },
            DeathCause::Starvation => VisualCue {
                kind: VisualCueKind::Wilt,
                color: [0.45, 0.45, 0.5],
                accent_color: [0.3, 0.3, 0.35],
                intensity: 0.5,
                radius: 5.0,
                duration_ticks: 36,
            },
            DeathCause::Aging | DeathCause::Unknown => VisualCue {
                kind: VisualCueKind::Wilt,
                color: [0.55, 0.6, 0.75],
                accent_color: [0.4, 0.45, 0.6],
                intensity: 0.55,
                radius: 5.0,
                duration_ticks: 36,
            },
        },
        WorldVisualEvent::CombatHit { damage } => {
            let normalized = clamp01(damage / COMBAT_DAMAGE_REFERENCE);
            VisualCue {
                kind: VisualCueKind::SparkCone,
                color: [1.0, 0.95, 0.6],
                accent_color: [1.0, 0.7, 0.25],
                intensity: 0.5 + 0.5 * normalized,
                radius: 4.0 + 4.0 * normalized,
                duration_ticks: 8,
            }
        }
        WorldVisualEvent::Eat { amount } => {
            let normalized = clamp01(amount.abs());
            VisualCue {
                kind: VisualCueKind::Nibble,
                color: [0.5, 0.9, 0.4],
                accent_color: [0.8, 1.0, 0.5],
                intensity: 0.3 + 0.3 * normalized,
                radius: 2.0,
                duration_ticks: 12,
            }
        }
        WorldVisualEvent::Reproduce => VisualCue {
            kind: VisualCueKind::PulseRing,
            color: [1.0, 0.75, 0.4],
            accent_color: [0.6, 0.9, 1.0],
            intensity: 0.8,
            radius: 10.0,
            duration_ticks: 28,
        },
        WorldVisualEvent::SpikeExtend => VisualCue {
            kind: VisualCueKind::Flash,
            color: [1.0, 1.0, 1.0],
            accent_color: [1.0, 0.9, 0.6],
            intensity: 0.7,
            radius: 3.0,
            duration_ticks: 6,
        },
    }
}

// ---------------------------------------------------------------------------
// Terrain splat weights (bd-2z0.14.1.2.1): per-tile biome blend weights for
// the PBR splat shader. The shader blends across tile boundaries; this pure
// function owns the per-tile weight rules so the GPU material and any CPU
// reference agree exactly.
// ---------------------------------------------------------------------------

/// Number of splat layers (one per terrain kind, in [`TERRAIN_BASE_COLORS`]
/// order).
pub const SPLAT_LAYERS: usize = 6;

/// Rule thresholds (documented in the art bible, bd-2z0.14.3.6).
/// Above this slope, living biomes give way to rock.
pub const SPLAT_SLOPE_ROCK_THRESHOLD: f32 = 0.55;
/// Below this elevation, dry biomes blend toward sand (the waterline band).
pub const SPLAT_WATERLINE_ELEVATION: f32 = 0.22;
/// Above this elevation, living biomes blend toward rock (alpine rule).
pub const SPLAT_ALPINE_ELEVATION: f32 = 0.85;

/// Inputs for the splat-weight rules of one tile.
#[derive(Debug, Clone, Copy)]
pub struct SplatInput {
    /// Tile kind (the dominant biome).
    pub kind: TerrainKind,
    /// Normalized elevation in `[0, 1]`.
    pub elevation: f32,
    /// Local slope magnitude in `[0, 1]`.
    pub slope: f32,
    /// Hydrology water depth above the tile (world units, >= 0).
    pub water_depth: f32,
}

/// Per-tile splat weights over the six biome layers, normalized to sum 1.
///
/// Rule order (later rules blend into the result of earlier ones):
/// 1. One-hot on the tile kind.
/// 2. Waterline: below [`SPLAT_WATERLINE_ELEVATION`], land biomes blend
///    toward sand proportionally to how far below the line they are.
/// 3. Steep slopes: above [`SPLAT_SLOPE_ROCK_THRESHOLD`], living biomes
///    (grass/bloom/sand) blend toward rock with slope overhang.
/// 4. Alpine: above [`SPLAT_ALPINE_ELEVATION`], living biomes blend toward
///    rock with elevation overhang.
/// 5. Flooded: any positive water depth blends dry land toward the matching
///    water layer (deep vs shallow by depth), capped at full replacement.
///
/// Water kinds short-circuit rules 2-4 (a lakebed does not become sandy
/// cliffs), and rule 5 never applies to them. Output is always finite and
/// sums to `1 +/- 1e-5`.
#[must_use]
pub fn splat_weights(input: &SplatInput) -> [f32; SPLAT_LAYERS] {
    let kind_index = match input.kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    };
    let mut w = [0.0_f32; SPLAT_LAYERS];
    w[kind_index] = 1.0;

    let is_water = matches!(
        input.kind,
        TerrainKind::DeepWater | TerrainKind::ShallowWater
    );
    if !is_water {
        let elevation = clamp01(input.elevation);
        let slope = clamp01(input.slope);

        // Rule 2: waterline sand.
        if elevation < SPLAT_WATERLINE_ELEVATION {
            let t = (SPLAT_WATERLINE_ELEVATION - elevation) / SPLAT_WATERLINE_ELEVATION;
            blend_toward(&mut w, 2, t * 0.8);
        }
        // Rule 3: steep-slope rock (only for living biomes; bare sand already reads dry).
        if slope > SPLAT_SLOPE_ROCK_THRESHOLD {
            let t = (slope - SPLAT_SLOPE_ROCK_THRESHOLD) / (1.0 - SPLAT_SLOPE_ROCK_THRESHOLD);
            blend_toward(&mut w, 5, t * 0.85);
        }
        // Rule 4: alpine rock.
        if elevation > SPLAT_ALPINE_ELEVATION {
            let t = (elevation - SPLAT_ALPINE_ELEVATION) / (1.0 - SPLAT_ALPINE_ELEVATION);
            blend_toward(&mut w, 5, t * 0.7);
        }
        // Rule 5: flooding replaces dry land with the depth-matched water layer.
        if input.water_depth.is_finite() && input.water_depth > 0.0 {
            let deep = input.water_depth >= 3.0;
            let layer = usize::from(!deep); // 0 = deep, 1 = shallow
            let t = (input.water_depth / 3.0).clamp(0.0, 1.0);
            blend_toward(&mut w, layer, t);
        }
    }

    // Defensive renormalization keeps the sum exact under extreme inputs.
    let sum: f32 = w.iter().sum();
    if sum.is_finite() && sum > 0.0 {
        for v in &mut w {
            *v /= sum;
        }
    } else {
        w = [0.0; SPLAT_LAYERS];
        w[kind_index] = 1.0;
    }
    w
}

/// Blend `amount` of total weight into `layer`, taking proportionally from
/// all other layers so the sum stays 1.
fn blend_toward(w: &mut [f32; SPLAT_LAYERS], layer: usize, amount: f32) {
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    let current = w[layer];
    let target_share = current + (1.0 - current) * amount;
    let remaining = (1.0 - target_share).max(0.0);
    let others_sum = 1.0 - current;
    for (i, v) in w.iter_mut().enumerate() {
        if i == layer {
            *v = target_share;
        } else if others_sum > 0.0 {
            *v = (*v / others_sum) * remaining;
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic procedural biome texture baking (bd-2z0.14.1.2.1): the
// albedo/detail layers for the splat shader. Byte-identical across runs and
// platforms — the golden hash test proves it — so the repo stays asset-free
// and reproducible.
// ---------------------------------------------------------------------------

/// Lattice-hash value-noise sample in `[-1, 1]`.
///
/// Deterministic across platforms: integer hashing (no float hash inputs),
/// bilinear smoothstep interpolation, two octaves at fixed weights. The seed
/// domain-separates biomes so no two layers share a pattern.
#[must_use]
pub fn value_noise_2d(seed: u64, x: f32, y: f32) -> f32 {
    fn lattice(seed: u64, ix: i64, iy: i64) -> f32 {
        let mut h = seed ^ (ix as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= (iy as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        h = h.wrapping_mul(0x1656_67B1_9E37_79F9);
        h ^= h >> 29;
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 32;
        // Map to [-1, 1] via the top 24 bits for cross-platform stability.
        ((h >> 40) as f32 / 8_388_607.5) - 1.0
    }
    fn smooth(t: f32) -> f32 {
        t * t * (3.0 - 2.0 * t)
    }
    fn octave(seed: u64, x: f32, y: f32) -> f32 {
        let ix = x.floor() as i64;
        let iy = y.floor() as i64;
        let fx = smooth(x - x.floor());
        let fy = smooth(y - y.floor());
        let a = lattice(seed, ix, iy);
        let b = lattice(seed, ix + 1, iy);
        let c = lattice(seed, ix, iy + 1);
        let d = lattice(seed, ix + 1, iy + 1);
        let top = a + (b - a) * fx;
        let bottom = c + (d - c) * fx;
        top + (bottom - top) * fy
    }
    let base = octave(seed, x, y);
    let detail = octave(seed ^ 0xA5A5_5A5A_3C3C_F0F0, x * 2.13, y * 2.13);
    (base * 0.7 + detail * 0.3).clamp(-1.0, 1.0)
}

/// Per-biome texture parameters for [`bake_biome_texture`].
#[derive(Debug, Clone, Copy)]
pub struct BiomeTextureSpec {
    /// Grain frequency across the tile (cells of the noise lattice).
    pub grain_scale: f32,
    /// Brightness variation amplitude in `[0, 1]`.
    pub grain_amplitude: f32,
}

/// The texture spec for each biome layer, in [`TERRAIN_BASE_COLORS`] order.
#[must_use]
pub const fn biome_texture_specs() -> [BiomeTextureSpec; SPLAT_LAYERS] {
    [
        BiomeTextureSpec {
            grain_scale: 3.0,
            grain_amplitude: 0.10,
        }, // deep water: calm
        BiomeTextureSpec {
            grain_scale: 4.0,
            grain_amplitude: 0.14,
        }, // shallow water
        BiomeTextureSpec {
            grain_scale: 8.0,
            grain_amplitude: 0.20,
        }, // sand: ripples
        BiomeTextureSpec {
            grain_scale: 10.0,
            grain_amplitude: 0.26,
        }, // grass: tufts
        BiomeTextureSpec {
            grain_scale: 7.0,
            grain_amplitude: 0.30,
        }, // bloom: petals
        BiomeTextureSpec {
            grain_scale: 5.0,
            grain_amplitude: 0.34,
        }, // rock: strata
    ]
}

/// Bake one deterministic biome albedo texture as RGBA8 (row-major,
/// `size * size * 4` bytes).
///
/// The biome's [`TERRAIN_BASE_COLORS`] base is modulated by
/// [`value_noise_2d`] at the layer's grain parameters; alpha is fully opaque.
/// Identical `(kind, seed, size)` inputs produce byte-identical output on
/// every platform (integer hash lattice; no transcendental float inputs).
#[must_use]
pub fn bake_biome_texture(kind: TerrainKind, seed: u64, size: u32) -> Vec<u8> {
    let index = match kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    };
    let base = TERRAIN_BASE_COLORS[index];
    let spec = biome_texture_specs()[index];
    let domain = seed ^ (index as u64).wrapping_mul(0xB529_7A4D_2A95_5F31);
    let size = size.max(1);
    let mut out = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let n = value_noise_2d(
                domain,
                x as f32 / size as f32 * spec.grain_scale,
                y as f32 / size as f32 * spec.grain_scale,
            );
            let m = 1.0 + n * spec.grain_amplitude;
            for &channel in &base {
                let v = (channel * m).clamp(0.0, 1.0);
                out.push((v * 255.0 + 0.5) as u8);
            }
            out.push(255);
        }
    }
    out
}

/// Bake the full six-layer biome albedo atlas side by side
/// (`size * 6 x size` RGBA8), the layout the splat shader samples.
#[must_use]
pub fn bake_biome_atlas(seed: u64, size: u32) -> Vec<u8> {
    let kinds = [
        TerrainKind::DeepWater,
        TerrainKind::ShallowWater,
        TerrainKind::Sand,
        TerrainKind::Grass,
        TerrainKind::Bloom,
        TerrainKind::Rock,
    ];
    let size = size.max(1) as usize;
    let mut atlas = vec![0_u8; size * size * 6 * 4];
    for (layer, kind) in kinds.iter().enumerate() {
        let tex = bake_biome_texture(*kind, seed, size as u32);
        for y in 0..size {
            let src = y * size * 4;
            let dst = (y * size * 6 + layer * size) * 4;
            atlas[dst..dst + size * 4].copy_from_slice(&tex[src..src + size * 4]);
        }
    }
    atlas
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    fn assert_rgb_close(actual: [f32; 3], expected: [f32; 3], label: &str) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() < 1.0e-4,
                "{label}: channel {i} expected {}, got {}",
                expected[i],
                actual[i]
            );
        }
    }

    #[test]
    fn natural_palette_is_identity_and_clamps() {
        let rgb = [0.2, 0.5, 0.9];
        assert_eq!(
            apply_accessibility_palette(rgb, AccessibilityPalette::Natural),
            rgb
        );
        // Out-of-range inputs are clamped before matching.
        let clamped = apply_accessibility_palette([1.4, -0.2, 0.5], AccessibilityPalette::Natural);
        assert_eq!(clamped, [1.0, 0.0, 0.5]);
    }

    #[test]
    fn cvd_matrices_match_legacy_constants() {
        // Golden vectors computed from the legacy GPUI/Bevy matrices: identical
        // constants must produce identical outputs forever.
        let rgb = [0.8, 0.4, 0.2];
        let deut = apply_accessibility_palette(rgb, AccessibilityPalette::Deuteranopia);
        assert_rgb_close(
            deut,
            [
                (0.8_f32 * 0.43 + 0.4 * 0.72 + 0.2 * -0.15).clamp(0.0, 1.0),
                (0.8_f32 * 0.34 + 0.4 * 0.57 + 0.2 * 0.09).clamp(0.0, 1.0),
                (0.8_f32 * -0.02 + 0.4 * 0.03 + 0.2 * 0.97).clamp(0.0, 1.0),
            ],
            "deuteranopia",
        );
        let trit = apply_accessibility_palette([0.1, 0.9, 0.3], AccessibilityPalette::Tritanopia);
        for c in trit {
            assert!((0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn high_contrast_brightens_light_and_darkens_dark() {
        let light =
            apply_accessibility_palette([0.9, 0.9, 0.9], AccessibilityPalette::HighContrast);
        assert!(light[0] > 0.9, "light colors brighten: {light:?}");
        let dark = apply_accessibility_palette([0.1, 0.1, 0.1], AccessibilityPalette::HighContrast);
        assert!(dark[0] < 0.1, "dark colors darken: {dark:?}");
        // Luminance threshold: exactly mid-grey goes dark branch (luminance !> 0.5).
        let mid = apply_accessibility_palette([0.5, 0.5, 0.5], AccessibilityPalette::HighContrast);
        assert!((mid[0] - 0.3).abs() < EPS, "mid grey darkens: {mid:?}");
    }

    #[test]
    fn palette_transform_is_stable_over_domain() {
        // Property: every palette keeps every swept color in [0, 1] and finite.
        let palettes = [
            AccessibilityPalette::Natural,
            AccessibilityPalette::Deuteranopia,
            AccessibilityPalette::Protanopia,
            AccessibilityPalette::Tritanopia,
            AccessibilityPalette::HighContrast,
        ];
        for palette in palettes {
            for r in [0.0, 0.25, 0.5, 0.75, 1.0] {
                for g in [0.0, 0.33, 0.66, 1.0] {
                    for b in [0.0, 0.5, 1.0] {
                        let out = apply_accessibility_palette([r, g, b], palette);
                        for c in out {
                            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn health_factor_uses_canonical_scale_with_visibility_floor() {
        assert!(
            (health_factor(0.0) - 0.45).abs() < EPS,
            "dying stays visible"
        );
        assert!((health_factor(2.0) - 1.0).abs() < EPS, "full health");
        assert!(
            (health_factor(100.0) - 1.0).abs() < EPS,
            "legacy /100 bug class"
        );
        assert!((health_factor(-3.0) - 0.45).abs() < EPS, "negative clamps");
    }

    #[test]
    fn age_factor_desaturates_toward_a_floor() {
        assert!(
            (age_factor(0, 10_000) - 1.0).abs() < EPS,
            "newborns full color"
        );
        assert!(
            (age_factor(10_000, 10_000) - 0.82).abs() < 1.0e-4,
            "reference age reaches the floor"
        );
        assert!(
            (age_factor(99_999, 10_000) - 0.82).abs() < 1.0e-4,
            "ancient agents stay at the floor"
        );
        assert!(
            (age_factor(5, 0) - 1.0).abs() < EPS,
            "zero reference is a no-op"
        );
        let greyed = apply_saturation([0.9, 0.2, 0.3], age_factor(10_000, 10_000));
        assert!(
            (greyed[0] - greyed[1]).abs() < (0.9 - 0.2),
            "desaturation pulls channels toward luminance"
        );
        for c in greyed {
            assert!((0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn diet_stripe_endpoints_and_temperature_blend() {
        let carn = diet_stripe_color(0.0, 0.5);
        let herb = diet_stripe_color(1.0, 0.5);
        assert!(
            carn[0] > carn[1] && herb[1] > herb[0],
            "carnivore reads red, herbivore reads green: {carn:?} vs {herb:?}"
        );
        // Exact legacy value at herbivore endpoint with neutral temperature:
        // stripe = mix(CARNIVORE, HERBIVORE, 1) then 18% toward temp accent at 0.5.
        let temp_accent = mix_vec3(TEMP_COLD_RGB, TEMP_WARM_RGB, 0.5);
        let expected = mix_vec3(HERBIVORE_RGB, temp_accent, 0.18);
        assert_rgb_close(herb, expected, "herbivore endpoint");
        // Temperature extremes shift the stripe measurably.
        let cold = diet_stripe_color(0.5, 0.0);
        let warm = diet_stripe_color(0.5, 1.0);
        assert!(
            (cold[2] - warm[2]).abs() > 0.05,
            "temperature preference must tint the stripe: {cold:?} vs {warm:?}"
        );
    }

    #[test]
    fn agent_params_selection_ramp_and_clamps() {
        let base = AgentVisualInput {
            genome_color: [0.5, 0.6, 0.7],
            health: 2.0,
            ..AgentVisualInput::default()
        };
        let none = agent_visual_params(&base);
        let hovered = agent_visual_params(&AgentVisualInput {
            selection: VisualSelection::Hovered,
            ..base
        });
        let selected = agent_visual_params(&AgentVisualInput {
            selection: VisualSelection::Selected,
            ..base
        });
        assert!(
            none.body_emissive[2] < hovered.body_emissive[2]
                && hovered.body_emissive[2] < selected.body_emissive[2],
            "selection ramp increases emissive"
        );
        // Degenerate inputs never escape [0, 1].
        let wild = agent_visual_params(&AgentVisualInput {
            genome_color: [4.0, -1.0, f32::NAN],
            health: f32::INFINITY,
            wheel_left: 1e9,
            wheel_right: -1e9,
            food_delta: -1e9,
            sound_output: 1e9,
            sound_multiplier: 1e9,
            sound_level: 1e9,
            herbivore_tendency: -4.0,
            temperature_preference: 9.0,
            ..AgentVisualInput::default()
        });
        let all = [
            wild.body_color,
            wild.body_emissive,
            wild.stripe_color,
            wild.stripe_emissive,
            wild.mouth_color,
            wild.nose_color,
            wild.wheel_colors[0],
            wild.wheel_colors[1],
            wild.wheel_emissives[0],
            wild.wheel_emissives[1],
        ];
        for rgb in all {
            for c in rgb {
                assert!(
                    (0.0..=1.0).contains(&c) || c.is_finite() && c <= 1.0,
                    "visual params stay displayable: {rgb:?}"
                );
            }
        }
        assert!((0.0..=1.0).contains(&wild.mouth_activity));
        assert!((0.0..=1.0).contains(&wild.spike_readiness));
    }

    #[test]
    fn spike_readiness_tracks_extension_and_fractional_growth() {
        let mut input = AgentVisualInput {
            spike_length: 0.3,
            spike_extended: false,
            ..AgentVisualInput::default()
        };
        assert!((agent_visual_params(&input).spike_readiness - 0.3).abs() < EPS);
        input.spike_extended = true;
        assert!((agent_visual_params(&input).spike_readiness - 1.0).abs() < EPS);
    }

    #[test]
    fn terrain_base_colors_match_legacy_table() {
        assert_eq!(
            terrain_kind_base_color(TerrainKind::DeepWater),
            [0.117_647, 0.247_059, 0.4]
        );
        assert_eq!(
            terrain_kind_base_color(TerrainKind::Rock),
            [0.662_745, 0.694_118, 0.729_412]
        );
    }

    #[test]
    fn terrain_shading_stays_in_gamut_and_respects_daylight() {
        for kind in [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::Sand,
            TerrainKind::Grass,
            TerrainKind::Bloom,
            TerrainKind::Rock,
        ] {
            for daylight in [0.0, 0.3, 0.65, 1.0] {
                let color = terrain_shaded_color(&TerrainShadeInput {
                    kind,
                    moisture: 0.6,
                    elevation: 0.5,
                    slope: 0.4,
                    accent: 0.7,
                    daylight,
                });
                for c in color {
                    assert!(c.is_finite() && (0.0..=1.0).contains(&c));
                }
                let night = terrain_shaded_color(&TerrainShadeInput {
                    kind,
                    moisture: 0.6,
                    elevation: 0.5,
                    slope: 0.4,
                    accent: 0.7,
                    daylight: 0.0,
                });
                let noon = terrain_shaded_color(&TerrainShadeInput {
                    kind,
                    moisture: 0.6,
                    elevation: 0.5,
                    slope: 0.4,
                    accent: 0.7,
                    daylight: 1.0,
                });
                let night_lum: f32 = night.iter().sum();
                let noon_lum: f32 = noon.iter().sum();
                assert!(
                    noon_lum >= night_lum,
                    "{kind:?}: more daylight must not darken ({night_lum} vs {noon_lum})"
                );
            }
        }
    }

    #[test]
    fn daylight_curve_static_and_cyclic() {
        assert!((daylight_factor(999, 0, 0.0) - DAYLIGHT_STATIC).abs() < EPS);
        // Cycle: noon brightest, midnight at floor.
        let noon = daylight_factor(250, 1000, 0.0); // phase 0.25
        let midnight = daylight_factor(750, 1000, 0.0); // phase 0.75
        assert!((noon - 1.0).abs() < 1.0e-5, "noon is peak: {noon}");
        assert!(
            (midnight - DAYLIGHT_NIGHT_FLOOR).abs() < 1.0e-5,
            "midnight is floor: {midnight}"
        );
        // Determinism and wrap-around.
        assert_eq!(
            daylight_factor(1250, 1000, 0.0),
            daylight_factor(250, 1000, 0.0)
        );
        // start_phase shifts the curve (dawn phase != noon phase).
        assert_ne!(
            daylight_factor(0, 1000, 0.0),
            daylight_factor(0, 1000, 0.25)
        );
        // Range invariant over a full cycle.
        for tick in 0..1000 {
            let v = daylight_factor(tick, 1000, 0.3);
            assert!((DAYLIGHT_NIGHT_FLOOR..=1.0).contains(&v));
        }
    }

    #[test]
    fn food_ramp_endpoints_and_monotonic_green_channel() {
        assert_rgb_close(food_density_color(0.0), FOOD_SPARSE_RGB, "sparse");
        assert_rgb_close(food_density_color(1.0), FOOD_DENSE_RGB, "dense");
        assert_rgb_close(food_density_color(0.5), FOOD_MID_RGB, "mid");
        let mut prev = food_density_color(0.0)[2] + food_density_color(0.0)[0];
        for i in 1..=10 {
            let d = i as f32 / 10.0;
            let c = food_density_color(d);
            // Overall brightness rises with density.
            let lum = c[0] + c[1] + c[2];
            let prev_color = food_density_color((i - 1) as f32 / 10.0);
            let prev_lum = prev_color[0] + prev_color[1] + prev_color[2];
            assert!(lum >= prev_lum - EPS, "ramp brightens with density at {d}");
            prev = lum;
        }
        let _ = prev;
        // Clamps.
        assert_rgb_close(food_density_color(-1.0), FOOD_SPARSE_RGB, "clamp low");
        assert_rgb_close(food_density_color(2.0), FOOD_DENSE_RGB, "clamp high");
    }

    #[test]
    fn shimmer_is_deterministic_bounded_and_cell_specific() {
        let a = shimmer(42, 3, 7);
        let b = shimmer(42, 3, 7);
        assert_eq!(a, b, "same inputs, same shimmer");
        assert!((0.0..=1.0).contains(&a));
        let other_cell = shimmer(42, 4, 7);
        assert_ne!(a, other_cell, "neighboring cells desynchronize");
        // Tick wrap: phase repeats every period for the same cell.
        assert_eq!(
            shimmer(0, 3, 7),
            shimmer(SHIMMER_PERIOD_TICKS, 3, 7),
            "period wrap"
        );
        // Range over a full period.
        for tick in 0..SHIMMER_PERIOD_TICKS {
            let v = shimmer(tick, 11, 13);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn splat_weights_sum_to_one_across_the_domain() {
        for kind in [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::Sand,
            TerrainKind::Grass,
            TerrainKind::Bloom,
            TerrainKind::Rock,
        ] {
            for (elevation, slope, depth) in [
                (0.0, 0.0, 0.0),
                (0.1, 0.2, 0.0),
                (0.5, 0.7, 0.0),
                (0.9, 0.1, 0.0),
                (0.99, 0.95, 0.0),
                (0.3, 0.6, 1.5),
                (0.3, 0.6, 4.0),
                (0.1, 0.9, 10.0),
            ] {
                let w = splat_weights(&SplatInput {
                    kind,
                    elevation,
                    slope,
                    water_depth: depth,
                });
                let sum: f32 = w.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1.0e-5,
                    "{kind:?} e{elevation} s{slope} d{depth}: sum {sum}"
                );
                for v in w {
                    assert!(v.is_finite() && (0.0..=1.0).contains(&v));
                }
            }
        }
    }

    #[test]
    fn splat_one_hot_at_neutral_conditions() {
        let w = splat_weights(&SplatInput {
            kind: TerrainKind::Grass,
            elevation: 0.5,
            slope: 0.1,
            water_depth: 0.0,
        });
        assert!((w[3] - 1.0).abs() < EPS, "flat midland grass is pure grass");
        let w = splat_weights(&SplatInput {
            kind: TerrainKind::DeepWater,
            elevation: 0.0,
            slope: 0.9,
            water_depth: 0.0,
        });
        assert!((w[0] - 1.0).abs() < EPS, "water kinds ignore land rules");
    }

    #[test]
    fn splat_waterline_band_produces_sand() {
        let w = splat_weights(&SplatInput {
            kind: TerrainKind::Grass,
            elevation: 0.02,
            slope: 0.1,
            water_depth: 0.0,
        });
        assert!(
            w[2] > 0.5,
            "near-waterline grass blends mostly to sand: {w:?}"
        );
        let w_high = splat_weights(&SplatInput {
            kind: TerrainKind::Grass,
            elevation: 0.9,
            slope: 0.1,
            water_depth: 0.0,
        });
        assert!(w_high[2] < 0.01, "highland grass has no sand: {w_high:?}");
    }

    #[test]
    fn splat_steep_slopes_produce_rock() {
        let w = splat_weights(&SplatInput {
            kind: TerrainKind::Grass,
            elevation: 0.5,
            slope: 0.95,
            water_depth: 0.0,
        });
        assert!(w[5] > 0.6, "steep grass gives way to rock: {w:?}");
        let gentle = splat_weights(&SplatInput {
            kind: TerrainKind::Grass,
            elevation: 0.5,
            slope: 0.2,
            water_depth: 0.0,
        });
        assert!(gentle[5] < 0.01, "gentle grass keeps no rock: {gentle:?}");
    }

    #[test]
    fn splat_flooding_selects_depth_matched_water_layer() {
        let shallow = splat_weights(&SplatInput {
            kind: TerrainKind::Sand,
            elevation: 0.1,
            slope: 0.05,
            water_depth: 1.0,
        });
        assert!(
            shallow[1] > 0.2,
            "shallow flood adds shallow-water layer: {shallow:?}"
        );
        let deep = splat_weights(&SplatInput {
            kind: TerrainKind::Sand,
            elevation: 0.1,
            slope: 0.05,
            water_depth: 10.0,
        });
        assert!(
            deep[0] > 0.8,
            "deep flood replaces land with deep water: {deep:?}"
        );
    }

    #[test]
    fn biome_texture_baking_is_deterministic_and_opaque() {
        let a = bake_biome_texture(TerrainKind::Grass, 42, 64);
        let b = bake_biome_texture(TerrainKind::Grass, 42, 64);
        assert_eq!(a, b, "same inputs, byte-identical texture");
        assert_eq!(a.len(), 64 * 64 * 4);
        assert!(
            a.as_chunks::<4>().0.iter().all(|px| px[3] == 255),
            "fully opaque"
        );
        let other = bake_biome_texture(TerrainKind::Rock, 42, 64);
        assert_ne!(a, other, "biomes differ");
        let other_seed = bake_biome_texture(TerrainKind::Grass, 43, 64);
        assert_ne!(a, other_seed, "seeds differ");
        // Colors stay near the biome base color (modulation is bounded).
        let base = terrain_kind_base_color(TerrainKind::Grass);
        let spec_grass = biome_texture_specs()[3];
        for (i, px) in a.as_chunks::<4>().0.iter().enumerate().take(256) {
            for c in 0..3 {
                let v = f32::from(px[c]) / 255.0;
                let lo = base[c] * (1.0 - spec_grass.grain_amplitude) - 0.02;
                let hi = base[c] * (1.0 + spec_grass.grain_amplitude) + 0.02;
                assert!(
                    (lo..=hi).contains(&v),
                    "pixel {i} channel {c} out of grain bounds: {v} not in [{lo}, {hi}]"
                );
            }
        }
    }

    #[test]
    fn biome_atlas_layout_is_six_side_by_side_layers() {
        let atlas = bake_biome_atlas(7, 32);
        assert_eq!(atlas.len(), 32 * 32 * 6 * 4);
        // Layer 2 (sand) in the atlas equals the standalone sand bake.
        let sand = bake_biome_texture(TerrainKind::Sand, 7, 32);
        for y in 0..32_usize {
            let src = &sand[y * 32 * 4..(y + 1) * 32 * 4];
            let dst = &atlas[(y * 32 * 6 + 2 * 32) * 4..(y * 32 * 6 + 3 * 32) * 4];
            assert_eq!(src, dst, "row {y} of the sand layer matches");
        }
    }

    #[test]
    fn noise_is_bounded_and_deterministic() {
        for seed in [0_u64, 1, 42, u64::MAX] {
            for i in 0..64 {
                let v = value_noise_2d(seed, i as f32 * 0.37, i as f32 * 0.91);
                assert!(v.is_finite() && (-1.0..=1.0).contains(&v));
            }
        }
        assert_eq!(
            value_noise_2d(9, 3.25, 7.5),
            value_noise_2d(9, 3.25, 7.5),
            "deterministic"
        );
    }

    #[test]
    fn cue_table_is_exhaustive_and_sane() {
        let events = [
            WorldVisualEvent::Birth {
                origin: BirthOrigin::Born,
            },
            WorldVisualEvent::Birth {
                origin: BirthOrigin::Seeded,
            },
            WorldVisualEvent::Birth {
                origin: BirthOrigin::Injected,
            },
            WorldVisualEvent::Death {
                cause: DeathCause::CombatCarnivore,
            },
            WorldVisualEvent::Death {
                cause: DeathCause::CombatHerbivore,
            },
            WorldVisualEvent::Death {
                cause: DeathCause::Starvation,
            },
            WorldVisualEvent::Death {
                cause: DeathCause::Aging,
            },
            WorldVisualEvent::Death {
                cause: DeathCause::Unknown,
            },
            WorldVisualEvent::CombatHit { damage: 0.0 },
            WorldVisualEvent::CombatHit { damage: 100.0 },
            WorldVisualEvent::Eat { amount: 0.5 },
            WorldVisualEvent::Eat { amount: -0.2 },
            WorldVisualEvent::Reproduce,
            WorldVisualEvent::SpikeExtend,
        ];
        for event in events {
            let cue = visual_cue_for_event(&event);
            for c in cue.color.iter().chain(cue.accent_color.iter()) {
                assert!(c.is_finite() && (0.0..=1.0).contains(c));
            }
            assert!((0.0..=1.0).contains(&cue.intensity));
            assert!(cue.radius.is_finite() && cue.radius > 0.0);
            assert!(cue.duration_ticks > 0);
        }
    }

    #[test]
    fn combat_hit_intensity_scales_with_damage() {
        let tap = visual_cue_for_event(&WorldVisualEvent::CombatHit { damage: 0.1 });
        let slam = visual_cue_for_event(&WorldVisualEvent::CombatHit { damage: 100.0 });
        assert!(
            slam.intensity > tap.intensity && slam.radius > tap.radius,
            "bigger hits read bigger: {tap:?} vs {slam:?}"
        );
        assert!(slam.intensity <= 1.0);
    }

    #[test]
    fn birth_origins_are_visually_distinct() {
        let born = visual_cue_for_event(&WorldVisualEvent::Birth {
            origin: BirthOrigin::Born,
        });
        let seeded = visual_cue_for_event(&WorldVisualEvent::Birth {
            origin: BirthOrigin::Seeded,
        });
        let injected = visual_cue_for_event(&WorldVisualEvent::Birth {
            origin: BirthOrigin::Injected,
        });
        assert_ne!(born.color, seeded.color);
        assert_ne!(seeded.color, injected.color);
        assert!(
            born.intensity > seeded.intensity,
            "natural births outshine seeding"
        );
    }
}
