//! Renderer-neutral visual semantics (bd-2z0.14.3.2).
//!
//! One implementation of every "what should this look like" decision, consumed
//! identically by the Bevy 3D frontend, the `FrankenTUI` terminal canvas, the
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

// Exact floating-point equality is used only where the visual contract defines
// an exact identity value; deterministic arithmetic and casts are justified at
// their narrow call sites so new numerical operations remain lint-visible.
#![allow(clippy::float_cmp)]
#![allow(clippy::too_many_lines)]

use crate::{AccessibilityPalette, BirthOrigin, DeathCause, TerrainKind};

// ---------------------------------------------------------------------------
// Art direction (bd-9pqz / bd-l4gu).
// ---------------------------------------------------------------------------

/// Renderer-neutral sRGB triplet.
pub type Srgb = [f32; 3];

/// Physical material response shared by every renderer.
///
/// Colors remain bounded sRGB values while `emissive_gain` is an independent
/// HDR multiplier. Accessibility transforms therefore change chroma without
/// accidentally clamping bloom energy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaterialStyle {
    /// Base surface color.
    pub albedo_srgb: Srgb,
    /// Emissive surface color before HDR gain.
    pub emissive_srgb: Srgb,
    /// HDR multiplier applied after color-space/accessibility conversion.
    pub emissive_gain: f32,
    /// Perceptual roughness in `[0, 1]`.
    pub perceptual_roughness: f32,
    /// Specular reflectance in `[0, 1]`.
    pub reflectance: f32,
    /// Relative normal-map strength in `[0, 1]`.
    pub normal_strength: f32,
}

/// Terrain material alias documenting the canonical six-layer array.
pub type TerrainMaterialStyle = MaterialStyle;

/// Dark-field world substrate colors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubstrateStyle {
    /// Near-black outer field.
    pub abyss_srgb: Srgb,
    /// Primary world substrate.
    pub base_srgb: Srgb,
    /// Deeper blue-violet depth cue.
    pub depth_violet_srgb: Srgb,
    /// Distant atmospheric haze.
    pub distant_haze_srgb: Srgb,
}

/// Agent palette and visibility response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentStyle {
    /// Pure-herbivore end of the diet ramp.
    pub herbivore_srgb: Srgb,
    /// Pure-carnivore end of the diet ramp.
    pub carnivore_srgb: Srgb,
    /// Lowest health-driven luminance multiplier.
    pub health_luminance_floor: f32,
    /// Lowest age-driven saturation/luminance multiplier.
    pub age_luminance_floor: f32,
    /// Baseline HDR emissive gain.
    pub base_emissive_gain: f32,
    /// Hovered HDR emissive gain.
    pub hover_emissive_gain: f32,
    /// Selected HDR emissive gain.
    pub selected_emissive_gain: f32,
    /// Boosting HDR emissive gain.
    pub boost_emissive_gain: f32,
    /// Wheel material color.
    pub wheel_srgb: Srgb,
    /// Selected-agent rim color.
    pub selection_rim_srgb: Srgb,
    /// Extended-spike core color.
    pub spike_srgb: Srgb,
    /// Extended-spike HDR emissive gain.
    pub spike_emissive_gain: f32,
    /// Hearing-organ (ear) material color.
    ///
    /// Added by bd-sqji so the ear has an authority at all. Backends were each
    /// authoring their own; the value here is the one world-gfx already used, so
    /// adopting it MOVES authority without restyling anything. Choosing a
    /// different hue is a palette decision and belongs in its own change.
    pub ear_srgb: Srgb,
    /// Eye sclera color.
    ///
    /// See [`AgentStyle::ear_srgb`] on why this carries the pre-existing value.
    pub eye_sclera_srgb: Srgb,
    /// Eye pupil color.
    ///
    /// See [`AgentStyle::ear_srgb`] on why this carries the pre-existing value.
    pub eye_pupil_srgb: Srgb,
}

/// Stable-hue food mote response.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodStyle {
    /// Bright mote core.
    pub core_srgb: Srgb,
    /// Soft outer halo.
    pub halo_srgb: Srgb,
    /// Sparse-cell HDR gain.
    pub sparse_emissive_gain: f32,
    /// Dense-cell HDR gain.
    pub dense_emissive_gain: f32,
    /// Sparse-cell alpha.
    pub sparse_alpha: f32,
    /// Dense-cell alpha.
    pub dense_alpha: f32,
    /// Sparse-cell radius relative to a cell.
    pub sparse_radius: f32,
    /// Dense-cell radius relative to a cell.
    pub dense_radius: f32,
}

/// One event's two-tone HDR cue.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventCueStyle {
    /// Primary cue color.
    pub core_srgb: Srgb,
    /// Secondary cue color.
    pub accent_srgb: Srgb,
    /// HDR emissive gain.
    pub emissive_gain: f32,
    /// Deterministic lifetime in science ticks.
    pub duration_ticks: u32,
}

/// Complete world-event palette.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EventStyle {
    /// Spike impact flash.
    pub combat: EventCueStyle,
    /// New-agent bloom.
    pub birth: EventCueStyle,
    /// Fading death ember.
    pub death: EventCueStyle,
    /// Food-colored eating fleck.
    pub eat: EventCueStyle,
    /// Reproduction pulse.
    pub reproduce: EventCueStyle,
    /// Extended-spike flash.
    pub spike: EventCueStyle,
    /// Movement-boost motion trail.
    pub boost: EventCueStyle,
}

/// Shared application chrome tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterfaceStyle {
    /// Base panel surface.
    pub surface_srgb: Srgb,
    /// Raised panel surface.
    pub elevated_srgb: Srgb,
    /// Panel/control border.
    pub border_srgb: Srgb,
    /// Primary text.
    pub primary_text_srgb: Srgb,
    /// Secondary text.
    pub muted_text_srgb: Srgb,
    /// Cool data accent.
    pub accent_cyan_srgb: Srgb,
    /// Hot data accent.
    pub accent_magenta_srgb: Srgb,
    /// Warning state.
    pub warning_srgb: Srgb,
    /// Danger state.
    pub danger_srgb: Srgb,
}

/// Shared atmospheric/post-processing defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereStyle {
    /// Fog color.
    pub fog_srgb: Srgb,
    /// Linear HDR bloom threshold.
    pub bloom_threshold: f32,
    /// Bloom blend intensity.
    pub bloom_intensity: f32,
    /// Vignette strength.
    pub vignette: f32,
    /// Scene exposure.
    pub exposure: f32,
}

/// Versioned, renderer-neutral appearance contract.
///
/// This value is the sole literal authority for application colors. Renderers
/// may convert these values into backend-specific types, but may not author a
/// competing world palette.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualStyleV1 {
    /// World substrate.
    pub substrate: SubstrateStyle,
    /// Terrain materials in `TerrainKind` canonical order.
    pub terrain: [TerrainMaterialStyle; 6],
    /// Agent visual response.
    pub agents: AgentStyle,
    /// Food mote response.
    pub food: FoodStyle,
    /// Typed world-event response.
    pub events: EventStyle,
    /// Application chrome.
    pub chrome: InterfaceStyle,
    /// Atmosphere and post-processing defaults.
    pub atmosphere: AtmosphereStyle,
}

/// Bioluminescent dark-field microscopy art direction.
///
/// Terrain stays inside one blue-violet family and differentiates biomes by
/// value and material response. Agents and food carry scene luminance.
pub const BIOLUMINESCENT_DARK_FIELD_V1: VisualStyleV1 = {
    let herbivore = [0.18, 0.86, 1.00];
    let carnivore = [1.00, 0.60, 0.92];
    let food_core = [0.55, 1.00, 0.82];
    let food_halo = [0.18, 0.88, 0.70];
    let combat_core = [1.00, 0.22, 0.45];
    let spike_core = [1.00, 0.82, 0.96];
    VisualStyleV1 {
        substrate: SubstrateStyle {
            abyss_srgb: [0.010, 0.012, 0.028],
            base_srgb: [0.022, 0.028, 0.070],
            depth_violet_srgb: [0.045, 0.035, 0.110],
            distant_haze_srgb: [0.095, 0.075, 0.180],
        },
        terrain: [
            MaterialStyle {
                albedo_srgb: [0.018, 0.028, 0.075],
                emissive_srgb: [0.006, 0.014, 0.045],
                emissive_gain: 0.04,
                perceptual_roughness: 0.16,
                reflectance: 0.72,
                normal_strength: 0.45,
            },
            MaterialStyle {
                albedo_srgb: [0.028, 0.055, 0.115],
                emissive_srgb: [0.010, 0.025, 0.075],
                emissive_gain: 0.08,
                perceptual_roughness: 0.22,
                reflectance: 0.62,
                normal_strength: 0.55,
            },
            MaterialStyle {
                albedo_srgb: [0.105, 0.095, 0.145],
                emissive_srgb: [0.015, 0.012, 0.030],
                emissive_gain: 0.02,
                perceptual_roughness: 0.78,
                reflectance: 0.18,
                normal_strength: 0.60,
            },
            MaterialStyle {
                albedo_srgb: [0.070, 0.090, 0.130],
                emissive_srgb: [0.010, 0.018, 0.030],
                emissive_gain: 0.03,
                perceptual_roughness: 0.62,
                reflectance: 0.24,
                normal_strength: 0.72,
            },
            MaterialStyle {
                albedo_srgb: [0.095, 0.075, 0.150],
                emissive_srgb: [0.040, 0.025, 0.095],
                emissive_gain: 0.16,
                perceptual_roughness: 0.48,
                reflectance: 0.30,
                normal_strength: 0.80,
            },
            MaterialStyle {
                albedo_srgb: [0.060, 0.065, 0.095],
                emissive_srgb: [0.006, 0.006, 0.012],
                emissive_gain: 0.00,
                perceptual_roughness: 0.86,
                reflectance: 0.20,
                normal_strength: 1.00,
            },
        ],
        agents: AgentStyle {
            herbivore_srgb: herbivore,
            carnivore_srgb: carnivore,
            health_luminance_floor: 0.42,
            age_luminance_floor: 0.78,
            base_emissive_gain: 2.40,
            hover_emissive_gain: 3.20,
            selected_emissive_gain: 4.40,
            boost_emissive_gain: 5.20,
            wheel_srgb: [0.09, 0.12, 0.22],
            selection_rim_srgb: [0.72, 0.94, 1.00],
            spike_srgb: spike_core,
            spike_emissive_gain: 6.00,
            ear_srgb: [0.32, 0.62, 0.92],
            eye_sclera_srgb: [0.97, 0.98, 1.00],
            eye_pupil_srgb: [0.08, 0.11, 0.18],
        },
        food: FoodStyle {
            core_srgb: food_core,
            halo_srgb: food_halo,
            sparse_emissive_gain: 0.80,
            dense_emissive_gain: 2.80,
            sparse_alpha: 0.28,
            dense_alpha: 0.92,
            sparse_radius: 0.35,
            dense_radius: 1.15,
        },
        events: EventStyle {
            combat: EventCueStyle {
                core_srgb: combat_core,
                accent_srgb: [1.00, 0.78, 0.34],
                emissive_gain: 6.0,
                duration_ticks: 8,
            },
            birth: EventCueStyle {
                core_srgb: [0.55, 0.92, 1.00],
                accent_srgb: [0.86, 0.62, 1.00],
                emissive_gain: 3.5,
                duration_ticks: 24,
            },
            death: EventCueStyle {
                core_srgb: [1.00, 0.32, 0.12],
                accent_srgb: [0.24, 0.12, 0.22],
                emissive_gain: 2.2,
                duration_ticks: 36,
            },
            eat: EventCueStyle {
                core_srgb: food_core,
                accent_srgb: food_halo,
                emissive_gain: 1.8,
                duration_ticks: 12,
            },
            reproduce: EventCueStyle {
                core_srgb: [0.72, 0.70, 1.00],
                accent_srgb: [0.40, 0.92, 1.00],
                emissive_gain: 3.0,
                duration_ticks: 28,
            },
            spike: EventCueStyle {
                core_srgb: [1.00, 1.00, 1.00],
                accent_srgb: spike_core,
                emissive_gain: 5.5,
                duration_ticks: 6,
            },
            // Cool exhaust reading as SPEED rather than damage: deliberately far from
            // `combat` (hot magenta) and `spike` (white) so a boosting agent is never
            // mistaken for an attacking one at a glance. Short duration -- a trail that
            // outlives the boost reads as a smear.
            boost: EventCueStyle {
                core_srgb: [0.62, 0.96, 1.00],
                accent_srgb: herbivore,
                emissive_gain: 2.6,
                duration_ticks: 10,
            },
        },
        chrome: InterfaceStyle {
            surface_srgb: [0.025, 0.030, 0.070],
            elevated_srgb: [0.045, 0.050, 0.105],
            border_srgb: [0.16, 0.20, 0.32],
            primary_text_srgb: [0.88, 0.92, 1.00],
            muted_text_srgb: [0.52, 0.59, 0.72],
            accent_cyan_srgb: herbivore,
            accent_magenta_srgb: carnivore,
            warning_srgb: [1.00, 0.72, 0.28],
            danger_srgb: combat_core,
        },
        atmosphere: AtmosphereStyle {
            fog_srgb: [0.025, 0.030, 0.080],
            bloom_threshold: 0.70,
            bloom_intensity: 0.55,
            vignette: 0.35,
            exposure: 1.0,
        },
    }
};

/// Return the canonical visual style.
#[must_use]
pub const fn visual_style() -> &'static VisualStyleV1 {
    &BIOLUMINESCENT_DARK_FIELD_V1
}

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
pub const fn apply_accessibility_palette(rgb: [f32; 3], palette: AccessibilityPalette) -> [f32; 3] {
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
pub const HERBIVORE_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.agents.herbivore_srgb;
/// Carnivore end of the diet stripe ramp.
pub const CARNIVORE_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.agents.carnivore_srgb;
/// Legacy cold-accent alias; temperature no longer changes agent hue.
pub const TEMP_COLD_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.agents.herbivore_srgb;
/// Legacy warm-accent alias; temperature no longer changes agent hue.
pub const TEMP_WARM_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.agents.carnivore_srgb;
/// Neutral wheel body color.
pub const WHEEL_BASE_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.agents.wheel_srgb;

/// Selection/hover state shared by every frontend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VisualSelection {
    /// Neither hovered nor selected: baseline HDR gain (2.40).
    #[default]
    None,
    /// Pointer is over the agent: hover HDR gain (3.20).
    Hovered,
    /// Agent is the active selection: selected HDR gain (4.40).
    Selected,
}

impl VisualSelection {
    /// Canonical HDR emissive gain for this selection state.
    #[must_use]
    pub const fn emissive_gain(self) -> f32 {
        match self {
            Self::None => BIOLUMINESCENT_DARK_FIELD_V1.agents.base_emissive_gain,
            Self::Hovered => BIOLUMINESCENT_DARK_FIELD_V1.agents.hover_emissive_gain,
            Self::Selected => BIOLUMINESCENT_DARK_FIELD_V1.agents.selected_emissive_gain,
        }
    }

    /// Legacy name for [`Self::emissive_gain`].
    #[must_use]
    pub const fn highlight(self) -> f32 {
        self.emissive_gain()
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
    /// Genome-inherited color, retained only as a restrained lineage accent.
    pub genome_color: [f32; 3],
    /// Current health on the simulation's `0..=2` scale.
    pub health: f32,
    /// Current age in completed science ticks.
    pub age_ticks: u64,
    /// Age at which the canonical weathering floor is reached.
    pub reference_age_ticks: u64,
    /// Diet axis: 0 = pure carnivore, 1 = pure herbivore.
    pub herbivore_tendency: f32,
    /// Temperature preference axis. Retained as semantic input but does not
    /// change hue; diet is the sole body-color axis.
    pub temperature_preference: f32,
    /// Requested left/right wheel efforts (sign = direction).
    pub wheel_left: f32,
    /// Requested right wheel effort; same sign/magnitude convention as
    /// `wheel_left`.
    pub wheel_right: f32,
    /// Facing angle in radians (bd-grbc).
    ///
    /// Zero points along +X and the angle increases toward +Y, matching the convention the
    /// locomotion model pinned under bd-2i1. Core states it here so GPUI, Bevy, the TUI and
    /// WASM cannot each pick their own zero-angle and winding.
    pub heading: f32,
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
    /// Diet-led body color after health and age response (natural palette).
    pub body_color: [f32; 3],
    /// Body emissive chroma; selection changes gain/rim, not body hue.
    pub body_emissive: [f32; 3],
    /// HDR emissive gain, kept separate from clamped sRGB.
    pub body_emissive_gain: f32,
    /// Diet stripe color after age desaturation.
    pub stripe_color: [f32; 3],
    /// Stripe emissive chroma, with HDR energy carried separately.
    pub stripe_emissive: [f32; 3],
    /// Wheel base colors after speed brightening.
    pub wheel_colors: [[f32; 3]; 2],
    /// Wheel emissive colors (speed-scaled, cool-tinted).
    pub wheel_emissives: [[f32; 3]; 2],
    /// Mouth activity in `[0, 1]` (eat/yell/sound composite).
    pub mouth_activity: f32,
    /// Mouth color interpolated inside the canonical hot-event palette.
    pub mouth_color: [f32; 3],
    /// Nose tint from the smell trait.
    pub nose_color: [f32; 3],
    /// Spike readiness in `[0, 1]`: 1.0 when extended, else fractional growth.
    pub spike_readiness: f32,
    /// Canonical selected-agent rim color.
    pub selection_rim_color: [f32; 3],
    /// Canonical spike core color.
    pub spike_color: [f32; 3],
    /// Canonical spike HDR emissive gain.
    pub spike_emissive_gain: f32,
    /// Unit vector the agent faces (bd-grbc).
    ///
    /// SETTLED CONTRACT, confirmed by a real consumer. `scriptbots-world-gfx::AgentInstance`
    /// declares `heading: [f32; 2]` and the GPUI path assigns this value straight into it
    /// (`scriptbots-render/src/lib.rs`, `resolve_agent_visual` -> `AgentInstance`), so the
    /// vector is uploaded directly into a GPU instance buffer.
    ///
    /// KEEP IT A VECTOR. Returning radians instead would force every renderer to call
    /// `sin_cos` per agent per frame and to re-pick a zero-angle and winding direction, which
    /// is how four frontends end up disagreeing about which way an agent points. A 2x2 rotation
    /// matrix was the other candidate and is strictly more work for a consumer that only needs
    /// the axis. This is why the shape looks redundant with `AgentVisualInput::heading` and
    /// should stay that way: scalar in, vector out, computed once in core.
    pub facing: [f32; 2],
    /// Unit vector 90 degrees clockwise from `facing`, i.e. the agent's right-hand side
    /// (bd-grbc). Supplied so wheel placement does not depend on each renderer getting the
    /// perpendicular's sign right -- a mirrored agent is a bug nobody notices until the wheels
    /// disagree with the spike.
    pub right: [f32; 2],
    /// Offset from body centre to spike tip in world units (bd-grbc).
    ///
    /// `facing` scaled by the spike length, composed here so the length/direction pairing has
    /// a single definition.
    pub spike_tip_offset: [f32; 2],
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
/// Carnivore hot-magenta ↔ herbivore cool-cyan, mixed only by diet.
///
/// Temperature deliberately does not change hue: diet is the sole chromatic
/// semantic, while health and age carry luminance/saturation.
#[must_use]
pub const fn diet_stripe_color(herbivore_tendency: f32, _temperature_preference: f32) -> [f32; 3] {
    let herbivore = clamp01(herbivore_tendency);
    mix_vec3(CARNIVORE_RGB, HERBIVORE_RGB, herbivore)
}

/// The body health factor: the normalized `0..=2` health range mapped from the
/// style's visibility floor through full luminance.
///
/// The floor keeps even dying agents visible without flattening every low-health
/// agent into the same luminance class. The divisor is the simulation's
/// canonical `0..=2` health scale (a legacy Bevy HUD path once normalized
/// against 100 — that bug class is impossible here because this is the only
/// implementation).
#[must_use]
pub const fn health_factor(health: f32) -> f32 {
    if health.is_finite() {
        let normalized = (health / 2.0).clamp(0.0, 1.0);
        let floor = BIOLUMINESCENT_DARK_FIELD_V1.agents.health_luminance_floor;
        floor + (1.0 - floor) * normalized
    } else {
        BIOLUMINESCENT_DARK_FIELD_V1.agents.health_luminance_floor
    }
}

/// Age desaturation/weathering factor from the style floor through `1.0`.
///
/// Multiplies body saturation so elders read as seasoned rather than freshly
/// spawned; the floor keeps ancient agents colorful enough to stay
/// identifiable. `reference_age` is the age at which the factor reaches its
/// floor (frontends pass the observed maximum or a scenario constant); age
/// beyond it simply stays at the floor. Pure and deterministic.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    reason = "visual age ratios intentionally resolve into the renderer's f32 color contract"
)]
pub const fn age_factor(age_ticks: u64, reference_age: u64) -> f32 {
    if reference_age == 0 {
        return 1.0;
    }
    let t = (age_ticks as f32 / reference_age as f32).min(1.0);
    1.0 - (1.0 - BIOLUMINESCENT_DARK_FIELD_V1.agents.age_luminance_floor) * t
}

/// Apply a saturation factor (e.g. [`age_factor`]) to a color, preserving hue
/// and luminance bias toward the rec.709 grey point.
#[must_use]
pub const fn apply_saturation(rgb: [f32; 3], factor: f32) -> [f32; 3] {
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
#[allow(
    clippy::suboptimal_flops,
    reason = "pinned visual composition order is shared by every renderer and its goldens"
)]
pub fn agent_visual_params(input: &AgentVisualInput) -> AgentVisualParams {
    let genome_rgb = [
        clamp01(input.genome_color[0]),
        clamp01(input.genome_color[1]),
        clamp01(input.genome_color[2]),
    ];
    let hf = health_factor(input.health);
    let age = age_factor(input.age_ticks, input.reference_age_ticks);
    let diet = diet_stripe_color(input.herbivore_tendency, input.temperature_preference);
    let aged_diet = apply_saturation(diet, age);
    // Diet is the body authority. Genome color survives only as a restrained
    // lineage accent so it cannot erase the cyan-to-magenta semantic.
    let body_base = mix_vec3(aged_diet, genome_rgb, 0.08);
    let luminance = hf * age;
    let body_color = [
        body_base[0] * luminance,
        body_base[1] * luminance,
        body_base[2] * luminance,
    ];

    let body_emissive_gain = if input.boosting {
        BIOLUMINESCENT_DARK_FIELD_V1.agents.boost_emissive_gain
    } else {
        input.selection.emissive_gain()
    };
    let body_emissive = body_color;

    let stripe = aged_diet;
    let stripe_emissive = stripe;

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
    let mouth_color = mix_vec3(
        BIOLUMINESCENT_DARK_FIELD_V1.events.death.core_srgb,
        BIOLUMINESCENT_DARK_FIELD_V1.events.combat.core_srgb,
        mouth_activity,
    );
    let nose_color = mix_vec3(
        BIOLUMINESCENT_DARK_FIELD_V1.food.halo_srgb,
        BIOLUMINESCENT_DARK_FIELD_V1.food.core_srgb,
        clamp01(input.trait_smell * 0.4),
    );

    let spike_readiness = if input.spike_extended {
        1.0
    } else {
        clamp01(input.spike_length)
    };

    // bd-grbc: one definition of facing, so no renderer has to guess a zero-angle or winding
    // direction. `right` is `facing` rotated -90 degrees (clockwise in a +Y-up frame).
    let (sin_heading, cos_heading) = input.heading.sin_cos();
    let facing = [cos_heading, sin_heading];
    let right = [sin_heading, -cos_heading];
    let spike_tip_offset = [
        facing[0] * input.spike_length,
        facing[1] * input.spike_length,
    ];

    AgentVisualParams {
        body_color,
        body_emissive,
        body_emissive_gain,
        stripe_color: stripe,
        stripe_emissive,
        wheel_colors: [left_rgb, right_rgb],
        wheel_emissives: [left_emissive, right_emissive],
        mouth_activity,
        mouth_color,
        nose_color,
        spike_readiness,
        selection_rim_color: BIOLUMINESCENT_DARK_FIELD_V1.agents.selection_rim_srgb,
        spike_color: BIOLUMINESCENT_DARK_FIELD_V1.agents.spike_srgb,
        spike_emissive_gain: BIOLUMINESCENT_DARK_FIELD_V1.agents.spike_emissive_gain,
        facing,
        right,
        spike_tip_offset,
    }
}

// ---------------------------------------------------------------------------
// Terrain semantics (base palette lifted from the Bevy chunk renderer, which
// itself inherited the GPUI canvas hues; brightness factors made explicit).
// ---------------------------------------------------------------------------

/// Base color per terrain kind (natural palette, `sRGB` bytes as floats).
pub const TERRAIN_BASE_COLORS: [Srgb; 6] = [
    BIOLUMINESCENT_DARK_FIELD_V1.terrain[0].albedo_srgb,
    BIOLUMINESCENT_DARK_FIELD_V1.terrain[1].albedo_srgb,
    BIOLUMINESCENT_DARK_FIELD_V1.terrain[2].albedo_srgb,
    BIOLUMINESCENT_DARK_FIELD_V1.terrain[3].albedo_srgb,
    BIOLUMINESCENT_DARK_FIELD_V1.terrain[4].albedo_srgb,
    BIOLUMINESCENT_DARK_FIELD_V1.terrain[5].albedo_srgb,
];

const fn terrain_kind_index(kind: TerrainKind) -> usize {
    match kind {
        TerrainKind::DeepWater => 0,
        TerrainKind::ShallowWater => 1,
        TerrainKind::Sand => 2,
        TerrainKind::Grass => 3,
        TerrainKind::Bloom => 4,
        TerrainKind::Rock => 5,
    }
}

/// Canonical physical material for a terrain kind.
#[must_use]
pub const fn terrain_material(kind: TerrainKind) -> TerrainMaterialStyle {
    BIOLUMINESCENT_DARK_FIELD_V1.terrain[terrain_kind_index(kind)]
}

/// Base color for a terrain kind.
#[must_use]
pub const fn terrain_kind_base_color(kind: TerrainKind) -> [f32; 3] {
    terrain_material(kind).albedo_srgb
}

/// The daylight level used when the day/night cycle is off (the historical
/// constant both renderers hard-coded).
pub const DAYLIGHT_STATIC: f32 = 0.65;
/// Night floor of the day/night curve: scenes stay readable while emissives
/// carry the frame.
pub const DAYLIGHT_NIGHT_FLOOR: f32 = 0.15;

/// Default day/night cycle length in ticks for a run that does not choose one (bd-lhml).
///
/// Equal to the crate's `LEGACY_EPOCH_TICKS`, the epoch the agents' own CLOCK1/CLOCK2 sensor
/// channels run on (`next_tick % LEGACY_EPOCH_TICKS`). Tying the visible day to the sensed
/// clock means what the world LOOKS like and what an agent FEELS about time of day cannot
/// disagree — one epoch, two surfaces.
///
/// This exists because `render.day_night` defaulting to `None` meant [`daylight_factor`]
/// returned [`DAYLIGHT_STATIC`] for every tick of every run: the curve was live on the
/// interactive canvas and structurally incapable of moving. See
/// `tests/day_night_default_probe.rs`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "LEGACY_EPOCH_TICKS is the fixed 10,000-tick visual and sensor epoch"
)]
pub const DEFAULT_DAY_NIGHT_CYCLE_TICKS: u32 = crate::LEGACY_EPOCH_TICKS as u32;

/// Default starting phase: noon, so a fresh run opens at full light rather than mid-dusk.
pub const DEFAULT_DAY_NIGHT_START_PHASE: f32 = 0.25;

/// Resolve the effective `(cycle_ticks, start_phase)` a frontend should pass to
/// [`daylight_factor`].
///
/// ONE definition of what an unset day/night block means, so the GPUI canvas, the Bevy
/// renderer and the terminal cannot each invent their own fallback — which is exactly how the
/// static default survived: the renderer's inline `.map_or((0, 0.25), ..)` was the only place
/// the question was answered, and it answered "no cycle".
///
/// `None` for the whole block, or an absent `cycle_ticks`, now means the DEFAULT cycle.
/// An explicit `Some(0)` still means static lighting, so a run can deliberately freeze the
/// clock and that intent survives.
#[must_use]
pub fn resolve_day_night(cycle_ticks: Option<u32>, start_phase: Option<f32>) -> (u32, f32) {
    (
        cycle_ticks.unwrap_or(DEFAULT_DAY_NIGHT_CYCLE_TICKS),
        start_phase.unwrap_or(DEFAULT_DAY_NIGHT_START_PHASE),
    )
}

/// Shared day/night curve.
///
/// `cycle_ticks == 0` is the historical static lighting and returns
/// [`DAYLIGHT_STATIC`]. Otherwise the phase advances one full cycle per
/// `cycle_ticks` ticks (plus `start_phase` in `[0, 1)`), and the curve is a
/// cosine between [`DAYLIGHT_NIGHT_FLOOR`] at midnight and `1.0` at noon, so
/// the Bevy sun and the terminal tint can never disagree about time of day.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "the renderer consumes an f32 phase and the established curve order is golden-pinned"
)]
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
pub const fn terrain_shaded_color(input: &TerrainShadeInput) -> [f32; 3] {
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

/// Weight of the signed `fertility_bias` channel when it is folded into moisture.
///
/// `TerrainTile::fertility_bias` is a signed `[-1, 1]` food-fertility term, while
/// [`TerrainShadeInput::moisture`] is documented as a COMBINED moisture/fertility channel.
/// This is the constant that reconciles the two, and it lives here so no frontend has to
/// guess it (bd-1lls).
pub const FERTILITY_LUSHNESS_WEIGHT: f32 = 0.25;

/// Fold a tile's signed fertility bias into its moisture channel.
///
/// The result is what [`TerrainShadeInput::moisture`] means: not raw ground wetness, but how
/// *lush* the tile reads. Fertile ground looks greener than its moisture alone implies, and
/// barren ground looks drier; [`FERTILITY_LUSHNESS_WEIGHT`] sets how much.
///
/// Both inputs are clamped defensively, so a caller sampling a partially initialized field
/// cannot push the shading path out of gamut.
#[must_use]
pub fn terrain_lushness(moisture: f32, fertility_bias: f32) -> f32 {
    let moisture = clamp01(moisture);
    let bias = if fertility_bias.is_finite() {
        fertility_bias.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    clamp01(moisture + bias * FERTILITY_LUSHNESS_WEIGHT)
}

/// Direction the key light comes FROM, in world-space XY. Normalized.
///
/// Down-right, so slopes facing up-left catch the light. Fixed rather than orbiting with
/// `daylight`: a rotating key light would make every terrain golden non-reproducible across
/// the tick at which it was captured, and the day/night signal is carried by intensity
/// ([`daylight_factor`]) instead.
pub const TERRAIN_LIGHT_DIR_XY: [f32; 2] = [0.554_7, -0.832_05];

/// Vertical component of the key light at full daylight.
///
/// The light lowers toward the horizon as `daylight` falls, which lengthens the apparent
/// shading gradient at dusk without ever letting the term reach zero.
pub const TERRAIN_LIGHT_HEIGHT: f32 = 0.85;

/// How far the normal-lighting term is allowed to push a tile's brightness.
///
/// The multiplier is BOUNDED on purpose: terrain lighting modulates an already-shaded color
/// from [`terrain_shaded_color`], so an unbounded term would silently blow out the palette
/// that bd-9pqz established.
pub const TERRAIN_LIGHT_FACTOR_RANGE: (f32, f32) = (0.72, 1.28);

/// Bounded brightness multiplier from surface normal versus the key light.
///
/// `gradient` is the elevation slope `[dh/dx, dh/dy]` in elevation-units per world-unit, as a
/// frontend would compute it from neighbouring samples. The surface normal is reconstructed as
/// `normalize([-gx * s, -gy * s, 1])`, where `s` is the per-kind
/// [`MaterialStyle::normal_strength`] — so rock reads as craggy and water reads as flat
/// from the same gradient, which is the whole reason this is keyed on `kind`.
///
/// The return value is a MULTIPLIER centered near 1.0 and clamped to
/// [`TERRAIN_LIGHT_FACTOR_RANGE`], intended to scale a color that
/// [`terrain_shaded_color`] already produced. It is not a lighting model in its own right and
/// deliberately has no ambient/specular terms; those belong to whichever renderer wants them.
///
/// Every constant it depends on lives in this module (bd-1lls): the renderer supplies geometry
/// and gets appearance back, and there is exactly one definition of what a lit slope looks like.
#[must_use]
#[allow(
    clippy::suboptimal_flops,
    reason = "normal-light evaluation order is shared renderer output, not an FMA optimization site"
)]
pub fn terrain_normal_light_factor(kind: TerrainKind, gradient: [f32; 2], daylight: f32) -> f32 {
    let (gx, gy) = match (gradient[0].is_finite(), gradient[1].is_finite()) {
        (true, true) => (gradient[0], gradient[1]),
        _ => return 1.0,
    };
    let daylight = clamp01(daylight);
    let strength = terrain_material(kind).normal_strength;

    // Surface normal from the height gradient.
    let nx = -gx * strength;
    let ny = -gy * strength;
    let n_len = (nx * nx + ny * ny + 1.0).sqrt();

    // Key light: fixed azimuth, elevation rising with daylight so dusk rakes across slopes.
    let lz = (TERRAIN_LIGHT_HEIGHT * daylight).max(0.15);
    let l_len = (TERRAIN_LIGHT_DIR_XY[0] * TERRAIN_LIGHT_DIR_XY[0]
        + TERRAIN_LIGHT_DIR_XY[1] * TERRAIN_LIGHT_DIR_XY[1]
        + lz * lz)
        .sqrt();

    let n_dot_l =
        (nx * TERRAIN_LIGHT_DIR_XY[0] + ny * TERRAIN_LIGHT_DIR_XY[1] + lz) / (n_len * l_len);

    // A flat tile has n_dot_l == lz/l_len; that case must map to exactly 1.0 so flat ground is
    // never darkened or brightened by merely being lit.
    let flat_reference = lz / l_len;
    let (lo, hi) = TERRAIN_LIGHT_FACTOR_RANGE;
    (1.0 + (n_dot_l - flat_reference)).clamp(lo, hi)
}

// ---------------------------------------------------------------------------
// Food semantics: density ramp + deterministic shimmer phase.
// ---------------------------------------------------------------------------

/// Legacy sparse-color alias; density changes visibility, not hue.
pub const FOOD_SPARSE_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.food.core_srgb;
/// Legacy midpoint-color alias; density changes visibility, not hue.
pub const FOOD_MID_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.food.core_srgb;
/// Legacy dense-color alias; density changes visibility, not hue.
pub const FOOD_DENSE_RGB: Srgb = BIOLUMINESCENT_DARK_FIELD_V1.food.core_srgb;

/// Fully resolved food-mote presentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodVisualParams {
    /// Stable mint-cyan core.
    pub core_srgb: Srgb,
    /// Stable mint-cyan halo.
    pub halo_srgb: Srgb,
    /// HDR emissive gain.
    pub emissive_gain: f32,
    /// Display alpha.
    pub alpha: f32,
    /// Radius relative to one terrain cell.
    pub relative_radius: f32,
}

/// Resolve food visibility from normalized density without changing its hue.
#[must_use]
pub const fn food_visual_params(density: f32) -> FoodVisualParams {
    let d = clamp01(density);
    let food = BIOLUMINESCENT_DARK_FIELD_V1.food;
    FoodVisualParams {
        core_srgb: food.core_srgb,
        halo_srgb: food.halo_srgb,
        emissive_gain: food.sparse_emissive_gain
            + (food.dense_emissive_gain - food.sparse_emissive_gain) * d,
        alpha: food.sparse_alpha + (food.dense_alpha - food.sparse_alpha) * d,
        relative_radius: food.sparse_radius + (food.dense_radius - food.sparse_radius) * d,
    }
}

/// Canonical food core color.
///
/// The parameter remains part of this established API, but density now drives
/// alpha, radius, and HDR energy through [`food_visual_params`] rather than
/// changing hue from green to gold.
#[must_use]
pub const fn food_density_color(density: f32) -> [f32; 3] {
    food_visual_params(density).core_srgb
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "coordinate bytes and the hash modulo are explicitly bounded by the stable phase codec"
)]
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
#[allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    reason = "both modulo operands are at most 120 and the replay-stable sine order is pinned"
)]
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
    Birth {
        /// How the agent entered the world (spawn, reproduction, etc.); selects the cue color.
        origin: BirthOrigin,
    },
    /// An agent died. Effect follows the cause.
    Death {
        /// What killed the agent (combat, starvation, age); selects the cue family.
        cause: DeathCause,
    },
    /// A spike connected. `damage` is the dealt amount (pre-normalization).
    CombatHit {
        /// Damage dealt by the connecting spike, pre-normalization; drives shard intensity.
        damage: f32,
    },
    /// An agent ate. `amount` is the intake delta.
    Eat {
        /// Energy intake delta from the bite; scales the nibble fleck effect.
        amount: f32,
    },
    /// A reproduction pulse (distinct from the child's Birth cue).
    Reproduce,
    /// A spike began extending (telegraph).
    SpikeExtend,
    /// Movement boost engaged this tick.
    ///
    /// Defined here rather than left to the renderer (WildDuck's `vfx.rs` question): boost is
    /// a real, observable agent action with an existing output channel, so it gets a canonical
    /// cue like every other action. Leaving it undefined is what produces a second palette --
    /// the renderer would have had to invent a colour, and then core and GPUI would disagree
    /// about what "boosting" looks like.
    Boost {
        /// Engaged drive magnitude in `[0, 1]`; scales trail length and brightness.
        magnitude: f32,
    },
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
    /// Short motion trail drawn BEHIND the agent along its heading (movement boost).
    ///
    /// Distinct from every other family: it is directional and persistent-per-tick rather
    /// than a one-shot burst, so a renderer must place it using the agent's heading rather
    /// than centring it on the body.
    BoostTrail,
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

const fn normalized_event_gain(style: EventCueStyle) -> f32 {
    (style.emissive_gain / BIOLUMINESCENT_DARK_FIELD_V1.events.combat.emissive_gain).clamp(0.0, 1.0)
}

/// Resolve a world event to its visual cue.
///
/// Pure table: same event, same cue, on every renderer. Returns the canonical
/// colors/durations from the art bible (bd-2z0.14.3.6); intensity is scaled
/// by caller-measured magnitudes where applicable.
#[must_use]
#[allow(
    clippy::suboptimal_flops,
    reason = "event cue arithmetic is renderer-neutral golden output with pinned evaluation order"
)]
pub fn visual_cue_for_event(event: &WorldVisualEvent) -> VisualCue {
    match *event {
        WorldVisualEvent::Birth { origin } => {
            let style = BIOLUMINESCENT_DARK_FIELD_V1.events.birth;
            let (color, intensity) = match origin {
                BirthOrigin::Born => (style.core_srgb, normalized_event_gain(style)),
                BirthOrigin::Seeded => (
                    mix_vec3(style.core_srgb, style.accent_srgb, 0.45),
                    normalized_event_gain(style) * 0.75,
                ),
                BirthOrigin::Injected => (style.accent_srgb, normalized_event_gain(style) * 0.9),
            };
            VisualCue {
                kind: VisualCueKind::Sparkle,
                color,
                accent_color: style.accent_srgb,
                intensity,
                radius: 6.0,
                duration_ticks: style.duration_ticks,
            }
        }
        WorldVisualEvent::Death { cause } => {
            let style = BIOLUMINESCENT_DARK_FIELD_V1.events.death;
            let combat = matches!(
                cause,
                DeathCause::CombatCarnivore | DeathCause::CombatHerbivore
            );
            VisualCue {
                kind: if combat {
                    VisualCueKind::Shards
                } else {
                    VisualCueKind::Wilt
                },
                color: style.core_srgb,
                accent_color: style.accent_srgb,
                intensity: normalized_event_gain(style) * if combat { 1.0 } else { 0.75 },
                radius: if combat { 8.0 } else { 5.0 },
                duration_ticks: style.duration_ticks,
            }
        }
        WorldVisualEvent::CombatHit { damage } => {
            let style = BIOLUMINESCENT_DARK_FIELD_V1.events.combat;
            let normalized = clamp01(damage / COMBAT_DAMAGE_REFERENCE);
            VisualCue {
                kind: VisualCueKind::SparkCone,
                color: style.core_srgb,
                accent_color: style.accent_srgb,
                intensity: 0.5 + 0.5 * normalized,
                radius: 4.0 + 4.0 * normalized,
                duration_ticks: style.duration_ticks,
            }
        }
        WorldVisualEvent::Eat { amount } => {
            let style = BIOLUMINESCENT_DARK_FIELD_V1.events.eat;
            let normalized = clamp01(amount.abs());
            VisualCue {
                kind: VisualCueKind::Nibble,
                color: style.core_srgb,
                accent_color: style.accent_srgb,
                intensity: normalized_event_gain(style) * (0.75 + 0.25 * normalized),
                radius: 2.0,
                duration_ticks: style.duration_ticks,
            }
        }
        WorldVisualEvent::Reproduce => {
            let style = BIOLUMINESCENT_DARK_FIELD_V1.events.reproduce;
            VisualCue {
                kind: VisualCueKind::PulseRing,
                color: style.core_srgb,
                accent_color: style.accent_srgb,
                intensity: normalized_event_gain(style),
                radius: 10.0,
                duration_ticks: style.duration_ticks,
            }
        }
        WorldVisualEvent::SpikeExtend => {
            let style = BIOLUMINESCENT_DARK_FIELD_V1.events.spike;
            VisualCue {
                kind: VisualCueKind::Flash,
                color: style.core_srgb,
                accent_color: style.accent_srgb,
                intensity: normalized_event_gain(style),
                radius: 3.0,
                duration_ticks: style.duration_ticks,
            }
        }
        WorldVisualEvent::Boost { magnitude } => {
            let style = BIOLUMINESCENT_DARK_FIELD_V1.events.boost;
            // `radius` is the trail LENGTH for this family, laid along the agent's heading
            // rather than a radius about its body. Scaled by magnitude so a hard boost reads
            // as a longer streak; the floor keeps a light boost visible rather than absent.
            let magnitude = clamp01(magnitude);
            VisualCue {
                kind: VisualCueKind::BoostTrail,
                color: style.core_srgb,
                accent_color: style.accent_srgb,
                intensity: normalized_event_gain(style) * (0.45 + 0.55 * magnitude),
                radius: 4.0 + 6.0 * magnitude,
                duration_ticks: style.duration_ticks,
            }
        }
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

/// Final renderer-neutral terrain-color inputs for one sample.
///
/// `splat_weights` may be the direct result of [`splat_weights`] for a cell, or a
/// convex interpolation of neighbouring cell weights for a continuous fragment
/// sample. Keeping those weights explicit lets a smooth renderer preserve
/// biome-boundary interpolation without reimplementing the color composition.
#[derive(Debug, Clone, Copy)]
pub struct TerrainSurfaceInput {
    /// Weights over [`TERRAIN_BASE_COLORS`] order.
    pub splat_weights: [f32; SPLAT_LAYERS],
    /// Combined moisture/fertility channel in `[0, 1]`.
    pub moisture: f32,
    /// Normalized elevation channel in `[0, 1]`.
    pub elevation: f32,
    /// Local slope magnitude in `[0, 1]`.
    pub slope: f32,
    /// Deterministic per-sample accent.
    pub accent: f32,
    /// Tick-derived daylight level.
    pub daylight: f32,
    /// Final display accessibility transform.
    pub accessibility: AccessibilityPalette,
}

// ---------------------------------------------------------------------------
// Per-pixel terrain sampling (bd-grbc, settled)
// ---------------------------------------------------------------------------

/// Borrowed view of the terrain fields a shading pass needs (bd-grbc).
///
/// `TerrainShadeInput` and `SplatInput` describe ONE cell, so a fragment path had no way to ask
/// "what are the fields at world (x, y)?" and would have had to invent its own interpolation
/// between cell centres. Two renderers would invent two, and neither would match the CPU
/// semantic raster the goldens compare against.
///
/// This is a borrowed view rather than a `WorldState` method on purpose: it keeps appearance
/// logic in `visual.rs`, it is testable headlessly without building a world, and it does not
/// require touching `lib.rs`.
#[derive(Debug, Clone, Copy)]
pub struct TerrainFieldView<'a> {
    /// Grid width in cells.
    pub width: u32,
    /// Grid height in cells.
    pub height: u32,
    /// World units per cell.
    pub cell_size: f32,
    /// Per-cell terrain kind, row-major, `width * height` entries.
    pub kinds: &'a [TerrainKind],
    /// Per-cell moisture/fertility in `[0, 1]`.
    pub moisture: &'a [f32],
    /// Per-cell normalized elevation in `[0, 1]`.
    pub elevation: &'a [f32],
    /// Per-cell slope magnitude in `[0, 1]`.
    pub slope: &'a [f32],
    /// Per-cell hydrology water depth in world units, `>= 0`.
    pub water_depth: &'a [f32],
}

/// The four cells surrounding a sample point, with their bilinear weights (bd-grbc).
///
/// OPTION B of bd-grbc. Returned when the caller wants to interpolate on the GPU: sample this
/// once per cell on the CPU, upload the corners, and let the fragment shader blend. Prefer this
/// over [`terrain_fields_at`] if the draw path is per-pixel, since it avoids a CPU call per
/// pixel while keeping the CORNER SELECTION and SEAM WRAP in core.
#[derive(Debug, Clone, Copy)]
pub struct TerrainSampleCorners {
    /// Flat indices of the four surrounding cells: `[x0y0, x1y0, x0y1, x1y1]`.
    pub indices: [usize; 4],
    /// Bilinear weights matching `indices`, summing to 1.
    pub weights: [f32; 4],
}

impl TerrainFieldView<'_> {
    fn cell_count(&self) -> usize {
        (self.width as usize).saturating_mul(self.height as usize)
    }

    /// Wrap a cell coordinate onto the torus.
    ///
    /// Uses `rem_euclid` rather than a single add/subtract correction. bd-b09u and bd-p095 both
    /// proved this codebase gets minimum-image arithmetic wrong whenever a site rolls its own,
    /// and a sampler is exactly such a site.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "rem_euclid bounds the result below the u32 extent before conversion to usize"
    )]
    fn wrap_cell(coordinate: i64, extent: u32) -> usize {
        if extent == 0 {
            return 0;
        }
        (coordinate.rem_euclid(i64::from(extent))) as usize
    }

    /// The four surrounding cells and their bilinear weights at world position `(x, y)`.
    ///
    /// Sample points are taken at CELL CENTRES, so a point exactly at a centre returns that cell
    /// with weight 1. The seam wraps.
    ///
    /// This is OPTION B of bd-grbc and the HOT path: the caller receives indices and weights and
    /// does the blend itself, which is what a fragment shader wants. Prefer it per-pixel.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "finite floor coordinates intentionally enter the integer torus-wrap boundary"
    )]
    pub fn sample_corners(&self, x: f32, y: f32) -> TerrainSampleCorners {
        if self.cell_count() == 0 || self.cell_size <= 0.0 || !x.is_finite() || !y.is_finite() {
            return TerrainSampleCorners {
                indices: [0; 4],
                weights: [1.0, 0.0, 0.0, 0.0],
            };
        }
        // Shift by half a cell so integer coordinates land on centres.
        let gx = x / self.cell_size - 0.5;
        let gy = y / self.cell_size - 0.5;
        let x0 = gx.floor();
        let y0 = gy.floor();
        let tx = gx - x0;
        let ty = gy - y0;

        let cx0 = Self::wrap_cell(x0 as i64, self.width);
        let cy0 = Self::wrap_cell(y0 as i64, self.height);
        let cx1 = Self::wrap_cell(x0 as i64 + 1, self.width);
        let cy1 = Self::wrap_cell(y0 as i64 + 1, self.height);

        let w = self.width as usize;
        TerrainSampleCorners {
            indices: [cy0 * w + cx0, cy0 * w + cx1, cy1 * w + cx0, cy1 * w + cx1],
            weights: [
                (1.0 - tx) * (1.0 - ty),
                tx * (1.0 - ty),
                (1.0 - tx) * ty,
                tx * ty,
            ],
        }
    }

    fn blend(values: &[f32], corners: &TerrainSampleCorners) -> f32 {
        let mut total = 0.0;
        for (index, weight) in corners.indices.iter().zip(corners.weights.iter()) {
            total += values.get(*index).copied().unwrap_or(0.0) * weight;
        }
        total
    }

    /// Bilinearly sampled shading inputs at world position `(x, y)`.
    ///
    /// OPTION A of bd-grbc. Convenient for CPU shading and for the semantic reference raster;
    /// use [`Self::sample_corners`] instead if the fragment shader should do the blending.
    ///
    /// Both options are kept deliberately. This one exists so the golden lane and the shader
    /// share a single definition of "the fields at (x, y)"; deleting it as redundant would put
    /// the reference raster back to inventing its own sampling.
    ///
    /// `kind` is NEAREST, not blended -- a terrain kind is categorical, and interpolating an
    /// enum discriminant would be meaningless. `splat_weights` is the mechanism for smooth
    /// transitions between biomes.
    #[must_use]
    pub fn shade_input_at(&self, x: f32, y: f32, daylight: f32, accent: f32) -> TerrainShadeInput {
        let corners = self.sample_corners(x, y);
        let dominant = corners
            .weights
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (slot, weight)| {
                if *weight > best.1 {
                    (slot, *weight)
                } else {
                    best
                }
            })
            .0;
        let kind = self
            .kinds
            .get(corners.indices[dominant])
            .copied()
            .unwrap_or(TerrainKind::Grass);
        TerrainShadeInput {
            kind,
            moisture: Self::blend(self.moisture, &corners),
            elevation: Self::blend(self.elevation, &corners),
            slope: Self::blend(self.slope, &corners),
            accent,
            daylight,
        }
    }

    /// Bilinearly sampled splat inputs at world position `(x, y)`.
    ///
    /// OPTION A of bd-grbc. Same nearest-kind rule as [`Self::shade_input_at`].
    #[must_use]
    pub fn splat_input_at(&self, x: f32, y: f32) -> SplatInput {
        let corners = self.sample_corners(x, y);
        let dominant = corners
            .weights
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (slot, weight)| {
                if *weight > best.1 {
                    (slot, *weight)
                } else {
                    best
                }
            })
            .0;
        SplatInput {
            kind: self
                .kinds
                .get(corners.indices[dominant])
                .copied()
                .unwrap_or(TerrainKind::Grass),
            elevation: Self::blend(self.elevation, &corners),
            slope: Self::blend(self.slope, &corners),
            water_depth: Self::blend(self.water_depth, &corners),
        }
    }
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

/// Compose the final semantic terrain color in display-referred sRGB.
///
/// This is the shared color boundary for GPUI, Bevy, and world-gfx. Renderers
/// may rasterize, light, and post-process differently, but none may own a
/// competing six-layer blend or accessibility transform.
#[must_use]
pub fn terrain_surface_srgb(input: &TerrainSurfaceInput) -> [f32; 3] {
    const KINDS: [TerrainKind; SPLAT_LAYERS] = [
        TerrainKind::DeepWater,
        TerrainKind::ShallowWater,
        TerrainKind::Sand,
        TerrainKind::Grass,
        TerrainKind::Bloom,
        TerrainKind::Rock,
    ];

    let mut rgb = [0.0_f32; 3];
    for (kind, weight) in KINDS.into_iter().zip(input.splat_weights) {
        let shaded = terrain_shaded_color(&TerrainShadeInput {
            kind,
            moisture: input.moisture,
            elevation: input.elevation,
            slope: input.slope,
            accent: input.accent,
            daylight: input.daylight,
        });
        rgb[0] += shaded[0] * weight;
        rgb[1] += shaded[1] * weight;
        rgb[2] += shaded[2] * weight;
    }
    apply_accessibility_palette(rgb, input.accessibility)
}

/// Blend `amount` of total weight into `layer`, taking proportionally from
/// all other layers so the sum stays 1.
#[allow(
    clippy::suboptimal_flops,
    reason = "splat interpolation order is shared shader-reference output and remains bit-pinned"
)]
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    reason = "the stable lattice codec and interpolation order are covered by byte-hash goldens"
)]
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
        let top_left = lattice(seed, ix, iy);
        let top_right = lattice(seed, ix + 1, iy);
        let bottom_left = lattice(seed, ix, iy + 1);
        let bottom_right = lattice(seed, ix + 1, iy + 1);
        let top = top_left + (top_right - top_left) * fx;
        let bottom = bottom_left + (bottom_right - bottom_left) * fx;
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
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded texture coordinates and clamped RGBA channels intentionally narrow to the output codec"
)]
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
#[allow(
    clippy::cast_possible_truncation,
    reason = "the retained u32 texture size is round-tripped only for the existing atlas codec"
)]
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

    /// bd-lhml: the resolved default must actually move the curve.
    ///
    /// The whole defect was a default that produced a constant, so the constant asserted here
    /// is the fix itself: resolving an unset block must yield a cycle whose daylight varies.
    #[test]
    fn the_resolved_default_day_night_cycle_actually_varies() {
        let (cycle_ticks, start_phase) = resolve_day_night(None, None);
        assert_eq!(cycle_ticks, DEFAULT_DAY_NIGHT_CYCLE_TICKS);
        assert!(
            cycle_ticks > 0,
            "an unset day/night block must select a real cycle"
        );

        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for step in 0..64u64 {
            let tick = step * u64::from(cycle_ticks) / 64;
            let daylight = daylight_factor(tick, cycle_ticks, start_phase);
            minimum = minimum.min(daylight);
            maximum = maximum.max(daylight);
        }
        assert!(
            maximum - minimum > 0.5,
            "the default cycle must sweep a real range, got {minimum}..={maximum}"
        );
    }

    /// An explicit zero must still mean static lighting.
    ///
    /// Without this, "give the default a cycle" would silently remove a run's ability to
    /// freeze the clock on purpose — turning one lost intent into another.
    #[test]
    fn an_explicit_zero_cycle_still_means_static_lighting() {
        let (cycle_ticks, _) = resolve_day_night(Some(0), None);
        assert_eq!(cycle_ticks, 0, "explicit zero must survive resolution");
        assert!(
            (daylight_factor(12_345, cycle_ticks, 0.25) - DAYLIGHT_STATIC).abs() < EPS,
            "a deliberately frozen clock must still return DAYLIGHT_STATIC"
        );
    }

    /// bd-1lls: flat ground must not be tinted by merely being lit.
    ///
    /// This is the property that makes the helper safe to multiply into an already-shaded
    /// color. If a zero gradient returned anything but exactly 1.0, every renderer adopting it
    /// would shift the whole bd-9pqz palette by a constant and the goldens would move for a
    /// reason nobody could name.
    #[test]
    fn flat_terrain_is_neutral_under_normal_lighting_at_every_kind_and_daylight() {
        for kind in [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::Sand,
            TerrainKind::Grass,
            TerrainKind::Bloom,
            TerrainKind::Rock,
        ] {
            for daylight in [0.0, 0.15, 0.5, DAYLIGHT_STATIC, 1.0] {
                let factor = terrain_normal_light_factor(kind, [0.0, 0.0], daylight);
                assert!(
                    (factor - 1.0).abs() < EPS,
                    "flat {kind:?} at daylight {daylight} returned {factor}, expected exactly 1.0"
                );
            }
        }
    }

    /// A slope facing the key light must be brighter than the same slope facing away, and both
    /// must stay inside the declared bound so the palette cannot be blown out.
    #[test]
    fn normal_lighting_is_directional_and_stays_within_its_declared_bound() {
        let (lo, hi) = TERRAIN_LIGHT_FACTOR_RANGE;
        // TERRAIN_LIGHT_DIR_XY is (+x, -y), so a normal tilted toward (+x, -y) faces the light.
        // The normal is -gradient, hence a gradient of (-1, +1) tilts the normal INTO the light.
        let toward = terrain_normal_light_factor(TerrainKind::Rock, [-1.0, 1.0], 1.0);
        let away = terrain_normal_light_factor(TerrainKind::Rock, [1.0, -1.0], 1.0);
        assert!(
            toward > away,
            "a slope facing the key light ({toward}) must be brighter than one facing away \
             ({away})"
        );
        for (label, factor) in [("toward", toward), ("away", away)] {
            assert!(
                (lo..=hi).contains(&factor),
                "{label} slope produced {factor}, outside TERRAIN_LIGHT_FACTOR_RANGE {lo}..={hi}"
            );
        }
    }

    /// Rock declares a higher `normal_strength` than water, so identical geometry must produce
    /// a stronger lighting response on rock. This is the reason the helper takes `kind` at all.
    #[test]
    fn normal_lighting_respects_per_kind_normal_strength() {
        let gradient = [-0.6, 0.6];
        let rock = terrain_normal_light_factor(TerrainKind::Rock, gradient, 1.0);
        let water = terrain_normal_light_factor(TerrainKind::DeepWater, gradient, 1.0);
        assert!(
            terrain_material(TerrainKind::Rock).normal_strength
                > terrain_material(TerrainKind::DeepWater).normal_strength,
            "fixture assumption broken: rock must declare more normal_strength than deep water"
        );
        assert!(
            (rock - 1.0).abs() > (water - 1.0).abs(),
            "rock ({rock}) must respond more strongly than deep water ({water}) to the same \
             gradient, because it declares a higher normal_strength"
        );
    }

    /// Non-finite geometry must not propagate into a color.
    #[test]
    fn normal_lighting_rejects_non_finite_gradients() {
        for gradient in [
            [f32::NAN, 0.0],
            [0.0, f32::NAN],
            [f32::INFINITY, 0.0],
            [0.0, f32::NEG_INFINITY],
        ] {
            let factor = terrain_normal_light_factor(TerrainKind::Grass, gradient, 1.0);
            assert!(
                (factor - 1.0).abs() < EPS,
                "gradient {gradient:?} must fall back to the neutral 1.0, got {factor}"
            );
        }
    }

    /// bd-1lls: fertility moves lushness in the direction of its sign, and the result stays a
    /// legal `[0, 1]` channel even when both inputs are pushed to their extremes.
    #[test]
    fn lushness_folds_fertility_in_signed_and_stays_in_range() {
        let neutral = terrain_lushness(0.5, 0.0);
        assert!(
            (neutral - 0.5).abs() < EPS,
            "zero bias must pass moisture through"
        );
        assert!(
            terrain_lushness(0.5, 1.0) > neutral,
            "positive fertility must read lusher"
        );
        assert!(
            terrain_lushness(0.5, -1.0) < neutral,
            "negative fertility must read drier"
        );
        for moisture in [-1.0, 0.0, 0.5, 1.0, 2.0] {
            for bias in [-2.0, -1.0, 0.0, 1.0, 2.0, f32::NAN] {
                let lushness = terrain_lushness(moisture, bias);
                assert!(
                    (0.0..=1.0).contains(&lushness),
                    "terrain_lushness({moisture}, {bias}) = {lushness} escaped [0, 1]"
                );
            }
        }
    }

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

    fn luminance(rgb: [f32; 3]) -> f32 {
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    }

    #[test]
    fn dark_field_style_is_finite_in_gamut_and_materially_coherent() {
        let style = visual_style();
        let colors = [
            style.substrate.abyss_srgb,
            style.substrate.base_srgb,
            style.substrate.depth_violet_srgb,
            style.substrate.distant_haze_srgb,
            style.food.core_srgb,
            style.food.halo_srgb,
            style.agents.herbivore_srgb,
            style.agents.carnivore_srgb,
            style.agents.wheel_srgb,
            style.agents.selection_rim_srgb,
            style.agents.spike_srgb,
            style.chrome.surface_srgb,
            style.chrome.elevated_srgb,
            style.chrome.border_srgb,
            style.chrome.primary_text_srgb,
            style.chrome.muted_text_srgb,
            style.chrome.accent_cyan_srgb,
            style.chrome.accent_magenta_srgb,
            style.chrome.warning_srgb,
            style.chrome.danger_srgb,
            style.atmosphere.fog_srgb,
        ];
        for rgb in colors {
            for channel in rgb {
                assert!(
                    channel.is_finite() && (0.0..=1.0).contains(&channel),
                    "style channel must be finite sRGB: {rgb:?}"
                );
            }
        }
        for material in style.terrain {
            for channel in material
                .albedo_srgb
                .into_iter()
                .chain(material.emissive_srgb)
            {
                assert!(channel.is_finite() && (0.0..=1.0).contains(&channel));
            }
            assert!(material.emissive_gain.is_finite() && material.emissive_gain >= 0.0);
            assert!((0.0..=1.0).contains(&material.perceptual_roughness));
            assert!((0.0..=1.0).contains(&material.reflectance));
            assert!((0.0..=1.0).contains(&material.normal_strength));
        }

        let deep_water = terrain_material(TerrainKind::DeepWater);
        let shallow_water = terrain_material(TerrainKind::ShallowWater);
        let land = [
            terrain_material(TerrainKind::Sand),
            terrain_material(TerrainKind::Grass),
            terrain_material(TerrainKind::Bloom),
            terrain_material(TerrainKind::Rock),
        ];
        let darkest_land = land
            .iter()
            .map(|material| luminance(material.albedo_srgb))
            .fold(f32::INFINITY, f32::min);
        let smoothest_land = land
            .iter()
            .map(|material| material.perceptual_roughness)
            .fold(f32::INFINITY, f32::min);
        let most_reflective_land = land
            .iter()
            .map(|material| material.reflectance)
            .fold(f32::NEG_INFINITY, f32::max);
        for water in [deep_water, shallow_water] {
            assert!(
                luminance(water.albedo_srgb) < darkest_land,
                "water must remain darker than land"
            );
            assert!(
                water.perceptual_roughness < smoothest_land,
                "water must remain smoother than land"
            );
            assert!(
                water.reflectance > most_reflective_land,
                "water must remain more reflective than land"
            );
        }

        let event_styles = [
            style.events.combat,
            style.events.birth,
            style.events.death,
            style.events.eat,
            style.events.reproduce,
            style.events.spike,
        ];
        for event in event_styles {
            for channel in event.core_srgb.into_iter().chain(event.accent_srgb) {
                assert!(channel.is_finite() && (0.0..=1.0).contains(&channel));
            }
            assert!(event.emissive_gain.is_finite() && event.emissive_gain >= 0.0);
            assert!(event.duration_ticks > 0);
        }

        let food = style.food;
        assert!(food.sparse_emissive_gain.is_finite() && food.sparse_emissive_gain >= 0.0);
        assert!(food.dense_emissive_gain >= food.sparse_emissive_gain);
        assert!((0.0..=1.0).contains(&food.sparse_alpha));
        assert!((food.sparse_alpha..=1.0).contains(&food.dense_alpha));
        assert!(food.sparse_radius.is_finite() && food.sparse_radius > 0.0);
        assert!(food.dense_radius >= food.sparse_radius);

        let agent_gains = [
            style.agents.base_emissive_gain,
            style.agents.hover_emissive_gain,
            style.agents.selected_emissive_gain,
            style.agents.boost_emissive_gain,
            style.agents.spike_emissive_gain,
        ];
        assert!(
            agent_gains
                .iter()
                .all(|gain| gain.is_finite() && *gain >= 0.0)
        );
        assert!(agent_gains.windows(2).all(|pair| pair[0] < pair[1]));

        let atmosphere = style.atmosphere;
        for scalar in [
            atmosphere.bloom_threshold,
            atmosphere.bloom_intensity,
            atmosphere.vignette,
            atmosphere.exposure,
        ] {
            assert!(scalar.is_finite() && scalar >= 0.0);
        }
    }

    #[test]
    fn terrain_material_and_splat_layer_orders_are_identical() {
        let kinds = [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::Sand,
            TerrainKind::Grass,
            TerrainKind::Bloom,
            TerrainKind::Rock,
        ];
        for (expected_index, kind) in kinds.into_iter().enumerate() {
            assert_eq!(
                terrain_material(kind),
                BIOLUMINESCENT_DARK_FIELD_V1.terrain[expected_index]
            );
            let weights = splat_weights(&SplatInput {
                kind,
                elevation: 0.5,
                slope: 0.1,
                water_depth: 0.0,
            });
            let dominant = weights
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .expect("six canonical terrain layers");
            assert_eq!(
                dominant, expected_index,
                "{kind:?} material and splat slots must stay aligned"
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
            (health_factor(0.0) - BIOLUMINESCENT_DARK_FIELD_V1.agents.health_luminance_floor).abs()
                < EPS,
            "dying stays visible"
        );
        assert!((health_factor(2.0) - 1.0).abs() < EPS, "full health");
        assert!(
            (health_factor(100.0) - 1.0).abs() < EPS,
            "legacy /100 bug class"
        );
        assert!(
            (health_factor(-3.0) - BIOLUMINESCENT_DARK_FIELD_V1.agents.health_luminance_floor)
                .abs()
                < EPS,
            "negative clamps"
        );
    }

    #[test]
    fn age_factor_desaturates_toward_a_floor() {
        assert!(
            (age_factor(0, 10_000) - 1.0).abs() < EPS,
            "newborns full color"
        );
        assert!(
            (age_factor(10_000, 10_000) - BIOLUMINESCENT_DARK_FIELD_V1.agents.age_luminance_floor)
                .abs()
                < 1.0e-4,
            "reference age reaches the floor"
        );
        assert!(
            (age_factor(99_999, 10_000) - BIOLUMINESCENT_DARK_FIELD_V1.agents.age_luminance_floor)
                .abs()
                < 1.0e-4,
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
    fn diet_endpoints_are_luminance_matched_and_temperature_neutral() {
        let carn = diet_stripe_color(0.0, 0.5);
        let herb = diet_stripe_color(1.0, 0.5);
        assert_rgb_close(carn, CARNIVORE_RGB, "carnivore endpoint");
        assert_rgb_close(herb, HERBIVORE_RGB, "herbivore endpoint");
        assert!(
            (luminance(carn) - luminance(herb)).abs() <= 0.03,
            "diet must change hue rather than implied health: {carn:?} vs {herb:?}"
        );
        let cold = diet_stripe_color(0.5, 0.0);
        let warm = diet_stripe_color(0.5, 1.0);
        assert_rgb_close(cold, warm, "temperature-neutral diet hue");
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
            none.body_emissive_gain < hovered.body_emissive_gain
                && hovered.body_emissive_gain < selected.body_emissive_gain,
            "selection ramp increases HDR emissive gain"
        );
        let boosted = agent_visual_params(&AgentVisualInput {
            boosting: true,
            ..base
        });
        assert_eq!(
            boosted.body_emissive_gain,
            BIOLUMINESCENT_DARK_FIELD_V1.agents.boost_emissive_gain
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
    fn health_and_age_change_luminance_without_reordering_diet_hues() {
        let params = |health, age_ticks, herbivore_tendency| {
            agent_visual_params(&AgentVisualInput {
                genome_color: diet_stripe_color(herbivore_tendency, 0.5),
                health,
                age_ticks,
                reference_age_ticks: 10_000,
                herbivore_tendency,
                ..AgentVisualInput::default()
            })
        };

        let dying = params(0.0, 0, 0.5);
        let recovering = params(1.0, 0, 0.5);
        let healthy = params(2.0, 0, 0.5);
        assert!(luminance(dying.body_color) < luminance(recovering.body_color));
        assert!(luminance(recovering.body_color) < luminance(healthy.body_color));

        let newborn = params(2.0, 0, 0.5);
        let middle_aged = params(2.0, 5_000, 0.5);
        let old = params(2.0, 10_000, 0.5);
        assert!(luminance(newborn.body_color) > luminance(middle_aged.body_color));
        assert!(luminance(middle_aged.body_color) > luminance(old.body_color));
        let chroma = |rgb: Srgb| {
            rgb.into_iter().fold(f32::NEG_INFINITY, f32::max)
                - rgb.into_iter().fold(f32::INFINITY, f32::min)
        };
        assert!(chroma(newborn.body_color) > chroma(old.body_color));

        // Weathering must not collapse health into an unreadable low-end
        // plateau. These are the exact full/mid/floor classes exercised by the
        // controlled 1280x720 GPUI proof for bd-ydym.
        let old_full = params(2.0, 10_000, 0.5);
        let old_mid = params(1.2, 10_000, 0.5);
        let old_floor = params(0.05, 10_000, 0.5);
        assert!(
            luminance(old_full.body_color) > luminance(old_mid.body_color) + 0.10,
            "old full-health agents must remain brighter than mid-health agents"
        );
        assert!(
            luminance(old_mid.body_color) > luminance(old_floor.body_color) + 0.10,
            "old mid-health agents must remain brighter than floor-health agents"
        );

        for age_ticks in [0, 5_000, 10_000] {
            for health in [0.0, 1.0, 2.0] {
                let carnivore = params(health, age_ticks, 0.0);
                let herbivore = params(health, age_ticks, 1.0);
                assert!(carnivore.body_color[0] > herbivore.body_color[0]);
                assert!(herbivore.body_color[1] > carnivore.body_color[1]);
            }
        }
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
    fn terrain_base_colors_are_aliases_into_the_style() {
        assert_eq!(
            terrain_kind_base_color(TerrainKind::DeepWater),
            BIOLUMINESCENT_DARK_FIELD_V1.terrain[0].albedo_srgb
        );
        assert_eq!(
            terrain_kind_base_color(TerrainKind::Rock),
            BIOLUMINESCENT_DARK_FIELD_V1.terrain[5].albedo_srgb
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
    fn food_keeps_one_hue_while_visibility_and_energy_rise() {
        assert_rgb_close(food_density_color(0.0), FOOD_SPARSE_RGB, "sparse");
        assert_rgb_close(food_density_color(1.0), FOOD_DENSE_RGB, "dense");
        assert_rgb_close(food_density_color(0.5), FOOD_MID_RGB, "mid");
        let mut previous = food_visual_params(0.0);
        for i in 1..=10 {
            let d = i as f32 / 10.0;
            let current = food_visual_params(d);
            assert_eq!(
                current.core_srgb, previous.core_srgb,
                "density must not change food hue"
            );
            assert!(current.alpha >= previous.alpha);
            assert!(current.emissive_gain >= previous.emissive_gain);
            assert!(current.relative_radius >= previous.relative_radius);
            previous = current;
        }
        assert_rgb_close(food_density_color(-1.0), FOOD_SPARSE_RGB, "clamp low");
        assert_rgb_close(food_density_color(2.0), FOOD_DENSE_RGB, "clamp high");
    }

    #[test]
    fn weakest_agent_remains_brighter_than_the_brightest_resolved_terrain() {
        let weak = agent_visual_params(&AgentVisualInput {
            genome_color: [0.0; 3],
            health: 0.0,
            age_ticks: 10_000,
            reference_age_ticks: 10_000,
            herbivore_tendency: 0.5,
            ..AgentVisualInput::default()
        });
        let mut brightest_terrain = f32::NEG_INFINITY;
        for kind in [
            TerrainKind::DeepWater,
            TerrainKind::ShallowWater,
            TerrainKind::Sand,
            TerrainKind::Grass,
            TerrainKind::Bloom,
            TerrainKind::Rock,
        ] {
            for moisture in [0.0, 1.0] {
                for elevation in [0.0, 1.0] {
                    for slope in [0.0, 1.0] {
                        for accent in [0.0, 1.0] {
                            for daylight in [0.0, 1.0] {
                                brightest_terrain = brightest_terrain.max(luminance(
                                    terrain_shaded_color(&TerrainShadeInput {
                                        kind,
                                        moisture,
                                        elevation,
                                        slope,
                                        accent,
                                        daylight,
                                    }),
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            luminance(weak.body_color) > brightest_terrain,
            "figure/ground invariant: {:?} vs {brightest_terrain}",
            weak.body_color
        );
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

    // --- bd-grbc settled surface -------------------------------------------------------
    //
    // These cover the orientation contract and the world-space terrain sampler. Both were
    // offered as proposals and have since been adopted by real consumers, so the surface is
    // settled, not provisional: scriptbots-world-gfx::AgentInstance takes `heading` straight
    // from `facing`, and scriptbots-render calls `splat_weights` / `TerrainFieldView` /
    // `sample_corners` on its live paths. The tests below are therefore a contract, not a
    // sketch — changing what they assert changes what those consumers already rely on.

    /// Orientation must be ONE stated convention, not something each renderer re-derives.
    #[test]
    fn agent_facing_and_right_follow_one_stated_convention() {
        let mut input = AgentVisualInput {
            heading: 0.0,
            spike_length: 2.0,
            ..AgentVisualInput::default()
        };

        // Heading 0 points along +X; `right` is 90 degrees clockwise, i.e. -Y.
        let params = agent_visual_params(&input);
        assert!(
            (params.facing[0] - 1.0).abs() < 1e-6,
            "facing {:?}",
            params.facing
        );
        assert!(params.facing[1].abs() < 1e-6, "facing {:?}", params.facing);
        assert!(
            params.right[1] < 0.0,
            "right must be clockwise, got {:?}",
            params.right
        );

        // Quarter turn toward +Y.
        input.heading = core::f32::consts::FRAC_PI_2;
        let params = agent_visual_params(&input);
        assert!(params.facing[0].abs() < 1e-6, "facing {:?}", params.facing);
        assert!(
            (params.facing[1] - 1.0).abs() < 1e-6,
            "facing {:?}",
            params.facing
        );

        // Both vectors stay unit length and perpendicular right around the circle.
        for step in 0..16 {
            input.heading = step as f32 * core::f32::consts::TAU / 16.0;
            let p = agent_visual_params(&input);
            let fl = p.facing[0].hypot(p.facing[1]);
            let rl = p.right[0].hypot(p.right[1]);
            let dot = p.facing[0] * p.right[0] + p.facing[1] * p.right[1];
            assert!((fl - 1.0).abs() < 1e-5, "facing not unit at step {step}");
            assert!((rl - 1.0).abs() < 1e-5, "right not unit at step {step}");
            assert!(
                dot.abs() < 1e-5,
                "facing/right not perpendicular at step {step}"
            );
        }
    }

    /// The spike tip is composed from the same facing the body uses, so a renderer cannot pair
    /// a length with a direction of its own.
    #[test]
    fn spike_tip_offset_is_facing_scaled_by_length() {
        let input = AgentVisualInput {
            heading: core::f32::consts::FRAC_PI_2,
            spike_length: 3.0,
            ..AgentVisualInput::default()
        };
        let p = agent_visual_params(&input);
        assert!(p.spike_tip_offset[0].abs() < 1e-5);
        assert!((p.spike_tip_offset[1] - 3.0).abs() < 1e-5);
    }

    fn ramp_fields() -> (Vec<TerrainKind>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let kinds = vec![TerrainKind::Grass; 16];
        let moisture = vec![0.5; 16];
        let mut elevation = vec![0.0; 16];
        for y in 0..4 {
            for x in 0..4 {
                elevation[y * 4 + x] = x as f32 / 3.0;
            }
        }
        (kinds, moisture, elevation, vec![0.25; 16], vec![0.0; 16])
    }

    /// Sampling exactly at a cell centre returns that cell rather than a blend of neighbours.
    #[test]
    fn sampling_a_cell_centre_returns_that_cell() {
        let (kinds, moisture, elevation, slope, water) = ramp_fields();
        let view = TerrainFieldView {
            width: 4,
            height: 4,
            cell_size: 10.0,
            kinds: &kinds,
            moisture: &moisture,
            elevation: &elevation,
            slope: &slope,
            water_depth: &water,
        };
        // Centre of cell (2, 1) is world (25, 15).
        let corners = view.sample_corners(25.0, 15.0);
        let dominant = corners.weights.iter().copied().fold(0.0f32, f32::max);
        assert!(
            (dominant - 1.0).abs() < 1e-5,
            "a centre sample must be unblended, weights {:?}",
            corners.weights
        );
        let input = view.shade_input_at(25.0, 15.0, 1.0, 0.0);
        assert!(
            (input.elevation - 2.0 / 3.0).abs() < 1e-5,
            "got {}",
            input.elevation
        );
    }

    /// The sampler must wrap at the toroidal seam. bd-b09u and bd-p095 both caught this
    /// codebase getting minimum-image arithmetic wrong at exactly such sites.
    #[test]
    fn sampling_wraps_across_the_toroidal_seam() {
        let (kinds, moisture, elevation, slope, water) = ramp_fields();
        let view = TerrainFieldView {
            width: 4,
            height: 4,
            cell_size: 10.0,
            kinds: &kinds,
            moisture: &moisture,
            elevation: &elevation,
            slope: &slope,
            water_depth: &water,
        };

        // Just past the right edge must blend column 3 with column 0, not clamp.
        let corners = view.sample_corners(39.0, 15.0);
        assert!(
            corners.indices.iter().any(|i| i % 4 == 0),
            "seam sample must reach column 0, got {:?}",
            corners.indices
        );

        // Weights form a partition of unity everywhere, including far off-grid coordinates.
        for (x, y) in [
            (0.0f32, 0.0f32),
            (39.9, 39.9),
            (-25.0, -15.0),
            (400.0, 400.0),
        ] {
            let c = view.sample_corners(x, y);
            let total: f32 = c.weights.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "weights at ({x},{y}) sum to {total}"
            );
            for index in c.indices {
                assert!(index < 16, "index {index} out of range at ({x},{y})");
            }
        }
    }

    /// Terrain kind is categorical, so it is nearest-sampled rather than blended: interpolating
    /// an enum discriminant would be meaningless. splat_weights is the smooth-transition path.
    #[test]
    fn terrain_kind_is_nearest_not_interpolated() {
        let (mut kinds, moisture, elevation, slope, water) = ramp_fields();
        kinds[0] = TerrainKind::Rock;
        let view = TerrainFieldView {
            width: 4,
            height: 4,
            cell_size: 10.0,
            kinds: &kinds,
            moisture: &moisture,
            elevation: &elevation,
            slope: &slope,
            water_depth: &water,
        };
        assert_eq!(
            view.shade_input_at(5.0, 5.0, 1.0, 0.0).kind,
            TerrainKind::Rock
        );
        assert_eq!(view.splat_input_at(5.0, 5.0).kind, TerrainKind::Rock);
    }
}
