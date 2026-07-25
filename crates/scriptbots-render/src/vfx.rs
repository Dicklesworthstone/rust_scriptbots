//! Deterministic, presentation-only world-event effects.
//!
//! This module deliberately knows nothing about [`scriptbots_core::WorldState`].
//! A completed science step supplies immutable, located events; the projection
//! retains them by science tick and the painter reads snapshots. Repainting,
//! pausing, or opening a second window therefore cannot advance an effect or
//! mutate the simulation.

use std::sync::Arc;

use gpui::{PathBuilder, Window, point, px};
use scriptbots_core::visual::{VisualCue, VisualCueKind, WorldVisualEvent, visual_cue_for_event};
use scriptbots_core::{AgentUid, Position};

use crate::camera::CameraSnapshot;
use crate::{
    ColorPaletteMode, append_arc_polyline, append_circle_polygon, apply_palette,
    rgba_from_triplet_with_alpha,
};

/// Maximum number of simultaneously retained presentation effects.
///
/// The core event batch is bounded independently. This second bound protects a
/// slow or paused frontend when many long-lived cues overlap.
pub(crate) const MAX_ACTIVE_VFX: usize = 8_192;

/// Renderer-level effect family.
///
/// Core owns the cue vocabulary and all colors. This label only selects a
/// geometric painter; it must never become another palette.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum VfxFamily {
    Birth,
    Death,
    CombatHit,
    Eat,
    Reproduce,
    SpikeExtend,
    /// Dormant geometry until core defines a canonical boost event and cue.
    ///
    /// Render must not invent boost colors, intensity, radius, or lifetime.
    BoostTrail,
}

impl VfxFamily {
    /// Resolve the geometric family from the core-owned cue kind.
    #[must_use]
    pub(crate) const fn from_cue(cue: &VisualCue) -> Self {
        match cue.kind {
            VisualCueKind::Sparkle => Self::Birth,
            VisualCueKind::Shards | VisualCueKind::Wilt => Self::Death,
            VisualCueKind::Nibble => Self::Eat,
            VisualCueKind::SparkCone => Self::CombatHit,
            VisualCueKind::PulseRing => Self::Reproduce,
            VisualCueKind::Flash => Self::SpikeExtend,
        }
    }

    const fn rank(self) -> u8 {
        match self {
            Self::Birth => 0,
            Self::Death => 1,
            Self::CombatHit => 2,
            Self::Eat => 3,
            Self::Reproduce => 4,
            Self::SpikeExtend => 5,
            Self::BoostTrail => 6,
        }
    }
}

/// One located, already-resolved cue from a completed science tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LocatedVfx {
    started_tick: u64,
    /// Primary accepted-event order within `started_tick`.
    ///
    /// The full immutable payload is a deterministic tie-breaker if an upstream
    /// producer accidentally repeats an ordinal.
    ordinal: u32,
    source: Option<AgentUid>,
    target: Option<AgentUid>,
    position: Position,
    /// Unit direction in world coordinates, or `[0, 0]` for radial effects.
    direction: [f32; 2],
    family: VfxFamily,
    cue: VisualCue,
}

impl LocatedVfx {
    /// Resolve and locate one core-owned world event.
    #[must_use]
    pub(crate) fn from_world_event(
        started_tick: u64,
        ordinal: u32,
        source: Option<AgentUid>,
        target: Option<AgentUid>,
        position: Position,
        direction: [f32; 2],
        event: &WorldVisualEvent,
    ) -> Self {
        Self::from_cue(
            started_tick,
            ordinal,
            source,
            target,
            position,
            direction,
            visual_cue_for_event(event),
        )
    }

    /// Construct a located cue while containing malformed direction input.
    fn from_cue(
        started_tick: u64,
        ordinal: u32,
        source: Option<AgentUid>,
        target: Option<AgentUid>,
        position: Position,
        direction: [f32; 2],
        cue: VisualCue,
    ) -> Self {
        Self {
            started_tick,
            ordinal,
            source,
            target,
            position,
            direction: normalized_direction(direction),
            family: VfxFamily::from_cue(&cue),
            cue,
        }
    }

    /// Integer age at or after a science boundary.
    ///
    /// A future event has no age. Keeping that state explicit prevents a
    /// saturating subtraction from presenting it as a just-started event.
    #[must_use]
    pub(crate) const fn age_at(&self, tick: u64) -> Option<u64> {
        tick.checked_sub(self.started_tick)
    }

    /// Normalized lifetime progress in `[0, 1]`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn progress_at(&self, tick: u64) -> Option<f32> {
        let age = self.age_at(tick)?;
        let duration = self.cue.duration_ticks.max(1);
        Some((age as f32 / duration as f32).clamp(0.0, 1.0))
    }

    /// Quadratic fade used by the emissive overlay.
    #[must_use]
    pub(crate) fn fade_at(&self, tick: u64) -> Option<f32> {
        self.progress_at(tick).map(|progress| {
            let remaining = 1.0 - progress;
            remaining * remaining
        })
    }

    const fn is_live_at(&self, tick: u64) -> bool {
        tick >= self.started_tick && tick - self.started_tick < u64::from(self.cue.duration_ticks)
    }
}

/// Shared immutable effect list materialized at one science boundary.
///
/// Keeping the tick beside the events prevents a caller from repainting an
/// old snapshot with a newer effect age.
#[derive(Debug, Clone)]
pub(crate) struct VfxFrame {
    tick: u64,
    events: Arc<[LocatedVfx]>,
}

impl VfxFrame {
    #[must_use]
    pub(crate) fn empty(tick: u64) -> Self {
        Self {
            tick,
            events: Arc::default(),
        }
    }

    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.events.len()
    }

    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// Successfully painted effect counts returned by the two effect passes.
///
/// Captures use this as a non-vacuity rail: a pixel delta must correspond to a
/// production effect family, not an unrelated animation elsewhere in the frame.
/// A pass records an effect only when at least one of its paths was built and
/// painted. Test instrumentation merges the passes with a per-family maximum,
/// so dual-pass birth/death effects are not counted twice. This is deliberately
/// a conservative visibility rail rather than a scientific accounting stream.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct VfxPaintCounts {
    by_family: [u32; 7],
}

impl VfxPaintCounts {
    fn record(&mut self, family: VfxFamily) {
        let slot = usize::from(family.rank());
        self.by_family[slot] = self.by_family[slot].saturating_add(1);
    }

    fn record_if_painted(&mut self, family: VfxFamily, painted: bool) {
        if painted {
            self.record(family);
        }
    }

    /// Number of painted effects from one family.
    #[cfg(test)]
    pub(crate) const fn family(self, family: VfxFamily) -> u32 {
        self.by_family[usize::from(family.rank())]
    }

    /// Merge two paint passes.
    #[cfg(test)]
    pub(crate) fn merge(&mut self, other: Self) {
        for (left, right) in self.by_family.iter_mut().zip(other.by_family) {
            // Birth and death deliberately have both an underlay and overlay.
            // Each pass counts successful effects so a failed PathBuilder in
            // one pass cannot make the proof vacuous; max makes the merge
            // idempotent instead of counting the same located event twice.
            *left = (*left).max(right);
        }
    }
}

/// Tick-indexed presentation ring shared by the two GPUI views.
#[derive(Debug, Default)]
pub(crate) struct VfxProjection {
    latest_tick: Option<u64>,
    active: Vec<LocatedVfx>,
    dropped: u64,
}

impl VfxProjection {
    /// Ingest one completed science boundary exactly once.
    ///
    /// A duplicate boundary is a no-op. A backwards tick denotes a reset or a
    /// different run, so stale effects are cleared before accepting it.
    pub(crate) fn ingest<I>(&mut self, tick: u64, events: I)
    where
        I: IntoIterator<Item = LocatedVfx>,
    {
        match self.latest_tick {
            Some(previous) if tick == previous => return,
            Some(previous) if tick < previous => {
                self.active.clear();
                self.dropped = 0;
            }
            _ => {}
        }
        self.latest_tick = Some(tick);
        self.active.retain(|event| event.is_live_at(tick));
        self.active.extend(
            events
                .into_iter()
                .filter(|event| event.started_tick == tick && event.cue.duration_ticks > 0),
        );
        self.active.sort_by_key(stable_event_key);

        if self.active.len() > MAX_ACTIVE_VFX {
            let overflow = self.active.len() - MAX_ACTIVE_VFX;
            self.active.drain(..overflow);
            self.dropped = self
                .dropped
                .saturating_add(u64::try_from(overflow).unwrap_or(u64::MAX));
        }
    }

    /// Snapshot live effects for a render frame without consuming them.
    #[must_use]
    pub(crate) fn frame_at(&self, tick: u64) -> VfxFrame {
        let events = self
            .active
            .iter()
            .copied()
            .filter(|event| event.is_live_at(tick))
            .collect::<Vec<_>>()
            .into();
        VfxFrame { tick, events }
    }

    /// Total effects evicted by the frontend retention cap.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn dropped(&self) -> u64 {
        self.dropped
    }
}

/// Paint soft glows and trails below agents.
#[must_use]
pub(crate) fn paint_underlay(
    frame: &VfxFrame,
    camera: &CameraSnapshot,
    palette: ColorPaletteMode,
    window: &mut Window,
) -> VfxPaintCounts {
    let tick = frame.tick;
    let scale = camera.last_scale;
    let mut counts = VfxPaintCounts::default();
    for event in frame.events.iter() {
        let Some((x, y)) = event_screen_position(event, camera) else {
            continue;
        };
        let Some(progress) = event.progress_at(tick) else {
            continue;
        };
        let remaining = 1.0 - progress;
        let fade = remaining * remaining;
        if fade <= f32::EPSILON {
            continue;
        }
        let intensity = event.cue.intensity.clamp(0.0, 1.0);
        let base_radius = (event.cue.radius * scale).max(3.0);
        match event.family {
            VfxFamily::Birth => {
                let painted = paint_soft_disc(
                    window,
                    x,
                    y,
                    base_radius * (1.2 + progress * 2.6),
                    event.cue.accent_color,
                    (0.08 + intensity * 0.22) * fade,
                    palette,
                );
                counts.record_if_painted(event.family, painted);
            }
            VfxFamily::Death => {
                let painted = paint_soft_disc(
                    window,
                    x,
                    y,
                    base_radius * (1.8 - progress * 0.7),
                    event.cue.accent_color,
                    (0.12 + intensity * 0.25) * fade,
                    palette,
                );
                counts.record_if_painted(event.family, painted);
            }
            VfxFamily::Reproduce => {
                let outer = paint_ring(
                    window,
                    x,
                    y,
                    base_radius * (1.0 + progress * 3.0),
                    (1.0 + scale.sqrt()).clamp(1.0, 3.5),
                    event.cue.accent_color,
                    (0.18 + intensity * 0.30) * fade,
                    palette,
                );
                let inner = paint_ring(
                    window,
                    x,
                    y,
                    base_radius * (0.72 + progress * 2.4),
                    (0.8 + scale.sqrt()).clamp(1.0, 3.0),
                    event.cue.color,
                    (0.12 + intensity * 0.24) * fade,
                    palette,
                );
                counts.record_if_painted(event.family, outer || inner);
            }
            VfxFamily::BoostTrail => {
                let direction = fallback_direction(event.direction, [1.0, 0.0]);
                let mut painted = false;
                for step in 1_u8..=3 {
                    let step_f = f32::from(step);
                    let distance = base_radius * (step_f * 0.85 + progress * 1.4);
                    painted |= paint_soft_disc(
                        window,
                        x - direction[0] * distance,
                        y - direction[1] * distance,
                        (base_radius * (0.55 - step_f * 0.10)).max(1.5),
                        if step == 3 {
                            event.cue.accent_color
                        } else {
                            event.cue.color
                        },
                        (0.12 + intensity * 0.26) * fade / step_f,
                        palette,
                    );
                }
                counts.record_if_painted(event.family, painted);
            }
            VfxFamily::CombatHit | VfxFamily::Eat | VfxFamily::SpikeExtend => {}
        }
    }
    counts
}

/// Paint hot cores, impact cones, and feeding motes above agents.
#[must_use]
pub(crate) fn paint_overlay(
    frame: &VfxFrame,
    camera: &CameraSnapshot,
    palette: ColorPaletteMode,
    window: &mut Window,
) -> VfxPaintCounts {
    let tick = frame.tick;
    let scale = camera.last_scale;
    let mut counts = VfxPaintCounts::default();
    for event in frame.events.iter() {
        let Some((x, y)) = event_screen_position(event, camera) else {
            continue;
        };
        let Some(progress) = event.progress_at(tick) else {
            continue;
        };
        let remaining = 1.0 - progress;
        let fade = remaining * remaining;
        if fade <= f32::EPSILON {
            continue;
        }
        let intensity = event.cue.intensity.clamp(0.0, 1.0);
        let base_radius = (event.cue.radius * scale).max(3.0);
        match event.family {
            VfxFamily::Birth => {
                let painted = paint_sparkle(
                    window,
                    x,
                    y,
                    base_radius * (0.8 + progress),
                    event.cue.color,
                    (0.35 + intensity * 0.55) * fade,
                    palette,
                );
                counts.record_if_painted(event.family, painted);
            }
            VfxFamily::Death => {
                let painted = if event.cue.kind == VisualCueKind::Shards {
                    paint_sparkle(
                        window,
                        x,
                        y,
                        base_radius * (0.85 + fade * 0.65),
                        event.cue.color,
                        (0.25 + intensity * 0.55) * fade,
                        palette,
                    )
                } else {
                    paint_disc(
                        window,
                        x,
                        y,
                        (base_radius * (0.65 + fade * 0.45)).max(2.0),
                        event.cue.color,
                        (0.25 + intensity * 0.55) * fade,
                        palette,
                    )
                };
                counts.record_if_painted(event.family, painted);
            }
            VfxFamily::CombatHit => {
                let cone = paint_directional_cone(
                    window,
                    (x, y),
                    fallback_direction(event.direction, [1.0, 0.0]),
                    base_radius,
                    event.cue.color,
                    (0.45 + intensity * 0.50) * fade,
                    palette,
                );
                let core = paint_disc(
                    window,
                    x,
                    y,
                    (base_radius * 0.42).max(2.0),
                    event.cue.accent_color,
                    (0.40 + intensity * 0.55) * fade,
                    palette,
                );
                counts.record_if_painted(event.family, cone || core);
            }
            VfxFamily::Eat => {
                let direction = fallback_direction(event.direction, [0.0, -1.0]);
                let mut painted = false;
                for mote in 0_u8..3 {
                    // Direction is subject -> food. Motes begin at the food and
                    // converge on the subject as progress reaches one.
                    let mote_position =
                        feeding_mote_position((x, y), direction, base_radius, progress, mote);
                    painted |= paint_disc(
                        window,
                        mote_position.0,
                        mote_position.1,
                        (base_radius * 0.22).max(1.5),
                        if mote == 1 {
                            event.cue.accent_color
                        } else {
                            event.cue.color
                        },
                        (0.30 + intensity * 0.55) * fade,
                        palette,
                    );
                }
                counts.record_if_painted(event.family, painted);
            }
            VfxFamily::Reproduce | VfxFamily::BoostTrail => {
                // The expanding ring is the complete effect and is painted below
                // agents so it never obscures the offspring. Boost trails also
                // stay below bodies so orientation remains legible.
            }
            VfxFamily::SpikeExtend => {
                // Core defines Flash as a centered full-body telegraph. The
                // directional cone belongs exclusively to a connected hit.
                let halo = paint_soft_disc(
                    window,
                    x,
                    y,
                    base_radius * 2.8,
                    event.cue.accent_color,
                    (0.18 + intensity * 0.38) * fade,
                    palette,
                );
                let body = paint_disc(
                    window,
                    x,
                    y,
                    base_radius * 1.15,
                    event.cue.color,
                    (0.45 + intensity * 0.50) * fade,
                    palette,
                );
                counts.record_if_painted(event.family, halo || body);
            }
        }
    }
    counts
}

fn event_screen_position(event: &LocatedVfx, camera: &CameraSnapshot) -> Option<(f32, f32)> {
    if !event.position.x.is_finite() || !event.position.y.is_finite() {
        return None;
    }
    let position = camera.world_to_screen((event.position.x, event.position.y))?;
    (position.0.is_finite() && position.1.is_finite()).then_some(position)
}

fn feeding_mote_position(
    subject: (f32, f32),
    direction_to_food: [f32; 2],
    base_radius: f32,
    progress: f32,
    mote: u8,
) -> (f32, f32) {
    let mote_f = f32::from(mote);
    let start = base_radius * (2.2 + mote_f * 0.75);
    let distance = start * (1.0 - progress.clamp(0.0, 1.0));
    let perpendicular = [-direction_to_food[1], direction_to_food[0]];
    let stagger = (mote_f - 1.0) * base_radius * 0.35;
    (
        subject.0 + direction_to_food[0] * distance + perpendicular[0] * stagger,
        subject.1 + direction_to_food[1] * distance + perpendicular[1] * stagger,
    )
}

fn paint_disc(
    window: &mut Window,
    x: f32,
    y: f32,
    radius: f32,
    color: [f32; 3],
    alpha: f32,
    palette: ColorPaletteMode,
) -> bool {
    let mut disc = PathBuilder::fill();
    append_circle_polygon(&mut disc, x, y, radius.max(1.0));
    if let Ok(path) = disc.build() {
        window.paint_path(
            path,
            apply_palette(
                rgba_from_triplet_with_alpha(color, alpha.clamp(0.0, 1.0)),
                palette,
            ),
        );
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_soft_disc(
    window: &mut Window,
    x: f32,
    y: f32,
    radius: f32,
    color: [f32; 3],
    alpha: f32,
    palette: ColorPaletteMode,
) -> bool {
    let mut painted = false;
    for (radius_scale, alpha_scale) in [(1.0, 0.18), (0.72, 0.30), (0.42, 0.52)] {
        painted |= paint_disc(
            window,
            x,
            y,
            radius * radius_scale,
            color,
            alpha * alpha_scale,
            palette,
        );
    }
    painted
}

#[allow(clippy::too_many_arguments)]
fn paint_ring(
    window: &mut Window,
    x: f32,
    y: f32,
    radius: f32,
    width: f32,
    color: [f32; 3],
    alpha: f32,
    palette: ColorPaletteMode,
) -> bool {
    let mut ring = PathBuilder::stroke(px(width));
    append_arc_polyline(&mut ring, x, y, radius.max(1.0), 0.0, 360.0);
    if let Ok(path) = ring.build() {
        window.paint_path(
            path,
            apply_palette(
                rgba_from_triplet_with_alpha(color, alpha.clamp(0.0, 1.0)),
                palette,
            ),
        );
        true
    } else {
        false
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_directional_cone(
    window: &mut Window,
    origin: (f32, f32),
    direction: [f32; 2],
    radius: f32,
    color: [f32; 3],
    alpha: f32,
    palette: ColorPaletteMode,
) -> bool {
    let perpendicular = [-direction[1], direction[0]];
    let half_width = radius * 0.55;
    let back = radius * 0.35;
    let tip = radius * 2.4;
    let left = (
        origin.0 - direction[0] * back + perpendicular[0] * half_width,
        origin.1 - direction[1] * back + perpendicular[1] * half_width,
    );
    let right = (
        origin.0 - direction[0] * back - perpendicular[0] * half_width,
        origin.1 - direction[1] * back - perpendicular[1] * half_width,
    );
    let tip = (origin.0 + direction[0] * tip, origin.1 + direction[1] * tip);
    let mut cone = PathBuilder::fill();
    cone.move_to(point(px(left.0), px(left.1)));
    cone.line_to(point(px(right.0), px(right.1)));
    cone.line_to(point(px(tip.0), px(tip.1)));
    cone.close();
    if let Ok(path) = cone.build() {
        window.paint_path(
            path,
            apply_palette(
                rgba_from_triplet_with_alpha(color, alpha.clamp(0.0, 1.0)),
                palette,
            ),
        );
        true
    } else {
        false
    }
}

fn paint_sparkle(
    window: &mut Window,
    x: f32,
    y: f32,
    radius: f32,
    color: [f32; 3],
    alpha: f32,
    palette: ColorPaletteMode,
) -> bool {
    let mut sparkle = PathBuilder::stroke(px((radius * 0.16).clamp(1.0, 3.0)));
    sparkle.move_to(point(px(x - radius), px(y)));
    sparkle.line_to(point(px(x + radius), px(y)));
    sparkle.move_to(point(px(x), px(y - radius)));
    sparkle.line_to(point(px(x), px(y + radius)));
    let diagonal = radius * 0.7;
    sparkle.move_to(point(px(x - diagonal), px(y - diagonal)));
    sparkle.line_to(point(px(x + diagonal), px(y + diagonal)));
    sparkle.move_to(point(px(x - diagonal), px(y + diagonal)));
    sparkle.line_to(point(px(x + diagonal), px(y - diagonal)));
    if let Ok(path) = sparkle.build() {
        window.paint_path(
            path,
            apply_palette(
                rgba_from_triplet_with_alpha(color, alpha.clamp(0.0, 1.0)),
                palette,
            ),
        );
        true
    } else {
        false
    }
}

fn fallback_direction(direction: [f32; 2], fallback: [f32; 2]) -> [f32; 2] {
    if direction == [0.0, 0.0] {
        fallback
    } else {
        direction
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct StableEventKey {
    started_tick: u64,
    ordinal: u32,
    source: Option<AgentUid>,
    family: u8,
    target: Option<AgentUid>,
    position: [u32; 2],
    direction: [u32; 2],
    cue_kind: u8,
    color: [u32; 3],
    accent_color: [u32; 3],
    intensity: u32,
    radius: u32,
    duration_ticks: u32,
}

fn stable_event_key(event: &LocatedVfx) -> StableEventKey {
    StableEventKey {
        started_tick: event.started_tick,
        ordinal: event.ordinal,
        source: event.source,
        family: event.family.rank(),
        target: event.target,
        position: [event.position.x.to_bits(), event.position.y.to_bits()],
        direction: [event.direction[0].to_bits(), event.direction[1].to_bits()],
        cue_kind: cue_kind_rank(event.cue.kind),
        color: event.cue.color.map(f32::to_bits),
        accent_color: event.cue.accent_color.map(f32::to_bits),
        intensity: event.cue.intensity.to_bits(),
        radius: event.cue.radius.to_bits(),
        duration_ticks: event.cue.duration_ticks,
    }
}

const fn cue_kind_rank(kind: VisualCueKind) -> u8 {
    match kind {
        VisualCueKind::Sparkle => 0,
        VisualCueKind::Shards => 1,
        VisualCueKind::Wilt => 2,
        VisualCueKind::Nibble => 3,
        VisualCueKind::SparkCone => 4,
        VisualCueKind::PulseRing => 5,
        VisualCueKind::Flash => 6,
    }
}

fn normalized_direction(direction: [f32; 2]) -> [f32; 2] {
    if !direction[0].is_finite() || !direction[1].is_finite() {
        return [0.0, 0.0];
    }
    let length_squared = direction[0].mul_add(direction[0], direction[1] * direction[1]);
    if length_squared <= f32::EPSILON {
        return [0.0, 0.0];
    }
    let inverse_length = length_squared.sqrt().recip();
    [direction[0] * inverse_length, direction[1] * inverse_length]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(kind: VisualCueKind, duration_ticks: u32) -> VisualCue {
        VisualCue {
            kind,
            color: [1.0, 0.25, 0.5],
            accent_color: [1.0, 0.8, 0.3],
            intensity: 1.0,
            radius: 8.0,
            duration_ticks,
        }
    }

    fn event(tick: u64, uid: u64, kind: VisualCueKind, duration_ticks: u32) -> LocatedVfx {
        let cue = cue(kind, duration_ticks);
        LocatedVfx::from_cue(
            tick,
            u32::try_from(uid).unwrap_or(u32::MAX),
            Some(AgentUid(uid)),
            None,
            Position::new(10.0, 20.0),
            [3.0, 4.0],
            cue,
        )
    }

    #[test]
    fn duplicate_repaint_boundary_does_not_duplicate_or_age_events() {
        let mut projection = VfxProjection::default();
        projection.ingest(7, [event(7, 1, VisualCueKind::Sparkle, 4)]);
        let first = projection.frame_at(7);
        projection.ingest(7, [event(7, 2, VisualCueKind::SparkCone, 4)]);
        let repaint = projection.frame_at(7);

        assert_eq!(first.events.as_ref(), repaint.events.as_ref());
        assert_eq!(repaint.len(), 1);
        assert_eq!(repaint.events[0].age_at(7), Some(0));
        assert_eq!(repaint.events[0].fade_at(7), Some(1.0));
    }

    #[test]
    fn located_event_uses_the_core_visual_cue_table() {
        let event = WorldVisualEvent::CombatHit { damage: 1.25 };
        let located = LocatedVfx::from_world_event(
            7,
            3,
            Some(AgentUid(1)),
            Some(AgentUid(2)),
            Position::new(10.0, 20.0),
            [1.0, 0.0],
            &event,
        );

        assert_eq!(located.cue, visual_cue_for_event(&event));
        assert_eq!(located.family, VfxFamily::CombatHit);
    }

    #[test]
    fn event_is_invisible_before_its_completed_science_boundary() {
        let mut projection = VfxProjection::default();
        projection.ingest(7, [event(7, 1, VisualCueKind::Sparkle, 4)]);

        let future = projection.active[0];
        assert_eq!(future.age_at(6), None);
        assert_eq!(future.progress_at(6), None);
        assert_eq!(future.fade_at(6), None);
        assert!(projection.frame_at(6).is_empty());
        assert_eq!(projection.frame_at(7).len(), 1);
    }

    #[test]
    fn lifetimes_advance_only_with_science_ticks_and_expire_at_duration() {
        let mut projection = VfxProjection::default();
        projection.ingest(10, [event(10, 1, VisualCueKind::Wilt, 3)]);

        assert_eq!(projection.frame_at(10).len(), 1);
        assert_eq!(projection.frame_at(11).len(), 1);
        assert_eq!(projection.frame_at(12).len(), 1);
        assert!(projection.frame_at(13).is_empty());
        assert!(projection.frame_at(50).is_empty());
    }

    #[test]
    fn backwards_tick_clears_effects_from_the_previous_run() {
        let mut projection = VfxProjection::default();
        projection.ingest(50, [event(50, 1, VisualCueKind::PulseRing, 28)]);
        projection.ingest(2, [event(2, 2, VisualCueKind::Nibble, 12)]);

        let frame = projection.frame_at(2);
        assert_eq!(frame.len(), 1);
        assert_eq!(frame.events[0].source, Some(AgentUid(2)));
    }

    #[test]
    fn malformed_directions_are_contained_and_valid_directions_are_normalized() {
        let valid = event(1, 1, VisualCueKind::SparkCone, 8);
        assert!((valid.direction[0] - 0.6).abs() <= f32::EPSILON);
        assert!((valid.direction[1] - 0.8).abs() <= f32::EPSILON);

        let invalid = LocatedVfx::from_cue(
            1,
            0,
            None,
            None,
            Position::new(0.0, 0.0),
            [f32::NAN, 1.0],
            cue(VisualCueKind::SparkCone, 8),
        );
        assert_eq!(invalid.direction, [0.0, 0.0]);
    }

    #[test]
    fn feeding_motes_travel_from_food_toward_the_subject() {
        let subject = (10.0, 20.0);
        let at_food = feeding_mote_position(subject, [1.0, 0.0], 8.0, 0.0, 1);
        let at_subject = feeding_mote_position(subject, [1.0, 0.0], 8.0, 1.0, 1);

        assert!(at_food.0 > subject.0);
        assert!((at_food.1 - subject.1).abs() <= f32::EPSILON);
        assert!((at_subject.0 - subject.0).abs() <= f32::EPSILON);
        assert!((at_subject.1 - subject.1).abs() <= f32::EPSILON);
    }

    #[test]
    fn stable_order_is_independent_of_batch_order() {
        let left = [
            event(4, 9, VisualCueKind::Flash, 8),
            event(4, 2, VisualCueKind::Sparkle, 8),
            event(4, 2, VisualCueKind::SparkCone, 8),
        ];
        let right = [left[2], left[0], left[1]];
        let mut a = VfxProjection::default();
        let mut b = VfxProjection::default();
        a.ingest(4, left);
        b.ingest(4, right);

        assert_eq!(a.frame_at(4).events.as_ref(), b.frame_at(4).events.as_ref());
    }

    #[test]
    fn repeated_ordinals_use_the_full_payload_as_a_stable_tie_breaker() {
        let mut left = event(4, 2, VisualCueKind::Sparkle, 8);
        left.ordinal = 7;
        left.position = Position::new(30.0, 20.0);
        let mut right = event(4, 2, VisualCueKind::Sparkle, 8);
        right.ordinal = 7;
        right.position = Position::new(10.0, 20.0);

        let mut a = VfxProjection::default();
        let mut b = VfxProjection::default();
        a.ingest(4, [left, right]);
        b.ingest(4, [right, left]);

        assert_eq!(a.frame_at(4).events.as_ref(), b.frame_at(4).events.as_ref());
    }

    #[test]
    fn stable_key_distinguishes_absent_uid_from_the_maximum_uid() {
        let mut left = event(4, 2, VisualCueKind::Sparkle, 8);
        left.ordinal = 7;
        left.source = None;
        left.target = Some(AgentUid(u64::MAX));
        let mut right = left;
        right.source = Some(AgentUid(u64::MAX));
        right.target = None;

        assert_ne!(stable_event_key(&left), stable_event_key(&right));

        let mut a = VfxProjection::default();
        let mut b = VfxProjection::default();
        a.ingest(4, [left, right]);
        b.ingest(4, [right, left]);

        assert_eq!(a.frame_at(4).events.as_ref(), b.frame_at(4).events.as_ref());
    }

    #[test]
    fn paint_counts_require_success_and_merge_dual_pass_effects_once() {
        let mut underlay = VfxPaintCounts::default();
        underlay.record_if_painted(VfxFamily::Birth, false);
        assert_eq!(underlay.family(VfxFamily::Birth), 0);
        underlay.record_if_painted(VfxFamily::Birth, true);

        let mut overlay = VfxPaintCounts::default();
        overlay.record_if_painted(VfxFamily::Birth, true);
        overlay.record_if_painted(VfxFamily::CombatHit, true);

        underlay.merge(overlay);
        assert_eq!(underlay.family(VfxFamily::Birth), 1);
        assert_eq!(underlay.family(VfxFamily::CombatHit), 1);
    }

    #[test]
    fn spike_extension_uses_the_core_full_body_flash_cue() {
        let world_event = WorldVisualEvent::SpikeExtend;
        let located = LocatedVfx::from_world_event(
            7,
            3,
            Some(AgentUid(1)),
            None,
            Position::new(10.0, 20.0),
            [0.0, 1.0],
            &world_event,
        );

        assert_eq!(located.family, VfxFamily::SpikeExtend);
        assert_eq!(located.cue.kind, VisualCueKind::Flash);
        assert_eq!(located.cue, visual_cue_for_event(&world_event));
    }

    #[test]
    fn frontend_cap_evicts_oldest_events_and_reports_the_loss() {
        let mut projection = VfxProjection::default();
        let events = (0..=MAX_ACTIVE_VFX).map(|uid| {
            event(
                1,
                u64::try_from(uid).expect("test UID fits u64"),
                VisualCueKind::Sparkle,
                4,
            )
        });
        projection.ingest(1, events);

        assert_eq!(projection.frame_at(1).len(), MAX_ACTIVE_VFX);
        assert_eq!(projection.dropped(), 1);
    }
}
