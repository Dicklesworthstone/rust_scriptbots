//! Parametric creature mesh generator (bd-2z0.14.1.5.1).
//!
//! The ScriptBots silhouette — capsule body, two wheels, spike, eyes, mouth,
//! ears, nose — generated in code (no asset files) as ONE base mesh family
//! per LOD tier. Every vertex carries a [`CreaturePart`] tag so the A1
//! instanced-pipeline shader can deform proportions per-instance (body
//! length/girth, wheel radius, spike length, eye placement) without distinct
//! meshes: the same base geometry serves 10k differently-shaped critters in
//! a handful of draw calls.
//!
//! The mesh is built at unit proportions around a documented anchor layout
//! (heading = +X, up = +Y, axle = ±Z): deformation is a per-part transform of
//! the anchor layout, never a topology change. All geometry is deterministic
//! — identical LOD input produces byte-identical buffers.
//!
//! This module is pure geometry: no Bevy app, no GPU. The
//! [`CreatureMeshData::to_bevy_mesh`] adapter converts for the render world;
//! tests validate the raw buffers directly.

use bevy::render::render_resource::VertexFormat;
use bevy_mesh::{Indices, Mesh, MeshVertexAttribute};

/// Custom per-vertex attribute carrying the [`CreaturePart::tag`] of each
/// vertex into the instance shader (id chosen far from the built-ins).
pub const ATTRIBUTE_CREATURE_PART: MeshVertexAttribute =
    MeshVertexAttribute::new("Vertex_CreaturePart", 9_001, VertexFormat::Uint32);

/// The ScriptBots eye count (mirrors `scriptbots_core::NUM_EYES`; duplicated
/// here so the geometry kit does not depend on brain constants).
pub const NUM_EYES: usize = 4;

/// Which body part a vertex belongs to. The A1 instance shader reads this
/// attribute to apply per-part deformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreaturePart {
    /// Capsule body shell.
    Body,
    /// Left/right wheel torus.
    WheelLeft,
    /// Right wheel.
    WheelRight,
    /// Tapered attack spike.
    Spike,
    /// Sclera (white) of eye N.
    EyeSclera(u8),
    /// Pupil of eye N.
    EyePupil(u8),
    /// Mouth band.
    Mouth,
    /// Left ear fin.
    EarLeft,
    /// Right ear fin.
    EarRight,
    /// Smell-trait nose knob.
    Nose,
}

impl CreaturePart {
    /// Stable numeric tag for the vertex attribute.
    #[must_use]
    pub const fn tag(self) -> u32 {
        match self {
            Self::Body => 0,
            Self::WheelLeft => 1,
            Self::WheelRight => 2,
            Self::Spike => 3,
            // Widening `as` casts: `u32::from` is not const-stable on the
            // pinned toolchain and `tag()` must stay `const`.
            Self::EyeSclera(n) => 10 + n as u32,
            Self::EyePupil(n) => 20 + n as u32,
            Self::Mouth => 30,
            Self::EarLeft => 31,
            Self::EarRight => 32,
            Self::Nose => 33,
        }
    }
}

/// Level-of-detail tier for the creature mesh family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreatureLod {
    /// Full detail (~2-6k triangles): hero shots and near camera.
    Lod0,
    /// Reduced detail (~800 triangles): mid distances.
    Lod1,
    /// Impostor: two crossed quads textured from the LOD0 bake produced by
    /// [`bake_impostor_atlas`] (tile 0 = profile, tile 1 = front).
    Lod2,
}

/// Raw creature geometry: plain buffers plus the per-vertex part tags.
#[derive(Debug, Clone)]
pub struct CreatureMeshData {
    /// Vertex positions (unit proportions; heading = +X, up = +Y).
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (unit length).
    pub normals: Vec<[f32; 3]>,
    /// UVs in `[0, 1]` (procedural-texture space; LOD2 = impostor atlas).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices (u32), CCW winding.
    pub indices: Vec<u32>,
    /// One [`CreaturePart`] per vertex.
    pub parts: Vec<CreaturePart>,
}

impl CreatureMeshData {
    /// Vertex count.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Triangle count.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Vertex ranges per part (test/debug): contiguous runs in build order.
    #[must_use]
    pub fn part_vertex_ranges(&self) -> Vec<(CreaturePart, core::ops::Range<usize>)> {
        let mut ranges: Vec<(CreaturePart, core::ops::Range<usize>)> = Vec::new();
        for (index, &part) in self.parts.iter().enumerate() {
            match ranges.last_mut() {
                Some((last_part, range)) if *last_part == part => range.end = index + 1,
                _ => ranges.push((part, index..index + 1)),
            }
        }
        ranges
    }

    /// Convert into a real Bevy mesh with the creature-part custom attribute
    /// and computed tangents (required for PBR normal mapping).
    #[must_use]
    pub fn to_bevy_mesh(&self) -> Mesh {
        let mut mesh = Mesh::new(
            bevy::render::render_resource::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.positions.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs.clone());
        mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, self.compute_tangents());
        mesh.insert_attribute(
            ATTRIBUTE_CREATURE_PART,
            bevy_mesh::VertexAttributeValues::Uint32(
                self.parts.iter().map(|part| part.tag()).collect(),
            ),
        );
        mesh.insert_indices(Indices::U32(self.indices.clone()));
        mesh
    }

    /// Per-vertex tangents (`[tangent.xyz, handedness]`) derived from the UV
    /// gradients (Lengyel's method), orthogonalized against the normals.
    ///
    /// Deterministic: fixed triangle order, plain f32 arithmetic, no hash
    /// maps or threading. Vertices whose triangles all have degenerate UV
    /// area receive an arbitrary-but-stable tangent orthogonal to their
    /// normal (handedness +1).
    #[must_use]
    pub fn compute_tangents(&self) -> Vec<[f32; 4]> {
        let vertex_count = self.positions.len();
        let mut tan1 = vec![[0.0_f32; 3]; vertex_count];
        let mut tan2 = vec![[0.0_f32; 3]; vertex_count];
        for tri in self.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let (p0, p1, p2) = (self.positions[i0], self.positions[i1], self.positions[i2]);
            let (uv0, uv1, uv2) = (self.uvs[i0], self.uvs[i1], self.uvs[i2]);
            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
            let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];
            let det = duv1[0] * duv2[1] - duv2[0] * duv1[1];
            if det.abs() < 1e-12 {
                continue;
            }
            let r = 1.0 / det;
            let t = [
                (e1[0] * duv2[1] - e2[0] * duv1[1]) * r,
                (e1[1] * duv2[1] - e2[1] * duv1[1]) * r,
                (e1[2] * duv2[1] - e2[2] * duv1[1]) * r,
            ];
            let b = [
                (e2[0] * duv1[0] - e1[0] * duv2[0]) * r,
                (e2[1] * duv1[0] - e1[1] * duv2[0]) * r,
                (e2[2] * duv1[0] - e1[2] * duv2[0]) * r,
            ];
            for &i in &[i0, i1, i2] {
                for k in 0..3 {
                    tan1[i][k] += t[k];
                    tan2[i][k] += b[k];
                }
            }
        }
        let mut tangents = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let n = self.normals[i];
            let t = tan1[i];
            let t_len2 = t[0].mul_add(t[0], t[1] * t[1]) + t[2] * t[2];
            if t_len2 < 1e-12 {
                // Degenerate UVs: any stable tangent orthogonal to n.
                let axis = if n[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let fallback = normalize([
                    n[1] * axis[2] - n[2] * axis[1],
                    n[2] * axis[0] - n[0] * axis[2],
                    n[0] * axis[1] - n[1] * axis[0],
                ]);
                tangents.push([fallback[0], fallback[1], fallback[2], 1.0]);
                continue;
            }
            // Gram-Schmidt: t' = normalize(t - n * dot(n, t)).
            let dot_nt = n[0].mul_add(t[0], n[1] * t[1]) + n[2] * t[2];
            let ortho = normalize([
                t[0] - n[0] * dot_nt,
                t[1] - n[1] * dot_nt,
                t[2] - n[2] * dot_nt,
            ]);
            // Handedness: sign of dot(cross(n, t), bitangent).
            let cross = [
                n[1] * t[2] - n[2] * t[1],
                n[2] * t[0] - n[0] * t[2],
                n[0] * t[1] - n[1] * t[0],
            ];
            let b = tan2[i];
            let handedness = if cross[0].mul_add(b[0], cross[1] * b[1]) + cross[2] * b[2] < 0.0 {
                -1.0
            } else {
                1.0
            };
            tangents.push([ortho[0], ortho[1], ortho[2], handedness]);
        }
        tangents
    }
}

/// Per-instance deformation parameters (unit-less multipliers/offsets in the
/// anchor layout). Bounds are validated so trait extremes cannot invert the
/// base mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CreatureProportions {
    /// Body length multiplier along +X in `[0.7, 1.6]`.
    pub body_length: f32,
    /// Body girth multiplier in `[0.7, 1.5]`.
    pub body_girth: f32,
    /// Wheel radius multiplier in `[0.7, 1.4]`.
    pub wheel_radius: f32,
    /// Spike extension in `[0, 2]` (0 = fully retracted).
    pub spike_extension: f32,
    /// Nose knob scale multiplier in `[0.7, 1.8]` (smell trait).
    pub nose_scale: f32,
    /// Ear fin scale multiplier in `[0.7, 1.8]` (hearing trait).
    pub ear_scale: f32,
    /// Eye spread multiplier in `[0.8, 1.3]`.
    pub eye_spread: f32,
}

impl Default for CreatureProportions {
    fn default() -> Self {
        Self {
            body_length: 1.0,
            body_girth: 1.0,
            wheel_radius: 1.0,
            spike_extension: 0.0,
            nose_scale: 1.0,
            ear_scale: 1.0,
            eye_spread: 1.0,
        }
    }
}

impl CreatureProportions {
    /// Validate every bound; returns the violating field name on failure.
    pub fn validate(&self) -> Result<(), &'static str> {
        let checks: [(f32, f32, f32, &str); 7] = [
            (self.body_length, 0.7, 1.6, "body_length"),
            (self.body_girth, 0.7, 1.5, "body_girth"),
            (self.wheel_radius, 0.7, 1.4, "wheel_radius"),
            (self.spike_extension, 0.0, 2.0, "spike_extension"),
            (self.nose_scale, 0.7, 1.8, "nose_scale"),
            (self.ear_scale, 0.7, 1.8, "ear_scale"),
            (self.eye_spread, 0.8, 1.3, "eye_spread"),
        ];
        for (value, lo, hi, name) in checks {
            if !value.is_finite() || value < lo || value > hi {
                return Err(name);
            }
        }
        Ok(())
    }
}

/// Map trait modifiers (as exposed by `scriptbots_core::visual`'s agent
/// semantics) to deformation proportions. Traits are clamped defensively;
/// outputs always satisfy [`CreatureProportions::validate`].
#[must_use]
pub fn proportions_for_traits(
    trait_smell: f32,
    trait_hearing: f32,
    trait_eye: f32,
    spike_length: f32,
) -> CreatureProportions {
    let clamp01 = |v: f32| {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let smell = clamp01(trait_smell);
    let hearing = clamp01(trait_hearing);
    let eye = clamp01(trait_eye);
    let spike = if spike_length.is_finite() {
        spike_length.clamp(0.0, 2.0)
    } else {
        0.0
    };
    CreatureProportions {
        body_length: 0.85 + 0.45 * clamp01(0.5 + spike * 0.25),
        body_girth: 0.9 + 0.35 * clamp01(1.0 - spike * 0.2),
        wheel_radius: 1.0,
        spike_extension: spike,
        nose_scale: 0.7 + 1.1 * smell,
        ear_scale: 0.7 + 1.1 * hearing,
        eye_spread: 0.85 + 0.4 * eye,
    }
}

// ---------------------------------------------------------------------------
// Geometry construction. All builders append into a shared builder struct so
// parts share one index space (the A1 pipeline wants a single mesh per LOD).
// ---------------------------------------------------------------------------

const TAU: f32 = core::f32::consts::TAU;

#[derive(Default)]
struct MeshBuilder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
    parts: Vec<CreaturePart>,
}

impl MeshBuilder {
    fn vertex(
        &mut self,
        part: CreaturePart,
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
    ) -> u32 {
        let index = self.positions.len() as u32;
        self.positions.push(position);
        self.normals.push(normalize(normal));
        self.uvs.push(uv);
        self.parts.push(part);
        index
    }

    fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend_from_slice(&[a, b, c]);
    }

    /// Lat-long ellipsoid shell (shared seam column: watertight by
    /// construction). `segments` around the equator, `rings` pole rows.
    fn ellipsoid(
        &mut self,
        part: CreaturePart,
        center: [f32; 3],
        radii: [f32; 3],
        segments: u32,
        rings: u32,
        uv_scale: [f32; 2],
    ) {
        let mut grid: Vec<Vec<u32>> = Vec::with_capacity(rings as usize + 1);
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let theta = v * core::f32::consts::PI; // 0 = north pole, PI = south
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let mut row = Vec::with_capacity(segments as usize);
            for seg in 0..segments {
                let u = seg as f32 / segments as f32;
                let phi = u * TAU;
                // Ellipsoid point and its normal (gradient of the implicit form).
                let px = radii[0] * sin_t * phi.cos();
                let py = radii[1] * cos_t;
                let pz = radii[2] * sin_t * phi.sin();
                let normal = [
                    px / (radii[0] * radii[0]).max(1e-6),
                    py / (radii[1] * radii[1]).max(1e-6),
                    pz / (radii[2] * radii[2]).max(1e-6),
                ];
                row.push(self.vertex(
                    part,
                    [center[0] + px, center[1] + py, center[2] + pz],
                    normal,
                    [u * uv_scale[0], v * uv_scale[1]],
                ));
            }
            grid.push(row);
        }
        for ring in 0..rings as usize {
            for seg in 0..segments as usize {
                let next_seg = (seg + 1) % segments as usize;
                let a = grid[ring][seg];
                let b = grid[ring][next_seg];
                let c = grid[ring + 1][seg];
                let d = grid[ring + 1][next_seg];
                if ring == 0 {
                    // North cap fan.
                    if a != c && b != c {
                        self.triangle(a, d, c);
                    }
                } else if ring + 1 == rings as usize {
                    // South cap fan.
                    if a != d && a != c {
                        self.triangle(a, b, d);
                    }
                } else {
                    self.triangle(a, b, c);
                    self.triangle(b, d, c);
                }
            }
        }
    }

    /// Open cylinder (used as capsule mid-section); ends are closed by the
    /// caller's hemispheres, so the shell itself is intentionally open and
    /// the union is watertight.
    fn cylinder_x(
        &mut self,
        part: CreaturePart,
        x0: f32,
        x1: f32,
        radius: f32,
        segments: u32,
        uv_scale: [f32; 2],
    ) -> (Vec<u32>, Vec<u32>) {
        let mut ring_a = Vec::with_capacity(segments as usize);
        let mut ring_b = Vec::with_capacity(segments as usize);
        for seg in 0..segments {
            let u = seg as f32 / segments as f32;
            let phi = u * TAU;
            let (s, c) = (phi.sin(), phi.cos());
            ring_a.push(self.vertex(
                part,
                [x0, radius * c, radius * s],
                [0.0, c, s],
                [u * uv_scale[0], 0.0],
            ));
            ring_b.push(self.vertex(
                part,
                [x1, radius * c, radius * s],
                [0.0, c, s],
                [u * uv_scale[0], 1.0 * uv_scale[1]],
            ));
        }
        for seg in 0..segments as usize {
            let next = (seg + 1) % segments as usize;
            self.triangle(ring_a[seg], ring_b[seg], ring_a[next]);
            self.triangle(ring_b[seg], ring_b[next], ring_a[next]);
        }
        (ring_a, ring_b)
    }

    /// Torus around the Y axis at `center` (wheel: axle along Z, so we build
    /// around Y then rotate 90 degrees about X).
    fn torus(
        &mut self,
        part: CreaturePart,
        center: [f32; 3],
        major_radius: f32,
        minor_radius: f32,
        major_segments: u32,
        minor_segments: u32,
    ) {
        let mut grid: Vec<Vec<u32>> = Vec::with_capacity(major_segments as usize);
        for major in 0..major_segments {
            let u = major as f32 / major_segments as f32;
            let phi = u * TAU;
            let (sp, cp) = (phi.sin(), phi.cos());
            let mut row = Vec::with_capacity(minor_segments as usize);
            for minor in 0..minor_segments {
                let v = minor as f32 / minor_segments as f32;
                let theta = v * TAU;
                let (st, ct) = (theta.sin(), theta.cos());
                // Torus around the Z axis (wheel axle): ring in the XY plane.
                let radial = major_radius + minor_radius * ct;
                let px = radial * cp;
                let py = radial * sp;
                let pz = minor_radius * st;
                let normal = [ct * cp, ct * sp, st];
                row.push(self.vertex(
                    part,
                    [center[0] + px, center[1] + py, center[2] + pz],
                    normal,
                    [u, v],
                ));
            }
            grid.push(row);
        }
        for major in 0..major_segments as usize {
            let next_major = (major + 1) % major_segments as usize;
            for minor in 0..minor_segments as usize {
                let next_minor = (minor + 1) % minor_segments as usize;
                let a = grid[major][minor];
                let b = grid[next_major][minor];
                let c = grid[major][next_minor];
                let d = grid[next_major][next_minor];
                self.triangle(a, b, c);
                self.triangle(b, d, c);
            }
        }
    }

    /// Tapered cone along +X from `base_x` to `tip_x` with `base_radius`.
    fn cone_x(
        &mut self,
        part: CreaturePart,
        base_x: f32,
        tip_x: f32,
        base_radius: f32,
        segments: u32,
    ) {
        let length = tip_x - base_x;
        let mut ring = Vec::with_capacity(segments as usize);
        let slope = base_radius / length.max(1e-6);
        for seg in 0..segments {
            let u = seg as f32 / segments as f32;
            let phi = u * TAU;
            let (s, c) = (phi.sin(), phi.cos());
            // Cone side normal: radial direction tilted along -X by the slope.
            let normal = normalize([slope, c, s]);
            ring.push(self.vertex(
                part,
                [base_x, base_radius * c, base_radius * s],
                normal,
                [u, 0.0],
            ));
        }
        let tip = self.vertex(part, [tip_x, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0]);
        // Base cap.
        let cap_center = self.vertex(part, [base_x, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.5, 0.5]);
        for seg in 0..segments as usize {
            let next = (seg + 1) % segments as usize;
            self.triangle(ring[seg], tip, ring[next]);
            self.triangle(cap_center, ring[next], ring[seg]);
        }
    }

    /// Single quad (mouth band, ear fins, impostor cards).
    fn quad(
        &mut self,
        part: CreaturePart,
        corners: [[f32; 3]; 4],
        normal: [f32; 3],
        uvs: [[f32; 2]; 4],
        double_sided: bool,
    ) {
        let mut ids = [0u32; 4];
        for (i, corner) in corners.iter().enumerate() {
            ids[i] = self.vertex(part, *corner, normal, uvs[i]);
        }
        self.triangle(ids[0], ids[1], ids[2]);
        self.triangle(ids[0], ids[2], ids[3]);
        if double_sided {
            self.triangle(ids[0], ids[2], ids[1]);
            self.triangle(ids[0], ids[3], ids[2]);
        }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0].mul_add(v[0], v[1] * v[1]) + v[2] * v[2]).sqrt();
    if len > 1e-9 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

// ---------------------------------------------------------------------------
// Anchor layout (unit proportions). These constants are the contract the
// instance shader deforms against; changing them is a visual redesign.
// ---------------------------------------------------------------------------

/// Body capsule mid-radius.
const BODY_RADIUS: f32 = 0.5;
/// Body capsule half-length (cylinder part, before caps).
const BODY_HALF_LEN: f32 = 0.45;
/// Wheel vertical offset below body center.
const WHEEL_DROP: f32 = 0.38 * BODY_RADIUS;
/// Wheel lateral offset from body center.
const WHEEL_OFFSET_Z: f32 = 1.12 * BODY_RADIUS;
/// Wheel major radius.
const WHEEL_MAJOR: f32 = 0.34;
/// Wheel minor (tube) radius.
const WHEEL_MINOR: f32 = 0.12;
/// Spike base radius.
const SPIKE_BASE_RADIUS: f32 = 0.16;
/// Spike length at unit extension.
const SPIKE_LEN: f32 = 0.7;
/// Eye sphere radius (sclera).
const EYE_RADIUS: f32 = 0.13;
/// Pupil radius factor of the sclera.
const PUPIL_FACTOR: f32 = 0.55;

/// Half-extent of the impostor card in unit proportions. Must bound the
/// whole creature: the spike tip reaches x = 0.45 + 0.175 + 0.7 = 1.325,
/// wheels bottom near y = -0.83, ears top near y = 0.63, wheels z = ±0.68.
/// 1.4 covers the spike with margin; shared by the LOD2 card geometry and
/// the impostor bake projection.
const IMPOSTOR_EXTENT: f32 = 1.4;

/// Build the creature mesh for one LOD tier.
///
/// LOD budgets (asserted in tests): Lod0 ≤ 6000 vertices, Lod1 ≤ 2000,
/// Lod2 = exactly 8 vertices / 24 indices (two double-sided crossed quads).
#[must_use]
pub fn build_creature_mesh(lod: CreatureLod) -> CreatureMeshData {
    let mut builder = MeshBuilder::default();
    match lod {
        CreatureLod::Lod2 => {
            // Two crossed quads centered at the anchor origin, sampling the
            // two-tile impostor atlas from [`bake_impostor_atlas`]: the
            // profile quad (X-Y plane) reads tile 0 (u in [0, 0.5]) and the
            // front quad (Z-Y plane) reads tile 1 (u in [0.5, 1]).
            let extent = IMPOSTOR_EXTENT;
            builder.quad(
                CreaturePart::Body,
                [
                    [-extent, -extent, 0.0],
                    [extent, -extent, 0.0],
                    [extent, extent, 0.0],
                    [-extent, extent, 0.0],
                ],
                [0.0, 0.0, 1.0],
                [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
                true,
            );
            builder.quad(
                CreaturePart::Body,
                [
                    [0.0, -extent, -extent],
                    [0.0, -extent, extent],
                    [0.0, extent, extent],
                    [0.0, extent, -extent],
                ],
                [1.0, 0.0, 0.0],
                [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                true,
            );
        }
        CreatureLod::Lod0 | CreatureLod::Lod1 => {
            // Segment budgets target the bead's triangle bands
            // (Lod0 ~2-4k tris, Lod1 ~800 tris); asserted in tests.
            let (
                body_segments,
                body_rings,
                wheel_major_segs,
                wheel_minor_segs,
                cone_segs,
                eye_segs,
            ) = match lod {
                CreatureLod::Lod0 => (24, 12, 16, 10, 12, 8),
                CreatureLod::Lod1 => (10, 6, 10, 6, 6, 5),
                CreatureLod::Lod2 => unreachable!(),
            };

            // Body: capsule along +X = cylinder + two hemispherical caps.
            let body_x0 = -BODY_HALF_LEN;
            let body_x1 = BODY_HALF_LEN;
            // UV scale [1, 1]: the closing seam wraps cleanly under REPEAT
            // sampling only when u spans exactly one period; a 2x scale
            // leaves a fractional-period jump (and breaks the [0,1] UV
            // invariant asserted for impostor-safe sampling).
            let (ring_a, ring_b) = builder.cylinder_x(
                CreaturePart::Body,
                body_x0,
                body_x1,
                BODY_RADIUS,
                body_segments,
                [1.0, 1.0],
            );
            // Caps: hemispheres whose equators coincide with the cylinder
            // rings. Build them as half-ellipsoids sharing the ring positions
            // (duplicate vertices at the seam are acceptable here — the cap
            // and cylinder share exact positions, so the UNION is closed; the
            // watertightness test checks each shell independently).
            let cap_rings = body_rings / 2;
            // Rear cap (centered at body_x0, pointing -X).
            cap_hemisphere(
                &mut builder,
                body_x0,
                BODY_RADIUS,
                -1.0,
                body_segments,
                cap_rings,
                &ring_a,
            );
            // Front cap (centered at body_x1, pointing +X).
            cap_hemisphere(
                &mut builder,
                body_x1,
                BODY_RADIUS,
                1.0,
                body_segments,
                cap_rings,
                &ring_b,
            );

            // Wheels: tori around Z at ±WHEEL_OFFSET_Z, dropped by WHEEL_DROP.
            let wheel_y = -WHEEL_DROP - BODY_RADIUS * 0.35;
            builder.torus(
                CreaturePart::WheelLeft,
                [0.0, wheel_y, WHEEL_OFFSET_Z],
                WHEEL_MAJOR,
                WHEEL_MINOR,
                wheel_major_segs,
                wheel_minor_segs,
            );
            builder.torus(
                CreaturePart::WheelRight,
                [0.0, wheel_y, -WHEEL_OFFSET_Z],
                WHEEL_MAJOR,
                WHEEL_MINOR,
                wheel_major_segs,
                wheel_minor_segs,
            );

            // Spike: cone from the front cap pole forward.
            let spike_base_x = body_x1 + BODY_RADIUS * 0.35;
            builder.cone_x(
                CreaturePart::Spike,
                spike_base_x,
                spike_base_x + SPIKE_LEN,
                SPIKE_BASE_RADIUS,
                cone_segs,
            );

            // Eyes: NUM_EYES spheres on the front face, arranged in a small
            // arc; pupils are smaller spheres offset slightly forward.
            let face_x = body_x1 + BODY_RADIUS * 0.55;
            for eye in 0..NUM_EYES {
                let t = (eye as f32 + 0.5) / NUM_EYES as f32 - 0.5;
                let eye_y = 0.18 + (eye % 2) as f32 * 0.1;
                let eye_z = t * 0.55;
                builder.ellipsoid(
                    CreaturePart::EyeSclera(eye as u8),
                    [face_x, eye_y, eye_z],
                    [EYE_RADIUS; 3],
                    eye_segs,
                    eye_segs,
                    [1.0, 1.0],
                );
                builder.ellipsoid(
                    CreaturePart::EyePupil(eye as u8),
                    [face_x + EYE_RADIUS * 0.7, eye_y, eye_z + t * 0.05],
                    [EYE_RADIUS * PUPIL_FACTOR; 3],
                    eye_segs,
                    eye_segs,
                    [1.0, 1.0],
                );
            }

            // Mouth: a band at the lower front of the body.
            let mouth_x = body_x1 + BODY_RADIUS * 0.5;
            let mouth_y = -BODY_RADIUS * 0.45;
            builder.quad(
                CreaturePart::Mouth,
                [
                    [mouth_x, mouth_y, -BODY_RADIUS * 0.5],
                    [mouth_x, mouth_y, BODY_RADIUS * 0.5],
                    [mouth_x + 0.06, mouth_y - 0.08, BODY_RADIUS * 0.45],
                    [mouth_x + 0.06, mouth_y - 0.08, -BODY_RADIUS * 0.45],
                ],
                [0.9, -0.3, 0.0],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                true,
            );

            // Ears: two fins on the upper sides.
            for (part, zsign) in [
                (CreaturePart::EarLeft, 1.0_f32),
                (CreaturePart::EarRight, -1.0_f32),
            ] {
                let base_z = zsign * BODY_RADIUS * 0.45;
                builder.quad(
                    part,
                    [
                        [-0.15, BODY_RADIUS * 0.8, base_z],
                        [0.15, BODY_RADIUS * 0.8, base_z],
                        [0.05, BODY_RADIUS * 1.25, base_z + zsign * 0.18],
                        [-0.05, BODY_RADIUS * 1.25, base_z + zsign * 0.18],
                    ],
                    [0.0, 0.7, zsign * 0.7],
                    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                    true,
                );
            }

            // Nose: small knob at the upper front.
            builder.ellipsoid(
                CreaturePart::Nose,
                [face_x + 0.02, 0.32, 0.0],
                [0.09, 0.09, 0.09],
                eye_segs,
                eye_segs,
                [1.0, 1.0],
            );
        }
    }
    CreatureMeshData {
        positions: builder.positions,
        normals: builder.normals,
        uvs: builder.uvs,
        indices: builder.indices,
        parts: builder.parts,
    }
}

/// Build a hemispherical cap whose equator matches `ring`'s positions.
/// The cap shares exact positions with the ring (a closed union), but writes
/// its own poleward vertices with ellipsoid normals.
fn cap_hemisphere(
    builder: &mut MeshBuilder,
    center_x: f32,
    radius: f32,
    direction: f32,
    segments: u32,
    cap_rings: u32,
    equator: &[u32],
) {
    let mut previous_row: Vec<u32> = equator.to_vec();
    let mut row_index = 0u32;
    let pole_vertex;
    loop {
        row_index += 1;
        let t = row_index as f32 / (cap_rings + 1) as f32;
        let theta = t * core::f32::consts::FRAC_PI_2;
        let ring_radius = radius * theta.cos();
        let offset_x = direction * radius * theta.sin();
        if theta >= core::f32::consts::FRAC_PI_2 - 1e-4 {
            pole_vertex = builder.vertex(
                CreaturePart::Body,
                [center_x + direction * radius, 0.0, 0.0],
                [direction, 0.0, 0.0],
                [0.5, 1.0],
            );
            break;
        }
        let mut row = Vec::with_capacity(segments as usize);
        for seg in 0..segments {
            let u = seg as f32 / segments as f32;
            let phi = u * TAU;
            let (s, c) = (phi.sin(), phi.cos());
            let px = center_x + offset_x;
            let py = ring_radius * c;
            let pz = ring_radius * s;
            // Normal: outward from the sphere center.
            let normal = [direction * theta.sin(), theta.cos() * c, theta.cos() * s];
            row.push(builder.vertex(CreaturePart::Body, [px, py, pz], normal, [u, t]));
        }
        for seg in 0..segments as usize {
            let next = (seg + 1) % segments as usize;
            // Winding depends on the cap direction; keep CCW-outward.
            if direction > 0.0 {
                builder.triangle(previous_row[seg], row[seg], previous_row[next]);
                builder.triangle(row[seg], row[next], previous_row[next]);
            } else {
                builder.triangle(previous_row[seg], previous_row[next], row[seg]);
                builder.triangle(row[seg], previous_row[next], row[next]);
            }
        }
        previous_row = row;
    }
    for seg in 0..segments as usize {
        let next = (seg + 1) % segments as usize;
        if direction > 0.0 {
            builder.triangle(previous_row[seg], pole_vertex, previous_row[next]);
        } else {
            builder.triangle(previous_row[seg], previous_row[next], pole_vertex);
        }
    }
}

// ---------------------------------------------------------------------------
// Impostor bake: deterministic CPU rasterization of the LOD0 creature into
// the two-tile atlas sampled by the LOD2 crossed quads. A CPU bake (not GPU)
// because the atlas must be byte-identical across adapters/platforms — it is
// a build-time artifact with a deterministic content hash, not a per-run
// render target.
// ---------------------------------------------------------------------------

/// Baked impostor atlas for the LOD2 crossed quads.
///
/// Layout: two `tile_size x tile_size` tiles side by side (`width() =
/// 2 * tile_size`, `height() = tile_size`). Tile 0 is the profile view
/// (viewer at +Z, sees the X-Y silhouette with wheels), tile 1 is the front
/// view (viewer at +X, sees face/spike). Buffers are row-major RGBA8 with
/// row 0 = the top of the creature (v = 1), so a direct PNG dump is upright.
/// Uncovered texels are `(0, 0, 0, 0)` in both maps.
///
/// The albedo map holds canonical per-part tints ([`part_albedo`]): neutral
/// near-white body surfaces so the per-instance primary tint multiplies
/// cleanly, dark pupils/wheels that stay dark under that tint. The normal
/// map stores view-space normals per tile encoded as `n * 0.5 + 0.5`.
#[derive(Debug, Clone)]
pub struct ImpostorAtlas {
    tile_size: u32,
    albedo: Vec<u8>,
    normals: Vec<u8>,
}

impl ImpostorAtlas {
    /// Per-tile resolution; the atlas is two tiles wide.
    #[must_use]
    pub const fn tile_size(&self) -> u32 {
        self.tile_size
    }

    /// Atlas width in pixels (two tiles).
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.tile_size * 2
    }

    /// Atlas height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.tile_size
    }

    /// Albedo + coverage alpha, RGBA8 row-major.
    #[must_use]
    pub fn albedo_rgba8(&self) -> &[u8] {
        &self.albedo
    }

    /// View-space normal map (`n * 0.5 + 0.5`), RGBA8 row-major; alpha 255
    /// marks covered texels.
    #[must_use]
    pub fn normals_rgba8(&self) -> &[u8] {
        &self.normals
    }

    /// FNV-1a64 over both buffers: the provenance/regression fingerprint
    /// used by the determinism test and capture manifests.
    #[must_use]
    pub fn content_hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &byte in self.albedo.iter().chain(self.normals.iter()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }
}

/// Canonical per-part albedo tints for the impostor bake.
///
/// Body-adjacent surfaces are near-white so the A1/A5 instance shader's
/// per-instance tint multiplies without hue shift; intrinsically dark
/// features (pupils, wheels) are baked dark so they survive the same
/// multiplicative tint. Wheel gray comes from the shared visual semantics
/// (bd-2z0.14.3.2) instead of a second convention.
fn part_albedo(part: CreaturePart) -> [f32; 3] {
    match part {
        CreaturePart::Body => [0.90, 0.90, 0.93],
        CreaturePart::WheelLeft | CreaturePart::WheelRight => {
            scriptbots_core::visual::WHEEL_BASE_RGB
        }
        CreaturePart::Spike => [0.72, 0.75, 0.82],
        CreaturePart::EyeSclera(_) => [0.98, 0.98, 0.98],
        CreaturePart::EyePupil(_) => [0.04, 0.04, 0.06],
        CreaturePart::Mouth => [0.80, 0.30, 0.26],
        CreaturePart::EarLeft | CreaturePart::EarRight => [0.84, 0.84, 0.88],
        CreaturePart::Nose => [0.60, 0.50, 0.46],
    }
}

/// One orthographic impostor view: world axes mapping to (u, v, depth).
struct ImpostorView {
    u_axis: [f32; 3],
    v_axis: [f32; 3],
    depth_axis: [f32; 3],
    /// Tile column in the atlas (0 = left/profile, 1 = right/front).
    tile: u32,
}

/// Profile (viewer at +Z) and front (viewer at +X) views. `u` runs with
/// world +X (profile) / +Z (front), `v` runs with world +Y (up), depth is
/// nearest-to-viewer-wins along the view axis.
const IMPOSTOR_VIEWS: [ImpostorView; 2] = [
    ImpostorView {
        u_axis: [1.0, 0.0, 0.0],
        v_axis: [0.0, 1.0, 0.0],
        depth_axis: [0.0, 0.0, 1.0],
        tile: 0,
    },
    ImpostorView {
        u_axis: [0.0, 0.0, 1.0],
        v_axis: [0.0, 1.0, 0.0],
        depth_axis: [1.0, 0.0, 0.0],
        tile: 1,
    },
];

/// Bake the two-tile impostor atlas by software-rasterizing the LOD0
/// creature from the two [`IMPOSTOR_VIEWS`].
///
/// Deterministic: single-threaded, fixed triangle order, plain f32
/// arithmetic, no hash maps. Two calls with the same `tile_size` produce
/// byte-identical buffers (asserted in tests).
///
/// # Panics
/// Panics if `tile_size` is outside `8..=1024` (programmer error; callers
/// bake once at startup with a fixed size).
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn bake_impostor_atlas(tile_size: u32) -> ImpostorAtlas {
    assert!(
        (8..=1024).contains(&tile_size),
        "impostor tile_size {tile_size} outside 8..=1024"
    );
    let mesh = build_creature_mesh(CreatureLod::Lod0);
    let size = tile_size as usize;
    let width = size * 2;
    let mut albedo = vec![0_u8; width * size * 4];
    let mut normals = vec![0_u8; width * size * 4];
    let mut zbuf = vec![f32::NEG_INFINITY; size * size];

    for view in &IMPOSTOR_VIEWS {
        zbuf.fill(f32::NEG_INFINITY);
        let tile_x = view.tile as usize * size;
        for tri in mesh.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let part = mesh.parts[i0];
            let mut pu = [0.0_f32; 3];
            let mut pv = [0.0_f32; 3];
            let mut pd = [0.0_f32; 3];
            for (corner, &idx) in [i0, i1, i2].iter().enumerate() {
                let p = mesh.positions[idx];
                pu[corner] =
                    p[0].mul_add(view.u_axis[0], p[1] * view.u_axis[1]) + p[2] * view.u_axis[2];
                pv[corner] =
                    p[0].mul_add(view.v_axis[0], p[1] * view.v_axis[1]) + p[2] * view.v_axis[2];
                pd[corner] = p[0].mul_add(view.depth_axis[0], p[1] * view.depth_axis[1])
                    + p[2] * view.depth_axis[2];
            }
            // Map [-EXTENT, EXTENT] -> pixel space; row 0 = top (v = 1).
            let to_px = |u: f32| (u / IMPOSTOR_EXTENT * 0.5 + 0.5) * tile_size as f32;
            let mut sx = [0.0_f32; 3];
            let mut sy = [0.0_f32; 3];
            for corner in 0..3 {
                sx[corner] = to_px(pu[corner]);
                // Flip v so image row 0 is the top of the creature.
                sy[corner] = (tile_size as f32) - to_px(pv[corner]);
            }
            let area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0]);
            if area.abs() < 1e-12 {
                continue;
            }
            let inv_area = 1.0 / area;
            let min_x = sx
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
                .floor()
                .max(0.0) as usize;
            let max_x = sx
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .ceil()
                .min(tile_size as f32) as usize;
            let min_y = sy
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
                .floor()
                .max(0.0) as usize;
            let max_y = sy
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
                .ceil()
                .min(tile_size as f32) as usize;
            let color = part_albedo(part);
            for py in min_y..max_y {
                for px in min_x..max_x {
                    let cx = px as f32 + 0.5;
                    let cy = py as f32 + 0.5;
                    let w0 = ((sx[1] - cx) * (sy[2] - cy) - (sx[2] - cx) * (sy[1] - cy)) * inv_area;
                    let w1 = ((sx[2] - cx) * (sy[0] - cy) - (sx[0] - cx) * (sy[2] - cy)) * inv_area;
                    let w2 = 1.0 - w0 - w1;
                    // Accept either winding: `inv_area` carries the sign, so
                    // positive barycentrics mean inside for both.
                    if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                        continue;
                    }
                    let depth = w0.mul_add(pd[0], w1 * pd[1]) + w2 * pd[2];
                    let cell = py * size + px;
                    if depth <= zbuf[cell] {
                        continue;
                    }
                    zbuf[cell] = depth;
                    let n_raw = [
                        w0.mul_add(mesh.normals[i0][0], w1 * mesh.normals[i1][0])
                            + w2 * mesh.normals[i2][0],
                        w0.mul_add(mesh.normals[i0][1], w1 * mesh.normals[i1][1])
                            + w2 * mesh.normals[i2][1],
                        w0.mul_add(mesh.normals[i0][2], w1 * mesh.normals[i1][2])
                            + w2 * mesh.normals[i2][2],
                    ];
                    let n = normalize(n_raw);
                    let n_view = [
                        n[0].mul_add(view.u_axis[0], n[1] * view.u_axis[1]) + n[2] * view.u_axis[2],
                        n[0].mul_add(view.v_axis[0], n[1] * view.v_axis[1]) + n[2] * view.v_axis[2],
                        n[0].mul_add(view.depth_axis[0], n[1] * view.depth_axis[1])
                            + n[2] * view.depth_axis[2],
                    ];
                    let out = ((py * width) + tile_x + px) * 4;
                    for k in 0..3 {
                        albedo[out + k] = (color[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        normals[out + k] =
                            ((n_view[k] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                    albedo[out + 3] = 255;
                    normals[out + 3] = 255;
                }
            }
        }
    }
    ImpostorAtlas {
        tile_size,
        albedo,
        normals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn assert_normals_valid(mesh: &CreatureMeshData) {
        for (i, normal) in mesh.normals.iter().enumerate() {
            for component in normal {
                assert!(component.is_finite(), "normal {i} has non-finite component");
            }
            let len = (normal[0].mul_add(normal[0], normal[1] * normal[1]) + normal[2] * normal[2])
                .sqrt();
            assert!(
                (len - 1.0).abs() < 0.01,
                "normal {i} not unit length: {len} ({normal:?})"
            );
        }
    }

    fn assert_indices_and_uvs_valid(mesh: &CreatureMeshData) {
        let vertex_count = mesh.vertex_count() as u32;
        assert!(mesh.indices.len().is_multiple_of(3));
        for &index in &mesh.indices {
            assert!(
                index < vertex_count,
                "index {index} out of bounds {vertex_count}"
            );
        }
        for uv in &mesh.uvs {
            assert!(uv[0].is_finite() && uv[1].is_finite());
            assert!((0.0..=1.0).contains(&uv[0]), "uv.x out of range: {uv:?}");
            assert!((0.0..=1.0).contains(&uv[1]), "uv.y out of range: {uv:?}");
        }
        assert_eq!(mesh.positions.len(), mesh.normals.len());
        assert_eq!(mesh.positions.len(), mesh.uvs.len());
        assert_eq!(mesh.positions.len(), mesh.parts.len());
    }

    /// Watertightness of a single shell: every undirected edge appears in
    /// exactly two triangles (when restricted to the given part).
    fn assert_shell_watertight(mesh: &CreatureMeshData, part: CreaturePart, label: &str) {
        let part_vertices: std::collections::HashSet<u32> = mesh
            .parts
            .iter()
            .enumerate()
            .filter(|(_, p)| **p == part)
            .map(|(i, _)| i as u32)
            .collect();
        let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
        for tri in mesh.indices.as_chunks::<3>().0 {
            let in_part = tri.iter().filter(|v| part_vertices.contains(v)).count();
            if in_part < 3 {
                continue;
            }
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let key = if edge.0 < edge.1 {
                    edge
                } else {
                    (edge.1, edge.0)
                };
                *edges.entry(key).or_insert(0) += 1;
            }
        }
        let open: Vec<_> = edges.iter().filter(|(_, count)| **count != 2).collect();
        assert!(
            open.is_empty(),
            "{label}: {} open edges of {}",
            open.len(),
            edges.len()
        );
    }

    #[test]
    fn lod0_mesh_is_valid_and_within_budget() {
        let mesh = build_creature_mesh(CreatureLod::Lod0);
        assert_normals_valid(&mesh);
        assert_indices_and_uvs_valid(&mesh);
        assert!(
            mesh.vertex_count() <= 6000,
            "Lod0 budget: {} vertices",
            mesh.vertex_count()
        );
        assert!(
            mesh.triangle_count() >= 1500,
            "Lod0 detail: {}",
            mesh.triangle_count()
        );
        // Every expected part is present.
        for part in [
            CreaturePart::Body,
            CreaturePart::WheelLeft,
            CreaturePart::WheelRight,
            CreaturePart::Spike,
            CreaturePart::Mouth,
            CreaturePart::EarLeft,
            CreaturePart::EarRight,
            CreaturePart::Nose,
        ] {
            assert!(mesh.parts.contains(&part), "missing part {part:?}");
        }
        for eye in 0..NUM_EYES as u8 {
            assert!(mesh.parts.contains(&CreaturePart::EyeSclera(eye)));
            assert!(mesh.parts.contains(&CreaturePart::EyePupil(eye)));
        }
    }

    #[test]
    fn lod1_mesh_is_cheaper_than_lod0_and_valid() {
        let lod0 = build_creature_mesh(CreatureLod::Lod0);
        let lod1 = build_creature_mesh(CreatureLod::Lod1);
        assert_normals_valid(&lod1);
        assert_indices_and_uvs_valid(&lod1);
        assert!(lod1.vertex_count() < lod0.vertex_count());
        assert!(
            lod1.vertex_count() <= 2000,
            "Lod1 budget: {} vertices",
            lod1.vertex_count()
        );
    }

    #[test]
    fn lod2_impostor_is_two_crossed_quads() {
        let mesh = build_creature_mesh(CreatureLod::Lod2);
        assert_eq!(mesh.vertex_count(), 8, "two quads = 8 vertices");
        assert_eq!(
            mesh.indices.len(),
            24,
            "two quads x two sides x two tris x three indices"
        );
        assert_indices_and_uvs_valid(&mesh);
        // UV layout matches the two-tile impostor atlas: the profile quad
        // (X-Y plane, first 4 vertices) samples tile 0 (u in [0, 0.5]) and
        // the front quad (Z-Y plane, last 4) samples tile 1 (u in [0.5, 1]).
        for uv in &mesh.uvs[..4] {
            assert!(uv[0] <= 0.5, "profile quad uv in tile 0: {uv:?}");
        }
        for uv in &mesh.uvs[4..] {
            assert!(uv[0] >= 0.5, "front quad uv in tile 1: {uv:?}");
        }
    }

    #[test]
    fn wheels_and_spike_shells_are_watertight() {
        let mesh = build_creature_mesh(CreatureLod::Lod0);
        assert_shell_watertight(&mesh, CreaturePart::WheelLeft, "wheel left");
        assert_shell_watertight(&mesh, CreaturePart::WheelRight, "wheel right");
        assert_shell_watertight(&mesh, CreaturePart::Spike, "spike");
    }

    #[test]
    fn mesh_build_is_deterministic() {
        let a = build_creature_mesh(CreatureLod::Lod0);
        let b = build_creature_mesh(CreatureLod::Lod0);
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.normals, b.normals);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.parts, b.parts);
    }

    #[test]
    fn part_tags_and_ranges_are_consistent() {
        let mesh = build_creature_mesh(CreatureLod::Lod0);
        let ranges = mesh.part_vertex_ranges();
        let mut covered = 0usize;
        for (_part, range) in &ranges {
            covered += range.end - range.start;
        }
        assert_eq!(covered, mesh.vertex_count(), "ranges cover every vertex");
        // Tags are unique per part variant.
        assert_eq!(CreaturePart::Body.tag(), 0);
        assert_ne!(
            CreaturePart::EyeSclera(0).tag(),
            CreaturePart::EyePupil(0).tag()
        );
        assert_ne!(
            CreaturePart::EyeSclera(1).tag(),
            CreaturePart::EyeSclera(0).tag()
        );
    }

    #[test]
    fn proportions_for_traits_stays_valid_at_extremes() {
        for (smell, hearing, eye, spike) in [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 2.0),
            (0.5, 0.25, 0.75, 1.0),
            (f32::NAN, f32::INFINITY, -1.0, 99.0),
        ] {
            let proportions = proportions_for_traits(smell, hearing, eye, spike);
            assert!(
                proportions.validate().is_ok(),
                "extremes ({smell},{hearing},{eye},{spike}) must stay in bounds: {proportions:?}"
            );
        }
        // Trait response is monotonic.
        let small_nose = proportions_for_traits(0.0, 0.5, 0.5, 0.0).nose_scale;
        let big_nose = proportions_for_traits(1.0, 0.5, 0.5, 0.0).nose_scale;
        assert!(big_nose > small_nose, "smell trait grows the nose");
        let invalid = CreatureProportions {
            body_length: 99.0,
            ..CreatureProportions::default()
        };
        assert_eq!(invalid.validate(), Err("body_length"));
    }

    #[test]
    fn bevy_mesh_adapter_carries_all_attributes() {
        let mesh = build_creature_mesh(CreatureLod::Lod1);
        let bevy_mesh = mesh.to_bevy_mesh();
        assert!(bevy_mesh.attribute(Mesh::ATTRIBUTE_POSITION.id).is_some());
        assert!(bevy_mesh.attribute(Mesh::ATTRIBUTE_NORMAL.id).is_some());
        assert!(bevy_mesh.attribute(Mesh::ATTRIBUTE_UV_0.id).is_some());
        assert!(bevy_mesh.attribute(Mesh::ATTRIBUTE_TANGENT.id).is_some());
        assert!(bevy_mesh.attribute(ATTRIBUTE_CREATURE_PART.id).is_some());
        assert_eq!(bevy_mesh.count_vertices(), mesh.vertex_count());
    }

    #[test]
    fn tangents_are_finite_orthonormal_for_all_lods() {
        for lod in [CreatureLod::Lod0, CreatureLod::Lod1, CreatureLod::Lod2] {
            let mesh = build_creature_mesh(lod);
            let tangents = mesh.compute_tangents();
            assert_eq!(tangents.len(), mesh.vertex_count());
            for (i, tangent) in tangents.iter().enumerate() {
                for component in tangent {
                    assert!(component.is_finite(), "{lod:?} tangent {i} non-finite");
                }
                let len = (tangent[0].mul_add(tangent[0], tangent[1] * tangent[1])
                    + tangent[2] * tangent[2])
                    .sqrt();
                assert!((len - 1.0).abs() < 0.01, "{lod:?} tangent {i} length {len}");
                assert!(
                    tangent[3].abs() == 1.0,
                    "{lod:?} tangent {i} handedness {}",
                    tangent[3]
                );
                let n = mesh.normals[i];
                let dot = n[0].mul_add(tangent[0], n[1] * tangent[1]) + n[2] * tangent[2];
                assert!(
                    dot.abs() < 1e-3,
                    "{lod:?} tangent {i} not orthogonal: {dot}"
                );
            }
        }
    }

    #[test]
    fn impostor_bake_is_deterministic_and_shaped() {
        let a = bake_impostor_atlas(64);
        let b = bake_impostor_atlas(64);
        assert_eq!(a.albedo_rgba8(), b.albedo_rgba8(), "albedo byte-identical");
        assert_eq!(
            a.normals_rgba8(),
            b.normals_rgba8(),
            "normals byte-identical"
        );
        assert_eq!(a.content_hash(), b.content_hash());
        assert_eq!(a.width(), 128);
        assert_eq!(a.height(), 64);
        assert_eq!(a.albedo_rgba8().len(), 128 * 64 * 4);
        assert_eq!(a.normals_rgba8().len(), 128 * 64 * 4);
        // Smaller bake differs from larger bake (resolution actually matters).
        let small = bake_impostor_atlas(16);
        assert_ne!(a.content_hash(), small.content_hash());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn impostor_bake_renders_expected_features() {
        let atlas = bake_impostor_atlas(64);
        let size = atlas.tile_size() as usize;
        let width = atlas.width() as usize;
        let albedo = atlas.albedo_rgba8();
        let tile_stats = |tile: usize| {
            let mut covered = 0_usize;
            let mut min_lum = u8::MAX;
            let mut max_lum = u8::MIN;
            for py in 0..size {
                for px in 0..size {
                    let out = ((py * width) + tile * size + px) * 4;
                    if albedo[out + 3] == 0 {
                        continue;
                    }
                    covered += 1;
                    let lum = albedo[out].max(albedo[out + 1]).max(albedo[out + 2]);
                    min_lum = min_lum.min(lum);
                    max_lum = max_lum.max(lum);
                }
            }
            (covered, min_lum, max_lum)
        };
        for (tile, label) in [(0_usize, "profile"), (1_usize, "front")] {
            let (covered, min_lum, max_lum) = tile_stats(tile);
            let fill = covered as f32 / (size * size) as f32;
            assert!(
                (0.05..=0.85).contains(&fill),
                "{label} tile fill {fill} outside 5%..85% (creature neither absent nor square-filling)"
            );
            // Near-white body/sclera present.
            assert!(max_lum >= 220, "{label} max luminance {max_lum}");
            // Dark features (pupils, wheels) present.
            assert!(min_lum <= 60, "{label} min luminance {min_lum}");
        }
        // Normal map: covered texels carry alpha and plausible normals.
        let normals = atlas.normals_rgba8();
        for py in 0..size {
            for px in 0..width {
                let out = (py * width + px) * 4;
                let covered = albedo[out + 3] == 255;
                assert_eq!(
                    normals[out + 3] == 255,
                    covered,
                    "normal alpha disagrees with coverage at ({px},{py})"
                );
            }
        }
    }
}
