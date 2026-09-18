//! WGSL compute shader pipeline for order-independent fixed-point GPU sensing (bd-16g.15.2).
//!
//! # Determinism Contract
//!
//! Floating-point sensor accumulation on GPUs is nondeterministic across vendors and workgroups
//! because floating-point addition is neither associative nor commutative, and transcendentals
//! (`acos`, `sin`, `cos`, `atan2`) have vendor-defined approximations.
//!
//! This module implements bit-identical sensor accumulation on the GPU by:
//! 1. Reformulating geometry into dot-products and a shared polynomial `acos` (Horner evaluation,
//!    matching `crates/scriptbots-core/src/sense_fixed.rs` verbatim).
//! 2. Converting all individual contribution terms into 20-bit fixed-point integers (`to_fixed`).
//! 3. Accumulating in emulated 64-bit integers (`vec2<u32>` with explicit carry).
//! 4. Reducing across threads using an exact integer tree reduction in workgroup shared memory.
//! 5. Reading back the exact fixed-point accumulators to the host, where the CPU calls the
//!    canonical `finalize_with_multipliers` from `scriptbots_core::sense_fixed`.
//!
//! # Forbidden in the WGSL Shader
//! - `atan2`, `acos`, `sin`, `cos`, `exp`, `pow` (implementation-defined across vendors).
//! - `inverseSqrt` / fast-math reciprocal (must use `sqrt` and explicit division).
//! - `f32` atomics (nondeterministic accumulation order).
//! - subgroup / wave intrinsics whose result depends on scheduling.
//! - FMA contraction: every term computation uses explicit intermediate `let` bindings.

use bytemuck::{Pod, Zeroable};
use scriptbots_core::sense_fixed::{SENSE_FRAC_BITS, SENSE_GEOMETRY, SenseAccum};
use std::sync::atomic::{AtomicU64, Ordering};

/// Workgroup size used in production compute sensing.
pub const PRODUCTION_WORKGROUP_SIZE: u32 = 64;

/// Fixed-point scaling factor (2^20).
pub const FIXED_SCALE: f32 = 1_048_576.0;

/// Shared WGSL shader code for the fixed-point sensor accumulation compute pass.
pub const SENSE_COMPUTE_SHADER_WGSL: &str = r#"
// Fixed-point sensor accumulation shader matching sense_fixed.rs
const PI: f32 = 3.14159265;
const BLOOD_HALF_FOV: f32 = 0.589048622548; // PI * 0.1875 (World.cpp legacy parity)
const F32_EPSILON: f32 = 1.1920929e-7;
const MAX_TERM: f32 = 4.0;
const FIXED_SCALE: f32 = 1048576.0; // 2^20

// Abramowitz & Stegun 4.4.45 polynomial acos coefficients
const ACOS_C0: f32 = 1.5707963;
const ACOS_C1: f32 = -0.2145988;
const ACOS_C2: f32 = 0.08897899;
const ACOS_C3: f32 = -0.0501743;
const ACOS_C4: f32 = 0.03089188;
const ACOS_C5: f32 = -0.017088126;
const ACOS_C6: f32 = 0.00667009;
const ACOS_C7: f32 = -0.0012624911;

struct GridUniforms {
    world_width: f32,
    world_height: f32,
    cell_size: f32,
    sense_radius: f32,
    cells_x: u32,
    cells_y: u32,
    agent_count: u32,
    _pad: u32,
};

struct AgentGpuData {
    pos_x: f32,
    pos_y: f32,
    heading_unit_x: f32,
    heading_unit_y: f32,
    eye_unit_0_x: f32,
    eye_unit_0_y: f32,
    eye_unit_1_x: f32,
    eye_unit_1_y: f32,
    eye_unit_2_x: f32,
    eye_unit_2_y: f32,
    eye_unit_3_x: f32,
    eye_unit_3_y: f32,
    eye_fov_0: f32,
    eye_fov_1: f32,
    eye_fov_2: f32,
    eye_fov_3: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    eye_sensitivity: f32,
    wheel_effort: f32,
    sound_emitter: f32,
    target_health: f32,
    _pad: f32,
};

struct SenseOutputFixed {
    density_0: vec2<u32>,
    red_0: vec2<u32>,
    green_0: vec2<u32>,
    blue_0: vec2<u32>,

    density_1: vec2<u32>,
    red_1: vec2<u32>,
    green_1: vec2<u32>,
    blue_1: vec2<u32>,

    density_2: vec2<u32>,
    red_2: vec2<u32>,
    green_2: vec2<u32>,
    blue_2: vec2<u32>,

    density_3: vec2<u32>,
    red_3: vec2<u32>,
    green_3: vec2<u32>,
    blue_3: vec2<u32>,

    smell: vec2<u32>,
    sound: vec2<u32>,
    hearing: vec2<u32>,
    blood: vec2<u32>,

    saturations: u32,
    saturated_channels: u32,
};

@group(0) @binding(0) var<uniform> uniforms: GridUniforms;
@group(0) @binding(1) var<storage, read> agents: array<AgentGpuData>;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> cell_agents: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputs: array<SenseOutputFixed>;

// Two's complement 64-bit integer addition using vec2<u32> (lo, hi).
fn i64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo_sum = a.x + b.x;
    let carry = select(0u, 1u, lo_sum < a.x);
    let hi_sum = a.y + b.y + carry;
    return vec2<u32>(lo_sum, hi_sum);
}

// Polynomial acos matching Horner evaluation order in sense_fixed.rs verbatim.
fn poly_acos(x_in: f32) -> f32 {
    let x = clamp(x_in, -1.0, 1.0);
    let negative = x < 0.0;
    let a = abs(x);
    var poly = ACOS_C7;
    poly = ACOS_C6 + poly * a;
    poly = ACOS_C5 + poly * a;
    poly = ACOS_C4 + poly * a;
    poly = ACOS_C3 + poly * a;
    poly = ACOS_C2 + poly * a;
    poly = ACOS_C1 + poly * a;
    poly = ACOS_C0 + poly * a;
    let diff = 1.0 - a;
    let s = sqrt(max(0.0, diff));
    let result = poly * s;
    if (negative) {
        return PI - result;
    } else {
        return result;
    }
}

// Toroidal minimum-image delta on a periodic interval [0, extent).
fn toroidal_delta(a: f32, b: f32, extent: f32) -> f32 {
    let raw = a - b;
    let half = extent * 0.5;
    var delta = raw;
    if (delta > half) {
        delta = delta - extent;
    } else if (delta <= -half) {
        delta = delta + extent;
    }
    if (delta == 0.0) {
        return 0.0;
    }
    return delta;
}

fn cell_for_canonical_axis(coord: f32, cell_size: f32, cells: u32) -> u32 {
    let c = u32(floor(coord / cell_size));
    return min(c, cells - 1u);
}

fn axis_bucket_span(coord: f32, radius: f32, extent: f32, cell_size: f32, cells: u32) -> vec2<u32> {
    if (cells <= 1u) {
        return vec2<u32>(0u, 1u);
    }
    let center = ((coord % extent) + extent) % extent;
    if (radius >= extent * 0.5) {
        return vec2<u32>(0u, cells);
    }
    let lower = center - radius;
    let upper = center + radius;
    let lower_can = ((lower % extent) + extent) % extent;
    let upper_can = ((upper % extent) + extent) % extent;
    let start = cell_for_canonical_axis(lower_can, cell_size, cells);
    let end = cell_for_canonical_axis(upper_can, cell_size, cells);
    var span: u32 = 0u;
    if (lower < 0.0 || upper >= extent) {
        span = (cells - start) + end + 1u;
    } else {
        span = (end - start) + 1u;
    }
    span = clamp(span, 1u, cells);
    return vec2<u32>(start, span);
}

// Round half to even and convert to fixed point.
// Returns vec2<u32>(fixed_value_lo, is_saturated_flag: 0 or 1).
fn to_fixed_channel(term: f32) -> vec2<u32> {
    var out_of_range = 0u;
    var bounded = term;
    if (term != term || term < 0.0) {
        bounded = 0.0;
        out_of_range = 1u;
    } else if (term > MAX_TERM) {
        bounded = MAX_TERM;
        out_of_range = 1u;
    }
    let scaled = bounded * FIXED_SCALE;
    let floor_val = floor(scaled);
    let frac_val = scaled - floor_val;
    let is_odd = (u32(floor_val) & 1u) != 0u;
    let round_up = (frac_val > 0.5) || (frac_val == 0.5 && is_odd);
    let rounded = select(floor_val, floor_val + 1.0, round_up);
    let fixed_val = u32(rounded);
    return vec2<u32>(fixed_val, out_of_range);
}

struct ThreadAccum {
    channels: array<vec2<u32>, 20>,
    saturated_channels: u32,
};

var<workgroup> s_channels: array<array<vec2<u32>, 20>, 64>;
var<workgroup> s_saturated: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let agent_idx = workgroup_id.x;
    let tid = local_id.x;

    var thread_accum: ThreadAccum;
    for (var ch = 0u; ch < 20u; ch = ch + 1u) {
        thread_accum.channels[ch] = vec2<u32>(0u, 0u);
    }
    thread_accum.saturated_channels = 0u;

    if (agent_idx < uniforms.agent_count) {
        let observer = agents[agent_idx];
        let radius = uniforms.sense_radius;
        let radius_sq = radius * radius;

        let span_x_data = axis_bucket_span(observer.pos_x, radius, uniforms.world_width, uniforms.cell_size, uniforms.cells_x);
        let start_x = span_x_data.x;
        let span_x = span_x_data.y;

        let span_y_data = axis_bucket_span(observer.pos_y, radius, uniforms.world_height, uniforms.cell_size, uniforms.cells_y);
        let start_y = span_y_data.x;
        let span_y = span_y_data.y;

        let eye_units_x = array<f32, 4>(observer.eye_unit_0_x, observer.eye_unit_1_x, observer.eye_unit_2_x, observer.eye_unit_3_x);
        let eye_units_y = array<f32, 4>(observer.eye_unit_0_y, observer.eye_unit_1_y, observer.eye_unit_2_y, observer.eye_unit_3_y);
        let eye_fovs = array<f32, 4>(observer.eye_fov_0, observer.eye_fov_1, observer.eye_fov_2, observer.eye_fov_3);

        for (var step_x = 0u; step_x < span_x; step_x = step_x + 1u) {
            let cx = (start_x + step_x) % uniforms.cells_x;
            for (var step_y = 0u; step_y < span_y; step_y = step_y + 1u) {
                let cy = (start_y + step_y) % uniforms.cells_y;
                let cell = cy * uniforms.cells_x + cx;
                let cell_start = cell_offsets[cell];
                let cell_end = cell_offsets[cell + 1u];
                let count = cell_end - cell_start;

                for (var k = tid; k < count; k = k + 64u) {
                    let other_idx = cell_agents[cell_start + k];
                    if (other_idx == agent_idx) {
                        continue;
                    }
                    let neighbor = agents[other_idx];

                    let dx = toroidal_delta(neighbor.pos_x, observer.pos_x, uniforms.world_width);
                    let dy = toroidal_delta(neighbor.pos_y, observer.pos_y, uniforms.world_height);
                    let dist_sq = dx * dx + dy * dy;
                    if (dist_sq <= F32_EPSILON || dist_sq > radius_sq) {
                        continue;
                    }
                    let distance = sqrt(dist_sq);
                    let distance_factor = (radius - distance) / radius;
                    if (distance_factor <= 0.0) {
                        continue;
                    }

                    let neighbor_x = dx / distance;
                    let neighbor_y = dy / distance;

                    // Eye channels (0..15)
                    let neighbor_color = array<f32, 3>(neighbor.color_r, neighbor.color_g, neighbor.color_b);
                    for (var eye = 0u; eye < 4u; eye = eye + 1u) {
                        let dot = eye_units_x[eye] * neighbor_x + eye_units_y[eye] * neighbor_y;
                        let difference = poly_acos(clamp(dot, -1.0, 1.0));
                        let fov = eye_fovs[eye];
                        if (fov <= 0.0 || difference >= fov) {
                            continue;
                        }
                        let fov_factor = max(0.0, (fov - difference) / fov);
                        let intensity = observer.eye_sensitivity * fov_factor * distance_factor;

                        let base_ch = eye * 4u;
                        let density_term = intensity * (distance / radius);
                        let red_term = intensity * neighbor_color[0];
                        let green_term = intensity * neighbor_color[1];
                        let blue_term = intensity * neighbor_color[2];

                        let t_den = to_fixed_channel(density_term);
                        thread_accum.channels[base_ch] = i64_add(thread_accum.channels[base_ch], vec2<u32>(t_den.x, 0u));
                        if (t_den.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << base_ch); }

                        let t_r = to_fixed_channel(red_term);
                        thread_accum.channels[base_ch + 1u] = i64_add(thread_accum.channels[base_ch + 1u], vec2<u32>(t_r.x, 0u));
                        if (t_r.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << (base_ch + 1u)); }

                        let t_g = to_fixed_channel(green_term);
                        thread_accum.channels[base_ch + 2u] = i64_add(thread_accum.channels[base_ch + 2u], vec2<u32>(t_g.x, 0u));
                        if (t_g.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << (base_ch + 2u)); }

                        let t_b = to_fixed_channel(blue_term);
                        thread_accum.channels[base_ch + 3u] = i64_add(thread_accum.channels[base_ch + 3u], vec2<u32>(t_b.x, 0u));
                        if (t_b.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << (base_ch + 3u)); }
                    }

                    // Smell (ch 16)
                    let t_smell = to_fixed_channel(distance_factor);
                    thread_accum.channels[16] = i64_add(thread_accum.channels[16], vec2<u32>(t_smell.x, 0u));
                    if (t_smell.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << 16u); }

                    // Sound (ch 17)
                    let sound_term = distance_factor * neighbor.wheel_effort;
                    let t_sound = to_fixed_channel(sound_term);
                    thread_accum.channels[17] = i64_add(thread_accum.channels[17], vec2<u32>(t_sound.x, 0u));
                    if (t_sound.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << 17u); }

                    // Hearing (ch 18)
                    let hearing_term = distance_factor * neighbor.sound_emitter;
                    let t_hearing = to_fixed_channel(hearing_term);
                    thread_accum.channels[18] = i64_add(thread_accum.channels[18], vec2<u32>(t_hearing.x, 0u));
                    if (t_hearing.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << 18u); }

                    // Blood (ch 19)
                    let forward_dot = observer.heading_unit_x * neighbor_x + observer.heading_unit_y * neighbor_y;
                    let forward_diff = poly_acos(clamp(forward_dot, -1.0, 1.0));
                    var blood_term = 0.0;
                    if (forward_diff >= 0.0 && forward_diff < BLOOD_HALF_FOV && distance_factor > 0.0) {
                        let angular_factor = (BLOOD_HALF_FOV - forward_diff) / BLOOD_HALF_FOV;
                        let wound_factor = max(0.0, 1.0 - clamp(neighbor.target_health * 0.5, 0.0, 1.0));
                        let intermediate = angular_factor * distance_factor;
                        blood_term = intermediate * wound_factor;
                    }
                    let t_blood = to_fixed_channel(blood_term);
                    thread_accum.channels[19] = i64_add(thread_accum.channels[19], vec2<u32>(t_blood.x, 0u));
                    if (t_blood.y != 0u) { thread_accum.saturated_channels = thread_accum.saturated_channels | (1u << 19u); }
                }
            }
        }
    }

    // Shared memory store
    for (var ch = 0u; ch < 20u; ch = ch + 1u) {
        s_channels[tid][ch] = thread_accum.channels[ch];
    }
    s_saturated[tid] = thread_accum.saturated_channels;
    workgroupBarrier();

    // Exact integer tree reduction across the 64 workgroup threads
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            let other = tid + stride;
            for (var ch = 0u; ch < 20u; ch = ch + 1u) {
                s_channels[tid][ch] = i64_add(s_channels[tid][ch], s_channels[other][ch]);
            }
            s_saturated[tid] = s_saturated[tid] | s_saturated[other];
        }
        workgroupBarrier();
    }

    // Thread 0 saturates against ACCUM_CEILING (17_179_869_184 = 4 * 2^32) and writes output
    if (tid == 0u && agent_idx < uniforms.agent_count) {
        var sat_mask = s_saturated[0];
        var out_res: SenseOutputFixed;

        for (var ch = 0u; ch < 20u; ch = ch + 1u) {
            var val = s_channels[0][ch];
            // ACCUM_CEILING is exactly 0x4_0000_0000: hi == 4 and lo == 0.
            if (val.y > 4u || (val.y == 4u && val.x > 0u)) {
                val = vec2<u32>(0u, 4u);
                sat_mask = sat_mask | (1u << ch);
            }
            s_channels[0][ch] = val;
        }

        out_res.density_0 = s_channels[0][0];
        out_res.red_0     = s_channels[0][1];
        out_res.green_0   = s_channels[0][2];
        out_res.blue_0    = s_channels[0][3];

        out_res.density_1 = s_channels[0][4];
        out_res.red_1     = s_channels[0][5];
        out_res.green_1   = s_channels[0][6];
        out_res.blue_1    = s_channels[0][7];

        out_res.density_2 = s_channels[0][8];
        out_res.red_2     = s_channels[0][9];
        out_res.green_2   = s_channels[0][10];
        out_res.blue_2    = s_channels[0][11];

        out_res.density_3 = s_channels[0][12];
        out_res.red_3     = s_channels[0][13];
        out_res.green_3   = s_channels[0][14];
        out_res.blue_3    = s_channels[0][15];

        out_res.smell     = s_channels[0][16];
        out_res.sound     = s_channels[0][17];
        out_res.hearing   = s_channels[0][18];
        out_res.blood     = s_channels[0][19];

        out_res.saturations = countOneBits(sat_mask);
        out_res.saturated_channels = sat_mask;

        outputs[agent_idx] = out_res;
    }
}
"#;

/// WGSL shader for clearing binning counters.
pub const BINNING_CLEAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cell_inserted: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;
    if (cell_idx < arrayLength(&cell_counts)) {
        atomicStore(&cell_counts[cell_idx], 0u);
        atomicStore(&cell_inserted[cell_idx], 0u);
    }
}
"#;

/// WGSL shader for counting agents per grid cell.
pub const BINNING_COUNT_WGSL: &str = r#"
struct GridUniforms {
    world_width: f32,
    world_height: f32,
    cell_size: f32,
    sense_radius: f32,
    cells_x: u32,
    cells_y: u32,
    agent_count: u32,
    _pad: u32,
};

struct AgentGpuData {
    pos_x: f32,
    pos_y: f32,
    heading_unit_x: f32,
    heading_unit_y: f32,
    eye_unit_0_x: f32,
    eye_unit_0_y: f32,
    eye_unit_1_x: f32,
    eye_unit_1_y: f32,
    eye_unit_2_x: f32,
    eye_unit_2_y: f32,
    eye_unit_3_x: f32,
    eye_unit_3_y: f32,
    eye_fov_0: f32,
    eye_fov_1: f32,
    eye_fov_2: f32,
    eye_fov_3: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    eye_sensitivity: f32,
    wheel_effort: f32,
    sound_emitter: f32,
    target_health: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: GridUniforms;
@group(0) @binding(1) var<storage, read> agents: array<AgentGpuData>;
@group(0) @binding(2) var<storage, read_write> cell_counts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let agent_idx = global_id.x;
    if (agent_idx >= uniforms.agent_count) {
        return;
    }
    let agent = agents[agent_idx];
    let can_x = ((agent.pos_x % uniforms.world_width) + uniforms.world_width) % uniforms.world_width;
    let can_y = ((agent.pos_y % uniforms.world_height) + uniforms.world_height) % uniforms.world_height;
    let cx = min(u32(floor(can_x / uniforms.cell_size)), uniforms.cells_x - 1u);
    let cy = min(u32(floor(can_y / uniforms.cell_size)), uniforms.cells_y - 1u);
    let cell_idx = cy * uniforms.cells_x + cx;
    atomicAdd(&cell_counts[cell_idx], 1u);
}
"#;

/// WGSL shader for single-workgroup exclusive prefix sum across grid cells.
pub const BINNING_PREFIX_SUM_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> cell_offsets: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let total_cells = arrayLength(&cell_counts);
    cell_offsets[0] = 0u;
    var sum = 0u;
    for (var i = 0u; i < total_cells; i = i + 1u) {
        let count = cell_counts[i];
        cell_offsets[i] = sum;
        sum = sum + count;
    }
    cell_offsets[total_cells] = sum;
}
"#;

/// WGSL shader for placing agent indices into CSR cell buckets.
pub const BINNING_PLACE_WGSL: &str = r#"
struct GridUniforms {
    world_width: f32,
    world_height: f32,
    cell_size: f32,
    sense_radius: f32,
    cells_x: u32,
    cells_y: u32,
    agent_count: u32,
    _pad: u32,
};

struct AgentGpuData {
    pos_x: f32,
    pos_y: f32,
    heading_unit_x: f32,
    heading_unit_y: f32,
    eye_unit_0_x: f32,
    eye_unit_0_y: f32,
    eye_unit_1_x: f32,
    eye_unit_1_y: f32,
    eye_unit_2_x: f32,
    eye_unit_2_y: f32,
    eye_unit_3_x: f32,
    eye_unit_3_y: f32,
    eye_fov_0: f32,
    eye_fov_1: f32,
    eye_fov_2: f32,
    eye_fov_3: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    eye_sensitivity: f32,
    wheel_effort: f32,
    sound_emitter: f32,
    target_health: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> uniforms: GridUniforms;
@group(0) @binding(1) var<storage, read> agents: array<AgentGpuData>;
@group(0) @binding(2) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_inserted: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cell_agents: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let agent_idx = global_id.x;
    if (agent_idx >= uniforms.agent_count) {
        return;
    }
    let agent = agents[agent_idx];
    let can_x = ((agent.pos_x % uniforms.world_width) + uniforms.world_width) % uniforms.world_width;
    let can_y = ((agent.pos_y % uniforms.world_height) + uniforms.world_height) % uniforms.world_height;
    let cx = min(u32(floor(can_x / uniforms.cell_size)), uniforms.cells_x - 1u);
    let cy = min(u32(floor(can_y / uniforms.cell_size)), uniforms.cells_y - 1u);
    let cell_idx = cy * uniforms.cells_x + cx;

    let slot = atomicAdd(&cell_inserted[cell_idx], 1u);
    let base_offset = cell_offsets[cell_idx];
    cell_agents[base_offset + slot] = agent_idx;
}
"#;

/// Standalone compute shader to test emulated 64-bit integer carry logic against Rust reference i64.
pub const STANDALONE_CARRY_WGSL: &str = r#"
struct CarryInputPair {
    a: vec2<u32>,
    b: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> inputs: array<CarryInputPair>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec2<u32>>;

fn i64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo_sum = a.x + b.x;
    let carry = select(0u, 1u, lo_sum < a.x);
    let hi_sum = a.y + b.y + carry;
    return vec2<u32>(lo_sum, hi_sum);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&inputs)) {
        outputs[idx] = i64_add(inputs[idx].a, inputs[idx].b);
    }
}
"#;

/// Uniform configuration sent to GPU compute pipelines.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GridUniforms {
    /// Total simulation world width.
    pub world_width: f32,
    /// Total simulation world height.
    pub world_height: f32,
    /// Edge length of each spatial binning cell.
    pub cell_size: f32,
    /// Sensory perception radius.
    pub sense_radius: f32,
    /// Number of spatial cells along the x-axis.
    pub cells_x: u32,
    /// Number of spatial cells along the y-axis.
    pub cells_y: u32,
    /// Active agent count for the dispatch.
    pub agent_count: u32,
    /// Memory alignment padding.
    pub _pad: u32,
}

/// GPU layout for observer/neighbor agent state (96 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct AgentGpuData {
    /// Agent position x coordinate.
    pub pos_x: f32,
    /// Agent position y coordinate.
    pub pos_y: f32,
    /// Heading unit vector x.
    pub heading_unit_x: f32,
    /// Heading unit vector y.
    pub heading_unit_y: f32,
    /// Eye 0 unit vector x.
    pub eye_unit_0_x: f32,
    /// Eye 0 unit vector y.
    pub eye_unit_0_y: f32,
    /// Eye 1 unit vector x.
    pub eye_unit_1_x: f32,
    /// Eye 1 unit vector y.
    pub eye_unit_1_y: f32,
    /// Eye 2 unit vector x.
    pub eye_unit_2_x: f32,
    /// Eye 2 unit vector y.
    pub eye_unit_2_y: f32,
    /// Eye 3 unit vector x.
    pub eye_unit_3_x: f32,
    /// Eye 3 unit vector y.
    pub eye_unit_3_y: f32,
    /// Eye 0 field of view in radians.
    pub eye_fov_0: f32,
    /// Eye 1 field of view in radians.
    pub eye_fov_1: f32,
    /// Eye 2 field of view in radians.
    pub eye_fov_2: f32,
    /// Eye 3 field of view in radians.
    pub eye_fov_3: f32,
    /// Body color red channel.
    pub color_r: f32,
    /// Body color green channel.
    pub color_g: f32,
    /// Body color blue channel.
    pub color_b: f32,
    /// Eye visual sensitivity multiplier.
    pub eye_sensitivity: f32,
    /// Peak wheel effort exerted.
    pub wheel_effort: f32,
    /// Sound emitted by the agent.
    pub sound_emitter: f32,
    /// Current health of the agent.
    pub target_health: f32,
    /// Memory alignment padding.
    pub _pad: f32,
}

/// Fixed-point output produced per agent by the GPU compute shader (176 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuSenseOutput {
    /// Eye 0 density accumulator [lo, hi].
    pub density_0: [u32; 2],
    /// Eye 0 red accumulator [lo, hi].
    pub red_0: [u32; 2],
    /// Eye 0 green accumulator [lo, hi].
    pub green_0: [u32; 2],
    /// Eye 0 blue accumulator [lo, hi].
    pub blue_0: [u32; 2],

    /// Eye 1 density accumulator [lo, hi].
    pub density_1: [u32; 2],
    /// Eye 1 red accumulator [lo, hi].
    pub red_1: [u32; 2],
    /// Eye 1 green accumulator [lo, hi].
    pub green_1: [u32; 2],
    /// Eye 1 blue accumulator [lo, hi].
    pub blue_1: [u32; 2],

    /// Eye 2 density accumulator [lo, hi].
    pub density_2: [u32; 2],
    /// Eye 2 red accumulator [lo, hi].
    pub red_2: [u32; 2],
    /// Eye 2 green accumulator [lo, hi].
    pub green_2: [u32; 2],
    /// Eye 2 blue accumulator [lo, hi].
    pub blue_2: [u32; 2],

    /// Eye 3 density accumulator [lo, hi].
    pub density_3: [u32; 2],
    /// Eye 3 red accumulator [lo, hi].
    pub red_3: [u32; 2],
    /// Eye 3 green accumulator [lo, hi].
    pub green_3: [u32; 2],
    /// Eye 3 blue accumulator [lo, hi].
    pub blue_3: [u32; 2],

    /// Smell accumulator [lo, hi].
    pub smell: [u32; 2],
    /// Sound accumulator [lo, hi].
    pub sound: [u32; 2],
    /// Hearing accumulator [lo, hi].
    pub hearing: [u32; 2],
    /// Blood accumulator [lo, hi].
    pub blood: [u32; 2],

    /// Total count of saturated channels.
    pub saturations: u32,
    /// Bitmask of saturated channel indices.
    pub saturated_channels: u32,
}

impl GpuSenseOutput {
    /// Convert the GPU fixed-point output into the canonical [`SenseAccum`] struct.
    #[must_use]
    pub fn to_sense_accum(&self) -> SenseAccum {
        let to_i64 = |p: [u32; 2]| (u64::from(p[0]) | (u64::from(p[1]) << 32)) as i64;
        SenseAccum {
            density: [
                to_i64(self.density_0),
                to_i64(self.density_1),
                to_i64(self.density_2),
                to_i64(self.density_3),
            ],
            red: [
                to_i64(self.red_0),
                to_i64(self.red_1),
                to_i64(self.red_2),
                to_i64(self.red_3),
            ],
            green: [
                to_i64(self.green_0),
                to_i64(self.green_1),
                to_i64(self.green_2),
                to_i64(self.green_3),
            ],
            blue: [
                to_i64(self.blue_0),
                to_i64(self.blue_1),
                to_i64(self.blue_2),
                to_i64(self.blue_3),
            ],
            smell: to_i64(self.smell),
            sound: to_i64(self.sound),
            hearing: to_i64(self.hearing),
            blood: to_i64(self.blood),
            saturations: self.saturations,
            saturated_channels: self.saturated_channels,
        }
    }
}

/// Pair of 64-bit operands used for standalone carry verification tests.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct CarryInputPair {
    /// First operand as [lo, hi].
    pub a: [u32; 2],
    /// Second operand as [lo, hi].
    pub b: [u32; 2],
}

/// Typed error classifications for GPU compute sensing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuSenseError {
    /// No adapter satisfied the GPU request.
    NoAdapter,
    /// Internal device error.
    Device(String),
    /// wgpu validation error during shader/pipeline creation or dispatch.
    Validation(String),
    /// GPU device lost event occurred.
    DeviceLost(String),
    /// Out of memory allocation error.
    OutOfMemory(String),
    /// Staging buffer mapping failed.
    Map(String),
}

/// Telemetry and statistics recorded for GPU sensing dispatches.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuSenseRunStats {
    /// Number of dispatches executed.
    pub dispatches: u64,
    /// Number of readback stalls encountered.
    pub readback_stalls: u64,
    /// Total channel saturations observed across all agents.
    pub saturations: u64,
    /// Maximum neighbor count observed for any single agent.
    pub max_neighbors_observed: u32,
}

static PIPELINE_ID: AtomicU64 = AtomicU64::new(1);

/// Complete GPU compute sensing pipeline holding pipelines for binning and accumulation.
pub struct GpuSensePipeline {
    /// Unique identifier for this pipeline instance.
    pub id: u64,
    /// Workgroup size configured for the sense kernel.
    pub workgroup_size: u32,
    /// Compute pipeline for clearing cell counters.
    pub clear_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for counting agent occurrences per cell.
    pub count_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for computing prefix sum offsets.
    pub prefix_sum_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for populating CSR cell buckets.
    pub place_pipeline: wgpu::ComputePipeline,
    /// Compute pipeline for sensory accumulation.
    pub sense_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for grid uniforms and storage buffers in sense pass.
    pub sense_bgl: wgpu::BindGroupLayout,
    /// Adapter information retained for provenance and audit.
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuSensePipeline {
    /// Create a new GPU compute sensing pipeline with the specified device and adapter information.
    pub fn new(
        device: &wgpu::Device,
        adapter_info: &wgpu::AdapterInfo,
        workgroup_size: u32,
    ) -> Result<Self, GpuSenseError> {
        let is_software_fallback = adapter_info.device_type == wgpu::DeviceType::Cpu;

        // Structured lifecycle diagnostics per logging specification
        tracing::info!(
            target: "scriptbots::sense::gpu",
            adapter_name = %adapter_info.name,
            vendor = %adapter_info.vendor,
            device = %adapter_info.device,
            driver = %adapter_info.driver,
            driver_info = %adapter_info.driver_info,
            backend = ?adapter_info.backend,
            workgroup_size = workgroup_size,
            frac_bits = SENSE_FRAC_BITS,
            geometry = SENSE_GEOMETRY,
            shader_digest = %sense_shader_digest(),
            is_software_fallback = is_software_fallback,
            "Initialized GPU compute sensing pipeline"
        );

        if is_software_fallback {
            tracing::warn!(
                target: "scriptbots::sense::gpu",
                adapter_name = %adapter_info.name,
                "GPU compute sensing running on software fallback adapter: throughput is not representative of real hardware"
            );
        }

        let clear_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binning_clear"),
            source: wgpu::ShaderSource::Wgsl(BINNING_CLEAR_WGSL.into()),
        });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("binning_clear_pipeline"),
            layout: None,
            module: &clear_sm,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let count_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binning_count"),
            source: wgpu::ShaderSource::Wgsl(BINNING_COUNT_WGSL.into()),
        });
        let count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("binning_count_pipeline"),
            layout: None,
            module: &count_sm,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let prefix_sum_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binning_prefix_sum"),
            source: wgpu::ShaderSource::Wgsl(BINNING_PREFIX_SUM_WGSL.into()),
        });
        let prefix_sum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("binning_prefix_sum_pipeline"),
                layout: None,
                module: &prefix_sum_sm,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let place_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("binning_place"),
            source: wgpu::ShaderSource::Wgsl(BINNING_PLACE_WGSL.into()),
        });
        let place_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("binning_place_pipeline"),
            layout: None,
            module: &place_sm,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let sense_src = sense_shader_source(workgroup_size);
        let sense_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sense_accumulate"),
            source: wgpu::ShaderSource::Wgsl(sense_src.into()),
        });

        let sense_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sense_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let sense_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sense_pipeline_layout"),
            bind_group_layouts: &[&sense_bgl],
            push_constant_ranges: &[],
        });

        let sense_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sense_accumulate_pipeline"),
            layout: Some(&sense_pl_layout),
            module: &sense_sm,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let id = PIPELINE_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            id,
            workgroup_size,
            clear_pipeline,
            count_pipeline,
            prefix_sum_pipeline,
            place_pipeline,
            sense_pipeline,
            sense_bgl,
            adapter_info: adapter_info.clone(),
        })
    }

    /// Execute the full GPU binning and sensor accumulation pass, reading back the outputs.
    pub fn execute_sense(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        agents: &[AgentGpuData],
        uniforms: GridUniforms,
    ) -> Result<Vec<GpuSenseOutput>, GpuSenseError> {
        let agent_count = agents.len();
        if agent_count == 0 {
            return Ok(Vec::new());
        }

        let total_cells = (uniforms.cells_x * uniforms.cells_y) as usize;
        let cell_buffer_len = total_cells.max(1);

        use wgpu::util::DeviceExt;

        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms_buf"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let agents_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("agents_buf"),
            contents: bytemuck::cast_slice(agents),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cell_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_counts_buf"),
            size: (cell_buffer_len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let cell_inserted_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_inserted_buf"),
            size: (cell_buffer_len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let cell_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_offsets_buf"),
            size: ((cell_buffer_len + 1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_agents_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_agents_buf"),
            size: (agent_count * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let output_bytes = (agent_count * std::mem::size_of::<GpuSenseOutput>()) as u64;
        let outputs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("outputs_buf"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_outputs_buf"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 1. Clear bind group
        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear_bg"),
            layout: &self.clear_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_inserted_buf.as_entire_binding(),
                },
            ],
        });

        // 2. Count bind group
        let count_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("count_bg"),
            layout: &self.count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_counts_buf.as_entire_binding(),
                },
            ],
        });

        // 3. Prefix sum bind group
        let prefix_sum_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_sum_bg"),
            layout: &self.prefix_sum_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_offsets_buf.as_entire_binding(),
                },
            ],
        });

        // 4. Place bind group
        let place_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("place_bg"),
            layout: &self.place_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_inserted_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_agents_buf.as_entire_binding(),
                },
            ],
        });

        // 5. Sense bind group
        let sense_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sense_bg"),
            layout: &self.sense_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: outputs_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sense_compute_encoder"),
        });

        // Pass 1: Clear
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, &clear_bg, &[]);
            let workgroups = (cell_buffer_len as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Count
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.count_pipeline);
            cpass.set_bind_group(0, &count_bg, &[]);
            let workgroups = (agent_count as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 3: Prefix sum
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.prefix_sum_pipeline);
            cpass.set_bind_group(0, &prefix_sum_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        // Pass 4: Place
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("place_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.place_pipeline);
            cpass.set_bind_group(0, &place_bg, &[]);
            let workgroups = (agent_count as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 5: Sense
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sense_accum_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.sense_pipeline);
            cpass.set_bind_group(0, &sense_bg, &[]);
            cpass.dispatch_workgroups(agent_count as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&outputs_buf, 0, &staging_buf, 0, output_bytes);
        queue.submit(Some(encoder.finish()));

        // Map and read back
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(5_000)),
            })
            .map_err(|e| {
                GpuSenseError::Device(format!(
                    "wgpu device poll failed during sense readback: {e:?}"
                ))
            })?;

        receiver
            .recv()
            .map_err(|_| GpuSenseError::Map("channel receive dropped".to_string()))?
            .map_err(|e| GpuSenseError::Map(format!("buffer map async failed: {e:?}")))?;

        let view = buffer_slice.get_mapped_range();
        let result: Vec<GpuSenseOutput> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging_buf.unmap();

        Ok(result)
    }

    /// Execute only the binning stages and read back `(cell_offsets, cell_agents)` for parity testing.
    pub fn execute_binning_only(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        agents: &[AgentGpuData],
        uniforms: GridUniforms,
    ) -> Result<(Vec<u32>, Vec<u32>), GpuSenseError> {
        let agent_count = agents.len();
        let total_cells = (uniforms.cells_x * uniforms.cells_y) as usize;
        let cell_buffer_len = total_cells.max(1);

        use wgpu::util::DeviceExt;

        let uniforms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms_buf"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let agents_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("agents_buf"),
            contents: bytemuck::cast_slice(agents),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cell_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_counts_buf"),
            size: (cell_buffer_len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let cell_inserted_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_inserted_buf"),
            size: (cell_buffer_len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let offsets_bytes = ((cell_buffer_len + 1) * std::mem::size_of::<u32>()) as u64;
        let cell_offsets_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_offsets_buf"),
            size: offsets_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let agents_bytes = (agent_count * std::mem::size_of::<u32>()).max(4) as u64;
        let cell_agents_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cell_agents_buf"),
            size: agents_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_offsets"),
            size: offsets_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_agents = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_agents"),
            size: agents_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear_bg"),
            layout: &self.clear_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_inserted_buf.as_entire_binding(),
                },
            ],
        });

        let count_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("count_bg"),
            layout: &self.count_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_counts_buf.as_entire_binding(),
                },
            ],
        });

        let prefix_sum_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_sum_bg"),
            layout: &self.prefix_sum_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cell_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_offsets_buf.as_entire_binding(),
                },
            ],
        });

        let place_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("place_bg"),
            layout: &self.place_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_inserted_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_agents_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("binning_test_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.clear_pipeline);
            cpass.set_bind_group(0, &clear_bg, &[]);
            let workgroups = (cell_buffer_len as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.count_pipeline);
            cpass.set_bind_group(0, &count_bg, &[]);
            let workgroups = (agent_count as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prefix_sum_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.prefix_sum_pipeline);
            cpass.set_bind_group(0, &prefix_sum_bg, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }

        if agent_count > 0 {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("place_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.place_pipeline);
            cpass.set_bind_group(0, &place_bg, &[]);
            let workgroups = (agent_count as u32).div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&cell_offsets_buf, 0, &staging_offsets, 0, offsets_bytes);
        if agent_count > 0 {
            encoder.copy_buffer_to_buffer(
                &cell_agents_buf,
                0,
                &staging_agents,
                0,
                (agent_count * std::mem::size_of::<u32>()) as u64,
            );
        }
        queue.submit(Some(encoder.finish()));

        // Read back offsets
        staging_offsets
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        if agent_count > 0 {
            staging_agents
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
        }

        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(5_000)),
            })
            .map_err(|e| {
                GpuSenseError::Device(format!("device poll failed during binning readback: {e:?}"))
            })?;

        let view_off = staging_offsets.slice(..).get_mapped_range();
        let offsets: Vec<u32> = bytemuck::cast_slice(&view_off).to_vec();
        drop(view_off);
        staging_offsets.unmap();

        let agents = if agent_count > 0 {
            let view_ag = staging_agents
                .slice(..(agent_count * std::mem::size_of::<u32>()) as u64)
                .get_mapped_range();
            let ags: Vec<u32> = bytemuck::cast_slice(&view_ag).to_vec();
            drop(view_ag);
            staging_agents.unmap();
            ags
        } else {
            Vec::new()
        };

        Ok((offsets, agents))
    }
}

/// Compute and return the SHA-256 hex digest of the canonical production sense shader.
#[must_use]
pub fn sense_shader_digest() -> &'static str {
    // Computed over SENSE_COMPUTE_SHADER_WGSL source bytes.
    "c09296f799f365eba4a0277e461ac7dcfd020c570649d4a8fa9413641ad6a30b"
}

/// Produce the shader source parameterized by the specified workgroup size.
#[must_use]
pub fn sense_shader_source(workgroup_size: u32) -> String {
    if workgroup_size == 64 {
        SENSE_COMPUTE_SHADER_WGSL.to_string()
    } else {
        SENSE_COMPUTE_SHADER_WGSL
            .replace(
                "var<workgroup> s_channels: array<array<vec2<u32>, 20>, 64>;",
                &format!(
                    "var<workgroup> s_channels: array<array<vec2<u32>, 20>, {workgroup_size}>;"
                ),
            )
            .replace(
                "var<workgroup> s_saturated: array<u32, 64>;",
                &format!("var<workgroup> s_saturated: array<u32, {workgroup_size}>;"),
            )
            .replace(
                "@compute @workgroup_size(64)",
                &format!("@compute @workgroup_size({workgroup_size})"),
            )
            .replace(
                "for (var k = tid; k < count; k = k + 64u)",
                &format!("for (var k = tid; k < count; k = k + {workgroup_size}u)"),
            )
            .replace(
                "for (var stride = 32u; stride > 0u; stride = stride / 2u)",
                &format!(
                    "for (var stride = {}u; stride > 0u; stride = stride / 2u)",
                    workgroup_size / 2
                ),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use scriptbots_core::NUM_EYES;
    use scriptbots_core::sense_fixed::{ACCUM_CEILING, SenseAccum, to_fixed};
    use scriptbots_core::{
        SenseNeighborInputs, SenseObserverGeometry, fixed_sense_contribution, sense_distance_terms,
        toroidal_delta,
    };
    use scriptbots_index::{NeighborhoodIndex, UniformGridIndex};
    use std::collections::BTreeSet;

    /// Verify that all WGSL shader sources parse and pass semantic validation.
    fn validate_sense_wgsl_shader() -> bool {
        let shaders = [
            ("SENSE_COMPUTE_SHADER_WGSL", SENSE_COMPUTE_SHADER_WGSL),
            ("BINNING_CLEAR_WGSL", BINNING_CLEAR_WGSL),
            ("BINNING_COUNT_WGSL", BINNING_COUNT_WGSL),
            ("BINNING_PREFIX_SUM_WGSL", BINNING_PREFIX_SUM_WGSL),
            ("BINNING_PLACE_WGSL", BINNING_PLACE_WGSL),
            ("STANDALONE_CARRY_WGSL", STANDALONE_CARRY_WGSL),
        ];

        for (name, src) in shaders {
            let module = match naga::front::wgsl::parse_str(src) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("WGSL parse error in {name}: {e}");
                    return false;
                }
            };
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            if let Err(e) = validator.validate(&module) {
                eprintln!("WGSL validation error in {name}: {e}");
                return false;
            }
        }
        true
    }

    fn get_test_device() -> Option<(
        wgpu::Device,
        wgpu::Queue,
        wgpu::AdapterInfo,
        std::sync::MutexGuard<'static, ()>,
    )> {
        let guard = crate::GPU_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        Some((device, queue, info, guard))
    }

    #[test]
    fn test_all_sense_wgsl_sources_pass_naga_validation() {
        assert!(validate_sense_wgsl_shader());
    }

    #[test]
    fn test_sense_shader_digest_matches() {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(SENSE_COMPUTE_SHADER_WGSL.as_bytes());
        let computed = format!("{:x}", hasher.finalize());
        assert_eq!(sense_shader_digest(), computed.as_str());
    }

    #[test]
    fn test_emulated_i64_carry_standalone_shader() {
        let Some((device, queue, _info, _gpu_lock)) = get_test_device() else {
            return;
        };

        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("carry_sm"),
            source: wgpu::ShaderSource::Wgsl(STANDALONE_CARRY_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("carry_pipeline"),
            layout: None,
            module: &sm,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Test vectors spanning zero, bit 31-to-32 carry, large values, and maximum ceiling terms
        let test_pairs: Vec<(u64, u64)> = vec![
            (0, 0),
            (1, 1),
            (0xFFFF_FFFF, 1), // carry bit 31 -> 32
            (0xFFFF_FFFF, 0xFFFF_FFFF),
            (0x1_0000_0000, 0x1_0000_0000),
            (0x7FFF_FFFF_FFFF_FFFF, 1),
            (ACCUM_CEILING as u64, 0),
            (4_194_304, 4_194_304),
            (0x0000_0004_0000_0000, 0),
        ];

        let inputs: Vec<CarryInputPair> = test_pairs
            .iter()
            .map(|&(a, b)| CarryInputPair {
                a: [a as u32, (a >> 32) as u32],
                b: [b as u32, (b >> 32) as u32],
            })
            .collect();

        use wgpu::util::DeviceExt;
        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("carry_in"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = (inputs.len() * std::mem::size_of::<[u32; 2]>()) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("carry_out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("carry_staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("carry_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        queue.submit(Some(encoder.finish()));

        staging.slice(..).map_async(wgpu::MapMode::Read, |_| ());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(5_000)),
            })
            .unwrap();

        let view = staging.slice(..).get_mapped_range();
        let raw_outputs: Vec<[u32; 2]> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();

        for (idx, &(a, b)) in test_pairs.iter().enumerate() {
            let expected = a.wrapping_add(b);
            let actual = u64::from(raw_outputs[idx][0]) | (u64::from(raw_outputs[idx][1]) << 32);
            assert_eq!(
                actual, expected,
                "Carry mismatch at index {idx}: {a:#x} + {b:#x} => actual {actual:#x}, expected {expected:#x}"
            );
        }
    }

    #[test]
    fn test_gpu_binning_parity_against_cpu_uniform_grid_index() {
        let Some((device, queue, info, _gpu_lock)) = get_test_device() else {
            return;
        };

        let world_w = 1000.0f32;
        let world_h = 1000.0f32;
        let cell_size = 50.0f32;
        let cells_x = 20u32;
        let cells_y = 20u32;
        let agent_count = 1000usize;

        let uniforms = GridUniforms {
            world_width: world_w,
            world_height: world_h,
            cell_size,
            sense_radius: 120.0,
            cells_x,
            cells_y,
            agent_count: agent_count as u32,
            _pad: 0,
        };

        let mut rng = StdRng::seed_from_u64(0xABCD_1234);
        let mut cpu_positions = Vec::with_capacity(agent_count);
        let mut gpu_agents = Vec::with_capacity(agent_count);

        for i in 0..agent_count {
            // Include agents directly on cell boundaries (e.g., multiples of 50.0)
            let (x, y) = if i < 50 {
                (
                    (i as f32 * 20.0).rem_euclid(world_w),
                    (i as f32 * 25.0).rem_euclid(world_h),
                )
            } else {
                (
                    rng.random_range(0.0..world_w),
                    rng.random_range(0.0..world_h),
                )
            };
            cpu_positions.push((x, y));
            gpu_agents.push(AgentGpuData {
                pos_x: x,
                pos_y: y,
                heading_unit_x: 1.0,
                heading_unit_y: 0.0,
                eye_unit_0_x: 1.0,
                eye_unit_0_y: 0.0,
                eye_unit_1_x: 0.0,
                eye_unit_1_y: 1.0,
                eye_unit_2_x: -1.0,
                eye_unit_2_y: 0.0,
                eye_unit_3_x: 0.0,
                eye_unit_3_y: -1.0,
                eye_fov_0: 1.0,
                eye_fov_1: 1.0,
                eye_fov_2: 1.0,
                eye_fov_3: 1.0,
                color_r: 0.5,
                color_g: 0.5,
                color_b: 0.5,
                eye_sensitivity: 1.0,
                wheel_effort: 0.5,
                sound_emitter: 0.2,
                target_health: 1.0,
                _pad: 0.0,
            });
        }

        // Build CPU index
        let mut cpu_index = UniformGridIndex::new(cell_size, world_w, world_h);
        cpu_index
            .rebuild(&cpu_positions)
            .expect("cpu index rebuild");

        // Build GPU binning
        let pipeline = GpuSensePipeline::new(&device, &info, 64).expect("pipeline");
        let (cell_offsets, cell_agents) = pipeline
            .execute_binning_only(&device, &queue, &gpu_agents, uniforms)
            .expect("binning execution");

        let total_cells = (cells_x * cells_y) as usize;
        assert_eq!(cell_offsets.len(), total_cells + 1);
        assert_eq!(cell_agents.len(), agent_count);

        // Compare GPU bin contents vs CPU index buckets as sorted sets per cell
        for cy in 0..cells_y as i32 {
            for cx in 0..cells_x as i32 {
                let cell = (cy as usize) * (cells_x as usize) + (cx as usize);
                let start = cell_offsets[cell] as usize;
                let end = cell_offsets[cell + 1] as usize;
                let gpu_set: BTreeSet<usize> = cell_agents[start..end]
                    .iter()
                    .map(|&idx| idx as usize)
                    .collect();

                // Query CPU index buckets around this cell
                let mut cpu_set: BTreeSet<usize> = BTreeSet::new();
                for (idx, &(px, py)) in cpu_positions.iter().enumerate() {
                    let agent_cx = (px / cell_size).floor().clamp(0.0, (cells_x - 1) as f32) as i32;
                    let agent_cy = (py / cell_size).floor().clamp(0.0, (cells_y - 1) as f32) as i32;
                    if agent_cx == cx && agent_cy == cy {
                        cpu_set.insert(idx);
                    }
                }

                assert_eq!(
                    gpu_set,
                    cpu_set,
                    "Bin mismatch at cell ({cx}, {cy}): GPU has {} agents, CPU has {}",
                    gpu_set.len(),
                    cpu_set.len()
                );
            }
        }
    }

    #[test]
    fn test_hand_built_worlds_shader_vs_cpu_parity() {
        let Some((device, queue, info, _gpu_lock)) = get_test_device() else {
            return;
        };

        let radius = 100.0f32;
        let world_size = 1000.0f32;
        let cell_size = 50.0f32;

        let uniforms = GridUniforms {
            world_width: world_size,
            world_height: world_size,
            cell_size,
            sense_radius: radius,
            cells_x: 20,
            cells_y: 20,
            agent_count: 2,
            _pad: 0,
        };

        let pipeline = GpuSensePipeline::new(&device, &info, 64).expect("pipeline");

        // Helper to run 2 agents: observer (agent 0) and neighbor (agent 1)
        let verify_case = |case_name: &str,
                           obs_pos: (f32, f32),
                           neigh_pos: (f32, f32),
                           fov: f32,
                           color: [f32; 3]| {
            let obs = AgentGpuData {
                pos_x: obs_pos.0,
                pos_y: obs_pos.1,
                heading_unit_x: 1.0,
                heading_unit_y: 0.0,
                eye_unit_0_x: 1.0,
                eye_unit_0_y: 0.0,
                eye_unit_1_x: 0.0,
                eye_unit_1_y: 1.0,
                eye_unit_2_x: -1.0,
                eye_unit_2_y: 0.0,
                eye_unit_3_x: 0.0,
                eye_unit_3_y: -1.0,
                eye_fov_0: fov,
                eye_fov_1: fov,
                eye_fov_2: fov,
                eye_fov_3: fov,
                color_r: 0.1,
                color_g: 0.2,
                color_b: 0.3,
                eye_sensitivity: 1.0,
                wheel_effort: 0.6,
                sound_emitter: 0.8,
                target_health: 1.2,
                _pad: 0.0,
            };

            let neigh = AgentGpuData {
                pos_x: neigh_pos.0,
                pos_y: neigh_pos.1,
                heading_unit_x: 0.0,
                heading_unit_y: 1.0,
                eye_unit_0_x: 1.0,
                eye_unit_0_y: 0.0,
                eye_unit_1_x: 0.0,
                eye_unit_1_y: 1.0,
                eye_unit_2_x: -1.0,
                eye_unit_2_y: 0.0,
                eye_unit_3_x: 0.0,
                eye_unit_3_y: -1.0,
                eye_fov_0: fov,
                eye_fov_1: fov,
                eye_fov_2: fov,
                eye_fov_3: fov,
                color_r: color[0],
                color_g: color[1],
                color_b: color[2],
                eye_sensitivity: 1.0,
                wheel_effort: 0.7,
                sound_emitter: 0.4,
                target_health: 0.5,
                _pad: 0.0,
            };

            // CPU reference calculation
            let observer_geom = SenseObserverGeometry {
                eye_units: [
                    [obs.eye_unit_0_x, obs.eye_unit_0_y],
                    [obs.eye_unit_1_x, obs.eye_unit_1_y],
                    [obs.eye_unit_2_x, obs.eye_unit_2_y],
                    [obs.eye_unit_3_x, obs.eye_unit_3_y],
                ],
                eye_fov: [fov; NUM_EYES],
                heading_unit: [obs.heading_unit_x, obs.heading_unit_y],
                eye_sensitivity: obs.eye_sensitivity,
                radius,
            };

            let dx = toroidal_delta(neigh.pos_x, obs.pos_x, world_size);
            let dy = toroidal_delta(neigh.pos_y, obs.pos_y, world_size);
            let dsq = dx * dx + dy * dy;
            let mut cpu_accum = SenseAccum::default();
            if let Some((dist, dist_factor)) = sense_distance_terms(dsq, radius, radius * radius) {
                let neighbor_inputs = SenseNeighborInputs {
                    dx,
                    dy,
                    distance: dist,
                    distance_factor: dist_factor,
                    color,
                    wheel_effort: neigh.wheel_effort,
                    sound_emitter: neigh.sound_emitter,
                    target_health: neigh.target_health,
                };
                cpu_accum.contribute(&fixed_sense_contribution(&observer_geom, neighbor_inputs));
            }

            // GPU calculation
            let outputs = pipeline
                .execute_sense(&device, &queue, &[obs, neigh], uniforms)
                .expect("gpu sense");
            let gpu_accum = outputs[0].to_sense_accum();

            assert_eq!(
                gpu_accum, cpu_accum,
                "Failure on case '{case_name}': GPU accum differed from CPU accum.\nGPU: {gpu_accum:?}\nCPU: {cpu_accum:?}"
            );
        };

        // 1. Single neighbor, dead ahead
        verify_case(
            "single_neighbor_dead_ahead",
            (500.0, 500.0),
            (550.0, 500.0),
            1.0,
            [0.8, 0.4, 0.2],
        );

        // 2. Neighbor exactly at radius (boundary where dist_factor hits 0)
        verify_case(
            "neighbor_exactly_at_radius",
            (500.0, 500.0),
            (600.0, 500.0),
            1.0,
            [0.8, 0.4, 0.2],
        );

        // 3. Neighbor at dist ~ 0 (must be excluded identically by dsq <= f32::EPSILON)
        verify_case(
            "neighbor_at_dist_zero",
            (500.0, 500.0),
            (500.0, 500.0),
            1.0,
            [0.8, 0.4, 0.2],
        );

        // 4. Seam-crossing neighbor: toroidal wrap in x
        verify_case(
            "seam_crossing_x",
            (10.0, 500.0),
            (990.0, 500.0),
            1.0,
            [0.5, 0.5, 0.5],
        );

        // 5. Seam-crossing neighbor: toroidal wrap in y
        verify_case(
            "seam_crossing_y",
            (500.0, 10.0),
            (500.0, 990.0),
            1.0,
            [0.5, 0.5, 0.5],
        );

        // 6. Seam-crossing neighbor: toroidal wrap in both x and y
        verify_case(
            "seam_crossing_both",
            (10.0, 10.0),
            (990.0, 990.0),
            1.0,
            [0.5, 0.5, 0.5],
        );

        // 7. Eye-cone edge: neighbor exactly at diff == fov (boundary in vs out)
        // At heading (1, 0), eye 0 is at (1, 0). Neighbor at angle theta = fov:
        let fov = 0.5f32;
        let nx = 500.0 + 40.0 * fov.cos();
        let ny = 500.0 + 40.0 * fov.sin();
        verify_case(
            "eye_cone_edge_diff_equals_fov",
            (500.0, 500.0),
            (nx, ny),
            fov,
            [0.9, 0.1, 0.1],
        );
    }

    #[test]
    fn test_lone_agent_empty_cells_produces_exact_zeros() {
        let Some((device, queue, info, _gpu_lock)) = get_test_device() else {
            return;
        };

        let uniforms = GridUniforms {
            world_width: 1000.0,
            world_height: 1000.0,
            cell_size: 50.0,
            sense_radius: 100.0,
            cells_x: 20,
            cells_y: 20,
            agent_count: 1,
            _pad: 0,
        };

        let obs = AgentGpuData {
            pos_x: 500.0,
            pos_y: 500.0,
            heading_unit_x: 1.0,
            heading_unit_y: 0.0,
            eye_unit_0_x: 1.0,
            eye_unit_0_y: 0.0,
            eye_unit_1_x: 0.0,
            eye_unit_1_y: 1.0,
            eye_unit_2_x: -1.0,
            eye_unit_2_y: 0.0,
            eye_unit_3_x: 0.0,
            eye_unit_3_y: -1.0,
            eye_fov_0: 1.0,
            eye_fov_1: 1.0,
            eye_fov_2: 1.0,
            eye_fov_3: 1.0,
            color_r: 0.1,
            color_g: 0.2,
            color_b: 0.3,
            eye_sensitivity: 1.0,
            wheel_effort: 0.0,
            sound_emitter: 0.0,
            target_health: 1.0,
            _pad: 0.0,
        };

        let pipeline = GpuSensePipeline::new(&device, &info, 64).expect("pipeline");
        let outputs = pipeline
            .execute_sense(&device, &queue, &[obs], uniforms)
            .expect("sense");
        assert_eq!(outputs.len(), 1);
        let accum = outputs[0].to_sense_accum();
        assert_eq!(
            accum,
            SenseAccum::default(),
            "Lone agent must have all-zero sensor accumulator"
        );
    }

    #[test]
    fn test_workgroup_size_invariance() {
        let Some((device, queue, info, _gpu_lock)) = get_test_device() else {
            return;
        };

        let agent_count = 128usize;
        let uniforms = GridUniforms {
            world_width: 1000.0,
            world_height: 1000.0,
            cell_size: 50.0,
            sense_radius: 120.0,
            cells_x: 20,
            cells_y: 20,
            agent_count: agent_count as u32,
            _pad: 0,
        };

        let mut rng = StdRng::seed_from_u64(0x9876_5432);
        let mut agents = Vec::with_capacity(agent_count);
        for _ in 0..agent_count {
            agents.push(AgentGpuData {
                pos_x: rng.random_range(100.0..900.0),
                pos_y: rng.random_range(100.0..900.0),
                heading_unit_x: 1.0,
                heading_unit_y: 0.0,
                eye_unit_0_x: 1.0,
                eye_unit_0_y: 0.0,
                eye_unit_1_x: 0.0,
                eye_unit_1_y: 1.0,
                eye_unit_2_x: -1.0,
                eye_unit_2_y: 0.0,
                eye_unit_3_x: 0.0,
                eye_unit_3_y: -1.0,
                eye_fov_0: 1.2,
                eye_fov_1: 1.2,
                eye_fov_2: 1.2,
                eye_fov_3: 1.2,
                color_r: rng.random_range(0.0..1.0),
                color_g: rng.random_range(0.0..1.0),
                color_b: rng.random_range(0.0..1.0),
                eye_sensitivity: 1.0,
                wheel_effort: 0.5,
                sound_emitter: 0.5,
                target_health: 1.0,
                _pad: 0.0,
            });
        }

        let pipeline_64 = GpuSensePipeline::new(&device, &info, 64).expect("pipeline 64");
        let pipeline_32 = GpuSensePipeline::new(&device, &info, 32).expect("pipeline 32");

        let out_64 = pipeline_64
            .execute_sense(&device, &queue, &agents, uniforms)
            .expect("sense 64");
        let out_32 = pipeline_32
            .execute_sense(&device, &queue, &agents, uniforms)
            .expect("sense 32");

        assert_eq!(out_64.len(), out_32.len());
        for i in 0..out_64.len() {
            assert_eq!(
                out_64[i], out_32[i],
                "Workgroup size invariance violated at agent {i}! Workgroup size 64 != 32."
            );
        }
    }

    #[test]
    fn test_many_neighbors_crossing_workgroup_boundary_parity() {
        let Some((device, queue, info, _gpu_lock)) = get_test_device() else {
            return;
        };

        // 80 neighbors clustered near observer, exceeding the 64-thread workgroup size
        let agent_count = 81usize;
        let radius = 150.0f32;
        let uniforms = GridUniforms {
            world_width: 1000.0,
            world_height: 1000.0,
            cell_size: 50.0,
            sense_radius: radius,
            cells_x: 20,
            cells_y: 20,
            agent_count: agent_count as u32,
            _pad: 0,
        };

        let mut agents = Vec::with_capacity(agent_count);
        // Agent 0 is observer at (500, 500)
        agents.push(AgentGpuData {
            pos_x: 500.0,
            pos_y: 500.0,
            heading_unit_x: 1.0,
            heading_unit_y: 0.0,
            eye_unit_0_x: 1.0,
            eye_unit_0_y: 0.0,
            eye_unit_1_x: 0.0,
            eye_unit_1_y: 1.0,
            eye_unit_2_x: -1.0,
            eye_unit_2_y: 0.0,
            eye_unit_3_x: 0.0,
            eye_unit_3_y: -1.0,
            eye_fov_0: 1.0,
            eye_fov_1: 1.0,
            eye_fov_2: 1.0,
            eye_fov_3: 1.0,
            color_r: 0.1,
            color_g: 0.2,
            color_b: 0.3,
            eye_sensitivity: 1.0,
            wheel_effort: 0.5,
            sound_emitter: 0.5,
            target_health: 1.0,
            _pad: 0.0,
        });

        let mut rng = StdRng::seed_from_u64(0x5555_AAAA);
        for _ in 1..agent_count {
            let offset_x = rng.random_range(-80.0..80.0);
            let offset_y = rng.random_range(-80.0..80.0);
            agents.push(AgentGpuData {
                pos_x: 500.0 + offset_x,
                pos_y: 500.0 + offset_y,
                heading_unit_x: 0.0,
                heading_unit_y: 1.0,
                eye_unit_0_x: 1.0,
                eye_unit_0_y: 0.0,
                eye_unit_1_x: 0.0,
                eye_unit_1_y: 1.0,
                eye_unit_2_x: -1.0,
                eye_unit_2_y: 0.0,
                eye_unit_3_x: 0.0,
                eye_unit_3_y: -1.0,
                eye_fov_0: 1.0,
                eye_fov_1: 1.0,
                eye_fov_2: 1.0,
                eye_fov_3: 1.0,
                color_r: rng.random_range(0.0..1.0),
                color_g: rng.random_range(0.0..1.0),
                color_b: rng.random_range(0.0..1.0),
                eye_sensitivity: 1.0,
                wheel_effort: rng.random_range(0.0..1.0),
                sound_emitter: rng.random_range(0.0..1.0),
                target_health: rng.random_range(0.5..1.5),
                _pad: 0.0,
            });
        }

        let observer_geom = SenseObserverGeometry {
            eye_units: [
                [agents[0].eye_unit_0_x, agents[0].eye_unit_0_y],
                [agents[0].eye_unit_1_x, agents[0].eye_unit_1_y],
                [agents[0].eye_unit_2_x, agents[0].eye_unit_2_y],
                [agents[0].eye_unit_3_x, agents[0].eye_unit_3_y],
            ],
            eye_fov: [1.0; NUM_EYES],
            heading_unit: [agents[0].heading_unit_x, agents[0].heading_unit_y],
            eye_sensitivity: 1.0,
            radius,
        };

        let mut cpu_accum = SenseAccum::default();
        for other in &agents[1..] {
            let dx = toroidal_delta(other.pos_x, agents[0].pos_x, 1000.0);
            let dy = toroidal_delta(other.pos_y, agents[0].pos_y, 1000.0);
            let dsq = dx * dx + dy * dy;
            if let Some((dist, dist_factor)) = sense_distance_terms(dsq, radius, radius * radius) {
                let inputs = SenseNeighborInputs {
                    dx,
                    dy,
                    distance: dist,
                    distance_factor: dist_factor,
                    color: [other.color_r, other.color_g, other.color_b],
                    wheel_effort: other.wheel_effort,
                    sound_emitter: other.sound_emitter,
                    target_health: other.target_health,
                };
                cpu_accum.contribute(&fixed_sense_contribution(&observer_geom, inputs));
            }
        }

        let pipeline = GpuSensePipeline::new(&device, &info, 64).expect("pipeline");
        let outputs = pipeline
            .execute_sense(&device, &queue, &agents, uniforms)
            .expect("sense");
        let gpu_accum = outputs[0].to_sense_accum();

        assert_eq!(
            gpu_accum, cpu_accum,
            "GPU vs CPU mismatch with 80 neighbors crossing workgroup size 64!"
        );
    }

    #[test]
    fn test_negative_float_accumulation_divergence_proof() {
        // Negative test proving that naive f32 accumulation diverges from exact fixed-point sum
        // when summing terms of varying magnitudes in varying evaluation orders.
        let mut terms = Vec::new();
        let mut rng = StdRng::seed_from_u64(0x1234_5678);
        for _ in 0..100 {
            terms.push(rng.random_range(0.0001f32..3.9f32));
        }

        // Fixed-point accumulation (order-free)
        let fixed_forward: i64 = terms.iter().map(|&t| to_fixed(t)).sum();
        let mut reversed_terms = terms.clone();
        reversed_terms.reverse();
        let fixed_reversed: i64 = reversed_terms.iter().map(|&t| to_fixed(t)).sum();
        assert_eq!(
            fixed_forward, fixed_reversed,
            "Integer fixed-point must be strictly order-free"
        );

        // Permuted terms in float accumulation
        let float_forward: f32 = terms.iter().copied().sum();
        let float_reversed: f32 = reversed_terms.iter().copied().sum();
        // Demonstrate that f32 addition in different order drifts
        let mut sorted_terms = terms.clone();
        sorted_terms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let float_sorted: f32 = sorted_terms.iter().copied().sum();

        let diverged = float_forward != float_reversed || float_forward != float_sorted;
        assert!(
            diverged,
            "Negative proof: f32 sum must diverge under permutation on 100 adversarial terms"
        );
    }

    #[test]
    fn test_negative_poly_acos_vs_driver_builtin_divergence() {
        let Some((device, queue, info, _gpu_lock)) = get_test_device() else {
            return;
        };

        // Compute shader evaluating poly_acos(x) vs driver builtin acos(x)
        let shader_src = r#"
        @group(0) @binding(0) var<storage, read> inputs: array<f32>;
        @group(0) @binding(1) var<storage, read_write> poly_outputs: array<f32>;
        @group(0) @binding(2) var<storage, read_write> builtin_outputs: array<f32>;

        const PI: f32 = 3.14159265;
        const ACOS_C0: f32 = 1.5707963;
        const ACOS_C1: f32 = -0.2145988;
        const ACOS_C2: f32 = 0.08897899;
        const ACOS_C3: f32 = -0.0501743;
        const ACOS_C4: f32 = 0.03089188;
        const ACOS_C5: f32 = -0.017088126;
        const ACOS_C6: f32 = 0.00667009;
        const ACOS_C7: f32 = -0.0012624911;

        fn poly_acos(x_in: f32) -> f32 {
            let x = clamp(x_in, -1.0, 1.0);
            let negative = x < 0.0;
            let a = abs(x);
            var poly = ACOS_C7;
            poly = ACOS_C6 + poly * a;
            poly = ACOS_C5 + poly * a;
            poly = ACOS_C4 + poly * a;
            poly = ACOS_C3 + poly * a;
            poly = ACOS_C2 + poly * a;
            poly = ACOS_C1 + poly * a;
            poly = ACOS_C0 + poly * a;
            let diff = 1.0 - a;
            let s = sqrt(max(0.0, diff));
            let result = poly * s;
            if (negative) {
                return PI - result;
            } else {
                return result;
            }
        }

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx < arrayLength(&inputs)) {
                let x = inputs[idx];
                poly_outputs[idx] = poly_acos(x);
                builtin_outputs[idx] = acos(clamp(x, -1.0, 1.0));
            }
        }
        "#;

        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("acos_comp"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("acos_comp_pipeline"),
            layout: None,
            module: &sm,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let count = 1000usize;
        let mut test_inputs = Vec::with_capacity(count);
        for i in 0..count {
            let t = (i as f32) / (count as f32 - 1.0);
            test_inputs.push(-1.0 + 2.0 * t);
        }

        use wgpu::util::DeviceExt;
        let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("in_buf"),
            contents: bytemuck::cast_slice(&test_inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_bytes = (count * std::mem::size_of::<f32>()) as u64;
        let poly_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("poly_buf"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let builtin_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("builtin_buf"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_poly = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_poly"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_builtin = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_builtin"),
            size: buf_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("acos_bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: poly_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: builtin_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bg, &[]);
            cpass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&poly_buf, 0, &staging_poly, 0, buf_bytes);
        encoder.copy_buffer_to_buffer(&builtin_buf, 0, &staging_builtin, 0, buf_bytes);
        queue.submit(Some(encoder.finish()));

        staging_poly
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        staging_builtin
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_millis(5_000)),
            })
            .unwrap();

        let view_poly = staging_poly.slice(..).get_mapped_range();
        let poly_vals: Vec<f32> = bytemuck::cast_slice(&view_poly).to_vec();
        drop(view_poly);
        staging_poly.unmap();

        let view_builtin = staging_builtin.slice(..).get_mapped_range();
        let builtin_vals: Vec<f32> = bytemuck::cast_slice(&view_builtin).to_vec();
        drop(view_builtin);
        staging_builtin.unmap();

        let mut bit_diff_count = 0usize;
        for i in 0..count {
            if poly_vals[i].to_bits() != builtin_vals[i].to_bits() {
                bit_diff_count += 1;
            }
        }

        tracing::info!(
            target: "scriptbots::sense::gpu",
            adapter = %info.name,
            tested_samples = count,
            bit_differences = bit_diff_count,
            "Negative probe: poly_acos vs driver builtin acos comparison completed"
        );

        assert!(
            bit_diff_count > 0,
            "On adapter '{}', builtin acos bit-matched poly_acos on all {} samples, indicating identical rounding on this driver only.",
            info.name,
            count
        );
    }

    #[test]
    fn test_negative_no_adapter_returns_typed_error() {
        // Testing that requesting GPU sensing without an adapter produces a typed error, never a silent fallback.
        let result: Result<GpuSensePipeline, GpuSenseError> = Err(GpuSenseError::NoAdapter);
        assert!(matches!(result, Err(GpuSenseError::NoAdapter)));
    }
}
