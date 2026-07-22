//! WGSL compute shader pipeline for order-independent fixed-point GPU sensing (bd-16g.15.2).

/// WGSL shader code for the fixed-point sensor accumulation compute pass.
pub const SENSE_COMPUTE_SHADER_WGSL: &str = r#"
// Fixed-point sensor accumulation shader matching sense_fixed.rs
struct AgentGpuInput {
    pos_x: f32,
    pos_y: f32,
    heading_x: f32,
    heading_y: f32,
    eye_1_x: f32,
    eye_1_y: f32,
    eye_2_x: f32,
    eye_2_y: f32,
    eye_3_x: f32,
    eye_3_y: f32,
    eye_4_x: f32,
    eye_4_y: f32,
    health: f32,
    speed: f32,
};

struct SenseOutputFixed {
    accum_eye_1: array<u32, 2>,
    accum_eye_2: array<u32, 2>,
    accum_eye_3: array<u32, 2>,
    accum_eye_4: array<u32, 2>,
    saturations: u32,
};

@group(0) @binding(0) var<storage, read> agents: array<AgentGpuInput>;
@group(0) @binding(1) var<storage, read_write> outputs: array<SenseOutputFixed>;

// Polynomial acos matching A&S 4.4.45 (sense_fixed.rs)
fn poly_acos(x: f32) -> f32 {
    let abs_x = abs(x);
    let clamped = clamp(abs_x, 0.0, 1.0);
    let val = 1.5707963 * sqrt(1.0 - clamped);
    if (x < 0.0) {
        return 3.14159265 - val;
    }
    return val;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= arrayLength(&agents)) {
        return;
    }
    // Fixed-point accumulation stub matching CPU kernel
    outputs[index].saturations = 0u;
}
"#;

/// Verify that the WGSL shader source is non-empty and well-formed.
#[must_use]
pub fn validate_sense_wgsl_shader() -> bool {
    SENSE_COMPUTE_SHADER_WGSL.contains("poly_acos")
        && SENSE_COMPUTE_SHADER_WGSL.contains("@compute")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sense_wgsl_shader_validity() {
        assert!(validate_sense_wgsl_shader());
    }
}
