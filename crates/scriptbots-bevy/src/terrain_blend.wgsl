#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}
#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

@group(#{MATERIAL_BIND_GROUP}) @binding(100) var water_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var water_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(102) var grass_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(103) var grass_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(104) var rock_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(105) var rock_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(106) var<uniform> dry_light_strength: f32;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
#ifdef VERTEX_UVS_A
#ifdef VERTEX_UVS_B
    // UV1 carries the normalized water and grass shares; rock is the remainder.
    let weights = clamp(in.uv_b, vec2<f32>(0.0), vec2<f32>(1.0));
    let rock_weight = max(0.0, 1.0 - weights.x - weights.y);
    let total = max(weights.x + weights.y + rock_weight, 0.00001);
    let photo_detail = (textureSample(water_texture, water_sampler, in.uv).rgb * weights.x
        + textureSample(grass_texture, grass_sampler, in.uv).rgb * weights.y
        + textureSample(rock_texture, rock_sampler, in.uv).rgb * rock_weight) / total;
    // Ease the dark photo modulation without washing out its local detail.
    // The square-root curve lifts shadows more than highlights, stays within
    // the base biome tint, and leaves the white accessibility texture white.
    let detail = mix(photo_detail, sqrt(max(photo_detail, vec3<f32>(0.0))), 0.6);
    // Rock and sand share this texture group. Lift their natural-palette light
    // response while retaining hue and avoiding channel clipping; the uniform
    // disables the lift for every accessibility palette.
    let base = pbr_input.material.base_color.rgb;
    let peak = max(max(base.r, base.g), base.b);
    let max_gain = max(1.0, 1.0 / max(peak, 0.00001));
    let gain = min(1.0 + dry_light_strength * rock_weight / total, max_gain);
    pbr_input.material.base_color = vec4<f32>(base * gain * detail,
                                            pbr_input.material.base_color.a);
#endif
#endif
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif
    return out;
}
