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
    // Photos are sampled as linear RGB. Their mean is much darker than white,
    // while the vertex color already carries the canonical biome and lighting
    // tint. Lift the detail in linear space while retaining most of its local
    // contrast. Clamp at white so accessibility white remains unchanged and
    // the texture cannot brighten a semantic color beyond its base value.
    let detail = min(vec3<f32>(1.0), vec3<f32>(0.55) + 0.75 * photo_detail);
    pbr_input.material.base_color = vec4<f32>(pbr_input.material.base_color.rgb * detail,
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
