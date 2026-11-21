// Bevy 0.17's StandardMaterial lighting/post-processing path, with colors from
// the stable MeshTag row instead of a unique material uniform per creature part.
#import bevy_pbr::{
    pbr_types,
    pbr_functions,
    pbr_functions::alpha_discard,
    pbr_fragment::pbr_input_from_standard_material,
    mesh_functions::get_tag,
    decal::clustered::apply_decal_base_color,
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
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
}
#endif

#ifdef OIT_ENABLED
#import bevy_core_pipeline::oit::oit_draw
#endif
#ifdef FORWARD_DECAL
#import bevy_pbr::decal::forward::get_forward_decal_info
#endif

struct AppearanceRow {
    base: vec4<f32>,
    emissive: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<storage, read> appearance: array<AppearanceRow>;

@fragment
fn fragment(vertex_output: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var in = vertex_output;
#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif
#ifdef FORWARD_DECAL
    let forward_decal_info = get_forward_decal_info(in);
    in.world_position = forward_decal_info.world_position;
    in.uv = forward_decal_info.uv;
#endif

    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let row = appearance[get_tag(in.instance_index)];
    // Multiplication preserves the selection ring's vertex colors. The shared
    // material is white, so this is the same order as StandardMaterial tinting.
    pbr_input.material.base_color *= row.base;
    pbr_input.material.emissive = row.emissive;
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);
    pbr_input.material.base_color = apply_decal_base_color(
        in.world_position.xyz,
        in.position.xy,
        pbr_input.material.base_color
    );

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

#ifdef OIT_ENABLED
    let alpha_mode = pbr_input.material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;
    if alpha_mode != pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE {
        oit_draw(in.position, out.color);
        discard;
    }
#endif
#ifdef FORWARD_DECAL
    out.color.a = min(forward_decal_info.alpha, out.color.a);
#endif
    return out;
}
