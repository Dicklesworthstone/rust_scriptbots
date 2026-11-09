// Creature classes have no textures and use only Opaque, Blend and Add.
// Keep Bevy 0.17's depth/normal/motion outputs and its shadow alpha cutoff.
// Using the shared white StandardMaterial alpha here would cast false shadows
// for nearly transparent mouth/indicator/sound parts.
#import bevy_pbr::{
    prepass_io,
    pbr_bindings::material,
    pbr_types,
    pbr_functions,
    pbr_prepass_functions,
    mesh_functions::get_tag,
}

struct AppearanceRow {
    base: vec4<f32>,
    emissive: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<storage, read> appearance: array<AppearanceRow>;

// Match Bevy 0.17's prepass cutoff. Its module-local constant is not exported
// through naga_oil, so referring to it through the module fails shader composition.
const PREMULTIPLIED_ALPHA_CUTOFF: f32 = 0.05;

fn discard_appearance(in: prepass_io::VertexOutput) {
#ifdef VISIBILITY_RANGE_DITHER
    pbr_functions::visibility_range_dither(in.position, in.visibility_range_dither);
#endif
#ifdef MAY_DISCARD
    let alpha_mode = material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;
    if (alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND ||
        alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_ADD) &&
        appearance[get_tag(in.instance_index)].base.a < PREMULTIPLIED_ALPHA_CUTOFF {
        discard;
    }
#endif
}

#ifdef PREPASS_FRAGMENT
@fragment
fn fragment(in: prepass_io::VertexOutput, @builtin(front_facing) is_front: bool) -> prepass_io::FragmentOutput {
    discard_appearance(in);
    var out: prepass_io::FragmentOutput;
#ifdef UNCLIPPED_DEPTH_ORTHO_EMULATION
    out.frag_depth = in.unclipped_depth;
#endif
#ifdef NORMAL_PREPASS
    if (material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        let double_sided = (material.flags & pbr_types::STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u;
        let normal = pbr_functions::prepare_world_normal(in.world_normal, double_sided, is_front);
        out.normal = vec4(normal * 0.5 + vec3(0.5), 1.0);
    } else {
        out.normal = vec4(in.world_normal * 0.5 + vec3(0.5), 1.0);
    }
#endif
#ifdef MOTION_VECTOR_PREPASS
    out.motion_vector = pbr_prepass_functions::calculate_motion_vector(in.world_position, in.previous_world_position);
#endif
    return out;
}
#else
@fragment
fn fragment(in: prepass_io::VertexOutput) {
    discard_appearance(in);
}
#endif
