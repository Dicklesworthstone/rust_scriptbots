#import bevy_pbr::mesh_view_bindings::view

// Six vec4 rows: explicit 16-byte field alignment and a 96-byte array stride.
struct ParticleRow {
    position_gravity: vec4<f32>,
    velocity_drag: vec4<f32>,
    age_duration_size: vec4<f32>,
    rotation_spin_intensity_sprite: vec4<f32>,
    color: vec4<f32>,
    accent: vec4<f32>,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<storage, read> particles: array<ParticleRow>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var<uniform> parameters: vec4<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var atlas: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(3) var atlas_sampler: sampler;

struct Vertex {
    @location(0) corner_slot: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vertex(input: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0);
    out.color = vec4<f32>(0.0);
    let slot = u32(input.corner_slot.z);
    if slot >= u32(parameters.x) {
        return out;
    }
    let particle = particles[slot];
    // Only bounded, spawn-relative integer ages reach the GPU.
    let age = particle.age_duration_size.x + parameters.y;
    let duration = particle.age_duration_size.y;
    if age < 0.0 || age >= duration || duration <= 0.0 {
        return out;
    }
    let life = clamp(age / duration, 0.0, 1.0);
    let damping = 1.0 - clamp(particle.velocity_drag.w, 0.0, 1.0);
    // Closed form of the pool's damped per-tick integration. This also
    // defines fractional presentation ages without consuming another tick.
    var displacement = age;
    var gravity_displacement = age * (age + 1.0) * 0.5;
    if damping < 0.99999 {
        displacement = damping * (1.0 - pow(damping, age)) / (1.0 - damping);
        gravity_displacement = damping * (age - displacement) / (1.0 - damping);
    }
    var center = particle.position_gravity.xyz + particle.velocity_drag.xyz * displacement;
    center.y -= particle.position_gravity.w * gravity_displacement;
    // The same finite world periods apply to both simulation axes.
    center.x -= floor((center.x + parameters.z * 0.5) / parameters.z) * parameters.z;
    // Simulation y grows opposite to z: retain y=0 at the positive z edge.
    center.z -= ceil((center.z - parameters.w * 0.5) / parameters.w) * parameters.w;

    let angle = particle.rotation_spin_intensity_sprite.x
        + particle.rotation_spin_intensity_sprite.y * age;
    let cosine = cos(angle);
    let sine = sin(angle);
    let corner = input.corner_slot.xy;
    let rotated = vec2<f32>(
        corner.x * cosine - corner.y * sine,
        corner.x * sine + corner.y * cosine,
    );
    let size = mix(particle.age_duration_size.z, particle.age_duration_size.w, life);
    let world = center + size * (
        view.world_from_view[0].xyz * rotated.x
        + view.world_from_view[1].xyz * rotated.y
    );
    out.clip = view.clip_from_world * vec4<f32>(world, 1.0);

    let dimensions = vec2<f32>(textureDimensions(atlas));
    let tile_pixels = dimensions.y;
    let tile = particle.rotation_spin_intensity_sprite.w;
    let local_uv = corner * 0.5 + vec2<f32>(0.5);
    // Half-texel inset prevents neighboring sprite tiles bleeding under filtering.
    out.uv = vec2<f32>(
        tile * tile_pixels + 0.5 + local_uv.x * (tile_pixels - 1.0),
        0.5 + local_uv.y * (tile_pixels - 1.0),
    ) / dimensions;
    out.color = vec4<f32>(
        mix(particle.color.rgb, particle.accent.rgb, life),
        particle.rotation_spin_intensity_sprite.z * (1.0 - life * life),
    );
    return out;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4<f32> {
    let sprite = textureSample(atlas, atlas_sampler, input.uv);
    let alpha = sprite.a * clamp(input.color.a, 0.0, 1.0);
    if alpha < 0.005 {
        discard;
    }
    // Bevy AlphaMode::Add selects the premultiplied pipeline: zero output
    // alpha preserves the destination while premultiplied RGB adds emission.
    return vec4<f32>(input.color.rgb * sprite.rgb * alpha, 0.0);
}
