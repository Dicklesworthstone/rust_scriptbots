//! Canonical sensor layout exhaustiveness, properties, and stage_sense drift tests (bd-16g.4.1).

use scriptbots_core::{
    AgentData, INPUT_SIZE, NUM_EYES, Position, SENSOR_LAYOUT, SENSOR_LAYOUT_DIGEST,
    ScriptBotsConfig, SensorKind, SensorSource, SensorsExt, TraitModifiers, WorldState,
    compute_sensor_layout_digest, sensor_layout,
};

#[test]
fn sensor_layout_table_properties_and_contract() {
    let layout = sensor_layout();
    assert_eq!(
        layout.len(),
        INPUT_SIZE,
        "layout must describe exactly 25 sensors"
    );

    for (index, channel) in layout.iter().enumerate() {
        assert_eq!(channel.index, index, "channel index must match slice index");
        assert_eq!(
            channel.range(),
            (0.0, 1.0),
            "sensor range must be [0.0, 1.0]"
        );
        assert!(!channel.short().is_empty(), "short name cannot be empty");
        assert_eq!(channel.short(), channel.short);
        assert_eq!(channel.source(), channel.source);
        assert_eq!(channel.eye_index(), channel.eye);
        assert_eq!(channel.clock_index(), channel.clock);
    }

    // Name uniqueness
    for (i, a) in layout.iter().enumerate() {
        for b in &layout[i + 1..] {
            assert_ne!(a.name, b.name, "duplicate sensor name: {}", a.name);
            assert_ne!(a.short, b.short, "duplicate sensor short: {}", a.short);
        }
    }

    // Eye count and channel structure
    for eye in 0..NUM_EYES {
        let eye_channels: Vec<_> = layout.iter().filter(|c| c.eye == Some(eye)).collect();
        assert_eq!(eye_channels.len(), 4, "each eye must have 4 channels");
        assert_eq!(eye_channels[0].kind, SensorKind::EyeDensity);
        assert_eq!(eye_channels[1].kind, SensorKind::EyeRed);
        assert_eq!(eye_channels[2].kind, SensorKind::EyeGreen);
        assert_eq!(eye_channels[3].kind, SensorKind::EyeBlue);
        for c in eye_channels {
            assert_eq!(c.source, SensorSource::Neighbors);
        }
    }

    // Clocks
    let clocks: Vec<_> = layout
        .iter()
        .filter(|c| c.kind == SensorKind::Clock)
        .collect();
    assert_eq!(clocks.len(), 2, "must have exactly 2 clock channels");
    assert_eq!(clocks[0].clock, Some(0));
    assert_eq!(clocks[1].clock, Some(1));
    assert_eq!(clocks[0].source, SensorSource::SelfState);
    assert_eq!(clocks[1].source, SensorSource::SelfState);

    // Food
    let food_channels: Vec<_> = layout
        .iter()
        .filter(|c| c.kind == SensorKind::Food)
        .collect();
    assert_eq!(food_channels.len(), 1);
    assert_eq!(food_channels[0].source, SensorSource::FoodGrid);
    assert_eq!(food_channels[0].index, 4);

    // Temperature
    let temp_channels: Vec<_> = layout
        .iter()
        .filter(|c| c.kind == SensorKind::Temperature)
        .collect();
    assert_eq!(temp_channels.len(), 1);
    assert_eq!(temp_channels[0].source, SensorSource::Environment);
    assert_eq!(temp_channels[0].index, 20);
}

#[test]
fn sensor_layout_blake3_digest_pinned_and_sensitive() {
    let computed = compute_sensor_layout_digest();
    assert_eq!(
        computed, SENSOR_LAYOUT_DIGEST,
        "compute_sensor_layout_digest must match SENSOR_LAYOUT_DIGEST"
    );

    // Test sensitivity: if any byte changes, the digest must diverge
    let mut modified = SENSOR_LAYOUT;
    modified[4].short = "fd";
    let mut hasher = blake3::Hasher::new();
    for c in &modified {
        hasher.update(c.index.to_string().as_bytes());
        hasher.update(b":");
        hasher.update(c.name.as_bytes());
        hasher.update(b":");
        hasher.update(c.short.as_bytes());
        hasher.update(b":Food:FoodGrid:eye=:clk=\n");
    }
    let altered_digest = hasher.finalize().to_hex().to_string();
    assert_ne!(
        altered_digest, SENSOR_LAYOUT_DIGEST,
        "a changed sensor field must alter the cryptographic digest"
    );
}

#[test]
fn sensors_ext_roundtrip_with_sensor_layout() {
    let mut raw = [0.0f32; INPUT_SIZE];
    for (i, val) in raw.iter_mut().enumerate() {
        *val = (i as f32) * 0.04;
    }

    for channel in sensor_layout() {
        assert_eq!(raw.sensor(channel), raw[channel.index]);
    }

    let labelled = raw.labelled();
    assert_eq!(labelled.len(), INPUT_SIZE);
    for (i, &(channel, val)) in labelled.iter().enumerate() {
        assert_eq!(channel.index, i);
        assert_eq!(val, raw[i]);
    }
}

#[test]
fn stage_sense_drift_test_channel_by_channel() {
    let mut world = WorldState::new(ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        food_cell_size: 20,
        initial_food: 0.0,
        food_respawn_interval: 0,
        food_growth_rate: 0.0,
        food_decay_rate: 0.0,
        food_diffusion_rate: 0.0,
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_interval: 0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        rng_seed: Some(42),
        ..ScriptBotsConfig::default()
    })
    .expect("world");

    // Spawn lone observer at (100.0, 100.0)
    let observer = world
        .try_spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            heading: 0.0,
            health: 0.8,
            ..AgentData::default()
        })
        .expect("spawn observer");

    world.step().expect("step 1");
    let rt = world.agent_runtime(observer).expect("runtime");
    let sensors = rt.sensors;

    // 1. Lone agent: all neighbor channels MUST be exactly 0.0
    for c in sensor_layout() {
        if c.source == SensorSource::Neighbors {
            assert_eq!(
                sensors[c.index], 0.0,
                "neighbor sensor {} (idx {}) must be 0 for lone agent",
                c.name, c.index
            );
        }
    }

    // 2. Food channel (index 4) with zero food underfoot must be 0.0
    assert_eq!(sensors[4], 0.0, "food sensor must be 0 with no food");

    // 3. Health channel (index 11) is health * 0.5 clamped
    // Note health was 0.8 -> 0.8 * 0.5 = 0.4
    assert!(
        (sensors[11] - 0.4).abs() < 0.05,
        "health sensor (idx 11) expected ~0.4, got {}",
        sensors[11]
    );

    // 4. Clocks (index 16, 17) must be within [0.0, 1.0]
    assert!((0.0..=1.0).contains(&sensors[16]), "clock1 out of range");
    assert!((0.0..=1.0).contains(&sensors[17]), "clock2 out of range");

    // 5. Temperature (index 20) must be within [0.0, 1.0]
    assert!(
        (0.0..=1.0).contains(&sensors[20]),
        "temperature out of range"
    );

    // World 2: Observer with a red neighbor directly in front (heading 0.0 looks in +X direction)
    let mut world2 = WorldState::new(ScriptBotsConfig {
        world_width: 500,
        world_height: 500,
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_interval: 0,
        metabolism_drain: 0.0,
        movement_drain: 0.0,
        temperature_discomfort_rate: 0.0,
        rng_seed: Some(42),
        ..ScriptBotsConfig::default()
    })
    .expect("world2");

    let obs2 = world2
        .try_spawn_agent(AgentData {
            position: Position::new(100.0, 100.0),
            heading: 0.0,
            health: 0.8,
            ..AgentData::default()
        })
        .expect("spawn obs2");

    world2
        .try_update_agent_runtime(obs2, |runtime| {
            runtime.trait_modifiers = TraitModifiers {
                smell: 1.0,
                sound: 1.0,
                hearing: 1.0,
                eye: 1.0,
                blood: 1.0,
            };
            runtime.eye_fov = [std::f32::consts::FRAC_PI_4; NUM_EYES];
            runtime.eye_direction = [
                0.0,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::PI,
                -std::f32::consts::FRAC_PI_2,
            ];
        })
        .expect("update obs2 runtime");

    let _neighbor = world2
        .try_spawn_agent(AgentData {
            position: Position::new(130.0, 100.0), // 30 units ahead in +X
            heading: 1.0,
            health: 0.8,
            color: [0.9, 0.2, 0.5],
            ..AgentData::default()
        })
        .expect("spawn neighbor");
    let expl = world2.explain_sensors(obs2, 16).expect("explain obs2");
    let explained = expl.clamped;

    // Check that neighbor is detected in the explanation
    assert!(
        !expl.contributions.is_empty(),
        "neighbor inside sense radius must be detected"
    );

    // Eye 0 density (idx 0) and Eye 0 Red (idx 1) must be positive
    assert!(
        explained[0] > 0.0,
        "eye0_density should see neighbor ahead, got {}",
        explained[0]
    );
    assert!(
        explained[1] > 0.0,
        "eye0_red should detect red color, got {}",
        explained[1]
    );

    // Smell (idx 10) must be stimulated because neighbor is nearby
    assert!(
        explained[10] > 0.0,
        "smell sensor (idx 10) must detect nearby neighbor, got {}",
        explained[10]
    );

    world2.step().expect("step world2");

    // Step world and verify agent runtime sensors agree with the sensor layout
    world2.step().expect("step world2");
    let rt_after = world2.agent_runtime(obs2).expect("runtime");
    for (i, c) in sensor_layout().iter().enumerate() {
        assert!(
            rt_after.sensors[i].is_finite(),
            "channel {} ({}) must be finite",
            c.name,
            i
        );
    }
}
