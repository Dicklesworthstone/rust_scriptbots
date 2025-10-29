use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use scriptbots_core::ScriptBotsConfig;
use scriptbots_core::WorldState;
use std::time::Duration;

fn bench_world_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("world_step");
    // Increase iteration time for more stable results and allow env overrides
    let samples: usize = std::env::var("SB_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(30);
    let warm: u64 = std::env::var("SB_BENCH_WARMUP_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(2);
    let measure: u64 = std::env::var("SB_BENCH_MEASURE_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(10);
    group.sample_size(samples);
    group.warm_up_time(Duration::from_secs(warm));
    group.measurement_time(Duration::from_secs(measure));
    // Steps per bench iteration (can override via SB_BENCH_STEPS)
    let steps: usize = std::env::var("SB_BENCH_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(64);
    let agents_list: Vec<usize> = std::env::var("SB_BENCH_AGENTS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![2000_usize, 5000, 10000]);
    for &agents in &agents_list {
        group.bench_function(format!("steps{}_agents{}_ticks", steps, agents), |b| {
            b.iter_batched(
                || {
                    let mut config = ScriptBotsConfig::default();
                    // Smaller world to stress neighbor density
                    config.world_width = 800;
                    config.world_height = 800;
                    config.food_cell_size = 20;
                    config.rng_seed = Some(0xBEEFu64);
                    config.history_capacity = 0;
                    // Ensure all validated fields are set to sane values
                    config.food_max = 2.0;
                    config.food_growth_rate = 0.02;
                    config.food_decay_rate = 0.0;
                    config.food_diffusion_rate = 0.1;
                    config.food_waste_rate = 0.0;
                    config.reproduction_attempt_interval = 1;
                    config.reproduction_attempt_chance = 0.2;
                    config.reproduction_rate_herbivore = 1.0;
                    config.reproduction_rate_carnivore = 1.0;
                    config.spike_radius = 4.0;
                    config.spike_damage = 0.0;
                    config.spike_energy_cost = 0.0;
                    config.spike_min_length = 0.0;
                    config.spike_alignment_cosine = 0.5;
                    config.spike_speed_damage_bonus = 0.0;
                    config.spike_length_damage_bonus = 0.0;
                    config.carnivore_threshold = 0.5;
                    config.history_capacity = 1;
                    config.metabolism_drain = 0.001;
                    config.movement_drain = 0.001;
                    config.metabolism_ramp_floor = 0.0;
                    config.metabolism_ramp_rate = 0.0;
                    config.temperature_discomfort_rate = 0.0;
                    config.temperature_comfort_band = 0.2;
                    config.temperature_gradient_exponent = 1.0;
                    config.temperature_discomfort_exponent = 1.0;
                    config.aging_tick_interval = 1;
                    config.aging_health_decay_rate = 0.0;
                    config.aging_health_decay_max = 0.0;
                    config.aging_energy_penalty_rate = 0.0;
                    config.carcass_distribution_radius = 0.0;
                    config.carcass_health_reward = 0.0;
                    config.carcass_reproduction_reward = 0.0;
                    config.carcass_energy_share_rate = 0.0;
                    config.carcass_indicator_scale = 0.0;
                    config.boost_multiplier = 1.2;
                    config.spike_growth_rate = 0.01;
                    config.population_spawn_count = 1;
                    config.population_crossover_chance = 0.0;
                    config.sense_radius = 20.0;
                    config.sense_max_neighbors = 16.0;
                    config.bot_radius = 2.0;
                    config.bot_speed = 1.0;
                    let mut world = WorldState::new(config).expect("world");
                    for seed in 0..agents as u32 {
                        let pos_x = (seed % 800) as f32;
                        let pos_y = ((seed * 37) % 800) as f32;
                        let data = scriptbots_core::AgentData::new(
                            scriptbots_core::Position::new(pos_x, pos_y),
                            scriptbots_core::Velocity::default(),
                            0.0,
                            1.0,
                            [0.5, 0.5, 0.5],
                            0.0,
                            false,
                            0,
                            scriptbots_core::Generation(0),
                        );
                        world.spawn_agent(data);
                    }
                    world
                },
                |mut world| {
                    for _ in 0..steps {
                        world.step();
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_world_steps);
criterion_main!(benches);
