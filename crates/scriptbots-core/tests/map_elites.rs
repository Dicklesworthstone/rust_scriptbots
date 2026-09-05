//! Comprehensive integration and unit test suite for MAP-Elites behavioral archive (bd-16g.6.1).

use scriptbots_brain::mlp::{MlpBrain, MlpBrainFamily};
use scriptbots_core::map_elites::{
    ArchiveEntry, ArchiveProvenance, Axis, BehaviorDescriptor, BehaviorSpaceV0, InsertionResult,
    MAX_ARCHIVE_CELLS, MapElitesArchive, PhenotypeFeature, QdError, QualityMetric,
    compute_novelty_score,
};
use scriptbots_core::{
    AgentData, AgentUid, BrainFamilyId, BrainGenomeEnvelope, BrainProvenance, Generation,
    ScriptBotsConfig, Tick, WorldState,
};

fn sample_genome(byte: u8) -> BrainGenomeEnvelope {
    let family_id = BrainFamilyId::new("mlp").expect("family_id");
    BrainGenomeEnvelope::new(family_id, 1, 1, vec![byte; 32], BrainProvenance::default())
        .expect("envelope")
}

#[test]
fn test_table_driven_binning_boundaries_and_clamping() {
    struct Case {
        val: f32,
        expected_bin: Result<u8, ()>,
    }

    let axis = Axis::new("test_axis", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 5).expect("axis");

    let cases = vec![
        // Exact lower bound -> bin 0
        Case {
            val: 0.0,
            expected_bin: Ok(0),
        },
        // Below domain -> clamp to bin 0
        Case {
            val: -10.0,
            expected_bin: Ok(0),
        },
        Case {
            val: -0.0001,
            expected_bin: Ok(0),
        },
        // Inside first bin [0.0, 2.0)
        Case {
            val: 1.0,
            expected_bin: Ok(0),
        },
        Case {
            val: 1.999,
            expected_bin: Ok(0),
        },
        // Inside second bin [2.0, 4.0)
        Case {
            val: 2.0,
            expected_bin: Ok(1),
        },
        Case {
            val: 3.5,
            expected_bin: Ok(1),
        },
        // Inside third bin [4.0, 6.0)
        Case {
            val: 4.0,
            expected_bin: Ok(2),
        },
        // Inside fourth bin [6.0, 8.0)
        Case {
            val: 6.0,
            expected_bin: Ok(3),
        },
        // Inside fifth bin [8.0, 10.0)
        Case {
            val: 8.0,
            expected_bin: Ok(4),
        },
        Case {
            val: 9.999,
            expected_bin: Ok(4),
        },
        // Exact upper bound -> LAST bin (4), NOT out-of-range
        Case {
            val: 10.0,
            expected_bin: Ok(4),
        },
        // Above domain -> clamp to LAST bin (4)
        Case {
            val: 10.001,
            expected_bin: Ok(4),
        },
        Case {
            val: 1000.0,
            expected_bin: Ok(4),
        },
        // Non-finite values -> typed error
        Case {
            val: f32::NAN,
            expected_bin: Err(()),
        },
        Case {
            val: f32::INFINITY,
            expected_bin: Err(()),
        },
        Case {
            val: f32::NEG_INFINITY,
            expected_bin: Err(()),
        },
    ];

    for (idx, case) in cases.into_iter().enumerate() {
        let result = axis.discretize(case.val, 0);
        match case.expected_bin {
            Ok(expected) => {
                assert_eq!(
                    result.expect("valid bin"),
                    expected,
                    "case {idx} with val {}",
                    case.val
                );
            }
            Err(()) => {
                assert!(
                    matches!(result, Err(QdError::NonFiniteValue { .. })),
                    "case {idx} with val {} must be NonFiniteValue",
                    case.val
                );
            }
        }
    }
}

#[test]
fn test_zero_width_and_inverted_domain_rejection() {
    assert!(matches!(
        Axis::new("zero", PhenotypeFeature::MeanSpeed, (1.0, 1.0), 5),
        Err(QdError::InvalidDomain { .. })
    ));

    assert!(matches!(
        Axis::new("inverted", PhenotypeFeature::MeanSpeed, (5.0, 2.0), 5),
        Err(QdError::InvalidDomain { .. })
    ));

    assert!(matches!(
        Axis::new("zero_bins", PhenotypeFeature::MeanSpeed, (0.0, 1.0), 0),
        Err(QdError::ZeroBins { .. })
    ));
}

#[test]
fn test_mixed_radix_exhaustiveness_and_invertibility() {
    // 3 axes with 3, 4, 2 bins = 24 total cells
    let space = BehaviorSpaceV0::new(
        0,
        vec![
            Axis::new("a", PhenotypeFeature::MeanSpeed, (0.0, 3.0), 3).expect("axis a"),
            Axis::new("b", PhenotypeFeature::DietTendency, (0.0, 4.0), 4).expect("axis b"),
            Axis::new("c", PhenotypeFeature::SpikeUsageRate, (0.0, 2.0), 2).expect("axis c"),
        ],
    );
    assert_eq!(space.total_cells().expect("total"), 24);

    for a in 0..3 {
        for b in 0..4 {
            for c in 0..2 {
                let desc =
                    BehaviorDescriptor::new(vec![a as f32 + 0.5, b as f32 + 0.5, c as f32 + 0.5]);
                let cell_id = space.cell_index(&desc).expect("cell_id");
                let expected_mixed = a as u64 + (b as u64 * 3) + (c as u64 * 3 * 4);
                assert_eq!(cell_id.get(), expected_mixed);

                let decoded = space.decode_cell_coords(cell_id).expect("decoded");
                assert_eq!(decoded, vec![a as u8, b as u8, c as u8]);
            }
        }
    }
}

#[test]
fn test_cell_cap_named_at_validate() {
    // 10 axes of 10 bins each = 10^10 cells, exceeding 1,000,000
    // Wait, D <= 8 is dimension limit, so 8 axes with 10 bins each = 10^8 = 100,000,000 > 1,000,000
    let axes = (0..8)
        .map(|i| {
            Axis::new(
                format!("ax_{i}"),
                PhenotypeFeature::MeanSpeed,
                (0.0, 10.0),
                10,
            )
            .expect("axis")
        })
        .collect();
    let space = BehaviorSpaceV0::new(0, axes);
    let err = space.validate().expect_err("should reject");
    match err {
        QdError::CellCapacityExceeded {
            total_cells,
            max_cells,
        } => {
            assert_eq!(total_cells, 100_000_000);
            assert_eq!(max_cells, MAX_ARCHIVE_CELLS);
        }
        other => panic!("expected CellCapacityExceeded, got {other:?}"),
    }
}

#[test]
fn test_dimension_mismatch_and_empty_space() {
    let empty_space = BehaviorSpaceV0::new(0, vec![]);
    assert!(matches!(empty_space.validate(), Err(QdError::EmptySpace)));

    let space = BehaviorSpaceV0::default();
    assert_eq!(space.axes.len(), 6);

    // 5 dims descriptor vs 6 dims space
    let desc_wrong = BehaviorDescriptor::new(vec![0.5; 5]);
    assert!(matches!(
        space.cell_index(&desc_wrong),
        Err(QdError::DimensionMismatch {
            expected: 6,
            actual: 5
        })
    ));
}

#[test]
fn test_insertion_replacement_semantics() {
    let space = BehaviorSpaceV0::new(
        0,
        vec![Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 5).expect("axis")],
    );
    let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 1_000_000)
        .expect("archive");

    let cell0_desc = BehaviorDescriptor::new(vec![1.0]);

    // 1. Insert new entry with UID 10, quality 50.0
    let res = archive
        .insert(ArchiveEntry {
            uid: AgentUid(10),
            tick_inserted: Tick(100),
            descriptor: cell0_desc.clone(),
            quality: 50.0,
            genome: sample_genome(1),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        })
        .expect("insert");
    assert_eq!(res, InsertionResult::InsertedNew);
    assert_eq!(archive.coverage_count(), 1);
    assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(10));

    // 2. Candidate with worse quality (40.0) -> Rejected
    let res = archive
        .insert(ArchiveEntry {
            uid: AgentUid(5),
            tick_inserted: Tick(105),
            descriptor: cell0_desc.clone(),
            quality: 40.0,
            genome: sample_genome(2),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        })
        .expect("insert");
    assert_eq!(res, InsertionResult::RejectedWorseOrEqual);
    assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(10));

    // 3. Candidate with equal quality (50.0) but higher UID (25 > 10) -> Rejected
    let res = archive
        .insert(ArchiveEntry {
            uid: AgentUid(25),
            tick_inserted: Tick(110),
            descriptor: cell0_desc.clone(),
            quality: 50.0,
            genome: sample_genome(3),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        })
        .expect("insert");
    assert_eq!(res, InsertionResult::RejectedWorseOrEqual);
    assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(10));

    // 4. Candidate with equal quality (50.0) and lower UID (4 < 10) -> ReplacedTieBreak
    let res = archive
        .insert(ArchiveEntry {
            uid: AgentUid(4),
            tick_inserted: Tick(115),
            descriptor: cell0_desc.clone(),
            quality: 50.0,
            genome: sample_genome(4),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        })
        .expect("insert");
    assert_eq!(
        res,
        InsertionResult::ReplacedTieBreak {
            displaced_uid: AgentUid(10),
            displaced_quality: 50.0
        }
    );
    assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(4));

    // 5. Candidate with strictly better quality (75.0) -> ReplacedBetter
    let res = archive
        .insert(ArchiveEntry {
            uid: AgentUid(99),
            tick_inserted: Tick(120),
            descriptor: cell0_desc,
            quality: 75.0,
            genome: sample_genome(5),
            provenance: ArchiveProvenance {
                run_id: "test".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        })
        .expect("insert");
    assert_eq!(
        res,
        InsertionResult::ReplacedBetter {
            displaced_uid: AgentUid(4),
            displaced_quality: 50.0
        }
    );
    assert_eq!(archive.cells.values().next().unwrap().uid, AgentUid(99));
    assert_eq!(archive.cells.values().next().unwrap().quality, 75.0);
    assert_eq!(archive.qd_score(), 75.0);
}

#[test]
fn test_byte_cap_bounds_without_silent_eviction() {
    let space = BehaviorSpaceV0::new(
        0,
        vec![Axis::new("speed", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 10).expect("axis")],
    );
    // 80 bytes capacity
    let mut archive =
        MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 80).expect("archive");

    let entry = ArchiveEntry {
        uid: AgentUid(1),
        tick_inserted: Tick(10),
        descriptor: BehaviorDescriptor::new(vec![1.0]),
        quality: 100.0,
        genome: sample_genome(1),
        provenance: ArchiveProvenance {
            run_id: "test".to_string(),
            parent_uid: None,
            generation: Generation(0),
        },
    };

    let err = archive.insert(entry).expect_err("should reject byte cap");
    match err {
        QdError::ByteCapExceeded {
            current_bytes,
            entry_bytes,
            cap_bytes,
        } => {
            assert_eq!(current_bytes, 0);
            assert!(entry_bytes > 80);
            assert_eq!(cap_bytes, 80);
        }
        other => panic!("expected ByteCapExceeded, got {other:?}"),
    }
    // No silent insertion or eviction: archive remains empty
    assert_eq!(archive.coverage_count(), 0);
}

#[test]
fn test_determinism_permutation_invariance_and_sorted_iteration() {
    let space = BehaviorSpaceV0::default();
    let mut arc_a = MapElitesArchive::new(
        space.clone(),
        QualityMetric::LifetimeIntake,
        100,
        10_000_000,
    )
    .expect("arc_a");
    let mut arc_b = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 10_000_000)
        .expect("arc_b");

    let entries = vec![
        ArchiveEntry {
            uid: AgentUid(100),
            tick_inserted: Tick(10),
            descriptor: BehaviorDescriptor::new(vec![0.8, 4.0, 0.9, 0.1, 0.2, 1.5]),
            quality: 88.0,
            genome: sample_genome(1),
            provenance: ArchiveProvenance {
                run_id: "run1".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        },
        ArchiveEntry {
            uid: AgentUid(200),
            tick_inserted: Tick(20),
            descriptor: BehaviorDescriptor::new(vec![0.2, 1.0, 0.1, 0.9, 0.8, 0.2]),
            quality: 45.0,
            genome: sample_genome(2),
            provenance: ArchiveProvenance {
                run_id: "run1".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        },
        ArchiveEntry {
            uid: AgentUid(300),
            tick_inserted: Tick(30),
            descriptor: BehaviorDescriptor::new(vec![0.5, 2.5, 0.5, 0.5, 0.5, 1.0]),
            quality: 60.0,
            genome: sample_genome(3),
            provenance: ArchiveProvenance {
                run_id: "run1".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        },
    ];

    // Insert forward order
    for e in entries.clone() {
        arc_a.insert(e).expect("insert a");
    }
    // Insert reverse order
    for e in entries.into_iter().rev() {
        arc_b.insert(e).expect("insert b");
    }

    // Both archives must have identical cell keys in strictly sorted order
    let keys_a = arc_a.cell_ids_sorted();
    let keys_b = arc_b.cell_ids_sorted();
    assert_eq!(keys_a, keys_b);
    for window in keys_a.windows(2) {
        assert!(window[0] < window[1]);
    }

    // Serialized representation must be byte-identical
    let json_a = serde_json::to_vec(&arc_a).expect("serialize a");
    let json_b = serde_json::to_vec(&arc_b).expect("serialize b");
    assert_eq!(json_a, json_b);
}

#[test]
fn test_novelty_knn_calculation() {
    let space = BehaviorSpaceV0::new(
        0,
        vec![Axis::new("x", PhenotypeFeature::MeanSpeed, (0.0, 10.0), 10).expect("axis")],
    );
    let mut archive = MapElitesArchive::new(space, QualityMetric::LifetimeIntake, 100, 1_000_000)
        .expect("archive");

    archive
        .insert(ArchiveEntry {
            uid: AgentUid(1),
            tick_inserted: Tick(0),
            descriptor: BehaviorDescriptor::new(vec![0.0]),
            quality: 10.0,
            genome: sample_genome(1),
            provenance: ArchiveProvenance {
                run_id: "r".to_string(),
                parent_uid: None,
                generation: Generation(0),
            },
        })
        .expect("insert");

    let cand = BehaviorDescriptor::new(vec![3.0]);
    let novelty = compute_novelty_score(&cand, &archive, 1);
    assert!((novelty - 3.0).abs() < 1e-5);
}

#[test]
fn test_config_validate_rejects_past_cell_cap_with_cap_named() {
    let axes = (0..8)
        .map(|i| {
            Axis::new(
                format!("ax_{i}"),
                PhenotypeFeature::MeanSpeed,
                (0.0, 10.0),
                10,
            )
            .expect("axis")
        })
        .collect();
    let space = BehaviorSpaceV0::new(0, axes);
    let config = ScriptBotsConfig {
        archive_space: space,
        ..ScriptBotsConfig::default()
    };
    let err = config
        .validate()
        .expect_err("should reject exceeding cell cap");
    let err_str = err.to_string();
    assert!(
        err_str.contains("1000000"),
        "error message must name the cell cap: {err_str}"
    );
}

#[test]
fn config_updates_preserve_real_elites_until_archive_meaning_changes() {
    let config = ScriptBotsConfig {
        world_width: 200,
        world_height: 200,
        closed: true,
        population_minimum: 0,
        population_spawn_interval: 0,
        reproduction_attempt_chance: 0.0,
        persistence_interval: 0,
        archive_enabled: true,
        archive_interval: 1,
        archive_min_lifetime_ticks: 1,
        archive_quality_metric: QualityMetric::AgeAtEvaluation,
        archive_space: BehaviorSpaceV0::new(
            0,
            vec![
                Axis::new("diet", PhenotypeFeature::DietTendency, (0.0, 1.0), 2)
                    .expect("bounded axis"),
            ],
        ),
        rng_seed: Some(0x00A2_C41E),
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("archive world");
    let key = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register production MLP");
    let agent = world
        .try_spawn_agent(AgentData::default())
        .expect("spawn agent");
    assert!(world.bind_agent_brain(agent, key).expect("bind MLP"));
    world.step().expect("collect a real elite");
    let archive = world.archive().expect("enabled archive").clone();
    assert_eq!(archive.coverage_count(), 1);
    assert!(archive.current_bytes > 0);
    assert!(world.config().archive_max_cells > archive.max_archive_cells);

    let unchanged = world.config().clone();
    let mut cadence = unchanged.clone();
    cadence.archive_interval = 2;
    let mut capacity = cadence.clone();
    capacity.archive_max_cells = archive
        .space
        .total_cells()
        .expect("actual grid cardinality");
    let mut chart_cadence = capacity.clone();
    chart_cadence.chart_flush_interval = unchanged.chart_flush_interval + 1;
    for updated in [unchanged, cadence, capacity, chart_cadence] {
        world
            .apply_config_update(updated)
            .expect("valid operational update");
        assert_eq!(
            world.archive(),
            Some(&archive),
            "preserve every elite and byte count"
        );
    }

    let mut changed = world.config().clone();
    changed.archive_quality_metric = QualityMetric::LifetimeIntake;
    world
        .apply_config_update(changed)
        .expect("new archive quality definition");
    let rebuilt = world.archive().expect("rebuilt archive");
    assert_eq!(rebuilt.quality_metric, QualityMetric::LifetimeIntake);
    assert_eq!(rebuilt.coverage_count(), 0);
    assert_eq!(rebuilt.current_bytes, 0);
}

#[test]
fn test_archive_eligibility_filter_rejects_young_agents() {
    let config = ScriptBotsConfig {
        population_minimum: 10,
        population_spawn_interval: 0,
        archive_enabled: true,
        archive_interval: 50,
        archive_min_lifetime_ticks: 100,
        rng_seed: Some(9999),
        ..ScriptBotsConfig::default()
    };
    let mut world = WorldState::new(config).expect("world");
    let key = world
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register MLP");

    for _ in 0..10 {
        let id = world.try_spawn_agent(AgentData::default()).expect("spawn");
        world.bind_agent_brain(id, key).expect("bind");
    }

    // Step to tick 50 (cadence fires, but agents are age 50 < 100)
    for _ in 1..=50 {
        world.step().expect("step");
    }

    // All agents must be rejected by eligibility filter
    let archive = world.archive().expect("archive exists");
    assert_eq!(
        archive.coverage_count(),
        0,
        "agents with age 50 < min_lifetime_ticks 100 must be rejected"
    );

    // Step to tick 100 (agents reach age 100 >= 100)
    for _ in 51..=100 {
        world.step().expect("step");
    }

    let archive = world.archive().expect("archive exists");
    assert!(
        archive.coverage_count() > 0,
        "agents with age 100 >= min_lifetime_ticks 100 must now be evaluated into archive"
    );
}

#[test]
fn test_negative_5k_tick_pinned_seed_archive_inertness() {
    let base_config = ScriptBotsConfig {
        population_minimum: 15,
        population_spawn_interval: 10,
        reproduction_meta_mutation_chance: 0.0,
        rng_seed: Some(4242_1337),
        ..ScriptBotsConfig::default()
    };

    // World A: Archive DISABLED
    let mut config_disabled = base_config.clone();
    config_disabled.archive_enabled = false;
    let mut world_a = WorldState::new(config_disabled).expect("world_a");
    let key_a = world_a
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register MLP a");
    for _ in 0..15 {
        let id = world_a
            .try_spawn_agent(AgentData::default())
            .expect("spawn a");
        world_a.bind_agent_brain(id, key_a).expect("bind a");
    }

    // World B: Archive ENABLED (cadence every 100 ticks, min lifetime 200 ticks)
    let mut config_enabled = base_config;
    config_enabled.archive_enabled = true;
    config_enabled.archive_interval = 100;
    config_enabled.archive_min_lifetime_ticks = 200;
    let mut world_b = WorldState::new(config_enabled).expect("world_b");
    let key_b = world_b
        .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
        .expect("register MLP b");
    for _ in 0..15 {
        let id = world_b
            .try_spawn_agent(AgentData::default())
            .expect("spawn b");
        world_b.bind_agent_brain(id, key_b).expect("bind b");
    }

    // Step both worlds for 5,000 ticks
    for _ in 1..=5000 {
        world_a.step().expect("step a");
        world_b.step().expect("step b");
    }

    let digest_disabled = world_a
        .characterization_digest_v0()
        .expect("digest_disabled");
    let digest_enabled = world_b
        .characterization_digest_v0()
        .expect("digest_enabled");

    assert_eq!(
        digest_disabled, digest_enabled,
        "Archive must be purely an observer: enabling the MAP-Elites archive MUST NOT change characterization_digest_v0"
    );

    // Verify the archive actually ran and accumulated elites
    let archive_b = world_b.archive().expect("archive must exist on world_b");
    assert!(
        archive_b.coverage_count() > 0,
        "archive must have evaluated and inserted elites across 5k ticks"
    );
}
