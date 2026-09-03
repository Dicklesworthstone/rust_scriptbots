//! End-to-end integration tests for Phylogeny Event Stream and Hint Cross-Validation (bd-16g.3.3).
//!
//! Acceptance criteria verified:
//! 1. Quiet run (stable single-species world): ZERO speciation events emitted.
//! 2. Total hint reconciliation: EVERY emitted detector hint terminates in exactly one [`HintVerdict`].
//! 3. Speciation evidence reconstruction: evidence payload (observed cross-mating rate, founders,
//!    persisted samples, separation kind) is reconstructible from ancestry and birth records.
//! 4. Determinism: repeat run at same seed produces byte-identical serialized events and verdicts.

use scriptbots_core::ancestry::AncestryGraph;
use scriptbots_core::phylo::{
    DetectorHint, DetectorHintKind, HintId, HintVerdict, PhyloEngineState, PhyloEvent,
    PhyloEventParams, SeparationKind, step_phylo_events,
};
use scriptbots_core::species::{Species, SpeciesId, SpeciesTable};
use scriptbots_core::{AgentUid, BirthOrigin, BirthRecord, Generation, Position, Tick};

fn make_agent_birth(uid: u64, tick: u64, pa: Option<u64>, pb: Option<u64>) -> BirthRecord {
    BirthRecord {
        tick: Tick(tick),
        agent_uid: AgentUid(uid),
        spawn_ordinal: uid,
        birth_ordinal: Some(uid),
        origin: if pa.is_some() {
            BirthOrigin::Born
        } else {
            BirthOrigin::Seeded
        },
        parent_a: pa.map(AgentUid),
        parent_b: pb.map(AgentUid),
        brain_kind: None,
        brain_key: None,
        herbivore_tendency: 0.5,
        generation: Generation(1),
        position: Position::default(),
        is_hybrid: false,
    }
}

fn make_species(id: u64, members: &[u64], tick: u64) -> Species {
    Species {
        id: SpeciesId(id),
        name: format!("Species-{id}"),
        founders: members.iter().copied().take(1).map(AgentUid).collect(),
        members: members.iter().copied().map(AgentUid).collect(),
        centroid: vec![0.5; 6],
        spread: 0.1,
        first_tick: Tick(0),
        last_seen_tick: Tick(tick),
    }
}

/// 1. Quiet run: stable single-species world produces ZERO speciation events.
#[test]
fn test_phylo_events_quiet_run_zero_speciation() {
    let mut ancestry = AncestryGraph::new();
    for uid in 1..=30 {
        let _ = ancestry.apply_birth(&make_agent_birth(uid, 0, None, None));
    }

    let params = PhyloEventParams::default();
    let mut state = PhyloEngineState::default();

    // 5 cadence steps with a single stable species
    for step in 1..=5 {
        let tick = step * 10;
        let mut table = SpeciesTable {
            tick: Tick(tick),
            ..Default::default()
        };
        table
            .species
            .push(make_species(1, &(1..=30).collect::<Vec<_>>(), tick));

        // Panmictic births within species 1
        let mut births = Vec::new();
        for b in 0..5 {
            let child_uid = 100 + step * 10 + b;
            let birth = make_agent_birth(child_uid, tick, Some(1), Some(2));
            let _ = ancestry.apply_birth(&birth);
            births.push(birth);
        }

        let output = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);

        let speciation_count = output
            .events
            .iter()
            .filter(|(_, e)| matches!(e, PhyloEvent::Speciation { .. }))
            .count();
        assert_eq!(
            speciation_count, 0,
            "quiet single-species world must emit zero speciation events (step {step})"
        );
    }
}

/// 2. Total hint reconciliation: every detector hint receives exactly one verdict.
#[test]
fn test_phylo_events_total_hint_reconciliation() {
    let mut ancestry = AncestryGraph::new();
    for uid in 1..=20 {
        let _ = ancestry.apply_birth(&make_agent_birth(uid, 0, None, None));
    }

    let params = PhyloEventParams {
        persistence_samples: 2,
        ..Default::default()
    };
    let mut state = PhyloEngineState::default();

    // Step 1: initial state
    let mut table1 = SpeciesTable {
        tick: Tick(10),
        ..Default::default()
    };
    table1
        .species
        .push(make_species(1, &(1..=10).collect::<Vec<_>>(), 10));
    table1
        .species
        .push(make_species(2, &(11..=20).collect::<Vec<_>>(), 10));

    // Two hints submitted: one bimodality hint on species 1, one change-point hint on non-existent radiation
    let hints = vec![
        DetectorHint {
            id: HintId(101),
            tick: Tick(10),
            kind: DetectorHintKind::Bimodality,
            score: 0.92,
            metric: "phenotype_bimodality".to_string(),
            target_species: Some(SpeciesId(1)),
        },
        DetectorHint {
            id: HintId(102),
            tick: Tick(10),
            kind: DetectorHintKind::ChangePoint,
            score: 1.10,
            metric: "population_cusum".to_string(),
            target_species: Some(SpeciesId(1)),
        },
    ];

    let mut births = Vec::new();
    for b in 0..6 {
        let birth = make_agent_birth(200 + b, 10, Some(1), Some(2));
        let _ = ancestry.apply_birth(&birth);
        births.push(birth);
    }

    let output1 = step_phylo_events(&table1, &ancestry, &births, &hints, &params, &mut state);

    // Every hint must receive a verdict!
    assert_eq!(
        output1.verdicts.len(),
        hints.len(),
        "every detector hint must be reconciled with a typed verdict"
    );

    // Hint 102 (change-point without radiation) must be rejected with evidence
    let h102_verdict = output1.verdicts.iter().find(|v| match v {
        HintVerdict::Rejected { evidence, .. } => evidence.hint_id == HintId(102),
        HintVerdict::Confirmed(_) => false,
    });
    assert!(
        h102_verdict.is_some(),
        "unconfirmed change point must be rejected"
    );

    // Step 2: persists -> speciation confirmed for hint 101!
    let mut table2 = SpeciesTable {
        tick: Tick(20),
        ..Default::default()
    };
    table2
        .species
        .push(make_species(1, &(1..=10).collect::<Vec<_>>(), 20));
    table2
        .species
        .push(make_species(2, &(11..=20).collect::<Vec<_>>(), 20));

    let output2 = step_phylo_events(&table2, &ancestry, &births, &[], &params, &mut state);
    assert_eq!(
        output2.events.len(),
        1,
        "speciation confirmed at persistence K=2"
    );
    assert_eq!(output2.verdicts.len(), 1);
    assert_eq!(
        output2.verdicts[0],
        HintVerdict::Confirmed(output2.events[0].0)
    );
}

/// 3. Speciation evidence reconstruction: evidence payload is fully verifiable
///    against ancestry and birth records.
#[test]
fn test_phylo_events_evidence_reconstruction() {
    let mut ancestry = AncestryGraph::new();
    for uid in 1..=20 {
        let _ = ancestry.apply_birth(&make_agent_birth(uid, 0, None, None));
    }

    let mut births = Vec::new();
    // 8 within species 1, 8 within species 2, 0 cross
    for b in 0..8 {
        let birth_a = make_agent_birth(300 + b, 10, Some(1), Some(2));
        let birth_b = make_agent_birth(400 + b, 10, Some(11), Some(12));
        let _ = ancestry.apply_birth(&birth_a);
        let _ = ancestry.apply_birth(&birth_b);
        births.push(birth_a);
        births.push(birth_b);
    }

    #[allow(clippy::cast_precision_loss)]
    let params = PhyloEventParams {
        persistence_samples: 1,
        ..Default::default()
    };
    let mut state = PhyloEngineState::default();

    let mut table = SpeciesTable {
        tick: Tick(10),
        ..Default::default()
    };
    table
        .species
        .push(make_species(1, &(1..=10).collect::<Vec<_>>(), 10));
    table
        .species
        .push(make_species(2, &(11..=20).collect::<Vec<_>>(), 10));

    let hint = DetectorHint {
        id: HintId(777),
        tick: Tick(10),
        kind: DetectorHintKind::Bimodality,
        score: 0.98,
        metric: "phenotype_bimodality".to_string(),
        target_species: Some(SpeciesId(1)),
    };

    let output = step_phylo_events(&table, &ancestry, &births, &[hint], &params, &mut state);
    assert_eq!(output.events.len(), 1);

    let (_, event) = &output.events[0];
    if let PhyloEvent::Speciation {
        parent,
        children,
        founders,
        separation,
        cross_mating_rate,
        persisted_samples,
        hint: matched_hint,
        tick,
    } = event
    {
        // Reconstruct and verify directly from raw inputs:
        assert_eq!(*parent, SpeciesId(1));
        assert_eq!(*children, [SpeciesId(1), SpeciesId(2)]);
        assert_eq!(*founders, vec![AgentUid(11)]);
        assert_eq!(*separation, SeparationKind::Phenotypic);
        assert_eq!(*persisted_samples, 1);
        assert_eq!(*matched_hint, Some(HintId(777)));
        assert_eq!(*tick, Tick(10));

        // Reconstruct observed cross mating rate from birth records:
        let mut within = 0;
        let mut cross = 0;
        for b in &births {
            if let (Some(pa), Some(pb)) = (b.parent_a, b.parent_b) {
                let in_1 = (1..=10).contains(&pa.0) && (1..=10).contains(&pb.0);
                let in_2 = (11..=20).contains(&pa.0) && (11..=20).contains(&pb.0);
                if in_1 || in_2 {
                    within += 1;
                } else {
                    cross += 1;
                }
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let reconstructed_rate = cross as f32 / (within + cross) as f32;
        assert!((*cross_mating_rate - reconstructed_rate).abs() < f32::EPSILON);
    } else {
        panic!("expected Speciation event");
    }
}

/// 4. Determinism: identical seeded inputs produce byte-identical serialized outputs.
#[test]
fn test_phylo_events_determinism_byte_identical() {
    let run_simulation = || {
        let mut ancestry = AncestryGraph::new();
        for uid in 1..=30 {
            let _ = ancestry.apply_birth(&make_agent_birth(uid, 0, None, None));
        }

        let mut births = Vec::new();
        for b in 0..10 {
            let birth = make_agent_birth(500 + b, 10, Some(1), Some(2));
            let _ = ancestry.apply_birth(&birth);
            births.push(birth);
        }

        let params = PhyloEventParams::default();
        let mut state = PhyloEngineState::default();

        let mut table = SpeciesTable {
            tick: Tick(10),
            ..Default::default()
        };
        table
            .species
            .push(make_species(1, &(1..=15).collect::<Vec<_>>(), 10));
        table
            .species
            .push(make_species(2, &(16..=30).collect::<Vec<_>>(), 10));

        let hint = DetectorHint {
            id: HintId(888),
            tick: Tick(10),
            kind: DetectorHintKind::Bimodality,
            score: 0.89,
            metric: "phenotype_bimodality".to_string(),
            target_species: Some(SpeciesId(1)),
        };

        step_phylo_events(&table, &ancestry, &births, &[hint], &params, &mut state)
    };

    let out_a = run_simulation();
    let out_b = run_simulation();

    let bytes_a = serde_json::to_vec(&out_a).expect("serialize A");
    let bytes_b = serde_json::to_vec(&out_b).expect("serialize B");

    assert_eq!(
        bytes_a, bytes_b,
        "repeat run must produce byte-identical serialized output"
    );
}
