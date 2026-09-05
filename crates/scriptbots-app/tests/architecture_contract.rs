//! Architecture references and literal scenario-input checks.
//!
//! The guide is included in crate documentation and its Rust programs execute in
//! `cargo test -p scriptbots-app --doc`. These integration tests supplement those
//! executions; source-reference checks alone do not prove a runtime capability.
//!
//! Verifies:
//! 1. `docs/ARCHITECTURE.md` explicitly distinguishes Current/Transitional (`SharedWorld`, `bd-2z0.4.9`)
//!    from Target (`HostCore`-only, `bd-k7nq`).
//! 2. Every relative path cited in `docs/ARCHITECTURE.md` exists on disk.
//! 3. The explicitly enumerated Rust symbols compile and route strings exist in source.
//! 4. The literal scenario is normalized and stepped, including both interventions.
//! 5. Mutated literal scenario input rejects unknown fields and detects missing effects.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("determine repository root from CARGO_MANIFEST_DIR")
        .to_path_buf()
}

fn load_architecture_guide() -> String {
    let guide_path = repo_root().join("docs/ARCHITECTURE.md");
    std::fs::read_to_string(&guide_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", guide_path.display()))
}

#[test]
fn architecture_guide_exists_and_is_substantial() {
    let guide = load_architecture_guide();
    assert!(
        guide.len() > 15_000,
        "docs/ARCHITECTURE.md must be substantial (found {} bytes)",
        guide.len()
    );
    assert!(
        guide.contains("# ScriptBots architecture & contribution guide"),
        "guide must contain standard header"
    );
}

#[test]
fn architecture_guide_distinguishes_current_transitional_and_target_state() {
    let guide = load_architecture_guide();

    // Must link the HostCore-only target and bd-k7nq
    assert!(
        guide.contains("Target State: Dedicated HostCore Ownership (`bd-k7nq`)")
            || guide.contains("[Target State"),
        "guide must explicitly annotate Target State"
    );
    assert!(
        guide.contains("bd-k7nq"),
        "guide must retain the historical bd-k7nq context without inferring caller migration"
    );

    // Must document current SharedWorld bridge and bd-2z0.4.9
    assert!(
        guide.contains("Current & Transitional State") || guide.contains("[Current State]"),
        "guide must explicitly annotate Current & Transitional State"
    );
    assert!(
        guide.contains("SharedWorld"),
        "guide must cite SharedWorld as current transitional mutex bridge"
    );
    assert!(
        guide.contains("bd-2z0.4.9"),
        "guide must cite closed bead bd-2z0.4.9 for eliminating discarded control receipts"
    );
}

#[test]
fn architecture_guide_cited_paths_exist_on_disk() {
    let root = repo_root();
    let guide = load_architecture_guide();

    let cited_paths = [
        "crates/scriptbots-core/src/lib.rs",
        "crates/scriptbots-core/src/checkpoint.rs",
        "crates/scriptbots-core/src/rng_domains.rs",
        "crates/scriptbots-core/src/interventions.rs",
        "crates/scriptbots-runtime/src/lib.rs",
        "crates/scriptbots-storage/src/lib.rs",
        "crates/scriptbots-app/src/lib.rs",
        "crates/scriptbots-app/src/control.rs",
        "crates/scriptbots-app/src/servers.rs",
        "crates/scriptbots-brain/src/lib.rs",
        "crates/scriptbots-analytics/src/lib.rs",
        "crates/scriptbots-web/Cargo.toml",
        "ci/check_wasm_graph.sh",
    ];

    for path_str in cited_paths {
        assert!(
            guide.contains(path_str),
            "docs/ARCHITECTURE.md must cite expected path `{path_str}`"
        );
        let path = root.join(path_str);
        assert!(
            path.exists(),
            "cited path `{path_str}` does not exist on disk at {}",
            path.display()
        );
    }
}

#[test]
fn architecture_guide_cited_symbols_exist() {
    // Compile-time and symbol presence check for symbols cited in ARCHITECTURE.md
    use scriptbots_app::{
        CharacterizationLimitationsV0, ScenarioDocumentV1, SharedWorld,
        apply_scenario_interventions, install_brains, seed_founding_population,
    };
    use scriptbots_core::{
        AgentUid, BrainAdapterIdentityV1, BrainFamilyCodec, BrainRunner, SmallRngStream,
        WorldCheckpointV1, WorldDigestV1, WorldState,
        interventions::InterventionRecord,
        rng_domains::{AgentSubstreamProtocolV1, DomainStreams},
    };
    use scriptbots_runtime::{
        ClientProjection, CommandEnvelope, CommandId, CommandValidationError, HostCommand,
        HostCore, HostEventKind, ProjectionLimits, ProjectionRequest, RenderSnapshot,
        StatusCombinationError, project_snapshot,
    };
    use scriptbots_storage::{Storage, StoragePipeline, StorageReader};

    // Instantiate / touch symbols to ensure compiler acknowledges their presence and types
    let _ = std::mem::size_of::<WorldState>();
    let _ = std::mem::size_of::<HostCore>();
    let _ = std::mem::size_of::<HostCommand>();
    let _ = std::mem::size_of::<CommandEnvelope>();
    let _ = std::mem::size_of::<CommandId>();
    let _ = std::mem::size_of::<CommandValidationError>();
    let _ = std::mem::size_of::<StatusCombinationError>();
    let _ = std::mem::size_of::<RenderSnapshot>();
    let _ = std::mem::size_of::<HostEventKind>();
    let _ = std::mem::size_of::<AgentUid>();
    let _ = std::mem::size_of::<BrainAdapterIdentityV1>();
    let _ = std::mem::size_of::<SmallRngStream>();
    let _ = std::mem::size_of::<DomainStreams>();
    let _ = std::mem::size_of::<AgentSubstreamProtocolV1>();
    let _ = std::mem::size_of::<WorldDigestV1>();
    let _ = std::mem::size_of::<WorldCheckpointV1>();
    let _ = std::mem::size_of::<Storage>();
    let _ = std::mem::size_of::<StorageReader>();
    let _ = std::mem::size_of::<StoragePipeline>();
    let _ = std::mem::size_of::<SharedWorld>();
    let _ = std::mem::size_of::<ScenarioDocumentV1>();
    let _ = std::mem::size_of::<InterventionRecord>();
    let _ = std::mem::size_of::<CharacterizationLimitationsV0>();

    // Assert function symbols and trait implementations exist
    let _ = install_brains;
    let _ = apply_scenario_interventions;
    let _ = <scriptbots_brain::MlpBrain as scriptbots_brain::Brain>::tick;
    fn _assert_traits<T: BrainRunner + BrainFamilyCodec>() {}
    let _ = seed_founding_population
        as fn(&mut WorldState, &[u64]) -> Result<(), scriptbots_app::ScenarioRunError>;
    let _ = project_snapshot
        as fn(
            &RenderSnapshot,
            &ProjectionRequest,
            ProjectionLimits,
        ) -> Result<ClientProjection, scriptbots_runtime::ProjectionError>;
}

#[test]
fn architecture_guide_cited_rest_routes_exist() {
    let servers_code =
        std::fs::read_to_string(repo_root().join("crates/scriptbots-app/src/servers.rs"))
            .expect("read servers.rs");

    let expected_routes = [
        "/api/control/status/{command_id}",
        "/api/control/pause",
        "/api/control/resume",
        "/api/control/step",
        "/api/control/speed",
        "/api/control/shutdown",
        "/api/selection",
        "/api/scenario",
        "/api/presets",
        "/api/presets/apply",
        "/mcp",
        "/health",
    ];

    let _ = load_architecture_guide();
    for route in expected_routes {
        assert!(
            servers_code.contains(route),
            "route `{route}` cited in architecture docs must exist in servers.rs"
        );
    }
}

#[test]
fn architecture_guide_negative_controls() {
    // Negative Control 1: Non-existent file path must be detected and rejected
    fn verify_path(path_str: &str, root: &Path) -> Result<(), String> {
        let p = root.join(path_str);
        if !p.exists() {
            return Err(format!("path `{path_str}` does not exist"));
        }
        Ok(())
    }
    let root = repo_root();
    assert!(
        verify_path(
            "crates/scriptbots-core/src/nonexistent_phantom_file.rs",
            &root
        )
        .is_err(),
        "negative control: non-existent file path must fail"
    );

    // Negative Control 2: Missing state annotation must be detected and rejected
    fn verify_state_annotations(doc: &str) -> Result<(), String> {
        if !doc.contains("Target State") {
            return Err("missing Target State annotation".into());
        }
        if !doc.contains("Current") {
            return Err("missing Current State annotation".into());
        }
        if !doc.contains("bd-k7nq") {
            return Err("missing bead reference bd-k7nq".into());
        }
        Ok(())
    }
    let bad_doc = "# Architecture\nAll components communicate cleanly.";
    assert!(
        verify_state_annotations(bad_doc).is_err(),
        "negative control: unannotated architecture doc must fail"
    );

    // Negative Control 3: Non-existent route must fail route verification
    let servers_code = std::fs::read_to_string(root.join("crates/scriptbots-app/src/servers.rs"))
        .expect("read servers.rs");
    assert!(
        !servers_code.contains("/api/phantom_fake_nonexistent_route"),
        "negative control: phantom route must not exist"
    );
}

fn recipe_block<'a>(guide: &'a str, name: &str, language: &str) -> Result<&'a str, String> {
    let marker = format!("<!-- recipe:{name} -->");
    let section = guide
        .split_once(&marker)
        .ok_or_else(|| format!("missing {marker}"))?
        .1;
    let fence = format!("```{language}\n");
    let source = section
        .trim_start()
        .strip_prefix(&fence)
        .ok_or_else(|| format!("{name} must be a runnable {language} block"))?;
    source
        .split_once("\n```")
        .map(|(body, _)| body)
        .ok_or_else(|| format!("unclosed {name} block"))
}

#[test]
fn literal_rust_recipes_remain_enrolled_in_documentation_tests() {
    let guide = load_architecture_guide();
    for name in ["brain", "scenario", "meteor", "frontend"] {
        let body = recipe_block(&guide, name, "rust").expect("runnable recipe");
        assert!(
            body.contains("fn main()"),
            "{name} must be a complete program"
        );
        let marker = format!("<!-- recipe:{name} -->\n```rust");
        let disabled = guide.replacen(&marker, &format!("{marker},ignore"), 1);
        assert_ne!(
            disabled, guide,
            "the negative must mutate the literal fence"
        );
        assert!(recipe_block(&disabled, name, "rust").is_err());
    }
    let library = std::fs::read_to_string(repo_root().join("crates/scriptbots-app/src/lib.rs"))
        .expect("library source");
    assert!(library.contains("#![doc = include_str!(\"../../../docs/ARCHITECTURE.md\")]"));
    // This is enrollment evidence only. The companion cargo --doc run must
    // report actual executed examples; this test cannot substitute for it.
}

fn execute_literal_scenario(text: &str) -> anyhow::Result<Vec<(u64, usize, f32)>> {
    use scriptbots_app::{
        BrainPreset, ScenarioDocumentV1, apply_scenario_interventions, install_brains,
        precedence::{ConfigLayerKind, ConfigLayerStatement, resolve_config_layers},
        seed_founding_population,
    };
    use scriptbots_core::{ScriptBotsConfig, WorldState};
    let document = ScenarioDocumentV1::parse_toml(text.as_bytes())?;
    let defaults = serde_json::to_value(ScriptBotsConfig::default())?;
    let mut current = resolve_config_layers(
        &defaults,
        &[
            ConfigLayerStatement {
                kind: ConfigLayerKind::File,
                label: document.id.clone(),
                fields: document.config.clone(),
            },
            ConfigLayerStatement {
                kind: ConfigLayerKind::Cli,
                label: "recipe-seed".into(),
                fields: serde_json::json!({"rng_seed": 42}),
            },
        ],
    )
    .merged;
    let config = serde_json::from_value(current.clone())?;
    let mut world = WorldState::new(config)?;
    let roster = install_brains(&mut world, BrainPreset::Mlp)?;
    seed_founding_population(&mut world, roster.population())?;
    let mut effects = Vec::new();
    for tick in 0..=1000 {
        anyhow::ensure!(world.tick().0 == tick, "recipe tick drift");
        let count =
            apply_scenario_interventions(&mut world, &mut current, &document.interventions, tick)?;
        if count != 0 {
            effects.push((tick, count, world.config().food_growth_rate));
        }
        world.step()?;
    }
    Ok(effects)
}

#[test]
fn literal_scenario_applies_both_effects_and_rejects_mutated_inputs() {
    let guide = load_architecture_guide();
    let text = recipe_block(&guide, "scenario-document", "toml").expect("literal scenario");
    let expected = vec![(500, 1, 0.001_f32), (1000, 1, 0.05_f32)];
    let observed = execute_literal_scenario(text).expect("execute documented scenario");
    assert_eq!(observed, expected);
    let unknown = text.replacen("food_max", "food_max_typo", 1);
    assert_ne!(unknown, text);
    assert!(execute_literal_scenario(&unknown).is_err());
    let no_recovery = text.replacen(
        "set = { food_growth_rate = 0.05 }",
        "set = { food_growth_rate = 0.001 }",
        1,
    );
    assert_ne!(no_recovery, text);
    assert_ne!(
        execute_literal_scenario(&no_recovery).expect("valid but ineffective recovery"),
        expected
    );
    println!(
        "{}",
        serde_json::json!({
            "literal_input_blake3": blake3::hash(text.as_bytes()).to_hex().to_string(),
            "observed_effects": observed, "final_tick": 1001,
            "scope": "production scenario configuration and simulation steps; no frontend claim",
        })
    );
}

#[test]
fn custom_family_is_a_library_extension_not_an_invented_cli_preset() {
    use clap::ValueEnum;
    use scriptbots_app::BrainPreset;
    assert_eq!(
        BrainPreset::from_str("mlp", false).unwrap(),
        BrainPreset::Mlp
    );
    assert!(BrainPreset::from_str("custom", false).is_err());
    let guide = load_architecture_guide();
    let brain = recipe_block(&guide, "brain", "rust").unwrap();
    assert!(brain.contains("world.register_brain_family("));
    assert!(guide.contains("`--brain custom` does not exist"));
}
