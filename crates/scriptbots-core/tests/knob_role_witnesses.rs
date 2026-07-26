//! Behavioural witnesses for the public configuration knob surface (bd-dorx).
//!
//! `ControlHandle::list_knobs` publishes every serialized `ScriptBotsConfig` leaf as a REST/MCP
//! knob, and `apply_patch` accepts every declared path. A field added to the config therefore
//! becomes a public scientific control with no proof that any model transition reads it. Hashing
//! the config cannot supply that proof: a dead field still changes the config lane by
//! construction, which is exactly the ghost-control failure mode recorded on bd-yw1j.
//!
//! This file owns the enumeration half of the fix. It derives the knob surface the same way the
//! control plane does -- serialize the config, then treat every non-object node as a leaf -- so
//! the list here cannot drift from the list callers actually see, and cannot be invalidated by a
//! `#[serde(rename)]` that source-level transcription would miss.

use scriptbots_core::knob_roles::KNOB_ROLES;
use scriptbots_core::{AgentData, Position, ScriptBotsConfig, WorldDigestV1, WorldState};
use serde_json::Value;

/// Flatten a serialized config into dotted leaf paths.
///
/// Mirrors `flatten_value` in `scriptbots-app`: recurse into objects only, and treat every other
/// node -- including arrays and nulls -- as a single leaf. Anything else would report a surface
/// the control plane does not actually expose.
fn flatten_paths(prefix: &mut String, value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            let base = prefix.len();
            for (key, nested) in map {
                if base != 0 {
                    prefix.push('.');
                }
                prefix.push_str(key);
                flatten_paths(prefix, nested, out);
                prefix.truncate(base);
            }
        }
        _ => out.push(prefix.clone()),
    }
}

/// Every knob path the control plane publishes for a default config.
fn default_knob_paths() -> Vec<String> {
    let value = serde_json::to_value(ScriptBotsConfig::default())
        .expect("the public config must serialize; the control plane depends on it");
    let mut paths = Vec::new();
    flatten_paths(&mut String::new(), &value, &mut paths);
    paths.sort();
    paths
}

/// Enumerate the knob surface so the classification registry can be built against fact.
///
/// Deliberately not an assertion on a hardcoded count: the point is to print the authoritative
/// list. A count assertion here would be the same class of unfounded claim the bead is about.
#[test]
fn bd_dorx_enumerate_the_published_knob_surface() {
    let paths = default_knob_paths();
    println!("PUBLISHED_KNOB_COUNT={}", paths.len());
    for path in &paths {
        println!("KNOB\t{path}");
    }
    assert!(
        !paths.is_empty(),
        "the config must publish at least one knob or list_knobs is meaningless"
    );
}

/// The flattener must agree with the control plane on what counts as one leaf.
///
/// Nested objects expand; arrays and scalars do not. If this ever changes, the registry's notion
/// of "every knob" silently stops matching the surface callers can patch.
#[test]
fn bd_dorx_flattening_expands_objects_and_stops_at_every_other_node() {
    let value = serde_json::json!({
        "scalar": 1,
        "nested": { "inner": true, "deeper": { "leaf": "x" } },
        "array": [1, 2, 3],
        "null": null,
    });
    let mut paths = Vec::new();
    flatten_paths(&mut String::new(), &value, &mut paths);
    paths.sort();

    assert_eq!(
        paths,
        vec![
            "array".to_owned(),
            "nested.deeper.leaf".to_owned(),
            "nested.inner".to_owned(),
            "null".to_owned(),
            "scalar".to_owned(),
        ],
        "an array is one knob, not one knob per element, and a null is still a knob"
    );
}

// ---------------------------------------------------------------------------
// bd-dorx acceptance item 2: behavioural counterfactual witnesses.
//
// The bead is explicit that neither a source grep nor `WorldDigest.config` sensitivity counts as
// proof a knob is consumed, and it is right: a dead field changes the config lane BY
// CONSTRUCTION, so any config-derived evidence is satisfied by a field nobody reads. The only
// thing that distinguishes a live knob from a ghost is running two worlds that differ in exactly
// that knob and observing MATERIAL, NON-CONFIG state diverge.
//
// `WorldDigestV1` already separates its lanes, which is what makes this checkable rather than
// hand-wavy: `config` is excluded here, and so is `overall` (it folds the config lane in, so it
// would happily "prove" a ghost). `rng` is excluded too, deliberately and more strictly than
// necessary -- a knob that only changes HOW MANY random numbers get drawn, without changing any
// outcome, should not earn a passing witness.
// ---------------------------------------------------------------------------

/// The lanes a witness is allowed to count as evidence.
///
/// Everything here is material simulation state. `config`, `overall` and `rng` are deliberately
/// absent; see the module comment above.
fn material_lanes(digest: &WorldDigestV1) -> Vec<(&'static str, String)> {
    vec![
        ("agents", digest.agents.clone()),
        ("brains", digest.brains.clone()),
        ("food", digest.food.clone()),
        ("terrain", digest.terrain.clone()),
        ("hydrology", format!("{:?}", digest.hydrology)),
        ("counters", digest.counters.clone()),
        ("effects", digest.effects.clone()),
        ("derived_transition", digest.derived_transition.clone()),
        ("origins", digest.origins.clone()),
    ]
}

/// Deterministic world for a witness run: fixed seed, fixed agent layout, fixed tick budget.
///
/// Agents are seeded explicitly rather than left to population dynamics, because a witness must
/// differ from its baseline ONLY in the knob under test.
///
/// The error reports which STAGE failed, not merely that something did.
///
/// The first version collapsed four distinct failures -- config rejected, spawn refused, step
/// errored, digest unavailable -- into one `None` and reported them all as "cannot build a world".
/// That sent me looking at world construction when the real cause was a config validator
/// (`world_width must be divisible by food_cell_size`). A witness harness whose own diagnostics
/// mislead is not much better than no harness.
fn run_material(
    config: ScriptBotsConfig,
    ticks: u64,
) -> Result<Vec<(&'static str, String)>, String> {
    let mut world = WorldState::new(config).map_err(|e| format!("config rejected: {e:?}"))?;
    for index in 0..6_u32 {
        let agent = world
            .try_spawn_agent(AgentData {
                position: Position::new(30.0 + f32::from(index as u16) * 9.0, 45.0),
                ..AgentData::default()
            })
            .map_err(|e| format!("spawn refused: {e:?}"))?;
        world
            .try_update_agent_runtime(agent, |runtime| {
                runtime.energy = 1.2;
            })
            .map_err(|e| format!("runtime update refused: {e:?}"))?;
    }
    for tick in 0..ticks {
        world
            .step()
            .map_err(|e| format!("step {tick} errored: {e:?}"))?;
    }
    let digest = world
        .world_digest_v1()
        .map_err(|e| format!("digest unavailable: {e:?}"))?;
    Ok(material_lanes(&digest))
}

/// Apply a dotted-path override to the serialized config, exactly as `apply_patch` addresses it.
///
/// Going through JSON rather than the struct is the point: it witnesses the knob at the SAME
/// address the control plane publishes, so a witness cannot accidentally prove that some
/// adjacent Rust field is live while the published path is a ghost.
fn perturbed_config(path: &str, value: Value) -> Option<ScriptBotsConfig> {
    let mut root = serde_json::to_value(ScriptBotsConfig::default()).ok()?;
    let mut cursor = &mut root;
    let mut segments = path.split('.').peekable();
    while let Some(segment) = segments.next() {
        let map = cursor.as_object_mut()?;
        if segments.peek().is_none() {
            map.insert(segment.to_owned(), value);
            return serde_json::from_value(root).ok();
        }
        cursor = map.get_mut(segment)?;
    }
    None
}

/// One counterfactual: the published path, the value to set, and how long to run.
struct Witness {
    path: &'static str,
    value: fn() -> Value,
    ticks: u64,
}

/// Witnesses for scientific knobs whose consumption is proven by a two-world differential.
///
/// Each entry perturbs exactly one published path away from its default and requires at least one
/// material lane to move. Values stay inside the config's own validation bounds; a witness that
/// cannot build a world is a failure, not a skip.
static WITNESSES: &[Witness] = &[
    Witness {
        path: "rng_seed",
        value: || Value::from(987_654_u64),
        ticks: 4,
    },
    Witness {
        path: "world_width",
        value: || Value::from(6_500),
        ticks: 2,
    },
    Witness {
        path: "world_height",
        value: || Value::from(3_500),
        ticks: 2,
    },
    Witness {
        path: "food_cell_size",
        value: || Value::from(25),
        ticks: 2,
    },
    Witness {
        path: "food_growth_rate",
        value: || Value::from(0.4),
        ticks: 6,
    },
    Witness {
        path: "food_decay_rate",
        value: || Value::from(0.25),
        ticks: 6,
    },
    Witness {
        path: "food_intake_rate",
        value: || Value::from(0.5),
        ticks: 6,
    },
    Witness {
        path: "food_max",
        value: || Value::from(4.0),
        ticks: 6,
    },
    Witness {
        path: "metabolism_drain",
        value: || Value::from(0.25),
        ticks: 6,
    },
    Witness {
        path: "movement_drain",
        value: || Value::from(0.25),
        ticks: 6,
    },
    Witness {
        path: "bot_speed",
        value: || Value::from(2.5),
        ticks: 4,
    },
    Witness {
        path: "bot_radius",
        value: || Value::from(25.0),
        ticks: 4,
    },
    Witness {
        path: "sense_radius",
        value: || Value::from(60.0),
        ticks: 6,
    },
    Witness {
        path: "temperature_discomfort_rate",
        value: || Value::from(0.4),
        ticks: 6,
    },
    // TWO KNOBS ARE DELIBERATELY ABSENT, and the reason is a real limit of this harness shape
    // rather than an opinion about the knobs.
    //
    // aging_health_decay_rate CANNOT be witnessed by perturbing one published path from the
    // default config. aging_health_decay_max defaults to 0.0 and the validator requires
    // max >= rate, so every positive rate is rejected before a world is ever built. Witnessing it
    // needs a non-default BASELINE with decay already enabled -- a different harness shape than
    // "perturb exactly one knob away from default", because two knobs have to move together.
    // Cross-field invariants like this one mean single-knob perturbation cannot reach the whole
    // scientific surface, which is the main thing standing between 15 witnesses and 88.
    //
    // initial_food is absent for the same class of reason (it may not exceed food_max, which
    // defaults to 0.5) and simply has not been re-verified since. Every entry in this table has
    // been OBSERVED to move material state on a real run; nothing is here on the strength of an
    // argument that it ought to.
    Witness {
        path: "aging_tick_interval",
        value: || Value::from(1),
        ticks: 8,
    },
];

/// THE WITNESS GATE. Every listed knob must move material, non-config world state.
///
/// A failure here means one of two things, and both matter: either the knob is a ghost -- public,
/// patchable, and read by nothing -- or its consuming code path is unreachable under this fixture.
/// The second is not a lesser finding; an unreachable consumer is how bd-pdx5's Combat category
/// spent its entire life reporting coverage it did not have.
#[test]
fn bd_dorx_every_witnessed_knob_moves_material_world_state() {
    let baseline = run_material(ScriptBotsConfig::default(), 8)
        .expect("the default config must build and step a world");

    // Collect every failure rather than panicking on the first. One run then tells you about all
    // 17 witnesses instead of one per run, which matters when a verification cycle is minutes long
    // and the lane is contended.
    let mut ghosts = Vec::new();
    let mut broken = Vec::new();
    for witness in WITNESSES {
        let Some(config) = perturbed_config(witness.path, (witness.value)()) else {
            broken.push(format!("{}: not a published path", witness.path));
            continue;
        };
        match run_material(config, witness.ticks.max(8)) {
            Err(reason) => broken.push(format!("{}: {reason}", witness.path)),
            Ok(perturbed) => {
                let moved: Vec<&str> = baseline
                    .iter()
                    .zip(perturbed.iter())
                    .filter(|((_, before), (_, after))| before != after)
                    .map(|((lane, _), _)| *lane)
                    .collect();
                if moved.is_empty() {
                    ghosts.push(witness.path);
                } else {
                    println!("WITNESS\t{}\tmoved={moved:?}", witness.path);
                }
            }
        }
    }

    assert!(
        broken.is_empty(),
        "these witnesses could not run, which is a defect in the WITNESS not proof about the knob: {broken:#?}"
    );
    assert!(
        ghosts.is_empty(),
        "these knobs are published and patchable but moved no material world state: {ghosts:?}"
    );
}

/// Witnesses must address knobs this registry actually calls scientific.
///
/// Without this, a witness could quietly drift onto an Operational or Presentation path and
/// inflate the apparent coverage of the scientific surface.
#[test]
fn bd_dorx_every_witness_targets_a_scientific_knob() {
    for witness in WITNESSES {
        let spec = KNOB_ROLES
            .iter()
            .find(|spec| spec.path == witness.path)
            .unwrap_or_else(|| panic!("{} is witnessed but not classified", witness.path));
        assert!(
            spec.role.is_scientific(),
            "{} is witnessed as scientific but classified {:?}",
            witness.path,
            spec.role
        );
    }
}

/// Ratchet for scientific witness coverage. It may only ever be raised.
///
/// Deliberately a floor rather than an equality with the scientific count: 88 knobs are
/// classified scientific and far fewer are witnessed today, so asserting completeness would be a
/// lie of exactly the kind bd-dorx exists to stop. A floor makes the debt visible in the test
/// output every run while making it impossible to quietly delete a witness.
///
/// 15 is the OBSERVED count -- every one of these was seen to move material state on a real run.
/// It is not an aspiration. An earlier value of 17 was aspirational and left this gate red,
/// which is the failure it exists to prevent, so the number now tracks evidence only.
const WITNESS_COVERAGE_FLOOR: usize = 15;

/// Report scientific coverage, and hold the line against it regressing.
///
/// This is deliberately NOT an assertion that all 88 scientific knobs are witnessed: they are not,
/// and claiming otherwise is the exact false assurance this bead exists to prevent. It asserts
/// that coverage never goes DOWN, so the debt can only shrink. The remaining entries are tracked
/// on bd-dorx rather than silently tolerated here.
#[test]
fn bd_dorx_scientific_witness_coverage_does_not_regress() {
    let scientific: Vec<&str> = KNOB_ROLES
        .iter()
        .filter(|spec| spec.role.is_scientific())
        .map(|spec| spec.path)
        .collect();
    let witnessed: Vec<&str> = WITNESSES
        .iter()
        .filter(|w| scientific.contains(&w.path))
        .map(|w| w.path)
        .collect();

    println!(
        "SCIENTIFIC_KNOBS={} WITNESSED={} UNWITNESSED={}",
        scientific.len(),
        witnessed.len(),
        scientific.len() - witnessed.len()
    );
    for path in &scientific {
        if !witnessed.contains(path) {
            println!("UNWITNESSED\t{path}");
        }
    }

    assert!(
        witnessed.len() >= WITNESS_COVERAGE_FLOOR,
        "scientific witness coverage regressed: {} witnessed, floor is {WITNESS_COVERAGE_FLOOR}. \
         Raise the floor when you add witnesses; never lower it to make this pass",
        witnessed.len()
    );
    assert!(
        scientific.len() >= witnessed.len(),
        "a witness targeted a path the registry does not call scientific"
    );
}

/// Paths `KNOB_RANGES` declares that the control plane does not publish on a default config.
///
/// This set is PINNED rather than merely reported, so it cannot grow quietly. Two distinct things
/// live in it and they must not be conflated:
///
/// * The nine `render.*` entries are LEGITIMATE. `render.post`, `render.day_night` and
///   `render.auto_exposure` are `Option` fields that serialize to `null` when unset, and the
///   flattener stops at any non-object node, so each is one leaf on a default world and expands
///   only once populated. The deeper paths are real and reachable — just not discoverable from a
///   default config. That is the discovery/mutation asymmetry recorded on bd-dorx.
/// * `mutation.primary` and `mutation.secondary` are STALE, and are a live defect. No
///   `ScriptBotsConfig` field produces either path: the config carries
///   `reproduction_mutation_scale`, `reproduction_meta_mutation_chance` and
///   `reproduction_meta_mutation_scale`. The `mutation_primary` / `mutation_secondary` identifiers
///   in `lib.rs` belong to an agent PROJECTION struct, and the only other `mutation.primary.*`
///   references in the tree are analytics metric names — a different namespace entirely. They
///   declare ranges for paths no config can ever emit.
///
/// Unknown paths are simply not range-checked (`validate` does `if let Some(range) = ...find(...)`),
/// so the stale pair is inert rather than harmful — which is precisely why it survived. Removing
/// them needs an edit to `crates/scriptbots-core/src/lib.rs` (~4749), which was leased by another
/// agent when this gate was written.
const KNOB_RANGES_NOT_PUBLISHED_BY_DEFAULT: &[&str] = &[
    "mutation.primary",
    "mutation.secondary",
    "render.auto_exposure.speed_brighten",
    "render.auto_exposure.speed_darken",
    "render.day_night.cycle_ticks",
    "render.day_night.night_ambient",
    "render.day_night.start_phase",
    "render.post.bloom.intensity",
    "render.post.bloom.threshold",
    "render.post.vignette.intensity",
    "render.post.vignette.smoothness",
];

/// `KNOB_RANGES` must not drift from the surface the control plane actually publishes.
///
/// Asserting EQUALITY with the pinned set, not merely containment, is the point. Containment would
/// let a new stale entry slip in, and it would also let the two known-stale entries be removed
/// without anyone updating this list — so the gate would keep passing while its own documentation
/// rotted. Equality means both directions are load-bearing: adding a stale range fails, and fixing
/// one fails until the fix is recorded here.
///
/// This is the same defect class `no_spec_is_stale` catches for `KNOB_ROLES`. That gate covers the
/// new registry only; `KNOB_RANGES` predates it and was never checked.
#[test]
fn bd_dorx_knob_ranges_declare_no_unexpected_unpublished_path() {
    let published = default_knob_paths();
    let mut unpublished: Vec<&str> = scriptbots_core::KNOB_RANGES
        .iter()
        .map(|range| range.path)
        .filter(|path| !published.iter().any(|known| known == path))
        .collect();
    unpublished.sort_unstable();

    let mut expected: Vec<&str> = KNOB_RANGES_NOT_PUBLISHED_BY_DEFAULT.to_vec();
    expected.sort_unstable();

    assert_eq!(
        unpublished, expected,
        "KNOB_RANGES drifted from the published surface. A path here that is NOT under \
         render.auto_exposure/render.day_night/render.post is a stale range declaring a knob no \
         config can emit. If you fixed one, delete it from KNOB_RANGES_NOT_PUBLISHED_BY_DEFAULT too"
    );
}
