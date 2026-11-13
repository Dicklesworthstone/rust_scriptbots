//! Complete, bounded browser launch recipes. These links restart initial
//! conditions; saved owner continuations use the separate checkpoint codec.

use std::collections::BTreeMap;

use scriptbots_core::permalink::to_canonical_value;
use scriptbots_core::{CoreBuildIdentityV0, ScriptBotsConfig};
use serde::{Deserialize, Serialize};

use super::{BrainPreset, DiffRow, SeedStrategy, SimSpec, SnapshotFormat};

pub(super) const PREFIX: &str = "sbb1.";
pub(super) const MAX_PAYLOAD_BYTES: usize = 65_536;
const MAX_LINK_BYTES: usize = PREFIX.len() + MAX_PAYLOAD_BYTES.div_ceil(3) * 4 + 17;
const MAX_FOUNDERS: usize = 50_000;
const MAX_GRID_CELLS: u64 = 262_144;
const MAX_HISTORY: usize = 65_536;
const MAX_CHANGES: usize = 64;
const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

mod captured {
    include!(concat!(env!("OUT_DIR"), "/browser_build.rs"));

    pub(super) fn inputs() -> super::BrowserBuildInputs {
        super::BrowserBuildInputs {
            source_fingerprint: SOURCE.map(str::to_owned),
            source_inventory: SOURCE_INVENTORY
                .iter()
                .map(|&(path, fingerprint)| super::SourceInput {
                    path: path.to_owned(),
                    fingerprint: fingerprint.to_owned(),
                })
                .collect(),
            lockfile_fingerprint: LOCKFILE.map(str::to_owned),
            toolchain_fingerprint: TOOLCHAIN.map(str::to_owned),
            compiler: COMPILER.map(str::to_owned),
            target: TARGET.map(str::to_owned),
            rustflags: RUSTFLAGS.map(str::to_owned),
            profile: PROFILE.map(str::to_owned),
            opt_level: OPT_LEVEL.map(str::to_owned),
            debug: DEBUG.map(str::to_owned),
            target_features: TARGET_FEATURES.map(str::to_owned),
            core_lane: scriptbots_core::CoreBuildIdentityV0::current(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SourceInput {
    path: String,
    fingerprint: String,
}

/// Recorded inputs, not an attestation or proof of cross-build determinism.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct BrowserBuildInputs {
    pub(super) source_fingerprint: Option<String>,
    source_inventory: Vec<SourceInput>,
    pub(super) lockfile_fingerprint: Option<String>,
    pub(super) toolchain_fingerprint: Option<String>,
    compiler: Option<String>,
    target: Option<String>,
    rustflags: Option<String>,
    profile: Option<String>,
    opt_level: Option<String>,
    debug: Option<String>,
    target_features: Option<String>,
    core_lane: CoreBuildIdentityV0,
}

impl BrowserBuildInputs {
    pub(super) fn current() -> Self {
        captured::inputs()
    }

    fn complete(&self) -> bool {
        !self.source_inventory.is_empty()
            && [
                &self.source_fingerprint,
                &self.lockfile_fingerprint,
                &self.toolchain_fingerprint,
                &self.compiler,
                &self.target,
                &self.rustflags,
                &self.profile,
                &self.opt_level,
                &self.debug,
                &self.target_features,
            ]
            .iter()
            .all(|value| value.is_some())
    }

    pub(super) fn comparison(&self, local: &Self) -> &'static str {
        if !self.complete() || !local.complete() {
            "unavailable"
        } else if self == local {
            "recorded_match"
        } else {
            "mismatch"
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.source_inventory.len() > 256
            || self.source_inventory.iter().any(|entry| {
                entry.path.is_empty()
                    || entry.path.len() > 256
                    || !valid_fingerprint(&entry.fingerprint)
            })
            || self
                .source_inventory
                .windows(2)
                .any(|pair| pair[0].path >= pair[1].path)
        {
            return Err("browser source inventory is oversized or noncanonical".to_owned());
        }
        for value in [
            &self.source_fingerprint,
            &self.lockfile_fingerprint,
            &self.toolchain_fingerprint,
        ]
        .into_iter()
        .flatten()
        {
            if !valid_fingerprint(value) {
                return Err("browser build fingerprint is malformed".to_owned());
            }
        }
        for value in [
            &self.compiler,
            &self.target,
            &self.rustflags,
            &self.profile,
            &self.opt_level,
            &self.debug,
            &self.target_features,
        ]
        .into_iter()
        .flatten()
        {
            if value.len() > 16_384 {
                return Err("browser build input exceeds its byte bound".to_owned());
            }
        }
        Ok(())
    }
}

fn valid_fingerprint(value: &str) -> bool {
    value.strip_prefix("fnv1a64:").is_some_and(|hex| {
        hex.len() == 16
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

/// Metadata used to explain what a link can reproduce before a user shares it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LaunchInfo {
    /// `browser_launch`, `gallery_reference`, or `checkpoint_continuation`.
    pub kind: String,
    /// A launch restarts initial conditions; it never forks the current tick.
    pub semantics: String,
    /// Whether this handle retains a complete supported initial recipe.
    pub shareable: bool,
    /// Exact root seed, represented as decimal text for JavaScript.
    pub seed: Option<String>,
    /// Explicit browser founders; absent for gallery references and continuations.
    pub initial_population: Option<usize>,
    /// Actual browser family or explicit `wander`/`none` strategy.
    pub brain_preset: Option<String>,
    /// The browser founder recipe's version, when known.
    pub recipe_version: Option<u16>,
    /// Typed explanatory scope or refusal reason.
    pub reason: Option<String>,
}

/// Version 1 freezes uniform browser seeding and records the complete effective
/// configuration. Changes are explanatory rows against the original launch,
/// not a replacement for the configuration or executable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct BrowserLaunch {
    schema_version: u16,
    founder_recipe_version: u16,
    pub(super) config: ScriptBotsConfig,
    seed: u64,
    initial_population: u32,
    snapshot_format: SnapshotFormat,
    seed_strategy: SeedStrategy,
    default_brain: Option<BrainPreset>,
    pub(super) build: BrowserBuildInputs,
    pub(super) changes: Vec<DiffRow>,
}

impl BrowserLaunch {
    pub(super) fn from_fresh(spec: &SimSpec, config: ScriptBotsConfig) -> Result<Self, String> {
        let launch = Self {
            schema_version: 1,
            founder_recipe_version: 1,
            seed: config
                .rng_seed
                .ok_or_else(|| "a browser launch requires the actual resolved seed".to_owned())?,
            config,
            initial_population: u32::try_from(spec.initial_population)
                .map_err(|_| "browser founder count does not fit its protocol".to_owned())?,
            snapshot_format: spec.snapshot_format,
            seed_strategy: spec.seed_strategy,
            default_brain: spec.default_brain,
            build: BrowserBuildInputs::current(),
            changes: spec
                .launch
                .as_ref()
                .map_or_else(Vec::new, |launch| launch.changes.clone()),
        };
        // The world has already passed browser/core admission. Recording its
        // inputs must not make an otherwise usable owner fail merely because a
        // future source inventory or recipe is too large for a URL. Encoding
        // and metadata report that separate sharing refusal explicitly.
        Ok(launch)
    }

    pub(super) fn into_spec(self) -> SimSpec {
        let mut spec = SimSpec::new(
            self.config.clone(),
            self.initial_population as usize,
            Some(self.seed),
            self.snapshot_format,
            self.seed_strategy,
            self.default_brain,
        );
        // The link already records the effective population policy. Applying the
        // fresh-demo override again would discard a deliberate fork assignment.
        spec.preserve_population_config = true;
        spec.launch = Some(self);
        spec
    }

    pub(super) fn info(&self) -> LaunchInfo {
        let refusal = self.encode().err();
        LaunchInfo {
            kind: "browser_launch".to_owned(),
            semantics: "initial_conditions_restart".to_owned(),
            shareable: refusal.is_none(),
            seed: Some(self.seed.to_string()),
            initial_population: Some(self.initial_population as usize),
            brain_preset: Some(
                self.default_brain
                    .map_or_else(
                        || match self.seed_strategy {
                            SeedStrategy::Wander => "wander",
                            SeedStrategy::None => "none",
                        },
                        BrainPreset::name,
                    )
                    .to_owned(),
            ),
            recipe_version: Some(self.founder_recipe_version),
            reason: Some(refusal.map_or_else(
                || "Restarts the recorded founders and configuration at tick zero; live interventions and current state are not included.".to_owned(),
                |error| format!("This initial recipe cannot be shared: {error}. The running owner remains usable; save its checkpoint for continuation."),
            )),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.founder_recipe_version != 1 {
            return Err("unsupported browser launch or founder recipe version".to_owned());
        }
        if self.config.rng_seed != Some(self.seed) {
            return Err(
                "browser launch seed disagrees with its effective configuration".to_owned(),
            );
        }
        if self.default_brain.is_some() && self.seed_strategy != SeedStrategy::None {
            return Err(
                "browser launch declares conflicting brain and wander strategies".to_owned(),
            );
        }
        validate_config(&self.config, self.initial_population as usize)?;
        self.build.validate()?;
        if self.changes.len() > MAX_CHANGES
            || self
                .changes
                .windows(2)
                .any(|rows| rows[0].knob >= rows[1].knob)
        {
            return Err("browser launch changes are oversized, duplicated or unsorted".to_owned());
        }
        let config = to_canonical_value(&self.config).map_err(|error| error.to_string())?;
        for row in &self.changes {
            // Rows report the exact stored f32 value as f64. Its rounded value
            // can sit a few bits beyond a decimal registry endpoint; admission
            // validates requested assignments before applying that conversion.
            if scriptbots_core::knob_range(&row.knob).is_none()
                || !row.this_value.is_finite()
                || !row.parent_value.is_finite()
                || !super::extract_json_number(&config, &row.knob)
                    .is_some_and(|value| value.to_bits() == row.this_value.to_bits())
            {
                return Err(format!(
                    "browser launch diff disagrees with config at '{}'",
                    row.knob
                ));
            }
        }
        Ok(())
    }

    pub(super) fn fork(&self, patch: &[(String, f64)]) -> Result<Self, String> {
        // Compare the actual stored f32 values widened to f64 while their type is
        // known. Plain to_value changes their meaning when another workspace
        // dependency enables serde_json/arbitrary_precision.
        let original = to_canonical_value(&self.config).map_err(|error| error.to_string())?;
        validate_patch(&original, patch)?;
        let mut next = self.clone();
        scriptbots_core::gallery::apply_knob_diff(&mut next.config, patch)?;
        validate_config(&next.config, self.initial_population as usize)?;
        let configured = to_canonical_value(&next.config).map_err(|error| error.to_string())?;
        let mut changes: BTreeMap<_, _> = self
            .changes
            .iter()
            .map(|row| (row.knob.clone(), row.clone()))
            .collect();
        for (path, _) in patch {
            let before = super::extract_json_number(&original, path).ok_or_else(|| {
                format!("knob '{path}' is not an available numeric configuration field")
            })?;
            let after = super::extract_json_number(&configured, path)
                .ok_or_else(|| format!("knob '{path}' disappeared after validation"))?;
            let baseline = changes.get(path).map_or(before, |row| row.parent_value);
            if baseline.to_bits() == after.to_bits() {
                changes.remove(path);
            } else {
                changes.insert(
                    path.clone(),
                    DiffRow {
                        knob: path.clone(),
                        parent_value: baseline,
                        this_value: after,
                    },
                );
            }
        }
        next.changes = changes.into_values().collect();
        next.build = BrowserBuildInputs::current();
        next.validate()?;
        Ok(next)
    }

    pub(super) fn encode(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        if bytes.len() > MAX_PAYLOAD_BYTES {
            return Err(format!(
                "browser launch exceeds {MAX_PAYLOAD_BYTES} encoded bytes"
            ));
        }
        Ok(encode_payload(&bytes))
    }

    pub(super) fn decode(link: &str) -> Result<Self, String> {
        // Check the textual cap before any proportional allocation or parsing.
        if link.len() > MAX_LINK_BYTES {
            return Err("browser launch link exceeds its byte bound".to_owned());
        }
        let body = link
            .strip_prefix(PREFIX)
            .ok_or_else(|| "invalid browser launch prefix".to_owned())?;
        let (encoded, checksum) = body
            .split_once('.')
            .ok_or_else(|| "browser launch is missing its corruption checksum".to_owned())?;
        if checksum.len() != 16
            || !checksum
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("browser launch checksum is malformed".to_owned());
        }
        let bytes = decode_base64(encoded)?;
        if checksum != format!("{:016x}", payload_fingerprint(&bytes)) {
            return Err("browser launch corruption checksum mismatch".to_owned());
        }
        let launch: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid browser launch: {error}"))?;
        launch.validate()?;
        // This also rejects ignored nested fields, duplicate nested keys, missing
        // defaulted fields and alternative numeric spellings. We produce this
        // exact canonical form; it is not a permissive general JSON import API.
        if serde_json::to_vec(&launch).map_err(|error| error.to_string())? != bytes {
            return Err("browser launch payload is not canonical".to_owned());
        }
        Ok(launch)
    }
}

pub(super) fn validate_config(config: &ScriptBotsConfig, founders: usize) -> Result<(), String> {
    if founders > MAX_FOUNDERS
        || config.population_minimum > MAX_FOUNDERS
        || u64::from(config.population_spawn_count) > MAX_FOUNDERS as u64
    {
        return Err(format!(
            "browser founder and population injection counts must not exceed {MAX_FOUNDERS}"
        ));
    }
    let (width, height) = config
        .food_dimensions()
        .map_err(|error| error.to_string())?;
    if u64::from(width) * u64::from(height) > MAX_GRID_CELLS {
        return Err(format!(
            "browser dense grids must not exceed {MAX_GRID_CELLS} cells"
        ));
    }
    if config.history_capacity > MAX_HISTORY
        || config.narrative_capacity > MAX_HISTORY
        || u64::from(config.communication_mi_window) > MAX_HISTORY as u64
        || config.reproduction_gene_log_capacity > MAX_HISTORY
        || config.replay_event_tick_cap > MAX_HISTORY
        || config.interaction_event_tick_cap > MAX_HISTORY
        || config.archive_max_bytes > 64 * 1024 * 1024
        || config.archive_space.axes.len() > 8
    {
        return Err(
            "browser history, event, archive or nested configuration exceeds its admission bound"
                .to_owned(),
        );
    }
    Ok(())
}

pub(super) fn validate_patch(
    config: &serde_json::Value,
    patch: &[(String, f64)],
) -> Result<(), String> {
    if patch.len() > MAX_CHANGES {
        return Err(format!(
            "a browser restart accepts at most {MAX_CHANGES} knob assignments"
        ));
    }
    let mut unique = std::collections::BTreeSet::new();
    for (path, value) in patch {
        validate_assignment(config, path, *value)?;
        if !unique.insert(path) {
            return Err(format!("duplicate knob assignment '{path}'"));
        }
    }
    Ok(())
}

fn validate_assignment(config: &serde_json::Value, path: &str, value: f64) -> Result<(), String> {
    let range = scriptbots_core::knob_range(path)
        .ok_or_else(|| format!("unknown knob id '{path}' not in registry"))?;
    if !value.is_finite() || value < range.min || value > range.max {
        return Err(format!(
            "knob '{path}' must be finite and within [{}, {}]",
            range.min, range.max
        ));
    }
    let mut field = config;
    for part in path.split('.') {
        field = field
            .get(part)
            .ok_or_else(|| format!("knob '{path}' is absent from the effective configuration"))?;
    }
    if !field.is_number() {
        return Err(format!(
            "knob '{path}' is not an available numeric configuration field"
        ));
    }
    if (field.is_u64() || field.is_i64()) && value.fract() != 0.0 {
        return Err(format!("integer knob '{path}' requires an integral value"));
    }
    Ok(())
}

fn payload_fingerprint(bytes: &[u8]) -> u64 {
    b"scriptbots.browser-launch.v1\0"
        .iter()
        .chain(bytes)
        .fold(OFFSET, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(PRIME)
        })
}

fn encode_payload(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(PREFIX.len() + bytes.len().div_ceil(3) * 4 + 17);
    text.push_str(PREFIX);
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or_default();
        let c = chunk.get(2).copied().unwrap_or_default();
        text.push(char::from(ALPHABET[usize::from(a >> 2)]));
        text.push(char::from(ALPHABET[usize::from(((a & 3) << 4) | (b >> 4))]));
        if chunk.len() > 1 {
            text.push(char::from(
                ALPHABET[usize::from(((b & 15) << 2) | (c >> 6))],
            ));
        }
        if chunk.len() > 2 {
            text.push(char::from(ALPHABET[usize::from(c & 63)]));
        }
    }
    use std::fmt::Write as _;
    let _ = write!(text, ".{:016x}", payload_fingerprint(bytes));
    text
}

fn decode_base64(text: &str) -> Result<Vec<u8>, String> {
    if text.is_empty() || text.len() % 4 == 1 {
        return Err("browser launch base64 length is invalid".to_owned());
    }
    let byte_count = text.len() / 4 * 3
        + match text.len() % 4 {
            2 => 1,
            3 => 2,
            _ => 0,
        };
    if byte_count > MAX_PAYLOAD_BYTES {
        return Err("browser launch payload exceeds its byte bound".to_owned());
    }
    let mut bytes = Vec::with_capacity(byte_count);
    for chunk in text.as_bytes().chunks(4) {
        let mut values = [0_u8; 4];
        for (index, &byte) in chunk.iter().enumerate() {
            values[index] = match byte {
                b'A'..=b'Z' => byte - b'A',
                b'a'..=b'z' => byte - b'a' + 26,
                b'0'..=b'9' => byte - b'0' + 52,
                b'-' => 62,
                b'_' => 63,
                _ => return Err("browser launch contains invalid base64url bytes".to_owned()),
            };
        }
        if (chunk.len() == 2 && values[1] & 15 != 0) || (chunk.len() == 3 && values[2] & 3 != 0) {
            return Err("browser launch base64url has nonzero padding bits".to_owned());
        }
        bytes.push((values[0] << 2) | (values[1] >> 4));
        if chunk.len() > 2 {
            bytes.push((values[1] << 4) | (values[2] >> 2));
        }
        if chunk.len() > 3 {
            bytes.push((values[2] << 6) | values[3]);
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;
    use crate::{SimHandle, Simulation, check_build_match_identity, init_from_permalink};

    fn fresh(brain: Option<BrainPreset>, strategy: SeedStrategy, seed: Option<u64>) -> Simulation {
        Simulation::new(SimSpec::new(
            ScriptBotsConfig {
                world_width: 240,
                world_height: 160,
                food_cell_size: 40,
                food_growth_rate: 0.037,
                persistence_interval: 0,
                reproduction_energy_threshold: f32::MAX,
                ..Default::default()
            },
            7,
            seed,
            SnapshotFormat::Json,
            strategy,
            brain,
        ))
        .unwrap()
    }

    fn handle(simulation: Simulation) -> SimHandle {
        SimHandle {
            inner: Rc::new(RefCell::new(simulation)),
        }
    }

    fn recipe() -> BrowserLaunch {
        fresh(Some(BrainPreset::Mlp), SeedStrategy::None, Some(17))
            .spec
            .launch
            .unwrap()
    }

    fn assert_same_world(left: &SimHandle, right: &SimHandle) {
        assert_eq!(
            left.inner.borrow().core.world().world_digest_v1().unwrap(),
            right.inner.borrow().core.world().world_digest_v1().unwrap()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn browser_launch_roundtrips_actual_founders_config_and_all_families() {
        for brain in [BrainPreset::Mlp, BrainPreset::Dwraon, BrainPreset::Assembly] {
            let original = handle(fresh(Some(brain), SeedStrategy::None, Some(u64::MAX - 7)));
            let link = original.permalink_string().unwrap();
            assert!(link.starts_with(PREFIX));
            assert!(link.len() <= MAX_LINK_BYTES);
            println!("{} browser launch URL bytes: {}", brain.name(), link.len());
            let metadata = original.launch_info();
            assert_eq!(
                metadata.seed.as_deref(),
                Some((u64::MAX - 7).to_string().as_str())
            );
            assert_eq!(metadata.initial_population, Some(7));
            assert_eq!(metadata.brain_preset.as_deref(), Some(brain.name()));
            assert_eq!(metadata.semantics, "initial_conditions_restart");
            assert!(metadata.shareable);
            assert_eq!(crate::permalink_info_details(&link).unwrap(), metadata);
            let restored = init_from_permalink(&link).unwrap();
            assert_same_world(&original, &restored);
            assert_eq!(restored.inner.borrow().core.world().agent_count(), 7);
            assert_eq!(
                serde_json::to_vec(original.inner.borrow().core.world().config()).unwrap(),
                serde_json::to_vec(restored.inner.borrow().core.world().config()).unwrap()
            );
            for id in restored.inner.borrow().core.world().agents().iter_handles() {
                let simulation = restored.inner.borrow();
                assert!(simulation.core.world().agent_brain_genome(id).is_some());
                assert_eq!(
                    simulation
                        .core
                        .world()
                        .agent_runtime(id)
                        .unwrap()
                        .brain
                        .kind(),
                    Some(brain.kind())
                );
            }
            original.tick_snapshot(8).unwrap();
            restored.tick_snapshot(8).unwrap();
            assert_same_world(&original, &restored);
            assert_eq!(
                original.permalink_string().unwrap(),
                link,
                "elapsed state never replaces the initial recipe"
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn explicit_demo_strategies_roundtrip_without_becoming_mlp() {
        for strategy in [SeedStrategy::None, SeedStrategy::Wander] {
            let original = handle(fresh(None, strategy, Some(23)));
            let restored = init_from_permalink(&original.permalink_string().unwrap()).unwrap();
            assert_eq!(restored.inner.borrow().spec.default_brain, None);
            assert_eq!(restored.inner.borrow().spec.seed_strategy, strategy);
            assert_eq!(
                serde_json::to_vec(&original.tick_snapshot(5).unwrap()).unwrap(),
                serde_json::to_vec(&restored.tick_snapshot(5).unwrap()).unwrap()
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn entropy_seed_is_shareable_and_resets_keep_original_option_semantics() {
        let original = handle(fresh(Some(BrainPreset::Mlp), SeedStrategy::None, None));
        let actual = original
            .inner
            .borrow()
            .core
            .world()
            .config()
            .rng_seed
            .unwrap();
        let restored = init_from_permalink(&original.permalink_string().unwrap()).unwrap();
        assert_eq!(original.launch_info().seed, Some(actual.to_string()));
        assert_same_world(&original, &restored);
        original.tick_snapshot(4).unwrap();
        restored.tick_snapshot(4).unwrap();
        assert_same_world(&original, &restored);
        original.inner.borrow_mut().reset(Some(29)).unwrap();
        assert_eq!(original.launch_info().seed.as_deref(), Some("29"));
        original.inner.borrow_mut().reset(None).unwrap();
        {
            let simulation = original.inner.borrow();
            assert_eq!(simulation.spec.seed, None);
            assert_eq!(simulation.spec.base_config.rng_seed, None);
            assert_eq!(
                simulation.spec.launch.as_ref().unwrap().seed,
                simulation
                    .core
                    .world()
                    .agent_substream_protocol_v1()
                    .root_seed()
            );
        }
        let reset = init_from_permalink(&original.permalink_string().unwrap()).unwrap();
        assert_same_world(&original, &reset);

        let mut nested = fresh(Some(BrainPreset::Dwraon), SeedStrategy::None, None).spec;
        nested.base_config.rng_seed = Some(31);
        let mut simulation = Simulation::new(nested).unwrap();
        simulation.reset(Some(37)).unwrap();
        assert_eq!(simulation.spec.launch.as_ref().unwrap().seed, 37);
        simulation.reset(None).unwrap();
        assert_eq!(simulation.spec.launch.as_ref().unwrap().seed, 31);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn browser_forks_preserve_full_recipe_and_restart_at_tick_zero() {
        let original = handle(fresh(
            Some(BrainPreset::Assembly),
            SeedStrategy::None,
            Some(43),
        ));
        let root_link = original.permalink_string().unwrap();
        let root = BrowserLaunch::decode(&root_link).unwrap();
        original.tick_snapshot(5).unwrap();
        let before = original
            .inner
            .borrow()
            .core
            .world()
            .world_digest_v1()
            .unwrap();
        let child_link = original
            .fork_patch(&[("food_growth_rate".to_owned(), 0.2)])
            .unwrap();
        assert_eq!(
            original
                .inner
                .borrow()
                .core
                .world()
                .world_digest_v1()
                .unwrap(),
            before
        );
        assert_eq!(original.permalink_string().unwrap(), root_link);
        let child = BrowserLaunch::decode(&child_link).unwrap();
        assert_eq!(child.seed, 43);
        assert_eq!(child.initial_population, 7);
        assert_eq!(child.default_brain, Some(BrainPreset::Assembly));
        assert_eq!(child.seed_strategy, SeedStrategy::None);
        let mut expected_config = root.config.clone();
        expected_config.food_growth_rate = 0.2;
        assert_eq!(
            serde_json::to_vec(&child.config).unwrap(),
            serde_json::to_vec(&expected_config).unwrap()
        );
        let restarted = init_from_permalink(&child_link).unwrap();
        assert_eq!(restarted.inner.borrow().core.world_tick().0, 0);
        let mut expected_spec = SimSpec::new(
            expected_config,
            7,
            Some(43),
            SnapshotFormat::Json,
            SeedStrategy::None,
            Some(BrainPreset::Assembly),
        );
        expected_spec.preserve_population_config = true;
        let expected = handle(Simulation::new(expected_spec).unwrap());
        assert_same_world(&restarted, &expected);
        restarted.tick_snapshot(5).unwrap();
        expected.tick_snapshot(5).unwrap();
        assert_same_world(&restarted, &expected);
        let grandchild = BrowserLaunch::decode(
            &restarted
                .fork_patch(&[("population_minimum".to_owned(), 12.0)])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            grandchild.config.population_minimum, 12,
            "reconstruction must preserve a deliberate population policy"
        );
        assert_eq!(grandchild.changes.len(), 2);
        let rows = crate::permalink_diff_rows(&child_link).unwrap();
        assert_eq!(
            rows[0].parent_value,
            f64::from(root.config.food_growth_rate)
        );
        assert_eq!(rows[0].this_value, f64::from(0.2_f32));
        let restored_baseline = child
            .fork(&[(
                "food_growth_rate".to_owned(),
                f64::from(root.config.food_growth_rate),
            )])
            .unwrap();
        assert!(restored_baseline.changes.is_empty());
        for patch in [
            vec![("population_minimum".to_owned(), 3.5)],
            vec![("food_growth_rate".to_owned(), f64::NAN)],
            vec![("missing".to_owned(), 1.0)],
            vec![("world_width".to_owned(), 241.0)],
            vec![
                ("food_growth_rate".to_owned(), 0.1),
                ("food_growth_rate".to_owned(), 0.2),
            ],
        ] {
            assert!(original.fork_patch(&patch).is_err());
        }
        assert_eq!(
            original
                .inner
                .borrow()
                .core
                .world()
                .world_digest_v1()
                .unwrap(),
            before
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn browser_fork_diffs_preserve_typed_float_values_with_arbitrary_precision() {
        let mut root = recipe();
        root.config.render.day_night = Some(scriptbots_core::RenderDayNightSettings {
            start_phase: Some(0.123),
            cycle_ticks: Some(1_024),
            ..Default::default()
        });
        let changed = root
            .fork(&[
                ("food_growth_rate".to_owned(), 0.2),
                ("population_minimum".to_owned(), 12.0),
                ("render.day_night.start_phase".to_owned(), 0.3),
            ])
            .unwrap();
        let decoded = BrowserLaunch::decode(&changed.encode().unwrap()).unwrap();
        for (path, before, after) in [
            ("food_growth_rate", f64::from(0.037_f32), f64::from(0.2_f32)),
            ("population_minimum", 0.0, 12.0),
            (
                "render.day_night.start_phase",
                f64::from(0.123_f32),
                f64::from(0.3_f32),
            ),
        ] {
            let row = decoded.changes.iter().find(|row| row.knob == path).unwrap();
            assert_eq!(
                row.parent_value.to_bits(),
                before.to_bits(),
                "{path} original value"
            );
            assert_eq!(
                row.this_value.to_bits(),
                after.to_bits(),
                "{path} stored value"
            );
        }
        // Returning to the real stored values removes the complete diff even
        // when the first fork was encoded and decoded in between.
        let restored = decoded
            .fork(&[
                (
                    "food_growth_rate".to_owned(),
                    f64::from(root.config.food_growth_rate),
                ),
                ("population_minimum".to_owned(), 0.0),
                (
                    "render.day_night.start_phase".to_owned(),
                    f64::from(0.123_f32),
                ),
            ])
            .unwrap();
        assert!(restored.changes.is_empty());
        assert_eq!(restored.config, root.config);
        let mut understated = changed;
        understated.changes[0].this_value = 0.2;
        assert!(
            understated.encode().is_err(),
            "decimal metadata cannot replace the exact stored f32 observation"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn continuations_refuse_recipe_sharing_until_explicit_restart() {
        let mut original = fresh(Some(BrainPreset::Dwraon), SeedStrategy::None, Some(47));
        original.tick(4).unwrap();
        let continued =
            handle(Simulation::from_checkpoint(&original.checkpoint().unwrap()).unwrap());
        let before = continued
            .inner
            .borrow()
            .core
            .world()
            .world_digest_v1()
            .unwrap();
        let metadata = continued.launch_info();
        assert_eq!(metadata.kind, "checkpoint_continuation");
        assert_eq!(metadata.semantics, "saved_owner_continuation");
        assert!(!metadata.shareable);
        assert_eq!(metadata.initial_population, None);
        assert!(
            continued
                .permalink_string()
                .unwrap_err()
                .contains("original browser founders")
        );
        assert!(continued.fork_patch(&[]).is_err());
        assert_eq!(
            continued
                .inner
                .borrow()
                .core
                .world()
                .world_digest_v1()
                .unwrap(),
            before
        );
        continued.inner.borrow_mut().reset(Some(53)).unwrap();
        assert_eq!(continued.launch_info().kind, "browser_launch");
        assert!(continued.launch_info().shareable);
        let restarted = init_from_permalink(&continued.permalink_string().unwrap()).unwrap();
        assert_same_world(&continued, &restarted);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gallery_reset_and_family_change_discard_obsolete_reference_metadata() {
        let config = crate::base_config_for_scenario("meadow");
        let reference = scriptbots_core::permalink::Permalink {
            scenario_id: "meadow".to_owned(),
            seed: 59,
            knob_diff: Vec::new(),
            config_digest: crate::config_digest_with_diff(
                &serde_json::to_value(&config).unwrap(),
                &[],
            ),
            build: crate::canonical_build_link(),
        }
        .to_url_string();
        let gallery = init_from_permalink(&reference).unwrap();
        assert_eq!(gallery.permalink_string().unwrap(), reference);
        assert_eq!(gallery.launch_info().kind, "gallery_reference");
        assert_eq!(gallery.launch_info().initial_population, None);
        assert_eq!(
            check_build_match_identity(&reference)
                .unwrap()
                .identity_scope,
            "legacy_gallery_reference"
        );
        for patch in [
            vec![("world_width".to_owned(), 201.0)],
            vec![("world_width".to_owned(), 4100.0)],
            vec![("population_minimum".to_owned(), 50_001.0)],
        ] {
            assert!(
                gallery.fork_patch(&patch).is_err(),
                "legacy variants must be loadable before being offered"
            );
        }
        gallery.register_brain_js("assembly").unwrap();
        let changed = gallery.permalink_string().unwrap();
        assert!(changed.starts_with(PREFIX));
        assert_eq!(
            gallery.launch_info().brain_preset.as_deref(),
            Some("assembly")
        );
        assert_eq!(gallery.launch_info().seed.as_deref(), Some("59"));
        let restored = init_from_permalink(&changed).unwrap();
        assert_same_world(&gallery, &restored);
        assert!(gallery.register_brain_js("unsupported").is_err());
        assert_eq!(gallery.permalink_string().unwrap(), changed);

        let reset = init_from_permalink(&reference).unwrap();
        reset.inner.borrow_mut().reset(Some(61)).unwrap();
        let recipe = BrowserLaunch::decode(&reset.permalink_string().unwrap()).unwrap();
        assert_eq!(recipe.seed, 61);
        assert_eq!(recipe.config.rng_seed, Some(61));
        assert_eq!(recipe.config.population_minimum, config.population_minimum);
        assert_eq!(recipe.initial_population, 0);
        assert_eq!(recipe.default_brain, None);
        assert_eq!(reset.launch_info().kind, "browser_launch");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn launch_decoder_rejects_corruption_ambiguity_and_oversized_inputs() {
        let launch = recipe();
        let valid = launch.encode().unwrap();
        assert_eq!(
            BrowserLaunch::decode(&valid).unwrap().encode().unwrap(),
            valid
        );
        let mut corrupted = valid.clone();
        let last = corrupted.pop().unwrap();
        corrupted.push(if last == '0' { '1' } else { '0' });
        assert!(
            BrowserLaunch::decode(&corrupted)
                .unwrap_err()
                .contains("checksum")
        );
        assert!(BrowserLaunch::decode(&format!("{valid}.trailing")).is_err());
        assert!(BrowserLaunch::decode(&format!("{PREFIX}{}", "A".repeat(MAX_LINK_BYTES))).is_err());
        assert!(BrowserLaunch::decode(&encode_payload(&vec![0; MAX_PAYLOAD_BYTES + 1])).is_err());
        for invalid in ["A", "AB", "AAB", "AA=", "AA+", "AA/"] {
            assert!(decode_base64(invalid).is_err(), "reject {invalid}");
        }
        let json = serde_json::to_string(&launch).unwrap();
        for ambiguous in [
            json.replacen(
                "\"schema_version\":1",
                "\"schema_version\":1,\"schema_version\":1",
                1,
            ),
            json.replacen(
                "\"schema_version\":1",
                "\"unknown\":1,\"schema_version\":1",
                1,
            ),
            json.replacen("\"neuroflow\":{", "\"neuroflow\":{\"unknown\":1,", 1),
            json.replacen("\"neuroflow\":{", "\"neuroflow\":{\"enabled\":false,", 1),
            format!(" {json}"),
        ] {
            assert_ne!(ambiguous, json);
            assert!(BrowserLaunch::decode(&encode_payload(ambiguous.as_bytes())).is_err());
        }
        let mut changed = launch.clone();
        changed.schema_version = 2;
        assert!(
            BrowserLaunch::decode(&encode_payload(&serde_json::to_vec(&changed).unwrap())).is_err()
        );
        changed = launch.clone();
        changed.founder_recipe_version = 2;
        assert!(
            BrowserLaunch::decode(&encode_payload(&serde_json::to_vec(&changed).unwrap())).is_err()
        );
        changed = launch;
        changed.seed += 1;
        assert!(
            BrowserLaunch::decode(&encode_payload(&serde_json::to_vec(&changed).unwrap())).is_err()
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn browser_admission_checks_core_invariants_and_allocation_bounds() {
        let baseline = recipe();
        let mutations: [fn(&mut BrowserLaunch); 9] = [
            |launch: &mut BrowserLaunch| launch.initial_population = 50_001,
            |launch: &mut BrowserLaunch| launch.config.population_minimum = 50_001,
            |launch: &mut BrowserLaunch| launch.config.population_spawn_count = 50_001,
            |launch: &mut BrowserLaunch| {
                launch.config.world_width = 2048;
                launch.config.world_height = 2048;
                launch.config.food_cell_size = 1;
            },
            |launch: &mut BrowserLaunch| launch.config.food_max = -1.0,
            |launch: &mut BrowserLaunch| launch.config.history_capacity = MAX_HISTORY + 1,
            |launch: &mut BrowserLaunch| launch.config.narrative_capacity = MAX_HISTORY + 1,
            |launch: &mut BrowserLaunch| launch.config.archive_max_bytes = 64 * 1024 * 1024 + 1,
            |launch: &mut BrowserLaunch| launch.config.neuroflow.hidden_layers = vec![1; 1_000],
        ];
        for mutate in mutations {
            let mut changed = baseline.clone();
            mutate(&mut changed);
            let hostile = encode_payload(&serde_json::to_vec(&changed).unwrap());
            assert!(BrowserLaunch::decode(&hostile).is_err());
            assert!(init_from_permalink(&hostile).is_err());
        }
        let original = handle(fresh(Some(BrainPreset::Mlp), SeedStrategy::None, Some(67)));
        let before = original
            .inner
            .borrow()
            .core
            .world()
            .world_digest_v1()
            .unwrap();
        {
            let mut simulation = original.inner.borrow_mut();
            let launch = simulation.spec.launch.as_mut().unwrap();
            launch.build.rustflags = Some("x".repeat(MAX_PAYLOAD_BYTES));
        }
        assert!(!original.launch_info().shareable);
        assert!(original.permalink_string().is_err());
        assert_eq!(
            original
                .inner
                .borrow()
                .core
                .world()
                .world_digest_v1()
                .unwrap(),
            before
        );
        original.tick_snapshot(1).unwrap();
        assert!(
            original.inner.borrow_mut().checkpoint().is_ok(),
            "URL admission cannot disable the running owner or checkpoint route"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn build_comparison_is_about_captured_inputs_and_missing_evidence_is_explicit() {
        let local = BrowserBuildInputs::current();
        assert!(
            local
                .source_inventory
                .iter()
                .any(|input| input.path == "crates/scriptbots-core/src/lib.rs")
        );
        assert!(
            local
                .source_inventory
                .iter()
                .any(|input| input.path == "crates/scriptbots-web/src/launch.rs")
        );
        let expected = if local.complete() {
            "recorded_match"
        } else {
            "unavailable"
        };
        assert_eq!(local.comparison(&local), expected);
        let launch = recipe();
        let assessment = check_build_match_identity(&launch.encode().unwrap()).unwrap();
        assert_eq!(assessment.status, expected);
        assert_eq!(assessment.identity_scope, "recorded_browser_build_inputs");
        assert!(assessment.evidence.contains("noncryptographic"));
        let mutations: [fn(&mut BrowserBuildInputs); 4] = [
            |inputs: &mut BrowserBuildInputs| inputs.target = Some("another-target".to_owned()),
            |inputs: &mut BrowserBuildInputs| inputs.rustflags = Some("different-flags".to_owned()),
            |inputs: &mut BrowserBuildInputs| inputs.compiler = Some("another compiler".to_owned()),
            |inputs: &mut BrowserBuildInputs| {
                inputs.source_inventory[0].fingerprint = "fnv1a64:0000000000000000".to_owned()
            },
        ];
        for mutate in mutations {
            let mut changed = local.clone();
            mutate(&mut changed);
            assert_eq!(
                changed.comparison(&local),
                if local.complete() {
                    "mismatch"
                } else {
                    "unavailable"
                }
            );
        }
        let mut missing = local;
        missing.compiler = None;
        assert_eq!(missing.comparison(&missing), "unavailable");
        missing.validate().unwrap();
    }
}
