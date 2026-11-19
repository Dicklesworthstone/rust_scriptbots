//! Canonical production brain installer and discovery for ScriptBots applications.
//!
//! Bead `bd-16g.13.4.2`. Moves the brain installation and preset dispatch logic into a
//! reusable app-owned production entry point consumed by shipped startup, servers, and
//! integration tests.

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use scriptbots_brain::{
    assembly::{AssemblyBrain, AssemblyFamilyAdapter},
    dwraon::{DwraonBrain, DwraonFamilyAdapter},
    mlp::{MlpBrain, MlpBrainFamily},
};
#[cfg(feature = "brain-ft")]
use scriptbots_brain_ml::{FT_BRAIN_KIND, FtBrainFamily};
#[cfg(feature = "neuro")]
use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig, NeuroflowFamilyAdapter};
use scriptbots_core::{
    BrainFamilyAdapter, BrainProtocolError, BrainRegistry, BrainRegistryHereditySnapshotV1,
    CheckpointBrainRegistryRequirementsV1, ScriptBotsConfig, WorldCheckpointV1, WorldState,
};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Selectable brain architecture presets for world initialization.
#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrainPreset {
    Mixed,
    Mlp,
    Dwraon,
    Assembly,
    Ft,
    Neuro,
}

impl BrainPreset {
    /// Return the canonical string identifier for the preset.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::Mlp => "mlp",
            Self::Dwraon => "dwraon",
            Self::Assembly => "assembly",
            Self::Ft => "ft",
            Self::Neuro => "neuro",
        }
    }
}

/// Founder-preset recipes recognized after exact saved-registry reconstruction.
///
/// Saved registries may contain other supported family combinations. Recognition
/// must still match the complete registry, and never relabels those combinations
/// as a current founder preset. Feature-disabled recipes are never constructed.
pub(crate) const CHECKPOINT_BRAIN_PRESETS: &[Option<BrainPreset>] = &[
    None,
    Some(BrainPreset::Mlp),
    Some(BrainPreset::Dwraon),
    Some(BrainPreset::Assembly),
    Some(BrainPreset::Mixed),
    #[cfg(feature = "brain-ft")]
    Some(BrainPreset::Ft),
    #[cfg(feature = "neuro")]
    Some(BrainPreset::Neuro),
];

/// Registered brain families, split by whether they may found a population.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledBrains {
    /// Versioned protocol families admitted to seed the founding population.
    pub population: Vec<u64>,
    /// Legacy registrations withheld from default populations, with an inspectable reason label.
    /// A family can have both a legacy registration and a separate versioned adapter; admission
    /// describes the registration actually installed by this preset.
    pub withheld: Vec<(String, u64)>,
}

impl InstalledBrains {
    /// How many families are REGISTERED — eligible plus withheld.
    ///
    /// Registration and population eligibility are different questions: a withheld registration
    /// can still be bound explicitly, but may not found a population without a versioned protocol.
    #[must_use]
    pub fn registered(&self) -> usize {
        self.population.len() + self.withheld.len()
    }

    /// Read-only slice of admitted founding population registry keys.
    #[must_use]
    pub fn population(&self) -> &[u64] {
        &self.population
    }

    /// Read-only slice of registered but withheld families.
    #[must_use]
    pub fn withheld(&self) -> &[(String, u64)] {
        &self.withheld
    }

    /// Whether no brain families were registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.population.is_empty() && self.withheld.is_empty()
    }

    /// Query the canonical heredity capability snapshot from the live world's registry.
    ///
    /// # Errors
    ///
    /// Returns a typed protocol error if any registered brain descriptor is malformed or invalid.
    pub fn heredity_capabilities(
        &self,
        world: &WorldState,
    ) -> Result<BrainRegistryHereditySnapshotV1, BrainProtocolError> {
        world.brain_registry().heredity_capabilities()
    }
}

#[cfg(feature = "neuro")]
/// Validates the configured NeuroFlow brain settings if enabled.
///
/// # Errors
///
/// Returns an error if the configured layer dimensions or parameters are invalid.
pub fn validated_neuroflow_config(
    config: &ScriptBotsConfig,
) -> Result<Option<NeuroflowBrainConfig>> {
    if !config.neuroflow.enabled {
        return Ok(None);
    }
    let adapter = NeuroflowBrainConfig::from_settings(&config.neuroflow);
    adapter
        .validate()
        .context("failed to validate configured NeuroFlow brain")?;
    Ok(Some(adapter))
}

/// Installs the brain family adapters for the given preset into the target world.
///
/// The explicit `neuro` preset installs the configured f64 genome/evaluator-state protocol,
/// regardless of the Mixed-preset enable toggle. Mixed preserves its existing founding roster
/// and, when enabled, its withheld legacy NeuroFlow registration.
///
/// # Errors
///
/// Returns an error if registration fails or if a requested non-default feature is missing.
pub fn install_brains(world: &mut WorldState, preset: BrainPreset) -> Result<InstalledBrains> {
    #[cfg(feature = "neuro")]
    let neuro_config = if preset == BrainPreset::Mixed {
        validated_neuroflow_config(world.config())?
    } else {
        None
    };

    #[cfg(feature = "neuro")]
    let mut withheld = Vec::new();
    #[cfg(not(feature = "neuro"))]
    let withheld = Vec::new();
    let mut population = Vec::new();

    // Founding-population admission is structural: every eligible entry must own a versioned
    // genome codec, evaluator-state codec, offspring-state policy, and evaluator constructor.
    let register_mlp = |world: &mut WorldState| {
        world
            .register_brain_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
            .context("failed to register the versioned MLP brain family")
    };
    let register_dwraon = |world: &mut WorldState| {
        world
            .register_brain_family(
                DwraonBrain::KIND.as_str(),
                Box::new(DwraonFamilyAdapter::default()),
            )
            .context("failed to register the versioned DWRAON brain family")
    };
    let register_assembly = |world: &mut WorldState| {
        let assembly = AssemblyFamilyAdapter::new()
            .context("failed to construct the versioned Assembly brain family")?;
        world
            .register_brain_family(AssemblyBrain::KIND.as_str(), Box::new(assembly))
            .context("failed to register the versioned Assembly brain family")
    };
    #[cfg(feature = "brain-ft")]
    let register_ft = |world: &mut WorldState| {
        world
            .register_brain_family(FT_BRAIN_KIND, Box::new(FtBrainFamily::default()))
            .context("failed to register the versioned Frankentorch brain family")
    };

    match preset {
        BrainPreset::Mixed => {
            population.push(register_mlp(world)?);
            population.push(register_dwraon(world)?);
            population.push(register_assembly(world)?);
            #[cfg(feature = "brain-ft")]
            population.push(register_ft(world)?);
        }
        BrainPreset::Mlp => population.push(register_mlp(world)?),
        BrainPreset::Dwraon => population.push(register_dwraon(world)?),
        BrainPreset::Assembly => population.push(register_assembly(world)?),
        BrainPreset::Ft => {
            #[cfg(feature = "brain-ft")]
            population.push(register_ft(world)?);
            #[cfg(not(feature = "brain-ft"))]
            bail!(
                "brain preset `ft` requires a scriptbots-app build with the non-default \
                 `brain-ft` feature"
            );
        }
        BrainPreset::Neuro => {
            #[cfg(feature = "neuro")]
            {
                let adapter = NeuroflowFamilyAdapter::new(NeuroflowBrainConfig::from_settings(
                    &world.config().neuroflow,
                ))
                .context("failed to construct the configured versioned NeuroFlow brain family")?;
                let key = world
                    .register_brain_family(NeuroflowBrain::KIND.as_str(), Box::new(adapter))
                    .context("failed to register the versioned NeuroFlow brain family")?;
                population.push(key);
            }
            #[cfg(not(feature = "neuro"))]
            bail!("brain preset `neuro` requires a scriptbots-app build with the `neuro` feature");
        }
    }

    #[cfg(feature = "neuro")]
    if preset == BrainPreset::Mixed
        && let Some(config) = neuro_config
    {
        let key = NeuroflowBrain::register(world, config)
            .context("failed to register configured NeuroFlow brain")?;
        let label = world
            .brain_registry()
            .kind(key)
            .unwrap_or("neuroflow")
            .to_owned();
        warn!(
            brain = %label,
            key,
            "Mixed retains its legacy NeuroFlow registration, withheld from the founding \
             population. Select the explicit neuro preset to use the versioned f64 genome \
             and evaluator-state protocol."
        );
        withheld.push((label, key));
    }

    let installed = InstalledBrains {
        population,
        withheld,
    };

    if installed.withheld.is_empty() {
        info!(
            registered = installed.registered(),
            eligible = installed.population.len(),
            "every registered brain family implements the versioned genome/evaluator protocol; all are eligible to found the population"
        );
    } else {
        let withheld_labels: Vec<&str> = installed
            .withheld
            .iter()
            .map(|(label, _)| label.as_str())
            .collect();
        warn!(
            registered = installed.registered(),
            eligible = installed.population.len(),
            withheld = ?withheld_labels,
            "some installed registrations use legacy runners without a versioned \
             genome/evaluator-state protocol and are withheld from the founding population"
        );
    }

    Ok(installed)
}

/// Construct a registry of versioned families using their default constructor settings.
///
/// Use [`create_brain_registry_for_config`] when restoring a configured NeuroFlow run.
///
/// # Errors
///
/// Returns an error if adapter registration fails or if a requested non-default feature is missing.
pub fn create_brain_registry(preset: BrainPreset) -> Result<BrainRegistry> {
    let mut registry = BrainRegistry::new();
    match preset {
        BrainPreset::Mixed => {
            registry
                .register_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
                .context("failed to register versioned MLP brain family")?;
            registry
                .register_family(
                    DwraonBrain::KIND.as_str(),
                    Box::new(DwraonFamilyAdapter::default()),
                )
                .context("failed to register versioned DWRAON brain family")?;
            let assembly = AssemblyFamilyAdapter::new()
                .context("failed to construct versioned Assembly brain family")?;
            registry
                .register_family(AssemblyBrain::KIND.as_str(), Box::new(assembly))
                .context("failed to register versioned Assembly brain family")?;
            #[cfg(feature = "brain-ft")]
            {
                registry
                    .register_family(FT_BRAIN_KIND, Box::new(FtBrainFamily::default()))
                    .context("failed to register versioned Frankentorch brain family")?;
            }
        }
        BrainPreset::Mlp => {
            registry
                .register_family(MlpBrain::KIND.as_str(), Box::new(MlpBrainFamily::new()))
                .context("failed to register versioned MLP brain family")?;
        }
        BrainPreset::Dwraon => {
            registry
                .register_family(
                    DwraonBrain::KIND.as_str(),
                    Box::new(DwraonFamilyAdapter::default()),
                )
                .context("failed to register versioned DWRAON brain family")?;
        }
        BrainPreset::Assembly => {
            let assembly = AssemblyFamilyAdapter::new()
                .context("failed to construct versioned Assembly brain family")?;
            registry
                .register_family(AssemblyBrain::KIND.as_str(), Box::new(assembly))
                .context("failed to register versioned Assembly brain family")?;
        }
        BrainPreset::Ft => {
            #[cfg(feature = "brain-ft")]
            {
                registry
                    .register_family(FT_BRAIN_KIND, Box::new(FtBrainFamily::default()))
                    .context("failed to register versioned Frankentorch brain family")?;
            }
            #[cfg(not(feature = "brain-ft"))]
            bail!(
                "brain preset `ft` requires a scriptbots-app build with the non-default `brain-ft` feature"
            );
        }
        BrainPreset::Neuro => {
            #[cfg(feature = "neuro")]
            registry
                .register_family(
                    NeuroflowBrain::KIND.as_str(),
                    Box::new(NeuroflowFamilyAdapter::default()),
                )
                .context("failed to register versioned NeuroFlow brain family")?;
            #[cfg(not(feature = "neuro"))]
            bail!("brain preset `neuro` requires a scriptbots-app build with the `neuro` feature");
        }
    }
    Ok(registry)
}

/// Construct the exact registry selected by a preset and its constructor configuration.
///
/// Explicit NeuroFlow uses its versioned f64 protocol. Mixed retains its configured legacy
/// NeuroFlow registration, which remains withheld from founders.
///
/// `config` must describe the settings used when the registry was originally installed.
/// A raw checkpoint stores the adapter identity but not a previous constructor recipe; after
/// a live edit to NeuroFlow settings, passing the edited configuration correctly causes a
/// registry mismatch at restore. Callers with launch provenance should supply that recipe.
///
/// # Errors
///
/// Returns an error if adapter registration fails or if a requested non-default feature is missing.
pub fn create_brain_registry_for_config(
    preset: BrainPreset,
    config: &ScriptBotsConfig,
) -> Result<BrainRegistry> {
    #[cfg(feature = "neuro")]
    if preset == BrainPreset::Neuro {
        let adapter =
            NeuroflowFamilyAdapter::new(NeuroflowBrainConfig::from_settings(&config.neuroflow))
                .context("failed to construct the configured versioned NeuroFlow brain family")?;
        let mut registry = BrainRegistry::new();
        registry
            .register_family(NeuroflowBrain::KIND.as_str(), Box::new(adapter))
            .context("failed to register versioned NeuroFlow brain family")?;
        return Ok(registry);
    }
    #[cfg(not(feature = "neuro"))]
    let _ = config;

    let registry = create_brain_registry(preset)?;
    #[cfg(feature = "neuro")]
    let mut registry = registry;
    #[cfg(feature = "neuro")]
    if preset == BrainPreset::Mixed
        && let Some(neuro_cfg) = validated_neuroflow_config(config)?
    {
        let neuro_cfg = std::sync::Arc::new(neuro_cfg);
        registry.register_with_heredity_exclusion(
            scriptbots_brain_neuro::NeuroflowBrain::KIND.as_str(),
            scriptbots_core::BrainHeredityExclusionV1::NoVersionedGenomeProtocol,
            move |rng| {
                scriptbots_brain_neuro::NeuroflowBrain::runner((*neuro_cfg).clone(), rng).map_err(
                    |source| {
                        scriptbots_core::BrainSpawnError::new(
                            scriptbots_brain_neuro::NeuroflowBrain::KIND.as_str(),
                            source,
                        )
                    },
                )
            },
        );
    }
    Ok(registry)
}

/// Reconstruct the saved registry from the application's compiled adapter catalog.
///
/// Registry keys and order come from the complete checkpoint, including families
/// with no surviving agents. Enabling another optional family does not add it to
/// an existing world. Labels alone never authorize an adapter: protocol family,
/// constructor-dependent semantic identity and the core restore checks must agree.
///
/// The shipped constructors allocate contiguous keys. Retired-key layouts and
/// unknown or duplicated families remain explicit refusals.
pub(crate) fn create_brain_registry_for_checkpoint(
    checkpoint: &WorldCheckpointV1,
) -> Result<BrainRegistry> {
    create_required_checkpoint_registry(&checkpoint.required_brain_registry(), checkpoint.config())
}

fn compiled_checkpoint_adapter(
    kind: &str,
    config: &ScriptBotsConfig,
) -> Result<Box<dyn BrainFamilyAdapter>> {
    #[cfg(not(feature = "neuro"))]
    let _ = config;
    match kind {
        kind if kind == MlpBrain::KIND.as_str() => Ok(Box::new(MlpBrainFamily::new())),
        kind if kind == DwraonBrain::KIND.as_str() => Ok(Box::new(DwraonFamilyAdapter::default())),
        kind if kind == AssemblyBrain::KIND.as_str() => Ok(Box::new(AssemblyFamilyAdapter::new()?)),
        #[cfg(feature = "brain-ft")]
        FT_BRAIN_KIND => Ok(Box::new(FtBrainFamily::default())),
        #[cfg(feature = "neuro")]
        kind if kind == NeuroflowBrain::KIND.as_str() => Ok(Box::new(
            NeuroflowFamilyAdapter::new(NeuroflowBrainConfig::from_settings(&config.neuroflow))?,
        )),
        _ => bail!("checkpoint brain family {kind:?} is not available in this application build"),
    }
}

fn create_required_checkpoint_registry(
    required: &CheckpointBrainRegistryRequirementsV1,
    config: &ScriptBotsConfig,
) -> Result<BrainRegistry> {
    let family_limit = CHECKPOINT_BRAIN_PRESETS
        .iter()
        .flatten()
        .filter(|preset| **preset != BrainPreset::Mixed)
        .count();
    ensure!(
        required.entries.len() <= family_limit
            && required.next_key == u64::try_from(required.entries.len())?,
        "checkpoint brain registry has an unsupported allocation cursor or family count"
    );
    let mut registry = BrainRegistry::new();
    let mut kinds = std::collections::BTreeSet::new();
    for (index, entry) in required.entries.iter().enumerate() {
        ensure!(
            entry.key == u64::try_from(index)? && kinds.insert(entry.kind.as_str()),
            "checkpoint brain registry has a retired, reordered or duplicate key/family"
        );
        ensure!(
            entry.factory_state_digest.is_none(),
            "checkpoint brain family {:?} requires an unavailable legacy factory identity",
            entry.kind
        );
        #[cfg(feature = "neuro")]
        if entry.kind == NeuroflowBrain::KIND.as_str() && entry.protocol_family.is_none() {
            ensure!(
                entry.adapter_identity.is_none(),
                "legacy NeuroFlow checkpoint entry cannot claim a protocol adapter"
            );
            let settings = validated_neuroflow_config(config)?
                .context("checkpoint requires the original enabled legacy NeuroFlow recipe")?;
            let settings = std::sync::Arc::new(settings);
            let key = registry.register_with_heredity_exclusion(
                NeuroflowBrain::KIND.as_str(),
                scriptbots_core::BrainHeredityExclusionV1::NoVersionedGenomeProtocol,
                move |rng| {
                    NeuroflowBrain::runner((*settings).clone(), rng).map_err(|source| {
                        scriptbots_core::BrainSpawnError::new(NeuroflowBrain::KIND.as_str(), source)
                    })
                },
            );
            ensure!(key == entry.key, "legacy checkpoint registry key changed");
            continue;
        }
        let adapter = compiled_checkpoint_adapter(&entry.kind, config)?;
        ensure!(
            entry.protocol_family.as_ref() == Some(adapter.family_id())
                && entry.adapter_identity == Some(adapter.adapter_identity()),
            "checkpoint brain family {:?} does not match the compiled protocol and constructor identity",
            entry.kind
        );
        let key = registry.register_family(entry.kind.clone(), adapter)?;
        ensure!(key == entry.key, "checkpoint registry key changed");
    }
    Ok(registry)
}

/// Recognize a current founder preset only when its complete registry matches.
///
/// A supported saved combination can be replayable without being a founder
/// preset. Callers that require a preset (such as the current study-plan schema)
/// must keep that distinction rather than relabel every combination as Mixed.
pub(crate) fn checkpoint_brain_preset(checkpoint: &WorldCheckpointV1) -> Option<BrainPreset> {
    let required = checkpoint.required_brain_registry();
    for &preset in CHECKPOINT_BRAIN_PRESETS.iter().flatten() {
        let Ok(registry) = create_brain_registry_for_config(preset, checkpoint.config()) else {
            continue;
        };
        let descriptors = registry.descriptors();
        if u64::try_from(descriptors.len()).ok() == Some(required.next_key)
            && descriptors.len() == required.entries.len()
            && required
                .entries
                .iter()
                .zip(&descriptors)
                .all(|(entry, (key, kind))| {
                    entry.key == *key
                        && entry.kind == *kind
                        && entry.factory_state_digest.is_none()
                        && entry.adapter_identity == registry.adapter_identity(*key)
                        && entry.protocol_family.as_ref()
                            == registry.family(*key).map(|family| family.family_id())
                })
        {
            return Some(preset);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::BrainFamilyCodec as _;

    fn ordered_checkpoint_world(kinds: &[&str], living_families: usize) -> WorldState {
        let config = ScriptBotsConfig {
            rng_seed: Some(42_103),
            persistence_interval: 0,
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            initial_food: 0.0,
            food_growth_rate: 0.0,
            food_respawn_interval: 0,
            closed: true,
            population_minimum: 0,
            reproduction_attempt_chance: 0.0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config.clone()).unwrap();
        for (index, &kind) in kinds.iter().enumerate() {
            let key = world
                .register_brain_family(
                    kind.to_owned(),
                    compiled_checkpoint_adapter(kind, &config).unwrap(),
                )
                .unwrap();
            let agent = world
                .try_spawn_agent(scriptbots_core::AgentData {
                    health: if index < living_families { 1.0 } else { 0.0 },
                    ..scriptbots_core::AgentData::default()
                })
                .unwrap();
            assert!(world.bind_agent_brain(agent, key).unwrap());
        }
        for _ in 0..3 {
            world.step().unwrap();
        }
        world
    }

    #[test]
    fn saved_registry_restores_order_subsets_and_extinct_factories() {
        let combinations = [
            vec![],
            vec![MlpBrain::KIND.as_str()],
            vec![DwraonBrain::KIND.as_str(), MlpBrain::KIND.as_str()],
            vec![
                AssemblyBrain::KIND.as_str(),
                MlpBrain::KIND.as_str(),
                DwraonBrain::KIND.as_str(),
            ],
        ];
        for kinds in combinations {
            for living in [0, kinds.len()] {
                let mut original = ordered_checkpoint_world(&kinds, living);
                if living == 0 {
                    assert_eq!(
                        original.agent_count(),
                        0,
                        "all founder families became extinct"
                    );
                }
                let checkpoint = original.checkpoint_v1().unwrap();
                let bytes = checkpoint.encode().unwrap();
                let registry = create_brain_registry_for_checkpoint(&checkpoint).unwrap();
                assert_eq!(
                    registry.descriptors(),
                    original.brain_registry().descriptors()
                );
                let mut restored = WorldState::restore_checkpoint_v1(&checkpoint, registry).unwrap();
                assert_eq!(
                    restored.checkpoint_v1().unwrap().required_brain_registry(),
                    checkpoint.required_brain_registry()
                );
                // Empty/extinct populations still need their exact factories to bind
                // future brains without changing the source's keyed RNG streams.
                for key in 0..u64::try_from(kinds.len()).unwrap() {
                    let first = original
                        .try_spawn_agent(scriptbots_core::AgentData::default())
                        .unwrap();
                    let second = restored
                        .try_spawn_agent(scriptbots_core::AgentData::default())
                        .unwrap();
                    assert!(original.bind_agent_brain(first, key).unwrap());
                    assert!(restored.bind_agent_brain(second, key).unwrap());
                }
                for _ in 0..4 {
                    original.step().unwrap();
                    restored.step().unwrap();
                    assert_eq!(
                        original.world_digest_v1().unwrap(),
                        restored.world_digest_v1().unwrap()
                    );
                    assert_eq!(
                        original.checkpoint_v1().unwrap().encode().unwrap(),
                        restored.checkpoint_v1().unwrap().encode().unwrap()
                    );
                }
                assert_eq!(checkpoint.encode().unwrap(), bytes);
            }
        }
    }

    #[test]
    fn saved_registry_rejects_unknown_duplicate_and_substituted_adapters() {
        let world = ordered_checkpoint_world(&[MlpBrain::KIND.as_str()], 1);
        let checkpoint = world.checkpoint_v1().unwrap();
        let required = checkpoint.required_brain_registry();
        let mut unknown = required.clone();
        unknown.entries[0].kind = "unknown.family".into();
        let mut duplicate = required.clone();
        duplicate.entries.push(duplicate.entries[0].clone());
        duplicate.entries[1].key = 1;
        duplicate.next_key = 2;
        let mut cursor = required.clone();
        cursor.next_key = u64::MAX;
        let mut retired = required.clone();
        retired.entries[0].key = 1;
        let mut semantic = required.clone();
        semantic.entries[0].adapter_identity = Some(
            scriptbots_core::BrainAdapterIdentityV1::from_semantic_descriptor(
                semantic.entries[0].protocol_family.as_ref().unwrap(),
                99,
                b"changed",
            ),
        );
        let mut protocol = required.clone();
        protocol.entries[0].protocol_family =
            Some(DwraonFamilyAdapter::default().family_id().clone());
        let mut legacy = required.clone();
        legacy.entries[0].protocol_family = None;
        legacy.entries[0].adapter_identity = None;
        let mut factory = required.clone();
        factory.entries[0].factory_state_digest = Some(7);
        for invalid in [
            unknown, duplicate, cursor, retired, semantic, protocol, legacy, factory,
        ] {
            assert!(create_required_checkpoint_registry(&invalid, world.config()).is_err());
        }
        assert!(create_required_checkpoint_registry(&required, world.config()).is_ok());
    }

    #[test]
    fn saved_combinations_are_not_mislabeled_as_founder_presets() {
        let world =
            ordered_checkpoint_world(&[DwraonBrain::KIND.as_str(), MlpBrain::KIND.as_str()], 2);
        let checkpoint = world.checkpoint_v1().unwrap();
        assert!(create_brain_registry_for_checkpoint(&checkpoint).is_ok());
        assert_eq!(checkpoint_brain_preset(&checkpoint), None);
        let world = ordered_checkpoint_world(&[MlpBrain::KIND.as_str()], 1);
        assert_eq!(
            checkpoint_brain_preset(&world.checkpoint_v1().unwrap()),
            Some(BrainPreset::Mlp)
        );
    }

    #[cfg(feature = "brain-ft")]
    #[test]
    fn enabling_ft_does_not_add_a_family_to_a_saved_three_family_world() {
        let mut original = ordered_checkpoint_world(
            &[
                MlpBrain::KIND.as_str(),
                DwraonBrain::KIND.as_str(),
                AssemblyBrain::KIND.as_str(),
            ],
            3,
        );
        let checkpoint = original.checkpoint_v1().unwrap();
        assert!(matches!(
            WorldState::restore_checkpoint_v1(
                &checkpoint,
                create_brain_registry(BrainPreset::Mixed).unwrap(),
            ),
            Err(scriptbots_core::WorldCheckpointError::RegistryMismatch { .. })
        ));
        let registry = create_brain_registry_for_checkpoint(&checkpoint).unwrap();
        let mut restored = WorldState::restore_checkpoint_v1(&checkpoint, registry).unwrap();
        original.step().unwrap();
        restored.step().unwrap();
        assert_eq!(
            original.world_digest_v1().unwrap(),
            restored.world_digest_v1().unwrap()
        );
        assert_eq!(checkpoint_brain_preset(&checkpoint), None);
    }

    #[cfg(feature = "neuro")]
    fn configured_neuro() -> ScriptBotsConfig {
        let mut config = ScriptBotsConfig {
            rng_seed: Some(42_103),
            persistence_interval: 0,
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            ..ScriptBotsConfig::default()
        };
        // The explicit preset overrides only this Mixed-preset admission toggle.
        config.neuroflow.enabled = false;
        config.neuroflow.hidden_layers = vec![7, 3];
        config.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Sigmoid;
        config
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn explicit_neuro_restores_configured_protocol_and_continues_exact_science() {
        let config = configured_neuro();
        let mut world = WorldState::new(config.clone()).unwrap();
        let installed = install_brains(&mut world, BrainPreset::Neuro).unwrap();
        assert_eq!(installed.population.len(), 1);
        assert!(installed.withheld.is_empty());
        let key = installed.population[0];
        assert!(world.brain_registry().is_protocol_family(key));

        let agent = world
            .try_spawn_agent(scriptbots_core::AgentData::default())
            .unwrap();
        assert!(world.bind_agent_brain(agent, key).unwrap());
        for _ in 0..3 {
            world.step().unwrap();
        }
        assert!(world.agent_brain_genome(agent).is_some());
        assert!(world.agent_brain_evaluator_state(agent).unwrap().is_some());

        let checkpoint = world.checkpoint_v1().unwrap();
        assert!(checkpoint.source_digest().evaluator_state_covered);
        assert!(checkpoint.source_digest().factory_state_covered);
        let bytes = checkpoint.encode().unwrap();
        let decoded = scriptbots_core::WorldCheckpointV1::decode(&bytes).unwrap();
        let registry = create_brain_registry_for_config(BrainPreset::Neuro, &config).unwrap();
        let mut restored = WorldState::restore_checkpoint_v1(&decoded, registry).unwrap();
        assert_eq!(
            world.world_digest_v1().unwrap(),
            restored.world_digest_v1().unwrap()
        );
        for _ in 0..6 {
            world.step().unwrap();
            restored.step().unwrap();
            assert_eq!(
                world.world_digest_v1().unwrap(),
                restored.world_digest_v1().unwrap()
            );
            assert_eq!(
                world.checkpoint_v1().unwrap().encode().unwrap(),
                restored.checkpoint_v1().unwrap().encode().unwrap()
            );
        }
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuro_checkpoint_requires_the_original_constructor_recipe_even_without_agents() {
        let config = configured_neuro();
        let mut world = WorldState::new(config.clone()).unwrap();
        install_brains(&mut world, BrainPreset::Neuro).unwrap();
        let checkpoint = world.checkpoint_v1().unwrap();

        let defaults = create_brain_registry(BrainPreset::Neuro).unwrap();
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&checkpoint, defaults),
            Err(scriptbots_core::WorldCheckpointError::RegistryMismatch { .. })
        ));
        let mut architecture = config.clone();
        architecture.neuroflow.hidden_layers = vec![11, 5];
        let mut activation = config.clone();
        activation.neuroflow.activation = scriptbots_core::NeuroflowActivationKind::Relu;
        for changed in [architecture, activation] {
            let registry = create_brain_registry_for_config(BrainPreset::Neuro, &changed).unwrap();
            assert!(matches!(
                WorldState::restore_checkpoint_v1(&checkpoint, registry),
                Err(scriptbots_core::WorldCheckpointError::RegistryMismatch { .. })
            ));
        }
        let registry = create_brain_registry_for_config(BrainPreset::Neuro, &config).unwrap();
        let restored = WorldState::restore_checkpoint_v1(&checkpoint, registry).unwrap();
        assert_eq!(
            restored.world_digest_v1().unwrap(),
            world.world_digest_v1().unwrap()
        );

        // A live settings edit changes configuration, not the already installed constructor.
        // Empty populations must still bind the factory that will create future agents.
        let mut edited = config.clone();
        edited.neuroflow.hidden_layers = vec![11, 5];
        world.apply_config_update(edited.clone()).unwrap();
        let edited_checkpoint = world.checkpoint_v1().unwrap();
        assert!(create_brain_registry_for_checkpoint(&edited_checkpoint).is_err());
        let current_recipe = create_brain_registry_for_config(BrainPreset::Neuro, &edited).unwrap();
        assert!(matches!(
            WorldState::restore_checkpoint_v1(&edited_checkpoint, current_recipe),
            Err(scriptbots_core::WorldCheckpointError::RegistryMismatch { .. })
        ));
        let original_recipe =
            create_brain_registry_for_config(BrainPreset::Neuro, &config).unwrap();
        let resumed =
            WorldState::restore_checkpoint_v1(&edited_checkpoint, original_recipe).unwrap();
        assert_eq!(
            resumed.world_digest_v1().unwrap(),
            world.world_digest_v1().unwrap()
        );
    }

    #[cfg(feature = "neuro")]
    #[test]
    fn neuro_protocol_budget_refuses_before_registration_changes_science() {
        let mut config = configured_neuro();
        config.neuroflow.hidden_layers = vec![256, 256, 256];
        let mut world = WorldState::new(config.clone()).unwrap();
        let before = world.world_digest_v1().unwrap();
        assert!(install_brains(&mut world, BrainPreset::Neuro).is_err());
        assert_eq!(before, world.world_digest_v1().unwrap());
        assert!(create_brain_registry_for_config(BrainPreset::Neuro, &config).is_err());
    }

    #[cfg(not(feature = "neuro"))]
    #[test]
    fn neuro_preset_requires_its_compiled_feature_before_registration() {
        let config = ScriptBotsConfig {
            rng_seed: Some(42_103),
            persistence_interval: 0,
            ..ScriptBotsConfig::default()
        };
        let mut world = WorldState::new(config.clone()).unwrap();
        let before = world.world_digest_v1().unwrap();
        assert!(install_brains(&mut world, BrainPreset::Neuro).is_err());
        assert_eq!(before, world.world_digest_v1().unwrap());
        assert!(create_brain_registry(BrainPreset::Neuro).is_err());
        assert!(create_brain_registry_for_config(BrainPreset::Neuro, &config).is_err());
    }
}
