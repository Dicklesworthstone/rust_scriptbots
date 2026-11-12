//! Canonical production brain installer and discovery for ScriptBots applications.
//!
//! Bead `bd-16g.13.4.2`. Moves the brain installation and preset dispatch logic into a
//! reusable app-owned production entry point consumed by shipped startup, servers, and
//! integration tests.

#[cfg(any(not(feature = "brain-ft"), not(feature = "neuro")))]
use anyhow::bail;
use anyhow::{Context, Result};
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
    BrainProtocolError, BrainRegistry, BrainRegistryHereditySnapshotV1, ScriptBotsConfig,
    WorldState,
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

#[cfg(test)]
mod tests {
    use super::*;

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
