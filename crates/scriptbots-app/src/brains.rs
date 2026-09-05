//! Canonical production brain installer and discovery for ScriptBots applications.
//!
//! Bead `bd-16g.13.4.2`. Moves the brain installation and preset dispatch logic into a
//! reusable app-owned production entry point consumed by shipped startup, servers, and
//! integration tests.

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use scriptbots_brain::{
    assembly::{AssemblyBrain, AssemblyFamilyAdapter},
    dwraon::{DwraonBrain, DwraonFamilyAdapter},
    mlp::{MlpBrain, MlpBrainFamily},
};
#[cfg(feature = "brain-ft")]
use scriptbots_brain_ml::{FT_BRAIN_KIND, FtBrainFamily};
#[cfg(feature = "neuro")]
use scriptbots_brain_neuro::{NeuroflowBrain, NeuroflowBrainConfig};
use scriptbots_core::{
    BrainProtocolError, BrainRegistryHereditySnapshotV1, ScriptBotsConfig, WorldState,
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
    /// Families registered for explicit selection but withheld from default populations,
    /// with the reason label, so the exclusion is inspectable rather than folklore.
    pub withheld: Vec<(String, u64)>,
}

impl InstalledBrains {
    /// How many families are REGISTERED — eligible plus withheld.
    ///
    /// Registration and population-eligibility are different questions: a withheld family is
    /// still registered (can be bound explicitly by an experiment that genuinely wants it),
    /// but may not found a population until it implements the versioned protocol.
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
                let mut neuro_settings = world.config().neuroflow.clone();
                neuro_settings.enabled = true;
                let adapter = NeuroflowBrainConfig::from_settings(&neuro_settings);
                adapter
                    .validate()
                    .context("failed to validate configured NeuroFlow brain")?;
                let key = NeuroflowBrain::register(world, adapter)
                    .context("failed to register configured NeuroFlow brain")?;
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
            "NeuroFlow remains available as an explicitly selected legacy runner, but it \
             has no versioned genome/evaluator-state protocol codec and is WITHHELD from \
             the founding population. Admitting it would reintroduce an opaque hereditary \
             state beside the canonical protocol families."
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
            "SOME BRAIN FAMILIES ARE WITHHELD FROM THE FOUNDING POPULATION because they do \
             not implement the versioned genome/evaluator-state protocol. They remain registered \
             and can still be selected explicitly, but no founder will be seeded with an opaque \
             legacy hereditary state."
        );
    }

    Ok(installed)
}
