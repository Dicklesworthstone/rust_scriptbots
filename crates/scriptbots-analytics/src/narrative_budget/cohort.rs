//! Compare one fixed scenario budget across declared, distinct root seeds.
//!
//! Example specification (unqualified members use the CLI's positional database):
//! ```json
//! {"schema_version":1,
//!  "budget":{"schema_version":1,"expected_ticks":10000,"scenario_id":"meadow","rule":{"type":"quiet"}},
//!  "members":[
//!    {"run_id":"00000000000000000000000000000001","root_seed":41},
//!    {"run_id":"00000000000000000000000000000002","root_seed":42,"database":"second.sqlite"},
//!    {"run_id":"00000000000000000000000000000003","root_seed":43,"database":"third.sqlite"}]}
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use scriptbots_runtime::RunId;
use scriptbots_storage::{RunManifestRecord, StorageReader};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{NarrativeBudget, NarrativeBudgetReport, verify_narrative_budget};
use crate::AnalyticsError;

const MAX_RUNS: usize = 32;

/// A single budget and the independently specified runs and seeds to check.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NarrativeCohort {
    /// Specification format version, currently 1.
    pub schema_version: u32,
    /// Same named single-world scenario and event rules for all replicates.
    pub budget: NarrativeBudget,
    /// Three to 32 unique run/seed pairs, in report order.
    pub members: Vec<CohortMember>,
}

/// A pinned run and its independently expected root seed.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortMember {
    /// Exact retained run identity in its database.
    pub run_id: RunId,
    /// Required to equal the attributed root seed in this run's manifest.
    pub root_seed: u64,
    /// Optional database path relative to the specification file.
    pub database: Option<PathBuf>,
}

fn bad(reason: &str) -> AnalyticsError {
    AnalyticsError::BadParam {
        name: "narrative cohort".to_owned(),
        reason: reason.to_owned(),
    }
}

impl NarrativeCohort {
    /// Resolve the input database of each replicate using the same paths as verification.
    pub fn database_paths(&self, default_database: &Path, spec_directory: &Path) -> Vec<PathBuf> {
        self.members
            .iter()
            .map(|member| {
                member.database.as_ref().map_or_else(
                    || default_database.to_owned(),
                    |path| spec_directory.join(path),
                )
            })
            .collect()
    }

    fn validate(&self) -> Result<(), AnalyticsError> {
        self.budget.validate()?;
        if self.schema_version != 1 || !(3..=MAX_RUNS).contains(&self.members.len()) {
            return Err(bad("expected schema_version 1 and between 3 and 32 independent runs"));
        }
        if self.budget.scenario_id.as_ref().is_none_or(|id| id.trim().is_empty())
            || self.budget.root_seed.is_some() || self.budget.island_id.is_some()
        {
            return Err(bad("supply one named single-world scenario and declare seeds per member"));
        }
        let mut runs = BTreeSet::new();
        let mut seeds = BTreeSet::new();
        for member in &self.members {
            if member.run_id.get() == 0 || !runs.insert(member.run_id) || !seeds.insert(member.root_seed) {
                return Err(bad("run IDs must be nonzero; run IDs and root seeds must each be distinct"));
            }
        }
        Ok(())
    }
}

/// Only the root seed may vary in the normalized scientific configuration. Check
/// the recorded scenario recipe (including scheduled interventions), build, brain
/// roster, RNG protocol and worker policy explicitly. Counters and evolved state
/// legitimately differ across seeds and are verified by the per-run replay.
fn launch_digest(record: &RunManifestRecord) -> Result<Option<String>, AnalyticsError> {
    if record.manifest_schema_version != 3 || !record.reproducible {
        return Ok(None);
    }
    let raw: Value = serde_json::from_str(&record.manifest_json)?;
    let mut config: Value = serde_json::from_str(&record.normalized_config_json)?;
    let object = config.as_object_mut().ok_or_else(|| bad("expected a normalized configuration object"))?;
    if let Some(seed) = object.remove("rng_seed")
        && !seed.is_null() && seed.as_u64() != Some(record.root_seed)
    {
        return Ok(None);
    }
    let mut protocol = raw["agent_substream_protocol"].clone();
    let protocol_seed = protocol.as_object_mut().and_then(|fields| fields.remove("root_seed"));
    if protocol_seed.and_then(|value| value.as_u64()) != Some(record.root_seed) {
        return Ok(None);
    }
    let mut scenario = raw["scenario"].clone();
    let identity = scenario.as_object_mut().ok_or_else(|| bad("missing scenario identity"))?;
    // A config layer can carry the replicate seed. Compare its fully resolved
    // settings below; each original layer digest stays bound by the run manifest.
    let _ = identity.remove("ordered_config_layer_digests");
    let comparable = json!({
        "schema": raw["schema"],
        "scenario": scenario,
        "configuration_except_rng_seed": config,
        "build": raw["build"],
        "source_bundle_digest": record.source_bundle_digest,
        "rng_algorithm": record.rng_algorithm,
        "rng_version": record.rng_version,
        "agent_substream_protocol": protocol,
        "thread_policy": raw["thread_policy"],
        "sense_policy": raw["sense_policy"],
        "brain_roster": raw["brain_roster"],
    });
    let bytes = serde_json::to_vec(&comparable)?;
    Ok(Some(blake3::hash(&bytes).to_hex().to_string()))
}

/// One member's complete scientific verdict and launch-comparability checks.
#[derive(Debug, Serialize)]
pub struct CohortResult<'a> {
    /// Expected seed supplied independently of the retained manifest.
    pub expected_root_seed: u64,
    /// Digest of the common launch design, absent for unusable provenance.
    pub launch_digest: Option<String>,
    /// Extra cohort-level source failures for this run.
    pub violations: Vec<&'static str>,
    /// Complete parity, counts, coverage and per-run budget evidence.
    pub result: NarrativeBudgetReport<'a>,
}

/// Complete bounded per-run results, in specification order.
#[derive(Debug, Serialize)]
pub struct NarrativeCohortReport<'a> {
    /// Report version.
    pub schema_version: u32,
    /// Every run matched its seed, common launch design and shared budget.
    pub passed: bool,
    /// Number of distinct, attributed root seeds actually observed.
    pub distinct_observed_seeds: usize,
    /// Every individual result, including all observed divergences.
    pub members: Vec<CohortResult<'a>>,
}

/// Qualify a budget against independent saved runs. One lease per input file
/// stays held until all comparisons complete, including multiple runs in a
/// shared database. A failed budget never skips a remaining replicate.
/// This measures finite runs; it does not estimate population-wide efficacy.
pub fn verify_cohort<'a>(
    spec: &'a NarrativeCohort,
    default_database: &Path,
    spec_directory: &Path,
) -> Result<NarrativeCohortReport<'a>, AnalyticsError> {
    spec.validate()?;
    let paths = spec.database_paths(default_database, spec_directory);
    let mut sources = BTreeMap::new();
    for (member, path) in spec.members.iter().zip(&paths) {
        sources.entry(path.as_path()).or_insert(member.run_id);
    }
    let mut readers = BTreeMap::new();
    for (path, run_id) in sources {
        let text = path.to_str().ok_or_else(|| bad("database paths must be UTF-8"))?;
        readers.insert(path, StorageReader::open_finished_for_run(text, run_id)?);
    }
    let mut members = Vec::with_capacity(spec.members.len());
    let mut baseline = None;
    let mut observed = BTreeSet::new();
    let mut passed = true;
    for (member, path) in spec.members.iter().zip(&paths) {
        let reader = readers.get_mut(path.as_path()).ok_or_else(|| bad("missing leased database"))?;
        reader.switch_finished_run(member.run_id)?;
        let manifest = reader.run_manifest()?;
        let digest = launch_digest(&manifest)?;
        let result = verify_narrative_budget(reader, &spec.budget)?;
        let mut violations = Vec::new();
        if let Some(seed) = result.observed_root_seed {
            observed.insert(seed);
        }
        if result.observed_root_seed != Some(member.root_seed) {
            violations.push("unexpected_root_seed");
        }
        if let Some(launch) = &digest {
            if baseline.get_or_insert_with(|| launch.clone()).as_str() != launch {
                violations.push("launch_design_mismatch");
            }
        } else {
            violations.push("unverified_launch_design");
        }
        passed &= violations.is_empty() && result.passed;
        members.push(CohortResult {
            expected_root_seed: member.root_seed,
            launch_digest: digest,
            violations,
            result,
        });
    }
    Ok(NarrativeCohortReport {
        schema_version: 1,
        passed: passed && observed.len() == members.len(),
        distinct_observed_seeds: observed.len(),
        members,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(ids: [u128; 3], seeds: [u64; 3]) -> NarrativeCohort {
        serde_json::from_value(json!({
            "schema_version": 1,
            "budget": {"schema_version": 1, "expected_ticks": 10000, "scenario_id":"quiet",
                       "rule": {"type":"quiet"}},
            "members": (0..3).map(|i| json!({"run_id": RunId::new(ids[i]), "root_seed": seeds[i]}))
                .collect::<Vec<_>>()
        })).unwrap()
    }

    #[test]
    fn aliases_and_repeated_seeds_do_not_count_as_independent_replicates() {
        spec([1,2,3], [0,1,u64::MAX]).validate().unwrap();
        assert!(spec([1,1,3], [0,1,2]).validate().is_err());
        assert!(spec([1,2,3], [0,0,2]).validate().is_err());
        let mut short = spec([1,2,3], [0,1,2]);
        short.members.pop();
        assert!(short.validate().is_err());
    }

    // This tests the seed-independent projection of already-validated input;
    // the database reader owns full manifest validation.
    fn launch(seed: u64, habitat: &str, recipe: &str) -> RunManifestRecord {
        let mut record = RunManifestRecord::unattributed(RunId::new(u128::from(seed) + 1));
        record.reproducible = true;
        record.manifest_schema_version = 3;
        record.root_seed = seed;
        record.normalized_config_json = json!({"rng_seed":seed,"habitat":habitat}).to_string();
        record.manifest_json = json!({
            "schema":"v3", "scenario":{"id":"meadow","population_recipe":recipe,"ordered_config_layer_digests":[format!("layer-{seed}")]},
            "build":{"source_revision":"same-source"},
            "agent_substream_protocol":{"version":1,"root_seed":seed}
        }).to_string();
        record
    }

    #[test]
    fn design_digest_ignores_only_seed_variation_in_config_and_rng_contract() {
        let first = launch_digest(&launch(1,"meadow","founders")).unwrap().unwrap();
        assert_eq!(launch_digest(&launch(u64::MAX,"meadow","founders")).unwrap().unwrap(), first);
        assert_ne!(launch_digest(&launch(2,"desert","founders")).unwrap().unwrap(), first);
        assert_ne!(launch_digest(&launch(3,"meadow","bottleneck")).unwrap().unwrap(), first);
        let mut mismatch = launch(4,"meadow","founders");
        mismatch.root_seed = 5;
        assert!(launch_digest(&mismatch).unwrap().is_none());
        mismatch.root_seed = 4;
        mismatch.reproducible = false;
        assert!(launch_digest(&mismatch).unwrap().is_none());
    }
}
