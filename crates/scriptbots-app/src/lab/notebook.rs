//! Notebook renderer with provenance enforcement and reproduce.sh emission (bd-16g.1.5).

use super::stats::{
    AnalysisParams, ConfidenceIntervalProcedure, MatchedEffect, MatchedSeedAnalysis,
    PValueProcedure, RunSummary, StatsError, analyze_matched_seed_runs,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

const MAX_RETAINED_SUMMARY_BYTES: u64 = 4_096;

/// Immutable run provenance reference required for all empirical claims.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunRef {
    pub run_id: String,
    pub arm_id: u16,
    pub seed: u64,
    pub config_digest: String,
    pub digest: String,
    pub total_ticks: u64,
    pub summary_artifact_digest: String,
    pub analysis_input_digest: String,
    pub summary_path: Option<String>,
    /// Canonical arm identity, so a reproduction can rebuild the arm rather than
    /// recognize its digest (bd-16g.1.7 item 3).
    pub variant_id: String,
    /// The arm's exact config overrides.
    pub config_overrides: BTreeMap<String, serde_json::Value>,
    /// Provenance schema version this run was written at (bd-2z0.5.6 policy).
    pub provenance_version: u32,
}

/// Provenance support payload attached to a scientific claim.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum Support {
    Effect(Box<EffectRef>),
    Descriptive(Vec<RunRef>),
}

/// Effect size reference carrying statistically derived run evidence.
///
/// The fields are deliberately private and the type is not deserializable: empirical
/// effect claims can only be constructed by [`claims_from_analysis`], which rederives the
/// complete correction family from the bound run summaries before issuing the reference.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EffectRef {
    runs: Vec<RunRef>,
    effect: MatchedEffect,
    params: AnalysisParams,
    p_value_procedure: PValueProcedure,
    confidence_interval_procedure: ConfidenceIntervalProcedure,
}

/// Scientific claim requiring mandatory provenance support and falsifier criteria.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Claim {
    pub text: String,
    pub support: Support,
    pub falsifier: String,
}

/// Errors raised during notebook rendering and provenance validation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NotebookRenderError {
    #[error("Missing mandatory section: {0}")]
    MissingSection(String),
    #[error("Claim {index} has no statement text")]
    MissingClaimText { index: usize },
    #[error("Claim {index} has no falsification criteria")]
    MissingFalsifier { index: usize },
    #[error("Notebook text field {field} contains disallowed character U+{codepoint:04X}")]
    UnsafeText { field: &'static str, codepoint: u32 },
    #[error(
        "Invalid session id: expected 1-128 ASCII letters, digits, periods, underscores, or hyphens"
    )]
    InvalidSessionId,
    #[error("Non-finite float value in output")]
    NonFiniteFloat,
    #[error("Missing run_id {0} in claim support")]
    MissingRunSupport(String),
    #[error("Digest mismatch for run {run_id}: expected {expected}, found {found}")]
    DigestMismatch {
        run_id: String,
        expected: String,
        found: String,
    },
    #[error("Run provenance mismatch for {run_id}: field {field}")]
    RunProvenanceMismatch { run_id: String, field: &'static str },
    #[error(
        "Effect {metric} arm {treatment_arm} references {found} runs; expected {expected} matched control/treatment rows"
    )]
    EffectRunCountMismatch {
        metric: String,
        treatment_arm: u16,
        expected: usize,
        found: usize,
    },
    #[error("Effect {metric} violates field contract {field}")]
    EffectContractMismatch { metric: String, field: &'static str },
    #[error("Canonical analysis could not be rederived from its run evidence: {0}")]
    AnalysisVerification(#[source] StatsError),
    #[error("Supplied analysis differs from the canonical result rederived from its run evidence")]
    AnalysisResultMismatch,
    #[error("Known run list contains duplicate run_id {0}")]
    DuplicateKnownRun(String),
    #[error("Run {0} has no retained summary artifact path")]
    MissingRetainedArtifactPath(String),
    #[error("Retained summary artifact for run {run_id} exceeds {limit} bytes")]
    RetainedArtifactTooLarge { run_id: String, limit: u64 },
    #[error("Retained summary artifact for run {run_id} does not match its BLAKE3 digest")]
    RetainedArtifactDigestMismatch { run_id: String },
    #[error("IO error: {0}")]
    Io(String),
}

impl From<&RunSummary> for RunRef {
    fn from(summary: &RunSummary) -> Self {
        Self {
            run_id: summary.run_id.clone(),
            arm_id: summary.arm_id,
            seed: summary.seed,
            config_digest: summary.config_digest.clone(),
            digest: summary.digest.clone(),
            total_ticks: summary.ticks,
            summary_artifact_digest: summary.summary_artifact_digest.clone(),
            analysis_input_digest: summary.analysis_input_digest().to_owned(),
            summary_path: summary.summary_path.clone(),
            variant_id: summary.variant_id.clone(),
            config_overrides: summary.config_overrides.clone(),
            provenance_version: summary.provenance_version,
        }
    }
}

/// Stable run provenance list for a completed analysis cohort.
#[must_use]
pub fn run_refs(summaries: &[RunSummary]) -> Vec<RunRef> {
    let mut refs = summaries.iter().map(RunRef::from).collect::<Vec<_>>();
    refs.sort_by(|left, right| {
        left.arm_id
            .cmp(&right.arm_id)
            .then_with(|| left.seed.cmp(&right.seed))
            .then_with(|| left.run_id.cmp(&right.run_id))
    });
    refs
}

/// Turn the canonical analysis output into notebook claims backed by the exact run rows.
///
/// # Errors
///
/// Refuses an analysis effect whose control/treatment provenance cannot be reconstructed
/// exactly from the supplied completed summaries.
pub fn claims_from_analysis(
    analysis: &MatchedSeedAnalysis,
    summaries: &[RunSummary],
    declared_hypothesis: &str,
    falsifier: &str,
) -> Result<Vec<Claim>, NotebookRenderError> {
    let rederived = analyze_matched_seed_runs(summaries, &analysis.metrics, analysis.params)
        .map_err(NotebookRenderError::AnalysisVerification)?;
    if rederived != *analysis {
        return Err(NotebookRenderError::AnalysisResultMismatch);
    }

    let mut by_arm = BTreeMap::<u16, Vec<&RunSummary>>::new();
    for summary in summaries {
        by_arm.entry(summary.arm_id).or_default().push(summary);
    }
    for rows in by_arm.values_mut() {
        rows.sort_by_key(|summary| summary.seed);
    }

    analysis
        .effects
        .iter()
        .map(|effect| {
            let runs = by_arm
                .get(&effect.control_arm)
                .into_iter()
                .flatten()
                .chain(by_arm.get(&effect.treatment_arm).into_iter().flatten())
                .map(|summary| RunRef::from(*summary))
                .collect::<Vec<_>>();
            let expected = effect.n_pairs.saturating_mul(2);
            if runs.len() != expected {
                return Err(NotebookRenderError::EffectRunCountMismatch {
                    metric: effect.metric.clone(),
                    treatment_arm: effect.treatment_arm,
                    expected,
                    found: runs.len(),
                });
            }
            Ok(Claim {
                text: format!(
                    "Matched-seed evaluation of the declared hypothesis {:?} \
                     [metric={}, control_arm={}, treatment_arm={}]",
                    declared_hypothesis, effect.metric, effect.control_arm, effect.treatment_arm
                ),
                support: Support::Effect(Box::new(EffectRef {
                    runs,
                    effect: effect.clone(),
                    params: analysis.params,
                    p_value_procedure: analysis.p_value_procedure,
                    confidence_interval_procedure: analysis.confidence_interval_procedure,
                })),
                falsifier: falsifier.to_owned(),
            })
        })
        .collect()
}

/// Format float value safely, rejecting non-finite numbers.
pub fn format_float_safe(val: f64) -> Result<String, NotebookRenderError> {
    if !val.is_finite() {
        return Err(NotebookRenderError::NonFiniteFloat);
    }
    Ok(format!("{val:.4}"))
}

fn is_spoofing_format_character(character: char) -> bool {
    matches!(
        character,
        '\u{00AD}'
            | '\u{0600}'..='\u{0605}'
            | '\u{061C}'
            | '\u{06DD}'
            | '\u{070F}'
            | '\u{0890}'..='\u{0891}'
            | '\u{08E2}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{206F}'
            | '\u{FEFF}'
            | '\u{FFF9}'..='\u{FFFB}'
            | '\u{110BD}'
            | '\u{110CD}'
            | '\u{13430}'..='\u{13455}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0001}'
            | '\u{E0020}'..='\u{E007F}'
    )
}

/// Render caller-controlled text as one literal Markdown paragraph fragment.
///
/// Newlines are shown as the two visible characters `\n`; every Markdown structural
/// character is escaped, and HTML metacharacters become entities. This keeps model text and
/// persisted identifiers visible without allowing them to create headings, links, raw HTML,
/// code spans, lists, or tables in the trusted notebook structure.
fn markdown_literal(field: &'static str, value: &str) -> Result<String, NotebookRenderError> {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\n' => escaped.push_str("\\n"),
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            '\\' => escaped.push_str("\\\\"),
            '`' => escaped.push_str("\\`"),
            '*' => escaped.push_str("\\*"),
            '_' => escaped.push_str("\\_"),
            '{' => escaped.push_str("\\{"),
            '}' => escaped.push_str("\\}"),
            '[' => escaped.push_str("\\["),
            ']' => escaped.push_str("\\]"),
            '(' => escaped.push_str("\\("),
            ')' => escaped.push_str("\\)"),
            '#' => escaped.push_str("\\#"),
            '+' => escaped.push_str("\\+"),
            '-' => escaped.push_str("\\-"),
            '.' => escaped.push_str("\\."),
            '!' => escaped.push_str("\\!"),
            '|' => escaped.push_str("\\|"),
            '~' => escaped.push_str("\\~"),
            ':' => escaped.push_str("\\:"),
            '@' => escaped.push_str("\\@"),
            '$' => escaped.push_str("\\$"),
            '=' => escaped.push_str("\\="),
            character
                if character.is_control()
                    || (character.is_whitespace() && character != ' ')
                    || is_spoofing_format_character(character) =>
            {
                return Err(NotebookRenderError::UnsafeText {
                    field,
                    codepoint: u32::from(character),
                });
            }
            character => escaped.push(character),
        }
    }
    Ok(escaped)
}

fn validate_session_id(session_id: &str) -> Result<(), NotebookRenderError> {
    let bytes = session_id.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 128
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(*byte, b'.' | b'_' | b'-'))
    {
        return Err(NotebookRenderError::InvalidSessionId);
    }
    Ok(())
}

/// Notebook renderer enforcing claim provenance and emitting executable reproduce.sh.
pub struct NotebookRenderer;

impl NotebookRenderer {
    /// Render deterministic notebook Markdown after validating every referenced run and
    /// statistical value.
    ///
    /// # Errors
    ///
    /// Refuses missing/mismatched provenance, malformed effect support, non-finite values, and
    /// an empty goal.
    pub fn render_markdown(
        goal: &str,
        claims: &[Claim],
        known_runs: &[RunRef],
    ) -> Result<String, NotebookRenderError> {
        if goal.trim().is_empty() {
            return Err(NotebookRenderError::MissingSection("Goal".into()));
        }
        let rendered_goal = markdown_literal("goal", goal)?;
        for (index, claim) in claims.iter().enumerate() {
            if claim.text.trim().is_empty() {
                return Err(NotebookRenderError::MissingClaimText { index: index + 1 });
            }
            if claim.falsifier.trim().is_empty() {
                return Err(NotebookRenderError::MissingFalsifier { index: index + 1 });
            }
            markdown_literal("claim", &claim.text)?;
            markdown_literal("falsifier", &claim.falsifier)?;
        }

        let mut known_run_map = BTreeMap::new();
        for run in known_runs {
            if known_run_map.insert(run.run_id.as_str(), run).is_some() {
                return Err(NotebookRenderError::DuplicateKnownRun(run.run_id.clone()));
            }
        }

        for claim in claims {
            let referenced_runs = match &claim.support {
                Support::Effect(reference) => {
                    validate_effect(reference)?;
                    &reference.runs
                }
                Support::Descriptive(runs) => runs,
            };
            for run_ref in referenced_runs {
                let known = known_run_map.get(run_ref.run_id.as_str()).ok_or_else(|| {
                    NotebookRenderError::MissingRunSupport(run_ref.run_id.clone())
                })?;
                if known.digest != run_ref.digest {
                    return Err(NotebookRenderError::DigestMismatch {
                        run_id: run_ref.run_id.clone(),
                        expected: known.digest.clone(),
                        found: run_ref.digest.clone(),
                    });
                }
                for (matches, field) in [
                    (known.arm_id == run_ref.arm_id, "arm_id"),
                    (known.seed == run_ref.seed, "seed"),
                    (
                        known.config_digest == run_ref.config_digest,
                        "config_digest",
                    ),
                    (known.total_ticks == run_ref.total_ticks, "total_ticks"),
                    (
                        known.summary_artifact_digest == run_ref.summary_artifact_digest,
                        "summary_artifact_digest",
                    ),
                    (
                        known.analysis_input_digest == run_ref.analysis_input_digest,
                        "analysis_input_digest",
                    ),
                    (known.summary_path == run_ref.summary_path, "summary_path"),
                    (known.variant_id == run_ref.variant_id, "variant_id"),
                    (
                        known.config_overrides == run_ref.config_overrides,
                        "config_overrides",
                    ),
                    (
                        known.provenance_version == run_ref.provenance_version,
                        "provenance_version",
                    ),
                ] {
                    if !matches {
                        return Err(NotebookRenderError::RunProvenanceMismatch {
                            run_id: run_ref.run_id.clone(),
                            field,
                        });
                    }
                }
            }
        }

        let mut markdown = String::new();
        markdown.push_str("# ScriptBots Autonomous Science Lab Notebook\n\n");
        markdown.push_str("## 1. Goal\n");
        markdown.push_str(&rendered_goal);
        markdown.push_str("\n\n");
        markdown.push_str("## 2. Methods\n");
        markdown.push_str("- Runner: MatchedSeedExperimentRunner\n");
        markdown.push_str("- Statistics Authority: scriptbots_app::lab::stats\n");
        markdown.push_str("- Verification: BLAKE3 Checksums + WorldDigestV1\n");
        markdown.push_str(&format!("- Total Runs Executed: {}\n\n", known_runs.len()));

        markdown.push_str("## 3. Results & Claims\n");
        for (index, claim) in claims.iter().enumerate() {
            markdown.push_str(&format!("### Claim {}\n", index + 1));
            markdown.push_str(&format!(
                "**Statement**: {}\n\n",
                markdown_literal("claim", &claim.text)?
            ));
            markdown.push_str(&format!(
                "**Falsification Criteria**: {}\n\n",
                markdown_literal("falsifier", &claim.falsifier)?
            ));

            match &claim.support {
                Support::Effect(reference) => render_effect(&mut markdown, reference)?,
                Support::Descriptive(runs) => {
                    markdown.push_str("- **Descriptive Runs**:\n");
                    for run in runs {
                        markdown.push_str(&format!(
                            "  - run_id: {}, arm: {}, seed: {}\n",
                            markdown_literal("run_id", &run.run_id)?,
                            run.arm_id,
                            run.seed
                        ));
                    }
                }
            }
            markdown.push('\n');
        }

        // THE HONESTY GATE (bd-16g.1.7). The parent bead's design requires the template to
        // force an explicit "what would falsify this" and "what I did not test" section.
        //
        // Both are DERIVED from data already validated above rather than accepted as prose
        // from a caller. That is deliberate on two counts: a derived section cannot be
        // omitted (there is no argument a caller can forget to pass), and it cannot be
        // fabricated -- every sentence below restates a fact this function has already
        // checked against `known_runs`. A notebook whose limitations section is optional
        // free text is a notebook whose limitations section is empty.
        markdown.push_str("## 4. What Would Falsify This\n");
        if claims.is_empty() {
            markdown.push_str(
                "No claims were made, so there is nothing to falsify. A session that \
                 completed no claims establishes nothing.\n\n",
            );
        } else {
            for (index, claim) in claims.iter().enumerate() {
                markdown.push_str(&format!(
                    "- **Claim {}**: {}\n",
                    index + 1,
                    markdown_literal("falsifier", &claim.falsifier)?
                ));
            }
            markdown.push('\n');
        }

        markdown.push_str("## 5. What I Did Not Test\n");
        Self::render_scope_limits(&mut markdown, known_runs);

        markdown.push_str("## 6. Reproducibility\n");
        markdown.push_str(
            "`./reproduce.sh` does two things. It verifies the retained summary artifacts \
             are unmodified, and -- with `SCRIPTBOTS_BIN` set to the simulator binary -- it \
             re-executes every arm x seed from the exact emitted config layer and compares \
             each re-run world digest against the one cited here, exiting nonzero if any \
             run fails or any digest differs. The emitted configs are retained beside this \
             notebook as evidence. Summary and adjusted-p table re-derivation is not yet \
             included; digest agreement is the reproduction claim it makes.\n",
        );
        Ok(markdown)
    }

    /// Restate the experiment's boundaries as explicit non-claims.
    ///
    /// Everything here is read off the executed run set. It deliberately reports what the
    /// session DID cover and then names the boundary, rather than asserting "no other knob
    /// was tested" -- that stronger claim needs the full knob space, which this layer does
    /// not have. Overstating the limitations section would be its own dishonesty.
    fn render_scope_limits(markdown: &mut String, known_runs: &[RunRef]) {
        if known_runs.is_empty() {
            markdown.push_str(
                "- No runs completed, so NOTHING in this notebook is empirically \
                 supported.\n\n",
            );
            return;
        }

        let mut arms: Vec<u16> = known_runs.iter().map(|run| run.arm_id).collect();
        arms.sort_unstable();
        arms.dedup();
        let mut seeds: Vec<u64> = known_runs.iter().map(|run| run.seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        let min_ticks = known_runs
            .iter()
            .map(|run| run.total_ticks)
            .min()
            .unwrap_or(0);
        let max_ticks = known_runs
            .iter()
            .map(|run| run.total_ticks)
            .max()
            .unwrap_or(0);

        markdown.push_str(&format!(
            "- Only {} arm(s) were executed. Any configuration outside these arms is \
             untested by this session.\n",
            arms.len()
        ));
        markdown.push_str(&format!(
            "- Only {} distinct seed(s) were executed. Behaviour under other seeds is \
             untested; matched-seed pairing controls for seed variance WITHIN this \
             cohort, it does not generalize beyond it.\n",
            seeds.len()
        ));
        if min_ticks == max_ticks {
            markdown.push_str(&format!(
                "- Every run stopped at {min_ticks} ticks. Nothing is established about \
                 behaviour after that horizon.\n"
            ));
        } else {
            markdown.push_str(&format!(
                "- Runs stopped between {min_ticks} and {max_ticks} ticks. Nothing is \
                 established about behaviour after that horizon, and the UNEQUAL horizons \
                 are themselves a confound: arms observed for different durations are not \
                 directly comparable.\n"
            ));
        }
        if seeds.len() < 3 {
            markdown.push_str(
                "- **Underpowered**: fewer than three distinct seeds. Any effect reported \
                 here should be treated as a pilot observation, not a result.\n",
            );
        }
        markdown.push('\n');
    }

    /// Renders the complete notebook artifact directory and reproduce.sh script.
    pub fn render_notebook(
        session_id: &str,
        goal: &str,
        claims: &[Claim],
        known_runs: &[RunRef],
        out_dir: &Path,
    ) -> Result<PathBuf, NotebookRenderError> {
        validate_session_id(session_id)?;
        let markdown = Self::render_markdown(goal, claims, known_runs)?;
        let mut verified_paths = Vec::with_capacity(known_runs.len());
        for run in known_runs {
            let summary_path = run.summary_path.as_deref().ok_or_else(|| {
                NotebookRenderError::MissingRetainedArtifactPath(run.run_id.clone())
            })?;
            let file = fs::File::open(summary_path)
                .map_err(|error| NotebookRenderError::Io(error.to_string()))?;
            let mut bytes = Vec::new();
            file.take(MAX_RETAINED_SUMMARY_BYTES + 1)
                .read_to_end(&mut bytes)
                .map_err(|error| NotebookRenderError::Io(error.to_string()))?;
            if bytes.len() > usize::try_from(MAX_RETAINED_SUMMARY_BYTES).unwrap_or(usize::MAX) {
                return Err(NotebookRenderError::RetainedArtifactTooLarge {
                    run_id: run.run_id.clone(),
                    limit: MAX_RETAINED_SUMMARY_BYTES,
                });
            }
            if blake3::hash(&bytes).to_hex().as_str() != run.summary_artifact_digest {
                return Err(NotebookRenderError::RetainedArtifactDigestMismatch {
                    run_id: run.run_id.clone(),
                });
            }
            verified_paths.push((run, summary_path));
        }

        fs::create_dir_all(out_dir).map_err(|e| NotebookRenderError::Io(e.to_string()))?;
        let notebook_path = out_dir.join("notebook.md");
        fs::write(&notebook_path, markdown).map_err(|e| NotebookRenderError::Io(e.to_string()))?;

        // Emit a real retained-evidence verifier. Re-execution is intentionally not faked.
        let mut script = String::new();
        script.push_str("#!/usr/bin/env bash\nset -euo pipefail\n\n");
        script.push_str("# Retained-evidence verifier for session: ");
        script.push_str(session_id);
        script.push_str("\n# Verifies retained evidence, then re-executes every arm x seed.\n\n");
        script.push_str(
            "command -v b3sum >/dev/null 2>&1 || { \
             echo 'b3sum is required to verify BLAKE3 evidence' >&2; exit 2; }\n",
        );
        for (run, summary_path) in verified_paths {
            script.push_str("expected=");
            script.push_str(&shell_single_quote(&run.summary_artifact_digest));
            script.push('\n');
            script.push_str("path=");
            script.push_str(&shell_single_quote(summary_path));
            script.push('\n');
            script.push_str(
                "actual=\"$(b3sum -- \"$path\")\"\nactual=\"${actual%% *}\"\n\
                 test \"$actual\" = \"$expected\" || { \
                 echo \"digest mismatch: $path\" >&2; exit 1; }\n",
            );
        }
        script.push_str("echo '[VERIFY] Retained analysis inputs match the notebook evidence.'\n");

        // === RE-EXECUTION (bd-16g.1.7 item 3) ===
        //
        // The block above hashes retained files. That is an integrity check and proves
        // nothing about reproducibility, which is why the acceptance says re-execution must
        // never be substituted by retained-file hashing. Now that RunRef carries the arm's
        // exact config_overrides (12ba9ab09e), the script can write the real config layer
        // and run it.
        //
        // rng_seed is a CONFIG FIELD rather than a flag, so one emitted file fully
        // determines the run -- there is no second channel through which a seed could
        // disagree with the config it was paired with.
        //
        // The emitted configs are written NEXT TO THE NOTEBOOK and never cleaned up. They
        // are evidence: they are the exact bytes the reproduction fed the simulator, and a
        // script that deletes them on exit destroys the artifact that makes its own result
        // inspectable.
        script.push_str(
            "\n# --- Re-execution: rerun every arm x seed and compare world digests ---\n",
        );
        script.push_str(
            "if [ -z \"${SCRIPTBOTS_BIN:-}\" ]; then\n               echo 'set SCRIPTBOTS_BIN to the scriptbots-app binary to re-execute' >&2\n               exit 2\nfi\n",
        );
        script.push_str(
            "work=\"$(cd \"$(dirname \"$0\")\" && pwd)/reproduce-configs\"\nmkdir -p \"$work\"\n",
        );
        script.push_str("rerun_failures=0\n");

        for run in known_runs {
            // The emitted config is the arm's overrides plus the pinned seed, serialized
            // with serde so the file is exactly what the runner parses -- never hand-built
            // text that could drift from the real schema.
            let mut layer = serde_json::Map::new();
            for (key, value) in &run.config_overrides {
                layer.insert(key.clone(), value.clone());
            }
            layer.insert("rng_seed".to_owned(), serde_json::json!(run.seed));
            let layer_json = serde_json::to_string(&serde_json::Value::Object(layer))
                .map_err(|error| NotebookRenderError::Io(error.to_string()))?;

            let safe_name = run
                .run_id
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect::<String>();
            script.push_str("cfg=\"$work/");
            script.push_str(&safe_name);
            script.push_str(".json\"\n");
            script.push_str("printf '%s' ");
            script.push_str(&shell_single_quote(&layer_json));
            script.push_str(" > \"$cfg\"\n");

            script.push_str("out=\"$(SCRIPTBOTS_DET_RUN=1 SCRIPTBOTS_DET_TICKS=");
            script.push_str(&run.total_ticks.to_string());
            script.push_str(" \"$SCRIPTBOTS_BIN\" --config \"$cfg\")\" || { echo 'run failed: ");
            script.push_str(&safe_name);
            script.push_str("' >&2; rerun_failures=$((rerun_failures+1)); }\n");

            // Compare the RE-EXECUTED world digest against the cited one. This assertion is
            // what makes the script a reproduction rather than a checksum.
            script.push_str("expected_digest=");
            script.push_str(&shell_single_quote(&run.digest));
            script.push('\n');
            script.push_str(
                "case \"$out\" in *\"$expected_digest\"*) ;; *) echo 'digest differs on re-execution: ",
            );
            script.push_str(&safe_name);
            script.push_str("' >&2; rerun_failures=$((rerun_failures+1));; esac\n");
        }

        script.push_str(
            "test \"$rerun_failures\" -eq 0 || { echo \"[FAIL] $rerun_failures run(s) did not reproduce\" >&2; exit 1; }\n",
        );
        script.push_str(
            "echo '[VERIFY] Every arm x seed re-executed and matched its cited digest.'\n",
        );

        let reproduce_path = out_dir.join("reproduce.sh");
        fs::write(&reproduce_path, &script).map_err(|e| NotebookRenderError::Io(e.to_string()))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&reproduce_path, fs::Permissions::from_mode(0o755));
        }

        Ok(notebook_path)
    }
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn validate_effect(reference: &EffectRef) -> Result<(), NotebookRenderError> {
    let effect = &reference.effect;
    let params = reference.params;
    let contract_error = |field| NotebookRenderError::EffectContractMismatch {
        metric: effect.metric.clone(),
        field,
    };
    if effect.n_pairs == 0 || effect.control_arm == effect.treatment_arm {
        return Err(contract_error("arm_and_sample_contract"));
    }
    let expected = effect.n_pairs.saturating_mul(2);
    if reference.runs.len() != expected {
        return Err(NotebookRenderError::EffectRunCountMismatch {
            metric: effect.metric.clone(),
            treatment_arm: effect.treatment_arm,
            expected,
            found: reference.runs.len(),
        });
    }
    if reference
        .runs
        .iter()
        .any(|run| run.arm_id != effect.control_arm && run.arm_id != effect.treatment_arm)
    {
        return Err(contract_error("support_arm_membership"));
    }
    let control_seeds = reference
        .runs
        .iter()
        .filter(|run| run.arm_id == effect.control_arm)
        .map(|run| run.seed)
        .collect::<Vec<_>>();
    let treatment_seeds = reference
        .runs
        .iter()
        .filter(|run| run.arm_id == effect.treatment_arm)
        .map(|run| run.seed)
        .collect::<Vec<_>>();
    let control_set = control_seeds.iter().copied().collect::<BTreeSet<_>>();
    let treatment_set = treatment_seeds.iter().copied().collect::<BTreeSet<_>>();
    if control_seeds.len() != effect.n_pairs
        || treatment_seeds.len() != effect.n_pairs
        || control_set.len() != effect.n_pairs
        || treatment_set.len() != effect.n_pairs
        || control_set != treatment_set
    {
        return Err(contract_error("matched_seed_membership"));
    }
    for value in [
        Some(effect.mean_difference),
        Some(effect.raw_p_value),
        Some(effect.adjusted.raw_p_value),
        Some(effect.adjusted.adjusted_p_value),
        Some(effect.adjusted.adjusted_alpha),
        effect.standardized_effect,
        effect.ci_95.map(|interval| interval.0),
        effect.ci_95.map(|interval| interval.1),
    ]
    .into_iter()
    .flatten()
    {
        format_float_safe(value)?;
    }
    if effect.raw_p_value != effect.adjusted.raw_p_value
        || effect.alternative != params.alternative
        || effect.adjusted.correction != params.correction
        || effect.adjusted.rank == 0
        || params.bootstrap_iterations == 0
        || params.permutation_iterations == 0
        || params.recommended_pairs == 0
        || !params.alpha.is_finite()
        || !(0.0..1.0).contains(&params.alpha)
        || !(0.0..=1.0).contains(&effect.raw_p_value)
        || !(0.0..=1.0).contains(&effect.adjusted.adjusted_p_value)
        || effect.adjusted.adjusted_alpha <= 0.0
        || effect.adjusted.adjusted_alpha >= 1.0
    {
        return Err(contract_error("multiple_comparison_evidence"));
    }
    match (effect.ci_95, effect.ci_undefined_reason) {
        (Some((lower, upper)), None) if lower <= upper => {}
        (None, Some(super::stats::UndefinedReason::InsufficientPairs { have, need }))
            if have == effect.n_pairs && need == 2 => {}
        _ => return Err(contract_error("confidence_interval_contract")),
    }
    match (
        effect.n_pairs < params.recommended_pairs,
        effect.underpowered_reason,
    ) {
        (
            true,
            Some(super::stats::UnderpoweredReason::FewerThanRecommendedPairs { have, recommended }),
        ) if have == effect.n_pairs && recommended == params.recommended_pairs => {}
        (false, None) => {}
        _ => return Err(contract_error("underpowered_contract")),
    }
    Ok(())
}

fn render_effect(markdown: &mut String, reference: &EffectRef) -> Result<(), NotebookRenderError> {
    let effect = &reference.effect;
    markdown.push_str(&format!(
        "- **Metric**: {}\n- **Control / Treatment Arm**: {} / {}\n\
         - **Estimator**: {}\n- **P-value Procedure**: {}\n\
         - **Permutation Iterations**: {}\n- **Confidence Interval Procedure**: {}\n\
         - **Bootstrap Iterations**: {}\n- **Resampling Seed**: {}\n\
         - **Alternative**: {}\n- **Family Alpha**: {}\n- **Recommended Pairs**: {}\n\
         - **Matched Pairs**: {}\n\
         - **Mean Difference (treatment - control)**: {}\n",
        markdown_literal("metric", &effect.metric)?,
        effect.control_arm,
        effect.treatment_arm,
        effect.estimator.as_str(),
        reference.p_value_procedure.as_str(),
        reference.params.permutation_iterations,
        reference.confidence_interval_procedure.as_str(),
        reference.params.bootstrap_iterations,
        reference.params.resampling_seed,
        effect.alternative.as_str(),
        format_float_safe(reference.params.alpha)?,
        reference.params.recommended_pairs,
        effect.n_pairs,
        format_float_safe(effect.mean_difference)?,
    ));
    match (effect.standardized_effect, effect.undefined_reason) {
        (Some(value), None) => markdown.push_str(&format!(
            "- **Standardized Effect (Cohen's dz)**: {}\n",
            format_float_safe(value)?
        )),
        (None, Some(reason)) => markdown.push_str(&format!(
            "- **Standardized Effect (Cohen's dz)**: undefined ({})\n",
            reason.description()
        )),
        _ => {
            return Err(NotebookRenderError::EffectContractMismatch {
                metric: effect.metric.clone(),
                field: "standardized_effect_contract",
            });
        }
    }
    match (effect.ci_95, effect.ci_undefined_reason) {
        (Some((lower, upper)), None) => markdown.push_str(&format!(
            "- **95% Bootstrap CI**: [{}, {}]\n",
            format_float_safe(lower)?,
            format_float_safe(upper)?,
        )),
        (None, Some(reason)) => markdown.push_str(&format!(
            "- **95% Bootstrap CI**: undefined ({})\n",
            reason.description()
        )),
        _ => {
            return Err(NotebookRenderError::EffectContractMismatch {
                metric: effect.metric.clone(),
                field: "confidence_interval_contract",
            });
        }
    }
    let observed_direction = if effect.mean_difference > 0.0 {
        "treatment_above_control"
    } else if effect.mean_difference < 0.0 {
        "treatment_below_control"
    } else {
        "no_observed_difference"
    };
    markdown.push_str(&format!(
        "- **Observed Direction**: {observed_direction}\n- **Raw p-value**: {}\n\
         - **Correction**: {}\n- **Adjusted p-value**: {}\n\
         - **Rank-specific alpha**: {}\n- **Null Rejected**: {}\n",
        format_float_safe(effect.raw_p_value)?,
        effect.adjusted.correction.as_str(),
        format_float_safe(effect.adjusted.adjusted_p_value)?,
        format_float_safe(effect.adjusted.adjusted_alpha)?,
        effect.adjusted.rejected,
    ));
    if let Some(reason) = effect.underpowered_reason {
        markdown.push_str(&format!(
            "- **Caveat (Underpowered)**: {}\n",
            reason.description()
        ));
    }
    markdown.push_str("- **Structured Input Evidence**:\n");
    for run in &reference.runs {
        markdown.push_str(&format!(
            "  - run_id={}, arm={}, seed={}, config_digest={}, world_digest={}, \
             summary_digest={}, analysis_input_digest={}\n",
            markdown_literal("run_id", &run.run_id)?,
            run.arm_id,
            run.seed,
            markdown_literal("config_digest", &run.config_digest)?,
            markdown_literal("world_digest", &run.digest)?,
            markdown_literal("summary_artifact_digest", &run.summary_artifact_digest)?,
            markdown_literal("analysis_input_digest", &run.analysis_input_digest)?,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab::stats::{
        AdjustedComparison, AlternativeHypothesis, Correction, RunSummaryParts, TestName,
        UndefinedReason, UnderpoweredReason,
    };
    use std::collections::BTreeMap;

    fn run(run_id: &str, arm_id: u16, seed: u64) -> RunRef {
        RunRef {
            run_id: run_id.to_owned(),
            arm_id,
            seed,
            config_digest: format!("config-{arm_id}"),
            digest: format!("digest-{run_id}"),
            total_ticks: 100,
            summary_artifact_digest: format!("summary-{run_id}"),
            analysis_input_digest: format!("analysis-{run_id}"),
            summary_path: None,
            variant_id: format!("arm-{arm_id:03}"),
            config_overrides: BTreeMap::from([(
                "food_regrowth_rate".to_owned(),
                serde_json::json!(0.1 * f64::from(arm_id + 1)),
            )]),
            provenance_version: super::super::stats::LAB_RUN_SUMMARY_VERSION,
        }
    }

    fn effect() -> MatchedEffect {
        MatchedEffect {
            metric: "alive_agents".to_owned(),
            control_arm: 0,
            treatment_arm: 1,
            n_pairs: 1,
            estimator: TestName::PairedDifference,
            alternative: AlternativeHypothesis::TwoSided,
            mean_difference: 3.0,
            standardized_effect: None,
            undefined_reason: Some(UndefinedReason::InsufficientPairs { have: 1, need: 2 }),
            ci_95: None,
            ci_undefined_reason: Some(UndefinedReason::InsufficientPairs { have: 1, need: 2 }),
            raw_p_value: 0.5,
            adjusted: AdjustedComparison {
                original_index: 0,
                rank: 1,
                raw_p_value: 0.5,
                adjusted_p_value: 0.5,
                adjusted_alpha: 0.05,
                rejected: false,
                correction: Correction::BenjaminiHochberg,
            },
            underpowered_reason: Some(UnderpoweredReason::FewerThanRecommendedPairs {
                have: 1,
                recommended: 10,
            }),
        }
    }

    fn effect_ref(runs: Vec<RunRef>, effect: MatchedEffect) -> EffectRef {
        EffectRef {
            runs,
            effect,
            params: AnalysisParams::default(),
            p_value_procedure: PValueProcedure::PairedSignFlipMonteCarlo,
            confidence_interval_procedure: ConfidenceIntervalProcedure::PercentileBootstrap,
        }
    }

    fn summary(run_id: &str, arm_id: u16, seed: u64, value: f64) -> RunSummary {
        RunSummary::from_verified_parts(RunSummaryParts {
            run_id: run_id.to_owned(),
            arm_id,
            seed,
            config_digest: format!("config-{arm_id}-{seed}"),
            digest: format!("world-{run_id}"),
            ticks: 100,
            metrics: BTreeMap::from([("alive_agents".to_owned(), value)]),
            summary_artifact_digest: format!("summary-{run_id}"),
            summary_path: None,
            variant_id: format!("arm-{arm_id:03}"),
            config_overrides: BTreeMap::from([(
                "food_regrowth_rate".to_owned(),
                serde_json::json!(0.1 * f64::from(arm_id + 1)),
            )]),
        })
    }

    fn claim(runs: Vec<RunRef>) -> Claim {
        Claim {
            text: "Treatment changes the surviving population".to_owned(),
            support: Support::Effect(Box::new(effect_ref(runs, effect()))),
            falsifier: "The adjusted interval and test show no change".to_owned(),
        }
    }

    #[test]
    fn provenance_enforcement_rejects_missing_and_mismatched_runs() {
        let control = run("control", 0, 42);
        let missing = run("missing", 1, 42);
        assert_eq!(
            NotebookRenderer::render_markdown(
                "Test Goal",
                &[claim(vec![control.clone(), missing])],
                std::slice::from_ref(&control),
            ),
            Err(NotebookRenderError::MissingRunSupport("missing".into()))
        );

        let mut forged = run("treatment", 1, 42);
        let treatment = forged.clone();
        forged.config_digest = "forged-config".to_owned();
        assert_eq!(
            NotebookRenderer::render_markdown(
                "Test Goal",
                &[claim(vec![control.clone(), forged])],
                &[control, treatment],
            ),
            Err(NotebookRenderError::RunProvenanceMismatch {
                run_id: "treatment".to_owned(),
                field: "config_digest",
            })
        );
    }

    #[test]
    fn claims_rederive_and_reject_mutated_analysis_outputs() {
        let summaries = vec![
            summary("control-1", 0, 1, 10.0),
            summary("control-2", 0, 2, 12.0),
            summary("treatment-1", 1, 1, 14.0),
            summary("treatment-2", 1, 2, 13.0),
        ];
        let params = AnalysisParams {
            bootstrap_iterations: 64,
            permutation_iterations: 64,
            ..AnalysisParams::default()
        };
        let mut analysis =
            analyze_matched_seed_runs(&summaries, &["alive_agents".to_owned()], params).unwrap();
        analysis.effects[0].adjusted.rejected = !analysis.effects[0].adjusted.rejected;

        assert_eq!(
            claims_from_analysis(&analysis, &summaries, "hypothesis", "falsifier"),
            Err(NotebookRenderError::AnalysisResultMismatch)
        );
    }

    #[test]
    fn notebook_artifact_refuses_a_zero_check_verifier() {
        let runs = vec![run("control", 0, 42), run("treatment", 1, 42)];
        let temp_dir = tempfile::tempdir().unwrap();
        assert_eq!(
            NotebookRenderer::render_notebook(
                "session-missing",
                "Population study",
                &[claim(runs.clone())],
                &runs,
                temp_dir.path(),
            ),
            Err(NotebookRenderError::MissingRetainedArtifactPath(
                "control".to_owned()
            ))
        );

        let mut oversized = runs;
        let oversized_path = temp_dir.path().join("oversized.csv");
        let oversized_bytes = vec![b'x'; 4_097];
        std::fs::write(&oversized_path, &oversized_bytes).unwrap();
        oversized[0].summary_path = Some(oversized_path.to_string_lossy().into_owned());
        oversized[0].summary_artifact_digest = blake3::hash(&oversized_bytes).to_hex().to_string();
        assert_eq!(
            NotebookRenderer::render_notebook(
                "session-oversized",
                "Population study",
                &[claim(oversized.clone())],
                &oversized,
                temp_dir.path(),
            ),
            Err(NotebookRenderError::RetainedArtifactTooLarge {
                run_id: "control".to_owned(),
                limit: 4_096,
            })
        );
    }

    #[test]
    fn typed_effect_evidence_is_rendered_and_byte_stable() {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut runs = vec![run("control", 0, 42), run("treatment", 1, 42)];
        for run in &mut runs {
            let bytes = format!("summary evidence for {}\n", run.run_id).into_bytes();
            let path = temp_dir.path().join(format!("{}.csv", run.run_id));
            std::fs::write(&path, &bytes).unwrap();
            run.summary_artifact_digest = blake3::hash(&bytes).to_hex().to_string();
            run.summary_path = Some(path.to_string_lossy().into_owned());
        }
        let rendered_claim = claim(runs.clone());
        let first = NotebookRenderer::render_markdown(
            "Population study",
            std::slice::from_ref(&rendered_claim),
            &runs,
        )
        .unwrap();
        let second =
            NotebookRenderer::render_markdown("Population study", &[rendered_claim], &runs)
                .unwrap();
        assert_eq!(first, second);
        for required in [
            "Statistics Authority: scriptbots_app::lab::stats",
            "Estimator**: paired_difference",
            "P-value Procedure**: paired_sign_flip_monte_carlo",
            "Permutation Iterations**: 4000",
            "Confidence Interval Procedure**: percentile_bootstrap",
            "Bootstrap Iterations**: 2000",
            "Resampling Seed**: 29779",
            "Alternative**: two_sided",
            "Recommended Pairs**: 10",
            "Matched Pairs**: 1",
            "undefined (insufficient_pairs(have=1, need=2))",
            "95% Bootstrap CI**: undefined",
            "Correction**: benjamini_hochberg",
            "Adjusted p-value**: 0.5000",
            "fewer_than_recommended_pairs(have=1, recommended=10)",
        ] {
            assert!(
                first.contains(required),
                "missing `{required}` in:\n{first}"
            );
        }

        let notebook_dir = temp_dir.path().join("notebook");
        let path = NotebookRenderer::render_notebook(
            "session-2",
            "Population study",
            &[claim(runs.clone())],
            &runs,
            &notebook_dir,
        )
        .unwrap();
        assert_eq!(std::fs::read_to_string(path).unwrap(), first);
        let verifier = std::fs::read_to_string(notebook_dir.join("reproduce.sh")).unwrap();
        assert!(verifier.contains("b3sum is required"));
        assert!(verifier.contains("b3sum -- \"$path\""));
        // Was: asserted the script does NOT re-run. bd-16g.1.7 item 3 made it re-run, so
        // the retained-evidence check this test guards now coexists with re-execution
        // rather than standing in for it.
        assert!(verifier.contains("re-executes every arm x seed"));
    }

    #[test]
    fn non_finite_or_internally_inconsistent_effects_are_refused() {
        let runs = vec![run("control", 0, 42), run("treatment", 1, 42)];
        let mut non_finite = effect();
        non_finite.raw_p_value = f64::NAN;
        let bad = Claim {
            text: "bad".to_owned(),
            support: Support::Effect(Box::new(effect_ref(runs.clone(), non_finite))),
            falsifier: "bad".to_owned(),
        };
        assert_eq!(
            NotebookRenderer::render_markdown("goal", &[bad], &runs),
            Err(NotebookRenderError::NonFiniteFloat)
        );

        let mut inconsistent = effect();
        inconsistent.standardized_effect = Some(2.0);
        let bad = Claim {
            text: "bad".to_owned(),
            support: Support::Effect(Box::new(effect_ref(runs.clone(), inconsistent))),
            falsifier: "bad".to_owned(),
        };
        assert_eq!(
            NotebookRenderer::render_markdown("goal", &[bad], &runs),
            Err(NotebookRenderError::EffectContractMismatch {
                metric: "alive_agents".to_owned(),
                field: "standardized_effect_contract",
            })
        );

        let rogue = run("rogue", 2, 42);
        assert_eq!(
            NotebookRenderer::render_markdown(
                "goal",
                &[claim(vec![runs[0].clone(), rogue.clone()])],
                &[runs[0].clone(), rogue],
            ),
            Err(NotebookRenderError::EffectContractMismatch {
                metric: "alive_agents".to_owned(),
                field: "support_arm_membership",
            })
        );

        let mut inconsistent_p = effect();
        inconsistent_p.adjusted.raw_p_value = 0.4;
        let bad = Claim {
            text: "bad".to_owned(),
            support: Support::Effect(Box::new(effect_ref(runs.clone(), inconsistent_p))),
            falsifier: "bad".to_owned(),
        };
        assert_eq!(
            NotebookRenderer::render_markdown("goal", &[bad], &runs),
            Err(NotebookRenderError::EffectContractMismatch {
                metric: "alive_agents".to_owned(),
                field: "multiple_comparison_evidence",
            })
        );
    }

    #[test]
    fn hostile_notebook_text_is_literal_and_byte_stable() {
        let mut control = run("control](https://run.example)", 0, 42);
        control.config_digest = "<script>config()</script>".to_owned();
        control.digest = "# forged-world-heading".to_owned();
        let mut treatment = run("treatment", 1, 42);
        treatment.summary_artifact_digest = "[summary](https://artifact.example)".to_owned();
        treatment.analysis_input_digest = "<img src=x onerror=alert(1)>".to_owned();
        let runs = vec![control, treatment];

        let mut hostile_effect = effect();
        hostile_effect.metric = "alive_agents <script>alert(1)</script>".to_owned();
        let hostile_claim = Claim {
            text: "## Forged result\n<script>alert(1)</script> [click](https://evil.example)"
                .to_owned(),
            support: Support::Effect(Box::new(effect_ref(runs.clone(), hostile_effect))),
            falsifier: "> trusted quote\n![pixel](https://evil.example/pixel)".to_owned(),
        };
        let goal = "# Forged goal\n<img src=x onerror=alert(1)>";

        let first =
            NotebookRenderer::render_markdown(goal, std::slice::from_ref(&hostile_claim), &runs)
                .unwrap();
        let second = NotebookRenderer::render_markdown(goal, &[hostile_claim], &runs).unwrap();
        assert_eq!(first, second);

        for forbidden in [
            "<script>",
            "<img",
            "](https://",
            "\n## Forged result",
            "\n# Forged goal",
        ] {
            assert!(
                !first.contains(forbidden),
                "hostile structure `{forbidden}` reached trusted output:\n{first}"
            );
        }
        for required in [
            "\\# Forged goal\\n&lt;img",
            "\\#\\# Forged result\\n&lt;script&gt;",
            "\\[click\\]\\(https\\://evil\\.example\\)",
            "alive\\_agents &lt;script&gt;",
            "control\\]\\(https\\://run\\.example\\)",
            "&lt;img src\\=x onerror\\=alert\\(1\\)&gt;",
        ] {
            assert!(
                first.contains(required),
                "escaped literal `{required}` missing from:\n{first}"
            );
        }
    }

    #[test]
    fn empty_claim_text_and_falsifier_are_typed_errors() {
        let runs = vec![run("control", 0, 42), run("treatment", 1, 42)];
        let mut missing_text = claim(runs.clone());
        missing_text.text = " \n ".to_owned();
        assert_eq!(
            NotebookRenderer::render_markdown("goal", &[missing_text], &runs),
            Err(NotebookRenderError::MissingClaimText { index: 1 })
        );

        let mut missing_falsifier = claim(runs.clone());
        missing_falsifier.falsifier.clear();
        assert_eq!(
            NotebookRenderer::render_markdown("goal", &[missing_falsifier], &runs),
            Err(NotebookRenderError::MissingFalsifier { index: 1 })
        );
    }

    #[test]
    fn disallowed_control_characters_are_refused() {
        let runs = vec![run("control", 0, 42), run("treatment", 1, 42)];
        assert_eq!(markdown_literal("run_id", " run "), Ok(" run ".to_owned()));
        assert_ne!(
            markdown_literal("run_id", " run "),
            markdown_literal("run_id", "run")
        );
        assert_eq!(
            markdown_literal("run_id", "\nrun\n"),
            Ok("\\nrun\\n".to_owned())
        );
        assert_eq!(
            NotebookRenderer::render_markdown(
                "population\u{0007}study",
                &[claim(runs.clone())],
                &runs
            ),
            Err(NotebookRenderError::UnsafeText {
                field: "goal",
                codepoint: 7,
            })
        );
        assert_eq!(
            NotebookRenderer::render_markdown("population study\t", &[claim(runs.clone())], &runs),
            Err(NotebookRenderError::UnsafeText {
                field: "goal",
                codepoint: 9,
            })
        );
        assert_eq!(
            NotebookRenderer::render_markdown(
                "population\u{202E}study",
                &[claim(runs.clone())],
                &runs
            ),
            Err(NotebookRenderError::UnsafeText {
                field: "goal",
                codepoint: 0x202e,
            })
        );

        let mut unsafe_run = runs[0].clone();
        unsafe_run.digest = "digest\u{2028}second-line".to_owned();
        let support = vec![unsafe_run.clone(), runs[1].clone()];
        assert_eq!(
            NotebookRenderer::render_markdown(
                "goal",
                &[claim(support)],
                &[unsafe_run, runs[1].clone()],
            ),
            Err(NotebookRenderError::UnsafeText {
                field: "world_digest",
                codepoint: 0x2028,
            })
        );
    }

    #[test]
    fn session_id_cannot_inject_shell_lines() {
        let temp_dir = tempfile::tempdir().unwrap();
        for unsafe_id in [
            "",
            "session\nprintf PWNED",
            "session;printf-PWNED",
            "session/path",
        ] {
            assert_eq!(
                NotebookRenderer::render_notebook(unsafe_id, "goal", &[], &[], temp_dir.path(),),
                Err(NotebookRenderError::InvalidSessionId)
            );
        }
        assert!(!temp_dir.path().join("notebook.md").exists());
        assert!(!temp_dir.path().join("reproduce.sh").exists());
    }

    /// The honesty gate must be present and must be derived, not decorative.
    #[test]
    fn bd_16g_1_7_notebook_renders_both_honesty_sections() {
        let runs = vec![run("r1", 0, 7), run("r2", 1, 7)];
        let claims = vec![Claim {
            text: "Higher food regrowth raises equilibrium population.".to_owned(),
            support: Support::Descriptive(runs.clone()),
            falsifier: "Equilibrium population does not rise when regrowth is raised.".to_owned(),
        }];
        let md = NotebookRenderer::render_markdown("goal", &claims, &runs).expect("renders");

        assert!(md.contains("## 4. What Would Falsify This"), "{md}");
        assert!(md.contains("## 5. What I Did Not Test"), "{md}");
        assert!(
            md.contains("Equilibrium population does not rise"),
            "the falsifier must be surfaced in its own section, not only inline: {md}"
        );

        // Derived from the run set, so it cannot be a blank heading.
        assert!(md.contains("Only 2 arm(s)"), "{md}");
        assert!(md.contains("Only 1 distinct seed(s)"), "{md}");
        assert!(md.contains("Every run stopped at 100 ticks"), "{md}");
    }

    /// A one-seed cohort must be labelled underpowered, not presented as a result.
    #[test]
    fn bd_16g_1_7_thin_seed_cohorts_are_flagged_underpowered() {
        let thin = vec![run("r1", 0, 7), run("r2", 1, 7)];
        let md = NotebookRenderer::render_markdown("goal", &[], &thin).expect("renders");
        assert!(
            md.contains("Underpowered"),
            "two runs sharing one seed is a pilot, not a result: {md}"
        );

        let wide = vec![
            run("r1", 0, 1),
            run("r2", 0, 2),
            run("r3", 0, 3),
            run("r4", 1, 1),
        ];
        let md = NotebookRenderer::render_markdown("goal", &[], &wide).expect("renders");
        assert!(
            !md.contains("Underpowered"),
            "three distinct seeds clears the pilot floor: {md}"
        );
    }

    /// Unequal horizons are a confound and must be named as one.
    #[test]
    fn bd_16g_1_7_unequal_tick_horizons_are_reported_as_a_confound() {
        let mut runs = vec![run("r1", 0, 1), run("r2", 1, 2)];
        runs[1].total_ticks = 5_000;
        let md = NotebookRenderer::render_markdown("goal", &[], &runs).expect("renders");
        assert!(md.contains("between 100 and 5000 ticks"), "{md}");
        assert!(
            md.contains("confound"),
            "arms observed for different durations are not comparable, and the notebook \
             must say so rather than printing two numbers: {md}"
        );
    }

    /// A session that completed nothing must say so, not render an empty gap.
    #[test]
    fn bd_16g_1_7_a_zero_run_session_states_that_nothing_is_supported() {
        let md = NotebookRenderer::render_markdown("goal", &[], &[]).expect("renders");
        assert!(
            md.contains("NOTHING in this notebook is empirically supported"),
            "{md}"
        );
        assert!(
            md.contains("No claims were made, so there is nothing to falsify"),
            "{md}"
        );
    }

    /// The reproducibility blurb must not imply re-execution it does not perform.
    #[test]
    fn bd_16g_1_7_reproduce_blurb_does_not_overclaim_reproduction() {
        let runs = vec![run("r1", 0, 1)];
        let md = NotebookRenderer::render_markdown("goal", &[], &runs).expect("renders");
        // The blurb used to disclose that nothing was re-executed. It now re-executes, so
        // the honest disclosure moved: what it must NOT overclaim is table re-derivation,
        // which is still absent. A test that pins a limitation outlives the limitation.
        assert!(
            md.contains("re-executes every arm x seed"),
            "the notebook must state what reproduce.sh actually does: {md}"
        );
        assert!(
            md.contains("table re-derivation is not yet"),
            "the remaining gap must stay disclosed rather than quietly dropped: {md}"
        );
    }

    /// The emitted script must actually RE-EXECUTE, not only re-hash.
    ///
    /// The defect this bead names is that reproduce.sh "explicitly hashes retained
    /// summaries instead of rerunning every arm and seed". These assertions fail if anyone
    /// reverts to that shape.
    #[test]
    fn bd_16g_1_7_reproduce_script_reruns_every_arm_and_seed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let runs = vec![run("r1", 0, 11), run("r2", 1, 12)];
        let mut summaries = Vec::new();
        for item in &runs {
            let path = dir.path().join(format!("{}.json", item.run_id));
            std::fs::write(&path, b"{}").expect("write summary");
            let mut owned = item.clone();
            owned.summary_path = Some(path.to_string_lossy().into_owned());
            owned.summary_artifact_digest = blake3::hash(b"{}").to_hex().to_string();
            summaries.push(owned);
        }
        let out = dir.path().join("nb");
        NotebookRenderer::render_notebook("session-1", "goal", &[], &summaries, &out)
            .expect("renders");
        let script = std::fs::read_to_string(out.join("reproduce.sh")).expect("script");

        // It invokes the real runner, once per run, with the pinned tick budget.
        assert_eq!(
            script.matches("SCRIPTBOTS_DET_RUN=1").count(),
            summaries.len(),
            "every arm x seed must be re-executed:\n{script}"
        );
        assert!(script.contains("SCRIPTBOTS_DET_TICKS=100"), "{script}");

        // It writes the EXACT config layer, seed included, rather than assuming defaults.
        assert!(
            script.contains("\"rng_seed\":11") && script.contains("\"rng_seed\":12"),
            "each run's seed must be pinned into its emitted config: {script}"
        );
        assert!(
            script.contains("food_regrowth_rate"),
            "the arm's overrides must reach the emitted config: {script}"
        );

        // It compares re-executed digests and fails the script on any difference.
        assert!(
            script.contains("digest differs on re-execution"),
            "{script}"
        );
        assert!(script.contains("did not reproduce"), "{script}");
        assert!(
            script.contains("exit 1"),
            "a mismatch must exit nonzero: {script}"
        );

        // And it never deletes the evidence it just produced.
        assert!(
            !script.contains("rm -"),
            "emitted configs are evidence and must not be cleaned up: {script}"
        );
    }
}
