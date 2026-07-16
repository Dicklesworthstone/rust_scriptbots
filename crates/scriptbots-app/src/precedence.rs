//! Startup configuration precedence, written down once and enforced in one place.
//!
//! Thread count can be decided by five different layers — an explicit `--threads`
//! flag, an exported `SCRIPTBOTS_MAX_THREADS`, an auto-tune probe, `--low-power`,
//! or the built-in default — and until now the decision was made by a chain of
//! `if`/`else if` branches scattered through startup, each writing the answer into
//! a process-global environment variable.
//!
//! That arrangement had a specific, silent bug: a user who exported
//! `SCRIPTBOTS_MAX_THREADS=16` and then passed `--low-power` had their explicit
//! value overwritten with `2`, with no warning and no record. They asked for
//! sixteen threads; they got two; nothing told them.
//!
//! # The order, and why it is this order
//!
//! Most SPECIFIC layer wins, not merely the last one to run:
//!
//! 1. `--threads N` — the user naming the exact number on the command line.
//! 2. `SCRIPTBOTS_MAX_THREADS` — the user naming the exact number in their
//!    environment. Still an explicit choice of a specific value, so it beats any
//!    layer that is only supplying a default.
//! 3. The auto-tune probe — a measured recommendation, which is evidence rather
//!    than a preference, so it must never override a preference the user stated.
//! 4. `--low-power` — a MODE, not a number. It supplies a conservative default
//!    for a knob the user did not set; it does not get to overrule a knob they
//!    did.
//! 5. The built-in default (let Rayon use the machine).
//!
//! The rule that makes this coherent: a layer that names a SPECIFIC VALUE always
//! beats a layer that merely expresses a PREFERENCE about defaults.
//!
//! # Probe containment
//!
//! The auto-tune probe's recommendation is applied only when neither of the two
//! explicit layers spoke. That is what stops a probe child's tuning from leaking
//! into a final run whose operator had already made the decision themselves.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::fmt;

/// Which layer actually decided the thread count.
///
/// Recorded in run provenance: a run that used two threads because the operator
/// asked for two is a different run from one that used two because low-power mode
/// quietly decided for them, and a manifest that cannot tell them apart cannot
/// explain its own performance numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadSource {
    /// `--threads N`.
    CliFlag,
    /// An exported `SCRIPTBOTS_MAX_THREADS`.
    Environment,
    /// A measurement from the auto-tune probe.
    AutoTune,
    /// The conservative default that `--low-power` supplies.
    LowPowerDefault,
    /// Nobody asked; let Rayon decide.
    BuiltinDefault,
}

impl ThreadSource {
    /// Stable identifier for PERSISTED records — the run manifest.
    ///
    /// Deliberately separate from [`fmt::Display`]. The Display string is prose written for a
    /// human reading a log line, and someone will eventually reword it. This is a WIRE VALUE that
    /// lands in a manifest on disk and is compared across runs. Tying the two together would mean
    /// that improving a log message silently changed a persisted provenance record — and every
    /// manifest written before the reword would disagree with every one written after, for no
    /// reason a reader could see.
    #[must_use]
    pub const fn wire_tag(self) -> &'static str {
        match self {
            Self::CliFlag => "cli-flag",
            Self::Environment => "environment",
            Self::AutoTune => "auto-tune",
            Self::LowPowerDefault => "low-power-default",
            Self::BuiltinDefault => "builtin-default",
        }
    }
}

impl fmt::Display for ThreadSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::CliFlag => "cli:--threads",
            Self::Environment => "env:SCRIPTBOTS_MAX_THREADS",
            Self::AutoTune => "auto-tune",
            Self::LowPowerDefault => "low-power default",
            Self::BuiltinDefault => "builtin default",
        };
        f.write_str(name)
    }
}

/// The thread count `--low-power` falls back to when nothing more specific spoke.
pub const LOW_POWER_THREADS: usize = 2;

/// The resolved decision, and who made it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadPolicy {
    /// The thread cap, or `None` to leave the machine's default alone.
    pub threads: Option<usize>,
    /// Which layer won.
    pub source: ThreadSource,
    /// Whether a layer's suggestion was DECLINED because a more specific layer
    /// had already spoken.
    ///
    /// Not a warning about a mistake — it is the normal, correct outcome of the
    /// precedence rules. But it must be visible: a user who passes `--low-power`
    /// alongside `--threads 16` deserves to know that low-power did not lower
    /// their thread count, rather than discovering it from a power bill.
    pub overridden: Option<ThreadSource>,
}

/// Resolve the thread count from every layer that has an opinion.
///
/// Pure: no environment reads, no writes, no logging. The caller supplies what
/// each layer said and gets back the decision plus its provenance, which is what
/// makes the matrix below testable at all.
#[must_use]
pub fn resolve_thread_policy(
    cli_threads: Option<usize>,
    env_threads: Option<usize>,
    auto_tune_threads: Option<usize>,
    low_power: bool,
) -> ThreadPolicy {
    // The most specific layer that spoke, in order. Everything after the winner
    // is recorded as declined rather than silently discarded.
    let declined = |winner: ThreadSource| -> Option<ThreadSource> {
        // Report the most specific layer that WOULD have decided, had the winner
        // not spoken. Only layers strictly less specific than the winner can be
        // overridden, so this is the next one down that actually has an opinion.
        match winner {
            ThreadSource::CliFlag => {
                if env_threads.is_some() {
                    Some(ThreadSource::Environment)
                } else if auto_tune_threads.is_some() {
                    Some(ThreadSource::AutoTune)
                } else if low_power {
                    Some(ThreadSource::LowPowerDefault)
                } else {
                    None
                }
            }
            ThreadSource::Environment => {
                if auto_tune_threads.is_some() {
                    Some(ThreadSource::AutoTune)
                } else if low_power {
                    Some(ThreadSource::LowPowerDefault)
                } else {
                    None
                }
            }
            ThreadSource::AutoTune => low_power.then_some(ThreadSource::LowPowerDefault),
            ThreadSource::LowPowerDefault | ThreadSource::BuiltinDefault => None,
        }
    };

    if let Some(threads) = cli_threads {
        return ThreadPolicy {
            threads: Some(threads),
            source: ThreadSource::CliFlag,
            overridden: declined(ThreadSource::CliFlag),
        };
    }
    // An exported SCRIPTBOTS_MAX_THREADS is the user naming a specific number.
    // It beats low-power, which only supplies a default — this is the case the
    // old code got wrong, silently replacing the user's value with 2.
    if let Some(threads) = env_threads {
        return ThreadPolicy {
            threads: Some(threads),
            source: ThreadSource::Environment,
            overridden: declined(ThreadSource::Environment),
        };
    }
    // The probe's recommendation applies only where nobody expressed a
    // preference. This is what keeps a probe child's tuning from leaking into a
    // final run whose operator had already decided.
    if let Some(threads) = auto_tune_threads {
        return ThreadPolicy {
            threads: Some(threads),
            source: ThreadSource::AutoTune,
            overridden: declined(ThreadSource::AutoTune),
        };
    }
    if low_power {
        return ThreadPolicy {
            threads: Some(LOW_POWER_THREADS),
            source: ThreadSource::LowPowerDefault,
            overridden: None,
        };
    }
    ThreadPolicy {
        threads: None,
        source: ThreadSource::BuiltinDefault,
        overridden: None,
    }
}

// ============================================================================
// Configuration layering: defaults -> scenario files -> environment -> CLI
// ============================================================================
//
// The thread lane above resolved ONE knob. The config itself has hundreds, and
// until this resolver existed they were layered by a pile of one-off
// `if let Ok(value) = env::var(..)` blocks plus in-place CLI mutations — the
// same shape as the thread bug before it was fixed: the winner was whichever
// branch happened to run last, rather than the layer the user most explicitly
// stated.
//
// The rule is the same one that made the thread lane coherent: the more
// specific layer wins. A CLI flag names a value for THIS invocation; an
// environment variable names it for this shell; a scenario file names it for
// anyone who runs the file; the defaults speak for nobody in particular.
//
//   defaults -> scenario files (in order) -> environment -> CLI
//
// And the same discipline applies: resolution is a PURE function of what each
// layer said. No environment reads, no file reads, no logging — the caller
// gathers the statements, the resolver merges them and returns the provenance
// of every field where one layer displaced another. That purity is what makes
// the matrix below testable at all.

/// Which kind of layer a configuration statement came from.
///
/// Declared in application order: a later kind is more specific than an
/// earlier one, and the resolver applies statements in exactly the order the
/// caller supplies them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigLayerKind {
    /// The built-in `ScriptBotsConfig` defaults — the layer that speaks for
    /// nobody in particular and therefore loses to everyone.
    Defaults,
    /// A scenario/configuration file supplied with `--config`.
    File,
    /// Environment variables (`SCRIPTBOTS_*`).
    Environment,
    /// Command-line flags — the user naming a value for this exact invocation.
    Cli,
}

impl ConfigLayerKind {
    /// Stable identifier for PERSISTED records — the run manifest.
    ///
    /// Deliberately separate from [`fmt::Display`] for the same reason as
    /// [`ThreadSource::wire_tag`]: a Display string is prose someone will
    /// eventually reword, while this is a wire value compared across runs.
    #[must_use]
    pub const fn wire_tag(self) -> &'static str {
        match self {
            Self::Defaults => "defaults",
            Self::File => "file",
            Self::Environment => "environment",
            Self::Cli => "cli",
        }
    }
}

impl fmt::Display for ConfigLayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.wire_tag())
    }
}

/// What one configuration layer actually said.
///
/// `fields` is a partial configuration as a JSON object tree: only the paths
/// the layer spoke about are present. The caller performs whatever I/O and
/// parsing produced it (reading a file, decoding environment variables,
/// interpreting flags); by the time a statement reaches the resolver it is
/// pure data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigLayerStatement {
    /// Which kind of layer is speaking.
    pub kind: ConfigLayerKind,
    /// Human-readable identity of the speaker — a file path, a variable name,
    /// a flag name. Carried into override records so a reader can tell WHICH
    /// file lost to WHICH variable, not merely that "a file" lost.
    pub label: String,
    /// The partial configuration the layer stated.
    pub fields: JsonValue,
}

/// One configuration field where a later layer displaced an earlier layer's value.
///
/// The config analogue of [`ThreadPolicy::overridden`]: the normal, correct
/// outcome of the precedence rules rather than a mistake — but it must be
/// visible. A user whose scenario file said one thing and whose environment
/// said another deserves to see that in the run record rather than discover it
/// from the results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigFieldOverride {
    /// Dotted path of the displaced field (for example `neuroflow.enabled`).
    /// Configuration field names never contain dots, so the path is
    /// unambiguous.
    pub path: String,
    /// Label of the layer whose value was displaced.
    pub losing_layer: String,
    /// Kind of the layer whose value was displaced.
    pub losing_kind: ConfigLayerKind,
    /// The value that was displaced.
    pub losing_value: JsonValue,
    /// Label of the layer whose value now stands.
    pub winning_layer: String,
    /// Kind of the layer whose value now stands.
    pub winning_kind: ConfigLayerKind,
    /// The value that now stands.
    pub winning_value: JsonValue,
}

/// The merged configuration tree plus the provenance of every displacement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedConfigLayers {
    /// The configuration after every layer was applied in order.
    pub merged: JsonValue,
    /// Every cross-layer displacement, in application order.
    ///
    /// Displacing a DEFAULT is not recorded here — that is simply what
    /// configuring means. An entry exists only where two explicit layers both
    /// named the same path and disagreed; when several layers disagree in
    /// sequence, each displacement names the most recent earlier writer, so a
    /// three-way conflict yields a chain of two records.
    pub overrides: Vec<ConfigFieldOverride>,
}

/// Merge every layer statement over the defaults, in order, tracking who wrote
/// what.
///
/// Pure: no environment reads, no file reads, no logging. The caller supplies
/// what each layer said and gets back the merged tree plus the provenance of
/// every field where one explicit layer displaced another — which is what
/// makes the layering matrix testable at all.
///
/// Merging is the same deep-merge the file layers have always used: objects
/// merge key by key, everything else (scalars and arrays alike) replaces
/// wholesale. Restating the standing value is not a displacement, but it does
/// make the restating layer the field's most recent writer, so a later
/// conflicting layer names the runner-up — exactly as the thread lane's
/// `overridden` names the next-most-specific declined layer.
#[must_use]
pub fn resolve_config_layers(
    defaults: &JsonValue,
    layers: &[ConfigLayerStatement],
) -> ResolvedConfigLayers {
    let mut merged = defaults.clone();
    let mut provenance: BTreeMap<String, usize> = BTreeMap::new();
    let mut overrides = Vec::new();
    for (index, layer) in layers.iter().enumerate() {
        merge_tracked(
            &mut merged,
            &layer.fields,
            "",
            index,
            layers,
            &mut provenance,
            &mut overrides,
        );
    }
    ResolvedConfigLayers { merged, overrides }
}

/// Canonical content bytes for one layer's statement, for the ordered layer
/// digests in the run manifest.
///
/// Canonical by explicit recursive key-sorting, NOT by trusting the map type:
/// this workspace's dependency graph enables `serde_json`'s `preserve_order`
/// feature, so a `Value` serializes in insertion order and two statements with
/// identical content but different construction histories would otherwise
/// digest differently. File layers keep digesting their exact source bytes —
/// those bytes ARE the layer; this exists for the environment and CLI layers,
/// which have no source file.
#[must_use]
pub fn canonical_layer_bytes(fields: &JsonValue) -> Vec<u8> {
    canonical_value(fields).to_string().into_bytes()
}

/// Rebuild a JSON tree with every object's keys in sorted order.
///
/// Array order is preserved — element order is data, while object key order is
/// an accident of construction.
fn canonical_value(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            let mut sorted = serde_json::Map::new();
            for key in keys {
                if let Some(child) = map.get(key) {
                    sorted.insert(key.clone(), canonical_value(child));
                }
            }
            JsonValue::Object(sorted)
        }
        JsonValue::Array(items) => JsonValue::Array(items.iter().map(canonical_value).collect()),
        other => other.clone(),
    }
}

/// Recursive worker for [`resolve_config_layers`].
///
/// `provenance` maps each dotted leaf path to the index of the layer that most
/// recently wrote it. Only explicit layers appear; paths still owned by the
/// defaults are absent, which is how "displacing a default is not an override"
/// falls out of the structure instead of being a special case.
fn merge_tracked(
    target: &mut JsonValue,
    incoming: &JsonValue,
    path: &str,
    layer_index: usize,
    layers: &[ConfigLayerStatement],
    provenance: &mut BTreeMap<String, usize>,
    overrides: &mut Vec<ConfigFieldOverride>,
) {
    if let (JsonValue::Object(target_map), JsonValue::Object(incoming_map)) =
        (&mut *target, incoming)
    {
        for (key, value) in incoming_map {
            let child_path = join_path(path, key);
            if let Some(existing) = target_map.get_mut(key) {
                merge_tracked(
                    existing,
                    value,
                    &child_path,
                    layer_index,
                    layers,
                    provenance,
                    overrides,
                );
            } else {
                target_map.insert(key.clone(), value.clone());
                mark_subtree(provenance, &child_path, value, layer_index);
            }
        }
        return;
    }

    // Leaf write: a scalar or array, or a shape conflict where one side is an
    // object and the other is not. Either way the incoming value replaces the
    // standing subtree wholesale, so every earlier claim at or below this path
    // is displaced together.
    record_displacement(
        target,
        incoming,
        path,
        layer_index,
        layers,
        provenance,
        overrides,
    );
    purge_subtree(provenance, path);
    mark_subtree(provenance, path, incoming, layer_index);
    *target = incoming.clone();
}

/// Record one displacement if this leaf write takes a path an earlier explicit
/// layer had written differently.
fn record_displacement(
    target: &JsonValue,
    incoming: &JsonValue,
    path: &str,
    layer_index: usize,
    layers: &[ConfigLayerStatement],
    provenance: &BTreeMap<String, usize>,
    overrides: &mut Vec<ConfigFieldOverride>,
) {
    if target == incoming {
        // Restating the standing value displaces nothing.
        return;
    }
    let Some(loser_index) = most_recent_writer(provenance, path) else {
        // The standing value came from the defaults; replacing it is simply
        // what configuring means.
        return;
    };
    let (Some(loser), Some(winner)) = (layers.get(loser_index), layers.get(layer_index)) else {
        return;
    };
    overrides.push(ConfigFieldOverride {
        path: path.to_owned(),
        losing_layer: loser.label.clone(),
        losing_kind: loser.kind,
        losing_value: target.clone(),
        winning_layer: winner.label.clone(),
        winning_kind: winner.kind,
        winning_value: incoming.clone(),
    });
}

/// The most recent explicit layer that wrote this exact path or anything
/// strictly below it (a subtree replacement displaces every claim inside).
fn most_recent_writer(provenance: &BTreeMap<String, usize>, path: &str) -> Option<usize> {
    let exact = provenance.get(path).copied();
    let below = descendants(provenance, path).map(|(_, index)| index).max();
    exact.into_iter().chain(below).max()
}

/// Mark every leaf inside `value` as written by `layer_index`.
fn mark_subtree(
    provenance: &mut BTreeMap<String, usize>,
    path: &str,
    value: &JsonValue,
    layer_index: usize,
) {
    if let JsonValue::Object(map) = value {
        if map.is_empty() {
            // An empty object is still a statement at this path.
            provenance.insert(path.to_owned(), layer_index);
            return;
        }
        for (key, child) in map {
            let child_path = join_path(path, key);
            mark_subtree(provenance, &child_path, child, layer_index);
        }
        return;
    }
    provenance.insert(path.to_owned(), layer_index);
}

/// Forget every claim at or below `path`; its subtree has been replaced.
fn purge_subtree(provenance: &mut BTreeMap<String, usize>, path: &str) {
    if path.is_empty() {
        provenance.clear();
        return;
    }
    provenance.remove(path);
    let doomed: Vec<String> = descendants(provenance, path)
        .map(|(key, _)| key.to_owned())
        .collect();
    for key in doomed {
        provenance.remove(&key);
    }
}

/// Iterate provenance entries strictly below `path` in the dotted hierarchy.
fn descendants<'p>(
    provenance: &'p BTreeMap<String, usize>,
    path: &str,
) -> impl Iterator<Item = (&'p str, usize)> {
    let prefix = if path.is_empty() {
        String::new()
    } else {
        format!("{path}.")
    };
    provenance
        .range(prefix.clone()..)
        .take_while(move |(key, _)| key.starts_with(&prefix))
        .map(|(key, &index)| (key.as_str(), index))
}

/// Join a dotted parent path with a child key.
fn join_path(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_owned()
    } else {
        format!("{path}.{key}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The precedence matrix, in full.
    ///
    /// Table-driven because the interesting bugs here are not in any single
    /// branch but in the INTERACTIONS between layers — and an interaction nobody
    /// enumerated is an interaction nobody tested.
    #[test]
    fn the_precedence_matrix_is_complete_and_specific_beats_general() {
        struct Case {
            name: &'static str,
            cli: Option<usize>,
            env: Option<usize>,
            auto: Option<usize>,
            low_power: bool,
            want_threads: Option<usize>,
            want_source: ThreadSource,
        }

        let cases = [
            Case {
                name: "nobody speaks: let the machine decide",
                cli: None,
                env: None,
                auto: None,
                low_power: false,
                want_threads: None,
                want_source: ThreadSource::BuiltinDefault,
            },
            Case {
                name: "low-power alone supplies its conservative default",
                cli: None,
                env: None,
                auto: None,
                low_power: true,
                want_threads: Some(LOW_POWER_THREADS),
                want_source: ThreadSource::LowPowerDefault,
            },
            Case {
                name: "the probe decides when nobody stated a preference",
                cli: None,
                env: None,
                auto: Some(6),
                low_power: false,
                want_threads: Some(6),
                want_source: ThreadSource::AutoTune,
            },
            Case {
                name: "the probe outranks low-power: evidence beats a default",
                cli: None,
                env: None,
                auto: Some(6),
                low_power: true,
                want_threads: Some(6),
                want_source: ThreadSource::AutoTune,
            },
            Case {
                // THE BUG. The old code overwrote this user's exported 16 with a
                // silent 2, because low-power ran later in the if/else chain.
                name: "an exported thread count SURVIVES --low-power",
                cli: None,
                env: Some(16),
                auto: None,
                low_power: true,
                want_threads: Some(16),
                want_source: ThreadSource::Environment,
            },
            Case {
                name: "an exported thread count is not clobbered by the probe either",
                cli: None,
                env: Some(16),
                auto: Some(6),
                low_power: false,
                want_threads: Some(16),
                want_source: ThreadSource::Environment,
            },
            Case {
                name: "--threads beats the environment",
                cli: Some(8),
                env: Some(16),
                auto: None,
                low_power: false,
                want_threads: Some(8),
                want_source: ThreadSource::CliFlag,
            },
            Case {
                name: "--threads beats absolutely everything",
                cli: Some(8),
                env: Some(16),
                auto: Some(6),
                low_power: true,
                want_threads: Some(8),
                want_source: ThreadSource::CliFlag,
            },
        ];

        for case in cases {
            let policy = resolve_thread_policy(case.cli, case.env, case.auto, case.low_power);
            assert_eq!(
                policy.threads, case.want_threads,
                "`{}`: wrong thread count",
                case.name
            );
            assert_eq!(
                policy.source, case.want_source,
                "`{}`: wrong deciding layer",
                case.name
            );
        }
    }

    #[test]
    fn a_declined_layer_is_reported_rather_than_silently_dropped() {
        // A user who passes --low-power alongside --threads 16 deserves to learn
        // that low-power did NOT lower their thread count from the program, not
        // from their power bill.
        let policy = resolve_thread_policy(Some(16), None, None, true);
        assert_eq!(policy.source, ThreadSource::CliFlag);
        assert_eq!(policy.overridden, Some(ThreadSource::LowPowerDefault));

        // And the case that used to be a silent clobber is now an explicit,
        // reported decline.
        let policy = resolve_thread_policy(None, Some(16), None, true);
        assert_eq!(policy.threads, Some(16));
        assert_eq!(policy.source, ThreadSource::Environment);
        assert_eq!(policy.overridden, Some(ThreadSource::LowPowerDefault));
    }

    #[test]
    fn the_probe_cannot_leak_into_a_run_whose_operator_already_decided() {
        // Probe containment. An auto-tune child measures the machine and returns
        // a recommendation; a recommendation must never overrule an instruction.
        for explicit in [8usize, 1, 64] {
            let via_cli = resolve_thread_policy(Some(explicit), None, Some(6), true);
            assert_eq!(via_cli.threads, Some(explicit));
            assert_eq!(via_cli.source, ThreadSource::CliFlag);

            let via_env = resolve_thread_policy(None, Some(explicit), Some(6), true);
            assert_eq!(via_env.threads, Some(explicit));
            assert_eq!(via_env.source, ThreadSource::Environment);
        }
    }

    #[test]
    fn resolution_is_a_pure_function_of_its_inputs() {
        // Same inputs, same answer, every time: startup configuration that
        // depended on evaluation order would make a run's thread count a
        // property of when it was launched rather than of what was asked for.
        let inputs = [
            (None, None, None, false),
            (None, Some(16), Some(6), true),
            (Some(8), Some(16), Some(6), true),
            (None, None, Some(4), true),
        ];
        for (cli, env, auto, low) in inputs {
            let first = resolve_thread_policy(cli, env, auto, low);
            let second = resolve_thread_policy(cli, env, auto, low);
            assert_eq!(first, second);
        }
    }
}

#[cfg(test)]
mod config_layering_tests {
    use super::*;
    use serde_json::json;

    fn statement(kind: ConfigLayerKind, label: &str, fields: JsonValue) -> ConfigLayerStatement {
        ConfigLayerStatement {
            kind,
            label: label.to_owned(),
            fields,
        }
    }

    fn defaults() -> JsonValue {
        json!({
            "world_width": 1600,
            "world_height": 900,
            "rng_seed": null,
            "neuroflow": { "enabled": false, "hidden_layers": [16, 8] },
        })
    }

    fn at<'v>(merged: &'v JsonValue, dotted: &str) -> Option<&'v JsonValue> {
        let pointer = format!("/{}", dotted.replace('.', "/"));
        merged.pointer(&pointer)
    }

    /// The layering matrix, in full — the config analogue of the thread matrix
    /// above, and table-driven for the same reason: the interesting bugs are
    /// not in any single branch but in the INTERACTIONS between layers, and an
    /// interaction nobody enumerated is an interaction nobody tested.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn the_layering_matrix_covers_every_pair_and_specific_beats_general() {
        struct Case {
            name: &'static str,
            layers: Vec<ConfigLayerStatement>,
            want: Vec<(&'static str, JsonValue)>,
            /// `(path, losing_layer, winning_layer)` triples, in order.
            want_overrides: Vec<(&'static str, &'static str, &'static str)>,
        }

        let cases = [
            Case {
                name: "nobody speaks: the defaults stand and nothing is an override",
                layers: vec![],
                want: vec![("world_width", json!(1600))],
                want_overrides: vec![],
            },
            Case {
                name: "a file beating a default is configuration, not an override",
                layers: vec![statement(
                    ConfigLayerKind::File,
                    "file:base.toml",
                    json!({"world_width": 2048}),
                )],
                want: vec![("world_width", json!(2048))],
                want_overrides: vec![],
            },
            Case {
                name: "the environment displaces the file, and the record says so",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"world_width": 2048}),
                    ),
                    statement(
                        ConfigLayerKind::Environment,
                        "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                        json!({"world_width": 1024}),
                    ),
                ],
                want: vec![("world_width", json!(1024))],
                want_overrides: vec![(
                    "world_width",
                    "file:base.toml",
                    "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                )],
            },
            Case {
                name: "the CLI displaces the environment",
                layers: vec![
                    statement(
                        ConfigLayerKind::Environment,
                        "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                        json!({"world_width": 1024}),
                    ),
                    statement(
                        ConfigLayerKind::Cli,
                        "cli:--set",
                        json!({"world_width": 512}),
                    ),
                ],
                want: vec![("world_width", json!(512))],
                want_overrides: vec![(
                    "world_width",
                    "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                    "cli:--set",
                )],
            },
            Case {
                name: "the CLI displaces the file when the environment is silent",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"world_width": 2048}),
                    ),
                    statement(
                        ConfigLayerKind::Cli,
                        "cli:--set",
                        json!({"world_width": 512}),
                    ),
                ],
                want: vec![("world_width", json!(512))],
                want_overrides: vec![("world_width", "file:base.toml", "cli:--set")],
            },
            Case {
                name: "all three speak: the CLI wins and every displacement is on the record",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"world_width": 2048}),
                    ),
                    statement(
                        ConfigLayerKind::Environment,
                        "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                        json!({"world_width": 1024}),
                    ),
                    statement(
                        ConfigLayerKind::Cli,
                        "cli:--set",
                        json!({"world_width": 512}),
                    ),
                ],
                want: vec![("world_width", json!(512))],
                want_overrides: vec![
                    (
                        "world_width",
                        "file:base.toml",
                        "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                    ),
                    (
                        "world_width",
                        "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                        "cli:--set",
                    ),
                ],
            },
            Case {
                name: "a later file displaces an earlier file, by name",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:first.toml",
                        json!({"rng_seed": 41}),
                    ),
                    statement(
                        ConfigLayerKind::File,
                        "file:second.toml",
                        json!({"rng_seed": 42}),
                    ),
                ],
                want: vec![("rng_seed", json!(42))],
                want_overrides: vec![("rng_seed", "file:first.toml", "file:second.toml")],
            },
            Case {
                name: "agreement is not displacement",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"world_width": 2048}),
                    ),
                    statement(
                        ConfigLayerKind::Environment,
                        "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                        json!({"world_width": 2048}),
                    ),
                ],
                want: vec![("world_width", json!(2048))],
                want_overrides: vec![],
            },
            Case {
                name: "an empty statement changes nothing and displaces nothing",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"world_width": 2048}),
                    ),
                    statement(ConfigLayerKind::Environment, "env:typed", json!({})),
                ],
                want: vec![("world_width", json!(2048))],
                want_overrides: vec![],
            },
            Case {
                name: "nested fields are tracked by dotted path",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"neuroflow": {"enabled": true}}),
                    ),
                    statement(
                        ConfigLayerKind::Cli,
                        "cli:--set",
                        json!({"neuroflow": {"enabled": false}}),
                    ),
                ],
                want: vec![
                    ("neuroflow.enabled", json!(false)),
                    // Sibling keys the later layer did not mention survive.
                    ("neuroflow.hidden_layers", json!([16, 8])),
                ],
                want_overrides: vec![("neuroflow.enabled", "file:base.toml", "cli:--set")],
            },
            Case {
                name: "arrays replace wholesale rather than merging elementwise",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"neuroflow": {"hidden_layers": [64, 32]}}),
                    ),
                    statement(
                        ConfigLayerKind::Environment,
                        "env:SCRIPTBOTS_NEUROFLOW_HIDDEN",
                        json!({"neuroflow": {"hidden_layers": [8]}}),
                    ),
                ],
                want: vec![("neuroflow.hidden_layers", json!([8]))],
                want_overrides: vec![(
                    "neuroflow.hidden_layers",
                    "file:base.toml",
                    "env:SCRIPTBOTS_NEUROFLOW_HIDDEN",
                )],
            },
            Case {
                name: "a subtree replacement displaces the claims written inside it",
                layers: vec![
                    statement(
                        ConfigLayerKind::File,
                        "file:base.toml",
                        json!({"neuroflow": {"enabled": true}}),
                    ),
                    statement(
                        ConfigLayerKind::Cli,
                        "cli:--set",
                        json!({"neuroflow": "disabled"}),
                    ),
                ],
                want: vec![("neuroflow", json!("disabled"))],
                want_overrides: vec![("neuroflow", "file:base.toml", "cli:--set")],
            },
        ];

        for case in cases {
            let resolved = resolve_config_layers(&defaults(), &case.layers);
            for (path, want_value) in &case.want {
                assert_eq!(
                    at(&resolved.merged, path),
                    Some(want_value),
                    "`{}`: wrong merged value at `{path}`",
                    case.name
                );
            }
            let got: Vec<(&str, &str, &str)> = resolved
                .overrides
                .iter()
                .map(|o| {
                    (
                        o.path.as_str(),
                        o.losing_layer.as_str(),
                        o.winning_layer.as_str(),
                    )
                })
                .collect();
            let want: Vec<(&str, &str, &str)> = case.want_overrides.clone();
            assert_eq!(got, want, "`{}`: wrong override record", case.name);
        }
    }

    #[test]
    fn an_override_record_carries_both_values_and_both_kinds() {
        let resolved = resolve_config_layers(
            &defaults(),
            &[
                statement(
                    ConfigLayerKind::File,
                    "file:base.toml",
                    json!({"world_width": 2048}),
                ),
                statement(
                    ConfigLayerKind::Environment,
                    "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                    json!({"world_width": 1024}),
                ),
            ],
        );
        assert_eq!(
            resolved.overrides,
            vec![ConfigFieldOverride {
                path: "world_width".to_owned(),
                losing_layer: "file:base.toml".to_owned(),
                losing_kind: ConfigLayerKind::File,
                losing_value: json!(2048),
                winning_layer: "env:SCRIPTBOTS_CONFIG_OVERRIDES".to_owned(),
                winning_kind: ConfigLayerKind::Environment,
                winning_value: json!(1024),
            }]
        );
    }

    #[test]
    fn a_restated_value_makes_the_restater_the_runner_up() {
        // File says 800, the environment restates 800, the CLI says 900. The
        // displacement names the environment — the most recent earlier writer —
        // exactly as the thread lane's `overridden` names the next-most-specific
        // declined layer rather than the whole queue behind it.
        let resolved = resolve_config_layers(
            &defaults(),
            &[
                statement(
                    ConfigLayerKind::File,
                    "file:base.toml",
                    json!({"world_width": 800}),
                ),
                statement(
                    ConfigLayerKind::Environment,
                    "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                    json!({"world_width": 800}),
                ),
                statement(
                    ConfigLayerKind::Cli,
                    "cli:--set",
                    json!({"world_width": 900}),
                ),
            ],
        );
        assert_eq!(resolved.overrides.len(), 1);
        assert_eq!(
            resolved.overrides[0].losing_layer,
            "env:SCRIPTBOTS_CONFIG_OVERRIDES"
        );
        assert_eq!(resolved.overrides[0].winning_layer, "cli:--set");
    }

    #[test]
    fn the_merged_tree_deserializes_into_the_real_config() {
        // The bead's acceptance chain, end to end on the REAL config type: a
        // scenario file that sets `world_width` loses to an environment
        // variable that sets it, which loses to a CLI flag.
        let defaults = serde_json::to_value(scriptbots_core::ScriptBotsConfig::default())
            .expect("default config serializes");
        let resolved = resolve_config_layers(
            &defaults,
            &[
                statement(
                    ConfigLayerKind::File,
                    "file:scenario.toml",
                    json!({"world_width": 2000}),
                ),
                statement(
                    ConfigLayerKind::Environment,
                    "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                    json!({"world_width": 1000}),
                ),
                statement(
                    ConfigLayerKind::Cli,
                    "cli:--set",
                    json!({"world_width": 500}),
                ),
            ],
        );
        let config: scriptbots_core::ScriptBotsConfig =
            serde_json::from_value(resolved.merged).expect("merged tree deserializes");
        assert_eq!(config.world_width, 500);
        assert_eq!(resolved.overrides.len(), 2);
        // The winning values must also form a VALID config — a chain proven only on a
        // tree the validator would reject proves nothing about a real run.
        config
            .validate()
            .expect("the resolved configuration must satisfy config validation");
    }

    #[test]
    fn resolution_is_a_pure_function_of_its_statements() {
        let layers = [
            statement(
                ConfigLayerKind::File,
                "file:base.toml",
                json!({"world_width": 2048, "neuroflow": {"enabled": true}}),
            ),
            statement(
                ConfigLayerKind::Environment,
                "env:SCRIPTBOTS_CONFIG_OVERRIDES",
                json!({"world_width": 1024}),
            ),
            statement(
                ConfigLayerKind::Cli,
                "cli:--set",
                json!({"neuroflow": {"enabled": false}}),
            ),
        ];
        let first = resolve_config_layers(&defaults(), &layers);
        let second = resolve_config_layers(&defaults(), &layers);
        assert_eq!(first, second);
    }

    #[test]
    fn canonical_bytes_are_construction_order_independent() {
        // The workspace dependency graph enables serde_json's `preserve_order`
        // feature, so a Value serializes in INSERTION order. Canonicalization
        // must therefore sort keys explicitly, or two statements with the same
        // content would digest differently depending on how they were
        // assembled. This test failed against the naive implementation and
        // exists so nobody reintroduces it.
        let mut forward = serde_json::Map::new();
        forward.insert("alpha".to_owned(), json!(1));
        forward.insert("omega".to_owned(), json!(2));
        let mut backward = serde_json::Map::new();
        backward.insert("omega".to_owned(), json!(2));
        backward.insert("alpha".to_owned(), json!(1));

        assert_eq!(
            canonical_layer_bytes(&JsonValue::Object(forward)),
            canonical_layer_bytes(&JsonValue::Object(backward)),
        );
        assert_ne!(
            canonical_layer_bytes(&json!({"alpha": 1})),
            canonical_layer_bytes(&json!({"alpha": 2})),
        );
    }

    #[test]
    fn wire_tags_are_stable_persisted_values() {
        // These land in run manifests and are compared across runs; changing
        // one is a schema decision, not a wording tweak.
        assert_eq!(ConfigLayerKind::Defaults.wire_tag(), "defaults");
        assert_eq!(ConfigLayerKind::File.wire_tag(), "file");
        assert_eq!(ConfigLayerKind::Environment.wire_tag(), "environment");
        assert_eq!(ConfigLayerKind::Cli.wire_tag(), "cli");
    }
}
