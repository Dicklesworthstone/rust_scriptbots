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
