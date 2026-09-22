//! Highlight reel selection, scoring, and clip-window math (`bd-16g.9.1`, `bd-16g.9.3`).

use crate::narrative::{EventKind, EventRecord, SubjectRef};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;
use thiserror::Error;

/// Version identifier for reel event scoring formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ScoringVersion {
    /// Version 1 scoring model (population/extinction weighted).
    V1,
    /// Version 2 scoring model (speciation/regime/evolution weighted).
    V2,
}

/// Errors returned during reel selection and scoring.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReelSelectionError {
    /// A numeric field on an event is NaN or infinite.
    #[error("non-finite numeric field '{field}' in event at tick {tick}: {value}")]
    NonFiniteNumeric {
        /// Tick of the offending event.
        tick: u64,
        /// Name of the non-finite field.
        field: &'static str,
        /// Observed non-finite value.
        value: f64,
    },
    /// Event severity is negative.
    #[error("negative severity in event at tick {tick}: {severity}")]
    InvalidSeverity {
        /// Tick of the offending event.
        tick: u64,
        /// Observed negative severity.
        severity: f32,
    },
}

impl PartialEq for ReelSelectionError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::NonFiniteNumeric {
                    tick: t1,
                    field: f1,
                    value: v1,
                },
                Self::NonFiniteNumeric {
                    tick: t2,
                    field: f2,
                    value: v2,
                },
            ) => {
                t1 == t2
                    && f1 == f2
                    && (v1.to_bits() == v2.to_bits() || (v1.is_nan() && v2.is_nan()))
            }
            (
                Self::InvalidSeverity {
                    tick: t1,
                    severity: s1,
                },
                Self::InvalidSeverity {
                    tick: t2,
                    severity: s2,
                },
            ) => t1 == t2 && s1.to_bits() == s2.to_bits(),
            _ => false,
        }
    }
}

/// Reference to a narrative event embedded within a selected clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventRef {
    /// Tick on which the event occurred.
    pub tick: u64,
    /// Event kind representation.
    pub kind: String,
    /// Score assigned to this event.
    pub score: f32,
    /// Human-readable text describing the event.
    pub human_text: String,
    /// Stable event identity key.
    #[serde(default)]
    pub identity: String,
}

/// A selected clip window containing one or more narrative events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    /// 1-based rank of the clip in the reel.
    pub rank: usize,
    /// Starting tick of the clip (inclusive).
    pub start: u64,
    /// Ending tick of the clip (inclusive).
    pub end: u64,
    /// Events occurring within this clip window.
    pub events: Vec<EventRef>,
    /// Maximum score among member events.
    pub score: f32,
    /// Number of raw clip windows merged into this clip.
    pub merged_from: usize,
}

/// Configuration parameters for clip selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionConfig {
    /// Scoring formula version.
    pub scoring_version: ScoringVersion,
    /// Maximum number of clips to select.
    pub top_k: usize,
    /// Maximum clips allowed for any single event kind.
    pub max_per_kind: usize,
    /// Ticks to include before the event tick.
    pub pre_window_ticks: u64,
    /// Ticks to include after the event tick.
    pub post_window_ticks: u64,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            scoring_version: ScoringVersion::V1,
            top_k: 6,
            max_per_kind: 2,
            pre_window_ticks: 50,
            post_window_ticks: 100,
        }
    }
}

/// Canonical stable identity string for an `EventRecord`.
///
/// Combines tick, event kind, metric, subject, and human-readable text.
#[must_use]
pub fn event_identity(ev: &EventRecord) -> String {
    let subject_str = ev
        .subject
        .map_or_else(|| "none".to_string(), SubjectRef::to_db_string);
    format!(
        "{}:{}:{}:{}:{}",
        ev.tick.0,
        ev.kind.as_str(),
        ev.metric,
        subject_str,
        ev.human_text
    )
}

/// Compute weight for an event kind under a given scoring version.
#[must_use]
pub const fn kind_weight(kind: EventKind, version: ScoringVersion) -> f32 {
    match version {
        ScoringVersion::V1 => match kind {
            EventKind::Extinction => 3.0,
            EventKind::SpeciationHint => 2.5,
            EventKind::PredatorEmergence
            | EventKind::AltruismOnset
            | EventKind::ResourceCollapse
            | EventKind::CombatSurge => 2.0,
            EventKind::PopulationCrash | EventKind::EnergyCollapse => 1.8,
            EventKind::PopulationBoom | EventKind::EnergyRecovery => 1.5,
            EventKind::DietShift => 1.2,
            EventKind::RegimeChange => 1.0,
            EventKind::FloorEngaged => 0.8,
        },
        ScoringVersion::V2 => match kind {
            EventKind::SpeciationHint => 3.5,
            EventKind::RegimeChange => 3.0,
            EventKind::PredatorEmergence => 2.8,
            EventKind::AltruismOnset => 2.6,
            EventKind::DietShift => 2.4,
            EventKind::Extinction => 2.0,
            EventKind::ResourceCollapse => 1.8,
            EventKind::CombatSurge => 1.5,
            EventKind::PopulationCrash | EventKind::EnergyCollapse => 1.4,
            EventKind::PopulationBoom | EventKind::EnergyRecovery => 1.2,
            EventKind::FloorEngaged => 0.5,
        },
    }
}

/// Compute rarity multiplier based on occurrence count in run.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "V1 reel scoring rounds occurrence counts to f32 before its logarithm; widening changes versioned scores"
)]
pub fn rarity_weight(count_of_kind: usize) -> f32 {
    1.0 / (2.0 + count_of_kind as f32).log2()
}

/// Per-kind magnitude extrema and degenerate distribution status.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MagnitudeExtrema {
    /// Minimum observed magnitude for this event kind.
    pub min: f64,
    /// Maximum observed magnitude for this event kind.
    pub max: f64,
    /// True if all events of this kind have identical magnitude (or single event).
    pub is_degenerate: bool,
}

impl MagnitudeExtrema {
    /// Normalize a magnitude value into `[0.1, 1.0]`.
    ///
    /// For degenerate distributions (`max - min <= 1e-9`), returns `1.0`.
    /// Otherwise, maps linearly from `min` (0.1) to `max` (1.0).
    #[must_use]
    pub fn normalize(&self, magnitude: f64) -> f32 {
        if self.is_degenerate {
            1.0
        } else {
            let spread = self.max - self.min;
            let t = ((magnitude - self.min) / spread).clamp(0.0, 1.0);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "normalized float ratios fit within f32"
            )]
            let norm = 0.9f32.mul_add(t as f32, 0.1);
            norm.clamp(0.1, 1.0)
        }
    }
}

/// Score a single event record with explicit normalized magnitude.
#[must_use]
pub fn score_event_normalized(
    event: &EventRecord,
    norm_mag: f32,
    count_of_kind_in_run: usize,
    version: ScoringVersion,
) -> f32 {
    let kw = kind_weight(event.kind, version);
    let rw = rarity_weight(count_of_kind_in_run);
    let mag_norm = norm_mag.clamp(0.1, 1.0);
    let sev_norm = event.severity.clamp(0.1, 1.0);
    kw * mag_norm * sev_norm * rw
}

/// Score a single event record in isolation (degenerate distribution where normalized magnitude is 1.0).
#[must_use]
pub fn score_event(
    event: &EventRecord,
    count_of_kind_in_run: usize,
    version: ScoringVersion,
) -> f32 {
    score_event_normalized(event, 1.0, count_of_kind_in_run, version)
}

/// Compute a clamped clip tick range around an event tick.
#[must_use]
pub fn clip_window(tick: u64, pre: u64, post: u64, last_tick: u64) -> Range<u64> {
    let start = tick.saturating_sub(pre);
    let end = (tick.saturating_add(post)).min(last_tick);
    start..end
}

/// Merge overlapping or adjacent clip ranges into unified clips.
#[must_use]
pub fn merge_clips(
    scored_events: &[(f32, &EventRecord)],
    config: &SelectionConfig,
    last_tick: u64,
) -> Vec<Clip> {
    struct RawItem {
        range: Range<u64>,
        score: f32,
        identity: String,
        event: EventRef,
    }

    if scored_events.is_empty() {
        return Vec::new();
    }

    // Convert scored events to raw clip items with tick ranges
    let mut items: Vec<RawItem> = scored_events
        .iter()
        .map(|&(score, ev)| {
            let identity = event_identity(ev);
            RawItem {
                range: clip_window(
                    ev.tick.0,
                    config.pre_window_ticks,
                    config.post_window_ticks,
                    last_tick,
                ),
                score,
                identity: identity.clone(),
                event: EventRef {
                    tick: ev.tick.0,
                    kind: format!("{:?}", ev.kind),
                    score,
                    human_text: ev.human_text.clone(),
                    identity,
                },
            }
        })
        .collect();

    // Sort by range start, then range end, then tick, then identity
    items.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then_with(|| a.range.end.cmp(&b.range.end))
            .then_with(|| a.event.tick.cmp(&b.event.tick))
            .then_with(|| a.identity.cmp(&b.identity))
    });

    let mut merged: Vec<Clip> = Vec::new();

    for item in items {
        if let Some(last) = merged.last_mut() {
            // Check if ranges overlap or are adjacent
            if item.range.start <= last.end + 1 {
                last.end = last.end.max(item.range.end);
                last.score = last.score.max(item.score); // MAX NOT SUM
                last.events.push(item.event);
                last.merged_from += 1;
                continue;
            }
        }

        merged.push(Clip {
            rank: 0,
            start: item.range.start,
            end: item.range.end,
            events: vec![item.event],
            score: item.score,
            merged_from: 1,
        });
    }

    // Sort merged clips by max score descending, assigning 1-based ranks
    merged.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.start.cmp(&b.start))
            .then_with(|| a.end.cmp(&b.end))
    });
    for (idx, clip) in merged.iter_mut().enumerate() {
        clip.rank = idx + 1;
        // Sort internal events by tick ASC, score DESC, identity ASC
        clip.events.sort_by(|a, b| {
            a.tick
                .cmp(&b.tick)
                .then_with(|| b.score.total_cmp(&a.score))
                .then_with(|| a.identity.cmp(&b.identity))
        });
    }

    merged
}

/// Structured editorial decision log for auditing reel selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditorialDecisionLog {
    /// Optional run identifier.
    pub run_id: Option<String>,
    /// Scoring formula version used.
    pub scoring_version: ScoringVersion,
    /// Total events input to selection.
    pub total_events: usize,
    /// Event count per kind.
    pub kind_counts: BTreeMap<String, usize>,
    /// Magnitude bounds and degenerate status per kind.
    pub kind_magnitude_extrema: BTreeMap<String, MagnitudeExtrema>,
    /// All scored candidate events.
    pub scored_candidates: Vec<ScoredCandidateLog>,
    /// Selected candidates with assigned rank.
    pub selected_ranks: Vec<SelectedRankLog>,
    /// Merged clip windows.
    pub merged_windows: Vec<MergedWindowLog>,
    /// Events excluded from the final reel and reasons.
    pub exclusions: Vec<ExclusionLog>,
}

/// Audit record for a scored candidate event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredCandidateLog {
    /// Stable event identity.
    pub event_identity: String,
    /// Tick of occurrence.
    pub tick: u64,
    /// Event kind string.
    pub kind: String,
    /// Raw un-normalized magnitude.
    pub raw_magnitude: f64,
    /// Normalized magnitude in `[0.1, 1.0]`.
    pub normalized_magnitude: f32,
    /// Raw severity.
    pub raw_severity: f32,
    /// Normalized severity in `[0.1, 1.0]`.
    pub normalized_severity: f32,
    /// Kind weight factor.
    pub kind_weight: f32,
    /// Rarity multiplier.
    pub rarity_multiplier: f32,
    /// Final composite score.
    pub final_score: f32,
}

/// Audit record for an event selected into the top-K.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectedRankLog {
    /// 1-based rank among selected candidates.
    pub rank: usize,
    /// Stable event identity.
    pub event_identity: String,
    /// Tick of occurrence.
    pub tick: u64,
    /// Event kind string.
    pub kind: String,
    /// Final composite score.
    pub score: f32,
}

/// Audit record for a merged clip window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergedWindowLog {
    /// 1-based clip rank.
    pub rank: usize,
    /// Starting tick (inclusive).
    pub start_tick: u64,
    /// Ending tick (inclusive).
    pub end_tick: u64,
    /// Number of events in this clip window.
    pub event_count: usize,
    /// Maximum score among member events.
    pub max_score: f32,
    /// Identities of member events.
    pub event_identities: Vec<String>,
}

/// Audit record for an excluded event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExclusionLog {
    /// Stable event identity.
    pub event_identity: String,
    /// Event kind string.
    pub kind: String,
    /// Tick of occurrence.
    pub tick: u64,
    /// Final composite score.
    pub score: f32,
    /// Reason for exclusion.
    pub reason: String,
}

struct ScoredItem<'a> {
    score: f32,
    norm_mag: f32,
    sev_norm: f32,
    rw: f32,
    kw: f32,
    identity: String,
    event: &'a EventRecord,
}

/// Select top highlight clips and produce a structured editorial decision log.
///
/// # Errors
///
/// Returns [`ReelSelectionError`] if any event contains non-finite numeric fields
/// or invalid severity.
#[allow(clippy::too_many_lines)]
pub fn select_clips_with_log(
    events: &[EventRecord],
    last_tick: u64,
    config: &SelectionConfig,
    run_id: Option<&str>,
) -> Result<(Vec<Clip>, EditorialDecisionLog), ReelSelectionError> {
    // 1. Validate finite numbers across all events
    for ev in events {
        if !ev.magnitude.is_finite() {
            return Err(ReelSelectionError::NonFiniteNumeric {
                tick: ev.tick.0,
                field: "magnitude",
                value: ev.magnitude,
            });
        }
        if !ev.severity.is_finite() {
            return Err(ReelSelectionError::NonFiniteNumeric {
                tick: ev.tick.0,
                field: "severity",
                value: f64::from(ev.severity),
            });
        }
        if !ev.score.is_finite() {
            return Err(ReelSelectionError::NonFiniteNumeric {
                tick: ev.tick.0,
                field: "score",
                value: ev.score,
            });
        }
        if !ev.before.is_finite() {
            return Err(ReelSelectionError::NonFiniteNumeric {
                tick: ev.tick.0,
                field: "before",
                value: ev.before,
            });
        }
        if !ev.after.is_finite() {
            return Err(ReelSelectionError::NonFiniteNumeric {
                tick: ev.tick.0,
                field: "after",
                value: ev.after,
            });
        }
        if ev.severity < 0.0 {
            return Err(ReelSelectionError::InvalidSeverity {
                tick: ev.tick.0,
                severity: ev.severity,
            });
        }
    }

    let total_events = events.len();
    if events.is_empty() || config.top_k == 0 {
        let log = EditorialDecisionLog {
            run_id: run_id.map(str::to_string),
            scoring_version: config.scoring_version,
            total_events,
            kind_counts: BTreeMap::new(),
            kind_magnitude_extrema: BTreeMap::new(),
            scored_candidates: Vec::new(),
            selected_ranks: Vec::new(),
            merged_windows: Vec::new(),
            exclusions: Vec::new(),
        };
        return Ok((Vec::new(), log));
    }

    // 2. Count occurrences and find magnitude extrema per kind
    let mut kind_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut kind_magnitudes: BTreeMap<String, (f64, f64)> = BTreeMap::new();

    for ev in events {
        let kind_str = format!("{:?}", ev.kind);
        *kind_counts.entry(kind_str.clone()).or_insert(0) += 1;
        kind_magnitudes
            .entry(kind_str)
            .and_modify(|(min_m, max_m)| {
                *min_m = min_m.min(ev.magnitude);
                *max_m = max_m.max(ev.magnitude);
            })
            .or_insert((ev.magnitude, ev.magnitude));
    }

    let mut kind_magnitude_extrema: BTreeMap<String, MagnitudeExtrema> = BTreeMap::new();
    for (kind_str, (min_m, max_m)) in &kind_magnitudes {
        let is_degenerate = (*max_m - *min_m) <= 1e-9;
        kind_magnitude_extrema.insert(
            kind_str.clone(),
            MagnitudeExtrema {
                min: *min_m,
                max: *max_m,
                is_degenerate,
            },
        );
    }

    // 3. Score all candidates with normalized magnitude and stable identity
    let mut candidates: Vec<ScoredItem<'_>> = Vec::with_capacity(events.len());
    for ev in events {
        let kind_str = format!("{:?}", ev.kind);
        let count = kind_counts.get(&kind_str).copied().unwrap_or(1);
        let extrema = kind_magnitude_extrema
            .get(&kind_str)
            .copied()
            .unwrap_or(MagnitudeExtrema {
                min: ev.magnitude,
                max: ev.magnitude,
                is_degenerate: true,
            });
        let norm_mag = extrema.normalize(ev.magnitude);
        let kw = kind_weight(ev.kind, config.scoring_version);
        let rw = rarity_weight(count);
        let sev_norm = ev.severity.clamp(0.1, 1.0);
        let score = kw * norm_mag * sev_norm * rw;
        let identity = event_identity(ev);

        candidates.push(ScoredItem {
            score,
            norm_mag,
            sev_norm,
            rw,
            kw,
            identity,
            event: ev,
        });
    }

    // 4. Sort candidates: score DESC, tick ASC, identity ASC (tie-free total order)
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.event.tick.0.cmp(&b.event.tick.0))
            .then_with(|| a.identity.cmp(&b.identity))
    });

    let mut scored_candidates_log: Vec<ScoredCandidateLog> = Vec::with_capacity(candidates.len());
    for c in &candidates {
        scored_candidates_log.push(ScoredCandidateLog {
            event_identity: c.identity.clone(),
            tick: c.event.tick.0,
            kind: format!("{:?}", c.event.kind),
            raw_magnitude: c.event.magnitude,
            normalized_magnitude: c.norm_mag,
            raw_severity: c.event.severity,
            normalized_severity: c.sev_norm,
            kind_weight: c.kw,
            rarity_multiplier: c.rw,
            final_score: c.score,
        });
    }

    // 5. Select top-K respecting diversity limits
    let mut selected: Vec<(f32, &EventRecord)> = Vec::new();
    let mut selected_ranks_log: Vec<SelectedRankLog> = Vec::new();
    let mut exclusions_log: Vec<ExclusionLog> = Vec::new();
    let mut kind_selected: BTreeMap<String, usize> = BTreeMap::new();

    for c in candidates {
        let kind_str = format!("{:?}", c.event.kind);
        let count = kind_selected.entry(kind_str.clone()).or_insert(0);

        if selected.len() >= config.top_k {
            exclusions_log.push(ExclusionLog {
                event_identity: c.identity,
                kind: kind_str,
                tick: c.event.tick.0,
                score: c.score,
                reason: format!("below_top_k({})", config.top_k),
            });
            continue;
        }

        if *count >= config.max_per_kind {
            exclusions_log.push(ExclusionLog {
                event_identity: c.identity,
                kind: kind_str,
                tick: c.event.tick.0,
                score: c.score,
                reason: format!("exceeded_max_per_kind({})", config.max_per_kind),
            });
            continue;
        }

        *count += 1;
        let rank = selected.len() + 1;
        selected_ranks_log.push(SelectedRankLog {
            rank,
            event_identity: c.identity.clone(),
            tick: c.event.tick.0,
            kind: kind_str,
            score: c.score,
        });
        selected.push((c.score, c.event));
    }

    // 6. Merge clips
    let clips = merge_clips(&selected, config, last_tick);

    let mut merged_windows_log: Vec<MergedWindowLog> = Vec::with_capacity(clips.len());
    for clip in &clips {
        merged_windows_log.push(MergedWindowLog {
            rank: clip.rank,
            start_tick: clip.start,
            end_tick: clip.end,
            event_count: clip.events.len(),
            max_score: clip.score,
            event_identities: clip.events.iter().map(|e| e.identity.clone()).collect(),
        });
    }

    let decision_log = EditorialDecisionLog {
        run_id: run_id.map(str::to_string),
        scoring_version: config.scoring_version,
        total_events,
        kind_counts,
        kind_magnitude_extrema,
        scored_candidates: scored_candidates_log,
        selected_ranks: selected_ranks_log,
        merged_windows: merged_windows_log,
        exclusions: exclusions_log,
    };

    Ok((clips, decision_log))
}

/// Select top highlight clips from a list of narrative event records.
///
/// # Errors
///
/// Returns [`ReelSelectionError`] if any event contains non-finite numeric fields
/// or invalid severity.
pub fn select_clips(
    events: &[EventRecord],
    last_tick: u64,
    config: &SelectionConfig,
) -> Result<Vec<Clip>, ReelSelectionError> {
    select_clips_with_log(events, last_tick, config, None).map(|(clips, _)| clips)
}

#[cfg(test)]
#[allow(
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]
mod tests {
    use super::*;
    use crate::Tick;
    use crate::narrative::{EventKind, EventRecord, SubjectRef};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;

    fn sample_event(tick: u64, kind: EventKind, mag: f64) -> EventRecord {
        sample_event_full(
            tick,
            kind,
            mag,
            0.8,
            "population",
            None,
            &format!("{kind:?} at tick {tick}"),
        )
    }

    fn sample_event_full(
        tick: u64,
        kind: EventKind,
        mag: f64,
        severity: f32,
        metric: &str,
        subject: Option<SubjectRef>,
        text: &str,
    ) -> EventRecord {
        EventRecord {
            schema_version: 1,
            tick: Tick(tick),
            kind,
            severity,
            magnitude: mag,
            window: (tick.saturating_sub(10), tick),
            metric: metric.into(),
            before: 100.0,
            after: 50.0,
            score: 0.8,
            subject,
            human_text: text.into(),
        }
    }

    #[test]
    fn test_empty_events_returns_empty_reel() {
        let clips = select_clips(&[], 1000, &SelectionConfig::default()).expect("select clips");
        assert_eq!(clips.len(), 0);
    }

    #[test]
    fn test_clip_window_clamping() {
        let window = clip_window(10, 50, 100, 1000);
        assert_eq!(window, 0..110);

        let window_end = clip_window(980, 50, 100, 1000);
        assert_eq!(window_end, 930..1000);
    }

    #[test]
    fn test_non_finite_numeric_rejected() {
        let mut ev = sample_event(10, EventKind::PopulationCrash, f64::NAN);
        let err = select_clips(&[ev.clone()], 100, &SelectionConfig::default()).unwrap_err();
        assert_eq!(
            err,
            ReelSelectionError::NonFiniteNumeric {
                tick: 10,
                field: "magnitude",
                value: f64::NAN
            }
        );

        ev.magnitude = 10.0;
        ev.severity = f32::INFINITY;
        let err = select_clips(&[ev.clone()], 100, &SelectionConfig::default()).unwrap_err();
        assert_eq!(
            err,
            ReelSelectionError::NonFiniteNumeric {
                tick: 10,
                field: "severity",
                value: f64::INFINITY
            }
        );

        ev.severity = 0.5;
        ev.score = f64::NEG_INFINITY;
        let err = select_clips(&[ev], 100, &SelectionConfig::default()).unwrap_err();
        assert_eq!(
            err,
            ReelSelectionError::NonFiniteNumeric {
                tick: 10,
                field: "score",
                value: f64::NEG_INFINITY
            }
        );
    }

    #[test]
    fn test_negative_severity_rejected() {
        let mut ev = sample_event(25, EventKind::PopulationBoom, 15.0);
        ev.severity = -0.1;
        let err = select_clips(&[ev], 100, &SelectionConfig::default()).unwrap_err();
        assert_eq!(
            err,
            ReelSelectionError::InvalidSeverity {
                tick: 25,
                severity: -0.1
            }
        );
    }

    #[test]
    fn test_degenerate_magnitude_distribution_normalizes_to_one() {
        let events = vec![
            sample_event(100, EventKind::Extinction, 42.0),
            sample_event(200, EventKind::Extinction, 42.0),
        ];
        let (_, log) =
            select_clips_with_log(&events, 1000, &SelectionConfig::default(), None).unwrap();
        let extrema = log.kind_magnitude_extrema.get("Extinction").unwrap();
        assert!(extrema.is_degenerate);
        assert_eq!(extrema.min, 42.0);
        assert_eq!(extrema.max, 42.0);
        for cand in &log.scored_candidates {
            assert_eq!(cand.normalized_magnitude, 1.0);
        }
    }

    #[test]
    fn test_non_degenerate_magnitude_normalization_spread() {
        let events = vec![
            sample_event(100, EventKind::CombatSurge, 10.0),
            sample_event(200, EventKind::CombatSurge, 20.0),
            sample_event(300, EventKind::CombatSurge, 30.0),
        ];
        let (_, log) =
            select_clips_with_log(&events, 1000, &SelectionConfig::default(), None).unwrap();
        let extrema = log.kind_magnitude_extrema.get("CombatSurge").unwrap();
        assert!(!extrema.is_degenerate);
        assert_eq!(extrema.min, 10.0);
        assert_eq!(extrema.max, 30.0);

        let min_cand = log
            .scored_candidates
            .iter()
            .find(|c| c.tick == 100)
            .unwrap();
        assert!((min_cand.normalized_magnitude - 0.1).abs() < 1e-5);

        let mid_cand = log
            .scored_candidates
            .iter()
            .find(|c| c.tick == 200)
            .unwrap();
        assert!((mid_cand.normalized_magnitude - 0.55).abs() < 1e-5);

        let max_cand = log
            .scored_candidates
            .iter()
            .find(|c| c.tick == 300)
            .unwrap();
        assert!((max_cand.normalized_magnitude - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rarity_weighting_promotes_rare_event_400_common_vs_1_rare() {
        let mut events = Vec::new();
        // 400 common PopulationCrash events
        for i in 0..400 {
            events.push(sample_event(i * 10, EventKind::PopulationCrash, 20.0));
        }
        // 1 rare Extinction event
        events.push(sample_event(2500, EventKind::Extinction, 20.0));

        let clips = select_clips(&events, 5000, &SelectionConfig::default()).expect("select clips");
        assert_ne!(clips.len(), 0);
        let has_extinction = clips
            .iter()
            .any(|c| c.events.iter().any(|e| e.kind == "Extinction"));
        assert!(
            has_extinction,
            "rare extinction event must appear in top clips over 400 common events"
        );
        // Extinction should be the rank 1 clip
        assert_eq!(clips[0].events[0].kind, "Extinction");
    }

    #[test]
    fn test_diversity_limit_max_per_kind() {
        let mut events = Vec::new();
        for i in 0..10 {
            events.push(sample_event(
                i * 100,
                EventKind::PopulationBoom,
                (i + 1) as f64 * 10.0,
            ));
        }
        events.push(sample_event(2000, EventKind::Extinction, 50.0));

        let config = SelectionConfig {
            top_k: 6,
            max_per_kind: 2,
            ..SelectionConfig::default()
        };
        let (clips, log) = select_clips_with_log(&events, 3000, &config, None).unwrap();
        let boom_count: usize = clips
            .iter()
            .map(|c| {
                c.events
                    .iter()
                    .filter(|e| e.kind == "PopulationBoom")
                    .count()
            })
            .sum();
        assert_eq!(
            boom_count, 2,
            "diversity limit max_per_kind=2 must restrict population boom events to 2"
        );
        let boom_exclusions = log
            .exclusions
            .iter()
            .filter(|e| e.kind == "PopulationBoom" && e.reason.starts_with("exceeded_max_per_kind"))
            .count();
        assert_eq!(boom_exclusions, 8);
    }

    #[test]
    fn test_top_k_zero_and_exceeding_count() {
        let events = vec![
            sample_event(100, EventKind::PopulationCrash, 10.0),
            sample_event(400, EventKind::SpeciationHint, 20.0),
        ];

        let config_k0 = SelectionConfig {
            top_k: 0,
            ..SelectionConfig::default()
        };
        let clips0 = select_clips(&events, 500, &config_k0).unwrap();
        assert_eq!(clips0.len(), 0);

        let config_k100 = SelectionConfig {
            top_k: 100,
            ..SelectionConfig::default()
        };
        let clips100 = select_clips(&events, 500, &config_k100).unwrap();
        assert_eq!(clips100.len(), 2);
    }

    #[test]
    fn test_separated_and_chained_overlaps_merging() {
        let events = vec![
            sample_event(100, EventKind::PopulationCrash, 10.0), // window: 50..200
            sample_event(150, EventKind::DietShift, 20.0), // window: 100..250 (overlaps with 100)
            sample_event(220, EventKind::CombatSurge, 30.0), // window: 170..320 (overlaps with 150)
            sample_event(800, EventKind::Extinction, 40.0), // window: 750..900 (separated)
        ];
        let config = SelectionConfig {
            pre_window_ticks: 50,
            post_window_ticks: 100,
            top_k: 10,
            max_per_kind: 5,
            scoring_version: ScoringVersion::V1,
        };
        let clips = select_clips(&events, 1000, &config).unwrap();
        assert_eq!(
            clips.len(),
            2,
            "three chained events should merge into 1, fourth separated into 2nd"
        );

        let chained = clips.iter().find(|c| c.events.len() == 3).unwrap();
        assert_eq!(chained.start, 50);
        assert_eq!(chained.end, 320);
        assert_eq!(chained.merged_from, 3);

        let separated = clips.iter().find(|c| c.events.len() == 1).unwrap();
        assert_eq!(separated.start, 750);
        assert_eq!(separated.end, 900);
        assert_eq!(separated.merged_from, 1);
    }

    #[test]
    fn test_max_not_sum_merged_scores() {
        let ev1 = sample_event(100, EventKind::PopulationCrash, 10.0);
        let ev2 = sample_event(120, EventKind::CombatSurge, 20.0);
        let events = vec![ev1, ev2];
        let config = SelectionConfig {
            pre_window_ticks: 50,
            post_window_ticks: 50,
            ..SelectionConfig::default()
        };
        let (clips, log) = select_clips_with_log(&events, 500, &config, None).unwrap();
        assert_eq!(clips.len(), 1);
        let clip = &clips[0];
        let s1 = log
            .scored_candidates
            .iter()
            .find(|c| c.tick == 100)
            .unwrap()
            .final_score;
        let s2 = log
            .scored_candidates
            .iter()
            .find(|c| c.tick == 120)
            .unwrap()
            .final_score;
        let expected_max = s1.max(s2);
        assert_eq!(
            clip.score, expected_max,
            "merged clip score must be MAX of member events, not sum"
        );
    }

    #[test]
    fn test_annotation_order_within_clip() {
        let ev1 = sample_event_full(
            150,
            EventKind::CombatSurge,
            10.0,
            0.5,
            "combat",
            None,
            "combat event",
        );
        let ev2 = sample_event_full(
            100,
            EventKind::PopulationCrash,
            20.0,
            0.9,
            "pop",
            None,
            "pop crash",
        );
        let ev3 = sample_event_full(
            150,
            EventKind::Extinction,
            30.0,
            0.9,
            "ext",
            None,
            "extinction event",
        );
        let events = vec![ev1, ev2, ev3];

        let config = SelectionConfig {
            pre_window_ticks: 100,
            post_window_ticks: 100,
            ..SelectionConfig::default()
        };
        let clips = select_clips(&events, 500, &config).unwrap();
        assert_eq!(clips.len(), 1);
        let clip_events = &clips[0].events;
        assert_eq!(clip_events.len(), 3);
        // Annotation order: tick ASC, then score DESC, then identity ASC
        assert_eq!(clip_events[0].tick, 100);
        assert_eq!(clip_events[1].tick, 150);
        assert_eq!(clip_events[2].tick, 150);
        assert!(clip_events[1].score >= clip_events[2].score);
    }

    #[test]
    fn test_semantic_scoring_version_v1_vs_v2() {
        let events = vec![
            sample_event(100, EventKind::RegimeChange, 25.0),
            sample_event(200, EventKind::PopulationCrash, 25.0),
        ];

        let config_v1 = SelectionConfig {
            scoring_version: ScoringVersion::V1,
            ..SelectionConfig::default()
        };
        let (_, log_v1) = select_clips_with_log(&events, 500, &config_v1, None).unwrap();
        let regime_v1 = log_v1
            .scored_candidates
            .iter()
            .find(|c| c.kind == "RegimeChange")
            .unwrap();
        let pop_v1 = log_v1
            .scored_candidates
            .iter()
            .find(|c| c.kind == "PopulationCrash")
            .unwrap();
        // In V1, PopulationCrash (weight 1.8) beats RegimeChange (weight 1.0)
        assert!(pop_v1.final_score > regime_v1.final_score);

        let config_v2 = SelectionConfig {
            scoring_version: ScoringVersion::V2,
            ..SelectionConfig::default()
        };
        let (_, log_v2) = select_clips_with_log(&events, 500, &config_v2, None).unwrap();
        let regime_v2 = log_v2
            .scored_candidates
            .iter()
            .find(|c| c.kind == "RegimeChange")
            .unwrap();
        let pop_v2 = log_v2
            .scored_candidates
            .iter()
            .find(|c| c.kind == "PopulationCrash")
            .unwrap();
        // In V2, RegimeChange (weight 3.0) beats PopulationCrash (weight 1.4)
        assert!(regime_v2.final_score > pop_v2.final_score);
    }

    #[test]
    fn test_editorial_decision_log_recomputability() {
        let events = vec![
            sample_event(100, EventKind::PopulationCrash, 10.0),
            sample_event(400, EventKind::Extinction, 50.0),
            sample_event(700, EventKind::CombatSurge, 30.0),
        ];
        let config = SelectionConfig::default();
        let (clips, log) = select_clips_with_log(&events, 1000, &config, Some("run-42")).unwrap();

        assert_eq!(log.run_id.as_deref(), Some("run-42"));
        assert_eq!(log.total_events, 3);
        assert_eq!(log.selected_ranks.len(), 3);
        assert_eq!(log.merged_windows.len(), clips.len());
        for (i, clip) in clips.iter().enumerate() {
            let win = &log.merged_windows[i];
            assert_eq!(win.rank, clip.rank);
            assert_eq!(win.start_tick, clip.start);
            assert_eq!(win.end_tick, clip.end);
            assert_eq!(win.max_score, clip.score);
        }
    }

    #[test]
    fn test_checked_fixture_50_events_pinned_golden_and_100_permutations() {
        let mut fixture = Vec::with_capacity(50);
        let kinds = [
            EventKind::PopulationCrash,
            EventKind::PopulationBoom,
            EventKind::Extinction,
            EventKind::DietShift,
            EventKind::CombatSurge,
            EventKind::SpeciationHint,
            EventKind::PredatorEmergence,
        ];

        // Construct 50 deterministic events across 7 kinds with varied magnitudes, ticks, and severities
        for i in 0..50 {
            let kind = kinds[i % kinds.len()];
            let tick = (i as u64 + 1) * 60;
            let mag = ((i * 7 + 13) % 100) as f64 + 5.0;
            let sev = (((i * 11 + 17) % 90) as f32 / 100.0) + 0.1;
            let metric = match i % 4 {
                0 => "population",
                1 => "combat_activity",
                2 => "diet_herbivore_fraction",
                _ => "energy_mean",
            };
            let subject = if i % 3 == 0 {
                Some(SubjectRef::Agent(crate::AgentUid(i as u64 + 1000)))
            } else if i % 5 == 0 {
                Some(SubjectRef::Species(i as u64 + 200))
            } else {
                None
            };
            let text = format!("{kind:?} occurrence {i} at tick {tick}");
            fixture.push(sample_event_full(
                tick, kind, mag, sev, metric, subject, &text,
            ));
        }

        // Add 2 events with same tick and same magnitude/severity to test tie-breaker on identity
        fixture[5].tick = Tick(300);
        fixture[6].tick = Tick(300);

        let config = SelectionConfig {
            scoring_version: ScoringVersion::V1,
            top_k: 6,
            max_per_kind: 2,
            pre_window_ticks: 40,
            post_window_ticks: 80,
        };

        let (golden_clips, golden_log) =
            select_clips_with_log(&fixture, 5000, &config, Some("run_golden_50")).unwrap();
        let golden_json = serde_json::to_string_pretty(&golden_clips).unwrap();
        assert_eq!(golden_clips.len(), 4);

        // Pin the golden JSON representation
        let expected_golden_json = r#"[
  {
    "rank": 1,
    "start": 2240,
    "end": 2480,
    "events": [
      {
        "tick": 2280,
        "kind": "Extinction",
        "score": 0.6544921,
        "human_text": "Extinction occurrence 37 at tick 2280",
        "identity": "2280:extinction:combat_activity:none:Extinction occurrence 37 at tick 2280"
      },
      {
        "tick": 2400,
        "kind": "CombatSurge",
        "score": 0.56604725,
        "human_text": "CombatSurge occurrence 39 at tick 2400",
        "identity": "2400:combat_surge:energy_mean:agent:1039:CombatSurge occurrence 39 at tick 2400"
      }
    ],
    "score": 0.6544921,
    "merged_from": 2
  },
  {
    "rank": 2,
    "start": 260,
    "end": 380,
    "events": [
      {
        "tick": 300,
        "kind": "PredatorEmergence",
        "score": 0.5867647,
        "human_text": "PredatorEmergence occurrence 6 at tick 420",
        "identity": "300:predator_emergence:diet_herbivore_fraction:agent:1006:PredatorEmergence occurrence 6 at tick 420"
      }
    ],
    "score": 0.5867647,
    "merged_from": 1
  },
  {
    "rank": 3,
    "start": 740,
    "end": 860,
    "events": [
      {
        "tick": 780,
        "kind": "SpeciationHint",
        "score": 0.54417694,
        "human_text": "SpeciationHint occurrence 12 at tick 780",
        "identity": "780:speciation_hint:population:agent:1012:SpeciationHint occurrence 12 at tick 780"
      }
    ],
    "score": 0.54417694,
    "merged_from": 1
  },
  {
    "rank": 4,
    "start": 1220,
    "end": 1400,
    "events": [
      {
        "tick": 1260,
        "kind": "PredatorEmergence",
        "score": 0.40888837,
        "human_text": "PredatorEmergence occurrence 20 at tick 1260",
        "identity": "1260:predator_emergence:population:species:220:PredatorEmergence occurrence 20 at tick 1260"
      },
      {
        "tick": 1320,
        "kind": "PopulationCrash",
        "score": 0.40881404,
        "human_text": "PopulationCrash occurrence 21 at tick 1320",
        "identity": "1320:population_crash:combat_activity:agent:1021:PopulationCrash occurrence 21 at tick 1320"
      }
    ],
    "score": 0.40888837,
    "merged_from": 2
  }
]"#;

        assert_eq!(
            golden_json, expected_golden_json,
            "checked fixture output must match pinned golden JSON"
        );

        // Run 100 permutations using a seeded PRNG
        let mut rng = SmallRng::seed_from_u64(987_654_321);
        for perm in 0..100 {
            let mut shuffled = fixture.clone();
            shuffled.shuffle(&mut rng);
            let (shuffled_clips, shuffled_log) =
                select_clips_with_log(&shuffled, 5000, &config, Some("run_golden_50")).unwrap();
            assert_eq!(
                shuffled_clips, golden_clips,
                "permutation {perm} must produce identical clips"
            );
            assert_eq!(
                shuffled_log.selected_ranks, golden_log.selected_ranks,
                "permutation {perm} must produce identical selected ranks"
            );
            assert_eq!(
                shuffled_log.merged_windows, golden_log.merged_windows,
                "permutation {perm} must produce identical merged windows"
            );
        }
    }
}
