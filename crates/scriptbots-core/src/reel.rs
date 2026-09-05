//! Highlight reel selection, scoring, and clip-window math (`bd-16g.9.1`).

use crate::narrative::{EventKind, EventRecord};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;

/// Version identifier for reel event scoring formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ScoringVersion {
    /// Version 1 scoring model.
    V1,
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

/// Compute weight for an event kind.
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

/// Score a single event record.
#[must_use]
pub fn score_event(
    event: &EventRecord,
    count_of_kind_in_run: usize,
    version: ScoringVersion,
) -> f32 {
    let kw = kind_weight(event.kind, version);
    let rw = rarity_weight(count_of_kind_in_run);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "V1 reel scoring narrows magnitude to f32 before normalization and multiplication"
    )]
    let mag_norm = (event.magnitude as f32).max(0.1);
    let sev_norm = event.severity.max(0.1);
    kw * mag_norm * sev_norm * rw
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
        event: EventRef,
    }

    if scored_events.is_empty() {
        return Vec::new();
    }

    // Convert scored events to raw clip items with tick ranges
    let mut items: Vec<RawItem> = scored_events
        .iter()
        .map(|&(score, ev)| RawItem {
            range: clip_window(
                ev.tick.0,
                config.pre_window_ticks,
                config.post_window_ticks,
                last_tick,
            ),
            score,
            event: EventRef {
                tick: ev.tick.0,
                kind: format!("{:?}", ev.kind),
                score,
                human_text: ev.human_text.clone(),
            },
        })
        .collect();

    // Sort by range start, then range end
    items.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then_with(|| a.range.end.cmp(&b.range.end))
    });

    let mut merged: Vec<Clip> = Vec::new();

    for item in items {
        if let Some(last) = merged.last_mut() {
            // Check if ranges overlap or are adjacent
            if item.range.start <= last.end + 1 {
                last.end = last.end.max(item.range.end);
                last.score = last.score.max(item.score);
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
    });
    for (idx, clip) in merged.iter_mut().enumerate() {
        clip.rank = idx + 1;
        // Sort internal events by tick
        clip.events.sort_by_key(|e| e.tick);
    }

    merged
}

/// Select top highlight clips from a list of narrative event records.
#[must_use]
pub fn select_clips(events: &[EventRecord], last_tick: u64, config: &SelectionConfig) -> Vec<Clip> {
    if events.is_empty() || config.top_k == 0 {
        return Vec::new();
    }

    // Count occurrences per kind
    let mut kind_counts: BTreeMap<String, usize> = BTreeMap::new();
    for ev in events {
        *kind_counts.entry(format!("{:?}", ev.kind)).or_insert(0) += 1;
    }

    // Score all events with deterministic tiebreakers
    let mut scored: Vec<(f32, &EventRecord)> = events
        .iter()
        .map(|ev| {
            let count = kind_counts
                .get(&format!("{:?}", ev.kind))
                .copied()
                .unwrap_or(1);
            let s = score_event(ev, count, config.scoring_version);
            (s, ev)
        })
        .collect();

    // Sort by (score DESC, tick ASC, kind ASC) for total deterministic ordering
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.tick.0.cmp(&b.1.tick.0))
            .then_with(|| format!("{:?}", a.1.kind).cmp(&format!("{:?}", b.1.kind)))
    });

    // Apply diversity limit: max_per_kind
    let mut selected: Vec<(f32, &EventRecord)> = Vec::new();
    let mut kind_selected: BTreeMap<String, usize> = BTreeMap::new();

    for (score, ev) in scored {
        let kind_str = format!("{:?}", ev.kind);
        let count = kind_selected.entry(kind_str).or_insert(0);
        if *count < config.max_per_kind {
            *count += 1;
            selected.push((score, ev));
            if selected.len() >= config.top_k {
                break;
            }
        }
    }

    merge_clips(&selected, config, last_tick)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tick;
    use crate::narrative::{EventKind, EventRecord};

    fn sample_event(tick: u64, kind: EventKind, mag: f64) -> EventRecord {
        EventRecord {
            schema_version: 1,
            tick: Tick(tick),
            kind,
            severity: 0.8,
            magnitude: mag,
            window: (tick.saturating_sub(10), tick),
            metric: "population".into(),
            before: 100.0,
            after: 50.0,
            score: 0.8,
            subject: None,
            human_text: format!("{kind:?} at tick {tick}"),
        }
    }

    #[test]
    fn test_empty_events_returns_empty_reel() {
        let clips = select_clips(&[], 1000, &SelectionConfig::default());
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
    fn test_rarity_weighting_promotes_rare_event() {
        let mut events = Vec::new();
        // 50 common PopulationCrash events
        for i in 0..50 {
            events.push(sample_event(i * 20, EventKind::PopulationCrash, 0.2));
        }
        // 1 rare Extinction event
        events.push(sample_event(500, EventKind::Extinction, 0.5));

        let clips = select_clips(&events, 1000, &SelectionConfig::default());
        assert_ne!(clips.len(), 0);
        let has_extinction = clips
            .iter()
            .any(|c| c.events.iter().any(|e| e.kind == "Extinction"));
        assert!(
            has_extinction,
            "rare extinction event must appear in top clips"
        );
    }

    #[test]
    fn test_input_order_independence() {
        let events = vec![
            sample_event(100, EventKind::PopulationCrash, 0.5),
            sample_event(200, EventKind::Extinction, 0.8),
            sample_event(300, EventKind::SpeciationHint, 0.6),
        ];

        let config = SelectionConfig::default();
        let clips1 = select_clips(&events, 1000, &config);

        let mut shuffled = events;
        shuffled.reverse();
        let clips2 = select_clips(&shuffled, 1000, &config);

        assert_eq!(clips1, clips2, "selection must be input-order independent");
    }

    #[test]
    fn test_overlapping_clip_merging() {
        let events = vec![
            sample_event(100, EventKind::PopulationCrash, 0.5),
            sample_event(120, EventKind::SpeciationHint, 0.6),
        ];
        let config = SelectionConfig {
            pre_window_ticks: 50,
            post_window_ticks: 50,
            ..SelectionConfig::default()
        };

        let clips = select_clips(&events, 1000, &config);
        assert_eq!(
            clips.len(),
            1,
            "overlapping events should merge into 1 clip"
        );
        assert_eq!(clips[0].merged_from, 2);
    }
}
