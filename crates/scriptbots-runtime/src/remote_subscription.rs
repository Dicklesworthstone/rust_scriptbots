//! Queue-free remote subscriptions over one coherent immutable host publication.
//!
//! Preparing a frame does not advance a delivery cursor. The transport commits
//! that cursor only after its bounded write succeeds. Latest-value channels
//! expose skipped publications; narrative pages expose exact retention gaps.

use crate::RenderSnapshot;
use crate::remote::{
    REMOTE_MAX_AGENTS, REMOTE_MAX_EVENT_LABEL_BYTES, REMOTE_MAX_EVENT_TEXT_BYTES,
    REMOTE_MAX_EVENTS, REMOTE_PROTOCOL_VERSION, RemoteEventGap, RemoteEventPage, RemoteEventRecord,
    RemoteFrame, RemoteFrameIdentity, RemoteFramePayload, RemoteProtocolError,
    RemoteSequencedEvent,
};
use scriptbots_core::{Tick, TickSummary, TickSummaryDto};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Maximum inbound subscription or control message, independently of frame size.
pub const REMOTE_MAX_REQUEST_BYTES: usize = 16 * 1024;

/// Independently negotiated presentation channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteChannel {
    /// Canonical tick aggregates.
    Telemetry,
    /// Dynamic-world snapshots for the existing browser renderers.
    Canvas,
    /// Canonical narrative records with session-bound ordinals.
    Events,
}

/// Decimal strings avoid rounding identities in JavaScript JSON clients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteResumeRequest {
    /// Exact host session that originally issued these cursors.
    pub session_id: String,
    /// Last successfully consumed publication in each latest-value channel.
    pub after_revision: Option<String>,
    /// Last successfully consumed narrative ordinal.
    pub after_event: Option<String>,
}

/// One explicit replacement of a connection's negotiated stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteSubscriptionRequest {
    /// Control protocol version; version one is the only accepted value.
    pub protocol_version: u16,
    /// Optional client correlation, not a mutating host command identity.
    pub request_id: Option<String>,
    /// Must be exactly `subscribe`.
    pub command: String,
    /// One through three unique channels; omitted channels do no projection work.
    pub channels: Vec<RemoteChannel>,
    /// Requested upper bound on sampling frequency, from one to 120 Hz.
    #[serde(default = "default_max_fps")]
    pub max_fps: u16,
    /// Optional explicit continuation; foreign and future cursors are refused.
    pub resume: Option<RemoteResumeRequest>,
}

/// Consumer credit for exactly one received presentation frame.
///
/// These acknowledgements never submit an owner command. Decimal strings retain
/// every identity bit in browser clients; only narrative frames carry an event
/// cursor, including pages that report a retention gap without retained records.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteFrameAck {
    /// Version of the presentation protocol being acknowledged.
    pub protocol_version: u16,
    /// Must be exactly `stream_ack`.
    pub command: String,
    /// Exact host session that issued the consumed frame.
    pub session_id: String,
    /// Channel whose single outstanding frame was consumed.
    pub channel: RemoteChannel,
    /// Exact publication revision carried by that frame.
    pub publication_revision: String,
    /// Cursor reached by an events page; absent for telemetry and canvas.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_ack_event_cursor"
    )]
    pub after_event: Option<String>,
}

// An omitted cursor and an explicitly null cursor have different wire meanings:
// the former is required for snapshot channels; the latter is never an identity.
fn deserialize_ack_event_cursor<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    String::deserialize(deserializer).map(Some)
}

impl RemoteFrameAck {
    /// Parse one complete, bounded consumer acknowledgement.
    ///
    /// # Errors
    /// Refuses unsupported versions, unknown or duplicate fields, noncanonical
    /// decimal identities, and an event cursor on the wrong channel.
    pub fn parse(text: &str) -> Result<Self, RemoteSubscriptionError> {
        if text.len() > REMOTE_MAX_REQUEST_BYTES {
            return Err(RemoteSubscriptionError::Request(
                "frame acknowledgement exceeds 16 KiB".into(),
            ));
        }
        let acknowledgement: Self = serde_json::from_str(text)
            .map_err(|error| RemoteSubscriptionError::Request(error.to_string()))?;
        acknowledgement.validated_identity()?;
        Ok(acknowledgement)
    }

    fn validated_identity(&self) -> Result<(u128, u64, Option<u64>), RemoteSubscriptionError> {
        if self.protocol_version != REMOTE_PROTOCOL_VERSION || self.command != "stream_ack" {
            return Err(RemoteSubscriptionError::Request(
                "expected protocol_version 1 and command stream_ack".into(),
            ));
        }
        if (self.channel == RemoteChannel::Events) != self.after_event.is_some() {
            return Err(RemoteSubscriptionError::Request(
                "after_event is required for events acknowledgements and forbidden for other channels"
                    .into(),
            ));
        }
        Ok((
            parse_decimal(&self.session_id, "session_id")?,
            parse_decimal(&self.publication_revision, "publication_revision")?,
            self.after_event
                .as_deref()
                .map(|value| parse_decimal(value, "after_event"))
                .transpose()?,
        ))
    }
}

const fn default_max_fps() -> u16 {
    60
}

impl RemoteSubscriptionRequest {
    /// Parse and validate the complete bounded JSON request.
    ///
    /// # Errors
    /// Refuses unsupported versions, unknown fields, duplicate channels, invalid
    /// rates, excessive text and noncanonical decimal cursor strings.
    pub fn parse(text: &str) -> Result<Self, RemoteSubscriptionError> {
        if text.len() > REMOTE_MAX_REQUEST_BYTES {
            return Err(RemoteSubscriptionError::Request(
                "subscription exceeds 16 KiB".into(),
            ));
        }
        let mut request: Self = serde_json::from_str(text)
            .map_err(|error| RemoteSubscriptionError::Request(error.to_string()))?;
        if let Some(id) = &mut request.request_id {
            *id = id.trim().to_owned();
        }
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), RemoteSubscriptionError> {
        if self.protocol_version != REMOTE_PROTOCOL_VERSION || self.command != "subscribe" {
            return Err(RemoteSubscriptionError::Request(
                "expected protocol_version 1 and command subscribe".into(),
            ));
        }
        if self
            .request_id
            .as_ref()
            .is_some_and(|id| id.is_empty() || id.len() > 1024 || id.trim() != id)
        {
            return Err(RemoteSubscriptionError::Request(
                "request_id must contain 1..=1024 bytes".into(),
            ));
        }
        if self.channels.is_empty()
            || self.channels.len() > 3
            || self
                .channels
                .iter()
                .enumerate()
                .any(|(index, channel)| self.channels[..index].contains(channel))
        {
            return Err(RemoteSubscriptionError::Request(
                "channels must contain 1..=3 unique telemetry, canvas or events entries".into(),
            ));
        }
        if !(1..=120).contains(&self.max_fps) {
            return Err(RemoteSubscriptionError::Request(
                "max_fps must be in 1..=120".into(),
            ));
        }
        if let Some(resume) = &self.resume {
            parse_decimal::<u128>(&resume.session_id, "session_id")?;
            if let Some(revision) = &resume.after_revision {
                parse_decimal::<u64>(revision, "after_revision")?;
            }
            if let Some(event) = &resume.after_event {
                parse_decimal::<u64>(event, "after_event")?;
            }
        }
        Ok(())
    }
}

fn parse_decimal<T>(text: &str, field: &str) -> Result<T, RemoteSubscriptionError>
where
    T: std::str::FromStr + std::fmt::Display,
{
    let parsed = text
        .parse::<T>()
        .ok()
        .filter(|value| value.to_string() == text);
    parsed.ok_or_else(|| {
        RemoteSubscriptionError::Request(format!(
            "{field} must be a canonical unsigned decimal string within its integer range",
        ))
    })
}

/// Explicit refusal at a subscription or delivery boundary.
#[derive(Debug, Error)]
pub enum RemoteSubscriptionError {
    /// Malformed request; no existing subscription was changed.
    #[error("invalid remote subscription: {0}")]
    Request(String),
    /// Cursors cannot be silently reused across owner sessions.
    #[error("remote session changed: expected {expected}, observed {actual}")]
    SessionChanged {
        /// Session identified by the client.
        expected: u128,
        /// Session identified by the publication.
        actual: u128,
    },
    /// A cursor claims data that this session has not published.
    #[error("{field} cursor {requested} is ahead of published {current}")]
    CursorAhead {
        /// Cursor axis.
        field: &'static str,
        /// Client's requested position.
        requested: u64,
        /// Current published position.
        current: u64,
    },
    /// The source cannot be represented within the bounded remote protocol.
    #[error("remote projection exceeds {0}")]
    ProjectionLimit(&'static str),
    /// Transport attempted to commit a stale or unrelated prepared frame.
    #[error("remote delivery cursor mismatch")]
    DeliveryMismatch,
    /// Projection violated the shared frame contract.
    #[error(transparent)]
    Protocol(#[from] RemoteProtocolError),
}

#[derive(Debug, Clone, Copy)]
struct PendingRemoteDelivery {
    publication_revision: u64,
    after_event: Option<u64>,
    sent_at: Duration,
}

/// Bounded consumer credit: at most one outstanding frame in each channel.
///
/// Only scalar identities and monotonic times are retained. A slow browser
/// cannot authorize another frame merely by draining the network socket: its
/// consumer must acknowledge the exact frame first. The owner remains free to
/// publish newer snapshots while a channel has no available credit.
#[derive(Debug)]
pub struct RemoteDeliveryWindow {
    session_id: u128,
    timeout: Duration,
    pending: [Option<PendingRemoteDelivery>; 3],
    last_observed: Option<Duration>,
}

impl RemoteDeliveryWindow {
    /// Start an empty delivery window for one host session.
    #[must_use]
    pub const fn new(session_id: u128, timeout: Duration) -> Self {
        Self {
            session_id,
            timeout,
            pending: [None; 3],
            last_observed: None,
        }
    }

    /// Channels whose next projection must wait for consumer acknowledgement.
    #[must_use]
    pub fn awaiting_channels(&self) -> Vec<RemoteChannel> {
        [
            RemoteChannel::Telemetry,
            RemoteChannel::Canvas,
            RemoteChannel::Events,
        ]
        .into_iter()
        .filter(|channel| self.pending[channel_index(*channel)].is_some())
        .collect()
    }

    /// Whether every channel has consumer credit available.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.iter().all(Option::is_none)
    }

    /// Record a successfully written, already validated frame in constant time.
    ///
    /// The transport validates and encodes the payload before writing it. This
    /// method extracts only its channel, publication and final event ordinal; it
    /// never repeats a canvas or narrative validation scan on an async worker.
    ///
    /// # Errors
    /// Refuses a foreign session, a channel without available credit, or a
    /// regressed monotonic clock. Refusal leaves all existing credit unchanged.
    pub fn mark_sent(
        &mut self,
        frame: &RemoteFrame,
        now: Duration,
    ) -> Result<(), RemoteSubscriptionError> {
        if frame.identity.session_id != self.session_id {
            return Err(RemoteSubscriptionError::SessionChanged {
                expected: self.session_id,
                actual: frame.identity.session_id,
            });
        }
        self.validate_time(now)?;
        let (channel, after_event) = match &frame.payload {
            RemoteFramePayload::Telemetry(_) => (RemoteChannel::Telemetry, None),
            RemoteFramePayload::Canvas(_) => (RemoteChannel::Canvas, None),
            RemoteFramePayload::Events(page) => {
                (RemoteChannel::Events, Some(page.next_after_event()))
            }
        };
        let pending = &mut self.pending[channel_index(channel)];
        if pending.is_some() {
            return Err(RemoteSubscriptionError::DeliveryMismatch);
        }
        *pending = Some(PendingRemoteDelivery {
            publication_revision: frame.identity.publication_revision,
            after_event,
            sent_at: now,
        });
        self.last_observed = Some(now);
        Ok(())
    }

    /// Release exactly the acknowledged channel and return its consumer latency.
    ///
    /// # Errors
    /// Refuses malformed, foreign, missing, duplicate or mismatched identities,
    /// regressed time, and acknowledgement at or after the delivery deadline.
    /// None of these refusals release credit or move the clock observation.
    pub fn acknowledge(
        &mut self,
        acknowledgement: &RemoteFrameAck,
        now: Duration,
    ) -> Result<Duration, RemoteSubscriptionError> {
        let (session_id, publication_revision, after_event) =
            acknowledgement.validated_identity()?;
        if session_id != self.session_id {
            return Err(RemoteSubscriptionError::SessionChanged {
                expected: self.session_id,
                actual: session_id,
            });
        }
        self.validate_time(now)?;
        let index = channel_index(acknowledgement.channel);
        let pending = self.pending[index].ok_or(RemoteSubscriptionError::DeliveryMismatch)?;
        if pending.publication_revision != publication_revision
            || pending.after_event != after_event
        {
            return Err(RemoteSubscriptionError::DeliveryMismatch);
        }
        let elapsed = now
            .checked_sub(pending.sent_at)
            .ok_or_else(delivery_clock_regression)?;
        if elapsed >= self.timeout {
            return Err(RemoteSubscriptionError::Request(
                "remote frame acknowledgement exceeded its delivery deadline".into(),
            ));
        }
        self.pending[index] = None;
        self.last_observed = Some(now);
        Ok(elapsed)
    }

    /// Whether any outstanding frame has reached its consumer deadline.
    ///
    /// A regressed clock with outstanding credit fails closed as expired;
    /// subtraction never wraps or treats a future send as a fresh delivery.
    #[must_use]
    pub fn expired(&self, now: Duration) -> bool {
        !self.is_empty()
            && (self.validate_time(now).is_err()
                || self.pending.iter().flatten().any(|pending| {
                    now.checked_sub(pending.sent_at)
                        .is_none_or(|elapsed| elapsed >= self.timeout)
                }))
    }

    fn validate_time(&self, now: Duration) -> Result<(), RemoteSubscriptionError> {
        if self
            .last_observed
            .is_some_and(|previous| now.checked_sub(previous).is_none())
        {
            return Err(delivery_clock_regression());
        }
        Ok(())
    }
}

const fn channel_index(channel: RemoteChannel) -> usize {
    match channel {
        RemoteChannel::Telemetry => 0,
        RemoteChannel::Canvas => 1,
        RemoteChannel::Events => 2,
    }
}

fn delivery_clock_regression() -> RemoteSubscriptionError {
    RemoteSubscriptionError::Request("remote delivery clock moved backwards".into())
}

/// Per-connection cursors; no snapshots or outbound messages are retained here.
#[derive(Debug, Clone)]
pub struct RemoteSubscription {
    session_id: u128,
    channels: Vec<RemoteChannel>,
    max_fps: u16,
    telemetry_revision: Option<u64>,
    canvas_revision: Option<u64>,
    after_event: u64,
    initial_events: bool,
}

impl RemoteSubscription {
    /// Validate a replacement subscription against one immutable publication.
    ///
    /// # Errors
    /// Refuses malformed requests and foreign/future resume cursors.
    pub fn new(
        request: &RemoteSubscriptionRequest,
        snapshot: &RenderSnapshot,
    ) -> Result<Self, RemoteSubscriptionError> {
        request.validate()?;
        let identity = RemoteFrameIdentity::from(snapshot);
        let (_, latest_event) = event_bounds(snapshot)?;
        let (after_revision, after_event) = if let Some(resume) = &request.resume {
            let session_id = parse_decimal(&resume.session_id, "session_id")?;
            if session_id != identity.session_id {
                return Err(RemoteSubscriptionError::SessionChanged {
                    expected: session_id,
                    actual: identity.session_id,
                });
            }
            let revision = resume
                .after_revision
                .as_deref()
                .map(|value| parse_decimal(value, "after_revision"))
                .transpose()?;
            let event = resume
                .after_event
                .as_deref()
                .map(|value| parse_decimal(value, "after_event"))
                .transpose()?
                .unwrap_or(0);
            if let Some(revision) = revision {
                reject_ahead("after_revision", revision, identity.publication_revision)?;
            }
            reject_ahead("after_event", event, latest_event)?;
            (revision, event)
        } else {
            (None, 0)
        };
        Ok(Self {
            session_id: identity.session_id,
            channels: request.channels.clone(),
            max_fps: request.max_fps,
            telemetry_revision: after_revision,
            canvas_revision: after_revision,
            after_event,
            initial_events: true,
        })
    }

    /// Negotiated channels in the client's requested order.
    #[must_use]
    pub fn channels(&self) -> &[RemoteChannel] {
        &self.channels
    }

    /// Requested minimum interval; transports may back off for a slow consumer.
    #[must_use]
    pub fn interval(&self) -> Duration {
        Duration::from_secs_f64(1.0 / f64::from(self.max_fps))
    }

    /// Current narrative cursor, advanced only by a successful delivered page.
    #[must_use]
    pub const fn after_event(&self) -> u64 {
        self.after_event
    }

    /// Prepare at most one bounded frame for a selected channel, without mutation.
    ///
    /// # Errors
    /// Refuses a changed session, regressed source, oversized projection or invalid
    /// canonical payload. An unselected or unchanged channel returns `None`.
    pub fn next_frame(
        &self,
        channel: RemoteChannel,
        snapshot: &RenderSnapshot,
    ) -> Result<Option<RemoteFrame>, RemoteSubscriptionError> {
        if !self.channels.contains(&channel) {
            return Ok(None);
        }
        let identity = RemoteFrameIdentity::from(snapshot);
        if identity.session_id != self.session_id {
            return Err(RemoteSubscriptionError::SessionChanged {
                expected: self.session_id,
                actual: identity.session_id,
            });
        }
        let previous_revision = match channel {
            RemoteChannel::Telemetry => self.telemetry_revision,
            RemoteChannel::Canvas => self.canvas_revision,
            RemoteChannel::Events => None,
        };
        if let Some(previous) = previous_revision {
            reject_ahead("after_revision", previous, identity.publication_revision)?;
            if previous == identity.publication_revision {
                return Ok(None);
            }
        }
        let payload = match channel {
            RemoteChannel::Telemetry => RemoteFramePayload::Telemetry(tick_summary(snapshot)),
            RemoteChannel::Canvas => {
                if snapshot.world.agents.len() > REMOTE_MAX_AGENTS {
                    return Err(RemoteSubscriptionError::ProjectionLimit(
                        "canvas agent limit",
                    ));
                }
                RemoteFramePayload::Canvas(snapshot.world.clone())
            }
            RemoteChannel::Events => {
                let (earliest_retained, latest_event) = event_bounds(snapshot)?;
                reject_ahead("after_event", self.after_event, latest_event)?;
                if self.after_event == latest_event && !self.initial_events {
                    return Ok(None);
                }
                let gap = (self.after_event < snapshot.narrative_dropped_events).then(|| {
                    RemoteEventGap {
                        first_missing: self.after_event + 1,
                        last_missing: snapshot.narrative_dropped_events,
                    }
                });
                let offset = self
                    .after_event
                    .saturating_sub(snapshot.narrative_dropped_events);
                let offset = usize::try_from(offset)
                    .map_err(|_| RemoteSubscriptionError::ProjectionLimit("event ordinal range"))?;
                let mut events =
                    Vec::with_capacity(REMOTE_MAX_EVENTS.min(snapshot.narrative_events.len()));
                for (index, record) in snapshot
                    .narrative_events
                    .iter()
                    .enumerate()
                    .skip(offset)
                    .take(REMOTE_MAX_EVENTS)
                {
                    if record.metric.len() > REMOTE_MAX_EVENT_LABEL_BYTES
                        || record.human_text.len() > REMOTE_MAX_EVENT_TEXT_BYTES
                    {
                        return Err(RemoteSubscriptionError::ProjectionLimit(
                            "narrative text limit",
                        ));
                    }
                    events.push(RemoteSequencedEvent {
                        sequence: earliest_retained + index as u64,
                        record: RemoteEventRecord::from(record),
                    });
                }
                RemoteFramePayload::Events(RemoteEventPage {
                    after_event: self.after_event,
                    earliest_retained,
                    latest_event,
                    gap,
                    events,
                })
            }
        };
        let frame = RemoteFrame {
            identity,
            playback: snapshot.playback,
            coalesced_publications: previous_revision
                .map_or(0, |previous| identity.publication_revision - previous - 1),
            payload,
        };
        frame.validate()?;
        Ok(Some(frame))
    }

    /// Commit exactly the frame whose socket write completed successfully.
    ///
    /// # Errors
    /// Refuses unrelated sessions, disabled channels, repeated snapshot revisions
    /// and event pages that do not start at this connection's current cursor.
    pub fn delivered(&mut self, frame: &RemoteFrame) -> Result<(), RemoteSubscriptionError> {
        frame.validate()?;
        if frame.identity.session_id != self.session_id {
            return Err(RemoteSubscriptionError::DeliveryMismatch);
        }
        let channel = match &frame.payload {
            RemoteFramePayload::Telemetry(_) => RemoteChannel::Telemetry,
            RemoteFramePayload::Canvas(_) => RemoteChannel::Canvas,
            RemoteFramePayload::Events(_) => RemoteChannel::Events,
        };
        if !self.channels.contains(&channel) {
            return Err(RemoteSubscriptionError::DeliveryMismatch);
        }
        match &frame.payload {
            RemoteFramePayload::Telemetry(_) | RemoteFramePayload::Canvas(_) => {
                let previous = if channel == RemoteChannel::Telemetry {
                    &mut self.telemetry_revision
                } else {
                    &mut self.canvas_revision
                };
                if previous.is_some_and(|revision| revision >= frame.identity.publication_revision)
                {
                    return Err(RemoteSubscriptionError::DeliveryMismatch);
                }
                *previous = Some(frame.identity.publication_revision);
            }
            RemoteFramePayload::Events(page) => {
                if page.after_event != self.after_event
                    || (!self.initial_events && page.next_after_event() == self.after_event)
                {
                    return Err(RemoteSubscriptionError::DeliveryMismatch);
                }
                self.after_event = page.next_after_event();
                self.initial_events = false;
            }
        }
        Ok(())
    }
}

const fn reject_ahead(
    field: &'static str,
    requested: u64,
    current: u64,
) -> Result<(), RemoteSubscriptionError> {
    if requested > current {
        return Err(RemoteSubscriptionError::CursorAhead {
            field,
            requested,
            current,
        });
    }
    Ok(())
}

fn event_bounds(snapshot: &RenderSnapshot) -> Result<(u64, u64), RemoteSubscriptionError> {
    let limit = || RemoteSubscriptionError::ProjectionLimit("narrative ordinal range");
    let retained = u64::try_from(snapshot.narrative_events.len()).map_err(|_| limit())?;
    let earliest = snapshot
        .narrative_dropped_events
        .checked_add(1)
        .ok_or_else(limit)?;
    let latest = snapshot
        .narrative_dropped_events
        .checked_add(retained)
        .ok_or_else(limit)?;
    Ok((earliest, latest))
}

fn tick_summary(snapshot: &RenderSnapshot) -> TickSummaryDto {
    TickSummaryDto::from(
        snapshot
            .completed_summary
            .clone()
            .or_else(|| snapshot.summary_history.last().cloned())
            .unwrap_or_else(|| TickSummary {
                tick: Tick(snapshot.world.tick),
                agent_count: snapshot.world.agents.len(),
                births: snapshot.world.summary.births,
                deaths: snapshot.world.summary.deaths,
                total_energy: snapshot.world.summary.total_energy,
                average_energy: snapshot.world.summary.average_energy,
                average_health: snapshot.world.summary.average_health,
                max_age: snapshot
                    .world
                    .agents
                    .iter()
                    .map(|agent| agent.age)
                    .max()
                    .unwrap_or(0),
                spike_hits: 0,
            }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::{decode_remote_frame, encode_remote_frame};
    use crate::{
        HostCore, HostCoreOptions, HostSessionId, ManualInstant, NullFrontend, PlaybackSnapshot,
    };
    use scriptbots_core::narrative::{EVENT_RECORD_SCHEMA_VERSION, EventKind, EventRecord};
    use scriptbots_core::{AgentData, ScriptBotsConfig, WorldState};
    use std::sync::Arc;

    fn owner() -> HostCore {
        let mut world = WorldState::new(ScriptBotsConfig {
            world_width: 64,
            world_height: 64,
            food_cell_size: 16,
            rng_seed: Some(7241),
            persistence_interval: 0,
            closed: true,
            ..ScriptBotsConfig::default()
        })
        .unwrap();
        world
            .try_spawn_agent(AgentData {
                age: 17,
                ..AgentData::default()
            })
            .unwrap();
        HostCore::new(
            HostSessionId::new(891),
            world,
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: true,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
        )
        .unwrap()
    }

    fn request(channels: Vec<RemoteChannel>) -> RemoteSubscriptionRequest {
        RemoteSubscriptionRequest {
            protocol_version: 1,
            request_id: Some("viewer-1".into()),
            command: "subscribe".into(),
            channels,
            max_fps: 60,
            resume: None,
        }
    }

    fn narrative_record(tick: u64) -> EventRecord {
        EventRecord {
            schema_version: EVENT_RECORD_SCHEMA_VERSION,
            tick: Tick(tick),
            kind: EventKind::PopulationBoom,
            severity: 0.5,
            magnitude: 4.0,
            window: (0, tick),
            metric: "population".into(),
            before: 1.0,
            after: 5.0,
            score: 0.75,
            subject: None,
            human_text: "The population increased.".into(),
        }
    }

    fn frame_ack(frame: &RemoteFrame) -> RemoteFrameAck {
        let (channel, after_event) = match &frame.payload {
            RemoteFramePayload::Telemetry(_) => (RemoteChannel::Telemetry, None),
            RemoteFramePayload::Canvas(_) => (RemoteChannel::Canvas, None),
            RemoteFramePayload::Events(page) => (
                RemoteChannel::Events,
                Some(page.next_after_event().to_string()),
            ),
        };
        RemoteFrameAck {
            protocol_version: REMOTE_PROTOCOL_VERSION,
            command: "stream_ack".into(),
            session_id: frame.identity.session_id.to_string(),
            channel,
            publication_revision: frame.identity.publication_revision.to_string(),
            after_event,
        }
    }

    #[test]
    fn remote_delivery_window_preserves_latest_value_while_consumer_has_no_credit() {
        let mut owner = owner();
        let mut frontend = NullFrontend::new(owner.local_port(), 10);
        let initial = owner.latest_snapshot();
        let mut stream =
            RemoteSubscription::new(&request(vec![RemoteChannel::Canvas]), &initial).unwrap();
        let first = stream
            .next_frame(RemoteChannel::Canvas, &initial)
            .unwrap()
            .unwrap();
        let mut window =
            RemoteDeliveryWindow::new(first.identity.session_id, Duration::from_secs(5));
        assert!(window.is_empty());
        assert!(!window.expired(Duration::MAX));
        window.mark_sent(&first, Duration::from_secs(1)).unwrap();
        stream.delivered(&first).unwrap();
        assert_eq!(window.awaiting_channels(), vec![RemoteChannel::Canvas]);
        assert!(window.mark_sent(&first, Duration::from_secs(2)).is_err());

        let acknowledgement = frame_ack(&first);
        for malformed in [
            RemoteFrameAck {
                channel: RemoteChannel::Telemetry,
                ..acknowledgement.clone()
            },
            RemoteFrameAck {
                session_id: (first.identity.session_id + 1).to_string(),
                ..acknowledgement.clone()
            },
            RemoteFrameAck {
                publication_revision: (first.identity.publication_revision + 1).to_string(),
                ..acknowledgement.clone()
            },
        ] {
            assert!(
                window
                    .acknowledge(&malformed, Duration::from_secs(2))
                    .is_err()
            );
            assert_eq!(window.awaiting_channels(), vec![RemoteChannel::Canvas]);
        }

        // Owner progress does not need consumer credit. The socket lane holds
        // one scalar identity while several scientific publications replace it.
        for now in 1..=3 {
            frontend.step().unwrap();
            for _ in 0..4 {
                frontend
                    .drive_at(&mut owner, ManualInstant::from_nanos(now))
                    .unwrap();
            }
        }
        let latest = owner.latest_snapshot();
        assert_eq!(latest.world.tick, 3);
        let newest = stream
            .next_frame(RemoteChannel::Canvas, &latest)
            .unwrap()
            .unwrap();
        assert!(window.mark_sent(&newest, Duration::from_secs(2)).is_err());
        assert_eq!(
            window
                .acknowledge(&acknowledgement, Duration::from_secs(2))
                .unwrap(),
            Duration::from_secs(1)
        );
        assert!(window.is_empty());
        assert_eq!(newest.identity.tick, 3);
        assert_eq!(
            newest.coalesced_publications,
            latest.revision.get() - initial.revision.get() - 1
        );
        window.mark_sent(&newest, Duration::from_secs(3)).unwrap();
        assert!(
            window
                .acknowledge(&acknowledgement, Duration::from_secs(4))
                .is_err(),
            "an old acknowledgement cannot release the replacement frame"
        );
        let latest_ack = frame_ack(&newest);
        window
            .acknowledge(&latest_ack, Duration::from_secs(4))
            .unwrap();
        assert!(
            window
                .acknowledge(&latest_ack, Duration::from_secs(4))
                .is_err(),
            "duplicate credit must not be manufactured"
        );
        assert!(window.is_empty());
    }

    #[test]
    fn remote_delivery_window_tracks_independent_channels_and_same_revision_event_pages() {
        let owner = owner();
        let mut snapshot = owner.latest_snapshot().as_ref().clone();
        snapshot.narrative_dropped_events = 7;
        snapshot.narrative_events = Arc::new(vec![narrative_record(0); REMOTE_MAX_EVENTS + 1]);
        let mut stream = RemoteSubscription::new(
            &request(vec![
                RemoteChannel::Canvas,
                RemoteChannel::Telemetry,
                RemoteChannel::Events,
            ]),
            &snapshot,
        )
        .unwrap();
        let canvas = stream
            .next_frame(RemoteChannel::Canvas, &snapshot)
            .unwrap()
            .unwrap();
        let telemetry = stream
            .next_frame(RemoteChannel::Telemetry, &snapshot)
            .unwrap()
            .unwrap();
        let first_page = stream
            .next_frame(RemoteChannel::Events, &snapshot)
            .unwrap()
            .unwrap();
        let mut window =
            RemoteDeliveryWindow::new(canvas.identity.session_id, Duration::from_secs(5));
        for frame in [&canvas, &telemetry, &first_page] {
            window.mark_sent(frame, Duration::ZERO).unwrap();
            stream.delivered(frame).unwrap();
        }
        assert_eq!(
            window.awaiting_channels(),
            vec![
                RemoteChannel::Telemetry,
                RemoteChannel::Canvas,
                RemoteChannel::Events
            ]
        );
        window
            .acknowledge(&frame_ack(&canvas), Duration::from_millis(1))
            .unwrap();
        assert_eq!(
            window.awaiting_channels(),
            vec![RemoteChannel::Telemetry, RemoteChannel::Events]
        );

        let first_ack = frame_ack(&first_page);
        assert_eq!(
            first_ack.after_event,
            Some((7 + REMOTE_MAX_EVENTS as u64).to_string())
        );
        let mut wrong_cursor = first_ack.clone();
        wrong_cursor.after_event = Some("7".into());
        assert!(
            window
                .acknowledge(&wrong_cursor, Duration::from_millis(2))
                .is_err()
        );
        assert_eq!(
            window.awaiting_channels(),
            vec![RemoteChannel::Telemetry, RemoteChannel::Events]
        );
        window
            .acknowledge(&first_ack, Duration::from_millis(2))
            .unwrap();

        let second_page = stream
            .next_frame(RemoteChannel::Events, &snapshot)
            .unwrap()
            .unwrap();
        assert_eq!(second_page.identity, first_page.identity);
        window
            .mark_sent(&second_page, Duration::from_millis(3))
            .unwrap();
        assert!(
            window
                .acknowledge(&first_ack, Duration::from_millis(4))
                .is_err(),
            "event ordinals distinguish pages sharing a publication"
        );
        window
            .acknowledge(&frame_ack(&second_page), Duration::from_millis(4))
            .unwrap();
        assert_eq!(window.awaiting_channels(), vec![RemoteChannel::Telemetry]);
        window
            .acknowledge(&frame_ack(&telemetry), Duration::from_millis(5))
            .unwrap();
        assert!(window.is_empty());
    }

    #[test]
    fn remote_delivery_window_refuses_deadline_and_clock_regression_without_releasing_credit() {
        let owner = owner();
        let snapshot = owner.latest_snapshot();
        let stream = RemoteSubscription::new(
            &request(vec![RemoteChannel::Canvas, RemoteChannel::Telemetry]),
            &snapshot,
        )
        .unwrap();
        let canvas = stream
            .next_frame(RemoteChannel::Canvas, &snapshot)
            .unwrap()
            .unwrap();
        let telemetry = stream
            .next_frame(RemoteChannel::Telemetry, &snapshot)
            .unwrap()
            .unwrap();
        let mut window =
            RemoteDeliveryWindow::new(canvas.identity.session_id, Duration::from_secs(5));
        window.mark_sent(&canvas, Duration::from_secs(10)).unwrap();
        assert!(
            window.expired(Duration::from_secs(9)),
            "regressed clocks fail closed"
        );
        assert!(!window.expired(Duration::from_millis(14_999)));
        assert!(window.expired(Duration::from_secs(15)));
        assert!(
            window
                .acknowledge(&frame_ack(&canvas), Duration::from_secs(9))
                .is_err()
        );
        assert!(
            window
                .acknowledge(&frame_ack(&canvas), Duration::from_secs(15))
                .is_err(),
            "a late exact acknowledgement cannot revive expired credit"
        );
        assert_eq!(window.awaiting_channels(), vec![RemoteChannel::Canvas]);

        let mut timely =
            RemoteDeliveryWindow::new(canvas.identity.session_id, Duration::from_secs(5));
        timely.mark_sent(&canvas, Duration::from_secs(10)).unwrap();
        assert_eq!(
            timely
                .acknowledge(&frame_ack(&canvas), Duration::from_millis(14_999))
                .unwrap(),
            Duration::from_millis(4_999)
        );
        assert!(timely.mark_sent(&canvas, Duration::from_secs(14)).is_err());
        assert!(timely.is_empty());

        let mut window =
            RemoteDeliveryWindow::new(canvas.identity.session_id, Duration::from_secs(5));
        window.mark_sent(&canvas, Duration::from_secs(10)).unwrap();
        window
            .mark_sent(&telemetry, Duration::from_secs(12))
            .unwrap();
        assert!(
            window
                .acknowledge(&frame_ack(&canvas), Duration::from_secs(11))
                .is_err(),
            "clock monotonicity spans all channels"
        );
        window
            .acknowledge(&frame_ack(&canvas), Duration::from_secs(14))
            .unwrap();
        assert!(!window.expired(Duration::from_secs(16)));
        assert!(
            window.expired(Duration::from_secs(17)),
            "the remaining channel keeps its original deadline"
        );
    }

    #[test]
    fn remote_frame_ack_requires_exact_bounded_channel_identity() {
        let valid = serde_json::json!({
            "protocol_version": 1, "command": "stream_ack", "session_id": u128::MAX.to_string(),
            "channel": "canvas", "publication_revision": u64::MAX.to_string(),
        });
        let ack = RemoteFrameAck::parse(&valid.to_string()).unwrap();
        assert_eq!(serde_json::to_value(ack).unwrap(), valid);
        for (field, value) in [
            ("protocol_version", serde_json::json!(2)),
            ("command", serde_json::json!("subscribe")),
            ("session_id", serde_json::json!("01")),
            (
                "session_id",
                serde_json::json!("340282366920938463463374607431768211456"),
            ),
            ("publication_revision", serde_json::json!(1)),
            (
                "publication_revision",
                serde_json::json!("18446744073709551616"),
            ),
            ("publication_revision", serde_json::json!("-1")),
            ("channel", serde_json::json!("events")),
            ("channel", serde_json::json!("unknown")),
            ("after_event", serde_json::json!("0")),
            ("after_event", serde_json::Value::Null),
            ("request_id", serde_json::json!("not-an-owner-command")),
        ] {
            let mut invalid = valid.clone();
            invalid[field] = value;
            assert!(
                RemoteFrameAck::parse(&invalid.to_string()).is_err(),
                "accepted {invalid}"
            );
        }
        let mut events = valid.clone();
        events["channel"] = serde_json::json!("events");
        events["after_event"] = serde_json::json!("0");
        assert!(RemoteFrameAck::parse(&events.to_string()).is_ok());
        for cursor in [
            serde_json::Value::Null,
            serde_json::json!(0),
            serde_json::json!("00"),
        ] {
            events["after_event"] = cursor;
            assert!(RemoteFrameAck::parse(&events.to_string()).is_err());
        }
        let duplicate = valid
            .to_string()
            .replacen('{', "{\"channel\":\"canvas\",", 1);
        assert!(RemoteFrameAck::parse(&duplicate).is_err());
        assert!(RemoteFrameAck::parse("null").is_err());
        assert!(RemoteFrameAck::parse("[]").is_err());
        assert!(RemoteFrameAck::parse(&" ".repeat(REMOTE_MAX_REQUEST_BYTES + 1)).is_err());
    }

    #[test]
    fn remote_subscription_masks_and_delivery_are_independent_of_science() {
        let owner = owner();
        let snapshot = owner.latest_snapshot();
        let before = owner.scientific_digest_cut().unwrap();
        let mut stream =
            RemoteSubscription::new(&request(vec![RemoteChannel::Telemetry]), &snapshot).unwrap();
        assert!(
            stream
                .next_frame(RemoteChannel::Canvas, &snapshot)
                .unwrap()
                .is_none()
        );
        assert!(
            stream
                .next_frame(RemoteChannel::Events, &snapshot)
                .unwrap()
                .is_none()
        );
        let first = stream
            .next_frame(RemoteChannel::Telemetry, &snapshot)
            .unwrap()
            .unwrap();
        let again = stream
            .next_frame(RemoteChannel::Telemetry, &snapshot)
            .unwrap()
            .unwrap();
        assert_eq!(
            first, again,
            "preparing a failed or delayed write cannot consume a cursor"
        );
        let RemoteFramePayload::Telemetry(summary) = &first.payload else {
            panic!("telemetry")
        };
        assert_eq!(summary.max_age, 17);
        assert_eq!(summary.tick, 0);
        assert_eq!(
            decode_remote_frame(&encode_remote_frame(&first).unwrap()).unwrap(),
            first
        );
        stream.delivered(&first).unwrap();
        assert!(
            stream
                .next_frame(RemoteChannel::Telemetry, &snapshot)
                .unwrap()
                .is_none()
        );
        assert!(stream.delivered(&first).is_err());
        assert_eq!(before.digest, owner.scientific_digest_cut().unwrap().digest);
    }

    #[test]
    fn remote_subscription_publishes_paused_control_changes_and_coalesces_per_channel() {
        let mut owner = owner();
        let mut frontend = NullFrontend::new(owner.local_port(), 10);
        let initial = owner.latest_snapshot();
        let before = owner.scientific_digest_cut().unwrap();
        let mut stream = RemoteSubscription::new(
            &request(vec![RemoteChannel::Canvas, RemoteChannel::Telemetry]),
            &initial,
        )
        .unwrap();
        let canvas = stream
            .next_frame(RemoteChannel::Canvas, &initial)
            .unwrap()
            .unwrap();
        stream.delivered(&canvas).unwrap();
        for speed in [2.0, 3.0] {
            frontend.set_speed(speed).unwrap();
            for _ in 0..4 {
                frontend
                    .drive_at(&mut owner, ManualInstant::from_nanos(1))
                    .unwrap();
            }
        }
        let changed = owner.latest_snapshot();
        assert_eq!(changed.world.tick, initial.world.tick);
        assert!(changed.revision > initial.revision);
        let canvas = stream
            .next_frame(RemoteChannel::Canvas, &changed)
            .unwrap()
            .unwrap();
        assert_eq!(canvas.playback.speed_multiplier, 3.0);
        assert_eq!(
            canvas.coalesced_publications,
            changed.revision.get() - initial.revision.get() - 1
        );
        stream.delivered(&canvas).unwrap();
        let telemetry = stream
            .next_frame(RemoteChannel::Telemetry, &changed)
            .unwrap()
            .unwrap();
        assert_eq!(
            telemetry.coalesced_publications, 0,
            "undelivered channel has its own cursor"
        );
        assert_eq!(before.digest, owner.scientific_digest_cut().unwrap().digest);
        frontend.step().unwrap();
        for _ in 0..4 {
            frontend
                .drive_at(&mut owner, ManualInstant::from_nanos(2))
                .unwrap();
        }
        let stepped = owner.latest_snapshot();
        let frame = stream
            .next_frame(RemoteChannel::Canvas, &stepped)
            .unwrap()
            .unwrap();
        assert_eq!(frame.identity.tick, 1);
        assert!(frame.identity.scientific_revision > canvas.identity.scientific_revision);
    }

    #[test]
    fn remote_subscription_pages_retained_narrative_and_reports_exact_gap() {
        let owner = owner();
        let mut snapshot = owner.latest_snapshot().as_ref().clone();
        snapshot.narrative_dropped_events = 7;
        snapshot.narrative_events = Arc::new(vec![narrative_record(0); REMOTE_MAX_EVENTS + 1]);
        let mut stream =
            RemoteSubscription::new(&request(vec![RemoteChannel::Events]), &snapshot).unwrap();
        let first = stream
            .next_frame(RemoteChannel::Events, &snapshot)
            .unwrap()
            .unwrap();
        let RemoteFramePayload::Events(page) = &first.payload else {
            panic!("events")
        };
        assert_eq!(page.events.len(), REMOTE_MAX_EVENTS);
        assert_eq!(
            page.gap,
            Some(RemoteEventGap {
                first_missing: 1,
                last_missing: 7
            })
        );
        assert_eq!(page.events[0].sequence, 8);
        assert_eq!(
            page.events[0].record.to_event_record().unwrap(),
            snapshot.narrative_events[0]
        );
        assert_eq!(
            stream.after_event(),
            0,
            "a prepared page is not a delivered page"
        );
        assert_eq!(
            decode_remote_frame(&encode_remote_frame(&first).unwrap()).unwrap(),
            first
        );
        stream.delivered(&first).unwrap();
        assert_eq!(stream.after_event(), 7 + REMOTE_MAX_EVENTS as u64);
        let second = stream
            .next_frame(RemoteChannel::Events, &snapshot)
            .unwrap()
            .unwrap();
        assert_eq!(
            first.identity, second.identity,
            "backlog drains without requiring a new tick"
        );
        let RemoteFramePayload::Events(page) = &second.payload else {
            panic!("events")
        };
        assert_eq!(page.events.len(), 1);
        assert_eq!(page.gap, None);
        stream.delivered(&second).unwrap();
        assert!(
            stream
                .next_frame(RemoteChannel::Events, &snapshot)
                .unwrap()
                .is_none()
        );
        assert!(
            stream.delivered(&first).is_err(),
            "an earlier delivered page cannot rewind the cursor"
        );

        let mut resumed = request(vec![RemoteChannel::Events]);
        resumed.resume = Some(RemoteResumeRequest {
            session_id: snapshot.session_id.get().to_string(),
            after_revision: Some(snapshot.revision.get().to_string()),
            after_event: Some(stream.after_event().to_string()),
        });
        let mut reconnect = RemoteSubscription::new(&resumed, &snapshot).unwrap();
        let confirmation = reconnect
            .next_frame(RemoteChannel::Events, &snapshot)
            .unwrap()
            .unwrap();
        let RemoteFramePayload::Events(page) = &confirmation.payload else {
            panic!("events")
        };
        assert_eq!(page.events, Vec::<RemoteSequencedEvent>::new());
        reconnect.delivered(&confirmation).unwrap();
        assert!(
            reconnect
                .next_frame(RemoteChannel::Events, &snapshot)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn remote_subscription_refuses_foreign_future_and_oversized_sources() {
        let owner = owner();
        let snapshot = owner.latest_snapshot();
        let mut req = request(vec![RemoteChannel::Canvas, RemoteChannel::Events]);
        req.resume = Some(RemoteResumeRequest {
            session_id: "892".into(),
            after_revision: None,
            after_event: None,
        });
        assert!(matches!(
            RemoteSubscription::new(&req, &snapshot),
            Err(RemoteSubscriptionError::SessionChanged { .. })
        ));
        req.resume.as_mut().unwrap().session_id = snapshot.session_id.get().to_string();
        req.resume.as_mut().unwrap().after_revision =
            Some((snapshot.revision.get() + 1).to_string());
        assert!(matches!(
            RemoteSubscription::new(&req, &snapshot),
            Err(RemoteSubscriptionError::CursorAhead {
                field: "after_revision",
                ..
            })
        ));
        req.resume.as_mut().unwrap().after_revision = None;
        req.resume.as_mut().unwrap().after_event = Some("1".into());
        assert!(matches!(
            RemoteSubscription::new(&req, &snapshot),
            Err(RemoteSubscriptionError::CursorAhead {
                field: "after_event",
                ..
            })
        ));
        req.resume = None;
        let stream = RemoteSubscription::new(&req, &snapshot).unwrap();
        let mut too_large = snapshot.as_ref().clone();
        too_large.world.agents = vec![snapshot.world.agents[0].clone(); REMOTE_MAX_AGENTS + 1];
        assert!(matches!(
            stream.next_frame(RemoteChannel::Canvas, &too_large),
            Err(RemoteSubscriptionError::ProjectionLimit(_))
        ));
        let mut record = narrative_record(0);
        record.human_text = "x".repeat(REMOTE_MAX_EVENT_TEXT_BYTES + 1);
        too_large.narrative_events = Arc::new(vec![record]);
        assert!(matches!(
            stream.next_frame(RemoteChannel::Events, &too_large),
            Err(RemoteSubscriptionError::ProjectionLimit(_))
        ));
        too_large.session_id = HostSessionId::new(892);
        assert!(matches!(
            stream.next_frame(RemoteChannel::Events, &too_large),
            Err(RemoteSubscriptionError::SessionChanged { .. })
        ));
    }

    #[test]
    fn remote_subscription_request_validation_refuses_ambiguous_shapes() {
        let valid = serde_json::to_value(request(vec![RemoteChannel::Canvas])).unwrap();
        assert!(RemoteSubscriptionRequest::parse(&valid.to_string()).is_ok());
        for (key, value) in [
            ("channels", serde_json::json!([])),
            ("channels", serde_json::json!(["canvas", "canvas"])),
            ("channels", serde_json::json!(["full-state"])),
            ("max_fps", serde_json::json!(0)),
            ("max_fps", serde_json::json!(121)),
            ("max_fps", serde_json::json!(60.5)),
            ("protocol_version", serde_json::json!(2)),
            ("request_id", serde_json::json!(" ")),
            ("unknown", serde_json::json!(true)),
            ("resume", serde_json::json!({"session_id": "01"})),
            (
                "resume",
                serde_json::json!({"session_id": "1", "after_event": 1}),
            ),
            (
                "resume",
                serde_json::json!({"session_id": "1", "after_revision": "18446744073709551616"}),
            ),
        ] {
            let mut malformed = valid.clone();
            malformed[key] = value;
            assert!(
                RemoteSubscriptionRequest::parse(&malformed.to_string()).is_err(),
                "accepted {malformed}"
            );
        }
        assert!(
            RemoteSubscriptionRequest::parse(&" ".repeat(REMOTE_MAX_REQUEST_BYTES + 1)).is_err()
        );
        assert!(RemoteSubscriptionRequest::parse("null").is_err());
    }
}
