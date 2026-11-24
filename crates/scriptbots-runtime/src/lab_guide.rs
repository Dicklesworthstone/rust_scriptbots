//! Shared, opt-in Lab walkthrough driven only by observed host evidence.
//!
//! This model owns no world, client, worker, or artifact bytes. Native adapters
//! submit through their normal controls, then feed their real receipts here.
//! A hidden guide keeps its progress after being opened. Merely constructing
//! the model never makes unrelated user actions count as tutorial progress.

use scriptbots_core::{AgentUid, ControlCommand};

use crate::lab_workflows::{
    LabBranchReport, LabCheckpoint, LabCheckpointBranch, LabCheckpointSave, LabCheckpointSaveState,
    branch_report_matches_receipt, checkpoint_receipt_is_bounded,
    checkpoint_request_identity_is_valid,
};
use crate::{
    ApplicationState, AppliedCommand, CommandId, CommandStatus, HostHealth, HostLifecycle,
    HostRevisions, HostSessionId, JournalState, RenderSnapshot, SnapshotRevision,
};

/// Stable identity of the shared native guide contract.
pub const LAB_GUIDE_SCHEMA: &str = "scriptbots.lab-guide.v1";
const MAX_GUIDE_TEXT_BYTES: usize = 1024;

/// Ordered learning steps shared by all native adapters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LabGuideStep {
    /// Read the active owner's provenance.
    #[default]
    Identify,
    /// Inspect one live organism by stable identity.
    Inspect,
    /// Confirm a submitted Pause at its committed boundary.
    Pause,
    /// Observe exactly one explicit scientific step.
    Step,
    /// Apply one growth-only configuration change.
    Configure,
    /// Save a fresh cut from the observed paused source.
    Checkpoint,
    /// Load a completed control and intervention comparison.
    Branch,
    /// Read the bounded descriptive outcome.
    Outcome,
}

impl LabGuideStep {
    /// The complete ordered registry; adapters must derive their count from it.
    pub const ALL: [Self; 8] = [
        Self::Identify,
        Self::Inspect,
        Self::Pause,
        Self::Step,
        Self::Configure,
        Self::Checkpoint,
        Self::Branch,
        Self::Outcome,
    ];

    /// Stable identifier independent of presentation.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::Identify => "identify",
            Self::Inspect => "inspect",
            Self::Pause => "pause",
            Self::Step => "step",
            Self::Configure => "configure",
            Self::Checkpoint => "checkpoint",
            Self::Branch => "branch",
            Self::Outcome => "outcome",
        }
    }

    /// Zero-based position in the shared step registry.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Identify => 0,
            Self::Inspect => 1,
            Self::Pause => 2,
            Self::Step => 3,
            Self::Configure => 4,
            Self::Checkpoint => 5,
            Self::Branch => 6,
            Self::Outcome => 7,
        }
    }

    /// Human-readable heading shared by the native frontends.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Identify => "Identify the live world",
            Self::Inspect => "Inspect one organism",
            Self::Pause => "Pause at a scientific boundary",
            Self::Step => "Advance exactly one tick",
            Self::Configure => "Change food growth",
            Self::Checkpoint => "Save the current cut",
            Self::Branch => "Compare a control and intervention",
            Self::Outcome => "Read the observed outcome",
        }
    }

    /// Actionable instruction; adapters add their own control labels.
    #[must_use]
    pub const fn instruction(self) -> &'static str {
        match self {
            Self::Identify => {
                "Read the live owner, resolved seed, tick, and population below, then continue."
            }
            Self::Inspect => "Use the inspector to select one living organism by its stable UID.",
            Self::Pause => {
                "Use Pause. Wait for its applied receipt, committed journal, and paused publication."
            }
            Self::Step => {
                "While paused, submit one Step. Wait for exactly one completed scientific tick."
            }
            Self::Configure => {
                "While paused, change only food_growth_rate to a different valid value and apply it."
            }
            Self::Checkpoint => {
                "Save a new checkpoint of this paused world. Wait for the saved receipt."
            }
            Self::Branch => {
                "Branch the checkpoint saved in this guide with an intervention. Load its completed two-arm report."
            }
            Self::Outcome => {
                "Compare the control and treatment. This single-seed result is descriptive; it does not establish a general effect."
            }
        }
    }
}

/// Compact provenance of the most recent accepted live publication.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabGuideSource {
    /// Identity of the sole-owner host producing this publication.
    pub session_id: HostSessionId,
    /// Monotonic publication revision within the owner.
    pub revision: SnapshotRevision,
    /// Independent control, scientific, and configuration revisions.
    pub revisions: HostRevisions,
    /// Completed scientific tick.
    pub tick: u64,
    /// The resolved root seed, including worlds launched from entropy.
    pub seed: Option<u64>,
    /// Whether automatic playback is paused.
    pub paused: bool,
    /// Published food-growth parameter used in the configuration exercise.
    pub food_growth_rate: f32,
    /// Number of living organisms in this publication.
    pub population: usize,
}

impl LabGuideSource {
    fn from_snapshot(snapshot: &RenderSnapshot) -> Self {
        Self {
            session_id: snapshot.session_id,
            revision: snapshot.revision,
            revisions: snapshot.revisions,
            tick: snapshot.world.tick,
            seed: Some(snapshot.visual_seed),
            paused: snapshot.playback.paused,
            food_growth_rate: snapshot.config.food_growth_rate,
            population: snapshot.world.agents.len(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GuideCommand {
    Pause,
    Step,
    Configure { food_growth_rate: f32 },
}

#[derive(Clone, Copy, Debug)]
enum Commitment {
    Volatile,
    Durable,
}

impl Commitment {
    const fn label(self) -> &'static str {
        match self {
            Self::Volatile => "CommittedVolatile (memory only)",
            Self::Durable => "Durable",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PendingCommand {
    id: CommandId,
    command: GuideCommand,
    before: LabGuideSource,
    applied: Option<AppliedCommand>,
    commitment: Option<Commitment>,
    observed: Option<LabGuideSource>,
}

impl PendingCommand {
    fn matches_applied(&self, applied: AppliedCommand) -> bool {
        if applied.revisions.control <= self.before.revisions.control {
            return false;
        }
        match self.command {
            GuideCommand::Pause => {
                applied.tick.0 >= self.before.tick
                    && applied.revisions.scientific >= self.before.revisions.scientific
                    && applied.revisions.config == self.before.revisions.config
            }
            GuideCommand::Step => {
                Some(applied.tick.0) == self.before.tick.checked_add(1)
                    && Some(applied.revisions.scientific)
                        == self.before.revisions.scientific.checked_next()
                    && applied.revisions.config == self.before.revisions.config
            }
            GuideCommand::Configure { .. } => {
                applied.tick.0 == self.before.tick
                    && applied.revisions.scientific == self.before.revisions.scientific
                    && Some(applied.revisions.config) == self.before.revisions.config.checked_next()
            }
        }
    }

    fn matches_source(&self, source: LabGuideSource) -> bool {
        if source.session_id != self.before.session_id
            || source.revision <= self.before.revision
            || !source.paused
            || !self.matches_applied(AppliedCommand {
                tick: scriptbots_core::Tick(source.tick),
                revisions: source.revisions,
            })
        {
            return false;
        }
        let expected_growth = match self.command {
            GuideCommand::Configure { food_growth_rate } => food_growth_rate,
            GuideCommand::Pause | GuideCommand::Step => self.before.food_growth_rate,
        };
        source.food_growth_rate.to_bits() == expected_growth.to_bits()
    }

    fn matching_receipt(&self) -> Option<(AppliedCommand, Commitment)> {
        let applied = self.applied?;
        let observed = self.observed?;
        let commitment = self.commitment?;
        // Later selection and other control-only publications are safe. A later
        // scientific tick or configuration revision cannot prove this boundary.
        (observed.tick == applied.tick.0
            && observed.revisions.scientific == applied.revisions.scientific
            && observed.revisions.config == applied.revisions.config
            && observed.revisions.control >= applied.revisions.control)
            .then_some((applied, commitment))
    }
}

#[derive(Clone, Debug)]
struct PendingSave {
    request_id: String,
    before: LabGuideSource,
}

/// Bounded, renderer-neutral observation state for the first-run Lab guide.
///
/// At most one command, one save request, one saved receipt, and one compact
/// outcome are retained. No full snapshot, configuration, or report is stored.
#[derive(Clone, Debug)]
pub struct LabGuide {
    /// Whether an adapter currently presents the guide. Hiding preserves progress.
    pub visible: bool,
    started: bool,
    generation: u64,
    step: LabGuideStep,
    source: Option<LabGuideSource>,
    available: bool,
    evidence_ready: bool,
    finished: bool,
    pending: Option<PendingCommand>,
    save_request: Option<PendingSave>,
    checkpoint: Option<LabCheckpoint>,
    outcome: Option<String>,
    status: String,
    evidence: String,
}

impl Default for LabGuide {
    fn default() -> Self {
        Self {
            visible: false,
            started: false,
            generation: 0,
            step: LabGuideStep::Identify,
            source: None,
            available: false,
            evidence_ready: false,
            finished: false,
            pending: None,
            save_request: None,
            checkpoint: None,
            outcome: None,
            status: "Open the guide to begin.".to_owned(),
            evidence: String::new(),
        }
    }
}

impl LabGuide {
    /// Start the opt-in guide or show its retained progress.
    pub fn open(&mut self) {
        self.visible = true;
        if self.generation == u64::MAX {
            return;
        }
        self.started = true;
        if !self.evidence_ready && self.pending.is_none() && self.save_request.is_none() {
            self.status = self.step.instruction().to_owned();
        }
    }

    /// Hide the guide while ordinary application controls remain available.
    pub const fn close(&mut self) {
        self.visible = false;
    }

    /// Restart at Identify; the current live source is retained, but no evidence is.
    pub fn restart(&mut self) {
        self.clear_progress();
        self.started = self.generation != u64::MAX;
    }

    fn clear_progress(&mut self) {
        self.generation = self.generation.saturating_add(1);
        self.step = LabGuideStep::Identify;
        self.evidence_ready = false;
        self.finished = false;
        self.pending = None;
        self.save_request = None;
        self.checkpoint = None;
        self.outcome = None;
        self.evidence.clear();
        self.status = self.step.instruction().to_owned();
        if self.generation == u64::MAX {
            self.started = false;
            self.available = false;
            self.status =
                "Guide presentation identity exhausted; open a new application session.".to_owned();
        }
    }

    /// Presentation identity; changes on restart and owner replacement.
    ///
    /// Adapters can capture this alongside a step to reject callbacks from an
    /// older guide that subsequently reached the same step again. The maximum
    /// value is a permanently disabled sentinel, so an active token is never reused.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Current learning step, retained while the guide is hidden.
    #[must_use]
    pub const fn step(&self) -> LabGuideStep {
        self.step
    }

    /// Current bounded instruction, progress, or refusal text.
    #[must_use]
    pub fn status(&self) -> &str {
        &self.status
    }

    /// The actual bounded evidence supporting the current step.
    #[must_use]
    pub fn evidence(&self) -> &str {
        &self.evidence
    }

    /// Most recent accepted compact owner provenance.
    #[must_use]
    pub const fn source(&self) -> Option<LabGuideSource> {
        self.source
    }

    /// Whether the user explicitly advanced through the final Outcome step.
    #[must_use]
    pub const fn finished(&self) -> bool {
        self.finished
    }

    /// Whether actual evidence and an available owner permit advancing now.
    #[must_use]
    pub const fn ready(&self) -> bool {
        self.started
            && self.available
            && self.evidence_ready
            && !self.finished
            && self.generation != u64::MAX
    }

    /// The single command whose application and commitment are still awaited.
    #[must_use]
    pub fn pending_command_id(&self) -> Option<CommandId> {
        self.pending.as_ref().map(|pending| pending.id)
    }

    /// The fresh cut saved during this guide, if its receipt has been observed.
    #[must_use]
    pub const fn saved_checkpoint(&self) -> Option<&LabCheckpoint> {
        self.checkpoint.as_ref()
    }

    /// Mark a disconnected or otherwise unavailable live owner without erasing progress.
    pub fn observe_unavailable(&mut self) {
        self.available = false;
        self.status = "Live owner unavailable. Reconnect to continue.".to_owned();
    }

    /// Continue only after evidence for this exact step has been observed.
    pub fn advance(&mut self) -> bool {
        if !self.ready() {
            return false;
        }
        if self.step == LabGuideStep::Outcome {
            self.finished = true;
            self.status =
                "Guide complete. The comparison describes one seed and one saved cut.".to_owned();
            return true;
        }
        self.step = LabGuideStep::ALL[self.step.index() + 1];
        self.evidence_ready = false;
        self.pending = None;
        self.save_request = None;
        self.evidence.clear();
        self.status = self.step.instruction().to_owned();
        if self.step == LabGuideStep::Outcome
            && let Some(outcome) = &self.outcome
        {
            self.evidence.clone_from(outcome);
            self.evidence_ready = true;
        }
        true
    }

    /// Observe the active owner's publication and an actual selected stable UID.
    ///
    /// Older publications and regressing scientific/configuration axes are
    /// ignored. A different owner resets all evidence, even when the seed matches.
    pub fn observe_snapshot(
        &mut self,
        snapshot: &RenderSnapshot,
        selected_uid: Option<AgentUid>,
    ) -> bool {
        if self.generation == u64::MAX {
            return false;
        }
        let source = LabGuideSource::from_snapshot(snapshot);
        if let Some(previous) = self.source {
            if source.session_id != previous.session_id {
                self.clear_progress();
                if self.generation == u64::MAX {
                    return false;
                }
            } else if source.revision < previous.revision
                || source.tick < previous.tick
                || source.revisions.control < previous.revisions.control
                || source.revisions.scientific < previous.revisions.scientific
                || source.revisions.config < previous.revisions.config
                || (source.revision == previous.revision && source != previous)
            {
                return false;
            }
        }
        self.source = Some(source);
        let was_available = self.available;
        self.available = snapshot.lifecycle == HostLifecycle::Running
            && matches!(&snapshot.health, HostHealth::Healthy);
        if !self.available {
            self.status =
                "Live owner is blocked, faulted, or stopped. Resolve it to continue.".to_owned();
            return true;
        }
        if !self.started || self.finished {
            return true;
        }
        if !was_available {
            self.status = if self.evidence_ready {
                "Live owner available; retained evidence permits continuing.".to_owned()
            } else {
                self.step.instruction().to_owned()
            };
        }
        if let Some(pending) = self.pending.as_mut()
            && pending.matches_source(source)
        {
            pending.observed = Some(source);
        }
        self.reconcile_command();
        match self.step {
            LabGuideStep::Identify => {
                self.evidence = bounded_text(format!(
                    "Owner {} | resolved seed {} | tick {} | population {}",
                    source.session_id.get(),
                    snapshot.visual_seed,
                    source.tick,
                    source.population,
                ));
                self.evidence_ready = true;
                self.status = "Live source identified. Continue when you have read its provenance."
                    .to_owned();
            }
            LabGuideStep::Inspect => {
                let selected = selected_uid
                    .filter(|uid| snapshot.world.agents.iter().any(|agent| agent.uid == *uid));
                self.evidence_ready = selected.is_some();
                if let Some(uid) = selected {
                    self.evidence = format!(
                        "Living agent UID {uid} observed in owner {} at tick {}.",
                        source.session_id.get(),
                        source.tick,
                    );
                    self.status = "Stable organism identity observed.".to_owned();
                } else {
                    self.evidence.clear();
                    self.status = "Select a living organism in the inspector.".to_owned();
                }
            }
            _ => {}
        }
        true
    }

    /// Track one submitted command identity for the current learning step.
    ///
    /// Call after an actual submission attempt, with the publication used to
    /// compose it. An ambiguous transport failure may retain the same identity;
    /// only exact applied and committed receipts can unlock Next. Hidden guides
    /// keep working after open(), while ordinary controls before opt-in never count.
    pub fn record_command(
        &mut self,
        command: &ControlCommand,
        command_id: CommandId,
        before: &RenderSnapshot,
    ) -> bool {
        if self.source != Some(LabGuideSource::from_snapshot(before)) {
            return false;
        }
        self.record_command_from_source(command, command_id, before, None)
    }

    /// Register a real admission that returned after newer publications were observed.
    ///
    /// Capture the generation, step and complete proposal publication before dispatch.
    /// A restart, different owner, retired step or regressing source refuses this late
    /// result. The current source never moves backwards. A matching current publication
    /// may supply observation evidence, but actual application and journal receipts are
    /// still required before the step can complete.
    pub fn record_deferred_command(
        &mut self,
        command: &ControlCommand,
        command_id: CommandId,
        before: &RenderSnapshot,
        expected_generation: u64,
        expected_step: LabGuideStep,
    ) -> bool {
        let source = LabGuideSource::from_snapshot(before);
        let Some(current) = self.source else {
            return false;
        };
        if self.generation == u64::MAX
            || self.generation != expected_generation
            || self.step != expected_step
            || current.session_id != source.session_id
            || current.revision < source.revision
            || current.tick < source.tick
            || current.revisions.control < source.revisions.control
            || current.revisions.scientific < source.revisions.scientific
            || current.revisions.config < source.revisions.config
            || (current.revision == source.revision && current != source)
        {
            return false;
        }
        self.record_command_from_source(command, command_id, before, Some(current))
    }

    fn record_command_from_source(
        &mut self,
        command: &ControlCommand,
        command_id: CommandId,
        before: &RenderSnapshot,
        observed: Option<LabGuideSource>,
    ) -> bool {
        let source = LabGuideSource::from_snapshot(before);
        if !self.started
            || self.finished
            || !self.available
            || self.evidence_ready
            || self.pending.is_some()
            || before.lifecycle != HostLifecycle::Running
            || !matches!(&before.health, HostHealth::Healthy)
            || command.validate().is_err()
        {
            return false;
        }
        let kind = match (self.step, command) {
            (LabGuideStep::Pause, ControlCommand::Pause) => GuideCommand::Pause,
            (LabGuideStep::Pause, ControlCommand::UpdateSimulation(update))
                if update.paused == Some(true)
                    && update.speed_multiplier.is_none()
                    && !update.step_once =>
            {
                GuideCommand::Pause
            }
            (LabGuideStep::Step, ControlCommand::Step) if source.paused => GuideCommand::Step,
            (LabGuideStep::Step, ControlCommand::UpdateSimulation(update))
                if source.paused
                    && update.paused != Some(false)
                    && update.speed_multiplier.is_none()
                    && update.step_once =>
            {
                GuideCommand::Step
            }
            (LabGuideStep::Configure, ControlCommand::UpdateConfig(config)) if source.paused => {
                let growth = config.food_growth_rate;
                if growth.to_bits() == source.food_growth_rate.to_bits() {
                    return false;
                }
                let mut expected = before.config.as_ref().clone();
                expected.food_growth_rate = growth;
                if &expected != config.as_ref() {
                    return false;
                }
                GuideCommand::Configure {
                    food_growth_rate: growth,
                }
            }
            _ => return false,
        };
        let mut pending = PendingCommand {
            id: command_id,
            command: kind,
            before: source,
            applied: None,
            commitment: None,
            observed: None,
        };
        pending.observed = observed.filter(|source| pending.matches_source(*source));
        self.pending = Some(pending);
        self.evidence.clear();
        self.status =
            "Submission attempted. Waiting for application, journal commitment, and publication."
                .to_owned();
        true
    }

    /// Correlate both receipt axes with the tracked identity and observed owner.
    pub fn observe_status(&mut self, session_id: HostSessionId, status: &CommandStatus) -> bool {
        if self
            .source
            .is_none_or(|source| source.session_id != session_id)
            || self
                .pending
                .is_none_or(|pending| pending.id != status.command_id())
            || status.validate().is_err()
        {
            return false;
        }
        let Some(mut pending) = self.pending.take() else {
            return false;
        };
        match status.application() {
            ApplicationState::Admitted => {
                if pending.applied.is_some() {
                    self.pending = Some(pending);
                    return false;
                }
                self.status = "Admitted; waiting for the command to apply.".to_owned();
            }
            ApplicationState::Rejected(_) | ApplicationState::Failed(_) => {
                if pending.applied.is_some() {
                    self.pending = Some(pending);
                    return false;
                }
                self.evidence_ready = false;
                self.evidence.clear();
                self.status =
                    "Command rejected or failed. Refresh the live source and retry this step."
                        .to_owned();
                return true;
            }
            ApplicationState::Applied(applied) => {
                if !pending.matches_applied(*applied)
                    || pending.applied.is_some_and(|previous| previous != *applied)
                {
                    self.pending = Some(pending);
                    self.status = "Receipt boundary does not match this step. Restart the guide after refreshing the source."
                        .to_owned();
                    return false;
                }
                pending.applied = Some(*applied);
                match status.journal() {
                    JournalState::CommittedVolatile => {
                        if !matches!(pending.commitment, Some(Commitment::Durable)) {
                            pending.commitment = Some(Commitment::Volatile);
                        }
                        self.status = "CommittedVolatile (memory only); waiting for the matching paused publication."
                            .to_owned();
                    }
                    JournalState::Durable => {
                        pending.commitment = Some(Commitment::Durable);
                        self.status =
                            "Durable; waiting for the matching paused publication.".to_owned();
                    }
                    JournalState::Pending if pending.commitment.is_none() => {
                        self.status = "Applied; journal commitment is still pending.".to_owned();
                    }
                    JournalState::NotRequired if pending.commitment.is_none() => {
                        self.status = "Applied without a committed journal receipt; this step is not complete."
                            .to_owned();
                    }
                    JournalState::Failed(_) if pending.commitment.is_none() => {
                        self.status = "Applied but journal commitment failed. Resolve persistence before continuing."
                            .to_owned();
                    }
                    _ => {}
                }
            }
        }
        self.pending = Some(pending);
        self.reconcile_command();
        true
    }

    fn reconcile_command(&mut self) {
        let Some(pending) = self.pending else {
            return;
        };
        let Some((applied, commitment)) = pending.matching_receipt() else {
            return;
        };
        self.evidence = bounded_text(format!(
            "Command {} applied at tick {} | control {} | science {} | config {} | {}",
            pending.id,
            applied.tick.0,
            applied.revisions.control.get(),
            applied.revisions.scientific.get(),
            applied.revisions.config.get(),
            commitment.label(),
        ));
        self.status = format!(
            "Applied boundary and publication matched. Journal: {}.",
            commitment.label(),
        );
        self.evidence_ready = true;
        self.pending = None;
    }

    /// Observe a fresh save request and its matching result at Checkpoint.
    ///
    /// A Saved receipt by itself, including an old catalog entry, cannot count.
    /// A Pending request must first have been observed at this learning step.
    pub fn observe_checkpoint_save(&mut self, save: &LabCheckpointSave) -> bool {
        let Some(source) = self.source else {
            return false;
        };
        if !self.started
            || self.finished
            || self.step != LabGuideStep::Checkpoint
            || self.evidence_ready
            || !self.available
            || source.session_id != save.session_id
            || !source.paused
            || !checkpoint_request_identity_is_valid(&save.request_id)
        {
            return false;
        }
        match &save.state {
            LabCheckpointSaveState::Pending => {
                if let Some(pending) = &self.save_request {
                    return pending.request_id == save.request_id;
                }
                self.save_request = Some(PendingSave {
                    request_id: save.request_id.clone(),
                    before: source,
                });
                self.status =
                    "Checkpoint requested; waiting for this save identity to complete.".to_owned();
                true
            }
            LabCheckpointSaveState::Retryable { .. } => {
                if self
                    .save_request
                    .as_ref()
                    .is_none_or(|pending| pending.request_id != save.request_id)
                {
                    return false;
                }
                // The native adapters may retry with a fresh request identity.
                // Require another actual Pending observation before any Saved
                // result can count, including a late result for this failed try.
                self.save_request = None;
                self.status = "Checkpoint has not saved. Retry saving after resolving its refusal."
                    .to_owned();
                true
            }
            LabCheckpointSaveState::Saved(checkpoint) => {
                let Some(pending) = &self.save_request else {
                    return false;
                };
                if pending.request_id != save.request_id
                    || checkpoint.tick != pending.before.tick
                    || source.tick != pending.before.tick
                    || source.revisions.scientific != pending.before.revisions.scientific
                    || source.revisions.config != pending.before.revisions.config
                    || checkpoint.agent_count != source.population
                    || !checkpoint_receipt_is_bounded(checkpoint)
                {
                    return false;
                }
                self.evidence = bounded_text(format!(
                    "Saved {} at tick {} | population {} | world digest {}",
                    checkpoint.checkpoint_id,
                    checkpoint.tick,
                    checkpoint.agent_count,
                    checkpoint.world_digest,
                ));
                self.checkpoint = Some(checkpoint.clone());
                self.save_request = None;
                self.evidence_ready = true;
                self.status = "The new saved cut is ready for a branch comparison.".to_owned();
                true
            }
        }
    }

    /// Observe a completed report returned by the existing checksum-verifying loader.
    ///
    /// This method checks the loaded DTO against its completed receipt and the
    /// cut saved during this guide. It does not hash artifact bytes it never owns.
    /// An unchanged, single-arm continuation cannot satisfy the intervention step.
    pub fn observe_branch_report(
        &mut self,
        session_id: HostSessionId,
        branch: &LabCheckpointBranch,
        report: &LabBranchReport,
    ) -> bool {
        let (Some(source), Some(checkpoint), Some(treatment)) = (
            self.source,
            self.checkpoint.as_ref(),
            report.treatment.as_ref(),
        ) else {
            return false;
        };
        if !self.started
            || self.finished
            || self.step != LabGuideStep::Branch
            || self.evidence_ready
            || !self.available
            || session_id != source.session_id
            || branch.checkpoint_id != checkpoint.checkpoint_id
            || report.source_tick != checkpoint.tick
            || report.source_world_digest != checkpoint.world_digest
            || report.source_byte_size != checkpoint.byte_size
            || Some(report.base_seed) != source.seed
            || report.control.baseline_population != checkpoint.agent_count
            || treatment.command_applied_tick != Some(checkpoint.tick)
            || !branch_report_matches_receipt(report, branch)
        {
            return false;
        }
        let summary = bounded_text(format!(
            "Seed {} | saved tick {} -> {} | control population {}, treatment {} | control food {:.3e}, treatment {:.3e} | control energy {:.3e}, treatment {:.3e}. One seed: descriptive, not a general causal conclusion.",
            report.base_seed,
            report.source_tick,
            report.target_tick,
            report.control.population,
            treatment.population,
            report.control.total_food,
            treatment.total_food,
            report.control.total_energy,
            treatment.total_energy,
        ));
        self.outcome = Some(summary);
        self.evidence = bounded_text(format!(
            "Completed branch {} | artifact {} | source digest {} | both arms reached tick {}.",
            branch.branch_id, report.artifact_id, report.source_world_digest, report.target_tick,
        ));
        self.evidence_ready = true;
        self.status = "Completed control and intervention report matches the saved cut.".to_owned();
        true
    }
}

fn bounded_text(mut text: String) -> String {
    if text.len() > MAX_GUIDE_TEXT_BYTES {
        let mut boundary = MAX_GUIDE_TEXT_BYTES;
        while !text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        text.truncate(boundary);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lab_workflows::{LabBranchDivergence, LabBranchOutcome};
    use crate::{
        AdmissionSequence, ApplicationFailure, CommandEnvelope, HostCommand, HostCore,
        HostCoreOptions, HostPort, JournalFailure, ManualHostDriver, ManualInstant,
        PlaybackSnapshot, RejectionReason,
    };
    use scriptbots_core::{AgentData, ScriptBotsConfig, SimulationCommand, Tick, WorldState};

    fn owner() -> HostCore {
        let mut world = WorldState::new(ScriptBotsConfig {
            rng_seed: Some(42),
            world_width: 120,
            world_height: 120,
            food_cell_size: 20,
            closed: true,
            population_minimum: 0,
            population_spawn_interval: 0,
            persistence_interval: 0,
            reproduction_energy_threshold: f32::MAX,
            ..ScriptBotsConfig::default()
        })
        .expect("valid compact world");
        for _ in 0..3 {
            world
                .try_spawn_agent(AgentData::default())
                .expect("real founder");
        }
        HostCore::new(
            HostSessionId::new(41),
            world,
            HostCoreOptions {
                initial_playback: PlaybackSnapshot {
                    paused: false,
                    speed_multiplier: 1.0,
                },
                ..HostCoreOptions::default()
            },
        )
        .expect("real volatile owner")
    }

    fn at_pause(snapshot: &RenderSnapshot) -> LabGuide {
        let mut guide = LabGuide::default();
        guide.open();
        assert!(guide.observe_snapshot(snapshot, None));
        assert!(guide.ready());
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Inspect);
        assert!(guide.observe_snapshot(snapshot, Some(snapshot.world.agents[0].uid)));
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Pause);
        guide
    }

    fn status(
        id: CommandId,
        application: ApplicationState,
        journal: JournalState,
    ) -> CommandStatus {
        CommandStatus::try_new(id, Some(AdmissionSequence::new(1)), application, journal)
            .expect("valid independent receipt axes")
    }

    fn apply(owner: &mut HostCore, guide: &mut LabGuide, command: ControlCommand, id: u128) {
        let before = owner.snapshot_hub().latest();
        let id = CommandId::new(id);
        assert!(guide.record_command(&command, id, &before));
        let mut port = owner.local_port();
        let admitted = port
            .submit(CommandEnvelope::new(
                id,
                HostCommand::try_from(command).expect("supported command"),
            ))
            .expect("actual command admission");
        assert!(guide.observe_status(before.session_id, &admitted));
        assert!(!guide.ready(), "admission is not completion");
        for _ in 0..4 {
            owner
                .drive(ManualInstant::from_nanos(0))
                .expect("owner boundary");
            let receipt = port
                .command_status(id)
                .expect("status read")
                .expect("known command");
            guide.observe_status(before.session_id, &receipt);
            let frame = owner.snapshot_hub().latest();
            guide.observe_snapshot(&frame, frame.world.agents.first().map(|agent| agent.uid));
            if guide.ready() {
                break;
            }
        }
        assert!(
            guide.ready(),
            "actual command and volatile journal must settle"
        );
        assert!(guide.evidence().contains("CommittedVolatile (memory only)"));
        assert!(guide.pending_command_id().is_none());
    }

    fn at_step(owner: &mut HostCore) -> LabGuide {
        let mut guide = at_pause(&owner.snapshot_hub().latest());
        apply(owner, &mut guide, ControlCommand::Pause, 1);
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Step);
        guide
    }

    fn at_configure(owner: &mut HostCore) -> LabGuide {
        let mut guide = at_step(owner);
        let before = owner.world_tick();
        apply(owner, &mut guide, ControlCommand::Step, 2);
        assert_eq!(owner.world_tick().0, before.0 + 1);
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Configure);
        guide
    }

    fn at_checkpoint(owner: &mut HostCore) -> LabGuide {
        let mut guide = at_configure(owner);
        let mut config = owner.snapshot_hub().latest().config.as_ref().clone();
        config.food_growth_rate = 0.075;
        apply(
            owner,
            &mut guide,
            ControlCommand::UpdateConfig(Box::new(config)),
            3,
        );
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Checkpoint);
        guide
    }

    fn save_pair(guide: &LabGuide) -> (LabCheckpointSave, LabCheckpointSave) {
        let source = guide.source().expect("observed source");
        let pending = LabCheckpointSave {
            request_id: "guide-save-1".to_owned(),
            session_id: source.session_id,
            state: LabCheckpointSaveState::Pending,
        };
        let saved = LabCheckpointSave {
            state: LabCheckpointSaveState::Saved(LabCheckpoint {
                checkpoint_id: "guide-cut-1".to_owned(),
                tick: source.tick,
                agent_count: source.population,
                byte_size: 4096,
                world_digest: "0123456789abcdef".to_owned(),
            }),
            ..pending.clone()
        };
        (pending, saved)
    }

    fn at_branch(owner: &mut HostCore) -> LabGuide {
        let mut guide = at_checkpoint(owner);
        let (pending, saved) = save_pair(&guide);
        assert!(guide.observe_checkpoint_save(&pending));
        assert!(!guide.ready());
        assert!(guide.observe_checkpoint_save(&saved));
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Branch);
        guide
    }

    // These are structurally valid, bounded loader DTOs. Artifact hash
    // verification itself belongs to the existing artifact-loader tests.
    fn report_pair(guide: &LabGuide) -> (LabCheckpointBranch, LabBranchReport) {
        let cut = guide.saved_checkpoint().expect("guide's own saved cut");
        let target = cut.tick + 5;
        let branch = LabCheckpointBranch {
            branch_id: "guide-study".to_owned(),
            checkpoint_id: cut.checkpoint_id.clone(),
            branch_name: "drought".to_owned(),
            status: "completed".to_owned(),
            source_tick: cut.tick,
            target_tick: target,
            world_digest: cut.world_digest.clone(),
            artifact_id: Some("guide-artifact".to_owned()),
            error_reason: None,
        };
        let control = LabBranchOutcome {
            tick: target,
            population: cut.agent_count,
            total_food: 20.0,
            total_energy: 6.0,
            average_health: Some(1.0),
            final_digest: "2222222222222222".to_owned(),
            command_applied_tick: None,
            minimum_population: cut.agent_count,
            extinction_tick: None,
            baseline_population: cut.agent_count,
            recovery_ratio: Some(1.0),
            first_divergence: None,
        };
        let treatment = LabBranchOutcome {
            total_food: 10.0,
            total_energy: 5.5,
            final_digest: "3333333333333333".to_owned(),
            command_applied_tick: Some(cut.tick),
            first_divergence: Some(LabBranchDivergence {
                tick: cut.tick + 1,
                stage: "food".to_owned(),
                control_digest: "2222222222222222".to_owned(),
                branch_digest: "3333333333333333".to_owned(),
                command_applied_tick: cut.tick,
                details: "Fixture has an observed resource divergence.".to_owned(),
            }),
            ..control.clone()
        };
        let report = LabBranchReport {
            branch_id: branch.branch_id.clone(),
            artifact_id: branch.artifact_id.clone().expect("completed artifact"),
            base_seed: guide.source().expect("source").seed.expect("resolved seed"),
            source_tick: cut.tick,
            target_tick: target,
            source_world_digest: cut.world_digest.clone(),
            source_byte_size: cut.byte_size,
            source_checksum_blake3: "a".repeat(64),
            report_byte_size: 1024,
            report_checksum_blake3: "b".repeat(64),
            report_checksum_sha256: "c".repeat(64),
            control,
            treatment: Some(treatment),
        };
        assert!(branch_report_matches_receipt(&report, &branch));
        (branch, report)
    }

    #[test]
    fn shared_registry_has_stable_unique_ids_and_zero_based_positions() {
        assert_eq!(LAB_GUIDE_SCHEMA, "scriptbots.lab-guide.v1");
        let mut ids = std::collections::BTreeSet::new();
        for (index, step) in LabGuideStep::ALL.into_iter().enumerate() {
            assert_eq!(step.index(), index);
            assert!(ids.insert(step.id()));
            assert!(!step.title().is_empty());
            assert!(!step.instruction().is_empty());
        }
    }

    #[test]
    fn opt_in_real_uid_and_monotonic_source_are_required() {
        let owner = owner();
        let frame = owner.snapshot_hub().latest();
        let mut guide = LabGuide::default();
        assert!(guide.observe_snapshot(&frame, Some(frame.world.agents[0].uid)));
        assert!(!guide.ready());
        assert!(!guide.advance());
        assert!(!guide.record_command(&ControlCommand::Pause, CommandId::new(1), &frame));
        guide.open();
        assert!(guide.observe_snapshot(&frame, None));
        assert_eq!(guide.source().expect("source").seed, Some(42));
        assert_eq!(guide.source().expect("source").population, 3);
        assert!(guide.advance());
        guide.observe_snapshot(&frame, Some(AgentUid(u64::MAX)));
        assert!(
            !guide.ready(),
            "a missing stable UID is not an inspected organism"
        );
        guide.observe_snapshot(&frame, Some(frame.world.agents[0].uid));
        assert!(guide.ready());
        let mut regressed = frame.as_ref().clone();
        regressed.revision = frame.revision.checked_next().expect("next");
        regressed.revisions.control = frame.revisions.control.checked_next().expect("next");
        assert!(guide.observe_snapshot(&regressed, Some(frame.world.agents[0].uid)));
        assert!(!guide.observe_snapshot(&frame, Some(frame.world.agents[0].uid)));
        assert_eq!(guide.source().expect("source").revision, regressed.revision);
        let generation = guide.generation();
        guide.close();
        guide.open();
        assert_eq!(guide.generation(), generation);
        guide.restart();
        assert!(guide.generation() > generation);
        assert_eq!(guide.step(), LabGuideStep::Identify);
        assert!(!guide.ready());
    }

    #[test]
    fn command_needs_matching_owner_id_application_journal_and_publication() {
        let owner = owner();
        let before = owner.snapshot_hub().latest();
        let mut guide = at_pause(&before);
        let id = CommandId::new(91);
        assert!(guide.record_command(&ControlCommand::Pause, id, &before));
        assert!(!guide.record_command(&ControlCommand::Pause, CommandId::new(92), &before));
        let mut after = before.as_ref().clone();
        after.revision = before.revision.checked_next().expect("next");
        after.revisions.control = before.revisions.control.checked_next().expect("next");
        after.playback.paused = true;
        let applied = AppliedCommand {
            tick: Tick(after.world.tick),
            revisions: after.revisions,
        };
        let committed = status(
            id,
            ApplicationState::Applied(applied),
            JournalState::CommittedVolatile,
        );
        assert!(!guide.observe_status(HostSessionId::new(999), &committed));
        assert!(!guide.observe_status(
            before.session_id,
            &status(
                CommandId::new(92),
                ApplicationState::Applied(applied),
                JournalState::Durable,
            )
        ));
        assert_eq!(guide.pending_command_id(), Some(id));
        assert!(guide.observe_status(before.session_id, &committed));
        assert!(
            !guide.ready(),
            "a receipt alone is not a published observation"
        );
        let mut pending_only = at_pause(&before);
        assert!(pending_only.record_command(&ControlCommand::Pause, id, &before));
        assert!(pending_only.observe_snapshot(&after, None));
        for journal in [
            JournalState::NotRequired,
            JournalState::Pending,
            JournalState::Failed(JournalFailure {
                code: "disk".into(),
                message: "unavailable".into(),
            }),
        ] {
            assert!(pending_only.observe_status(
                before.session_id,
                &status(id, ApplicationState::Applied(applied), journal,)
            ));
            assert!(!pending_only.ready());
            assert!(!pending_only.advance());
        }
        assert!(pending_only.observe_status(
            before.session_id,
            &status(
                id,
                ApplicationState::Applied(applied),
                JournalState::Durable,
            )
        ));
        assert!(pending_only.ready());
        assert!(pending_only.evidence().contains("Durable"));
        after.revisions.control = after
            .revisions
            .control
            .checked_next()
            .expect("selection control");
        assert!(guide.observe_snapshot(&after, None));
        assert!(
            guide.ready(),
            "later control-only observation still proves the boundary"
        );
        assert!(guide.evidence().contains("CommittedVolatile (memory only)"));
        assert!(!guide.evidence().contains("Durable"));
    }

    #[test]
    fn deferred_commands_match_real_late_receipts_without_rewinding_the_source() {
        for step in [
            LabGuideStep::Pause,
            LabGuideStep::Step,
            LabGuideStep::Configure,
        ] {
            let mut owner = owner();
            let mut guide = match step {
                LabGuideStep::Pause => at_pause(&owner.snapshot_hub().latest()),
                LabGuideStep::Step => at_step(&mut owner),
                LabGuideStep::Configure => at_configure(&mut owner),
                _ => unreachable!("command exercise"),
            };
            let before = owner.snapshot_hub().latest();
            let generation = guide.generation();
            let command = match step {
                LabGuideStep::Pause => ControlCommand::Pause,
                LabGuideStep::Step => ControlCommand::Step,
                LabGuideStep::Configure => {
                    let mut config = before.config.as_ref().clone();
                    config.food_growth_rate = 0.075;
                    ControlCommand::UpdateConfig(Box::new(config))
                }
                _ => unreachable!("command exercise"),
            };
            let id = CommandId::new(100);
            let mut port = owner.local_port();
            let admitted = port
                .submit(CommandEnvelope::new(
                    id,
                    HostCommand::try_from(command.clone()).unwrap(),
                ))
                .unwrap();
            owner.drive(ManualInstant::from_nanos(0)).unwrap();
            let after = owner.snapshot_hub().latest();
            assert!(after.revision > before.revision);
            assert!(guide.observe_snapshot(&after, None));
            let current_source = guide.source();
            assert!(
                !guide.record_command(&command, id, &before),
                "synchronous source gate stays strict"
            );
            assert!(guide.record_deferred_command(&command, id, &before, generation, step));
            assert_eq!(guide.source(), current_source);
            assert_eq!(guide.pending_command_id(), Some(id));
            assert!(
                !guide.ready(),
                "a published change does not supply a receipt"
            );
            assert!(guide.observe_status(before.session_id, &admitted));
            assert!(!guide.ready(), "actual admission alone is insufficient");
            let applied = port.command_status(id).unwrap().unwrap();
            assert!(matches!(
                applied.application(),
                ApplicationState::Applied(_)
            ));
            assert_eq!(applied.journal(), &JournalState::Pending);
            assert!(guide.observe_status(before.session_id, &applied));
            assert!(!guide.ready(), "application must still commit");
            owner.drive(ManualInstant::from_nanos(0)).unwrap();
            let committed = port.command_status(id).unwrap().unwrap();
            assert_eq!(committed.journal(), &JournalState::CommittedVolatile);
            assert!(guide.observe_status(before.session_id, &committed));
            assert!(guide.ready());
            assert!(guide.evidence().contains("CommittedVolatile (memory only)"));
            assert!(!guide.record_deferred_command(&command, id, &before, generation, step));
            assert!(guide.advance());
            assert!(!guide.record_deferred_command(&command, id, &before, generation, step));
        }
    }

    #[test]
    fn deferred_commands_refuse_retired_busy_and_inconsistent_sources() {
        let owner = owner();
        let before = owner.snapshot_hub().latest();
        let baseline = at_pause(&before);
        let generation = baseline.generation();
        let rejects = |mut guide: LabGuide, source: &RenderSnapshot, token, step| {
            let current = guide.source();
            let pending = guide.pending_command_id();
            let evidence = guide.evidence().to_owned();
            assert!(!guide.record_deferred_command(
                &ControlCommand::Pause,
                CommandId::new(101),
                source,
                token,
                step,
            ));
            assert_eq!(guide.source(), current);
            assert_eq!(guide.pending_command_id(), pending);
            assert_eq!(guide.evidence(), evidence);
        };
        rejects(
            LabGuide::default(),
            &before,
            generation,
            LabGuideStep::Pause,
        );
        rejects(
            baseline.clone(),
            &before,
            generation + 1,
            LabGuideStep::Pause,
        );
        rejects(baseline.clone(), &before, generation, LabGuideStep::Step);
        let mut restarted = baseline.clone();
        restarted.restart();
        assert!(restarted.observe_snapshot(&before, None));
        assert!(restarted.advance());
        assert!(restarted.observe_snapshot(&before, Some(before.world.agents[0].uid)));
        assert!(restarted.advance());
        assert_eq!(restarted.step(), LabGuideStep::Pause);
        rejects(restarted, &before, generation, LabGuideStep::Pause);
        let mut replacement = before.as_ref().clone();
        replacement.session_id = HostSessionId::new(before.session_id.get() + 1);
        rejects(
            at_pause(&replacement),
            &before,
            generation,
            LabGuideStep::Pause,
        );
        let mut exhausted = baseline.clone();
        exhausted.generation = u64::MAX - 1;
        exhausted.restart();
        rejects(exhausted, &before, u64::MAX, LabGuideStep::Identify);
        let mut unavailable = baseline.clone();
        unavailable.observe_unavailable();
        rejects(unavailable, &before, generation, LabGuideStep::Pause);
        let mut busy = baseline.clone();
        assert!(busy.record_command(&ControlCommand::Pause, CommandId::new(102), &before));
        rejects(busy, &before, generation, LabGuideStep::Pause);

        let mut observed = before.as_ref().clone();
        observed.revision = before.revision.checked_next().unwrap();
        let mut current = baseline;
        assert!(current.observe_snapshot(&observed, None));
        let mut invalid_sources = Vec::new();
        let mut future = before.as_ref().clone();
        future.revision = observed.revision.checked_next().unwrap();
        invalid_sources.push(future);
        let mut future = before.as_ref().clone();
        future.world.tick += 1;
        invalid_sources.push(future);
        let mut future = before.as_ref().clone();
        future.revisions.control = future.revisions.control.checked_next().unwrap();
        invalid_sources.push(future);
        let mut future = before.as_ref().clone();
        future.revisions.scientific = future.revisions.scientific.checked_next().unwrap();
        invalid_sources.push(future);
        let mut future = before.as_ref().clone();
        future.revisions.config = future.revisions.config.checked_next().unwrap();
        invalid_sources.push(future);
        let mut inconsistent = observed;
        inconsistent.playback.paused = !inconsistent.playback.paused;
        invalid_sources.push(inconsistent);
        for invalid in invalid_sources {
            rejects(current.clone(), &invalid, generation, LabGuideStep::Pause);
        }
    }

    #[test]
    fn deferred_step_does_not_use_a_later_scientific_tick_as_completion_evidence() {
        let mut owner = owner();
        let mut guide = at_step(&mut owner);
        let before = owner.snapshot_hub().latest();
        let generation = guide.generation();
        let id = CommandId::new(103);
        let mut port = owner.local_port();
        for command_id in [id, CommandId::new(104)] {
            port.submit(CommandEnvelope::new(command_id, HostCommand::Step))
                .unwrap();
            owner.drive(ManualInstant::from_nanos(0)).unwrap();
            owner.drive(ManualInstant::from_nanos(0)).unwrap();
        }
        let after = owner.snapshot_hub().latest();
        assert_eq!(after.world.tick, before.world.tick + 2);
        assert!(guide.observe_snapshot(&after, None));
        assert!(guide.record_deferred_command(
            &ControlCommand::Step,
            id,
            &before,
            generation,
            LabGuideStep::Step,
        ));
        let receipt = port.command_status(id).unwrap().unwrap();
        assert_eq!(receipt.journal(), &JournalState::CommittedVolatile);
        assert!(guide.observe_status(before.session_id, &receipt));
        assert!(
            !guide.ready(),
            "a later tick cannot witness the requested one-tick boundary"
        );
        assert!(!guide.advance());
        assert_eq!(guide.pending_command_id(), Some(id));
        assert_eq!(guide.source().unwrap().tick, after.world.tick);
    }

    #[test]
    fn deferred_real_rejection_never_counts_as_a_completed_step() {
        let mut owner = owner();
        let before = owner.snapshot_hub().latest();
        let mut guide = at_pause(&before);
        let generation = guide.generation();
        let id = CommandId::new(105);
        let mut port = owner.local_port();
        port.submit(
            CommandEnvelope::new(id, HostCommand::Pause)
                .expecting_scientific_revision(before.revisions.scientific.checked_next().unwrap()),
        )
        .unwrap();
        owner.drive(ManualInstant::from_nanos(0)).unwrap();
        owner.drive(ManualInstant::from_nanos(0)).unwrap();
        let after = owner.snapshot_hub().latest();
        assert!(guide.observe_snapshot(&after, None));
        assert!(guide.record_deferred_command(
            &ControlCommand::Pause,
            id,
            &before,
            generation,
            LabGuideStep::Pause,
        ));
        let receipt = port.command_status(id).unwrap().unwrap();
        assert!(matches!(
            receipt.application(),
            ApplicationState::Rejected(_)
        ));
        assert_eq!(receipt.journal(), &JournalState::CommittedVolatile);
        assert!(guide.observe_status(before.session_id, &receipt));
        assert!(guide.pending_command_id().is_none());
        assert!(!guide.ready());
        assert!(!guide.advance());
        assert_eq!(guide.source(), Some(LabGuideSource::from_snapshot(&after)));
    }

    #[test]
    fn step_refuses_running_inputs_two_tick_receipts_and_wrong_scientific_axis() {
        let mut owner = owner();
        let guide = at_step(&mut owner);
        let before = owner.snapshot_hub().latest();
        let id = CommandId::new(70);
        let mut running = before.as_ref().clone();
        running.revision = running.revision.checked_next().expect("next");
        running.playback.paused = false;
        let mut running_guide = guide.clone();
        running_guide.observe_snapshot(&running, None);
        assert!(!running_guide.record_command(&ControlCommand::Step, id, &running));
        let mut tracked = guide.clone();
        assert!(tracked.record_command(&ControlCommand::Step, id, &before));
        let mut wrong = AppliedCommand {
            tick: Tick(before.world.tick + 2),
            revisions: HostRevisions {
                control: before.revisions.control.checked_next().expect("next"),
                scientific: before.revisions.scientific.checked_next().expect("next"),
                config: before.revisions.config,
            },
        };
        assert!(!tracked.observe_status(
            before.session_id,
            &status(id, ApplicationState::Applied(wrong), JournalState::Durable,)
        ));
        wrong.tick = Tick(before.world.tick + 1);
        wrong.revisions.scientific = before.revisions.scientific;
        assert!(!tracked.observe_status(
            before.session_id,
            &status(id, ApplicationState::Applied(wrong), JournalState::Durable,)
        ));
        assert!(!tracked.advance());
        assert!(!tracked.record_command(
            &ControlCommand::UpdateSimulation(SimulationCommand {
                paused: Some(false),
                speed_multiplier: None,
                step_once: true,
            }),
            CommandId::new(71),
            &before
        ));
    }

    #[test]
    fn configuration_is_a_real_growth_only_change_at_the_paused_boundary() {
        let mut owner = owner();
        let mut guide = at_configure(&mut owner);
        let before = owner.snapshot_hub().latest();
        let id = CommandId::new(80);
        let original = before.config.as_ref().clone();
        assert!(!guide.record_command(
            &ControlCommand::UpdateConfig(Box::new(original.clone())),
            id,
            &before
        ));
        let mut wrong = original.clone();
        wrong.food_growth_rate = 0.075;
        wrong.food_decay_rate += 0.001;
        assert!(!guide.record_command(&ControlCommand::UpdateConfig(Box::new(wrong)), id, &before));
        let mut invalid = original.clone();
        invalid.food_growth_rate = f32::NAN;
        assert!(!guide.record_command(
            &ControlCommand::UpdateConfig(Box::new(invalid)),
            id,
            &before
        ));
        let mut right = original;
        right.food_growth_rate = 0.075;
        apply(
            &mut owner,
            &mut guide,
            ControlCommand::UpdateConfig(Box::new(right)),
            80,
        );
        assert_eq!(owner.world_tick().0, before.world.tick);
        let after = owner.snapshot_hub().latest();
        assert_eq!(after.revisions.scientific, before.revisions.scientific);
        assert_eq!(
            Some(after.revisions.config),
            before.revisions.config.checked_next()
        );
        assert_eq!(after.config.food_growth_rate.to_bits(), 0.075_f32.to_bits());
    }

    #[test]
    fn failed_and_stale_commands_allow_retry_but_unavailable_owner_cannot_advance() {
        let owner = owner();
        let before = owner.snapshot_hub().latest();
        let mut guide = at_pause(&before);
        let id = CommandId::new(83);
        assert!(guide.record_command(&ControlCommand::Pause, id, &before));
        let stale = status(
            id,
            ApplicationState::Rejected(RejectionReason::ControlRevisionConflict {
                expected: before.revisions.control,
                actual: before.revisions.control.checked_next().expect("next"),
            }),
            JournalState::Durable,
        );
        assert!(guide.observe_status(before.session_id, &stale));
        assert!(!guide.ready());
        assert!(guide.pending_command_id().is_none());
        let id = CommandId::new(84);
        assert!(guide.record_command(&ControlCommand::Pause, id, &before));
        assert!(guide.observe_status(
            before.session_id,
            &status(
                id,
                ApplicationState::Failed(ApplicationFailure {
                    code: "owner".into(),
                    message: "failure".into()
                }),
                JournalState::Durable,
            )
        ));
        assert!(guide.pending_command_id().is_none());
        guide.observe_unavailable();
        assert!(!guide.ready());
        assert!(!guide.record_command(&ControlCommand::Pause, CommandId::new(85), &before));
        assert!(guide.observe_snapshot(&before, None));
        assert!(guide.record_command(&ControlCommand::Pause, CommandId::new(85), &before));
        guide.close();
        assert_eq!(guide.pending_command_id(), Some(CommandId::new(85)));
        guide.open();
        assert_eq!(guide.step(), LabGuideStep::Pause);
    }

    #[test]
    fn checkpoint_requires_fresh_pending_identity_and_unchanged_saved_source() {
        let mut owner = owner();
        let mut guide = at_checkpoint(&mut owner);
        let (pending, saved) = save_pair(&guide);
        assert!(
            !guide.observe_checkpoint_save(&saved),
            "an old catalog receipt cannot count"
        );
        let mut foreign = pending.clone();
        foreign.session_id = HostSessionId::new(99);
        assert!(!guide.observe_checkpoint_save(&foreign));
        assert!(guide.observe_checkpoint_save(&pending));
        let mut other_id = saved.clone();
        other_id.request_id = "older-save".into();
        assert!(!guide.observe_checkpoint_save(&other_id));
        let mut wrong_cut = saved.clone();
        if let LabCheckpointSaveState::Saved(cut) = &mut wrong_cut.state {
            cut.tick += 1;
        }
        assert!(!guide.observe_checkpoint_save(&wrong_cut));
        let mut bad_digest = saved.clone();
        if let LabCheckpointSaveState::Saved(cut) = &mut bad_digest.state {
            cut.world_digest = "a".repeat(64);
        }
        assert!(!guide.observe_checkpoint_save(&bad_digest));
        let mut moved_guide = guide.clone();
        let mut moved = owner.snapshot_hub().latest().as_ref().clone();
        moved.revision = moved.revision.checked_next().expect("next");
        moved.world.tick += 1;
        moved.revisions.scientific = moved.revisions.scientific.checked_next().expect("next");
        assert!(moved_guide.observe_snapshot(&moved, None));
        assert!(!moved_guide.observe_checkpoint_save(&saved));
        let retryable = LabCheckpointSave {
            state: LabCheckpointSaveState::Retryable {
                reason: "Temporary save refusal".to_owned(),
            },
            ..pending.clone()
        };
        assert!(guide.observe_checkpoint_save(&retryable));
        assert!(
            !guide.observe_checkpoint_save(&saved),
            "late failed-try result needs a fresh Pending"
        );
        let mut next_pending = pending;
        next_pending.request_id = "guide-save-retry".to_owned();
        let mut next_saved = saved;
        next_saved.request_id.clone_from(&next_pending.request_id);
        assert!(guide.observe_checkpoint_save(&next_pending));
        assert!(guide.observe_checkpoint_save(&next_saved));
        assert!(guide.ready());
    }

    #[test]
    fn only_matching_completed_two_arm_report_unlocks_descriptive_outcome() {
        let mut owner = owner();
        let mut guide = at_branch(&mut owner);
        let session = guide.source().expect("source").session_id;
        let (branch, report) = report_pair(&guide);
        let mut single_arm = report.clone();
        single_arm.treatment = None;
        assert!(!guide.observe_branch_report(session, &branch, &single_arm));
        let mut missing_intervention = report.clone();
        missing_intervention
            .treatment
            .as_mut()
            .expect("treatment")
            .command_applied_tick = None;
        assert!(!guide.observe_branch_report(session, &branch, &missing_intervention));
        let mut wrong_source = report.clone();
        wrong_source.source_byte_size += 1;
        assert!(!guide.observe_branch_report(session, &branch, &wrong_source));
        let mut wrong_seed = report.clone();
        wrong_seed.base_seed += 1;
        assert!(!guide.observe_branch_report(session, &branch, &wrong_seed));
        let mut wrong_artifact = report.clone();
        wrong_artifact.artifact_id = "other-artifact".into();
        assert!(!guide.observe_branch_report(session, &branch, &wrong_artifact));
        let mut old_cut = branch.clone();
        old_cut.checkpoint_id = "old-catalog-cut".into();
        assert!(!guide.observe_branch_report(session, &old_cut, &report));
        let mut running = branch.clone();
        running.status = "running".into();
        running.artifact_id = None;
        assert!(!guide.observe_branch_report(session, &running, &report));
        assert!(!guide.observe_branch_report(HostSessionId::new(99), &branch, &report));
        assert!(!guide.advance());
        assert!(guide.observe_branch_report(session, &branch, &report));
        guide.close();
        assert!(guide.ready());
        guide.open();
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Outcome);
        assert!(guide.evidence().contains("control population"));
        assert!(guide.evidence().contains("One seed: descriptive"));
        assert!(guide.evidence().len() <= MAX_GUIDE_TEXT_BYTES);
        assert!(guide.advance());
        assert!(guide.finished());
        assert!(!guide.ready());
        assert!(!guide.advance());
    }

    #[test]
    fn owner_switch_even_with_same_seed_resets_all_saved_and_command_evidence() {
        let mut owner = owner();
        let mut guide = at_branch(&mut owner);
        let old_source = guide.source().expect("source");
        let old_generation = guide.generation();
        assert!(guide.saved_checkpoint().is_some());
        let mut replacement = owner.snapshot_hub().latest().as_ref().clone();
        replacement.session_id = HostSessionId::new(old_source.session_id.get() + 1);
        replacement.revision = SnapshotRevision::new(0);
        assert!(guide.observe_snapshot(&replacement, None));
        assert_eq!(guide.step(), LabGuideStep::Identify);
        assert!(guide.generation() > old_generation);
        assert!(guide.saved_checkpoint().is_none());
        assert!(guide.pending_command_id().is_none());
        assert!(!guide.finished());
        assert_eq!(
            guide.source().expect("new owner").session_id,
            replacement.session_id
        );
        guide.observe_unavailable();
        assert!(!guide.advance());
        guide.observe_snapshot(&replacement, None);
        assert!(guide.advance());
        assert_eq!(guide.step(), LabGuideStep::Inspect);
    }

    #[test]
    fn generation_exhaustion_never_reuses_an_active_callback_identity() {
        let owner = owner();
        let frame = owner.snapshot_hub().latest();
        let mut guide = at_pause(&frame);
        guide.generation = u64::MAX - 1;
        guide.restart();
        assert_eq!(guide.generation(), u64::MAX);
        guide.open();
        assert!(!guide.observe_snapshot(&frame, None));
        assert!(!guide.ready());
        assert!(!guide.advance());
        assert!(!guide.record_command(&ControlCommand::Pause, CommandId::new(1), &frame));
        guide.restart();
        assert_eq!(guide.generation(), u64::MAX);
        assert!(!guide.ready());
    }
}
