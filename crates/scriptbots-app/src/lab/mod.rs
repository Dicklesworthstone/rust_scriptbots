//! The autonomous lab: the model proposes experiments, the simulation runs them.
//!
//! This module owns the provider boundary and nothing else yet. The loop, the
//! experiment schema, the statistics, and the notebook each land in their own
//! bead — but they can all be written and tested against [`llm::ScriptedClient`]
//! with no API key and no network, which is the entire reason this seam exists.
//!
//! The lab's action space is deliberately narrow: it may READ knobs and config,
//! and its only write is to PROPOSE an experiment. It never gets `apply_patch`.
//! That is not squeamishness — `ControlHandle::apply_patch` enqueues against the
//! LIVE world, so a lab that mutated it would produce confounded science and
//! could wedge a user's running simulation. Experiments run in fresh worlds.

pub mod llm;
pub mod spec;
