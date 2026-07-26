//! The autonomous lab: the model proposes experiments, the simulation runs them.
//!
//! [`spec::ExperimentSpec`] is the single proposal contract. Its derived tool
//! schema is offered through [`llm`], then the state machine deserializes and
//! validates that same type before the executor boundary can be crossed.
//! [`llm::ScriptedClient`] exercises the identical provider parser without an
//! API key or network.
//!
//! The lab's action space is deliberately narrow: it may READ knobs and config,
//! and its only write is to PROPOSE an experiment. It never gets `apply_patch`.
//! That is not squeamishness — `ControlHandle::apply_patch` enqueues against the
//! LIVE world, so a lab that mutated it would produce confounded science and
//! could wedge a user's running simulation. Experiments run in fresh worlds.

pub mod llm;
pub mod notebook;
pub mod spec;
pub mod stats;
