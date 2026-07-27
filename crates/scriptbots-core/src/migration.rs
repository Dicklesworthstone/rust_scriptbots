//! Cross-world agent transfer: the `emigrate`/`immigrate` pair (bd-16g.5.2).
//!
//! Before this module a [`WorldState`] had exactly two doors. Agents entered
//! through the reproduction/population stages and left through the internal
//! death pipeline. Neither is usable for migration: an arrival is not a birth
//! and a departure is not a death, and recording them as such would put
//! fabricated organisms and fabricated corpses into the lifecycle metrics that
//! every downstream science claim is computed from.
//!
//! # What crosses, and what must not
//!
//! [`MigratingAgent`] is the *whole organism* — its dense scalar row
//! ([`AgentData`]) and its complete phenotype ([`AgentRuntime`], which owns the
//! live brain executor). Carrying the brain is the entire point: an agent that
//! arrived with a freshly initialized brain would make gene flow between
//! islands a fiction, and the allopatric-speciation experiment (bd-16g.5) would
//! be measuring nothing.
//!
//! Three things deliberately do **not** cross:
//!
//! 1. **Per-agent generator state.** bd-16g.5.3 requires that "an immigrating
//!    agent carries its genome and its phenotype — never a generator, never a
//!    generator's state." [`AgentRngCountersV1`] exists per agent, so this is a
//!    live hazard rather than a hypothetical one: carrying the counters would
//!    make the destination's future draws a function of the *source's* history,
//!    silently coupling two islands whose independence is the experiment. The
//!    counters are dropped at departure and re-derived on arrival from the
//!    destination's own allocator — see [`WorldState::immigrate`].
//! 2. **The source's `AgentUid`.** Every world mints UIDs from its own private
//!    counter, so `AgentUid(1)` exists on every island and names a different
//!    organism on each. An arrival takes a fresh UID from the destination. The
//!    source UID is retained on the [`MigratingAgent`] as provenance
//!    ([`MigratingAgent::origin_uid`]) so the mover can journal *which* organism
//!    moved; it is never reused as identity.
//! 3. **Lineage parent UIDs.** `runtime.lineage` holds bare source-island UIDs.
//!    Feeding those into the destination's ancestry DAG would merge distinct
//!    organisms into one node — the exact bd-8jlj hazard — producing a
//!    phylogeny that is plausible, publishable and wrong. Arrivals are recorded
//!    with no local parents; the cross-island edge lives in the migration
//!    record, which is the only witness that can carry `(island, uid)` pairs.
//!
//! # Position across differently-sized worlds
//!
//! Islands may legally differ in world dimensions. A migrant's position is
//! captured at departure as the normalized fraction `(x / width, y / height)`
//! and re-expanded into the destination's bounds on arrival. Absolute
//! coordinates would place an agent outside a smaller destination; a fixed
//! spawn point would inject a systematic spatial bias into every migration.
//!
//! # A move is not a death plus a birth
//!
//! Nothing here writes a [`DeathRecord`](crate::DeathRecord). An arrival is
//! recorded as [`BirthOrigin::Injected`], which is the honest category (the
//! agent did enter this world, and it was not born here) and is already
//! excluded from the born-agent lifecycle metrics. The consequence, stated
//! plainly because it constrains every caller: **population state alone cannot
//! witness a migration.** From the source's census the emigrant is simply gone;
//! from the destination's it simply appeared. Conservation across a barrier is
//! therefore only checkable against the mover's own migration record, never
//! against two population counts.

use crate::{
    AgentData, AgentRuntime, AgentUid, BirthOrigin, BrainBinding, CombatEventFlags, Generation,
    ScientificStateError, SelectionState, WorldState, validated_world_unit_f32,
};
use thiserror::Error;

/// Typed rejection at a cross-world agent-transfer boundary.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum MigrationTransferError {
    /// No living agent on the source world carries the requested UID.
    #[error("no living agent with uid {uid} to emigrate")]
    UnknownAgent {
        /// The requested source-world UID.
        uid: u64,
    },
    /// The agent's brain attachment has no live executor, so moving it would
    /// deliver a brainless organism to the destination.
    ///
    /// This is refused rather than repaired: silently re-instantiating the
    /// family from the destination registry would hand the arrival a *random*
    /// brain while every population count still balanced, which is the shape of
    /// failure this module exists to make impossible.
    #[error("agent uid {uid} has brain kind '{kind}' with no live executor to transfer")]
    BrainNotTransferable {
        /// The source-world UID.
        uid: u64,
        /// The brain family kind label.
        kind: String,
    },
    /// The destination's brain registry does not offer the migrant's family
    /// under the same key and kind.
    ///
    /// Archipelago construction proves registry-lane equality across islands,
    /// so this cannot fire there — but this API is not archipelago-only, and a
    /// mismatched registry must fail loudly at the boundary rather than produce
    /// an agent whose registry key resolves to a different family.
    #[error(
        "destination brain registry has no family for key {registry_key} with kind '{kind}' \
         (found: {found})"
    )]
    UnknownBrainFamily {
        /// The migrant's registry key.
        registry_key: u64,
        /// The migrant's brain family kind label.
        kind: String,
        /// What the destination registry actually holds under that key.
        found: String,
    },
    /// The transfer crossed a scientific-state boundary that refused it.
    #[error(transparent)]
    ScientificState(#[from] ScientificStateError),
}

/// One complete organism in transit between worlds.
///
/// Constructed only by [`WorldState::emigrate`] and consumed only by
/// [`WorldState::immigrate`]. The value owns the agent outright: while it
/// exists the organism is present in no world at all, which is exactly why the
/// mover must treat the round trip as a single transaction and must journal it.
///
/// See the [module documentation](self) for what does and does not cross.
#[derive(Debug)]
pub struct MigratingAgent {
    data: AgentData,
    runtime: AgentRuntime,
    origin_uid: AgentUid,
    /// Departure position as `(x / width, y / height)` of the *source* world.
    normalized_position: [f32; 2],
}

impl MigratingAgent {
    /// The UID this organism held on the world it departed.
    ///
    /// Provenance only. It is never reinstated as identity on arrival, because
    /// UIDs are private to each world's allocator; pairing it with the source
    /// island id is what makes a move auditable.
    #[must_use]
    pub const fn origin_uid(&self) -> AgentUid {
        self.origin_uid
    }

    /// Departure position as a fraction of the source world's extent.
    #[must_use]
    pub const fn normalized_position(&self) -> [f32; 2] {
        self.normalized_position
    }

    /// Heritable generation counter carried across the move.
    #[must_use]
    pub const fn generation(&self) -> Generation {
        self.data.generation
    }

    /// Completed ticks the organism had lived when it departed.
    #[must_use]
    pub const fn age(&self) -> u32 {
        self.data.age
    }

    /// Energy reserve carried across the move.
    #[must_use]
    pub const fn energy(&self) -> f32 {
        self.runtime.energy
    }

    /// Brain family kind label, or `None` for an unbound organism.
    #[must_use]
    pub fn brain_kind(&self) -> Option<&str> {
        match &self.runtime.brain {
            BrainBinding::Unbound => None,
            BrainBinding::Protocol { kind, .. } | BrainBinding::Legacy { kind, .. } => Some(kind),
        }
    }
}

/// Live-executor and registry-key view of one brain attachment.
enum BrainTransferCheck<'a> {
    /// Nothing to check: the organism carries no brain.
    Unbound,
    /// A live executor is attached; `registry_key` is `Some` when the family is
    /// expected to exist in the destination registry under that key.
    Live {
        registry_key: Option<u64>,
        kind: &'a str,
    },
    /// The attachment names a family but holds no executor to move.
    Dead { kind: &'a str },
}

fn inspect_brain(binding: &BrainBinding) -> BrainTransferCheck<'_> {
    match binding {
        BrainBinding::Unbound => BrainTransferCheck::Unbound,
        BrainBinding::Protocol {
            registry_key,
            kind,
            evaluator,
            ..
        } => {
            if evaluator.is_some() {
                BrainTransferCheck::Live {
                    registry_key: Some(*registry_key),
                    kind,
                }
            } else {
                BrainTransferCheck::Dead { kind }
            }
        }
        BrainBinding::Legacy {
            runner,
            registry_key,
            kind,
        } => {
            if runner.is_some() {
                BrainTransferCheck::Live {
                    registry_key: *registry_key,
                    kind,
                }
            } else {
                BrainTransferCheck::Dead { kind }
            }
        }
    }
}

impl WorldState {
    /// Remove one living agent from this world and return it as a transferable
    /// organism.
    ///
    /// The agent is dropped from the arena, the identity map, the per-agent RNG
    /// counter map, the runtime map, and any pending persistence tail — every
    /// per-agent structure this world owns — so no half-present ghost can be
    /// left behind for a later stage to trip over.
    ///
    /// **This is not a death.** No [`DeathRecord`](crate::DeathRecord) is
    /// produced and no death metric moves, because the organism is not dead.
    /// The caller owns the returned value and owns the obligation to deliver it
    /// to exactly one destination via [`Self::immigrate`]; dropping it destroys
    /// the organism without any record that it ever left.
    ///
    /// # Errors
    ///
    /// - [`MigrationTransferError::UnknownAgent`] if no living agent holds `uid`.
    /// - [`MigrationTransferError::BrainNotTransferable`] if the agent's brain
    ///   attachment names a family but holds no live executor.
    /// - [`MigrationTransferError::ScientificState`] if an unresolved
    ///   persistence boundary forbids mutating scientific state right now.
    pub fn emigrate(&mut self, uid: AgentUid) -> Result<MigratingAgent, MigrationTransferError> {
        self.ensure_scientific_mutation_allowed("agents.emigrate")?;

        let Some(id) = self
            .agents
            .iter_handles()
            .find(|id| self.identities.get(*id).map(|identity| identity.uid) == Some(uid))
        else {
            return Err(MigrationTransferError::UnknownAgent { uid: uid.get() });
        };

        // Refuse BEFORE any removal: a failed emigration must leave the source
        // world exactly as it was, not holding a partially detached agent.
        if let Some(runtime) = self.runtime.get(id)
            && let BrainTransferCheck::Dead { kind } = inspect_brain(&runtime.brain)
        {
            return Err(MigrationTransferError::BrainNotTransferable {
                uid: uid.get(),
                kind: kind.to_owned(),
            });
        }

        let Some(runtime) = self.runtime.remove(id) else {
            return Err(MigrationTransferError::UnknownAgent { uid: uid.get() });
        };
        self.identities.remove(id);
        self.agent_rng_counters.remove(id);
        self.pending_persistence_runtime_tail.remove(id);
        self.pending_deaths.retain(|pending| *pending != id);
        let Some(data) = self.agents.remove(id) else {
            return Err(MigrationTransferError::UnknownAgent { uid: uid.get() });
        };

        let width = validated_world_unit_f32(self.config.world_width);
        let height = validated_world_unit_f32(self.config.world_height);
        let normalized_position = [
            normalized_fraction(data.position.x, width),
            normalized_fraction(data.position.y, height),
        ];

        diag_info!(
            uid = uid.get(),
            tick = self.tick.0,
            age = data.age,
            generation = data.generation.0,
            energy = runtime.energy,
            normalized_x = normalized_position[0],
            normalized_y = normalized_position[1],
            "agent emigrated from world"
        );

        Ok(MigratingAgent {
            data,
            runtime,
            origin_uid: uid,
            normalized_position,
        })
    }

    /// Admit a transferred organism into this world under a freshly minted
    /// local identity, and return that identity.
    ///
    /// Four things happen here that the caller cannot do for itself, and each
    /// one is a place where a plausible shortcut corrupts the science:
    ///
    /// 1. **A fresh `AgentUid`** comes from this world's own allocator. Reusing
    ///    the source UID would collide with a different organism already living
    ///    here.
    /// 2. **`AgentRngCountersV1` is re-derived, not carried.** The arrival
    ///    starts at this world's default ordinals, so its future draws come
    ///    from `(this world's root seed, its new local UID, ordinal)` and owe
    ///    nothing to the source's history. This is bd-16g.5.3's rule, and it is
    ///    load-bearing rather than decorative: carrying the counters would
    ///    couple two islands through a channel no digest gate inspects.
    /// 3. **Position is re-expanded** from the departure fraction into this
    ///    world's bounds, so a migrant into a smaller world lands inside it.
    /// 4. **Lineage is cleared.** The parent UIDs are the source world's, and
    ///    the destination's ancestry DAG keys on bare UIDs (bd-8jlj); grafting
    ///    them in would merge unrelated organisms. The cross-island parentage
    ///    edge belongs in the migration record, which can express `(island,
    ///    uid)`.
    ///
    /// The arrival is recorded as [`BirthOrigin::Injected`]: it entered this
    /// world without being born in it.
    ///
    /// # Errors
    ///
    /// - [`MigrationTransferError::UnknownBrainFamily`] if this world's brain
    ///   registry does not offer the migrant's family under the same key and
    ///   kind.
    /// - [`MigrationTransferError::BrainNotTransferable`] if the attachment
    ///   holds no live executor.
    /// - [`MigrationTransferError::ScientificState`] if the payload fails
    ///   validation or an unresolved persistence boundary forbids the mutation.
    pub fn immigrate(
        &mut self,
        migrant: MigratingAgent,
    ) -> Result<AgentUid, MigrationTransferError> {
        self.ensure_scientific_mutation_allowed("agents.immigrate")?;

        let MigratingAgent {
            mut data,
            mut runtime,
            origin_uid,
            normalized_position,
        } = migrant;

        match inspect_brain(&runtime.brain) {
            // Nothing to resolve: an unbound organism carries no family, and a
            // runner constructed outside the registry travels with the agent.
            BrainTransferCheck::Unbound
            | BrainTransferCheck::Live {
                registry_key: None, ..
            } => {}
            BrainTransferCheck::Dead { kind } => {
                return Err(MigrationTransferError::BrainNotTransferable {
                    uid: origin_uid.get(),
                    kind: kind.to_owned(),
                });
            }
            BrainTransferCheck::Live {
                registry_key: Some(key),
                kind,
            } => {
                let found = self.brain_registry.kind(key);
                if found != Some(kind) {
                    return Err(MigrationTransferError::UnknownBrainFamily {
                        registry_key: key,
                        kind: kind.to_owned(),
                        found: found.map_or_else(
                            || "<unregistered>".to_owned(),
                            std::borrow::ToOwned::to_owned,
                        ),
                    });
                }
            }
        }

        let width = validated_world_unit_f32(self.config.world_width);
        let height = validated_world_unit_f32(self.config.world_height);
        data.position.x = expand_fraction(normalized_position[0], width);
        data.position.y = expand_fraction(normalized_position[1], height);
        data.validate_at("migration.arrival.agent")?;

        // Parentage is source-local; see the method docs and bd-8jlj.
        runtime.lineage = [None, None];
        // Per-tick transient state from another world's last tick must not
        // reach this world's first tick: a stale spike flag would let the
        // destination attribute a death to combat that happened elsewhere.
        runtime.combat = CombatEventFlags::default();
        runtime.selection = SelectionState::None;
        runtime.validate_at("migration.arrival.agent.runtime")?;

        let record_tick = self.tick;
        // `insert_agent` mints the identity from this world's allocator and
        // installs DEFAULT `AgentRngCountersV1` — that default is precisely the
        // re-derivation bd-16g.5.3 requires, since every keyed substream is a
        // function of (this world's root seed, this new UID, ordinal).
        let id = self.insert_agent(data, runtime, record_tick, BirthOrigin::Injected);
        let uid = self
            .agent_uid(id)
            .ok_or_else(|| ScientificStateError::MissingAgentIdentity {
                path: "migration.arrival.identity".to_owned(),
            })?;

        diag_info!(
            origin_uid = origin_uid.get(),
            local_uid = uid.get(),
            tick = record_tick.0,
            position_x = data.position.x,
            position_y = data.position.y,
            "agent immigrated into world under a fresh local identity"
        );

        Ok(uid)
    }
}

/// Departure position as a fraction of the source extent, clamped to `[0, 1]`.
///
/// A zero or non-finite extent cannot produce a meaningful fraction; centering
/// is the only choice that is both defined and unbiased, and world validation
/// already rejects such configurations before a world exists.
fn normalized_fraction(value: f32, extent: f32) -> f32 {
    if extent > 0.0 && value.is_finite() {
        (value / extent).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Re-expand a departure fraction into the destination extent.
///
/// The result is held strictly inside the destination bounds: a coordinate
/// exactly equal to the extent is outside every world-space grid cell, and an
/// agent that lands there would be a boundary special case at every stage that
/// indexes the food or terrain grid.
fn expand_fraction(fraction: f32, extent: f32) -> f32 {
    if extent > 0.0 && fraction.is_finite() {
        let inside = f32::EPSILON.mul_add(-extent.max(1.0), extent);
        (fraction.clamp(0.0, 1.0) * extent).clamp(0.0, inside)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AgentId, BrainRunner, BrainSpawnError, INPUT_SIZE, OUTPUT_SIZE, Position, RandomStream,
        ScriptBotsConfig, Velocity,
    };

    const TEST_BRAIN_KIND: &str = "migration-test-brain";

    #[derive(Debug, Clone)]
    struct TestBrain {
        weight: f32,
    }

    impl BrainRunner for TestBrain {
        fn kind(&self) -> &'static str {
            TEST_BRAIN_KIND
        }

        fn tick(&mut self, inputs: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
            let mut outputs = [0.0f32; OUTPUT_SIZE];
            for (index, output) in outputs.iter_mut().enumerate() {
                *output = (inputs[index % INPUT_SIZE] * self.weight).clamp(0.0, 1.0);
            }
            outputs
        }

        fn state_digest(&self) -> Option<u64> {
            Some(u64::from(self.weight.to_bits()))
        }

        fn clone_runner(&self) -> Result<Option<Box<dyn BrainRunner>>, BrainSpawnError> {
            Ok(Some(Box::new(self.clone())))
        }

        fn mutate(
            &mut self,
            _rng: &mut dyn RandomStream,
            _rate: f32,
            _scale: f32,
        ) -> Result<(), BrainSpawnError> {
            Ok(())
        }
    }

    fn config(width: u32, height: u32) -> ScriptBotsConfig {
        ScriptBotsConfig {
            world_width: width,
            world_height: height,
            food_cell_size: 50,
            rng_seed: Some(0x5eed_1234_5678_9abc),
            persistence_interval: 0,
            population_minimum: 0,
            population_spawn_interval: 0,
            ..ScriptBotsConfig::default()
        }
    }

    /// A world with one registered brain family and `count` bound agents.
    fn world_with_agents(width: u32, height: u32, count: usize) -> (WorldState, u64) {
        let mut world = WorldState::new(config(width, height)).expect("valid config");
        let key = world
            .brain_registry_mut()
            .expect("registry mutable before first tick")
            .register_with_state_digest(TEST_BRAIN_KIND, 0xABCD, |_rng| {
                Ok(Box::new(TestBrain { weight: 0.5 }) as Box<dyn BrainRunner>)
            });
        for index in 0..count {
            #[allow(clippy::cast_precision_loss)]
            let offset = index as f32;
            let agent = AgentData::new(
                Position::new(10.0 + offset, 20.0 + offset),
                Velocity::default(),
                0.25,
                1.0,
                [0.1, 0.2, 0.3],
                0.0,
                false,
                7,
                Generation(3),
            );
            let id = world.try_spawn_agent(agent).expect("agent spawns");
            assert!(
                world.bind_agent_brain(id, key).expect("brain binds"),
                "the test brain family must bind"
            );
        }
        (world, key)
    }

    fn uids(world: &WorldState) -> Vec<AgentUid> {
        world
            .ordered_agent_rng_counters_v1()
            .expect("counters readable")
            .iter()
            .map(|state| state.agent_uid())
            .collect()
    }

    /// Resolve the live handle for a stable UID, failing loudly if it is gone.
    fn handle_of(world: &WorldState, uid: AgentUid) -> AgentId {
        world
            .agents()
            .iter_handles()
            .find(|id| world.agent_uid(*id) == Some(uid))
            .expect("the test's target agent must still be alive")
    }

    /// The whole organism crosses: phenotype, brain, age, generation, energy.
    #[test]
    fn bd_16g_5_2_round_trip_carries_the_whole_organism() {
        let (mut source, _) = world_with_agents(600, 300, 3);
        let (mut destination, _) = world_with_agents(600, 300, 2);

        let target = uids(&source)[1];
        let source_energy = {
            let id = handle_of(&source, target);
            source.agent_runtime(id).expect("runtime present").energy
        };

        let migrant = source.emigrate(target).expect("emigration succeeds");
        assert_eq!(migrant.origin_uid(), target);
        assert_eq!(migrant.age(), 7);
        assert_eq!(migrant.generation(), Generation(3));
        assert_eq!(migrant.brain_kind(), Some(TEST_BRAIN_KIND));

        assert_eq!(source.agent_count(), 2, "the emigrant left the source");
        assert!(
            !uids(&source).contains(&target),
            "the source must not retain the departed uid"
        );

        let arrived = destination
            .immigrate(migrant)
            .expect("immigration succeeds");
        assert_eq!(destination.agent_count(), 3, "the immigrant arrived");

        let id = handle_of(&destination, arrived);
        let runtime = destination.agent_runtime(id).expect("runtime present");
        assert_eq!(
            runtime.brain.kind(),
            Some(TEST_BRAIN_KIND),
            "the brain must cross, or gene flow between islands is a fiction"
        );
        assert!(
            runtime.brain.runner().is_some(),
            "the LIVE executor must cross, not just the family label"
        );
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(runtime.energy, source_energy, "energy is phenotype");
        }
        let data = destination.agents().snapshot(id).expect("row present");
        assert_eq!(data.age, 7);
        assert_eq!(data.generation, Generation(3));
    }

    /// An arrival takes a FRESH local uid; the source uid is provenance only.
    ///
    /// Both worlds mint from 1, so the source uid is guaranteed to already name
    /// a different organism in the destination. Reinstating it would not merely
    /// be untidy — it would make two organisms indistinguishable.
    #[test]
    fn bd_16g_5_2_arrival_takes_a_fresh_destination_uid() {
        let (mut source, _) = world_with_agents(600, 300, 2);
        let (mut destination, _) = world_with_agents(600, 300, 2);

        let target = uids(&source)[0];
        assert!(
            uids(&destination).contains(&target),
            "the destination must already hold this bare uid, or the test proves nothing"
        );

        let migrant = source.emigrate(target).expect("emigration succeeds");
        let arrived = destination
            .immigrate(migrant)
            .expect("immigration succeeds");

        assert_ne!(
            arrived, target,
            "the arrival must not reuse the source uid, which already names another organism"
        );
        let after = uids(&destination);
        assert_eq!(after.len(), 3);
        let mut unique = after.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 3, "every local uid must remain unique");
        assert!(after.contains(&arrived));
    }

    /// Per-agent RNG counters are RE-DERIVED on arrival, never carried.
    ///
    /// This is bd-16g.5.3's rule and the one whose violation is completely
    /// silent: carried counters make the destination's future draws a function
    /// of the source's history while every determinism gate stays green,
    /// because those gates vary thread counts, not island counts.
    #[test]
    fn bd_16g_5_2_rng_counters_are_rederived_not_smuggled() {
        let (mut source, _) = world_with_agents(600, 300, 1);
        let (mut destination, _) = world_with_agents(600, 300, 1);

        // Advance the emigrant's counters so "carried" and "re-derived" differ.
        let target = uids(&source)[0];
        let id = handle_of(&source, target);
        source
            .agent_rng_counters
            .get_mut(id)
            .expect("counters present")
            .take_reproduction_attempt()
            .expect("ordinal available");
        let carried = source.agent_rng_counters(id).expect("counters present");
        assert_ne!(
            carried.reproduction_attempt_ordinal(),
            0,
            "the source counters must be non-default, or nothing distinguishes the two outcomes"
        );

        let migrant = source.emigrate(target).expect("emigration succeeds");
        let arrived = destination
            .immigrate(migrant)
            .expect("immigration succeeds");

        let arrived_id = handle_of(&destination, arrived);
        let counters = destination
            .agent_rng_counters(arrived_id)
            .expect("counters present");
        assert_eq!(
            counters.reproduction_attempt_ordinal(),
            0,
            "the arrival's ordinals must restart under the destination's allocator"
        );
        assert_eq!(counters.birth_ordinal(), 0);
        assert_eq!(counters.brain_initialization_ordinal(), 0);
    }

    /// Lineage parent uids are source-local and must not follow the organism.
    #[test]
    fn bd_16g_5_2_lineage_is_cleared_on_arrival() {
        let (mut source, _) = world_with_agents(600, 300, 1);
        let (mut destination, _) = world_with_agents(600, 300, 1);

        let target = uids(&source)[0];
        let id = handle_of(&source, target);
        source.runtime.get_mut(id).expect("runtime present").lineage =
            [Some(AgentUid(41)), Some(AgentUid(42))];

        let migrant = source.emigrate(target).expect("emigration succeeds");
        let arrived = destination
            .immigrate(migrant)
            .expect("immigration succeeds");
        let arrived_id = handle_of(&destination, arrived);
        assert_eq!(
            destination
                .agent_runtime(arrived_id)
                .expect("runtime present")
                .lineage,
            [None, None],
            "bare source-island parent uids would merge unrelated organisms (bd-8jlj)"
        );
    }

    /// A migrant into a SMALLER world lands inside it, at the same relative spot.
    #[test]
    fn bd_16g_5_2_position_remaps_into_destination_bounds() {
        let (mut source, _) = world_with_agents(1000, 500, 1);
        let (mut destination, _) = world_with_agents(200, 100, 1);

        let target = uids(&source)[0];
        let id = handle_of(&source, target);
        source
            .try_update_agent(id, |data, _| {
                data.position = Position::new(750.0, 400.0);
            })
            .expect("position update accepted");

        let migrant = source.emigrate(target).expect("emigration succeeds");
        let fraction = migrant.normalized_position();
        assert!((fraction[0] - 0.75).abs() < 1e-6, "got {}", fraction[0]);
        assert!((fraction[1] - 0.8).abs() < 1e-6, "got {}", fraction[1]);

        let arrived = destination
            .immigrate(migrant)
            .expect("immigration succeeds");
        let arrived_id = handle_of(&destination, arrived);
        let landed = destination
            .agents()
            .snapshot(arrived_id)
            .expect("row present");
        assert!(
            (landed.position.x - 150.0).abs() < 1e-3,
            "x remapped to {}",
            landed.position.x
        );
        assert!(
            (landed.position.y - 80.0).abs() < 1e-3,
            "y remapped to {}",
            landed.position.y
        );
        assert!(
            landed.position.x < 200.0 && landed.position.y < 100.0,
            "the arrival must land strictly inside the destination bounds"
        );
    }

    /// An agent at the far edge of a large world still lands inside a small one.
    #[test]
    fn bd_16g_5_2_far_edge_migrant_stays_inside_a_smaller_world() {
        let (mut source, _) = world_with_agents(4000, 4000, 1);
        let (mut destination, _) = world_with_agents(100, 100, 1);

        let target = uids(&source)[0];
        let id = handle_of(&source, target);
        source
            .try_update_agent(id, |data, _| {
                data.position = Position::new(3999.9, 3999.9);
            })
            .expect("position update accepted");

        let migrant = source.emigrate(target).expect("emigration succeeds");
        let arrived = destination
            .immigrate(migrant)
            .expect("immigration succeeds");
        let arrived_id = handle_of(&destination, arrived);
        let landed = destination
            .agents()
            .snapshot(arrived_id)
            .expect("row present");
        assert!(
            landed.position.x >= 0.0 && landed.position.x < 100.0,
            "x landed at {}",
            landed.position.x
        );
        assert!(
            landed.position.y >= 0.0 && landed.position.y < 100.0,
            "y landed at {}",
            landed.position.y
        );
    }

    /// Emigrating an absent uid is a typed refusal, not a panic or a silent no-op.
    #[test]
    fn bd_16g_5_2_unknown_uid_is_a_typed_refusal() {
        let (mut source, _) = world_with_agents(600, 300, 1);
        let before = source.agent_count();
        let error = source
            .emigrate(AgentUid(9999))
            .expect_err("an absent uid must be refused");
        assert_eq!(error, MigrationTransferError::UnknownAgent { uid: 9999 });
        assert_eq!(
            source.agent_count(),
            before,
            "a refused emigration must not disturb the source population"
        );
    }

    /// A destination without the migrant's brain family refuses the arrival,
    /// and refuses it WITHOUT admitting the agent.
    #[test]
    fn bd_16g_5_2_destination_without_the_family_refuses_the_arrival() {
        let (mut source, key) = world_with_agents(600, 300, 1);
        let mut destination = WorldState::new(config(600, 300)).expect("valid config");

        let target = uids(&source)[0];
        let migrant = source.emigrate(target).expect("emigration succeeds");
        let error = destination
            .immigrate(migrant)
            .expect_err("an unknown family must be refused");
        assert!(
            matches!(
                error,
                MigrationTransferError::UnknownBrainFamily { registry_key, .. }
                    if registry_key == key
            ),
            "unexpected error: {error:?}"
        );
        assert_eq!(
            destination.agent_count(),
            0,
            "a refused arrival must not be admitted"
        );
    }

    /// Departure and arrival are population-symmetric across a pair of worlds.
    ///
    /// This is the arithmetic half of conservation and it is deliberately NOT
    /// claimed to be the whole of it: censuses cannot tell a move from a death
    /// plus a birth, because the arrival necessarily takes a fresh local uid.
    /// The migration record is the only witness of the movement itself.
    #[test]
    fn bd_16g_5_2_transfer_conserves_the_pair_population() {
        let (mut source, _) = world_with_agents(600, 300, 5);
        let (mut destination, _) = world_with_agents(300, 600, 4);
        let total = source.agent_count() + destination.agent_count();

        for _ in 0..3 {
            let target = uids(&source)[0];
            let migrant = source.emigrate(target).expect("emigration succeeds");
            assert_eq!(
                source.agent_count() + destination.agent_count(),
                total - 1,
                "while in transit the organism belongs to no world; that gap is exactly \
                 why the transfer must be one transaction"
            );
            destination
                .immigrate(migrant)
                .expect("immigration succeeds");
            assert_eq!(
                source.agent_count() + destination.agent_count(),
                total,
                "the pair population must be restored by arrival"
            );
        }
        assert_eq!(source.agent_count(), 2);
        assert_eq!(destination.agent_count(), 7);
    }
}
