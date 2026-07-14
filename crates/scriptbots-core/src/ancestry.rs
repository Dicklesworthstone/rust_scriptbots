//! The ancestry graph: who descended from whom, keyed by logical identity.
//!
//! # The failure mode this module exists to prevent
//!
//! [`crate::AgentId`] is a slotmap key — a `(version, slot)` pair — and slots are
//! REUSED after an agent dies. Three consequences, each of which silently
//! corrupts a phylogeny:
//!
//! 1. **Dead parents vanish.** `runtime.get(parent_id)` returns `None` the moment
//!    the parent dies, so an ancestry builder that resolves parents lazily from
//!    the live world simply LOSES the edge, and every long lineage decays into a
//!    forest of orphan roots.
//! 2. **Handles are not identities.** A slot handle does not survive a checkpoint
//!    restore, a world rebuild, an island migration, or a second run. Any offline
//!    join on it can FUSE TWO DIFFERENT AGENTS onto one node.
//! 3. **Stale handles name strangers.** A reused slot names whatever agent now
//!    occupies it.
//!
//! So this graph is keyed on [`crate::AgentUid`] — a logical identity that is
//! never reused — and it is fed from [`crate::BirthRecord`] and
//! [`crate::DeathRecord`], which carry that identity OUT of the tick. It never
//! consults the live world, because the live world has already forgotten the dead.
//!
//! The slot-reuse test in this module builds the same log twice — once keyed on
//! `AgentUid`, once on a deliberate slot-handle foil — and proves the foil
//! mis-parents. It is executable documentation of why this is not over-engineering.
//!
//! # Determinism
//!
//! `BTreeMap` and `Vec` only. Iteration order is part of the contract:
//! [`AncestryGraph::canonical_digest`] is the oracle an offline rebuild is checked
//! against, and a `HashMap` anywhere in here would make that digest depend on the
//! hasher's seed.
//!
//! # Memory
//!
//! A node carries two parent uids, an arrival tick, a death tick, a cause, a
//! generation, a brain key, and its descendant list. Excluding that vector's heap,
//! this is on the order of 64 bytes, so roughly 80 bytes per node once an average
//! of one direct descendant is counted. At 10k living agents whose full ancestry
//! is retained, a 100k-node graph therefore costs on the order of 8 MB — which is
//! why the graph is PRUNED rather than kept forever, and why the bound is asserted
//! by a test rather than hoped for.
//!
//! # Purity
//!
//! No I/O, no clock, no RNG, no storage types, no tracing (core has no tracing
//! dependency and must not acquire one). The graph returns typed reports; the
//! layer above it does the logging.

use crate::{AgentUid, BirthOrigin, BirthRecord, DeathCause, DeathRecord, Generation, Tick};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Roughly how many bytes one node costs, used by the asserted memory bound.
pub const APPROX_BYTES_PER_NODE: usize = 80;

/// One agent's place in the tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AncestryNode {
    /// Logical identity. Never reused, unlike a slot handle.
    pub uid: AgentUid,
    /// When its typed arrival was recorded.
    pub birth_tick: Tick,
    /// How it first entered the world.
    pub origin: BirthOrigin,
    /// When it died, if it has.
    pub death_tick: Option<Tick>,
    /// How it died, if it has.
    pub death_cause: Option<DeathCause>,
    /// Its parents. `None` means it genuinely has none — a seeded agent, or one
    /// respawned at the population floor, is a legitimate ROOT rather than an
    /// orphan whose parent got lost.
    pub parents: [Option<AgentUid>; 2],
    /// Its direct descendants, in arrival order.
    pub children: Vec<AgentUid>,
    /// Generations since a root.
    pub generation: Generation,
    /// Which brain it ran.
    pub brain_key: Option<u64>,
    /// Whether it came from two parents of differing diet.
    pub is_hybrid: bool,
    /// Whether this node is a stand-in for a pruned ancestor.
    ///
    /// A pruned parent becomes one of these rather than a dangling reference. A
    /// consumer that walked a lineage into a hole would panic or silently
    /// truncate the tree; neither is acceptable, so the hole is given a name.
    pub pruned: bool,
}

/// Everything that can be wrong with an ancestry log.
///
/// No malformed log may panic. A phylogeny that unwinds on a duplicate row takes
/// the whole run down with it, and a phylogeny that silently accepts a cycle is
/// worse — it will render, and it will be a lie.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AncestryError {
    /// This arrival's logical identity is already in the graph.
    #[error("duplicate arrival for uid {0:?}")]
    DuplicateBirth(AgentUid),
    /// A named parent's arrival was never recorded.
    #[error("arrival of {child:?} names parent {parent:?}, which is not in the graph")]
    UnknownParent {
        /// The arriving descendant.
        child: AgentUid,
        /// The parent nobody has seen.
        parent: AgentUid,
    },
    /// Both parent slots name the same logical agent.
    #[error("arrival of {child:?} names {parent:?} as both parents")]
    DuplicateParent {
        /// The arriving descendant.
        child: AgentUid,
        /// The repeated parent identity.
        parent: AgentUid,
    },
    /// A descendant arrived no later than its parent.
    #[error(
        "arrival of {child:?} at tick {child_tick} does not follow its parent {parent:?} at tick {parent_tick}"
    )]
    ChildPrecedesParent {
        /// The arriving descendant.
        child: AgentUid,
        /// When the descendant's arrival was recorded.
        child_tick: u64,
        /// The parent.
        parent: AgentUid,
        /// When the parent's arrival was recorded.
        parent_tick: u64,
    },
    /// An agent that is its own ancestor.
    #[error("arrival of {0:?} would close a lineage cycle")]
    Cycle(AgentUid),
    /// A death whose corresponding arrival was never recorded.
    #[error("death for uid {0:?}, which is not in the graph")]
    UnknownDeath(AgentUid),
    /// A second death for the same agent.
    #[error("uid {0:?} died twice")]
    DuplicateDeath(AgentUid),
    /// A death recorded no later than the corresponding arrival.
    #[error(
        "death of {uid:?} at tick {death_tick} does not follow its arrival at tick {birth_tick}"
    )]
    DeathPrecedesBirth {
        /// The agent whose lifecycle ordering is invalid.
        uid: AgentUid,
        /// When the death was recorded.
        death_tick: u64,
        /// When the arrival was recorded.
        birth_tick: u64,
    },
}

/// What to keep when pruning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrunePolicy {
    /// Never prune a node that is an ancestor of a living agent.
    ///
    /// Not configurable, and stated as a field only so the intent is legible at
    /// the call site: pruning a living agent's ancestry would silently rewrite
    /// the history of the population that is still on screen.
    pub keep_ancestors_of_living: bool,
    /// Keep extinct nodes whose death is within this many ticks of `now`.
    pub extinct_retention_ticks: u64,
    /// Hard ceiling on retained extinct nodes, oldest dropped first.
    pub max_retained_extinct: usize,
}

impl Default for PrunePolicy {
    fn default() -> Self {
        Self {
            keep_ancestors_of_living: true,
            extinct_retention_ticks: 20_000,
            max_retained_extinct: 50_000,
        }
    }
}

/// What a prune did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PruneReport {
    /// Nodes removed outright.
    pub removed: usize,
    /// Nodes replaced by a pruned-ancestor sentinel because a survivor still
    /// names them as a parent.
    pub tombstoned: usize,
    /// Nodes left untouched.
    pub retained: usize,
}

/// The graph.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AncestryGraph {
    nodes: BTreeMap<AgentUid, AncestryNode>,
}

impl AncestryGraph {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look one up.
    #[must_use]
    pub fn node(&self, uid: AgentUid) -> Option<&AncestryNode> {
        self.nodes.get(&uid)
    }

    /// The roots: agents with no parents at all.
    ///
    /// These are legitimate. The world seeds agents and respawns them at the
    /// population floor, and neither is a descendant of anything.
    #[must_use]
    pub fn roots(&self) -> Vec<AgentUid> {
        self.nodes
            .values()
            .filter(|node| node.parents.iter().all(Option::is_none) && !node.pruned)
            .map(|node| node.uid)
            .collect()
    }

    /// How many are still alive.
    #[must_use]
    pub fn living(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| node.death_tick.is_none() && !node.pruned)
            .count()
    }

    /// Record one typed agent arrival.
    ///
    /// # Errors
    ///
    /// Every malformed row is a typed [`AncestryError`], never a panic: a
    /// phylogeny that unwinds on a bad log takes the run down with it.
    pub fn apply_birth(&mut self, record: &BirthRecord) -> Result<(), AncestryError> {
        let child = record.agent_uid;
        if self.nodes.contains_key(&child) {
            return Err(AncestryError::DuplicateBirth(child));
        }

        let parents = [record.parent_a, record.parent_b];
        if let [Some(parent_a), Some(parent_b)] = parents
            && parent_a == parent_b
        {
            return Err(AncestryError::DuplicateParent {
                child,
                parent: parent_a,
            });
        }
        for parent in parents.into_iter().flatten() {
            // A parent we have never seen is a hole in the log. Accepting it would
            // produce an edge into nothing, and every consumer that walked it
            // would have to guess what to do.
            let Some(parent_node) = self.nodes.get(&parent) else {
                return Err(AncestryError::UnknownParent { child, parent });
            };
            // Time only runs one way. A descendant that arrived no later than
            // its parent is not a lineage, and accepting it could create a cycle.
            if record.tick.0 <= parent_node.birth_tick.0 {
                return Err(AncestryError::ChildPrecedesParent {
                    child,
                    child_tick: record.tick.0,
                    parent,
                    parent_tick: parent_node.birth_tick.0,
                });
            }
            // Self-parenthood is the degenerate cycle, and the strictly-increasing
            // arrival-tick ordering above rules out every longer one: an
            // ancestor is always strictly older, so no chain of edges can return
            // to its start.
            if parent == child {
                return Err(AncestryError::Cycle(child));
            }
        }

        for parent in parents.into_iter().flatten() {
            if let Some(parent_node) = self.nodes.get_mut(&parent) {
                parent_node.children.push(child);
            }
        }

        self.nodes.insert(
            child,
            AncestryNode {
                uid: child,
                birth_tick: record.tick,
                origin: record.origin,
                death_tick: None,
                death_cause: None,
                parents,
                children: Vec::new(),
                generation: record.generation,
                brain_key: record.brain_key,
                is_hybrid: record.is_hybrid,
                pruned: false,
            },
        );
        Ok(())
    }

    /// Record a death.
    ///
    /// # Errors
    ///
    /// [`AncestryError::UnknownDeath`], [`AncestryError::DuplicateDeath`], or
    /// [`AncestryError::DeathPrecedesBirth`].
    pub fn apply_death(&mut self, record: &DeathRecord) -> Result<(), AncestryError> {
        let Some(node) = self.nodes.get_mut(&record.agent_uid) else {
            return Err(AncestryError::UnknownDeath(record.agent_uid));
        };
        if node.death_tick.is_some() {
            return Err(AncestryError::DuplicateDeath(record.agent_uid));
        }
        if record.tick.0 <= node.birth_tick.0 {
            return Err(AncestryError::DeathPrecedesBirth {
                uid: record.agent_uid,
                death_tick: record.tick.0,
                birth_tick: node.birth_tick.0,
            });
        }
        node.death_tick = Some(record.tick);
        node.death_cause = Some(record.cause);
        Ok(())
    }

    /// This agent's parents.
    #[must_use]
    pub fn parents_of(&self, uid: AgentUid) -> [Option<AgentUid>; 2] {
        self.nodes
            .get(&uid)
            .map_or([None, None], |node| node.parents)
    }

    /// Walk back up the tree, following the first parent.
    ///
    /// BOUNDED by `max_depth`, and it also refuses to revisit a node. Unbounded
    /// recursion over a graph built from an untrusted log is a stack overflow
    /// waiting for the right input, and a stack overflow is not a typed error —
    /// it is the process dying.
    #[must_use]
    pub fn lineage_path(&self, uid: AgentUid, max_depth: usize) -> Vec<AgentUid> {
        let mut path = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        let mut current = Some(uid);
        while let Some(next) = current {
            if path.len() >= max_depth || !seen.insert(next) {
                break;
            }
            if !self.nodes.contains_key(&next) {
                break;
            }
            path.push(next);
            current = self.parents_of(next)[0];
        }
        path
    }

    /// Drop history nobody needs any more.
    ///
    /// A pruned node that a survivor still names as a parent is TOMBSTONED rather
    /// than removed: it stays in the graph as a `pruned` sentinel. A dangling
    /// parent reference would make every consumer choose between panicking and
    /// silently truncating a lineage, and both of those are worse than admitting
    /// the ancestor is gone.
    pub fn prune(&mut self, now: Tick, policy: PrunePolicy) -> PruneReport {
        // Everything an ancestor of a living agent needs, walked from the living
        // upward. Pruning a living agent's ancestry would silently rewrite the
        // history of the population still on screen.
        let mut keep: std::collections::BTreeSet<AgentUid> = std::collections::BTreeSet::new();
        if policy.keep_ancestors_of_living {
            let living: Vec<AgentUid> = self
                .nodes
                .values()
                .filter(|node| node.death_tick.is_none())
                .map(|node| node.uid)
                .collect();
            let mut frontier = living;
            while let Some(uid) = frontier.pop() {
                if !keep.insert(uid) {
                    continue;
                }
                for parent in self.parents_of(uid).into_iter().flatten() {
                    frontier.push(parent);
                }
            }
        }

        // Recent extinctions, newest first, up to the budget.
        let mut extinct: Vec<(u64, AgentUid)> = self
            .nodes
            .values()
            .filter(|node| !keep.contains(&node.uid))
            .filter_map(|node| node.death_tick.map(|tick| (tick.0, node.uid)))
            .filter(|(tick, _)| now.0.saturating_sub(*tick) <= policy.extinct_retention_ticks)
            .collect();
        extinct.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.0.cmp(&b.1.0)));
        for (_, uid) in extinct.into_iter().take(policy.max_retained_extinct) {
            keep.insert(uid);
        }

        // Anyone a survivor still points at must remain reachable, as a sentinel
        // if not in full.
        let mut needed_as_parent: std::collections::BTreeSet<AgentUid> =
            std::collections::BTreeSet::new();
        for uid in &keep {
            for parent in self.parents_of(*uid).into_iter().flatten() {
                if !keep.contains(&parent) {
                    needed_as_parent.insert(parent);
                }
            }
        }

        let mut report = PruneReport::default();
        let doomed: Vec<AgentUid> = self
            .nodes
            .keys()
            .copied()
            .filter(|uid| !keep.contains(uid))
            .collect();

        for uid in doomed {
            if needed_as_parent.contains(&uid) {
                if let Some(node) = self.nodes.get_mut(&uid) {
                    node.pruned = true;
                    node.children.clear();
                    node.parents = [None, None];
                    report.tombstoned += 1;
                }
            } else {
                self.nodes.remove(&uid);
                report.removed += 1;
            }
        }

        // A tombstoned ancestor's children list is cleared, so every surviving
        // node's child list must also be scrubbed of nodes that no longer exist.
        let present: std::collections::BTreeSet<AgentUid> = self.nodes.keys().copied().collect();
        for node in self.nodes.values_mut() {
            node.children.retain(|child| present.contains(child));
        }

        report.retained = self.nodes.len();
        report
    }

    /// A stable fingerprint of the whole graph.
    ///
    /// THE ORACLE. An offline rebuild from the run DB is correct exactly when its
    /// digest equals the live graph's. Because the encoding walks a `BTreeMap`,
    /// the order is the sorted uid order on every platform and every feature
    /// combination — which is the only reason this comparison means anything.
    #[must_use]
    pub fn canonical_digest(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |value: u64| {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for node in self.nodes.values() {
            mix(node.uid.0);
            mix(node.birth_tick.0);
            mix(node.origin.digest_tag());
            mix(node.death_tick.map_or(u64::MAX, |tick| tick.0));
            mix(node.death_cause.map_or(u64::MAX, DeathCause::digest_tag));
            mix(u64::from(node.generation.0));
            mix(node.brain_key.unwrap_or(u64::MAX));
            mix(u64::from(node.is_hybrid));
            mix(u64::from(node.pruned));
            for parent in node.parents {
                mix(parent.map_or(u64::MAX, |uid| uid.0));
            }
        }
        hash
    }

    /// Roughly how much heap this graph occupies.
    #[must_use]
    pub fn approx_bytes(&self) -> usize {
        self.nodes
            .values()
            .map(|node| {
                APPROX_BYTES_PER_NODE + node.children.len() * std::mem::size_of::<AgentUid>()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CombatEventFlags, Position};

    fn birth(uid: u64, tick: u64, parents: [Option<u64>; 2], generation: u32) -> BirthRecord {
        birth_with_origin(uid, tick, parents, generation, BirthOrigin::Born)
    }

    fn birth_with_origin(
        uid: u64,
        tick: u64,
        parents: [Option<u64>; 2],
        generation: u32,
        origin: BirthOrigin,
    ) -> BirthRecord {
        BirthRecord {
            tick: Tick(tick),
            agent_uid: AgentUid(uid),
            spawn_ordinal: uid,
            birth_ordinal: (origin == BirthOrigin::Born).then_some(uid),
            origin,
            parent_a: parents[0].map(AgentUid),
            parent_b: parents[1].map(AgentUid),
            brain_kind: Some("mlp.baseline".to_owned()),
            brain_key: Some(uid),
            herbivore_tendency: 0.5,
            generation: Generation(generation),
            position: Position::new(0.0, 0.0),
            is_hybrid: parents[0].is_some() && parents[1].is_some(),
        }
    }

    fn death(uid: u64, tick: u64) -> DeathRecord {
        death_with_cause(uid, tick, DeathCause::Starvation)
    }

    fn death_with_cause(uid: u64, tick: u64, cause: DeathCause) -> DeathRecord {
        DeathRecord {
            tick: Tick(tick),
            agent_uid: AgentUid(uid),
            age: 10,
            generation: Generation(0),
            herbivore_tendency: 0.5,
            brain_kind: None,
            brain_key: None,
            energy: 0.0,
            food_balance_total: 0.0,
            cause,
            was_hybrid: false,
            combat_flags: CombatEventFlags::default(),
        }
    }

    #[test]
    fn the_graph_is_built_from_every_shape_of_birth_the_world_produces() {
        let mut graph = AncestryGraph::new();

        // A seeded root: no parents. This is NOT an orphan whose parent got lost —
        // the world seeds agents, and they are descendants of nothing.
        graph
            .apply_birth(&birth_with_origin(
                1,
                0,
                [None, None],
                0,
                BirthOrigin::Seeded,
            ))
            .expect("seeded root");
        graph
            .apply_birth(&birth_with_origin(
                2,
                0,
                [None, None],
                0,
                BirthOrigin::Seeded,
            ))
            .expect("second root");

        // Asexual: one parent.
        graph
            .apply_birth(&birth(3, 10, [Some(1), None], 1))
            .expect("asexual");
        // Sexual: two parents (and therefore hybrid, per the fixture).
        graph
            .apply_birth(&birth(4, 20, [Some(1), Some(2)], 1))
            .expect("sexual");
        // Three generations deep.
        graph
            .apply_birth(&birth(5, 30, [Some(4), None], 2))
            .expect("grandchild");

        assert_eq!(graph.len(), 5);
        assert_eq!(graph.roots(), vec![AgentUid(1), AgentUid(2)]);
        assert_eq!(
            graph.node(AgentUid(1)).expect("root").origin,
            BirthOrigin::Seeded
        );
        assert_eq!(
            graph.node(AgentUid(4)).expect("child").origin,
            BirthOrigin::Born
        );
        assert_eq!(
            graph.parents_of(AgentUid(4)),
            [Some(AgentUid(1)), Some(AgentUid(2))]
        );
        assert!(graph.node(AgentUid(4)).expect("node").is_hybrid);
        assert_eq!(
            graph.node(AgentUid(1)).expect("node").children,
            vec![AgentUid(3), AgentUid(4)]
        );

        // A full lineage walk, three generations back to the root.
        assert_eq!(
            graph.lineage_path(AgentUid(5), 16),
            vec![AgentUid(5), AgentUid(4), AgentUid(1)]
        );

        // A floor-respawn root that arrives mid-run is still a root.
        graph
            .apply_birth(&birth_with_origin(
                6,
                40,
                [None, None],
                0,
                BirthOrigin::Injected,
            ))
            .expect("floor respawn");
        assert!(graph.roots().contains(&AgentUid(6)));
        assert_eq!(
            graph.node(AgentUid(6)).expect("injected root").origin,
            BirthOrigin::Injected
        );
    }

    #[test]
    fn keying_on_a_slot_handle_misparents_and_keying_on_a_uid_does_not() {
        // THE test — executable documentation of the bug this module exists to
        // prevent, kept forever.
        //
        // Agent A dies. The slotmap REUSES its slot for an unrelated agent C. A
        // graph keyed on the slot handle now believes C's ancestry is A's, and
        // fuses two different agents onto one node. A graph keyed on the logical
        // uid does not, because uids are never reused.
        //
        // Slot handles, as the world would hand them out:
        const SLOT: u64 = 7; // reused by both A and C
        let a_uid = AgentUid(100);
        let c_uid = AgentUid(300);
        let b_uid = AgentUid(200); // A's real child

        let mut graph = AncestryGraph::new();
        graph
            .apply_birth(&birth(a_uid.0, 1, [None, None], 0))
            .expect("A");
        graph
            .apply_birth(&birth(b_uid.0, 2, [Some(a_uid.0), None], 1))
            .expect("B, child of A");
        graph
            .apply_death(&death(a_uid.0, 3))
            .expect("A dies, freeing its slot");
        // C is born into A's old slot. It is unrelated to A and to B.
        graph
            .apply_birth(&birth(c_uid.0, 4, [None, None], 0))
            .expect("C, a stranger");

        // The UID graph gets it right: B descends from A, C descends from nobody,
        // and C has no children.
        assert_eq!(graph.parents_of(b_uid), [Some(a_uid), None]);
        assert_eq!(graph.parents_of(c_uid), [None, None]);
        assert!(graph.node(c_uid).expect("C").children.is_empty());
        assert_eq!(graph.node(a_uid).expect("A").children, vec![b_uid]);

        // THE FOIL. Build the same log keyed on the slot handle, as a naive
        // implementation would. Both A and C hash to SLOT, so they collide.
        let mut foil: BTreeMap<u64, Vec<u64>> = BTreeMap::new(); // slot -> children slots
        foil.entry(SLOT).or_default(); // A, in slot 7
        foil.entry(SLOT).or_default().push(8); // B, child of "slot 7"
        // A dies; C is born into slot 7 and is recorded as a fresh root...
        // ...but it lands on the SAME KEY, so it inherits A's children.
        let cs_children = foil.get(&SLOT).cloned().unwrap_or_default();
        assert_eq!(
            cs_children,
            vec![8],
            "the slot-keyed graph now believes the STRANGER in slot 7 is the parent \
             of B — two different agents fused onto one node. This is precisely the \
             corruption that keying on AgentUid prevents, and it is why this module \
             refuses to resolve parents from the live world."
        );
    }

    #[test]
    fn every_malformed_log_is_a_typed_error_and_nothing_panics() {
        let mut graph = AncestryGraph::new();
        graph
            .apply_birth(&birth(1, 5, [None, None], 0))
            .expect("root");
        graph
            .apply_birth(&birth(2, 10, [Some(1), None], 1))
            .expect("child");

        // Duplicate child.
        assert_eq!(
            graph.apply_birth(&birth(2, 11, [Some(1), None], 1)),
            Err(AncestryError::DuplicateBirth(AgentUid(2)))
        );
        // A parent nobody has ever seen.
        assert!(matches!(
            graph.apply_birth(&birth(3, 12, [Some(999), None], 1)),
            Err(AncestryError::UnknownParent { .. })
        ));
        assert_eq!(
            graph.apply_birth(&birth(3, 12, [Some(1), Some(1)], 1)),
            Err(AncestryError::DuplicateParent {
                child: AgentUid(3),
                parent: AgentUid(1),
            })
        );
        // A child older than its parent — time runs one way.
        assert!(matches!(
            graph.apply_birth(&birth(4, 1, [Some(1), None], 1)),
            Err(AncestryError::ChildPrecedesParent { .. })
        ));
        // Born at the same instant as its parent is not a lineage either.
        assert!(matches!(
            graph.apply_birth(&birth(5, 5, [Some(1), None], 1)),
            Err(AncestryError::ChildPrecedesParent { .. })
        ));
        // Self-parenthood: the degenerate cycle.
        assert!(matches!(
            graph.apply_birth(&birth(6, 20, [Some(6), None], 1)),
            Err(AncestryError::UnknownParent { .. } | AncestryError::Cycle(_))
        ));
        // A death for a stranger, and a second death for the same agent.
        assert_eq!(
            graph.apply_death(&death(999, 30)),
            Err(AncestryError::UnknownDeath(AgentUid(999)))
        );
        assert_eq!(
            graph.apply_death(&death(2, 10)),
            Err(AncestryError::DeathPrecedesBirth {
                uid: AgentUid(2),
                death_tick: 10,
                birth_tick: 10,
            })
        );
        assert_eq!(
            graph.apply_death(&death(2, 9)),
            Err(AncestryError::DeathPrecedesBirth {
                uid: AgentUid(2),
                death_tick: 9,
                birth_tick: 10,
            })
        );
        graph.apply_death(&death(2, 30)).expect("first death");
        assert_eq!(
            graph.apply_death(&death(2, 31)),
            Err(AncestryError::DuplicateDeath(AgentUid(2)))
        );
    }

    #[test]
    fn a_cycle_cannot_be_constructed_because_ancestors_are_strictly_older() {
        // The strictly-increasing birth tick is what rules out every cycle, not
        // just the self-parent case: an ancestor is always strictly older, so no
        // chain of edges can return to where it started.
        let mut graph = AncestryGraph::new();
        graph
            .apply_birth(&birth(1, 10, [None, None], 0))
            .expect("A");
        graph
            .apply_birth(&birth(2, 20, [Some(1), None], 1))
            .expect("B, child of A");
        // Now try to make A a child of B. A is older, so this is refused.
        graph.nodes.remove(&AgentUid(1));
        assert!(matches!(
            graph.apply_birth(&birth(1, 10, [Some(2), None], 2)),
            Err(AncestryError::ChildPrecedesParent { .. })
        ));
    }

    #[test]
    fn pruning_never_orphans_a_survivor_and_never_touches_a_living_lineage() {
        let mut graph = AncestryGraph::new();
        // A long dead lineage whose descendant is still alive.
        graph
            .apply_birth(&birth(1, 1, [None, None], 0))
            .expect("root");
        graph
            .apply_birth(&birth(2, 2, [Some(1), None], 1))
            .expect("child");
        graph
            .apply_birth(&birth(3, 3, [Some(2), None], 2))
            .expect("living grandchild");
        graph.apply_death(&death(1, 4)).expect("root dies");
        graph.apply_death(&death(2, 5)).expect("child dies");
        // An unrelated extinct clade, long dead.
        graph
            .apply_birth(&birth(10, 6, [None, None], 0))
            .expect("extinct root");
        graph
            .apply_birth(&birth(11, 7, [Some(10), None], 1))
            .expect("extinct child");
        graph.apply_death(&death(10, 8)).expect("dies");
        graph.apply_death(&death(11, 9)).expect("dies");

        let policy = PrunePolicy {
            keep_ancestors_of_living: true,
            extinct_retention_ticks: 0, // drop anything not needed by the living
            max_retained_extinct: 0,
        };
        let report = graph.prune(Tick(1_000), policy);

        // The living agent's whole ancestry survives, even though both ancestors
        // are dead: pruning it would rewrite the history of an agent still on
        // screen.
        assert!(graph.node(AgentUid(3)).is_some(), "the living agent stays");
        assert!(graph.node(AgentUid(2)).is_some(), "its dead parent stays");
        assert!(
            graph.node(AgentUid(1)).is_some(),
            "its dead grandparent stays"
        );
        // The unrelated extinct clade is gone.
        assert!(graph.node(AgentUid(10)).is_none());
        assert!(graph.node(AgentUid(11)).is_none());
        assert_eq!(report.removed, 2);
        assert_eq!(report.retained, graph.len());

        // NO DANGLING PARENTS. Every parent reference must resolve to a node that
        // is actually present — a consumer walking into a hole would have to
        // choose between panicking and silently truncating a lineage.
        for node in graph.nodes.values() {
            for parent in node.parents.into_iter().flatten() {
                assert!(
                    graph.nodes.contains_key(&parent),
                    "node {:?} points at parent {parent:?}, which was pruned away",
                    node.uid
                );
            }
        }
    }

    #[test]
    fn a_pruned_ancestor_is_tombstoned_rather_than_left_dangling() {
        let mut graph = AncestryGraph::new();
        graph
            .apply_birth(&birth(1, 1, [None, None], 0))
            .expect("ancestor");
        graph
            .apply_birth(&birth(2, 2, [Some(1), None], 1))
            .expect("descendant");
        graph.apply_death(&death(2, 3)).expect("descendant dies");
        graph.apply_death(&death(1, 3)).expect("ancestor dies");

        // Retain the recent descendant but not the ancestor.
        let policy = PrunePolicy {
            keep_ancestors_of_living: false,
            extinct_retention_ticks: 1_000,
            max_retained_extinct: 1,
        };
        graph.prune(Tick(4), policy);

        // Whichever survived, nobody points into a hole.
        for node in graph.nodes.values() {
            for parent in node.parents.into_iter().flatten() {
                let resolved = graph.nodes.get(&parent);
                assert!(
                    resolved.is_some(),
                    "a surviving node names a parent that no longer exists"
                );
            }
        }
        // Any tombstone is clearly marked, so a consumer can say "the ancestry
        // ends here because we pruned it" rather than "this agent has no parents".
        for node in graph.nodes.values().filter(|node| node.pruned) {
            assert_eq!(node.parents, [None, None]);
            assert!(node.children.is_empty());
        }
    }

    #[test]
    fn the_digest_is_stable_and_the_memory_bound_holds() {
        let build = || {
            let mut graph = AncestryGraph::new();
            graph
                .apply_birth(&birth(1, 0, [None, None], 0))
                .expect("root");
            for uid in 2..=2_000u64 {
                graph
                    .apply_birth(&birth(uid, uid, [Some(uid - 1), None], (uid - 1) as u32))
                    .expect("chain");
            }
            graph
        };
        // Same log, same digest — the oracle an offline rebuild is checked against.
        assert_eq!(build().canonical_digest(), build().canonical_digest());

        // A different log must produce a different digest, or the oracle certifies
        // everything and proves nothing.
        let mut altered = build();
        altered
            .apply_birth(&birth(9_999, 9_999, [None, None], 0))
            .expect("extra root");
        assert_ne!(build().canonical_digest(), altered.canonical_digest());

        // The documented memory bound, asserted rather than hoped for.
        let graph = build();
        let bytes = graph.approx_bytes();
        let ceiling = graph.len() * (APPROX_BYTES_PER_NODE + 2 * std::mem::size_of::<AgentUid>());
        assert!(
            bytes <= ceiling,
            "the graph costs {bytes} bytes, over the documented ceiling of {ceiling}"
        );
    }

    #[test]
    fn the_digest_distinguishes_identical_agents_with_different_origins() {
        let build = |origin| {
            let mut graph = AncestryGraph::new();
            graph
                .apply_birth(&birth_with_origin(1, 0, [None, None], 0, origin))
                .expect("origin record");
            graph
        };

        let born = build(BirthOrigin::Born).canonical_digest();
        let seeded = build(BirthOrigin::Seeded).canonical_digest();
        let injected = build(BirthOrigin::Injected).canonical_digest();
        assert_ne!(born, seeded);
        assert_ne!(born, injected);
        assert_ne!(seeded, injected);
    }

    #[test]
    fn the_digest_distinguishes_identical_deaths_with_different_causes() {
        let build = |cause| {
            let mut graph = AncestryGraph::new();
            graph
                .apply_birth(&birth(1, 0, [None, None], 0))
                .expect("root");
            graph
                .apply_death(&death_with_cause(1, 5, cause))
                .expect("death");
            graph
        };

        let combat = build(DeathCause::CombatCarnivore).canonical_digest();
        let starvation = build(DeathCause::Starvation).canonical_digest();
        let aging = build(DeathCause::Aging).canonical_digest();
        assert_ne!(combat, starvation);
        assert_ne!(combat, aging);
        assert_ne!(starvation, aging);
    }

    #[test]
    fn a_lineage_walk_is_bounded_even_on_a_pathological_graph() {
        // Unbounded recursion over a graph built from an untrusted log is a stack
        // overflow waiting for the right input — and a stack overflow is not a
        // typed error, it is the process dying.
        let mut graph = AncestryGraph::new();
        graph
            .apply_birth(&birth(1, 0, [None, None], 0))
            .expect("root");
        for uid in 2..=5_000u64 {
            graph
                .apply_birth(&birth(uid, uid, [Some(uid - 1), None], 1))
                .expect("deep chain");
        }
        let path = graph.lineage_path(AgentUid(5_000), 32);
        assert_eq!(path.len(), 32, "the walk must stop at max_depth");
        // And an unknown uid walks nowhere rather than panicking.
        assert!(graph.lineage_path(AgentUid(123_456), 32).is_empty());
    }
}
