//! Incremental phylogeny layout with indexed level-of-detail (LOD) queries.
//!
//! # Memory & Performance Guarantees
//!
//! Compact leaf ranks are intentionally not used: inserting an earlier leaf
//! would shift an arbitrary suffix of the tree. Each leaf instead owns an
//! immutable founder/agent anchor. Internal Y coordinates are derived from an
//! exact integer `(sum, count)` over descendant leaves, so an insert, detach, or
//! late-parent repair changes only the affected ancestry paths. No production
//! update contains a whole-tree relayout fallback.
//!
//! The retained-memory report counts vector capacities, child allocations,
//! pending-parent storage, both ordered indexes, and conservative `BTreeMap`
//! entry overhead. The default 10,000-node budget is 32 MiB. This is a derived
//! renderer-neutral view and therefore is deliberately not deserializable:
//! accepting serialized cache indexes would permit dangling indices and cycles.
//!
//! LOD is backed by a deterministic augmented treap over integer Y coordinates.
//! Subtree time/Y bounds prune the viewport and cached best ranks drive a
//! best-first exact top-k query without a per-frame full scan or sort.

use crate::ancestry::AncestryGraph;
use crate::species::{SpeciesId, SpeciesTable};
use crate::{AgentUid, BirthRecord, DeathCause, Tick};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::ops::Range;

const DEFAULT_LAYOUT_BYTES: usize = 32 << 20;
const CONSERVATIVE_NODE_GROWTH_BYTES: usize = 2_048;
const BTREE_ENTRY_OVERHEAD_BYTES: usize = 48;

/// Stable identifier for a phylogeny tree node (species clade, individual agent, or sentinel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PhyloKey {
    /// Sentinel node representing ancestors that have been pruned from memory.
    PrunedAncestor,
    /// Species clade identified by a numeric species ID.
    Species(u64),
    /// Individual agent identified by its stable UID.
    Agent(AgentUid),
}

/// Index into the layout's node storage vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LayoutIdx(u32);

impl LayoutIdx {
    fn from_usize(value: usize) -> Option<Self> {
        u32::try_from(value).ok().map(Self)
    }

    fn as_usize(self) -> usize {
        usize::try_from(self.0).expect("u32 layout index fits usize")
    }
}

/// 2D Bounding Box for layout elements.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rect {
    /// Lower horizontal bound in layout coordinates.
    pub x_min: f32,
    /// Lower vertical bound in layout coordinates.
    pub y_min: f32,
    /// Upper horizontal bound in layout coordinates.
    pub x_max: f32,
    /// Upper vertical bound in layout coordinates.
    pub y_max: f32,
}

impl Default for Rect {
    fn default() -> Self {
        Self {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.0,
            y_max: 0.0,
        }
    }
}

/// Simulation tick viewport range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TickRange {
    /// Inclusive first tick in the viewport.
    pub start: Tick,
    /// Inclusive last tick in the viewport.
    pub end: Tick,
}

/// Memory and rendering budgets for layout operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutBudget {
    /// Maximum number of nodes to retain or display.
    pub max_nodes: usize,
    /// Maximum tree depth for expansion.
    pub max_depth: usize,
    /// Conservative ceiling for all retained layout/index allocations.
    pub max_bytes: usize,
}

impl Default for LayoutBudget {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            max_depth: 100,
            max_bytes: DEFAULT_LAYOUT_BYTES,
        }
    }
}

/// Declared ancestry relation for one layout node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParentRef {
    /// A genuine root.
    Root,
    /// A parent that exists now or may arrive in a later delta.
    Known(PhyloKey),
    /// The authoritative ancestry store explicitly pruned the parent.
    Pruned,
}

/// Single node in the 2D phylogeny layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutNode {
    /// Stable graph identity represented by this node.
    pub key: PhyloKey,
    /// Resolved parent slot, absent for roots and unresolved parents.
    pub parent: Option<LayoutIdx>,
    /// Immutable declared ancestry relation.
    pub declared_parent: ParentRef,
    /// Stable founder used to order species leaves. Agent-scale nodes use
    /// their own [`AgentUid`] as the exact coordinate anchor.
    pub founder_uid: AgentUid,
    /// X coordinate (derived deterministically from tick).
    pub x: f32,
    /// Y coordinate (derived from deterministic tidy-tree leaf ordering).
    pub y: f32,
    /// Visual thickness scaling (e.g. proportional to population).
    pub thickness: f32,
    /// Current observed population.
    pub population: u32,
    /// Birth tick of the represented agent or species.
    pub first_tick: Tick,
    /// Observed death tick, absent while the node remains alive.
    pub last_tick: Option<Tick>,
    /// Whether the node's descendants are hidden by a collapse operation.
    pub collapsed: bool,
    /// Largest population observed for this node.
    pub peak_population: u32,
}

/// Report emitted after incremental layout extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutDelta {
    /// Slots allocated for newly accepted nodes.
    pub added: Vec<LayoutIdx>,
    /// Existing slots whose layout projection changed.
    pub moved: Vec<LayoutIdx>,
    /// Whether this update performed a whole-tree relayout.
    pub full_relayout: bool,
    /// Input rows examined, including idempotent duplicates.
    pub updates_seen: usize,
    /// Ancestor nodes whose exact aggregate was updated.
    pub ancestor_steps: usize,
    /// Child slots shifted by canonical ordered insertion/removal.
    pub child_slots_shifted: usize,
    /// Previously unresolved children attached when a late parent arrived.
    pub pending_children_resolved: usize,
    /// Augmented LOD-index nodes touched by removes/inserts.
    pub index_nodes_visited: usize,
    /// Retained allocation growth observed during this update.
    pub allocated_bytes: usize,
    /// Vector-capacity growths plus conservative ordered-index entry
    /// allocations observed during this update.
    pub allocation_growths: usize,
    /// Typed rows rejected without mutating their node.
    pub issues: Vec<LayoutIssue>,
}

/// A malformed or budget-refused layout update.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutIssue {
    /// Conflicting duplicate rows for one key appeared in the same batch.
    ConflictingDuplicate {
        /// Identity with conflicting rows.
        key: PhyloKey,
    },
    /// An existing node's immutable identity/topology fields changed.
    ImmutableConflict {
        /// Existing identity whose immutable fields differed.
        key: PhyloKey,
    },
    /// A node declared itself, or one of its descendants, as parent.
    Cycle {
        /// Node whose proposed parent would create a cycle.
        key: PhyloKey,
        /// Proposed parent closing the cycle.
        parent: PhyloKey,
    },
    /// A known parent was born after its child.
    ParentBornAfterChild {
        /// Child with the earlier birth tick.
        child: PhyloKey,
        /// Proposed parent with the later birth tick.
        parent: PhyloKey,
    },
    /// Death precedes birth.
    InvalidLifetime {
        /// Identity whose death tick precedes its birth.
        key: PhyloKey,
    },
    /// Retaining another node would cross the configured node/byte ceiling.
    BudgetExceeded {
        /// Identity that could not be retained within the budget.
        key: PhyloKey,
    },
}

/// Input update for a node in the phylogeny graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhyloNodeUpdate {
    /// Stable identity to insert or update.
    pub key: PhyloKey,
    /// Declared ancestry relation, including explicit pruning.
    pub parent: ParentRef,
    /// Immutable founder used to anchor species ordering.
    pub founder_uid: AgentUid,
    /// First tick of the represented lifetime.
    pub birth_tick: Tick,
    /// Final tick of the lifetime, absent while alive.
    pub death_tick: Option<Tick>,
    /// Current observed population.
    pub population: u32,
}

/// Batch of node updates to incorporate into the layout.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhyloDelta {
    /// Node observations to validate and incorporate.
    pub updates: Vec<PhyloNodeUpdate>,
}

/// Report returned when expanding a collapsed clade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpandReport {
    /// Number of nodes expanded by this operation.
    pub expanded_nodes: usize,
    /// Whether the expansion stopped at its node or depth budget.
    pub truncated: bool,
}

/// Allocation-aware retained-memory estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReport {
    /// Inline size of the layout owner.
    pub inline_bytes: usize,
    /// Bytes reserved by the layout-node vector.
    pub node_capacity_bytes: usize,
    /// Bytes reserved by the exact subtree-aggregate vector.
    pub aggregate_capacity_bytes: usize,
    /// Bytes reserved by topology metadata vectors.
    pub topology_capacity_bytes: usize,
    /// Bytes reserved by per-node child vectors.
    pub child_capacity_bytes: usize,
    /// Conservative retained bytes for the identity-to-slot index.
    pub key_index_bytes: usize,
    /// Conservative retained bytes for unresolved-parent entries and child lists.
    pub pending_index_bytes: usize,
    /// Conservative retained bytes for the augmented spatial index.
    pub lod_index_bytes: usize,
    /// Sum of the reported inline and retained-allocation estimates.
    pub total_retained_bytes: usize,
}

/// Indexed LOD result plus inspectable query work.
#[derive(Debug)]
pub struct LodReport<'a> {
    /// Selected nodes in deterministic level-of-detail rank order.
    pub nodes: Vec<&'a LayoutNode>,
    /// Spatial-index nodes examined by this query.
    pub index_nodes_visited: usize,
    /// Height of the retained spatial index.
    pub index_height: usize,
    /// More index candidates remained after the explicit output budget.
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubtreeAggregate {
    sum: u128,
    leaves: u64,
}

impl SubtreeAggregate {
    const fn leaf(anchor: u64) -> Self {
        Self {
            sum: anchor as u128,
            leaves: 1,
        }
    }

    fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            sum: self.sum.checked_add(other.sum)?,
            leaves: self.leaves.checked_add(other.leaves)?,
        })
    }

    fn checked_replace(self, old: Self, new: Self) -> Option<Self> {
        Some(Self {
            sum: self.sum.checked_sub(old.sum)?.checked_add(new.sum)?,
            leaves: self
                .leaves
                .checked_sub(old.leaves)?
                .checked_add(new.leaves)?,
        })
    }

    fn checked_sub(self, other: Self) -> Option<Self> {
        Some(Self {
            sum: self.sum.checked_sub(other.sum)?,
            leaves: self.leaves.checked_sub(other.leaves)?,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the exact integer leaf aggregate is projected to the layout's f32 coordinate; retain the existing f64 division before rounding"
    )]
    fn y(self) -> f32 {
        debug_assert!(self.leaves > 0);
        ((self.sum as f64) / (self.leaves as f64)) as f32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SpatialKey {
    y_bits: u32,
    key: PhyloKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LodRank {
    peak_population: u32,
    duration: u64,
    founder_uid: AgentUid,
    key: PhyloKey,
}

impl Ord for LodRank {
    fn cmp(&self, other: &Self) -> Ordering {
        self.peak_population
            .cmp(&other.peak_population)
            .then_with(|| self.duration.cmp(&other.duration))
            // Smaller founder UIDs and keys win exact rank ties.
            .then_with(|| other.founder_uid.cmp(&self.founder_uid))
            .then_with(|| other.key.cmp(&self.key))
    }
}

impl PartialOrd for LodRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
struct LodIndexNode {
    spatial: SpatialKey,
    layout_idx: LayoutIdx,
    priority: u64,
    rank: LodRank,
    first_tick: u64,
    last_tick: u64,
    left: Option<usize>,
    right: Option<usize>,
    best_rank: LodRank,
    min_y_bits: u32,
    max_y_bits: u32,
    min_first_tick: u64,
    max_last_tick: u64,
    height: usize,
}

impl LodIndexNode {
    fn new(spatial: SpatialKey, layout_idx: LayoutIdx, node: &LayoutNode) -> Self {
        let rank = lod_rank(node);
        Self {
            spatial,
            layout_idx,
            priority: deterministic_priority(node.key),
            rank,
            first_tick: node.first_tick.0,
            last_tick: node.last_tick.map_or(u64::MAX, |tick| tick.0),
            left: None,
            right: None,
            best_rank: rank,
            min_y_bits: spatial.y_bits,
            max_y_bits: spatial.y_bits,
            min_first_tick: node.first_tick.0,
            max_last_tick: node.last_tick.map_or(u64::MAX, |tick| tick.0),
            height: 1,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct LodIndex {
    root: Option<usize>,
    slots: Vec<Option<LodIndexNode>>,
    free: Vec<usize>,
}

impl LodIndex {
    fn insert(
        &mut self,
        spatial: SpatialKey,
        layout_idx: LayoutIdx,
        node: &LayoutNode,
        visited: &mut usize,
    ) {
        let slot = if let Some(slot) = self.free.pop() {
            self.slots[slot] = Some(LodIndexNode::new(spatial, layout_idx, node));
            slot
        } else {
            let slot = self.slots.len();
            self.slots
                .push(Some(LodIndexNode::new(spatial, layout_idx, node)));
            slot
        };
        self.root = Some(self.insert_at(self.root, slot, visited));
    }

    fn remove(&mut self, spatial: SpatialKey, visited: &mut usize) -> bool {
        let (root, removed) = self.remove_at(self.root, spatial, visited);
        self.root = root;
        if let Some(slot) = removed {
            self.slots[slot] = None;
            self.free.push(slot);
            true
        } else {
            false
        }
    }

    fn insert_at(&mut self, root: Option<usize>, inserted: usize, visited: &mut usize) -> usize {
        let Some(root) = root else {
            return inserted;
        };
        *visited = visited.saturating_add(1);
        let inserted_key = self.node(inserted).spatial;
        let root_key = self.node(root).spatial;
        if inserted_key < root_key {
            let left = self.node(root).left;
            let new_left = self.insert_at(left, inserted, visited);
            self.node_mut(root).left = Some(new_left);
            self.pull(root);
            if self.higher_priority(new_left, root) {
                self.rotate_right(root)
            } else {
                root
            }
        } else {
            debug_assert_ne!(inserted_key, root_key);
            let right = self.node(root).right;
            let new_right = self.insert_at(right, inserted, visited);
            self.node_mut(root).right = Some(new_right);
            self.pull(root);
            if self.higher_priority(new_right, root) {
                self.rotate_left(root)
            } else {
                root
            }
        }
    }

    fn remove_at(
        &mut self,
        root: Option<usize>,
        spatial: SpatialKey,
        visited: &mut usize,
    ) -> (Option<usize>, Option<usize>) {
        let Some(root) = root else {
            return (None, None);
        };
        *visited = visited.saturating_add(1);
        match spatial.cmp(&self.node(root).spatial) {
            Ordering::Less => {
                let (left, removed) = self.remove_at(self.node(root).left, spatial, visited);
                self.node_mut(root).left = left;
                self.pull(root);
                (Some(root), removed)
            }
            Ordering::Greater => {
                let (right, removed) = self.remove_at(self.node(root).right, spatial, visited);
                self.node_mut(root).right = right;
                self.pull(root);
                (Some(root), removed)
            }
            Ordering::Equal => {
                let merged = self.merge(self.node(root).left, self.node(root).right);
                (merged, Some(root))
            }
        }
    }

    fn merge(&mut self, left: Option<usize>, right: Option<usize>) -> Option<usize> {
        match (left, right) {
            (None, other) | (other, None) => other,
            (Some(left), Some(right)) => {
                if self.higher_priority(left, right) {
                    let merged = self.merge(self.node(left).right, Some(right));
                    self.node_mut(left).right = merged;
                    self.pull(left);
                    Some(left)
                } else {
                    let merged = self.merge(Some(left), self.node(right).left);
                    self.node_mut(right).left = merged;
                    self.pull(right);
                    Some(right)
                }
            }
        }
    }

    fn rotate_right(&mut self, root: usize) -> usize {
        let pivot = self
            .node(root)
            .left
            .expect("right rotation requires a left child");
        let transfer = self.node(pivot).right;
        self.node_mut(root).left = transfer;
        self.node_mut(pivot).right = Some(root);
        self.pull(root);
        self.pull(pivot);
        pivot
    }

    fn rotate_left(&mut self, root: usize) -> usize {
        let pivot = self
            .node(root)
            .right
            .expect("left rotation requires a right child");
        let transfer = self.node(pivot).left;
        self.node_mut(root).right = transfer;
        self.node_mut(pivot).left = Some(root);
        self.pull(root);
        self.pull(pivot);
        pivot
    }

    fn pull(&mut self, slot: usize) {
        let (left, right, rank, y_bits, first_tick, last_tick) = {
            let node = self.node(slot);
            (
                node.left,
                node.right,
                node.rank,
                node.spatial.y_bits,
                node.first_tick,
                node.last_tick,
            )
        };
        let mut best_rank = rank;
        let mut min_y_bits = y_bits;
        let mut max_y_bits = y_bits;
        let mut min_first_tick = first_tick;
        let mut max_last_tick = last_tick;
        let mut height = 1;
        for child in [left, right].into_iter().flatten() {
            let child = self.node(child);
            best_rank = best_rank.max(child.best_rank);
            min_y_bits = min_y_bits.min(child.min_y_bits);
            max_y_bits = max_y_bits.max(child.max_y_bits);
            min_first_tick = min_first_tick.min(child.min_first_tick);
            max_last_tick = max_last_tick.max(child.max_last_tick);
            height = height.max(child.height.saturating_add(1));
        }
        let node = self.node_mut(slot);
        node.best_rank = best_rank;
        node.min_y_bits = min_y_bits;
        node.max_y_bits = max_y_bits;
        node.min_first_tick = min_first_tick;
        node.max_last_tick = max_last_tick;
        node.height = height;
    }

    fn higher_priority(&self, left: usize, right: usize) -> bool {
        let left = self.node(left);
        let right = self.node(right);
        left.priority > right.priority
            || (left.priority == right.priority && left.spatial < right.spatial)
    }

    fn node(&self, slot: usize) -> &LodIndexNode {
        self.slots[slot].as_ref().expect("live LOD index slot")
    }

    fn node_mut(&mut self, slot: usize) -> &mut LodIndexNode {
        self.slots[slot].as_mut().expect("live LOD index slot")
    }

    fn height(&self) -> usize {
        self.root.map_or(0, |root| self.node(root).height)
    }

    fn y_bounds(&self) -> Option<(f32, f32)> {
        self.root.map(|root| {
            let root = self.node(root);
            (
                f32::from_bits(root.min_y_bits),
                f32::from_bits(root.max_y_bits),
            )
        })
    }

    const fn retained_bytes(&self) -> usize {
        self.slots
            .capacity()
            .saturating_mul(std::mem::size_of::<Option<LodIndexNode>>())
            .saturating_add(
                self.free
                    .capacity()
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateKind {
    Subtree(usize),
    Item(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LodCandidate {
    rank: LodRank,
    kind: CandidateKind,
}

impl Ord for LodCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank).then_with(|| {
            let self_key = match self.kind {
                CandidateKind::Subtree(slot) | CandidateKind::Item(slot) => slot,
            };
            let other_key = match other.kind {
                CandidateKind::Subtree(slot) | CandidateKind::Item(slot) => slot,
            };
            other_key.cmp(&self_key)
        })
    }
}

impl PartialOrd for LodCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn lod_rank(node: &LayoutNode) -> LodRank {
    LodRank {
        peak_population: node.peak_population,
        // Extant clades rank as ongoing. This is deliberately viewport
        // independent so the index never needs query-time reranking.
        duration: node
            .last_tick
            .map_or(u64::MAX, |end| end.0.saturating_sub(node.first_tick.0)),
        founder_uid: node.founder_uid,
        key: node.key,
    }
}

const fn deterministic_priority(key: PhyloKey) -> u64 {
    let raw = match key {
        PhyloKey::PrunedAncestor => 0xA076_1D64_78BD_642F,
        PhyloKey::Species(id) => id ^ 0xE703_7ED1_A0B4_28DB,
        PhyloKey::Agent(uid) => uid.get() ^ 0x8EBC_6AF0_9C88_C6E3,
    };
    let mut value = raw.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Deterministic 2D layout engine for phylogeny graphs.
#[derive(Debug, Clone)]
pub struct TreeLayout {
    nodes: Vec<LayoutNode>,
    index: BTreeMap<PhyloKey, LayoutIdx>,
    children: Vec<Vec<LayoutIdx>>,
    aggregates: Vec<SubtreeAggregate>,
    spatial_keys: Vec<SpatialKey>,
    pending_by_parent: BTreeMap<PhyloKey, Vec<LayoutIdx>>,
    lod_index: LodIndex,
    child_capacity_bytes: usize,
    pending_capacity_bytes: usize,
    bounds: Rect,
    budget: LayoutBudget,
}

impl TreeLayout {
    /// Creates a new layout engine with the specified budget constraints.
    #[must_use]
    pub fn new(budget: LayoutBudget) -> Self {
        Self {
            nodes: Vec::new(),
            index: BTreeMap::new(),
            children: Vec::new(),
            aggregates: Vec::new(),
            spatial_keys: Vec::new(),
            pending_by_parent: BTreeMap::new(),
            lod_index: LodIndex::default(),
            child_capacity_bytes: 0,
            pending_capacity_bytes: 0,
            bounds: Rect::default(),
            budget,
        }
    }

    /// Returns a slice of all nodes in the layout.
    #[must_use]
    pub fn nodes(&self) -> &[LayoutNode] {
        &self.nodes
    }

    /// Returns the total number of nodes stored in the layout.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the layout contains no nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the bounding box of the entire layout.
    #[must_use]
    pub const fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Looks up a node index by its [`PhyloKey`].
    #[must_use]
    pub fn get_idx(&self, key: &PhyloKey) -> Option<LayoutIdx> {
        self.index.get(key).copied()
    }

    /// Looks up a layout node by its [`PhyloKey`].
    #[must_use]
    pub fn get_node(&self, key: &PhyloKey) -> Option<&LayoutNode> {
        self.index.get(key).map(|idx| &self.nodes[idx.as_usize()])
    }

    /// Conservative allocation-aware memory retained by this layout.
    #[must_use]
    pub fn memory_report(&self) -> MemoryReport {
        let inline_bytes = std::mem::size_of::<Self>();
        let node_capacity_bytes = self
            .nodes
            .capacity()
            .saturating_mul(std::mem::size_of::<LayoutNode>());
        let aggregate_capacity_bytes = self
            .aggregates
            .capacity()
            .saturating_mul(std::mem::size_of::<SubtreeAggregate>());
        let topology_capacity_bytes = self
            .children
            .capacity()
            .saturating_mul(std::mem::size_of::<Vec<LayoutIdx>>())
            .saturating_add(
                self.spatial_keys
                    .capacity()
                    .saturating_mul(std::mem::size_of::<SpatialKey>()),
            );
        let child_capacity_bytes = self.child_capacity_bytes;
        let key_index_bytes = self.index.len().saturating_mul(
            std::mem::size_of::<(PhyloKey, LayoutIdx)>().saturating_add(BTREE_ENTRY_OVERHEAD_BYTES),
        );
        let pending_index_bytes = self
            .pending_by_parent
            .len()
            .saturating_mul(
                std::mem::size_of::<(PhyloKey, Vec<LayoutIdx>)>()
                    .saturating_add(BTREE_ENTRY_OVERHEAD_BYTES),
            )
            .saturating_add(self.pending_capacity_bytes);
        let lod_index_bytes = self.lod_index.retained_bytes();
        let total_retained_bytes = [
            inline_bytes,
            node_capacity_bytes,
            aggregate_capacity_bytes,
            topology_capacity_bytes,
            child_capacity_bytes,
            key_index_bytes,
            pending_index_bytes,
            lod_index_bytes,
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add);
        MemoryReport {
            inline_bytes,
            node_capacity_bytes,
            aggregate_capacity_bytes,
            topology_capacity_bytes,
            child_capacity_bytes,
            key_index_bytes,
            pending_index_bytes,
            lod_index_bytes,
            total_retained_bytes,
        }
    }

    /// Extends the layout without any whole-tree coordinate pass.
    pub fn extend(&mut self, delta: &PhyloDelta) -> LayoutDelta {
        let memory_before = self.memory_report().total_retained_bytes;
        let mut report = LayoutDelta {
            added: Vec::new(),
            moved: Vec::new(),
            full_relayout: false,
            updates_seen: delta.updates.len(),
            ancestor_steps: 0,
            child_slots_shifted: 0,
            pending_children_resolved: 0,
            index_nodes_visited: 0,
            allocated_bytes: 0,
            allocation_growths: 0,
            issues: Vec::new(),
        };
        if delta.updates.is_empty() {
            return report;
        }

        let mut canonical = BTreeMap::<PhyloKey, PhyloNodeUpdate>::new();
        let mut conflicts = BTreeSet::new();
        for update in &delta.updates {
            match canonical.get(&update.key) {
                Some(previous) if previous != update => {
                    conflicts.insert(update.key);
                }
                Some(_) => {}
                None => {
                    canonical.insert(update.key, update.clone());
                }
            }
        }
        for key in conflicts {
            canonical.remove(&key);
            report
                .issues
                .push(LayoutIssue::ConflictingDuplicate { key });
        }

        let invalid = canonical
            .iter()
            .filter_map(|(&key, update)| {
                self.validate_update(update, &canonical)
                    .err()
                    .map(|issue| (key, issue))
            })
            .collect::<BTreeMap<_, _>>();

        let mut new_nodes = Vec::new();
        for (key, update) in &canonical {
            if let Some(issue) = invalid.get(key) {
                report.issues.push(issue.clone());
                continue;
            }
            if let Some(&idx) = self.index.get(key) {
                self.apply_mutable_update(idx, update, &mut report);
                continue;
            }
            let sentinel_needed = matches!(update.parent, ParentRef::Pruned)
                && !self.index.contains_key(&PhyloKey::PrunedAncestor);
            let required = 1 + usize::from(sentinel_needed);
            if !self.can_retain(required) {
                report
                    .issues
                    .push(LayoutIssue::BudgetExceeded { key: *key });
                continue;
            }
            let idx = self.push_node(update, &mut report);
            new_nodes.push(idx);
        }

        // Every batch node is indexed before topology resolution, so input order
        // cannot turn a same-batch parent into a fake pruned ancestor.
        for &idx in &new_nodes {
            self.resolve_declared_parent(idx, &mut report);
        }
        for &idx in &new_nodes {
            let key = self.nodes[idx.as_usize()].key;
            self.resolve_pending_children(key, idx, &mut report);
        }

        self.refresh_bounds();
        report.moved.sort_unstable();
        report.moved.dedup();
        report.allocated_bytes = self
            .memory_report()
            .total_retained_bytes
            .saturating_sub(memory_before);
        report
    }

    fn validate_update(
        &self,
        update: &PhyloNodeUpdate,
        batch: &BTreeMap<PhyloKey, PhyloNodeUpdate>,
    ) -> Result<(), LayoutIssue> {
        if update.key == PhyloKey::PrunedAncestor {
            return Err(LayoutIssue::ImmutableConflict { key: update.key });
        }
        if update
            .death_tick
            .is_some_and(|death| death < update.birth_tick)
        {
            return Err(LayoutIssue::InvalidLifetime { key: update.key });
        }
        if let Some(&idx) = self.index.get(&update.key) {
            let node = &self.nodes[idx.as_usize()];
            if node.first_tick != update.birth_tick
                || node.declared_parent != update.parent
                || node.founder_uid != update.founder_uid
            {
                return Err(LayoutIssue::ImmutableConflict { key: update.key });
            }
        }

        let ParentRef::Known(direct_parent) = update.parent else {
            return Ok(());
        };
        if direct_parent == update.key {
            return Err(LayoutIssue::Cycle {
                key: update.key,
                parent: direct_parent,
            });
        }
        if let Some(parent_birth) = self.declared_birth(direct_parent, batch)
            && parent_birth > update.birth_tick
        {
            return Err(LayoutIssue::ParentBornAfterChild {
                child: update.key,
                parent: direct_parent,
            });
        }

        let mut seen = BTreeSet::from([update.key]);
        let mut cursor = direct_parent;
        loop {
            if !seen.insert(cursor) {
                return Err(LayoutIssue::Cycle {
                    key: update.key,
                    parent: direct_parent,
                });
            }
            let Some(parent) = self.declared_parent(cursor, batch) else {
                break;
            };
            match parent {
                ParentRef::Known(next) => cursor = next,
                ParentRef::Root | ParentRef::Pruned => break,
            }
        }
        Ok(())
    }

    fn declared_birth(
        &self,
        key: PhyloKey,
        batch: &BTreeMap<PhyloKey, PhyloNodeUpdate>,
    ) -> Option<Tick> {
        batch.get(&key).map_or_else(
            || self.get_node(&key).map(|node| node.first_tick),
            |update| Some(update.birth_tick),
        )
    }

    fn declared_parent(
        &self,
        key: PhyloKey,
        batch: &BTreeMap<PhyloKey, PhyloNodeUpdate>,
    ) -> Option<ParentRef> {
        batch.get(&key).map_or_else(
            || self.get_node(&key).map(|node| node.declared_parent),
            |update| Some(update.parent),
        )
    }

    fn can_retain(&self, additional_nodes: usize) -> bool {
        let count_fits = self.nodes.len().saturating_add(additional_nodes)
            <= self
                .budget
                .max_nodes
                .min(usize::try_from(u32::MAX).unwrap_or(usize::MAX));
        let bytes_fit = self
            .memory_report()
            .total_retained_bytes
            .saturating_add(additional_nodes.saturating_mul(CONSERVATIVE_NODE_GROWTH_BYTES))
            <= self.budget.max_bytes;
        count_fits && bytes_fit
    }

    fn push_node(&mut self, update: &PhyloNodeUpdate, report: &mut LayoutDelta) -> LayoutIdx {
        let idx =
            LayoutIdx::from_usize(self.nodes.len()).expect("retention preflight enforces u32");
        let aggregate = SubtreeAggregate::leaf(update_anchor(update.key, update.founder_uid));
        let y = aggregate.y();
        let node = LayoutNode {
            key: update.key,
            parent: None,
            declared_parent: update.parent,
            founder_uid: update.founder_uid,
            x: tick_x(update.birth_tick),
            y,
            thickness: population_thickness(update.population),
            population: update.population,
            first_tick: update.birth_tick,
            last_tick: update.death_tick,
            collapsed: false,
            peak_population: update.population,
        };
        let node_capacity = self.nodes.capacity();
        let topology_capacity = self.children.capacity();
        let aggregate_capacity = self.aggregates.capacity();
        let spatial_capacity = self.spatial_keys.capacity();
        self.nodes.push(node);
        self.children.push(Vec::new());
        self.aggregates.push(aggregate);
        let spatial = SpatialKey {
            y_bits: y.to_bits(),
            key: update.key,
        };
        self.spatial_keys.push(spatial);
        self.index.insert(update.key, idx);
        report.allocation_growths = report
            .allocation_growths
            .saturating_add(usize::from(self.nodes.capacity() > node_capacity))
            .saturating_add(usize::from(self.children.capacity() > topology_capacity))
            .saturating_add(usize::from(self.aggregates.capacity() > aggregate_capacity))
            .saturating_add(usize::from(self.spatial_keys.capacity() > spatial_capacity))
            // Ordered-map allocation is deliberately counted
            // conservatively per retained entry.
            .saturating_add(1);
        let lod_slots_capacity = self.lod_index.slots.capacity();
        let lod_free_capacity = self.lod_index.free.capacity();
        self.lod_index.insert(
            spatial,
            idx,
            &self.nodes[idx.as_usize()],
            &mut report.index_nodes_visited,
        );
        report.allocation_growths = report
            .allocation_growths
            .saturating_add(usize::from(
                self.lod_index.slots.capacity() > lod_slots_capacity,
            ))
            .saturating_add(usize::from(
                self.lod_index.free.capacity() > lod_free_capacity,
            ));
        if self.nodes.len() == 1 {
            self.bounds.x_min = tick_x(update.birth_tick);
            self.bounds.x_max = tick_x(update.birth_tick);
        } else {
            self.bounds.x_min = self.bounds.x_min.min(tick_x(update.birth_tick));
            self.bounds.x_max = self.bounds.x_max.max(tick_x(update.birth_tick));
        }
        report.added.push(idx);
        idx
    }

    fn ensure_pruned_sentinel(&mut self, report: &mut LayoutDelta) -> Option<LayoutIdx> {
        if let Some(&idx) = self.index.get(&PhyloKey::PrunedAncestor) {
            return Some(idx);
        }
        if !self.can_retain(1) {
            return None;
        }
        let update = PhyloNodeUpdate {
            key: PhyloKey::PrunedAncestor,
            parent: ParentRef::Root,
            founder_uid: AgentUid(0),
            birth_tick: Tick(0),
            death_tick: None,
            population: 0,
        };
        Some(self.push_node(&update, report))
    }

    fn apply_mutable_update(
        &mut self,
        idx: LayoutIdx,
        update: &PhyloNodeUpdate,
        report: &mut LayoutDelta,
    ) {
        let node_index = idx.as_usize();
        let unchanged = self.nodes[node_index].population == update.population
            && self.nodes[node_index].last_tick == update.death_tick;
        if unchanged {
            return;
        }
        let rank_changed = self.nodes[node_index].last_tick != update.death_tick
            || update.population > self.nodes[node_index].peak_population;
        if rank_changed {
            self.remove_lod_node(idx, report);
        }
        let node = &mut self.nodes[node_index];
        node.population = update.population;
        node.last_tick = update.death_tick;
        node.peak_population = node.peak_population.max(update.population);
        node.thickness = population_thickness(update.population);
        if rank_changed {
            self.insert_lod_node(idx, report);
        }
    }

    fn resolve_declared_parent(&mut self, child: LayoutIdx, report: &mut LayoutDelta) {
        let declared = self.nodes[child.as_usize()].declared_parent;
        match declared {
            ParentRef::Root => {}
            ParentRef::Pruned => {
                if let Some(parent) = self.ensure_pruned_sentinel(report) {
                    self.attach_child(parent, child, report);
                } else {
                    report.issues.push(LayoutIssue::BudgetExceeded {
                        key: self.nodes[child.as_usize()].key,
                    });
                }
            }
            ParentRef::Known(parent_key) => {
                if let Some(&parent) = self.index.get(&parent_key) {
                    self.attach_child(parent, child, report);
                } else {
                    self.insert_pending(parent_key, child, report);
                }
            }
        }
    }

    fn resolve_pending_children(
        &mut self,
        parent_key: PhyloKey,
        parent: LayoutIdx,
        report: &mut LayoutDelta,
    ) {
        let Some(children) = self.pending_by_parent.remove(&parent_key) else {
            return;
        };
        self.pending_capacity_bytes = self.pending_capacity_bytes.saturating_sub(
            children
                .capacity()
                .saturating_mul(std::mem::size_of::<LayoutIdx>()),
        );
        for child in children {
            let child_key = self.nodes[child.as_usize()].key;
            if self.nodes[parent.as_usize()].first_tick > self.nodes[child.as_usize()].first_tick {
                report.issues.push(LayoutIssue::ParentBornAfterChild {
                    child: child_key,
                    parent: parent_key,
                });
                self.insert_pending(parent_key, child, report);
                continue;
            }
            if self.would_cycle(child, parent) {
                report.issues.push(LayoutIssue::Cycle {
                    key: child_key,
                    parent: parent_key,
                });
                self.insert_pending(parent_key, child, report);
                continue;
            }
            self.attach_child(parent, child, report);
            report.pending_children_resolved = report.pending_children_resolved.saturating_add(1);
        }
    }

    fn insert_pending(&mut self, parent_key: PhyloKey, child: LayoutIdx, report: &mut LayoutDelta) {
        let child_order = self.node_order(child);
        let new_parent_entry = !self.pending_by_parent.contains_key(&parent_key);
        let position = self
            .pending_by_parent
            .get(&parent_key)
            .map_or(Ok(0), |pending| {
                pending.binary_search_by(|candidate| {
                    let node = &self.nodes[candidate.as_usize()];
                    (node.founder_uid, node.key).cmp(&child_order)
                })
            })
            .unwrap_or_else(|position| position);
        let pending = self.pending_by_parent.entry(parent_key).or_default();
        if pending.get(position) == Some(&child) {
            return;
        }
        report.child_slots_shifted = report
            .child_slots_shifted
            .saturating_add(pending.len().saturating_sub(position));
        let old_capacity = pending.capacity();
        pending.insert(position, child);
        let capacity_growth = pending.capacity().saturating_sub(old_capacity);
        self.pending_capacity_bytes = self
            .pending_capacity_bytes
            .saturating_add(capacity_growth.saturating_mul(std::mem::size_of::<LayoutIdx>()));
        report.allocation_growths = report
            .allocation_growths
            .saturating_add(usize::from(new_parent_entry))
            .saturating_add(usize::from(capacity_growth > 0));
    }

    fn node_order(&self, idx: LayoutIdx) -> (AgentUid, PhyloKey) {
        let node = &self.nodes[idx.as_usize()];
        (node.founder_uid, node.key)
    }

    fn would_cycle(&self, child: LayoutIdx, mut parent: LayoutIdx) -> bool {
        for _ in 0..=self.nodes.len() {
            if parent == child {
                return true;
            }
            let Some(next) = self.nodes[parent.as_usize()].parent else {
                return false;
            };
            parent = next;
        }
        true
    }

    fn attach_child(&mut self, parent: LayoutIdx, child: LayoutIdx, report: &mut LayoutDelta) {
        if self.nodes[child.as_usize()].parent == Some(parent) {
            return;
        }
        if let Some(old_parent) = self.nodes[child.as_usize()].parent {
            self.detach_child(old_parent, child, report);
        }
        if self.would_cycle(child, parent) {
            report.issues.push(LayoutIssue::Cycle {
                key: self.nodes[child.as_usize()].key,
                parent: self.nodes[parent.as_usize()].key,
            });
            return;
        }

        let parent_index = parent.as_usize();
        let order = self.node_order(child);
        let position = self.children[parent_index]
            .binary_search_by(|candidate| self.node_order(*candidate).cmp(&order))
            .unwrap_or_else(|position| position);
        report.child_slots_shifted = report
            .child_slots_shifted
            .saturating_add(self.children[parent_index].len().saturating_sub(position));
        let was_leaf = self.children[parent_index].is_empty();
        let old_parent = self.aggregates[parent_index];
        let old_capacity = self.children[parent_index].capacity();
        self.children[parent_index].insert(position, child);
        let capacity_growth = self.children[parent_index]
            .capacity()
            .saturating_sub(old_capacity);
        self.child_capacity_bytes = self
            .child_capacity_bytes
            .saturating_add(capacity_growth.saturating_mul(std::mem::size_of::<LayoutIdx>()));
        report.allocation_growths = report
            .allocation_growths
            .saturating_add(usize::from(capacity_growth > 0));
        self.nodes[child.as_usize()].parent = Some(parent);
        let child_aggregate = self.aggregates[child.as_usize()];
        let new_parent = if was_leaf {
            child_aggregate
        } else {
            old_parent
                .checked_add(child_aggregate)
                .expect("aggregate bounded by max_nodes")
        };
        self.propagate_aggregate_change(parent, old_parent, new_parent, report);
    }

    fn detach_child(&mut self, parent: LayoutIdx, child: LayoutIdx, report: &mut LayoutDelta) {
        let parent_index = parent.as_usize();
        let order = self.node_order(child);
        let Ok(position) = self.children[parent_index]
            .binary_search_by(|candidate| self.node_order(*candidate).cmp(&order))
        else {
            return;
        };
        let old_parent = self.aggregates[parent_index];
        let child_aggregate = self.aggregates[child.as_usize()];
        report.child_slots_shifted = report.child_slots_shifted.saturating_add(
            self.children[parent_index]
                .len()
                .saturating_sub(position + 1),
        );
        self.children[parent_index].remove(position);
        self.nodes[child.as_usize()].parent = None;
        let new_parent = if self.children[parent_index].is_empty() {
            SubtreeAggregate::leaf(node_anchor(&self.nodes[parent_index]))
        } else {
            old_parent
                .checked_sub(child_aggregate)
                .expect("attached child aggregate is contained by parent")
        };
        self.propagate_aggregate_change(parent, old_parent, new_parent, report);
    }

    fn propagate_aggregate_change(
        &mut self,
        mut current: LayoutIdx,
        mut old: SubtreeAggregate,
        mut new: SubtreeAggregate,
        report: &mut LayoutDelta,
    ) {
        while old != new {
            let current_index = current.as_usize();
            self.aggregates[current_index] = new;
            report.ancestor_steps = report.ancestor_steps.saturating_add(1);
            let y = new.y();
            if self.nodes[current_index].y.to_bits() != y.to_bits() {
                self.remove_lod_node(current, report);
                self.nodes[current_index].y = y;
                self.spatial_keys[current_index] = SpatialKey {
                    y_bits: y.to_bits(),
                    key: self.nodes[current_index].key,
                };
                self.insert_lod_node(current, report);
                report.moved.push(current);
            }
            let Some(parent) = self.nodes[current_index].parent else {
                break;
            };
            let parent_old = self.aggregates[parent.as_usize()];
            let parent_new = parent_old
                .checked_replace(old, new)
                .expect("ancestor aggregate contains its child aggregate");
            current = parent;
            old = parent_old;
            new = parent_new;
        }
    }

    fn remove_lod_node(&mut self, idx: LayoutIdx, report: &mut LayoutDelta) {
        let spatial = self.spatial_keys[idx.as_usize()];
        let free_capacity = self.lod_index.free.capacity();
        let removed = self
            .lod_index
            .remove(spatial, &mut report.index_nodes_visited);
        debug_assert!(removed);
        report.allocation_growths = report
            .allocation_growths
            .saturating_add(usize::from(self.lod_index.free.capacity() > free_capacity));
    }

    fn insert_lod_node(&mut self, idx: LayoutIdx, report: &mut LayoutDelta) {
        let spatial = self.spatial_keys[idx.as_usize()];
        let slots_capacity = self.lod_index.slots.capacity();
        let free_capacity = self.lod_index.free.capacity();
        self.lod_index.insert(
            spatial,
            idx,
            &self.nodes[idx.as_usize()],
            &mut report.index_nodes_visited,
        );
        report.allocation_growths = report
            .allocation_growths
            .saturating_add(usize::from(
                self.lod_index.slots.capacity() > slots_capacity,
            ))
            .saturating_add(usize::from(self.lod_index.free.capacity() > free_capacity));
    }

    fn refresh_bounds(&mut self) {
        if self.nodes.is_empty() {
            self.bounds = Rect::default();
            return;
        }
        if let Some((y_min, y_max)) = self.lod_index.y_bounds() {
            self.bounds.y_min = y_min;
            self.bounds.y_max = y_max;
        }
    }

    /// Queries visible nodes using the augmented index.
    ///
    /// The result is in deterministic descending `(peak population,
    /// lifetime)` order with a stable key tie-break. `index_nodes_visited`
    /// exposes the actual amount of index work instead of hiding a fallback
    /// scan behind the output budget.
    #[must_use]
    pub fn lod_report(
        &self,
        viewport: TickRange,
        y_range: Range<f32>,
        budget: usize,
    ) -> LodReport<'_> {
        let mut report = LodReport {
            nodes: Vec::new(),
            index_nodes_visited: 0,
            index_height: self.lod_index.height(),
            truncated: false,
        };
        if budget == 0
            || viewport.start > viewport.end
            || !y_range.start.is_finite()
            || !y_range.end.is_finite()
            || y_range.start > y_range.end
            || y_range.end < 0.0
        {
            return report;
        }
        let y_start = y_range.start.max(0.0).to_bits();
        let y_end = y_range.end.to_bits();
        let Some(root) = self.lod_index.root else {
            return report;
        };
        let mut candidates = BinaryHeap::new();
        if self.lod_subtree_intersects(root, viewport, y_start, y_end) {
            candidates.push(LodCandidate {
                rank: self.lod_index.node(root).best_rank,
                kind: CandidateKind::Subtree(root),
            });
        }

        while report.nodes.len() < budget {
            let Some(candidate) = candidates.pop() else {
                break;
            };
            match candidate.kind {
                CandidateKind::Item(slot) => {
                    let idx = self.lod_index.node(slot).layout_idx;
                    report.nodes.push(&self.nodes[idx.as_usize()]);
                }
                CandidateKind::Subtree(slot) => {
                    report.index_nodes_visited = report.index_nodes_visited.saturating_add(1);
                    let indexed = self.lod_index.node(slot);
                    if Self::lod_item_matches(indexed, viewport, y_start, y_end) {
                        candidates.push(LodCandidate {
                            rank: indexed.rank,
                            kind: CandidateKind::Item(slot),
                        });
                    }
                    for child in [indexed.left, indexed.right].into_iter().flatten() {
                        if self.lod_subtree_intersects(child, viewport, y_start, y_end) {
                            candidates.push(LodCandidate {
                                rank: self.lod_index.node(child).best_rank,
                                kind: CandidateKind::Subtree(child),
                            });
                        }
                    }
                }
            }
        }
        report.truncated = !candidates.is_empty();
        report
    }

    fn lod_subtree_intersects(
        &self,
        slot: usize,
        viewport: TickRange,
        y_start: u32,
        y_end: u32,
    ) -> bool {
        let node = self.lod_index.node(slot);
        node.max_y_bits >= y_start
            && node.min_y_bits <= y_end
            && node.min_first_tick <= viewport.end.0
            && node.max_last_tick >= viewport.start.0
    }

    const fn lod_item_matches(
        node: &LodIndexNode,
        viewport: TickRange,
        y_start: u32,
        y_end: u32,
    ) -> bool {
        node.spatial.y_bits >= y_start
            && node.spatial.y_bits <= y_end
            && node.first_tick <= viewport.end.0
            && node.last_tick >= viewport.start.0
    }

    /// Convenience iterator over [`Self::lod_report`].
    pub fn lod(
        &self,
        viewport: TickRange,
        y_range: Range<f32>,
        budget: usize,
    ) -> impl Iterator<Item = &LayoutNode> {
        self.lod_report(viewport, y_range, budget).nodes.into_iter()
    }

    /// Performs a 2D hit test finding the closest node key within `tol` Euclidean distance.
    #[must_use]
    pub fn hit_test(&self, x: f32, y: f32, tol: f32) -> Option<PhyloKey> {
        if !x.is_finite() || !y.is_finite() || !tol.is_finite() || tol < 0.0 {
            return None;
        }
        let tol_sq = tol * tol;
        let mut best: Option<(f32, PhyloKey)> = None;

        for node in &self.nodes {
            let dx = node.x - x;
            let dy = node.y - y;
            #[expect(
                clippy::suboptimal_flops,
                reason = "hit-test tolerance and exact tie decisions use the existing separately rounded squared-distance sum"
            )]
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= tol_sq {
                match best {
                    Some((best_dist, best_key)) => {
                        if dist_sq < best_dist
                            || (dist_sq.partial_cmp(&best_dist) == Some(Ordering::Equal)
                                && node.key < best_key)
                        {
                            best = Some((dist_sq, node.key));
                        }
                    }
                    None => {
                        best = Some((dist_sq, node.key));
                    }
                }
            }
        }

        best.map(|(_, key)| key)
    }

    /// Expands a collapsed clade subtree up to the depth and node budget limits.
    pub fn expand(
        &mut self,
        root: PhyloKey,
        depth_budget: usize,
        node_budget: usize,
    ) -> ExpandReport {
        let Some(&root_idx) = self.index.get(&root) else {
            return ExpandReport {
                expanded_nodes: 0,
                truncated: false,
            };
        };

        let effective_depth_budget = depth_budget.min(self.budget.max_depth);
        let effective_node_budget = node_budget.min(self.budget.max_nodes);
        let mut count = 0;
        let mut truncated = false;
        let mut stack = vec![(root_idx, 0)];

        while let Some((idx, depth)) = stack.pop() {
            if count >= effective_node_budget {
                truncated = true;
                break;
            }
            self.nodes[idx.as_usize()].collapsed = false;
            count += 1;

            let children = &self.children[idx.as_usize()];
            if depth < effective_depth_budget {
                for &child_idx in children.iter().rev() {
                    stack.push((child_idx, depth + 1));
                }
            } else if !children.is_empty() {
                truncated = true;
            }
        }

        ExpandReport {
            expanded_nodes: count,
            truncated,
        }
    }
}

const fn update_anchor(key: PhyloKey, founder_uid: AgentUid) -> u64 {
    match key {
        PhyloKey::PrunedAncestor => 0,
        PhyloKey::Species(_) => founder_uid.get().saturating_add(1),
        PhyloKey::Agent(uid) => uid.get().saturating_add(1),
    }
}

const fn node_anchor(node: &LayoutNode) -> u64 {
    update_anchor(node.key, node.founder_uid)
}

#[allow(clippy::cast_precision_loss)]
const fn tick_x(tick: Tick) -> f32 {
    tick.0 as f32
}

#[allow(clippy::cast_precision_loss)]
fn population_thickness(population: u32) -> f32 {
    (population.max(1) as f32).sqrt()
}

// =========================================================================
// Phylogeny Event Stream & Hint Cross-Validation Engine (bd-16g.3.3)
// =========================================================================

/// Monotonically increasing identifier for an emitted phylogeny event (bd-16g.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EventId(pub u64);

/// Monotonically increasing or tagged identifier for a detector kernel hint (bd-16g.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HintId(pub u64);

/// Mechanism explaining reproductive isolation between clades (bd-16g.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeparationKind {
    /// True evolutionary divergence in physical and behavioral phenotype.
    Phenotypic,
    /// Separation mechanically forced by the simulation's brain-family mating gate.
    ///
    /// The simulation gates sexual crossover on `parent_kind == partner_kind`.
    /// Two clusters carrying different brain kinds are a mechanical consequence
    /// of the mating gate, not evolutionary speciation.
    BrainKindGated,
    /// Geographic separation across isolated islands (bd-16g.5).
    Allopatric,
    /// Combination of mechanisms.
    Mixed,
}

/// Reason why a candidate speciation split or detector hint was rejected (bd-16g.3.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    /// Candidate split reverted or vanished before persisting for K consecutive samples.
    Transient,
    /// Realized cross-cluster mating rate exceeded the reproductive isolation threshold.
    Interbreeding,
    /// One or both sub-clusters had fewer than `min_species_size` members.
    BelowMinSize,
    /// Insufficient two-parent births in the window to establish a statistically valid rate.
    ///
    /// Protects against declaring reproductive isolation from an empty or near-zero denominator.
    NoAncestralSupport,
    /// Apparent isolation was entirely an artifact of the brain-family mating gate.
    BrainKindArtifact,
}

/// Self-describing evidence attached to a detector hint verdict (bd-16g.3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HintEvidence {
    /// Identity of the hint being resolved.
    pub hint_id: HintId,
    /// Tick at which the hint was evaluated.
    pub tick: Tick,
    /// Observed score or magnitude from the detector.
    pub score: f32,
    /// Human-readable explanation of why the hint was confirmed or rejected.
    pub detail: String,
    /// Quantitative metrics governing the decision.
    pub metrics: BTreeMap<String, f32>,
}

/// Final verdict on a detector hint (bd-16g.3.3).
///
/// Every hint from the detector kernel must terminate in exactly one verdict: Confirmed or Rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HintVerdict {
    /// Confirmed by an emitted phylogeny event with the given [`EventId`].
    Confirmed(EventId),
    /// Rejected with specific failure reason and quantitative evidence.
    Rejected {
        /// Reason for rejection.
        reason: RejectReason,
        /// Attached evidence.
        evidence: HintEvidence,
    },
}

/// Typed phylogeny event emitted into the append-only stream (bd-16g.3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhyloEvent {
    /// Speciation: an ancestral species splits into distinct clades.
    Speciation {
        /// Ancestral species ID.
        parent: SpeciesId,
        /// The two diverging child species IDs.
        children: [SpeciesId; 2],
        /// Founding agents of the newly emerged lineage.
        founders: Vec<AgentUid>,
        /// Categorized mechanism of reproductive isolation.
        separation: SeparationKind,
        /// Realized cross-cluster mating rate over the observation window.
        cross_mating_rate: f32,
        /// Number of consecutive segmentation samples the split persisted.
        /// Uses the same count representation as [`SplitTracking::samples_held`].
        persisted_samples: usize,
        /// Cross-validated detector hint ID, if any.
        hint: Option<HintId>,
        /// Simulation tick at which speciation was confirmed.
        tick: Tick,
    },
    /// Extinction: a species has zero living members.
    Extinction {
        /// Extinct species ID.
        species: SpeciesId,
        /// UID of the last surviving member to die.
        last_member: AgentUid,
        /// Historical peak population size observed for this species.
        peak_size: usize,
        /// Histogram of mortality causes for this species.
        cause_histogram: BTreeMap<DeathCause, u32>,
        /// Simulation tick at which extinction was recorded.
        tick: Tick,
    },
    /// Radiation: a species experiences rapid population expansion.
    Radiation {
        /// Radiating species ID.
        species: SpeciesId,
        /// Population size before radiation window.
        from_size: usize,
        /// Population size after radiation window.
        to_size: usize,
        /// Tick window over which radiation occurred.
        window: TickRange,
        /// Relative expansion score.
        score: f32,
        /// Cross-validated detector hint ID, if any.
        hint: Option<HintId>,
        /// Simulation tick at which radiation was confirmed.
        tick: Tick,
    },
}

impl PhyloEvent {
    /// Simulation tick at which the event occurred.
    #[must_use]
    pub const fn tick(&self) -> Tick {
        match self {
            Self::Speciation { tick, .. }
            | Self::Extinction { tick, .. }
            | Self::Radiation { tick, .. } => *tick,
        }
    }

    /// Primary species identity associated with the event.
    #[must_use]
    pub const fn species_id(&self) -> SpeciesId {
        match self {
            Self::Speciation { parent, .. } => *parent,
            Self::Extinction { species, .. } | Self::Radiation { species, .. } => *species,
        }
    }

    /// Discriminant for deterministic total ordering.
    #[must_use]
    pub const fn kind_discriminant(&self) -> u8 {
        match self {
            Self::Speciation { .. } => 0,
            Self::Extinction { .. } => 1,
            Self::Radiation { .. } => 2,
        }
    }
}

impl Ord for PhyloEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tick()
            .0
            .cmp(&other.tick().0)
            .then(self.kind_discriminant().cmp(&other.kind_discriminant()))
            .then(self.species_id().0.cmp(&other.species_id().0))
    }
}

impl PartialOrd for PhyloEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PhyloEvent {}

/// Candidate hint from the detector kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectorHint {
    /// Stable hint ID.
    pub id: HintId,
    /// Tick when hint was produced.
    pub tick: Tick,
    /// Primitive kind.
    pub kind: DetectorHintKind,
    /// Score / magnitude.
    pub score: f32,
    /// Monitored metric name.
    pub metric: String,
    /// Target species predicted by hint, if known.
    pub target_species: Option<SpeciesId>,
}

/// Primitive kind of detector hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorHintKind {
    /// Bimodality in phenotype or metric series.
    Bimodality,
    /// Change-point in population or metric series.
    ChangePoint,
}

/// Parameters configuring phylogeny event confirmation and rejection rules (bd-16g.3.3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhyloEventParams {
    /// Consecutive segmentation samples a split must hold (default 3, calibrated in bd-3l5d).
    pub persistence_samples: usize,
    /// Maximum cross-mating rate allowed for speciation (default 0.05, calibrated in bd-3l5d).
    pub max_cross_mating: f32,
    /// Minimum members required in both child sub-clusters (default 2).
    pub min_species_size: usize,
    /// Minimum two-parent births required in the window to avoid empty denominator (default 2).
    pub min_two_parent_births: usize,
    /// Minimum members required to qualify as radiation (default 5).
    pub radiation_min_members: usize,
    /// Expansion factor required for radiation (default 2.0).
    pub radiation_growth_factor: f32,
    /// Lookback tick window within which a hint can match an event (default 20).
    pub hint_match_window: u64,
}

impl Default for PhyloEventParams {
    fn default() -> Self {
        Self {
            persistence_samples: 3,
            max_cross_mating: 0.05,
            min_species_size: 2,
            min_two_parent_births: 2,
            radiation_min_members: 5,
            radiation_growth_factor: 2.0,
            hint_match_window: 20,
        }
    }
}

/// Candidate split key identifying the ancestral parent and diverging sub-clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CandidateSplitKey {
    /// Ancestor species.
    pub parent: SpeciesId,
    /// First child clade.
    pub child_a: SpeciesId,
    /// Second child clade.
    pub child_b: SpeciesId,
}

/// Tracking state for a candidate split across consecutive segmentation samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitTracking {
    /// Tick when the split was first observed.
    pub first_seen_tick: Tick,
    /// Number of consecutive samples the split has satisfied all criteria.
    pub samples_held: usize,
    /// Most recent observed cross-mating rate.
    pub last_cross_mating_rate: f32,
    /// Most recent sample sizes (`child_a`, `child_b`).
    pub last_sample_sizes: (usize, usize),
    /// Bound detector hint ID, if any.
    pub matched_hint: Option<HintId>,
    /// Last rejection reason observed, if failed.
    pub last_rejection_reason: Option<RejectReason>,
    /// Last separation kind evaluated.
    pub separation_kind: SeparationKind,
}

/// Running state maintained by the phylogeny event engine across simulation ticks.
#[derive(Debug, Clone, Default)]
pub struct PhyloEngineState {
    /// Next monotonic event sequence ID.
    pub next_event_id: u64,
    /// Already extinct species IDs (guarantees extinction idempotence).
    pub extinct_species: BTreeSet<SpeciesId>,
    /// Historical peak population size observed per species.
    pub peak_sizes: BTreeMap<SpeciesId, usize>,
    /// Last known living members observed per species.
    pub last_known_members: BTreeMap<SpeciesId, Vec<AgentUid>>,
    /// Mortality cause histogram accumulated per species.
    pub cause_histograms: BTreeMap<SpeciesId, BTreeMap<DeathCause, u32>>,
    /// Candidate splits currently tracked across samples.
    pub pending_splits: BTreeMap<CandidateSplitKey, SplitTracking>,
    /// Species sizes from previous segmentation sample: `species_id` -> `member_count`.
    pub prev_species_sizes: BTreeMap<SpeciesId, usize>,
    /// Tick of previous segmentation sample.
    pub prev_sample_tick: Option<Tick>,
    /// Cumulative counter of anomalies (e.g. clustering dropped species with living members).
    pub anomaly_count: usize,
    /// Cumulative counter of unhinted speciation events emitted.
    pub events_unhinted: usize,
}

/// Output produced by a single execution of the phylogeny event engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhyloEngineOutput {
    /// Monotonically ordered stream of emitted phylogeny events.
    pub events: Vec<(EventId, PhyloEvent)>,
    /// Verdict for every hint evaluated at this step.
    pub verdicts: Vec<HintVerdict>,
    /// Anomaly count encountered at this step.
    pub anomalies: usize,
    /// Number of unhinted events emitted at this step.
    pub unhinted_count: usize,
}

/// Evaluates species table progression, ancestry, birth records, and detector hints
/// to produce a total-ordered stream of [`PhyloEvent`]s and reconciled [`HintVerdict`]s (bd-16g.3.3).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn step_phylo_events(
    current_table: &SpeciesTable,
    ancestry: &AncestryGraph,
    births: &[BirthRecord],
    hints: &[DetectorHint],
    params: &PhyloEventParams,
    state: &mut PhyloEngineState,
) -> PhyloEngineOutput {
    let mut emitted_events = Vec::new();
    let mut verdicts = Vec::new();
    let mut unhinted_this_step = 0;
    let mut anomalies_this_step = 0;

    let current_tick = current_table.tick;
    let prev_tick = state.prev_sample_tick.unwrap_or(Tick(0));

    // Map species_id -> species in current_table
    let current_species_map: BTreeMap<SpeciesId, &crate::species::Species> =
        current_table.species.iter().map(|s| (s.id, s)).collect();

    // 1. Update peak sizes and last known members for all active species
    for species in &current_table.species {
        let size = species.members.len();
        state
            .peak_sizes
            .entry(species.id)
            .and_modify(|p| *p = (*p).max(size))
            .or_insert(size);
        if !species.members.is_empty() {
            state
                .last_known_members
                .insert(species.id, species.members.clone());
        }
    }

    // 2. Discover and track candidate splits
    // Match any incoming bimodality hints to existing or new candidate splits
    for hint in hints {
        if let (DetectorHintKind::Bimodality, Some(target)) = (hint.kind, hint.target_species) {
            for species in &current_table.species {
                if species.id != target {
                    let key = CandidateSplitKey {
                        parent: target,
                        child_a: target,
                        child_b: species.id,
                    };
                    state
                        .pending_splits
                        .entry(key)
                        .or_insert_with(|| SplitTracking {
                            first_seen_tick: current_tick,
                            samples_held: 0,
                            last_cross_mating_rate: 0.0,
                            last_sample_sizes: (0, 0),
                            matched_hint: Some(hint.id),
                            last_rejection_reason: None,
                            separation_kind: SeparationKind::Phenotypic,
                        });
                }
            }
        }
    }

    // Auto-discover candidate splits between current species
    for (idx_a, sa) in current_table.species.iter().enumerate() {
        for sb in current_table.species.iter().skip(idx_a + 1) {
            let parent_id = sa.id.min(sb.id);
            let key = CandidateSplitKey {
                parent: parent_id,
                child_a: sa.id,
                child_b: sb.id,
            };
            state
                .pending_splits
                .entry(key)
                .or_insert_with(|| SplitTracking {
                    first_seen_tick: current_tick,
                    samples_held: 0,
                    last_cross_mating_rate: 0.0,
                    last_sample_sizes: (0, 0),
                    matched_hint: None,
                    last_rejection_reason: None,
                    separation_kind: SeparationKind::Phenotypic,
                });
        }
    }

    // 3. Evaluate pending splits
    let mut resolved_split_keys = Vec::new();
    for (key, tracking) in &mut state.pending_splits {
        let (Some(sa), Some(sb)) = (
            current_species_map.get(&key.child_a),
            current_species_map.get(&key.child_b),
        ) else {
            // Split reverted / dropped!
            tracking.last_rejection_reason = Some(RejectReason::Transient);
            resolved_split_keys.push(*key);
            continue;
        };

        tracking.last_sample_sizes = (sa.members.len(), sb.members.len());

        // Check (c) MIN SIZE: both sub-clusters >= min_species_size
        if sa.members.len() < params.min_species_size || sb.members.len() < params.min_species_size
        {
            tracking.last_rejection_reason = Some(RejectReason::BelowMinSize);
            tracking.samples_held = 0;
            continue;
        }

        // Check (b) REPRODUCTIVE SEPARATION and (e) EMPTY DENOMINATOR:
        let mut within_matings = 0;
        let mut cross_matings = 0;
        for birth in births {
            let (Some(pa), Some(pb)) = (birth.parent_a, birth.parent_b) else {
                continue;
            };
            if pa == pb {
                continue; // self-parented / asexual budding
            }
            let in_a = (sa.members.contains(&pa), sa.members.contains(&pb));
            let in_b = (sb.members.contains(&pa), sb.members.contains(&pb));

            if (in_a.0 && in_a.1) || (in_b.0 && in_b.1) {
                within_matings += 1;
            } else if (in_a.0 && in_b.1) || (in_b.0 && in_a.1) {
                cross_matings += 1;
            }
        }

        let total_matings = within_matings + cross_matings;
        if total_matings < params.min_two_parent_births {
            // Empty denominator!
            tracking.last_rejection_reason = Some(RejectReason::NoAncestralSupport);
            tracking.samples_held = 0;
            continue;
        }

        #[allow(clippy::cast_precision_loss)]
        let cross_rate = cross_matings as f32 / total_matings as f32;
        tracking.last_cross_mating_rate = cross_rate;

        if cross_rate > params.max_cross_mating {
            // High interbreeding!
            tracking.last_rejection_reason = Some(RejectReason::Interbreeding);
            tracking.samples_held = 0;
            continue;
        }

        // Check Brain-Kind gating artifact
        let brains_a: BTreeSet<Option<u64>> = sa
            .members
            .iter()
            .map(|uid| ancestry.node(*uid).and_then(|n| n.brain_key))
            .collect();
        let brains_b: BTreeSet<Option<u64>> = sb
            .members
            .iter()
            .map(|uid| ancestry.node(*uid).and_then(|n| n.brain_key))
            .collect();
        let is_brain_gated = !brains_a.is_empty()
            && !brains_b.is_empty()
            && brains_a.intersection(&brains_b).next().is_none();

        let separation = if is_brain_gated {
            SeparationKind::BrainKindGated
        } else {
            SeparationKind::Phenotypic
        };
        tracking.separation_kind = separation;

        // All confirmation criteria satisfied for this sample!
        tracking.samples_held += 1;
        tracking.last_rejection_reason = None;

        if tracking.samples_held >= params.persistence_samples {
            // Emit Speciation!
            state.next_event_id += 1;
            let event_id = EventId(state.next_event_id);
            let speciation = PhyloEvent::Speciation {
                parent: key.parent,
                children: [key.child_a, key.child_b],
                founders: sb.founders.clone(),
                separation,
                cross_mating_rate: cross_rate,
                persisted_samples: tracking.samples_held,
                hint: tracking.matched_hint,
                tick: current_tick,
            };
            emitted_events.push((event_id, speciation));

            if let Some(_hid) = tracking.matched_hint {
                verdicts.push(HintVerdict::Confirmed(event_id));
            } else {
                unhinted_this_step += 1;
            }

            resolved_split_keys.push(*key);
        }
    }

    // Clean up resolved / transient splits and reconcile hints for failed splits
    for key in resolved_split_keys {
        if let Some(tracking) = state.pending_splits.remove(&key)
            && let Some(reason) = tracking.last_rejection_reason
            && let Some(hid) = tracking.matched_hint
        {
            #[expect(
                clippy::cast_precision_loss,
                reason = "hint metrics store approximate f32 counts; exact member counts remain in the species table"
            )]
            let child_sizes = (
                tracking.last_sample_sizes.0 as f32,
                tracking.last_sample_sizes.1 as f32,
            );
            let mut metrics = BTreeMap::new();
            metrics.insert(
                "cross_mating_rate".to_string(),
                tracking.last_cross_mating_rate,
            );
            metrics.insert("child_a_size".to_string(), child_sizes.0);
            metrics.insert("child_b_size".to_string(), child_sizes.1);
            verdicts.push(HintVerdict::Rejected {
                reason,
                evidence: HintEvidence {
                    hint_id: hid,
                    tick: current_tick,
                    score: 0.0,
                    detail: format!("Candidate split {key:?} rejected: {reason:?}"),
                    metrics,
                },
            });
        }
    }

    // 4. Extinction detection with idempotence & living member verification
    for (&prev_sp_id, &prev_size) in &state.prev_species_sizes {
        if prev_size > 0 && !current_species_map.contains_key(&prev_sp_id) {
            // Species not present in current clustering table
            if state.extinct_species.contains(&prev_sp_id) {
                continue; // Idempotent: already extinct, never emit again!
            }

            // Verify against AncestryGraph: are there any living members?
            let known_members = state
                .last_known_members
                .get(&prev_sp_id)
                .cloned()
                .unwrap_or_default();
            let has_living_members = known_members.iter().any(|uid| {
                ancestry
                    .node(*uid)
                    .is_some_and(|node| node.death_tick.is_none() && !node.pruned)
            });

            if has_living_members {
                // ANOMALY: clustering dropped a species that still has living members!
                anomalies_this_step += 1;
                state.anomaly_count += 1;
                // Emit warning counter rather than false extinction!
                continue;
            }

            // Living count is zero -> GENUINE EXTINCTION!
            state.extinct_species.insert(prev_sp_id);

            // Find last member and accumulate cause histogram
            let mut last_member = known_members.first().copied().unwrap_or(AgentUid(0));
            let mut latest_death = Tick(0);
            let mut cause_histogram = BTreeMap::new();

            for uid in &known_members {
                if let Some(node) = ancestry.node(*uid) {
                    if let Some(dt) = node.death_tick
                        && dt.0 >= latest_death.0
                    {
                        latest_death = dt;
                        last_member = *uid;
                    }
                    if let Some(cause) = node.death_cause {
                        *cause_histogram.entry(cause).or_insert(0) += 1;
                    }
                }
            }

            state.next_event_id += 1;
            let event_id = EventId(state.next_event_id);
            let peak_size = state
                .peak_sizes
                .get(&prev_sp_id)
                .copied()
                .unwrap_or(prev_size);
            let extinction = PhyloEvent::Extinction {
                species: prev_sp_id,
                last_member,
                peak_size,
                cause_histogram,
                tick: current_tick,
            };
            emitted_events.push((event_id, extinction));
        }
    }

    // 5. Radiation detection: doubling inside window with min size
    for (&sp_id, species) in &current_species_map {
        if let Some(&prev_size) = state.prev_species_sizes.get(&sp_id) {
            let curr_size = species.members.len();
            #[expect(
                clippy::cast_precision_loss,
                reason = "radiation uses the existing f32 population ratio and configured f32 growth threshold; widening would change boundary decisions"
            )]
            let is_radiation = curr_size >= params.radiation_min_members
                && prev_size >= 1
                && (curr_size as f32) >= (prev_size as f32) * params.radiation_growth_factor;
            if is_radiation {
                // Radiation detected!
                state.next_event_id += 1;
                let event_id = EventId(state.next_event_id);
                #[allow(clippy::cast_precision_loss)]
                let growth_score = (curr_size as f32) / (prev_size as f32);

                // Check for matching change-point hint in window
                let matched_hint = hints.iter().find(|h| {
                    h.kind == DetectorHintKind::ChangePoint
                        && (h.target_species.is_none() || h.target_species == Some(sp_id))
                        && h.tick.0 <= current_tick.0
                        && current_tick.0 <= h.tick.0 + params.hint_match_window
                });

                let hint_id = matched_hint.map(|h| h.id);
                if matched_hint.is_some() {
                    verdicts.push(HintVerdict::Confirmed(event_id));
                }

                let radiation = PhyloEvent::Radiation {
                    species: sp_id,
                    from_size: prev_size,
                    to_size: curr_size,
                    window: TickRange {
                        start: prev_tick,
                        end: current_tick,
                    },
                    score: growth_score,
                    hint: hint_id,
                    tick: current_tick,
                };
                emitted_events.push((event_id, radiation));
            }
        }
    }

    // 6. Total hint reconciliation: ensure EVERY input hint has a verdict
    for hint in hints {
        let already_reconciled = verdicts.iter().any(|v| match v {
            HintVerdict::Confirmed(eid) => emitted_events.iter().any(|(id, e)| {
                id == eid
                    && (match e {
                        PhyloEvent::Speciation { hint: h, .. }
                        | PhyloEvent::Radiation { hint: h, .. } => *h == Some(hint.id),
                        PhyloEvent::Extinction { .. } => false,
                    })
            }),
            HintVerdict::Rejected { evidence, .. } => evidence.hint_id == hint.id,
        });

        if !already_reconciled {
            // Hint was not confirmed by any event at this step
            let mut metrics = BTreeMap::new();
            metrics.insert("hint_score".to_string(), hint.score);
            let reason = match hint.kind {
                DetectorHintKind::Bimodality => RejectReason::Transient,
                DetectorHintKind::ChangePoint => RejectReason::BelowMinSize,
            };
            verdicts.push(HintVerdict::Rejected {
                reason,
                evidence: HintEvidence {
                    hint_id: hint.id,
                    tick: current_tick,
                    score: hint.score,
                    detail: format!(
                        "Hint {:?} had no confirming phylogeny event in window",
                        hint.id
                    ),
                    metrics,
                },
            });
        }
    }

    // 7. Update state for next step
    state.prev_species_sizes = current_table
        .species
        .iter()
        .map(|s| (s.id, s.members.len()))
        .collect();
    state.prev_sample_tick = Some(current_tick);
    state.events_unhinted += unhinted_this_step;

    // 8. Deterministic total ordering of emitted events: sorted by (tick, kind discriminant, species id)
    emitted_events.sort_by(|(_, a), (_, b)| a.cmp(b));

    // Sort verdicts by hint_id for byte-identical determinism
    verdicts.sort_by_key(|v| match v {
        HintVerdict::Confirmed(eid) => (eid.0, 0),
        HintVerdict::Rejected { evidence, .. } => (evidence.hint_id.0, 1),
    });

    PhyloEngineOutput {
        events: emitted_events,
        verdicts,
        anomalies: anomalies_this_step,
        unhinted_count: unhinted_this_step,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn species(
        id: u64,
        parent: ParentRef,
        founder: u64,
        birth: u64,
        death: Option<u64>,
        population: u32,
    ) -> PhyloNodeUpdate {
        PhyloNodeUpdate {
            key: PhyloKey::Species(id),
            parent,
            founder_uid: AgentUid(founder),
            birth_tick: Tick(birth),
            death_tick: death.map(Tick),
            population,
        }
    }

    fn assert_same_coordinates(left: &TreeLayout, right: &TreeLayout) {
        assert_eq!(left.len(), right.len());
        for left_node in left.nodes() {
            let right_node = right
                .get_node(&left_node.key)
                .expect("reference layout contains every incremental node");
            assert_eq!(
                left_node.x.to_bits(),
                right_node.x.to_bits(),
                "x divergence for {:?}",
                left_node.key
            );
            assert_eq!(
                left_node.y.to_bits(),
                right_node.y.to_bits(),
                "y divergence for {:?}",
                left_node.key
            );
            assert_eq!(left_node.first_tick, right_node.first_tick);
            assert_eq!(left_node.last_tick, right_node.last_tick);
            assert_eq!(left_node.peak_population, right_node.peak_population);
            let left_parent = left_node.parent.map(|idx| left.nodes[idx.as_usize()].key);
            let right_parent = right_node.parent.map(|idx| right.nodes[idx.as_usize()].key);
            assert_eq!(
                left_parent, right_parent,
                "parent divergence for {:?}",
                left_node.key
            );
        }
        assert_eq!(
            left.bounds().x_min.to_bits(),
            right.bounds().x_min.to_bits()
        );
        assert_eq!(
            left.bounds().x_max.to_bits(),
            right.bounds().x_max.to_bits()
        );
        assert_eq!(
            left.bounds().y_min.to_bits(),
            right.bounds().y_min.to_bits()
        );
        assert_eq!(
            left.bounds().y_max.to_bits(),
            right.bounds().y_max.to_bits()
        );
    }

    fn brute_force_keys(
        layout: &TreeLayout,
        viewport: TickRange,
        y_range: Range<f32>,
        budget: usize,
    ) -> Vec<PhyloKey> {
        let mut visible = layout
            .nodes()
            .iter()
            .filter(|node| {
                node.first_tick <= viewport.end
                    && node.last_tick.unwrap_or(Tick(u64::MAX)) >= viewport.start
                    && node.y >= y_range.start
                    && node.y <= y_range.end
            })
            .collect::<Vec<_>>();
        visible.sort_by(|left, right| lod_rank(right).cmp(&lod_rank(left)));
        visible
            .into_iter()
            .take(budget)
            .map(|node| node.key)
            .collect()
    }

    fn indexed_keys(report: &LodReport<'_>) -> Vec<PhyloKey> {
        report.nodes.iter().map(|node| node.key).collect()
    }

    fn reference_aggregate(
        key: PhyloKey,
        updates: &BTreeMap<PhyloKey, PhyloNodeUpdate>,
        children: &BTreeMap<PhyloKey, Vec<PhyloKey>>,
    ) -> SubtreeAggregate {
        let child_keys = children.get(&key).map_or(&[][..], Vec::as_slice);
        if child_keys.is_empty() {
            let update = updates.get(&key).expect("reference update exists");
            return SubtreeAggregate::leaf(update_anchor(update.key, update.founder_uid));
        }
        child_keys
            .iter()
            .map(|child| reference_aggregate(*child, updates, children))
            .reduce(|left, right| {
                left.checked_add(right)
                    .expect("test aggregate remains bounded")
            })
            .expect("non-empty child list")
    }

    #[test]
    fn memory_bound_accounts_for_every_retained_allocation_category() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        let mut updates = Vec::with_capacity(10_000);
        updates.push(species(0, ParentRef::Root, 0, 0, None, 10_000));
        for id in 1_u64..10_000 {
            updates.push(species(
                id,
                ParentRef::Known(PhyloKey::Species(0)),
                id,
                id,
                None,
                1,
            ));
        }

        let retained_before = layout.memory_report().total_retained_bytes;
        let delta = layout.extend(&PhyloDelta { updates });
        let memory = layout.memory_report();
        assert!(delta.issues.is_empty(), "{:?}", delta.issues);
        assert_eq!(layout.len(), 10_000);
        assert!(!delta.full_relayout);
        assert_eq!(
            delta.allocated_bytes,
            memory.total_retained_bytes - retained_before
        );
        assert!(memory.node_capacity_bytes > 0);
        assert!(memory.aggregate_capacity_bytes > 0);
        assert!(memory.topology_capacity_bytes > 0);
        assert!(memory.child_capacity_bytes > 0);
        assert!(memory.key_index_bytes > 0);
        assert!(memory.lod_index_bytes > 0);
        assert_eq!(memory.pending_index_bytes, 0);
        assert!(
            memory.total_retained_bytes <= DEFAULT_LAYOUT_BYTES,
            "{} retained bytes exceeded the documented {}-byte budget",
            memory.total_retained_bytes,
            DEFAULT_LAYOUT_BYTES
        );
        assert!(
            delta.allocation_growths <= layout.len().saturating_add(128),
            "unexpected non-amortized allocation growth: {}",
            delta.allocation_growths
        );
    }

    #[test]
    fn empty_single_budget_zero_and_hit_test_boundaries_are_safe() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        assert!(layout.is_empty());
        assert_eq!(layout.hit_test(0.0, 0.0, 5.0), None);
        assert_eq!(layout.hit_test(f32::NAN, 0.0, 5.0), None);
        assert!(
            layout
                .lod(
                    TickRange {
                        start: Tick(0),
                        end: Tick(100),
                    },
                    0.0..10.0,
                    0,
                )
                .next()
                .is_none()
        );

        let delta = PhyloDelta {
            updates: vec![species(1, ParentRef::Root, 7, 10, Some(20), 50)],
        };

        let res = layout.extend(&delta);
        assert_eq!(res.added.len(), 1);
        assert_eq!(layout.len(), 1);

        let node = layout
            .get_node(&PhyloKey::Species(1))
            .expect("single node exists");
        assert_eq!(node.x, 10.0);
        assert_eq!(node.y, 8.0);
        assert_eq!(layout.hit_test(10.5, 8.0, 0.5), Some(PhyloKey::Species(1)));
        assert_eq!(layout.hit_test(10.6, 8.0, 0.5), None);
        assert_eq!(layout.hit_test(10.0, 8.0, -1.0), None);
        assert!(
            layout
                .lod(
                    TickRange {
                        start: Tick(0),
                        end: Tick(9),
                    },
                    0.0..20.0,
                    1,
                )
                .next()
                .is_none()
        );
        assert!(
            layout
                .lod(
                    TickRange {
                        start: Tick(21),
                        end: Tick(30),
                    },
                    0.0..20.0,
                    1,
                )
                .next()
                .is_none()
        );
    }

    #[test]
    fn drift_oracle_matches_scratch_and_independent_exact_aggregates() {
        let mut updates = Vec::new();
        for id in 0_u64..255 {
            let parent = if id == 0 {
                ParentRef::Root
            } else {
                ParentRef::Known(PhyloKey::Species((id - 1) / 2))
            };
            updates.push(species(
                id,
                parent,
                id.wrapping_mul(37) % 257,
                id,
                None,
                u32::try_from((id % 31) + 1).expect("small population"),
            ));
        }

        let mut inc_layout = TreeLayout::new(LayoutBudget::default());
        for chunk in updates.chunks(17) {
            let report = inc_layout.extend(&PhyloDelta {
                updates: chunk.to_vec(),
            });
            assert!(!report.full_relayout);
            assert!(report.issues.is_empty(), "{:?}", report.issues);
        }

        let mut scratch_layout = TreeLayout::new(LayoutBudget::default());
        let scratch_report = scratch_layout.extend(&PhyloDelta {
            updates: updates.iter().rev().cloned().collect(),
        });
        assert!(scratch_report.issues.is_empty());
        assert_same_coordinates(&inc_layout, &scratch_layout);

        let by_key = updates
            .iter()
            .cloned()
            .map(|update| (update.key, update))
            .collect::<BTreeMap<_, _>>();
        let mut children = BTreeMap::<PhyloKey, Vec<PhyloKey>>::new();
        for update in &updates {
            if let ParentRef::Known(parent) = update.parent {
                children.entry(parent).or_default().push(update.key);
            }
        }
        for node in inc_layout.nodes() {
            let expected = reference_aggregate(node.key, &by_key, &children);
            assert_eq!(
                node.y.to_bits(),
                expected.y().to_bits(),
                "independent aggregate divergence for {:?}",
                node.key
            );
        }
    }

    #[test]
    fn explicit_pruned_parent_reattaches_to_sentinel_without_dangling_index() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        let report = layout.extend(&PhyloDelta {
            updates: vec![species(10, ParentRef::Pruned, 42, 50, None, 15)],
        });
        assert!(report.issues.is_empty());
        let node = layout
            .get_node(&PhyloKey::Species(10))
            .expect("pruned child exists");
        let parent_idx = node.parent.expect("pruned child has sentinel parent");
        let parent_node = &layout.nodes()[parent_idx.as_usize()];
        assert_eq!(parent_node.key, PhyloKey::PrunedAncestor);
        assert_eq!(layout.get_idx(&PhyloKey::PrunedAncestor), Some(parent_idx));
    }

    #[test]
    fn late_parent_duplicate_and_invalid_lineage_updates_are_typed() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        let child = species(2, ParentRef::Known(PhyloKey::Species(1)), 20, 20, None, 5);
        let pending = layout.extend(&PhyloDelta {
            updates: vec![child.clone()],
        });
        assert!(pending.issues.is_empty());
        assert_eq!(
            layout
                .get_node(&PhyloKey::Species(2))
                .expect("pending child retained")
                .parent,
            None
        );
        assert!(layout.memory_report().pending_index_bytes > 0);

        let resolved = layout.extend(&PhyloDelta {
            updates: vec![species(1, ParentRef::Root, 10, 10, None, 8)],
        });
        assert_eq!(resolved.pending_children_resolved, 1);
        assert_eq!(
            layout
                .get_node(&PhyloKey::Species(2))
                .expect("resolved child exists")
                .parent,
            layout.get_idx(&PhyloKey::Species(1))
        );
        assert_eq!(layout.memory_report().pending_index_bytes, 0);

        let retained = layout.memory_report().total_retained_bytes;
        let duplicate = layout.extend(&PhyloDelta {
            updates: vec![child.clone(), child],
        });
        assert!(duplicate.added.is_empty());
        assert!(duplicate.moved.is_empty());
        assert!(duplicate.issues.is_empty());
        assert_eq!(duplicate.allocated_bytes, 0);
        assert_eq!(layout.memory_report().total_retained_bytes, retained);

        let conflicting_key = PhyloKey::Species(30);
        let conflicting = layout.extend(&PhyloDelta {
            updates: vec![
                species(30, ParentRef::Root, 30, 30, None, 1),
                species(30, ParentRef::Root, 31, 30, None, 1),
            ],
        });
        assert_eq!(
            conflicting.issues,
            vec![LayoutIssue::ConflictingDuplicate {
                key: conflicting_key
            }]
        );
        assert!(layout.get_node(&conflicting_key).is_none());

        let invalid = layout.extend(&PhyloDelta {
            updates: vec![
                species(40, ParentRef::Known(PhyloKey::Species(40)), 40, 40, None, 1),
                species(41, ParentRef::Root, 41, 50, Some(49), 1),
            ],
        });
        assert!(invalid.issues.contains(&LayoutIssue::Cycle {
            key: PhyloKey::Species(40),
            parent: PhyloKey::Species(40),
        }));
        assert!(invalid.issues.contains(&LayoutIssue::InvalidLifetime {
            key: PhyloKey::Species(41),
        }));
    }

    #[test]
    fn indexed_lod_matches_brute_force_across_pan_zoom_and_ties() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        let mut updates = Vec::new();
        for id in 0_u64..512 {
            updates.push(species(
                id,
                ParentRef::Root,
                id.wrapping_mul(97) % 521,
                100 + (id % 80),
                Some(180 + (id % 160)),
                u32::try_from((id.wrapping_mul(17) % 101) + 1).expect("small population"),
            ));
        }
        // Exact rank tie: smaller founder wins even though its species key is larger.
        updates.push(species(10_001, ParentRef::Root, 900, 120, Some(220), 700));
        updates.push(species(10_002, ParentRef::Root, 800, 120, Some(220), 700));
        let update = layout.extend(&PhyloDelta { updates });
        assert!(update.issues.is_empty());

        let queries = [
            (
                TickRange {
                    start: Tick(100),
                    end: Tick(400),
                },
                0.0..2_000.0,
                17,
            ),
            (
                TickRange {
                    start: Tick(150),
                    end: Tick(190),
                },
                100.0..300.0,
                31,
            ),
            (
                TickRange {
                    start: Tick(210),
                    end: Tick(240),
                },
                250.0..450.0,
                9,
            ),
        ];
        for (viewport, y_range, budget) in queries {
            let indexed = layout.lod_report(viewport, y_range.start..y_range.end, budget);
            let repeated = layout.lod_report(viewport, y_range.start..y_range.end, budget);
            let expected = brute_force_keys(&layout, viewport, y_range, budget);
            assert_eq!(indexed_keys(&indexed), expected);
            assert_eq!(indexed_keys(&repeated), expected);
            assert!(indexed.nodes.len() <= budget);
            assert!(indexed.index_nodes_visited <= layout.len());
        }

        let tied = layout.lod_report(
            TickRange {
                start: Tick(120),
                end: Tick(220),
            },
            0.0..2_000.0,
            2,
        );
        assert_eq!(
            indexed_keys(&tied),
            vec![PhyloKey::Species(10_002), PhyloKey::Species(10_001)]
        );
    }

    #[test]
    fn indexed_lod_prunes_extinction_heavy_outside_viewports() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        let updates = (0_u64..2_048)
            .map(|id| {
                species(
                    id,
                    ParentRef::Root,
                    id,
                    1_000 + id,
                    Some(2_000 + id),
                    u32::try_from((id % 47) + 1).expect("small population"),
                )
            })
            .collect();
        layout.extend(&PhyloDelta { updates });

        let before = layout.lod_report(
            TickRange {
                start: Tick(0),
                end: Tick(999),
            },
            0.0..10_000.0,
            10,
        );
        let after = layout.lod_report(
            TickRange {
                start: Tick(4_048),
                end: Tick(5_000),
            },
            0.0..10_000.0,
            10,
        );
        assert!(before.nodes.is_empty());
        assert!(after.nodes.is_empty());
        assert_eq!(before.index_nodes_visited, 0);
        assert_eq!(after.index_nodes_visited, 0);
    }

    #[test]
    fn chain_star_and_balanced_updates_expose_only_affected_work() {
        let chain_len = 256_u64;
        let chain_updates = (0..chain_len)
            .map(|id| {
                let parent = if id == 0 {
                    ParentRef::Root
                } else {
                    ParentRef::Known(PhyloKey::Species(id - 1))
                };
                species(id, parent, id, id, None, 1)
            })
            .collect();
        let mut chain = TreeLayout::new(LayoutBudget::default());
        let chain_report = chain.extend(&PhyloDelta {
            updates: chain_updates,
        });
        let expected_chain_steps =
            usize::try_from(chain_len.saturating_mul(chain_len - 1) / 2).expect("small chain");
        assert_eq!(chain_report.ancestor_steps, expected_chain_steps);
        assert!(!chain_report.full_relayout);

        let mut star = TreeLayout::new(LayoutBudget::default());
        star.extend(&PhyloDelta {
            updates: vec![species(10_000, ParentRef::Root, 0, 0, None, 500)],
        });
        let star_updates = (1_u64..2_049)
            .map(|id| {
                species(
                    10_000 + id,
                    ParentRef::Known(PhyloKey::Species(10_000)),
                    id,
                    id,
                    None,
                    1,
                )
            })
            .collect();
        let star_report = star.extend(&PhyloDelta {
            updates: star_updates,
        });
        assert_eq!(star_report.ancestor_steps, 2_048);
        assert!(!star_report.full_relayout);

        let leaf_update = species(
            10_001,
            ParentRef::Known(PhyloKey::Species(10_000)),
            1,
            1,
            Some(50),
            20,
        );
        let mutable_report = star.extend(&PhyloDelta {
            updates: vec![leaf_update],
        });
        assert_eq!(mutable_report.ancestor_steps, 0);
        assert!(mutable_report.moved.is_empty());
        assert!(!mutable_report.full_relayout);

        let balanced_len = 1_023_u64;
        let balanced_updates = (0..balanced_len)
            .map(|id| {
                let parent = if id == 0 {
                    ParentRef::Root
                } else {
                    ParentRef::Known(PhyloKey::Species(20_000 + ((id - 1) / 2)))
                };
                species(20_000 + id, parent, id, id, None, 1)
            })
            .collect();
        let mut balanced = TreeLayout::new(LayoutBudget::default());
        let balanced_report = balanced.extend(&PhyloDelta {
            updates: balanced_updates,
        });
        assert!(
            balanced_report.ancestor_steps
                <= usize::try_from(balanced_len)
                    .expect("small balanced tree")
                    .saturating_mul(10)
        );
        assert!(!balanced_report.full_relayout);
    }

    #[test]
    fn node_and_byte_budget_boundaries_refuse_without_partial_sentinel() {
        let mut layout = TreeLayout::new(LayoutBudget {
            max_nodes: 1,
            max_depth: 1,
            max_bytes: DEFAULT_LAYOUT_BYTES,
        });
        let report = layout.extend(&PhyloDelta {
            updates: vec![species(1, ParentRef::Pruned, 1, 1, None, 1)],
        });
        assert!(layout.is_empty());
        assert_eq!(
            report.issues,
            vec![LayoutIssue::BudgetExceeded {
                key: PhyloKey::Species(1)
            }]
        );

        let mut no_bytes = TreeLayout::new(LayoutBudget {
            max_nodes: 10,
            max_depth: 10,
            max_bytes: 0,
        });
        let report = no_bytes.extend(&PhyloDelta {
            updates: vec![species(2, ParentRef::Root, 2, 2, None, 1)],
        });
        assert!(no_bytes.is_empty());
        assert_eq!(report.issues.len(), 1);
    }

    #[test]
    fn deterministic_hundred_thousand_node_stress_stays_incremental() {
        const NODE_COUNT: usize = 100_000;
        let mut layout = TreeLayout::new(LayoutBudget {
            max_nodes: NODE_COUNT,
            max_depth: 100,
            max_bytes: 256 << 20,
        });
        let mut updates = Vec::with_capacity(NODE_COUNT);
        updates.push(species(0, ParentRef::Root, 0, 0, None, 100_000));
        for id in 1..NODE_COUNT {
            let id = u64::try_from(id).expect("node count fits u64");
            updates.push(species(
                id,
                ParentRef::Known(PhyloKey::Species(0)),
                id,
                id,
                Some(id + 1_000),
                u32::try_from((id % 101) + 1).expect("small population"),
            ));
        }

        let started = Instant::now();
        let update = layout.extend(&PhyloDelta { updates });
        let elapsed = started.elapsed();
        let viewport = TickRange {
            start: Tick(40_000),
            end: Tick(60_000),
        };
        let lod = layout.lod_report(viewport, 45_000.0..55_000.0, 128);
        let memory = layout.memory_report();
        eprintln!(
            "phylo_stress nodes={} added={} ancestor_steps={} index_update_visits={} \
             allocation_growths={} allocated_bytes={} elapsed_ms={} lod_visits={} \
             lod_results={} lod_height={} first_divergence=none",
            layout.len(),
            update.added.len(),
            update.ancestor_steps,
            update.index_nodes_visited,
            update.allocation_growths,
            update.allocated_bytes,
            elapsed.as_millis(),
            lod.index_nodes_visited,
            lod.nodes.len(),
            lod.index_height,
        );
        assert!(update.issues.is_empty(), "{:?}", update.issues);
        assert_eq!(layout.len(), NODE_COUNT);
        assert_eq!(update.ancestor_steps, NODE_COUNT - 1);
        assert!(!update.full_relayout);
        assert!(update.allocation_growths <= NODE_COUNT.saturating_add(160));
        assert!(memory.total_retained_bytes <= 256 << 20);
        assert_eq!(lod.nodes.len(), 128);
        assert!(
            lod.index_nodes_visited
                <= lod
                    .index_height
                    .saturating_mul(4)
                    .saturating_add(lod.nodes.len().saturating_mul(8)),
            "indexed query visited {} nodes at height {} for {} results",
            lod.index_nodes_visited,
            lod.index_height,
            lod.nodes.len()
        );
    }

    // =========================================================================
    // Unit Tests for Phylogeny Event Stream & Hint Cross-Validation (bd-16g.3.3)
    // =========================================================================

    fn make_test_species(id: u64, members: &[u64]) -> crate::species::Species {
        crate::species::Species {
            id: SpeciesId(id),
            name: format!("Species-{id}"),
            founders: members.iter().copied().take(1).map(AgentUid).collect(),
            members: members.iter().copied().map(AgentUid).collect(),
            centroid: vec![0.5; 6],
            spread: 0.1,
            first_tick: Tick(0),
            last_seen_tick: Tick(100),
        }
    }

    fn make_birth(uid: u64, pa: Option<u64>, pb: Option<u64>) -> BirthRecord {
        BirthRecord {
            tick: Tick(10),
            agent_uid: AgentUid(uid),
            spawn_ordinal: uid,
            birth_ordinal: Some(uid),
            origin: crate::BirthOrigin::Born,
            parent_a: pa.map(AgentUid),
            parent_b: pb.map(AgentUid),
            brain_kind: None,
            brain_key: None,
            herbivore_tendency: 0.5,
            generation: crate::Generation(1),
            position: crate::Position::default(),
            is_hybrid: false,
        }
    }

    fn make_death(uid: u64, cause: DeathCause, tick: Tick) -> crate::DeathRecord {
        crate::DeathRecord {
            tick,
            agent_uid: AgentUid(uid),
            age: 50,
            generation: crate::Generation(1),
            herbivore_tendency: 0.5,
            brain_kind: None,
            brain_key: None,
            energy: 0.0,
            food_balance_total: 0.0,
            cause,
            was_hybrid: false,
            combat_flags: crate::CombatEventFlags::default(),
        }
    }

    fn build_test_ancestry(members: &[u64], brain_keys: Option<&[(u64, u64)]>) -> AncestryGraph {
        let mut graph = AncestryGraph::new();
        for &uid in members {
            let mut record = make_birth(uid, None, None);
            if let Some(bkeys) = brain_keys {
                if let Some(&(_, bk)) = bkeys.iter().find(|(u, _)| *u == uid) {
                    record.brain_key = Some(bk);
                }
            }
            let _ = graph.apply_birth(&record);
        }
        graph
    }

    /// (c) Split with high observed cross-cluster mating -> NO event, hint Rejected{Interbreeding}.
    /// THIS IS THE TEST THAT STOPS THE FEATURE FROM LYING; write it first.
    #[test]
    fn test_bd_16g_3_3_c_high_cross_mating_is_rejected_as_interbreeding() {
        let mut table = SpeciesTable::default();
        table.tick = Tick(10);
        table
            .species
            .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));
        table
            .species
            .push(make_test_species(2, &(11..=20).collect::<Vec<_>>()));

        let ancestry = build_test_ancestry(&(1..=20).collect::<Vec<_>>(), None);

        // 20 births: 10 within, 10 cross-mating -> cross_rate = 0.50 >> max_cross_mating (0.05)
        let mut births = Vec::new();
        for idx in 0..10 {
            births.push(make_birth(100 + idx, Some(1), Some(2))); // within species 1
            births.push(make_birth(200 + idx, Some(1), Some(11))); // cross: species 1 x species 2
        }

        let hint = DetectorHint {
            id: HintId(1),
            tick: Tick(10),
            kind: DetectorHintKind::Bimodality,
            score: 0.85,
            metric: "phenotype_bimodality".to_string(),
            target_species: Some(SpeciesId(1)),
        };

        let params = PhyloEventParams::default();
        let mut state = PhyloEngineState::default();

        let output = step_phylo_events(&table, &ancestry, &births, &[hint], &params, &mut state);

        // Assert NO speciation event emitted
        assert!(
            output.events.is_empty(),
            "interbreeding population must not emit speciation"
        );
        let split_key = CandidateSplitKey {
            parent: SpeciesId(1),
            child_a: SpeciesId(1),
            child_b: SpeciesId(2),
        };
        let tracking = state
            .pending_splits
            .get(&split_key)
            .expect("tracking exists");
        assert_eq!(tracking.samples_held, 0);
        assert_eq!(
            tracking.last_rejection_reason,
            Some(RejectReason::Interbreeding)
        );
    }

    /// (a) Clean split persisting K samples with zero cross-mating -> exactly one Speciation,
    /// hint Confirmed, separation=Phenotypic.
    #[test]
    fn test_bd_16g_3_3_a_clean_split_persisting_k_samples_speciation_confirmed() {
        let mut table = SpeciesTable::default();
        table
            .species
            .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));
        table
            .species
            .push(make_test_species(2, &(11..=20).collect::<Vec<_>>()));

        let ancestry = build_test_ancestry(&(1..=20).collect::<Vec<_>>(), None);

        // Births with zero cross-mating
        let mut births = Vec::new();
        for idx in 0..10 {
            births.push(make_birth(100 + idx, Some(1), Some(2))); // within species 1
            births.push(make_birth(200 + idx, Some(11), Some(12))); // within species 2
        }

        let hint = DetectorHint {
            id: HintId(42),
            tick: Tick(10),
            kind: DetectorHintKind::Bimodality,
            score: 0.95,
            metric: "phenotype_bimodality".to_string(),
            target_species: Some(SpeciesId(1)),
        };

        let params = PhyloEventParams {
            persistence_samples: 3,
            ..Default::default()
        };
        let mut state = PhyloEngineState::default();

        // Sample 1: first observation
        table.tick = Tick(10);
        let out1 = step_phylo_events(
            &table,
            &ancestry,
            &births,
            &[hint.clone()],
            &params,
            &mut state,
        );
        assert!(out1.events.is_empty());

        // Sample 2: held for 2 samples
        table.tick = Tick(20);
        let out2 = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);
        assert!(out2.events.is_empty());

        // Sample 3: reaches K=3 -> speciation confirmed!
        table.tick = Tick(30);
        let out3 = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);
        assert_eq!(out3.events.len(), 1, "exactly one speciation event at K=3");

        let (eid, event) = &out3.events[0];
        match event {
            PhyloEvent::Speciation {
                parent,
                children,
                separation,
                cross_mating_rate,
                persisted_samples,
                hint: matched_hint,
                tick,
                ..
            } => {
                assert_eq!(*parent, SpeciesId(1));
                assert_eq!(*children, [SpeciesId(1), SpeciesId(2)]);
                assert_eq!(*separation, SeparationKind::Phenotypic);
                assert!((*cross_mating_rate - 0.0).abs() < f32::EPSILON);
                assert_eq!(*persisted_samples, 3);
                assert_eq!(*matched_hint, Some(HintId(42)));
                assert_eq!(*tick, Tick(30));
            }
            _ => panic!("expected Speciation event"),
        }

        assert_eq!(out3.verdicts.len(), 1);
        assert_eq!(out3.verdicts[0], HintVerdict::Confirmed(*eid));
    }

    #[test]
    fn speciation_preserves_sample_counts_at_representation_boundaries() {
        let mut sample_counts = vec![2, 3, usize::MAX];
        if let Ok(above_u32) = usize::try_from(u64::from(u32::MAX) + 1) {
            sample_counts.push(above_u32);
        }
        let ancestry = build_test_ancestry(&[1, 2, 3, 4], None);
        let births = [make_birth(5, Some(1), Some(2))];
        let split_key = CandidateSplitKey {
            parent: SpeciesId(1),
            child_a: SpeciesId(1),
            child_b: SpeciesId(2),
        };

        for expected_count in sample_counts {
            let mut table = SpeciesTable {
                tick: Tick(10),
                species: vec![make_test_species(1, &[1, 2]), make_test_species(2, &[3, 4])],
                ..SpeciesTable::default()
            };
            let params = PhyloEventParams {
                persistence_samples: expected_count,
                min_species_size: 2,
                min_two_parent_births: 1,
                ..PhyloEventParams::default()
            };
            let mut state = PhyloEngineState::default();
            state.pending_splits.insert(
                split_key,
                SplitTracking {
                    first_seen_tick: Tick(0),
                    samples_held: expected_count - 2,
                    last_cross_mating_rate: 0.0,
                    last_sample_sizes: (2, 2),
                    matched_hint: None,
                    last_rejection_reason: None,
                    separation_kind: SeparationKind::Phenotypic,
                },
            );

            let before = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);
            assert!(
                before.events.is_empty(),
                "one more sample is still required"
            );
            assert_eq!(
                state.pending_splits[&split_key].samples_held,
                expected_count - 1
            );
            table.tick = Tick(20);
            let output = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);
            assert_eq!(output.events.len(), 1);
            assert!(!state.pending_splits.contains_key(&split_key));
            let expected_u64 = u64::try_from(expected_count).expect("sample count fits u64");
            let observed = match &output.events[0].1 {
                PhyloEvent::Speciation {
                    persisted_samples, ..
                } => Some(u64::try_from(*persisted_samples).expect("reported count fits u64")),
                PhyloEvent::Extinction { .. } | PhyloEvent::Radiation { .. } => None,
            };
            assert_eq!(observed, Some(expected_u64));

            let json = serde_json::to_vec(&output).expect("encode phylogeny output as JSON");
            let value: serde_json::Value = serde_json::from_slice(&json).expect("read JSON fields");
            assert_eq!(
                value["events"][0][1]["Speciation"]["persisted_samples"].as_u64(),
                Some(expected_u64)
            );
            let json_output: PhyloEngineOutput =
                serde_json::from_slice(&json).expect("decode phylogeny JSON");
            assert_eq!(json_output, output);
            let binary = postcard::to_allocvec(&output).expect("encode phylogeny postcard");
            let binary_output: PhyloEngineOutput =
                postcard::from_bytes(&binary).expect("decode phylogeny postcard");
            assert_eq!(binary_output, output);
        }
    }

    /// (b) Split that reverts after K-1 samples -> NO event, hint Rejected{Transient}.
    #[test]
    fn test_bd_16g_3_3_b_split_reverting_after_k_minus_one_samples_is_transient() {
        let mut table = SpeciesTable::default();
        table
            .species
            .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));
        table
            .species
            .push(make_test_species(2, &(11..=20).collect::<Vec<_>>()));

        let ancestry = build_test_ancestry(&(1..=20).collect::<Vec<_>>(), None);
        let mut births = Vec::new();
        for idx in 0..10 {
            births.push(make_birth(100 + idx, Some(1), Some(2)));
            births.push(make_birth(200 + idx, Some(11), Some(12)));
        }

        let hint = DetectorHint {
            id: HintId(77),
            tick: Tick(10),
            kind: DetectorHintKind::Bimodality,
            score: 0.90,
            metric: "phenotype_bimodality".to_string(),
            target_species: Some(SpeciesId(1)),
        };

        let params = PhyloEventParams {
            persistence_samples: 3,
            ..Default::default()
        };
        let mut state = PhyloEngineState::default();

        // Sample 1 & 2: holds
        table.tick = Tick(10);
        let _ = step_phylo_events(&table, &ancestry, &births, &[hint], &params, &mut state);
        table.tick = Tick(20);
        let _ = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);

        // Sample 3: species 2 vanished / merged back!
        let mut reverted_table = SpeciesTable::default();
        reverted_table.tick = Tick(30);
        reverted_table
            .species
            .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));

        let out3 = step_phylo_events(
            &reverted_table,
            &ancestry,
            &births,
            &[],
            &params,
            &mut state,
        );
        assert!(
            out3.events.is_empty(),
            "reverted split must not emit Speciation"
        );
        assert_eq!(out3.verdicts.len(), 1);
        match &out3.verdicts[0] {
            HintVerdict::Rejected { reason, evidence } => {
                assert_eq!(*reason, RejectReason::Transient);
                assert_eq!(evidence.hint_id, HintId(77));
            }
            _ => panic!("expected Rejected{{Transient}}"),
        }
    }

    /// (d) Split entirely explained by brain-kind gating -> event emitted with separation=BrainKindGated,
    /// and an assertion that it is NOT reported as Phenotypic.
    #[test]
    fn test_bd_16g_3_3_d_brain_kind_gated_separation_distinguished() {
        let mut table = SpeciesTable::default();
        table
            .species
            .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));
        table
            .species
            .push(make_test_species(2, &(11..=20).collect::<Vec<_>>()));

        // Species 1 members have brain 1, Species 2 members have brain 2
        let mut brain_keys = Vec::new();
        for id in 1..=10 {
            brain_keys.push((id, 1));
        }
        for id in 11..=20 {
            brain_keys.push((id, 2));
        }
        let ancestry = build_test_ancestry(&(1..=20).collect::<Vec<_>>(), Some(&brain_keys));

        let mut births = Vec::new();
        for idx in 0..10 {
            births.push(make_birth(100 + idx, Some(1), Some(2)));
            births.push(make_birth(200 + idx, Some(11), Some(12)));
        }

        let params = PhyloEventParams {
            persistence_samples: 1, // evaluate immediately
            ..Default::default()
        };
        let mut state = PhyloEngineState::default();

        let out = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);
        assert_eq!(out.events.len(), 1);
        let (_, event) = &out.events[0];
        match event {
            PhyloEvent::Speciation { separation, .. } => {
                assert_eq!(*separation, SeparationKind::BrainKindGated);
                assert_ne!(
                    *separation,
                    SeparationKind::Phenotypic,
                    "must NOT report as Phenotypic"
                );
            }
            _ => panic!("expected Speciation event"),
        }
    }

    /// (e) Window containing zero two-parent births -> Rejected{NoAncestralSupport}, not Confirmed
    /// (the empty-denominator test).
    #[test]
    fn test_bd_16g_3_3_e_empty_two_parent_denominator_rejected_no_ancestral_support() {
        let mut table = SpeciesTable::default();
        table
            .species
            .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));
        table
            .species
            .push(make_test_species(2, &(11..=20).collect::<Vec<_>>()));

        let ancestry = build_test_ancestry(&(1..=20).collect::<Vec<_>>(), None);

        // Zero two-parent births (e.g. all asexual births or empty)
        let births: Vec<BirthRecord> = (0..5).map(|i| make_birth(100 + i, Some(1), None)).collect();

        let params = PhyloEventParams {
            persistence_samples: 1,
            min_two_parent_births: 2,
            ..Default::default()
        };
        let mut state = PhyloEngineState::default();

        let out = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);
        assert!(
            out.events.is_empty(),
            "empty denominator must not confirm speciation"
        );

        let split_key = CandidateSplitKey {
            parent: SpeciesId(1),
            child_a: SpeciesId(1),
            child_b: SpeciesId(2),
        };
        let tracking = state
            .pending_splits
            .get(&split_key)
            .expect("tracking exists");
        assert_eq!(
            tracking.last_rejection_reason,
            Some(RejectReason::NoAncestralSupport)
        );
    }

    /// (f) Sub-cluster below min_size -> Rejected{BelowMinSize}.
    #[test]
    fn test_bd_16g_3_3_f_sub_cluster_below_min_size_rejected() {
        let mut table = SpeciesTable::default();
        table.species.push(make_test_species(1, &[1, 2, 3]));
        table.species.push(make_test_species(2, &[4])); // only 1 member!

        let ancestry = build_test_ancestry(&[1, 2, 3, 4], None);
        let births = vec![
            make_birth(100, Some(1), Some(2)),
            make_birth(101, Some(1), Some(3)),
        ];

        let params = PhyloEventParams {
            persistence_samples: 1,
            min_species_size: 2,
            ..Default::default()
        };
        let mut state = PhyloEngineState::default();

        let out = step_phylo_events(&table, &ancestry, &births, &[], &params, &mut state);
        assert!(
            out.events.is_empty(),
            "sub-cluster below min_size must not confirm"
        );

        let split_key = CandidateSplitKey {
            parent: SpeciesId(1),
            child_a: SpeciesId(1),
            child_b: SpeciesId(2),
        };
        let tracking = state
            .pending_splits
            .get(&split_key)
            .expect("tracking exists");
        assert_eq!(
            tracking.last_rejection_reason,
            Some(RejectReason::BelowMinSize)
        );
    }

    /// (g) Species size -> 0 -> exactly ONE Extinction, and no further events ever emitted
    /// for that SpeciesId (idempotence of extinction).
    #[test]
    fn test_bd_16g_3_3_g_species_size_zero_extinction_idempotence() {
        let mut ancestry = build_test_ancestry(&(1..=5).collect::<Vec<_>>(), None);

        // Step 1: species 1 has 5 members alive
        let mut table1 = SpeciesTable::default();
        table1.tick = Tick(10);
        table1
            .species
            .push(make_test_species(1, &(1..=5).collect::<Vec<_>>()));

        let params = PhyloEventParams::default();
        let mut state = PhyloEngineState::default();
        let out1 = step_phylo_events(&table1, &ancestry, &[], &[], &params, &mut state);
        assert!(out1.events.is_empty());

        // Now all members die:
        let _ = ancestry.apply_death(&make_death(1, DeathCause::CombatCarnivore, Tick(20)));
        let _ = ancestry.apply_death(&make_death(2, DeathCause::CombatCarnivore, Tick(22)));
        let _ = ancestry.apply_death(&make_death(3, DeathCause::Starvation, Tick(24)));
        let _ = ancestry.apply_death(&make_death(4, DeathCause::Starvation, Tick(26)));
        let _ = ancestry.apply_death(&make_death(5, DeathCause::Aging, Tick(30))); // last member!

        // Step 2: species 1 has 0 members and is absent from clustering table
        let mut table2 = SpeciesTable::default();
        table2.tick = Tick(35);
        let out2 = step_phylo_events(&table2, &ancestry, &[], &[], &params, &mut state);

        assert_eq!(out2.events.len(), 1, "exactly ONE Extinction event");
        let (_, event) = &out2.events[0];
        match event {
            PhyloEvent::Extinction {
                species,
                last_member,
                peak_size,
                cause_histogram,
                tick,
            } => {
                assert_eq!(*species, SpeciesId(1));
                assert_eq!(*last_member, AgentUid(5));
                assert_eq!(*peak_size, 5);
                assert_eq!(*tick, Tick(35));
                assert_eq!(cause_histogram.get(&DeathCause::CombatCarnivore), Some(&2));
                assert_eq!(cause_histogram.get(&DeathCause::Starvation), Some(&2));
                assert_eq!(cause_histogram.get(&DeathCause::Aging), Some(&1));
            }
            _ => panic!("expected Extinction event"),
        }

        // Step 3: subsequent step with species 1 still absent
        let mut table3 = SpeciesTable::default();
        table3.tick = Tick(40);
        let out3 = step_phylo_events(&table3, &ancestry, &[], &[], &params, &mut state);

        assert!(
            out3.events.is_empty(),
            "extinction must be idempotent (zero new events)"
        );
    }

    /// (h) Species size doubling inside the window -> one Radiation with the correct from/to and score.
    #[test]
    fn test_bd_16g_3_3_h_radiation_doubling_inside_window() {
        let ancestry = build_test_ancestry(&(1..=15).collect::<Vec<_>>(), None);
        let params = PhyloEventParams {
            radiation_min_members: 5,
            radiation_growth_factor: 2.0,
            ..Default::default()
        };
        let mut state = PhyloEngineState::default();

        // Step 1: species 1 has 5 members
        let mut table1 = SpeciesTable::default();
        table1.tick = Tick(10);
        table1
            .species
            .push(make_test_species(1, &(1..=5).collect::<Vec<_>>()));
        let _ = step_phylo_events(&table1, &ancestry, &[], &[], &params, &mut state);

        // Step 2: species 1 doubles to 12 members
        let mut table2 = SpeciesTable::default();
        table2.tick = Tick(20);
        table2
            .species
            .push(make_test_species(1, &(1..=12).collect::<Vec<_>>()));

        let hint = DetectorHint {
            id: HintId(99),
            tick: Tick(15),
            kind: DetectorHintKind::ChangePoint,
            score: 2.4,
            metric: "population_cusum".to_string(),
            target_species: Some(SpeciesId(1)),
        };

        let out2 = step_phylo_events(&table2, &ancestry, &[], &[hint], &params, &mut state);
        assert_eq!(out2.events.len(), 1, "exactly one Radiation event");
        let (eid, event) = &out2.events[0];
        match event {
            PhyloEvent::Radiation {
                species,
                from_size,
                to_size,
                window,
                score,
                hint: matched_hint,
                tick,
            } => {
                assert_eq!(*species, SpeciesId(1));
                assert_eq!(*from_size, 5);
                assert_eq!(*to_size, 12);
                assert_eq!(window.start, Tick(10));
                assert_eq!(window.end, Tick(20));
                assert!((*score - 2.4).abs() < f32::EPSILON);
                assert_eq!(*matched_hint, Some(HintId(99)));
                assert_eq!(*tick, Tick(20));
            }
            _ => panic!("expected Radiation event"),
        }

        assert_eq!(out2.verdicts.len(), 1);
        assert_eq!(out2.verdicts[0], HintVerdict::Confirmed(*eid));
    }

    /// (i) Clustering drops a species that still has living members -> no Extinction,
    /// a warn, and a typed anomaly counter.
    #[test]
    fn test_bd_16g_3_3_i_clustering_drops_species_with_living_members_anomaly() {
        let ancestry = build_test_ancestry(&(1..=5).collect::<Vec<_>>(), None);
        let params = PhyloEventParams::default();
        let mut state = PhyloEngineState::default();

        // Step 1: species 1 has 5 members
        let mut table1 = SpeciesTable::default();
        table1.tick = Tick(10);
        table1
            .species
            .push(make_test_species(1, &(1..=5).collect::<Vec<_>>()));
        let _ = step_phylo_events(&table1, &ancestry, &[], &[], &params, &mut state);

        // Members 1..=5 are still ALIVE in ancestry (no death records)
        // Step 2: clustering table drops species 1!
        let mut table2 = SpeciesTable::default();
        table2.tick = Tick(20);
        let out2 = step_phylo_events(&table2, &ancestry, &[], &[], &params, &mut state);

        // Must NOT emit Extinction! Must record anomaly!
        assert!(
            out2.events.is_empty(),
            "must not emit extinction when members are alive"
        );
        assert_eq!(out2.anomalies, 1);
        assert_eq!(state.anomaly_count, 1);
    }

    /// Determinism: the same seeded fixture twice -> byte-identical event list and verdict list.
    #[test]
    fn test_bd_16g_3_3_determinism_identical_runs_produce_byte_identical_events() {
        let run_fixture = || {
            let ancestry = build_test_ancestry(&(1..=20).collect::<Vec<_>>(), None);
            let mut births = Vec::new();
            for idx in 0..10 {
                births.push(make_birth(100 + idx, Some(1), Some(2)));
                births.push(make_birth(200 + idx, Some(11), Some(12)));
            }

            let params = PhyloEventParams {
                persistence_samples: 2,
                ..Default::default()
            };
            let mut state = PhyloEngineState::default();

            let mut table1 = SpeciesTable::default();
            table1.tick = Tick(10);
            table1
                .species
                .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));
            table1
                .species
                .push(make_test_species(2, &(11..=20).collect::<Vec<_>>()));
            let out1 = step_phylo_events(&table1, &ancestry, &births, &[], &params, &mut state);

            let mut table2 = SpeciesTable::default();
            table2.tick = Tick(20);
            table2
                .species
                .push(make_test_species(1, &(1..=10).collect::<Vec<_>>()));
            table2
                .species
                .push(make_test_species(2, &(11..=20).collect::<Vec<_>>()));
            let out2 = step_phylo_events(&table2, &ancestry, &births, &[], &params, &mut state);

            (out1, out2)
        };

        let run_a = run_fixture();
        let run_b = run_fixture();

        let bytes_a = serde_json::to_vec(&run_a).expect("serialize run a");
        let bytes_b = serde_json::to_vec(&run_b).expect("serialize run b");

        assert_eq!(
            bytes_a, bytes_b,
            "same fixture twice must produce byte-identical serialized events and verdicts"
        );
    }
}
