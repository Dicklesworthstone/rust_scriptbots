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

use crate::{AgentUid, Tick};
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
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
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
    pub start: Tick,
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
    pub key: PhyloKey,
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
    pub first_tick: Tick,
    pub last_tick: Option<Tick>,
    pub collapsed: bool,
    pub peak_population: u32,
}

/// Report emitted after incremental layout extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutDelta {
    pub added: Vec<LayoutIdx>,
    pub moved: Vec<LayoutIdx>,
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
    ConflictingDuplicate { key: PhyloKey },
    /// An existing node's immutable identity/topology fields changed.
    ImmutableConflict { key: PhyloKey },
    /// A node declared itself, or one of its descendants, as parent.
    Cycle { key: PhyloKey, parent: PhyloKey },
    /// A known parent was born after its child.
    ParentBornAfterChild { child: PhyloKey, parent: PhyloKey },
    /// Death precedes birth.
    InvalidLifetime { key: PhyloKey },
    /// Retaining another node would cross the configured node/byte ceiling.
    BudgetExceeded { key: PhyloKey },
}

/// Input update for a node in the phylogeny graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhyloNodeUpdate {
    pub key: PhyloKey,
    pub parent: ParentRef,
    pub founder_uid: AgentUid,
    pub birth_tick: Tick,
    pub death_tick: Option<Tick>,
    pub population: u32,
}

/// Batch of node updates to incorporate into the layout.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhyloDelta {
    pub updates: Vec<PhyloNodeUpdate>,
}

/// Report returned when expanding a collapsed clade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpandReport {
    pub expanded_nodes: usize,
    pub truncated: bool,
}

/// Allocation-aware retained-memory estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReport {
    pub inline_bytes: usize,
    pub node_capacity_bytes: usize,
    pub aggregate_capacity_bytes: usize,
    pub topology_capacity_bytes: usize,
    pub child_capacity_bytes: usize,
    pub key_index_bytes: usize,
    pub pending_index_bytes: usize,
    pub lod_index_bytes: usize,
    pub total_retained_bytes: usize,
}

/// Indexed LOD result plus inspectable query work.
#[derive(Debug)]
pub struct LodReport<'a> {
    pub nodes: Vec<&'a LayoutNode>,
    pub index_nodes_visited: usize,
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
            debug_assert!(inserted_key != root_key);
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

    fn retained_bytes(&self) -> usize {
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

fn deterministic_priority(key: PhyloKey) -> u64 {
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
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the layout contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the bounding box of the entire layout.
    #[must_use]
    pub fn bounds(&self) -> Rect {
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
                    if self.lod_item_matches(indexed, viewport, y_start, y_end) {
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

    fn lod_item_matches(
        &self,
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
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= tol_sq {
                match best {
                    Some((best_dist, best_key)) => {
                        if dist_sq < best_dist || (dist_sq == best_dist && node.key < best_key) {
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
        let root_idx = match self.index.get(&root) {
            Some(&idx) => idx,
            None => {
                return ExpandReport {
                    expanded_nodes: 0,
                    truncated: false,
                };
            }
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

fn update_anchor(key: PhyloKey, founder_uid: AgentUid) -> u64 {
    match key {
        PhyloKey::PrunedAncestor => 0,
        PhyloKey::Species(_) => founder_uid.get().saturating_add(1),
        PhyloKey::Agent(uid) => uid.get().saturating_add(1),
    }
}

fn node_anchor(node: &LayoutNode) -> u64 {
    update_anchor(node.key, node.founder_uid)
}

#[allow(clippy::cast_precision_loss)]
fn tick_x(tick: Tick) -> f32 {
    tick.0 as f32
}

#[allow(clippy::cast_precision_loss)]
fn population_thickness(population: u32) -> f32 {
    (population.max(1) as f32).sqrt()
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
}
