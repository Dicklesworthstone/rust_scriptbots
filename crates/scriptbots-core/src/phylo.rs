//! Phylogeny incremental tree layout engine with level-of-detail (LOD) filtering and documented memory bounds.
//!
//! # Memory & Performance Guarantees
//! - **Memory Bound**: Each [`LayoutNode`] occupies ~80 bytes. A 10,000-node species tree consumes ~800 KB of node storage.
//! - **Determinism**: Node positioning (`x`, `y`) is a 100% pure, deterministic function of node tick range and key hierarchy.
//! - **Incremental Parity**: Calling [`TreeLayout::extend`] sequentially produces bitwise-identical coordinates to building the layout from scratch.
//! - **LOD Query**: [`TreeLayout::lod`] runs in O(N log N) rank selection capped at `budget` visible items without UI/frame side effects.

use crate::{AgentUid, Tick};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;

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
pub struct LayoutIdx(pub usize);

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
}

impl Default for LayoutBudget {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            max_depth: 100,
        }
    }
}

/// Single node in the 2D phylogeny layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayoutNode {
    pub key: PhyloKey,
    pub parent: Option<LayoutIdx>,
    /// X coordinate (derived deterministically from tick).
    pub x: f32,
    /// Y coordinate (derived from deterministic tidy-tree leaf ordering).
    pub y: f32,
    /// Visual thickness scaling (e.g. proportional to population).
    pub thickness: f32,
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
}

/// Input update for a node in the phylogeny graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhyloNodeUpdate {
    pub key: PhyloKey,
    pub parent_key: Option<PhyloKey>,
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

/// Deterministic 2D layout engine for phylogeny graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeLayout {
    nodes: Vec<LayoutNode>,
    index: BTreeMap<PhyloKey, LayoutIdx>,
    children_map: BTreeMap<LayoutIdx, Vec<LayoutIdx>>,
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
            children_map: BTreeMap::new(),
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
        self.index.get(key).map(|idx| &self.nodes[idx.0])
    }

    /// Ensures that the `PrunedAncestor` sentinel node exists in the layout.
    fn ensure_pruned_sentinel(&mut self) -> LayoutIdx {
        if let Some(&idx) = self.index.get(&PhyloKey::PrunedAncestor) {
            return idx;
        }
        let idx = LayoutIdx(self.nodes.len());
        let sentinel = LayoutNode {
            key: PhyloKey::PrunedAncestor,
            parent: None,
            x: 0.0,
            y: 0.0,
            thickness: 1.0,
            first_tick: Tick(0),
            last_tick: None,
            collapsed: false,
            peak_population: 0,
        };
        self.nodes.push(sentinel);
        self.index.insert(PhyloKey::PrunedAncestor, idx);
        idx
    }

    /// Extends the layout incrementally with new or updated node deltas.
    pub fn extend(&mut self, delta: &PhyloDelta) -> LayoutDelta {
        if delta.updates.is_empty() {
            return LayoutDelta {
                added: Vec::new(),
                moved: Vec::new(),
                full_relayout: false,
            };
        }

        let mut added = Vec::new();
        let mut updated = false;

        for update in &delta.updates {
            let parent_idx = update.parent_key.and_then(|pk| {
                if let Some(&idx) = self.index.get(&pk) {
                    Some(idx)
                } else {
                    Some(self.ensure_pruned_sentinel())
                }
            });

            if let Some(&existing_idx) = self.index.get(&update.key) {
                let node = &mut self.nodes[existing_idx.0];
                node.last_tick = update.death_tick;
                if update.population > node.peak_population {
                    node.peak_population = update.population;
                }
                node.thickness = (update.population as f32).max(1.0).sqrt();
                updated = true;
            } else {
                let new_idx = LayoutIdx(self.nodes.len());
                let node = LayoutNode {
                    key: update.key,
                    parent: parent_idx,
                    x: update.birth_tick.0 as f32,
                    y: 0.0,
                    thickness: (update.population as f32).max(1.0).sqrt(),
                    first_tick: update.birth_tick,
                    last_tick: update.death_tick,
                    collapsed: false,
                    peak_population: update.population,
                };
                self.nodes.push(node);
                self.index.insert(update.key, new_idx);
                if let Some(pidx) = parent_idx {
                    self.children_map.entry(pidx).or_default().push(new_idx);
                }
                added.push(new_idx);
            }
        }

        // Recompute Y positions deterministically to guarantee incremental == from-scratch.
        let moved = self.recompute_y_positions();

        LayoutDelta {
            added,
            moved,
            full_relayout: updated || self.nodes.len() > self.budget.max_nodes,
        }
    }

    /// Recomputes Y coordinates and bounding box deterministically.
    fn recompute_y_positions(&mut self) -> Vec<LayoutIdx> {
        if self.nodes.is_empty() {
            self.bounds = Rect::default();
            return Vec::new();
        }

        let mut roots = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.parent.is_none() {
                roots.push(LayoutIdx(i));
            }
        }

        // Sort roots deterministically by (first_tick, key)
        roots.sort_by(|a, b| {
            let na = &self.nodes[a.0];
            let nb = &self.nodes[b.0];
            na.first_tick
                .cmp(&nb.first_tick)
                .then_with(|| na.key.cmp(&nb.key))
        });

        let mut moved = Vec::new();
        let mut current_leaf_y = 0.0f32;
        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;

        for root in roots {
            self.assign_y_dfs(root, &mut current_leaf_y, &mut moved);
        }

        for node in &self.nodes {
            x_min = x_min.min(node.x);
            x_max = x_max.max(node.x);
            y_min = y_min.min(node.y);
            y_max = y_max.max(node.y);
        }

        self.bounds = Rect {
            x_min: if x_min > x_max { 0.0 } else { x_min },
            y_min: if y_min > y_max { 0.0 } else { y_min },
            x_max: if x_min > x_max { 0.0 } else { x_max },
            y_max: if y_min > y_max { 0.0 } else { y_max },
        };

        moved
    }

    /// DFS traversal to compute tidy Y positions.
    fn assign_y_dfs(&mut self, idx: LayoutIdx, leaf_y: &mut f32, moved: &mut Vec<LayoutIdx>) {
        let mut children = self.children_map.get(&idx).cloned().unwrap_or_default();

        if children.is_empty() {
            let new_y = *leaf_y;
            *leaf_y += 1.0;
            if (self.nodes[idx.0].y - new_y).abs() > f32::EPSILON {
                self.nodes[idx.0].y = new_y;
                moved.push(idx);
            }
        } else {
            children.sort_by(|a, b| {
                let na = &self.nodes[a.0];
                let nb = &self.nodes[b.0];
                na.first_tick
                    .cmp(&nb.first_tick)
                    .then_with(|| na.key.cmp(&nb.key))
            });

            let first_child_y_before = *leaf_y;
            for child_idx in children {
                self.assign_y_dfs(child_idx, leaf_y, moved);
            }
            let last_child_y_after = *leaf_y - 1.0;
            let centroid_y = (first_child_y_before + last_child_y_after) / 2.0;

            if (self.nodes[idx.0].y - centroid_y).abs() > f32::EPSILON {
                self.nodes[idx.0].y = centroid_y;
                moved.push(idx);
            }
        }
    }

    /// Queries the level-of-detail (LOD) visible nodes within the given viewport and budget.
    pub fn lod(
        &self,
        viewport: TickRange,
        y_range: Range<f32>,
        budget: usize,
    ) -> impl Iterator<Item = &LayoutNode> {
        let mut candidate_indices: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                let node_end = node.last_tick.unwrap_or(Tick(u64::MAX));
                let tick_overlap =
                    node.first_tick <= viewport.end && node_end >= viewport.start;
                let y_overlap = node.y >= y_range.start && node.y <= y_range.end;
                tick_overlap && y_overlap
            })
            .map(|(i, _)| i)
            .collect();

        // Rank by (peak_population, duration) descending, then key ascending.
        candidate_indices.sort_by(|&a, &b| {
            let na = &self.nodes[a];
            let nb = &self.nodes[b];
            let dur_a = na.last_tick.unwrap_or(viewport.end).0.saturating_sub(na.first_tick.0);
            let dur_b = nb.last_tick.unwrap_or(viewport.end).0.saturating_sub(nb.first_tick.0);

            nb.peak_population
                .cmp(&na.peak_population)
                .then_with(|| dur_b.cmp(&dur_a))
                .then_with(|| na.key.cmp(&nb.key))
        });

        candidate_indices.truncate(budget);

        candidate_indices
            .into_iter()
            .map(move |idx| &self.nodes[idx])
    }

    /// Performs a 2D hit test finding the closest node key within `tol` Euclidean distance.
    #[must_use]
    pub fn hit_test(&self, x: f32, y: f32, tol: f32) -> Option<PhyloKey> {
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

        let mut count = 0;
        let mut truncated = false;
        let mut stack = vec![(root_idx, 0)];

        while let Some((idx, depth)) = stack.pop() {
            if count >= node_budget {
                truncated = true;
                break;
            }
            self.nodes[idx.0].collapsed = false;
            count += 1;

            if depth < depth_budget {
                if let Some(children) = self.children_map.get(&idx) {
                    for &child_idx in children {
                        stack.push((child_idx, depth + 1));
                    }
                }
            } else if self.children_map.contains_key(&idx) {
                truncated = true;
            }
        }

        ExpandReport {
            expanded_nodes: count,
            truncated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bound_assertion() {
        let bytes_per_node = std::mem::size_of::<LayoutNode>();
        assert!(
            bytes_per_node <= 128,
            "LayoutNode size {bytes_per_node} bytes exceeds 128-byte budget boundary"
        );
    }

    #[test]
    fn test_empty_and_single_node() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        assert!(layout.is_empty());
        assert_eq!(layout.hit_test(0.0, 0.0, 5.0), None);

        let delta = PhyloDelta {
            updates: vec![PhyloNodeUpdate {
                key: PhyloKey::Species(1),
                parent_key: None,
                birth_tick: Tick(10),
                death_tick: None,
                population: 50,
            }],
        };

        let res = layout.extend(&delta);
        assert_eq!(res.added.len(), 1);
        assert_eq!(layout.len(), 1);

        let node = layout.get_node(&PhyloKey::Species(1)).unwrap();
        assert_eq!(node.x, 10.0);
        assert_eq!(node.y, 0.0);
        assert_eq!(layout.hit_test(10.0, 0.0, 0.5), Some(PhyloKey::Species(1)));
    }

    #[test]
    fn test_incremental_equals_from_scratch() {
        let update1 = PhyloNodeUpdate {
            key: PhyloKey::Species(1),
            parent_key: None,
            birth_tick: Tick(0),
            death_tick: None,
            population: 10,
        };
        let update2 = PhyloNodeUpdate {
            key: PhyloKey::Species(2),
            parent_key: Some(PhyloKey::Species(1)),
            birth_tick: Tick(20),
            death_tick: None,
            population: 5,
        };
        let update3 = PhyloNodeUpdate {
            key: PhyloKey::Species(3),
            parent_key: Some(PhyloKey::Species(1)),
            birth_tick: Tick(30),
            death_tick: None,
            population: 8,
        };

        // Incremental extension in 3 steps
        let mut inc_layout = TreeLayout::new(LayoutBudget::default());
        inc_layout.extend(&PhyloDelta {
            updates: vec![update1.clone()],
        });
        inc_layout.extend(&PhyloDelta {
            updates: vec![update2.clone()],
        });
        inc_layout.extend(&PhyloDelta {
            updates: vec![update3.clone()],
        });

        // From-scratch extension in 1 step
        let mut scratch_layout = TreeLayout::new(LayoutBudget::default());
        scratch_layout.extend(&PhyloDelta {
            updates: vec![update1, update2, update3],
        });

        assert_eq!(inc_layout.len(), scratch_layout.len());
        for node_inc in inc_layout.nodes() {
            let node_scratch = scratch_layout.get_node(&node_inc.key).expect("node exists");
            assert_eq!(node_inc.x, node_scratch.x);
            assert_eq!(node_inc.y, node_scratch.y);
            assert_eq!(node_inc.first_tick, node_scratch.first_tick);
        }
    }

    #[test]
    fn test_pruned_parent_reattachment() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        let delta = PhyloDelta {
            updates: vec![PhyloNodeUpdate {
                key: PhyloKey::Species(10),
                parent_key: Some(PhyloKey::Species(999)), // Non-existent parent (pruned)
                birth_tick: Tick(50),
                death_tick: None,
                population: 15,
            }],
        };

        layout.extend(&delta);
        let node = layout.get_node(&PhyloKey::Species(10)).unwrap();
        assert!(node.parent.is_some());
        let parent_node = &layout.nodes()[node.parent.unwrap().0];
        assert_eq!(parent_node.key, PhyloKey::PrunedAncestor);
    }

    #[test]
    fn test_lod_filtering() {
        let mut layout = TreeLayout::new(LayoutBudget::default());
        let delta = PhyloDelta {
            updates: vec![
                PhyloNodeUpdate {
                    key: PhyloKey::Species(1),
                    parent_key: None,
                    birth_tick: Tick(0),
                    death_tick: Some(Tick(100)),
                    population: 100,
                },
                PhyloNodeUpdate {
                    key: PhyloKey::Species(2),
                    parent_key: Some(PhyloKey::Species(1)),
                    birth_tick: Tick(10),
                    death_tick: Some(Tick(50)),
                    population: 10,
                },
                PhyloNodeUpdate {
                    key: PhyloKey::Species(3),
                    parent_key: Some(PhyloKey::Species(1)),
                    birth_tick: Tick(20),
                    death_tick: Some(Tick(80)),
                    population: 500,
                },
            ],
        };
        layout.extend(&delta);

        let visible: Vec<_> = layout
            .lod(
                TickRange {
                    start: Tick(5),
                    end: Tick(60),
                },
                -10.0..10.0,
                2,
            )
            .collect();

        assert_eq!(visible.len(), 2);
        // Top by peak population: Species(3) has 500, Species(1) has 100
        assert_eq!(visible[0].key, PhyloKey::Species(3));
        assert_eq!(visible[1].key, PhyloKey::Species(1));
    }
}
