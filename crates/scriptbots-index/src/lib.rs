//! Spatial indexing abstractions for agent neighborhood queries.

use ordered_float::OrderedFloat;
use std::collections::HashMap;
use thiserror::Error;

type BucketVisitor<'a> = dyn FnMut(&[f32], &[f32], &[usize]) + 'a;

/// Errors emitted by spatial index implementations.
#[derive(Debug, Error)]
pub enum IndexError {
    /// Indicates configuration values that cannot be used (e.g., non-positive cell size).
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
}

/// Common behaviour exposed by neighborhood indices.
pub trait NeighborhoodIndex {
    /// Rebuild internal structures from agent positions.
    fn rebuild(&mut self, positions: &[(f32, f32)]) -> Result<(), IndexError>;

    /// Visit neighbors of `agent_idx` within the provided squared radius.
    fn neighbors_within(
        &self,
        agent_idx: usize,
        radius_sq: f32,
        visitor: &mut dyn FnMut(usize, OrderedFloat<f32>),
    );

    /// Visit candidate neighbor bucket slices around `agent_idx` spanning cells that intersect `radius`.
    /// This does not perform distance checks; callers should filter by distance as needed.
    fn visit_neighbor_buckets(
        &self,
        agent_idx: usize,
        radius: f32,
        visitor: &mut dyn FnMut(&[usize]),
    );
}

/// Baseline uniform grid index backing neighbor queries.
#[derive(Debug, Clone)]
pub struct UniformGridIndex {
    /// Edge length of each grid cell used for bucketing agents.
    pub cell_size: f32,
    width: f32,
    height: f32,
    inv_cell_size: f32,
    cells_x: i32,
    cells_y: i32,
    buckets: Buckets,
    agent_cells: Vec<(i32, i32)>,
    positions: Vec<(f32, f32)>,
}

#[derive(Debug, Clone)]
enum Buckets {
    Dense(Vec<Vec<usize>>),
    Sparse(HashMap<(i32, i32), Vec<usize>>),
}

impl Default for Buckets {
    fn default() -> Self {
        Self::Sparse(HashMap::new())
    }
}

const DENSE_BUCKET_MAX_CELLS: usize = 1_000_000; // guard against excessive memory use

impl UniformGridIndex {
    /// Create a new uniform grid with the provided cell size and world dimensions.
    #[must_use]
    pub fn new(cell_size: f32, width: f32, height: f32) -> Self {
        let inv_cell_size = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
        let cells_x = if cell_size > 0.0 {
            Self::cells_for_dimension(width, cell_size)
        } else {
            1
        };
        let cells_y = if cell_size > 0.0 {
            Self::cells_for_dimension(height, cell_size)
        } else {
            1
        };
        Self {
            cell_size,
            width,
            height,
            inv_cell_size,
            cells_x,
            cells_y,
            buckets: Buckets::Sparse(HashMap::new()),
            agent_cells: Vec::new(),
            positions: Vec::new(),
        }
    }

    /// Visit candidate neighbor buckets and provide structure-of-arrays (`x`, `y`) slices using caller scratch buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn visit_neighbor_bucket_positions_with_scratch(
        &self,
        agent_idx: usize,
        radius: f32,
        scratch_x: &mut Vec<f32>,
        scratch_y: &mut Vec<f32>,
        visitor: &mut BucketVisitor<'_>,
    ) {
        if agent_idx >= self.positions.len() || radius < 0.0 {
            return;
        }
        let (cell_x, cell_y) = self.agent_cells[agent_idx];
        let cell_radius = Self::discretize_positive(radius * self.inv_cell_size);
        let span_x = Self::wrapped_span(cell_radius, self.cells_x);
        let span_y = Self::wrapped_span(cell_radius, self.cells_y);

        for step_x in 0..span_x {
            for step_y in 0..span_y {
                let nx = Self::wrap(cell_x - cell_radius + step_x, self.cells_x);
                let ny = Self::wrap(cell_y - cell_radius + step_y, self.cells_y);
                match &self.buckets {
                    Buckets::Dense(b) => {
                        let lin = self.linear_index(nx, ny);
                        let indices = &b[lin];
                        if indices.is_empty() {
                            continue;
                        }
                        scratch_x.clear();
                        scratch_y.clear();
                        scratch_x.reserve(indices.len());
                        scratch_y.reserve(indices.len());
                        for &other_idx in indices {
                            let (x, y) = self.positions[other_idx];
                            scratch_x.push(x);
                            scratch_y.push(y);
                        }
                        visitor(scratch_x.as_slice(), scratch_y.as_slice(), indices);
                    }
                    Buckets::Sparse(m) => {
                        if let Some(indices) = m.get(&(nx, ny)) {
                            if indices.is_empty() {
                                continue;
                            }
                            scratch_x.clear();
                            scratch_y.clear();
                            scratch_x.reserve(indices.len());
                            scratch_y.reserve(indices.len());
                            for &other_idx in indices {
                                let (x, y) = self.positions[other_idx];
                                scratch_x.push(x);
                                scratch_y.push(y);
                            }
                            visitor(scratch_x.as_slice(), scratch_y.as_slice(), indices);
                        }
                    }
                }
            }
        }
    }

    #[inline]
    const fn wrap(value: i32, max: i32) -> i32 {
        ((value % max) + max) % max
    }

    /// Number of cells to scan along one axis so each wrapped cell is visited at most once.
    #[inline]
    const fn wrapped_span(cell_radius: i32, cells: i32) -> i32 {
        let span = cell_radius.saturating_mul(2).saturating_add(1);
        if span < cells { span } else { cells }
    }

    /// Minimum-image delta between two coordinates on a toroidal axis of the given extent.
    #[inline]
    const fn toroidal_delta(a: f32, b: f32, extent: f32) -> f32 {
        let mut delta = a - b;
        if delta > extent * 0.5 {
            delta -= extent;
        } else if delta < -extent * 0.5 {
            delta += extent;
        }
        delta
    }

    #[inline]
    fn cell_from_point(&self, x: f32, y: f32) -> (i32, i32) {
        let cx = Self::wrap(Self::discretize_cell(x * self.inv_cell_size), self.cells_x);
        let cy = Self::wrap(Self::discretize_cell(y * self.inv_cell_size), self.cells_y);
        (cx, cy)
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    const fn linear_index(&self, cx: i32, cy: i32) -> usize {
        // wrap() guarantees 0 <= cx < cells_x and 0 <= cy < cells_y
        (cy as usize) * (self.cells_x as usize) + (cx as usize)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn cells_for_dimension(dimension: f32, cell_size: f32) -> i32 {
        let raw = (dimension / cell_size).ceil().max(1.0);
        raw.min(i32::MAX as f32) as i32
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn discretize_cell(value: f32) -> i32 {
        let floored = value.floor();
        floored.max(i32::MIN as f32).min(i32::MAX as f32) as i32
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn discretize_positive(value: f32) -> i32 {
        value.ceil().max(0.0).min(i32::MAX as f32) as i32
    }
}

impl Default for UniformGridIndex {
    fn default() -> Self {
        Self::new(50.0, 1_000.0, 1_000.0)
    }
}

impl NeighborhoodIndex for UniformGridIndex {
    fn rebuild(&mut self, positions: &[(f32, f32)]) -> Result<(), IndexError> {
        if self.cell_size <= 0.0 {
            return Err(IndexError::InvalidConfig("cell_size must be positive"));
        }
        if self.width <= 0.0 || self.height <= 0.0 {
            return Err(IndexError::InvalidConfig(
                "world dimensions must be positive",
            ));
        }
        self.positions.clear();
        self.positions.extend_from_slice(positions);
        self.agent_cells.resize(positions.len(), (0, 0));

        // Decide dense vs sparse layout based on total cell count.
        let total_cells_u64 =
            i128::from(i64::from(self.cells_x)) * i128::from(i64::from(self.cells_y));
        let total_cells: Option<usize> = if total_cells_u64 >= 0 {
            usize::try_from(total_cells_u64).ok()
        } else {
            None
        };

        if let Some(cell_count) = total_cells.filter(|&c| c <= DENSE_BUCKET_MAX_CELLS) {
            // Dense path: two-pass build for precise capacity reservations
            let mut counts: Vec<usize> = vec![0; cell_count];
            for (idx, &(x, y)) in positions.iter().enumerate() {
                let (cx, cy) = self.cell_from_point(x, y);
                self.agent_cells[idx] = (cx, cy);
                let lin = self.linear_index(cx, cy);
                counts[lin] += 1;
            }

            let mut dense: Vec<Vec<usize>> = counts.into_iter().map(Vec::with_capacity).collect();

            for (idx, &(cx, cy)) in self.agent_cells.iter().enumerate() {
                let lin = self.linear_index(cx, cy);
                dense[lin].push(idx);
            }
            self.buckets = Buckets::Dense(dense);
        } else {
            // Sparse path: fallback HashMap to avoid huge allocations
            let mut map: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
            map.reserve(positions.len());
            for (idx, &(x, y)) in positions.iter().enumerate() {
                let key = self.cell_from_point(x, y);
                self.agent_cells[idx] = key;
                map.entry(key).or_default().push(idx);
            }
            self.buckets = Buckets::Sparse(map);
        }
        Ok(())
    }

    fn neighbors_within(
        &self,
        agent_idx: usize,
        radius_sq: f32,
        visitor: &mut dyn FnMut(usize, OrderedFloat<f32>),
    ) {
        if agent_idx >= self.positions.len() || radius_sq < 0.0 {
            return;
        }
        let (ax, ay) = self.positions[agent_idx];
        let (cell_x, cell_y) = self.agent_cells[agent_idx];
        let radius = radius_sq.sqrt();
        let cell_radius = Self::discretize_positive(radius * self.inv_cell_size);
        let span_x = Self::wrapped_span(cell_radius, self.cells_x);
        let span_y = Self::wrapped_span(cell_radius, self.cells_y);

        for step_x in 0..span_x {
            for step_y in 0..span_y {
                let nx = Self::wrap(cell_x - cell_radius + step_x, self.cells_x);
                let ny = Self::wrap(cell_y - cell_radius + step_y, self.cells_y);
                match &self.buckets {
                    Buckets::Dense(b) => {
                        let lin = self.linear_index(nx, ny);
                        let indices = &b[lin];
                        for &other_idx in indices {
                            if other_idx == agent_idx {
                                continue;
                            }
                            let (ox, oy) = self.positions[other_idx];
                            let dx = Self::toroidal_delta(ox, ax, self.width);
                            let dy = Self::toroidal_delta(oy, ay, self.height);
                            let dist_sq = dx.mul_add(dx, dy * dy);
                            if dist_sq <= radius_sq {
                                visitor(other_idx, OrderedFloat(dist_sq));
                            }
                        }
                    }
                    Buckets::Sparse(m) => {
                        if let Some(indices) = m.get(&(nx, ny)) {
                            for &other_idx in indices {
                                if other_idx == agent_idx {
                                    continue;
                                }
                                let (ox, oy) = self.positions[other_idx];
                                let dx = Self::toroidal_delta(ox, ax, self.width);
                                let dy = Self::toroidal_delta(oy, ay, self.height);
                                let dist_sq = dx.mul_add(dx, dy * dy);
                                if dist_sq <= radius_sq {
                                    visitor(other_idx, OrderedFloat(dist_sq));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn visit_neighbor_buckets(
        &self,
        agent_idx: usize,
        radius: f32,
        visitor: &mut dyn FnMut(&[usize]),
    ) {
        if agent_idx >= self.positions.len() || radius < 0.0 {
            return;
        }
        let (_ax, _ay) = self.positions[agent_idx];
        let (cell_x, cell_y) = self.agent_cells[agent_idx];
        let cell_radius = Self::discretize_positive(radius * self.inv_cell_size);
        let span_x = Self::wrapped_span(cell_radius, self.cells_x);
        let span_y = Self::wrapped_span(cell_radius, self.cells_y);

        for step_x in 0..span_x {
            for step_y in 0..span_y {
                let nx = Self::wrap(cell_x - cell_radius + step_x, self.cells_x);
                let ny = Self::wrap(cell_y - cell_radius + step_y, self.cells_y);
                match &self.buckets {
                    Buckets::Dense(b) => {
                        let lin = self.linear_index(nx, ny);
                        let indices = &b[lin];
                        if !indices.is_empty() {
                            visitor(indices);
                        }
                    }
                    Buckets::Sparse(m) => {
                        if let Some(indices) = m.get(&(nx, ny))
                            && !indices.is_empty()
                        {
                            visitor(indices);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index(
        cell_size: f32,
        width: f32,
        height: f32,
        positions: &[(f32, f32)],
    ) -> UniformGridIndex {
        let mut index = UniformGridIndex::new(cell_size, width, height);
        index.rebuild(positions).expect("rebuild should succeed");
        index
    }

    #[test]
    fn full_world_radius_visits_each_neighbor_exactly_once() {
        // 4x4 grid of cells; the query radius spans the whole world many times over,
        // so an unclamped scan would revisit every wrapped cell repeatedly.
        let positions = [
            (5.0, 5.0),
            (15.0, 5.0),
            (25.0, 15.0),
            (35.0, 35.0),
            (5.0, 25.0),
        ];
        let index = build_index(10.0, 40.0, 40.0, &positions);
        assert!(matches!(index.buckets, Buckets::Dense(_)));

        let mut counts: HashMap<usize, usize> = HashMap::new();
        index.neighbors_within(0, 10_000.0, &mut |idx, _| {
            *counts.entry(idx).or_insert(0) += 1;
        });

        assert_eq!(counts.len(), positions.len() - 1);
        for (idx, count) in counts {
            assert_eq!(count, 1, "neighbor {idx} visited {count} times");
        }
    }

    #[test]
    fn full_world_radius_surfaces_each_bucket_entry_exactly_once() {
        let positions = [
            (5.0, 5.0),
            (15.0, 5.0),
            (25.0, 15.0),
            (35.0, 35.0),
            (5.0, 25.0),
        ];
        let index = build_index(10.0, 40.0, 40.0, &positions);

        let mut bucket_counts: HashMap<usize, usize> = HashMap::new();
        index.visit_neighbor_buckets(0, 100.0, &mut |indices| {
            for &idx in indices {
                *bucket_counts.entry(idx).or_insert(0) += 1;
            }
        });
        assert_eq!(bucket_counts.len(), positions.len());
        for (idx, count) in bucket_counts {
            assert_eq!(count, 1, "agent {idx} surfaced {count} times");
        }

        let mut scratch_x = Vec::new();
        let mut scratch_y = Vec::new();
        let mut soa_counts: HashMap<usize, usize> = HashMap::new();
        index.visit_neighbor_bucket_positions_with_scratch(
            0,
            100.0,
            &mut scratch_x,
            &mut scratch_y,
            &mut |xs, ys, indices| {
                assert_eq!(xs.len(), indices.len());
                assert_eq!(ys.len(), indices.len());
                for &idx in indices {
                    *soa_counts.entry(idx).or_insert(0) += 1;
                }
            },
        );
        assert_eq!(soa_counts.len(), positions.len());
        for (idx, count) in soa_counts {
            assert_eq!(count, 1, "agent {idx} surfaced {count} times");
        }
    }

    #[test]
    fn neighbors_within_finds_pairs_across_the_world_seam() {
        // Dense bucket layout: 10x10 cells in a 100x100 world.
        let positions = [(1.0, 50.0), (99.0, 50.0)];
        let index = build_index(10.0, 100.0, 100.0, &positions);
        assert!(matches!(index.buckets, Buckets::Dense(_)));

        for (agent_idx, expected) in [(0_usize, 1_usize), (1, 0)] {
            let mut found = Vec::new();
            index.neighbors_within(agent_idx, 25.0, &mut |idx, dist_sq| {
                found.push((idx, dist_sq));
            });
            assert_eq!(found.len(), 1, "agent {agent_idx} should see one neighbor");
            let (idx, dist_sq) = found[0];
            assert_eq!(idx, expected);
            assert!(
                (dist_sq.into_inner() - 4.0).abs() < 1e-4,
                "seam distance should be 2 units, got dist_sq {dist_sq}"
            );
        }
    }

    #[test]
    fn neighbors_within_finds_seam_pairs_in_sparse_buckets() {
        // 1200x1200 cells exceeds the dense-bucket cap, forcing the sparse path.
        let positions = [(1.0, 300.0), (599.0, 300.0)];
        let index = build_index(0.5, 600.0, 600.0, &positions);
        assert!(matches!(index.buckets, Buckets::Sparse(_)));

        for (agent_idx, expected) in [(0_usize, 1_usize), (1, 0)] {
            let mut found = Vec::new();
            index.neighbors_within(agent_idx, 25.0, &mut |idx, dist_sq| {
                found.push((idx, dist_sq));
            });
            assert_eq!(found.len(), 1, "agent {agent_idx} should see one neighbor");
            let (idx, dist_sq) = found[0];
            assert_eq!(idx, expected);
            assert!(
                (dist_sq.into_inner() - 4.0).abs() < 1e-4,
                "seam distance should be 2 units, got dist_sq {dist_sq}"
            );
        }
    }
}
