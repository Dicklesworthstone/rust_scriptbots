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
                        if let Some(indices) = b.get(lin) {
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
        if max <= 0 {
            return 0;
        }
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
                        if let Some(indices) = b.get(lin) {
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
                        if let Some(indices) = b.get(lin) {
                            if !indices.is_empty() {
                                visitor(indices);
                            }
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
    use std::collections::{BTreeMap, HashSet};

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

    fn minimum_image_delta(a: f32, b: f32, extent: f32) -> f32 {
        let direct = (a - b).abs().rem_euclid(extent);
        direct.min(extent - direct)
    }

    fn brute_force_neighbors(
        positions: &[(f32, f32)],
        agent_idx: usize,
        width: f32,
        height: f32,
        radius_sq: f32,
    ) -> BTreeMap<usize, f32> {
        let (ax, ay) = positions[agent_idx];
        positions
            .iter()
            .enumerate()
            .filter_map(|(other_idx, &(ox, oy))| {
                if other_idx == agent_idx {
                    return None;
                }
                let dx = minimum_image_delta(ox, ax, width);
                let dy = minimum_image_delta(oy, ay, height);
                let distance_sq = dx.mul_add(dx, dy * dy);
                (distance_sq <= radius_sq).then_some((other_idx, distance_sq))
            })
            .collect()
    }

    fn queried_neighbors(
        index: &UniformGridIndex,
        agent_idx: usize,
        radius_sq: f32,
    ) -> BTreeMap<usize, f32> {
        let mut delivered = HashSet::new();
        let mut neighbors = BTreeMap::new();
        index.neighbors_within(agent_idx, radius_sq, &mut |other_idx, distance_sq| {
            assert!(
                delivered.insert(other_idx),
                "query delivered agent {other_idx} more than once"
            );
            neighbors.insert(other_idx, distance_sq.into_inner());
        });
        neighbors
    }

    fn bucket_candidates(
        index: &UniformGridIndex,
        agent_idx: usize,
        radius: f32,
    ) -> HashSet<usize> {
        let mut candidates = HashSet::new();
        index.visit_neighbor_buckets(agent_idx, radius, &mut |indices| {
            for &other_idx in indices {
                assert!(
                    candidates.insert(other_idx),
                    "bucket visitor delivered agent {other_idx} more than once"
                );
            }
        });
        candidates
    }

    fn scratch_bucket_candidates(
        index: &UniformGridIndex,
        positions: &[(f32, f32)],
        agent_idx: usize,
        radius: f32,
    ) -> HashSet<usize> {
        let mut scratch_x = Vec::new();
        let mut scratch_y = Vec::new();
        let mut candidates = HashSet::new();
        index.visit_neighbor_bucket_positions_with_scratch(
            agent_idx,
            radius,
            &mut scratch_x,
            &mut scratch_y,
            &mut |xs, ys, indices| {
                assert_eq!(xs.len(), indices.len());
                assert_eq!(ys.len(), indices.len());
                for ((&x, &y), &other_idx) in xs.iter().zip(ys).zip(indices) {
                    let expected = positions[other_idx];
                    assert_eq!(
                        (x.to_bits(), y.to_bits()),
                        (expected.0.to_bits(), expected.1.to_bits())
                    );
                    assert!(
                        candidates.insert(other_idx),
                        "scratch visitor delivered agent {other_idx} more than once"
                    );
                }
            },
        );
        candidates
    }

    fn filter_candidates(
        candidates: &HashSet<usize>,
        positions: &[(f32, f32)],
        agent_idx: usize,
        width: f32,
        height: f32,
        radius_sq: f32,
    ) -> BTreeMap<usize, f32> {
        let (ax, ay) = positions[agent_idx];
        candidates
            .iter()
            .filter_map(|&other_idx| {
                if other_idx == agent_idx {
                    return None;
                }
                let (ox, oy) = positions[other_idx];
                let dx = minimum_image_delta(ox, ax, width);
                let dy = minimum_image_delta(oy, ay, height);
                let distance_sq = dx.mul_add(dx, dy * dy);
                (distance_sq <= radius_sq).then_some((other_idx, distance_sq))
            })
            .collect()
    }

    fn assert_neighbor_maps_close(
        context: &str,
        expected: &BTreeMap<usize, f32>,
        actual: &BTreeMap<usize, f32>,
    ) {
        assert_eq!(
            expected.keys().collect::<Vec<_>>(),
            actual.keys().collect::<Vec<_>>(),
            "neighbor IDs disagree for {context}"
        );
        for (&other_idx, &expected_distance_sq) in expected {
            let actual_distance_sq = actual[&other_idx];
            let tolerance = expected_distance_sq.abs().max(1.0) * 1.0e-5;
            assert!(
                (expected_distance_sq - actual_distance_sq).abs() <= tolerance,
                "distance for neighbor {other_idx} disagrees in {context}: expected {expected_distance_sq}, got {actual_distance_sq}"
            );
        }
    }

    fn assert_all_query_surfaces_match_oracle(
        cell_size: f32,
        width: f32,
        height: f32,
        positions: &[(f32, f32)],
        radii: &[f32],
    ) {
        let index = build_index(cell_size, width, height, positions);
        for agent_idx in 0..positions.len() {
            for &radius in radii {
                let radius_sq = radius * radius;
                let context = format!(
                    "cell={cell_size} world={width}x{height} agent={agent_idx} radius={radius}"
                );
                let expected =
                    brute_force_neighbors(positions, agent_idx, width, height, radius_sq);
                let queried = queried_neighbors(&index, agent_idx, radius_sq);
                assert_neighbor_maps_close(&context, &expected, &queried);

                let candidates = bucket_candidates(&index, agent_idx, radius);
                let scratch_candidates =
                    scratch_bucket_candidates(&index, positions, agent_idx, radius);
                assert_eq!(
                    candidates, scratch_candidates,
                    "bucket visitor surfaces disagree for {context}"
                );
                let filtered =
                    filter_candidates(&candidates, positions, agent_idx, width, height, radius_sq);
                assert_neighbor_maps_close(&context, &expected, &filtered);
                assert_eq!(
                    expected.len(),
                    queried.len(),
                    "query count disagrees with the oracle for {context}"
                );
                assert_eq!(
                    queried.len(),
                    filtered.len(),
                    "query and filtered visitor counts disagree for {context}"
                );
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn deterministic_cell_positions(
        cell_size: f32,
        width: f32,
        height: f32,
        cells_x: usize,
        cells_y: usize,
    ) -> Vec<(f32, f32)> {
        let mut positions = Vec::with_capacity(cells_x * cells_y);
        for cell_y in 0..cells_y {
            for cell_x in 0..cells_x {
                let x_start = cell_x as f32 * cell_size;
                let y_start = cell_y as f32 * cell_size;
                let x_end = ((cell_x + 1) as f32 * cell_size).min(width);
                let y_end = ((cell_y + 1) as f32 * cell_size).min(height);
                let x_fraction = if (cell_x + cell_y) % 2 == 0 {
                    0.17
                } else {
                    0.83
                };
                let y_fraction = if (cell_x * 3 + cell_y) % 2 == 0 {
                    0.79
                } else {
                    0.23
                };
                positions.push((
                    (x_end - x_start)
                        .mul_add(x_fraction, x_start)
                        .rem_euclid(width),
                    (y_end - y_start)
                        .mul_add(y_fraction, y_start)
                        .rem_euclid(height),
                ));
            }
        }
        positions
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn deterministic_grid_sweep_matches_minimum_image_oracle_and_api_counts() {
        for &cell_size in &[0.75_f32, 3.25] {
            for cells_x in 1_usize..=6 {
                for cells_y in 1_usize..=5 {
                    let width_trim = if cells_x % 2 == 0 { 0.37 } else { 0.0 };
                    let height_trim = if cells_y % 2 == 0 { 0.23 } else { 0.0 };
                    let width = cell_size * (cells_x as f32 - width_trim);
                    let height = cell_size * (cells_y as f32 - height_trim);
                    let positions =
                        deterministic_cell_positions(cell_size, width, height, cells_x, cells_y);
                    let maximum_minimum_image_distance = (width * 0.5).hypot(height * 0.5);
                    let radii = [
                        0.0,
                        cell_size * 0.49,
                        cell_size,
                        cell_size * 1.01,
                        width.min(height) * 0.5,
                        maximum_minimum_image_distance,
                        width.max(height) * 4.0,
                    ];
                    assert_all_query_surfaces_match_oracle(
                        cell_size, width, height, &positions, &radii,
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_grid_query_and_bucket_surfaces_match_the_toroidal_oracle() {
        let cell_size = 0.5;
        let width = 600.0;
        let height = 600.0;
        let positions = [
            (0.1, 0.1),
            (599.9, 0.2),
            (300.0, 300.0),
            (1.0, 599.8),
            (598.5, 598.7),
            (20.0, 20.0),
        ];
        let index = build_index(cell_size, width, height, &positions);
        assert!(matches!(&index.buckets, Buckets::Sparse(_)));

        assert_all_query_surfaces_match_oracle(
            cell_size,
            width,
            height,
            &positions,
            &[0.0, 0.75, 3.0, 25.0],
        );
    }

    #[test]
    fn wrapped_translation_preserves_neighbor_ids_distances_and_counts() {
        let cell_size = 2.5;
        let width = 12.5;
        let height = 10.0;
        let positions = [
            (0.1, 0.2),
            (12.3, 9.8),
            (6.25, 5.0),
            (2.4, 7.6),
            (10.1, 1.7),
            (4.8, 9.9),
        ];
        let radii = [0.3_f32, 2.0, 4.25, 8.0];
        let original = build_index(cell_size, width, height, &positions);

        for &(offset_x, offset_y) in &[(3.75_f32, 6.25), (12.5, 10.0), (-4.5, 2.75)] {
            let translated = positions.map(|(x, y)| {
                (
                    (x + offset_x).rem_euclid(width),
                    (y + offset_y).rem_euclid(height),
                )
            });
            let shifted = build_index(cell_size, width, height, &translated);
            for agent_idx in 0..positions.len() {
                for radius in radii {
                    let radius_sq = radius * radius;
                    let baseline = queried_neighbors(&original, agent_idx, radius_sq);
                    let actual = queried_neighbors(&shifted, agent_idx, radius_sq);
                    let context = format!(
                        "translation=({offset_x},{offset_y}) agent={agent_idx} radius={radius}"
                    );
                    assert_neighbor_maps_close(&context, &baseline, &actual);

                    let candidates = bucket_candidates(&shifted, agent_idx, radius);
                    let filtered = filter_candidates(
                        &candidates,
                        &translated,
                        agent_idx,
                        width,
                        height,
                        radius_sq,
                    );
                    assert_neighbor_maps_close(&context, &actual, &filtered);
                    assert_eq!(
                        baseline.len(),
                        filtered.len(),
                        "count changed for {context}"
                    );
                }
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn oversized_radius_terminates_and_delivers_each_entry_once_in_tiny_worlds() {
        for cells_x in 1_usize..=4 {
            for cells_y in 1_usize..=4 {
                let cell_size = 1.0;
                let width = cells_x as f32;
                let height = cells_y as f32;
                let positions =
                    deterministic_cell_positions(cell_size, width, height, cells_x, cells_y);
                let index = build_index(cell_size, width, height, &positions);
                for agent_idx in 0..positions.len() {
                    let queried = queried_neighbors(&index, agent_idx, f32::MAX);
                    let candidates = bucket_candidates(&index, agent_idx, f32::MAX);
                    let scratch_candidates =
                        scratch_bucket_candidates(&index, &positions, agent_idx, f32::MAX);
                    assert_eq!(queried.len(), positions.len() - 1);
                    assert_eq!(candidates.len(), positions.len());
                    assert_eq!(candidates, scratch_candidates);
                }
            }
        }
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

    #[test]
    fn wrap_zero_max_does_not_panic() {
        assert_eq!(UniformGridIndex::wrap(5, 0), 0);
        assert_eq!(UniformGridIndex::wrap(5, -10), 0);
    }
}
