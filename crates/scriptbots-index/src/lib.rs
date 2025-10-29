//! Spatial indexing abstractions for agent neighborhood queries.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformGridIndex {
    /// Edge length of each grid cell used for bucketing agents.
    pub cell_size: f32,
    #[serde(skip)]
    width: f32,
    #[serde(skip)]
    height: f32,
    #[serde(skip)]
    inv_cell_size: f32,
    #[serde(skip)]
    cells_x: i32,
    #[serde(skip)]
    cells_y: i32,
    #[serde(skip)]
    buckets: Buckets,
    #[serde(skip)]
    agent_cells: Vec<(i32, i32)>,
    #[serde(skip)]
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

    /// Visit candidate neighbor buckets and provide SoA x[]/y[] slices using caller scratch buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn visit_neighbor_bucket_positions_with_scratch(
        &self,
        agent_idx: usize,
        radius: f32,
        scratch_x: &mut Vec<f32>,
        scratch_y: &mut Vec<f32>,
        visitor: &mut dyn FnMut(&[f32], &[f32], &[usize]),
    ) {
        if agent_idx >= self.positions.len() || radius < 0.0 {
            return;
        }
        let (cell_x, cell_y) = self.agent_cells[agent_idx];
        let cell_radius = Self::discretize_positive(radius * self.inv_cell_size);

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                let nx = Self::wrap(cell_x + dx, self.cells_x);
                let ny = Self::wrap(cell_y + dy, self.cells_y);
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
                        for &other_idx in indices.iter() {
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
                            for &other_idx in indices.iter() {
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

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                let nx = Self::wrap(cell_x + dx, self.cells_x);
                let ny = Self::wrap(cell_y + dy, self.cells_y);
                match &self.buckets {
                    Buckets::Dense(b) => {
                        let lin = self.linear_index(nx, ny);
                        let indices = &b[lin];
                        for &other_idx in indices {
                            if other_idx == agent_idx {
                                continue;
                            }
                            let (ox, oy) = self.positions[other_idx];
                            let dx = ox - ax;
                            let dy = oy - ay;
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
                                let dx = ox - ax;
                                let dy = oy - ay;
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

        for dx in -cell_radius..=cell_radius {
            for dy in -cell_radius..=cell_radius {
                let nx = Self::wrap(cell_x + dx, self.cells_x);
                let ny = Self::wrap(cell_y + dy, self.cells_y);
                match &self.buckets {
                    Buckets::Dense(b) => {
                        let lin = self.linear_index(nx, ny);
                        let indices = &b[lin];
                        if !indices.is_empty() {
                            visitor(indices);
                        }
                    }
                    Buckets::Sparse(m) => {
                        if let Some(indices) = m.get(&(nx, ny)) {
                            if !indices.is_empty() {
                                visitor(indices);
                            }
                        }
                    }
                }
            }
        }
    }
}
