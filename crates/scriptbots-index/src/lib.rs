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
    buckets: HashMap<(i32, i32), Vec<usize>>,
    #[serde(skip)]
    agent_cells: Vec<(i32, i32)>,
    #[serde(skip)]
    positions: Vec<(f32, f32)>,
}

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
            buckets: HashMap::new(),
            agent_cells: Vec::new(),
            positions: Vec::new(),
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
        self.buckets.clear();

        for (idx, &(x, y)) in positions.iter().enumerate() {
            let key = self.cell_from_point(x, y);
            self.agent_cells[idx] = key;
            self.buckets.entry(key).or_default().push(idx);
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
                if let Some(indices) = self.buckets.get(&(nx, ny)) {
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
