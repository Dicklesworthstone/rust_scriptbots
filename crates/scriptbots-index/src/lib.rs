//! Spatial indexing abstractions for agent neighborhood queries.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
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

/// Baseline uniform grid index; currently a placeholder until full implementation lands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniformGridIndex {
    /// Edge length of each grid cell used for bucketing agents.
    pub cell_size: f32,
    #[serde(skip)]
    agent_order: Vec<usize>,
}

impl UniformGridIndex {
    /// Create a new uniform grid with the provided cell size.
    #[must_use]
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            agent_order: Vec::new(),
        }
    }
}

impl Default for UniformGridIndex {
    fn default() -> Self {
        Self::new(50.0)
    }
}

impl NeighborhoodIndex for UniformGridIndex {
    fn rebuild(&mut self, positions: &[(f32, f32)]) -> Result<(), IndexError> {
        if self.cell_size <= 0.0 {
            return Err(IndexError::InvalidConfig("cell_size must be positive"));
        }
        self.agent_order.clear();
        self.agent_order
            .extend(positions.iter().enumerate().map(|(idx, _)| idx));
        Ok(())
    }

    fn neighbors_within(
        &self,
        agent_idx: usize,
        _radius_sq: f32,
        _visitor: &mut dyn FnMut(usize, OrderedFloat<f32>),
    ) {
        // Placeholder: full neighborhood traversal will be implemented with real grid buckets.
        let _ = agent_idx;
    }
}
