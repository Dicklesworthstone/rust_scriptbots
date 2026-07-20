//! Pure output attribution (bd-16g.4.3): which named sensor drove which named
//! output, computed from a bounded [`BrainActivations`] snapshot.
//!
//! The inspector's contract is that an attribution must never claim more than
//! its method can support, so the method is carried in the result and every
//! unsupported topology is an explicit [`AttributionMethod::Unavailable`]
//! rather than a silently empty list that reads as "nothing drives this
//! output".
//!
//! INDEX-SPACE CONVENTION (documented, because every consumer depends on it):
//! connection endpoints index the brain's flat state vector, in which slots
//! `0..INPUT_SIZE` are the sensor inputs (named by
//! [`crate::channels::SENSOR_LAYOUT`]) and the remaining slots are internal
//! nodes. Output `i` is produced by node `i` of the FINAL activation layer —
//! the same convention the legacy C++ uses when it reads `out[j]` from the
//! brain state.

use crate::channels::{BOOST_THRESHOLD, OutputChannel, SENSOR_LAYOUT};
use crate::{BrainActivations, INPUT_SIZE};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Depth cap for the path-product method. Deep enough to cross the hidden
/// structure of any brain family currently registered, shallow enough that the
/// cost stays bounded per probed agent per frame.
pub const PATH_PRODUCT_DEPTH_CAP: usize = 4;

/// How the contributions in an [`OutputExplanation`] were computed — or the
/// honest reason they were not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributionMethod {
    /// `|weight × activation|` along direct sensor→output edges.
    OneHopWeightActivation,
    /// Product of edge weights along depth-bounded sensor→output paths.
    PathProduct {
        /// The depth bound actually used.
        depth: usize,
    },
    /// Attribution is impossible here; the reason is explicit.
    Unavailable(AttributionUnavailable),
}

/// Why attribution is unavailable for a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributionUnavailable {
    /// The brain reported no activation layers at all.
    NoActivations,
    /// The brain reported layers but no weighted edges — there is no graph to
    /// attribute over, and an empty top-k here would read as "nothing drives
    /// this output".
    NoConnections,
    /// The agent's brain binding has no runner: the actuator outputs are an
    /// identity copy of sensors `0..OUTPUT_SIZE`, so any "attribution" would be
    /// a fabricated explanation of a passthrough.
    IdentityPassthrough,
    /// The snapshot's topology matches no supported attribution method.
    UnsupportedTopology,
}

impl AttributionUnavailable {
    /// Human-facing reason string, suitable for a panel's empty state.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::NoActivations => "brain reported no activation layers",
            Self::NoConnections => {
                "brain reported no weighted connections; there is no graph to attribute over"
            }
            Self::IdentityPassthrough => {
                "unbound brain: outputs are an identity copy of sensors 0..8"
            }
            Self::UnsupportedTopology => "topology not supported by any attribution method",
        }
    }
}

/// One sensor's contribution to one output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputAttribution {
    /// Sensor slot in the brain's input vector.
    pub input_index: usize,
    /// Canonical sensor name from [`crate::channels::SENSOR_LAYOUT`].
    pub sensor_name: &'static str,
    /// Signed contribution under the stated method.
    pub contribution: f32,
    /// Edge weight along the (first) hop.
    pub weight: f32,
    /// Sensor activation at capture time.
    pub activation: f32,
}

/// What the actuator actually does with an output, beyond the raw float.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectiveOutput {
    /// The raw value is used directly.
    Continuous(f32),
    /// The value is thresholded into a boolean; the panel must show both,
    /// because `raw 0.49 → OFF` is the whole story of a near-miss.
    Thresholded {
        /// The brain's raw output.
        raw: f32,
        /// The actuator-visible state.
        active: bool,
        /// The threshold applied.
        threshold: f32,
    },
    /// The value is clamped before use.
    Clamped {
        /// The brain's raw output.
        raw: f32,
        /// The value the actuator applies.
        applied: f32,
    },
}

/// The full explanation of one output: its name, its value, the method used,
/// and the top-k sensor contributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputExplanation {
    /// Actuator slot in the output vector.
    pub output_index: usize,
    /// Canonical actuator name from [`crate::channels::OutputChannel`].
    pub output_name: &'static str,
    /// The brain's raw output value at capture time.
    pub raw_value: f32,
    /// What the actuator does with the raw value.
    pub effective: EffectiveOutput,
    /// How `inputs` was computed — or why it is empty.
    pub method: AttributionMethod,
    /// Top-k sensor contributions, sorted by (`|contribution|` desc,
    /// `input_index` asc) — an explicit total order, so the panel cannot
    /// flicker between frames on ties.
    pub inputs: Vec<InputAttribution>,
    /// Non-finite weights or activations excluded from the attribution.
    pub non_finite_skipped: u32,
}

impl OutputExplanation {
    /// Build the explanation for an unbound (identity-passthrough) agent: no
    /// attribution rows, and the reason stated.
    #[must_use]
    pub fn identity_passthrough(output_index: usize, raw_value: f32) -> Self {
        let channel = OutputChannel::ALL[output_index.min(OutputChannel::ALL.len() - 1)];
        Self {
            output_index,
            output_name: channel.name(),
            raw_value,
            effective: effective_for(channel, raw_value),
            method: AttributionMethod::Unavailable(AttributionUnavailable::IdentityPassthrough),
            inputs: Vec::new(),
            non_finite_skipped: 0,
        }
    }
}

/// Errors the pure function can return.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AttributionError {
    /// The requested output index is outside the output vector.
    #[error("output index {output} is out of range for {outputs} outputs")]
    OutputOutOfRange {
        /// Requested index.
        output: usize,
        /// Output vector length.
        outputs: usize,
    },
}

fn effective_for(channel: OutputChannel, raw: f32) -> EffectiveOutput {
    if channel == OutputChannel::Boost {
        EffectiveOutput::Thresholded {
            raw,
            active: raw > BOOST_THRESHOLD,
            threshold: BOOST_THRESHOLD,
        }
    } else {
        EffectiveOutput::Clamped {
            raw,
            applied: raw.clamp(0.0, 1.0),
        }
    }
}

/// Explain every output of one agent: the single helper BOTH inspector panels
/// call (bd-16g.4.3), so the TUI and the GPU lab can never drift into
/// different stories about the same brain.
///
/// Unbound agents get [`AttributionUnavailable::IdentityPassthrough`] rows —
/// their outputs are an identity copy of sensors `0..OUTPUT_SIZE` and any
/// "attribution" would be fabricated. A bound agent without an activation
/// snapshot yields one `Unavailable(NoActivations)` table; with a snapshot,
/// each output goes through [`attribute_output`].
///
/// # Panics
///
/// Never in practice: the internal `expect` is guarded by the `OUTPUT_SIZE`
/// bound on the loop range, and `attribute_output` validates the same bound.
#[must_use]
pub fn explain_outputs(
    outputs: &[f32; crate::OUTPUT_SIZE],
    brain_bound: bool,
    activations: Option<&BrainActivations>,
    k: usize,
) -> Vec<OutputExplanation> {
    if !brain_bound {
        return outputs
            .iter()
            .enumerate()
            .map(|(index, raw)| OutputExplanation::identity_passthrough(index, *raw))
            .collect();
    }
    activations.map_or_else(
        || {
            (0..crate::OUTPUT_SIZE)
                .map(|output| OutputExplanation {
                    output_index: output,
                    output_name: OutputChannel::ALL[output].name(),
                    raw_value: outputs[output],
                    effective: effective_for(OutputChannel::ALL[output], outputs[output]),
                    method: AttributionMethod::Unavailable(AttributionUnavailable::NoActivations),
                    inputs: Vec::new(),
                    non_finite_skipped: 0,
                })
                .collect()
        },
        |act| {
            (0..crate::OUTPUT_SIZE)
                .map(|output| {
                    attribute_output(act, output, k).expect("output index bounded by OUTPUT_SIZE")
                })
                .collect()
        },
    )
}

/// Attribute one output of a captured brain to its driving sensors.
///
/// Pure and deterministic: the result depends only on `act`, `output`, and
/// `k`. Runs for the probed agent only, on demand.
///
/// # Errors
///
/// Returns [`AttributionError::OutputOutOfRange`] when `output` is not a valid
/// actuator slot.
///
/// # Panics
///
/// Never in practice: the internal `expect` fires only if the layers vector is
/// empty, a case this function handles before reaching it.
pub fn attribute_output(
    act: &BrainActivations,
    output: usize,
    k: usize,
) -> Result<OutputExplanation, AttributionError> {
    if output >= crate::OUTPUT_SIZE {
        return Err(AttributionError::OutputOutOfRange {
            output,
            outputs: crate::OUTPUT_SIZE,
        });
    }
    let channel = OutputChannel::ALL[output];
    let unavailable = |act: &BrainActivations, reason: AttributionUnavailable| OutputExplanation {
        output_index: output,
        output_name: channel.name(),
        raw_value: raw_output_value(act, output),
        effective: effective_for(channel, raw_output_value(act, output)),
        method: AttributionMethod::Unavailable(reason),
        inputs: Vec::new(),
        non_finite_skipped: 0,
    };

    if act.layers.is_empty() {
        return Ok(unavailable(act, AttributionUnavailable::NoActivations));
    }
    if act.connections.is_empty() {
        return Ok(unavailable(act, AttributionUnavailable::NoConnections));
    }

    // One hop: direct sensor→output edges.
    let mut non_finite_skipped = 0_u32;
    let final_layer = act.layers.last().expect("non-empty layers checked above");
    let node_activation = |index: usize| -> Option<f32> {
        // Sensor slots take the sensor value carried in the FINAL layer's flat
        // vector when present (the convention documented in the module docs);
        // anything outside the layer is not an activation we can use.
        final_layer.values.get(index).copied()
    };

    let mut one_hop: Vec<InputAttribution> = Vec::new();
    let mut has_internal_edges = false;
    for edge in &act.connections {
        if edge.to != output {
            continue;
        }
        has_internal_edges |= edge.from >= INPUT_SIZE;
        if edge.from >= INPUT_SIZE {
            continue;
        }
        if !edge.weight.is_finite() {
            non_finite_skipped += 1;
            continue;
        }
        let Some(activation) = node_activation(edge.from) else {
            non_finite_skipped += 1;
            continue;
        };
        if !activation.is_finite() {
            non_finite_skipped += 1;
            continue;
        }
        one_hop.push(InputAttribution {
            input_index: edge.from,
            sensor_name: SENSOR_LAYOUT[edge.from].name,
            contribution: edge.weight * activation,
            weight: edge.weight,
            activation,
        });
    }

    let method;
    let mut inputs = one_hop;
    if !inputs.is_empty() {
        method = AttributionMethod::OneHopWeightActivation;
    } else if has_internal_edges {
        // Depth-bounded path product: expand internal edges backwards until a
        // sensor is reached or the documented cap runs out.
        let (path_inputs, path_skipped) = path_product(act, output, PATH_PRODUCT_DEPTH_CAP);
        non_finite_skipped += path_skipped;
        if path_inputs.is_empty() {
            return Ok(unavailable(
                act,
                AttributionUnavailable::UnsupportedTopology,
            ));
        }
        method = AttributionMethod::PathProduct {
            depth: PATH_PRODUCT_DEPTH_CAP,
        };
        inputs = path_inputs;
    } else {
        // Edges exist but none touch this output at all: that is a topology the
        // method cannot speak about, not "nothing drives this output".
        return Ok(unavailable(
            act,
            AttributionUnavailable::UnsupportedTopology,
        ));
    }

    // Explicit total order: |contribution| descending, input_index ascending.
    // total_cmp keeps NaN out of the comparison path (NaN was filtered above).
    inputs.sort_by(|left, right| {
        right
            .contribution
            .abs()
            .total_cmp(&left.contribution.abs())
            .then(left.input_index.cmp(&right.input_index))
    });
    inputs.truncate(k);

    Ok(OutputExplanation {
        output_index: output,
        output_name: channel.name(),
        raw_value: raw_output_value(act, output),
        effective: effective_for(channel, raw_output_value(act, output)),
        method,
        inputs,
        non_finite_skipped,
    })
}

/// The output's raw value: node `output` of the final layer, or 0.0 when the
/// snapshot carries no such slot (the Unavailable paths still owe the panel a
/// value to display; runtime.outputs is the authoritative source at the
/// surface).
fn raw_output_value(act: &BrainActivations, output: usize) -> f32 {
    act.layers
        .last()
        .and_then(|layer| layer.values.get(output))
        .copied()
        .unwrap_or(0.0)
}

/// Depth-bounded backwards expansion from the output node to sensor sources,
/// accumulating the product of edge weights along each path. Cycles are cut
/// by the depth bound; the cap is [`PATH_PRODUCT_DEPTH_CAP`].
fn path_product(
    act: &BrainActivations,
    output: usize,
    depth_cap: usize,
) -> (Vec<InputAttribution>, u32) {
    let final_layer = act.layers.last().expect("non-empty layers checked above");
    let mut skipped = 0_u32;
    // Frontier of (node, accumulated weight product, depth).
    let mut frontier: Vec<(usize, f32, usize)> = vec![(output, 1.0, 0)];
    let mut out: Vec<InputAttribution> = Vec::new();
    while let Some((node, product, depth)) = frontier.pop() {
        if depth >= depth_cap {
            continue;
        }
        for edge in &act.connections {
            if edge.to != node {
                continue;
            }
            if !edge.weight.is_finite() {
                skipped += 1;
                continue;
            }
            let next_product = product * edge.weight;
            if edge.from < INPUT_SIZE {
                let Some(activation) = final_layer.values.get(edge.from).copied() else {
                    skipped += 1;
                    continue;
                };
                if !activation.is_finite() {
                    skipped += 1;
                    continue;
                }
                out.push(InputAttribution {
                    input_index: edge.from,
                    sensor_name: SENSOR_LAYOUT[edge.from].name,
                    contribution: next_product * activation,
                    weight: next_product,
                    activation,
                });
            } else {
                frontier.push((edge.from, next_product, depth + 1));
            }
        }
    }
    // Multiple paths can reach the same sensor; sum them so each sensor
    // appears once with its total contribution.
    let mut merged: Vec<InputAttribution> = Vec::new();
    for attribution in out {
        if let Some(existing) = merged
            .iter_mut()
            .find(|entry| entry.input_index == attribution.input_index)
        {
            existing.contribution += attribution.contribution;
        } else {
            merged.push(attribution);
        }
    }
    (merged, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ActivationEdge, ActivationLayer};

    fn fixture_layer(values: &[f32]) -> ActivationLayer {
        ActivationLayer {
            name: "state".to_owned(),
            width: values.len(),
            height: 1,
            values: values.to_vec(),
        }
    }

    fn fixture(layers: Vec<ActivationLayer>, connections: Vec<ActivationEdge>) -> BrainActivations {
        BrainActivations {
            layers,
            connections,
            truncated: false,
        }
    }

    fn edge(from: usize, to: usize, weight: f32) -> ActivationEdge {
        ActivationEdge { from, to, weight }
    }

    #[test]
    fn one_hop_top_k_is_exact_in_set_and_order_with_tie_break() {
        // Wheel-left (output 0) is driven by eye0_red (weight 2.0, act 0.5),
        // blood (weight -4.0, act 0.25), and food (weight 1.0, act 1.0):
        // contributions 1.0, -1.0, 1.0 — a deliberate three-way |contribution|
        // tie that the input_index tie-break must resolve deterministically.
        let mut values = vec![0.0_f32; INPUT_SIZE + 16];
        values[1] = 0.5; // eye0_red
        values[4] = 1.0; // food
        values[19] = 0.25; // blood
        let act = fixture(
            vec![fixture_layer(&values)],
            vec![edge(19, 0, -4.0), edge(4, 0, 1.0), edge(1, 0, 2.0)],
        );
        let explanation = attribute_output(&act, 0, 3).expect("attribution");
        assert_eq!(
            explanation.method,
            AttributionMethod::OneHopWeightActivation
        );
        let indices: Vec<usize> = explanation.inputs.iter().map(|i| i.input_index).collect();
        assert_eq!(
            indices,
            vec![1, 4, 19],
            "tied |contribution| must resolve by input_index ascending"
        );
        assert_eq!(explanation.inputs[0].sensor_name, "eye0_red");
        assert_eq!(explanation.inputs[2].sensor_name, "blood");
        // k truncates after the total order is fixed.
        let top1 = attribute_output(&act, 0, 1).expect("attribution");
        assert_eq!(top1.inputs.len(), 1);
        assert_eq!(top1.inputs[0].input_index, 1);
        assert_eq!(explanation.output_name, "wheel_left");
    }

    #[test]
    fn boost_explanation_shows_raw_and_thresholded_state() {
        let mut values = vec![0.0_f32; INPUT_SIZE + 16];
        values[OutputChannel::Boost.index()] = 0.49;
        let act = fixture(vec![fixture_layer(&values)], vec![edge(4, 6, 1.0)]);
        let explanation =
            attribute_output(&act, OutputChannel::Boost.index(), 1).expect("attribution");
        assert_eq!(explanation.output_name, "boost");
        assert_eq!(
            explanation.effective,
            EffectiveOutput::Thresholded {
                raw: 0.49,
                active: false,
                threshold: BOOST_THRESHOLD,
            },
            "raw 0.49 -> boost OFF is the whole story of a near-miss"
        );
        assert_eq!(explanation.output_index, 6);
    }

    #[test]
    fn no_activations_and_no_connections_are_distinct_honest_reasons() {
        let empty = fixture(Vec::new(), Vec::new());
        let no_act = attribute_output(&empty, 0, 3).expect("no-activations explanation");
        assert_eq!(
            no_act.method,
            AttributionMethod::Unavailable(AttributionUnavailable::NoActivations)
        );
        assert!(no_act.inputs.is_empty());

        let layers_only = fixture(vec![fixture_layer(&[0.0; INPUT_SIZE + 16])], Vec::new());
        let no_conn = attribute_output(&layers_only, 0, 3).expect("no-connections explanation");
        assert_eq!(
            no_conn.method,
            AttributionMethod::Unavailable(AttributionUnavailable::NoConnections),
            "an empty top-k here would read as 'nothing drives this output'"
        );
        assert!(no_conn.inputs.is_empty());
    }

    #[test]
    fn non_finite_weights_and_activations_are_excluded_and_counted() {
        let mut values = vec![0.0_f32; INPUT_SIZE + 16];
        values[1] = f32::NAN;
        values[4] = f32::INFINITY;
        values[9] = 0.5; // sound
        let act = fixture(
            vec![fixture_layer(&values)],
            vec![
                edge(1, 0, 1.0),
                edge(4, 0, 1.0),
                edge(9, 0, f32::NAN),
                edge(9, 0, 2.0),
            ],
        );
        let explanation = attribute_output(&act, 0, 4).expect("attribution");
        assert_eq!(
            explanation.non_finite_skipped, 3,
            "NaN activation, +inf activation, and NaN weight are all excluded"
        );
        let indices: Vec<usize> = explanation.inputs.iter().map(|i| i.input_index).collect();
        assert_eq!(indices, vec![9], "only the finite sensor survives");
        assert!(
            explanation
                .inputs
                .iter()
                .all(|i| i.contribution.is_finite()),
            "no NaN may poison a displayed contribution"
        );
    }

    #[test]
    fn path_product_crosses_one_internal_hop() {
        // Output 0 has no direct sensor edge; node N (index INPUT_SIZE+3) links
        // sensors eye0_density (w 2.0) and sound (w -1.0) to the output (w 0.5).
        let mut values = vec![0.0_f32; INPUT_SIZE + 16];
        values[0] = 0.25; // eye0_density
        values[9] = 0.5; // sound
        let internal = INPUT_SIZE + 3;
        let act = fixture(
            vec![fixture_layer(&values)],
            vec![
                edge(0, internal, 2.0),
                edge(9, internal, -1.0),
                edge(internal, 0, 0.5),
            ],
        );
        let explanation = attribute_output(&act, 0, 4).expect("attribution");
        assert_eq!(
            explanation.method,
            AttributionMethod::PathProduct {
                depth: PATH_PRODUCT_DEPTH_CAP
            }
        );
        let by_index: std::collections::HashMap<usize, f32> = explanation
            .inputs
            .iter()
            .map(|i| (i.input_index, i.contribution))
            .collect();
        assert_eq!(by_index.len(), 2);
        assert!(
            (by_index[&0] - 0.25).abs() < f32::EPSILON,
            "2.0*0.5*0.25 = 0.25"
        );
        assert!(
            (by_index[&9] + 0.25).abs() < f32::EPSILON,
            "-1.0*0.5*0.5 = -0.25"
        );
    }

    #[test]
    fn unsupported_topology_is_named_not_silent() {
        // Edges exist but none touch output 3: the panel must say the topology
        // is unsupported, not "nothing drives this output".
        let act = fixture(
            vec![fixture_layer(&[0.5; INPUT_SIZE + 16])],
            vec![edge(0, 1, 1.0)],
        );
        let explanation = attribute_output(&act, 3, 3).expect("attribution");
        assert_eq!(
            explanation.method,
            AttributionMethod::Unavailable(AttributionUnavailable::UnsupportedTopology)
        );
        assert!(explanation.inputs.is_empty());
    }

    #[test]
    fn out_of_range_output_is_a_typed_error() {
        let act = fixture(
            vec![fixture_layer(&[0.0; INPUT_SIZE + 16])],
            vec![edge(0, 0, 1.0)],
        );
        assert_eq!(
            attribute_output(&act, crate::OUTPUT_SIZE, 3),
            Err(AttributionError::OutputOutOfRange {
                output: crate::OUTPUT_SIZE,
                outputs: crate::OUTPUT_SIZE,
            })
        );
    }

    #[test]
    // Test-only value synthesis: index-derived f32 fixtures, not simulation
    // math; the precision of the fixture values is irrelevant to the assertion.
    #[allow(clippy::cast_precision_loss)]
    fn attribution_is_deterministic_for_all_outputs() {
        let mut values = vec![0.0_f32; INPUT_SIZE + 16];
        for (index, value) in values.iter_mut().enumerate() {
            *value = ((index * 37 % 11) as f32) / 10.0;
        }
        let connections: Vec<ActivationEdge> = (0..INPUT_SIZE)
            .flat_map(|from| {
                (0..crate::OUTPUT_SIZE)
                    .map(move |to| edge(from, to, ((from * 13 + to * 7) % 9) as f32 / 4.0 - 1.0))
            })
            .collect();
        let act = fixture(vec![fixture_layer(&values)], connections);
        let first: Vec<OutputExplanation> = (0..crate::OUTPUT_SIZE)
            .map(|output| attribute_output(&act, output, 5).expect("attribution"))
            .collect();
        let second: Vec<OutputExplanation> = (0..crate::OUTPUT_SIZE)
            .map(|output| attribute_output(&act, output, 5).expect("attribution"))
            .collect();
        assert_eq!(first, second, "same fixture must explain identically");
    }

    #[test]
    fn identity_passthrough_carries_the_reason_and_no_rows() {
        let explanation = OutputExplanation::identity_passthrough(2, 0.75);
        assert_eq!(
            explanation.method,
            AttributionMethod::Unavailable(AttributionUnavailable::IdentityPassthrough)
        );
        assert!(explanation.inputs.is_empty());
        assert!(!explanation.output_name.is_empty());
        assert!(
            AttributionUnavailable::IdentityPassthrough
                .reason()
                .contains("identity copy")
        );
    }

    #[test]
    fn unbound_agent_outputs_are_an_identity_copy_of_sensors() {
        // THE UNBOUND-BRAIN TRAP (bd-16g.4.3): with no runner, stage_brains calls
        // default_outputs, which copies the first 9 sensors straight into the
        // outputs. Such an agent has perfect-looking "attribution" that means
        // absolutely nothing; the panel must refuse to explain it.
        let mut world = crate::WorldState::new(crate::ScriptBotsConfig {
            world_width: 200,
            world_height: 200,
            food_cell_size: 20,
            rng_seed: Some(0x00A7_7E57),
            ..crate::ScriptBotsConfig::default()
        })
        .expect("world");
        let agent = world
            .try_spawn_agent(crate::AgentData::default())
            .expect("unbound agent spawn");
        assert!(
            !world
                .agent_runtime(agent)
                .expect("runtime")
                .brain
                .is_bound(),
            "fixture agent must be unbound"
        );
        world.step().expect("one tick");
        let runtime = world.agent_runtime(agent).expect("runtime");
        for index in 0..crate::OUTPUT_SIZE {
            // The identity copy is a BITWISE property, not an approximation.
            assert_eq!(
                runtime.outputs[index].to_bits(),
                runtime.sensors[index].to_bits(),
                "unbound output {index} must be an identity copy of sensor {index}"
            );
        }
        // The panel's response to this agent is the passthrough explanation, not
        // an attribution over the identity mapping.
        let explanation = OutputExplanation::identity_passthrough(0, runtime.outputs[0]);
        assert!(explanation.inputs.is_empty());
        assert_eq!(
            explanation.method,
            AttributionMethod::Unavailable(AttributionUnavailable::IdentityPassthrough)
        );
    }

    #[test]
    fn channel_names_are_the_centralized_wire_map() {
        // Regression for the historical mislabeling (combat read outputs[3],
        // the GREEN COLOR channel, as if it were boost): the panel's names come
        // from bd-2z0.2.4's centralized map, never a hand-rolled table.
        assert_eq!(OutputChannel::ALL[6].name(), "boost");
        assert_eq!(OutputChannel::ALL[3].name(), "color_green");
        assert_eq!(OutputChannel::ALL[0].name(), "wheel_left");
        assert_eq!(OutputChannel::ALL[8].name(), "give_intent");
    }
}
