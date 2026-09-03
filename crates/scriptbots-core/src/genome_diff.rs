//! Protocol-level genome locus addressing and structural diffing (bd-16g.13.1).
//!
//! Evolution in this project happens at a level nothing could observe until now: the
//! genome. This module defines the canonical locus address space ([`Locus`]), the typed
//! delta vocabulary ([`GenomeDelta`]), and the only sanctioned comparison
//! ([`diff_genomes`]) — all computed from protocol envelopes through the family codec's
//! [`BrainFamilyCodec::genome_loci`], never from live brain objects, never by text-diffing
//! serialized dumps. A reordered field must never read as a mutation, and a retarget from
//! 88 to 14 must read as exactly one typed [`GenomeDelta::Retarget`].

use crate::{AgentUid, BrainFamilyCodec, BrainFamilyId, BrainGenomeEnvelope, Tick};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One addressable genome locus, in canonical order (index order, never a hash walk).
///
/// Two genomes of the same family and schema produce loci in the same order, so the diff
/// is a single aligned pass and its bytes are reproducible on any platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Locus {
    /// Bias of the node with the given index.
    NodeBias(u32),
    /// Damping factor of the node with the given index.
    NodeDamping(u32),
    /// Gain of the node with the given index.
    NodeGain(u32),
    /// Weight of connection `conn` feeding node `node`.
    NodeWeight {
        /// Index of the node that owns the connection.
        node: u32,
        /// Incoming connection slot within the node.
        conn: u8,
    },
    /// Function kind of connection `conn` feeding node `node`.
    NodeKind {
        /// Index of the node that owns the connection.
        node: u32,
        /// Incoming connection slot within the node.
        conn: u8,
    },
    /// Target node of connection `conn` leaving node `node`.
    NodeTarget {
        /// Index of the node that owns the connection.
        node: u32,
        /// Outgoing connection slot within the node.
        conn: u8,
    },
    /// A family-level hyperparameter locus (reserved; families name their own).
    Hyper(u8),
    /// Scalar cell at the given index in an assembly-style genome.
    Cell(u32),
}

impl Locus {
    /// The string the UI shows ("node 47 weight 2", "node 130 conn 1 kind").
    #[must_use]
    pub fn human(&self) -> String {
        match *self {
            Self::NodeBias(node) => format!("node {node} bias"),
            Self::NodeDamping(node) => format!("node {node} damping"),
            Self::NodeGain(node) => format!("node {node} gain"),
            Self::NodeWeight { node, conn } => format!("node {node} weight {conn}"),
            Self::NodeKind { node, conn } => format!("node {node} conn {conn} kind"),
            Self::NodeTarget { node, conn } => format!("node {node} conn {conn} target"),
            Self::Hyper(id) => format!("hyperparameter {id}"),
            Self::Cell(index) => format!("cell {index}"),
        }
    }
}

/// One decoded locus value. Float comparison for scalars is BITWISE (`f32::to_bits`) —
/// heredity is an exact claim, and an epsilon-tolerant diff would hide exactly the
/// corruption it exists to find.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LocusValue {
    /// A continuous scalar parameter; compared bitwise via `f32::to_bits`.
    Scalar(f32),
    /// A connection target: the index of the node a connection points at.
    Target(u32),
    /// A connection function-kind tag.
    Kind(u8),
}

/// One typed change between two aligned genomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GenomeDelta {
    /// A scalar parameter whose bits changed.
    Scalar {
        /// The locus that changed.
        locus: Locus,
        /// Parent value.
        before: f32,
        /// Child value.
        after: f32,
    },
    /// A connection whose target node changed.
    Retarget {
        /// The locus that changed.
        locus: Locus,
        /// Parent target node index.
        before: u32,
        /// Child target node index.
        after: u32,
    },
    /// A connection whose function kind changed.
    KindFlip {
        /// The locus that changed.
        locus: Locus,
        /// Parent kind tag.
        before: u8,
        /// Child kind tag.
        after: u8,
    },
}

/// The kind of a delta, for the by-kind summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DeltaKind {
    /// Scalar value change.
    Scalar,
    /// Connection retarget.
    Retarget,
    /// Function-kind change.
    KindFlip,
}

/// Aggregate diff statistics, computed in one pass over the canonical delta list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffSummary {
    /// Number of loci that differ (equal to the delta count).
    pub changed_loci: usize,
    /// Total loci compared (genome length).
    pub total_loci: usize,
    /// Sum of |after - before| over scalar deltas.
    pub l1: f64,
    /// Max |after - before| over scalar deltas.
    pub linf: f64,
    /// Delta count per kind.
    pub by_kind: BTreeMap<DeltaKind, usize>,
}

/// The complete structural diff between two genomes of one family and schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeDiff {
    /// Brain family both genomes belong to.
    pub family: BrainFamilyId,
    /// Genome schema version both envelopes share.
    pub schema_version: u32,
    /// Typed deltas in canonical locus order.
    pub deltas: Vec<GenomeDelta>,
    /// Aggregate statistics over `deltas`.
    pub summary: DiffSummary,
}

/// A single point sample along a lineage trace for a specific locus.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocusSample {
    /// Lineage generation.
    pub generation: u32,
    /// Stable Agent UID.
    pub agent_uid: AgentUid,
    /// Tick when the sample was recorded.
    pub tick: Tick,
    /// Value at this locus, or `None` if the locus is absent in an older schema/shape.
    pub value: Option<LocusValue>,
}

/// Trace a specific locus value across a sequence of lineage ancestor envelopes.
#[must_use]
pub fn trace_lineage_locus(
    codec: &dyn BrainFamilyCodec,
    lineage_samples: &[(u32, AgentUid, Tick, BrainGenomeEnvelope)],
    locus: Locus,
) -> Vec<LocusSample> {
    lineage_samples
        .iter()
        .map(|(generation, agent_uid, tick, envelope)| {
            let value = codec.genome_loci(envelope).ok().and_then(|loci| {
                loci.into_iter()
                    .find(|(loc, _)| *loc == locus)
                    .map(|(_, val)| val)
            });
            LocusSample {
                generation: *generation,
                agent_uid: *agent_uid,
                tick: *tick,
                value,
            }
        })
        .collect()
}

/// Export a locus trace as a CSV string.
#[must_use]
pub fn export_locus_trace_csv(samples: &[LocusSample], locus: Locus) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Locus Trace: {}\n", locus.human()));
    out.push_str("generation,agent_uid,tick,value_type,value\n");
    for s in samples {
        match s.value {
            Some(LocusValue::Scalar(v)) => {
                out.push_str(&format!(
                    "{},{},{},scalar,{v}\n",
                    s.generation, s.agent_uid.0, s.tick.0
                ));
            }
            Some(LocusValue::Target(v)) => {
                out.push_str(&format!(
                    "{},{},{},target,{v}\n",
                    s.generation, s.agent_uid.0, s.tick.0
                ));
            }
            Some(LocusValue::Kind(v)) => {
                out.push_str(&format!(
                    "{},{},{},kind,{v}\n",
                    s.generation, s.agent_uid.0, s.tick.0
                ));
            }
            None => {
                out.push_str(&format!(
                    "{},{},{},gap,GAP\n",
                    s.generation, s.agent_uid.0, s.tick.0
                ));
            }
        }
    }
    out
}

/// Export a locus trace as an SVG chart string.
#[must_use]
pub fn export_locus_trace_svg(samples: &[LocusSample], locus: Locus) -> String {
    let width = 600.0;
    let height = 300.0;
    let padding = 40.0;

    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"##
    ));
    svg.push_str(r##"<rect width="100%" height="100%" fill="#1e1e2e"/>"##);
    svg.push_str(&format!(
        r##"<text x="{}" y="25" fill="#cdd6f4" font-size="14" font-family="sans-serif" text-anchor="middle">Locus Trace: {}</text>"##,
        width / 2.0,
        locus.human()
    ));

    let valid_scalars: Vec<(f32, f32)> = samples
        .iter()
        .filter_map(|s| match s.value {
            Some(LocusValue::Scalar(v)) => Some((s.generation as f32, v)),
            _ => None,
        })
        .collect();

    if valid_scalars.len() >= 2 {
        let min_gen = valid_scalars
            .iter()
            .map(|(g, _)| *g)
            .fold(f32::INFINITY, f32::min);
        let max_gen = valid_scalars
            .iter()
            .map(|(g, _)| *g)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_val = valid_scalars
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::INFINITY, f32::min);
        let max_val = valid_scalars
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);

        let gen_span = (max_gen - min_gen).max(1.0);
        let val_span = (max_val - min_val).max(0.001);

        let mut points = String::new();
        for (g, v) in &valid_scalars {
            let x = padding + (g - min_gen) / gen_span * (width - 2.0 * padding);
            let y = height - padding - (v - min_val) / val_span * (height - 2.0 * padding);
            if !points.is_empty() {
                points.push(' ');
            }
            points.push_str(&format!("{x:.1},{y:.1}"));
        }
        svg.push_str(&format!(
            "<polyline fill=\"none\" stroke=\"#89b4fa\" stroke-width=\"2\" points=\"{points}\"/>"
        ));
    }

    svg.push_str("</svg>");
    svg
}

fn crc32_chunk(chunk_type: &[u8; 4], data: &[u8]) -> u32 {
    let mut crc: u32 = 0xffff_ffff;
    for &byte in chunk_type.iter().chain(data.iter()) {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xedb8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn write_png_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let len = u32::try_from(data.len()).expect("PNG chunk length fits u32");
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let crc = crc32_chunk(chunk_type, data);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn encode_rgba_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    // 1. Signature
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    // 2. IHDR Chunk
    let mut ihdr_data = Vec::with_capacity(13);
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // 8-bit depth
    ihdr_data.push(6); // RGBA color type
    ihdr_data.push(0); // compression method 0
    ihdr_data.push(0); // filter method 0
    ihdr_data.push(0); // interlace method 0
    write_png_chunk(&mut out, b"IHDR", &ihdr_data);

    // 3. IDAT Chunk
    let scanline_len = 1 + (width as usize) * 4;
    let mut raw_scanlines = Vec::with_capacity((height as usize) * scanline_len);
    for row in 0..(height as usize) {
        raw_scanlines.push(0); // Filter type 0: None
        let start = row * (width as usize) * 4;
        let end = start + (width as usize) * 4;
        raw_scanlines.extend_from_slice(&rgba[start..end]);
    }

    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&raw_scanlines, 6);
    write_png_chunk(&mut out, b"IDAT", &compressed);

    // 4. IEND Chunk
    write_png_chunk(&mut out, b"IEND", &[]);

    out
}

/// Export a locus trace as a headless PNG image buffer.
#[must_use]
pub fn export_locus_trace_png(samples: &[LocusSample], locus: Locus) -> Vec<u8> {
    let _ = locus;
    const WIDTH: usize = 600;
    const HEIGHT: usize = 300;
    const PADDING: usize = 40;

    let mut pixels = [30u8, 30u8, 46u8, 255u8].repeat(WIDTH * HEIGHT);

    let put_pixel = |pixels: &mut [u8], x: i32, y: i32, color: [u8; 4]| {
        if x >= 0 && (x as usize) < WIDTH && y >= 0 && (y as usize) < HEIGHT {
            let idx = ((y as usize) * WIDTH + (x as usize)) * 4;
            pixels[idx..idx + 4].copy_from_slice(&color);
        }
    };

    let draw_line =
        |pixels: &mut [u8], mut x0: i32, mut y0: i32, x1: i32, y1: i32, color: [u8; 4]| {
            let dx = (x1 - x0).abs();
            let dy = -(y1 - y0).abs();
            let sx = if x0 < x1 { 1 } else { -1 };
            let sy = if y0 < y1 { 1 } else { -1 };
            let mut err = dx + dy;
            loop {
                put_pixel(pixels, x0, y0, color);
                put_pixel(pixels, x0 + 1, y0, color);
                put_pixel(pixels, x0, y0 + 1, color);
                if x0 == x1 && y0 == y1 {
                    break;
                }
                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    x0 += sx;
                }
                if e2 <= dx {
                    err += dx;
                    y0 += sy;
                }
            }
        };

    let draw_dot = |pixels: &mut [u8], cx: i32, cy: i32, radius: i32, color: [u8; 4]| {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    put_pixel(pixels, cx + dx, cy + dy, color);
                }
            }
        }
    };

    let draw_cross = |pixels: &mut [u8], cx: i32, cy: i32, size: i32, color: [u8; 4]| {
        for d in -size..=size {
            put_pixel(pixels, cx + d, cy + d, color);
            put_pixel(pixels, cx + d, cy - d, color);
        }
    };

    // Draw frame/axes
    let axis_color = [69, 71, 90, 255]; // Catppuccin surface0
    draw_line(
        &mut pixels,
        PADDING as i32,
        (HEIGHT - PADDING) as i32,
        (WIDTH - PADDING) as i32,
        (HEIGHT - PADDING) as i32,
        axis_color,
    );
    draw_line(
        &mut pixels,
        PADDING as i32,
        PADDING as i32,
        PADDING as i32,
        (HEIGHT - PADDING) as i32,
        axis_color,
    );
    draw_line(
        &mut pixels,
        (WIDTH - PADDING) as i32,
        PADDING as i32,
        (WIDTH - PADDING) as i32,
        (HEIGHT - PADDING) as i32,
        axis_color,
    );
    draw_line(
        &mut pixels,
        PADDING as i32,
        PADDING as i32,
        (WIDTH - PADDING) as i32,
        PADDING as i32,
        axis_color,
    );

    let valid_scalars: Vec<(f32, f32)> = samples
        .iter()
        .filter_map(|s| match s.value {
            Some(LocusValue::Scalar(v)) => Some((s.generation as f32, v)),
            _ => None,
        })
        .collect();

    if !valid_scalars.is_empty() {
        let min_gen = valid_scalars
            .iter()
            .map(|(g, _)| *g)
            .fold(f32::INFINITY, f32::min);
        let max_gen = valid_scalars
            .iter()
            .map(|(g, _)| *g)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_val = valid_scalars
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::INFINITY, f32::min);
        let max_val = valid_scalars
            .iter()
            .map(|(_, v)| *v)
            .fold(f32::NEG_INFINITY, f32::max);

        let gen_span = (max_gen - min_gen).max(1.0);
        let val_span = (max_val - min_val).max(0.001);

        let mut prev_pt: Option<(i32, i32)> = None;
        let line_color = [137, 180, 250, 255]; // Catppuccin blue
        let dot_color = [205, 214, 244, 255]; // Catppuccin text

        for (g, v) in &valid_scalars {
            let x = (PADDING as f32 + (g - min_gen) / gen_span * (WIDTH - 2 * PADDING) as f32)
                .round() as i32;
            let y = ((HEIGHT - PADDING) as f32
                - (v - min_val) / val_span * (HEIGHT - 2 * PADDING) as f32)
                .round() as i32;
            if let Some((px, py)) = prev_pt {
                draw_line(&mut pixels, px, py, x, y, line_color);
            }
            draw_dot(&mut pixels, x, y, 3, dot_color);
            prev_pt = Some((x, y));
        }

        // Draw gaps as crosses on bottom axis
        let gap_color = [243, 139, 168, 255]; // Catppuccin red
        for s in samples {
            if s.value.is_none() {
                let g = s.generation as f32;
                let x = (PADDING as f32 + (g - min_gen) / gen_span * (WIDTH - 2 * PADDING) as f32)
                    .round() as i32;
                let y = (HEIGHT - PADDING) as i32;
                draw_cross(&mut pixels, x, y, 4, gap_color);
            }
        }
    }

    encode_rgba_png(
        u32::try_from(WIDTH).expect("fits u32"),
        u32::try_from(HEIGHT).expect("fits u32"),
        &pixels,
    )
}

/// Degenerate inputs are typed, never a panic and never a silent empty diff.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GenomeDiffError {
    /// The two genomes belong to different brain families.
    #[error("cannot diff across brain families: {a} vs {b}")]
    FamilyMismatch {
        /// Family of the parent genome.
        a: BrainFamilyId,
        /// Family of the child genome.
        b: BrainFamilyId,
    },
    /// The two genomes use different schema versions.
    #[error("cannot diff across genome schemas: {a} vs {b}")]
    SchemaMismatch {
        /// Schema version of the parent genome.
        a: u32,
        /// Schema version of the child genome.
        b: u32,
    },
    /// The decoded locus lists differ in length.
    #[error("genome shape mismatch: {a_loci} loci vs {b_loci} loci")]
    ShapeMismatch {
        /// Locus count of the parent genome.
        a_loci: usize,
        /// Locus count of the child genome.
        b_loci: usize,
    },
    /// The decoded locus address sequences disagree at an index.
    #[error(
        "locus address mismatch at index {index}: parent has {parent_human} ({parent:?}) vs child has {child_human} ({child:?})"
    )]
    LocusMismatch {
        /// Index of the first mismatched locus address.
        index: usize,
        /// Parent locus address.
        parent: Locus,
        /// Child locus address.
        child: Locus,
        /// Human-readable parent locus description.
        parent_human: String,
        /// Human-readable child locus description.
        child_human: String,
    },
    /// The value kinds disagree at a shared locus address.
    #[error(
        "locus value type mismatch at index {index} ({locus:?}): parent is {parent_type} vs child is {child_type}"
    )]
    ValueTypeMismatch {
        /// Index of the locus with mismatched value kinds.
        index: usize,
        /// The locus address.
        locus: Locus,
        /// Parent value kind name.
        parent_type: &'static str,
        /// Child value kind name.
        child_type: &'static str,
    },
    /// The family codec could not expose typed loci.
    #[error("family {family} cannot expose typed loci: {reason}")]
    Unsupported {
        /// Family that failed to decode.
        family: BrainFamilyId,
        /// Codec-reported failure detail.
        reason: String,
    },
}

const fn locus_value_type_name(val: &LocusValue) -> &'static str {
    match val {
        LocusValue::Scalar(_) => "scalar",
        LocusValue::Target(_) => "target",
        LocusValue::Kind(_) => "kind",
    }
}

/// Compute the typed structural diff between two genomes of the same family and schema.
///
/// Decoded through the family codec's locus view. Deltas are emitted in canonical locus
/// order, so two invocations produce identical bytes on any platform at any thread count.
/// An empty delta list means "no mutations" — and nothing else.
pub fn diff_genomes(
    codec: &dyn BrainFamilyCodec,
    parent: &BrainGenomeEnvelope,
    child: &BrainGenomeEnvelope,
) -> Result<GenomeDiff, GenomeDiffError> {
    if parent.family_id() != child.family_id() {
        return Err(GenomeDiffError::FamilyMismatch {
            a: parent.family_id().clone(),
            b: child.family_id().clone(),
        });
    }
    if parent.schema_version() != child.schema_version() {
        return Err(GenomeDiffError::SchemaMismatch {
            a: parent.schema_version(),
            b: child.schema_version(),
        });
    }
    let family = parent.family_id().clone();
    let schema_version = parent.schema_version();
    let parent_loci = codec
        .genome_loci(parent)
        .map_err(|error| GenomeDiffError::Unsupported {
            family: family.clone(),
            reason: error.to_string(),
        })?;
    let child_loci = codec
        .genome_loci(child)
        .map_err(|error| GenomeDiffError::Unsupported {
            family: family.clone(),
            reason: error.to_string(),
        })?;
    if parent_loci.len() != child_loci.len() {
        return Err(GenomeDiffError::ShapeMismatch {
            a_loci: parent_loci.len(),
            b_loci: child_loci.len(),
        });
    }

    let mut deltas = Vec::new();
    let mut l1 = 0.0_f64;
    let mut linf = 0.0_f64;
    let mut by_kind: BTreeMap<DeltaKind, usize> = BTreeMap::new();
    for (index, ((locus, before), (child_locus, after))) in
        parent_loci.iter().zip(child_loci.iter()).enumerate()
    {
        if locus != child_locus {
            return Err(GenomeDiffError::LocusMismatch {
                index,
                parent: *locus,
                child: *child_locus,
                parent_human: locus.human(),
                child_human: child_locus.human(),
            });
        }
        match (before, after) {
            (LocusValue::Scalar(before), LocusValue::Scalar(after)) => {
                if before.to_bits() != after.to_bits() {
                    let magnitude = f64::from((*after - *before).abs());
                    l1 += magnitude;
                    linf = linf.max(magnitude);
                    *by_kind.entry(DeltaKind::Scalar).or_insert(0) += 1;
                    deltas.push(GenomeDelta::Scalar {
                        locus: *locus,
                        before: *before,
                        after: *after,
                    });
                }
            }
            (LocusValue::Target(before), LocusValue::Target(after)) => {
                if before != after {
                    *by_kind.entry(DeltaKind::Retarget).or_insert(0) += 1;
                    deltas.push(GenomeDelta::Retarget {
                        locus: *locus,
                        before: *before,
                        after: *after,
                    });
                }
            }
            (LocusValue::Kind(before), LocusValue::Kind(after)) => {
                if before != after {
                    *by_kind.entry(DeltaKind::KindFlip).or_insert(0) += 1;
                    deltas.push(GenomeDelta::KindFlip {
                        locus: *locus,
                        before: *before,
                        after: *after,
                    });
                }
            }
            _ => {
                return Err(GenomeDiffError::ValueTypeMismatch {
                    index,
                    locus: *locus,
                    parent_type: locus_value_type_name(before),
                    child_type: locus_value_type_name(after),
                });
            }
        }
    }

    Ok(GenomeDiff {
        family,
        schema_version,
        summary: DiffSummary {
            changed_loci: deltas.len(),
            total_loci: parent_loci.len(),
            l1,
            linf,
            by_kind,
        },
        deltas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locus_human_strings_name_the_gene() {
        assert_eq!(Locus::NodeBias(47).human(), "node 47 bias");
        assert_eq!(
            Locus::NodeWeight { node: 47, conn: 2 }.human(),
            "node 47 weight 2"
        );
        assert_eq!(
            Locus::NodeKind { node: 130, conn: 1 }.human(),
            "node 130 conn 1 kind"
        );
        assert_eq!(
            Locus::NodeTarget { node: 3, conn: 0 }.human(),
            "node 3 conn 0 target"
        );
        assert_eq!(Locus::Cell(47).human(), "cell 47");
    }

    #[test]
    fn test_trace_lineage_locus_and_exporters() {
        let locus = Locus::NodeBias(0);
        let samples = vec![
            LocusSample {
                generation: 1,
                agent_uid: AgentUid(100),
                tick: Tick(10),
                value: Some(LocusValue::Scalar(0.5)),
            },
            LocusSample {
                generation: 2,
                agent_uid: AgentUid(101),
                tick: Tick(20),
                value: Some(LocusValue::Scalar(0.75)),
            },
            LocusSample {
                generation: 3,
                agent_uid: AgentUid(102),
                tick: Tick(30),
                value: None, // gap
            },
        ];

        let csv = export_locus_trace_csv(&samples, locus);
        assert!(csv.contains("generation,agent_uid,tick,value_type,value"));
        assert!(csv.contains("1,100,10,scalar,0.5"));
        assert!(csv.contains("3,102,30,gap,GAP"));

        let svg = export_locus_trace_svg(&samples, locus);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Locus Trace: node 0 bias"));
        assert!(svg.contains("<polyline"));

        let png = export_locus_trace_png(&samples, locus);
        assert_eq!(
            &png[0..8],
            &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
        );
        assert!(png.len() > 100);
    }
}
