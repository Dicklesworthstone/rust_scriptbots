//! Shared PBR classes with stable, reusable per-part appearance rows.
//!
//! The existing mesh hierarchy still owns transforms, visibility and picking.
//! MeshTag indexes only linear color/emission data; it never identifies a
//! scientific agent. Keeping identical mesh/material handles lets Bevy batch
//! eligible instances while retaining its normal lighting and shadow paths.

use anyhow::{Result, anyhow};
use bevy::app::AppExit;
use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::pbr::{ExtendedMaterial, MaterialExtension};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{AsBindGroup, BufferId, BufferUsages};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderStorageBuffer, ShaderStorageBuffer};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::shader::ShaderRef;

const ROW_BYTES: usize = 32;
// Bound the persistent high-water allocation independently of population input.
// The adapter's storage binding limit can lower this bound further.
const MAX_ROWS: usize = 1_048_576;
const MIN_CAPACITY: usize = 32;
const MAX_CLEAN_GAP: usize = 4;

pub(crate) type AppearanceMaterial = ExtendedMaterial<StandardMaterial, AppearanceExtension>;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub(crate) struct AppearanceExtension {
    #[storage(100, read_only)]
    colors: Handle<ShaderStorageBuffer>,
}

impl MaterialExtension for AppearanceExtension {
    fn fragment_shader() -> ShaderRef {
        "embedded://scriptbots_bevy/appearance.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "embedded://scriptbots_bevy/appearance.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "embedded://scriptbots_bevy/appearance_prepass.wgsl".into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AppearanceSlot {
    pub(crate) index: u32,
    generation: u64,
}

#[derive(Clone)]
pub(crate) struct AppearancePart {
    pub(crate) slot: AppearanceSlot,
    pub(crate) material: Handle<AppearanceMaterial>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct AppearanceRow {
    base: [f32; 4],
    emissive: [f32; 4],
}

impl AppearanceRow {
    fn new(base: Color, emissive: Color) -> Self {
        let mut emissive = LinearRgba::from(emissive).to_f32_array();
        // StandardMaterial uses this component for emissive exposure weight,
        // not opacity. Preserve the previous materials' default value of zero.
        emissive[3] = 0.0;
        Self {
            base: LinearRgba::from(base).to_f32_array(),
            emissive,
        }
    }

    fn append_bytes(self, bytes: &mut Vec<u8>) {
        for value in self.base.into_iter().chain(self.emissive) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct MaterialClass {
    alpha: AlphaMode,
    unlit: bool,
    double_sided: bool,
}

impl MaterialClass {
    fn new(alpha: AlphaMode, unlit: bool, double_sided: bool) -> Result<Self> {
        // The prepass implements exactly the modes used by creature parts.
        // Reject future unsupported modes instead of silently changing shadows.
        if !matches!(alpha, AlphaMode::Opaque | AlphaMode::Blend | AlphaMode::Add) {
            return Err(anyhow!("unsupported shared creature alpha mode: {alpha:?}"));
        }
        Ok(Self {
            alpha,
            unlit,
            double_sided,
        })
    }
}

#[derive(Clone, Copy, Default)]
struct SlotRow {
    appearance: AppearanceRow,
    revision: u64,
    generation: u64,
    occupied: bool,
}

#[derive(Resource)]
pub(crate) struct AppearanceTable {
    rows: Vec<SlotRow>,
    free: Vec<usize>,
    revision: u64,
    capacity: usize,
    max_rows: usize,
    buffer: Option<Handle<ShaderStorageBuffer>>,
    classes: Vec<(MaterialClass, Handle<AppearanceMaterial>)>,
    error: Option<String>,
}

impl Default for AppearanceTable {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            free: Vec::new(),
            revision: 0,
            capacity: 0,
            max_rows: MAX_ROWS,
            buffer: None,
            classes: Vec::new(),
            error: None,
        }
    }
}

impl AppearanceTable {
    pub(crate) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub(crate) fn fail(&mut self, error: impl std::fmt::Display) {
        if self.error.is_none() {
            let message = error.to_string();
            tracing::error!(%message, "shared creature appearance failed");
            self.error = Some(message);
        }
    }

    /// Keep allocations and material handles across capture sessions, but make
    /// every former slot available. Reuse increments each slot's generation.
    pub(crate) fn reset(&mut self) {
        for row in &mut self.rows {
            row.occupied = false;
        }
        self.free.clear();
        self.free.extend((0..self.rows.len()).rev());
        self.error = None;
    }

    fn next_revision(&mut self) -> Result<u64> {
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| anyhow!("creature appearance revision overflow"))?;
        Ok(self.revision)
    }

    /// Allocate GPU storage only when the population exceeds the retained
    /// capacity. Ordinary color updates never modify the storage-buffer asset:
    /// Bevy 0.17 would otherwise create a new GPU buffer for each modification.
    pub(crate) fn reserve(
        &mut self,
        required: usize,
        materials: &mut Assets<AppearanceMaterial>,
        buffers: &mut Assets<ShaderStorageBuffer>,
    ) -> Result<()> {
        if required > self.max_rows {
            return Err(anyhow!(
                "creature appearance needs {required} rows; device/application limit is {}",
                self.max_rows
            ));
        }
        if required <= self.capacity {
            return Ok(());
        }
        let capacity = required
            .max(MIN_CAPACITY)
            .next_power_of_two()
            .min(self.max_rows);
        let mut buffer =
            ShaderStorageBuffer::with_size(capacity * ROW_BYTES, RenderAssetUsages::default());
        buffer.buffer_description.label = Some("scriptbots creature appearance");
        buffer.buffer_description.usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let buffer = buffers.add(buffer);
        for (_, handle) in &self.classes {
            let material = materials
                .get_mut(handle)
                .ok_or_else(|| anyhow!("shared creature material disappeared"))?;
            material.extension.colors = buffer.clone();
        }
        self.buffer = Some(buffer);
        self.capacity = capacity;
        Ok(())
    }

    pub(crate) fn allocate(
        &mut self,
        materials: &mut Assets<AppearanceMaterial>,
        base: Color,
        emissive: Color,
        alpha: AlphaMode,
        unlit: bool,
        double_sided: bool,
    ) -> Result<AppearancePart> {
        let class = MaterialClass::new(alpha, unlit, double_sided)?;
        let buffer = self
            .buffer
            .as_ref()
            .ok_or_else(|| anyhow!("creature appearance storage was not reserved"))?;
        let material = if let Some((_, handle)) = self.classes.iter().find(|(key, _)| *key == class)
        {
            handle.clone()
        } else {
            let handle = materials.add(AppearanceMaterial {
                base: StandardMaterial {
                    base_color: Color::WHITE,
                    emissive: LinearRgba::BLACK,
                    alpha_mode: alpha,
                    unlit,
                    double_sided,
                    ..Default::default()
                },
                extension: AppearanceExtension {
                    colors: buffer.clone(),
                },
            });
            self.classes.push((class, handle.clone()));
            handle
        };
        let index = if let Some(index) = self.free.pop() {
            index
        } else {
            if self.rows.len() >= self.capacity {
                return Err(anyhow!("creature appearance exhausted its reserved slots"));
            }
            self.rows.push(SlotRow::default());
            self.rows.len() - 1
        };
        let generation = self.rows[index]
            .generation
            .checked_add(1)
            .ok_or_else(|| anyhow!("creature appearance slot generation overflow"))?;
        let revision = self.next_revision()?;
        self.rows[index] = SlotRow {
            appearance: AppearanceRow::new(base, emissive),
            revision,
            generation,
            occupied: true,
        };
        Ok(AppearancePart {
            slot: AppearanceSlot {
                index: index as u32,
                generation,
            },
            material,
        })
    }

    fn validate_slot(&self, slot: AppearanceSlot) -> Result<usize> {
        let index = slot.index as usize;
        if self
            .rows
            .get(index)
            .is_some_and(|row| row.occupied && row.generation == slot.generation)
        {
            Ok(index)
        } else {
            Err(anyhow!("stale creature appearance slot {}", slot.index))
        }
    }

    pub(crate) fn update(
        &mut self,
        slot: AppearanceSlot,
        base: Color,
        emissive: Color,
    ) -> Result<()> {
        let index = self.validate_slot(slot)?;
        let appearance = AppearanceRow::new(base, emissive);
        if self.rows[index].appearance != appearance {
            let revision = self.next_revision()?;
            self.rows[index].appearance = appearance;
            self.rows[index].revision = revision;
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, slot: AppearanceSlot) -> Result<()> {
        let index = self.validate_slot(slot)?;
        let revision = self.next_revision()?;
        self.rows[index].appearance = AppearanceRow::default();
        self.rows[index].revision = revision;
        self.rows[index].occupied = false;
        self.free.push(index);
        Ok(())
    }
}

#[derive(Resource, Default)]
struct ExtractedAppearance {
    rows: Vec<AppearanceRow>,
    revisions: Vec<u64>,
    revision: u64,
    buffer: Option<Handle<ShaderStorageBuffer>>,
    uploaded_buffer: Option<BufferId>,
    uploaded_revision: u64,
    bytes: Vec<u8>,
}

impl ExtractedAppearance {
    fn extract(&mut self, table: &AppearanceTable) {
        let replaced = self.buffer != table.buffer || table.revision < self.revision;
        if !replaced && self.revision == table.revision {
            return;
        }
        if replaced {
            self.buffer = table.buffer.clone();
            self.uploaded_buffer = None;
            self.uploaded_revision = 0;
        }
        self.rows.resize(table.rows.len(), AppearanceRow::default());
        self.revisions.resize(table.rows.len(), 0);
        for (index, row) in table.rows.iter().enumerate() {
            if replaced || row.revision > self.revision {
                self.rows[index] = row.appearance;
                self.revisions[index] = row.revision;
            }
        }
        self.revision = table.revision;
    }

    /// The revision stays pending while the render asset is unavailable.
    /// Coalesce nearby dirty rows to bound queue calls, retaining the staging
    /// vector's capacity between frames. A new GPU allocation gets a full copy.
    fn write_pending(&mut self, full: bool, mut write: impl FnMut(u64, &[u8])) {
        if !full && self.uploaded_revision == self.revision {
            return;
        }
        let mut cursor = 0;
        while cursor < self.rows.len() {
            let Some(first) = (cursor..self.rows.len())
                .find(|index| full || self.revisions[*index] > self.uploaded_revision)
            else {
                break;
            };
            let mut end = first + 1;
            if full {
                end = self.rows.len();
            } else {
                for next in first + 1..self.rows.len() {
                    if self.revisions[next] > self.uploaded_revision {
                        if next - end > MAX_CLEAN_GAP {
                            break;
                        }
                        end = next + 1;
                    } else if next - end > MAX_CLEAN_GAP {
                        break;
                    }
                }
            }
            self.bytes.clear();
            self.bytes.reserve((end - first) * ROW_BYTES);
            for row in &self.rows[first..end] {
                row.append_bytes(&mut self.bytes);
            }
            write((first * ROW_BYTES) as u64, &self.bytes);
            cursor = end;
        }
        self.uploaded_revision = self.revision;
    }
}

fn extract_appearance(
    table: Extract<Res<AppearanceTable>>,
    mut extracted: ResMut<ExtractedAppearance>,
) {
    extracted.extract(&table);
}

fn upload_appearance(
    mut extracted: ResMut<ExtractedAppearance>,
    buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
    queue: Res<RenderQueue>,
) {
    let Some(buffer) = extracted
        .buffer
        .as_ref()
        .and_then(|handle| buffers.get(handle))
    else {
        return;
    };
    let id = buffer.buffer.id();
    let full = extracted.uploaded_buffer != Some(id);
    extracted.write_pending(full, |offset, bytes| {
        queue.write_buffer(&buffer.buffer, offset, bytes)
    });
    extracted.uploaded_buffer = Some(id);
}

fn report_appearance_failure(
    table: Res<AppearanceTable>,
    mut exit: MessageWriter<AppExit>,
    mut reported: Local<bool>,
) {
    if table.error().is_some() && !*reported {
        *reported = true;
        exit.write(AppExit::error());
    }
}

pub(crate) struct AppearancePlugin;

impl Plugin for AppearancePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "appearance.wgsl");
        embedded_asset!(app, "appearance_prepass.wgsl");
        app.init_resource::<AppearanceTable>()
            .add_plugins(MaterialPlugin::<AppearanceMaterial>::default())
            .add_systems(Last, report_appearance_failure);
        if let Some(render) = app.get_sub_app_mut(RenderApp) {
            render
                .init_resource::<ExtractedAppearance>()
                .add_systems(ExtractSchedule, extract_appearance)
                .add_systems(
                    Render,
                    upload_appearance.in_set(RenderSystems::PrepareResources),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        if let Some(device) = app.world().get_resource::<RenderDevice>() {
            let limit = device.limits().max_storage_buffer_binding_size as usize / ROW_BYTES;
            app.world_mut().resource_mut::<AppearanceTable>().max_rows = limit.min(MAX_ROWS);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccessibilityState, AgentMeshes, AgentRegistry, ColorPaletteMode, ReflectionProbeAssets,
        SnapshotState, TerrainChunkRegistry, TerrainMaterial, WheelSuspensionJoint, WorldSnapshot,
        agent_parts, sync_world,
    };
    use bevy::mesh::MeshTag;
    use scriptbots_core::{AgentUid, NUM_EYES, SelectionState};

    fn allocate(
        table: &mut AppearanceTable,
        materials: &mut Assets<AppearanceMaterial>,
        value: f32,
    ) -> AppearancePart {
        table
            .allocate(
                materials,
                Color::linear_rgb(value, 0.2, 0.3),
                Color::BLACK,
                AlphaMode::Opaque,
                false,
                false,
            )
            .unwrap()
    }

    #[test]
    fn appearance_slots_reuse_storage_and_reject_stale_owners() {
        let mut table = AppearanceTable::default();
        let mut materials = Assets::default();
        let mut buffers = Assets::default();
        table.reserve(2, &mut materials, &mut buffers).unwrap();
        let buffer = table.buffer.clone();
        let first = allocate(&mut table, &mut materials, 0.4);
        let second = allocate(&mut table, &mut materials, 0.8);
        assert_eq!(first.material, second.material);
        assert_ne!(first.slot.index, second.slot.index);
        assert_eq!(materials.len(), 1);
        assert_eq!(buffers.len(), 1);
        let before = table.revision;
        table
            .update(first.slot, Color::linear_rgb(0.4, 0.2, 0.3), Color::BLACK)
            .unwrap();
        assert_eq!(table.revision, before, "unchanged appearance is not dirty");
        table.release(first.slot).unwrap();
        let replacement = allocate(&mut table, &mut materials, 0.6);
        assert_eq!(replacement.slot.index, first.slot.index);
        assert_ne!(replacement.slot.generation, first.slot.generation);
        assert!(
            table
                .update(first.slot, Color::WHITE, Color::BLACK)
                .is_err()
        );
        assert!(table.release(first.slot).is_err());
        assert_eq!(
            table.rows[replacement.slot.index as usize].appearance.base[0],
            0.6
        );
        assert_eq!(table.buffer, buffer);
        assert_eq!(table.rows.len(), 2);
        table.reset();
        let next_session = allocate(&mut table, &mut materials, 0.1);
        assert!(table.validate_slot(replacement.slot).is_err());
        assert!(table.validate_slot(second.slot).is_err());
        assert!(table.validate_slot(next_session.slot).is_ok());
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.buffer, buffer);
    }

    #[test]
    fn appearance_growth_rebinds_only_shared_classes_and_enforces_device_bound() {
        let mut table = AppearanceTable {
            max_rows: 70,
            ..default()
        };
        let mut materials = Assets::default();
        let mut buffers = Assets::default();
        table.reserve(1, &mut materials, &mut buffers).unwrap();
        let a = allocate(&mut table, &mut materials, 0.2);
        let b = table
            .allocate(
                &mut materials,
                Color::WHITE,
                Color::BLACK,
                AlphaMode::Blend,
                true,
                true,
            )
            .unwrap();
        let first = table.buffer.clone().unwrap();
        assert_ne!(a.material, b.material);
        assert_eq!(table.capacity, 32);
        for value in [0.4, 0.6, 0.8] {
            table
                .update(a.slot, Color::linear_rgb(value, 0.2, 0.3), Color::BLACK)
                .unwrap();
            table.reserve(30, &mut materials, &mut buffers).unwrap();
            assert_eq!(table.buffer.as_ref(), Some(&first));
            assert_eq!(buffers.len(), 1);
        }
        table.reserve(65, &mut materials, &mut buffers).unwrap();
        let grown = table.buffer.clone().unwrap();
        assert_ne!(grown, first);
        assert_eq!(table.capacity, 70);
        for part in [&a, &b] {
            assert_eq!(
                materials.get(&part.material).unwrap().extension.colors,
                grown
            );
            assert!(table.validate_slot(part.slot).is_ok());
        }
        let descriptor = &buffers.get(&grown).unwrap().buffer_description;
        assert_eq!(descriptor.size, 70 * ROW_BYTES as u64);
        assert!(
            descriptor
                .usage
                .contains(BufferUsages::STORAGE | BufferUsages::COPY_DST)
        );
        assert!(buffers.get(&grown).unwrap().data.is_none());
        assert!(table.reserve(71, &mut materials, &mut buffers).is_err());
        assert_eq!(table.buffer.as_ref(), Some(&grown));
        assert!(
            table
                .allocate(
                    &mut materials,
                    Color::WHITE,
                    Color::BLACK,
                    AlphaMode::Mask(0.5),
                    false,
                    false
                )
                .is_err()
        );
        assert_eq!(materials.len(), 2);
    }

    fn writes(extracted: &mut ExtractedAppearance, full: bool) -> Vec<(u64, Vec<u8>)> {
        let mut output = Vec::new();
        extracted.write_pending(full, |offset, bytes| output.push((offset, bytes.to_vec())));
        output
    }

    fn float_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn appearance_dirty_uploads_preserve_skipped_publications_and_reuse_staging() {
        let mut table = AppearanceTable::default();
        let mut materials = Assets::default();
        let mut buffers = Assets::default();
        table.reserve(24, &mut materials, &mut buffers).unwrap();
        let parts: Vec<_> = (0..24)
            .map(|index| allocate(&mut table, &mut materials, index as f32 / 24.0))
            .collect();
        let mut extracted = ExtractedAppearance::default();
        extracted.extract(&table);
        let initial = writes(&mut extracted, true);
        assert_eq!(initial.len(), 1);
        assert_eq!(initial[0].0, 0);
        assert_eq!(initial[0].1.len(), 24 * ROW_BYTES);
        let allocation = extracted.bytes.as_ptr();
        let capacity = extracted.bytes.capacity();
        assert!(writes(&mut extracted, false).is_empty());

        table
            .update(
                parts[1].slot,
                Color::linear_rgba(0.7, 0.2, 0.3, 0.4),
                Color::linear_rgb(2.0, 3.0, 4.0),
            )
            .unwrap();
        extracted.extract(&table);
        // No upload while a render asset is pending. A later publication must
        // retain this first row's change as well as the newest row's change.
        table
            .update(
                parts[20].slot,
                Color::linear_rgb(0.9, 0.2, 0.3),
                Color::BLACK,
            )
            .unwrap();
        extracted.extract(&table);
        let delta = writes(&mut extracted, false);
        assert_eq!(
            delta
                .iter()
                .map(|(offset, bytes)| (*offset, bytes.len()))
                .collect::<Vec<_>>(),
            vec![
                (ROW_BYTES as u64, ROW_BYTES),
                (20 * ROW_BYTES as u64, ROW_BYTES)
            ]
        );
        assert_eq!(float_at(&delta[0].1, 0), 0.7);
        assert_eq!(float_at(&delta[0].1, 12), 0.4);
        assert_eq!(float_at(&delta[0].1, 16), 2.0);
        assert_eq!(
            float_at(&delta[0].1, 28),
            0.0,
            "emissive exposure weight, not alpha"
        );
        assert_eq!(float_at(&delta[1].1, 0), 0.9);
        assert_eq!(extracted.bytes.as_ptr(), allocation);
        assert_eq!(extracted.bytes.capacity(), capacity);
        assert!(writes(&mut extracted, false).is_empty());

        // Extraction can skip an entire main-world update too.
        table
            .update(
                parts[4].slot,
                Color::linear_rgb(0.4, 0.2, 0.3),
                Color::BLACK,
            )
            .unwrap();
        table
            .update(
                parts[7].slot,
                Color::linear_rgb(0.8, 0.2, 0.3),
                Color::BLACK,
            )
            .unwrap();
        extracted.extract(&table);
        let coalesced = writes(&mut extracted, false);
        assert_eq!(coalesced.len(), 1);
        assert_eq!(coalesced[0].0, 4 * ROW_BYTES as u64);
        assert_eq!(coalesced[0].1.len(), 4 * ROW_BYTES);
        assert_eq!(float_at(&coalesced[0].1, 3 * ROW_BYTES), 0.8);

        // Buffer growth keeps existing row values, forcing a complete copy to
        // the replacement allocation even when no appearances changed.
        table.reserve(65, &mut materials, &mut buffers).unwrap();
        extracted.extract(&table);
        let replacement = writes(&mut extracted, true);
        assert_eq!(replacement.len(), 1);
        assert_eq!(replacement[0].1.len(), 24 * ROW_BYTES);
        assert_eq!(float_at(&replacement[0].1, ROW_BYTES), 0.7);
        assert_eq!(float_at(&replacement[0].1, 20 * ROW_BYTES), 0.9);
        // Device-side recreation also requires a full replay, independent of
        // the unchanged source asset/revision (the runtime compares BufferId).
        assert_eq!(writes(&mut extracted, true), replacement);
    }

    fn scene(shared: bool, snapshot: WorldSnapshot) -> App {
        let mut app = App::new();
        let mut images = Assets::<Image>::default();
        app.insert_resource(ReflectionProbeAssets::fallback(&mut images))
            .insert_resource(images)
            .insert_resource(SnapshotState {
                latest: Some(snapshot),
                ..default()
            })
            .init_resource::<AgentRegistry>()
            .init_resource::<TerrainChunkRegistry>()
            .insert_resource(AgentMeshes {
                base_radius: 1.0,
                ..default()
            })
            .init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<TerrainMaterial>>()
            .init_resource::<AccessibilityState>()
            .add_systems(Update, sync_world);
        if shared {
            app.init_resource::<AppearanceTable>()
                .init_resource::<Assets<AppearanceMaterial>>()
                .init_resource::<Assets<ShaderStorageBuffer>>();
        }
        app
    }

    fn fixture() -> WorldSnapshot {
        let mut snapshot = crate::terrain_tests::sample_world_snapshot();
        let first = crate::terrain_tests::sample_agent_visual();
        let mut second = first.clone();
        second.uid = AgentUid(2);
        second.color = [0.9, 0.2, 0.1];
        second.selection = SelectionState::None;
        second.boost = 0.0;
        second.sound_output = 0.0;
        second.sound_level = 0.0;
        second.spike_length = 0.0;
        snapshot.agents = vec![first, second];
        snapshot
    }

    fn assert_parity(reference: &App, shared: &App) {
        let reference_world = reference.world();
        let shared_world = shared.world();
        let table = shared_world.resource::<AppearanceTable>();
        assert!(table.error().is_none());
        let reference_materials = reference_world.resource::<Assets<StandardMaterial>>();
        let shared_materials = shared_world.resource::<Assets<AppearanceMaterial>>();
        for (uid, expected_record) in &reference_world.resource::<AgentRegistry>().records {
            let actual_record = &shared_world.resource::<AgentRegistry>().records[uid];
            assert_eq!(
                reference_world.get::<Transform>(expected_record.root),
                shared_world.get::<Transform>(actual_record.root)
            );
            for (expected, actual) in agent_parts(expected_record).zip(agent_parts(actual_record)) {
                let expected_material = reference_materials
                    .get(expected.material.as_ref().unwrap())
                    .unwrap();
                let part = actual.appearance.as_ref().unwrap();
                let actual_material = shared_materials.get(&part.material).unwrap();
                assert_eq!(
                    table.rows[part.slot.index as usize].appearance,
                    AppearanceRow::new(
                        expected_material.base_color,
                        Color::LinearRgba(expected_material.emissive)
                    )
                );
                assert_eq!(
                    actual_material.base.alpha_mode,
                    expected_material.alpha_mode
                );
                assert_eq!(actual_material.base.unlit, expected_material.unlit);
                assert_eq!(
                    actual_material.base.double_sided,
                    expected_material.double_sided
                );
                assert_eq!(
                    actual_material.base.perceptual_roughness,
                    expected_material.perceptual_roughness
                );
                assert_eq!(
                    actual_material.base.reflectance,
                    expected_material.reflectance
                );
                assert_eq!(
                    actual_material.base.emissive_exposure_weight,
                    expected_material.emissive_exposure_weight
                );
                assert_eq!(
                    shared_world.get::<MeshTag>(actual.entity).unwrap().0,
                    part.slot.index
                );
                assert_eq!(
                    reference_world.get::<Transform>(expected.entity),
                    shared_world.get::<Transform>(actual.entity)
                );
                assert_eq!(
                    reference_world.get::<Visibility>(expected.entity),
                    shared_world.get::<Visibility>(actual.entity)
                );
                assert!(
                    shared_world
                        .get::<MeshMaterial3d<StandardMaterial>>(actual.entity)
                        .is_none()
                );
            }
            assert!(
                shared_world
                    .get::<bevy::light::NotShadowCaster>(actual_record.selection.entity)
                    .is_some()
            );
        }
        assert_eq!(
            shared_materials.len(),
            4,
            "only four current pipeline classes"
        );
        assert_eq!(shared_world.resource::<Assets<StandardMaterial>>().len(), 0);
    }

    #[test]
    fn appearance_shared_scene_matches_reference_parts_across_palettes_and_state_changes() {
        let mut reference = scene(false, fixture());
        let mut shared = scene(true, fixture());
        for palette in [
            ColorPaletteMode::Natural,
            ColorPaletteMode::Deuteranopia,
            ColorPaletteMode::Protanopia,
            ColorPaletteMode::Tritanopia,
            ColorPaletteMode::HighContrast,
        ] {
            for app in [&mut reference, &mut shared] {
                app.world_mut().resource_mut::<AccessibilityState>().palette = palette;
                app.update();
            }
            assert_parity(&reference, &shared);
        }
        let body_slots: Vec<_> = shared
            .world()
            .resource::<AgentRegistry>()
            .records
            .values()
            .map(|record| record.body.appearance.as_ref().unwrap().slot.index)
            .collect();
        let mut joints = shared
            .world_mut()
            .query_filtered::<&MeshTag, With<WheelSuspensionJoint>>();
        let tags: Vec<_> = joints.iter(shared.world()).map(|tag| tag.0).collect();
        assert_eq!(tags.len(), 4);
        for slot in body_slots {
            assert_eq!(tags.iter().filter(|tag| **tag == slot).count(), 2);
        }

        let (revision, buffer) = {
            let table = shared.world().resource::<AppearanceTable>();
            (table.revision, table.buffer.clone())
        };
        shared
            .world_mut()
            .resource_mut::<SnapshotState>()
            .latest
            .as_mut()
            .unwrap()
            .revision += 1;
        shared.update();
        assert_eq!(
            shared.world().resource::<AppearanceTable>().revision,
            revision
        );
        assert_eq!(shared.world().resource::<AppearanceTable>().buffer, buffer);
        for app in [&mut reference, &mut shared] {
            let mut snapshots = app.world_mut().resource_mut::<SnapshotState>();
            let snapshot = snapshots.latest.as_mut().unwrap();
            snapshot.revision += 1;
            snapshot.agents[0].health = 0.2;
            snapshot.agents[0].selection = SelectionState::Hovered;
            snapshot.agents[0].sound_output = 0.0;
            snapshot.agents[0].sound_level = 0.0;
            snapshot.agents[0].boost = 0.0;
            snapshot.agents[0].eye_dirs[0] += 0.4;
            snapshot.agents[1].selection = SelectionState::Selected;
            app.update();
        }
        assert_parity(&reference, &shared);
        assert!(shared.world().resource::<AppearanceTable>().revision > revision);
        assert_eq!(shared.world().resource::<AppearanceTable>().buffer, buffer);
    }

    #[test]
    fn appearance_population_turnover_reuses_rows_and_capacity_refusal_is_visible() {
        let mut app = scene(true, fixture());
        app.update();
        let buffer = app.world().resource::<AppearanceTable>().buffer.clone();
        let rows = 2 * (14 + 2 * NUM_EYES);
        for wave in 1..8 {
            let mut snapshot = fixture();
            snapshot.revision += wave;
            for agent in &mut snapshot.agents {
                agent.uid = AgentUid(agent.uid.0 + wave * 2);
            }
            app.world_mut().resource_mut::<SnapshotState>().latest = Some(snapshot);
            app.update();
            let table = app.world().resource::<AppearanceTable>();
            assert!(table.error().is_none());
            assert_eq!(table.rows.len(), rows);
            assert!(table.rows.iter().all(|row| row.occupied));
            assert_eq!(table.buffer, buffer);
            assert_eq!(app.world().resource::<AgentRegistry>().records.len(), 2);
        }
        let mut empty = fixture();
        empty.revision += 10;
        empty.agents.clear();
        app.world_mut().resource_mut::<SnapshotState>().latest = Some(empty);
        app.update();
        let table = app.world().resource::<AppearanceTable>();
        assert_eq!(table.free.len(), rows);
        assert!(table.rows.iter().all(|row| !row.occupied));
        assert!(app.world().resource::<AgentRegistry>().records.is_empty());
        assert_eq!(table.buffer, buffer);

        let mut limited = scene(true, fixture());
        limited
            .world_mut()
            .resource_mut::<AppearanceTable>()
            .max_rows = rows - 1;
        limited.update();
        assert!(
            limited
                .world()
                .resource::<AppearanceTable>()
                .error()
                .unwrap()
                .contains("limit")
        );
        assert!(
            limited
                .world()
                .resource::<AgentRegistry>()
                .records
                .is_empty()
        );
        assert_eq!(
            limited
                .world()
                .resource::<SnapshotState>()
                .last_applied_revision,
            0,
            "an unsupported population must not be marked as displayed"
        );
    }
}
