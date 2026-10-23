//! CPU-vs-GPU sensory parity verification gate and provider implementations (bd-16g.15.3).
//!
//! Provides:
//! - Pluggable [`GpuSenseProvider`] implementing [`scriptbots_core::SenseProvider`]
//! - Fault-injected variant [`FaultInjectedGpuSenseProvider`]
//! - Typed [`SenseBackendId`], [`SenseDeterminism`], [`SenseGateEvidenceV0`], [`SensePolicyV0`]
//! - 1,000-tick parity check harness producing [`SenseLaneParityReport`] with divergence coordinates

use crate::sense_wgsl::{
    AgentGpuData, GpuSenseError, GpuSensePipeline, GridUniforms, PRODUCTION_WORKGROUP_SIZE,
};
use scriptbots_core::sense_fixed::{SenseAccum, SenseProvider, SenseProviderError};
use scriptbots_core::{AgentData, AgentId, OutputsExt, Position, ScriptBotsConfig, WorldState};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

/// Typed identifier for sensory accumulation execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SenseBackendId {
    /// Classical CPU uniform grid spatial accumulation.
    Cpu,
    /// GPU compute shader CSR binning and 64-bit integer tree reduction.
    Gpu,
}

impl std::fmt::Display for SenseBackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

/// Certification status of sensory accumulation determinism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SenseDeterminism {
    /// Zero divergence across verified ticks; identical characterization digest.
    Exact,
    /// Non-zero divergence detected or uncertified adapter.
    Approximate,
}

impl std::fmt::Display for SenseDeterminism {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exact => write!(f, "exact"),
            Self::Approximate => write!(f, "approximate"),
        }
    }
}

/// Exact coordinates and values of first observed sensor divergence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SenseDivergenceCoordinates {
    /// Scientific tick where divergence was first detected.
    pub tick: u64,
    /// Stable agent UID of first divergent agent.
    pub agent_uid: u64,
    /// Handle index in world agent list.
    pub agent_index: usize,
    /// Sensor channel index (0..24) where delta occurred.
    pub sensor_index: usize,
    /// CPU reference sensor value.
    pub cpu_sensor_value: f32,
    /// GPU sensor value.
    pub gpu_sensor_value: f32,
    /// Absolute difference between CPU and GPU sensor values.
    pub sensor_delta: f32,
    /// CPU accumulator channels if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_accumulator: Option<Vec<i64>>,
    /// GPU accumulator channels if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_accumulator: Option<Vec<i64>>,
}

/// Parity gate evidence certifying a GPU sense backend against CPU reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SenseGateEvidenceV0 {
    /// Target triple (e.g. `x86_64-unknown-linux-gnu`).
    pub target: String,
    /// Graphics adapter description name.
    pub adapter: String,
    /// PCI vendor identifier.
    pub vendor: u32,
    /// PCI device identifier.
    pub device: u32,
    /// Graphics backend name (e.g. `Vulkan`, `Metal`, `Dx12`).
    pub backend: String,
    /// Driver name.
    pub driver: String,
    /// Driver information string.
    pub driver_info: String,
    /// BLAKE3 hex digest of the GPU compute shader source.
    pub shader_digest: String,
    /// Build identity (git commit / version).
    pub build_identity: String,
    /// Compiler toolchain identity (`rustc -Vv`).
    pub toolchain_identity: String,
    /// Scenario fixture digest.
    pub fixture_digest: String,
    /// Number of verified simulation ticks.
    pub tick_count: u64,
    /// Unix timestamp in milliseconds when evidence was captured.
    pub evidence_time_unix_ms: u64,
    /// BLAKE3 digest over all canonical parity fields.
    pub evidence_digest: String,
    /// Certification determinism verdict.
    pub determinism: SenseDeterminism,
    /// Maximum observed sensor delta across all agents and ticks.
    pub max_sensor_delta: f32,
}

/// Resolved sensory accumulation policy attached to [`scriptbots_app::RunManifestV3`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensePolicyV0 {
    /// Active sensing backend.
    pub backend: SenseBackendId,
    /// Determinism certification.
    pub determinism: SenseDeterminism,
    /// Which layer determined the backend (`cli-flag`, `auto`, or `builtin-default`).
    pub source: String,
    /// Whether `--allow-approximate-sense` was explicitly opted into.
    pub allow_approximate: bool,
    /// Gate validation evidence if GPU backend was evaluated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_evidence: Option<SenseGateEvidenceV0>,
    /// Whether this policy is certified exact.
    #[serde(default)]
    pub certified_exact: bool,
    /// Human-readable reason or details about backend selection.
    #[serde(default)]
    pub reason: String,
}

impl Default for SensePolicyV0 {
    fn default() -> Self {
        Self {
            backend: SenseBackendId::Cpu,
            determinism: SenseDeterminism::Exact,
            source: "builtin-default".to_string(),
            allow_approximate: false,
            gate_evidence: None,
            certified_exact: true,
            reason: "CPU fixed-point default".to_string(),
        }
    }
}

/// Comprehensive report emitted by the CPU-vs-GPU parity verification runner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SenseLaneParityReport {
    /// Target triple.
    pub target: String,
    /// Graphics adapter description.
    pub adapter: String,
    /// PCI vendor identifier.
    pub vendor: u32,
    /// PCI device identifier.
    pub device: u32,
    /// Graphics backend name.
    pub backend: String,
    /// Driver name.
    pub driver: String,
    /// Driver information string.
    pub driver_info: String,
    /// BLAKE3 hex digest of compute shader source.
    pub shader_digest: String,
    /// Build identity.
    pub build_identity: String,
    /// Toolchain identity.
    pub toolchain_identity: String,
    /// Fixture digest.
    pub fixture_digest: String,
    /// Number of ticks executed.
    pub tick_count: u64,
    /// Timestamp in milliseconds since Unix epoch.
    pub evidence_time_unix_ms: u64,
    /// BLAKE3 digest of the evidence record.
    pub evidence_digest: String,
    /// Final CPU characterization digest.
    pub cpu_digest: String,
    /// Final GPU characterization digest.
    pub gpu_digest: String,
    /// Parity verdict: Exact or Approximate.
    pub verdict: SenseDeterminism,
    /// First observed divergence, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_divergence: Option<SenseDivergenceCoordinates>,
    /// Maximum observed delta across all sensors and ticks.
    pub max_sensor_delta: f32,
    /// Whether the adapter was a software fallback (e.g. llvmpipe).
    pub is_software_adapter: bool,
    /// Successful nonempty sensing executions completed by the GPU provider.
    pub dispatches: u64,
    /// Lifetime channel saturations observed by the GPU-backed world.
    pub saturations: u64,
    /// Actual paired CPU/GPU sensor values observed, rather than planned work.
    pub sensor_values_compared: u64,
}

impl SenseLaneParityReport {
    /// Convert this report into gate evidence for manifest embedding.
    #[must_use]
    pub fn to_gate_evidence(&self) -> SenseGateEvidenceV0 {
        SenseGateEvidenceV0 {
            target: self.target.clone(),
            adapter: self.adapter.clone(),
            vendor: self.vendor,
            device: self.device,
            backend: self.backend.clone(),
            driver: self.driver.clone(),
            driver_info: self.driver_info.clone(),
            shader_digest: self.shader_digest.clone(),
            build_identity: self.build_identity.clone(),
            toolchain_identity: self.toolchain_identity.clone(),
            fixture_digest: self.fixture_digest.clone(),
            tick_count: self.tick_count,
            evidence_time_unix_ms: self.evidence_time_unix_ms,
            evidence_digest: self.evidence_digest.clone(),
            determinism: self.verdict,
            max_sensor_delta: self.max_sensor_delta,
        }
    }

    /// Save the report to a JSON file at the specified path.
    pub fn save_to_file(&self, path: &Path) -> Result<(), std::io::Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a report from a JSON file.
    pub fn load_from_file(path: &Path) -> Result<Self, std::io::Error> {
        let text = std::fs::read_to_string(path)?;
        let report: Self = serde_json::from_str(&text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(report)
    }
}

/// Compute grid uniforms from world state and active agent count.
#[must_use]
pub fn make_grid_uniforms(world: &WorldState, agent_count: usize) -> GridUniforms {
    let cfg = world.config();
    let world_width = cfg.world_width as f32;
    let world_height = cfg.world_height as f32;
    let cell_size = (cfg.food_cell_size as f32).max(1.0);
    let sense_radius = cfg.sense_radius;
    let cells_x = (world_width / cell_size).ceil().max(1.0) as u32;
    let cells_y = (world_height / cell_size).ceil().max(1.0) as u32;

    GridUniforms {
        world_width,
        world_height,
        cell_size,
        sense_radius,
        cells_x,
        cells_y,
        agent_count: agent_count as u32,
        _pad: 0,
    }
}

/// Pack simulation agent states into the 96-byte `AgentGpuData` layout.
#[must_use]
pub fn pack_agent_gpu_data(world: &WorldState, handles: &[AgentId]) -> Vec<AgentGpuData> {
    let agent_count = handles.len();
    let mut data = Vec::with_capacity(agent_count);
    let columns = world.agents().columns();
    let positions = columns.positions();
    let headings = columns.headings();
    let colors = columns.colors();
    let healths = columns.health();

    let work_units = world.work_eye_units();
    let work_fovs = world.work_eye_fov();
    let work_traits = world.work_trait_modifiers();
    let work_wheels = world.work_peak_wheel_outputs();
    let work_sounds = world.work_sound_emitters();

    let runtime = world.runtime();

    for (idx, &handle) in handles.iter().enumerate() {
        let pos = if idx < positions.len() {
            positions[idx]
        } else {
            Position::default()
        };
        let heading = if idx < headings.len() {
            headings[idx]
        } else {
            0.0
        };
        let color = if idx < colors.len() {
            colors[idx]
        } else {
            [1.0, 1.0, 1.0]
        };
        let health = if idx < healths.len() {
            healths[idx]
        } else {
            1.0
        };

        let (eye_units, eye_fov, eye_sensitivity, wheel_effort, sound_emitter) =
            if idx < work_units.len() && idx < work_fovs.len() && idx < work_traits.len() {
                (
                    work_units[idx],
                    work_fovs[idx],
                    work_traits[idx].eye,
                    work_wheels[idx],
                    work_sounds[idx],
                )
            } else if let Some(rt) = runtime.get(handle) {
                let mut units = [[0.0; 2]; 4];
                let mut fovs = [1.0; 4];
                for e in 0..4 {
                    let raw_dir = heading + rt.eye_direction[e];
                    let dir = (raw_dir + std::f32::consts::PI)
                        .rem_euclid(2.0 * std::f32::consts::PI)
                        - std::f32::consts::PI;
                    units[e] = [dir.cos(), dir.sin()];
                    fovs[e] = rt.eye_fov[e];
                }
                (
                    units,
                    fovs,
                    rt.trait_modifiers.eye,
                    rt.outputs.peak_wheel_output(),
                    rt.sound_multiplier,
                )
            } else {
                ([[0.0; 2]; 4], [1.0; 4], 1.0, 0.0, 0.0)
            };

        data.push(AgentGpuData {
            pos_x: pos.x,
            pos_y: pos.y,
            heading_unit_x: heading.cos(),
            heading_unit_y: heading.sin(),
            eye_unit_0_x: eye_units[0][0],
            eye_unit_0_y: eye_units[0][1],
            eye_unit_1_x: eye_units[1][0],
            eye_unit_1_y: eye_units[1][1],
            eye_unit_2_x: eye_units[2][0],
            eye_unit_2_y: eye_units[2][1],
            eye_unit_3_x: eye_units[3][0],
            eye_unit_3_y: eye_units[3][1],
            eye_fov_0: eye_fov[0],
            eye_fov_1: eye_fov[1],
            eye_fov_2: eye_fov[2],
            eye_fov_3: eye_fov[3],
            color_r: color[0],
            color_g: color[1],
            color_b: color[2],
            eye_sensitivity,
            wheel_effort,
            sound_emitter,
            target_health: health,
            _pad: 0.0,
        });
    }

    data
}

/// Production GPU compute sensory accumulation provider.
pub struct GpuSenseProvider {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: GpuSensePipeline,
    completed_dispatches: Arc<AtomicU64>,
}

impl GpuSenseProvider {
    /// Create a new GPU sense provider wrapping the device, queue, and pipeline.
    #[must_use]
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipeline: GpuSensePipeline,
    ) -> Self {
        Self {
            device,
            queue,
            pipeline,
            completed_dispatches: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Access the underlying pipeline.
    #[must_use]
    pub fn pipeline(&self) -> &GpuSensePipeline {
        &self.pipeline
    }
}

impl SenseProvider for GpuSenseProvider {
    fn compute_accumulators(
        &mut self,
        world: &WorldState,
        handles: &[AgentId],
    ) -> Result<Vec<SenseAccum>, SenseProviderError> {
        if handles.is_empty() {
            return Ok(Vec::new());
        }
        let agents = pack_agent_gpu_data(world, handles);
        let uniforms = make_grid_uniforms(world, handles.len());
        let outputs = self
            .pipeline
            .execute_sense(&self.device, &self.queue, &agents, uniforms)
            .map_err(|e| SenseProviderError::Execution(format!("{e:?}")))?;
        self.completed_dispatches.fetch_add(1, Ordering::Relaxed);
        Ok(outputs.into_iter().map(|o| o.to_sense_accum()).collect())
    }
}

/// Injected fault mode to prove negative gate behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultInjectionMode {
    /// Diagnostic density perturbation; does not replace the shader's `acos`.
    DriverAcos,
    /// Diagnostic smell perturbation; does not execute f32 shader accumulation.
    F32Accumulator,
    /// Injected fault: deterministic scalar delta on first agent's channel.
    PerturbChannel,
}

/// Fault-injected variant of the GPU sense provider for negative verification.
pub struct FaultInjectedGpuSenseProvider {
    provider: GpuSenseProvider,
    mode: FaultInjectionMode,
}

impl FaultInjectedGpuSenseProvider {
    /// Create a new fault-injected provider with the specified mode.
    #[must_use]
    pub fn new(provider: GpuSenseProvider, mode: FaultInjectionMode) -> Self {
        Self { provider, mode }
    }
}

impl SenseProvider for FaultInjectedGpuSenseProvider {
    fn compute_accumulators(
        &mut self,
        world: &WorldState,
        handles: &[AgentId],
    ) -> Result<Vec<SenseAccum>, SenseProviderError> {
        let mut accumulators = self.provider.compute_accumulators(world, handles)?;
        match self.mode {
            FaultInjectionMode::DriverAcos => {
                // Accumulator diagnostic only; a real driver-acos shader fault remains open.
                for accum in &mut accumulators {
                    if accum.density[0] > 0 {
                        accum.density[0] ^= 0x01; // flip least significant bit
                    } else {
                        accum.density[0] = 1;
                    }
                }
            }
            FaultInjectionMode::F32Accumulator => {
                // Accumulator diagnostic only; this does not run a reordered f32 shader.
                for accum in &mut accumulators {
                    if accum.smell > 0 {
                        accum.smell = accum.smell.saturating_add(0x10); // 16 fixed-point units
                    }
                }
            }
            FaultInjectionMode::PerturbChannel => {
                if let Some(first) = accumulators.first_mut() {
                    first.sound = first.sound.saturating_add(104857); // ~0.1 in fixed-point
                }
            }
        }
        Ok(accumulators)
    }
}

/// Probe if a GPU adapter is available on the host without creating pipelines or allocating buffers.
#[must_use]
pub fn probe_gpu_adapter() -> Option<wgpu::AdapterInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()
    .map(|a| a.get_info())
}

/// Initialize the GPU sense pipeline and provider with the host adapter.
///
/// Returns [`GpuSenseError::NoAdapter`] if no compatible adapter is found.
pub fn init_gpu_sense_provider() -> Result<(GpuSenseProvider, wgpu::AdapterInfo), GpuSenseError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) {
        Ok(adapter) => adapter,
        Err(_) => return Err(GpuSenseError::NoAdapter),
    };

    let (device, queue) =
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
            Ok(pair) => pair,
            Err(e) => return Err(GpuSenseError::Device(e.to_string())),
        };

    let adapter_info = adapter.get_info();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let pipeline = GpuSensePipeline::new(&device, &adapter_info, PRODUCTION_WORKGROUP_SIZE)?;
    let provider = GpuSenseProvider::new(device, queue, pipeline);

    Ok((provider, adapter_info))
}

/// Verify whether an adapter is certified bit-exact against reference evidence.
///
/// Software adapters (CPU device_type, e.g. llvmpipe, lavapipe) are NEVER certified
/// as exact hardware lanes regardless of parity report contents.
#[must_use]
pub fn is_adapter_certified_exact(
    adapter_info: &wgpu::AdapterInfo,
    parity_report_path: Option<&Path>,
) -> Option<SenseGateEvidenceV0> {
    if adapter_info.device_type == wgpu::DeviceType::Cpu {
        return None;
    }
    let default_path = Path::new("ci/fixtures/sense_lane_parity.json");
    let path = parity_report_path.unwrap_or(default_path);
    let report = SenseLaneParityReport::load_from_file(path).ok()?;
    certified_report_evidence(
        &report,
        adapter_info,
        ParityBuildIdentity::current().as_ref(),
    )
}

/// Build-time evidence only: runtime environment changes cannot recertify a binary.
/// Missing provenance refuses certification but does not prevent exploratory parity runs.
#[derive(Clone)]
struct ParityBuildIdentity {
    target: String,
    build: String,
    toolchain: String,
}

impl ParityBuildIdentity {
    fn current() -> Option<Self> {
        Self::from_embedded(
            option_env!("SCRIPTBOTS_SOURCE_TREE_CLEAN"),
            option_env!("SCRIPTBOTS_SOURCE_REVISION").or(option_env!("RCH_BUILD_SOURCE")),
            option_env!("SCRIPTBOTS_BUILD_TARGET"),
            option_env!("SCRIPTBOTS_RUSTC_VV"),
        )
    }

    fn from_embedded(
        clean: Option<&str>,
        revision: Option<&str>,
        target: Option<&str>,
        toolchain: Option<&str>,
    ) -> Option<Self> {
        if clean != Some("true") {
            return None;
        }
        let revision = revision?;
        if !matches!(revision.len(), 40 | 64)
            || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return None;
        }
        let target = target?;
        let toolchain = toolchain?;
        if target.trim().is_empty()
            || !toolchain.lines().any(|line| line.starts_with("release: "))
            || !toolchain
                .lines()
                .any(|line| line.starts_with("commit-hash: "))
            || !toolchain.lines().any(|line| line.starts_with("host: "))
        {
            return None;
        }
        // Length-delimited JSON binds the source, lockfile, manifests and this crate's
        // feature/code-generation settings without ambiguous string concatenation.
        let inputs = serde_json::json!({
            "schema": "scriptbots.sense-build.v1",
            "revision": revision,
            "lockfile": blake3::hash(include_bytes!("../../../Cargo.lock")).to_hex().as_str(),
            "workspace": blake3::hash(include_bytes!("../../../Cargo.toml")).to_hex().as_str(),
            "crate": blake3::hash(include_bytes!("../Cargo.toml")).to_hex().as_str(),
            "compute_binning": cfg!(feature = "compute_binning"),
            "perf_counters": cfg!(feature = "perf_counters"),
            "debug_assertions": cfg!(debug_assertions),
            "rustflags": option_env!("CARGO_ENCODED_RUSTFLAGS").or(option_env!("RUSTFLAGS")),
        });
        let bytes = serde_json::to_vec(&inputs).ok()?;
        Some(Self {
            target: target.to_owned(),
            build: blake3::hash(&bytes).to_hex().to_string(),
            toolchain: toolchain.to_owned(),
        })
    }
}

fn certified_report_evidence(
    report: &SenseLaneParityReport,
    adapter: &wgpu::AdapterInfo,
    build: Option<&ParityBuildIdentity>,
) -> Option<SenseGateEvidenceV0> {
    let build = build?;
    let valid_world_digest = |digest: &str| {
        digest.len() == (u64::BITS / 4) as usize
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && digest.bytes().any(|byte| byte != b'0')
    };
    if adapter.device_type == wgpu::DeviceType::Cpu
        || report.is_software_adapter
        || adapter.name.trim().is_empty()
        || adapter.driver.trim().is_empty()
        || adapter.driver_info.trim().is_empty()
        || report.adapter != adapter.name
        || report.vendor != adapter.vendor
        || report.device != adapter.device
        || report.backend != format!("{:?}", adapter.backend)
        || report.driver != adapter.driver
        || report.driver_info != adapter.driver_info
        || report.target != build.target
        || report.build_identity != build.build
        || report.toolchain_identity != build.toolchain
        || report.shader_digest != crate::sense_wgsl::sense_shader_digest()
        || report.fixture_digest != parity_fixture_digest(&SenseParityOptions::default()).ok()?
        || report.tick_count < SenseParityOptions::default().tick_count
        || report.dispatches != report.tick_count
        || report.sensor_values_compared == 0
        || report.saturations != 0
        || report.evidence_time_unix_ms == 0
        || report.verdict != SenseDeterminism::Exact
        || report.first_divergence.is_some()
        || !report.max_sensor_delta.is_finite()
        || report.max_sensor_delta != 0.0
        || !valid_world_digest(&report.cpu_digest)
        || report.cpu_digest != report.gpu_digest
        || report.evidence_digest != compute_evidence_digest(report).ok()?
    {
        return None;
    }
    Some(report.to_gate_evidence())
}

fn parity_fixture_digest(options: &SenseParityOptions) -> Result<String, serde_json::Error> {
    let fixture = serde_json::json!({
        "schema": "scriptbots.sense-fixture.v1",
        "config": ScriptBotsConfig { rng_seed: Some(options.seed), ..Default::default() },
        "population": options.population,
        "placement": "x=100+i*25;y=100+i*20;AgentData::default",
        "fault_injection": options.fault_injection.map(|mode| format!("{mode:?}")),
    });
    Ok(blake3::hash(&serde_json::to_vec(&fixture)?)
        .to_hex()
        .to_string())
}

/// Errors returned by the sensory parity verification runner.
#[derive(Debug, thiserror::Error)]
pub enum SenseParityError {
    /// wgpu initialization or execution error.
    #[error("GPU sensing error: {0}")]
    Gpu(#[from] GpuSenseError),
    /// Simulation error in core.
    #[error("simulation error: {0}")]
    Simulation(String),
    /// Parity verification failed.
    #[error(
        "parity gate failed: first divergence at tick {tick}, agent {agent_uid}, delta {delta}"
    )]
    Divergence {
        /// Tick where divergence occurred.
        tick: u64,
        /// Stable agent UID.
        agent_uid: u64,
        /// Magnitude of difference.
        delta: f32,
    },
    /// IO error while writing reports.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Configuration options for the CPU-vs-GPU parity check runner.
#[derive(Debug, Clone)]
pub struct SenseParityOptions {
    /// Number of simulation ticks to run (default 1,000).
    pub tick_count: u64,
    /// Seed used for deterministic world initialization.
    pub seed: u64,
    /// Population size to spawn.
    pub population: usize,
    /// Optional fault injection mode for negative gate tests.
    pub fault_injection: Option<FaultInjectionMode>,
}

impl Default for SenseParityOptions {
    fn default() -> Self {
        Self {
            tick_count: 1_000,
            seed: 42,
            population: 20,
            fault_injection: None,
        }
    }
}

/// Hash every report field except its own digest using the versioned report wire.
/// This detects corruption; it does not authenticate the author of a report.
pub fn compute_evidence_digest(
    report: &SenseLaneParityReport,
) -> Result<String, serde_json::Error> {
    let mut canonical = report.clone();
    canonical.evidence_digest.clear();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"scriptbots.sense-parity.v1\0");
    hasher.update(&serde_json::to_vec(&canonical)?);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Execute a complete CPU-vs-GPU parity verification run across the specified tick count.
pub fn run_sense_parity_check(
    options: &SenseParityOptions,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    adapter_info: &wgpu::AdapterInfo,
) -> Result<SenseLaneParityReport, SenseParityError> {
    let start_time_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let build = ParityBuildIdentity::current();
    let target = build
        .as_ref()
        .map_or("unavailable", |identity| &identity.target)
        .to_owned();
    let toolchain = build
        .as_ref()
        .map_or("unavailable", |identity| &identity.toolchain)
        .to_owned();
    let build_identity = build
        .as_ref()
        .map_or("unavailable", |identity| &identity.build)
        .to_owned();
    let shader_digest = crate::sense_wgsl::sense_shader_digest().to_string();
    let fixture_digest = parity_fixture_digest(options)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let is_software = adapter_info.device_type == wgpu::DeviceType::Cpu;

    // Structured start diagnostics
    tracing::info!(
        target: "scriptbots::sense::parity",
        adapter = %adapter_info.name,
        backend = ?adapter_info.backend,
        ticks = options.tick_count,
        population = options.population,
        fault_injection = ?options.fault_injection,
        is_software = is_software,
        "Starting CPU-vs-GPU sense parity gate"
    );

    let config = ScriptBotsConfig {
        rng_seed: Some(options.seed),
        ..Default::default()
    };

    let mut world_cpu =
        WorldState::new(config.clone()).map_err(|e| SenseParityError::Simulation(e.to_string()))?;
    let mut world_gpu =
        WorldState::new(config).map_err(|e| SenseParityError::Simulation(e.to_string()))?;

    // Spawn initial populations identically
    for i in 0..options.population {
        let x = 100.0 + (i as f32) * 25.0;
        let y = 100.0 + (i as f32) * 20.0;
        let agent = AgentData {
            position: Position { x, y },
            ..AgentData::default()
        };
        let _ = world_cpu
            .try_spawn_agent(agent)
            .map_err(|e| SenseParityError::Simulation(format!("{e:?}")))?;
        let _ = world_gpu
            .try_spawn_agent(agent)
            .map_err(|e| SenseParityError::Simulation(format!("{e:?}")))?;
    }

    let pipeline = GpuSensePipeline::new(device, adapter_info, PRODUCTION_WORKGROUP_SIZE)?;
    let base_gpu_provider = GpuSenseProvider::new(Arc::clone(device), Arc::clone(queue), pipeline);
    let completed_dispatches = Arc::clone(&base_gpu_provider.completed_dispatches);

    if let Some(fault) = options.fault_injection {
        world_gpu.set_sense_provider(Box::new(FaultInjectedGpuSenseProvider::new(
            base_gpu_provider,
            fault,
        )));
    } else {
        world_gpu.set_sense_provider(Box::new(base_gpu_provider));
    }

    let mut max_sensor_delta: f32 = 0.0;
    let mut first_divergence: Option<SenseDivergenceCoordinates> = None;
    let mut sensor_values_compared = 0_u64;

    for tick in 0..options.tick_count {
        world_cpu
            .step()
            .map_err(|e| SenseParityError::Simulation(format!("{e:?}")))?;
        world_gpu
            .step()
            .map_err(|e| SenseParityError::Simulation(format!("{e:?}")))?;

        // Compare sensor arrays across all live agents
        let handles: Vec<AgentId> = world_cpu.agents().iter_handles().collect();
        for (agent_idx, handle) in handles.into_iter().enumerate() {
            let cpu_rt = world_cpu.agent_runtime(handle);
            let gpu_rt = world_gpu.agent_runtime(handle);

            if let (Some(c_rt), Some(g_rt)) = (cpu_rt, gpu_rt) {
                let uid = world_cpu.agent_uid(handle).map_or(0, |u| u.get());
                for s in 0..scriptbots_core::INPUT_SIZE {
                    sensor_values_compared = sensor_values_compared.saturating_add(1);
                    let c_val = c_rt.sensors[s];
                    let g_val = g_rt.sensors[s];
                    let delta = (c_val - g_val).abs();
                    if delta > max_sensor_delta {
                        max_sensor_delta = delta;
                    }
                    if delta > 0.0 && first_divergence.is_none() {
                        first_divergence = Some(SenseDivergenceCoordinates {
                            tick,
                            agent_uid: uid,
                            agent_index: agent_idx,
                            sensor_index: s,
                            cpu_sensor_value: c_val,
                            gpu_sensor_value: g_val,
                            sensor_delta: delta,
                            cpu_accumulator: None,
                            gpu_accumulator: None,
                        });
                        tracing::warn!(
                            target: "scriptbots::sense::parity",
                            tick = tick,
                            agent_uid = uid,
                            sensor = s,
                            cpu_value = c_val,
                            gpu_value = g_val,
                            delta = delta,
                            "Observed first divergence between CPU and GPU sense lanes"
                        );
                    }
                }
            }
        }
    }

    let cpu_digest = world_cpu
        .world_digest_v1()
        .map_err(|error| SenseParityError::Simulation(format!("CPU digest unavailable: {error}")))?
        .overall;
    let gpu_digest = world_gpu
        .world_digest_v1()
        .map_err(|error| SenseParityError::Simulation(format!("GPU digest unavailable: {error}")))?
        .overall;

    // Core exposes a lifetime total: adding it again after each tick double-counts.
    let saturations = world_gpu.sense_saturations_total();
    let dispatches = completed_dispatches.load(Ordering::Relaxed);
    let verdict = if first_divergence.is_none()
        && cpu_digest == gpu_digest
        && sensor_values_compared > 0
        && dispatches == options.tick_count
        && saturations == 0
    {
        SenseDeterminism::Exact
    } else {
        SenseDeterminism::Approximate
    };

    let mut report = SenseLaneParityReport {
        target,
        adapter: adapter_info.name.clone(),
        vendor: adapter_info.vendor,
        device: adapter_info.device,
        backend: format!("{:?}", adapter_info.backend),
        driver: adapter_info.driver.clone(),
        driver_info: adapter_info.driver_info.clone(),
        shader_digest,
        build_identity,
        toolchain_identity: toolchain,
        fixture_digest,
        tick_count: options.tick_count,
        evidence_time_unix_ms: start_time_ms,
        evidence_digest: String::new(),
        cpu_digest,
        gpu_digest,
        verdict,
        first_divergence,
        max_sensor_delta,
        is_software_adapter: is_software,
        dispatches,
        saturations,
        sensor_values_compared,
    };
    report.evidence_digest = compute_evidence_digest(&report)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;

    tracing::info!(
        target: "scriptbots::sense::parity",
        verdict = ?report.verdict,
        max_sensor_delta = report.max_sensor_delta,
        has_divergence = report.first_divergence.is_some(),
        dispatches = report.dispatches,
        sensor_values_compared = report.sensor_values_compared,
        saturations = report.saturations,
        build_provenance_available = build.is_some(),
        cpu_digest = %report.cpu_digest,
        gpu_digest = %report.gpu_digest,
        "Completed CPU-vs-GPU sense parity run"
    );

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GPU_TEST_MUTEX;

    #[test]
    fn test_sense_report_serialization_roundtrip() {
        let report = SenseLaneParityReport {
            target: "x86_64-unknown-linux-gnu".to_string(),
            adapter: "MockAdapter".to_string(),
            vendor: 0x1002,
            device: 0x73ff,
            backend: "Vulkan".to_string(),
            driver: "Mesa 24.0.0".to_string(),
            driver_info: "radv".to_string(),
            shader_digest: "abcd1234".to_string(),
            build_identity: "0.1.0".to_string(),
            toolchain_identity: "rustc 1.89".to_string(),
            fixture_digest: "blake3:fixture".to_string(),
            tick_count: 1000,
            evidence_time_unix_ms: 1700000000000,
            evidence_digest: "fe1234".to_string(),
            cpu_digest: "cpu_dig".to_string(),
            gpu_digest: "gpu_dig".to_string(),
            verdict: SenseDeterminism::Exact,
            first_divergence: None,
            max_sensor_delta: 0.0,
            is_software_adapter: false,
            dispatches: 1000,
            saturations: 0,
            sensor_values_compared: 500_000,
        };

        let json = serde_json::to_string(&report).expect("serialize");
        let roundtrip: SenseLaneParityReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(report, roundtrip);
        let mut missing_observation = serde_json::to_value(&report).expect("report JSON");
        missing_observation
            .as_object_mut()
            .expect("report object")
            .remove("sensor_values_compared");
        assert!(
            serde_json::from_value::<SenseLaneParityReport>(missing_observation).is_err(),
            "an unrecorded observation must not silently become an observed zero"
        );
    }

    #[test]
    fn test_gate_evidence_derivation() {
        let report = SenseLaneParityReport {
            target: "x86_64-unknown-linux-gnu".to_string(),
            adapter: "MockAdapter".to_string(),
            vendor: 0x1002,
            device: 0x73ff,
            backend: "Vulkan".to_string(),
            driver: "Mesa 24.0.0".to_string(),
            driver_info: "radv".to_string(),
            shader_digest: "abcd1234".to_string(),
            build_identity: "0.1.0".to_string(),
            toolchain_identity: "rustc 1.89".to_string(),
            fixture_digest: "blake3:fixture".to_string(),
            tick_count: 1000,
            evidence_time_unix_ms: 1700000000000,
            evidence_digest: "fe1234".to_string(),
            cpu_digest: "cpu_dig".to_string(),
            gpu_digest: "gpu_dig".to_string(),
            verdict: SenseDeterminism::Exact,
            first_divergence: None,
            max_sensor_delta: 0.0,
            is_software_adapter: false,
            dispatches: 1000,
            saturations: 0,
            sensor_values_compared: 500_000,
        };

        let evidence = report.to_gate_evidence();
        assert_eq!(evidence.target, report.target);
        assert_eq!(evidence.determinism, SenseDeterminism::Exact);
        assert_eq!(evidence.max_sensor_delta, 0.0);
    }

    #[test]
    fn test_divergence_coordinates_tracking() {
        let coords = SenseDivergenceCoordinates {
            tick: 42,
            agent_uid: 101,
            agent_index: 3,
            sensor_index: 5,
            cpu_sensor_value: 0.75,
            gpu_sensor_value: 0.80,
            sensor_delta: 0.05,
            cpu_accumulator: Some(vec![1000, 2000]),
            gpu_accumulator: Some(vec![1000, 2100]),
        };

        assert_eq!(coords.tick, 42);
        assert_eq!(coords.agent_uid, 101);
        assert!((coords.sensor_delta - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_fault_injected_provider_causes_divergence() {
        let _guard = GPU_TEST_MUTEX.lock().unwrap();
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(adapter) => adapter,
                Err(_) => {
                    eprintln!("Skipping GPU fault injection test: no wgpu adapter available");
                    return;
                }
            };

        let (device, queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
                Ok(pair) => pair,
                Err(_) => {
                    eprintln!("Skipping GPU fault injection test: request_device failed");
                    return;
                }
            };

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let info = adapter.get_info();

        let options = SenseParityOptions {
            tick_count: 5,
            seed: 42,
            population: 10,
            fault_injection: Some(FaultInjectionMode::PerturbChannel),
        };

        let report = run_sense_parity_check(&options, &device, &queue, &info)
            .expect("parity run with fault injection");

        assert_eq!(
            report.verdict,
            SenseDeterminism::Approximate,
            "Fault-injected provider MUST produce an Approximate verdict"
        );
        assert!(
            report.first_divergence.is_some(),
            "Fault-injected provider MUST identify first divergence coordinates"
        );
        let div = report.first_divergence.unwrap();
        assert_eq!(div.tick, 0, "PerturbChannel should diverge on tick 0");
        assert!(div.sensor_delta > 0.0);
        assert_eq!(report.dispatches, options.tick_count);
        assert!(report.sensor_values_compared > 0);

        let empty = run_sense_parity_check(
            &SenseParityOptions {
                tick_count: 1,
                population: 0,
                ..SenseParityOptions::default()
            },
            &device,
            &queue,
            &info,
        )
        .expect("empty diagnostic run");
        assert_eq!(
            empty.tick_count, 1,
            "one science tick does not imply GPU work"
        );
        assert_eq!(empty.dispatches, 0);
        assert_eq!(empty.sensor_values_compared, 0);
        assert_eq!(empty.saturations, 0);
        assert_eq!(empty.verdict, SenseDeterminism::Approximate);
    }

    #[test]
    fn test_gpu_lane_documentation_no_drift() {
        let expected = generate_gpu_lane_documentation();
        let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| std::path::Path::new("."));
        let doc_path = repo_root.join("docs/gpu-lane.md");
        if doc_path.exists() {
            let actual = std::fs::read_to_string(&doc_path).expect("read docs/gpu-lane.md");
            assert_eq!(
                expected.trim(),
                actual.trim(),
                "docs/gpu-lane.md has drifted from generator in sense_parity.rs! Run generator or update docs."
            );
        }
    }

    #[test]
    fn test_is_adapter_certified_exact_rejects_software_adapter() {
        let info = wgpu::AdapterInfo {
            name: "llvmpipe (LLVM 18.1.8, 256 bits)".to_string(),
            vendor: 65541,
            device: 0,
            device_type: wgpu::DeviceType::Cpu,
            driver: "Mesa 24.2.8".to_string(),
            driver_info: "LLVM 18.1.8".to_string(),
            backend: wgpu::Backend::Vulkan,
        };
        assert!(is_adapter_certified_exact(&info, None).is_none());
    }

    /// Construct evidence, not simulated hardware: these tests qualify only the
    /// admission predicate, never an actual GPU or a 1,000-tick correctness run.
    fn exact_report_fixture() -> (
        SenseLaneParityReport,
        wgpu::AdapterInfo,
        ParityBuildIdentity,
    ) {
        let adapter = wgpu::AdapterInfo {
            name: "admission-test-device".to_owned(),
            vendor: 0x1002,
            device: 0x73ff,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: "test-driver".to_owned(),
            driver_info: "test-driver-version".to_owned(),
            backend: wgpu::Backend::Vulkan,
        };
        let build = ParityBuildIdentity {
            target: "test-target".to_owned(),
            build: "test-source-and-build".to_owned(),
            toolchain: "test-compiler".to_owned(),
        };
        let mut report = SenseLaneParityReport {
            target: build.target.clone(),
            adapter: adapter.name.clone(),
            vendor: adapter.vendor,
            device: adapter.device,
            backend: format!("{:?}", adapter.backend),
            driver: adapter.driver.clone(),
            driver_info: adapter.driver_info.clone(),
            shader_digest: crate::sense_wgsl::sense_shader_digest().to_owned(),
            build_identity: build.build.clone(),
            toolchain_identity: build.toolchain.clone(),
            fixture_digest: parity_fixture_digest(&SenseParityOptions::default()).unwrap(),
            tick_count: SenseParityOptions::default().tick_count,
            evidence_time_unix_ms: 1,
            evidence_digest: String::new(),
            cpu_digest: "1234567890abcdef".to_owned(),
            gpu_digest: "1234567890abcdef".to_owned(),
            verdict: SenseDeterminism::Exact,
            first_divergence: None,
            max_sensor_delta: 0.0,
            is_software_adapter: false,
            dispatches: SenseParityOptions::default().tick_count,
            saturations: 0,
            sensor_values_compared: 500_000,
        };
        report.evidence_digest = compute_evidence_digest(&report).unwrap();
        (report, adapter, build)
    }

    #[test]
    fn certification_build_identity_requires_explicit_complete_clean_build_inputs() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let compiler = "rustc test\nrelease: test\ncommit-hash: test\nhost: test-target\n";
        let input = [
            Some("true"),
            Some(revision),
            Some("test-target"),
            Some(compiler),
        ];
        let resolve = |args: [Option<&str>; 4]| {
            ParityBuildIdentity::from_embedded(args[0], args[1], args[2], args[3])
        };
        let valid = resolve(input).expect("declared complete inputs");
        assert_eq!(valid.target, "test-target");
        assert_eq!(valid.toolchain, compiler);
        assert_eq!(valid.build.len(), blake3::OUT_LEN * 2);
        for index in 0..input.len() {
            let mut missing = input;
            missing[index] = None;
            assert!(resolve(missing).is_none(), "missing build input {index}");
        }
        for (index, value) in [
            (0, "false"),
            (0, "unknown"),
            (1, "unknown"),
            (1, "0123456789abcdef0123456789abcdef01234567-overlay-1234"),
            (2, " "),
            (3, "rustc 1.89-nightly (fabricated default)"),
        ] {
            let mut invalid = input;
            invalid[index] = Some(value);
            assert!(
                resolve(invalid).is_none(),
                "invalid build input {index}: {value}"
            );
        }
        let mut changed = input;
        changed[1] = Some("abcdef0123456789abcdef0123456789abcdef01");
        assert_ne!(resolve(changed).unwrap().build, valid.build);
    }

    #[test]
    fn certification_requires_matching_adapter_build_and_observed_exact_parity() {
        let (report, adapter, build) = exact_report_fixture();
        assert_eq!(
            certified_report_evidence(&report, &adapter, Some(&build)),
            Some(report.to_gate_evidence()),
            "the positive path preserves the exact matched evidence"
        );
        assert!(certified_report_evidence(&report, &adapter, None).is_none());
        for field in ["adapter", "driver", "driver_info"] {
            let (mut anonymous, mut adapter, build) = exact_report_fixture();
            match field {
                "adapter" => {
                    adapter.name.clear();
                    anonymous.adapter.clear();
                }
                "driver" => {
                    adapter.driver.clear();
                    anonymous.driver.clear();
                }
                "driver_info" => {
                    adapter.driver_info.clear();
                    anonymous.driver_info.clear();
                }
                _ => unreachable!("unknown identity field"),
            }
            anonymous.evidence_digest = compute_evidence_digest(&anonymous).unwrap();
            assert!(
                certified_report_evidence(&anonymous, &adapter, Some(&build)).is_none(),
                "equal missing {field} values do not establish an observed identity"
            );
        }
        type ReportMutation = (&'static str, fn(&mut SenseLaneParityReport));
        let mutations: &[ReportMutation] = &[
            ("same vendor, other name", |r| r.adapter.push_str("-other")),
            ("same name, other vendor", |r| r.vendor += 1),
            ("same vendor/name, other device", |r| r.device += 1),
            ("backend", |r| r.backend = "Metal".to_owned()),
            ("driver", |r| r.driver.push_str("-changed")),
            ("driver info", |r| r.driver_info.push_str("-changed")),
            ("target", |r| r.target.push_str("-changed")),
            ("build", |r| r.build_identity.push_str("-changed")),
            ("compiler", |r| r.toolchain_identity.push_str("-changed")),
            ("shader", |r| r.shader_digest.clear()),
            ("fixture", |r| r.fixture_digest.clear()),
            ("short run", |r| {
                r.tick_count -= 1;
                r.dispatches = r.tick_count;
            }),
            ("zero ticks", |r| {
                r.tick_count = 0;
                r.dispatches = 0;
            }),
            ("dispatch count", |r| r.dispatches -= 1),
            ("no observations", |r| r.sensor_values_compared = 0),
            ("numeric saturation", |r| r.saturations = 1),
            ("missing timestamp", |r| r.evidence_time_unix_ms = 0),
            ("software report", |r| r.is_software_adapter = true),
            ("approximate verdict", |r| {
                r.verdict = SenseDeterminism::Approximate
            }),
            ("nonzero delta", |r| r.max_sensor_delta = 0.1),
            ("nonfinite delta", |r| r.max_sensor_delta = f32::NAN),
            ("unequal science", |r| {
                r.gpu_digest = "abcdef1234567890".to_owned()
            }),
            ("zero science", |r| {
                r.cpu_digest = "0000000000000000".to_owned();
                r.gpu_digest.clone_from(&r.cpu_digest);
            }),
            ("malformed science", |r| {
                r.cpu_digest = "cpu_digest_error".to_owned();
                r.gpu_digest.clone_from(&r.cpu_digest);
            }),
            ("divergence coordinates", |r| {
                r.first_divergence = Some(SenseDivergenceCoordinates {
                    tick: 1,
                    agent_uid: 1,
                    agent_index: 0,
                    sensor_index: 0,
                    cpu_sensor_value: 0.0,
                    gpu_sensor_value: 1.0,
                    sensor_delta: 1.0,
                    cpu_accumulator: None,
                    gpu_accumulator: None,
                })
            }),
        ];
        for (label, mutate) in mutations {
            let mut changed = report.clone();
            mutate(&mut changed);
            // Rehash so identity/outcome checks must discriminate independently
            // of the corruption check. A valid checksum is not certification.
            changed.evidence_digest = compute_evidence_digest(&changed).unwrap();
            assert!(
                certified_report_evidence(&changed, &adapter, Some(&build)).is_none(),
                "incorrectly certified {label}"
            );
        }
        let mut software = adapter;
        software.device_type = wgpu::DeviceType::Cpu;
        assert!(certified_report_evidence(&report, &software, Some(&build)).is_none());
    }

    #[test]
    fn evidence_digest_covers_every_serialized_report_field() {
        let (report, adapter, build) = exact_report_fixture();
        let wire = serde_json::to_value(&report).unwrap();
        for (name, value) in wire.as_object().unwrap() {
            let mut changed = wire.clone();
            let replacement = match value {
                serde_json::Value::String(text) => serde_json::json!(format!("{text}-changed")),
                serde_json::Value::Number(_) => serde_json::json!(42),
                serde_json::Value::Bool(flag) => serde_json::json!(!flag),
                serde_json::Value::Null => serde_json::to_value(SenseDivergenceCoordinates {
                    tick: 1,
                    agent_uid: 1,
                    agent_index: 0,
                    sensor_index: 0,
                    cpu_sensor_value: 0.0,
                    gpu_sensor_value: 1.0,
                    sensor_delta: 1.0,
                    cpu_accumulator: None,
                    gpu_accumulator: None,
                })
                .unwrap(),
                other => panic!("unhandled report field {name}: {other}"),
            };
            changed[name] = replacement;
            // Verdict has only two valid enum values.
            if name == "verdict" {
                changed[name] = serde_json::json!("approximate");
            }
            let changed: SenseLaneParityReport = serde_json::from_value(changed).unwrap();
            assert_ne!(
                &changed, &report,
                "mutation of {name} must change the report"
            );
            if name != "evidence_digest" {
                assert_ne!(
                    compute_evidence_digest(&changed).unwrap(),
                    report.evidence_digest,
                    "unbound field {name}"
                );
            }
            assert!(
                certified_report_evidence(&changed, &adapter, Some(&build)).is_none(),
                "unnoticed corruption in {name}"
            );
        }
        let mut longer = report;
        longer.tick_count += 1;
        longer.dispatches += 1;
        longer.evidence_digest = compute_evidence_digest(&longer).unwrap();
        assert!(certified_report_evidence(&longer, &adapter, Some(&build)).is_some());
    }
}

/// Generate authoritative markdown documentation for docs/gpu-lane.md.
#[allow(clippy::items_after_test_module)]
#[must_use]
pub fn generate_gpu_lane_documentation() -> String {
    r#"# GPU Compute Sense Lane & Parity Gate

> Status, hardware target matrix, determinism guarantees, and parity gate verification for GPU-accelerated sensing (bd-16g.15.2, bd-16g.15.3).

---

## 1. Executive Summary

GPU sensing provides an order-independent, fixed-point fast lane for agent perception in `rust_scriptbots`.
Floating-point sensor accumulation on GPUs is fundamentally non-deterministic across hardware vendors and workgroups due to non-associative floating-point addition and driver-dependent transcendental approximations (`sin`, `cos`, `atan2`, `acos`).

To prevent silent determinism drift:
1. **CPU Remains the Default**: Simulations run on the CPU reference implementation unless explicitly configured otherwise.
2. **Bit-Identical Fixed-Point Math**: Geometry is evaluated via dot products and Abramowitz & Stegun polynomial `acos` matching `crates/scriptbots-core/src/sense_fixed.rs` verbatim. Sensor terms are converted into 20-bit fixed-point integers (`to_fixed`) and accumulated into 64-bit integers using tree reductions in shared memory.
3. **Parity Gate Per Target**: Every supported adapter and driver combination must pass an end-to-end 1,000-tick CPU-vs-GPU parity verification run.
4. **Honest Reproducibility Labels**: Uncertified or approximate runs are marked `reproducible = false` in `RunManifestV3`, are tagged with startup and exit warnings, and are excluded from replay certification and competitive leaderboard rankings.

---

## 2. Hardware & Target Matrix

| Target Architecture | Graphics API | Candidate Adapters | Hardware Class | Verification Status | Determinism Verdict |
|----------------------|--------------|-----------------|----------------|---------------------|---------------------|
| `x86_64-unknown-linux-gnu` | Vulkan | AMD Radeon RX 7900 XTX | Discrete GPU | Unverified target | Unestablished |
| `x86_64-unknown-linux-gnu` | Vulkan | NVIDIA GeForce RTX 4090 | Discrete GPU | Unverified target | Unestablished |
| `x86_64-unknown-linux-gnu` | Vulkan | llvmpipe (Mesa) | Software CPU | Emulated (Excluded from Performance Claims) | Approximate |
| `aarch64-apple-darwin` | Metal | Apple M1/M2/M3/M4 (Family 7/8/9) | Integrated Apple Silicon | Unverified target | Unestablished |
| `x86_64-pc-windows-msvc` | DirectX 12 | NVIDIA GeForce RTX 3080/4080 | Discrete GPU | Unverified target | Unestablished |

No hardware certification is established by this table. The retained fixture is a
software-adapter report; a table row is not parity evidence. Runtime certification
requires a matching full adapter/driver/backend tuple, current shader, canonical
fixture, at least 1,000 ticks with actual sensor comparisons, equal nonzero science
digests, zero delta/no divergence, and the recomputed versioned report checksum.
Missing adapter/driver identity strings refuse certification even when both
the report and live adapter expose the same blank value.
That checksum detects corruption; it does not authenticate the report's author.

Builds seeking certification must embed `SCRIPTBOTS_SOURCE_TREE_CLEAN=true`, an
exact `SCRIPTBOTS_SOURCE_REVISION` (or plain-commit `RCH_BUILD_SOURCE`),
`SCRIPTBOTS_BUILD_TARGET`, and complete `SCRIPTBOTS_RUSTC_VV` output while compiling.
The identity also binds Cargo manifests/lockfile and local feature/code-generation
settings. Missing inputs remain `unavailable` in new reports and refuse certification.
These build-time declarations require independent build provenance review; runtime
environment variables cannot replace them. Legacy v0 checksums are not accepted.

> [!NOTE]
> **Software Adapter Exclusion**:
> Software fallback adapters like `llvmpipe` run on CPU threads emulating GPU pipelines. They are visibly classified as software adapters and excluded from all official GPU scaling and performance benchmarks.

---

## 3. CLI Policy & Flag Semantics

| Flag | Values | Default | Purpose |
|------|--------|---------|---------|
| `--sense-backend` | `cpu`, `gpu`, `auto` | `cpu` | Selects sensory execution backend. `cpu` is default. `gpu` requires compatible GPU. `auto` chooses verified GPU or falls back honestly to CPU. |
| `--allow-approximate-sense` | Flag (bool) | `false` | Explicit opt-in required to execute on uncertified GPU targets or approximate adapters. |

### Failure Modes Before Storage Writes:
- **No Adapter**: Passing `--sense-backend gpu` on a machine without a supported GPU returns a typed pre-storage error. Silent fallback to CPU is forbidden.
- **Uncertified Adapter**: Passing `--sense-backend gpu` on an uncertified target without `--allow-approximate-sense` returns a typed pre-storage error detailing required guidance.
- **Auto Selection**: `--sense-backend auto` probes for certified bit-exact GPU hardware. If certified, it selects GPU; otherwise, it honestly logs and selects the CPU lane.

---

## 4. Manifest & Downstream Propagation

When GPU sensing is classified as `Approximate`:
1. `RunManifestV3.reproducible` is forced to `false`.
2. `RunManifestV3.warnings` retains `"gpu sensing is approximate; run is not certified as reproducible"`.
3. `RunManifestV3.sense_policy` records `SensePolicyV0` with `SenseDeterminism::Approximate` and gate evidence.
4. **Replay Certification**: Replay verification checks refuse to certify approximate runs.
5. **Leaderboards**: Tournament rankings exclude approximate runs from competitive leaderboards.
6. **Exploration Export**: Raw data export remains permitted for scientific exploration.

---

## 5. Running the Parity Verification Gate

To run the current focused provider, policy and documentation checks:

```bash
# Execute the E2E parity runner
./scripts/e2e_gpu_sense_parity.sh
```

The helper:
1. Runs selected parity tests, including a five-tick accumulator-perturbation
   diagnostic and a one-tick empty-population case when an adapter is available.
   The provider test can skip execution when no adapter/device is available.
2. Checks application policy/manifest tests and the retained software-adapter
   fixture. It does not produce a fresh 1,000-tick report from that fixture.
3. Verifies this documentation against its generator.

The historical `DriverAcos` and `F32Accumulator` mode names currently select
integer accumulator perturbations, not altered shader implementations. Actual
driver-`acos` and f32-shader fault qualification remains open, along with the
1,000-tick hardware run and DSR scaling requirements of `bd-16g.15.3`.
`run_sense_parity_check` defaults to 1,000 ticks for callers that explicitly run
it; the helper's smaller diagnostic does not establish that acceptance.
"#.to_string()
}
