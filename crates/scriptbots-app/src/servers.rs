use std::{
    env,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{Arc, Mutex, mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use axum::extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::response::sse::{Event, Sse};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use fastmcp_rust::{
    Content, Cx, JsonRpcRequest, McpError, McpErrorCode, McpResult, NotificationSender,
    PendingRequests, RequestSender, Server, Session, ToolHandler,
};
use futures_util::stream::{Stream, StreamExt};
use scriptbots_core::PresetKind;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::Infallible;
use tokio::sync::watch;
use tokio_stream::wrappers::IntervalStream;
use tracing::{debug, error, info, warn};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use crate::ScenarioIdentityV0;
use crate::SharedWorld;
use crate::command::{
    CommandDrain, CommandSubmit, create_command_bus, make_command_drain, make_command_submit,
};
use crate::control::{
    AgentScoreEntry, CommandReporter, CommandStatusDto, ConfigSnapshot, ControlError,
    ControlHandle, DietClassDto, EventEntry, EventKind, HydrologySnapshot, KnobEntry, KnobUpdate,
    Scoreboard, SelectionModeDto, SelectionStateDto, SharedLatestSummary, SimulationStatusDto,
    SpeedRequest,
};
use scriptbots_core::{AgentDebugInfo, AgentDebugQuery, AgentDebugSort, Position, SelectionUpdate};
// keep image out of servers unless needed
use scriptbots_core::ConfigAuditEntry;
use scriptbots_core::TickSummaryDto;

/// Default loopback address for the REST control surface.
pub const DEFAULT_CONTROL_REST_ADDRESS: SocketAddr =
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8088);
/// Default loopback address for the MCP HTTP control surface.
pub const DEFAULT_CONTROL_MCP_HTTP_ADDRESS: SocketAddr =
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 8090);
/// Default Swagger UI route served by the REST control surface.
pub const DEFAULT_CONTROL_SWAGGER_PATH: &str = "/docs";

const CONTROL_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
const CONTROL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// Return the control CLI's default REST base URL from the server's socket authority.
#[must_use]
pub fn default_control_rest_base_url() -> String {
    format!("http://{DEFAULT_CONTROL_REST_ADDRESS}")
}

/// The last terminal frame actually PRESENTED to a user.
///
/// `GET /api/screenshot/ascii` used to answer by re-rasterizing terrain and food
/// into its own ≤96x48 ASCII grid — a THIRD renderer, agreeing with neither the
/// sub-cell canvas nor the flat map, and containing no panels at all. It was not a
/// low-fidelity screenshot; it was a different renderer's output wearing the
/// screenshot's name, which is the same defect `save_ascii_snapshot` had before
/// bd-2z0.14.2.6 item (1) replaced it (bd-0oro).
///
/// The buffer is published rather than a serialized string so the render path pays
/// only for a clone: ANSI/plain serialization happens in the handler, on the rare
/// request, through the SAME `terminal::export` functions the `S` key uses. One
/// serializer means the served bytes and the saved bytes cannot drift.
#[derive(Debug, Clone)]
pub struct PresentedTerminalFrame {
    /// The simulation tick the frame was drawn for.
    pub tick: u64,
    /// The exact buffer that was presented.
    pub buffer: ratatui::buffer::Buffer,
}

/// Wait-free publication slot for [`PresentedTerminalFrame`].
///
/// Same shape as [`SharedLatestSummary`]: readers never block, and a request that
/// lands mid-frame gets the previous frame rather than a torn one.
pub type SharedPresentedFrame = Arc<arc_swap::ArcSwapOption<PresentedTerminalFrame>>;

/// A fresh, empty presented-frame slot.
#[must_use]
pub fn empty_presented_frame() -> SharedPresentedFrame {
    Arc::new(arc_swap::ArcSwapOption::from(None))
}

/// Configuration for the hosted control surfaces.
#[derive(Debug, Clone)]
pub struct ControlServerConfig {
    pub rest_address: SocketAddr,
    pub swagger_path: String,
    pub rest_enabled: bool,
    pub mcp_transport: McpTransportConfig,
    /// The run's scenario identity, surfaced read-only at `GET /api/scenario`.
    /// `None` only for embedded/test servers with no manifest context.
    pub scenario: Option<Arc<ScenarioIdentityV0>>,
    /// Environment parsing failures retained until the fallible launch boundary.
    #[doc(hidden)]
    pub environment_errors: Vec<String>,
    /// Shared slot the terminal frontend publishes presented frames into and
    /// `GET /api/screenshot/ascii` reads.
    ///
    /// Carried on the config for the same reason `ControlRuntime::command_reporter`
    /// is carried on the runtime (bd-tgfz): the writer is the terminal frontend,
    /// several calls from where the server is built, and widening four signatures to
    /// reach it would touch files other panes are actively editing. Defaulted, so no
    /// existing `..ControlServerConfig::default()` construction changes.
    pub presented_frame: SharedPresentedFrame,
}

impl Default for ControlServerConfig {
    fn default() -> Self {
        Self {
            rest_address: DEFAULT_CONTROL_REST_ADDRESS,
            swagger_path: DEFAULT_CONTROL_SWAGGER_PATH.to_string(),
            rest_enabled: true,
            mcp_transport: McpTransportConfig::Http {
                bind_address: DEFAULT_CONTROL_MCP_HTTP_ADDRESS,
            },
            scenario: None,
            environment_errors: Vec::new(),
            presented_frame: empty_presented_frame(),
        }
    }
}

impl ControlServerConfig {
    /// Build configuration from environment variables and retain parse failures
    /// for validation at the launch/reservation boundary.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Some(addr) = read_control_environment(
            "SCRIPTBOTS_CONTROL_REST_ADDR",
            &mut config.environment_errors,
        ) {
            match addr.parse::<SocketAddr>() {
                Ok(addr) => config.rest_address = addr,
                Err(error) => config.environment_errors.push(format!(
                    "SCRIPTBOTS_CONTROL_REST_ADDR={addr:?} is not a socket address: {error}"
                )),
            }
        }

        if let Some(path) = read_control_environment(
            "SCRIPTBOTS_CONTROL_SWAGGER_PATH",
            &mut config.environment_errors,
        ) {
            let sanitized = sanitize_swagger_path(&path);
            if sanitized != path {
                warn!(original = %path, sanitized = %sanitized, "sanitized swagger path");
            }
            config.swagger_path = sanitized;
        }

        if let Some(flag) = read_control_environment(
            "SCRIPTBOTS_CONTROL_REST_ENABLED",
            &mut config.environment_errors,
        ) {
            match flag.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => config.rest_enabled = true,
                "0" | "false" | "no" | "off" => config.rest_enabled = false,
                other => config.environment_errors.push(format!(
                    "SCRIPTBOTS_CONTROL_REST_ENABLED={other:?} must be one of true/false, yes/no, on/off, or 1/0"
                )),
            }
        }

        let mut http_override = None;
        if let Some(addr) = read_control_environment(
            "SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR",
            &mut config.environment_errors,
        ) {
            match parse_mcp_socket_addr(&addr) {
                Some(parsed) => http_override = Some(parsed),
                None if is_https_url(&addr) => config.environment_errors.push(format!(
                    "SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR={addr:?} requests TLS, but the ScriptBots MCP server is plaintext HTTP; use http:// or a bare socket address"
                )),
                None => config.environment_errors.push(format!(
                    "SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR={addr:?} is not a supported HTTP socket address"
                )),
            }
        }

        if let Some(raw) =
            read_control_environment("SCRIPTBOTS_CONTROL_MCP", &mut config.environment_errors)
        {
            let trimmed = raw.trim();
            match trimmed.to_ascii_lowercase().as_str() {
                "disabled" | "off" | "false" | "0" => {
                    config.mcp_transport = McpTransportConfig::Disabled;
                }
                "http" | "" => {
                    let bind_address = http_override.unwrap_or(DEFAULT_CONTROL_MCP_HTTP_ADDRESS);
                    config.mcp_transport = McpTransportConfig::Http { bind_address };
                }
                _other => {
                    if let Some(addr) = parse_mcp_socket_addr(trimmed) {
                        config.mcp_transport = McpTransportConfig::Http { bind_address: addr };
                    } else if is_https_url(trimmed) {
                        config.environment_errors.push(format!(
                            "SCRIPTBOTS_CONTROL_MCP={raw:?} requests TLS, but the ScriptBots MCP server is plaintext HTTP; use http:// or a bare socket address"
                        ));
                    } else {
                        config.environment_errors.push(format!(
                            "SCRIPTBOTS_CONTROL_MCP={raw:?} must be disabled, http, or an HTTP socket address"
                        ));
                    }
                }
            }
        } else if let Some(addr) = http_override {
            config.mcp_transport = McpTransportConfig::Http { bind_address: addr };
        }

        config
    }

    /// Build and validate configuration from the process environment.
    pub fn try_from_env() -> Result<Self> {
        let config = Self::from_env();
        config.validate_environment()?;
        Ok(config)
    }

    fn validate_environment(&self) -> Result<()> {
        if self.environment_errors.is_empty() {
            Ok(())
        } else {
            Err(anyhow!(
                "invalid control server environment: {}",
                self.environment_errors.join("; ")
            ))
        }
    }
}

fn read_control_environment(name: &str, errors: &mut Vec<String>) -> Option<String> {
    match env::var(name) {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => {
            errors.push(format!("{name} is not valid Unicode"));
            None
        }
    }
}

fn sanitize_swagger_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return DEFAULT_CONTROL_SWAGGER_PATH.to_string();
    }
    if trimmed.starts_with('/') {
        trimmed.to_string()
    } else {
        format!("/{trimmed}")
    }
}

fn parse_mcp_socket_addr(value: &str) -> Option<SocketAddr> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    if is_https_url(trimmed) {
        return None;
    }

    if let Ok(addr) = trimmed.parse::<SocketAddr>() {
        return Some(addr);
    }

    let normalized = trimmed.strip_prefix("http://").unwrap_or(trimmed);
    let host_port = normalized.split('/').next().unwrap_or(normalized);
    host_port.parse::<SocketAddr>().ok()
}

fn is_https_url(value: &str) -> bool {
    value.trim().to_ascii_lowercase().starts_with("https://")
}

fn parse_id_list(raw: &str) -> Result<Vec<u64>, AppError> {
    let mut ids = Vec::new();
    for part in raw.split(',') {
        let id_text = part.trim();
        if id_text.is_empty() {
            continue;
        }
        match id_text.parse::<u64>() {
            Ok(value) => ids.push(value),
            Err(_) => {
                return Err(AppError::bad_request(format!(
                    "unable to parse agent id '{id_text}'"
                )));
            }
        }
    }
    Ok(ids)
}

/// Supported transports for the MCP server.
#[derive(Debug, Clone)]
pub enum McpTransportConfig {
    Disabled,
    Http { bind_address: SocketAddr },
}

struct ReservedControlListener {
    address: SocketAddr,
    listener: std::net::TcpListener,
}

/// Transactional socket reservation for the REST and MCP control surfaces.
///
/// Preparing a reservation binds every enabled listener before world or
/// storage construction. Launch consumes those exact listeners, eliminating
/// check-then-bind races and partial control-plane publication.
pub struct ControlServerReservation {
    config: ControlServerConfig,
    rest: Option<ReservedControlListener>,
    mcp: Option<ReservedControlListener>,
}

impl ControlServerReservation {
    /// Validate configuration and reserve every enabled control socket.
    pub fn prepare(config: ControlServerConfig) -> Result<Self> {
        config.validate_environment()?;

        let rest = if config.rest_enabled {
            Some(reserve_control_listener("REST", config.rest_address)?)
        } else {
            None
        };
        let mcp = match &config.mcp_transport {
            McpTransportConfig::Disabled => None,
            McpTransportConfig::Http { bind_address } => {
                Some(reserve_control_listener("MCP HTTP", *bind_address)?)
            }
        };

        Ok(Self { config, rest, mcp })
    }

    /// Launch the control runtime using the exact listeners reserved earlier.
    pub fn launch(
        self,
        world: SharedWorld,
        latest_summary: SharedLatestSummary,
    ) -> Result<(ControlRuntime, CommandDrain, CommandSubmit)> {
        ControlRuntime::launch_reserved_with_timeout(
            world,
            latest_summary,
            self,
            CONTROL_STARTUP_TIMEOUT,
        )
    }

    /// Actual REST address, including the assigned port when configured with port zero.
    #[must_use]
    pub fn rest_address(&self) -> Option<SocketAddr> {
        self.rest.as_ref().map(|reserved| reserved.address)
    }

    /// Actual MCP HTTP address, including the assigned port when configured with port zero.
    #[must_use]
    pub fn mcp_http_address(&self) -> Option<SocketAddr> {
        self.mcp.as_ref().map(|reserved| reserved.address)
    }
}

fn reserve_control_listener(name: &str, address: SocketAddr) -> Result<ReservedControlListener> {
    let listener = std::net::TcpListener::bind(address)
        .with_context(|| format!("failed to reserve {name} address {address}"))?;
    listener
        .set_nonblocking(true)
        .with_context(|| format!("failed to make reserved {name} listener nonblocking"))?;
    let actual_address = listener
        .local_addr()
        .with_context(|| format!("failed to inspect reserved {name} listener"))?;
    Ok(ReservedControlListener {
        address: actual_address,
        listener,
    })
}

/// Observable lifecycle state for a running control runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlRuntimeStatus {
    Starting,
    Running,
    Disabled,
    Stopped,
    Failed(String),
}

/// Runtime guard for background control servers.
///
/// Shutdown is a level-triggered `watch` flag rather than `Notify`:
/// `notify_waiters` wakes only already-registered waiters, so a shutdown
/// issued before the server tasks reach their await would be lost and the
/// join below would hang forever.
pub struct ControlRuntime {
    /// Lets whatever applies drained commands report their outcome.
    ///
    /// Carried here rather than threaded through the launch tuple because the
    /// applier lives in the renderer or the terminal frontend, several calls
    /// away from where the bus is built, and widening four signatures to reach
    /// it would touch files other panes are actively editing (bd-tgfz).
    command_reporter: CommandReporter,
    /// The slot the terminal frontend publishes presented frames into, shared with
    /// the REST server so `GET /api/screenshot/ascii` serves what was displayed
    /// instead of re-rasterizing the world (bd-2z0.14.2.6).
    presented_frame: SharedPresentedFrame,
    shutdown: watch::Sender<bool>,
    thread: Option<JoinHandle<Result<()>>>,
    status: watch::Receiver<ControlRuntimeStatus>,
    #[cfg(test)]
    _dummy_status_guard: Option<watch::Sender<ControlRuntimeStatus>>,
}

enum ControlStartupSignal {
    Ready,
    Failed(String),
}

impl ControlRuntime {
    /// Spawn the control runtime and return only after every enabled listener is bound.
    pub fn launch(
        world: SharedWorld,
        latest_summary: SharedLatestSummary,
        config: ControlServerConfig,
    ) -> Result<(Self, CommandDrain, CommandSubmit)> {
        ControlServerReservation::prepare(config)?.launch(world, latest_summary)
    }

    fn launch_reserved_with_timeout(
        world: SharedWorld,
        latest_summary: SharedLatestSummary,
        reservation: ControlServerReservation,
        startup_timeout: Duration,
    ) -> Result<(Self, CommandDrain, CommandSubmit)> {
        let (command_tx, command_rx) = create_command_bus(32);
        let command_drain = make_command_drain(command_rx);
        let command_submit = make_command_submit(command_tx.clone());
        let (shutdown, shutdown_rx) = watch::channel(false);
        let shutdown_for_thread = shutdown.clone();
        let (startup_tx, startup_rx) = mpsc::sync_channel(1);
        let (status_tx, status_rx) = watch::channel(ControlRuntimeStatus::Starting);
        let status_for_thread = status_tx.clone();
        // Taken before the reservation moves into the server thread, so the runtime
        // and the REST state hold THE SAME slot rather than two empty ones.
        let presented_frame = reservation.config.presented_frame.clone();
        let handle = ControlHandle::new(world.clone(), command_tx.clone(), latest_summary);
        // Derived before the handle moves into the axum state: the reporter
        // shares the ledger, so an outcome recorded through it is the same
        // receipt a REST client polls.
        let command_reporter = handle.command_reporter();

        let thread = thread::Builder::new()
            .name("scriptbots-control".into())
            .spawn(move || -> Result<()> {
                let result = match tokio::runtime::Builder::new_multi_thread()
                    .thread_name("scriptbots-control-rt")
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime.block_on(run_control_servers(
                        handle,
                        reservation,
                        shutdown_for_thread,
                        shutdown_rx,
                        startup_tx,
                        status_for_thread.clone(),
                    )),
                    Err(source) => {
                        let error =
                            anyhow!("failed to build Tokio runtime for control servers: {source}");
                        let _ = startup_tx.send(ControlStartupSignal::Failed(format!("{error:#}")));
                        Err(error)
                    }
                };
                publish_control_runtime_result(&status_for_thread, &result);
                result
            })
            .context("failed to spawn control runtime thread")?;

        match startup_rx.recv_timeout(startup_timeout) {
            Ok(ControlStartupSignal::Ready) => Ok((
                Self {
                    command_reporter,
                    presented_frame,
                    shutdown,
                    thread: Some(thread),
                    status: status_rx,
                    #[cfg(test)]
                    _dummy_status_guard: None,
                },
                command_drain,
                command_submit,
            )),
            Ok(ControlStartupSignal::Failed(detail)) => match join_control_thread(thread) {
                Ok(()) => Err(anyhow!("control server startup failed: {detail}")),
                Err(error) => {
                    Err(error).context(format!("control server startup failed: {detail}"))
                }
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = shutdown.send(true);
                handoff_control_reaper(thread);
                Err(anyhow!(
                    "control servers did not report readiness within {} ms",
                    startup_timeout.as_millis()
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => match join_control_thread(thread) {
                Ok(()) => Err(anyhow!(
                    "control runtime exited before reporting startup readiness"
                )),
                Err(error) => {
                    Err(error).context("control runtime failed before reporting startup readiness")
                }
            },
        }
    }

    /// Subscribe to prompt runtime termination/failure notifications.
    #[must_use]
    pub fn subscribe_status(&self) -> watch::Receiver<ControlRuntimeStatus> {
        self.status.clone()
    }

    /// A reporter the applier uses to record what became of a drained command.
    #[must_use]
    pub fn command_reporter(&self) -> CommandReporter {
        Arc::clone(&self.command_reporter)
    }

    /// The slot a frontend publishes its presented frames into.
    ///
    /// Handed out as a clone of the Arc, so the frontend writes exactly where
    /// `GET /api/screenshot/ascii` reads (bd-2z0.14.2.6).
    #[must_use]
    pub fn presented_frame_slot(&self) -> SharedPresentedFrame {
        Arc::clone(&self.presented_frame)
    }

    /// Return the latest runtime lifecycle state without blocking.
    #[must_use]
    pub fn status(&self) -> ControlRuntimeStatus {
        self.status.borrow().clone()
    }

    /// Check runtime health without blocking or trusting a stale cached status.
    pub fn health(&self) -> std::result::Result<(), String> {
        control_runtime_health(&self.status)
    }

    /// Build an owned, dependency-neutral health callback for UI runtimes.
    pub fn health_probe(
        &self,
    ) -> impl Fn() -> std::result::Result<(), String> + Clone + Send + Sync + 'static {
        let status = self.status.clone();
        move || control_runtime_health(&status)
    }

    /// Trigger a graceful shutdown and block until the background thread exits.
    pub fn shutdown(mut self) -> Result<()> {
        let _ = self.shutdown.send(true);
        if let Some(handle) = self.thread.take() {
            join_control_thread(handle)?;
        }
        Ok(())
    }
}

fn join_control_thread(handle: JoinHandle<Result<()>>) -> Result<()> {
    handle
        .join()
        .map_err(|panic| anyhow!("control thread panicked: {panic:?}"))?
        .context("control runtime terminated with an error")
}

fn publish_control_runtime_result(
    status: &watch::Sender<ControlRuntimeStatus>,
    result: &Result<()>,
) {
    if matches!(*status.borrow(), ControlRuntimeStatus::Disabled) {
        return;
    }
    let final_status = match result {
        Ok(()) => ControlRuntimeStatus::Stopped,
        Err(error) => ControlRuntimeStatus::Failed(format!("{error:#}")),
    };
    status.send_replace(final_status);
}

fn control_runtime_health(
    status: &watch::Receiver<ControlRuntimeStatus>,
) -> std::result::Result<(), String> {
    let mut probe = status.clone();
    loop {
        let cached = probe.borrow_and_update().clone();
        match &cached {
            ControlRuntimeStatus::Failed(detail) => return Err(detail.clone()),
            ControlRuntimeStatus::Stopped | ControlRuntimeStatus::Disabled => return Ok(()),
            ControlRuntimeStatus::Starting | ControlRuntimeStatus::Running => {}
        }

        match probe.has_changed() {
            Ok(true) => continue,
            Err(_) => {
                let current = probe.borrow().clone();
                if matches!(
                    current,
                    ControlRuntimeStatus::Disabled | ControlRuntimeStatus::Stopped
                ) {
                    return Ok(());
                }
                return Err(
                    "control runtime terminated without publishing final status".to_string()
                );
            }
            Ok(false) => {
                return match cached {
                    ControlRuntimeStatus::Starting => {
                        Err("control runtime is still starting".to_string())
                    }
                    ControlRuntimeStatus::Running
                    | ControlRuntimeStatus::Disabled
                    | ControlRuntimeStatus::Stopped => Ok(()),
                    ControlRuntimeStatus::Failed(_) => {
                        unreachable!("terminal statuses returned before closure inspection")
                    }
                };
            }
        }
    }
}

fn handoff_control_reaper(handle: JoinHandle<Result<()>>) {
    if let Err(source) = thread::Builder::new()
        .name("scriptbots-control-reaper".into())
        .spawn(move || {
            if let Err(error) = join_control_thread(handle) {
                error!(%error, "timed-out control runtime terminated during supervised cleanup");
            }
        })
    {
        error!(%source, "failed to spawn control runtime cleanup supervisor");
    }
}

#[cfg(test)]
impl ControlRuntime {
    /// Create a no-op runtime for tests without starting background threads.
    pub fn dummy() -> (Self, CommandDrain, CommandSubmit) {
        let (command_tx, command_rx) = create_command_bus(4);
        let command_drain = make_command_drain(command_rx);
        let command_submit = make_command_submit(command_tx);
        let (status_guard, status) = watch::channel(ControlRuntimeStatus::Running);
        let runtime = Self {
            // The dummy runtime has no ledger, so its reporter is a sink. It is
            // explicitly a no-op rather than a panic: tests that never apply a
            // command should not have to care that they hold one.
            command_reporter: Arc::new(|_, _| {}),
            presented_frame: empty_presented_frame(),
            shutdown: watch::channel(false).0,
            thread: None,
            status,
            _dummy_status_guard: Some(status_guard),
        };
        (runtime, command_drain, command_submit)
    }
}

impl Drop for ControlRuntime {
    fn drop(&mut self) {
        let _ = self.shutdown.send(true);
        if let Some(handle) = self.thread.take()
            && let Err(error) = join_control_thread(handle)
        {
            error!(%error, "control runtime failed during drop");
        }
    }
}

async fn run_control_servers(
    handle: ControlHandle,
    reservation: ControlServerReservation,
    shutdown_signal: watch::Sender<bool>,
    shutdown: watch::Receiver<bool>,
    startup: mpsc::SyncSender<ControlStartupSignal>,
    status: watch::Sender<ControlRuntimeStatus>,
) -> Result<()> {
    let mut servers = match start_control_servers(handle, reservation, shutdown.clone()).await {
        Ok(servers) => servers,
        Err(error) => {
            let _ = startup.send(ControlStartupSignal::Failed(format!("{error:#}")));
            return Err(error);
        }
    };

    if servers.rest.is_none() && servers.mcp.is_none() {
        status.send_replace(ControlRuntimeStatus::Disabled);
    } else {
        status.send_replace(ControlRuntimeStatus::Running);
    }
    if startup.send(ControlStartupSignal::Ready).is_err() {
        let _ = shutdown_signal.send(true);
        return servers
            .shutdown()
            .await
            .context("startup receiver disappeared before control servers were published");
    }

    servers.supervise(&shutdown_signal, shutdown, &status).await
}

struct PreparedRestServer {
    address: SocketAddr,
    listener: tokio::net::TcpListener,
    router: Router,
}

struct PreparedMcpServer {
    address: SocketAddr,
    listener: tokio::net::TcpListener,
    router: Router,
}

type ControlServerTask = tokio::task::JoinHandle<Result<()>>;
type JoinedControlServer = std::result::Result<Result<()>, tokio::task::JoinError>;

struct RunningControlServers {
    rest: Option<ControlServerTask>,
    mcp: Option<ControlServerTask>,
}

enum ControlServerExit {
    Rest(JoinedControlServer),
    Mcp(JoinedControlServer),
}

enum SupervisionOutcome {
    Shutdown,
    ServerExit(ControlServerExit),
}

impl RunningControlServers {
    async fn supervise(
        &mut self,
        shutdown_signal: &watch::Sender<bool>,
        mut shutdown: watch::Receiver<bool>,
        status: &watch::Sender<ControlRuntimeStatus>,
    ) -> Result<()> {
        // Prefer an already-observed shutdown over a simultaneous successful
        // task completion. Otherwise a normal graceful exit can be mislabeled
        // as an unexpected server death.
        let outcome = match (self.rest.as_mut(), self.mcp.as_mut()) {
            (Some(rest), Some(mcp)) => tokio::select! {
                biased;
                _ = shutdown.wait_for(|stop| *stop) => SupervisionOutcome::Shutdown,
                result = rest => SupervisionOutcome::ServerExit(ControlServerExit::Rest(result)),
                result = mcp => SupervisionOutcome::ServerExit(ControlServerExit::Mcp(result)),
            },
            (Some(rest), None) => tokio::select! {
                biased;
                _ = shutdown.wait_for(|stop| *stop) => SupervisionOutcome::Shutdown,
                result = rest => SupervisionOutcome::ServerExit(ControlServerExit::Rest(result)),
            },
            (None, Some(mcp)) => tokio::select! {
                biased;
                _ = shutdown.wait_for(|stop| *stop) => SupervisionOutcome::Shutdown,
                result = mcp => SupervisionOutcome::ServerExit(ControlServerExit::Mcp(result)),
            },
            (None, None) => {
                let _ = shutdown.wait_for(|stop| *stop).await;
                SupervisionOutcome::Shutdown
            }
        };

        match outcome {
            SupervisionOutcome::Shutdown => self.shutdown().await,
            SupervisionOutcome::ServerExit(exit) => {
                let primary = self.take_unexpected_exit(exit);
                status.send_replace(ControlRuntimeStatus::Failed(format!("{primary:#}")));
                let _ = shutdown_signal.send(true);
                match self.shutdown().await {
                    Ok(()) => Err(primary),
                    Err(cleanup) => Err(primary)
                        .context(format!("control sibling shutdown also failed: {cleanup:#}")),
                }
            }
        }
    }

    fn take_unexpected_exit(&mut self, exit: ControlServerExit) -> anyhow::Error {
        match exit {
            ControlServerExit::Rest(result) => {
                self.rest = None;
                unexpected_server_exit("REST", result)
            }
            ControlServerExit::Mcp(result) => {
                self.mcp = None;
                unexpected_server_exit("MCP HTTP", result)
            }
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        let rest = self.rest.take();
        let mcp = self.mcp.take();
        let (rest_result, mcp_result) = tokio::join!(
            await_control_server_shutdown("REST", rest),
            await_control_server_shutdown("MCP HTTP", mcp),
        );

        match (rest_result, mcp_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(rest_error), Err(mcp_error)) => {
                Err(rest_error).context(format!("MCP HTTP shutdown also failed: {mcp_error:#}"))
            }
        }
    }
}

fn unexpected_server_exit(name: &str, result: JoinedControlServer) -> anyhow::Error {
    match result {
        Ok(Ok(())) => anyhow!("{name} control server stopped unexpectedly"),
        Ok(Err(error)) => error.context(format!("{name} control server terminated unexpectedly")),
        Err(error) => anyhow!("{name} control server task failed: {error}"),
    }
}

async fn await_control_server_shutdown(name: &str, task: Option<ControlServerTask>) -> Result<()> {
    let Some(mut task) = task else {
        return Ok(());
    };

    match tokio::time::timeout(CONTROL_SHUTDOWN_TIMEOUT, &mut task).await {
        Ok(joined) => joined
            .with_context(|| format!("{name} control server task panicked"))?
            .with_context(|| format!("{name} control server terminated with an error")),
        Err(_) => {
            task.abort();
            let _ = task.await;
            Err(anyhow!(
                "{name} control server did not stop within {} ms",
                CONTROL_SHUTDOWN_TIMEOUT.as_millis()
            ))
        }
    }
}

async fn start_control_servers(
    handle: ControlHandle,
    reservation: ControlServerReservation,
    shutdown: watch::Receiver<bool>,
) -> Result<RunningControlServers> {
    let ControlServerReservation {
        config,
        rest: reserved_rest,
        mcp: reserved_mcp,
    } = reservation;

    let prepared_rest = match reserved_rest {
        Some(reserved) => Some(prepare_rest_server(handle.clone(), &config, reserved)?),
        None => {
            info!("REST control server disabled via configuration");
            None
        }
    };

    // Every enabled socket was bound transactionally before this runtime and
    // its world existed. Build both routers before publishing either task.
    let prepared_mcp = match reserved_mcp {
        None => {
            info!("MCP control server disabled via configuration");
            None
        }
        Some(reserved) => Some(prepare_mcp_server(handle, reserved).await?),
    };

    let rest = prepared_rest.map(|prepared| {
        let shutdown = shutdown.clone();
        tokio::spawn(async move { serve_prepared_rest_server(prepared, shutdown).await })
    });
    let mcp = prepared_mcp.map(|prepared| {
        tokio::spawn(async move { serve_prepared_mcp_server(prepared, shutdown).await })
    });

    Ok(RunningControlServers { rest, mcp })
}

#[derive(Clone)]
struct ApiState {
    handle: ControlHandle,
    scenario: Option<Arc<ScenarioIdentityV0>>,
    /// The frame the terminal frontend last presented, or empty when no terminal
    /// frontend is running. Read by `GET /api/screenshot/ascii`.
    presented_frame: SharedPresentedFrame,
}

#[derive(Debug, Serialize, ToSchema)]
struct ErrorResponse {
    message: String,
}

#[derive(Debug, Serialize, ToSchema)]
struct PresetList {
    presets: Vec<&'static str>,
}

/// Read-only view of the run's scenario identity and bootstrap policy.
#[derive(Debug, Serialize, ToSchema)]
struct ScenarioView {
    /// Stable scenario identifier (from `--scenario` or the derived default).
    id: String,
    /// Scenario schema version (`0` = derived, `1` = first-class document).
    schema_version: u16,
    /// Explicit pre-frontend warmup ticks (`0` = launch at tick zero).
    bootstrap_ticks: u64,
    /// How the founding population was seeded (app-derived recipe).
    population_recipe: String,
    /// Ordered, kind-tagged digests of every configuration layer that built this run.
    ordered_config_layer_digests: Vec<String>,
}

impl From<&ScenarioIdentityV0> for ScenarioView {
    fn from(scenario: &ScenarioIdentityV0) -> Self {
        Self {
            id: scenario.id.clone(),
            schema_version: scenario.schema_version,
            bootstrap_ticks: scenario.bootstrap_ticks,
            population_recipe: scenario.population_recipe.clone(),
            ordered_config_layer_digests: scenario.ordered_config_layer_digests.clone(),
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
struct PresetApplyRequest {
    name: String,
}

#[derive(Debug, Serialize, ToSchema)]
struct AgentDebugResponse {
    agents: Vec<AgentDebugEntryDto>,
}

#[derive(Debug, Serialize, ToSchema)]
struct AgentDebugEntryDto {
    agent_id: u64,
    selection: SelectionStateDto,
    position: PositionDto,
    energy: f32,
    health: f32,
    age: u32,
    generation: u32,
    herbivore_tendency: f32,
    diet: DietClassDto,
    brain_kind: Option<String>,
    brain_key: Option<u64>,
    mutation_primary: f32,
    mutation_secondary: f32,
    indicator_intensity: f32,
    indicator_color: [f32; 3],
}

impl From<AgentDebugInfo> for AgentDebugEntryDto {
    fn from(info: AgentDebugInfo) -> Self {
        Self {
            agent_id: info.agent_id,
            selection: info.selection.into(),
            position: PositionDto::from(info.position),
            energy: info.energy,
            health: info.health,
            age: info.age,
            generation: info.generation,
            herbivore_tendency: info.herbivore_tendency,
            diet: DietClassDto::from(info.diet),
            brain_kind: info.brain_kind,
            brain_key: info.brain_key,
            mutation_primary: info.mutation_primary,
            mutation_secondary: info.mutation_secondary,
            indicator_intensity: info.indicator.intensity,
            indicator_color: info.indicator.color,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
struct PositionDto {
    x: f32,
    y: f32,
}

impl From<Position> for PositionDto {
    fn from(value: Position) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema, Default)]
struct AgentDebugQueryParams {
    #[serde(default)]
    ids: Option<String>,
    #[serde(default)]
    diet: Option<DietClassDto>,
    #[serde(default)]
    selection: Option<SelectionStateDto>,
    #[serde(default)]
    brain: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    sort: Option<String>,
}

#[derive(Debug, Deserialize, ToSchema)]
struct SelectionUpdateRequestBody {
    mode: SelectionModeDto,
    #[serde(default)]
    agent_ids: Vec<u64>,
    #[serde(default)]
    state: Option<SelectionStateDto>,
}

impl From<SelectionUpdateRequestBody> for SelectionUpdate {
    fn from(value: SelectionUpdateRequestBody) -> Self {
        SelectionUpdate {
            mode: value.mode.into(),
            agent_ids: value.agent_ids,
            state: value.state.unwrap_or_default().into(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ConfigPatchRequest {
    #[schema(value_type = Object, nullable = false)]
    pub patch: Value,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct KnobApplyRequest {
    pub updates: Vec<KnobUpdate>,
}

#[derive(Debug, Serialize, ToSchema)]
struct ConfigAuditEntryView {
    tick: u64,
    patch: Value,
}

impl From<ConfigAuditEntry> for ConfigAuditEntryView {
    fn from(entry: ConfigAuditEntry) -> Self {
        Self {
            tick: entry.tick,
            patch: entry.patch,
        }
    }
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct StepRequestBody {
    #[serde(default = "default_step_count")]
    pub count: u64,
}

fn default_step_count() -> u64 {
    1
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct SpeedRequestBody {
    pub speed: f32,
}

#[derive(OpenApi)]
#[openapi(
    paths(
        get_knobs,
        get_config,
        patch_config,
        apply_updates,
        get_latest_tick_summary,
        get_hydrology_snapshot,
        stream_ticks_sse,
        stream_ticks_ws,
        stream_ticks_ndjson,
        screenshot_ascii,
        screenshot_png,
        get_events_tail,
        get_narrative_search,
        get_scoreboard,
        get_agents_debug,
        get_config_audit,
        get_scenario,
        list_presets,
        apply_preset,
        post_selection,
        post_pause,
        post_resume,
        post_step,
        post_speed,
        post_control_pause,
        post_control_resume,
        post_control_step,
        post_control_speed,
        post_control_shutdown,
        get_control_status,
        get_status
    ),
    components(
        schemas(
            KnobEntry,
            KnobUpdate,
            ConfigSnapshot,
            ConfigPatchRequest,
            KnobApplyRequest,
            ConfigAuditEntryView,
            ScenarioView,
            PresetList,
            PresetApplyRequest,
            ErrorResponse,
            EventEntry,
            NarrativeSearchHitDto,
            EventKind,
            DietClassDto,
            SelectionStateDto,
            SelectionModeDto,
            AgentScoreEntry,
            Scoreboard,
            HydrologySnapshot,
            AgentDebugResponse,
            AgentDebugEntryDto,
            PositionDto,
            SelectionUpdateRequestBody,
            CommandStatusDto,
            StepRequestBody,
            SpeedRequestBody,
            SimulationStatusDto
        )
    ),
    info(title = "ScriptBots Control API", version = "0.0.0"),
    tags((name = "control", description = "Runtime configuration controls"))
)]
struct ApiDoc;

struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    /// The request was well-formed but the process cannot satisfy it in its current
    /// mode — distinct from `bad_request` (the caller's fault) and `not_found` (the
    /// resource does not exist). Used by `/api/screenshot/ascii` when no terminal
    /// frontend has presented a frame.
    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    fn service_unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl From<ControlError> for AppError {
    fn from(err: ControlError) -> Self {
        match err {
            ControlError::UnknownPath(path) => {
                Self::bad_request(format!("unknown knob path: {path}"))
            }
            ControlError::InvalidPatch(msg) => Self::bad_request(msg),
            ControlError::Serialization(msg) => Self::internal(msg),
            ControlError::Lock => Self::service_unavailable("world state is currently unavailable"),
            ControlError::CommandQueueFull => {
                Self::service_unavailable("command queue is full; retry shortly")
            }
            ControlError::CommandQueueClosed => {
                Self::service_unavailable("command queue is closed")
            }
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = Json(ErrorResponse {
            message: self.message.clone(),
        });
        (self.status, body).into_response()
    }
}

/// Run a world-lock-taking control operation on the blocking thread pool.
///
/// Every `ControlHandle` read or mutation may contend on the world mutex with
/// the simulation driver, which holds it for a full scientific tick at a time.
/// Calling such an operation directly from a handler parks a tokio async
/// worker on a blocking lock; with one slow tick and `num_cpus` concurrent
/// clients the entire control plane (REST, Swagger, and MCP) freezes (bd-134).
/// A parked blocking-pool thread is cheap and bounded; a parked async worker
/// is the event loop.
async fn run_control<T, F>(operation: F) -> Result<T, AppError>
where
    F: FnOnce() -> Result<T, ControlError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|join_error| {
            AppError::internal(format!("control operation task failed: {join_error}"))
        })?
        .map_err(AppError::from)
}

#[utoipa::path(
    get,
    path = "/api/knobs",
    tag = "control",
    responses((status = 200, body = [KnobEntry]))
)]
async fn get_knobs(State(state): State<ApiState>) -> Result<Json<Vec<KnobEntry>>, AppError> {
    let mut knobs = run_control(move || state.handle.list_knobs()).await?;
    knobs.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(Json(knobs))
}

#[utoipa::path(
    get,
    path = "/api/config",
    tag = "control",
    responses((status = 200, body = ConfigSnapshot))
)]
async fn get_config(State(state): State<ApiState>) -> Result<Json<ConfigSnapshot>, AppError> {
    let snapshot = run_control(move || state.handle.snapshot()).await?;
    Ok(Json(snapshot))
}

/// Return the latest tick summary as JSON.
#[utoipa::path(
    get,
    path = "/api/ticks/latest",
    tag = "control",
    responses((status = 200, description = "Latest tick summary"))
)]
async fn get_latest_tick_summary(
    State(state): State<ApiState>,
) -> Result<Json<TickSummaryDto>, AppError> {
    let summary = run_control(move || state.handle.latest_summary()).await?;
    Ok(Json(summary.into()))
}

#[utoipa::path(
    get,
    path = "/api/hydrology",
    tag = "control",
    responses(
        (status = 200, body = HydrologySnapshot),
        (status = 404, description = "Hydrology state unavailable")
    )
)]
async fn get_hydrology_snapshot(
    State(state): State<ApiState>,
) -> Result<Json<HydrologySnapshot>, AppError> {
    match run_control(move || state.handle.hydrology_snapshot()).await? {
        Some(snapshot) => Ok(Json(snapshot)),
        None => Err(AppError::not_found("hydrology state unavailable")),
    }
}

/// Stream latest tick summaries as Server-Sent Events (SSE).
#[utoipa::path(
    get,
    path = "/api/ticks/stream",
    tag = "control",
    responses((status = 200, description = "SSE stream of tick summaries"))
)]
async fn stream_ticks_sse(
    State(state): State<ApiState>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    let handle = state.handle.clone();
    let stream =
        IntervalStream::new(tokio::time::interval(Duration::from_millis(500))).then(move |_| {
            let handle = handle.clone();
            async move {
                // Poll on the blocking pool: a contended world mutex must park
                // a blocking thread, never this stream's async worker (bd-134).
                let summary = tokio::task::spawn_blocking(move || handle.latest_summary()).await;
                let event = match summary {
                    Ok(Ok(summary)) => {
                        let json = serde_json::to_string(&TickSummaryDto::from(summary))
                            .unwrap_or_else(|_| "{}".to_string());
                        Event::default().data(json)
                    }
                    Ok(Err(_)) | Err(_) => Event::default().data("{}"),
                };
                Ok::<Event, Infallible>(event)
            }
        });
    Ok(Sse::new(stream))
}

/// Upgrade HTTP connection to a WebSocket binary/text real-time stream.
#[utoipa::path(
    get,
    path = "/api/ws/stream",
    tag = "control",
    responses((status = 101, description = "WebSocket upgrade for tick & event stream"))
)]
async fn stream_ticks_ws(ws: WebSocketUpgrade, State(state): State<ApiState>) -> Response {
    ws.on_upgrade(move |socket| handle_ws_stream(socket, state))
}

async fn handle_ws_stream(mut socket: WebSocket, state: ApiState) {
    let handle = state.handle.clone();
    let mut interval = tokio::time::interval(Duration::from_millis(50));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                let handle_clone = handle.clone();
                let summary_res = tokio::task::spawn_blocking(move || handle_clone.latest_summary()).await;
                if let Ok(Ok(summary)) = summary_res {
                    let dto = TickSummaryDto::from(summary);
                    if let Ok(bytes) = postcard::to_stdvec(&dto) {
                        if socket.send(WsMessage::Binary(bytes.into())).await.is_err() {
                            break;
                        }
                    }
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(WsMessage::Text(text))) => {
                        let text_str = text.trim();
                        // Every branch REPORTS its receipt. These four used to
                        // discard it with `let _ =`, so a websocket client sent
                        // "pause" and received nothing at all - no receipt, no
                        // error, no acknowledgement of any kind. It is the same
                        // class the other transports were cleared of, on a
                        // surface that is not even listed in the migration
                        // acceptance criteria alongside CLI/REST/MCP/SSE/NDJSON
                        // (bd-k7nq).
                        let outcome = if text_str.eq_ignore_ascii_case("pause") {
                            Some(handle.pause(None))
                        } else if text_str.eq_ignore_ascii_case("resume") {
                            Some(handle.resume(None))
                        } else if text_str.eq_ignore_ascii_case("step") {
                            Some(handle.step())
                        } else if let Ok(json) = serde_json::from_str::<Value>(text_str) {
                            match json.get("command").and_then(|c| c.as_str()) {
                                Some("speed") => json
                                    .get("speed")
                                    .and_then(|s| s.as_f64())
                                    .map(|speed| handle.set_speed(speed as f32, None)),
                                _ => None,
                            }
                        } else {
                            None
                        };
                        if let Some(result) = outcome {
                            let payload = match result {
                                Ok(status) => serde_json::json!({"receipt": status}),
                                Err(error) => serde_json::json!({"error": error.to_string()}),
                            };
                            if let Ok(text) = serde_json::to_string(&payload)
                                && socket.send(WsMessage::Text(text.into())).await.is_err()
                            {
                                break;
                            }
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
}

/// Return the exact terminal frame the user is looking at, as text.
///
/// UPGRADED FIDELITY, SAME ENDPOINT (bd-2z0.14.2.6 item 2). This used to answer by
/// calling `ascii_map()`, which re-rasterizes terrain and food into its own ≤96x48
/// grid: a third renderer, agreeing with neither the sub-cell canvas nor the flat
/// map, with no agents and no panels. Two visually different worlds could serve
/// identical bytes, and a frame with a broken widget served clean.
///
/// It now serves the buffer the terminal actually presented, serialized by the same
/// `terminal::export` functions the `S` key writes to disk — so the served bytes,
/// the saved bytes, and the displayed cells are one artifact rather than three.
///
/// WHEN NO TERMINAL FRONTEND IS RUNNING it REFUSES with 409 rather than falling
/// back to the old rasterization. A synthesized map returned under the name
/// "screenshot" is the exact defect this replaced; a caller that wants world state
/// has `/api/ticks/latest` and `/api/screenshot/png`, both of which say what they
/// are.
#[utoipa::path(
    get,
    path = "/api/screenshot/ascii",
    tag = "control",
    responses(
        (status = 200, description = "The exact presented terminal frame as text", content_type = "text/plain"),
        (status = 409, description = "no terminal frontend has presented a frame", body = ErrorResponse)
    )
)]
async fn screenshot_ascii(State(state): State<ApiState>) -> Result<Response, AppError> {
    let Some(frame) = state.presented_frame.load_full() else {
        return Err(AppError::conflict(
            "no terminal frame has been presented: this endpoint serves the exact \
             buffer the terminal frontend displayed, and this process is not running \
             one (headless, server-only, or GUI mode). It deliberately does not \
             substitute a re-rasterized world map, which would be a different \
             renderer's output under the name 'screenshot'.",
        ));
    };
    // Serialization is pure CPU over a cloned buffer and touches no lock, so it
    // does not need the blocking pool the world-lock handlers use.
    let text = crate::terminal::export::buffer_to_plain_text(&frame.buffer);
    Ok((
        StatusCode::OK,
        [
            ("x-scriptbots-frame-tick", frame.tick.to_string()),
            (
                "x-scriptbots-frame-size",
                format!("{}x{}", frame.buffer.area.width, frame.buffer.area.height),
            ),
        ],
        text,
    )
        .into_response())
}

/// Rasterize the current world offscreen and return it as a 1024x576 PNG.
///
/// Requires the `gui` application feature; a binary built without it answers
/// `400 Bad Request` rather than pretending to render.
#[utoipa::path(
    get,
    path = "/api/screenshot/png",
    tag = "control",
    responses(
        (status = 200, description = "1024x576 PNG render of the current world", content_type = "image/png"),
        (status = 400, description = "binary was built without the `gui` feature", body = ErrorResponse)
    )
)]
async fn screenshot_png(State(state): State<ApiState>) -> Result<Response, AppError> {
    // Rasterization is CPU-heavy on top of the lock acquisition, so the whole
    // operation belongs on the blocking pool (bd-134).
    let bytes = run_control(move || state.handle.snapshot_png(1024, 576)).await?;
    Ok((StatusCode::OK, axum::body::Bytes::from(bytes)).into_response())
}

// NDJSON tick stream for simple clients
#[utoipa::path(
    get,
    path = "/api/ticks/ndjson",
    tag = "control",
    responses((status = 200, description = "NDJSON stream of tick summaries"))
)]
async fn stream_ticks_ndjson(State(state): State<ApiState>) -> Result<Response, AppError> {
    let handle = state.handle.clone();
    let stream =
        IntervalStream::new(tokio::time::interval(Duration::from_millis(500))).then(move |_| {
            let handle = handle.clone();
            async move {
                // Poll on the blocking pool: a contended world mutex must park
                // a blocking thread, never this stream's async worker (bd-134).
                let summary = tokio::task::spawn_blocking(move || handle.latest_summary()).await;
                let line = match summary {
                    Ok(Ok(summary)) => {
                        let json = serde_json::to_string(&TickSummaryDto::from(summary))
                            .unwrap_or_else(|_| "{}".to_string());
                        format!("{json}\n")
                    }
                    Ok(Err(_)) | Err(_) => "{}\n".to_string(),
                };
                Ok::<axum::body::Bytes, Infallible>(axum::body::Bytes::from(line))
            }
        });

    let mut resp = Response::new(axum::body::Body::from_stream(stream));
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("application/x-ndjson"),
    );
    Ok(resp)
}

#[utoipa::path(
    get,
    path = "/api/events/tail",
    tag = "control",
    params(("limit" = usize, Query, description = "Max events to return", example = 32)),
    responses((status = 200, body = [EventEntry]))
)]
async fn get_events_tail(
    State(state): State<ApiState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<EventEntry>>, AppError> {
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32);
    let events = run_control(move || state.handle.events_tail(limit)).await?;
    Ok(Json(events))
}

#[derive(Serialize, Deserialize, ToSchema)]
pub struct NarrativeSearchHitDto {
    pub tick: u64,
    pub kind: String,
    pub severity: f32,
    pub human_text: String,
    pub score: f64,
}

#[utoipa::path(
    get,
    path = "/api/narrative/search",
    tag = "control",
    params(("q" = String, Query, description = "Search query")),
    responses((status = 200, body = [NarrativeSearchHitDto]))
)]
async fn get_narrative_search(
    State(_state): State<ApiState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Vec<NarrativeSearchHitDto>>, AppError> {
    let _q = params.get("q").cloned().unwrap_or_default();
    Ok(Json(vec![]))
}

#[utoipa::path(
    get,
    path = "/api/scoreboard",
    tag = "control",
    params(("limit" = usize, Query, description = "Max entries per list", example = 10)),
    responses((status = 200, body = Scoreboard))
)]
async fn get_scoreboard(
    State(state): State<ApiState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Result<Json<Scoreboard>, AppError> {
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    let board = run_control(move || state.handle.compute_scoreboard(limit)).await?;
    Ok(Json(board))
}

#[utoipa::path(
    get,
    path = "/api/agents/debug",
    tag = "control",
    params(
        ("ids" = String, Query, description = "Comma-separated list of agent ids to include"),
        ("diet" = DietClassDto, Query, description = "Filter by dietary class"),
        ("selection" = SelectionStateDto, Query, description = "Filter by selection state"),
        ("brain" = String, Query, description = "Substring match against brain kind"),
        ("limit" = usize, Query, description = "Maximum number of agents to return"),
        ("sort" = String, Query, description = "Sort order: 'energy' (default) or 'age'")
    ),
    responses((status = 200, body = AgentDebugResponse), (status = 400, body = ErrorResponse))
)]
async fn get_agents_debug(
    State(state): State<ApiState>,
    axum::extract::Query(params): axum::extract::Query<AgentDebugQueryParams>,
) -> Result<Json<AgentDebugResponse>, AppError> {
    let mut query = AgentDebugQuery::default();

    if let Some(raw_ids) = params.ids.as_deref() {
        let ids = parse_id_list(raw_ids)?;
        if !ids.is_empty() {
            query.ids = Some(ids);
        }
    }
    query.diet = params.diet.map(|dto| dto.into());
    query.selection = params.selection.map(|dto| dto.into());
    query.brain_kind = params.brain.clone();
    query.limit = params.limit;
    if let Some(sort) = params.sort.as_deref() {
        query.sort = match sort.to_ascii_lowercase().as_str() {
            "energy" | "" => AgentDebugSort::EnergyDesc,
            "age" => AgentDebugSort::AgeDesc,
            other => {
                return Err(AppError::bad_request(format!(
                    "unknown sort mode '{other}'; expected 'energy' or 'age'"
                )));
            }
        };
    }

    let query_for_world = query.clone();
    let mut agents: Vec<AgentDebugEntryDto> =
        run_control(move || state.handle.debug_agents(query_for_world))
            .await?
            .into_iter()
            .map(AgentDebugEntryDto::from)
            .collect();
    // Ensure deterministic ordering and explicit tie-breaks
    match query.sort {
        AgentDebugSort::EnergyDesc => agents.sort_by(|a, b| {
            b.energy
                .partial_cmp(&a.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.health
                        .partial_cmp(&a.health)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| b.age.cmp(&a.age))
                .then_with(|| a.agent_id.cmp(&b.agent_id))
        }),
        AgentDebugSort::AgeDesc => agents.sort_by(|a, b| {
            b.age
                .cmp(&a.age)
                .then_with(|| {
                    b.energy
                        .partial_cmp(&a.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.health
                        .partial_cmp(&a.health)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.agent_id.cmp(&b.agent_id))
        }),
    }
    Ok(Json(AgentDebugResponse { agents }))
}

#[utoipa::path(
    post,
    path = "/api/selection",
    tag = "control",
    request_body = SelectionUpdateRequestBody,
    responses((status = 202, body = CommandStatusDto), (status = 400, body = ErrorResponse))
)]
async fn post_selection(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<SelectionUpdateRequestBody>,
) -> Result<(StatusCode, Json<CommandStatusDto>), AppError> {
    let key = idempotency_key(&headers);
    let update: SelectionUpdate = body.into();
    // Report the receipt the command actually produced. This used to discard
    // the call's result and answer with a hardcoded `queued: true` — a literal,
    // not a fact derived from anything — which left selection as the only
    // control surface with no command identity at all (bd-2z0.4.9). A client
    // could not poll it, correlate it, or tell two selections apart.
    //
    // 202 is still correct and is now honest rather than incidental: the body
    // says `admitted`, not `applied`, so the status code and the payload agree
    // that this command has been accepted and not yet applied.
    let status = run_control(move || state.handle.update_selection(update, key.as_deref())).await?;
    Ok((StatusCode::ACCEPTED, Json(status)))
}

#[utoipa::path(
    patch,
    path = "/api/config",
    tag = "control",
    request_body = ConfigPatchRequest,
    responses(
        (status = 202, body = CommandStatusDto),
        (status = 400, body = ErrorResponse)
    )
)]
async fn patch_config(
    State(state): State<ApiState>,
    Json(payload): Json<ConfigPatchRequest>,
) -> Result<(StatusCode, Json<CommandStatusDto>), AppError> {
    // 202 with a receipt, not 200 with the requested config echoed back as
    // though it were current. The old response projected future config: it was
    // built from the REQUESTED values and stamped with the tick at which those
    // values were NOT in effect (bd-k7nq). Read /api/config for applied truth.
    let status = run_control(move || state.handle.apply_patch(payload.patch)).await?;
    Ok((StatusCode::ACCEPTED, Json(status)))
}

#[utoipa::path(
    post,
    path = "/api/knobs/apply",
    tag = "control",
    request_body = KnobApplyRequest,
    responses(
        (status = 202, body = CommandStatusDto),
        (status = 400, body = ErrorResponse)
    )
)]
async fn apply_updates(
    State(state): State<ApiState>,
    Json(payload): Json<KnobApplyRequest>,
) -> Result<(StatusCode, Json<CommandStatusDto>), AppError> {
    if payload.updates.is_empty() {
        return Err(AppError::bad_request("updates cannot be empty"));
    }
    let status = run_control(move || state.handle.apply_updates(&payload.updates)).await?;
    Ok((StatusCode::ACCEPTED, Json(status)))
}

#[utoipa::path(
    get,
    path = "/api/config/audit",
    tag = "control",
    responses((status = 200, body = [ConfigAuditEntryView]))
)]
async fn get_config_audit(
    State(state): State<ApiState>,
) -> Result<Json<Vec<ConfigAuditEntryView>>, AppError> {
    let mut entries: Vec<ConfigAuditEntryView> = run_control(move || state.handle.audit())
        .await?
        .into_iter()
        .map(ConfigAuditEntryView::from)
        .collect();
    entries.sort_by_key(|entry| entry.tick);
    Ok(Json(entries))
}

#[utoipa::path(
    get,
    path = "/api/scenario",
    tag = "control",
    responses(
        (status = 200, body = ScenarioView),
        (status = 404, body = ErrorResponse, description = "no scenario identity available")
    )
)]
async fn get_scenario(State(state): State<ApiState>) -> Result<Json<ScenarioView>, AppError> {
    let scenario = state.scenario.as_deref().ok_or_else(|| {
        AppError::not_found("this server was launched without a scenario identity")
    })?;
    Ok(Json(ScenarioView::from(scenario)))
}

#[utoipa::path(
    get,
    path = "/api/presets",
    tag = "control",
    responses((status = 200, body = PresetList))
)]
async fn list_presets() -> Result<Json<PresetList>, AppError> {
    let presets = PresetKind::all().iter().map(|p| p.as_str()).collect();
    Ok(Json(PresetList { presets }))
}

#[utoipa::path(
    post,
    path = "/api/presets/apply",
    tag = "control",
    request_body = PresetApplyRequest,
    responses((status = 202, body = CommandStatusDto), (status = 400, body = ErrorResponse))
)]
async fn apply_preset(
    State(state): State<ApiState>,
    Json(payload): Json<PresetApplyRequest>,
) -> Result<(StatusCode, Json<CommandStatusDto>), AppError> {
    let Some(kind) = PresetKind::from_name(&payload.name) else {
        return Err(AppError::bad_request(format!(
            "unknown preset: {}",
            payload.name
        )));
    };
    let status = run_control(move || state.handle.apply_patch(kind.patch())).await?;
    Ok((StatusCode::ACCEPTED, Json(status)))
}

#[utoipa::path(
    post,
    path = "/api/control/pause",
    tag = "control",
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_control_pause(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<CommandStatusDto>, AppError> {
    let key = idempotency_key(&headers);
    let status = run_control(move || state.handle.pause(key.as_deref())).await?;
    Ok(Json(status))
}

#[utoipa::path(
    post,
    path = "/api/control/resume",
    tag = "control",
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_control_resume(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<CommandStatusDto>, AppError> {
    let key = idempotency_key(&headers);
    let status = run_control(move || state.handle.resume(key.as_deref())).await?;
    Ok(Json(status))
}

#[utoipa::path(
    post,
    path = "/api/control/step",
    tag = "control",
    request_body(content = Option<StepRequestBody>, description = "Optional step configuration"),
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_control_step(
    State(state): State<ApiState>,
    bytes: axum::body::Bytes,
) -> Result<Json<CommandStatusDto>, AppError> {
    let count = if bytes.iter().all(u8::is_ascii_whitespace) {
        1
    } else {
        let body: StepRequestBody = serde_json::from_slice(&bytes).map_err(|error| {
            AppError::bad_request(format!("invalid step request payload: {error}"))
        })?;
        body.count
    };
    let status = run_control(move || state.handle.step_count(count)).await?;
    Ok(Json(status))
}

#[utoipa::path(
    post,
    path = "/api/control/speed",
    tag = "control",
    request_body = SpeedRequest,
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_control_speed(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(payload): Json<SpeedRequest>,
) -> Result<Json<CommandStatusDto>, AppError> {
    let key = idempotency_key(&headers);
    let status = run_control(move || state.handle.set_speed(payload.speed, key.as_deref())).await?;
    Ok(Json(status))
}

#[utoipa::path(
    post,
    path = "/api/control/shutdown",
    tag = "control",
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_control_shutdown(
    State(state): State<ApiState>,
) -> Result<Json<CommandStatusDto>, AppError> {
    let status = run_control(move || state.handle.shutdown()).await?;
    Ok(Json(status))
}

#[utoipa::path(
    get,
    path = "/api/control/status/{command_id}",
    tag = "control",
    params(("command_id" = String, Path, description = "Command ID")),
    responses(
        (status = 200, body = CommandStatusDto),
        (status = 404, description = "Command ID not found")
    )
)]
async fn get_control_status(
    State(state): State<ApiState>,
    axum::extract::Path(command_id): axum::extract::Path<String>,
) -> Result<Json<CommandStatusDto>, AppError> {
    let id_for_handle = command_id.clone();
    let status = run_control(move || state.handle.command_status(&id_for_handle))
        .await?
        .ok_or_else(|| AppError::not_found(format!("unknown command id: {command_id}")))?;
    Ok(Json(status))
}

#[utoipa::path(
    post,
    path = "/api/pause",
    tag = "control",
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_pause(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<CommandStatusDto>, AppError> {
    let key = idempotency_key(&headers);
    let status = run_control(move || state.handle.pause(key.as_deref())).await?;
    Ok(Json(status))
}

/// The client-supplied idempotency key from an MCP tool call, if any.
///
/// Same absent-is-absent rule as the HTTP header: a blank string is not a key.
/// MCP carries it as a tool argument because there is no header to hang it on,
/// but the guarantee is identical, which is the point of wiring all the
/// transports in one change rather than one at a time (bd-k7nq).
fn mcp_idempotency_key(arguments: &serde_json::Map<String, Value>) -> Option<String> {
    let value = arguments.get("idempotency_key")?.as_str()?.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

/// The client-supplied idempotency key, if any.
///
/// `Idempotency-Key` is the conventional HTTP spelling, so clients and proxies
/// already understand it. A blank or non-ASCII value is treated as absent
/// rather than as a key: keying on an empty string would make every unkeyed
/// retry collide with every other one, which is worse than not keying at all
/// (bd-k7nq).
fn idempotency_key(headers: &HeaderMap) -> Option<String> {
    let value = headers.get("Idempotency-Key")?.to_str().ok()?.trim();
    (!value.is_empty()).then(|| value.to_owned())
}

#[utoipa::path(
    post,
    path = "/api/resume",
    tag = "control",
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_resume(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<CommandStatusDto>, AppError> {
    let key = idempotency_key(&headers);
    let status = run_control(move || state.handle.resume(key.as_deref())).await?;
    Ok(Json(status))
}

#[utoipa::path(
    post,
    path = "/api/step",
    tag = "control",
    request_body(content = Option<StepRequestBody>, description = "Optional step configuration"),
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_step(
    State(state): State<ApiState>,
    bytes: axum::body::Bytes,
) -> Result<Json<CommandStatusDto>, AppError> {
    let count = if bytes.iter().all(u8::is_ascii_whitespace) {
        1
    } else {
        let body: StepRequestBody = serde_json::from_slice(&bytes).map_err(|error| {
            AppError::bad_request(format!("invalid step request payload: {error}"))
        })?;
        body.count
    };
    // `step_count` submits `count` commands and returns the LAST receipt. The
    // reported id therefore identifies the final step, not the batch; a caller
    // polling it learns about that one command. Saying so beats implying the
    // receipt covers all of them.
    let status = run_control(move || state.handle.step_count(count)).await?;
    Ok(Json(status))
}

#[utoipa::path(
    post,
    path = "/api/speed",
    tag = "control",
    request_body = SpeedRequestBody,
    responses((status = 200, body = CommandStatusDto))
)]
async fn post_speed(
    State(state): State<ApiState>,
    headers: HeaderMap,
    Json(body): Json<SpeedRequestBody>,
) -> Result<Json<CommandStatusDto>, AppError> {
    let key = idempotency_key(&headers);
    let status = run_control(move || state.handle.set_speed(body.speed, key.as_deref())).await?;
    Ok(Json(status))
}

#[utoipa::path(
    get,
    path = "/api/status",
    tag = "control",
    responses((status = 200, body = SimulationStatusDto))
)]
async fn get_status(State(state): State<ApiState>) -> Result<Json<SimulationStatusDto>, AppError> {
    let status = run_control(move || state.handle.status()).await?;
    Ok(Json(status))
}

fn prepare_rest_server(
    handle: ControlHandle,
    config: &ControlServerConfig,
    reserved: ReservedControlListener,
) -> Result<PreparedRestServer> {
    let state = ApiState {
        handle,
        scenario: config.scenario.clone(),
        presented_frame: Arc::clone(&config.presented_frame),
    };
    let mut openapi = ApiDoc::openapi();
    openapi.info.version = env!("CARGO_PKG_VERSION").to_string();

    let api_router = Router::new()
        .route("/api/knobs", get(get_knobs))
        .route("/api/config", get(get_config).patch(patch_config))
        .route("/api/knobs/apply", post(apply_updates))
        // Tick summaries (JSON one-shot and SSE stream)
        .route("/api/ticks/latest", get(get_latest_tick_summary))
        .route("/api/ticks/stream", get(stream_ticks_sse))
        .route("/api/ws/stream", get(stream_ticks_ws))
        // Screenshots
        .route("/api/screenshot/ascii", get(screenshot_ascii))
        .route("/api/screenshot/png", get(screenshot_png))
        .route("/api/ticks/ndjson", get(stream_ticks_ndjson))
        .route("/api/hydrology", get(get_hydrology_snapshot))
        // Event tail and scoreboard
        .route("/api/events/tail", get(get_events_tail))
        .route("/api/narrative/search", get(get_narrative_search))
        .route("/api/scoreboard", get(get_scoreboard))
        .route("/api/agents/debug", get(get_agents_debug))
        .route("/api/selection", post(post_selection))
        // Scenario identity and presets
        .route("/api/scenario", get(get_scenario))
        .route("/api/presets", get(list_presets))
        .route("/api/presets/apply", post(apply_preset))
        .route("/api/config/audit", get(get_config_audit))
        // Simulation playback & status controls
        .route("/api/pause", post(post_pause))
        .route("/api/resume", post(post_resume))
        .route("/api/step", post(post_step))
        .route("/api/speed", post(post_speed))
        .route("/api/status", get(get_status))
        // Direct simulation control commands
        .route("/api/control/pause", post(post_control_pause))
        .route("/api/control/resume", post(post_control_resume))
        .route("/api/control/step", post(post_control_step))
        .route("/api/control/speed", post(post_control_speed))
        .route("/api/control/shutdown", post(post_control_shutdown))
        .route("/api/control/status/{command_id}", get(get_control_status))
        .with_state(state);

    let swagger_router: Router<_> = SwaggerUi::new(config.swagger_path.clone())
        .url("/api-docs/openapi.json", openapi)
        .into();

    let router = Router::new().merge(api_router).merge(swagger_router);

    let listener = tokio::net::TcpListener::from_std(reserved.listener)
        .context("failed to adopt reserved REST listener")?;

    Ok(PreparedRestServer {
        address: reserved.address,
        listener,
        router,
    })
}

async fn serve_prepared_rest_server(
    prepared: PreparedRestServer,
    shutdown: watch::Receiver<bool>,
) -> Result<()> {
    info!(address = %prepared.address, "REST control server listening");
    let mut shutdown_signal = shutdown.clone();
    axum::serve(prepared.listener, prepared.router)
        .with_graceful_shutdown(async move {
            let _ = shutdown_signal.wait_for(|stop| *stop).await;
        })
        .await
        .context("REST control server errored")
}

async fn prepare_mcp_server(
    handle: ControlHandle,
    reserved: ReservedControlListener,
) -> Result<PreparedMcpServer> {
    info!(address = %reserved.address, "Preparing MCP HTTP server");
    let builder = register_control_tools(
        fastmcp_rust::ServerBuilder::new("scriptbots-control", env!("CARGO_PKG_VERSION")),
        handle,
    );

    let server = Arc::new(builder.build());
    let session = Arc::new(Mutex::new(Session::new(
        server.info().clone(),
        server.capabilities().clone(),
    )));
    // HTTP has no persistent outbound channel, so server-initiated
    // notifications are only observable in the trace log. Tool callers never
    // depend on them; the request/response path below is the product surface.
    let notification_sender: NotificationSender = Arc::new(|request: JsonRpcRequest| {
        debug!(
            method = %request.method,
            "MCP notification not deliverable over stateless HTTP transport"
        );
    });
    let request_sender = RequestSender::new(
        Arc::new(PendingRequests::new()),
        Arc::new(|_message: &_| {
            Err("MCP HTTP transport does not support server-to-client requests".into())
        }),
    );

    let router = Router::new()
        .route("/mcp", post(handle_mcp_http_request))
        .route("/mcp/notify", post(handle_mcp_http_notification))
        .route("/mcp/events", get(handle_mcp_http_events))
        .route("/health", get(handle_mcp_http_health))
        .with_state(McpHttpState {
            server,
            session,
            notification_sender,
            request_sender,
        });
    let listener = tokio::net::TcpListener::from_std(reserved.listener)
        .context("failed to adopt reserved MCP HTTP listener")?;

    Ok(PreparedMcpServer {
        address: reserved.address,
        listener,
        router,
    })
}

/// Register the complete MCP control-tool roster onto `builder`.
///
/// Split out of [`prepare_mcp_server`] so the roster can be asserted without
/// binding a socket: this list is hand-maintained beside `ControlToolKind`, and a
/// variant that dispatches but is never registered here is simply an MCP tool that
/// does not exist. That is the same silent drift that removed eight routes from the
/// published OpenAPI document (bd-01dg), so `mcp_tool_roster_is_complete` pins it.
fn register_control_tools(
    builder: fastmcp_rust::ServerBuilder,
    handle: ControlHandle,
) -> fastmcp_rust::ServerBuilder {
    let mut builder = builder;

    builder = register_tool(
        builder,
        "list_presets",
        "List available scenario presets",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::ListPresets,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "apply_preset",
        "Apply a named scenario preset",
        json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        }),
        ControlToolKind::ApplyPreset,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "list_knobs",
        "List all exposed configuration knobs",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::ListKnobs,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "get_config",
        "Fetch the entire simulation configuration",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::GetConfig,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "apply_updates",
        "Apply one or more knob updates by path",
        json!({
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["path", "value"],
                        "properties": {
                            "path": {"type": "string"},
                            "value": {}
                        },
                        "additionalProperties": false
                    }
                }
            },
            "required": ["updates"],
            "additionalProperties": false
        }),
        ControlToolKind::ApplyUpdates,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "apply_patch",
        "Merge a JSON object patch into the configuration",
        json!({
            "type": "object",
            "properties": {
                "patch": {"type": "object"}
            },
            "required": ["patch"],
            "additionalProperties": false
        }),
        ControlToolKind::ApplyPatch,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "pause",
        "Pause the simulation",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::Pause,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "resume",
        "Resume the simulation",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::Resume,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "step",
        "Step the simulation by N ticks",
        json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1}
            },
            "additionalProperties": false
        }),
        ControlToolKind::Step,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "set_speed",
        "Set the simulation speed (target TPS)",
        json!({
            "type": "object",
            "properties": {
                "speed": {"type": "number"}
            },
            "required": ["speed"],
            "additionalProperties": false
        }),
        ControlToolKind::SetSpeed,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "get_status",
        "Retrieve current simulation status (tick, agent count, pause state, revision)",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::GetStatus,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "shutdown",
        "Request graceful simulation shutdown",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::Shutdown,
        handle.clone(),
    );

    builder = register_tool(
        builder,
        "get_command_status",
        "Look up status of a command by ID",
        json!({
            "type": "object",
            "properties": {
                "command_id": {"type": "string"}
            },
            "required": ["command_id"],
            "additionalProperties": false
        }),
        ControlToolKind::GetCommandStatus,
        handle,
    );

    builder
}

fn register_tool(
    builder: fastmcp_rust::ServerBuilder,
    name: &str,
    description: &str,
    schema: Value,
    kind: ControlToolKind,
    handle: ControlHandle,
) -> fastmcp_rust::ServerBuilder {
    builder.tool(ControlTool {
        handle,
        kind,
        name: name.to_string(),
        description: description.to_string(),
        schema,
    })
}

/// Shared per-process MCP HTTP state: one negotiated session guarded by
/// `dispatch_request_concurrent`, which releases the session mutex for
/// read-only tool calls so parallel `tools/call` probes stay concurrent.
#[derive(Clone)]
struct McpHttpState {
    server: Arc<Server>,
    session: Arc<Mutex<Session>>,
    notification_sender: NotificationSender,
    request_sender: RequestSender,
}

/// Dispatch one JSON-RPC request through the real fastmcp router.
///
/// Requests carrying an `id` receive a JSON-RPC response; notifications (no
/// `id`) are accepted with HTTP 202 and never produce a response body. The
/// registered 13-tool control roster is reachable exactly as MCP clients
/// expect: `initialize`, `tools/list`, then `tools/call`.
async fn handle_mcp_http_request(
    State(state): State<McpHttpState>,
    Json(request): Json<JsonRpcRequest>,
) -> Response {
    let method = request.method.clone();
    let id = request.id.clone();
    let cx = Cx::for_request();
    match state.server.dispatch_request_concurrent(
        &cx,
        &state.session,
        request,
        &state.notification_sender,
        &state.request_sender,
    ) {
        Some(message) => Json(message).into_response(),
        None => {
            if let Some(request_id) = id {
                let (code, message) = if method == "initialize" {
                    (
                        -32602,
                        "Unsupported protocol version; supported versions: [\"2024-11-05\"]",
                    )
                } else {
                    (
                        -32600,
                        "Invalid Request: unhandled or rejected by MCP server",
                    )
                };
                let err_response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": code,
                        "message": message
                    }
                });
                Json(err_response).into_response()
            } else {
                StatusCode::ACCEPTED.into_response()
            }
        }
    }
}

async fn handle_mcp_http_notification(Json(_notification): Json<Value>) -> StatusCode {
    StatusCode::OK
}

async fn handle_mcp_http_events() -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = futures_util::stream::pending::<Result<Event, Infallible>>();
    Sse::new(stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(30))
            .text("keep-alive"),
    )
}

async fn handle_mcp_http_health() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "transport": "http",
    }))
}

async fn serve_prepared_mcp_server(
    prepared: PreparedMcpServer,
    shutdown: watch::Receiver<bool>,
) -> Result<()> {
    info!(address = %prepared.address, "MCP HTTP server listening");
    let mut shutdown_signal = shutdown.clone();
    axum::serve(prepared.listener, prepared.router)
        .with_graceful_shutdown(async move {
            let _ = shutdown_signal.wait_for(|stop| *stop).await;
        })
        .await
        .context("MCP HTTP control server errored")
}

#[derive(Clone)]
struct ControlTool {
    handle: ControlHandle,
    kind: ControlToolKind,
    name: String,
    description: String,
    schema: Value,
}

#[derive(Clone, Copy)]
enum ControlToolKind {
    ListPresets,
    ApplyPreset,
    ListKnobs,
    GetConfig,
    ApplyUpdates,
    ApplyPatch,
    Pause,
    Resume,
    Step,
    SetSpeed,
    GetStatus,
    Shutdown,
    GetCommandStatus,
}

impl ToolHandler for ControlTool {
    fn definition(&self) -> fastmcp_rust::Tool {
        fastmcp_rust::Tool {
            name: self.name.clone(),
            description: Some(self.description.clone()),
            input_schema: self.schema.clone(),
            annotations: None,
            icon: None,
            version: None,
            tags: Vec::new(),
            output_schema: None,
        }
    }

    fn call(&self, _ctx: &fastmcp_rust::McpContext, arguments: Value) -> McpResult<Vec<Content>> {
        let arguments = arguments.as_object().cloned().unwrap_or_default();
        match self.kind {
            ControlToolKind::ListPresets => {
                let presets: Vec<&'static str> =
                    PresetKind::all().iter().map(|p| p.as_str()).collect();
                make_tool_result(presets)
            }
            ControlToolKind::ApplyPreset => {
                let name_value =
                    arguments
                        .get("name")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            McpError::new(McpErrorCode::InvalidParams, "missing 'name' field")
                        })?;
                let kind = PresetKind::from_name(name_value).ok_or_else(|| {
                    McpError::new(
                        McpErrorCode::InvalidParams,
                        format!("unknown preset: {}", name_value),
                    )
                })?;
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.apply_patch(kind.patch()))?;
                make_tool_result(status)
            }
            ControlToolKind::ListKnobs => {
                let handle = self.handle.clone();
                let knobs = run_control_mcp_sync(move || handle.list_knobs())?;
                make_tool_result(knobs)
            }
            ControlToolKind::GetConfig => {
                let handle = self.handle.clone();
                let snapshot = run_control_mcp_sync(move || handle.snapshot())?;
                make_tool_result(snapshot)
            }
            ControlToolKind::ApplyUpdates => {
                let updates_value = arguments.get("updates").ok_or_else(|| {
                    McpError::new(McpErrorCode::InvalidParams, "missing 'updates' field")
                })?;
                let updates: Vec<KnobUpdate> = serde_json::from_value(updates_value.clone())
                    .map_err(|err| {
                        McpError::new(
                            McpErrorCode::InvalidParams,
                            format!("invalid updates payload: {err}"),
                        )
                    })?;
                if updates.is_empty() {
                    return Err(McpError::new(
                        McpErrorCode::InvalidParams,
                        "updates cannot be empty",
                    ));
                }
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.apply_updates(&updates))?;
                make_tool_result(status)
            }
            ControlToolKind::ApplyPatch => {
                let patch_value = arguments.get("patch").cloned().ok_or_else(|| {
                    McpError::new(McpErrorCode::InvalidParams, "missing 'patch' field")
                })?;
                if !patch_value.is_object() {
                    return Err(McpError::new(
                        McpErrorCode::InvalidParams,
                        "patch must be a JSON object",
                    ));
                }
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.apply_patch(patch_value))?;
                make_tool_result(status)
            }
            // These four returned the REQUEST echoed back as though it were the
            // result: `{"paused": true}` restated what was asked for, and
            // `{"stepped": count}` / `{"speed": speed}` echoed the caller's own
            // arguments. None of it was observed — a fabricated confirmation
            // reads exactly like a real one, and the receipt that could have
            // answered honestly was built and discarded (bd-2z0.4.9). This also
            // left the GetCommandStatus tool below able to look up ids that no
            // other tool ever handed out.
            ControlToolKind::Pause => {
                let key = mcp_idempotency_key(&arguments);
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.pause(key.as_deref()))?;
                make_tool_result(status)
            }
            ControlToolKind::Resume => {
                let key = mcp_idempotency_key(&arguments);
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.resume(key.as_deref()))?;
                make_tool_result(status)
            }
            ControlToolKind::Step => {
                let count = arguments.get("count").and_then(|v| v.as_u64()).unwrap_or(1);
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.step_count(count))?;
                make_tool_result(status)
            }
            ControlToolKind::SetSpeed => {
                let speed = arguments
                    .get("speed")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| {
                        McpError::new(McpErrorCode::InvalidParams, "missing 'speed' parameter")
                    })? as f32;
                let key = mcp_idempotency_key(&arguments);
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.set_speed(speed, key.as_deref()))?;
                make_tool_result(status)
            }
            ControlToolKind::GetStatus => {
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.status())?;
                make_tool_result(status)
            }
            ControlToolKind::Shutdown => {
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.shutdown())?;
                make_tool_result(status)
            }
            ControlToolKind::GetCommandStatus => {
                let command_id = arguments
                    .get("command_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        McpError::new(
                            McpErrorCode::InvalidParams,
                            "missing 'command_id' parameter",
                        )
                    })?
                    .to_string();
                let handle = self.handle.clone();
                let status = run_control_mcp_sync(move || handle.command_status(&command_id))?;
                make_tool_result(status)
            }
        }
    }
}

fn make_tool_result<T>(value: T) -> McpResult<Vec<Content>>
where
    T: Serialize,
{
    let pretty = serde_json::to_string_pretty(&value).map_err(|err| {
        McpError::new(
            McpErrorCode::InternalError,
            format!("failed to format result: {err}"),
        )
    })?;

    Ok(vec![Content::text(pretty)])
}

fn run_control_mcp_sync<T, F>(operation: F) -> Result<T, McpError>
where
    F: FnOnce() -> Result<T, ControlError>,
{
    operation().map_err(map_control_error)
}

/// MCP twin of [`run_control`]: a contended world mutex parks a blocking-pool
/// thread instead of the MCP server's async worker (bd-134).
#[allow(dead_code)]
async fn run_control_mcp<T, F>(operation: F) -> Result<T, McpError>
where
    F: FnOnce() -> Result<T, ControlError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|join_error| {
            McpError::new(
                McpErrorCode::InternalError,
                format!("control operation task failed: {join_error}"),
            )
        })?
        .map_err(map_control_error)
}

fn map_control_error(err: ControlError) -> McpError {
    match err {
        ControlError::UnknownPath(path) => McpError::new(
            McpErrorCode::InvalidParams,
            format!("unknown knob path: {path}"),
        ),
        ControlError::InvalidPatch(msg) => McpError::new(McpErrorCode::InvalidParams, msg),
        ControlError::Serialization(msg) => McpError::new(McpErrorCode::InternalError, msg),
        ControlError::Lock => {
            McpError::new(McpErrorCode::InternalError, "world state is unavailable")
        }
        ControlError::CommandQueueFull => McpError::new(
            McpErrorCode::InternalError,
            "command queue is full; retry shortly",
        ),
        ControlError::CommandQueueClosed => {
            McpError::new(McpErrorCode::InternalError, "command queue is closed")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{
        APPLICATION_STATE_ADMITTED, JOURNAL_STATE_NOT_REQUIRED, empty_latest_summary,
    };
    use scriptbots_core::{ScriptBotsConfig, WorldState};
    use serial_test::serial;
    use std::{
        collections::BTreeSet,
        ffi::OsString,
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::atomic::{AtomicBool, Ordering},
    };

    /// No transport may discard a control command's receipt.
    ///
    /// This is a GUARD over a shape, not a fix for one site. The same defect
    /// had appeared independently on three surfaces and each one looked like a
    /// local oversight until they were counted together: four REST endpoints
    /// answering a hardcoded `success: true`, four MCP tools echoing the
    /// caller's own arguments back as the result, and a CLI printing
    /// "✔ Simulation paused" off an HTTP 2xx. Meanwhile both MCP and the CLI
    /// shipped a command-status lookup for ids that no command ever returned
    /// (bd-2z0.4.9).
    ///
    /// A ninth instance is easy to add and hard to notice, so the shape is
    /// checked instead of the instances. A control call whose line both starts
    /// the statement and ends it with `;` bound nothing — the receipt went
    /// nowhere. Bound calls continue onto an expression, so they do not match.
    #[test]
    fn no_control_command_discards_its_receipt() {
        let source = include_str!("servers.rs");
        // Assembled at runtime: a source-scanning test that spells out its own
        // needle matches itself and passes forever.
        let openers = [
            format!("run_control(move || {}.", "state.handle"),
            format!("run_control_mcp_sync(move || {}.", "handle"),
        ];

        let discarded: Vec<&str> = source
            .lines()
            .map(str::trim)
            .filter(|line| {
                openers.iter().any(|opener| line.starts_with(opener)) && line.ends_with(';')
            })
            .collect();

        assert!(
            discarded.is_empty(),
            "these control calls throw away the receipt they were handed, leaving the \
             caller with no command id to poll: {discarded:#?}"
        );

        // Positive control: the bound form must actually be present, or this
        // test would pass just as happily against a file where every control
        // call had been deleted.
        assert!(
            source.contains("let status = run_control(move || state.handle.pause())"),
            "the receipt-bearing form is missing; this guard is checking nothing"
        );
    }

    /// Both transports must read a key, and both must treat blank as absent.
    ///
    /// The header and the MCP argument are separate extractors, so nothing but
    /// a test stops them drifting apart — and a key wired on one transport but
    /// not another is the which-control-did-you-use inconsistency this work
    /// keeps removing (bd-k7nq).
    #[test]
    fn every_transport_reads_an_idempotency_key_the_same_way() {
        let mut headers = HeaderMap::new();
        assert_eq!(idempotency_key(&headers), None, "no header is no key");

        headers.insert("Idempotency-Key", "  ".parse().expect("blank header"));
        assert_eq!(
            idempotency_key(&headers),
            None,
            "a blank key must be absent, not an empty-string key that every unkeyed retry \
             would collide on"
        );

        headers.insert("Idempotency-Key", " abc-123 ".parse().expect("header"));
        assert_eq!(
            idempotency_key(&headers).as_deref(),
            Some("abc-123"),
            "surrounding whitespace must be trimmed, or the same key sent twice would not match"
        );

        let mut args = serde_json::Map::new();
        assert_eq!(mcp_idempotency_key(&args), None);
        args.insert("idempotency_key".into(), Value::from("   "));
        assert_eq!(
            mcp_idempotency_key(&args),
            None,
            "MCP must apply the same blank-is-absent rule as HTTP"
        );
        args.insert("idempotency_key".into(), Value::from(" abc-123 "));
        assert_eq!(
            mcp_idempotency_key(&args).as_deref(),
            Some("abc-123"),
            "MCP and HTTP must derive the SAME key from the same client intent"
        );

        // Non-string values are absent, not coerced: a numeric key silently
        // stringified would collide across clients that never agreed on a type.
        args.insert("idempotency_key".into(), Value::from(7));
        assert_eq!(mcp_idempotency_key(&args), None);
    }

    fn handle() -> (ControlHandle, crate::command::CommandReceiver) {
        let world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let (sender, receiver) = create_command_bus(2);
        let handle = ControlHandle::new(
            Arc::new(std::sync::Mutex::new(world)),
            sender,
            empty_latest_summary(),
        );
        (handle, receiver)
    }

    fn shared_world() -> SharedWorld {
        Arc::new(std::sync::Mutex::new(
            WorldState::new(ScriptBotsConfig::default()).expect("world"),
        ))
    }

    fn unused_loopback_address() -> SocketAddr {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("reserve test port");
        let address = listener.local_addr().expect("reserved test address");
        drop(listener);
        address
    }

    fn two_unused_loopback_addresses() -> (SocketAddr, SocketAddr) {
        let first = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("reserve first test port");
        let second = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("reserve second test port");
        let addresses = (
            first.local_addr().expect("first test address"),
            second.local_addr().expect("second test address"),
        );
        drop((first, second));
        addresses
    }

    fn http_get(address: SocketAddr, path: &str) -> String {
        let mut stream = TcpStream::connect(address).expect("connect HTTP test client");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set HTTP test timeout");
        write!(
            stream,
            "GET {path} HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\n\r\n"
        )
        .expect("write HTTP test request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("read HTTP test response");
        response
    }

    fn http_post_json(address: SocketAddr, path: &str, body: &Value) -> String {
        let body = serde_json::to_string(body).expect("serialize HTTP test body");
        let mut stream = TcpStream::connect(address).expect("connect HTTP test client");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set HTTP test timeout");
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .expect("write HTTP test request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("read HTTP test response");
        response
    }

    const CONTROL_ENVIRONMENT_VARIABLES: [&str; 5] = [
        "SCRIPTBOTS_CONTROL_REST_ADDR",
        "SCRIPTBOTS_CONTROL_SWAGGER_PATH",
        "SCRIPTBOTS_CONTROL_REST_ENABLED",
        "SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR",
        "SCRIPTBOTS_CONTROL_MCP",
    ];

    struct ControlEnvironmentGuard(Vec<(&'static str, Option<OsString>)>);

    impl ControlEnvironmentGuard {
        fn cleared() -> Self {
            let saved = CONTROL_ENVIRONMENT_VARIABLES
                .into_iter()
                .map(|name| (name, env::var_os(name)))
                .collect();
            for name in CONTROL_ENVIRONMENT_VARIABLES {
                // SAFETY: serial tests isolate these process-global environment changes.
                unsafe { env::remove_var(name) };
            }
            Self(saved)
        }
    }

    impl Drop for ControlEnvironmentGuard {
        fn drop(&mut self) {
            for (name, value) in &self.0 {
                // SAFETY: serial tests isolate these process-global environment changes.
                unsafe {
                    if let Some(value) = value {
                        env::set_var(name, value);
                    } else {
                        env::remove_var(name);
                    }
                }
            }
        }
    }

    #[test]
    fn control_defaults_share_one_rest_socket_authority() {
        let config = ControlServerConfig::default();
        assert_eq!(config.rest_address, DEFAULT_CONTROL_REST_ADDRESS);
        assert_eq!(config.swagger_path, DEFAULT_CONTROL_SWAGGER_PATH);
        assert_eq!(
            default_control_rest_base_url(),
            format!("http://{}", config.rest_address)
        );
        assert!(matches!(
            config.mcp_transport,
            McpTransportConfig::Http { bind_address }
                if bind_address == DEFAULT_CONTROL_MCP_HTTP_ADDRESS
        ));
    }

    #[test]
    fn mcp_socket_parser_rejects_tls_urls_for_the_plaintext_server() {
        assert_eq!(
            parse_mcp_socket_addr("http://127.0.0.1:8090"),
            Some(DEFAULT_CONTROL_MCP_HTTP_ADDRESS)
        );
        assert_eq!(parse_mcp_socket_addr("https://127.0.0.1:8090"), None);
        assert_eq!(parse_mcp_socket_addr("HTTPS://127.0.0.1:8090"), None);
    }

    #[test]
    fn rest_port_conflict_is_reported_before_runtime_publication() {
        let occupied = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("occupy REST port");
        let address = occupied.local_addr().expect("occupied REST address");
        let config = ControlServerConfig {
            rest_address: address,
            rest_enabled: true,
            mcp_transport: McpTransportConfig::Disabled,
            ..ControlServerConfig::default()
        };

        let error = ControlRuntime::launch(shared_world(), empty_latest_summary(), config)
            .err()
            .expect("occupied REST address must fail startup");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("failed to reserve REST address"),
            "{rendered}"
        );
        assert!(rendered.contains(&address.to_string()), "{rendered}");
    }

    #[test]
    fn mcp_port_conflict_is_reported_before_runtime_publication() {
        let occupied = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("occupy MCP port");
        let address = occupied.local_addr().expect("occupied MCP address");
        let config = ControlServerConfig {
            rest_enabled: false,
            mcp_transport: McpTransportConfig::Http {
                bind_address: address,
            },
            ..ControlServerConfig::default()
        };

        let error = ControlRuntime::launch(shared_world(), empty_latest_summary(), config)
            .err()
            .expect("occupied MCP address must fail startup");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("failed to reserve MCP HTTP address"),
            "{rendered}"
        );
        assert!(rendered.contains(&address.to_string()), "{rendered}");
    }

    #[test]
    fn second_surface_failure_releases_the_prepared_rest_listener() {
        let shared_address = unused_loopback_address();
        let config = ControlServerConfig {
            rest_address: shared_address,
            rest_enabled: true,
            mcp_transport: McpTransportConfig::Http {
                bind_address: shared_address,
            },
            ..ControlServerConfig::default()
        };

        let error = ControlRuntime::launch(shared_world(), empty_latest_summary(), config)
            .err()
            .expect("MCP must not share the prepared REST listener");
        assert!(
            format!("{error:#}").contains("failed to reserve MCP HTTP address"),
            "unexpected error: {error:#}"
        );

        let rebound = TcpListener::bind(shared_address)
            .expect("failed transactional startup must release the REST listener");
        drop(rebound);
    }

    #[test]
    fn acknowledged_rest_startup_and_shutdown_own_the_listener_exactly_once() {
        let rest_address = unused_loopback_address();
        let config = ControlServerConfig {
            rest_address,
            rest_enabled: true,
            mcp_transport: McpTransportConfig::Disabled,
            ..ControlServerConfig::default()
        };

        let (runtime, _drain, _submit) =
            ControlRuntime::launch(shared_world(), empty_latest_summary(), config)
                .expect("REST startup");
        let stream = TcpStream::connect(rest_address)
            .expect("readiness acknowledgement must follow a listening REST socket");
        drop(stream);
        assert!(
            TcpListener::bind(rest_address).is_err(),
            "live runtime lost exclusive ownership of its listener"
        );

        runtime.shutdown().expect("acknowledged REST shutdown");
        let rebound =
            TcpListener::bind(rest_address).expect("shutdown must release the REST listener");
        drop(rebound);
    }

    #[test]
    fn acknowledged_mcp_startup_and_shutdown_own_the_listener_exactly_once() {
        let mcp_address = unused_loopback_address();
        let config = ControlServerConfig {
            rest_enabled: false,
            mcp_transport: McpTransportConfig::Http {
                bind_address: mcp_address,
            },
            ..ControlServerConfig::default()
        };
        let reservation = ControlServerReservation::prepare(config).expect("MCP reservation");
        assert_eq!(reservation.mcp_http_address(), Some(mcp_address));
        assert!(
            TcpListener::bind(mcp_address).is_err(),
            "reservation must hold the MCP socket before world construction"
        );

        let (runtime, _drain, _submit) = reservation
            .launch(shared_world(), empty_latest_summary())
            .expect("MCP startup");
        assert_eq!(runtime.status(), ControlRuntimeStatus::Running);
        let response = http_get(mcp_address, "/health");
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        assert!(response.contains("\"status\":\"healthy\""), "{response}");

        let status = runtime.subscribe_status();
        runtime.shutdown().expect("acknowledged MCP shutdown");
        assert_eq!(*status.borrow(), ControlRuntimeStatus::Stopped);
        let rebound =
            TcpListener::bind(mcp_address).expect("shutdown must release the MCP listener");
        drop(rebound);
    }

    #[test]
    fn acknowledged_rest_and_mcp_shutdown_release_both_reserved_listeners() {
        let (rest_address, mcp_address) = two_unused_loopback_addresses();
        let config = ControlServerConfig {
            rest_address,
            rest_enabled: true,
            mcp_transport: McpTransportConfig::Http {
                bind_address: mcp_address,
            },
            ..ControlServerConfig::default()
        };
        let reservation =
            ControlServerReservation::prepare(config).expect("transactional reservation");
        assert_eq!(reservation.rest_address(), Some(rest_address));
        assert_eq!(reservation.mcp_http_address(), Some(mcp_address));
        assert!(TcpListener::bind(rest_address).is_err());
        assert!(TcpListener::bind(mcp_address).is_err());

        let (runtime, _drain, _submit) = reservation
            .launch(shared_world(), empty_latest_summary())
            .expect("REST plus MCP startup");
        assert!(http_get(rest_address, "/api/knobs").starts_with("HTTP/1.1 200"));
        assert!(http_get(mcp_address, "/health").starts_with("HTTP/1.1 200"));
        runtime.shutdown().expect("combined control shutdown");

        let rest = TcpListener::bind(rest_address).expect("REST listener released");
        let mcp = TcpListener::bind(mcp_address).expect("MCP listener released");
        drop((rest, mcp));
    }

    #[test]
    fn mcp_http_errors_use_the_json_rpc_error_member() {
        let mcp_address = unused_loopback_address();
        let config = ControlServerConfig {
            rest_enabled: false,
            mcp_transport: McpTransportConfig::Http {
                bind_address: mcp_address,
            },
            ..ControlServerConfig::default()
        };
        let (runtime, _drain, _submit) =
            ControlRuntime::launch(shared_world(), empty_latest_summary(), config)
                .expect("MCP startup");

        let response = http_post_json(
            mcp_address,
            "/mcp",
            &json!({
                "jsonrpc": "2.0",
                "id": 41,
                "method": "scriptbots/definitely-unknown",
                "params": {},
            }),
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
        let (_, body) = response
            .split_once("\r\n\r\n")
            .expect("HTTP response contains a body delimiter");
        let message: Value = serde_json::from_str(body).expect("JSON-RPC response body");
        assert_eq!(message["id"], json!(41));
        assert!(message.get("error").is_some(), "{message}");
        assert!(message.get("result").is_none(), "{message}");

        runtime.shutdown().expect("MCP shutdown");
    }

    #[tokio::test]
    async fn post_ready_rest_failure_stops_mcp_and_publishes_the_root_error() {
        let rest_listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("REST fixture");
        let rest_address = rest_listener.local_addr().expect("REST fixture address");
        let mcp_listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("MCP fixture");
        let mcp_address = mcp_listener.local_addr().expect("MCP fixture address");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (trigger_tx, trigger_rx) = tokio::sync::oneshot::channel::<()>();
        let sibling_stopped = Arc::new(AtomicBool::new(false));

        let rest: ControlServerTask = tokio::spawn(async move {
            let _listener = rest_listener;
            trigger_rx
                .await
                .context("REST failure trigger disappeared")?;
            Err(anyhow!("injected REST serve failure"))
        });
        let stopped = Arc::clone(&sibling_stopped);
        let mut mcp_shutdown = shutdown_rx.clone();
        let mcp: ControlServerTask = tokio::spawn(async move {
            let _listener = mcp_listener;
            let _ = mcp_shutdown.wait_for(|stop| *stop).await;
            stopped.store(true, Ordering::SeqCst);
            Ok(())
        });
        let mut servers = RunningControlServers {
            rest: Some(rest),
            mcp: Some(mcp),
        };
        let (status_tx, status_rx) = watch::channel(ControlRuntimeStatus::Running);

        let trigger = async move {
            tokio::task::yield_now().await;
            trigger_tx.send(()).expect("deliver REST failure");
        };
        let (result, ()) = tokio::join!(
            servers.supervise(&shutdown_tx, shutdown_rx, &status_tx),
            trigger,
        );
        let error = result.expect_err("post-ready REST failure must terminate supervision");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("injected REST serve failure"),
            "{rendered}"
        );
        assert!(sibling_stopped.load(Ordering::SeqCst));
        assert!(matches!(
            status_rx.borrow().clone(),
            ControlRuntimeStatus::Failed(detail)
                if detail.contains("injected REST serve failure")
        ));

        let rest = TcpListener::bind(rest_address).expect("failed REST task released its port");
        let mcp = TcpListener::bind(mcp_address).expect("stopped MCP sibling released its port");
        drop((rest, mcp));
    }

    #[tokio::test]
    async fn post_ready_mcp_success_is_an_error_and_stops_rest() {
        let rest_listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("REST fixture");
        let rest_address = rest_listener.local_addr().expect("REST fixture address");
        let mcp_listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("MCP fixture");
        let mcp_address = mcp_listener.local_addr().expect("MCP fixture address");
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (trigger_tx, trigger_rx) = tokio::sync::oneshot::channel::<()>();
        let sibling_stopped = Arc::new(AtomicBool::new(false));

        let stopped = Arc::clone(&sibling_stopped);
        let mut rest_shutdown = shutdown_rx.clone();
        let rest: ControlServerTask = tokio::spawn(async move {
            let _listener = rest_listener;
            let _ = rest_shutdown.wait_for(|stop| *stop).await;
            stopped.store(true, Ordering::SeqCst);
            Ok(())
        });
        let mcp: ControlServerTask = tokio::spawn(async move {
            let _listener = mcp_listener;
            trigger_rx
                .await
                .context("MCP completion trigger disappeared")?;
            Ok(())
        });
        let mut servers = RunningControlServers {
            rest: Some(rest),
            mcp: Some(mcp),
        };
        let (status_tx, status_rx) = watch::channel(ControlRuntimeStatus::Running);

        let trigger = async move {
            tokio::task::yield_now().await;
            trigger_tx.send(()).expect("deliver MCP completion");
        };
        let (result, ()) = tokio::join!(
            servers.supervise(&shutdown_tx, shutdown_rx, &status_tx),
            trigger,
        );
        let error = result.expect_err("post-ready MCP completion must terminate supervision");
        assert_eq!(
            error.to_string(),
            "MCP HTTP control server stopped unexpectedly"
        );
        assert!(sibling_stopped.load(Ordering::SeqCst));
        assert_eq!(
            status_rx.borrow().clone(),
            ControlRuntimeStatus::Failed(
                "MCP HTTP control server stopped unexpectedly".to_string()
            )
        );

        let rest = TcpListener::bind(rest_address).expect("stopped REST sibling released its port");
        let mcp = TcpListener::bind(mcp_address).expect("completed MCP task released its port");
        drop((rest, mcp));
    }

    #[test]
    fn disabled_control_surfaces_still_have_a_safe_runtime_lifecycle() {
        let config = ControlServerConfig {
            rest_enabled: false,
            mcp_transport: McpTransportConfig::Disabled,
            ..ControlServerConfig::default()
        };
        let (runtime, _drain, _submit) =
            ControlRuntime::launch(shared_world(), empty_latest_summary(), config)
                .expect("disabled runtime startup");
        runtime.shutdown().expect("disabled runtime shutdown");
    }

    #[test]
    fn health_rejects_a_closed_channel_with_a_stale_live_status() {
        for cached in [
            ControlRuntimeStatus::Starting,
            ControlRuntimeStatus::Running,
        ] {
            let (status_tx, status_rx) = watch::channel(cached);
            drop(status_tx);
            assert_eq!(
                control_runtime_health(&status_rx),
                Err("control runtime terminated without publishing final status".to_string())
            );
        }

        let (status_tx, status_rx) = watch::channel(ControlRuntimeStatus::Starting);
        status_tx.send_replace(ControlRuntimeStatus::Running);
        drop(status_tx);
        assert_eq!(
            control_runtime_health(&status_rx),
            Err("control runtime terminated without publishing final status".to_string())
        );

        let (status_tx, status_rx) = watch::channel(ControlRuntimeStatus::Failed(
            "published root failure".to_string(),
        ));
        drop(status_tx);
        assert_eq!(
            control_runtime_health(&status_rx),
            Err("published root failure".to_string())
        );

        let (status_tx, status_rx) = watch::channel(ControlRuntimeStatus::Running);
        status_tx.send_replace(ControlRuntimeStatus::Failed(
            "late published root failure".to_string(),
        ));
        drop(status_tx);
        assert_eq!(
            control_runtime_health(&status_rx),
            Err("late published root failure".to_string())
        );
    }

    #[test]
    #[serial]
    fn invalid_control_environment_is_actionable_before_socket_reservation() {
        let _guard = ControlEnvironmentGuard::cleared();
        // SAFETY: this serial test owns the control environment variables.
        unsafe { env::set_var("SCRIPTBOTS_CONTROL_REST_ADDR", "definitely-not-a-socket") };

        let error = ControlServerConfig::try_from_env()
            .expect_err("invalid REST environment must fail configuration");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("SCRIPTBOTS_CONTROL_REST_ADDR"),
            "{rendered}"
        );
        assert!(rendered.contains("not a socket address"), "{rendered}");

        let captured = ControlServerConfig::from_env();
        let error = ControlServerReservation::prepare(captured)
            .err()
            .expect("legacy constructor must still fail at the reservation boundary");
        assert!(
            format!("{error:#}").contains("SCRIPTBOTS_CONTROL_REST_ADDR"),
            "{error:#}"
        );

        // SAFETY: this serial test owns the control environment variables.
        unsafe {
            env::remove_var("SCRIPTBOTS_CONTROL_REST_ADDR");
            env::set_var("SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR", "https://127.0.0.1:8090");
        }
        let error = ControlServerConfig::try_from_env()
            .expect_err("TLS URL must fail for the plaintext MCP server");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("requests TLS"), "{rendered}");
        assert!(rendered.contains("plaintext HTTP"), "{rendered}");
    }

    #[cfg(unix)]
    #[test]
    #[serial]
    fn non_unicode_control_environment_is_actionable() {
        use std::os::unix::ffi::OsStringExt;

        let _guard = ControlEnvironmentGuard::cleared();
        // SAFETY: this serial test owns the control environment variables.
        unsafe {
            env::set_var(
                "SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR",
                OsString::from_vec(vec![0xff]),
            )
        };

        let error = ControlServerConfig::try_from_env()
            .expect_err("non-Unicode MCP address must fail configuration");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR is not valid Unicode"),
            "{rendered}"
        );
    }

    /// THE SCREENSHOT ENDPOINT SERVES THE PRESENTED BUFFER, BYTE FOR BYTE.
    ///
    /// The identity that matters is served == exported: both go through
    /// `terminal::export::buffer_to_plain_text` over the same published buffer, so a
    /// caller polling REST and a user pressing `S` cannot get two different pictures
    /// of one frame. Asserted against the real handler rather than by inspection.
    #[tokio::test]
    async fn screenshot_ascii_serves_the_exact_presented_buffer() {
        let (handle, _receiver) = handle();
        let slot = empty_presented_frame();

        // A buffer with content at a known cell, so "served the right frame" is
        // distinguishable from "served an empty one".
        let area = ratatui::layout::Rect::new(0, 0, 12, 3);
        let mut buffer = ratatui::buffer::Buffer::empty(area);
        buffer[(0, 0)].set_symbol("Z");
        buffer[(11, 2)].set_symbol("Q");
        slot.store(Some(std::sync::Arc::new(PresentedTerminalFrame {
            tick: 4242,
            buffer: buffer.clone(),
        })));

        let state = ApiState {
            handle,
            scenario: None,
            presented_frame: std::sync::Arc::clone(&slot),
        };
        let Ok(response) = screenshot_ascii(State(state)).await else {
            panic!("a published frame must be served, not refused");
        };
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("x-scriptbots-frame-tick")
                .and_then(|value| value.to_str().ok()),
            Some("4242"),
            "the response must identify WHICH frame it served"
        );
        assert_eq!(
            response
                .headers()
                .get("x-scriptbots-frame-size")
                .and_then(|value| value.to_str().ok()),
            Some("12x3")
        );

        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("read screenshot body");
        let served = String::from_utf8(body.to_vec()).expect("screenshot must be UTF-8");
        assert_eq!(
            served,
            crate::terminal::export::buffer_to_plain_text(&buffer),
            "the served bytes must equal the export of the SAME buffer; if these \
             diverge there are two serializers again"
        );
        // Non-vacuity: the export must actually carry the painted cells, or the
        // equality above would hold for two empty strings.
        assert!(
            served.contains('Z') && served.contains('Q'),
            "the served frame must contain the painted cells: {served:?}"
        );
    }

    /// WITH NO PRESENTED FRAME, THE ENDPOINT REFUSES INSTEAD OF SUBSTITUTING.
    ///
    /// This is the acceptance criterion's refusal path, and it is the whole point of
    /// the change: the old handler answered every request by re-rasterizing the
    /// world, so a headless or GUI-mode process returned a synthesized map under the
    /// name "screenshot". A 409 that explains itself is strictly more useful than a
    /// 200 that is not what it claims (bd-0oro).
    #[tokio::test]
    async fn screenshot_ascii_refuses_when_no_frame_was_presented() {
        let (handle, _receiver) = handle();
        let state = ApiState {
            handle,
            scenario: None,
            presented_frame: empty_presented_frame(),
        };
        let Err(error) = screenshot_ascii(State(state)).await else {
            panic!("an unpresented frame must refuse, not synthesize one");
        };
        assert_eq!(
            error.status,
            StatusCode::CONFLICT,
            "the refusal must be a 409: the request is well-formed and the process \
             simply has no frame, which is neither the caller's error nor a missing \
             resource"
        );
        assert!(
            error
                .message
                .contains("no terminal frame has been presented"),
            "the refusal must say what is missing: {}",
            error.message
        );
        assert!(
            error.message.contains("re-rasterized"),
            "and must name what it is deliberately NOT doing, so the next reader does \
             not restore the fallback as a convenience: {}",
            error.message
        );
    }

    #[tokio::test]
    async fn rest_patch_rejects_non_finite_value_with_field_path_before_admission() {
        let (handle, receiver) = handle();
        let state = ApiState {
            handle,
            scenario: None,
            presented_frame: empty_presented_frame(),
        };
        let result = patch_config(
            State(state.clone()),
            Json(ConfigPatchRequest {
                patch: json!({"food_growth_rate": "NaN"}),
            }),
        )
        .await;
        assert!(result.is_err(), "REST patch accepted non-finite input");
        let Err(error) = result else {
            return;
        };
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(
            error.message.contains("food_growth_rate"),
            "REST error did not identify field: {}",
            error.message
        );
        assert!(matches!(
            receiver.try_recv(),
            Err(crate::command::CommandRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn mcp_patch_rejects_non_finite_value_with_field_path_before_admission() {
        let (handle, receiver) = handle();
        let tool = ControlTool {
            handle,
            kind: ControlToolKind::ApplyPatch,
            name: "apply_patch".to_string(),
            description: "Merge a JSON object patch".to_string(),
            schema: json!({"type": "object"}),
        };
        let ctx = fastmcp_rust::McpContext::new(asupersync::Cx::for_testing(), 1);
        let arguments = json!({"patch": {"food_growth_rate": "Infinity"}});
        let error = tool
            .call(&ctx, arguments)
            .expect_err("MCP patch accepted non-finite input");
        let rendered = error.to_string();
        assert!(
            rendered.contains("food_growth_rate"),
            "MCP error did not identify field: {rendered}"
        );
        assert!(matches!(
            receiver.try_recv(),
            Err(crate::command::CommandRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn scenario_endpoint_reports_the_runs_identity_or_404() {
        // No scenario plumbed: the endpoint must say so honestly rather than invent one.
        let (bare_handle, _bare_receiver) = handle();
        let bare = ApiState {
            handle: bare_handle,
            scenario: None,
            presented_frame: empty_presented_frame(),
        };
        let missing = match get_scenario(State(bare)).await {
            Ok(_) => panic!("a server without scenario context must 404"),
            Err(error) => error,
        };
        assert_eq!(missing.status, StatusCode::NOT_FOUND);

        // With the run's scenario plumbed, the endpoint reports exactly it.
        let mut identity = ScenarioIdentityV0::caller_seeded("fixture-equilibrium-study");
        identity.schema_version = 1;
        identity.bootstrap_ticks = 12;
        let (state_handle, _state_receiver) = handle();
        let state = ApiState {
            handle: state_handle,
            scenario: Some(Arc::new(identity)),
            presented_frame: empty_presented_frame(),
        };
        let Json(view) = match get_scenario(State(state)).await {
            Ok(view) => view,
            Err(error) => panic!("scenario view must resolve: {}", error.message),
        };
        let rendered = serde_json::to_value(&view).expect("view serializes");
        assert_eq!(rendered["id"], "fixture-equilibrium-study");
        assert_eq!(rendered["schema_version"], 1);
        assert_eq!(rendered["bootstrap_ticks"], 12);
    }

    /// The two-axis status must stay honest all the way out to JSON.
    ///
    /// `bd-f65w` fixed the values at the `ControlHandle` boundary, where they had been
    /// hardcoded to `applied`/`durable` the instant a command was enqueued. Nothing
    /// asserted they survive serialization, so a future DTO or handler change could
    /// reintroduce the fabricated pair without a single test failing. These two routes
    /// are also part of the eight that had drifted out of the OpenAPI document
    /// (bd-01dg) and had no coverage at all.
    #[tokio::test]
    async fn control_routes_report_admitted_and_unjournaled_status_as_json() {
        let (control, _receiver) = handle();
        let state = ApiState {
            handle: control,
            scenario: None,
            presented_frame: empty_presented_frame(),
        };

        // `AppError` is deliberately not `Debug` (it carries a client-facing message),
        // so surface `message` explicitly rather than reaching for `expect`.
        let pause = match post_control_pause(State(state.clone()), HeaderMap::new()).await {
            Ok(Json(status)) => status,
            Err(error) => panic!("pause must be accepted: {}", error.message),
        };
        let body = serde_json::to_value(&pause).expect("status serializes");
        assert_eq!(
            body["application_state"], APPLICATION_STATE_ADMITTED,
            "enqueueing proves admission order, never application: {body}"
        );
        assert_eq!(
            body["journal_state"], JOURNAL_STATE_NOT_REQUIRED,
            "the legacy bus writes no lifecycle record, so no journal state may be claimed: {body}"
        );
        assert_eq!(body["admission_sequence"], 1);

        // The lookup route must return the same record rather than inventing progress:
        // nothing on this path can advance either axis.
        let looked_up = match get_control_status(
            State(state.clone()),
            axum::extract::Path(pause.command_id.clone()),
        )
        .await
        {
            Ok(Json(status)) => status,
            Err(error) => panic!("status lookup must succeed: {}", error.message),
        };
        assert_eq!(
            serde_json::to_value(&looked_up).expect("status serializes"),
            body
        );

        // An unknown ID is a typed 404 Not Found, not a fabricated terminal status or 200 null.
        let missing = get_control_status(
            State(state),
            axum::extract::Path("cmd-does-not-exist".to_owned()),
        )
        .await
        .expect_err("unknown command id must return 404 not found");
        assert_eq!(missing.status, StatusCode::NOT_FOUND);
    }

    /// The MCP roster README documents and that `ControlToolKind` dispatches.
    ///
    /// Registration is hand-written per tool, so a `ControlToolKind` variant can be
    /// added and dispatched while never being registered — an MCP tool that simply
    /// does not exist, invisible to `tools/list` and to every client. Asserting set
    /// equality against the built server catches both directions: a variant that lost
    /// its registration, and a tool registered under a name the docs do not claim.
    #[test]
    fn mcp_tool_roster_is_complete() {
        let (control, _receiver) = handle();
        let server = register_control_tools(
            fastmcp_rust::ServerBuilder::new("scriptbots-control-test", "0.0.0"),
            control,
        )
        .build();

        let registered: BTreeSet<String> =
            server.tools().into_iter().map(|tool| tool.name).collect();
        let expected: BTreeSet<String> = [
            "list_presets",
            "apply_preset",
            "list_knobs",
            "get_config",
            "apply_updates",
            "apply_patch",
            "pause",
            "resume",
            "step",
            "set_speed",
            "get_status",
            "shutdown",
            "get_command_status",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();

        assert_eq!(
            registered,
            expected,
            "MCP tool roster drifted; missing: {:?}, unexpected: {:?}",
            expected.difference(&registered).collect::<Vec<_>>(),
            registered.difference(&expected).collect::<Vec<_>>()
        );
    }

    /// Collect the `/api/...` literal that follows each occurrence of `marker`.
    ///
    /// Used to read the router's route table and the `#[utoipa::path]` annotations
    /// straight out of this file's implementation half, so the conformance test
    /// below compares real registrations instead of a hand-copied list.
    fn api_paths_following(source: &str, marker: &str) -> BTreeSet<String> {
        let mut found = BTreeSet::new();
        let mut rest = source;
        while let Some(offset) = rest.find(marker) {
            rest = &rest[offset + marker.len()..];
            let Some(open) = rest.find('"') else {
                break;
            };
            // Only a literal that starts immediately after the marker (modulo
            // whitespace) belongs to it; anything else is unrelated code.
            if rest[..open].trim().is_empty() {
                let literal = &rest[open + 1..];
                if let Some(close) = literal.find('"') {
                    let value = &literal[..close];
                    if value.starts_with("/api/") {
                        found.insert(value.to_owned());
                    }
                }
            }
        }
        found
    }

    /// The Swagger UI and `/api-docs/openapi.json` are the advertised control-API
    /// contract, so a handler that is routed but missing from `ApiDoc`'s `paths(...)`
    /// silently disappears from that contract (bd-01dg: eight endpoints, including the
    /// README-documented `/api/screenshot/ascii`, `/api/screenshot/png`, and
    /// `/api/ticks/ndjson`, had drifted out of the spec). Compare the three sets that
    /// must agree — routed, annotated, and published — rather than spot-checking names.
    #[test]
    fn test_openapi_spec_conformance() {
        // The test module mentions paths in assertions and fixtures; only the
        // implementation half of the file declares real routes.
        let implementation = include_str!("servers.rs")
            .split("#[cfg(test)]\nmod tests {")
            .next()
            .expect("split always yields the implementation prefix");

        let routed = api_paths_following(implementation, ".route(");
        let annotated = api_paths_following(implementation, "path = ");
        assert!(
            !routed.is_empty(),
            "route extraction found nothing; the router shape must have changed"
        );

        let spec = serde_json::to_value(ApiDoc::openapi()).expect("OpenAPI spec serializes");
        let published: BTreeSet<String> = spec["paths"]
            .as_object()
            .expect("OpenAPI documents always carry a paths object")
            .keys()
            .cloned()
            .collect();

        assert_eq!(
            annotated,
            routed,
            "every `#[utoipa::path]` annotation must describe a registered route and vice versa; \
             annotated-but-unrouted: {:?}, routed-but-unannotated: {:?}",
            annotated.difference(&routed).collect::<Vec<_>>(),
            routed.difference(&annotated).collect::<Vec<_>>()
        );
        assert_eq!(
            published,
            routed,
            "the published OpenAPI document must expose exactly the registered routes; \
             missing from spec: {:?}, present only in spec: {:?}",
            routed.difference(&published).collect::<Vec<_>>(),
            published.difference(&routed).collect::<Vec<_>>()
        );
    }
}
