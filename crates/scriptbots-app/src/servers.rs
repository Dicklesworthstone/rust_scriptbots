use std::{
    collections::HashMap,
    env,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::{Arc, mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use axum::response::sse::{Event, Sse};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::stream::{Stream, StreamExt};
use mcp_protocol_sdk::{
    core::error::McpResult,
    prelude::*,
    protocol::types::{
        JsonRpcError, JsonRpcMessage, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse,
        error_codes,
    },
    server::McpServer,
};
use scriptbots_core::PresetKind;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::convert::Infallible;
use tokio::sync::watch;
use tokio_stream::wrappers::IntervalStream;
use tracing::{error, info, warn};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use crate::SharedWorld;
use crate::command::{
    CommandDrain, CommandSubmit, create_command_bus, make_command_drain, make_command_submit,
};
use crate::control::{
    AgentScoreEntry, ConfigSnapshot, ControlError, ControlHandle, DietClassDto, EventEntry,
    EventKind, HydrologySnapshot, KnobEntry, KnobUpdate, Scoreboard, SelectionModeDto,
    SelectionStateDto,
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

/// Configuration for the hosted control surfaces.
#[derive(Debug, Clone)]
pub struct ControlServerConfig {
    pub rest_address: SocketAddr,
    pub swagger_path: String,
    pub rest_enabled: bool,
    pub mcp_transport: McpTransportConfig,
    /// Environment parsing failures retained until the fallible launch boundary.
    #[doc(hidden)]
    pub environment_errors: Vec<String>,
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
            environment_errors: Vec::new(),
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
    ) -> Result<(ControlRuntime, CommandDrain, CommandSubmit)> {
        ControlRuntime::launch_reserved_with_timeout(world, self, CONTROL_STARTUP_TIMEOUT)
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
        config: ControlServerConfig,
    ) -> Result<(Self, CommandDrain, CommandSubmit)> {
        ControlServerReservation::prepare(config)?.launch(world)
    }

    fn launch_reserved_with_timeout(
        world: SharedWorld,
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
        let handle = ControlHandle::new(world.clone(), command_tx.clone());

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
            ControlRuntimeStatus::Stopped => return Err("control runtime stopped".to_string()),
            ControlRuntimeStatus::Starting | ControlRuntimeStatus::Running => {}
        }

        match probe.has_changed() {
            Ok(true) => continue,
            Err(_) => {
                return Err(
                    "control runtime terminated without publishing final status".to_string()
                );
            }
            Ok(false) => {
                return match cached {
                    ControlRuntimeStatus::Starting => {
                        Err("control runtime is still starting".to_string())
                    }
                    ControlRuntimeStatus::Running => Ok(()),
                    ControlRuntimeStatus::Stopped | ControlRuntimeStatus::Failed(_) => {
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

    status.send_replace(ControlRuntimeStatus::Running);
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
}

#[derive(Debug, Serialize, ToSchema)]
struct ErrorResponse {
    message: String,
}

#[derive(Debug, Serialize, ToSchema)]
struct PresetList {
    presets: Vec<&'static str>,
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

#[derive(Debug, Serialize, ToSchema)]
struct SelectionAcknowledge {
    queued: bool,
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
        get_events_tail,
        get_scoreboard,
        get_agents_debug,
        get_config_audit,
        list_presets,
        apply_preset,
        post_selection
    ),
    components(
        schemas(
            KnobEntry,
            KnobUpdate,
            ConfigSnapshot,
            ConfigPatchRequest,
            KnobApplyRequest,
            ConfigAuditEntryView,
            PresetList,
            PresetApplyRequest,
            ErrorResponse,
            EventEntry,
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
            SelectionAcknowledge
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
                let summary =
                    tokio::task::spawn_blocking(move || handle.latest_summary()).await;
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

/// Return an ASCII snapshot of the current world mini-map.
#[utoipa::path(
    get,
    path = "/api/screenshot/ascii",
    tag = "control",
    responses((status = 200, description = "ASCII screenshot", content_type = "text/plain"))
)]
async fn screenshot_ascii(State(state): State<ApiState>) -> Result<Response, AppError> {
    let text = run_control(move || state.handle.ascii_map()).await?;
    Ok((StatusCode::OK, text).into_response())
}

/// Return a PNG screenshot placeholder for GPUI view (basic 1x1 pixel placeholder for now).
#[utoipa::path(
    get,
    path = "/api/screenshot/png",
    tag = "control",
    responses((status = 200, description = "PNG screenshot", content_type = "image/png"))
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
                let summary =
                    tokio::task::spawn_blocking(move || handle.latest_summary()).await;
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
    responses((status = 202, body = SelectionAcknowledge), (status = 400, body = ErrorResponse))
)]
async fn post_selection(
    State(state): State<ApiState>,
    Json(body): Json<SelectionUpdateRequestBody>,
) -> Result<(StatusCode, Json<SelectionAcknowledge>), AppError> {
    let update: SelectionUpdate = body.into();
    run_control(move || state.handle.update_selection(update)).await?;
    Ok((
        StatusCode::ACCEPTED,
        Json(SelectionAcknowledge { queued: true }),
    ))
}

#[utoipa::path(
    patch,
    path = "/api/config",
    tag = "control",
    request_body = ConfigPatchRequest,
    responses(
        (status = 200, body = ConfigSnapshot),
        (status = 400, body = ErrorResponse)
    )
)]
async fn patch_config(
    State(state): State<ApiState>,
    Json(payload): Json<ConfigPatchRequest>,
) -> Result<Json<ConfigSnapshot>, AppError> {
    let snapshot = run_control(move || state.handle.apply_patch(payload.patch)).await?;
    Ok(Json(snapshot))
}

#[utoipa::path(
    post,
    path = "/api/knobs/apply",
    tag = "control",
    request_body = KnobApplyRequest,
    responses(
        (status = 200, body = ConfigSnapshot),
        (status = 400, body = ErrorResponse)
    )
)]
async fn apply_updates(
    State(state): State<ApiState>,
    Json(payload): Json<KnobApplyRequest>,
) -> Result<Json<ConfigSnapshot>, AppError> {
    if payload.updates.is_empty() {
        return Err(AppError::bad_request("updates cannot be empty"));
    }
    let snapshot = run_control(move || state.handle.apply_updates(&payload.updates)).await?;
    Ok(Json(snapshot))
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
    responses((status = 200, body = ConfigSnapshot), (status = 400, body = ErrorResponse))
)]
async fn apply_preset(
    State(state): State<ApiState>,
    Json(payload): Json<PresetApplyRequest>,
) -> Result<Json<ConfigSnapshot>, AppError> {
    let Some(kind) = PresetKind::from_name(&payload.name) else {
        return Err(AppError::bad_request(format!(
            "unknown preset: {}",
            payload.name
        )));
    };
    let snapshot = run_control(move || state.handle.apply_patch(kind.patch())).await?;
    Ok(Json(snapshot))
}

fn prepare_rest_server(
    handle: ControlHandle,
    config: &ControlServerConfig,
    reserved: ReservedControlListener,
) -> Result<PreparedRestServer> {
    let state = ApiState { handle };
    let mut openapi = ApiDoc::openapi();
    openapi.info.version = env!("CARGO_PKG_VERSION").to_string();

    let api_router = Router::new()
        .route("/api/knobs", get(get_knobs))
        .route("/api/config", get(get_config).patch(patch_config))
        .route("/api/knobs/apply", post(apply_updates))
        // Tick summaries (JSON one-shot and SSE stream)
        .route("/api/ticks/latest", get(get_latest_tick_summary))
        .route("/api/ticks/stream", get(stream_ticks_sse))
        // Screenshots
        .route("/api/screenshot/ascii", get(screenshot_ascii))
        .route("/api/screenshot/png", get(screenshot_png))
        .route("/api/ticks/ndjson", get(stream_ticks_ndjson))
        .route("/api/hydrology", get(get_hydrology_snapshot))
        // Event tail and scoreboard
        .route("/api/events/tail", get(get_events_tail))
        .route("/api/scoreboard", get(get_scoreboard))
        .route("/api/agents/debug", get(get_agents_debug))
        .route("/api/selection", post(post_selection))
        // Presets and audit
        .route("/api/presets", get(list_presets))
        .route("/api/presets/apply", post(apply_preset))
        .route("/api/config/audit", get(get_config_audit))
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
    let server = Arc::new(McpServer::new(
        "scriptbots-control".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    ));

    register_tool(
        Arc::clone(&server),
        "list_presets",
        "List available scenario presets",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::ListPresets,
        handle.clone(),
    )
    .await
    .map_err(|err| anyhow!("failed to register list_presets tool: {err}"))?;

    register_tool(
        Arc::clone(&server),
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
    )
    .await
    .map_err(|err| anyhow!("failed to register apply_preset tool: {err}"))?;

    register_tool(
        Arc::clone(&server),
        "list_knobs",
        "List all exposed configuration knobs",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::ListKnobs,
        handle.clone(),
    )
    .await
    .map_err(|err| anyhow!("failed to register list_knobs tool: {err}"))?;

    register_tool(
        Arc::clone(&server),
        "get_config",
        "Fetch the entire simulation configuration",
        json!({"type": "object", "additionalProperties": false}),
        ControlToolKind::GetConfig,
        handle.clone(),
    )
    .await
    .map_err(|err| anyhow!("failed to register get_config tool: {err}"))?;

    register_tool(
        Arc::clone(&server),
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
    )
    .await
    .map_err(|err| anyhow!("failed to register apply_updates tool: {err}"))?;

    register_tool(
        Arc::clone(&server),
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
        handle,
    )
    .await
    .map_err(|err| anyhow!("failed to register apply_patch tool: {err}"))?;

    let router = Router::new()
        .route("/mcp", post(handle_mcp_http_request))
        .route("/mcp/notify", post(handle_mcp_http_notification))
        .route("/mcp/events", get(handle_mcp_http_events))
        .route("/health", get(handle_mcp_http_health))
        .with_state(server);
    let listener = tokio::net::TcpListener::from_std(reserved.listener)
        .context("failed to adopt reserved MCP HTTP listener")?;

    Ok(PreparedMcpServer {
        address: reserved.address,
        listener,
        router,
    })
}

async fn register_tool(
    server: Arc<McpServer>,
    name: &str,
    description: &str,
    schema: Value,
    kind: ControlToolKind,
    handle: ControlHandle,
) -> McpResult<()> {
    server
        .add_tool(
            name.to_string(),
            Some(description.to_string()),
            schema,
            ControlTool { handle, kind },
        )
        .await
}

async fn handle_mcp_http_request(
    State(server): State<Arc<McpServer>>,
    Json(request): Json<JsonRpcRequest>,
) -> Json<JsonRpcMessage> {
    let request_id = request.id.clone();
    match server.handle_request(request).await {
        Ok(response) => Json(normalize_mcp_http_response(response)),
        Err(error) => Json(JsonRpcMessage::Error(JsonRpcError::error(
            request_id,
            error_codes::INTERNAL_ERROR,
            error.to_string(),
            None,
        ))),
    }
}

fn normalize_mcp_http_response(response: JsonRpcResponse) -> JsonRpcMessage {
    let protocol_error = response.result.as_ref().and_then(|result| {
        let result = result.as_object()?;
        if result.len() != 1 {
            return None;
        }
        let error = result.get("error")?.as_object()?;
        let code = error
            .get("code")?
            .as_i64()
            .and_then(|code| i32::try_from(code).ok())?;
        let message = error.get("message")?.as_str()?.to_string();
        let data = error.get("data").cloned();
        Some((code, message, data))
    });

    if let Some((code, message, data)) = protocol_error {
        JsonRpcMessage::Error(JsonRpcError::error(response.id, code, message, data))
    } else {
        JsonRpcMessage::Response(response)
    }
}

async fn handle_mcp_http_notification(
    Json(_notification): Json<JsonRpcNotification>,
) -> StatusCode {
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
}

#[derive(Clone, Copy)]
enum ControlToolKind {
    ListPresets,
    ApplyPreset,
    ListKnobs,
    GetConfig,
    ApplyUpdates,
    ApplyPatch,
}

#[async_trait]
impl ToolHandler for ControlTool {
    async fn call(&self, arguments: HashMap<String, Value>) -> McpResult<ToolResult> {
        match self.kind {
            ControlToolKind::ListPresets => {
                let presets: Vec<&'static str> =
                    PresetKind::all().iter().map(|p| p.as_str()).collect();
                Ok(make_tool_result(presets)?)
            }
            ControlToolKind::ApplyPreset => {
                let name_value = arguments
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| McpError::Validation("missing 'name' field".into()))?;
                let kind = PresetKind::from_name(name_value).ok_or_else(|| {
                    McpError::Validation(format!("unknown preset: {}", name_value))
                })?;
                let handle = self.handle.clone();
                let snapshot = run_control_mcp(move || handle.apply_patch(kind.patch())).await?;
                Ok(make_tool_result(snapshot)?)
            }
            ControlToolKind::ListKnobs => {
                let handle = self.handle.clone();
                let knobs = run_control_mcp(move || handle.list_knobs()).await?;
                Ok(make_tool_result(knobs)?)
            }
            ControlToolKind::GetConfig => {
                let handle = self.handle.clone();
                let snapshot = run_control_mcp(move || handle.snapshot()).await?;
                Ok(make_tool_result(snapshot)?)
            }
            ControlToolKind::ApplyUpdates => {
                let updates_value = arguments
                    .get("updates")
                    .ok_or_else(|| McpError::Validation("missing 'updates' field".into()))?;
                let updates: Vec<KnobUpdate> = serde_json::from_value(updates_value.clone())
                    .map_err(|err| {
                        McpError::Validation(format!("invalid updates payload: {err}"))
                    })?;
                if updates.is_empty() {
                    return Err(McpError::Validation("updates cannot be empty".into()));
                }
                let handle = self.handle.clone();
                let snapshot = run_control_mcp(move || handle.apply_updates(&updates)).await?;
                Ok(make_tool_result(snapshot)?)
            }
            ControlToolKind::ApplyPatch => {
                let patch_value = arguments
                    .get("patch")
                    .cloned()
                    .ok_or_else(|| McpError::Validation("missing 'patch' field".into()))?;
                if !patch_value.is_object() {
                    return Err(McpError::Validation("patch must be a JSON object".into()));
                }
                let handle = self.handle.clone();
                let snapshot = run_control_mcp(move || handle.apply_patch(patch_value)).await?;
                Ok(make_tool_result(snapshot)?)
            }
        }
    }
}

fn make_tool_result<T>(value: T) -> McpResult<ToolResult>
where
    T: Serialize,
{
    let structured = serde_json::to_value(&value)
        .map_err(|err| McpError::Internal(format!("failed to serialize result: {err}")))?;
    let pretty = serde_json::to_string_pretty(&structured)
        .map_err(|err| McpError::Internal(format!("failed to format result: {err}")))?;

    Ok(ToolResult {
        content: vec![Content::text(pretty)],
        is_error: Some(false),
        structured_content: Some(structured),
        meta: None,
    })
}

/// MCP twin of [`run_control`]: a contended world mutex parks a blocking-pool
/// thread instead of the MCP server's async worker (bd-134).
async fn run_control_mcp<T, F>(operation: F) -> Result<T, McpError>
where
    F: FnOnce() -> Result<T, ControlError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|join_error| {
            McpError::Internal(format!("control operation task failed: {join_error}"))
        })?
        .map_err(map_control_error)
}

fn map_control_error(err: ControlError) -> McpError {
    match err {
        ControlError::UnknownPath(path) => {
            McpError::Validation(format!("unknown knob path: {path}"))
        }
        ControlError::InvalidPatch(msg) => McpError::Validation(msg),
        ControlError::Serialization(msg) => McpError::Internal(msg),
        ControlError::Lock => McpError::Internal("world state is unavailable".into()),
        ControlError::CommandQueueFull => {
            McpError::Internal("command queue is full; retry shortly".into())
        }
        ControlError::CommandQueueClosed => McpError::Internal("command queue is closed".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{ScriptBotsConfig, WorldState};
    use serial_test::serial;
    use std::{
        ffi::OsString,
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::atomic::{AtomicBool, Ordering},
    };

    fn handle() -> (ControlHandle, crate::command::CommandReceiver) {
        let world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        let (sender, receiver) = create_command_bus(2);
        let handle = ControlHandle::new(Arc::new(std::sync::Mutex::new(world)), sender);
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

        let error = ControlRuntime::launch(shared_world(), config)
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

        let error = ControlRuntime::launch(shared_world(), config)
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

        let error = ControlRuntime::launch(shared_world(), config)
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
            ControlRuntime::launch(shared_world(), config).expect("REST startup");
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

        let (runtime, _drain, _submit) = reservation.launch(shared_world()).expect("MCP startup");
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
            .launch(shared_world())
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
            ControlRuntime::launch(shared_world(), config).expect("MCP startup");

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
            ControlRuntime::launch(shared_world(), config).expect("disabled runtime startup");
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

    #[tokio::test]
    async fn rest_patch_rejects_non_finite_value_with_field_path_before_admission() {
        let (handle, receiver) = handle();
        let state = ApiState { handle };
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
        };
        let arguments =
            HashMap::from([("patch".to_owned(), json!({"food_growth_rate": "Infinity"}))]);
        let error = tool
            .call(arguments)
            .await
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
}
