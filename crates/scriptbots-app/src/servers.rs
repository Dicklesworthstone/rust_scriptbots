use std::{
    collections::HashMap,
    env,
    net::SocketAddr,
    sync::Arc,
    thread::{self, JoinHandle},
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
    server::{HttpMcpServer, McpServer},
    transport::http::HttpServerTransport,
};
use scriptbots_core::PresetKind;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{convert::Infallible, time::Duration};
use tokio::sync::{Mutex, Notify};
use tokio_stream::wrappers::IntervalStream;
use tracing::{error, info, warn};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use crate::SharedWorld;
use crate::command::{
    CommandDrain, CommandSubmit, create_command_bus, make_command_drain, make_command_submit,
};
use crate::control::{
    AgentScoreEntry, ConfigSnapshot, ControlError, ControlHandle, DietClass, EventEntry, EventKind,
    HydrologySnapshot, KnobEntry, KnobUpdate, Scoreboard,
};
use scriptbots_core::ConfigAuditEntry;
use scriptbots_core::TickSummaryDto;

const DEFAULT_MCP_HTTP_ADDR: &str = "127.0.0.1:8090";

/// Configuration for the hosted control surfaces.
#[derive(Debug, Clone)]
pub struct ControlServerConfig {
    pub rest_address: SocketAddr,
    pub swagger_path: String,
    pub rest_enabled: bool,
    pub mcp_transport: McpTransportConfig,
}

impl Default for ControlServerConfig {
    fn default() -> Self {
        Self {
            rest_address: "127.0.0.1:8088"
                .parse()
                .expect("hard-coded loopback socket"),
            swagger_path: "/docs".to_string(),
            rest_enabled: true,
            mcp_transport: McpTransportConfig::Http {
                bind_address: DEFAULT_MCP_HTTP_ADDR
                    .parse()
                    .expect("hard-coded MCP HTTP socket"),
            },
        }
    }
}

impl ControlServerConfig {
    /// Build configuration from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(addr) = env::var("SCRIPTBOTS_CONTROL_REST_ADDR") {
            match addr.parse::<SocketAddr>() {
                Ok(addr) => config.rest_address = addr,
                Err(err) => {
                    warn!(%addr, %err, "invalid SCRIPTBOTS_CONTROL_REST_ADDR; using default")
                }
            }
        }

        if let Ok(path) = env::var("SCRIPTBOTS_CONTROL_SWAGGER_PATH") {
            let sanitized = sanitize_swagger_path(&path);
            if sanitized != path {
                warn!(original = %path, sanitized = %sanitized, "sanitized swagger path");
            }
            config.swagger_path = sanitized;
        }

        if let Ok(flag) = env::var("SCRIPTBOTS_CONTROL_REST_ENABLED") {
            config.rest_enabled = matches!(
                flag.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }

        let mut http_override = None;
        if let Ok(addr) = env::var("SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR") {
            match parse_mcp_socket_addr(&addr) {
                Some(parsed) => http_override = Some(parsed),
                None => warn!(%addr, "invalid SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR; ignoring override"),
            }
        }

        if let Ok(raw) = env::var("SCRIPTBOTS_CONTROL_MCP") {
            let trimmed = raw.trim();
            match trimmed.to_ascii_lowercase().as_str() {
                "disabled" | "off" | "false" | "0" => {
                    config.mcp_transport = McpTransportConfig::Disabled;
                }
                "http" | "" => {
                    let bind_address = http_override
                        .or_else(|| parse_mcp_socket_addr(DEFAULT_MCP_HTTP_ADDR))
                        .expect("valid default MCP HTTP address");
                    config.mcp_transport = McpTransportConfig::Http { bind_address };
                }
                _other => {
                    if let Some(addr) = parse_mcp_socket_addr(trimmed) {
                        config.mcp_transport = McpTransportConfig::Http { bind_address: addr };
                    } else if let Some(addr) = http_override {
                        warn!(value = %raw, fallback = %addr, "could not parse SCRIPTBOTS_CONTROL_MCP; using HTTP override");
                        config.mcp_transport = McpTransportConfig::Http { bind_address: addr };
                    } else {
                        warn!(%raw, "unknown MCP transport; disabling MCP server");
                        config.mcp_transport = McpTransportConfig::Disabled;
                    }
                }
            }
        } else if let Some(addr) = http_override {
            config.mcp_transport = McpTransportConfig::Http { bind_address: addr };
        }

        config
    }
}

fn sanitize_swagger_path(path: &str) -> String {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return "/docs".to_string();
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

    if let Ok(addr) = trimmed.parse::<SocketAddr>() {
        return Some(addr);
    }

    let normalized = trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("https://"))
        .unwrap_or(trimmed);
    let host_port = normalized.split('/').next().unwrap_or(normalized);
    host_port.parse::<SocketAddr>().ok()
}

/// Supported transports for the MCP server.
#[derive(Debug, Clone)]
pub enum McpTransportConfig {
    Disabled,
    Http { bind_address: SocketAddr },
}

impl McpTransportConfig {
    fn is_enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

/// Runtime guard for background control servers.
pub struct ControlRuntime {
    shutdown: Arc<Notify>,
    thread: Option<JoinHandle<()>>,
}

impl ControlRuntime {
    /// Spawn the control runtime on a dedicated Tokio thread.
    pub fn launch(
        world: SharedWorld,
        config: ControlServerConfig,
    ) -> Result<(Self, CommandDrain, CommandSubmit)> {
        let (command_tx, command_rx) = create_command_bus(32);
        let command_drain = make_command_drain(command_rx);
        let command_submit = make_command_submit(command_tx.clone());
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();
        let handle = ControlHandle::new(world.clone(), command_tx.clone());

        let thread = thread::Builder::new()
            .name("scriptbots-control".into())
            .spawn(move || {
                match tokio::runtime::Builder::new_multi_thread()
                    .thread_name("scriptbots-control-rt")
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime.block_on(async move {
                        if let Err(err) =
                            run_control_servers(handle, config, shutdown_clone.clone()).await
                        {
                            error!(?err, "control servers terminated with error");
                        }
                    }),
                    Err(err) => error!(?err, "failed to build Tokio runtime for control servers"),
                }
            })?;

        Ok((
            Self {
                shutdown,
                thread: Some(thread),
            },
            command_drain,
            command_submit,
        ))
    }

    /// Trigger a graceful shutdown and block until the background thread exits.
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown.notify_waiters();
        if let Some(handle) = self.thread.take() {
            handle
                .join()
                .map_err(|err| anyhow!("control thread panicked: {err:?}"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
impl ControlRuntime {
    /// Create a no-op runtime for tests without starting background threads.
    pub fn dummy() -> (Self, CommandDrain, CommandSubmit) {
        let (command_tx, command_rx) = create_command_bus(4);
        let command_drain = make_command_drain(command_rx);
        let command_submit = make_command_submit(command_tx);
        let runtime = Self {
            shutdown: Arc::new(Notify::new()),
            thread: None,
        };
        (runtime, command_drain, command_submit)
    }
}

impl Drop for ControlRuntime {
    fn drop(&mut self) {
        self.shutdown.notify_waiters();
        if let Some(handle) = self.thread.take()
            && let Err(err) = handle.join()
        {
            error!(?err, "control runtime thread panicked during drop");
        }
    }
}

async fn run_control_servers(
    handle: ControlHandle,
    config: ControlServerConfig,
    shutdown: Arc<Notify>,
) -> Result<()> {
    let mut rest_handle = None;
    let mut mcp_handle = None;

    if config.rest_enabled {
        let rest_handle_clone = handle.clone();
        let rest_shutdown = shutdown.clone();
        let rest_config = config.clone();
        rest_handle = Some(tokio::spawn(async move {
            if let Err(err) = run_rest_server(rest_handle_clone, &rest_config, rest_shutdown).await
            {
                error!(?err, "REST control server exited unexpectedly");
            }
        }));
    } else {
        info!("REST control server disabled via configuration");
    }

    if config.mcp_transport.is_enabled() {
        let mcp_handle_clone = handle.clone();
        let transport = config.mcp_transport.clone();
        let mcp_shutdown = shutdown.clone();
        mcp_handle = Some(tokio::spawn(async move {
            if let Err(err) = run_mcp_server(mcp_handle_clone, transport, mcp_shutdown).await {
                error!(?err, "MCP server exited unexpectedly");
            }
        }));
    } else {
        info!("MCP control server disabled via configuration");
    }

    shutdown.notified().await;

    if let Some(handle) = rest_handle
        && let Err(err) = handle.await
    {
        error!(?err, "failed to await REST control server task");
    }

    if let Some(handle) = mcp_handle
        && let Err(err) = handle.await
    {
        error!(?err, "MCP server task join error");
    }

    Ok(())
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
        get_config_audit,
        list_presets,
        apply_preset
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
            DietClass,
            AgentScoreEntry,
            Scoreboard,
            HydrologySnapshot
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

#[utoipa::path(
    get,
    path = "/api/knobs",
    tag = "control",
    responses((status = 200, body = [KnobEntry]))
)]
async fn get_knobs(State(state): State<ApiState>) -> Result<Json<Vec<KnobEntry>>, AppError> {
    let knobs = state.handle.list_knobs()?;
    Ok(Json(knobs))
}

#[utoipa::path(
    get,
    path = "/api/config",
    tag = "control",
    responses((status = 200, body = ConfigSnapshot))
)]
async fn get_config(State(state): State<ApiState>) -> Result<Json<ConfigSnapshot>, AppError> {
    let snapshot = state.handle.snapshot()?;
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
    let summary = state.handle.latest_summary()?;
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
    match state.handle.hydrology_snapshot()? {
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
                let event = match handle.latest_summary() {
                    Ok(summary) => {
                        let json = serde_json::to_string(&TickSummaryDto::from(summary))
                            .unwrap_or_else(|_| "{}".to_string());
                        Event::default().data(json)
                    }
                    Err(_) => Event::default().data("{}"),
                };
                Ok::<Event, Infallible>(event)
            }
        });
    Ok(Sse::new(stream))
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
    let events = state.handle.events_tail(limit)?;
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
    let board = state.handle.compute_scoreboard(limit)?;
    Ok(Json(board))
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
    let snapshot = state.handle.apply_patch(payload.patch)?;
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
    let snapshot = state.handle.apply_updates(&payload.updates)?;
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
    let entries = state
        .handle
        .audit()?
        .into_iter()
        .map(ConfigAuditEntryView::from)
        .collect();
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
    let snapshot = state.handle.apply_patch(kind.patch())?;
    Ok(Json(snapshot))
}

async fn run_rest_server(
    handle: ControlHandle,
    config: &ControlServerConfig,
    shutdown: Arc<Notify>,
) -> Result<()> {
    let state = ApiState { handle };
    let swagger_path_static: &'static str = Box::leak(config.swagger_path.clone().into_boxed_str());
    let mut openapi = ApiDoc::openapi();
    openapi.info.version = env!("CARGO_PKG_VERSION").to_string();

    let api_router = Router::new()
        .route("/api/knobs", get(get_knobs))
        .route("/api/config", get(get_config).patch(patch_config))
        .route("/api/knobs/apply", post(apply_updates))
        // Tick summaries (JSON one-shot and SSE stream)
        .route("/api/ticks/latest", get(get_latest_tick_summary))
        .route("/api/ticks/stream", get(stream_ticks_sse))
        .route("/api/hydrology", get(get_hydrology_snapshot))
        // Event tail and scoreboard
        .route("/api/events/tail", get(get_events_tail))
        .route("/api/scoreboard", get(get_scoreboard))
        // Presets and audit
        .route("/api/presets", get(list_presets))
        .route("/api/presets/apply", post(apply_preset))
        .route("/api/config/audit", get(get_config_audit))
        .with_state(state);

    let swagger_router: Router<_> = SwaggerUi::new(swagger_path_static)
        .url("/api-docs/openapi.json", openapi)
        .into();

    let app = Router::new().merge(api_router).merge(swagger_router);

    let listener = tokio::net::TcpListener::bind(config.rest_address)
        .await
        .with_context(|| format!("failed to bind REST address {}", config.rest_address))?;

    info!(address = %config.rest_address, "REST control server listening");

    let shutdown_signal = shutdown.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal.notified().await;
        })
        .await
        .context("REST control server errored")
}

async fn run_mcp_server(
    handle: ControlHandle,
    transport: McpTransportConfig,
    shutdown: Arc<Notify>,
) -> Result<()> {
    match transport {
        McpTransportConfig::Disabled => Ok(()),
        McpTransportConfig::Http { bind_address } => {
            info!(address = %bind_address, "Starting MCP HTTP server");
            let mut http_server = HttpMcpServer::new(
                "scriptbots-control".to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
            );

            let server_arc = http_server.server().await;
            register_tool(
                server_arc.clone(),
                "list_presets",
                "List available scenario presets",
                json!({"type": "object", "additionalProperties": false}),
                ControlToolKind::ListPresets,
                handle.clone(),
            )
            .await
            .map_err(|err| anyhow!("failed to register list_presets tool: {err}"))?;

            register_tool(
                server_arc.clone(),
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
                server_arc.clone(),
                "list_knobs",
                "List all exposed configuration knobs",
                json!({"type": "object", "additionalProperties": false}),
                ControlToolKind::ListKnobs,
                handle.clone(),
            )
            .await
            .map_err(|err| anyhow!("failed to register list_knobs tool: {err}"))?;

            register_tool(
                server_arc.clone(),
                "get_config",
                "Fetch the entire simulation configuration",
                json!({"type": "object", "additionalProperties": false}),
                ControlToolKind::GetConfig,
                handle.clone(),
            )
            .await
            .map_err(|err| anyhow!("failed to register get_config tool: {err}"))?;

            register_tool(
                server_arc.clone(),
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
                server_arc,
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

            let transport = HttpServerTransport::new(bind_address.to_string());
            http_server
                .start(transport)
                .await
                .context("failed to start MCP HTTP server")?;

            info!(address = %bind_address, "MCP HTTP server listening");
            shutdown.notified().await;
            http_server
                .stop()
                .await
                .context("failed to stop MCP HTTP server")?;
            Ok(())
        }
    }
}

async fn register_tool(
    server: Arc<Mutex<McpServer>>,
    name: &str,
    description: &str,
    schema: Value,
    kind: ControlToolKind,
    handle: ControlHandle,
) -> McpResult<()> {
    let guard = server.lock().await;
    guard
        .add_tool(
            name.to_string(),
            Some(description.to_string()),
            schema,
            ControlTool { handle, kind },
        )
        .await
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
                let snapshot = self
                    .handle
                    .apply_patch(kind.patch())
                    .map_err(map_control_error)?;
                Ok(make_tool_result(snapshot)?)
            }
            ControlToolKind::ListKnobs => {
                let knobs = self.handle.list_knobs().map_err(map_control_error)?;
                Ok(make_tool_result(knobs)?)
            }
            ControlToolKind::GetConfig => {
                let snapshot = self.handle.snapshot().map_err(map_control_error)?;
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
                let snapshot = self
                    .handle
                    .apply_updates(&updates)
                    .map_err(map_control_error)?;
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
                let snapshot = self
                    .handle
                    .apply_patch(patch_value)
                    .map_err(map_control_error)?;
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
