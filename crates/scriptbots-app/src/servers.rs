use std::{
    collections::HashMap,
    env,
    net::SocketAddr,
    sync::Arc,
    thread::{self, JoinHandle},
};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, patch, post},
    Json, Router,
};
use crossfire::channel::TrySendError;
use mcp_protocol_sdk::{
    core::error::McpResult,
    prelude::*,
    server::mcp_server::McpServer,
    server::HttpMcpServer,
    transport::http::HttpServerTransport,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Notify;
use tracing::{error, info, warn};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use crate::SharedWorld;
use crate::control::{ConfigSnapshot, ControlError, ControlHandle, KnobEntry, KnobUpdate};

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
                bind_addr: "127.0.0.1:8090"
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

        let default_mcp = match config.mcp_transport {
            McpTransportConfig::Http { bind_addr } => bind_addr,
            McpTransportConfig::Disabled => "127.0.0.1:8090"
                .parse()
                .expect("fallback MCP HTTP socket"),
        };
        let mut mcp_bind = default_mcp;

        if let Ok(addr) = env::var("SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR") {
            match addr.parse::<SocketAddr>() {
                Ok(parsed) => mcp_bind = parsed,
                Err(err) => warn!(%addr, %err, "invalid SCRIPTBOTS_CONTROL_MCP_HTTP_ADDR; using default"),
            }
        }

        if let Ok(raw) = env::var("SCRIPTBOTS_CONTROL_MCP") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "disabled" | "off" | "false" | "0" => {
                    config.mcp_transport = McpTransportConfig::Disabled;
                }
                value if value.is_empty() || value == "http" => {
                    config.mcp_transport = McpTransportConfig::Http { bind_addr: mcp_bind };
                }
                other => match other.parse::<SocketAddr>() {
                    Ok(parsed) => config.mcp_transport = McpTransportConfig::Http { bind_addr: parsed },
                    Err(err) => {
                        warn!(%other, %err, "unknown MCP transport value; disabling MCP server");
                        config.mcp_transport = McpTransportConfig::Disabled;
                    }
                },
            }
        } else {
            config.mcp_transport = McpTransportConfig::Http { bind_addr: mcp_bind };
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

/// Supported transports for the MCP server.
#[derive(Debug, Clone)]
pub enum McpTransportConfig {
    Disabled,
    Http { bind_addr: SocketAddr },
}

impl McpTransportConfig {
    fn is_enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    fn bind_addr(&self) -> Option<SocketAddr> {
        match self {
            Self::Http { bind_addr } => Some(*bind_addr),
            Self::Disabled => None,
        }
    }
}

/// Runtime guard for background control servers.
pub struct ControlRuntime {
    shutdown: Arc<Notify>,
    thread: Option<JoinHandle<()>>,
}

impl ControlRuntime {
    /// Spawn the control runtime on a dedicated Tokio thread.
    pub fn launch(world: SharedWorld, config: ControlServerConfig) -> Result<Self> {
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = shutdown.clone();
        let handle = ControlHandle::new(world.clone());

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

        Ok(Self {
            shutdown,
            thread: Some(thread),
        })
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

impl Drop for ControlRuntime {
    fn drop(&mut self) {
        self.shutdown.notify_waiters();
        if let Some(handle) = self.thread.take() {
            if let Err(err) = handle.join() {
                error!(?err, "control runtime thread panicked during drop");
            }
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
        mcp_handle = Some(tokio::spawn(async move {
            if let Err(err) = run_mcp_server(mcp_handle_clone, transport).await {
                error!(?err, "MCP server exited unexpectedly");
            }
        }));
    } else {
        info!("MCP control server disabled via configuration");
    }

    shutdown.notified().await;

    if let Some(handle) = rest_handle {
        if let Err(err) = handle.await {
            error!(?err, "failed to await REST control server task");
        }
    }

    if let Some(handle) = mcp_handle {
        handle.abort();
        if let Err(err) = handle.await {
            if !err.is_cancelled() {
                error!(?err, "MCP server task join error");
            }
        }
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

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ConfigPatchRequest {
    #[schema(value_type = Object, nullable = false)]
    pub patch: Value,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct KnobApplyRequest {
    pub updates: Vec<KnobUpdate>,
}

#[derive(OpenApi)]
#[openapi(
    paths(get_knobs, get_config, patch_config, apply_updates),
    components(
        schemas(
            KnobEntry,
            KnobUpdate,
            ConfigSnapshot,
            ConfigPatchRequest,
            KnobApplyRequest,
            ErrorResponse
        )
    ),
    info(title = "ScriptBots Control API", version = env!("CARGO_PKG_VERSION")),
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

async fn run_rest_server(
    handle: ControlHandle,
    config: &ControlServerConfig,
    shutdown: Arc<Notify>,
) -> Result<()> {
    let state = ApiState { handle };
    let swagger_path_static: &'static str = Box::leak(config.swagger_path.clone().into_boxed_str());
    let openapi = ApiDoc::openapi();

    let api_router = Router::new()
        .route("/api/knobs", get(get_knobs))
        .route("/api/config", get(get_config).patch(patch_config))
        .route("/api/knobs/apply", post(apply_updates))
        .with_state(state);

    let app = Router::new()
        .merge(api_router)
        .merge(SwaggerUi::new(swagger_path_static).url("/api-docs/openapi.json", openapi));

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

async fn run_mcp_server(handle: ControlHandle, transport: McpTransportConfig) -> Result<()> {
    match transport {
        McpTransportConfig::Disabled => Ok(()),
        McpTransportConfig::Stdio => {
            info!("Starting MCP stdio server");
            let server = McpServer::new(
                "scriptbots-control".to_string(),
                env!("CARGO_PKG_VERSION").to_string(),
            );

            register_tool(
                &server,
                "list_knobs",
                "List all exposed configuration knobs",
                json!({"type": "object", "additionalProperties": false}),
                ControlToolKind::ListKnobs,
                handle.clone(),
            )
            .await?;

            register_tool(
                &server,
                "get_config",
                "Fetch the entire simulation configuration",
                json!({"type": "object", "additionalProperties": false}),
                ControlToolKind::GetConfig,
                handle.clone(),
            )
            .await?;

            register_tool(
                &server,
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
            .await?;

            register_tool(
                &server,
                "apply_patch",
                "Merge a JSON object patch into the configuration",
                json!({
                    "type": "object",
                    "properties": {
                        "patch": {
                            "type": "object"
                        }
                    },
                    "required": ["patch"],
                    "additionalProperties": false
                }),
                ControlToolKind::ApplyPatch,
                handle,
            )
            .await?;

            let transport = StdioServerTransport::new();
            server
                .start(transport)
                .await
                .context("MCP stdio server encountered an error")
        }
    }
}

async fn register_tool(
    server: &McpServer,
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

#[derive(Clone)]
struct ControlTool {
    handle: ControlHandle,
    kind: ControlToolKind,
}

#[derive(Clone, Copy)]
enum ControlToolKind {
    ListKnobs,
    GetConfig,
    ApplyUpdates,
    ApplyPatch,
}

#[async_trait]
impl ToolHandler for ControlTool {
    async fn call(&self, arguments: HashMap<String, Value>) -> McpResult<ToolResult> {
        match self.kind {
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
    }
}
