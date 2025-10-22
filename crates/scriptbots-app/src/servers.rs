//! Network control surfaces (REST, Swagger, MCP) are defined here.

use crate::{control::ControlHandle, SharedWorld};

/// Configuration for starting control servers.
#[derive(Debug, Clone)]
pub struct ControlServerConfig {
    pub rest_address: std::net::SocketAddr,
    pub swagger_path: String,
    pub mcp_transport: McpTransportConfig,
}

impl Default for ControlServerConfig {
    fn default() -> Self {
        Self {
            rest_address: "127.0.0.1:8080".parse().expect("valid default addr"),
            swagger_path: "/docs".to_string(),
            mcp_transport: McpTransportConfig::Stdio,
        }
    }
}

/// Transport configuration for the MCP server.
#[derive(Debug, Clone)]
pub enum McpTransportConfig {
    Stdio,
    #[allow(dead_code)]
    Tcp(std::net::SocketAddr),
}

/// Spawn REST and MCP servers.
pub async fn spawn_servers(
    _world: SharedWorld,
    _handle: ControlHandle,
    _config: ControlServerConfig,
) -> anyhow::Result<()> {
    anyhow::bail!("control servers not yet implemented")
}
