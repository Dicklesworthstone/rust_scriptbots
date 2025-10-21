use anyhow::Result;
use scriptbots_render::run_demo;
use tracing::info;

fn main() -> Result<()> {
    init_tracing();
    info!("Starting ScriptBots simulation shell");
    run_demo();
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}
