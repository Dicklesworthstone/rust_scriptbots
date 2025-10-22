use crossfire::{
    channel::TryRecvError,
    mpmc::{self, MAsyncTx, MRx},
    util,
};
use scriptbots_core::WorldState;
use scriptbots_core::ScriptBotsConfig;
use tracing::debug;

#[derive(Debug, Clone)]
pub enum ControlCommand {
    UpdateConfig(ScriptBotsConfig),
}

pub type CommandSender = MAsyncTx<ControlCommand>;
pub type CommandReceiver = MRx<ControlCommand>;

pub fn create_command_bus(capacity: usize) -> (CommandSender, CommandReceiver) {
    util::detect_backoff_cfg();
    mpmc::bounded_tx_async_rx_blocking(capacity)
}

pub fn drain_pending_commands(receiver: &CommandReceiver, world: &mut WorldState) {
    loop {
        match receiver.try_recv() {
            Ok(ControlCommand::UpdateConfig(config)) => {
                debug!("Applying config update via control command");
                *world.config_mut() = config;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }
}
