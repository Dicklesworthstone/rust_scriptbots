use crossfire::mpmc;
use crossfire::{MAsyncTx, MRx, TryRecvError, TrySendError, detect_backoff_cfg};
use scriptbots_core::{ControlCommand, WorldState, apply_control_command};
use std::sync::Arc;
use tracing::{debug, warn};

pub type CommandSender = MAsyncTx<ControlCommand>;
pub type CommandReceiver = MRx<ControlCommand>;
pub type CommandDrain = Arc<dyn Fn(&mut WorldState) + Send + Sync>;
pub type CommandSubmit = Arc<dyn Fn(ControlCommand) -> bool + Send + Sync>;

pub fn create_command_bus(capacity: usize) -> (CommandSender, CommandReceiver) {
    detect_backoff_cfg();
    mpmc::bounded_tx_async_rx_blocking(capacity)
}

pub fn drain_pending_commands(receiver: &CommandReceiver, world: &mut WorldState) {
    loop {
        match receiver.try_recv() {
            Ok(command) => {
                debug!(?command, "applying control command");
                apply_control_command(world, command);
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => break,
        }
    }
}

pub fn make_command_drain(receiver: CommandReceiver) -> CommandDrain {
    let receiver = Arc::new(receiver);
    Arc::new(move |world: &mut WorldState| {
        drain_pending_commands(&receiver, world);
    })
}

pub fn make_command_submit(sender: CommandSender) -> CommandSubmit {
    let sender = Arc::new(sender);
    Arc::new(
        move |command: ControlCommand| match sender.try_send(command) {
            Ok(()) => true,
            Err(TrySendError::Full(cmd)) => {
                warn!(?cmd, "control command queue full; dropping command");
                false
            }
            Err(TrySendError::Disconnected(cmd)) => {
                warn!(?cmd, "control command queue disconnected");
                false
            }
        },
    )
}
