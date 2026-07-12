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
                if let Err(err) = apply_control_command(world, command) {
                    warn!(?err, "failed to apply control command");
                }
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
    Arc::new(move |command: ControlCommand| {
        if let Err(error) = command.validate() {
            warn!(%error, "rejected invalid control command before queue admission");
            return false;
        }
        match sender.try_send(command) {
            Ok(()) => true,
            Err(TrySendError::Full(cmd)) => {
                warn!(?cmd, "control command queue full; dropping command");
                false
            }
            Err(TrySendError::Disconnected(cmd)) => {
                warn!(?cmd, "control command queue disconnected");
                false
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scriptbots_core::{ScriptBotsConfig, SimulationCommand};

    #[test]
    fn submit_rejects_non_finite_speed_before_queue_admission() {
        let (sender, receiver) = create_command_bus(1);
        let submit = make_command_submit(sender);
        assert!(!(submit)(ControlCommand::UpdateSimulation(
            SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(f32::NAN),
                step_once: false,
            }
        )));
        assert!(matches!(receiver.try_recv(), Err(TryRecvError::Empty)));
    }

    #[test]
    fn finite_speed_is_admitted_and_clamped_when_applied() {
        let (sender, receiver) = create_command_bus(1);
        let submit = make_command_submit(sender);
        assert!((submit)(ControlCommand::UpdateSimulation(
            SimulationCommand {
                paused: Some(false),
                speed_multiplier: Some(128.0),
                step_once: false,
            }
        )));

        let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world");
        drain_pending_commands(&receiver, &mut world);
        let pending = world.drain_simulation_commands();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].speed_multiplier, Some(32.0));
    }
}
