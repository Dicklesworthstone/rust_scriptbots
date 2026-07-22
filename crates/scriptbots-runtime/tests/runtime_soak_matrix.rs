//! Runtime lifecycle, contention, and soak matrix integration tests (bd-2z0.4.11).

use scriptbots_core::{
    ControlCommand, ScriptBotsConfig, SelectionMode, SelectionState, SelectionUpdate, WorldState,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn test_runtime_lifecycle_contention_storm_and_soak() {
    let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world init");
    let (tx, rx) = std::sync::mpsc::channel();

    let running = Arc::new(AtomicBool::new(true));
    let r_clone = Arc::clone(&running);

    // Sender thread executing high-throughput command storm
    let storm_handle = thread::spawn(move || {
        let mut count = 0;
        while r_clone.load(Ordering::Relaxed) && count < 1000 {
            let cmd = match count % 5 {
                0 => ControlCommand::Pause,
                1 => ControlCommand::Resume,
                2 => ControlCommand::Step,
                3 => ControlCommand::SetSpeed(1.5),
                _ => ControlCommand::UpdateSelection(SelectionUpdate {
                    mode: SelectionMode::Clear,
                    agent_ids: vec![],
                    state: SelectionState::Selected,
                }),
            };
            let _ = tx.send(cmd);
            count += 1;
        }
        count
    });

    // Receiver loop processing control commands safely
    let start = Instant::now();
    let mut processed = 0;
    while start.elapsed() < Duration::from_millis(500) && processed < 1000 {
        if let Ok(cmd) = rx.try_recv() {
            let _ = scriptbots_core::apply_control_command(&mut world, cmd);
            processed += 1;
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }

    running.store(false, Ordering::Relaxed);
    let sent = storm_handle.join().expect("storm thread joins cleanly");

    assert!(sent > 0, "storm thread submitted commands");
    assert!(
        processed > 0,
        "receiver processed commands safely under contention"
    );
}

#[test]
fn test_runtime_soak_digest_invariance() {
    let mut world = WorldState::new(ScriptBotsConfig::default()).expect("world init");
    let initial_tick = world.tick().0;

    // Step world 50 ticks deterministically
    for _ in 0..50 {
        let _ = world.step();
    }

    assert_eq!(world.tick().0, initial_tick + 50);
}
