# Vendored `crossterm` 0.29.0 (SIGWINCH coalescing fix)

This directory vendors **crossterm 0.29.0** with a single upstream fix applied to
`src/event/source/unix/mio.rs`, and the workspace root pins it via:

```toml
[patch.crates-io]
crossterm = { path = "vendor/crossterm" }
```

## Why

`scriptbots-app`'s terminal shell reads input through `ftui`'s crossterm backend
(`ftui-core` `crossterm` feature → the real `crossterm` 0.29.0 crate, via
`ftui_runtime::CrosstermEventSource` → `ftui_core::terminal_session::TerminalSession`
→ `crossterm::event::read()`). In crossterm 0.29.0, `UnixInternalEventSource::try_read`
iterates the ready mio tokens and **returns on the first event**. When a `SIGWINCH`
(terminal resize) and a pending key byte are ready in the *same* mio poll, the `TTY_TOKEN`
arm returns the key and the co-ready `SIGNAL_TOKEN` is abandoned. mio is edge-triggered,
so the un-serviced `SIGWINCH` is never reported again and the `Event::Resize` is **lost**.

This is the mechanism behind the flaky failure in
`crates/scriptbots-app/tests/terminal_end_to_end.rs::live_frankentui_pty_routes_and_restores_terminal`
(a shrink-resize immediately followed by a keypress left the presented frame at the pre-resize width
until the interaction-before oracle timed out).

Measured (running the test itself, unpatched vs patched builds): **0/5 → 5/5** passing on this
revision, and **3/20 → 20/20** on an earlier validation revision. At the crossterm level, a forced
key+SIGWINCH co-batch drops an event **40/40** of the time unpatched and **0/40** patched (both
token orders). The patched build passed every measured run (25/25).

## The fix

The patch is **crossterm upstream PR #1154 by mxaddict, verbatim** (fixes
[crossterm-rs/crossterm#1126](https://github.com/crossterm-rs/crossterm/issues/1126);
PR [crossterm-rs/crossterm#1154](https://github.com/crossterm-rs/crossterm/pull/1154),
"fix(event): keep a SIGWINCH that comes with input").
It changes `try_read` to handle **every** token in the batch before returning: the
`SIGNAL_TOKEN` arm queues the `Resize` onto `parser.internal_events` (instead of returning
it), the `TTY_TOKEN` read loop `break`s once events are queued (instead of returning), and a
single `parser.next()` after the loop returns one event — so a `SIGWINCH` co-batched with a
key byte is no longer dropped. Only `src/event/source/unix/mio.rs` is touched.

## Provenance / removal

- Base crate: `crossterm` 0.29.0, extracted from the **sha256-verified** `crossterm-0.29.0.crate`
  tarball from crates.io (its checksum matches the `crossterm 0.29.0` entry in the workspace
  `Cargo.lock`), unmodified except the hunk below.
- Applied change: PR #1154's `mio.rs` hunk, verbatim (see `vendor/crossterm/MIO_FIX.diff`).
- **Remove this vendored copy and the `[patch.crates-io] crossterm` line once a published
  `crossterm` release contains #1154** (then bump the normal dependency to that version).

The exact hunk is in `MIO_FIX.diff` beside this file.
