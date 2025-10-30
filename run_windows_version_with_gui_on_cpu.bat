@echo off
setlocal enableextensions enabledelayedexpansion

REM Force Windows MSVC target and isolate artifacts
set CARGO_TARGET_DIR=target-windows-msvc
set CARGO_BUILD_TARGET=
set CARGO_ENCODED_RUSTFLAGS=
set RUSTFLAGS=
set RUSTC_LINKER=

REM Use all cores for faster builds
set JOBS=%NUMBER_OF_PROCESSORS%

REM Force CPU canvas renderer and skip wgpu adapter probing
set SB_RENDERER=canvas
set SCRIPTBOTS_FORCE_GUI=1
set SCRIPTBOTS_FORCE_TERMINAL=
set WGPU_BACKEND=
set WGPU_POWER_PREFERENCE=
set SB_WGPU_PRESENT_MODE=
set SB_WGPU_RES_SCALE=
set SB_WGPU_MAX_FPS=

REM Launch ScriptBots in release mode with a GUI window driven by the CPU canvas
cargo run -p scriptbots-app --bin scriptbots-app --release --target x86_64-pc-windows-msvc -j %JOBS% --features gui -- --mode gui --threads 8

endlocal
