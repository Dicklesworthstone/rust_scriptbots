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

REM Launch ScriptBots with GPU GUI in release mode with tuned threads
set WGPU_BACKEND=Vulkan
set WGPU_POWER_PREFERENCE=high_performance
set SB_WGPU_PRESENT_MODE=diff
set SB_WGPU_RES_SCALE=1.0
set SB_WGPU_MAX_FPS=60
set SB_WGPU_BLOOM=1
set SB_WGPU_TONEMAP=1
set SB_WGPU_FOG=low
cargo run -p scriptbots-app --bin scriptbots-app --release --target x86_64-pc-windows-msvc -j %JOBS% --features gui -- --mode gui --threads 8

endlocal
