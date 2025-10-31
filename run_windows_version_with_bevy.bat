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

REM Prefer high-performance WGPU adapter for Bevy renderer (respect overrides)
if not defined WGPU_BACKEND set WGPU_BACKEND=Vulkan
if not defined WGPU_POWER_PREFERENCE set WGPU_POWER_PREFERENCE=high_performance
if not defined SB_WGPU_PRESENT_MODE set SB_WGPU_PRESENT_MODE=full
if not defined SB_WGPU_RES_SCALE set SB_WGPU_RES_SCALE=1.0
if not defined SB_WGPU_MAX_FPS set SB_WGPU_MAX_FPS=60
if not defined SB_WGPU_BLOOM set SB_WGPU_BLOOM=1
if not defined SB_WGPU_TONEMAP set SB_WGPU_TONEMAP=1
if not defined SB_WGPU_FOG set SB_WGPU_FOG=low

REM Launch ScriptBots with the Bevy renderer in release mode
cargo run -p scriptbots-app --bin scriptbots-app --release --target x86_64-pc-windows-msvc -j %JOBS% --features bevy_render -- --mode bevy --threads 8

endlocal
