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

REM Launch ScriptBots in terminal mode in release
set WGPU_BACKEND=
set SB_WGPU_PRESENT_MODE=
cargo run -p scriptbots-app --bin scriptbots-app --release --target x86_64-pc-windows-msvc -j %JOBS% -- --mode terminal

endlocal
