@echo off
setlocal enableextensions enabledelayedexpansion

REM Launch ScriptBots with GPU GUI in release mode with tuned threads
cargo run -p scriptbots-app --bin scriptbots-app --release --features gui -- --mode gui --threads 8

endlocal
