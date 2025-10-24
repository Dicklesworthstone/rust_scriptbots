@echo off
setlocal enableextensions enabledelayedexpansion

REM Launch ScriptBots in terminal mode in release
cargo run -p scriptbots-app --bin scriptbots-app --release -- --mode terminal

endlocal
