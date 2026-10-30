//! Observe the compiler that Cargo selected, before the executable is built.
//!
//! Keep the complete multiline output in OUT_DIR: Cargo's rustc-env protocol
//! carries single-line values. A failed probe leaves an empty observation,
//! which the manifest reader reports as unknown rather than guessing at launch.

use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let target = env::var("TARGET").expect("Cargo provides its resolved compilation target");
    println!("cargo:rustc-env=SCRIPTBOTS_COMPILE_TARGET={target}");

    let compiler = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let identity = match Command::new(compiler).arg("-vV").output() {
        Ok(output) => {
            if !output.stderr.is_empty() {
                println!(
                    "cargo:warning=compiler identity probe stderr: {:?}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            if output.status.success() {
                match String::from_utf8(output.stdout) {
                    Ok(text) => text,
                    Err(error) => {
                        println!("cargo:warning=compiler identity is not UTF-8: {error}");
                        String::new()
                    }
                }
            } else {
                println!(
                    "cargo:warning=compiler identity probe failed: {}",
                    output.status
                );
                String::new()
            }
        }
        Err(error) => {
            println!("cargo:warning=could not observe build-time compiler identity: {error}");
            String::new()
        }
    };
    let directory = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR"));
    fs::write(directory.join("rustc-vv.txt"), identity)
        .expect("write the build-time compiler observation into OUT_DIR");
}
