//! Record the inputs of this browser build without embedding source files in WASM.
//!
//! FNV-1a fingerprints detect ordinary input drift; they are not cryptographic
//! attestations and matching inputs do not prove cross-target replay equivalence.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

fn feed(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(PRIME);
    }
}

fn fingerprint(bytes: &[u8]) -> String {
    let mut hash = OFFSET;
    feed(&mut hash, bytes);
    format!("fnv1a64:{hash:016x}")
}

fn read_fingerprint(path: &Path) -> Option<String> {
    println!("cargo:rerun-if-changed={}", path.display());
    fs::read(path).map(|bytes| fingerprint(&bytes)).ok()
}

fn rust_sources(directory: &Path, paths: &mut Vec<PathBuf>) -> io::Result<()> {
    println!("cargo:rerun-if-changed={}", directory.display());
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() {
            rust_sources(&entry.path(), paths)?;
        } else if kind.is_file() && entry.path().extension().is_some_and(|ext| ext == "rs") {
            paths.push(entry.path());
        } else if kind.is_symlink() {
            return Err(io::Error::other(
                "source inventory contains a symbolic link",
            ));
        }
    }
    Ok(())
}

fn source_inventory(root: &Path) -> io::Result<(String, Vec<(String, String)>)> {
    let mut paths = vec![root.join("Cargo.toml")];
    for name in [
        "scriptbots-core",
        "scriptbots-index",
        "scriptbots-brain",
        "scriptbots-runtime",
        "scriptbots-web",
    ] {
        let directory = root.join("crates").join(name);
        paths.push(directory.join("Cargo.toml"));
        let build = directory.join("build.rs");
        println!("cargo:rerun-if-changed={}", build.display());
        if build.try_exists()? {
            paths.push(build);
        }
        rust_sources(&directory.join("src"), &mut paths)?;
    }
    let cargo_config = root.join(".cargo/config.toml");
    println!("cargo:rerun-if-changed={}", cargo_config.display());
    if cargo_config.try_exists()? {
        paths.push(cargo_config);
    }
    paths.sort();
    let mut hash = OFFSET;
    feed(&mut hash, b"scriptbots.browser-source-inventory.v1\0");
    let mut inventory = Vec::with_capacity(paths.len());
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path
            .strip_prefix(root)
            .map_err(io::Error::other)?
            .to_str()
            .ok_or_else(|| io::Error::other("source path is not UTF-8"))?
            .replace('\\', "/");
        let bytes = fs::read(path)?;
        feed(&mut hash, &(name.len() as u64).to_le_bytes());
        feed(&mut hash, name.as_bytes());
        feed(&mut hash, &(bytes.len() as u64).to_le_bytes());
        feed(&mut hash, &bytes);
        inventory.push((name, fingerprint(&bytes)));
    }
    Ok((format!("fnv1a64:{hash:016x}"), inventory))
}

fn main() -> io::Result<()> {
    let directory = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| io::Error::other("Cargo did not provide CARGO_MANIFEST_DIR"))?,
    );
    let root = directory
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| io::Error::other("browser crate has no workspace parent"))?;
    let inventory = source_inventory(root).ok();
    let source = inventory.as_ref().map(|(value, _)| value);
    let paths = inventory
        .as_ref()
        .map(|(_, values)| values.as_slice())
        .unwrap_or_default();
    let lockfile = read_fingerprint(&root.join("Cargo.lock"));
    let toolchain = read_fingerprint(&root.join("rust-toolchain.toml"));
    let compiler = env::var_os("RUSTC")
        .and_then(|compiler| {
            Command::new(compiler)
                .args(["--version", "--verbose"])
                .output()
                .ok()
        })
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok());
    let mut generated = format!(
        "const SOURCE: Option<&str> = {source:?};\n\
         const LOCKFILE: Option<&str> = {lockfile:?};\n\
         const TOOLCHAIN: Option<&str> = {toolchain:?};\n\
         const COMPILER: Option<&str> = {compiler:?};\n\
         const SOURCE_INVENTORY: &[(&str, &str)] = &{paths:?};\n"
    );
    for (constant, variable) in [
        ("TARGET", "TARGET"),
        ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS"),
        ("PROFILE", "PROFILE"),
        ("OPT_LEVEL", "OPT_LEVEL"),
        ("DEBUG", "DEBUG"),
        ("TARGET_FEATURES", "CARGO_CFG_TARGET_FEATURE"),
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
        let value = env::var(variable).ok();
        let _ = writeln!(generated, "const {constant}: Option<&str> = {value:?};");
    }
    println!("cargo:rerun-if-env-changed=RUSTC");
    let output = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("Cargo did not provide OUT_DIR"))?,
    );
    fs::write(output.join("browser_build.rs"), generated)
}
