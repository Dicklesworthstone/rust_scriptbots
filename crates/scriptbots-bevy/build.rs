//! Build-time provenance injection for capture artifacts (bd-2z0.14.3.10 item 1).
//!
//! Before this, `CaptureProvenance` could say which WORLD a frame depicted but
//! not which SOURCE TREE rendered it: `SCRIPTBOTS_RUSTC_VERSION` and
//! `SCRIPTBOTS_TARGET_TRIPLE` were read with `option_env!` and nothing in the
//! workspace ever set them, so they fell back to `CARGO_PKG_RUST_VERSION` and a
//! hand-written `cfg!` ladder. A golden mismatch could therefore not distinguish
//! "the renderer changed" from "a different build produced this".
//!
//! Everything here is best-effort by construction. A build script that can fail
//! the build over provenance would trade a working renderer for a nicer
//! manifest, so every step degrades to an absent variable and `option_env!`
//! keeps its existing fallback.
//!
//! No dependencies: this crate has no `[build-dependencies]` and deliberately
//! gains none, so the digest below is a small inline FNV-1a-64 rather than a
//! hashing crate.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// FNV-1a 64, the non-cryptographic fingerprint this workspace already uses for
/// frame and buffer identity. Chosen for consistency and zero dependencies; it
/// answers "is this the same file" and is not a security boundary.
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Run a git command in the workspace root, returning trimmed stdout on success.
///
/// Returns `None` when git is missing, this is not a repository (release
/// tarballs, vendored builds), or git reports failure — all normal states that
/// must not break the build.
fn git(root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

/// Git resolves linked-worktree metadata and shared refs to different places.
fn watch_git_metadata(root: &Path, name: &str, watch_parent_if_missing: bool) {
    let Some(raw) = git(root, &["rev-parse", "--git-path", name]) else {
        return;
    };
    let path = root.join(raw);
    if !path.exists() && !watch_parent_if_missing {
        return;
    }
    // A packed branch may have no loose ref yet. Watching its existing parent
    // detects creation without Cargo treating a missing file as always changed.
    let mut watched = path.as_path();
    while !watched.exists() {
        let Some(parent) = watched.parent() else {
            return;
        };
        watched = parent;
    }
    println!("cargo:rerun-if-changed={}", watched.display());
}

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo always sets CARGO_MANIFEST_DIR"),
    );
    // crates/scriptbots-bevy -> workspace root.
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map_or_else(|| manifest_dir.clone(), PathBuf::from);

    // The compile target as cargo actually resolved it, replacing the `cfg!`
    // ladder's guess. That ladder cannot express musl, windows-gnu, or any
    // cross-compile, and silently reported a plausible-looking wrong triple.
    if let Ok(target) = std::env::var("TARGET") {
        println!("cargo:rustc-env=SCRIPTBOTS_TARGET_TRIPLE={target}");
    }

    // Toolchain identity: the first line of `rustc -vV`, matching what the
    // provenance field documents itself as carrying.
    let rustc_bin = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    if let Some(rustc) = Command::new(rustc_bin)
        .arg("-vV")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|text| text.lines().next().map(str::to_string))
    {
        println!("cargo:rustc-env=SCRIPTBOTS_RUSTC_VERSION={rustc}");
    }

    // Source identity. `-dirty` is appended when the working tree carried
    // uncommitted changes at build-script run time; see the field docs in
    // capture.rs for why that is a warning rather than a cleanliness proof.
    if let Some(commit) = git(&workspace_root, &["rev-parse", "HEAD"]) {
        let dirty = git(&workspace_root, &["status", "--porcelain"]).is_some();
        let suffix = if dirty { "-dirty" } else { "" };
        println!("cargo:rustc-env=SCRIPTBOTS_SOURCE_COMMIT={commit}{suffix}");
    }

    // Dependency-graph identity. The lock file pins every version actually
    // linked, so two artifacts with equal lock digests were built from the same
    // resolved graph even if their tooling differed.
    let lock_path = workspace_root.join("Cargo.lock");
    if let Ok(bytes) = std::fs::read(&lock_path) {
        println!(
            "cargo:rustc-env=SCRIPTBOTS_LOCK_DIGEST={:016x}",
            fnv1a64(&bytes)
        );
        println!("cargo:rerun-if-changed={}", lock_path.display());
    }

    // A worktree has a .git FILE; an ordinary checkout's HEAD can remain the
    // same symbolic ref while its branch advances. Observe Git's actual paths.
    // Source edits alone still do not make the dirty marker authoritative.
    watch_git_metadata(&workspace_root, "HEAD", false);
    if let Some(reference) = git(&workspace_root, &["symbolic-ref", "-q", "HEAD"]) {
        watch_git_metadata(&workspace_root, &reference, true);
    }
    watch_git_metadata(&workspace_root, "packed-refs", false);
    println!("cargo:rerun-if-changed=build.rs");
}
