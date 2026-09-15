//! Automated guards and contract verification for visual art direction and reference capture (bd-9pqz.1).
//!
//! Validates:
//! 1. `docs/visual_art_direction.md` exists, conforms to single-source palette authority,
//!    fits the one-page budget, and has intact cross-references.
//! 2. `docs/rendering_reference/bioluminescent_dark_field_v1.png` exists, matches the
//!    exact SHA-256 bound in the provenance manifest, and satisfies PNG format invariants.
//! 3. `docs/rendering_reference/bioluminescent_dark_field_v1.provenance.json` provides
//!    complete metadata covering commit, backend, device, viewport, and human review notes.
//! 4. Negative controls: tampered image bytes fail hash validation, missing fields error out,
//!    and broken documentation links are detected.

use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonicalize repo root")
}

fn doc_path() -> PathBuf {
    repo_root().join("docs/visual_art_direction.md")
}

fn png_path() -> PathBuf {
    repo_root().join("docs/rendering_reference/bioluminescent_dark_field_v1.png")
}

fn provenance_path() -> PathBuf {
    repo_root().join("docs/rendering_reference/bioluminescent_dark_field_v1.provenance.json")
}

fn capture_log_path() -> PathBuf {
    repo_root().join("docs/rendering_reference/bioluminescent_dark_field_v1_capture.log")
}

#[allow(
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::chunks_exact_to_as_chunks
)]
/// Standard FIPS 180-4 SHA-256 digest in pure Rust.
fn compute_sha256(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let k: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, part) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([part[0], part[1], part[2], part[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut h_val = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h_val
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h_val = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(h_val);
    }

    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

#[derive(Debug, serde::Deserialize)]
struct ProvenanceManifest {
    schema_version: String,
    artifact_path: String,
    sha256: String,
    source_commit: String,
    git_dirty: bool,
    renderer: String,
    backend: String,
    device_name: String,
    toolchain: String,
    capture_command: String,
    viewport: ViewportMeta,
    fixture: FixtureMeta,
    human_review: HumanReviewMeta,
}

#[derive(Debug, serde::Deserialize)]
struct ViewportMeta {
    width: u32,
    height: u32,
}

#[derive(Debug, serde::Deserialize)]
struct FixtureMeta {
    seed: u64,
    tick: u64,
    world_width: f32,
    world_height: f32,
    cell_size: u32,
    terrain_grid: [u32; 2],
    agent_count: usize,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, serde::Deserialize)]
struct HumanReviewMeta {
    figure_ground_verified: bool,
    agent_visibility_verified: bool,
    water_terrain_distinction_verified: bool,
    food_event_vocabulary_verified: bool,
    hud_exclusion_verified: bool,
    notes: String,
}

fn parse_markdown_links(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = content;
    while let Some(start_bracket) = rest.find("](") {
        let after_bracket = &rest[start_bracket + 2..];
        if let Some(end_bracket) = after_bracket.find(')') {
            let target = after_bracket[..end_bracket].trim();
            if !target.starts_with("http://")
                && !target.starts_with("https://")
                && !target.is_empty()
            {
                links.push(target.to_string());
            }
            rest = &after_bracket[end_bracket + 1..];
        } else {
            break;
        }
    }
    links
}

#[test]
fn art_direction_note_conforms_to_specification_and_budget() {
    let path = doc_path();
    assert!(path.is_file(), "docs/visual_art_direction.md must exist");

    let content = fs::read_to_string(&path).expect("read visual_art_direction.md");
    let lines: Vec<&str> = content.lines().collect();

    // 1. One-page scope bound (< 200 lines).
    assert!(
        lines.len() <= 200,
        "visual_art_direction.md must fit one-page scope, got {} lines",
        lines.len()
    );
    assert!(
        lines.len() >= 80,
        "visual_art_direction.md is unexpectedly truncated, got {} lines",
        lines.len()
    );

    // 2. Required thematic sections.
    let required_sections = [
        "Single Numeric Authority & Core Rule",
        "Figure/Ground Rationale: Dark-Field Microscopy",
        "Visual Vocabulary",
        "Accessibility & Color Spaces",
        "Renderer-Consumption Boundaries & Open Owners",
        "Canonical Reference Capture & Human Review",
    ];
    for section in required_sections {
        assert!(
            content.contains(section),
            "missing required section in art direction note: '{section}'"
        );
    }

    // 3. Single source of truth enforcement: references core visual authority.
    assert!(
        content.contains("BIOLUMINESCENT_DARK_FIELD_V1"),
        "art direction note must reference BIOLUMINESCENT_DARK_FIELD_V1 as sole authority"
    );
    assert!(
        content.contains("apply_accessibility_palette"),
        "art direction note must document the final-stage accessibility boundary"
    );

    // 4. Verifies absence of competing numeric palette declarations (e.g. hex codes or raw float arrays).
    let prohibited_palette_tokens = ["const PALETTE", "let PALETTE", "#FF00", "#00FF", "0x00FF"];
    for token in prohibited_palette_tokens {
        assert!(
            !content.contains(token),
            "art direction note contains prohibited competing palette token: '{token}'"
        );
    }

    // 5. Link integrity check: all relative links resolve to existing files.
    let doc_parent = path.parent().expect("doc parent directory");
    let extracted_links = parse_markdown_links(&content);
    assert!(
        !extracted_links.is_empty(),
        "art direction note must contain verified cross-links"
    );
    for link in extracted_links {
        let clean_link = link.split('#').next().unwrap_or(&link);
        if clean_link.is_empty() {
            continue;
        }
        let resolved = doc_parent.join(clean_link);
        assert!(
            resolved.exists(),
            "broken relative link in visual_art_direction.md: '{clean_link}' -> '{}'",
            resolved.display()
        );
    }
}

#[test]
fn canonical_reference_image_and_provenance_manifest_match_exactly() {
    let image_path = png_path();
    assert!(
        image_path.is_file(),
        "canonical reference image docs/rendering_reference/bioluminescent_dark_field_v1.png must exist"
    );

    let png_bytes = fs::read(&image_path).expect("read canonical png bytes");
    assert!(
        png_bytes.len() > 10_000,
        "canonical PNG size is suspiciously small: {} bytes",
        png_bytes.len()
    );

    // Assert standard 8-byte PNG signature: \x89PNG\r\n\x1a\n
    assert_eq!(
        &png_bytes[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
        "file must be a valid PNG format"
    );

    // Assert IHDR chunk width (1600) and height (900)
    let width = u32::from_be_bytes([png_bytes[16], png_bytes[17], png_bytes[18], png_bytes[19]]);
    let height = u32::from_be_bytes([png_bytes[20], png_bytes[21], png_bytes[22], png_bytes[23]]);
    assert_eq!(
        (width, height),
        (1600, 900),
        "canonical PNG must have 1600x900 viewport"
    );

    // Compute SHA-256
    let calculated_hash = compute_sha256(&png_bytes);

    // Read and parse provenance manifest
    let prov_path = provenance_path();
    assert!(
        prov_path.is_file(),
        "canonical provenance manifest docs/rendering_reference/bioluminescent_dark_field_v1.provenance.json must exist"
    );
    let prov_raw = fs::read_to_string(&prov_path).expect("read provenance json");
    let manifest: ProvenanceManifest =
        serde_json::from_str(&prov_raw).expect("parse provenance manifest");

    // Exact SHA-256 agreement
    assert_eq!(
        calculated_hash, manifest.sha256,
        "image SHA-256 must match provenance manifest sha256"
    );

    // Verify manifest metadata requirements
    assert_eq!(manifest.schema_version, "1.0.0");
    assert_eq!(
        manifest.artifact_path,
        "docs/rendering_reference/bioluminescent_dark_field_v1.png"
    );
    assert_ne!(manifest.source_commit, "");
    assert_eq!(manifest.renderer, "scriptbots-world-gfx::WorldRenderer");
    assert_ne!(manifest.backend, "");
    assert_ne!(manifest.device_name, "");
    assert_eq!(manifest.viewport.width, 1600);
    assert_eq!(manifest.viewport.height, 900);
    assert!(!manifest.git_dirty);
    assert_eq!(manifest.toolchain, "rustc nightly (2024 edition)");
    assert_ne!(manifest.capture_command, "");
    assert_eq!(manifest.fixture.seed, 424_242);
    assert_eq!(manifest.fixture.tick, 120);
    assert_eq!(manifest.fixture.world_width, 1600.0);
    assert_eq!(manifest.fixture.world_height, 900.0);
    assert_eq!(manifest.fixture.cell_size, 50);
    assert_eq!(manifest.fixture.terrain_grid, [32, 18]);
    assert_eq!(manifest.fixture.agent_count, 6);

    // Verify human review assertions
    assert!(manifest.human_review.figure_ground_verified);
    assert!(manifest.human_review.agent_visibility_verified);
    assert!(manifest.human_review.water_terrain_distinction_verified);
    assert!(manifest.human_review.food_event_vocabulary_verified);
    assert!(manifest.human_review.hud_exclusion_verified);
    assert_ne!(manifest.human_review.notes, "");
}

#[test]
fn capture_log_is_retained_and_agrees_with_manifest_and_image() {
    let log_file = capture_log_path();
    assert!(log_file.is_file(), "capture log must exist");

    let log_content = fs::read_to_string(&log_file).expect("read capture log");
    assert!(log_content.contains("Renderer: scriptbots-world-gfx::WorldRenderer"));
    assert!(log_content.contains("Viewport: 1600x900"));
    assert!(log_content.contains("Status: SUCCESS"));

    // Verify SHA-256 declared in log matches image
    let png_bytes = fs::read(png_path()).expect("read canonical png bytes");
    let calculated_hash = compute_sha256(&png_bytes);
    assert!(
        log_content.contains(&calculated_hash),
        "capture log must record the exact computed SHA-256 of the canonical image"
    );
}

#[test]
fn born_red_negative_control_detects_tampered_image_bytes() {
    let mut tampered = fs::read(png_path()).expect("read canonical png bytes");
    // Tamper with a single byte in the middle of image payload
    let mid = tampered.len() / 2;
    tampered[mid] ^= 0xFF;

    let tampered_hash = compute_sha256(&tampered);
    let prov_raw = fs::read_to_string(provenance_path()).expect("read provenance json");
    let manifest: ProvenanceManifest =
        serde_json::from_str(&prov_raw).expect("parse provenance manifest");

    assert_ne!(
        tampered_hash, manifest.sha256,
        "tampered image bytes must produce a mismatched SHA-256"
    );
}

#[test]
fn born_red_negative_control_detects_missing_required_provenance_field() {
    // Provenance JSON missing the `sha256` field
    let malformed_json = r#"{
      "schema_version": "1.0.0",
      "artifact_path": "docs/rendering_reference/bioluminescent_dark_field_v1.png",
      "source_commit": "abcdef",
      "git_dirty": false,
      "renderer": "scriptbots-world-gfx::WorldRenderer",
      "backend": "Vulkan",
      "device_name": "llvmpipe",
      "toolchain": "rustc",
      "capture_command": "cargo test",
      "viewport": { "width": 1600, "height": 900 },
      "fixture": { "seed": 1, "tick": 1, "world_width": 100.0, "world_height": 100.0, "cell_size": 50, "terrain_grid": [2, 2], "agent_count": 1 },
      "human_review": {
        "figure_ground_verified": true,
        "agent_visibility_verified": true,
        "water_terrain_distinction_verified": true,
        "food_event_vocabulary_verified": true,
        "hud_exclusion_verified": true,
        "notes": "test"
      }
    }"#;

    let result: Result<ProvenanceManifest, _> = serde_json::from_str(malformed_json);
    assert!(
        result.is_err(),
        "manifest without required sha256 field must fail validation"
    );
}

#[test]
fn born_red_negative_control_detects_broken_document_link() {
    let doc_parent = doc_path().parent().expect("doc parent").to_path_buf();
    let fake_content = "[Broken Link](rendering_reference/nonexistent_artifact_file.xyz)";
    let links = parse_markdown_links(fake_content);
    assert_eq!(links.len(), 1);
    let resolved = doc_parent.join(&links[0]);
    assert!(
        !resolved.exists(),
        "broken link must correctly fail to resolve"
    );
}
