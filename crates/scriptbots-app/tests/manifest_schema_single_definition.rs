//! The manifest schema tags must have exactly one definition (`bd-k0wj`).
//!
//! `scriptbots-app` writes run manifests and `scriptbots-storage` validates them, so the
//! schema tag is a wire contract between them. Each crate used to declare its own `const`
//! for it. The copies agreed until `ff937dec6` bumped the bootstrap tag to
//! `scriptbots.run-manifest.v3.6` in the app alone, leaving storage validating v3.5 — the
//! writer emitted manifests the reader refused, and nothing failed at the seam itself. It
//! surfaced only indirectly, as five unrelated-looking failures in storage's manifest
//! tests.
//!
//! Storage now owns both tags and the app re-exports them, so there is one definition. This
//! test is the cheap guard that keeps it that way: it compares the two crates' public
//! constants, which is a tautology while the re-export stands and a failure the moment
//! anyone reintroduces a separate `const` in the app whose value drifts.
//!
//! It cannot catch a duplicate that happens to hold the same value today. That is the
//! accepted limit: divergence is the failure mode that actually bit, and it is what this
//! detects.

/// The two crates must agree on the continuation-complete manifest tag.
#[test]
fn app_and_storage_share_one_manifest_schema_definition() {
    assert_eq!(
        scriptbots_app::RUN_MANIFEST_V3_SCHEMA,
        scriptbots_storage::RUN_MANIFEST_V3_SCHEMA,
        "the app and storage disagree on the V3 manifest schema tag; they must share one \
         definition rather than each declaring a const (bd-k0wj)"
    );
}

/// The bootstrap tag is the one that actually drifted, so it gets its own assertion.
#[test]
fn app_and_storage_share_one_bootstrap_schema_definition() {
    assert_eq!(
        scriptbots_app::RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA,
        scriptbots_storage::RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA,
        "the app and storage disagree on the bootstrap manifest schema tag — this is the \
         exact drift of bd-k0wj, where the app emitted a tag storage refused (bd-k0wj)"
    );
}

/// A manifest the app would write must be one storage is willing to accept.
///
/// The equality checks above prove the constants match. This proves the value they hold is
/// one storage's validator actually admits, so bumping both in lockstep to a tag storage
/// rejects would still fail here rather than at a distant call site.
#[test]
fn the_shared_bootstrap_tag_is_one_storage_accepts() {
    let bootstrap = scriptbots_app::RUN_MANIFEST_V3_BOOTSTRAP_SCHEMA;
    assert!(
        !scriptbots_storage::manifest_schema_is_superseded(bootstrap),
        "the shared bootstrap tag {bootstrap:?} is one storage treats as superseded, so the \
         app would emit manifests storage refuses (bd-k0wj)"
    );
}
