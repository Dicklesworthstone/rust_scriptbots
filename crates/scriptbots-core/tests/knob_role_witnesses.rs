//! Behavioural witnesses for the public configuration knob surface (bd-dorx).
//!
//! `ControlHandle::list_knobs` publishes every serialized `ScriptBotsConfig` leaf as a REST/MCP
//! knob, and `apply_patch` accepts every declared path. A field added to the config therefore
//! becomes a public scientific control with no proof that any model transition reads it. Hashing
//! the config cannot supply that proof: a dead field still changes the config lane by
//! construction, which is exactly the ghost-control failure mode recorded on bd-yw1j.
//!
//! This file owns the enumeration half of the fix. It derives the knob surface the same way the
//! control plane does -- serialize the config, then treat every non-object node as a leaf -- so
//! the list here cannot drift from the list callers actually see, and cannot be invalidated by a
//! `#[serde(rename)]` that source-level transcription would miss.

use scriptbots_core::ScriptBotsConfig;
use serde_json::Value;

/// Flatten a serialized config into dotted leaf paths.
///
/// Mirrors `flatten_value` in `scriptbots-app`: recurse into objects only, and treat every other
/// node -- including arrays and nulls -- as a single leaf. Anything else would report a surface
/// the control plane does not actually expose.
fn flatten_paths(prefix: &mut String, value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            let base = prefix.len();
            for (key, nested) in map {
                if base != 0 {
                    prefix.push('.');
                }
                prefix.push_str(key);
                flatten_paths(prefix, nested, out);
                prefix.truncate(base);
            }
        }
        _ => out.push(prefix.clone()),
    }
}

/// Every knob path the control plane publishes for a default config.
fn default_knob_paths() -> Vec<String> {
    let value = serde_json::to_value(ScriptBotsConfig::default())
        .expect("the public config must serialize; the control plane depends on it");
    let mut paths = Vec::new();
    flatten_paths(&mut String::new(), &value, &mut paths);
    paths.sort();
    paths
}

/// Enumerate the knob surface so the classification registry can be built against fact.
///
/// Deliberately not an assertion on a hardcoded count: the point is to print the authoritative
/// list. A count assertion here would be the same class of unfounded claim the bead is about.
#[test]
fn bd_dorx_enumerate_the_published_knob_surface() {
    let paths = default_knob_paths();
    println!("PUBLISHED_KNOB_COUNT={}", paths.len());
    for path in &paths {
        println!("KNOB\t{path}");
    }
    assert!(
        !paths.is_empty(),
        "the config must publish at least one knob or list_knobs is meaningless"
    );
}

/// The flattener must agree with the control plane on what counts as one leaf.
///
/// Nested objects expand; arrays and scalars do not. If this ever changes, the registry's notion
/// of "every knob" silently stops matching the surface callers can patch.
#[test]
fn bd_dorx_flattening_expands_objects_and_stops_at_every_other_node() {
    let value = serde_json::json!({
        "scalar": 1,
        "nested": { "inner": true, "deeper": { "leaf": "x" } },
        "array": [1, 2, 3],
        "null": null,
    });
    let mut paths = Vec::new();
    flatten_paths(&mut String::new(), &value, &mut paths);
    paths.sort();

    assert_eq!(
        paths,
        vec![
            "array".to_owned(),
            "nested.deeper.leaf".to_owned(),
            "nested.inner".to_owned(),
            "null".to_owned(),
            "scalar".to_owned(),
        ],
        "an array is one knob, not one knob per element, and a null is still a knob"
    );
}
