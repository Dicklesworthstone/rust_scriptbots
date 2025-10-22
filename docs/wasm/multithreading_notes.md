# Multithreading & SharedArrayBuffer Requirements (Phase 1.6)

_Created: 2025-10-22 (UTC)_

## Executive Summary
- Enabling Rayon (or any multithreaded wasm) in browsers requires **cross-origin isolation** (COOP/COEP) so that `SharedArrayBuffer` is available. Chrome and Edge enforce this since Chrome 92; Firefox and Safari Tech Preview follow the same pattern.citeturn3search0turn3search3turn3search6
- `wasm-bindgen-rayon` 1.2+ provides glue code to spin up a web worker thread pool once cross-origin isolation is active.citeturn3search1turn3search4
- If cross-origin isolation cannot be guaranteed (e.g., embedded contexts), we must fall back to single-threaded execution and surface a UX warning.

## Header Requirements
Serve every HTML response with:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```
- COEP can also use the `credentialless` variant to relax CORS for third-party assets; evaluate implications for analytics/runtime CDNs.citeturn3search3
- Verify `self.crossOriginIsolated === true` at runtime; log failures for telemetry.citeturn3search6

## Initialization Flow
1. Main thread loads wasm bundle.
2. Call `initThreadPool` from `wasm-bindgen-rayon` with desired worker count (e.g., `navigator.hardwareConcurrency`).
3. Await thread pool initialization before invoking simulation entry points.

## Deployment Considerations
- Hosting environments without configurable headers (e.g., bare GitHub Pages) require an opt-in service worker shim such as `coi-serviceworker`.citeturn3search5
- Document COOP/COEP configuration for each supported CDN (Cloudflare Pages, Netlify, custom Nginx).
- Record fallback behavior (single-thread mode) metrics to assess prevalence.

## Open Questions
- Do we need to support Firefox/Safari versions without SharedArrayBuffer? If so, should we feature-detect and dynamically degrade to sequential iteration?
- Should we bundle a `coi-serviceworker.js` helper for development servers?

_Update this note once we prototype thread pool startup in the wasm harness._***
