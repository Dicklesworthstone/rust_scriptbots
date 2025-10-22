# Phase 1 Security & Hosting Notes

_Date: 2025-10-22 (UTC)_

## Cross-Origin Isolation (COOP/COEP)
- Required for enabling WebAssembly threads via `SharedArrayBuffer` on Chrome, Edge, Firefox, and Safari Tech Preview.citeturn3search0turn3search3turn3search6
- Headers to set:
  - `Cross-Origin-Opener-Policy: same-origin`
  - `Cross-Origin-Embedder-Policy: require-corp`
- Assets (including wasm, JS, textures) must be served with `Cross-Origin-Resource-Policy: same-origin` or appropriate `corp` headers.
- Evaluate service worker impact; ensure worker scripts also respect COEP. `coi-serviceworker` helps apply headers during local development.citeturn3search5

## Content Security Policy (CSP)
- Baseline recommended policy:
  ```
  Content-Security-Policy: default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline';
  ```
- `wasm-unsafe-eval` currently required for WebAssembly instantiation in some browsers; monitor for upcoming CSP Level 3 changes.
- Lock down `worker-src` and `child-src` to `'self'` when thread pools are introduced.

## Hosting Considerations
- Serve over HTTPS with HTTP/2 or HTTP/3 to minimize WASM download latency.
- Ensure Brotli compression for `.wasm`, `.js`, `.json`.
- CDN must support custom headers (COOP/COEP). Cloudflare Pages and Netlify support this via `_headers` file; GitHub Pages requires reverse proxy or Cloudflare fronting.

## SharedArrayBuffer Fallback Plan
- When COOP/COEP cannot be guaranteed (e.g., third-party embedding), fall back to single-thread simulation and surface a UI warning.
- Provide feature detection snippet:
  ```js
  const sabSupport = typeof SharedArrayBuffer !== 'undefined' && crossOriginIsolated;
  ```
- Log telemetry event when falling back to capture prevalence.

## Secure Storage & Export
- Downloads triggered via browser should use object URLs; revoke after use.
- Consider encrypting analytics exports if future remote storage is added.

## Outstanding Questions
- Do we need COEP exemptions for analytics CDNs or telemetry endpoints?
- How to integrate Subresource Integrity (SRI) when wasm filename includes hash?
- Should we bundle service worker for offline mode; if so, ensure cache poisoning mitigations.
