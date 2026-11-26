import init, { restore_checkpoint } from "./pkg/scriptbots_web.js";
import { runLocalStudy } from "./local-study.js";

self.onmessage = async ({ data }) => {
  try {
    await init();
    const digest = globalThis.crypto?.subtle
      ? new Uint8Array(await crypto.subtle.digest("SHA-256", data.bytes))
      : null;
    const checkpoint = { bytes: data.bytes.byteLength, sha256: digest
      ? Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("") : null };
    const result = runLocalStudy(data, restore_checkpoint, (arm, done, total) => {
      self.postMessage({ progress: { arm, done, total } });
    });
    self.postMessage({ result: { ...result, checkpoint } });
  } catch (error) {
    self.postMessage({ error: String(error?.message ?? error).slice(0, 1024) });
  }
};
