import init, { restore_checkpoint } from "./pkg/scriptbots_web.js";
import { runLocalStudy, validateLocalStudyJob } from "./local-study.js";

self.onmessage = async ({ data }) => {
  try {
    const job = validateLocalStudyJob(data);
    await init();
    const result = runLocalStudy(job, restore_checkpoint, (arm, done, total) => {
      self.postMessage({ progress: { arm, done, total } });
    });
    self.postMessage({ result });
  } catch (error) {
    self.postMessage({ error: String(error?.message ?? error).slice(0, 1024) });
  }
};
