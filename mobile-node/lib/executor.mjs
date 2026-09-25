/**
 * Sandboxed execution for allowlisted algorithms.
 *
 * Hard rules:
 *   - argv arrays only; never a shell string (no interpolation, no `sh -c`)
 *   - scrubbed environment (no secrets, no PATH surprises beyond a fixed PATH)
 *   - parameters arrive on stdin as JSON, never as argv
 *   - wall-clock timeout with SIGKILL escalation
 *   - bounded stdout/stderr capture
 */

import { spawn } from "node:child_process";
import { resolve } from "node:path";

const SAFE_ENV = {
  PATH: "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
  LANG: "C.UTF-8",
  TERM: "dumb",
};

export function runAlgorithm({ algorithm, sandboxArgv = [], params, cwd, env = {} }) {
  return new Promise((resolvePromise) => {
    const argv = [...sandboxArgv, ...algorithm.argv];
    if (argv.length === 0) {
      resolvePromise({ ok: false, error: "algorithm has no argv" });
      return;
    }

    const timeoutMs = Number(algorithm.timeoutMs ?? 30000);
    const maxOutputBytes = Number(algorithm.maxOutputBytes ?? 262144);

    let child;
    try {
      child = spawn(argv[0], argv.slice(1), {
        cwd: cwd ?? process.cwd(),
        env: { ...SAFE_ENV, ...env },
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (error) {
      resolvePromise({ ok: false, error: error instanceof Error ? error.message : String(error) });
      return;
    }

    let stdout = "";
    let stderr = "";
    let truncated = false;
    let settled = false;

    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolvePromise(result);
    };

    const timer = setTimeout(() => {
      truncated = true;
      child.kill("SIGKILL");
      finish({ ok: false, error: `timed out after ${timeoutMs} ms`, stdout, stderr });
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      if (stdout.length < maxOutputBytes) stdout += chunk.toString("utf8");
      else truncated = true;
    });
    child.stderr.on("data", (chunk) => {
      if (stderr.length < maxOutputBytes) stderr += chunk.toString("utf8");
      else truncated = true;
    });

    child.on("error", (error) => {
      finish({ ok: false, error: error.message, stdout, stderr });
    });

    child.on("close", (code) => {
      if (code === 0) {
        finish({ ok: true, stdout, stderr, truncated });
      } else {
        finish({ ok: false, error: `exit code ${code}`, stdout, stderr });
      }
    });

    child.stdin.on("error", () => {
      /* algorithm may ignore stdin */
    });
    child.stdin.end(JSON.stringify(params ?? {}));
  });
}

export function resolveAlgorithmCwd(baseDir) {
  return resolve(baseDir);
}
