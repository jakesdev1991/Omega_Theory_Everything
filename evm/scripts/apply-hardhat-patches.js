// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly

// Hardhat and patch-package are dev dependencies. A production-only install
// contains neither; a normal development install must apply both pinned patches.
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import path from "node:path";

const require = createRequire(import.meta.url);
try {
  require.resolve("hardhat/package.json");
} catch (error) {
  if (error.code === "MODULE_NOT_FOUND") process.exit(0);
  throw error;
}
const patchPackage = require.resolve("patch-package/package.json");
const cli = path.join(path.dirname(patchPackage), "index.js");
const result = spawnSync(process.execPath, [cli, "--error-on-fail"], {
  stdio: "inherit",
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
