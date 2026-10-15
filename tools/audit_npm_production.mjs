#!/usr/bin/env node
// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/**
 * Fail on high/critical advisories reachable from production dependency trees.
 *
 * This deliberately audits committed package locks (`--package-lock-only`) so
 * CI can check every JavaScript workspace without installing each workspace a
 * second time. Development-only dependency findings are not silently called
 * safe: they are tracked separately in docs/SOFTWARE_SUPPLY_CHAIN.md, while
 * this command protects the dependency surface shipped to users/operators.
 *
 * Run from the repository root:
 *   node tools/audit_npm_production.mjs
 */

import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

const ROOT = resolve(new URL("..", import.meta.url).pathname);
const WORKSPACES = ["amity", "desktop", "evm", "mobile-node", "nostr-client", "solana", "web"];
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
let failed = false;

for (const workspace of WORKSPACES) {
  const cwd = resolve(ROOT, workspace);
  const lockfile = resolve(cwd, "package-lock.json");
  if (!existsSync(lockfile)) {
    console.error(`Missing lockfile: ${lockfile}`);
    failed = true;
    continue;
  }

  console.log(`\n=== ${workspace}: production dependency audit ===`);
  const result = spawnSync(
    npmCommand,
    ["audit", "--omit=dev", "--package-lock-only", "--audit-level=high"],
    { cwd, stdio: "inherit" },
  );
  if (result.error || result.status !== 0) {
    console.error(`Production dependency audit failed for ${workspace}.`);
    failed = true;
  }
}

if (failed) {
  console.error("\nProduction dependency audit found a workspace failure.");
  process.exitCode = 1;
} else {
  console.log("\nAll committed npm production dependency trees passed the high/critical audit threshold.");
}
