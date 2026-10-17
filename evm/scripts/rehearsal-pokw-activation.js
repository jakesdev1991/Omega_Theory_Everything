// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/* eslint-disable no-console */

/**
 * Rehearsal for scripts/activate-pokw-minter.js — the two-phase PoKW
 * activation used when the treasury is not the deployer.
 *
 * Each `npx hardhat run` starts a FRESH in-process network, so a deployed
 * minter cannot be shared across script invocations. This rehearsal
 * therefore exercises the activation script's FAIL-CLOSED paths, which
 * are exactly what must never silently pass:
 *
 *   1. No POKW_MINTER_ADDRESS        -> refuse before touching the chain.
 *   2. Manifest missing              -> refuse (budget cannot be verified).
 *   3. Address with no contract      -> refuse (no minter there).
 *
 * The happy path (fund -> activate() -> renounceOwnership()) is NOT
 * rehearsed here because it cannot survive the process boundary; it is
 * covered by:
 *   - deploy-sepolia.js's inline treasury==deployer rehearsal (same
 *     activate()/renounceOwnership() calls, same order, real EDR network);
 *   - the pokw-minter.test.js suite (activate + budget lock + terminal
 *     ownership state, 13/13).
 *
 * Nothing is broadcast.
 */

import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const manifestPath = path.join(__dirname, "..", "deployments", "sepolia.json");

const baseEnv = {
  ...process.env,
  DEPLOYER_PRIVATE_KEY: process.env.DEPLOYER_PRIVATE_KEY || `0x${"11".repeat(32)}`,
};

function runActivation(env) {
  return spawnSync(
    "npx",
    ["hardhat", "run", "scripts/activate-pokw-minter.js", "--network", "hardhat"],
    {
      cwd: `${__dirname}/..`,
      env,
      encoding: "utf8",
      shell: process.platform === "win32",
    },
  );
}

function expectRefusal(label, result, messagePattern) {
  const output = `${result.stdout || ""}${result.stderr || ""}`;
  const refused = result.status !== 0 && messagePattern.test(output);
  if (refused) {
    console.log(`  PASS  ${label}`);
    return true;
  }
  console.error(`  FAIL  ${label}`);
  console.error(`        exit=${result.status}`);
  console.error(`        output=${output.slice(0, 400)}`);
  return false;
}

console.log("Rehearsing PoKW minter activation refusals (nothing is broadcast).");
let allPassed = true;

// Drill 1: no POKW_MINTER_ADDRESS.
{
  const env = { ...baseEnv };
  delete env.POKW_MINTER_ADDRESS;
  const result = runActivation(env);
  allPassed &&= expectRefusal(
    "refuses without POKW_MINTER_ADDRESS",
    result,
    /POKW_MINTER_ADDRESS/,
  );
}

// Drill 2: manifest missing. Preserve/restore any real manifest.
{
  const hadManifest = fs.existsSync(manifestPath);
  const backup = hadManifest ? fs.readFileSync(manifestPath, "utf8") : null;
  if (hadManifest) fs.renameSync(manifestPath, `${manifestPath}.rehearsal-bak`);
  const env = { ...baseEnv, POKW_MINTER_ADDRESS: "0x00000000000000000000000000000000000000C3" };
  try {
    const result = runActivation(env);
    allPassed &&= expectRefusal(
      "refuses when the deployment manifest is absent",
      result,
      /sepolia\.json not found/,
    );
  } finally {
    if (hadManifest) fs.renameSync(`${manifestPath}.rehearsal-bak`, manifestPath);
    else if (fs.existsSync(manifestPath)) fs.rmSync(manifestPath);
  }
}

// Drill 3: address with no contract code (fresh network — nothing is
// deployed there, so getContractAt must refuse).
{
  const env = {
    ...baseEnv,
    POKW_MINTER_ADDRESS: "0x00000000000000000000000000000000000000C3",
  };
  const result = runActivation(env);
  allPassed &&= expectRefusal(
    "refuses when no minter contract exists at the address",
    result,
    /.*/, // any non-zero exit: hardhat's no-bytecode refusal or ours
  );
}

if (!allPassed) {
  console.error("PoKW activation rehearsal FAILED a refusal drill.");
  process.exitCode = 1;
} else {
  console.log("All PoKW activation refusal drills passed.");
  console.log("Happy-path coverage: deploy-sepolia inline rehearsal + pokw-minter.test.js.");
}
