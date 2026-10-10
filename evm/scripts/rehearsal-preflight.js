// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/* eslint-disable no-console */

/**
 * Rehearses the Sepolia preflight against the in-process local network.
 *
 * The preflight's primary job is to refuse to proceed, so "does it still work?"
 * cannot be answered by the failure it gives without a deployer key. This runner
 * gives it a configuration that should pass and runs the same script against
 * `--network hardhat`, which hardhat.config.js deliberately pins to Sepolia's
 * chain ID (11155111) so guarded bytecode can be exercised offline.
 *
 * Every value here is a rehearsal value:
 *   - the key is never used for signing — the local network signs with its own
 *     funded accounts, and the preflight only requires the variable to be set;
 *   - the addresses are placeholders (any non-zero address satisfies the schema
 *     checks, which is what is being exercised);
 *   - nothing is broadcast: the preflight sends no transactions, by construction.
 *
 * Set DEPLOYER_PRIVATE_KEY / TREASURY_ADDRESS / GUARDIAN_ADDRESS in the
 * environment to rehearse with your own values instead.
 */

const { spawnSync } = require("node:child_process");

const rehearsalDefaults = {
  DEPLOYER_PRIVATE_KEY: `0x${"11".repeat(32)}`,
  TREASURY_ADDRESS: "0x00000000000000000000000000000000000000A1",
  GUARDIAN_ADDRESS: "0x00000000000000000000000000000000000000B2",
};

const env = { ...process.env };
for (const [name, value] of Object.entries(rehearsalDefaults)) {
  if (!env[name] || !env[name].trim()) {
    env[name] = value;
  }
}

console.log("Rehearsing the Sepolia preflight against the local network (nothing is broadcast).");
const result = spawnSync(
  "npx",
  ["hardhat", "run", "scripts/preflight-sepolia.js", "--network", "hardhat"],
  {
    cwd: `${__dirname}/..`,
    env,
    stdio: "inherit",
    shell: process.platform === "win32",
  },
);

if (result.error) {
  console.error(`Rehearsal could not run: ${result.error.message}`);
  process.exitCode = 1;
} else {
  process.exitCode = result.status ?? 1;
}
