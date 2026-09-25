// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/* eslint-disable no-console */

const fs = require("node:fs");
const path = require("node:path");

// EIP-170 limits runtime contract bytecode to 24,576 bytes on Ethereum.
const MAX_RUNTIME_BYTECODE_SIZE = 24_576;
const CONTRACTS = [
  "OmegaTestToken",
  "OmegaVotingEscrow",
  "OmegaLocking",
  "OmegaGovernor",
  "OmegaNovelGate",
];

let exceeded = false;
for (const contractName of CONTRACTS) {
  const artifactPath = path.join(
    __dirname,
    "..",
    "artifacts",
    "contracts",
    `${contractName}.sol`,
    `${contractName}.json`
  );
  const artifact = JSON.parse(fs.readFileSync(artifactPath, "utf8"));
  const runtimeBytecodeSize = (artifact.deployedBytecode.length - 2) / 2;
  const status = runtimeBytecodeSize < MAX_RUNTIME_BYTECODE_SIZE ? "OK" : "TOO LARGE";
  console.log(`${contractName.padEnd(20)} ${String(runtimeBytecodeSize).padStart(5)} bytes  ${status}`);
  exceeded ||= runtimeBytecodeSize >= MAX_RUNTIME_BYTECODE_SIZE;
}

if (exceeded) {
  throw new Error(`A pilot contract exceeds the EIP-170 ${MAX_RUNTIME_BYTECODE_SIZE}-byte limit.`);
}
