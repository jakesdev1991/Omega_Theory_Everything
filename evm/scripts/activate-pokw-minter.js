// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/* eslint-disable no-console */

/**
 * Two-phase activation for OmegaScienceMinter when the treasury is not
 * the deployer (the deploy script handles the inline case).
 *
 * Steps, in order — the script verifies each before the next:
 *   1. The treasury has already transferred the science budget to the
 *      minter (this script does NOT move treasury funds; the transfer
 *      is a treasury-side decision made out of band).
 *   2. The minter's held balance equals the manifest's
 *      `pokw.scienceBudget` — fail closed on mismatch, so a typo'd
 *      transfer can never silently lock the wrong budget.
 *   3. activate() — locks the budget immutably.
 *   4. renounceOwnership() — ends the deployer's setup authority.
 *   5. Verify the terminal state: activated, owner zero.
 *
 * Usage:
 *   POKW_MINTER_ADDRESS=0x... DEPLOYER_PRIVATE_KEY=0x... \
 *   node scripts/activate-pokw-minter.js --network sepolia
 */

import path from "node:path";
import { fileURLToPath } from "node:url";
import hre from "hardhat";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const connection = await hre.network.create();
const { ethers } = connection;
const SEPOLIA_CHAIN_ID = 11155111n;

async function main() {
  if (!process.env.DEPLOYER_PRIVATE_KEY) {
    throw new Error(
      "DEPLOYER_PRIVATE_KEY is required. Use the Sepolia deployer key only."
    );
  }
  const rawAddress = process.env.POKW_MINTER_ADDRESS;
  if (!rawAddress || !ethers.isAddress(rawAddress)) {
    throw new Error(
      "POKW_MINTER_ADDRESS must be the deployed minter address (see deployments/sepolia.json)."
    );
  }
  const minterAddress = ethers.getAddress(rawAddress);

  const chainId = BigInt(await connection.provider.send("eth_chainId"));
  if (chainId !== SEPOLIA_CHAIN_ID) {
    throw new Error(
      `Refusing to activate: expected Sepolia chain ID ${SEPOLIA_CHAIN_ID}, received ${chainId}.`
    );
  }

  const [deployer] = await ethers.getSigners();

  // Fail-closed ordering: verify CONFIGURATION before touching the chain.
  // The manifest is mandatory — the budget must be verifiable against it —
  // so a typo'd treasury transfer can never lock the wrong budget.
  const manifestPath = path.join(
    __dirname, "..", "deployments", "sepolia.json"
  );
  const fs = await import("node:fs");
  if (!fs.existsSync(manifestPath)) {
    throw new Error(
      "deployments/sepolia.json not found; refusing to activate without it."
    );
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (!manifest.pokw?.scienceBudget) {
    throw new Error(
      "Manifest has no pokw.scienceBudget; refusing to activate. " +
      "The budget must be pinned before it can be locked."
    );
  }
  const expectedBudget = BigInt(manifest.pokw.scienceBudget);
  if (manifest.contracts?.omegaScienceMinter
      && ethers.getAddress(manifest.contracts.omegaScienceMinter) !== minterAddress) {
    throw new Error(
      `POKW_MINTER_ADDRESS (${minterAddress}) does not match the manifest's ` +
      `minter (${manifest.contracts.omegaScienceMinter}).`
    );
  }

  const minter = await ethers.getContractAt(
    "OmegaScienceMinter",
    minterAddress
  );

  if (await minter.activated()) {
    throw new Error("Minter is already activated; nothing to do.");
  }
  if ((await minter.owner()) !== deployer.address) {
    throw new Error(
      `Refusing to activate: minter owner is ${await minter.owner()}, not this deployer.`
    );
  }

  const tokenAddress = await minter.omega();
  const token = await ethers.getContractAt("OmegaTestToken", tokenAddress);
  const held = await token.balanceOf(minterAddress);

  if (held !== expectedBudget) {
    throw new Error(
      `Budget mismatch: minter holds ${held.toString()} tOMEGA but the manifest says ${expectedBudget.toString()}. ` +
      "Transfer the correct science budget before activating."
    );
  }

  console.log("Activating OmegaScienceMinter");
  console.log(`  minter       ${minterAddress}`);
  console.log(`  budget       ${ethers.formatUnits(held, 18)} tOMEGA`);

  await (await minter.connect(deployer).activate()).wait();
  await (await minter.connect(deployer).renounceOwnership()).wait();

  // Terminal state checks — fail closed.
  if (!(await minter.activated())) {
    throw new Error("Activation failed: minter reports not activated.");
  }
  if ((await minter.owner()) !== ethers.ZeroAddress) {
    throw new Error("Renounce failed: minter owner is not zero.");
  }
  console.log("  activated + ownership renounced; the timelock now holds all post-activation power.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
