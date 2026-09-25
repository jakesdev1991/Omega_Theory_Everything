// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/* eslint-disable no-console */

const fs = require("node:fs");
const path = require("node:path");
const hre = require("hardhat");

const SEPOLIA_CHAIN_ID = 11155111n;

async function main() {
  if (!process.env.ETHERSCAN_API_KEY) {
    throw new Error("ETHERSCAN_API_KEY is required for explorer verification.");
  }

  const chainId = BigInt(await hre.network.provider.send("eth_chainId"));
  if (chainId !== SEPOLIA_CHAIN_ID) {
    throw new Error(
      `Refusing to verify: expected Sepolia chain ID ${SEPOLIA_CHAIN_ID}, received ${chainId}.`
    );
  }

  const manifestPath = path.join(__dirname, "..", "deployments", "sepolia.json");
  if (!fs.existsSync(manifestPath)) {
    throw new Error(
      `No deployment manifest found at ${manifestPath}. Deploy first, or restore the public manifest from the pilot record.`
    );
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (manifest.chainId !== Number(SEPOLIA_CHAIN_ID) || !manifest.testnetOnly) {
    throw new Error("Deployment manifest is not a Sepolia-only pilot manifest.");
  }

  const contracts = [
    {
      label: "OmegaTestToken",
      address: manifest.contracts.omegaTestToken,
      contract: "contracts/OmegaTestToken.sol:OmegaTestToken",
      args: manifest.constructorArguments.omegaTestToken,
    },
    {
      label: "OmegaVotingEscrow",
      address: manifest.contracts.omegaVotingEscrow,
      contract: "contracts/OmegaVotingEscrow.sol:OmegaVotingEscrow",
      args: manifest.constructorArguments.omegaVotingEscrow,
    },
    {
      label: "TimelockController",
      address: manifest.contracts.timelockController,
      contract: "@openzeppelin/contracts/governance/TimelockController.sol:TimelockController",
      args: manifest.constructorArguments.timelockController,
    },
    {
      label: "OmegaLocking",
      address: manifest.contracts.omegaLocking,
      contract: "contracts/OmegaLocking.sol:OmegaLocking",
      args: manifest.constructorArguments.omegaLocking,
    },
    {
      label: "OmegaGovernor",
      address: manifest.contracts.omegaGovernor,
      contract: "contracts/OmegaGovernor.sol:OmegaGovernor",
      args: manifest.constructorArguments.omegaGovernor,
    },
    {
      label: "OmegaNovelGate",
      address: manifest.contracts.omegaNovelGate,
      contract: "contracts/OmegaNovelGate.sol:OmegaNovelGate",
      args: manifest.constructorArguments.omegaNovelGate,
    },
  ];

  for (const entry of contracts) {
    console.log(`Verifying ${entry.label} at ${entry.address}…`);
    try {
      await hre.run("verify:verify", {
        address: entry.address,
        constructorArguments: entry.args,
        contract: entry.contract,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (/already verified|already been verified/i.test(message)) {
        console.log(`  ${entry.label} is already verified.`);
        continue;
      }
      throw error;
    }
  }

  console.log("All Sepolia pilot contracts submitted for verification.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
