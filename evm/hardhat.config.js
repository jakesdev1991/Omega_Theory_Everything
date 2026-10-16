// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly

import "dotenv/config";
import { createRequire } from "node:module";
import { defineConfig } from "hardhat/config";
import hardhatToolboxMochaEthers from "@nomicfoundation/hardhat-toolbox-mocha-ethers";

const require = createRequire(import.meta.url);
const solcPath = require.resolve("solc/soljson.js");

// Keep the solc-js binary pinned in package-lock.json so builds never need to
// download a compiler binary at build time.
const solcVersion = "0.8.26";
const sepoliaChainId = 11155111;
const sepoliaRpcUrl =
  process.env.SEPOLIA_RPC_URL || "https://ethereum-sepolia-rpc.publicnode.com";
const deployerPrivateKey = process.env.DEPLOYER_PRIVATE_KEY;
const simulatedSepolia = {
  type: "edr-simulated",
  chainType: "l1",
  chainId: sepoliaChainId,
};

export default defineConfig({
  plugins: [hardhatToolboxMochaEthers],
  solidity: {
    version: solcVersion,
    path: solcPath,
    npmFilesToBuild: ["@openzeppelin/contracts/governance/TimelockController.sol"],
    settings: {
      evmVersion: "cancun",
      optimizer: {
        enabled: true,
        runs: 200,
      },
      metadata: {
        bytecodeHash: "none",
      },
    },
  },
  networks: {
    // Hardhat 3's default network is used by `hardhat test`; mirror Sepolia's
    // chain ID so deployment guards can be exercised against a local simulator.
    default: simulatedSepolia,
    // Retain the named network for the existing offline preflight rehearsal.
    hardhat: simulatedSepolia,
    sepolia: {
      type: "http",
      chainType: "l1",
      chainId: sepoliaChainId,
      url: sepoliaRpcUrl,
      accounts: deployerPrivateKey ? [deployerPrivateKey] : [],
    },
  },
  verify: {
    etherscan: {
      apiKey: process.env.ETHERSCAN_API_KEY || "",
    },
  },
  paths: {
    sources: "./contracts",
    cache: "./cache",
    artifacts: "./artifacts",
  },
  test: {
    mocha: {
      timeout: 120000,
    },
  },
});
