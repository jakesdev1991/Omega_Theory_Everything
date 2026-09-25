// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly

require("dotenv").config();
require("@nomicfoundation/hardhat-toolbox");

const { subtask } = require("hardhat/config");
const { TASK_COMPILE_SOLIDITY_GET_SOLC_BUILD } = require("hardhat/builtin-tasks/task-names");

// Use the pinned solc-js package instead of downloading a compiler binary at
// build time. This keeps local and CI builds reproducible in restricted
// network environments.
subtask(TASK_COMPILE_SOLIDITY_GET_SOLC_BUILD, async (args, _hre, runSuper) => {
  if (args.solcVersion === "0.8.26") {
    return {
      compilerPath: require.resolve("solc/soljson.js"),
      isSolcJs: true,
      version: "0.8.26",
      longVersion: "0.8.26+commit.8a97fa7a",
    };
  }
  return runSuper();
});

const sepoliaRpcUrl =
  process.env.SEPOLIA_RPC_URL || "https://ethereum-sepolia-rpc.publicnode.com";
const deployerPrivateKey = process.env.DEPLOYER_PRIVATE_KEY;

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.26",
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
    // The contracts deliberately guard deployment to Sepolia's chain ID. Mirroring
    // that ID locally lets the same bytecode be exercised in deterministic tests.
    hardhat: {
      chainId: 11155111,
    },
    sepolia: {
      chainId: 11155111,
      url: sepoliaRpcUrl,
      accounts: deployerPrivateKey ? [deployerPrivateKey] : [],
    },
  },
  etherscan: {
    apiKey: {
      sepolia: process.env.ETHERSCAN_API_KEY || "",
    },
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts",
  },
  mocha: {
    timeout: 120000,
  },
};
