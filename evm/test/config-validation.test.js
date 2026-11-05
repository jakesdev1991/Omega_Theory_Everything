// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { validateNetworkUserConfig } from "../node_modules/hardhat/dist/src/internal/builtin-plugins/network-manager/type-validation.js";
import { validateSolidityTestUserConfig } from "../node_modules/hardhat/dist/src/internal/builtin-plugins/solidity-test/config.js";

describe("Hardhat configuration validation", function () {
  it("rejects an invalid deployer key with a configuration error", function () {
    const cli = fileURLToPath(new URL("../node_modules/hardhat/dist/src/cli.js", import.meta.url));
    const result = spawnSync(process.execPath, [cli, "compile"], {
      cwd: fileURLToPath(new URL("..", import.meta.url)),
      env: { ...process.env, DEPLOYER_PRIVATE_KEY: "not-a-private-key" },
      encoding: "utf8",
      timeout: 30000,
    });
    assert.equal(result.error, undefined);
    assert.notEqual(result.status, 0);
    assert.match(result.stdout + result.stderr, /HHE15: Invalid config/);
    assert.match(result.stdout + result.stderr, /sepolia\.accounts/);
    assert.doesNotMatch(result.stdout + result.stderr, /TypeError/);
  });

  it("validates network records and reports malformed nested entries", async function () {
    const network = { type: "http", url: "https://example.com", httpHeaders: { auth: "value" } };
    const chain = { name: "test", chainType: "l1", hardforkHistory: { london: { blockNumber: 1 } } };
    assert.deepEqual(await validateNetworkUserConfig({ networks: { remote: network }, chainDescriptors: { "1": chain } }), []);
    const badHeaders = await validateNetworkUserConfig({ networks: { remote: { ...network, httpHeaders: { auth: 123 } } } });
    assert.ok(badHeaders.some((issue) => issue.path.join(".") === "networks.remote.httpHeaders.auth"));
    const badFork = await validateNetworkUserConfig({ chainDescriptors: { "1": { ...chain, hardforkHistory: { london: { blockNumber: "bad" } } } } });
    assert.ok(badFork.some((issue) => issue.path.join(".") === "chainDescriptors.1.hardforkHistory.london"));
    const badNetwork = await validateNetworkUserConfig({ networks: { remote: { type: "invalid" } } });
    assert.ok(badNetwork.some((issue) => issue.path.join(".").startsWith("networks.remote")));
  });

  it("validates Solidity endpoint records and rejects conflicting profile shapes", function () {
    const config = (rpcEndpoints) => ({ test: { solidity: { forking: { rpcEndpoints } } } });
    assert.deepEqual(validateSolidityTestUserConfig(config({ mainnet: "https://example.com" })), []);
    const invalid = validateSolidityTestUserConfig(config({ mainnet: 42 }));
    assert.ok(invalid.some((issue) => issue.path.join(".") === "test.solidity.forking.rpcEndpoints.mainnet"));
    const conflict = validateSolidityTestUserConfig({ test: { solidity: { profiles: {}, fuzz: { runs: 2 } } } });
    assert.ok(conflict.some((issue) => issue.message.includes("incompatible with the flat solidity test config")));
  });
});
