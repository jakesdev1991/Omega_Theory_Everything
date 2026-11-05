// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly

import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { assertPatchFile, requireZod4Compatibility } from "../scripts/require-zod4-compatibility.js";
import { z } from "zod";
import { incompatibleFieldType, resolvedConfigurationVariableSchema, validateUserConfigZodType } from "@nomicfoundation/hardhat-zod-utils";
import { rpcAddress, rpcQuantity, rpcTransactionRequest, validateParams } from "@nomicfoundation/hardhat-zod-utils/rpc";
import mochaConfig from "../node_modules/@nomicfoundation/hardhat-mocha/dist/src/hookHandlers/config.js";
import { validateUserConfig as validateEthers } from "../node_modules/@nomicfoundation/hardhat-ethers/dist/src/internal/hook-handlers/config.js";
import { validateTypechainUserConfig } from "../node_modules/@nomicfoundation/hardhat-typechain/dist/src/internal/config/validation.js";
import { validateUserConfig as validateVerify } from "../node_modules/@nomicfoundation/hardhat-verify/dist/src/internal/hook-handlers/config.js";
import { validateSolidityUserConfig } from "../node_modules/hardhat/dist/src/internal/builtin-plugins/solidity/config.js";
import { validateCoverageUserConfig } from "../node_modules/hardhat/dist/src/internal/builtin-plugins/coverage/config.js";

const errorsAt = (errors, path) => errors.some((error) => error.path.join(".") === path);

describe("Pinned Hardhat Zod 4 compatibility", function () {
  it("fails closed when a compatibility patch is missing or changed", function () {
    requireZod4Compatibility();
    const dir = mkdtempSync(path.join(tmpdir(), "omega-patch-"));
    try {
      const filename = path.join(dir, "sample.js");
      const expected = createHash("sha256").update("patched").digest("hex");
      writeFileSync(filename, "patched");
      assert.doesNotThrow(() => assertPatchFile(filename, expected));
      writeFileSync(filename, "unpatched");
      assert.throws(() => assertPatchFile(filename, expected), /compatibility patch missing or changed/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("retains useful errors, custom messages and function validation", function () {
    const schema = z.object({ forbidden: incompatibleFieldType("not allowed") });
    assert.deepEqual(validateUserConfigZodType({}, schema), []);
    assert.equal(validateUserConfigZodType({ forbidden: true }, schema)[0].message, "not allowed");
    const variable = { _type: "ResolvedConfigurationVariable", get() {}, getUrl() {}, getBigInt() {}, getHexString() {} };
    assert.equal(resolvedConfigurationVariableSchema.safeParse(variable).success, true);
    assert.equal(resolvedConfigurationVariableSchema.safeParse({ ...variable, get: 5 }).success, false);
  });

  it("validates solidity compiler profiles and coverage settings", function () {
    const valid = { solidity: { compilers: [{ version: "0.8.26" }], overrides: { "A.sol": { version: "0.8.26" } } } };
    assert.deepEqual(validateSolidityUserConfig(valid), []);
    const bad = { solidity: { compilers: [{ version: "0.8.26" }], overrides: { "A.sol": { version: 42 } } } };
    assert.ok(errorsAt(validateSolidityUserConfig(bad), "solidity.overrides.A.sol.version"));
    assert.deepEqual(validateCoverageUserConfig({ coverage: { skipFiles: ["A.sol"] } }), []);
    assert.ok(errorsAt(validateCoverageUserConfig({ coverage: { skipFiles: [5] } }), "coverage.skipFiles.0"));
  });

  it("validates the installed Mocha, Ethers, Typechain and Verify plugins", async function () {
    const mocha = await mochaConfig();
    assert.deepEqual(await mocha.validateUserConfig({ test: { mocha: { timeout: 1000, rootHooks: { beforeAll: () => {} } } } }), []);
    assert.ok(errorsAt(await mocha.validateUserConfig({ test: { mocha: { timeout: false } } }), "test.mocha.timeout"));
    assert.deepEqual(await validateEthers({ networks: { local: { ethers: { waitForTransactionReceipt: true } } } }), []);
    assert.ok(errorsAt(await validateEthers({ networks: { local: { ethers: { waitForTransactionReceipt: "yes" } } } }), "ethers.waitForTransactionReceipt"));
    assert.deepEqual(await validateTypechainUserConfig({ typechain: { outDir: "/tmp/types", tsNocheck: true } }), []);
    assert.ok(errorsAt(await validateTypechainUserConfig({ typechain: { tsNocheck: "yes" } }), "tsNocheck"));
    assert.deepEqual(await validateVerify({ verify: { etherscan: { apiKey: "test" } } }), []);
    assert.deepEqual(await validateVerify({ verify: { etherscan: { enabled: false } } }), []);
    assert.ok(errorsAt(await validateVerify({ verify: { etherscan: {} } }), "verify.etherscan.apiKey"));
  });

  it("validates and decodes RPC inputs without accepting invalid transactions", function () {
    const address = "0x1111111111111111111111111111111111111111";
    const result = rpcTransactionRequest.safeParse({ from: address, gas: "0x10" });
    assert.equal(result.success, true);
    assert.equal(result.data.gas, 16n);
    assert.equal(Buffer.from(result.data.from).toString("hex"), address.slice(2));
    assert.equal(rpcTransactionRequest.safeParse({}).success, false);
    assert.equal(rpcTransactionRequest.safeParse({ from: "invalid" }).success, false);
    assert.equal(rpcTransactionRequest.safeParse({ from: address, gas: "invalid" }).success, false);
    assert.deepEqual(validateParams([address, "0x10"], rpcAddress, rpcQuantity).map((x) => typeof x === "bigint" ? x : Buffer.from(x).toString("hex")), [address.slice(2), 16n]);
    assert.throws(() => validateParams([address, "not-a-quantity"], rpcAddress, rpcQuantity));
  });
});
