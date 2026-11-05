// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly

// Fail closed if lifecycle scripts were skipped or the pinned packages changed.
// Update these hashes only after reviewing the corresponding patch and tests.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const patches = [
  ["@nomicfoundation/hardhat-zod-utils", "3.0.8", "dist/src/index.js", "d2664d277a91402431996f6ae27f05fb57c09863b48e9638ccacebfe65d1eb5c"],
  ["hardhat", "3.21.2", "dist/src/internal/builtin-plugins/network-manager/type-validation.js", "5344cf8b59ac161612505a74dc61f8da500094acba2f2202ab218f81d53ad288"],
  ["hardhat", "3.21.2", "dist/src/internal/builtin-plugins/solidity-test/config.js", "7b0a33913ba71806514dc72d131152428b1a1854cebaca4e2bfa8e67f1de87b3"],
];

export function assertPatchFile(filename, expected) {
  const actual = createHash("sha256").update(readFileSync(filename)).digest("hex");
  if (actual !== expected) {
    throw new Error(`Zod 4 compatibility patch missing or changed: ${filename}. Run npm ci with lifecycle scripts enabled.`);
  }
}

export function requireZod4Compatibility() {
  const zod = require("zod/package.json");
  if (zod.version !== "4.6.7") {
    throw new Error(`Unsupported Zod version ${zod.version}; expected 4.6.7. Review the compatibility patches before upgrading.`);
  }
  for (const [name, version, filename, hash] of patches) {
    const manifest = require.resolve(`${name}/package.json`);
    const installed = JSON.parse(readFileSync(manifest, "utf8"));
    if (installed.version !== version) {
      throw new Error(`Unsupported ${name} version ${installed.version}; expected ${version}. Review the compatibility patches before upgrading.`);
    }
    assertPatchFile(path.join(path.dirname(manifest), filename), hash);
  }
}
