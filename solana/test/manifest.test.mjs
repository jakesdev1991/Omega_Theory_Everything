import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { createManifest, readManifest, writeManifestNew } from "../lib/manifest.mjs";

test("manifest contains only public deployment data and enforces immutable authorities", async () => {
  const manifest = createManifest({
    deploymentConfig: {
      rpcUrl: "https://api.devnet.solana.com",
      treasuryAddress: "11111111111111111111111111111111",
      metadataUri: "https://arweave.net/metadata",
      metadataSha256: "a".repeat(64),
      initialSupplyTokens: "1000000000",
      initialSupplyBaseUnits: "1000000000000000000",
    },
    payerAddress: "11111111111111111111111111111111",
    mintAddress: "11111111111111111111111111111111",
    metadataAddress: "11111111111111111111111111111111",
    treasuryAtaAddress: "11111111111111111111111111111111",
    transactionSignature: "sig",
    metadataByteLength: 123,
  });

  assert.equal(manifest.mint.mintAuthority, null);
  assert.equal(manifest.mint.freezeAuthority, null);
  assert.equal(manifest.metadata.isMutable, false);
  assert.equal(JSON.stringify(manifest).includes("secretKey"), false);

  const directory = await mkdtemp(join(tmpdir(), "twc-manifest-"));
  const manifestPath = join(directory, "twc-devnet.json");
  await writeManifestNew(manifestPath, manifest);
  assert.match(await readFile(manifestPath, "utf8"), /"valueless": true/);
  assert.deepEqual(await readManifest(manifestPath), manifest);
  await assert.rejects(writeManifestNew(manifestPath, manifest), /Refusing to overwrite/);
});
