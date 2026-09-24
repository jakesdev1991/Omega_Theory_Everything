import { access, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import { AMITY_MANIFEST_VERSION, TESTNET_NETWORK } from "./constants.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(`Manifest error: ${message}`);
}

function normalizeOperatorFiles(operatorFiles) {
  assert(Array.isArray(operatorFiles) && operatorFiles.length === 4, "operatorFiles must contain the four expected AMITY operator file checks.");

  return operatorFiles.map((entry) => {
    assert(entry && typeof entry === "object", "each operator file entry must be an object.");
    assert(typeof entry.label === "string" && entry.label.length > 0, "each operator file entry needs a label.");
    assert(typeof entry.configured === "boolean", `operator file ${entry.label} must include configured=true|false.`);
    assert(typeof entry.exists === "boolean", `operator file ${entry.label} must include exists=true|false.`);
    return Object.freeze({
      label: entry.label,
      configured: entry.configured,
      exists: entry.exists,
    });
  });
}

export function createManifest({ config, operatorFiles }) {
  const normalizedFiles = normalizeOperatorFiles(operatorFiles);
  const operatorFilesPresent = normalizedFiles.every((entry) => entry.configured && entry.exists);

  return {
    manifestVersion: AMITY_MANIFEST_VERSION,
    generatedAt: new Date().toISOString(),
    pilot: {
      network: TESTNET_NETWORK,
      valueless: true,
      purpose: "AMITY Taproot Assets testnet operator scaffold snapshot",
      wiredIntoWebUnlock: false,
      liveHolderVerificationImplemented: false,
    },
    asset: {
      assetId: config.assetId,
      assetName: config.assetName,
      unitName: config.unitName,
      universeUrl: config.universeUrl,
    },
    services: {
      litd: {
        rpcHost: config.litdRpcHost,
        tlsCertConfigured: !!config.litdTlsCertPath,
        macaroonConfigured: !!config.litdMacaroonPath,
      },
      tapd: {
        rpcHost: config.tapdRpcHost,
        tlsCertConfigured: !!config.tapdTlsCertPath,
        macaroonConfigured: !!config.tapdMacaroonPath,
      },
    },
    operatorFiles: normalizedFiles,
    readiness: {
      requiredConfigComplete: true,
      operatorFilesPresent,
      holderVerificationReady: false,
    },
    notes: [
      "This manifest is a local operator scaffold snapshot only.",
      "It does not imply Taproot Asset issuance, Universe proof verification, or web unlock wiring.",
      "The live wallet/web unlock flow in this repository remains two rails only: $OMEGA and TWC.",
    ],
  };
}

export function assertValidManifest(manifest) {
  assert(manifest && typeof manifest === "object", "root must be an object.");
  assert(manifest.manifestVersion === AMITY_MANIFEST_VERSION, `unsupported manifestVersion (expected ${AMITY_MANIFEST_VERSION}).`);
  assert(typeof manifest.generatedAt === "string" && Number.isFinite(Date.parse(manifest.generatedAt)), "generatedAt must be an ISO timestamp.");

  assert(manifest.pilot?.network === TESTNET_NETWORK, `pilot.network must be ${TESTNET_NETWORK}.`);
  assert(manifest.pilot?.valueless === true, "pilot.valueless must be true.");
  assert(typeof manifest.pilot?.purpose === "string" && manifest.pilot.purpose.length > 0, "pilot.purpose is required.");
  assert(manifest.pilot?.wiredIntoWebUnlock === false, "pilot.wiredIntoWebUnlock must remain false until AMITY is actually integrated.");
  assert(manifest.pilot?.liveHolderVerificationImplemented === false, "pilot.liveHolderVerificationImplemented must remain false in the scaffold.");

  assert(/^[a-f0-9]{64}$/.test(manifest.asset?.assetId ?? ""), "asset.assetId must be a 64-character lowercase hex string.");
  assert(typeof manifest.asset?.assetName === "string" && manifest.asset.assetName.length > 0, "asset.assetName is required.");
  assert(typeof manifest.asset?.unitName === "string" && manifest.asset.unitName.length > 0, "asset.unitName is required.");
  assert(typeof manifest.asset?.universeUrl === "string" && /^https?:\/\//.test(manifest.asset.universeUrl), "asset.universeUrl must be an absolute HTTP(S) URL.");

  assert(typeof manifest.services?.litd?.rpcHost === "string" && /^[^:\s]+:\d{2,5}$/.test(manifest.services.litd.rpcHost), "services.litd.rpcHost must look like host:port.");
  assert(typeof manifest.services?.tapd?.rpcHost === "string" && /^[^:\s]+:\d{2,5}$/.test(manifest.services.tapd.rpcHost), "services.tapd.rpcHost must look like host:port.");
  assert(typeof manifest.services?.litd?.tlsCertConfigured === "boolean", "services.litd.tlsCertConfigured must be boolean.");
  assert(typeof manifest.services?.litd?.macaroonConfigured === "boolean", "services.litd.macaroonConfigured must be boolean.");
  assert(typeof manifest.services?.tapd?.tlsCertConfigured === "boolean", "services.tapd.tlsCertConfigured must be boolean.");
  assert(typeof manifest.services?.tapd?.macaroonConfigured === "boolean", "services.tapd.macaroonConfigured must be boolean.");

  const normalizedFiles = normalizeOperatorFiles(manifest.operatorFiles);
  assert(manifest.readiness?.requiredConfigComplete === true, "readiness.requiredConfigComplete must be true.");
  assert(typeof manifest.readiness?.operatorFilesPresent === "boolean", "readiness.operatorFilesPresent must be boolean.");
  assert(manifest.readiness?.holderVerificationReady === false, "readiness.holderVerificationReady must remain false in the scaffold.");
  assert(Array.isArray(manifest.notes) && manifest.notes.length >= 1, "notes must contain at least one operator note.");

  const computedFilesPresent = normalizedFiles.every((entry) => entry.configured && entry.exists);
  assert(computedFilesPresent === manifest.readiness.operatorFilesPresent, "readiness.operatorFilesPresent does not match operatorFiles.");
  return manifest;
}

export async function readManifest(manifestPath) {
  let manifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  } catch (error) {
    throw new Error(`Cannot read AMITY manifest at ${manifestPath}: ${error.message}`);
  }
  return assertValidManifest(manifest);
}

export async function writeManifest(manifestPath, manifest, { overwrite = false } = {}) {
  assertValidManifest(manifest);

  if (!overwrite) {
    try {
      await access(manifestPath);
      throw new Error(`Refusing to overwrite existing manifest: ${manifestPath}`);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }

  await mkdir(dirname(manifestPath), { recursive: true });
  const temporaryPath = `${manifestPath}.${process.pid}.${Date.now()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(manifest, null, 2)}\n`, { mode: 0o600 });
    await rename(temporaryPath, manifestPath);
  } catch (error) {
    throw new Error(`Could not write AMITY manifest: ${error.message}`);
  }
}
