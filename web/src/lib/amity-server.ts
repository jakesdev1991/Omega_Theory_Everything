// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import "server-only";

import { access, readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";

type EnvEntries = Record<string, string>;

type OperatorFileStatus = {
  label: string;
  configured: boolean;
  exists: boolean;
};

type AmityManifest = {
  manifestVersion?: unknown;
  pilot?: {
    network?: unknown;
    valueless?: unknown;
    wiredIntoWebUnlock?: unknown;
    liveHolderVerificationImplemented?: unknown;
  };
  asset?: {
    assetId?: unknown;
    assetName?: unknown;
    unitName?: unknown;
    universeUrl?: unknown;
  };
  services?: {
    litd?: {
      rpcHost?: unknown;
      tlsCertConfigured?: unknown;
      macaroonConfigured?: unknown;
    };
    tapd?: {
      rpcHost?: unknown;
      tlsCertConfigured?: unknown;
      macaroonConfigured?: unknown;
    };
  };
  operatorFiles?: Array<{
    label?: unknown;
    configured?: unknown;
    exists?: unknown;
  }>;
};

type ManifestInspection = {
  present: boolean;
  valid: boolean;
  warning: string | null;
  config: {
    assetId: string;
    assetName: string;
    unitName: string;
    universeUrl: string;
    litdRpcHost: string;
    tapdRpcHost: string;
    operatorFiles: OperatorFileStatus[];
  } | null;
};

type ManifestLoadResult = {
  present: boolean;
  manifest: AmityManifest | null;
  warning: string | null;
};

const TESTNET_NETWORK = "testnet";
const DEFAULT_AMITY_ENV_PATH = resolve(process.cwd(), "..", "amity", ".env");
const DEFAULT_AMITY_MANIFEST_PATH =
  process.env.AMITY_TESTNET_MANIFEST_PATH?.trim() ||
  resolve(process.cwd(), "..", "amity", "deployments", "amity-testnet.json");

const OPERATOR_FILE_SPECS = [
  { label: "LITD TLS cert", envKey: "LITD_TLS_CERT_PATH" },
  { label: "LITD macaroon", envKey: "LITD_MACAROON_PATH" },
  { label: "TAPD TLS cert", envKey: "TAPD_TLS_CERT_PATH" },
  { label: "TAPD macaroon", envKey: "TAPD_MACAROON_PATH" },
] as const;

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function firstString(...values: Array<unknown>) {
  for (const value of values) {
    if (isNonEmptyString(value)) {
      return value.trim();
    }
  }
  return undefined;
}

function isHex64(value: string) {
  return /^[a-fA-F0-9]{64}$/.test(value);
}

function isHostPort(value: string) {
  return /^[^:\s]+:\d{2,5}$/.test(value);
}

function isHttpUrl(value: string) {
  try {
    const parsed = new URL(value);
    return ["http:", "https:"].includes(parsed.protocol) && !parsed.username && !parsed.password;
  } catch {
    return false;
  }
}

async function exists(path: string) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function loadManifest(path: string): Promise<ManifestLoadResult> {
  if (!(await exists(path))) {
    return { present: false, manifest: null, warning: null };
  }

  try {
    return {
      present: true,
      manifest: JSON.parse(await readFile(path, "utf8")) as AmityManifest,
      warning: null,
    };
  } catch (error) {
    return {
      present: true,
      manifest: null,
      warning: error instanceof Error ? `Could not parse AMITY manifest JSON: ${error.message}` : "Could not parse AMITY manifest JSON.",
    };
  }
}

async function readEnvFile(path: string): Promise<EnvEntries> {
  if (!(await exists(path))) {
    return {};
  }

  const text = await readFile(path, "utf8");
  const entries: EnvEntries = {};

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const index = line.indexOf("=");
    if (index === -1) {
      continue;
    }

    const key = line.slice(0, index).trim();
    let value = line.slice(index + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    entries[key] = value;
  }

  return entries;
}

function parseOperatorFilesFromManifest(manifest: AmityManifest): OperatorFileStatus[] | null {
  if (!Array.isArray(manifest.operatorFiles) || manifest.operatorFiles.length !== 4) {
    return null;
  }

  const normalized: OperatorFileStatus[] = [];
  for (const entry of manifest.operatorFiles) {
    if (!isNonEmptyString(entry?.label) || typeof entry?.configured !== "boolean" || typeof entry?.exists !== "boolean") {
      return null;
    }

    normalized.push({
      label: entry.label,
      configured: entry.configured,
      exists: entry.exists,
    });
  }

  return normalized;
}

function inspectManifest(manifest: AmityManifest | null, present = false, loadWarning: string | null = null): ManifestInspection {
  if (!manifest) {
    return { present, valid: false, warning: loadWarning, config: null };
  }

  try {
    if (manifest.manifestVersion !== 1) {
      throw new Error("manifestVersion must be 1.");
    }
    if (manifest.pilot?.network !== TESTNET_NETWORK) {
      throw new Error(`pilot.network must be ${TESTNET_NETWORK}.`);
    }
    if (manifest.pilot?.valueless !== true) {
      throw new Error("pilot.valueless must be true.");
    }
    if (manifest.pilot?.wiredIntoWebUnlock !== false) {
      throw new Error("pilot.wiredIntoWebUnlock must remain false.");
    }
    if (manifest.pilot?.liveHolderVerificationImplemented !== false) {
      throw new Error("pilot.liveHolderVerificationImplemented must remain false.");
    }

    const assetId = firstString(manifest.asset?.assetId);
    const assetName = firstString(manifest.asset?.assetName);
    const unitName = firstString(manifest.asset?.unitName);
    const universeUrl = firstString(manifest.asset?.universeUrl);
    const litdRpcHost = firstString(manifest.services?.litd?.rpcHost);
    const tapdRpcHost = firstString(manifest.services?.tapd?.rpcHost);
    const operatorFiles = parseOperatorFilesFromManifest(manifest);

    if (!assetId || !isHex64(assetId)) {
      throw new Error("asset.assetId must be a 64-character hex string.");
    }
    if (!assetName) {
      throw new Error("asset.assetName is required.");
    }
    if (!unitName) {
      throw new Error("asset.unitName is required.");
    }
    if (!universeUrl || !isHttpUrl(universeUrl)) {
      throw new Error("asset.universeUrl must be an absolute HTTP(S) URL.");
    }
    if (!litdRpcHost || !isHostPort(litdRpcHost)) {
      throw new Error("services.litd.rpcHost must look like host:port.");
    }
    if (!tapdRpcHost || !isHostPort(tapdRpcHost)) {
      throw new Error("services.tapd.rpcHost must look like host:port.");
    }
    if (!operatorFiles) {
      throw new Error("operatorFiles must include the four expected file checks.");
    }

    return {
      present: true,
      valid: true,
      warning: loadWarning,
      config: {
        assetId: assetId.toLowerCase(),
        assetName,
        unitName,
        universeUrl,
        litdRpcHost,
        tapdRpcHost,
        operatorFiles,
      },
    };
  } catch (error) {
    return {
      present: true,
      valid: false,
      warning: error instanceof Error ? error.message : "Invalid AMITY manifest.",
      config: null,
    };
  }
}

async function inspectOperatorFilesFromEnv(env: EnvEntries) {
  const rows = await Promise.all(
    OPERATOR_FILE_SPECS.map(async ({ label, envKey }) => {
      const filePath = firstString(process.env[envKey], env[envKey]);
      if (!filePath) {
        return {
          label,
          configured: false,
          exists: false,
        } satisfies OperatorFileStatus;
      }

      try {
        const info = await stat(/* turbopackIgnore: true */ filePath);
        return {
          label,
          configured: true,
          exists: info.isFile(),
        } satisfies OperatorFileStatus;
      } catch {
        return {
          label,
          configured: true,
          exists: false,
        } satisfies OperatorFileStatus;
      }
    }),
  );

  return rows;
}

export async function getAmityScaffoldStatus() {
  const [envEntries, envPresent, manifestLoad] = await Promise.all([
    readEnvFile(DEFAULT_AMITY_ENV_PATH),
    exists(DEFAULT_AMITY_ENV_PATH),
    loadManifest(DEFAULT_AMITY_MANIFEST_PATH),
  ]);

  const manifest = inspectManifest(manifestLoad.manifest, manifestLoad.present, manifestLoad.warning);
  const warnings: string[] = [];
  if (manifest.warning) {
    warnings.push(`Manifest warning: ${manifest.warning}`);
  }

  const assetId = firstString(
    manifest.config?.assetId,
    process.env.AMITY_ASSET_ID,
    envEntries.AMITY_ASSET_ID,
  );
  const assetName = firstString(
    manifest.config?.assetName,
    process.env.AMITY_ASSET_NAME,
    envEntries.AMITY_ASSET_NAME,
    "AMITY Test Asset",
  );
  const unitName = firstString(
    manifest.config?.unitName,
    process.env.AMITY_UNIT_NAME,
    envEntries.AMITY_UNIT_NAME,
    "AMITY",
  );
  const universeUrl = firstString(
    manifest.config?.universeUrl,
    process.env.AMITY_UNIVERSE_URL,
    envEntries.AMITY_UNIVERSE_URL,
  );
  const explicitLitdRpcHost = firstString(
    manifest.config?.litdRpcHost,
    process.env.LITD_RPC_HOST,
    envEntries.LITD_RPC_HOST,
  );
  const explicitTapdRpcHost = firstString(
    manifest.config?.tapdRpcHost,
    process.env.TAPD_RPC_HOST,
    envEntries.TAPD_RPC_HOST,
  );
  const litdRpcHost = explicitLitdRpcHost ?? "127.0.0.1:8443";
  const tapdRpcHost = explicitTapdRpcHost ?? "127.0.0.1:10029";
  const network = firstString(process.env.AMITY_NETWORK, envEntries.AMITY_NETWORK, TESTNET_NETWORK) ?? TESTNET_NETWORK;

  const scaffoldConfigured =
    network === TESTNET_NETWORK &&
    !!assetId &&
    isHex64(assetId) &&
    !!universeUrl &&
    isHttpUrl(universeUrl) &&
    !!litdRpcHost &&
    isHostPort(litdRpcHost) &&
    !!tapdRpcHost &&
    isHostPort(tapdRpcHost);

  const processEnvConfigured = [
    "AMITY_NETWORK",
    "AMITY_ASSET_ID",
    "AMITY_UNIVERSE_URL",
    "AMITY_ASSET_NAME",
    "AMITY_UNIT_NAME",
    "LITD_RPC_HOST",
    "TAPD_RPC_HOST",
    "LITD_TLS_CERT_PATH",
    "LITD_MACAROON_PATH",
    "TAPD_TLS_CERT_PATH",
    "TAPD_MACAROON_PATH",
  ].some((key) => isNonEmptyString(process.env[key]));

  const operatorFiles = Object.keys(envEntries).length > 0 || processEnvConfigured
    ? await inspectOperatorFilesFromEnv(envEntries)
    : manifest.config?.operatorFiles ??
      OPERATOR_FILE_SPECS.map(({ label }) => ({ label, configured: false, exists: false } satisfies OperatorFileStatus));

  const operatorFilesReady = operatorFiles.every((entry) => entry.configured && entry.exists);
  const requiredFileCount = operatorFiles.filter((entry) => entry.exists).length;
  const envConfigured = envPresent || processEnvConfigured;
  const configSource = manifest.valid ? (envConfigured ? "manifest + env" : "manifest") : envConfigured ? "env" : "none";

  let message = "AMITY remains a separate testnet scaffold and is not wired into the live unlock flow.";
  if (!scaffoldConfigured) {
    message = "AMITY scaffold configuration is incomplete. Add amity/.env or generate amity/deployments/amity-testnet.json before attempting node/proof integration.";
  } else if (!operatorFilesReady) {
    message = "AMITY scaffold configuration is present, but TLS/macaroon operator files are still incomplete or missing.";
  } else {
    message = "AMITY scaffold configuration and local operator files are present, but live holder verification is still not implemented. The wallet/web unlock flow remains two rails only: $OMEGA and TWC.";
  }

  return {
    stage: "testnet-scaffold",
    wiredIntoUnlockFlow: false,
    holderVerificationReady: false,
    scaffoldConfigured,
    operatorFilesReady,
    network,
    manifestPath: DEFAULT_AMITY_MANIFEST_PATH,
    manifestPresent: manifest.present,
    manifestValid: manifest.valid,
    localEnvPath: DEFAULT_AMITY_ENV_PATH,
    localEnvPresent: envPresent,
    configSource,
    assetId: assetId ? assetId.toLowerCase() : null,
    assetName: assetName ?? null,
    unitName: unitName ?? null,
    universeUrl: universeUrl ?? null,
    litdRpcConfigured: !!explicitLitdRpcHost && isHostPort(explicitLitdRpcHost),
    tapdRpcConfigured: !!explicitTapdRpcHost && isHostPort(explicitTapdRpcHost),
    litdRpcStatus: manifest.config?.litdRpcHost ? "manifest" : explicitLitdRpcHost ? "env" : "default localhost",
    tapdRpcStatus: manifest.config?.tapdRpcHost ? "manifest" : explicitTapdRpcHost ? "env" : "default localhost",
    operatorFilesPresentCount: requiredFileCount,
    operatorFilesRequiredCount: operatorFiles.length,
    operatorFiles,
    message,
    warnings,
  };
}
