import { access, readFile, stat } from "node:fs/promises";
import { join } from "node:path";

import { AMITY_ROOT, TESTNET_NETWORK } from "./constants.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(`Configuration error: ${message}`);
}

async function loadDotEnv(path) {
  try {
    const text = await readFile(path, "utf8");
    const parsed = {};
    for (const rawLine of text.split(/\r?\n/)) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) continue;
      const index = line.indexOf("=");
      if (index === -1) continue;
      const key = line.slice(0, index).trim();
      let value = line.slice(index + 1).trim();
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      parsed[key] = value;
    }
    return parsed;
  } catch (error) {
    if (error?.code === "ENOENT") return {};
    throw error;
  }
}

function readValue(env, dotenv, key, fallback = undefined) {
  const value = env[key] ?? dotenv[key] ?? fallback;
  if (typeof value !== "string") return value;
  const trimmed = value.trim();
  return trimmed.length === 0 ? undefined : trimmed;
}

function requiredString(env, dotenv, key) {
  const value = readValue(env, dotenv, key);
  assert(typeof value === "string" && value.length > 0, `${key} is required.`);
  return value;
}

function optionalString(env, dotenv, key, fallback = undefined) {
  return readValue(env, dotenv, key, fallback);
}

function validateNetwork(value) {
  assert(value === TESTNET_NETWORK, `AMITY_NETWORK must be ${TESTNET_NETWORK}.`);
  return value;
}

function validateAssetId(value) {
  assert(/^[a-fA-F0-9]{64}$/.test(value), "AMITY_ASSET_ID must be a 64-character hex Taproot Asset ID.");
  return value.toLowerCase();
}

function validateUrl(value, label) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`Configuration error: ${label} must be an absolute HTTP(S) URL.`);
  }
  assert(["http:", "https:"].includes(parsed.protocol), `${label} must be an absolute HTTP(S) URL.`);
  assert(!parsed.username && !parsed.password, `${label} must not embed credentials.`);
  return parsed.toString();
}

function validateHostPort(value, label) {
  assert(/^[^:\s]+:\d{2,5}$/.test(value), `${label} must look like host:port.`);
  return value;
}

export async function readAmityConfig(env = process.env) {
  const dotenv = await loadDotEnv(join(AMITY_ROOT, ".env"));

  const network = validateNetwork(optionalString(env, dotenv, "AMITY_NETWORK", TESTNET_NETWORK));
  const assetId = validateAssetId(requiredString(env, dotenv, "AMITY_ASSET_ID"));
  const universeUrl = validateUrl(requiredString(env, dotenv, "AMITY_UNIVERSE_URL"), "AMITY_UNIVERSE_URL");
  const assetName = optionalString(env, dotenv, "AMITY_ASSET_NAME", "AMITY Test Asset");
  const unitName = optionalString(env, dotenv, "AMITY_UNIT_NAME", "AMITY");
  const litdRpcHost = validateHostPort(optionalString(env, dotenv, "LITD_RPC_HOST", "127.0.0.1:8443"), "LITD_RPC_HOST");
  const tapdRpcHost = validateHostPort(optionalString(env, dotenv, "TAPD_RPC_HOST", "127.0.0.1:10029"), "TAPD_RPC_HOST");
  const litdTlsCertPath = optionalString(env, dotenv, "LITD_TLS_CERT_PATH");
  const litdMacaroonPath = optionalString(env, dotenv, "LITD_MACAROON_PATH");
  const tapdTlsCertPath = optionalString(env, dotenv, "TAPD_TLS_CERT_PATH");
  const tapdMacaroonPath = optionalString(env, dotenv, "TAPD_MACAROON_PATH");

  return Object.freeze({
    network,
    assetId,
    universeUrl,
    assetName,
    unitName,
    litdRpcHost,
    tapdRpcHost,
    litdTlsCertPath,
    litdMacaroonPath,
    tapdTlsCertPath,
    tapdMacaroonPath,
  });
}

export async function inspectOperatorFiles(config) {
  const fileChecks = await Promise.all(
    [
      ["LITD_TLS_CERT_PATH", config.litdTlsCertPath],
      ["LITD_MACAROON_PATH", config.litdMacaroonPath],
      ["TAPD_TLS_CERT_PATH", config.tapdTlsCertPath],
      ["TAPD_MACAROON_PATH", config.tapdMacaroonPath],
    ].map(async ([label, filePath]) => {
      if (!filePath) {
        return { label, configured: false, exists: false, filePath: null };
      }
      try {
        const info = await stat(filePath);
        return { label, configured: true, exists: info.isFile(), filePath };
      } catch {
        return { label, configured: true, exists: false, filePath };
      }
    }),
  );

  return Object.freeze(fileChecks);
}

export async function amityEnvExists() {
  try {
    await access(join(AMITY_ROOT, ".env"));
    return true;
  } catch {
    return false;
  }
}
