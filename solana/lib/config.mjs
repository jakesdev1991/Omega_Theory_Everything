import { stat, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";

import dotenv from "dotenv";
import { address, createKeyPairSignerFromBytes } from "@solana/kit";

import {
  DEFAULT_INITIAL_SUPPLY_TOKENS,
  DEVNET_DEFAULT_RPC_URL,
  MAX_METADATA_URI_BYTES,
  MAX_U64,
  PILOT_TOKEN_DECIMALS,
  SOLANA_ROOT,
} from "./constants.mjs";

// A local .env is deliberately optional. The committed .env.example contains no
// deployer key, treasury, metadata URI, or confirmation value.
dotenv.config({ path: join(SOLANA_ROOT, ".env"), quiet: true });

function fail(message) {
  throw new Error(`Configuration error: ${message}`);
}

function requiredString(env, key) {
  const value = env[key];
  if (typeof value !== "string" || value.trim().length === 0) {
    fail(`${key} is required.`);
  }
  return value.trim();
}

function optionalString(env, key, fallback) {
  const value = env[key];
  if (value === undefined || value === "") {
    return fallback;
  }
  if (typeof value !== "string") {
    fail(`${key} must be a string.`);
  }
  return value.trim();
}

function assertHttpUrl(value, label) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    fail(`${label} must be an absolute HTTP(S) URL.`);
  }

  if (!["http:", "https:"].includes(parsed.protocol) || !parsed.hostname) {
    fail(`${label} must be an absolute HTTP(S) URL.`);
  }
  if (parsed.username || parsed.password) {
    fail(`${label} must not embed credentials.`);
  }
  return parsed;
}

function assertWsUrl(value, label) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    fail(`${label} must be an absolute WS(S) URL.`);
  }

  if (!["ws:", "wss:"].includes(parsed.protocol) || !parsed.hostname) {
    fail(`${label} must be an absolute WS(S) URL.`);
  }
  if (parsed.username || parsed.password) {
    fail(`${label} must not embed credentials.`);
  }
  return parsed;
}

export function deriveWebsocketUrl(rpcUrl) {
  const parsed = assertHttpUrl(rpcUrl, "SOLANA_DEVNET_RPC_URL");
  parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  return parsed.toString();
}

export function parseWholeTokenSupply(raw, decimals = PILOT_TOKEN_DECIMALS) {
  if (!Number.isInteger(decimals) || decimals < 0 || decimals > 255) {
    throw new Error("Token decimals must be an integer between 0 and 255.");
  }
  if (typeof raw !== "string" || !/^[1-9][0-9]*$/.test(raw)) {
    throw new Error("TWC_INITIAL_SUPPLY must be a positive, base-10 whole-token integer without separators or decimals.");
  }

  const wholeTokens = BigInt(raw);
  const baseUnits = wholeTokens * 10n ** BigInt(decimals);
  if (baseUnits > MAX_U64) {
    throw new Error("TWC_INITIAL_SUPPLY exceeds the SPL Token u64 supply limit after applying decimals.");
  }
  return { wholeTokens, baseUnits };
}

export function validateMetadataUri(value) {
  const uri = typeof value === "string" ? value.trim() : "";
  let parsed;
  try {
    parsed = new URL(uri);
  } catch {
    throw new Error("TWC_METADATA_URI must be an absolute HTTPS URL.");
  }

  if (parsed.protocol !== "https:" || !parsed.hostname) {
    throw new Error("TWC_METADATA_URI must be an absolute HTTPS URL.");
  }
  if (parsed.username || parsed.password) {
    throw new Error("TWC_METADATA_URI must not embed credentials.");
  }
  if (
    parsed.hostname === "localhost" ||
    parsed.hostname.endsWith(".localhost") ||
    parsed.hostname.endsWith(".invalid") ||
    parsed.hostname.endsWith(".example") ||
    /^127\./.test(parsed.hostname) ||
    parsed.hostname === "::1"
  ) {
    throw new Error("TWC_METADATA_URI must resolve to a public, non-placeholder HTTPS host.");
  }
  if (Buffer.byteLength(uri, "utf8") > MAX_METADATA_URI_BYTES) {
    throw new Error(`TWC_METADATA_URI must be at most ${MAX_METADATA_URI_BYTES} UTF-8 bytes for Metaplex metadata.`);
  }
  return uri;
}

export function validateSha256(value) {
  const hash = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (!/^[a-f0-9]{64}$/.test(hash)) {
    throw new Error("TWC_METADATA_SHA256 must be a lowercase or uppercase 64-character SHA-256 hex digest.");
  }
  return hash;
}

export function validateAddress(value, label) {
  try {
    return address(value);
  } catch (error) {
    throw new Error(`${label} must be a valid base58 Solana address: ${error.message}`);
  }
}

function expandHomePath(value) {
  if (value === "~") return homedir();
  if (value.startsWith("~/")) return join(homedir(), value.slice(2));
  return value;
}

export function resolveKeypairPath(value) {
  const expanded = expandHomePath(value);
  return isAbsolute(expanded) ? expanded : resolve(SOLANA_ROOT, expanded);
}

export function readDeploymentConfig(env = process.env) {
  const rpcUrl = optionalString(env, "SOLANA_DEVNET_RPC_URL", DEVNET_DEFAULT_RPC_URL);
  assertHttpUrl(rpcUrl, "SOLANA_DEVNET_RPC_URL");

  const configuredWsUrl = optionalString(env, "SOLANA_DEVNET_WS_URL", "");
  const wsUrl = configuredWsUrl ? configuredWsUrl : deriveWebsocketUrl(rpcUrl);
  assertWsUrl(wsUrl, "SOLANA_DEVNET_WS_URL");

  const deployerKeypairPath = resolveKeypairPath(requiredString(env, "SOLANA_DEPLOYER_KEYPAIR_PATH"));
  const treasuryAddress = validateAddress(requiredString(env, "TWC_TREASURY_ADDRESS"), "TWC_TREASURY_ADDRESS");
  const metadataUri = validateMetadataUri(requiredString(env, "TWC_METADATA_URI"));
  const metadataSha256 = validateSha256(requiredString(env, "TWC_METADATA_SHA256"));
  const supply = parseWholeTokenSupply(
    optionalString(env, "TWC_INITIAL_SUPPLY", DEFAULT_INITIAL_SUPPLY_TOKENS),
  );

  return Object.freeze({
    rpcUrl,
    wsUrl,
    deployerKeypairPath,
    treasuryAddress,
    metadataUri,
    metadataSha256,
    initialSupplyTokens: supply.wholeTokens,
    initialSupplyBaseUnits: supply.baseUnits,
  });
}

export function readVerificationConfig(env = process.env) {
  const rpcUrl = optionalString(env, "SOLANA_DEVNET_RPC_URL", DEVNET_DEFAULT_RPC_URL);
  assertHttpUrl(rpcUrl, "SOLANA_DEVNET_RPC_URL");
  return Object.freeze({ rpcUrl });
}

export async function loadKeypairSigner(keypairPath) {
  let fileStat;
  try {
    fileStat = await stat(keypairPath);
  } catch {
    throw new Error(`Deployer keypair file does not exist or is unreadable: ${keypairPath}`);
  }
  if (!fileStat.isFile()) {
    throw new Error(`Deployer keypair path is not a file: ${keypairPath}`);
  }

  // A Solana CLI keypair grants transaction-signing authority. Refuse common
  // group/world-readable modes rather than normalizing insecure permissions.
  if (process.platform !== "win32" && (fileStat.mode & 0o077) !== 0) {
    throw new Error(`Deployer keypair file must be owner-only (chmod 600): ${keypairPath}`);
  }

  let parsed;
  try {
    parsed = JSON.parse(await readFile(keypairPath, "utf8"));
  } catch {
    throw new Error("Deployer keypair file must contain a JSON array of 64 byte values.");
  }
  if (
    !Array.isArray(parsed) ||
    parsed.length !== 64 ||
    parsed.some((value) => !Number.isInteger(value) || value < 0 || value > 255)
  ) {
    throw new Error("Deployer keypair file must contain exactly 64 integer byte values in the range 0-255.");
  }

  return createKeyPairSignerFromBytes(Uint8Array.from(parsed));
}

export function deploymentConfigForManifest(config) {
  return {
    rpcUrl: config.rpcUrl,
    treasuryAddress: config.treasuryAddress,
    metadataUri: config.metadataUri,
    metadataSha256: config.metadataSha256,
    initialSupplyTokens: config.initialSupplyTokens.toString(),
    initialSupplyBaseUnits: config.initialSupplyBaseUnits.toString(),
  };
}
