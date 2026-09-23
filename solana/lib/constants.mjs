import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const thisDirectory = dirname(fileURLToPath(import.meta.url));

export const SOLANA_ROOT = join(thisDirectory, "..");
export const DEPLOYMENTS_DIRECTORY = join(SOLANA_ROOT, "deployments");
export const DEFAULT_MANIFEST_PATH = join(DEPLOYMENTS_DIRECTORY, "twc-devnet.json");

export const DEVNET_CLUSTER = "devnet";
export const DEVNET_DEFAULT_RPC_URL = "https://api.devnet.solana.com";
export const DEVNET_GENESIS_HASH = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG";

// This is intentionally a pilot identity. The canonical, future-mainnet identity
// remains Token of the World Citizen (TWC), subject to legal/trademark clearance.
export const CANONICAL_TOKEN_NAME = "Token of the World Citizen";
export const CANONICAL_TOKEN_SYMBOL = "TWC";
export const PILOT_TOKEN_NAME = "Token of the World Citizen";
export const PILOT_TOKEN_SYMBOL = "tTWC";
export const PILOT_TOKEN_DECIMALS = 9;
export const DEFAULT_INITIAL_SUPPLY_TOKENS = "1000000000";
export const MAX_U64 = (1n << 64n) - 1n;

export const MIN_DEPLOYER_BALANCE_LAMPORTS = 50_000_000n; // 0.05 Devnet SOL
export const MAX_METADATA_BYTES = 262_144;
export const MAX_METADATA_URI_BYTES = 200;
export const MANIFEST_VERSION = 1;

export const DEPLOYMENT_CONFIRMATION_VALUE = "DEVNET_TWC_PILOT";

export function devnetAddressExplorerUrl(address) {
  return `https://explorer.solana.com/address/${address}?cluster=${DEVNET_CLUSTER}`;
}

export function devnetTransactionExplorerUrl(signature) {
  return `https://explorer.solana.com/tx/${signature}?cluster=${DEVNET_CLUSTER}`;
}
