import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const thisDirectory = dirname(fileURLToPath(import.meta.url));

export const AMITY_ROOT = join(thisDirectory, "..");
export const DEPLOYMENTS_DIRECTORY = join(AMITY_ROOT, "deployments");
export const DEFAULT_MANIFEST_PATH = join(DEPLOYMENTS_DIRECTORY, "amity-testnet.json");

export const TESTNET_NETWORK = "testnet";
export const AMITY_MANIFEST_VERSION = 1;
export const AMITY_HOLDER_PROOF_PREAMBLE = "OMEGA AMITY TESTNET PILOT - RELEASE-DAY NOVEL UNLOCK";
export const AMITY_HOLDER_PROOF_PURPOSE = "Verify current AMITY testnet Taproot Asset holdings for release-day novel unlock.";
export const AMITY_HOLDER_PROOF_MAX_MESSAGE_BYTES = 1232;
