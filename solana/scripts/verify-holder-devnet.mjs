import { readFile } from "node:fs/promises";

import { DEFAULT_MANIFEST_PATH } from "../lib/constants.mjs";
import { readVerificationConfig } from "../lib/config.mjs";
import { verifyHolderUnlock } from "../lib/gate.mjs";
import { readManifest } from "../lib/manifest.mjs";

function usage() {
  console.error(
    "Usage: npm run verify:holder -- /path/to/proof.json [--manifest /path/to/twc-devnet.json] [--origin https://example.com] [--max-age-minutes 15]",
  );
}

function parseArguments(argv) {
  const options = {
    manifestPath: DEFAULT_MANIFEST_PATH,
    proofPath: undefined,
    expectedOrigin: undefined,
    maxAgeMs: undefined,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    switch (value) {
      case "--manifest":
        options.manifestPath = argv[++index];
        break;
      case "--origin":
        options.expectedOrigin = argv[++index];
        break;
      case "--max-age-minutes": {
        const rawMinutes = argv[++index];
        const minutes = Number(rawMinutes);
        if (!Number.isFinite(minutes) || minutes <= 0) {
          throw new Error("--max-age-minutes must be a positive number.");
        }
        options.maxAgeMs = minutes * 60_000;
        break;
      }
      case "-h":
      case "--help":
        usage();
        process.exit(0);
        break;
      default:
        if (value.startsWith("--")) {
          throw new Error(`Unknown option: ${value}`);
        }
        if (options.proofPath) {
          throw new Error(`Unexpected extra positional argument: ${value}`);
        }
        options.proofPath = value;
        break;
    }
  }

  if (!options.proofPath) {
    usage();
    throw new Error("A holder proof JSON file is required.");
  }

  if (!options.manifestPath) {
    throw new Error("A deployment manifest path is required.");
  }

  return options;
}

async function readJson(path, label) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new Error(`Unable to read ${label} at ${path}: ${error.message}`);
  }
}

try {
  const options = parseArguments(process.argv.slice(2));
  const config = readVerificationConfig();
  const manifest = await readManifest(options.manifestPath);
  const proof = await readJson(options.proofPath, "holder proof");

  const result = await verifyHolderUnlock({
    rpcUrl: config.rpcUrl,
    manifest,
    proof,
    expectedOrigin: options.expectedOrigin,
    maxAgeMs: options.maxAgeMs,
  });

  console.log(
    JSON.stringify(
      {
        ...result,
        manifestPath: options.manifestPath,
        proofPath: options.proofPath,
        rpcUrl: config.rpcUrl,
      },
      null,
      2,
    ),
  );
} catch (error) {
  console.error(`Holder verification failed: ${error.message}`);
  process.exitCode = 1;
}
