import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { parseAndValidatePilotMetadata, sha256Hex } from "../lib/metadata.mjs";

const sourcePath = process.argv[2];
if (!sourcePath) {
  console.error("Usage: npm run metadata:hash -- path/to/twc-devnet.json");
  process.exitCode = 1;
} else {
  try {
    const bytes = await readFile(resolve(process.cwd(), sourcePath));
    const document = parseAndValidatePilotMetadata(bytes);
    console.log(JSON.stringify({
      file: sourcePath,
      sha256: sha256Hex(bytes),
      bytes: bytes.byteLength,
      name: document.name,
      symbol: document.symbol,
    }, null, 2));
  } catch (error) {
    console.error(`Metadata validation failed: ${error.message}`);
    process.exitCode = 1;
  }
}
