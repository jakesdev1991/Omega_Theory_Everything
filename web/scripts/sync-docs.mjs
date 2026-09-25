#!/usr/bin/env node
/**
 * Syncs the C.A.R.E. Economy documents — the manifesto and the whitepapers —
 * from the repository root into the Next.js `content/docs/` tree so the website
 * can render them as first-class pages (/manifesto, /whitepapers, /whitepapers/[slug]).
 *
 * The documents stay authoritative at the repository root (`whitepapers/`, and the
 * tri-token blueprint). This script never edits them. The generated copies are
 * gitignored (`web/content/`) and rebuilt by `npm run dev`, `npm run build`,
 * and `npm run sync:docs`.
 */

import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptDir, "..");
const repoRoot = resolve(webRoot, "..");

const SOURCE_DIR = join(repoRoot, "whitepapers");
const TARGET_DIR = join(webRoot, "content", "docs");

// Whitelist: exactly the documents the site is allowed to publish.
const DOC_FILES = [
  "care_economy_whitepaper.md",
  "care_economy_manifesto.md",
  "twc_whitepaper.md",
  "omega_protocol_whitepaper.md",
  "tokamak_domain_token_whitepaper.md",
  "care_amity_protocol_whitepaper.md",
  "lucifer_hermes_omni_bridge_whitepaper.md",
];

const EXTRA_FILES = [
  { from: join(repoRoot, "tri_token_sovereign_economy_blueprint.md"), to: "tri_token_sovereign_economy_blueprint.md" },
  { from: join(repoRoot, "docs", "sovereign_economy_manifesto.md"), to: "sovereign_economy_manifesto.md" },
];

mkdirSync(TARGET_DIR, { recursive: true });

let synced = 0;
for (const name of DOC_FILES) {
  const source = join(SOURCE_DIR, name);
  try {
    copyFileSync(source, join(TARGET_DIR, name));
    synced += 1;
  } catch (error) {
    if (error.code === "ENOENT") {
      throw new Error(`C.A.R.E. Economy document is missing: whitepapers/${name}`);
    }
    throw error;
  }
}

for (const extra of EXTRA_FILES) {
  try {
    copyFileSync(extra.from, join(TARGET_DIR, extra.to));
    synced += 1;
  } catch (error) {
    if (error.code === "ENOENT") {
      throw new Error(`Referenced document is missing: ${extra.to}`);
    }
    throw error;
  }
}

console.log(`[sync-docs] Synced ${synced} C.A.R.E. Economy documents into content/docs/`);
