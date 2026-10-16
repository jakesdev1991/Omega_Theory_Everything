#!/usr/bin/env node
// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Publishes the repository-level ecosystem map as a static website asset.
 * The authoritative SVG stays in ../docs; this generated copy is ignored.
 */

import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(scriptDir, "..");
const repoRoot = resolve(webRoot, "..");
const source = join(repoRoot, "docs", "ecosystem-map.svg");
const target = join(webRoot, "public", "ecosystem-map.svg");

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log("[sync-ecosystem-map] Published docs/ecosystem-map.svg as /ecosystem-map.svg");
