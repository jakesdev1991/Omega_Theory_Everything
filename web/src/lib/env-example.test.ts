// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

/**
 * `.env.example` is the only place a reader learns which variables this app
 * understands, and it had drifted: eleven variables the code reads were not in
 * it (the whole AMITY rail, the two litd/tapd RPC hosts, the NIP-05 name map,
 * and the wallet bundle's port). A comment asking people to remember is not a
 * guard, so this test walks the source, collects every environment read, and
 * fails if one is not documented.
 *
 * It understands the three ways this codebase reads the environment:
 *   1. process.env.NAME / process.env["NAME"]
 *   2. `envKey: "NAME"` fields in descriptor tables (amity-server.ts)
 *   3. an array of "NAME" strings consumed by `.some(key => process.env[key])`
 * A new pattern would need a collector here — visible in the diff, which is the
 * point.
 */

// Supplied by Next.js itself, never configured by hand.
const FRAMEWORK_VARS = new Set(["NODE_ENV"]);

function findWebRoot(start: string): string {
  let dir = start;
  for (let depth = 0; depth < 6; depth += 1) {
    if (existsSync(join(dir, ".env.example")) && existsSync(join(dir, "src", "lib"))) {
      return dir;
    }
    dir = dirname(dir);
  }
  throw new Error(`could not find the web root above ${start}`);
}

function sourceFiles(root: string): string[] {
  const found: string[] = [];
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(path);
      } else if (/\.(ts|tsx|js|mjs)$/.test(entry.name)) {
        found.push(path);
      }
    }
  };
  walk(root);
  return found;
}

function envReadsIn(source: string): string[] {
  const names = new Set<string>();
  for (const match of source.matchAll(/process\.env\.([A-Z][A-Z0-9_]*)/g)) {
    names.add(match[1]);
  }
  for (const match of source.matchAll(/process\.env\[\s*["']([A-Z][A-Z0-9_]*)["']\s*\]/g)) {
    names.add(match[1]);
  }
  for (const match of source.matchAll(/envKey:\s*"([A-Z][A-Z0-9_]*)"/g)) {
    names.add(match[1]);
  }
  // No `s` flag: it needs an es2018 target and the app compiles below that.
  // `[^\]]` already matches newlines, so this still spans a multi-line list.
  const listConsumedByIndexing = /\[([^\]]*?)\]\s*\.some\(\s*\(?[\w$]+\)?\s*=>\s*isNonEmptyString\(\s*process\.env\[/g;
  for (const match of source.matchAll(listConsumedByIndexing)) {
    for (const name of match[1].matchAll(/"([A-Z][A-Z0-9_]*)"/g)) {
      names.add(name[1]);
    }
  }
  return [...names];
}

test("every environment variable the app reads is documented in .env.example", () => {
  const webRoot = findWebRoot(process.cwd());
  const example = readFileSync(join(webRoot, ".env.example"), "utf8");
  const documented = new Set<string>();
  for (const match of example.matchAll(/^#?\s*([A-Z][A-Z0-9_]*)=/gm)) {
    documented.add(match[1]);
  }

  const reads = new Map<string, string[]>();
  for (const dir of ["src", "scripts"]) {
    const root = join(webRoot, dir);
    if (!existsSync(root)) continue;
    for (const file of sourceFiles(root)) {
      // This file's own fixtures name variables that are deliberately not
      // documented anywhere else.
      if (file.endsWith("env-example.test.ts")) continue;
      for (const name of envReadsIn(readFileSync(file, "utf8"))) {
        reads.set(name, [...(reads.get(name) ?? []), file.slice(webRoot.length + 1)]);
      }
    }
  }

  assert.ok(reads.size > 10, `expected to find environment reads, found ${reads.size}`);

  const undocumented = [...reads.keys()]
    .filter((name) => !documented.has(name) && !FRAMEWORK_VARS.has(name))
    .sort();
  assert.deepEqual(
    undocumented,
    [],
    `these variables are read but missing from web/.env.example: ${undocumented.join(", ")}`,
  );
});

test("the collector sees all three read styles", () => {
  // A guard that silently stops matching would pass forever, so pin the shapes.
  const found = envReadsIn(
    'process.env.ALPHA_ONE; process.env["BETA_TWO"]; { envKey: "GAMMA_THREE" };\n' +
      'const configured = [\n' +
      '  "DELTA_FOUR",\n' +
      '  "EPSILON_FIVE",\n' +
      '].some((key) => isNonEmptyString(process.env[key]));',
  );
  assert.deepEqual(found.sort(), [
    "ALPHA_ONE",
    "BETA_TWO",
    "DELTA_FOUR",
    "EPSILON_FIVE",
    "GAMMA_THREE",
  ]);
});
