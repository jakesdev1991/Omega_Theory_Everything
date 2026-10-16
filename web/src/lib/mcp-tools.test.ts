// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import { MCP_TOOL_COUNT, MCP_TOOL_GROUPS } from "./mcp-tools";

interface McpManifest {
  tools: Array<{ name: string }>;
}

const manifestPath = join(process.cwd(), "..", "mcp", "server.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as McpManifest;

test("website MCP tool inventory matches the server manifest", () => {
  const displayedNames = MCP_TOOL_GROUPS.flatMap((group) => group.tools).sort();
  const serverNames = manifest.tools.map((tool) => tool.name).sort();

  assert.equal(MCP_TOOL_COUNT, 22);
  assert.equal(displayedNames.length, new Set(displayedNames).size, "displayed tool names must be unique");
  assert.deepEqual(displayedNames, serverNames);
});

test("MCP website lists the five planes and the separate house tools", () => {
  assert.deepEqual(
    MCP_TOOL_GROUPS.map((group) => group.plane),
    ["SOV", "USE", "CARE", "AMITY", "OMEGA", "HOUSE"],
  );
});
