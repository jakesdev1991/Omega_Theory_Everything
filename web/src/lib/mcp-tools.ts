// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

export interface McpToolGroup {
  plane: string;
  tools: readonly string[];
}

/**
 * Human-readable tool names shown by the website. Keep this list aligned with
 * mcp/server.json; mcp-tools.test.ts fails if the public inventory drifts.
 */
export const MCP_TOOL_GROUPS: readonly McpToolGroup[] = [
  { plane: "SOV", tools: ["sov_mint", "sov_burn", "sov_transfer", "sov_snapshot"] },
  { plane: "USE", tools: ["use_issue", "use_revoke", "use_snapshot"] },
  { plane: "CARE", tools: ["care_submit", "care_attest", "care_appeal", "care_snapshot"] },
  { plane: "AMITY", tools: ["amity_entitle", "amity_transfer", "amity_snapshot"] },
  { plane: "OMEGA", tools: ["omega_mint", "omega_lock", "omega_proposal", "omega_vote", "omega_snapshot"] },
  { plane: "HOUSE", tools: ["ledger", "events_since", "full_event_log"] },
];

export const MCP_TOOL_COUNT = MCP_TOOL_GROUPS.reduce(
  (total, group) => total + group.tools.length,
  0,
);
