# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple


@dataclass
class ToolManifest:
    tool_id: str
    version: str
    input_schema: dict
    output_schema: dict
    required_capabilities: List[str]
    security_requirements: dict
    network_requirements: List[str]
    filesystem_requirements: List[str]
    resource_limits: dict
    timeout_seconds: int
    deterministic: bool
    side_effects: List[str]
    rollback_supported: bool
    verification_requirements: List[str]


class ToolRegistry:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tools (tool_id TEXT PRIMARY KEY, record_json TEXT)"
            )

    def register(self, manifest: ToolManifest):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tools (tool_id, record_json) VALUES (?, ?)",
                (manifest.tool_id, json.dumps(asdict(manifest))),
            )

    def get(self, tool_id: str) -> Optional[ToolManifest]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT record_json FROM tools WHERE tool_id = ?", (tool_id,))
            row = cur.fetchone()
            if not row:
                return None
            return ToolManifest(**json.loads(row[0]))

    def list_tools(self) -> List[ToolManifest]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT record_json FROM tools")
            return [ToolManifest(**json.loads(r[0])) for r in cur.fetchall()]

    def validate_invocation(
        self, tool_id: str, input_data: dict
    ) -> Tuple[bool, List[str]]:
        m = self.get(tool_id)
        if not m:
            return False, ["Tool not found"]
        return True, []

    def unregister(self, tool_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tools WHERE tool_id = ?", (tool_id,))


class ToolExecutionPipeline:
    def __init__(self, registry: ToolRegistry, appa_engine, sandbox_selector=None):
        self.registry = registry
        self.appa_engine = appa_engine
        self.sandbox_selector = sandbox_selector

    def execute(self, tool_id: str, input_data: dict, agent_context: dict) -> dict:
        valid, errs = self.registry.validate_invocation(tool_id, input_data)
        if not valid:
            return {"status": "error", "reason": "schema_invalid", "errors": errs}
        return {"status": "success", "tool_id": tool_id, "result": "mock_execution"}
