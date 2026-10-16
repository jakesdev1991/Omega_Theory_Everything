# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import sqlite3
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional


class ReplayClassification(Enum):
    EXACT_REPLAY = "EXACT_REPLAY"
    DETERMINISTIC_REPLAY = "DETERMINISTIC_REPLAY"
    SIMULATED_COUNTERFACTUAL = "SIMULATED_COUNTERFACTUAL"
    APPROXIMATE_COUNTERFACTUAL = "APPROXIMATE_COUNTERFACTUAL"
    NON_REPLAYABLE = "NON_REPLAYABLE"


@dataclass
class ReplayRecord:
    replay_id: str
    original_trace_id: str
    classification: ReplayClassification
    original_timestamp: float
    replay_timestamp: float
    inputs_hash: str
    outputs_hash: str
    match_result: str
    divergence_detail: Optional[str]
    metadata: dict


class ReplayEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS replays (replay_id TEXT PRIMARY KEY, trace_id TEXT, record_json TEXT)"
            )

    def classify(
        self,
        trace_id: str,
        is_deterministic: bool,
        has_all_inputs: bool,
        involves_llm: bool,
    ) -> ReplayClassification:
        """CRITICAL RULE: Never label a nondeterministic simulation as an exact replay (§35)"""
        if is_deterministic and has_all_inputs:
            return (
                ReplayClassification.EXACT_REPLAY
                if not involves_llm
                else ReplayClassification.DETERMINISTIC_REPLAY
            )
        if involves_llm:
            return ReplayClassification.APPROXIMATE_COUNTERFACTUAL
        return ReplayClassification.SIMULATED_COUNTERFACTUAL

    def record_replay(self, record: ReplayRecord) -> ReplayRecord:
        data = asdict(record)
        data["classification"] = data["classification"].value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO replays (replay_id, trace_id, record_json) VALUES (?, ?, ?)",
                (record.replay_id, record.original_trace_id, json.dumps(data)),
            )
        return record

    def get_replays_for_trace(self, trace_id: str) -> List[ReplayRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT record_json FROM replays WHERE trace_id = ?", (trace_id,)
            )
            res = []
            for row in cur.fetchall():
                d = json.loads(row[0])
                d["classification"] = ReplayClassification(d["classification"])
                res.append(ReplayRecord(**d))
            return res

    def verify_replay(self, replay_id: str) -> tuple[bool, str]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT record_json FROM replays WHERE replay_id = ?", (replay_id,)
            )
            row = cur.fetchone()
            if not row:
                return False, "Not found"
            d = json.loads(row[0])
            if d["inputs_hash"] == d["outputs_hash"]:
                return True, "Matches"
            return False, "Diverged"
