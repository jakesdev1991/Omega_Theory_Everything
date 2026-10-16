# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Optional


class TraceStage(Enum):
    INPUT = "INPUT"
    CLASSIFICATION = "CLASSIFICATION"
    HERMES_RETRIEVAL = "HERMES_RETRIEVAL"
    LUCIFER_EXPLORATION = "LUCIFER_EXPLORATION"
    ROUTE_GENERATION = "ROUTE_GENERATION"
    ROUTE_SELECTION = "ROUTE_SELECTION"
    TOOL_INVOCATION = "TOOL_INVOCATION"
    SANDBOX = "SANDBOX"
    VERIFICATION = "VERIFICATION"
    AUDIT = "AUDIT"
    OUTCOME = "OUTCOME"
    EVOLUTION = "EVOLUTION"


@dataclass
class TraceSpan:
    span_id: str
    trace_id: str
    stage: TraceStage
    start_time: float
    end_time: Optional[float]
    parent_span_id: Optional[str]
    attributes: dict
    status: str


class TraceCorrelator:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS spans (span_id TEXT PRIMARY KEY, trace_id TEXT, start_time REAL, record_json TEXT)"
            )

    def start_trace(self, trace_id: str | None = None) -> str:
        return trace_id or str(uuid.uuid4())

    def start_span(
        self,
        trace_id: str,
        stage: TraceStage,
        parent_span_id: str | None = None,
        attributes: dict | None = None,
    ) -> TraceSpan:
        span = TraceSpan(
            str(uuid.uuid4()),
            trace_id,
            stage,
            time.time(),
            None,
            parent_span_id,
            attributes or {},
            "OK",
        )
        self._save(span)
        return span

    def end_span(
        self, span_id: str, status: str = "OK", attributes: dict | None = None
    ) -> TraceSpan:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT record_json FROM spans WHERE span_id = ?", (span_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError("Span not found")
            d = json.loads(row[0])
            d["stage"] = TraceStage(d["stage"])
            span = TraceSpan(**d)
            span.end_time = time.time()
            span.status = status
            if attributes:
                span.attributes.update(attributes)
            self._save(span)
            return span

    def _save(self, span: TraceSpan):
        d = asdict(span)
        d["stage"] = d["stage"].value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO spans (span_id, trace_id, start_time, record_json) VALUES (?, ?, ?, ?)",
                (span.span_id, span.trace_id, span.start_time, json.dumps(d)),
            )

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT record_json FROM spans WHERE trace_id = ? ORDER BY start_time ASC",
                (trace_id,),
            )
            res = []
            for row in cur.fetchall():
                d = json.loads(row[0])
                d["stage"] = TraceStage(d["stage"])
                res.append(TraceSpan(**d))
            return res

    def reconstruct_flow(self, trace_id: str) -> dict:
        spans = self.get_trace(trace_id)
        return {"trace_id": trace_id, "flow": [asdict(s) for s in spans]}
