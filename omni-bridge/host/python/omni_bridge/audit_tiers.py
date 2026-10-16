# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import List


class AuditVerdict(Enum):
    PASS = "PASS"
    FLAGGED_FOR_HUMAN = "FLAGGED_FOR_HUMAN"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"


@dataclass
class ScrutinyReport:
    trace_id: str
    verdict: AuditVerdict
    findings: List[str]
    timestamp: float


class Tier2Scrutiny:
    """Tier 2 Audit (Scrutiny) per §36. Runs asynchronously on recorded traces."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tier2_scrutiny (trace_id TEXT PRIMARY KEY, record_json TEXT)"
            )

    def analyze_trace(
        self, trace_id: str, ledger_records: List[dict]
    ) -> ScrutinyReport:
        findings = []
        verdict = AuditVerdict.PASS
        for r in ledger_records:
            if r.get("budget_used", 0) > 1000:
                findings.append(f"Excessive budget used: {r['budget_used']}")
                verdict = AuditVerdict.FLAGGED_FOR_HUMAN
        report = ScrutinyReport(trace_id, verdict, findings, time.time())
        self._save(report)
        return report

    def _save(self, report: ScrutinyReport):
        data = asdict(report)
        data["verdict"] = data["verdict"].value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tier2_scrutiny VALUES (?, ?)",
                (report.trace_id, json.dumps(data)),
            )


class Tier3MetaScrutiny:
    """Tier 3 Audit (Meta-Scrutiny) per §36. Looks for systemic drift across multiple traces."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS tier3_meta (audit_id TEXT PRIMARY KEY, record_json TEXT)"
            )

    def run_meta_audit(
        self, audit_id: str, scrutiny_reports: List[ScrutinyReport]
    ) -> dict:
        flagged_count = sum(
            1 for r in scrutiny_reports if r.verdict != AuditVerdict.PASS
        )
        drift_score = flagged_count / max(len(scrutiny_reports), 1)
        result = {
            "audit_id": audit_id,
            "traces_analyzed": len(scrutiny_reports),
            "flagged_count": flagged_count,
            "drift_score": drift_score,
            "systemic_warning": drift_score > 0.1,
        }
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tier3_meta VALUES (?, ?)",
                (audit_id, json.dumps(result)),
            )
        return result
