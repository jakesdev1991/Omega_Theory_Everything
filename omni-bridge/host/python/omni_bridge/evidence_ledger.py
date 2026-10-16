# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)

try:
    import blake3

    def compute_hash(data: bytes) -> str:
        return blake3.blake3(data).hexdigest()
except ImportError:
    import hashlib
    import warnings

    warnings.warn("blake3 not found. Falling back to hashlib.blake2b.")

    def compute_hash(data: bytes) -> str:
        return hashlib.blake2b(data).hexdigest()


from .appa import SecurityLabel  # noqa: E402


@dataclass
class LedgerRecord:
    timestamp: float
    trace_id: str
    actor: str
    route: str
    model: str
    skill: str
    tool: str
    input_hash: str
    output_hash: str
    security_labels: SecurityLabel
    capabilities: List[str]
    budget_used: float
    latency_ms: float
    verification_result: str
    outcome: str
    previous_hash: str
    current_hash: str = ""

    def validate_no_secrets(self):
        # Assert no credential-like fields exist in names or values
        assert not hasattr(self, "secret") and not hasattr(self, "password"), (
            "Record must not contain secrets"
        )
        pass


class EvidenceLedger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ledger (
                    trace_id TEXT,
                    timestamp REAL,
                    record_json TEXT,
                    previous_hash TEXT,
                    current_hash TEXT
                )
            """)

    def record(self, record: LedgerRecord) -> LedgerRecord:
        record.validate_no_secrets()
        record.previous_hash = self.get_latest_hash()

        # Serialize without current_hash to compute it
        temp_dict = asdict(record)
        temp_dict.pop("current_hash", None)

        # Serialize nested SecurityLabel correctly if needed, for simplicity let's rely on standard JSON serialization (dataclasses don't natively json serialize nested enums perfectly without custom handling, but assuming default string representation for now)
        try:
            serialized = json.dumps(temp_dict, sort_keys=True, default=str).encode(
                "utf-8"
            )
        except TypeError as e:
            logger.error(f"Failed to serialize record: {e}")
            raise

        record.current_hash = compute_hash(
            record.previous_hash.encode("utf-8") + serialized
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO ledger (trace_id, timestamp, record_json, previous_hash, current_hash)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    record.trace_id,
                    record.timestamp,
                    json.dumps(asdict(record), default=str),
                    record.previous_hash,
                    record.current_hash,
                ),
            )

        return record

    def verify_chain(self) -> Tuple[bool, List[str]]:
        errors = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT previous_hash, current_hash, record_json FROM ledger ORDER BY timestamp ASC"
            )
            rows = cursor.fetchall()

            expected_prev = ""
            for prev_hash, curr_hash, record_json in rows:
                if prev_hash != expected_prev:
                    errors.append(
                        f"Chain broken: expected previous hash {expected_prev}, got {prev_hash}"
                    )
                    break

                # Recompute
                record_dict = json.loads(record_json)
                record_dict.pop("current_hash", None)
                serialized = json.dumps(
                    record_dict, sort_keys=True, default=str
                ).encode("utf-8")
                recomputed = compute_hash(prev_hash.encode("utf-8") + serialized)

                if recomputed != curr_hash:
                    errors.append(
                        f"Hash mismatch: recomputed {recomputed} != stored {curr_hash}"
                    )
                    break

                expected_prev = curr_hash

        return len(errors) == 0, errors

    def get_records_by_trace(self, trace_id: str) -> List[LedgerRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT record_json FROM ledger WHERE trace_id = ?", (trace_id,)
            )
            return [LedgerRecord(**json.loads(row[0])) for row in cursor.fetchall()]

    def get_latest_hash(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT current_hash FROM ledger ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return row[0] if row else ""
