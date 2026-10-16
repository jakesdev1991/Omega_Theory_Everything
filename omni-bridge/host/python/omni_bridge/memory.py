# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import enum
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Similarity is not truth. Embedding distance is not proof. (per §33)
from .appa import ConfidentialityLevel, IntegrityLevel, SecurityLabel


class MemoryType(enum.Enum):
    WORKING = "WORKING"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"
    POLICY = "POLICY"
    AUDIT = "AUDIT"
    EXPERIMENTAL = "EXPERIMENTAL"


class ValidationStatus(enum.Enum):
    UNVALIDATED = "UNVALIDATED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"


@dataclass
class MemoryRecord:
    memory_id: str
    memory_type: MemoryType
    content: str
    source: str
    confidence: float
    provenance: str
    created_at: float
    security_label: SecurityLabel
    validation_status: ValidationStatus
    expiry: Optional[float]
    dependencies: List[str]


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT,
                    provenance TEXT,
                    expiry REAL,
                    record_json TEXT
                )
            """)

    def store(self, record: MemoryRecord) -> str:
        data = asdict(record)
        data["memory_type"] = data["memory_type"].value
        data["validation_status"] = data["validation_status"].value

        sl_data = asdict(record.security_label)
        sl_data["integrity"] = sl_data["integrity"].value
        sl_data["confidentiality"] = sl_data["confidentiality"].value
        data["security_label"] = sl_data

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO memory (memory_id, memory_type, provenance, expiry, record_json)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    record.memory_id,
                    record.memory_type.value,
                    record.provenance,
                    record.expiry,
                    json.dumps(data),
                ),
            )
        return record.memory_id

    def recall(
        self, memory_type: MemoryType, query: Optional[str] = None, limit: int = 10
    ) -> List[MemoryRecord]:
        records = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().timestamp()
            cursor.execute(
                """
                SELECT record_json FROM memory 
                WHERE memory_type = ? AND (expiry IS NULL OR expiry > ?)
                LIMIT ?
            """,
                (memory_type.value, now, limit),
            )
            for row in cursor.fetchall():
                records.append(self._parse_record(json.loads(row[0])))
        return records

    def validate(
        self, memory_id: str, status: ValidationStatus, reason: str
    ) -> MemoryRecord:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT record_json FROM memory WHERE memory_id = ?", (memory_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ValueError("Memory not found")

            data = json.loads(row[0])
            data["validation_status"] = status.value

            conn.execute(
                "UPDATE memory SET record_json = ? WHERE memory_id = ?",
                (json.dumps(data), memory_id),
            )
            return self._parse_record(data)

    def expire_stale(self) -> int:
        now = datetime.utcnow().timestamp()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM memory WHERE expiry IS NOT NULL AND expiry <= ?", (now,)
            )
            return cursor.rowcount

    def get_by_provenance(self, provenance_chain: str) -> List[MemoryRecord]:
        records = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT record_json FROM memory WHERE provenance = ?",
                (provenance_chain,),
            )
            for row in cursor.fetchall():
                records.append(self._parse_record(json.loads(row[0])))
        return records

    def _parse_record(self, data: Dict[str, Any]) -> MemoryRecord:
        data["memory_type"] = MemoryType(data["memory_type"])
        data["validation_status"] = ValidationStatus(data["validation_status"])

        sl_data = data["security_label"]
        sl_data["integrity"] = IntegrityLevel(sl_data["integrity"])
        sl_data["confidentiality"] = ConfidentialityLevel(sl_data["confidentiality"])
        data["security_label"] = SecurityLabel(**sl_data)

        return MemoryRecord(**data)
