# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import enum
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillState(enum.Enum):
    DISCOVERED = "DISCOVERED"
    CANDIDATE = "CANDIDATE"
    TESTING = "TESTING"
    AUDITED = "AUDITED"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    PROMOTED = "PROMOTED"
    MONITORED = "MONITORED"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


VALID_TRANSITIONS = {
    SkillState.DISCOVERED: [SkillState.CANDIDATE, SkillState.RETIRED],
    SkillState.CANDIDATE: [SkillState.TESTING, SkillState.RETIRED],
    SkillState.TESTING: [SkillState.AUDITED, SkillState.RETIRED],
    SkillState.AUDITED: [SkillState.SHADOW, SkillState.RETIRED],
    SkillState.SHADOW: [SkillState.CANARY, SkillState.RETIRED],
    SkillState.CANARY: [SkillState.PROMOTED, SkillState.RETIRED],
    SkillState.PROMOTED: [
        SkillState.MONITORED,
        SkillState.DEPRECATED,
        SkillState.RETIRED,
    ],
    SkillState.MONITORED: [SkillState.DEPRECATED, SkillState.RETIRED],
    SkillState.DEPRECATED: [SkillState.RETIRED],
    SkillState.RETIRED: [],
}


@dataclass
class SkillRecord:
    skill_id: str
    name: str
    version: str
    state: SkillState
    created_at: float
    updated_at: float
    author: str
    test_results: Dict[str, Any]
    audit_log: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SkillLifecycleManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id TEXT PRIMARY KEY,
                    record_json TEXT
                )
            """)

    def register_skill(
        self, skill_id: str, name: str, version: str, author: str
    ) -> SkillRecord:
        now = datetime.utcnow().timestamp()
        record = SkillRecord(
            skill_id=skill_id,
            name=name,
            version=version,
            state=SkillState.DISCOVERED,
            created_at=now,
            updated_at=now,
            author=author,
            test_results={},
            audit_log=[
                {"transition_to": SkillState.DISCOVERED.value, "timestamp": now}
            ],
            metadata={},
        )
        self._save_record(record)
        return record

    def transition(
        self,
        skill_id: str,
        target_state: SkillState,
        reason: str,
        evidence: Optional[Dict] = None,
    ) -> SkillRecord:
        record = self._get_record(skill_id)
        if not record:
            raise ValueError(f"Skill {skill_id} not found")

        if target_state not in VALID_TRANSITIONS[record.state]:
            raise ValueError(
                f"Illegal transition from {record.state} to {target_state}"
            )

        record.state = target_state
        record.updated_at = datetime.utcnow().timestamp()

        audit_entry = {
            "transition_to": target_state.value,
            "timestamp": record.updated_at,
            "reason": reason,
            "evidence": evidence or {},
        }
        record.audit_log.append(audit_entry)

        self._save_record(record)
        logger.info(
            f"Auditable transition: Skill {skill_id} -> {target_state.value} ({reason})"
        )
        return record

    def get_eligible_skills(
        self, min_state: SkillState = SkillState.PROMOTED
    ) -> List[SkillRecord]:
        # For simplistic implementation, just returning PROMOTED or MONITORED if min_state is PROMOTED.
        # Can be enhanced to define proper hierarchy
        eligible_states = {SkillState.PROMOTED, SkillState.MONITORED}
        records = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT record_json FROM skills")
            for row in cursor.fetchall():
                data = json.loads(row[0])
                data["state"] = SkillState(data["state"])
                rec = SkillRecord(**data)
                if rec.state in eligible_states:
                    records.append(rec)
        return records

    def retire_skill(self, skill_id: str, reason: str) -> SkillRecord:
        return self.transition(skill_id, SkillState.RETIRED, reason)

    def get_audit_trail(self, skill_id: str) -> List[Dict[str, Any]]:
        record = self._get_record(skill_id)
        if not record:
            return []
        return record.audit_log

    def _get_record(self, skill_id: str) -> Optional[SkillRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT record_json FROM skills WHERE skill_id = ?", (skill_id,)
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                data["state"] = SkillState(data["state"])
                return SkillRecord(**data)
        return None

    def _save_record(self, record: SkillRecord):
        data = asdict(record)
        data["state"] = data["state"].value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO skills (skill_id, record_json)
                VALUES (?, ?)
            """,
                (record.skill_id, json.dumps(data)),
            )
