# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import enum
import json
import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SourceType(enum.Enum):
    LEXICAL = "LEXICAL"
    DENSE = "DENSE"
    GRAPH = "GRAPH"


@dataclass
class RetrievalResult:
    doc_id: str
    content: str
    score: float
    source: SourceType
    metadata: Dict[str, Any]


class RetrievalEngine:
    def __init__(
        self,
        fts5_db_path: str,
        dense_enabled: bool = False,
        graph_enabled: bool = False,
    ):
        self.fts5_db_path = fts5_db_path
        self.dense_enabled = dense_enabled
        self.graph_enabled = graph_enabled
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.fts5_db_path) as conn:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
                    doc_id, content, metadata
                )
            """)

    def index_document(
        self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        meta_str = json.dumps(metadata) if metadata else "{}"
        with sqlite3.connect(self.fts5_db_path) as conn:
            conn.execute(
                """
                INSERT INTO docs(doc_id, content, metadata) VALUES(?, ?, ?)
            """,
                (doc_id, content, meta_str),
            )

    def search(self, query: str, limit: int = 10) -> List[RetrievalResult]:
        return self._lexical_search(query, limit)

    def hybrid_search(self, query: str, limit: int = 10) -> List[RetrievalResult]:
        results = self._lexical_search(query, limit)
        if self.dense_enabled:
            dense_results = self._dense_search(query, limit)
            results.extend(dense_results)
        return results[:limit]

    def _lexical_search(self, query: str, limit: int) -> List[RetrievalResult]:
        """FTS5-safe lexical search.

        FTS5 MATCH syntax treats ':' and other punctuation as operators and
        bare words as implicit AND. Natural-language queries with colons
        ('lookup: what is...') raise 'no such column'. Strategy:
          1. sanitize to bare tokens (strip FTS5 operators)
          2. try implicit AND over tokens
          3. fall back to token-OR
          4. fall back to individual longest tokens
        Never raises on adversarial input — returns [] instead.
        """
        import re as _re

        # strip FTS5 special chars; keep word chars, collapse whitespace
        tokens = [t for t in _re.split(r"[^A-Za-z0-9_]+", query) if len(t) >= 2]
        if not tokens:
            return []
        and_q = " ".join(f'"{t}"' for t in tokens)
        or_q = " OR ".join(f'"{t}"' for t in tokens)
        attempts = [and_q, or_q] + [
            f'"{t}"' for t in sorted(tokens, key=len, reverse=True)[:3]
        ]
        with sqlite3.connect(self.fts5_db_path) as conn:
            for q in attempts:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT doc_id, content, metadata, rank
                        FROM docs
                        WHERE docs MATCH ?
                        ORDER BY rank
                        LIMIT ?
                    """,
                        (q, limit),
                    )
                    rows = cursor.fetchall()
                except sqlite3.OperationalError:
                    continue
                if rows:
                    return [
                        RetrievalResult(
                            doc_id=row[0],
                            content=row[1],
                            score=row[3],
                            source=SourceType.LEXICAL,
                            metadata=json.loads(row[2]),
                        )
                        for row in rows
                    ]
        return []

    def _dense_search(self, query: str, limit: int) -> List[RetrievalResult]:
        raise NotImplementedError(
            "Dense retrieval requires Qdrant benchmark validation per §32"
        )

    def _graph_search(self, query: str, limit: int) -> List[RetrievalResult]:
        raise NotImplementedError(
            "Graph retrieval gated per §70: must beat FTS5+dense on recall, utility, false positives, cost, latency, poisoning robustness"
        )
