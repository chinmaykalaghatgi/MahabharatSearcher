"""Dogfood capture — append-only query/feedback log.

The unified `mbh` front door logs every query here so real usage becomes
the next eval set, instead of hand-picked canaries we've already tuned
against. Two record types, both append-only (no rewrites, so it's safe to
tail / concurrent-write):

  - ``{"type": "query", "id", "ts", "query", "mode", "kind", ...}``
  - ``{"type": "feedback", "ref_id", "ts", "flag", "note"}``

Harvest later by joining feedback.ref_id -> query.id; flagged queries are
the priority candidates for a curated eval set.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from uuid import uuid4


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class QueryLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_query(self, fields: dict) -> str:
        """Append a query record; return its id (for later feedback)."""
        qid = uuid4().hex[:12]
        self._append({"type": "query", "id": qid, "ts": _now(), **fields})
        return qid

    def log_feedback(self, ref_id: str, note: str, *, flag: str = "note") -> None:
        self._append(
            {
                "type": "feedback",
                "ref_id": ref_id,
                "ts": _now(),
                "flag": flag,
                "note": note,
            }
        )

    def _append(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
