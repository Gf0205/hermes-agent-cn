from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class RecallEvent:
    timestamp: str
    query: str
    limit: int
    top_ids: list[str]
    useful_at_1: float
    useful_at_3: float
    component_diversity_at_3: float
    weights: dict[str, float]
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "limit": self.limit,
            "top_ids": self.top_ids,
            "metrics": {
                "useful_at_1": self.useful_at_1,
                "useful_at_3": self.useful_at_3,
                "component_diversity_at_3": self.component_diversity_at_3,
            },
            "weights": self.weights,
            "extra": self.extra,
        }


class RecallLogger:
    def __init__(self, log_path: Optional[str] = None) -> None:
        base_dir = os.getenv("AGENT_DATA_DIR", str(Path.home() / ".hermes-cn"))
        default_path = Path(base_dir) / "recall" / "recall_logs.jsonl"
        self._path = Path(log_path or default_path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: RecallEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8", errors="replace", newline="\n") as f:
            f.write(line + "\n")

    def read_last(self, n: int = 50) -> list[dict[str, Any]]:
        n = max(0, int(n))
        if n <= 0:
            return []
        if not self._path.exists():
            return []

        # Simple & robust: read all then tail. Logs are expected small locally.
        lines = self._path.read_text(encoding="utf-8", errors="replace").splitlines()
        out: list[dict[str, Any]] = []
        for raw in lines[-n:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except Exception:
                continue
        return out


def now_iso() -> str:
    return datetime.now().isoformat()

