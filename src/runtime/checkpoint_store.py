"""
src/runtime/checkpoint_store.py - 会话断点存储
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CheckpointRecord:
    session_id: str
    goal: str
    context: str
    status: str
    updated_at: str
    completed_subgoals: list[str] = field(default_factory=list)
    total_subgoals: int = 0
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    final_answer: str = ""
    resumable_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckpointStore:
    """把会话执行状态持久化到 ~/.hermes-cn/checkpoints"""

    def __init__(self, checkpoints_dir: Path | None = None) -> None:
        base = Path.home() / ".hermes-cn" / "checkpoints"
        self._dir = (checkpoints_dir or base).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: CheckpointRecord) -> Path:
        record.updated_at = datetime.now().isoformat()
        target = self._dir / f"{record.session_id}.json"
        target.write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    def load(self, session_id: str) -> CheckpointRecord | None:
        target = self._dir / f"{session_id}.json"
        if not target.exists():
            return None
        data = json.loads(target.read_text(encoding="utf-8"))
        return CheckpointRecord(**data)

    def list_recent(self, limit: int = 20) -> list[CheckpointRecord]:
        files = sorted(
            self._dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]
        records: list[CheckpointRecord] = []
        for file in files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                records.append(CheckpointRecord(**data))
            except Exception:
                continue
        return records
