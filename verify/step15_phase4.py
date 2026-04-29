"""
Phase 4 P1 验证脚本（自动采纳审计日志 + 回滚）
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.event_bus import Event, EventType
from src.memory.skill_distiller import SkillDistiller


class FakeMemoryManager:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._skills_dir = data_dir / "skills"
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self.skills: dict[str, dict] = {}

    def save_skill(
        self,
        skill_name: str,
        description: str,
        steps: list[str],
        tags: list[str] | None = None,
    ) -> None:
        self.skills[skill_name] = {
            "name": skill_name,
            "description": description,
            "steps": steps,
            "tags": tags or [],
        }

    def load_skill(self, skill_name: str) -> dict | None:
        return self.skills.get(skill_name)

    def delete_skill(self, skill_name: str) -> bool:
        if skill_name not in self.skills:
            return False
        self.skills.pop(skill_name, None)
        return True


def _write_trace(traces_dir: Path, payload: dict) -> Path:
    trace_file = traces_dir / "audit.trace.json"
    trace_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return trace_file


def test_adoption_log_written_for_auto_adopt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        traces_dir = base / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_file = _write_trace(
            traces_dir,
            {
                "goal": "自动采纳并写审计",
                "success": True,
                "total_tool_calls": 5,
                "steps": [
                    {"tool_calls": [{"tool_name": "list_dir"}], "tool_results": [{"tool_name": "list_dir", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "read_file"}], "tool_results": [{"tool_name": "read_file", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "edit_file"}], "tool_results": [{"tool_name": "edit_file", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "success"}]},
                ],
            },
        )

        manager = FakeMemoryManager(base)
        fake_tracer = SimpleNamespace(
            _traces_dir=traces_dir,
            list_traces=lambda limit=1: [{"file": trace_file.name}],
        )
        distiller = SkillDistiller(
            manager,
            tracer=fake_tracer,
            auto_subscribe=False,
            min_tool_calls=2,
            auto_adopt_threshold=0.70,
        )
        distiller._on_agent_completed(Event(
            event_type=EventType.AGENT_COMPLETED,
            data={"goal": "自动采纳并写审计", "success": True},
            source="test",
        ))

        records = distiller.get_recent_adoption_records(limit=5)
        assert len(records) == 1
        assert records[0].source == "auto"
        assert records[0].rolled_back is False
        assert len(manager.skills) == 1


def test_rollback_last_auto_adopt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        traces_dir = base / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_file = _write_trace(
            traces_dir,
            {
                "goal": "自动采纳回滚案例",
                "success": True,
                "total_tool_calls": 5,
                "steps": [
                    {"tool_calls": [{"tool_name": "list_dir"}], "tool_results": [{"tool_name": "list_dir", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "read_file"}], "tool_results": [{"tool_name": "read_file", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "edit_file"}], "tool_results": [{"tool_name": "edit_file", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "success"}]},
                ],
            },
        )

        manager = FakeMemoryManager(base)
        fake_tracer = SimpleNamespace(
            _traces_dir=traces_dir,
            list_traces=lambda limit=1: [{"file": trace_file.name}],
        )
        distiller = SkillDistiller(
            manager,
            tracer=fake_tracer,
            auto_subscribe=False,
            min_tool_calls=2,
            auto_adopt_threshold=0.70,
        )
        distiller._on_agent_completed(Event(
            event_type=EventType.AGENT_COMPLETED,
            data={"goal": "自动采纳回滚案例", "success": True},
            source="test",
        ))

        assert len(manager.skills) == 1
        assert len(distiller.get_recent_drafts()) == 0

        result = distiller.rollback_last_auto_adopt()
        assert result is not None
        assert len(manager.skills) == 0
        assert len(distiller.get_recent_drafts()) == 1

        records = distiller.get_recent_adoption_records(limit=5)
        assert records[-1].rolled_back is True


def main() -> None:
    test_adoption_log_written_for_auto_adopt()
    print("[PASS] test_adoption_log_written_for_auto_adopt")
    test_rollback_last_auto_adopt()
    print("[PASS] test_rollback_last_auto_adopt")
    print("[DONE] Phase 4 P1 adoption audit and rollback checks passed")


if __name__ == "__main__":
    main()
