"""
Phase 4 P1 验证脚本（按 record_id 精确回滚）
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


def _write_trace(traces_dir: Path, name: str, goal: str) -> Path:
    payload = {
        "goal": goal,
        "success": True,
        "total_tool_calls": 5,
        "steps": [
            {"tool_calls": [{"tool_name": "list_dir"}], "tool_results": [{"tool_name": "list_dir", "status": "success"}]},
            {"tool_calls": [{"tool_name": "read_file"}], "tool_results": [{"tool_name": "read_file", "status": "success"}]},
            {"tool_calls": [{"tool_name": "edit_file"}], "tool_results": [{"tool_name": "edit_file", "status": "success"}]},
            {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "success"}]},
        ],
    }
    trace_file = traces_dir / name
    trace_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return trace_file


def test_rollback_specific_record_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        traces_dir = base / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        trace_a = _write_trace(traces_dir, "a.trace.json", "自动采纳A")
        trace_b = _write_trace(traces_dir, "b.trace.json", "自动采纳B")

        manager = FakeMemoryManager(base)
        current_trace = {"file": trace_a.name}
        fake_tracer = SimpleNamespace(
            _traces_dir=traces_dir,
            list_traces=lambda limit=1: [dict(current_trace)],
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
            data={"goal": "自动采纳A", "success": True},
            source="test",
        ))
        current_trace["file"] = trace_b.name
        distiller._on_agent_completed(Event(
            event_type=EventType.AGENT_COMPLETED,
            data={"goal": "自动采纳B", "success": True},
            source="test",
        ))

        records = distiller.get_recent_adoption_records(limit=10)
        assert len(records) == 2
        assert len(manager.skills) == 2

        target = records[0]
        other = records[1]
        rollback_result = distiller.rollback_auto_adopt(record_id=target.record_id)
        assert rollback_result is not None
        assert rollback_result["record_id"] == target.record_id

        # 仅回滚目标记录对应技能，不影响其它已采纳技能
        assert target.skill_name not in manager.skills
        assert other.skill_name in manager.skills

        refreshed = distiller.get_recent_adoption_records(limit=10)
        rolled_target = [r for r in refreshed if r.record_id == target.record_id][0]
        rolled_other = [r for r in refreshed if r.record_id == other.record_id][0]
        assert rolled_target.rolled_back is True
        assert rolled_other.rolled_back is False


def main() -> None:
    test_rollback_specific_record_id()
    print("[PASS] test_rollback_specific_record_id")
    print("[DONE] Phase 4 P1 targeted rollback checks passed")


if __name__ == "__main__":
    main()
