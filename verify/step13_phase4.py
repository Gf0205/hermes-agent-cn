"""
Phase 4 P1 验证脚本（技能草稿持久化）
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


def _make_distill_trace(traces_dir: Path, filename: str = "demo.trace.json") -> Path:
    trace_file = traces_dir / filename
    trace_payload = {
        "goal": "自动生成技能案例",
        "success": True,
        "total_tool_calls": 3,
        "steps": [
            {"tool_calls": [{"tool_name": "read_file"}], "tool_results": [{"tool_name": "read_file", "status": "success"}]},
            {"tool_calls": [{"tool_name": "edit_file"}], "tool_results": [{"tool_name": "edit_file", "status": "success"}]},
            {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "success"}]},
        ],
    }
    trace_file.write_text(json.dumps(trace_payload, ensure_ascii=False), encoding="utf-8")
    return trace_file


def test_draft_persist_and_reload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        traces_dir = base / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_file = _make_distill_trace(traces_dir)

        manager = FakeMemoryManager(base)
        fake_tracer = SimpleNamespace(
            _traces_dir=traces_dir,
            list_traces=lambda limit=1: [{"file": trace_file.name}],
        )
        distiller = SkillDistiller(manager, tracer=fake_tracer, auto_subscribe=False, min_tool_calls=2)
        distiller._on_agent_completed(Event(
            event_type=EventType.AGENT_COMPLETED,
            data={"goal": "自动生成技能案例", "success": True},
            source="test",
        ))

        assert len(distiller.get_recent_drafts()) == 1

        # 新实例应能加载已有草稿
        distiller2 = SkillDistiller(manager, tracer=fake_tracer, auto_subscribe=False, min_tool_calls=2)
        drafts = distiller2.get_recent_drafts()
        assert len(drafts) == 1
        assert drafts[0].name.startswith("auto_")


def test_adopt_will_remove_draft() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        traces_dir = base / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_file = _make_distill_trace(traces_dir, filename="adopt.trace.json")

        manager = FakeMemoryManager(base)
        fake_tracer = SimpleNamespace(
            _traces_dir=traces_dir,
            list_traces=lambda limit=1: [{"file": trace_file.name}],
        )
        distiller = SkillDistiller(manager, tracer=fake_tracer, auto_subscribe=False, min_tool_calls=2)
        distiller._on_agent_completed(Event(
            event_type=EventType.AGENT_COMPLETED,
            data={"goal": "自动生成技能案例", "success": True},
            source="test",
        ))
        assert len(distiller.get_recent_drafts()) == 1

        skill_name = distiller.adopt_draft(0)
        assert skill_name in manager.skills
        assert len(distiller.get_recent_drafts()) == 0


def main() -> None:
    test_draft_persist_and_reload()
    print("[PASS] test_draft_persist_and_reload")
    test_adopt_will_remove_draft()
    print("[PASS] test_adopt_will_remove_draft")
    print("[DONE] Phase 4 P1 draft persistence checks passed")


if __name__ == "__main__":
    main()
