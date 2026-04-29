"""
Phase 4 P0 验证脚本（Skill Distiller）
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
    def __init__(self) -> None:
        self.skills: dict[str, dict] = {}

    def save_skill(self, skill_name: str, description: str, steps: list[str], tags: list[str] | None = None) -> None:
        self.skills[skill_name] = {
            "name": skill_name,
            "description": description,
            "steps": steps,
            "tags": tags or [],
        }

    def load_skill(self, skill_name: str) -> dict | None:
        return self.skills.get(skill_name)


def test_distill_from_trace_success() -> None:
    manager = FakeMemoryManager()
    distiller = SkillDistiller(manager, auto_subscribe=False, min_tool_calls=2)
    trace_data = {
        "goal": "更新配置并验证",
        "success": True,
        "total_tool_calls": 4,
        "steps": [
            {"tool_calls": [{"tool_name": "read_file"}], "tool_results": [{"tool_name": "read_file", "status": "success"}]},
            {"tool_calls": [{"tool_name": "edit_file"}], "tool_results": [{"tool_name": "edit_file", "status": "success"}]},
            {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "failure", "error": "timeout"}]},
        ],
    }
    draft = distiller.distill_from_trace_data(trace_data, goal_hint="更新配置并验证")
    assert draft is not None
    assert draft.draft_id
    assert "auto_" in draft.name
    assert any("read_file" in step for step in draft.steps)
    assert any("失败" in step for step in draft.steps)


def test_distill_skip_small_trace() -> None:
    manager = FakeMemoryManager()
    distiller = SkillDistiller(manager, auto_subscribe=False, min_tool_calls=3)
    trace_data = {
        "goal": "简单任务",
        "success": True,
        "total_tool_calls": 1,
        "steps": [
            {"tool_calls": [{"tool_name": "read_file"}], "tool_results": [{"tool_name": "read_file", "status": "success"}]},
        ],
    }
    draft = distiller.distill_from_trace_data(trace_data, goal_hint="简单任务")
    assert draft is None


def test_on_agent_completed_persists_skill() -> None:
    manager = FakeMemoryManager()
    with tempfile.TemporaryDirectory() as tmp:
        traces_dir = Path(tmp)
        trace_file = traces_dir / "demo.trace.json"
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
        drafts = distiller.get_recent_drafts()
        assert len(drafts) == 1
        saved_name = distiller.adopt_draft(0)
        assert saved_name in manager.skills
        saved = manager.skills[saved_name]
        assert "自动生成技能案例" in saved["description"]


def main() -> None:
    test_distill_from_trace_success()
    print("[PASS] test_distill_from_trace_success")
    test_distill_skip_small_trace()
    print("[PASS] test_distill_skip_small_trace")
    test_on_agent_completed_persists_skill()
    print("[PASS] test_on_agent_completed_persists_skill")
    print("[DONE] Phase 4 P0 Skill Distiller checks passed")


if __name__ == "__main__":
    main()
