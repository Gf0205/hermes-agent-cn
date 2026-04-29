"""
Phase 4 P1 验证脚本（草稿质量评分 + 自动采纳阈值）
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


def _write_trace(traces_dir: Path, filename: str, payload: dict) -> Path:
    trace_file = traces_dir / filename
    trace_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return trace_file


def test_quality_score_discriminates_trace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        manager = FakeMemoryManager(base)
        distiller = SkillDistiller(manager, auto_subscribe=False, min_tool_calls=2, auto_adopt_threshold=0.0)

        high = distiller.distill_from_trace_data(
            {
                "goal": "高质量流程",
                "success": True,
                "total_tool_calls": 5,
                "steps": [
                    {"tool_calls": [{"tool_name": "list_dir"}, {"tool_name": "read_file"}], "tool_results": [{"tool_name": "read_file", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "edit_file"}], "tool_results": [{"tool_name": "edit_file", "status": "success"}]},
                    {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "success"}]},
                ],
            },
            goal_hint="高质量流程",
        )
        low = distiller.distill_from_trace_data(
            {
                "goal": "低质量流程",
                "success": True,
                "total_tool_calls": 2,
                "steps": [
                    {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "failure", "error": "timeout"}]},
                    {"tool_calls": [{"tool_name": "shell"}], "tool_results": [{"tool_name": "shell", "status": "failure", "error": "permission denied"}]},
                ],
            },
            goal_hint="低质量流程",
        )
        assert high is not None
        assert low is not None
        assert 0.0 <= high.quality_score <= 1.0
        assert 0.0 <= low.quality_score <= 1.0
        assert high.quality_score > low.quality_score


def test_auto_adopt_by_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        traces_dir = base / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        trace_file = _write_trace(
            traces_dir,
            "auto_adopt.trace.json",
            {
                "goal": "自动采纳技能案例",
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
            data={"goal": "自动采纳技能案例", "success": True},
            source="test",
        ))

        # 达到阈值后应自动采纳为正式技能，并清空草稿池
        assert len(manager.skills) == 1
        assert len(distiller.get_recent_drafts()) == 0


def main() -> None:
    test_quality_score_discriminates_trace()
    print("[PASS] test_quality_score_discriminates_trace")
    test_auto_adopt_by_threshold()
    print("[PASS] test_auto_adopt_by_threshold")
    print("[DONE] Phase 4 P1 quality score and auto-adopt checks passed")


if __name__ == "__main__":
    main()
