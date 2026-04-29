"""
Phase 3 P1 验证脚本（TUI 骨架）
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.event_bus import Event, EventType
from src.ui.tui_app import TUIState


def test_tui_state_event_updates() -> None:
    state = TUIState()
    state.on_event(Event(event_type=EventType.PLAN_CREATED, data={"sub_goal_count": 3}))
    state.on_event(Event(event_type=EventType.STATE_CHANGED, data={"to_state": "executing"}))
    state.on_event(Event(
        event_type=EventType.SUBGOAL_STARTED,
        data={"sub_goal_id": "sg_1", "description": "创建登录接口"},
    ))
    state.on_event(Event(event_type=EventType.SUBGOAL_COMPLETED, data={"sub_goal_id": "sg_1"}))
    state.on_event(Event(event_type=EventType.TOOL_CALLED, data={"tool_name": "read_file"}))
    state.on_event(Event(event_type=EventType.TOOL_RESULT, data={"status": "success"}))
    state.on_event(Event(
        event_type=EventType.LLM_RESPONSE,
        data={"tokens": 123, "estimated_cost_usd": 0.0012},
    ))
    state.on_event(Event(
        event_type=EventType.TOOL_RESULT,
        data={"tool_name": "grep_search", "status": "success", "execution_time_ms": 220.5},
    ))
    state.on_event(Event(
        event_type=EventType.TOOL_RESULT,
        data={"tool_name": "read_file", "status": "success", "execution_time_ms": 50.0},
    ))

    assert state.current_state == "executing"
    assert state.active_subgoal.startswith("创建登录接口")
    assert state.last_tool == "read_file"
    assert state.last_tool_status == "success"
    assert state.total_tokens == 123
    assert abs(state.total_cost_usd - 0.0012) < 1e-9
    assert state.plan_subgoal_count == 3
    assert state.subgoal_status["sg_1"] == "completed"
    assert len(state.recent_tool_timings) == 3
    assert state.recent_tool_timings[-2][0] == "grep_search"
    assert state.total_events == 9


def test_tui_render_smoke() -> None:
    state = TUIState()
    state.on_event(Event(event_type=EventType.STATE_CHANGED, data={"to_state": "planning"}))
    rendered = state.render()
    assert rendered is not None


def main() -> None:
    test_tui_state_event_updates()
    print("[PASS] test_tui_state_event_updates")
    test_tui_render_smoke()
    print("[PASS] test_tui_render_smoke")
    print("[DONE] Phase 3 P1 TUI 骨架验证通过")


if __name__ == "__main__":
    main()
