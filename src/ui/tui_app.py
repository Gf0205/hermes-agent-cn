"""
src/ui/tui_app.py - 轻量 TUI 事件面板
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from src.event_bus import Event, EventType, get_event_bus


@dataclass
class TUIState:
    current_state: str = "idle"
    active_subgoal: str = "-"
    last_tool: str = "-"
    last_tool_status: str = "-"
    total_events: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    plan_subgoal_count: int = 0
    subgoal_status: dict[str, str] = field(default_factory=dict)
    subgoal_desc: dict[str, str] = field(default_factory=dict)
    recent_tool_timings: deque[tuple[str, float, str]] = field(
        default_factory=lambda: deque(maxlen=30)
    )
    recent_events: deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def on_event(self, event: Event) -> None:
        self.total_events += 1
        self.recent_events.append(
            f"{event.timestamp.strftime('%H:%M:%S')} {event.event_type.value}"
        )

        if event.event_type == EventType.STATE_CHANGED:
            self.current_state = str(event.data.get("to_state", self.current_state))
        elif event.event_type == EventType.PLAN_CREATED:
            self.plan_subgoal_count = int(event.data.get("sub_goal_count", 0))
            self.subgoal_status.clear()
            self.subgoal_desc.clear()
        elif event.event_type == EventType.SUBGOAL_STARTED:
            subgoal_id = str(event.data.get("sub_goal_id", "unknown"))
            self.subgoal_status[subgoal_id] = "in_progress"
            self.subgoal_desc[subgoal_id] = str(event.data.get("description", ""))[:60]
            self.active_subgoal = str(event.data.get("description", "-"))[:80]
        elif event.event_type == EventType.SUBGOAL_COMPLETED:
            subgoal_id = str(event.data.get("sub_goal_id", "unknown"))
            self.subgoal_status[subgoal_id] = "completed"
        elif event.event_type == EventType.SUBGOAL_FAILED:
            subgoal_id = str(event.data.get("sub_goal_id", "unknown"))
            self.subgoal_status[subgoal_id] = "failed"
        elif event.event_type == EventType.TOOL_CALLED:
            self.last_tool = str(event.data.get("tool_name", "-"))
        elif event.event_type == EventType.TOOL_RESULT:
            self.last_tool_status = str(event.data.get("status", "-"))
            tool_name = str(event.data.get("tool_name", "-"))
            elapsed_ms = float(event.data.get("execution_time_ms", 0.0))
            status = str(event.data.get("status", "-"))
            self.recent_tool_timings.append((tool_name, elapsed_ms, status))
        elif event.event_type == EventType.LLM_RESPONSE:
            self.total_tokens += int(event.data.get("tokens", 0))
            self.total_cost_usd += float(event.data.get("estimated_cost_usd", 0.0))

    def render(self) -> RenderableType:
        summary = Table(show_header=False, box=None, pad_edge=False)
        summary.add_column("k", style="bold cyan", width=16)
        summary.add_column("v", style="white", width=68)
        summary.add_row("当前状态", self.current_state)
        summary.add_row("活跃子目标", self.active_subgoal)
        summary.add_row("最近工具", f"{self.last_tool} ({self.last_tool_status})")
        summary.add_row("累计事件", str(self.total_events))
        summary.add_row("累计Tokens", str(self.total_tokens))
        summary.add_row("累计费用", f"${self.total_cost_usd:.6f}")

        plan_table = Table(show_header=True, header_style="bold green")
        plan_table.add_column("子目标", style="white", width=24)
        plan_table.add_column("状态", style="cyan", width=14)
        if self.subgoal_status:
            for subgoal_id, status in list(self.subgoal_status.items())[:8]:
                desc = self.subgoal_desc.get(subgoal_id, "")
                label = f"{subgoal_id}: {desc}" if desc else subgoal_id
                plan_table.add_row(label[:24], status)
        else:
            total_hint = f"预计子目标数: {self.plan_subgoal_count}" if self.plan_subgoal_count else "暂无计划"
            plan_table.add_row(total_hint, "-")

        events = Table(show_header=True, header_style="bold magenta")
        events.add_column("最近事件", style="dim")
        for item in list(self.recent_events)[-8:]:
            events.add_row(item)
        if not self.recent_events:
            events.add_row("暂无事件")

        perf_table = Table(show_header=True, header_style="bold yellow")
        perf_table.add_column("工具", style="white", width=20)
        perf_table.add_column("耗时(ms)", style="yellow", width=12)
        perf_table.add_column("状态", style="cyan", width=10)
        if self.recent_tool_timings:
            slowest = sorted(
                list(self.recent_tool_timings),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            for tool_name, elapsed_ms, status in slowest:
                perf_table.add_row(tool_name[:20], f"{elapsed_ms:.1f}", status[:10])
        else:
            perf_table.add_row("暂无工具调用", "-", "-")

        return Group(
            Panel(summary, title="📡 实时状态", border_style="cyan"),
            Panel(plan_table, title="🗺️ 计划树（简版）", border_style="green"),
            Panel(perf_table, title="⏱️ 工具耗时 Top5（近窗口）", border_style="yellow"),
            Panel(events, title="🧾 事件流", border_style="magenta"),
        )


class TUIApp:
    """订阅事件总线并实时渲染状态面板"""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()
        self._bus = get_event_bus()
        self._state = TUIState()
        self._live: Live | None = None
        self._handlers: list[tuple[EventType, Callable[[Event], None]]] = []

    def start(self) -> None:
        if self._live is not None:
            return
        self._register_handlers()
        self._live = Live(
            self._state.render(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        self._unregister_handlers()
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _on_event(self, event: Event) -> None:
        self._state.on_event(event)
        if self._live is not None:
            self._live.update(self._state.render(), refresh=True)

    def _register_handlers(self) -> None:
        watch_types = [
            EventType.STATE_CHANGED,
            EventType.PLAN_CREATED,
            EventType.SUBGOAL_STARTED,
            EventType.SUBGOAL_COMPLETED,
            EventType.SUBGOAL_FAILED,
            EventType.TOOL_CALLED,
            EventType.TOOL_RESULT,
            EventType.LLM_RESPONSE,
        ]
        for event_type in watch_types:
            self._bus.subscribe(event_type, self._on_event)
            self._handlers.append((event_type, self._on_event))

    def _unregister_handlers(self) -> None:
        for event_type, handler in self._handlers:
            self._bus.unsubscribe(event_type, handler)
        self._handlers.clear()

    def __enter__(self) -> "TUIApp":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
