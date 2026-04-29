"""
src/execution/state_machine.py - Agent状态机

修订记录：
  v1 → v2:
  - 新增 REFLECTING → EXECUTING 合法转换
    （反思完一个子目标后继续执行下一个，这是正常流转）
  - 新增 transition_if_not() 辅助方法
    （已经在目标状态时静默跳过，避免 finally 块里的 IDLE→IDLE 问题）

面试要点：
"状态机的设计原则是：
 只建模真实存在的状态流转，不遗漏也不过度限制。
 遗漏 REFLECTING→EXECUTING 是一个设计遗漏，
 因为多子目标任务的正常节奏就是：
 执行→反思→执行→反思→...→完成。
 修复方法：补全转换矩阵，而不是绕过状态机。"
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from src.event_bus import Event, EventType, get_event_bus
from src.models import AgentState

logger = logging.getLogger(__name__)


# ==============================================================================
# 状态转换矩阵（有限状态机的核心）
# ==============================================================================

VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {
        AgentState.PLANNING,
        AgentState.ERROR,
    },
    AgentState.PLANNING: {
        AgentState.EXECUTING,
        AgentState.ERROR,
        AgentState.IDLE,       # 规划失败直接回 IDLE
    },
    AgentState.EXECUTING: {
        AgentState.REFLECTING, # 子目标执行完 → 反思
        AgentState.REPLANNING, # 跳过反思直接重规划（降级路径）
        AgentState.ERROR,
        AgentState.IDLE,       # 任务全部完成 → IDLE
    },
    AgentState.REFLECTING: {
        AgentState.EXECUTING,  # ✅ v2新增：反思完继续执行下一个子目标
        AgentState.REPLANNING, # 反思后决定需要重规划
        AgentState.ERROR,
        AgentState.IDLE,       # 反思后发现任务其实完成了
    },
    AgentState.REPLANNING: {
        AgentState.EXECUTING,  # 重规划完成，继续执行
        AgentState.ERROR,
        AgentState.IDLE,       # 重规划失败放弃
    },
    AgentState.ERROR: {
        AgentState.IDLE,       # 错误恢复
    },
}

# 用于打印的中文状态名
STATE_NAMES: dict[AgentState, str] = {
    AgentState.IDLE:       "空闲",
    AgentState.PLANNING:   "规划中",
    AgentState.EXECUTING:  "执行中",
    AgentState.REFLECTING: "反思中",
    AgentState.REPLANNING: "重规划中",
    AgentState.ERROR:      "错误",
}


class StateMachine:
    """
    Agent有限状态机

    状态流转图（v2）：
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  IDLE ──► PLANNING ──► EXECUTING ◄──── REFLECTING       │
    │    ▲          │            │    │          ▲  │          │
    │    │          │            │    └──────────┘  │          │
    │    │          ▼            ▼                  │          │
    │    │        IDLE         ERROR ──► IDLE        │          │
    │    │                       │                  │          │
    │    │                       └── REPLANNING ────┘          │
    │    │                              │                       │
    │    └──────────────────────────────┘                       │
    └──────────────────────────────────────────────────────────┘

    关键修复（v1→v2）：
    - REFLECTING → EXECUTING：多子目标任务的正常节奏
    """

    def __init__(self) -> None:
        self._state: AgentState = AgentState.IDLE
        self._previous_state: Optional[AgentState] = None
        self._state_entered_at: datetime = datetime.now()
        self._transition_count: int = 0
        self._bus = get_event_bus()

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def previous_state(self) -> Optional[AgentState]:
        return self._previous_state

    @property
    def time_in_current_state_ms(self) -> float:
        delta = datetime.now() - self._state_entered_at
        return delta.total_seconds() * 1000

    def transition(self, new_state: AgentState, reason: str = "") -> None:
        """
        状态转换（严格模式：非法转换直接抛异常）

        Raises:
            ValueError: 转换不在 VALID_TRANSITIONS 中
        """
        allowed = VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise ValueError(
                f"非法状态转换: {self._state.value} → {new_state.value}\n"
                f"从 {self._state.value} 只能转到: "
                f"{[s.value for s in sorted(allowed, key=lambda s: s.value)]}"
            )
        self._do_transition(new_state, reason)

    def transition_if_not(self, new_state: AgentState, reason: str = "") -> bool:
        """
        如果当前状态不是目标状态，则执行转换；否则静默跳过。

        面试要点：
        "这个方法解决了 finally 块中的幂等问题。
         finally 里需要确保回到 IDLE，但不知道当前是不是已经 IDLE 了
        （force_idle 可能已经被调用过）。
         用 transition_if_not(IDLE) 而不是 transition(IDLE)，
         已经是 IDLE 就直接返回 False，不报错也不重复发事件。
         这类似于 HTTP 的幂等性设计——多次调用结果一致。"

        Returns:
            True  = 执行了转换
            False = 已经是目标状态，跳过
        """
        if self._state == new_state:
            logger.debug(f"transition_if_not: 已是 {new_state.value}，跳过")
            return False
        self.transition(new_state, reason)
        return True

    def can_transition(self, new_state: AgentState) -> bool:
        """检查转换是否合法（不执行，只检查）"""
        return new_state in VALID_TRANSITIONS.get(self._state, set())

    def force_idle(self, reason: str = "强制重置") -> None:
        """
        强制重置到 IDLE（紧急中断/异常恢复用）

        面试要点：
        "force_idle 是状态机的'紧急出口'。
         当程序捕获到未预期异常时，不管当前状态是什么，
         都需要能回到干净的 IDLE 状态，让下一次任务可以正常启动。
         这比让程序卡死在 EXECUTING 状态要好得多。"
        """
        if self._state == AgentState.IDLE:
            return  # 已经是 IDLE，无需操作

        old_state = self._state
        self._state = AgentState.IDLE
        self._previous_state = old_state
        self._state_entered_at = datetime.now()

        logger.warning(
            f"⚡ 强制重置到IDLE | 来自: {old_state.value} | 原因: {reason}"
        )

        self._bus.publish(Event(
            event_type=EventType.STATE_CHANGED,
            data={
                "from_state": old_state.value,
                "to_state":   AgentState.IDLE.value,
                "reason":     f"[强制] {reason}",
                "forced":     True,
            },
            source="state_machine"
        ))

    def _do_transition(self, new_state: AgentState, reason: str) -> None:
        """执行实际的状态转换（内部方法）"""
        old_state = self._state
        self._previous_state = old_state
        self._state = new_state
        self._state_entered_at = datetime.now()
        self._transition_count += 1

        old_name = STATE_NAMES.get(old_state, old_state.value)
        new_name = STATE_NAMES.get(new_state, new_state.value)
        logger.debug(
            f"状态: {old_name} → {new_name}"
            + (f" | {reason}" if reason else "")
        )

        self._bus.publish(Event(
            event_type=EventType.STATE_CHANGED,
            data={
                "from_state":       old_state.value,
                "to_state":         new_state.value,
                "reason":           reason,
                "transition_count": self._transition_count,
            },
            source="state_machine"
        ))

    def is_idle(self) -> bool:
        return self._state == AgentState.IDLE

    def is_busy(self) -> bool:
        return self._state not in (AgentState.IDLE, AgentState.ERROR)

    def get_info(self) -> dict:
        return {
            "current_state":   self._state.value,
            "current_name":    STATE_NAMES.get(self._state, ""),
            "previous_state":  self._previous_state.value if self._previous_state else None,
            "time_in_state_ms": round(self.time_in_current_state_ms, 1),
            "total_transitions": self._transition_count,
        }