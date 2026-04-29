"""面试要点：
"事件总线是我解耦组件的核心机制。
 规划器不直接调用可观测性模块，
 而是发布事件；可观测性模块订阅事件。
 好处：
 1. 组件间零依赖，可以独立测试
 2. 新增功能（如Web Dashboard）只需订阅事件，不改现有代码
 3. 这是经典的Observer模式，也是微服务事件驱动架构的简化版"

设计参考：Python的logging模块也用了类似的Handler机制
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
# ==============================================================================
# 事件类型枚举
# ==============================================================================

class EventType(str, Enum):
    """所有Agent事件类型"""

    # Agent生命周期事件
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_ERROR = "agent.error"

    # 状态变更事件
    STATE_CHANGED = "state.changed"

    # 规划事件
    PLAN_CREATED = "plan.created"
    PLAN_UPDATED = "plan.updated"
    SUBGOAL_STARTED = "subgoal.started"
    SUBGOAL_COMPLETED = "subgoal.completed"
    SUBGOAL_FAILED = "subgoal.failed"

    # 执行事件
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"

    # LLM事件
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_STREAM_CHUNK = "llm.stream_chunk"

    # 反思事件
    REFLECTION_STARTED = "reflection.started"
    REFLECTION_COMPLETED = "reflection.completed"
    REPLAN_TRIGGERED = "replan.triggered"

    # 记忆事件
    MEMORY_SAVED = "memory.saved"
    MEMORY_RETRIEVED = "memory.retrieved"

    # 成本事件
    TOKEN_USED = "token.used"

# ==============================================================================
# 事件数据类
# ==============================================================================
@dataclass
class Event:
    """
    标准事件对象

    设计决策：
    用dataclass而不是dict，原因是：
    - 有类型提示，IDE可以自动补全
    - 访问属性比dict["key"]更安全
    - __repr__自动生成，便于调试
    """
    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # 事件来源组件名称

    def __repr__(self) -> str:
        return (f"Event({self.event_type.value}, "
                f"from={self.source}, "
                f"at={self.timestamp.strftime('%H:%M:%S.%f')[:-3]})")


# ==============================================================================
# 事件总线核心实现
# ==============================================================================


# 处理函数类型别名，提升可读性
EventHandler = Callable[[Event], None]


class EventBus:
    """
    同步事件总线 - 发布-订阅模式实现

    选择同步而非异步的原因：
    1. 代码更易读和调试
    2. 避免AsyncIO在Windows上的兼容性问题
    3. 可观测性模块通常是"旁观者"，不应影响主流程速度
    4. 如果将来需要异步，只需把publish改为async即可

    用法示例：
        bus = EventBus()

        # 订阅事件
        def on_tool_called(event: Event):
            print(f"工具被调用: {event.data['tool_name']}")

        bus.subscribe(EventType.TOOL_CALLED, on_tool_called)

        # 发布事件
        bus.publish(Event(
            event_type=EventType.TOOL_CALLED,
            data={"tool_name": "read_file.py", "args": {"path": "/tmp/test.txt"}},
            source="executor"
        ))
    """

    def __init__(self) -> None:
        # 订阅者注册表：事件类型 → 处理函数列表
        self._handlers: dict[EventType, list[EventHandler]] = {}
        # 全局处理器（订阅所有事件）
        self._global_handlers: list[EventHandler] = []
        # 事件历史（用于调试和回放）
        self._history: list[Event] = []
        self._max_history: int = 1000  # 最多保存1000条历史

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        订阅指定类型的事件

        Args:
            event_type: 要监听的事件类型
            handler: 事件处理函数，接收Event对象
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"订阅事件: {event_type.value} → {handler.__qualname__}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """订阅所有事件（用于日志记录、追踪等全局关注者）"""
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """取消订阅"""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def publish(self, event: Event) -> None:
        """
        发布事件，同步通知所有订阅者

        设计决策：
        发布方不关心订阅者的存在。
        如果没有订阅者，事件被静默丢弃（记录到history）。
        如果处理器抛出异常，我们只记录日志不传播，
        避免某个订阅者的bug影响主流程。
        """
        # 保存历史
        if len(self._history) < self._max_history:
            self._history.append(event)

        # 通知全局处理器
        for handler in self._global_handlers:
            self._safe_call(handler, event)

        # 通知类型特定处理器
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            self._safe_call(handler, event)

    def _safe_call(self, handler: EventHandler, event: Event) -> None:
        """安全调用处理器，捕获并记录异常"""
        try:
            handler(event)
        except Exception as e:
            # 订阅者的bug不应该影响主流程
            logger.error(
                f"事件处理器 {handler.__qualname__} 抛出异常: {e}",
                exc_info=True
            )

    def get_history(
            self,
            event_type: Optional[EventType] = None,
            limit: int = 100
    ) -> list[Event]:
        """
        获取事件历史（用于调试）

        Args:
            event_type: 过滤特定类型，None表示返回所有
            limit: 最多返回条数
        """
        history = self._history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]

    def clear_history(self) -> None:
        """清空历史（测试时使用）"""
        self._history.clear()
# ==============================================================================
# 全局单例
# ==============================================================================

# 模块级全局事件总线
#
# 面试要点：
# "我用模块级单例而不是依赖注入，
#  原因是：Agent的各组件天然共享同一个事件总线，
#  不需要到处传递bus实例。
#  这类似于Python的logging.getLogger()模式。"

_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """获取全局事件总线（懒加载单例）"""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """重置全局事件总线（测试时使用）"""
    global _global_bus
    _global_bus = None


