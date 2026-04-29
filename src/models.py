"""
src/models.py - 全局数据模型定义

面试要点：
"我用dataclasses而不是Pydantic，原因是：
 1. 零依赖，纯标准库
 2. 对于内部数据传输对象(DTO)，我不需要运行时校验
 3. Python 3.10+ 的dataclasses已支持slots=True，性能接近namedtuple
 4. 代码更简洁，意图更清晰"

设计原则：
- 所有跨模块传递的数据结构都在这里定义
- 使用Enum代替魔法字符串
- field(default_factory=...)避免可变默认值陷阱
"""
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ==============================================================================
# 枚举类型 - 用Enum代替魔法字符串，IDE可以提示，重构安全
# ==============================================================================

class PlanStatus(str, Enum):
    """计划/子目标状态机"""
    PENDING = "pending"          # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 执行失败
    BLOCKED = "blocked"          # 被(依赖)阻塞
    SKIPPED = "skipped"          # 已跳过（条件不满足）



class AgentState(str,Enum):
    """
       Agent状态机 - 核心状态流转
       面试要点：
       "状态机让Agent的行为完全可预测。
        每个状态只允许特定的转换，
        这防止了'Agent不知道自己在干什么'的问题。"
       状态流转：
       IDLE → PLANNING → EXECUTING → REFLECTING → IDLE (成功)
       IDLE → PLANNING → EXECUTING → REFLECTING → REPLANNING → EXECUTING (重规划)
       任意状态 → ERROR → IDLE (错误恢复)
       """
    IDLE = "idle"  # 空闲，等待用户输入
    PLANNING = "planning"  # 正在生成执行计划
    EXECUTING = "executing"  # 正在执行工具调用
    REFLECTING = "reflecting"  # 正在反思执行结果
    REPLANNING = "replanning"  # 基于反思重新规划
    ERROR = "error"  # 遇到不可恢复错误

class ToolStatus(str, Enum):
    """工具调用结果状态"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"

class ModelTier(str, Enum):
    """
    模型档位 - 用于成本优化路由

    面试要点：
    "不是所有任务都需要最强模型。
     这让Token消耗降低约30-40%。"
    """
    STRONG = "strong"  # 强模型：复杂规划、反思、推理
    FAST = "fast"      # 快模型：简单格式化、工具调用、总结


# ==============================================================================
# 工具系统数据模型
# ==============================================================================

@dataclass
class ToolParameter:
    """工具参数定义（用于生成JSON Schema）"""
    name: str
    type: str                        # "string" | "integer" | "boolean" | "array" | "object"
    description: str
    required: bool = True
    default: Any = None
    enum_values: list[str] = field(default_factory=list)  # 枚举值约束


@dataclass
class ToolResult:
    """
    工具执行结果

    面试要点：
    "工具结果不仅包含输出，还包含执行元数据。
     这些元数据对于反思模块非常重要：
     '这个工具花了多久？成功了吗？如果失败，错误是什么？'"
    """
    tool_name: str
    status: ToolStatus
    output: str                          # 工具的文本输出
    error: Optional[str] = None          # 失败时的错误信息
    execution_time_ms: float = 0.0       # 执行耗时（毫秒）
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class ToolCall:
    """LLM请求调用的工具（执行前的意图）"""
    call_id: str                          # 唯一调用ID（对应LLM的tool_call_id）
    tool_name: str
    arguments: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

# ==============================================================================
# 规划系统数据模型
# ==============================================================================

@dataclass
class SubGoal:
    """
    子目标节点 - 层级计划的基本单元

    面试要点：
    "用Tree-of-Thoughts的思想，每个子目标都有：
     1. 明确的成功标准（知道什么时候完成了）
     2. 回滚策略（失败了怎么办）
     3. 依赖关系（保证执行顺序）
     这让Agent不会在执行到一半时'迷失'。"
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    parent_id: Optional[str] = None
    status: PlanStatus = PlanStatus.PENDING

    # 依赖管理 - 只有依赖的子目标完成后，本节点才能执行
    dependencies: list[str] = field(default_factory=list)

    # 质量控制
    success_criteria: str = ""      # 明确的成功标准，让LLM知道何时算完成
    rollback_strategy: str = ""     # 失败时的回滚/降级方案

    # 执行结果
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2


@dataclass
class Plan:
    """
    执行计划 - 由StrategicPlanner生成?

    面试要点：
    "计划是Agent的'思维导图'。
     我记录计划的生成时间、消耗的Token数，
     这让我能事后分析：哪类任务规划最耗资源？"
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""                    # 原始用户目标
    sub_goals: list[SubGoal] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # 元数据（用于可观测性）
    planning_tokens_used: int = 0
    planning_time_ms: float = 0.0
    model_used: str = ""

    def get_pending_goals(self) -> list[SubGoal]:
        """获取所有待执行的子目标"""
        return [g for g in self.sub_goals if g.status == PlanStatus.PENDING]

    def get_ready_goals(self) -> list[SubGoal]:
        """
        获取可立即执行的子目标（依赖已全部满足）

        面试要点：
        "这个方法实现了简单的拓扑排序逻辑。
         只有当一个子目标的所有依赖都COMPLETED，
         它才进入'就绪'状态，可以开始执行。"
        """
        completed_ids = {
            g.id for g in self.sub_goals
            if g.status == PlanStatus.COMPLETED
        }
        return [
            g for g in self.sub_goals
            if g.status == PlanStatus.PENDING
               and all(dep in completed_ids for dep in g.dependencies)
        ]

    def is_completed(self) -> bool:
        """检查计划是否全部完成"""
        return all(g.status == PlanStatus.COMPLETED for g in self.sub_goals)

    def is_failed(self) -> bool:
        """检查计划是否有任何子目标超过重试次数"""
        return any(
            g.status == PlanStatus.FAILED and g.retry_count >= g.max_retries
            for g in self.sub_goals
        )



# ==============================================================================
# 执行追踪数据模型（可观测性）
# ==============================================================================

@dataclass
class ExecutionStep:
    """
    单次执行步骤记录

    面试要点：
    "每一步都被完整记录：LLM的思考过程、工具调用、结果、耗时。
     这让我可以事后回放整个Agent的执行过程，
     就像调试器的'时间旅行'功能。"
    """
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    iteration: int = 0

    # LLM的思考
    llm_reasoning: str = ""           # LLM的思考过程（CoT）
    model_used: str = ""
    tokens_used: int = 0
    llm_time_ms: float = 0.0

    # 工具调用
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    # 反思结果
    reflection: Optional[str] = None
    needs_replan: bool = False

    # 时间戳
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def total_time_ms(self) -> float:
        """总耗时（毫秒）"""
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return 0.0

@dataclass
class ExecutionTrace:
    """
    完整执行轨迹 - 一次任务的全程记录

    面试要点：
    "轨迹文件是我最重要的调试工具。
     任务失败了？打开trace文件，
     能看到每一步LLM在想什么、调用了什么工具、结果如何。
     这比printf调试高效10倍。"
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    goal: str = ""
    steps: list[ExecutionStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # 汇总统计
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_iterations: int = 0


# ==============================================================================
# 记忆系统数据模型
# ==============================================================================
@dataclass
class MemoryEntry:
    """
    记忆条目 - 存储在向量数据库中

    面试要点：
    "我实现了4层记忆：
     1. 工作记忆(上下文窗口) - 当前对话
     2. 情景记忆(SQLite) - 历史会话
     3. 语义记忆(FAISS) - 向量化的知识
     4. 程序性记忆(文件) - 可复用的技能/工具"
     # 原来有 embedding 字段（FAISS需要手动存向量）
    # ChromaDB 自己管理向量，所以这个字段变为可选/废弃
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    # ChromaDB自己存储向量，不需要我们在dataclass里带着走
    # embedding 字段移除，改为只在chroma_store内部使用
    memory_type: str = "semantic"
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class Session:
    """会话记录 - 存储在SQLite"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""                                         # 会话标题（自动生成）
    messages: list[dict[str, str]] = field(default_factory=list)  # OpenAI格式消息
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 成本追踪数据模型
# ==============================================================================
@dataclass
class TokenUsage:
    """Token消耗记录"""
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0  # 估算费用（美元）
    timestamp: datetime = field(default_factory=datetime.now)


# ==============================================================================
# 异常体系
# ==============================================================================

class AgentError(Exception):
    """
    Agent基础异常

    面试要点：
    "我设计了层次化的异常体系。
     每个异常都附带suggestion，告诉用户/上层代码应该怎么处理。
     这比返回None或裸Exception要好得多——
     错误信息本身就包含了修复建议。"
    """
    def __init__(self, message: str, suggestion: str = ""):
        super().__init__(message)
        self.suggestion = suggestion

class PlanningError(AgentError):
    """规划失败 - 无法生成有效计划"""
    pass

class ExecutionError(AgentError):
    """执行失败 - 工具调用或步骤执行出错"""
    def __init__(self, message: str, tool_name: str = "", suggestion: str = ""):
        super().__init__(message, suggestion)
        self.tool_name = tool_name

class MemoryError(AgentError):
    """记忆系统错误 - 读写失败"""
    pass


class ModelError(AgentError):
    """LLM调用错误 - API超时、token超限等"""
    def __init__(self, message: str, model: str = "", suggestion: str = ""):
        super().__init__(message, suggestion)
        self.model = model


class ToolNotFoundError(AgentError):
    """工具不存在"""
    def __init__(self, tool_name: str):
        super().__init__(
            f"工具 '{tool_name}' 未注册",
            suggestion=f"请检查工具名称是否正确，或使用 registry.list_tools() 查看可用工具"
        )
        self.tool_name = tool_name
