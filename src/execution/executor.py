"""
src/execution/executor.py - 执行引擎（ReAct循环）

面试要点（这是整个Agent最核心的部分！）：
"执行引擎实现了ReAct（Reasoning + Acting）模式。

 ReAct的核心循环：
 1. Reason：LLM思考当前状态，决定下一步
 2. Act：调用工具执行动作
 3. Observe：观察工具结果
 4. 重复，直到任务完成

 我的实现有几个关键设计：
 a) tool_calls和tool_results严格成对：
    LLM返回tool_call → 执行工具 → 把结果以tool角色回传给LLM
    这是OpenAI function calling的标准协议

 b) 终止条件：
    - LLM不再调用工具（说明任务完成）
    - 达到最大迭代次数（防止无限循环）
    - 显式返回TASK_COMPLETE信号

 c) 可观测性：
    每次迭代都记录ExecutionStep，包含LLM的思考和工具调用结果。

对比原版Hermes：
Hermes的executor.js是纯函数式，我用类封装状态，
这让追踪'当前执行到第几步'更清晰。"
"""

from __future__ import annotations

import logging
import os
import time
import json
from datetime import datetime
from typing import Any, Optional

from src.agent.context_compressor_v2 import ContextCompressorV2
from src.event_bus import Event, EventType, get_event_bus
from src.execution.parallel_executor import ParallelExecutor
from src.llm_client import LLMClient
from src.models import (
    AgentState, ExecutionStep, ExecutionTrace, ModelTier,
    Plan, SubGoal, ToolCall, ToolResult, ToolStatus,
)
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# Executor的系统提示词（核心！）
EXECUTOR_SYSTEM_PROMPT = """你是一个精确、高效的AI执行代理。

## 你的工作方式
1. 分析当前子目标和上下文
2. 选择最合适的工具执行操作
3. 观察结果，调整策略
4. 重复直到子目标完成

## 重要原则
- 每次只做一件事，不要同时调用过多工具
- 如果工具失败，分析原因后换策略重试
- 完成后明确说明"子目标已完成：[结果摘要]"
- 不确定时先read_file/list_dir了解现状，再行动

## 代码规范（如果需要写代码）
- Python文件使用UTF-8编码
- 包含适当的错误处理
- 变量名用snake_case

你有以下工具可以使用，根据任务需要选择合适的工具。
"""


class Executor:
    """
    执行引擎 - 驱动ReAct循环完成单个子目标

    工作流程：
    ┌─────────────────────────────────────────┐
    │  build_messages()                        │
    │      ↓                                   │
    │  llm.chat(tools=all_tools)              │
    │      ↓                                   │
    │  has tool_calls?                        │
    │    YES → execute_tools() → append result│
    │         → loop back to llm.chat()       │
    │    NO  → task complete, return          │
    └─────────────────────────────────────────┘
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        max_iterations: int = 15,
    ) -> None:
        self._llm = llm_client
        self._registry = tool_registry
        self._max_iterations = max_iterations
        self._bus = get_event_bus()
        self._parallel_executor = ParallelExecutor(max_workers=4)
        self._context_compressor = ContextCompressorV2(llm_client)
        self._model_context_limit = int(os.getenv("AGENT_MODEL_CONTEXT_LIMIT", "32000"))
        self._tool_timeout_retries = max(0, int(os.getenv("AGENT_TOOL_TIMEOUT_RETRIES", "1")))

    def execute_sub_goal(
        self,
        sub_goal: SubGoal,
        execution_prompt: str,
        trace: ExecutionTrace,
        iteration_offset: int = 0,
    ) -> tuple[bool, str]:
        """
        执行单个子目标（ReAct循环）

        Args:
            sub_goal: 要执行的子目标
            execution_prompt: 战术规划器生成的执行提示
            trace: 执行轨迹记录对象（由外部传入，追加记录）
            iteration_offset: 迭代计数偏移（多个子目标时保持连续计数）

        Returns:
            tuple(success: bool, result_summary: str)
        """
        # 构建初始消息列表
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT},
            {"role": "user", "content": execution_prompt},
        ]

        # 获取所有工具的OpenAI Schema
        openai_tools = self._registry.to_openai_tools()

        self._bus.publish(Event(
            event_type=EventType.SUBGOAL_STARTED,
            data={"sub_goal_id": sub_goal.id, "description": sub_goal.description},
            source="executor"
        ))

        for iteration in range(self._max_iterations):
            step_num = iteration_offset + iteration + 1
            step = ExecutionStep(iteration=step_num)

            self._bus.publish(Event(
                event_type=EventType.STEP_STARTED,
                data={"step": step_num, "sub_goal_id": sub_goal.id},
                source="executor"
            ))

            # ── 调用LLM ───────────────────────────────────────────────
            llm_start = time.time()
            try:
                messages = self._context_compressor.compress(
                    messages=messages,
                    max_tokens=self._model_context_limit,
                )
                response = self._llm.chat(
                    messages=messages,
                    tier=ModelTier.STRONG,
                    tools=openai_tools if openai_tools else None,
                )
            except Exception as e:
                logger.error(f"LLM调用失败 (迭代{step_num}): {e}")
                step.completed_at = datetime.now()
                trace.steps.append(step)
                return False, f"LLM调用失败: {e}"

            llm_elapsed = (time.time() - llm_start) * 1000
            step.llm_time_ms = llm_elapsed
            step.model_used = self._llm.config.strong_model

            if response.usage:
                step.tokens_used = response.usage.total_tokens
                trace.total_tokens += step.tokens_used

            assistant_message = response.choices[0].message

            # 记录LLM的思考内容
            step.llm_reasoning = assistant_message.content or ""

            # 把assistant的回复加入消息历史
            # 注意：必须先把assistant消息加入，再加tool结果
            messages.append(assistant_message.model_dump(exclude_none=True))

            # ── 检查是否有工具调用 ────────────────────────────────────
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                # LLM没有调用工具 → 任务完成（或需要用户输入）
                final_answer = assistant_message.content or "子目标已完成"
                step.completed_at = datetime.now()
                trace.steps.append(step)

                self._bus.publish(Event(
                    event_type=EventType.SUBGOAL_COMPLETED,
                    data={
                        "sub_goal_id": sub_goal.id,
                        "iterations": iteration + 1,
                        "final_answer": final_answer[:200],
                    },
                    source="executor"
                ))

                logger.info(
                    f"子目标完成: {sub_goal.id} "
                    f"| 迭代: {iteration+1} 次 "
                    f"| 最终回复: {final_answer[:100]}..."
                )
                return True, final_answer

            # ── 执行工具调用 ──────────────────────────────────────────
            prepared_calls: list[tuple[Any, str, dict[str, Any]]] = []
            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                prepared_calls.append((tc, tool_name, arguments))

                tool_call_record = ToolCall(
                    call_id=tc.id,
                    tool_name=tool_name,
                    arguments=arguments,
                )
                step.tool_calls.append(tool_call_record)

                self._bus.publish(Event(
                    event_type=EventType.TOOL_CALLED,
                    data={
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "call_id": tc.id,
                    },
                    source="executor"
                ))
            executed_calls = self._parallel_executor.execute_parallel_tools(
                tool_calls=prepared_calls,
                execute_fn=self._execute_single_tool,
            )
            for tc, tool_name, arguments, tool_result in executed_calls:
                tool_result = self._normalize_tool_result(tool_result, tool_name=tool_name)
                step.tool_results.append(tool_result)
                trace.total_tool_calls += 1

                self._bus.publish(Event(
                    event_type=EventType.TOOL_RESULT,
                    data={
                        "tool_name": tool_name,
                        "status": tool_result.status.value,
                        "output_preview": tool_result.output[:200],
                        "execution_time_ms": tool_result.execution_time_ms,
                    },
                    source="executor"
                ))

                # 把工具结果回传给LLM（OpenAI协议要求）
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": self._format_tool_result(tool_result),
                })

            step.completed_at = datetime.now()
            trace.steps.append(step)
            trace.total_iterations += 1

        # 达到最大迭代次数
        logger.warning(f"子目标 {sub_goal.id} 达到最大迭代次数 ({self._max_iterations})")
        self._bus.publish(Event(
            event_type=EventType.SUBGOAL_FAILED,
            data={
                "sub_goal_id": sub_goal.id,
                "reason": f"超过最大迭代次数 {self._max_iterations}",
            },
            source="executor"
        ))
        return False, f"超过最大迭代次数 {self._max_iterations}"

    def _execute_single_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """
        执行单个工具调用

        面试要点：
        "工具不存在和工具执行失败是两种不同的错误。
         前者说明LLM幻觉了一个不存在的工具名，
         后者是正常的业务失败（文件不存在、权限不足等）。
         我分开处理，错误信息也分开给LLM，
         这样LLM能更准确地判断下一步怎么做。"
        """
        try:
            tool = self._registry.get(tool_name)
        except KeyError:
            available = [t.name for t in self._registry.get_all()]
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.FAILURE,
                output="",
                error=f"Tool '{tool_name}' not found. Available: {available}",
            )

        start = time.time()
        attempts = self._tool_timeout_retries + 1
        for attempt in range(attempts):
            try:
                raw = tool.execute(**arguments)
                result = self._coerce_tool_result(raw, tool_name=tool_name)
                if result.execution_time_ms <= 0:
                    result.execution_time_ms = (time.time() - start) * 1000.0
                result.metadata["attempts"] = attempt + 1
                return result
            except Exception as e:
                status, retryable = self._classify_tool_exception(e)
                if retryable and attempt < attempts - 1:
                    logger.warning(
                        "Tool timeout, retrying | tool=%s | attempt=%s/%s | error=%s",
                        tool_name,
                        attempt + 1,
                        attempts,
                        e,
                    )
                    continue
                return ToolResult(
                    tool_name=tool_name,
                    status=status,
                    output="",
                    error=f"{type(e).__name__}: {e}",
                    execution_time_ms=(time.time() - start) * 1000.0,
                    metadata={
                        "attempts": attempt + 1,
                        "retryable_timeout": retryable,
                        "error_type": type(e).__name__,
                    },
                )

    def _format_tool_result(self, result: ToolResult) -> str:
        """
        格式化工具结果（供LLM消费）

        面试要点：
        "格式化是有讲究的：
         成功时只返回output（节省tokens）
         失败时返回error + 状态（让LLM知道失败了要换策略）"
        """
        if result.status == ToolStatus.SUCCESS:
            return result.output or "（工具执行成功，无输出）"
        else:
            return (
                f"❌ 工具执行失败\n"
                f"状态: {result.status.value}\n"
                f"错误: {result.error}"
            )

    def _classify_tool_exception(self, error: Exception) -> tuple[ToolStatus, bool]:
        message = str(error).lower()
        if isinstance(error, TimeoutError) or "timeout" in message or "timed out" in message:
            return ToolStatus.TIMEOUT, True
        if isinstance(error, PermissionError) or "permission" in message or "denied" in message:
            return ToolStatus.PERMISSION_DENIED, False
        return ToolStatus.FAILURE, False

    def _coerce_tool_result(self, raw: Any, tool_name: str) -> ToolResult:
        if isinstance(raw, ToolResult):
            return raw

        # 兼容少数工具可能返回非 ToolResult 的情况，避免主循环崩溃。
        return ToolResult(
            tool_name=tool_name,
            status=ToolStatus.SUCCESS,
            output=str(raw),
            metadata={"coerced_result": True},
        )

    def _normalize_tool_result(self, result: ToolResult, tool_name: str) -> ToolResult:
        status = result.status if isinstance(result.status, ToolStatus) else ToolStatus.FAILURE
        output = result.output if isinstance(result.output, str) else str(result.output)
        error: Optional[str]
        if result.error is None:
            error = None
        else:
            error = result.error if isinstance(result.error, str) else str(result.error)
        execution_time_ms = result.execution_time_ms if result.execution_time_ms >= 0 else 0.0
        metadata = dict(result.metadata or {})

        normalized = ToolResult(
            tool_name=tool_name or result.tool_name,
            status=status,
            output=output[:120000],
            error=error[:4000] if error else None,
            execution_time_ms=execution_time_ms,
            metadata=metadata,
        )
        return normalized