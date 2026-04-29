"""
src/agent_loop.py - Agent主循环

修订记录：
  v2 → v3:
  - 修复 _execute_plan 中 EXECUTING→EXECUTING 非法转换
    （循环内不再重复 transition(EXECUTING)，已在 _run_internal 中设置过）
  - 修复 finally 块 IDLE→IDLE 非法转换
    （改用 transition_if_not(IDLE)，幂等安全）
  - 修复反思后状态：REFLECTING → EXECUTING（补全转换路径）
  - 修复重规划后状态：已在 REPLANNING，直接 → EXECUTING
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from src.event_bus import Event, EventType, get_event_bus
from src.execution.executor import Executor
from src.execution.state_machine import StateMachine
from src.llm_client import LLMClient
from src.memory.memory_manager import MemoryManager
from src.models import (
    AgentState, ExecutionTrace, Plan, PlanStatus, SubGoal,
)
from src.observability.tracer import ExecutionTracer
from src.planning.strategic_planner import StrategicPlanner
from src.planning.tactical_planner import TacticalPlanner
from src.reflection.critic import Critic
from src.runtime.checkpoint_store import CheckpointRecord, CheckpointStore
from src.runtime.resume_manager import ResumeManager
from src.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    Agent主循环 - 协调所有组件完成用户目标

    状态机流转（v3修正后）：
    ┌─────────────────────────────────────────────────────────┐
    │  run() 被调用                                           │
    │    IDLE → PLANNING   （_run_internal开头）              │
    │    PLANNING → EXECUTING  （规划完成后）                 │
    │                                                         │
    │  _execute_plan() 循环（已经处于EXECUTING状态）          │
    │    子目标1：                                            │
    │      EXECUTING → REFLECTING  （反思）                  │
    │      REFLECTING → EXECUTING  （继续下一个子目标）       │
    │    子目标2：（同上）                                    │
    │      ...                                                │
    │    全部完成：                                           │
    │      EXECUTING → IDLE  （_run_internal末尾）           │
    │                                                         │
    │  finally块：                                            │
    │      transition_if_not(IDLE)  （幂等，已是IDLE则跳过）  │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        max_replan_attempts: int = 2,
        enable_reflection: bool = True,
    ) -> None:
        self._llm = llm_client
        self._registry = tool_registry
        self._memory = memory_manager
        self._max_replan_attempts = max_replan_attempts
        self._enable_reflection = enable_reflection
        self._bus = get_event_bus()

        self._state_machine = StateMachine()
        self._strategic      = StrategicPlanner(llm_client)
        self._tactical       = TacticalPlanner(llm_client)
        self._executor       = Executor(
            llm_client=llm_client,
            tool_registry=tool_registry,
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "15")),
        )
        self._critic  = Critic(llm_client)
        self._tracer  = ExecutionTracer()
        self._checkpoint_store = CheckpointStore()
        self._resume_manager = ResumeManager(self._checkpoint_store)
        self._current_session_id = ""
        self._current_goal = ""
        self._current_context = ""
        self._last_plan: Optional[Plan] = None

        logger.info(
            f"AgentLoop初始化完成 | "
            f"反思: {'开启' if enable_reflection else '关闭'} | "
            f"最大重规划次数: {max_replan_attempts}"
        )

    # ==================================================================
    # 公开接口
    # ==================================================================

    def run(self, goal: str, session_id: str = "", context: str = "") -> ExecutionTrace:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        self._current_session_id = session_id
        self._current_goal = goal
        self._current_context = context

        trace = ExecutionTrace(
            session_id=session_id,
            goal=goal,
            started_at=datetime.now(),
        )
        self._save_checkpoint("running", trace)

        self._bus.publish(Event(
            event_type=EventType.AGENT_STARTED,
            data={"goal": goal, "session_id": session_id},
            source="agent_loop"
        ))

        try:
            self._run_internal(goal, context, trace)

        except KeyboardInterrupt:
            logger.warning("\n⚠️  用户中断（Ctrl+C）")
            trace.success = False
            trace.final_answer = "用户中断"
            self._state_machine.force_idle("用户中断")
            self._save_checkpoint("interrupted", trace)

        except Exception as e:
            logger.error(f"Agent执行异常: {e}", exc_info=True)
            trace.success = False
            trace.final_answer = f"执行异常: {e}"
            self._state_machine.force_idle(f"异常: {e}")
            self._save_checkpoint("failed", trace)
            self._bus.publish(Event(
                event_type=EventType.AGENT_ERROR,
                data={"error": str(e), "goal": goal},
                source="agent_loop"
            ))

        finally:
            trace.completed_at = datetime.now()

            try:
                self._tracer.save_trace(trace)
            except Exception as e:
                logger.warning(f"保存轨迹失败: {e}")

            # ✅ FIX 1：用 transition_if_not 替代 transition
            # 原因：force_idle 可能已经把状态设为 IDLE，
            # 此时再调 transition(IDLE) 会触发 IDLE→IDLE 非法转换。
            # transition_if_not 是幂等的：已经是目标状态就直接返回。
            self._state_machine.transition_if_not(AgentState.IDLE, "任务结束")

            self._bus.publish(Event(
                event_type=EventType.AGENT_COMPLETED,
                data={
                    "goal":             goal,
                    "success":          trace.success,
                    "total_tokens":     trace.total_tokens,
                    "total_iterations": trace.total_iterations,
                },
                source="agent_loop"
            ))
            final_status = "completed" if trace.success else "failed"
            self._save_checkpoint(final_status, trace)

        return trace

    def resume(self, session_id: str) -> ExecutionTrace:
        record = self._resume_manager.get_resume_record(session_id)
        logger.info(
            "从断点恢复会话: %s | 状态: %s | 更新时间: %s",
            record.session_id,
            record.status,
            record.updated_at,
        )
        plan = self._restore_plan_from_checkpoint(record)
        if plan is None:
            logger.warning("断点缺少计划快照，回退到普通 run() 流程")
            return self.run(
                goal=record.goal,
                session_id=record.session_id,
                context=record.context,
            )

        self._current_session_id = record.session_id
        self._current_goal = record.goal
        self._current_context = record.context
        self._last_plan = plan

        trace = ExecutionTrace(
            session_id=record.session_id,
            goal=record.goal,
            started_at=datetime.now(),
            total_iterations=record.total_iterations,
            total_tool_calls=record.total_tool_calls,
            total_tokens=record.total_tokens,
        )

        self._bus.publish(Event(
            event_type=EventType.AGENT_STARTED,
            data={
                "goal": record.goal,
                "session_id": record.session_id,
                "resumed": True,
                "from_status": record.status,
            },
            source="agent_loop"
        ))

        try:
            self._run_internal(record.goal, record.context, trace, restored_plan=plan)
        except KeyboardInterrupt:
            logger.warning("\n⚠️  用户中断（Ctrl+C）")
            trace.success = False
            trace.final_answer = "用户中断"
            self._state_machine.force_idle("用户中断")
            self._save_checkpoint("interrupted", trace, plan)
        except Exception as e:
            logger.error(f"恢复执行异常: {e}", exc_info=True)
            trace.success = False
            trace.final_answer = f"恢复执行异常: {e}"
            self._state_machine.force_idle(f"恢复异常: {e}")
            self._save_checkpoint("failed", trace, plan)
            self._bus.publish(Event(
                event_type=EventType.AGENT_ERROR,
                data={"error": str(e), "goal": record.goal, "resumed": True},
                source="agent_loop"
            ))
        finally:
            trace.completed_at = datetime.now()
            try:
                self._tracer.save_trace(trace)
            except Exception as e:
                logger.warning(f"保存轨迹失败: {e}")
            self._state_machine.transition_if_not(AgentState.IDLE, "恢复任务结束")
            self._bus.publish(Event(
                event_type=EventType.AGENT_COMPLETED,
                data={
                    "goal": record.goal,
                    "success": trace.success,
                    "total_tokens": trace.total_tokens,
                    "total_iterations": trace.total_iterations,
                    "resumed": True,
                },
                source="agent_loop"
            ))
            final_status = "completed" if trace.success else "failed"
            self._save_checkpoint(final_status, trace, plan)

        return trace

    def list_checkpoints(self, limit: int = 20) -> list[CheckpointRecord]:
        return self._checkpoint_store.list_recent(limit=limit)

    def get_resume_preview(self, session_id: str) -> dict[str, Any]:
        record = self._resume_manager.get_resume_record(session_id)
        plan = self._restore_plan_from_checkpoint(record)
        total_subgoals = record.total_subgoals
        next_subgoal = ""

        if plan is not None:
            if total_subgoals <= 0:
                total_subgoals = len(plan.sub_goals)
            ready = plan.get_ready_goals()
            if ready:
                next_subgoal = f"{ready[0].id}: {ready[0].description}"
            else:
                pending = plan.get_pending_goals()
                if pending:
                    next_subgoal = f"{pending[0].id}: {pending[0].description}"

        return {
            "session_id": record.session_id,
            "status": record.status,
            "resumable_reason": record.resumable_reason,
            "completed_subgoals": len(record.completed_subgoals),
            "total_subgoals": total_subgoals,
            "next_subgoal": next_subgoal or "无（可能已完成或等待依赖）",
            "has_plan_snapshot": bool(record.metadata.get("plan_snapshot")),
        }

    def close(self) -> None:
        try:
            self._memory.close()
            logger.info("AgentLoop已关闭")
        except Exception as e:
            logger.warning(f"AgentLoop关闭时遇到警告: {e}")

    def __enter__(self) -> "AgentLoop":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ==================================================================
    # 内部执行逻辑
    # ==================================================================

    def _run_internal(
        self,
        goal: str,
        context: str,
        trace: ExecutionTrace,
        restored_plan: Optional[Plan] = None,
    ) -> None:
        logger.info(f"\n{'='*60}\n🎯 目标: {goal}\n{'='*60}")

        # 检索相关记忆
        relevant_memories = self._memory.format_memories_for_context(goal, top_k=3)
        if relevant_memories:
            logger.info(f"📚 相关记忆:\n{relevant_memories}")

        # 可选：注入跨会话检索（past sessions）结果到规划上下文
        sessions_ctx = ""
        if os.getenv("AGENT_RECALL_INJECT_SESSIONS", "").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                k = int(os.getenv("AGENT_RECALL_INJECT_SESSIONS_K", "3"))
            except Exception:
                k = 3
            k = max(1, min(8, k))
            sessions_ctx = self._memory.format_sessions_for_context(goal, top_k=k)
            # Guardrail: cap total injected chars to avoid prompt bloat.
            try:
                cap = int(os.getenv("AGENT_RECALL_INJECT_MAX_TOTAL_CHARS", "1200"))
            except Exception:
                cap = 1200
            cap = max(200, min(6000, cap))
            if sessions_ctx and len(sessions_ctx) > cap:
                sessions_ctx = sessions_ctx[: cap - 3] + "..."
            if sessions_ctx:
                logger.info(f"🧭 相关会话:\n{sessions_ctx}")

        full_context = "\n\n".join(filter(bool, [context, sessions_ctx, relevant_memories]))

        if restored_plan is None:
            # 战略规划
            self._state_machine.transition(AgentState.PLANNING, "开始规划")
            logger.info("🗺️  正在生成执行计划...")
            plan = self._strategic.decompose(goal, full_context)
            self._last_plan = plan
            self._log_plan(plan)
            self._save_checkpoint("running", trace, plan)
            # 开始执行（在 _execute_plan 循环之前统一设置 EXECUTING）
            self._state_machine.transition(AgentState.EXECUTING, "规划完成，开始执行")
        else:
            plan = restored_plan
            self._last_plan = plan
            logger.info("♻️ 使用断点计划继续执行（将跳过已完成子目标）")
            self._log_plan(plan)
            self._state_machine.transition(AgentState.EXECUTING, "断点恢复继续执行")
        success = self._execute_plan(plan, trace,global_context=full_context)

        # ✅ FIX 2：_execute_plan 结束后，当前状态可能是 EXECUTING 或 REFLECTING
        # 统一先回到 IDLE，然后再做收尾工作
        self._state_machine.transition_if_not(AgentState.IDLE, "执行循环结束")

        # 收尾
        trace.success = success
        trace.final_answer = self._generate_summary(goal, plan, trace)

        if success:
            self._memory.remember(
                content=(
                    f"成功完成目标：{goal}。"
                    f"共{trace.total_tool_calls}次工具调用，"
                    f"{trace.total_iterations}次迭代。"
                ),
                memory_type="episodic",
                importance=0.6,
                metadata={"goal": goal, "trace_id": trace.trace_id},
            )
            logger.info("\n✅ 目标完成！")
        else:
            logger.warning("\n❌ 目标未能完成")

        logger.info(trace.final_answer)

    def _execute_plan(self, plan: Plan, trace: ExecutionTrace,global_context: str = "") -> bool:
        """
        按依赖顺序执行计划中的所有子目标

        调用本方法时的前置状态：EXECUTING（由 _run_internal 设置）
        本方法退出时的状态：
          - 正常完成：EXECUTING（由调用方负责后续转换）
          - 反思后末尾：REFLECTING（由调用方负责后续转换）
          调用方统一用 transition_if_not(IDLE) 收尾

        面试要点：
        "状态机的职责边界：
         _execute_plan 不负责'进入EXECUTING'（那是调用方的职责），
         只负责循环内部的 EXECUTING↔REFLECTING↔REPLANNING 转换。
         职责清晰，每个方法只管自己的状态段。"
        """
        iteration_offset = trace.total_iterations
        replan_count = 0
        max_rounds = len(plan.sub_goals) * (self._max_replan_attempts + 1) * 5

        for _round in range(max_rounds):

            if plan.is_completed():
                return True

            if plan.is_failed():
                logger.error("计划中有子目标失败且超出重试次数")
                return False

            ready = plan.get_ready_goals()
            if not ready:
                logger.warning("没有可执行的子目标（可能存在未满足的依赖）")
                return False

            sub_goal = ready[0]
            sub_goal.status = PlanStatus.IN_PROGRESS

            logger.info(
                f"\n{'─'*50}\n"
                f"📍 [{sub_goal.id}] {sub_goal.description}\n"
                f"{'─'*50}"
            )
            execution_prompt = self._build_execution_prompt(
                sub_goal,
                global_context=global_context,  # ← 传入
            )

            # ✅ FIX 3：不在循环内重复 transition(EXECUTING)
            # 原因：第一次进入循环时已处于 EXECUTING（_run_internal 设置）；
            #       后续循环时从 REFLECTING→EXECUTING 由反思阶段末尾负责。
            # 这里只需确认当前状态是 EXECUTING 即可（断言，便于调试）
            assert self._state_machine.state == AgentState.EXECUTING, (
                f"进入子目标 {sub_goal.id} 执行时状态应为 EXECUTING，"
                f"实际: {self._state_machine.state.value}"
            )

            # 执行 ReAct 循环（在 EXECUTING 状态内）
            success, result = self._executor.execute_sub_goal(
                sub_goal=sub_goal,
                execution_prompt=execution_prompt,
                trace=trace,
                iteration_offset=iteration_offset,
            )
            iteration_offset = trace.total_iterations

            # ── 反思阶段 ──────────────────────────────────────────────
            reflection_result = None
            if self._enable_reflection:
                # EXECUTING → REFLECTING
                self._state_machine.transition(AgentState.REFLECTING, f"反思子目标 {sub_goal.id}")

                steps_this_goal = trace.steps[-max(1, trace.total_iterations - iteration_offset + 1):]
                reflection_result = self._critic.evaluate_step(
                    sub_goal=sub_goal,
                    steps=steps_this_goal if steps_this_goal else trace.steps[-3:],
                    success=success,
                    result_summary=result,
                )
                logger.info(
                    f"🤔 反思 | 质量分: {reflection_result.quality_score:.2f} "
                    f"| 需要重规划: {reflection_result.needs_replan}"
                )
                if reflection_result.reflection_text:
                    logger.info(f"   {reflection_result.reflection_text}")

            # ── 根据结果决定下一步 ────────────────────────────────────
            needs_replan = (
                reflection_result is not None and reflection_result.needs_replan
            )

            if success and not needs_replan:
                # ── 正常完成 ──────────────────────────────────────────
                sub_goal.status = PlanStatus.COMPLETED
                sub_goal.result = result
                logger.info(f"✅ 子目标完成: {sub_goal.id}")

                # 如果开了反思，当前在 REFLECTING；需要转回 EXECUTING 继续
                # 如果没开反思，当前在 EXECUTING；不需要转换
                if self._enable_reflection:
                    # ✅ FIX 4：REFLECTING → EXECUTING（新增的合法转换）
                    self._state_machine.transition(
                        AgentState.EXECUTING, "继续执行下一个子目标"
                    )
                self._save_checkpoint("running", trace, plan)
                # 没有开反思时：保持 EXECUTING，循环继续

            elif not success and replan_count < self._max_replan_attempts:
                # ── 触发重规划 ────────────────────────────────────────
                sub_goal.status = PlanStatus.FAILED
                sub_goal.error  = result
                replan_count += 1

                # 当前状态：REFLECTING（开了反思）或 EXECUTING（没开反思）
                # 两种情况都可以转到 REPLANNING
                self._state_machine.transition(
                    AgentState.REPLANNING,
                    f"子目标失败，第{replan_count}次重规划"
                )
                logger.info(f"🔄 触发重规划（第 {replan_count}/{self._max_replan_attempts} 次）")

                reflection_text = (
                    reflection_result.reflection_text
                    if reflection_result else f"执行失败: {result}"
                )
                plan = self._strategic.replan(plan, sub_goal, reflection_text)
                self._last_plan = plan
                self._log_plan(plan)
                self._save_checkpoint("running", trace, plan)

                # REPLANNING → EXECUTING，继续循环
                self._state_machine.transition(AgentState.EXECUTING, "重规划完成")

            else:
                # ── 超出重试次数或致命错误 ────────────────────────────
                sub_goal.status = PlanStatus.FAILED
                sub_goal.error  = result
                logger.error(f"❌ 子目标 {sub_goal.id} 最终失败，放弃")
                self._save_checkpoint("failed", trace, plan)
                return False

        logger.error(f"执行轮次超出安全上限 {max_rounds}")
        self._save_checkpoint("failed", trace, plan)
        return False

    def _build_execution_prompt(
            self,
            sub_goal,
            global_context: str = "",
    ) -> str:
        """
        构建子目标的完整执行提示词

        v3 修复：把全局 context（包含目标文件路径）注入到每个子目标的提示里。
        根因：StrategicPlanner 拆分子目标时，路径信息只在顶层 goal 里，
              子目标的 description 可能不包含完整路径。
              解决：把 global_context 透传到每个子目标的执行提示。
        """
        available_tools = [t.name for t in self._registry.get_all()]

        tactical_plan = self._tactical.plan_execution(
            sub_goal=sub_goal,
            available_tools=available_tools,
            context=global_context,  # ← 传入全局上下文
        )
        relevant_memories = self._memory.format_memories_for_context(
            sub_goal.description, top_k=2
        )
        return self._tactical.generate_execution_prompt(
            sub_goal=sub_goal,
            tactical_plan=tactical_plan,
            relevant_memories=relevant_memories,
            global_context=global_context,  # ← 传入全局上下文
        )

    def _generate_summary(self, goal: str, plan: Plan, trace: ExecutionTrace) -> str:
        completed = sum(1 for sg in plan.sub_goals if sg.status == PlanStatus.COMPLETED)
        total     = len(plan.sub_goals)
        duration  = ""
        if trace.completed_at and trace.started_at:
            secs = (trace.completed_at - trace.started_at).total_seconds()
            duration = f"{secs:.1f}秒"

        lines = [
            "─" * 50,
            f"目标:       {goal}",
            f"完成情况:   {completed}/{total} 个子目标",
            f"总迭代:     {trace.total_iterations} 次",
            f"工具调用:   {trace.total_tool_calls} 次",
            f"Token消耗:  {trace.total_tokens}",
            f"耗时:       {duration}",
            f"状态:       {'✅ 成功' if trace.success else '❌ 失败'}",
            "─" * 50,
        ]
        return "\n".join(lines)

    def _log_plan(self, plan: Plan) -> None:
        logger.info(f"\n📋 执行计划（{len(plan.sub_goals)} 个子目标）:")
        for i, sg in enumerate(plan.sub_goals, 1):
            dep_str = f" → 依赖: {sg.dependencies}" if sg.dependencies else ""
            logger.info(f"  {i}. [{sg.id}] {sg.description}{dep_str}")

    def _save_checkpoint(
        self,
        status: str,
        trace: ExecutionTrace,
        plan: Optional[Plan] = None,
    ) -> None:
        try:
            if plan is None:
                plan = self._last_plan
            completed_subgoals: list[str] = []
            total_subgoals = 0
            plan_snapshot: list[dict[str, object]] = []
            if plan is not None:
                completed_subgoals = [
                    sg.id for sg in plan.sub_goals if sg.status == PlanStatus.COMPLETED
                ]
                total_subgoals = len(plan.sub_goals)
                plan_snapshot = self._serialize_plan_snapshot(plan)
            failed_subgoal = self._extract_last_failed_subgoal(plan)
            record = CheckpointRecord(
                session_id=self._current_session_id or trace.session_id,
                goal=self._current_goal or trace.goal,
                context=self._current_context,
                status=status,
                updated_at=datetime.now().isoformat(),
                completed_subgoals=completed_subgoals,
                total_subgoals=total_subgoals,
                total_iterations=trace.total_iterations,
                total_tool_calls=trace.total_tool_calls,
                total_tokens=trace.total_tokens,
                final_answer=trace.final_answer or "",
                resumable_reason=self._checkpoint_reason(status),
                metadata={
                    "trace_id": trace.trace_id,
                    "plan_snapshot": plan_snapshot,
                    "last_failed_subgoal": failed_subgoal,
                },
            )
            self._checkpoint_store.save(record)
        except Exception as e:
            logger.warning(f"保存checkpoint失败: {e}")

    def _serialize_plan_snapshot(self, plan: Plan) -> list[dict[str, object]]:
        return [
            {
                "id": sg.id,
                "description": sg.description,
                "parent_id": sg.parent_id,
                "status": sg.status.value,
                "dependencies": list(sg.dependencies),
                "success_criteria": sg.success_criteria,
                "rollback_strategy": sg.rollback_strategy,
                "result": sg.result,
                "error": sg.error,
                "retry_count": sg.retry_count,
                "max_retries": sg.max_retries,
            }
            for sg in plan.sub_goals
        ]

    def _restore_plan_from_checkpoint(self, record: CheckpointRecord) -> Optional[Plan]:
        snapshot = record.metadata.get("plan_snapshot", [])
        if not isinstance(snapshot, list) or not snapshot:
            return None

        sub_goals: list[SubGoal] = []
        for item in snapshot:
            if not isinstance(item, dict):
                continue
            try:
                status_value = str(item.get("status", PlanStatus.PENDING.value))
                sub_goals.append(SubGoal(
                    id=str(item.get("id", "")),
                    description=str(item.get("description", "")),
                    parent_id=item.get("parent_id"),  # type: ignore[arg-type]
                    status=PlanStatus(status_value),
                    dependencies=list(item.get("dependencies", [])),  # type: ignore[arg-type]
                    success_criteria=str(item.get("success_criteria", "")),
                    rollback_strategy=str(item.get("rollback_strategy", "")),
                    result=item.get("result"),  # type: ignore[arg-type]
                    error=item.get("error"),  # type: ignore[arg-type]
                    retry_count=int(item.get("retry_count", 0)),
                    max_retries=int(item.get("max_retries", 2)),
                ))
            except Exception:
                continue

        if not sub_goals:
            return None

        return Plan(goal=record.goal, sub_goals=sub_goals)

    def _extract_last_failed_subgoal(self, plan: Optional[Plan]) -> str:
        if plan is None:
            return ""
        failed = [sg for sg in plan.sub_goals if sg.status == PlanStatus.FAILED]
        if not failed:
            return ""
        last = failed[-1]
        return f"{last.id}:{last.description[:40]}"

    def _checkpoint_reason(self, status: str) -> str:
        if status == "running":
            return "任务执行中，可中断后恢复"
        if status == "interrupted":
            return "用户中断，可从断点继续"
        if status == "failed":
            return "执行失败，可修复后重试"
        if status == "completed":
            return "已完成，无需恢复"
        return "未知状态"