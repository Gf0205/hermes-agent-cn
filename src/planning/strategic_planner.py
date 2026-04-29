"""
src/planning/strategic_planner.py - 战略规划器

面试要点：
"战略规划器的核心思想来自 Tree-of-Thoughts（ToT）论文。
 我把一个大目标分解成4-6个子目标，每个子目标：
 1. 有明确的成功标准（知道什么时候算完成）
 2. 有依赖关系（保证执行顺序）
 3. 有回滚策略（失败了怎么办）

 实现上，我用结构化输出（JSON mode）让LLM直接返回Plan对象，
 而不是从自然语言里解析，这大幅提高了解析成功率。

 对比原版Hermes：
 原版Hermes没有显式的规划层，直接进入ReAct循环。
 我增加了战略规划层，优点是Agent不会'只见树木不见森林'，
 在执行每个工具调用前都清楚自己在完成哪个子目标。"
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.event_bus import Event, EventType, get_event_bus
from src.llm_client import LLMClient
from src.models import ModelTier, Plan, PlanStatus, PlanningError, SubGoal

logger = logging.getLogger(__name__)


# 战略规划的系统提示词

STRATEGIC_PLANNING_PROMPT = """你是一个专业的任务规划专家。

你的工作是把用户的目标分解成具体可执行的子目标。

## 重要原则：识别单步任务

如果用户目标中包含以下特征，说明这是一个**单步任务**，
只需生成 1 个子目标，不要过度拆分：
- 明确说"单步任务"或"只需要调用一次工具"
- 目标非常具体（如"在路径X创建文件Y，内容是Z"）
- 已经包含完整的操作细节

## 分解原则（多步任务时）
1. 每个子目标必须具体、可验证
2. 依赖关系要准确（A依赖B意味着B必须先完成）
3. 每个子目标的成功标准要明确
4. 子目标数量：1-6个（单步任务就是1个）

## 输出格式
必须返回合法JSON，格式如下：
{
  "analysis": "对目标的简要分析（1-2句话）",
  "sub_goals": [
    {
      "id": "sg_1",
      "description": "子目标描述",
      "dependencies": [],
      "success_criteria": "成功标准",
      "rollback_strategy": "失败时的处理方案"
    }
  ]
}

## 约束
- id格式：sg_1, sg_2, sg_3...
- dependencies中填写前置子目标的id
- 只返回JSON，不要有其他文字
"""


class StrategicPlanner:
    """
    战略规划器 - 将用户目标分解为层级化子目标树

    面试可以说：
    "这相当于项目管理中的WBS（工作分解结构）。
     把一个大项目分成小任务，
     明确任务间的依赖关系，
     这样Agent执行时就有清晰的路线图。"
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._bus = get_event_bus()

    def decompose(self, goal: str, context: str = "") -> Plan:
        """
        将目标分解为子目标计划

        Args:
            goal: 用户目标（如"创建一个Flask CRUD API"）
            context: 额外上下文（如当前工作目录、已有文件等）

        Returns:
            Plan: 包含层级化子目标的计划

        Raises:
            PlanningError: 无法生成有效计划时
        """
        start_time = time.time()

        # 构建规划请求
        user_message = f"## 目标\n{goal}"
        if context:
            user_message += f"\n\n## 上下文\n{context}"

        messages = [
            {"role": "system", "content": STRATEGIC_PLANNING_PROMPT},
            {"role": "user", "content": user_message},
        ]

        logger.info(f"开始战略规划: {goal[:50]}...")

        # 调用强模型（规划是复杂推理任务）
        try:
            response = self._llm.chat(
                messages=messages,
                tier=ModelTier.STRONG,
                temperature=0.3,   # 低温度：规划需要确定性
                max_tokens=2000,
            )
        except Exception as e:
            raise PlanningError(
                f"LLM调用失败: {e}",
                suggestion="检查API密钥和网络连接"
            )

        content = response.choices[0].message.content or ""

        # 解析JSON输出
        plan_data = self._parse_plan_json(content, goal)

        # 构建Plan对象
        plan = self._build_plan(goal, plan_data, response)
        elapsed_ms = (time.time() - start_time) * 1000
        plan.planning_time_ms = elapsed_ms

        logger.info(
            f"规划完成 | {len(plan.sub_goals)} 个子目标 "
            f"| 耗时: {elapsed_ms:.0f}ms"
        )

        # 发布规划完成事件
        self._bus.publish(Event(
            event_type=EventType.PLAN_CREATED,
            data={
                "plan_id": plan.id,
                "goal": goal,
                "sub_goal_count": len(plan.sub_goals),
                "planning_time_ms": elapsed_ms,
            },
            source="strategic_planner"
        ))

        return plan

    def _parse_plan_json(self, content: str, goal: str) -> dict[str, Any]:
        """
        解析LLM输出的JSON

        面试要点：
        "LLM的JSON输出有时会被markdown代码块包裹（```json...```），
         我需要先strip掉这些包裹。
         这是处理LLM结构化输出的常见trick。"
        """
        # 清理markdown代码块
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # 移除首尾的 ``` 行
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # 尝试从内容中提取JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except Exception:
                    pass

            raise PlanningError(
                f"无法解析LLM返回的计划JSON: {e}",
                suggestion="这通常是模型输出格式问题，可以重试或简化目标描述"
            )

    def _build_plan(
        self,
        goal: str,
        plan_data: dict[str, Any],
        llm_response: Any,
    ) -> Plan:
        """将解析的JSON数据构建为Plan对象"""
        sub_goals_data = plan_data.get("sub_goals", [])

        if not sub_goals_data:
            raise PlanningError(
                "规划结果为空：LLM没有生成任何子目标",
                suggestion="请尝试更具体地描述你的目标"
            )

        sub_goals = []
        for sg_data in sub_goals_data:
            sub_goal = SubGoal(
                id=sg_data.get("id", f"sg_{len(sub_goals)+1}"),
                description=sg_data.get("description", ""),
                dependencies=sg_data.get("dependencies", []),
                success_criteria=sg_data.get("success_criteria", ""),
                rollback_strategy=sg_data.get("rollback_strategy", ""),
                status=PlanStatus.PENDING,
            )
            sub_goals.append(sub_goal)

        tokens_used = 0
        if llm_response.usage:
            tokens_used = llm_response.usage.total_tokens

        return Plan(
            goal=goal,
            sub_goals=sub_goals,
            status=PlanStatus.IN_PROGRESS,
            planning_tokens_used=tokens_used,
            model_used=self._llm.config.strong_model,
        )

    def replan(
        self,
        original_plan: Plan,
        failed_sub_goal: SubGoal,
        reflection: str,
    ) -> Plan:
        """
        基于失败和反思重新规划

        面试要点：
        "重规划是Agent从错误中恢复的关键。
         我把原始计划、失败的子目标、反思结论都传给LLM，
         让它生成修正后的计划。
         这比'从头重来'效率高得多——只重规划失败的部分。"
        """
        completed = [sg for sg in original_plan.sub_goals
                     if sg.status == PlanStatus.COMPLETED]
        completed_str = "\n".join(f"✅ {sg.description}" for sg in completed)

        replan_message = f"""
## 原始目标
{original_plan.goal}

## 已完成的步骤
{completed_str or "（无）"}

## 失败的步骤
❌ {failed_sub_goal.description}
失败原因: {failed_sub_goal.error or "未知"}

## 反思结论
{reflection}

请基于以上信息，重新规划剩余的步骤。
只规划还未完成的部分，已完成的步骤不要重复。
"""
        messages = [
            {"role": "system", "content": STRATEGIC_PLANNING_PROMPT},
            {"role": "user", "content": replan_message},
        ]

        response = self._llm.chat(
            messages=messages,
            tier=ModelTier.STRONG,
            temperature=0.2,
        )

        content = response.choices[0].message.content or ""
        plan_data = self._parse_plan_json(content, original_plan.goal)

        new_plan = self._build_plan(original_plan.goal, plan_data, response)

        self._bus.publish(Event(
            event_type=EventType.REPLAN_TRIGGERED,
            data={
                "original_plan_id": original_plan.id,
                "new_plan_id": new_plan.id,
                "failed_sub_goal": failed_sub_goal.id,
            },
            source="strategic_planner"
        ))

        return new_plan