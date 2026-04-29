"""
src/planning/tactical_planner.py - 战术规划器

面试要点：
"战略规划器生成'做什么'（子目标列表），
 战术规划器解决'怎么做'（具体工具调用序列）。

 这个两层设计的好处：
 1. 战略层保持稳定（不会因为一个工具调用失败就重新规划整个目标）
 2. 战术层可以灵活调整（换一个工具试试，或者分步执行）
 3. 职责分离：战略=What，战术=How

 对比：这类似于军事上的战略目标（占领高地）
       vs 战术计划（从左侧包抄还是正面强攻）"
"""

from __future__ import annotations

import logging
from typing import Any

from src.llm_client import LLMClient
from src.models import ModelTier, SubGoal

logger = logging.getLogger(__name__)


TACTICAL_PLANNING_PROMPT = """你是一个任务执行专家。

给定一个子目标，你需要分析并说明执行这个子目标的思路，
同时判断需要使用哪些工具，以及工具的调用顺序。

## 输出格式
以简洁的中文说明你的执行计划，包括：
1. 执行思路（2-3句话）
2. 需要的工具（列表）
3. 注意事项（可选）

保持简洁，不超过150字。
"""


class TacticalPlanner:
    """
    战术规划器 - 为单个子目标生成具体执行方案

    职责：
    - 分析子目标的技术实现路径
    - 建议使用的工具序列
    - 生成执行提示词（注入到Executor的system prompt）

    面试可以说：
    "战术规划器是战略和执行之间的桥梁。
     它把'创建数据库连接模块'这个子目标，
     翻译成'先read_file看现有结构，再write_file创建文件'的工具调用序列。"
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def plan_execution(
        self,
        sub_goal: SubGoal,
        available_tools: list[str],
        context: str = "",
    ) -> str:
        """
        为子目标生成战术执行计划

        Args:
            sub_goal: 要执行的子目标
            available_tools: 可用工具名称列表
            context: 当前环境上下文

        Returns:
            str: 执行计划说明（注入到Executor的用户消息中）
        """
        tools_str = ", ".join(available_tools)

        user_message = f"""
## 子目标
{sub_goal.description}

## 成功标准
{sub_goal.success_criteria or "完成上述子目标"}

## 可用工具
{tools_str}

## 当前上下文
{context or "无特殊上下文"}

请分析如何使用可用工具来完成这个子目标。
"""
        messages = [
            {"role": "system", "content": TACTICAL_PLANNING_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # 用快速模型（战术规划不需要强模型）
        response = self._llm.chat(
            messages=messages,
            tier=ModelTier.FAST,
            temperature=0.3,
            max_tokens=300,
        )

        tactical_plan = response.choices[0].message.content or ""
        logger.debug(f"战术规划完成: {sub_goal.id}")
        return tactical_plan

    def generate_execution_prompt(
            self,
            sub_goal: SubGoal,
            tactical_plan: str,
            relevant_memories: str = "",
            global_context: str = "",  # ← 新增参数
    ) -> str:
        """
        生成传给 Executor 的完整提示词

        v3 新增 global_context：
        确保每个子目标执行时都能看到全局上下文（包含目标文件绝对路径）。
        这解决了"路径在子目标间丢失"的问题。
        """
        parts = [
            f"## 当前子目标\n{sub_goal.description}",
            f"\n## 成功标准\n{sub_goal.success_criteria}" if sub_goal.success_criteria else "",
            f"\n## 执行计划\n{tactical_plan}" if tactical_plan else "",
            f"\n## 全局上下文（重要！）\n{global_context}" if global_context else "",
            f"\n## 相关记忆\n{relevant_memories}" if relevant_memories else "",
            (
                "\n## 指令\n"
                "按照执行计划，使用工具完成当前子目标。\n"
                "如果全局上下文中有文件路径，必须使用该绝对路径，不能用相对路径。\n"
                "完成后说明结果。"
            ),
        ]
        return "\n".join(filter(bool, parts))