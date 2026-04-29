"""
src/reflection/critic.py - LLM-as-Critic 自我反思

面试要点：
"自我反思（Self-Reflection）是让Agent从错误中学习的核心机制。

 传统程序：失败 → 抛异常 → 终止
 有反思的Agent：失败 → 分析失败原因 → 提出改进方案 → 重试

 我的实现参考了 Reflexion 论文（Shinn et al., 2023）的思路：
 1. Critic评估上一步的执行质量（不只看成功/失败，还评估效率）
 2. 生成自然语言的反思（'我犯了什么错？下次怎么做更好？'）
 3. 把反思结果注入下一次规划，避免重蹈覆辙

 关键设计：
 Critic是独立的LLM调用，用FAST模型（节省成本），
 它不执行任何动作，只做评估和建议。
 职责分离：Executor负责'做'，Critic负责'评'。"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.llm_client import LLMClient
from src.models import ExecutionStep, ModelTier, SubGoal, ToolStatus

logger = logging.getLogger(__name__)


# ==============================================================================
# 反思结果数据模型
# ==============================================================================

@dataclass
class ReflectionResult:
    """
    反思结果

    面试要点：
    "我把反思结果结构化，而不是只返回一段文字。
     needs_replan 是关键字段：True时触发重规划，
     False时只记录经验，继续执行原计划。
     这避免了'任何一点小问题都触发重规划'的过度反应。"
    """
    # 执行质量评分 0.0-1.0
    quality_score: float = 0.0

    # 是否需要重规划
    needs_replan: bool = False

    # 自然语言反思（注入到下一步的context中）
    reflection_text: str = ""

    # 具体问题列表
    issues: list[str] = field(default_factory=list)

    # 改进建议
    suggestions: list[str] = field(default_factory=list)

    # 是否是致命错误（需要立即停止）
    is_fatal: bool = False


# ==============================================================================
# Critic系统提示词
# ==============================================================================

CRITIC_SYSTEM_PROMPT = """你是一个AI Agent执行质量评估专家（Critic）。

你的任务是评估Agent刚刚完成的一个执行步骤，提供客观的质量反馈。

## 评估维度
1. 目标达成度：子目标是否被正确完成？
2. 工具选择合理性：选用的工具是否是最优选择？
3. 效率：是否有冗余的工具调用？
4. 错误处理：遇到错误时的处理方式是否合理？

## 输出要求
请严格按以下JSON格式输出，不要有其他文字：
{
  "quality_score": 0.85,
  "needs_replan": false,
  "reflection": "简洁的总体评价（1-2句话）",
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "is_fatal": false
}

## 评分标准
- 0.9-1.0：优秀，完全达成目标，工具选择高效
- 0.7-0.9：良好，基本达成目标，有小瑕疵
- 0.5-0.7：一般，部分达成，需要改进
- 0.0-0.5：差，未达成目标或有严重问题

## needs_replan 判断标准
只有以下情况才设为 true：
- 子目标根本没有完成
- 使用了错误的方法导致无法用原计划继续
- 发现了影响整个计划的前提假设错误
"""


class Critic:
    """
    执行质量评估器

    工作时机：
    - 每个子目标执行完成后（无论成功失败）
    - 工具调用失败超过阈值时

    使用 FAST 模型：
    Critic的评估不需要强大的推理能力，
    用 qwen-plus 而不是 qwen-max，
    在保证质量的同时节省约70%的Token成本。
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def evaluate_step(
        self,
        sub_goal: SubGoal,
        steps: list[ExecutionStep],
        success: bool,
        result_summary: str,
    ) -> ReflectionResult:
        """
        评估子目标的执行质量

        Args:
            sub_goal:       刚执行完的子目标
            steps:          该子目标对应的所有执行步骤
            success:        是否成功完成
            result_summary: 执行结果摘要

        Returns:
            ReflectionResult: 结构化的反思结果
        """
        # 构建评估上下文
        execution_summary = self._build_execution_summary(sub_goal, steps, success, result_summary)

        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user",   "content": execution_summary},
        ]

        try:
            response = self._llm.chat(
                messages=messages,
                tier=ModelTier.FAST,  # 用快速模型节省成本
                temperature=0.2,      # 低温度：评估需要稳定输出
                max_tokens=500,
            )

            content = response.choices[0].message.content or ""
            return self._parse_reflection(content, success)

        except Exception as e:
            logger.warning(f"Critic评估失败（降级处理）: {e}")
            # 降级：返回基于规则的简单反思
            return self._fallback_reflection(success, str(e))

    def _build_execution_summary(
        self,
        sub_goal: SubGoal,
        steps: list[ExecutionStep],
        success: bool,
        result_summary: str,
    ) -> str:
        """构建发给Critic的执行摘要"""
        # 统计工具调用情况
        total_calls = sum(len(s.tool_calls) for s in steps)
        failed_calls = sum(
            1 for s in steps
            for r in s.tool_results
            if r.status != ToolStatus.SUCCESS
        )
        total_tokens = sum(s.tokens_used for s in steps)

        # 工具调用历史（简化版，避免超Token）
        tool_history_lines = []
        for step in steps[-3:]:  # 只看最近3步，避免上下文过长
            for i, tc in enumerate(step.tool_calls):
                result = step.tool_results[i] if i < len(step.tool_results) else None
                status_icon = "✅" if (result and result.status == ToolStatus.SUCCESS) else "❌"
                args_preview = str(tc.arguments)[:60]
                output_preview = (result.output[:80] if result and result.output else "无输出")
                tool_history_lines.append(
                    f"  {status_icon} {tc.tool_name}({args_preview})\n"
                    f"     结果: {output_preview}"
                )

        tool_history = "\n".join(tool_history_lines) or "（无工具调用）"

        return f"""## 子目标
{sub_goal.description}

## 成功标准
{sub_goal.success_criteria or "（未指定）"}

## 执行结果
状态: {"✅ 成功" if success else "❌ 失败"}
摘要: {result_summary[:200]}

## 执行统计
- 总步骤数: {len(steps)}
- 工具调用次数: {total_calls}
- 失败调用次数: {failed_calls}
- Token消耗: {total_tokens}

## 最近工具调用记录
{tool_history}

请评估以上执行过程的质量。"""

    def _parse_reflection(self, content: str, success: bool) -> ReflectionResult:
        """解析Critic的JSON输出"""
        import json
        import re

        # 清理markdown包裹
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    return self._fallback_reflection(success, "JSON解析失败")
            else:
                return self._fallback_reflection(success, "无法提取JSON")

        return ReflectionResult(
            quality_score=float(data.get("quality_score", 0.7 if success else 0.3)),
            needs_replan=bool(data.get("needs_replan", not success)),
            reflection_text=data.get("reflection", ""),
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            is_fatal=bool(data.get("is_fatal", False)),
        )

    def _fallback_reflection(self, success: bool, reason: str) -> ReflectionResult:
        """
        基于规则的降级反思（Critic本身调用失败时使用）

        面试要点：
        "降级（Graceful Degradation）是生产级系统的必备设计。
         当Critic模块本身出错时，不能让整个Agent崩溃。
         基于规则的简单评估虽然不精确，
         但保证了系统的连续性。"
        """
        if success:
            return ReflectionResult(
                quality_score=0.7,
                needs_replan=False,
                reflection_text=f"子目标已完成（Critic降级，原因: {reason}）",
            )
        else:
            return ReflectionResult(
                quality_score=0.3,
                needs_replan=True,
                reflection_text=f"子目标失败，建议重规划（Critic降级，原因: {reason}）",
            )

    def quick_check(
        self,
        tool_name: str,
        error_message: str,
    ) -> str:
        """
        快速诊断单个工具调用失败的原因（轻量版，不走完整评估）

        使用场景：工具调用失败时，给Executor一个即时建议
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"工具 '{tool_name}' 调用失败，错误信息：\n{error_message}\n\n"
                    "请用一句话说明可能的原因和建议的修复方法。"
                )
            }
        ]
        try:
            response = self._llm.chat(
                messages=messages,
                tier=ModelTier.FAST,
                max_tokens=80,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return "工具调用失败，建议检查参数或换用其他工具"