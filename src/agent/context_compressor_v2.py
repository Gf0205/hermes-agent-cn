"""
src/agent/context_compressor_v2.py - 上下文压缩器 v2
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.llm_client import LLMClient
from src.models import ModelTier

logger = logging.getLogger(__name__)


class ContextCompressorV2:
    """
    上下文压缩策略：
    - 50% 预压缩：轻量摘要，保留更多 recent window
    - 75% 强压缩：深度摘要，缩短 recent window
    """

    def __init__(
        self,
        llm_client: LLMClient,
        precompress_ratio: float = 0.5,
        hardcompress_ratio: float = 0.75,
        ineffective_savings_threshold: float = 0.10,
        ineffective_limit: int = 2,
        summary_failure_cooldown_s: float = 20.0,
    ) -> None:
        self._llm = llm_client
        self._precompress_ratio = precompress_ratio
        self._hardcompress_ratio = hardcompress_ratio
        self._ineffective_savings_threshold = max(0.0, min(0.9, ineffective_savings_threshold))
        self._ineffective_limit = max(1, ineffective_limit)
        self._summary_failure_cooldown_s = max(1.0, summary_failure_cooldown_s)
        self._summary_version = 0
        self._compression_count = 0
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0
        self._skipped_due_to_thrashing = 0
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error = ""
        self._last_summary_fallback_used = False

    def compress(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        focus_topic: str | None = None,
    ) -> list[dict[str, Any]]:
        if len(messages) < 8:
            return messages

        estimated = self.estimate_tokens(messages)
        if estimated <= int(max_tokens * self._precompress_ratio):
            return messages

        if self._ineffective_compression_count >= self._ineffective_limit:
            self._skipped_due_to_thrashing += 1
            logger.warning(
                "Context v2 compression skipped due to anti-thrashing | ineffective_count=%s",
                self._ineffective_compression_count,
            )
            return messages

        self._compression_count += 1
        self._summary_version += 1
        hard_mode = estimated > int(max_tokens * self._hardcompress_ratio)

        pinned = self._build_pinned_facts(messages, focus_topic=focus_topic)
        keep_tail = 5 if hard_mode else 8
        head = messages[:1]
        tail = messages[-keep_tail:]
        middle = messages[1:-keep_tail]
        if not middle:
            return messages

        summary = self._summarize(middle, hard_mode=hard_mode, focus_topic=focus_topic)
        compressed = [
            *head,
            {
                "role": "system",
                "content": (
                    f"[ContextSummary v{self._summary_version}] "
                    f"compressions={self._compression_count}\n"
                    f"PinnedFacts:\n{pinned}\n\n"
                    f"RollingSummary:\n{summary}"
                ),
            },
            *tail,
        ]
        compressed_tokens = self.estimate_tokens(compressed)
        savings_pct = ((estimated - compressed_tokens) / max(1, estimated)) * 100.0
        self._last_compression_savings_pct = round(savings_pct, 2)
        if savings_pct < self._ineffective_savings_threshold * 100:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        logger.info(
            "Context v2 compression completed | hard_mode=%s | tokens=%s -> %s | savings=%.2f%%",
            hard_mode,
            estimated,
            compressed_tokens,
            savings_pct,
        )
        return compressed

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        payload = json.dumps(messages, ensure_ascii=False, default=str)
        return max(1, int(len(payload) / 3))

    def _build_pinned_facts(
        self,
        messages: list[dict[str, Any]],
        focus_topic: str | None = None,
    ) -> str:
        # 固定保真块：首条 system + 首条 user + 最近工具失败摘要
        system_content = ""
        user_goal = ""
        recent_failures: list[str] = []

        for message in messages:
            role = message.get("role", "")
            content = str(message.get("content", "")).strip()
            if role == "system" and not system_content and content:
                system_content = content[:500]
            if role == "user" and not user_goal and content:
                user_goal = content[:400]
            if role == "tool" and "工具执行失败" in content:
                recent_failures.append(content[:200])

        lines = [
            f"- GoalHint: {user_goal or 'N/A'}",
            f"- SystemConstraint: {system_content or 'N/A'}",
        ]
        if focus_topic and focus_topic.strip():
            lines.append(f"- FocusTopic: {focus_topic.strip()[:200]}")
        if recent_failures:
            lines.append("- RecentFailures:")
            lines.extend([f"  - {item}" for item in recent_failures[-3:]])
        return "\n".join(lines)

    def _summarize(
        self,
        middle: list[dict[str, Any]],
        hard_mode: bool,
        focus_topic: str | None = None,
    ) -> str:
        if self._is_summary_cooldown_active():
            self._last_summary_fallback_used = True
            return self._fallback_summary(
                middle=middle,
                hard_mode=hard_mode,
                reason="summary-cooldown",
                focus_topic=focus_topic,
            )

        mode_text = "深度摘要，优先保留决策链和未完成事项" if hard_mode else "轻量摘要，保留关键操作轨迹"
        prompt = (
            "请压缩以下历史消息，输出中文要点：\n"
            f"模式：{mode_text}\n"
            "必须包含：已完成动作、失败及原因、当前待办。\n"
            "输出最多12条要点。"
        )
        if focus_topic and focus_topic.strip():
            prompt += (
                f"\n重点保留与该主题相关的具体信息：{focus_topic.strip()[:200]}。"
                "非该主题的信息可以更简略。"
            )
        try:
            response = self._llm.chat(
                messages=[
                    {"role": "system", "content": "你是上下文压缩器，只输出摘要结果。"},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": json.dumps(middle, ensure_ascii=False, default=str)},
                ],
                tier=ModelTier.FAST,
                temperature=0.2,
                max_tokens=600 if hard_mode else 450,
            )
            self._last_summary_error = ""
            self._last_summary_fallback_used = False
            self._summary_failure_cooldown_until = 0.0
            return response.choices[0].message.content or "（摘要为空）"
        except Exception as e:
            logger.warning("Context v2 summary generation failed: %s", e)
            self._last_summary_error = str(e)
            self._last_summary_fallback_used = True
            self._summary_failure_cooldown_until = (
                time.monotonic() + self._summary_failure_cooldown_s
            )
            return self._fallback_summary(
                middle=middle,
                hard_mode=hard_mode,
                reason=f"summary-error:{str(e)[:120]}",
                focus_topic=focus_topic,
            )

    def _is_summary_cooldown_active(self) -> bool:
        return time.monotonic() < self._summary_failure_cooldown_until

    def _fallback_summary(
        self,
        middle: list[dict[str, Any]],
        hard_mode: bool,
        reason: str,
        focus_topic: str | None = None,
    ) -> str:
        window = 8 if hard_mode else 10
        trimmed = middle[-window:]
        parts: list[str] = [
            f"[fallback reason={reason}]",
            "最近上下文关键片段（简化保留）:",
        ]
        if focus_topic and focus_topic.strip():
            parts.append(f"Focus topic: {focus_topic.strip()[:120]}")
        for msg in trimmed:
            role = str(msg.get("role", "unknown"))
            content = str(msg.get("content", "")).replace("\n", " ").strip()
            if content:
                parts.append(f"- {role}: {content[:180]}")
        return "\n".join(parts)[:2000]

    def get_health_metrics(self) -> dict[str, Any]:
        now = time.monotonic()
        cooldown_remaining = max(0.0, self._summary_failure_cooldown_until - now)
        return {
            "compression_count": self._compression_count,
            "summary_version": self._summary_version,
            "last_compression_savings_pct": self._last_compression_savings_pct,
            "ineffective_compression_count": self._ineffective_compression_count,
            "skipped_due_to_thrashing": self._skipped_due_to_thrashing,
            "summary_cooldown_active": cooldown_remaining > 0,
            "summary_cooldown_remaining_s": round(cooldown_remaining, 3),
            "last_summary_error": self._last_summary_error,
            "last_summary_fallback_used": self._last_summary_fallback_used,
            "ineffective_limit": self._ineffective_limit,
        }
