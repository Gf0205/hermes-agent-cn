"""
src/agent/context_compressor.py - 上下文压缩器
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.llm_client import LLMClient
from src.models import ModelTier

logger = logging.getLogger(__name__)


class ContextCompressor:
    """在上下文接近上限时压缩中间历史消息"""

    def __init__(
        self,
        llm_client: LLMClient,
        trigger_ratio: float = 0.6,
        keep_tail_count: int = 5,
    ) -> None:
        self._llm = llm_client
        self._trigger_ratio = trigger_ratio
        self._keep_tail_count = keep_tail_count

    def compress(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages

        estimated = self.estimate_tokens(messages)
        if estimated <= int(max_tokens * self._trigger_ratio):
            return messages

        if len(messages) <= self._keep_tail_count + 1:
            return messages

        head = messages[0]
        tail = messages[-self._keep_tail_count:]
        middle = messages[1:-self._keep_tail_count]
        if not middle:
            return messages

        summary = self._summarize_middle(middle)
        compressed = [
            head,
            {
                "role": "system",
                "content": (
                    "以下是已压缩的历史执行上下文摘要（有损）：\n"
                    f"{summary}"
                ),
            },
            *tail,
        ]
        logger.info(
            "上下文已压缩 | 估算Token: %s -> %s",
            estimated,
            self.estimate_tokens(compressed),
        )
        return compressed

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        payload = json.dumps(messages, ensure_ascii=False, default=str)
        return max(1, int(len(payload) / 3))

    def _summarize_middle(self, middle: list[dict[str, Any]]) -> str:
        prompt = (
            "请将以下对话中间历史做结构化摘要，保留：\n"
            "1) 已完成的关键动作\n"
            "2) 失败尝试及原因\n"
            "3) 当前重要约束（文件路径、接口名、未完成事项）\n"
            "输出尽量精炼，使用中文要点。"
        )
        try:
            response = self._llm.chat(
                messages=[
                    {"role": "system", "content": "你是上下文压缩器，只输出摘要。"},
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": json.dumps(middle, ensure_ascii=False)},
                ],
                tier=ModelTier.FAST,
                temperature=0.2,
                max_tokens=500,
            )
            return response.choices[0].message.content or "（摘要为空）"
        except Exception as e:
            logger.warning(f"上下文压缩失败，回退到截断摘要: {e}")
            fallback = json.dumps(middle[-8:], ensure_ascii=False, default=str)
            return f"压缩失败，保留最近中间消息片段：{fallback[:1200]}"
