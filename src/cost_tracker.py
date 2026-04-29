"""
src/cost_tracker.py - LLM 成本统计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.event_bus import Event, EventType, get_event_bus


@dataclass
class CostStat:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class CostTracker:
    """订阅 LLM_RESPONSE 事件，累计 Token 与费用"""

    def __init__(self) -> None:
        self._bus = get_event_bus()
        self._session_total = CostStat()
        self._global_total = CostStat()
        self._by_model: dict[str, CostStat] = {}
        self._bus.subscribe(EventType.LLM_RESPONSE, self._on_llm_response)

    def _on_llm_response(self, event: Event) -> None:
        model = str(event.data.get("model", "unknown"))
        total_tokens = int(event.data.get("tokens", 0))
        cost_usd = float(event.data.get("estimated_cost_usd", 0.0))

        stat = self._by_model.setdefault(model, CostStat())
        stat.total_tokens += total_tokens
        stat.cost_usd += cost_usd

        self._session_total.total_tokens += total_tokens
        self._session_total.cost_usd += cost_usd

        self._global_total.total_tokens += total_tokens
        self._global_total.cost_usd += cost_usd

    def get_session_summary(self) -> dict[str, Any]:
        return {
            "session_tokens": self._session_total.total_tokens,
            "session_cost_usd": round(self._session_total.cost_usd, 6),
            "models": {
                model: {
                    "tokens": stat.total_tokens,
                    "cost_usd": round(stat.cost_usd, 6),
                }
                for model, stat in sorted(self._by_model.items())
            },
        }

    def get_global_summary(self) -> dict[str, Any]:
        return {
            "global_tokens": self._global_total.total_tokens,
            "global_cost_usd": round(self._global_total.cost_usd, 6),
        }
