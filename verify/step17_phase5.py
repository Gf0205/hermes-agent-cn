"""
Phase 5 P0 验证脚本（ContextCompressorV2 鲁棒性升级）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.context_compressor_v2 import ContextCompressorV2

# step17 intentionally injects summary failures; silence expected warning logs
# so regression output stays clean on Windows terminals.
logging.getLogger("src.agent.context_compressor_v2").setLevel(logging.ERROR)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class FakeLLMSuccess:
    def __init__(self, summary_text: str = "压缩摘要") -> None:
        self.summary_text = summary_text
        self.calls = 0
        self.last_messages: list[dict] = []

    def chat(self, messages: list[dict], **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.last_messages = messages
        return _FakeResponse(self.summary_text)


class FakeLLMFail:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict], **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise RuntimeError("simulated summary failure")


def _build_messages() -> list[dict]:
    msgs: list[dict] = [
        {"role": "system", "content": "你是执行代理，遵循安全约束。"},
        {"role": "user", "content": "请继续修复数据库连接问题，并补测试。"},
    ]
    for i in range(1, 10):
        role = "assistant" if i % 2 else "tool"
        msgs.append(
            {
                "role": role,
                "content": f"message-{i} " + ("x" * 260),
            }
        )
    return msgs


def test_anti_thrashing_skip_after_repeated_ineffective() -> None:
    llm = FakeLLMSuccess(summary_text="冗长摘要 " + ("y" * 1800))
    compressor = ContextCompressorV2(
        llm_client=llm,  # type: ignore[arg-type]
        ineffective_savings_threshold=0.95,
        ineffective_limit=2,
    )
    messages = _build_messages()
    max_tokens = 1200

    _ = compressor.compress(messages, max_tokens=max_tokens)
    _ = compressor.compress(messages, max_tokens=max_tokens)
    third = compressor.compress(messages, max_tokens=max_tokens)
    metrics = compressor.get_health_metrics()

    assert third == messages
    assert metrics["skipped_due_to_thrashing"] >= 1
    assert metrics["ineffective_compression_count"] >= 2


def test_summary_failure_cooldown_blocks_repeat_llm_calls() -> None:
    llm = FakeLLMFail()
    compressor = ContextCompressorV2(
        llm_client=llm,  # type: ignore[arg-type]
        summary_failure_cooldown_s=60.0,
        ineffective_limit=99,
    )
    messages = _build_messages()
    max_tokens = 1200

    out1 = compressor.compress(messages, max_tokens=max_tokens)
    calls_after_first = llm.calls
    out2 = compressor.compress(messages, max_tokens=max_tokens)
    metrics = compressor.get_health_metrics()

    assert calls_after_first == 1
    assert llm.calls == 1
    assert out1 != messages
    assert out2 != messages
    assert metrics["summary_cooldown_active"] is True
    assert metrics["last_summary_error"] != ""
    assert metrics["last_summary_fallback_used"] is True


def test_focus_topic_propagates_to_summary_and_pinned_facts() -> None:
    llm = FakeLLMSuccess(summary_text="重点保留数据库连接池优化记录")
    compressor = ContextCompressorV2(
        llm_client=llm,  # type: ignore[arg-type]
        ineffective_limit=99,
    )
    messages = _build_messages()
    max_tokens = 1200
    focus = "数据库连接池"

    out = compressor.compress(messages, max_tokens=max_tokens, focus_topic=focus)
    system_summaries = [
        str(msg.get("content", ""))
        for msg in out
        if msg.get("role") == "system" and "[ContextSummary" in str(msg.get("content", ""))
    ]

    assert system_summaries
    assert "FocusTopic: 数据库连接池" in system_summaries[0]
    assert llm.last_messages
    assert focus in str(llm.last_messages[1].get("content", ""))


def main() -> None:
    test_anti_thrashing_skip_after_repeated_ineffective()
    print("[PASS] test_anti_thrashing_skip_after_repeated_ineffective")
    test_summary_failure_cooldown_blocks_repeat_llm_calls()
    print("[PASS] test_summary_failure_cooldown_blocks_repeat_llm_calls")
    test_focus_topic_propagates_to_summary_and_pinned_facts()
    print("[PASS] test_focus_topic_propagates_to_summary_and_pinned_facts")
    print("[DONE] Phase 5 P0 ContextCompressorV2 resilience checks passed")


if __name__ == "__main__":
    main()
