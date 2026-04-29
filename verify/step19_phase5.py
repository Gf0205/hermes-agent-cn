"""
Phase 5 P0 验证脚本（执行鲁棒性对齐）
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.execution.executor import Executor
from src.execution.parallel_executor import ParallelExecutor
from src.models import ToolResult, ToolStatus


class FakeLLM:
    def __init__(self) -> None:
        self.config = type("Cfg", (), {"strong_model": "fake-strong"})()


class FakeRegistry:
    def __init__(self, tool) -> None:  # type: ignore[no-untyped-def]
        self._tool = tool

    def get(self, tool_name: str):  # type: ignore[no-untyped-def]
        return self._tool

    def get_all(self) -> list[object]:
        return []


class TimeoutThenSuccessTool:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("simulated timeout")
        return ToolResult(
            tool_name="timeout_tool",
            status=ToolStatus.SUCCESS,
            output="ok-after-retry",
        )


class PermissionDeniedTool:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        raise PermissionError("permission denied")


class NonStandardResultTool:
    def execute(self, **kwargs):  # type: ignore[no-untyped-def]
        return {"raw": "dict-result"}


def test_parallel_fallback_to_sequential_on_batch_error() -> None:
    executor = ParallelExecutor(max_workers=4)

    def execute_fn(name: str, args: dict) -> ToolResult:
        if args.get("raise_error"):
            raise RuntimeError("simulated parallel failure")
        return ToolResult(tool_name=name, status=ToolStatus.SUCCESS, output=f"ok:{args.get('id')}")

    prepared_calls = [
        ({"id": "1"}, "read_file", {"path": "a.txt", "id": 1}),
        ({"id": "2"}, "read_file", {"path": "a.txt", "id": 2, "raise_error": True}),
        ({"id": "3"}, "read_file", {"path": "a.txt", "id": 3}),
    ]

    results = executor.execute_parallel_tools(prepared_calls, execute_fn)
    assert len(results) == 3
    statuses = [item[3].status for item in results]
    assert statuses.count(ToolStatus.SUCCESS) == 2
    assert statuses.count(ToolStatus.FAILURE) == 1


def test_timeout_retry_and_status_classification() -> None:
    tool = TimeoutThenSuccessTool()
    reg = FakeRegistry(tool)
    exe = Executor(llm_client=FakeLLM(), tool_registry=reg, max_iterations=1)  # type: ignore[arg-type]

    result = exe._execute_single_tool("timeout_tool", {})
    assert result.status == ToolStatus.SUCCESS
    assert result.metadata.get("attempts") == 2
    assert tool.calls == 2


def test_permission_denied_no_retry() -> None:
    tool = PermissionDeniedTool()
    reg = FakeRegistry(tool)
    exe = Executor(llm_client=FakeLLM(), tool_registry=reg, max_iterations=1)  # type: ignore[arg-type]

    result = exe._execute_single_tool("permission_tool", {})
    assert result.status == ToolStatus.PERMISSION_DENIED
    assert result.metadata.get("attempts") == 1
    assert tool.calls == 1


def test_nonstandard_tool_result_is_coerced() -> None:
    tool = NonStandardResultTool()
    reg = FakeRegistry(tool)
    exe = Executor(llm_client=FakeLLM(), tool_registry=reg, max_iterations=1)  # type: ignore[arg-type]

    result = exe._execute_single_tool("nonstandard_tool", {})
    assert result.status == ToolStatus.SUCCESS
    assert isinstance(result.output, str)
    assert result.metadata.get("coerced_result") is True


def main() -> None:
    test_parallel_fallback_to_sequential_on_batch_error()
    print("[PASS] test_parallel_fallback_to_sequential_on_batch_error")
    test_timeout_retry_and_status_classification()
    print("[PASS] test_timeout_retry_and_status_classification")
    test_permission_denied_no_retry()
    print("[PASS] test_permission_denied_no_retry")
    test_nonstandard_tool_result_is_coerced()
    print("[PASS] test_nonstandard_tool_result_is_coerced")
    print("[DONE] Phase 5 P0 execution robustness checks passed")


if __name__ == "__main__":
    main()
