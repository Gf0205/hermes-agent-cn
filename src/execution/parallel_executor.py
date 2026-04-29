"""
src/execution/parallel_executor.py - 工具并发执行器
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from src.execution.scheduler_v2 import ExecutionBatchPlanner
from src.models import ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """执行一批工具调用，在可安全并发时自动并发"""

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._planner = ExecutionBatchPlanner()

    def execute_parallel_tools(
        self,
        tool_calls: list[Any] | list[tuple[Any, str, dict[str, Any]]],
        execute_fn: Callable[[str, dict[str, Any]], ToolResult],
    ) -> list[tuple[Any, str, dict[str, Any], ToolResult]]:
        """
        执行一批工具调用并保持原始顺序

        Returns:
            list[(原始tool_call对象, tool_name, arguments, result)]
        """
        if self._is_prepared(tool_calls):
            normalized = tool_calls  # type: ignore[assignment]
        else:
            normalized = [self._normalize_tool_call(tc) for tc in tool_calls]  # type: ignore[arg-type]
        if not normalized:
            return []
        return self._run_with_batches(normalized, execute_fn)

    def _normalize_tool_call(
        self,
        tool_call: Any,
    ) -> tuple[Any, str, dict[str, Any]]:
        tool_name = tool_call.function.name
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        return tool_call, tool_name, arguments

    def _is_prepared(self, tool_calls: list[Any] | list[tuple[Any, str, dict[str, Any]]]) -> bool:
        if not tool_calls:
            return True
        first = tool_calls[0]
        return isinstance(first, tuple) and len(first) == 3

    def _run_parallel(
        self,
        batch: list[tuple[int, tuple[Any, str, dict[str, Any]]]],
        execute_fn: Callable[[str, dict[str, Any]], ToolResult],
    ) -> list[tuple[int, Any, str, dict[str, Any], ToolResult]]:
        indexed: list[tuple[int, Any, str, dict[str, Any], ToolResult]] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                (
                    idx,
                    tc,
                    name,
                    args,
                    pool.submit(execute_fn, name, args),
                )
                for idx, (tc, name, args) in batch
            ]
            for idx, tc, name, args, future in futures:
                indexed.append((idx, tc, name, args, future.result()))

        return indexed

    def _run_with_batches(
        self,
        normalized: list[tuple[Any, str, dict[str, Any]]],
        execute_fn: Callable[[str, dict[str, Any]], ToolResult],
    ) -> list[tuple[Any, str, dict[str, Any], ToolResult]]:
        planned = self._planner.plan_batches(normalized)
        all_results: list[tuple[int, Any, str, dict[str, Any], ToolResult]] = []
        logger.debug("工具调用分批执行: %s 批", len(planned))

        for batch in planned:
            if len(batch) == 1:
                idx, (tc, name, args) = batch[0]
                all_results.append((idx, tc, name, args, self._safe_execute(execute_fn, name, args)))
            else:
                try:
                    all_results.extend(self._run_parallel(batch, execute_fn))
                except Exception as e:
                    logger.warning(
                        "Parallel batch failed, fallback to sequential | batch_size=%s | error=%s",
                        len(batch),
                        e,
                    )
                    for idx, (tc, name, args) in batch:
                        all_results.append((idx, tc, name, args, self._safe_execute(execute_fn, name, args)))

        all_results.sort(key=lambda item: item[0])
        return [(tc, name, args, result) for _, tc, name, args, result in all_results]

    def _safe_execute(
        self,
        execute_fn: Callable[[str, dict[str, Any]], ToolResult],
        name: str,
        args: dict[str, Any],
    ) -> ToolResult:
        start = time.time()
        try:
            result = execute_fn(name, args)
            if isinstance(result, ToolResult):
                return result
            return ToolResult(
                tool_name=name,
                status=ToolStatus.FAILURE,
                output="",
                error=f"Unexpected execute_fn return type: {type(result).__name__}",
                execution_time_ms=(time.time() - start) * 1000.0,
            )
        except Exception as e:
            return ToolResult(
                tool_name=name,
                status=ToolStatus.FAILURE,
                output="",
                error=f"Parallel executor caught exception: {e}",
                execution_time_ms=(time.time() - start) * 1000.0,
            )
