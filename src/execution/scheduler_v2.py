"""
src/execution/scheduler_v2.py - 冲突感知工具调度器
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass
class ToolAccess:
    tool_name: str
    mode: AccessMode
    resource: str


class ToolAccessAnalyzer:
    """根据工具名与参数推断读写模式与资源路径"""

    READ_TOOLS = {"read_file", "grep_search", "list_dir"}
    WRITE_TOOLS = {"write_file", "edit_file"}

    def analyze(self, tool_name: str, arguments: dict[str, Any]) -> ToolAccess:
        if tool_name in self.READ_TOOLS:
            return ToolAccess(
                tool_name=tool_name,
                mode=AccessMode.READ,
                resource=self._normalize_path(arguments.get("path", ".")),
            )
        if tool_name in self.WRITE_TOOLS:
            return ToolAccess(
                tool_name=tool_name,
                mode=AccessMode.WRITE,
                resource=self._normalize_path(arguments.get("path", ".")),
            )
        if tool_name == "shell":
            return ToolAccess(tool_name=tool_name, mode=AccessMode.WRITE, resource="*")
        # 保守策略：未知工具按全局写处理
        return ToolAccess(tool_name=tool_name, mode=AccessMode.WRITE, resource="*")

    def _normalize_path(self, path: Any) -> str:
        try:
            return str(Path(str(path)).resolve())
        except Exception:
            return str(path or ".")


class ExecutionBatchPlanner:
    """把一组工具调用按冲突关系切分为并发批次"""

    def __init__(self, analyzer: ToolAccessAnalyzer | None = None) -> None:
        self._analyzer = analyzer or ToolAccessAnalyzer()

    def plan_batches(
        self,
        calls: list[tuple[Any, str, dict[str, Any]]],
    ) -> list[list[tuple[int, tuple[Any, str, dict[str, Any]]]]]:
        batches: list[list[tuple[int, tuple[Any, str, dict[str, Any]]]]] = []
        current: list[tuple[int, tuple[Any, str, dict[str, Any]]]] = []
        current_accesses: list[ToolAccess] = []

        for idx, call in enumerate(calls):
            _, tool_name, arguments = call
            access = self._analyzer.analyze(tool_name, arguments)

            if current and self._has_conflict(access, current_accesses):
                batches.append(current)
                current = []
                current_accesses = []

            current.append((idx, call))
            current_accesses.append(access)

        if current:
            batches.append(current)

        return batches

    def _has_conflict(self, target: ToolAccess, existing: list[ToolAccess]) -> bool:
        for access in existing:
            if self._resource_conflict(target.resource, access.resource):
                # 同资源读读不冲突，其它都冲突
                if target.mode == AccessMode.READ and access.mode == AccessMode.READ:
                    continue
                return True
        return False

    def _resource_conflict(self, a: str, b: str) -> bool:
        if a == "*" or b == "*":
            return True
        return a == b
