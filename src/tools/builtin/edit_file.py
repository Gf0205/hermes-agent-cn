"""
src/tools/builtin/edit_file.py - 行级编辑工具
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from src.models import ToolParameter
from src.tools.base import BaseTool
from src.tools.registry import registry


class EditFileTool(BaseTool):
    """对已有文件执行行级编辑（replace/insert/delete）"""

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "对现有文件做行级编辑。"
            "支持 replace（替换行范围）、insert（在指定行后插入）、delete（删除行范围）。"
            "适用于精准修改大文件，避免整文件覆盖。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="目标文件路径",
                required=True,
            ),
            ToolParameter(
                name="operation",
                type="string",
                description="操作类型：replace/insert/delete",
                required=True,
                enum_values=["replace", "insert", "delete"],
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="起始行号（1-indexed）；insert 时表示在该行后插入",
                required=True,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="结束行号（replace/delete 必填，insert 可不填）",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="new_content",
                type="string",
                description="新内容（replace/insert 需要，delete 可为空）",
                required=False,
                default="",
            ),
        ]

    def _execute(
        self,
        path: str,
        operation: str,
        start_line: int,
        end_line: int | None = None,
        new_content: str = "",
        **kwargs: Any,
    ) -> str:
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if not file_path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        text = self._read_text(file_path)
        old_lines = text.splitlines()
        new_lines = old_lines.copy()

        total_lines = len(old_lines)
        self._validate_line_range(operation, start_line, end_line, total_lines)

        content_lines = new_content.splitlines()

        if operation == "replace":
            assert end_line is not None
            new_lines[start_line - 1:end_line] = content_lines
            changed_desc = f"替换第 {start_line}-{end_line} 行"
        elif operation == "insert":
            insert_idx = start_line
            new_lines[insert_idx:insert_idx] = content_lines
            changed_desc = f"在第 {start_line} 行后插入 {len(content_lines)} 行"
        elif operation == "delete":
            assert end_line is not None
            del new_lines[start_line - 1:end_line]
            changed_desc = f"删除第 {start_line}-{end_line} 行"
        else:
            raise ValueError(f"不支持的 operation: {operation}")

        result_text = "\n".join(new_lines)
        if text.endswith("\n") and result_text:
            result_text += "\n"

        file_path.write_text(result_text, encoding="utf-8")

        diff_preview = self._build_diff_preview(old_lines, new_lines, str(file_path))
        return (
            f"✅ edit_file 执行成功\n"
            f"文件: {file_path}\n"
            f"变更: {changed_desc}\n"
            f"原始行数: {len(old_lines)} -> 新行数: {len(new_lines)}\n"
            f"Diff摘要:\n{diff_preview}"
        )

    def _read_text(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="gbk", errors="replace")

    def _validate_line_range(
        self,
        operation: str,
        start_line: int,
        end_line: int | None,
        total_lines: int,
    ) -> None:
        if start_line < 1:
            raise ValueError("start_line 必须 >= 1")

        if operation in {"replace", "delete"}:
            if end_line is None:
                raise ValueError(f"{operation} 操作需要提供 end_line")
            if end_line < start_line:
                raise ValueError("end_line 不能小于 start_line")
            if start_line > total_lines or end_line > total_lines:
                raise ValueError(
                    f"行号越界：文件总行数 {total_lines}，请求范围 {start_line}-{end_line}"
                )
        elif operation == "insert":
            if start_line > total_lines:
                raise ValueError(
                    f"insert 行号越界：文件总行数 {total_lines}，start_line={start_line}"
                )

    def _build_diff_preview(
        self,
        before: list[str],
        after: list[str],
        file_label: str,
        max_lines: int = 30,
    ) -> str:
        diff_lines = list(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"{file_label} (before)",
                tofile=f"{file_label} (after)",
                lineterm="",
            )
        )
        if not diff_lines:
            return "（内容无变化）"
        if len(diff_lines) > max_lines:
            return "\n".join(diff_lines[:max_lines]) + "\n...（diff已截断）"
        return "\n".join(diff_lines)


registry.register(EditFileTool(), tags=["file", "write", "line-edit"])
