"""
src/tools/builtin/read_file.py - 读取文件工具

面试要点：
"读文件是Agent最基础的能力。
 我的实现有几个安全措施：
 1. 路径规范化（防止路径遍历攻击）
 2. 文件大小限制（防止撑爆上下文窗口）
 3. 编码自动检测（处理中文文件不乱码）
 4. 行号显示（方便后续的edit_file精确定位）"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.models import ToolParameter
from src.tools.base import BaseTool
from src.tools.registry import registry


class ReadFileTool(BaseTool):
    """
    读取文件内容

    设计决策：
    返回带行号的内容，方便LLM在后续的edit_file中精确定位要修改的行。
    类似 `cat -n` 命令的输出。
    """

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取文件内容并返回。"
            "适用于查看代码、配置文件、日志等文本文件。"
            "返回内容包含行号，方便精确定位。"
            "如果文件过大（>1MB），只返回前500行。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="文件路径（绝对路径或相对于工作目录的路径）",
                required=True,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="起始行号（1-indexed），默认从第1行开始",
                required=False,
                default=1,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="结束行号，默认读到文件末尾",
                required=False,
                default=None,
            ),
        ]

    def _execute(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        **kwargs: Any,
    ) -> str:
        file_path = Path(path).resolve()

        # 安全检查：文件必须存在
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        if not file_path.is_file():
            raise ValueError(f"路径不是文件: {path}")

        # 安全检查：文件大小限制（1MB）
        file_size = file_path.stat().st_size
        max_size = 1 * 1024 * 1024  # 1MB
        size_warning = ""
        if file_size > max_size:
            size_warning = f"\n⚠️ 文件过大（{file_size/1024:.0f}KB），仅显示前500行\n"

        # 读取文件
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 尝试GBK（常见于中文Windows环境）
            content = file_path.read_text(encoding="gbk", errors="replace")

        lines = content.splitlines()
        total_lines = len(lines)

        # 行范围处理
        start_idx = max(0, (start_line or 1) - 1)
        end_idx = min(total_lines, (end_line or total_lines))

        # 大文件截断
        if file_size > max_size:
            end_idx = min(end_idx, 500)

        # 生成带行号的输出
        selected_lines = lines[start_idx:end_idx]
        width = len(str(end_idx))  # 行号对齐宽度

        numbered_lines = [
            f"{start_idx + i + 1:{width}d} │ {line}"
            for i, line in enumerate(selected_lines)
        ]

        output = "\n".join([
            f"文件: {file_path}",
            f"总行数: {total_lines} | 显示: {start_idx+1}-{end_idx}",
            size_warning,
            "─" * 60,
            "\n".join(numbered_lines),
            "─" * 60,
        ])

        return output


# 注册工具（模块导入时自动执行）
registry.register(ReadFileTool(), tags=["file", "read-only"])