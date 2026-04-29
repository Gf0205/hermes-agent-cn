"""
src/tools/builtin/grep_search.py - 文本搜索工具

面试要点：
"grep_search是Agent定位代码的核心工具。
 我实现了两种搜索：
 1. 精确字符串匹配（快速）
 2. 正则表达式匹配（灵活）

 面试中可以说：
 '用Python的re模块实现，
  支持多文件递归搜索，
  返回文件名+行号+上下文，
  让LLM能精确定位到需要修改的代码行。'"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.models import ToolParameter
from src.tools.base import BaseTool
from src.tools.registry import registry


class GrepSearchTool(BaseTool):
    """在文件中搜索文本模式"""

    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return (
            "在文件或目录中搜索文本模式（支持正则表达式）。"
            "返回匹配的文件路径、行号和内容。"
            "适用于查找函数定义、变量使用、错误信息等。"
            "例如：搜索 'def train' 找所有训练函数定义。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="pattern",
                type="string",
                description="搜索模式（字符串或正则表达式）",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="搜索路径（文件或目录），默认为当前目录",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="file_pattern",
                type="string",
                description="文件名过滤模式（如 '*.py'），默认搜索所有文件",
                required=False,
                default="*",
            ),
            ToolParameter(
                name="use_regex",
                type="boolean",
                description="是否使用正则表达式，默认False（普通字符串搜索）",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="context_lines",
                type="integer",
                description="显示匹配行前后的上下文行数，默认2",
                required=False,
                default=2,
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="最多返回结果数，默认50",
                required=False,
                default=50,
            ),
        ]

    def _execute(
            self,
            pattern: str,
            path: str = ".",
            file_pattern: str = "*",
            use_regex: bool = False,
            context_lines: int = 2,
            max_results: int = 50,
            **kwargs: Any,
    ) -> str:
        search_path = Path(path).resolve()

        if not search_path.exists():
            raise FileNotFoundError(f"路径不存在: {path}")

        # 编译搜索模式
        if use_regex:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                raise ValueError(f"无效的正则表达式: {e}")
        else:
            # 普通字符串搜索：转义为正则
            compiled = re.compile(re.escape(pattern), re.IGNORECASE)

        # 收集要搜索的文件
        if search_path.is_file():
            files = [search_path]
        else:
            files = list(search_path.rglob(file_pattern))
            # 过滤常见不需要搜索的目录
            skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".hermes-cn"}
            files = [
                f for f in files
                if f.is_file() and not any(part in skip_dirs for part in f.parts)
            ]

        # 执行搜索
        results: list[str] = []
        total_matches = 0

        for file_path in sorted(files):
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
            except Exception:
                continue

            file_matches = []
            for line_idx, line in enumerate(lines):
                if compiled.search(line):
                    # 提取上下文
                    start = max(0, line_idx - context_lines)
                    end = min(len(lines), line_idx + context_lines + 1)

                    context = []
                    for ctx_idx in range(start, end):
                        prefix = "▶ " if ctx_idx == line_idx else "  "
                        context.append(f"  {ctx_idx + 1:4d}{prefix}{lines[ctx_idx]}")

                    file_matches.append("\n".join(context))
                    total_matches += 1

                    if total_matches >= max_results:
                        break

            if file_matches:
                rel_path = file_path.relative_to(search_path) if search_path.is_dir() else file_path
                results.append(f"\n📄 {rel_path} ({len(file_matches)} 处匹配):\n")
                results.extend(file_matches)

            if total_matches >= max_results:
                break

        if not results:
            return f"未找到匹配 '{pattern}' 的内容（搜索了 {len(files)} 个文件）"

        header = (
                f"搜索结果: '{pattern}'\n"
                f"共找到 {total_matches} 处匹配"
                + (f"（已达上限{max_results}条）" if total_matches >= max_results else "")
        )

        return header + "\n" + "─" * 60 + "\n".join(results)


registry.register(GrepSearchTool(), tags=["file", "search", "read-only"])