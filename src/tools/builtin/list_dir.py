"""
src/tools/builtin/list_dir.py - 目录列表工具
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import ToolParameter
from src.tools.base import BaseTool
from src.tools.registry import registry


class ListDirTool(BaseTool):
    """列出目录内容（类似tree命令）"""

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return (
            "列出目录中的文件和子目录（树状结构）。"
            "适用于了解项目结构、查找文件位置。"
            "忽略隐藏文件和常见的不重要目录（__pycache__、.git等）。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="path",
                type="string",
                description="要列出的目录路径，默认为当前目录",
                required=False,
                default=".",
            ),
            ToolParameter(
                name="depth",
                type="integer",
                description="递归深度，默认2层",
                required=False,
                default=2,
            ),
            ToolParameter(
                name="show_hidden",
                type="boolean",
                description="是否显示隐藏文件（以.开头），默认False",
                required=False,
                default=False,
            ),
        ]

    def _execute(
        self,
        path: str = ".",
        depth: int = 2,
        show_hidden: bool = False,
        **kwargs: Any,
    ) -> str:
        root = Path(path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"目录不存在: {path}")
        if not root.is_dir():
            raise ValueError(f"路径不是目录: {path}")

        # 忽略目录
        skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".idea", ".mypy_cache"}

        lines = [f"📁 {root}/"]
        self._tree(root, "", depth, 0, show_hidden, skip_dirs, lines)

        lines.append(f"\n（显示深度: {depth}层）")
        return "\n".join(lines)

    def _tree(
        self,
        directory: Path,
        prefix: str,
        max_depth: int,
        current_depth: int,
        show_hidden: bool,
        skip_dirs: set[str],
        lines: list[str],
    ) -> None:
        if current_depth >= max_depth:
            return

        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            lines.append(f"{prefix}└── [权限不足]")
            return

        # 过滤
        entries = [
            e for e in entries
            if (show_hidden or not e.name.startswith("."))
            and e.name not in skip_dirs
        ]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            icon = "📄 " if entry.is_file() else "📁 "

            if entry.is_file():
                size = entry.stat().st_size
                size_str = f" ({size/1024:.1f}KB)" if size > 1024 else f" ({size}B)"
                lines.append(f"{prefix}{connector}{icon}{entry.name}{size_str}")
            else:
                lines.append(f"{prefix}{connector}{icon}{entry.name}/")
                extension = "    " if is_last else "│   "
                self._tree(
                    entry, prefix + extension,
                    max_depth, current_depth + 1,
                    show_hidden, skip_dirs, lines
                )


registry.register(ListDirTool(), tags=["file", "read-only", "navigation"])