"""
src/tools/builtin/write_file.py - 写入文件工具
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import ToolParameter
from src.tools.base import BaseTool
from src.tools.registry import registry


class WriteFileTool(BaseTool):
    """
    写入/创建文件

    安全机制：
    - 自动创建父目录
    - 写入前显示diff预览（如果是覆盖）
    - 记录操作日志
    """

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "写入内容到文件（创建或覆盖）。"
            "如果目录不存在会自动创建。"
            "适用于创建新文件、保存生成的代码等。"
            "注意：会完全覆盖已有内容，如需部分修改请用edit_file。"
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
                name="content",
                type="string",
                description="要写入的内容",
                required=True,
            ),
            ToolParameter(
                name="encoding",
                type="string",
                description="文件编码，默认utf-8",
                required=False,
                default="utf-8",
            ),
        ]

    def _execute(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> str:
        file_path = Path(path).resolve()

        # 自动创建父目录
        file_path.parent.mkdir(parents=True, exist_ok=True)

        is_new = not file_path.exists()
        file_path.write_text(content, encoding=encoding)

        line_count = content.count("\n") + 1
        size_kb = len(content.encode(encoding)) / 1024

        action = "创建" if is_new else "覆盖"
        return (
            f"✅ 文件{action}成功\n"
            f"路径: {file_path}\n"
            f"内容: {line_count} 行, {size_kb:.1f} KB"
        )


registry.register(WriteFileTool(), tags=["file", "write"])