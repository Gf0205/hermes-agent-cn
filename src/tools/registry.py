"""
src/tools/registry.py - 工具注册表

面试要点：
"工具注册表是我的工具发现系统。
 借鉴了Hermes原版的registry.py设计：
 '无依赖，被所有工具文件导入'。
 每个工具文件调用 registry.register() 完成注册，
 主程序只需要 registry.get_all_tools() 就能拿到所有工具。

 这是经典的Plugin Registry模式，
 新增工具时不需要修改任何现有文件。"

对比原版Hermes：
 Hermes用registry.register()装饰器
 我用register()方法直接调用
 效果相同，但更显式，IDE跳转更友好
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.models import ToolNotFoundError
from src.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    工具注册表 - 管理所有可用工具

    设计特点：
    - 单例模式（通过模块级实例实现）
    - 支持按名称查找、按标签过滤
    - 自动生成LLM工具列表（OpenAI格式）
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._tags: dict[str, list[str]] = {}  # 工具名 → 标签列表

    def register(
        self,
        tool: BaseTool,
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        注册工具

        Args:
            tool: 工具实例
            tags: 标签列表（如 ["file", "read-only"]）
        """
        if tool.name in self._tools:
            logger.warning(f"工具 '{tool.name}' 已存在，将被覆盖")

        self._tools[tool.name] = tool
        self._tags[tool.name] = tags or []
        logger.debug(f"注册工具: {tool.name}")

    def get(self, name: str) -> BaseTool:
        """
        按名称获取工具

        Raises:
            ToolNotFoundError: 工具不存在时
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def get_all(self) -> list[BaseTool]:
        """获取所有已注册工具"""
        return list(self._tools.values())

    def get_by_tags(self, tags: list[str]) -> list[BaseTool]:
        """按标签过滤工具"""
        result = []
        for name, tool in self._tools.items():
            tool_tags = self._tags.get(name, [])
            if any(tag in tool_tags for tag in tags):
                result.append(tool)
        return result

    def to_openai_tools(
        self,
        tool_names: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        生成OpenAI格式的工具列表（用于API调用）

        Args:
            tool_names: 指定工具名称，None表示所有工具

        Returns:
            OpenAI tool_choice格式的列表

        面试要点：
        "在每次LLM调用时，我把当前可用工具列表传给LLM。
         LLM根据任务自动选择合适的工具。
         我可以按任务类型动态过滤工具，
         比如只读任务就不传写入相关工具，减少LLM的选择困难。"
        """
        if tool_names:
            tools = [self._tools[n] for n in tool_names if n in self._tools]
        else:
            tools = list(self._tools.values())

        return [tool.to_openai_schema() for tool in tools]

    def list_tools(self) -> None:
        """打印所有工具信息（调试用）"""
        print(f"\n📦 已注册工具 ({len(self._tools)} 个):\n")
        for name, tool in self._tools.items():
            tags = self._tags.get(name, [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            print(f"  • {name}{tag_str}")
            print(f"    {tool.description[:80]}...")
        print()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ==============================================================================
# 全局注册表单例
# ==============================================================================

# 模块级单例，类似Hermes原版的registry.py设计
registry = ToolRegistry()