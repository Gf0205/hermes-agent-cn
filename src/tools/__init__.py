# src/tools/__init__.py
"""工具系统 - 自动发现并注册内置工具"""

# 导入内置工具模块（触发各模块的registry.register()调用）
from src.tools.builtin import edit_file, grep_search, list_dir, read_file, shell, write_file

__all__ = ["read_file", "write_file", "edit_file", "shell", "grep_search", "list_dir"]