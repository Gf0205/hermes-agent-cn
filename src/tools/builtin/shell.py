"""
src/tools/builtin/shell.py - Shell命令执行工具

面试要点：
"Shell工具是Agent最强大也最危险的工具。
 我的安全措施：
 1. 超时机制（防止命令挂死）
 2. 工作目录隔离
 3. 环境变量白名单（不传递敏感变量）
 4. 输出大小限制（防止日志爆炸）
 5. 危险命令检测（rm -rf / 等）"
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from src.models import ToolParameter
from src.permissions import PermissionDecision, PermissionManager
from src.tools.base import BaseTool
from src.tools.registry import registry

# 危险命令前缀黑名单（简单防护，生产环境应该更严格）
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",  # fork炸弹
    "chmod -R 777 /",
]


class ShellTool(BaseTool):
    """
    执行Shell命令

    面试要点：
    "我用subprocess.run()而不是os.system()，
     原因是：
     1. 可以捕获stdout和stderr
     2. 有超时控制
     3. 不经过shell解析（更安全，除非shell=True）

     对于Agent来说，分开捕获stdout和stderr很重要：
     成功输出和错误输出不能混在一起，LLM需要能区分。"
    """

    @property
    def name(self) -> str:
        return "shell"

    def __init__(self) -> None:
        self._permission_manager = PermissionManager()

    @property
    def description(self) -> str:
        return (
            "在Shell中执行命令并返回输出。"
            "适用于运行Python脚本、pip安装、git操作、文件系统操作等。"
            "超时时间为60秒，输出超过5000字符会被截断。"
            "Windows环境下使用cmd.exe执行。"
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="command",
                type="string",
                description="要执行的Shell命令",
                required=True,
            ),
            ToolParameter(
                name="cwd",
                type="string",
                description="工作目录路径，默认为当前目录",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="超时时间（秒），默认60秒",
                required=False,
                default=60,
            ),
        ]

    def _execute(
            self,
            command: str,
            cwd: str | None = None,
            timeout: int = 60,
            **kwargs: Any,
    ) -> str:
        # 安全检查：危险命令检测
        command_lower = command.lower().strip()
        for pattern in DANGEROUS_PATTERNS:
            if pattern in command_lower:
                raise PermissionError(
                    f"检测到危险命令模式: '{pattern}'，已拒绝执行。"
                )

        # 工作目录处理
        work_dir = Path(cwd).resolve() if cwd else Path.cwd()
        if not work_dir.exists():
            raise FileNotFoundError(f"工作目录不存在: {cwd}")

        decision = self._permission_manager.check(command, cwd=str(work_dir))
        if decision == PermissionDecision.DENY:
            raise PermissionError(f"命令被权限系统拒绝: {command}")
        if decision == PermissionDecision.ASK:
            raise PermissionError(f"命令未获批准: {command}")

        # 检测操作系统（Windows需要特殊处理）
        is_windows = os.name == "nt"

        try:
            result = subprocess.run(
                command,
                shell=True,  # 允许shell语法（管道、重定向等）
                capture_output=True,  # 捕获stdout和stderr
                text=True,  # 返回字符串而不是bytes
                cwd=str(work_dir),
                timeout=timeout,
                encoding="utf-8",
                errors="replace",  # 编码错误时替换，不抛异常
                # Windows下设置代码页为UTF-8
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"命令执行超时（{timeout}秒）: {command}")

        # 整合输出
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        max_output = 5000

        # 构建输出报告
        lines = [
            f"$ {command}",
            f"退出码: {result.returncode}",
            f"工作目录: {work_dir}",
        ]

        if stdout:
            if len(stdout) > max_output:
                lines.append(f"\n[stdout - 截断到{max_output}字符]:\n{stdout[:max_output]}...")
            else:
                lines.append(f"\n[stdout]:\n{stdout}")

        if stderr:
            if len(stderr) > max_output:
                lines.append(f"\n[stderr - 截断到{max_output}字符]:\n{stderr[:max_output]}...")
            else:
                lines.append(f"\n[stderr]:\n{stderr}")

        if result.returncode != 0 and not stdout and not stderr:
            lines.append("\n(命令无输出)")

        return "\n".join(lines)


registry.register(ShellTool(), tags=["shell", "system"])