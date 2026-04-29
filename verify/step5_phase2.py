"""
Phase 2 验证脚本
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

# 确保从 verify 目录直接运行时也能导入 src 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.context_compressor import ContextCompressor
from src.execution.parallel_executor import ParallelExecutor
from src.permissions import PermissionDecision, PermissionManager
from src.tools.builtin.edit_file import EditFileTool


def test_edit_file() -> None:
    tool = EditFileTool()
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "demo.txt"
        target.write_text("a\nb\nc\nd\n", encoding="utf-8")
        result = tool.execute(
            path=str(target),
            operation="replace",
            start_line=3,
            end_line=3,
            new_content="CHANGED",
        )
        assert result.status.value == "success"
        assert target.read_text(encoding="utf-8").splitlines()[2] == "CHANGED"


def test_parallel_tools() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        files = []
        for idx in range(3):
            p = base / f"f{idx}.txt"
            p.write_text(f"file-{idx}", encoding="utf-8")
            files.append(p)

        calls = [
            SimpleNamespace(
                id=f"c{idx}",
                function=SimpleNamespace(
                    name="read_file",
                    arguments=f'{{"path":"{path.as_posix()}"}}',
                ),
            )
            for idx, path in enumerate(files)
        ]

        def execute_fn(name: str, args: dict) -> SimpleNamespace:
            time.sleep(0.2)
            content = Path(args["path"]).read_text(encoding="utf-8")
            return SimpleNamespace(status=SimpleNamespace(value="success"), output=content)

        parallel = ParallelExecutor(max_workers=4)

        t1 = time.perf_counter()
        for call in calls:
            args = json.loads(call.function.arguments)
            execute_fn(call.function.name, args)
        sequential_elapsed = time.perf_counter() - t1

        t2 = time.perf_counter()
        results = parallel.execute_parallel_tools(calls, execute_fn)
        parallel_elapsed = time.perf_counter() - t2

        assert len(results) == 3
        assert all(r[3].status.value == "success" for r in results)
        assert parallel_elapsed < sequential_elapsed * 0.7


def test_context_compressor() -> None:
    class DummyLLM:
        def chat(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="中间历史摘要"))]
            )

    compressor = ContextCompressor(DummyLLM())  # type: ignore[arg-type]
    messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"msg-{i}"} for i in range(100)
    ]
    compressed = compressor.compress(messages, max_tokens=200)
    assert len(compressed) < len(messages)
    assert compressed[0]["role"] == "system"
    assert compressed[0]["content"] == "sys"


def test_permission_deny() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manager = PermissionManager(config_path=Path(tmp) / "permissions.json")
        decision = manager.classify("rm -rf /")
        assert decision == PermissionDecision.DENY


def main() -> None:
    test_edit_file()
    print("✅ test_edit_file")
    test_parallel_tools()
    print("✅ test_parallel_tools")
    test_context_compressor()
    print("✅ test_context_compressor")
    test_permission_deny()
    print("✅ test_permission_deny")
    print("🎉 Phase 2 核心验证全部通过")


if __name__ == "__main__":
    main()
