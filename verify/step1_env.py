"""
verify/step1_env.py - Phase 1 环境验证脚本

运行方式：
    conda activate your-env
    python verify/step1_env.py

通过标准：
    ✅ 所有检查项均显示 PASS
"""

import sys
import importlib
from pathlib import Path


def check(name: str, condition: bool, detail: str = "") -> bool:
    """输出检查结果"""
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {name}")
    if not condition and detail:
        print(f"         └─ {detail}")
    return condition


def main() -> None:
    print("\n" + "=" * 60)
    print("  Phase 1 · Step 1 环境验证")
    print("=" * 60)

    all_pass = True

    # ── Python版本 ──────────────────────────────────────────
    print("\n📌 Python环境")
    py_ok = sys.version_info >= (3, 10)
    all_pass &= check(
        f"Python >= 3.10 (当前: {sys.version.split()[0]})",
        py_ok,
        "请使用 Python 3.10+ 或升级conda环境"
    )

    # ── 必需依赖 ────────────────────────────────────────────
    print("\n📌 核心依赖库")
    required_packages = [
        ("openai", "openai"),
        ("numpy", "numpy"),
        ("faiss", "faiss"),
        ("rich", "rich"),
        ("questionary", "questionary"),
        ("yaml", "pyyaml"),
        ("dotenv", "python-dotenv"),
        ("sqlite_utils", "sqlite-utils"),
    ]

    for import_name, package_name in required_packages:
        try:
            importlib.import_module(import_name)
            ok = True
        except ImportError:
            ok = False
        all_pass &= check(
            f"import {import_name}",
            ok,
            f"运行: pip install {package_name}"
        )

    # ── 环境变量 ────────────────────────────────────────────
    print("\n📌 环境变量配置")
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    key_ok = bool(api_key) and api_key.startswith("sk-")
    all_pass &= check(
        "DASHSCOPE_API_KEY 已设置",
        key_ok,
        "请在 .env 文件中设置: DASHSCOPE_API_KEY=sk-xxx"
    )

    base_url = os.getenv("DASHSCOPE_BASE_URL", "")
    all_pass &= check(
        "DASHSCOPE_BASE_URL 已设置",
        bool(base_url),
        "请在 .env 文件中设置 BASE_URL"
    )

    # ── 项目结构 ────────────────────────────────────────────
    print("\n📌 项目文件结构")
    root = Path(__file__).parent.parent
    required_files = [
        "src/__init__.py",
        "src/models.py",
        "src/event_bus.py",
        "src/llm_client.py",
        "src/tools/base.py",
        "src/tools/registry.py",
        "src/tools/builtin/read_file.py",
        "src/tools/builtin/write_file.py",
        "src/tools/builtin/shell.py",
        "src/tools/builtin/grep_search.py",
        "src/tools/builtin/list_dir.py",
        "pyproject.toml",
        ".env",
    ]

    for file_path in required_files:
        exists = (root / file_path).exists()
        all_pass &= check(f"存在 {file_path}", exists)

    # ── 模块导入测试 ────────────────────────────────────────
    print("\n📌 模块导入测试")
    # 添加项目根目录到路径
    sys.path.insert(0, str(root))

    modules = [
        "src.models",
        "src.event_bus",
        "src.llm_client",
        "src.tools.base",
        "src.tools.registry",
    ]
    for module in modules:
        try:
            importlib.import_module(module)
            ok = True
        except Exception as e:
            ok = False
            print(f"         └─ 错误: {e}")
        all_pass &= check(f"import {module}", ok)

    # ── 工具注册测试 ────────────────────────────────────────
    print("\n📌 工具注册验证")
    try:
        import src.tools  # 触发工具自动注册
        from src.tools.registry import registry
        tool_count = len(registry)
        tools_ok = tool_count >= 5
        all_pass &= check(
            f"工具注册表 ({tool_count} 个工具)",
            tools_ok,
            f"期望至少5个工具，当前{tool_count}个"
        )

        for tool_name in ["read_file", "write_file", "shell", "grep_search", "list_dir"]:
            registered = tool_name in registry
            all_pass &= check(f"  工具: {tool_name}", registered)
    except Exception as e:
        all_pass = False
        print(f"  ❌ FAIL  工具注册失败: {e}")

    # ── 汇总 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_pass:
        print("🎉 所有检查通过！Phase 1 Step 1 完成")
        print("\n下一步运行: python verify/step2_api.py")
    else:
        print("⚠️  部分检查未通过，请根据提示修复后重试")
    print("=" * 60 + "\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()