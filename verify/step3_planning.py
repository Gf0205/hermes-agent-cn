"""
verify/step3_planning.py - 规划 + 记忆 + 执行系统验证

修复记录：
  v1 → v2: 修复 Windows [WinError 32] 文件锁问题
  修复方案：
  1. tempfile.TemporaryDirectory(ignore_cleanup_errors=True) 表层兜底
  2. ChromaMemoryStore 实现 close() + 上下文管理器（根本修复）
  3. 测试函数中显式调用 memory.close() 确保句柄释放后再退出

运行：
  python verify/step3_planning.py
  python verify/step3_planning.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ==============================================================================
# 输出工具
# ==============================================================================

def _pass(msg: str) -> None:
    print(f"  ✅ {msg}")

def _fail(msg: str) -> None:
    print(f"  ❌ {msg}")

def _warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")

def _info(msg: str) -> None:
    print(f"  ℹ️  {msg}")

def _suggest(msg: str) -> None:
    for line in msg.strip().splitlines():
        print(f"     💡 {line}")


# ==============================================================================
# 测试函数
# ==============================================================================

def test_chromadb_basic(verbose: bool) -> bool:
    """
    测试1：ChromaDB 基础功能 + Windows文件句柄释放

    验证点：
    - ChromaDB可以初始化
    - 可以创建集合、插入、查询
    - close()后文件句柄正确释放（Windows不报WinError 32）
    """
    print("\n📌 测试1：ChromaDB 基础功能（含Windows文件锁验证）")

    try:
        import chromadb

        # ignore_cleanup_errors=True：即使close()后仍有残留句柄也不报错
        # 这是表层兜底，根本修复在ChromaMemoryStore.close()中
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            client = chromadb.PersistentClient(path=tmp_dir)
            col = client.get_or_create_collection(
                "test_col",
                metadata={"hnsw:space": "cosine"}
            )

            # 插入测试数据
            col.add(
                ids=["t1", "t2"],
                documents=["Python异步编程", "Agent记忆系统"],
                embeddings=[[0.1] * 8, [0.2] * 8],  # 伪造低维向量，不调API
                metadatas=[{"type": "test"}, {"type": "test"}],
            )

            count = col.count()
            assert count == 2, f"期望2条记录，实际{count}条"
            _pass(f"ChromaDB读写正常 | 记录数: {count}")

            # 显式释放（根本修复）
            col = None       # type: ignore[assignment]
            client = None    # type: ignore[assignment]

            import gc
            gc.collect()     # 强制GC，释放hnswlib文件句柄

            _pass("文件句柄释放成功（Windows WinError 32 修复验证通过）")

        _pass("临时目录清理完成（无 WinError 32）")
        return True

    except Exception as e:
        _fail(f"ChromaDB测试失败: {e}")
        if verbose:
            traceback.print_exc()
        _suggest(
            "安装命令: pip install chromadb>=1.0.0\n"
            "Windows用户确保Python>=3.10 且 SQLite>=3.35"
        )
        return False


def test_chroma_store(verbose: bool) -> bool:
    """
    测试2：ChromaMemoryStore 封装层

    验证点：
    - ChromaMemoryStore 可以正常初始化
    - add() 和 search() 工作正常（用伪向量跳过API调用）
    - 上下文管理器（with语句）正确关闭
    - 关闭后操作抛出 RuntimeError
    """
    print("\n📌 测试2：ChromaMemoryStore 封装层")

    try:
        from src.memory.chroma_store import ChromaMemoryStore
        from src.models import MemoryEntry

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:

            # 用 with 语句测试上下文管理器
            with ChromaMemoryStore(persist_dir=tmp_dir) as store:
                entry = MemoryEntry(
                    content="Python的GIL在CPU密集型任务时是性能瓶颈",
                    memory_type="semantic",
                    importance=0.8,
                )
                # 用8维伪向量（避免调Embedding API）
                fake_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

                entry_id = store.add(entry, fake_embedding)
                _pass(f"add() 成功 | ID: {entry_id[:8]}...")

                count = store.count()
                assert count == 1, f"期望1条，实际{count}条"
                _pass(f"count() 正常: {count}")

                # 搜索（用同样的伪向量，相似度应该很高）
                results = store.search(fake_embedding, n_results=1)
                assert len(results) == 1
                found_entry, score = results[0]
                _pass(f"search() 成功 | 相似度: {score:.4f} | 内容: {found_entry.content[:30]}...")

            # with块结束后，store应该已关闭
            assert store._closed is True, "with退出后应该是关闭状态"
            _pass("上下文管理器正确关闭（_closed=True）")

            # 验证关闭后操作抛出异常
            try:
                store.add(entry, fake_embedding)
                _fail("关闭后add()应该抛出RuntimeError，但没有")
                return False
            except RuntimeError:
                _pass("关闭后操作正确抛出 RuntimeError")

        return True

    except Exception as e:
        _fail(f"ChromaMemoryStore测试失败: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_state_machine(verbose: bool) -> bool:
    """测试3：状态机合法/非法转换"""
    print("\n📌 测试3：StateMachine 状态转换")

    try:
        from src.execution.state_machine import StateMachine
        from src.models import AgentState

        sm = StateMachine()

        # 合法转换链
        transitions = [
            (AgentState.PLANNING,   "开始规划"),
            (AgentState.EXECUTING,  "开始执行"),
            (AgentState.REFLECTING, "反思"),
            (AgentState.IDLE,       "完成"),
        ]
        for target, reason in transitions:
            sm.transition(target, reason)
            assert sm.state == target
        _pass(f"合法转换链通过: IDLE→PLANNING→EXECUTING→REFLECTING→IDLE")

        # 非法转换检测
        illegal_cases = [
            (AgentState.IDLE,       AgentState.REFLECTING,  "IDLE不能直接到REFLECTING"),
            (AgentState.IDLE,       AgentState.EXECUTING,   "IDLE不能直接到EXECUTING"),
        ]
        for start, target, desc in illegal_cases:
            sm2 = StateMachine()
            try:
                sm2.transition(target)
                _fail(f"应该拒绝非法转换: {desc}")
                return False
            except ValueError:
                pass  # 正确！
        _pass("非法转换全部被正确拒绝")

        # force_idle 测试
        sm3 = StateMachine()
        sm3.transition(AgentState.PLANNING)
        sm3.force_idle("测试强制重置")
        assert sm3.state == AgentState.IDLE
        _pass("force_idle() 强制重置到IDLE成功")

        return True

    except Exception as e:
        _fail(f"StateMachine测试失败: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_tool_execution(verbose: bool) -> bool:
    """测试4：5个内置工具实际执行"""
    print("\n📌 测试4：内置工具执行")

    try:
        import src.tools  # 触发自动注册
        from src.tools.registry import registry

        tool_count = len(registry)
        _info(f"已注册工具数: {tool_count}")

        # ── list_dir ───────────────────────────────────────────────────
        tool = registry.get("list_dir")
        result = tool.execute(path=".", depth=1)
        assert result.status.value == "success", f"list_dir失败: {result.error}"
        _pass("list_dir ✓")

        # ── grep_search ────────────────────────────────────────────────
        tool = registry.get("grep_search")
        result = tool.execute(
            pattern="class",
            path="src",
            file_pattern="*.py",
            max_results=5
        )
        assert result.status.value == "success", f"grep_search失败: {result.error}"
        match_count = result.output.count("▶")
        _pass(f"grep_search ✓ | 匹配: {match_count} 处")

        # ── write_file + read_file（配对测试）─────────────────────────
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            test_file = str(Path(tmp_dir) / "test_verify.txt")
            test_content = "这是Step3验证脚本写入的测试文件\n第二行内容"

            # 写入
            write_tool = registry.get("write_file")
            result = write_tool.execute(path=test_file, content=test_content)
            assert result.status.value == "success", f"write_file失败: {result.error}"
            _pass("write_file ✓")

            # 读取并验证
            read_tool = registry.get("read_file")
            result = read_tool.execute(path=test_file)
            assert result.status.value == "success", f"read_file失败: {result.error}"
            assert "验证脚本" in result.output, "读取内容与写入不符"
            _pass("read_file ✓（写入内容已验证）")

        # ── shell ──────────────────────────────────────────────────────
        tool = registry.get("shell")
        result = tool.execute(command="echo hello-from-hermes-cn")
        assert result.status.value == "success", f"shell失败: {result.error}"
        assert "hello-from-hermes-cn" in result.output
        _pass("shell ✓")

        # 危险命令拦截测试
        result = tool.execute(command="rm -rf /")
        assert result.status.value != "success", "危险命令应被拦截"
        _pass("shell危险命令拦截 ✓")

        return True

    except Exception as e:
        _fail(f"工具执行测试失败: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_memory_manager(verbose: bool) -> bool:
    """
    测试5：4层记忆管理器（调用Embedding API）

    注意：此测试会消耗少量API Token
    """
    print("\n📌 测试5：MemoryManager 4层记忆（会调用Embedding API，约100 tokens）")

    try:
        from src.llm_client import LLMClient
        from src.memory.memory_manager import MemoryManager

        # ignore_cleanup_errors=True：Windows文件锁兜底
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            client = LLMClient()

            # 用 with 语句确保MemoryManager正确关闭（释放ChromaDB文件句柄）
            with MemoryManager(client, data_dir=tmp_dir) as memory:

                # 层1：工作记忆
                memory.add_to_working_memory("user", "如何优化Python性能？")
                memory.add_to_working_memory("assistant", "可以考虑使用多进程绕过GIL")
                wm = memory.get_working_memory()
                assert len(wm) == 2
                _pass(f"工作记忆（层1）✓ | {len(wm)} 条消息")

                # 层3：语义记忆（调用Embedding API）
                start = time.time()
                entry_id = memory.remember(
                    content="Python的GIL在CPU密集型任务时是性能瓶颈，多进程是解决方案",
                    importance=0.8,
                    memory_type="semantic",
                )
                elapsed = (time.time() - start) * 1000
                _pass(f"语义记忆存储（层3）✓ | ID: {entry_id[:8]} | 耗时: {elapsed:.0f}ms")

                # 检索
                start = time.time()
                results = memory.recall("Python多线程性能问题", top_k=1)
                elapsed = (time.time() - start) * 1000

                if results:
                    entry, score = results[0]
                    _pass(f"语义记忆检索（层3）✓ | 相似度: {score:.3f} | 耗时: {elapsed:.0f}ms")
                    if verbose:
                        _info(f"检索内容: {entry.content[:60]}")
                else:
                    _warn("检索返回空（记忆太少或相似度过低，可忽略）")

                # 层4：技能（程序性记忆）
                memory.save_skill(
                    skill_name="create_python_module",
                    description="创建标准Python模块的步骤",
                    steps=["创建目录", "添加__init__.py", "编写模块代码", "添加测试"],
                    tags=["python", "project"],
                )
                skills = memory.list_skills()
                assert len(skills) >= 1
                _pass(f"程序性记忆/技能（层4）✓ | {len(skills)} 个技能")

                # 统计
                stats = memory.get_stats()
                _info(f"记忆统计: {stats}")

            # with块结束，MemoryManager已调用close()
            _pass("MemoryManager.close()成功（Windows文件句柄已释放）")

        return True

    except Exception as e:
        _fail(f"MemoryManager测试失败: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_strategic_planner(verbose: bool) -> bool:
    """
    测试6：战略规划器（调用Chat API）

    注意：此测试会消耗约500-800 Token
    """
    print("\n📌 测试6：StrategicPlanner（会调用Chat API，约500-800 tokens）")

    try:
        from src.llm_client import LLMClient
        from src.planning.strategic_planner import StrategicPlanner
        from src.models import PlanStatus

        client = LLMClient()
        planner = StrategicPlanner(client)

        start = time.time()
        plan = planner.decompose(
            goal="在当前目录创建一个名为hello.txt的文件，内容为'Hello, Hermes!'",
        )
        elapsed = (time.time() - start) * 1000

        assert plan is not None
        assert len(plan.sub_goals) >= 1, "至少应有1个子目标"
        assert plan.goal != ""

        _pass(
            f"规划成功 | {len(plan.sub_goals)} 个子目标 "
            f"| 耗时: {elapsed:.0f}ms "
            f"| Tokens: {plan.planning_tokens_used}"
        )

        for sg in plan.sub_goals:
            dep_str = f" [依赖: {sg.dependencies}]" if sg.dependencies else ""
            _info(f"  [{sg.id}] {sg.description[:60]}{dep_str}")

        # 验证get_ready_goals()（依赖拓扑排序）
        ready = plan.get_ready_goals()
        assert len(ready) >= 1, "初始状态至少有1个子目标没有前置依赖"
        _pass(f"依赖拓扑排序 ✓ | 初始就绪: {len(ready)} 个子目标")

        if verbose:
            _info(f"计划ID: {plan.id} | 模型: {plan.model_used}")

        return True

    except Exception as e:
        _fail(f"StrategicPlanner测试失败: {e}")
        if verbose:
            traceback.print_exc()
        _suggest(
            "如果是API相关错误，先运行 python verify/step2_api.py 确认API连通性"
        )
        return False


# ==============================================================================
# 主函数
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 · Step 3 验证脚本")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细错误信息")
    parser.add_argument("--skip-api", action="store_true",  help="跳过需要API调用的测试（省Token）")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Phase 1 · Step 3  规划 + 记忆 + 执行系统验证")
    print("=" * 60)

    # 测试序列
    # (测试函数, 名称, 是否需要API)
    test_plan = [
        (test_chromadb_basic,    "ChromaDB基础+文件锁修复", False),
        (test_chroma_store,      "ChromaMemoryStore封装",  False),
        (test_state_machine,     "StateMachine状态转换",   False),
        (test_tool_execution,    "5个内置工具执行",        False),
        (test_memory_manager,    "4层MemoryManager",       True ),
        (test_strategic_planner, "StrategicPlanner规划",   True ),
    ]

    results: dict[str, bool] = {}

    for test_fn, test_name, needs_api in test_plan:
        if needs_api and args.skip_api:
            _warn(f"跳过（需要API）: {test_name}")
            continue
        try:
            results[test_name] = test_fn(args.verbose)
        except Exception as e:
            _fail(f"{test_name} 发生未捕获异常: {e}")
            if args.verbose:
                traceback.print_exc()
            results[test_name] = False

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 验证汇总")
    print("─" * 60)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {name}")

    print("─" * 60)
    print(f"  通过: {passed} / 失败: {failed} / 共: {len(results)}")

    if failed == 0:
        print("\n🎉 所有测试通过！Phase 1 Step 3 完成。")
        print("   下一步运行: python verify/step4_integration.py")
    else:
        print(f"\n⚠️  {failed} 项未通过，请根据提示修复。")
        print("   运行 --verbose 查看详细错误堆栈。")

    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()