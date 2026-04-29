"""
verify/step2_api.py - Dashscope API 连通性诊断脚本

设计原则（吸取经验教训）：
  ✅ 先打印配置信息：让用户一眼看到实际使用的模型和端点
  ✅ 分类错误处理：区分认证/参数/网络/配额等不同错误类型
  ✅ 提供修复建议：每个错误都说清楚"问题在哪"和"怎么修复"
  ✅ 读取配置而非硬编码：模型名来自 .env，不在脚本里写死
  ✅ 超时控制：每项测试都有独立超时
  ✅ 不假设API行为：tool_calling测试只验证"调用成功"而非"一定触发工具"

运行：
  python verify/step2_api.py
  python verify/step2_api.py --verbose   # 显示更多调试信息
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# ── 路径初始化（必须在所有本地 import 之前）──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── 本地模块 ──────────────────────────────────────────────────────────────────
from src.llm_client import LLMClient, LLMConfig
from src.models import ModelError, ModelTier


# ==============================================================================
# 输出工具函数
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
    """打印修复建议（缩进格式）"""
    for line in msg.strip().splitlines():
        print(f"     💡 {line}")


# ==============================================================================
# 诊断测试函数
# ==============================================================================

def print_config_info(client: LLMClient) -> None:
    """
    第一步：打印当前配置信息

    原则：用户必须先知道"脚本在用什么配置"，
    才能判断后续错误是配置问题还是代码问题。
    """
    print("\n📌 当前配置信息")
    print("  " + "─" * 40)
    info = client.get_model_info()
    for key, value in info.items():
        # 隐藏API Key具体值，只显示前缀
        if "key" in key.lower():
            display = value[:8] + "..." if len(value) > 8 else "（未设置）"
        else:
            display = value
        print(f"  {key:<20} = {display}")
    print("  " + "─" * 40)


def test_fast_model_chat(client: LLMClient, verbose: bool) -> bool:
    """
    测试1：快速模型基础对话

    目的：验证API Key有效、端点可达、基本对话可用
    用快速模型：节省费用，响应更快
    """
    print(f"\n📌 测试1：快速模型对话（{client.config.fast_model}）")

    start = time.time()
    try:
        response = client.chat(
            messages=[{"role": "user", "content": "请用一句话（不超过20字）回答：1+1等于几？"}],
            tier=ModelTier.FAST,
            max_tokens=30,
            temperature=0.0,
        )
        elapsed = (time.time() - start) * 1000

        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        _pass(f"对话成功 | 耗时: {elapsed:.0f}ms | Tokens: {tokens}")
        _info(f"模型回复: {content.strip()[:60]}")

        if verbose:
            _info(f"完整响应 ID: {response.id}")
            _info(f"停止原因: {response.choices[0].finish_reason}")

        return True

    except ModelError as e:
        _fail(f"对话失败: {e}")
        _suggest(e.suggestion)
        if verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        _fail(f"未预期错误: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_strong_model_chat(client: LLMClient, verbose: bool) -> bool:
    """
    测试2：强力模型基础对话

    目的：验证强模型可用（规划器会用到）
    单独测试是因为 strong_model 可能和 fast_model 是同一个模型，
    也可能是需要不同参数的模型（如qwen3系列）
    """
    print(f"\n📌 测试2：强力模型对话（{client.config.strong_model}）")

    # 如果两个模型相同，跳过重复测试
    if client.config.strong_model == client.config.fast_model:
        _warn(
            f"strong_model 和 fast_model 相同（{client.config.strong_model}），"
            "跳过重复测试"
        )
        _suggest(
            "建议在 .env 中配置不同档位的模型，例如：\n"
            "AGENT_MODEL_STRONG=qwen-max\n"
            "AGENT_MODEL_FAST=qwen-plus"
        )
        return True

    start = time.time()
    try:
        response = client.chat(
            messages=[{"role": "user", "content": "请用一句话总结：什么是ReAct模式？"}],
            tier=ModelTier.STRONG,
            max_tokens=50,
            temperature=0.0,
        )
        elapsed = (time.time() - start) * 1000
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        _pass(f"对话成功 | 耗时: {elapsed:.0f}ms | Tokens: {tokens}")
        _info(f"模型回复: {content.strip()[:80]}")
        return True

    except ModelError as e:
        _fail(f"对话失败: {e}")
        _suggest(e.suggestion)
        if verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        _fail(f"未预期错误: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_embedding(client: LLMClient, verbose: bool) -> bool:
    """
    测试3：Embedding API

    目的：验证向量化功能可用（ChromaDB记忆系统依赖此接口）
    验证维度是否为1024（text-embedding-v3的标准维度）
    """
    print(f"\n📌 测试3：Embedding API（{client.config.embedding_model}）")

    test_texts = [
        "Python的GIL是什么？",
        "Agent的记忆系统如何设计？",
    ]

    start = time.time()
    try:
        embeddings = client.embed(test_texts)
        elapsed = (time.time() - start) * 1000

        if not embeddings or not embeddings[0]:
            _fail("返回了空向量")
            return False

        dim = len(embeddings[0])
        count = len(embeddings)

        _pass(f"Embedding成功 | 向量维度: {dim} | 数量: {count} | 耗时: {elapsed:.0f}ms")

        # 维度检查
        expected_dim = 1024
        if dim != expected_dim:
            _warn(
                f"向量维度 {dim} 与预期 {expected_dim} 不符。\n"
                f"如果你使用的不是 text-embedding-v3，请更新 AGENT_EMBEDDING_DIM 配置。"
            )
        else:
            _info(f"维度验证通过（{dim}维，与ChromaDB配置一致）")

        if verbose:
            # 打印第一个向量的前5个值（验证不是全零）
            first_5 = [round(v, 4) for v in embeddings[0][:5]]
            _info(f"向量前5维: {first_5}")

        return True

    except ModelError as e:
        _fail(f"Embedding失败: {e}")
        _suggest(e.suggestion)
        if verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        _fail(f"未预期错误: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_tool_calling(client: LLMClient, verbose: bool) -> bool:
    """
    测试4：Tool Calling（Function Calling）能力

    目的：验证LLM能正确解析工具定义并（在合适时）触发工具调用
    这是Agent ReAct循环的核心能力

    重要设计原则：
    不假设LLM一定会触发工具调用——
    有些简单问题LLM可能直接回答而不调用工具，这是正常行为。
    我们只验证"调用没有报错"，而不是"一定触发了tool_call"。
    """
    print(f"\n📌 测试4：Tool Calling能力（{client.config.fast_model}）")

    # 工具定义（JSON Schema格式）
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "文件路径"
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": "执行Shell命令",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "要执行的命令"
                        }
                    },
                    "required": ["command"]
                }
            }
        },
    ]

    # 使用强烈暗示需要工具的prompt，提高触发概率（但不强求）
    messages = [
        {
            "role": "user",
            "content": "请使用 read_file 工具读取 README.md 文件的内容"
        }
    ]

    start = time.time()
    try:
        response = client.chat(
            messages=messages,
            tier=ModelTier.FAST,
            tools=tools,
            max_tokens=100,
            temperature=0.0,
        )
        elapsed = (time.time() - start) * 1000

        finish_reason = response.choices[0].finish_reason
        message = response.choices[0].message
        tool_calls = message.tool_calls

        # ── 只要API调用成功就算通过 ─────────────────────────────────
        _pass(f"Tool Calling API调用成功 | 耗时: {elapsed:.0f}ms | finish_reason: {finish_reason}")

        if tool_calls:
            # LLM选择调用了工具
            for tc in tool_calls:
                _info(f"LLM选择调用工具: {tc.function.name}({tc.function.arguments[:60]})")
        else:
            # LLM选择直接回答（也是正常行为）
            content = message.content or ""
            _info(f"LLM选择直接回答（未触发工具）: {content[:60]}")
            _info("这是正常行为——LLM可能认为不需要工具也能回答该问题")

        if verbose:
            _info(f"完整message: {message}")

        return True

    except ModelError as e:
        _fail(f"Tool Calling失败: {e}")
        _suggest(e.suggestion)
        if verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        _fail(f"未预期错误: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_stream_chat(client: LLMClient, verbose: bool) -> bool:
    """
    测试5：流式输出

    目的：验证stream=True模式可用（TUI界面会用到）
    """
    print(f"\n📌 测试5：流式对话（{client.config.fast_model}）")

    messages = [{"role": "user", "content": "请用10个字以内介绍Python语言"}]

    start = time.time()
    chunks: list[str] = []

    try:
        print("  输出: ", end="", flush=True)
        for chunk in client.stream_chat(messages, tier=ModelTier.FAST):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        print()  # 换行

        elapsed = (time.time() - start) * 1000
        full_text = "".join(chunks)

        _pass(f"流式输出成功 | chunk数: {len(chunks)} | 耗时: {elapsed:.0f}ms")
        _info(f"完整文本: {full_text.strip()[:80]}")
        return True

    except ModelError as e:
        print()  # 换行
        _fail(f"流式输出失败: {e}")
        _suggest(e.suggestion)
        if verbose:
            traceback.print_exc()
        return False

    except Exception as e:
        print()
        _fail(f"未预期错误: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return False


def test_probe(client: LLMClient, verbose: bool) -> bool:
    """
    测试0（前置探针）：快速健康检查

    目的：在运行完整测试之前先做轻量探针，
    如果探针失败可以快速定位是基础连接问题还是参数问题
    """
    print(f"\n📌 测试0：API 健康探针")

    result = client.probe()

    if result["success"]:
        _pass(
            f"探针通过 | 模型: {result['model']} | "
            f"延迟: {result['latency_ms']}ms | "
            f"Tokens: {result.get('tokens_used', '?')}"
        )
        return True
    else:
        _fail(f"探针失败: {result.get('error', '未知错误')}")
        suggestion = result.get("suggestion", "")
        if suggestion:
            _suggest(suggestion)
        return False


# ==============================================================================
# 主函数
# ==============================================================================

def main() -> None:
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Dashscope API 连通性诊断脚本"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细调试信息（包括完整错误堆栈）"
    )
    parser.add_argument(
        "--skip-strong",
        action="store_true",
        help="跳过强力模型测试（节省费用）"
    )
    parser.add_argument(
        "--skip-stream",
        action="store_true",
        help="跳过流式输出测试"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Phase 1 · Step 2  API 连通性诊断")
    print("=" * 60)

    # ── 初始化客户端 ──────────────────────────────────────────────────────────
    try:
        client = LLMClient()
    except ValueError as e:
        print(f"\n❌ 客户端初始化失败: {e}")
        sys.exit(1)

    # ── 打印配置（第一步！用户必须能看到实际配置）────────────────────────────
    print_config_info(client)

    # ── 执行测试序列 ──────────────────────────────────────────────────────────
    results: dict[str, bool] = {}

    # 先跑探针，如果探针失败给出是否继续的选择
    probe_ok = test_probe(client, args.verbose)
    results["API探针"] = probe_ok

    if not probe_ok:
        print("\n" + "─" * 60)
        print("  ⚠️  探针失败，后续测试可能也会失败")
        print("  继续运行以获取详细诊断信息...")
        print("─" * 60)

    # 其余测试
    results["快速模型对话"] = test_fast_model_chat(client, args.verbose)

    if not args.skip_strong:
        results["强力模型对话"] = test_strong_model_chat(client, args.verbose)

    results["Embedding向量化"] = test_embedding(client, args.verbose)
    results["Tool Calling"] = test_tool_calling(client, args.verbose)

    if not args.skip_stream:
        results["流式输出"] = test_stream_chat(client, args.verbose)

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 诊断汇总")
    print("─" * 60)

    passed = 0
    failed = 0
    for test_name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {test_name}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("─" * 60)
    print(f"  通过: {passed} / 失败: {failed} / 共: {len(results)}")

    if failed == 0:
        print("\n🎉 所有测试通过！API 连通性正常。")
        print("\n下一步运行: python verify/step3_planning.py")
    elif passed == 0:
        print("\n💀 所有测试失败，请优先排查：")
        print("   1. DASHSCOPE_API_KEY 是否正确")
        print("   2. 网络是否能访问 dashscope.aliyuncs.com")
        print("   运行 --verbose 查看完整错误信息")
    else:
        print(f"\n⚠️  部分测试失败（{failed}个），请根据上方建议修复")
        print("   运行 python verify/step2_api.py --verbose 查看详细错误")

    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()