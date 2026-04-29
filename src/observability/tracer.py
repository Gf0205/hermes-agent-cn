"""
src/observability/tracer.py - 执行轨迹记录器

修订记录：
  v1 → v2：修复 _serialize_trace 无限递归问题

根因分析：
  原 convert() 函数的检查顺序错误：
    hasattr(obj, "__dict__")  ← 会匹配 Enum 实例
    obj.__dict__ 包含 __objclass__ → 指向枚举类
    枚举类 __dict__ 包含所有枚举成员
    每个成员又有 __objclass__ → 回到枚举类
    ∞ 递归

修复方案：
  1. isinstance(Enum) 检查必须在 hasattr(__dict__) 之前
  2. 用 dataclasses.fields() 替代 obj.__dict__（只取业务字段，跳过私有属性）
  3. 对 dict 类型显式处理（原来漏掉了）
  4. 兜底：无法序列化的对象转 str，绝不抛异常
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.event_bus import Event, EventType, get_event_bus
from src.models import ExecutionTrace

logger = logging.getLogger(__name__)


class ExecutionTracer:
    """执行轨迹追踪器"""

    def __init__(self, traces_dir: Optional[str] = None) -> None:
        base_dir = os.getenv("AGENT_DATA_DIR", str(Path.home() / ".hermes-cn"))
        self._traces_dir = Path(
            traces_dir or (Path(base_dir) / "traces")
        ).expanduser().resolve()
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._bus = get_event_bus()
        self._bus.subscribe(EventType.STATE_CHANGED,  self._on_state_changed)
        self._bus.subscribe(EventType.TOOL_CALLED,    self._on_tool_called)
        self._bus.subscribe(EventType.TOOL_RESULT,    self._on_tool_result)
        self._bus.subscribe(EventType.LLM_RESPONSE,   self._on_llm_response)

    # ==================================================================
    # 事件订阅回调
    # ==================================================================

    def _on_state_changed(self, event: Event) -> None:
        from_s  = event.data.get("from_state", "")
        to_s    = event.data.get("to_state", "")
        reason  = event.data.get("reason", "")
        logger.info(f"🔄 {from_s} → {to_s}" + (f"（{reason}）" if reason else ""))

    def _on_tool_called(self, event: Event) -> None:
        tool = event.data.get("tool_name", "")
        args = str(event.data.get("arguments", {}))[:80]
        logger.info(f"🔧 调用工具: {tool}({args})")

    def _on_tool_result(self, event: Event) -> None:
        tool    = event.data.get("tool_name", "")
        status  = event.data.get("status", "")
        elapsed = event.data.get("execution_time_ms", 0)
        icon    = "✅" if status == "success" else "❌"
        logger.info(f"{icon} {tool} | {status} | {elapsed:.0f}ms")

    def _on_llm_response(self, event: Event) -> None:
        tokens  = event.data.get("tokens", 0)
        elapsed = event.data.get("elapsed_ms", 0)
        cost    = event.data.get("estimated_cost_usd", 0)
        logger.debug(f"💭 Tokens: {tokens} | {elapsed:.0f}ms | ${cost:.4f}")

    # ==================================================================
    # 核心：序列化与持久化
    # ==================================================================

    def save_trace(self, trace: ExecutionTrace) -> Path:
        """将执行轨迹持久化为 JSON 文件"""
        timestamp  = trace.started_at.strftime("%Y%m%d_%H%M%S")
        goal_slug  = "".join(
            c if c.isalnum() else "_" for c in trace.goal[:30]
        ).strip("_")
        filename   = f"{timestamp}_{goal_slug}.trace.json"
        file_path  = self._traces_dir / filename

        trace_dict = self._serialize(trace)
        file_path.write_text(
            json.dumps(trace_dict, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"📝 轨迹已保存: {file_path}")
        return file_path

    # ------------------------------------------------------------------
    # 序列化核心（修复版）
    # ------------------------------------------------------------------

    def _serialize(self, obj: Any, _depth: int = 0) -> Any:
        """
        递归序列化任意对象为 JSON 兼容格式

        面试要点：
        "序列化有一个重要的优先级原则：
         特殊类型必须在通用类型之前检查，否则通用规则会'吞掉'特殊类型。
         原来的 bug 就是 Enum 被 hasattr(__dict__) 先捕获，
         导致序列化进入 Enum 的内部结构，产生循环引用。

         修复后的检查顺序（从具体到抽象）：
         1. None          → null
         2. 基础标量      → 原样返回
         3. Enum          → .value（必须在 __dict__ 检查之前！）
         4. datetime      → ISO 字符串
         5. dataclass     → 用 fields() 只取业务字段（跳过私有属性）
         6. dict          → 递归处理每个 value
         7. list/tuple    → 递归处理每个元素
         8. 有 __dict__   → 只取不以 _ 开头的公开属性
         9. 兜底          → str()，绝不抛异常"

        Args:
            obj:    任意 Python 对象
            _depth: 递归深度（安全上限 50，防止真正的循环引用）

        Returns:
            JSON 可序列化的对象（dict/list/str/int/float/bool/None）
        """
        # 安全深度上限（50 层已经足够所有正常数据结构）
        if _depth > 50:
            return f"<depth_limit: {type(obj).__name__}>"

        # ── 1. None ───────────────────────────────────────────────────
        if obj is None:
            return None

        # ── 2. 基础标量（直接返回，无需递归）────────────────────────
        if isinstance(obj, (bool, int, float, str)):
            return obj

        # ── 3. Enum（必须在 __dict__ 检查之前！）─────────────────────
        #    str+Enum 的实例同时满足 isinstance(str) 和 isinstance(Enum)
        #    但我们已在步骤2处理了纯 str，这里处理 str+Enum 复合类型
        if isinstance(obj, Enum):
            return obj.value

        # ── 4. datetime ───────────────────────────────────────────────
        if isinstance(obj, datetime):
            return obj.isoformat()

        # ── 5. dataclass（用 fields() 只取声明的业务字段）────────────
        #    比 obj.__dict__ 更安全：不会包含私有属性和内部状态
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            result = {}
            for f in dataclasses.fields(obj):
                try:
                    result[f.name] = self._serialize(
                        getattr(obj, f.name), _depth + 1
                    )
                except Exception as e:
                    result[f.name] = f"<serialize_error: {e}>"
            return result

        # ── 6. dict ───────────────────────────────────────────────────
        if isinstance(obj, dict):
            return {
                str(k): self._serialize(v, _depth + 1)
                for k, v in obj.items()
            }

        # ── 7. list / tuple ───────────────────────────────────────────
        if isinstance(obj, (list, tuple)):
            return [self._serialize(item, _depth + 1) for item in obj]

        # ── 8. 有 __dict__ 的普通对象（只取公开属性）─────────────────
        if hasattr(obj, "__dict__"):
            return {
                k: self._serialize(v, _depth + 1)
                for k, v in vars(obj).items()
                if not k.startswith("_")  # 跳过私有/内部属性
            }

        # ── 9. 兜底：转字符串，绝不抛异常 ────────────────────────────
        return str(obj)

    # ==================================================================
    # 可视化 & 查询
    # ==================================================================

    def generate_mermaid(self, trace: ExecutionTrace) -> str:
        """生成 Mermaid 流程图"""
        lines = [
            "```mermaid",
            "graph TD",
            f'  START([🎯 {trace.goal[:30]}])',
        ]
        prev_id = "START"
        for step in trace.steps:
            step_id = f"STEP{step.iteration}"
            if step.tool_calls:
                tool_names = ", ".join(tc.tool_name for tc in step.tool_calls)
                all_ok = all(
                    r.status.value == "success" for r in step.tool_results
                )
                icon = "✅" if all_ok else "❌"
                lines.append(
                    f'  {step_id}["{icon} Step {step.iteration}<br/>{tool_names}"]'
                )
            else:
                lines.append(
                    f'  {step_id}["💬 Step {step.iteration}<br/>完成推理"]'
                )
            lines.append(f"  {prev_id} --> {step_id}")
            prev_id = step_id

        final_icon = "✅" if trace.success else "❌"
        lines.append(f'  END([{final_icon} 结束])')
        lines.append(f"  {prev_id} --> END")
        lines.append("```")
        lines.append(
            f"\n> 迭代: {trace.total_iterations} | "
            f"工具调用: {trace.total_tool_calls} | "
            f"Tokens: {trace.total_tokens}"
        )
        return "\n".join(lines)

    def list_traces(self, limit: int = 10) -> list[dict[str, Any]]:
        """列出最近的执行轨迹"""
        files = sorted(
            self._traces_dir.glob("*.trace.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]

        traces = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                traces.append({
                    "file":             f.name,
                    "goal":             data.get("goal", "")[:50],
                    "success":          data.get("success", False),
                    "total_tokens":     data.get("total_tokens", 0),
                    "total_iterations": data.get("total_iterations", 0),
                    "started_at":       data.get("started_at", ""),
                })
            except Exception:
                pass
        return traces