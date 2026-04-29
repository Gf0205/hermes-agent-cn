"""
src/main.py - CLI 主入口

面试要点：
"我用 Rich 库构建终端界面，原因：
 1. 原生支持 Markdown 渲染、语法高亮、进度条
 2. 跨平台（Windows/Linux/macOS）颜色支持
 3. 不需要引入 Textual 这种重框架
 4. 对标 Claude Code 的终端体验

 Slash Commands 设计：
 以 '/' 开头的输入被拦截为命令，
 其余输入作为任务目标传给 AgentLoop。
 这让用户不需要记参数，
 在对话界面就能完成所有操作。"

支持的 Slash Commands：
  /help     - 显示帮助
  /memory   - 查看记忆统计
  /skills   - 列出所有技能
  /history  - 最近的执行轨迹
  /tools    - 列出所有可用工具
  /model    - 查看当前模型配置
  /clear    - 清空工作记忆
  /exit     - 退出
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

# 触发工具自动注册
import src.tools  # noqa: F401

from src.agent_loop import AgentLoop
from src.cost_tracker import CostTracker
from src.llm_client import LLMClient
from src.memory.skill_distiller import SkillDistiller
from src.memory.memory_manager import MemoryManager
from src.models import ModelError
from src.runtime.resume_manager import ResumeManager
from src.tools.registry import registry
from src.ui.tui_app import TUIApp

# ==============================================================================
# 全局配置
# ==============================================================================

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          Hermes Agent CN  v0.1.0                        ║
║          基于阿里云百炼 · 透明架构 · 可观测              ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
## 使用方法

直接输入你的任务目标，Agent 会自动分解并执行。

**示例任务：**
- `创建一个 Flask 应用，包含用户登录接口`
- `分析当前目录的 Python 文件，统计代码行数`
- `在 /tmp 目录下创建一个 README.md 文件`

## Slash 命令

| 命令 | 说明 |
|------|------|
| `/help` | 显示此帮助 |
| `/tools` | 列出所有可用工具 |
| `/memory` | 查看记忆系统统计 |
| `/skills` | 列出所有已保存的技能 |
| `/skills suggest` | 查看自动提炼的技能草稿 |
| `/skills adopt <index>` | 采纳草稿并保存技能 |
| `/skills log` | 查看最近自动/手动采纳日志 |
| `/skills rollback [record_id]` | 回滚最近或指定自动采纳 |
| `/skills govern [status|run]` | 查看或执行技能治理策略 |
| `/recall <query> [--explain]` | 跨会话检索历史任务（可选解释模式） |
| `/recall health [n]` | 汇总最近 N 次 recall 的可用性健康度 |
| `/history` | 最近的执行轨迹 |
| `/model` | 查看当前模型配置 |
| `/sessions` | 查看可恢复会话 |
| `/resume <id>` | 从断点恢复会话 |
| `/clear` | 清空工作记忆 |
| `/exit` | 退出 |

## Policy Engine 配置

- 项目级策略：`.hermes/policies.json`
- 用户级策略：`~/.hermes-cn/policies.json`
- 参考模板：`.hermes/policies.example.json`

## Recall 引用注入（可选）

- `AGENT_RECALL_INJECT_SESSIONS=1`：在规划前注入相似历史会话摘要
- `AGENT_RECALL_INJECT_SESSIONS_K=3`：注入条数（1-8）
- `AGENT_RECALL_INJECT_INCLUDE_EXCERPT=1`：是否包含 messages 摘要片段（0/1）
- `AGENT_RECALL_INJECT_EXCERPT_MAX_CHARS=140`：每条 excerpt 最大长度
- `AGENT_RECALL_INJECT_MAX_TOTAL_CHARS=1200`：注入总长度上限（防止 prompt 膨胀）
"""


# ==============================================================================
# 配置日志
# ==============================================================================

def setup_logging(level: str = "INFO") -> None:
    """配置日志输出格式"""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 自定义格式：时间 + 级别 + 消息
    fmt = "%(asctime)s │ %(levelname)-8s │ %(name)-20s │ %(message)s"
    date_fmt = "%H:%M:%S"

    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=date_fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 降低第三方库的日志级别，减少噪音
    for noisy_logger in ["httpx", "httpcore", "openai", "chromadb", "urllib3"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


# ==============================================================================
# Slash Command 处理器
# ==============================================================================

class SlashCommandHandler:
    """
    Slash 命令处理器

    面试要点：
    "Slash Commands 是一个简单的命令路由系统。
     用字典把命令字符串映射到处理函数，
     比 if/elif 链更易扩展（新增命令只需加一行）。
     这是 Command Pattern 的轻量实现。"
    """

    def __init__(
        self,
        agent: AgentLoop,
        memory_manager: MemoryManager,
        skill_distiller: SkillDistiller,
        tracer_dir: Path,
        cost_tracker: CostTracker,
    ) -> None:
        self._agent = agent
        self._memory  = memory_manager
        self._skill_distiller = skill_distiller
        self._tracer_dir = tracer_dir
        self._cost_tracker = cost_tracker

        # 命令路由表
        self._commands: dict[str, object] = {
            "/help":    self._cmd_help,
            "/tools":   self._cmd_tools,
            "/memory":  self._cmd_memory,
            "/skills":  self._cmd_skills,
            "/recall": self._cmd_recall,
            "/history": self._cmd_history,
            "/model":   self._cmd_model,
            "/sessions": self._cmd_sessions,
            "/resume": self._cmd_resume,
            "/clear":   self._cmd_clear,
            "/exit":    self._cmd_exit,
            "/quit":    self._cmd_exit,
        }

    def handle(self, command: str) -> bool:
        """
        处理 slash 命令

        Returns:
            True  = 命令已处理
            False = 不是已知命令（应该作为任务处理）
        """
        parts = command.strip().split()
        cmd_lower = parts[0].lower()
        args = parts[1:]
        handler = self._commands.get(cmd_lower)

        if handler is None:
            console.print(
                f"[yellow]未知命令: {cmd_lower}[/yellow]\n"
                f"输入 /help 查看所有可用命令"
            )
            return True  # 已处理（显示了错误提示）

        handler(args)  # type: ignore[operator]
        return True

    def is_command(self, text: str) -> bool:
        """判断输入是否为 slash 命令"""
        return text.strip().startswith("/")

    # ── 命令实现 ──────────────────────────────────────────────────────────

    def _cmd_help(self, args: list[str] | None = None) -> None:
        console.print(Markdown(HELP_TEXT))

    def _cmd_tools(self, args: list[str] | None = None) -> None:
        """列出所有工具"""
        table = Table(title="🔧 可用工具", show_header=True, header_style="bold cyan")
        table.add_column("工具名",   style="green",  width=20)
        table.add_column("标签",     style="yellow", width=20)
        table.add_column("描述",     style="white",  width=50)

        for tool in registry.get_all():
            # 从 registry 内部获取 tags（直接访问）
            tags = registry._tags.get(tool.name, [])
            tag_str = ", ".join(tags) if tags else "—"
            # 截断描述避免换行
            desc = tool.description.replace("\n", " ")[:60]
            table.add_row(tool.name, tag_str, desc)

        console.print(table)

    def _cmd_memory(self, args: list[str] | None = None) -> None:
        """显示记忆统计"""
        stats = self._memory.get_stats()
        table = Table(title="🧠 记忆系统统计", show_header=True, header_style="bold cyan")
        table.add_column("层级",   style="green")
        table.add_column("类型",   style="yellow")
        table.add_column("数量",   style="white")

        table.add_row("层1", "工作记忆（上下文窗口）", str(stats["working_memory_messages"]))
        table.add_row("层2", "情景记忆（SQLite会话）",  str(stats["episodic_sessions"]))
        table.add_row("层3", "语义记忆（ChromaDB）",    str(stats["semantic_memories"]))
        table.add_row("层4", "程序性记忆（技能）",       str(stats["skills"]))

        console.print(table)

    def _cmd_skills(self, args: list[str] | None = None) -> None:
        """技能命令：list / suggest / adopt / log / rollback / govern"""
        subcmd = (args[0].lower() if args else "").strip()
        if subcmd in {"suggest", "draft", "drafts"}:
            drafts = self._skill_distiller.get_recent_drafts()
            if not drafts:
                console.print("[yellow]暂无技能草稿（先完成一些复杂任务）[/yellow]")
                return
            table = Table(title="🧪 技能草稿（自动提炼）", show_header=True, header_style="bold cyan")
            table.add_column("索引", style="cyan", width=6)
            table.add_column("草稿ID", style="green", width=14)
            table.add_column("技能名", style="white", width=28)
            table.add_column("来源目标", style="yellow", width=34)
            table.add_column("质量分", style="magenta", width=8)
            table.add_column("建议采纳", style="cyan", width=10)
            for idx, draft in enumerate(drafts):
                table.add_row(
                    str(idx),
                    draft.draft_id[:14],
                    draft.name[:28],
                    draft.source_goal[:34],
                    f"{draft.quality_score:.2f}",
                    "YES" if draft.recommended else "NO",
                )
            console.print(table)
            threshold = self._skill_distiller.get_auto_adopt_threshold()
            if threshold > 0:
                console.print(f"[dim]自动采纳阈值: {threshold:.2f} (环境变量 AGENT_SKILL_AUTO_ADOPT_THRESHOLD)[/dim]")
            console.print("[dim]使用 /skills adopt <索引> 采纳草稿[/dim]")
            return

        if subcmd in {"adopt", "apply"}:
            if not args or len(args) < 2:
                console.print("[yellow]用法: /skills adopt <index>[/yellow]")
                return
            try:
                draft_idx = int(args[1])
                skill_name = self._skill_distiller.adopt_draft(draft_idx, source="manual")
            except ValueError:
                console.print("[red]index 必须是整数[/red]")
                return
            except IndexError:
                console.print("[red]草稿索引越界，请先用 /skills suggest 查看[/red]")
                return
            except Exception as e:
                console.print(f"[red]采纳失败: {e}[/red]")
                return
            console.print(f"[green]✅ 已采纳技能草稿: {skill_name}[/green]")
            return

        if subcmd in {"log", "logs", "history"}:
            records = self._skill_distiller.get_recent_adoption_records(limit=10)
            if not records:
                console.print("[yellow]暂无技能采纳日志[/yellow]")
                return
            table = Table(title="🧾 技能采纳日志（最近10条）", show_header=True, header_style="bold cyan")
            table.add_column("记录ID", style="dim", width=12)
            table.add_column("时间", style="dim", width=20)
            table.add_column("来源", style="green", width=8)
            table.add_column("技能名", style="white", width=28)
            table.add_column("质量分", style="magenta", width=8)
            table.add_column("去重", style="cyan", width=8)
            table.add_column("已回滚", style="yellow", width=8)
            for item in records:
                table.add_row(
                    item.record_id[:12],
                    item.timestamp[:19].replace("T", " "),
                    item.source.upper(),
                    item.skill_name[:28],
                    f"{item.quality_score:.2f}",
                    "YES" if item.deduplicated else "NO",
                    "YES" if item.rolled_back else "NO",
                )
            console.print(table)
            return

        if subcmd in {"rollback", "undo"}:
            target_record_id = args[1].strip() if len(args) >= 2 else None
            result = self._skill_distiller.rollback_auto_adopt(record_id=target_record_id)
            if result is None:
                if target_record_id:
                    console.print(f"[yellow]未找到可回滚记录: {target_record_id}[/yellow]")
                else:
                    console.print("[yellow]没有可回滚的自动采纳记录[/yellow]")
                return
            console.print(
                f"[green]✅ 回滚成功: {result['skill_name']} "
                f"({str(result['rolled_back_at'])[:19].replace('T', ' ')})[/green]"
            )
            return

        if subcmd in {"govern", "governance"}:
            action = (args[1].lower().strip() if len(args) >= 2 else "status")
            if action not in {"status", "run"}:
                console.print("[yellow]用法: /skills govern [status|run][/yellow]")
                return

            run_result: dict[str, object] | None = None
            if action == "run":
                run_result = self._skill_distiller.run_skill_governance()

            status = self._skill_distiller.get_governance_status()
            table = Table(title="🛡️ 技能治理状态", show_header=True, header_style="bold cyan")
            table.add_column("指标", style="green", width=28)
            table.add_column("值", style="white", width=24)
            table.add_row("草稿数", str(status["drafts"]))
            table.add_row("采纳记录数", str(status["adoption_records"]))
            table.add_row("去重记录数", str(status["deduplicated_records"]))
            table.add_row("已回滚记录数", str(status["rolled_back_records"]))
            table.add_row("自动技能数", str(status["auto_skills"]))
            table.add_row("低质量自动技能数", str(status["low_quality_auto_skills"]))
            table.add_row("去重阈值", f"{float(status['dedupe_similarity_threshold']):.2f}")
            table.add_row("衰减质量阈值", f"{float(status['decay_quality_threshold']):.2f}")
            table.add_row("最小年龄(天)", str(status["decay_min_age_days"]))
            table.add_row("每次衰减步长", str(status["decay_step"]))
            table.add_row("删除阈值", str(status["decay_remove_below"]))
            if run_result is not None:
                table.add_row("本次扫描自动技能", str(run_result.get("scanned_auto_skills", 0)))
                table.add_row("本次衰减更新", str(run_result.get("decayed_updated", 0)))
                table.add_row("本次移除技能", str(run_result.get("decayed_removed", 0)))
            console.print(table)
            return

        # 默认列出已保存技能
        skills = self._memory.list_skills()
        if not skills:
            console.print("[yellow]暂无已保存的技能[/yellow]")
            return

        table = Table(title="⚡ 已保存技能", show_header=True, header_style="bold cyan")
        table.add_column("技能名",   style="green",  width=25)
        table.add_column("标签",     style="yellow", width=20)
        table.add_column("使用次数", style="cyan",   width=8)
        table.add_column("描述",     style="white",  width=40)

        for skill in skills:
            tags = ", ".join(skill.get("tags", [])) or "—"
            table.add_row(
                skill.get("name", ""),
                tags,
                str(skill.get("use_count", 0)),
                skill.get("description", "")[:40],
            )
        console.print(table)

    def _cmd_history(self, args: list[str] | None = None) -> None:
        """显示最近执行轨迹"""
        from src.observability.tracer import ExecutionTracer
        tracer = ExecutionTracer()
        traces = tracer.list_traces(limit=5)

        if not traces:
            console.print("[yellow]暂无执行记录[/yellow]")
            return

        table = Table(title="📜 最近执行记录", show_header=True, header_style="bold cyan")
        table.add_column("时间",     style="dim",    width=20)
        table.add_column("目标",     style="white",  width=35)
        table.add_column("状态",     style="green",  width=6)
        table.add_column("Tokens",  style="yellow", width=8)
        table.add_column("迭代",    style="cyan",   width=6)

        for t in traces:
            status = "✅" if t["success"] else "❌"
            # 从文件名提取时间
            parts = t["file"].split("_")
            time_str = f"{parts[0]} {parts[1][:6]}" if len(parts) >= 2 else "—"
            table.add_row(
                time_str,
                t["goal"][:35],
                status,
                str(t["total_tokens"]),
                str(t["total_iterations"]),
            )
        console.print(table)

    def _cmd_recall(self, args: list[str] | None = None) -> None:
        argv = list(args or [])
        if not argv:
            console.print("[yellow]用法: /recall <关键词> [--explain][/yellow]")
            console.print("[yellow]用法: /recall health [N][/yellow]")
            return

        # Subcommand: health
        if argv and argv[0].strip().lower() == "health":
            from src.observability.recall_logger import RecallLogger

            n = 30
            if len(argv) >= 2:
                try:
                    n = int(argv[1])
                except Exception:
                    n = 30
            logger = RecallLogger()
            rows = logger.read_last(n=n)
            if not rows:
                console.print(f"[yellow]暂无 recall 日志：{logger.path}[/yellow]")
                return

            def _f(obj: dict, path: list[str], default: float = 0.0) -> float:
                cur: Any = obj
                for k in path:
                    if not isinstance(cur, dict) or k not in cur:
                        return default
                    cur = cur[k]
                try:
                    return float(cur)
                except Exception:
                    return default

            useful3 = [_f(r, ["metrics", "useful_at_3"]) for r in rows]
            div3 = [_f(r, ["metrics", "component_diversity_at_3"]) for r in rows]

            avg_useful3 = sum(useful3) / max(1, len(useful3))
            avg_div3 = sum(div3) / max(1, len(div3))

            try:
                useful3_min = float(os.getenv("AGENT_RECALL_HEALTH_USEFUL_AT_3_MIN", "0.80"))
            except Exception:
                useful3_min = 0.80
            try:
                div3_min = float(os.getenv("AGENT_RECALL_HEALTH_DIVERSITY_AT_3_MIN", "0.34"))
            except Exception:
                div3_min = 0.34

            status = "OK"
            if avg_useful3 < useful3_min or avg_div3 < div3_min:
                status = "WARN"

            table = Table(title=f"🩺 Recall Health (last {len(rows)}) [{status}]", show_header=True, header_style="bold cyan")
            table.add_column("指标", style="green", width=24)
            table.add_column("均值", style="yellow", width=10)
            table.add_column("阈值", style="dim", width=10)
            table.add_row("useful_at_3", f"{avg_useful3:.3f}", f">= {useful3_min:.2f}")
            table.add_row("component_diversity_at_3", f"{avg_div3:.3f}", f">= {div3_min:.2f}")
            console.print(table)
            console.print(f"[dim]日志文件: {logger.path}[/dim]")
            return

        explain = False
        if "--explain" in [a.strip().lower() for a in argv]:
            explain = True
            argv = [a for a in argv if a.strip().lower() != "--explain"]

        query = " ".join(argv).strip()
        if not query:
            console.print("[yellow]用法: /recall <关键词> [--explain][/yellow]")
            return

        hits = self._memory.search_sessions(query, limit=8)
        if not hits:
            console.print(f"[yellow]未检索到相关会话：{query}[/yellow]")
            return

        useful_reason_tokens = {
            "title-match",
            "goal-match",
            "messages-match",
            "semantic-strong",
            "semantic-related",
            "lexical-related",
        }

        def _is_useful(item: dict) -> bool:
            reason = str(item.get("match_reason", "")).strip()
            if not reason:
                return False
            tokens = {t.strip() for t in reason.split(",") if t.strip()}
            return any(t in useful_reason_tokens for t in tokens)

        def _dominant_component(item: dict) -> str:
            bd = item.get("score_breakdown", {}) or {}
            try:
                lexical = float(bd.get("lexical", 0.0) or 0.0)
                semantic = float(bd.get("semantic", 0.0) or 0.0)
                recency = float(bd.get("recency", 0.0) or 0.0)
            except Exception:
                return "-"

            # Deterministic tie-breaking: lexical > semantic > recency
            if lexical >= semantic and lexical >= recency:
                return "lexical"
            if semantic >= lexical and semantic >= recency:
                return "semantic"
            return "recency"

        def _fmt_breakdown(item: dict) -> str:
            bd = item.get("score_breakdown", {}) or {}
            try:
                l = float(bd.get("lexical", 0.0) or 0.0)
                s = float(bd.get("semantic", 0.0) or 0.0)
                r = float(bd.get("recency", 0.0) or 0.0)
            except Exception:
                return "-"
            return f"L{l:.2f} S{s:.2f} R{r:.2f}"

        def _weights_from_hits(items: list[dict]) -> dict[str, float]:
            for it in items:
                bd = it.get("score_breakdown", {}) or {}
                w = (bd.get("weights") or {}) if isinstance(bd, dict) else {}
                if isinstance(w, dict) and w:
                    try:
                        return {
                            "lexical": float(w.get("lexical", 0.0) or 0.0),
                            "semantic": float(w.get("semantic", 0.0) or 0.0),
                            "recency": float(w.get("recency", 0.0) or 0.0),
                        }
                    except Exception:
                        return {}
            return {}

        def _component_diversity_at_3(items: list[dict]) -> float:
            top3 = items[:3]
            if not top3:
                return 0.0
            comps = {_dominant_component(it) for it in top3 if _dominant_component(it) != "-"}
            return len(comps) / 3.0

        # Log recall event (best-effort).
        try:
            from src.observability.recall_logger import RecallEvent, RecallLogger, now_iso

            top3 = hits[:3]
            useful_at_1 = 1.0 if any(_is_useful(h) for h in hits[:1]) else 0.0
            useful_at_3 = 1.0 if any(_is_useful(h) for h in top3) else 0.0
            diversity_at_3 = _component_diversity_at_3(hits)
            weights = _weights_from_hits(hits)
            RecallLogger().append(
                RecallEvent(
                    timestamp=now_iso(),
                    query=query,
                    limit=8,
                    top_ids=[str(h.get("id", "")) for h in hits[:8]],
                    useful_at_1=useful_at_1,
                    useful_at_3=useful_at_3,
                    component_diversity_at_3=diversity_at_3,
                    weights=weights,
                    extra={"explain": explain},
                )
            )
        except Exception:
            pass

        table = Table(title=f"🔎 会话检索：{query}", show_header=True, header_style="bold cyan")
        table.add_column("Session", style="green", width=12)
        table.add_column("标题", style="white", width=24)
        table.add_column("目标", style="yellow", width=36)
        table.add_column("命中依据", style="cyan", width=22)
        table.add_column("可用", style="green", width=4)
        table.add_column("主导", style="blue", width=8)
        table.add_column("拆分", style="dim", width=18)
        table.add_column("分数", style="magenta", width=8)
        table.add_column("更新时间", style="dim", width=20)
        for item in hits:
            useful_flag = "Y" if _is_useful(item) else "N"
            table.add_row(
                str(item.get("id", ""))[:12],
                str(item.get("title", ""))[:24],
                str(item.get("goal", ""))[:36],
                str(item.get("match_reason", "-"))[:22],
                useful_flag,
                _dominant_component(item)[:8],
                _fmt_breakdown(item)[:18],
                f"{float(item.get('score', 0.0)):.2f}",
                str(item.get("updated_at", ""))[:19].replace("T", " "),
            )
        console.print(table)

        if explain:
            weights = _weights_from_hits(hits)
            diversity = _component_diversity_at_3(hits)
            useful3 = "Y" if any(_is_useful(h) for h in hits[:3]) else "N"
            console.print(
                f"[dim]useful@3={useful3} | component_diversity@3={diversity:.3f} | "
                f"weights={weights or '-'}[/dim]"
            )

    def _cmd_model(self, args: list[str] | None = None) -> None:
        """显示模型配置"""
        from src.llm_client import LLMClient
        client = LLMClient()
        info = client.get_model_info()
        cost = self._cost_tracker.get_session_summary()
        global_cost = self._cost_tracker.get_global_summary()

        table = Table(title="🤖 模型配置", show_header=True, header_style="bold cyan")
        table.add_column("配置项", style="green",  width=20)
        table.add_column("当前值", style="yellow", width=40)

        label_map = {
            "strong_model":    "强力模型（规划/反思）",
            "fast_model":      "快速模型（战术/评估）",
            "embedding_model": "向量模型（记忆）",
            "base_url":        "API端点",
            "enable_thinking": "思考链模式",
            "timeout":         "超时时间（秒）",
        }
        for key, value in info.items():
            label = label_map.get(key, key)
            table.add_row(label, value)

        table.add_row("会话累计Tokens", str(cost["session_tokens"]))
        table.add_row("会话累计费用", f"${cost['session_cost_usd']:.6f}")
        table.add_row("全局累计Tokens", str(global_cost["global_tokens"]))
        table.add_row("全局累计费用", f"${global_cost['global_cost_usd']:.6f}")
        console.print(table)

    def _cmd_sessions(self, args: list[str] | None = None) -> None:
        records = self._agent.list_checkpoints(limit=10)
        if not records:
            console.print("[yellow]暂无可恢复会话[/yellow]")
            return
        table = Table(title="🧩 可恢复会话", show_header=True, header_style="bold cyan")
        table.add_column("Session", style="green", width=12)
        table.add_column("可恢复", style="green", width=8)
        table.add_column("状态", style="yellow", width=12)
        table.add_column("更新时间", style="dim", width=24)
        table.add_column("目标", style="white", width=40)
        table.add_column("进度", style="cyan", width=12)
        table.add_column("可恢复原因", style="white", width=26)
        table.add_column("上次失败子目标", style="red", width=22)
        for r in records:
            if r.total_subgoals > 0:
                pct = int(len(r.completed_subgoals) / max(1, r.total_subgoals) * 100)
                progress = f"{len(r.completed_subgoals)}/{r.total_subgoals} ({pct}%)"
            else:
                progress = f"{len(r.completed_subgoals)}/?"
            resumable_flag = "✅" if ResumeManager.is_resumable_status(r.status) else "⛔"
            last_failed = str(r.metadata.get("last_failed_subgoal", "") or "-")
            table.add_row(
                r.session_id,
                resumable_flag,
                r.status,
                r.updated_at[:19].replace("T", " "),
                r.goal[:40],
                progress,
                (r.resumable_reason or "-")[:26],
                last_failed[:22],
            )
        console.print(table)

    def _cmd_resume(self, args: list[str] | None = None) -> None:
        if not args:
            console.print("[yellow]用法: /resume <session_id>[/yellow]")
            return
        session_id = args[0]
        try:
            preview = self._agent.get_resume_preview(session_id)
            total = int(preview["total_subgoals"])
            completed = int(preview["completed_subgoals"])
            if total > 0:
                progress = f"{completed}/{total} ({int(completed / max(1, total) * 100)}%)"
            else:
                progress = f"{completed}/?"
            console.print(Panel(
                f"[bold]恢复预览[/bold]\n"
                f"- 会话: {preview['session_id']}\n"
                f"- 当前状态: {preview['status']}\n"
                f"- 将跳过已完成子目标: {completed}\n"
                f"- 进度: {progress}\n"
                f"- 下一待执行子目标: {preview['next_subgoal']}\n"
                f"- 说明: {preview['resumable_reason'] or '-'}",
                title=f"♻️ 即将恢复 [{session_id}]",
                border_style="blue",
            ))
            trace = self._agent.resume(session_id)
        except Exception as e:
            console.print(f"[red]恢复失败: {e}[/red]")
            return

        if trace.success:
            console.print(Panel(
                Markdown(trace.final_answer or ""),
                title=f"✅ 恢复完成 [{session_id}]",
                border_style="green",
            ))
        else:
            console.print(Panel(
                trace.final_answer or "恢复执行失败",
                title=f"❌ 恢复失败 [{session_id}]",
                border_style="red",
            ))
        console.print(
            f"[dim]  📊 Tokens: {trace.total_tokens} | "
            f"迭代: {trace.total_iterations} | "
            f"工具调用: {trace.total_tool_calls}[/dim]"
        )
        _print_cost_summary(self._cost_tracker)

    def _cmd_clear(self, args: list[str] | None = None) -> None:
        """清空工作记忆"""
        self._memory.clear_working_memory()
        console.print("[green]✅ 工作记忆已清空[/green]")

    def _cmd_exit(self, args: list[str] | None = None) -> None:
        """退出程序"""
        console.print("\n[cyan]再见！👋[/cyan]\n")
        sys.exit(0)


# ==============================================================================
# 主入口
# ==============================================================================

def main() -> None:
    """CLI主函数"""
    # ── 参数解析 ──────────────────────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser(
        prog="hermes-cn",
        description="Hermes Agent CN - 透明架构 AI Agent"
    )
    parser.add_argument(
        "--log-level", "-l",
        default=os.getenv("AGENT_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）"
    )
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="禁用自我反思（节省Token，适合简单任务）"
    )
    parser.add_argument(
        "--goal", "-g",
        type=str,
        help="直接指定目标（非交互模式）"
    )
    parser.add_argument(
        "--context", "-c",
        type=str,
        default="",
        help="额外上下文（工作目录、项目信息等）"
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="开启实时 TUI 事件面板",
    )
    args = parser.parse_args()

    # ── 配置日志 ──────────────────────────────────────────────────────────────
    setup_logging(args.log_level)

    # ── 打印 Banner ───────────────────────────────────────────────────────────
    console.print(Panel(
        Text(BANNER, style="bold cyan", justify="center"),
        border_style="cyan",
        padding=(0, 2),
    ))

    # ── 初始化核心组件 ────────────────────────────────────────────────────────
    console.print("[dim]正在初始化...[/dim]")

    try:
        llm_client = LLMClient()
    except ValueError as e:
        console.print(f"[bold red]❌ 初始化失败: {e}[/bold red]")
        sys.exit(1)

    memory_manager = MemoryManager(llm_client)
    skill_distiller = SkillDistiller(memory_manager)
    cost_tracker = CostTracker()

    agent = AgentLoop(
        llm_client=llm_client,
        tool_registry=registry,
        memory_manager=memory_manager,
        enable_reflection=not args.no_reflection,
    )

    tracer_dir = Path.home() / ".hermes-cn" / "traces"

    slash_handler = SlashCommandHandler(
        agent=agent,
        memory_manager=memory_manager,
        skill_distiller=skill_distiller,
        tracer_dir=tracer_dir,
        cost_tracker=cost_tracker,
    )

    # 打印工具数量
    tool_count = len(registry)
    model_info = llm_client.get_model_info()
    console.print(
        f"[green]✅ 初始化完成[/green] | "
        f"工具: [cyan]{tool_count}[/cyan] 个 | "
        f"模型: [cyan]{model_info['strong_model']}[/cyan] | "
        f"反思: [cyan]{'开启' if not args.no_reflection else '关闭'}[/cyan]"
    )
    console.print("[dim]输入 /help 查看帮助，直接输入任务目标开始执行[/dim]\n")

    tui_app = TUIApp(console) if args.tui else None
    if args.tui:
        console.print("[cyan]🖥️ TUI 事件面板已开启[/cyan]")

    try:
        if tui_app:
            with tui_app:
                # ── 非交互模式（--goal 参数）─────────────────────────────────────────────
                if args.goal:
                    _run_single_goal(agent, args.goal, args.context, cost_tracker)
                    return
                # ── 交互模式（主循环）────────────────────────────────────────────────────
                _interactive_loop(agent, slash_handler, args.context, cost_tracker)
        else:
            # ── 非交互模式（--goal 参数）─────────────────────────────────────────────
            if args.goal:
                _run_single_goal(agent, args.goal, args.context, cost_tracker)
                return
            # ── 交互模式（主循环）────────────────────────────────────────────────────
            _interactive_loop(agent, slash_handler, args.context, cost_tracker)
    finally:
        agent.close()


def _run_single_goal(
    agent: AgentLoop,
    goal: str,
    context: str,
    cost_tracker: CostTracker,
) -> None:
    """非交互模式：执行单个目标后退出"""
    console.print(Panel(
        f"[bold]目标:[/bold] {goal}",
        title="🚀 开始执行",
        border_style="green",
    ))

    trace = agent.run(goal=goal, context=context)

    # 打印最终结果
    if trace.success:
        console.print(Panel(
            Markdown(trace.final_answer or "任务完成"),
            title="✅ 执行完成",
            border_style="green",
        ))
    else:
        console.print(Panel(
            trace.final_answer or "任务失败",
            title="❌ 执行失败",
            border_style="red",
        ))

    console.print(
        f"[dim]Tokens: {trace.total_tokens} | "
        f"迭代: {trace.total_iterations} | "
        f"工具调用: {trace.total_tool_calls}[/dim]"
    )
    _print_cost_summary(cost_tracker)


def _interactive_loop(
    agent: AgentLoop,
    slash_handler: SlashCommandHandler,
    default_context: str,
    cost_tracker: CostTracker,
) -> None:
    """
    交互式主循环

    面试要点：
    "交互循环的设计原则：
     1. 每次输入后给出明确反馈
     2. Ctrl+C 不退出程序，而是中断当前任务
     3. Slash 命令不走 Agent，直接处理
     4. 空输入直接忽略，不触发任何操作"
    """
    session_id_counter = 0

    while True:
        # ── 读取用户输入 ──────────────────────────────────────────────
        try:
            user_input = console.input("[bold cyan]你 ❯[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]再见！👋[/cyan]")
            break

        # 空输入跳过
        if not user_input:
            continue

        # ── Slash 命令处理 ────────────────────────────────────────────
        if slash_handler.is_command(user_input):
            slash_handler.handle(user_input)
            continue

        # ── 作为任务目标执行 ──────────────────────────────────────────
        session_id_counter += 1
        session_id = f"s{session_id_counter:03d}"

        console.print(Panel(
            f"[bold]目标:[/bold] {user_input}",
            title=f"🚀 执行任务 [{session_id}]",
            border_style="blue",
            padding=(0, 1),
        ))

        try:
            trace = agent.run(
                goal=user_input,
                session_id=session_id,
                context=default_context,
            )

            # 打印结果摘要
            if trace.success:
                console.print(Panel(
                    Markdown(trace.final_answer or ""),
                    title="✅ 任务完成",
                    border_style="green",
                ))
            else:
                console.print(Panel(
                    trace.final_answer or "任务未能完成",
                    title="❌ 任务失败",
                    border_style="red",
                ))

            # 底部统计栏
            console.print(
                f"[dim]  📊 Tokens: {trace.total_tokens} | "
                f"迭代: {trace.total_iterations} | "
                f"工具调用: {trace.total_tool_calls}[/dim]\n"
            )
            _print_cost_summary(cost_tracker)

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  任务已中断[/yellow]\n")


def _print_cost_summary(cost_tracker: CostTracker) -> None:
    summary = cost_tracker.get_session_summary()
    console.print(
        f"[dim]  💰 会话累计费用: ${summary['session_cost_usd']:.6f} | "
        f"累计Tokens: {summary['session_tokens']}[/dim]"
    )


if __name__ == "__main__":
    main()