"""
src/memory/memory_manager.py - 4层记忆管理器

面试要点（核心！）：
"我实现了认知科学中的4层记忆模型：

 层1 工作记忆（Working Memory）
   = 当前对话的messages列表（上下文窗口）
   特点：快速、有限（受Token限制）、会话结束后消失
   实现：内存中的list[dict]

 层2 情景记忆（Episodic Memory）
   = 历史会话记录（SQLite）
   特点：'上周我们讨论了什么？'
   实现：sqlite-utils，按session_id检索

 层3 语义记忆（Semantic Memory）
   = 知识、事实、用户偏好（ChromaDB向量搜索）
   特点：'什么是最优的Python异步模式？'
   实现：ChromaDB + Dashscope Embedding

 层4 程序性记忆（Procedural Memory）
   = 可复用技能（JSON文件）
   特点：'如何创建FastAPI项目？'— 步骤固定，直接复用
   实现：本地JSON文件，类似Hermes的Skills系统

这个分层设计让Agent能在不同粒度上检索信息，
避免把所有记忆都塞进上下文窗口导致Token爆炸。"
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.llm_client import LLMClient
from src.memory.chroma_store import ChromaMemoryStore
from src.models import MemoryEntry, Session

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    4层记忆统一管理器

    使用方式：
        memory = MemoryManager(llm_client)

        # 存储知识
        memory.remember("Python的GIL在CPU密集型任务时会成为瓶颈", importance=0.8)

        # 检索相关记忆
        relevant = memory.recall("如何优化Python性能？", top_k=3)

        # 保存会话
        memory.save_session(session)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        data_dir: Optional[str] = None,
    ) -> None:
        self._llm = llm_client

        # 数据根目录
        base_dir = data_dir or os.getenv(
            "AGENT_DATA_DIR",
            str(Path.home() / ".hermes-cn")
        )
        self._data_dir = Path(base_dir).expanduser().resolve()
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # 层1：工作记忆（运行时内存）
        self._working_memory: list[dict[str, str]] = []

        # 层2：情景记忆（SQLite）
        self._db_path = self._data_dir / "sessions.db"
        self._fts_enabled = False
        self._init_sqlite()

        # Hybrid recall ranking weights (lexical/semantic/recency).
        # Defaults keep backwards-compatible behavior.
        self._recall_weights: tuple[float, float, float] | None = None

        # 层3：语义记忆（ChromaDB）
        self._semantic_store = ChromaMemoryStore(
            persist_dir=str(self._data_dir / "chroma_db"),
        )

        # 层4：程序性记忆（JSON文件）
        self._skills_dir = self._data_dir / "skills"
        self._skills_dir.mkdir(exist_ok=True)

        logger.info(
            f"MemoryManager初始化 | 数据目录: {self._data_dir} | "
            f"语义记忆: {self._semantic_store.count()} 条"
        )

    def _load_recall_weights(self) -> tuple[float, float, float]:
        def _f(env_key: str, default: float) -> float:
            raw = os.getenv(env_key, "").strip()
            if not raw:
                return default
            try:
                v = float(raw)
                if math.isnan(v) or math.isinf(v):
                    return default
                return max(0.0, v)
            except Exception:
                return default

        w_lex = _f("AGENT_RECALL_WEIGHT_LEXICAL", 0.60)
        w_sem = _f("AGENT_RECALL_WEIGHT_SEMANTIC", 0.30)
        w_rec = _f("AGENT_RECALL_WEIGHT_RECENCY", 0.10)

        total = w_lex + w_sem + w_rec
        if total <= 0.0:
            return (0.60, 0.30, 0.10)
        return (w_lex / total, w_sem / total, w_rec / total)

    def _get_recall_weights(self) -> tuple[float, float, float]:
        if self._recall_weights is None:
            self._recall_weights = self._load_recall_weights()
        return self._recall_weights

    # ==================================================================
    # 层1：工作记忆（Working Memory）
    # ==================================================================

    def add_to_working_memory(self, role: str, content: str) -> None:
        """添加消息到工作记忆（当前对话上下文）"""
        self._working_memory.append({"role": role, "content": content})

    def get_working_memory(self) -> list[dict[str, str]]:
        """获取当前工作记忆（完整对话历史）"""
        return self._working_memory.copy()

    def clear_working_memory(self) -> None:
        """清空工作记忆（开始新任务时调用）"""
        self._working_memory.clear()

    def get_context_window(
        self,
        system_prompt: str,
        max_recent: int = 20,
    ) -> list[dict[str, str]]:
        """
        构建LLM调用的上下文窗口

        面试要点：
        "上下文窗口不是无限的（qwen-max约32K tokens）。
         我取最近N条消息而不是全部历史，
         并在开头注入系统提示和相关的语义记忆摘要，
         这样既保持了对话连贯性，又不超Token限制。"

        Args:
            system_prompt: 系统提示词
            max_recent: 保留的最近消息数量

        Returns:
            OpenAI格式的messages列表
        """
        messages = [{"role": "system", "content": system_prompt}]
        recent = self._working_memory[-max_recent:]
        messages.extend(recent)
        return messages

    # ==================================================================
    # 层2：情景记忆（Episodic Memory - SQLite）
    # ==================================================================

    def _init_sqlite(self) -> None:
        """初始化SQLite数据库表结构"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                messages TEXT,          -- JSON格式
                created_at TEXT,
                updated_at TEXT,
                tags TEXT,              -- 逗号分隔
                goal TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_created_at
            ON sessions(created_at)
        """)
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts
                USING fts5(
                    session_id UNINDEXED,
                    title,
                    messages,
                    goal
                )
            """)
            self._fts_enabled = True
        except sqlite3.OperationalError:
            # 某些SQLite构建不带FTS5，回退到LIKE查询
            self._fts_enabled = False
        conn.commit()
        conn.close()

    def save_session(self, session: Session) -> None:
        """保存会话到SQLite"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            INSERT OR REPLACE INTO sessions
            (id, title, messages, created_at, updated_at, tags, goal)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.title,
            json.dumps(session.messages, ensure_ascii=False),
            session.created_at.isoformat(),
            datetime.now().isoformat(),
            ",".join(session.tags),
            session.metadata.get("goal", ""),
        ))
        if self._fts_enabled:
            conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (session.id,))
            conn.execute(
                "INSERT INTO sessions_fts(session_id, title, messages, goal) VALUES (?, ?, ?, ?)",
                (
                    session.id,
                    session.title,
                    json.dumps(session.messages, ensure_ascii=False),
                    session.metadata.get("goal", ""),
                ),
            )
        conn.commit()
        conn.close()
        logger.debug(f"会话已保存: {session.id[:8]}")

    def load_recent_sessions(self, limit: int = 10) -> list[Session]:
        """加载最近的会话记录"""
        conn = sqlite3.connect(str(self._db_path))
        rows = conn.execute("""
            SELECT id, title, messages, created_at, tags, goal
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append(Session(
                id=row[0],
                title=row[1] or "",
                messages=json.loads(row[2] or "[]"),
                created_at=datetime.fromisoformat(row[3]),
                tags=row[4].split(",") if row[4] else [],
                metadata={"goal": row[5] or ""},
            ))
        return sessions

    def search_sessions(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        跨会话检索（候选召回 + 融合重排）

        流程：
        1) 候选召回：优先 FTS5，回退 LIKE；若仍为空，回退最近会话
        2) 词法打分：title/goal/messages 命中加权
        3) 语义打分：query 与候选文本 embedding 余弦相似度
        4) 新鲜度打分：越新的会话分数越高
        5) 融合排序：0.60 * lexical + 0.30 * semantic + 0.10 * recency
        """
        q = query.strip()
        if not q:
            return []

        candidate_limit = max(limit * 6, 20)
        candidates = self._search_session_candidates(q, candidate_limit=candidate_limit)
        if not candidates:
            return []

        semantic_scores = self._compute_semantic_scores(q, candidates)
        recency_scores = self._compute_recency_scores(candidates)
        w_lex, w_sem, w_rec = self._get_recall_weights()

        ranked: list[dict[str, Any]] = []
        for idx, row in enumerate(candidates):
            lexical = self._compute_lexical_score(q, row)
            semantic = semantic_scores[idx] if idx < len(semantic_scores) else 0.0
            recency = recency_scores[idx] if idx < len(recency_scores) else 0.0
            hybrid = w_lex * lexical + w_sem * semantic + w_rec * recency
            reasons = self._build_match_reasons(q, row, lexical, semantic)
            ranked.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "goal": row["goal"],
                    "updated_at": row["updated_at"],
                    "score": round(hybrid, 4),
                    "match_reason": ", ".join(reasons[:3]),
                    "score_breakdown": {
                        "lexical": round(lexical, 4),
                        "semantic": round(semantic, 4),
                        "recency": round(recency, 4),
                        "weights": {
                            "lexical": round(w_lex, 4),
                            "semantic": round(w_sem, 4),
                            "recency": round(w_rec, 4),
                        },
                    },
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        if self._is_diversify_enabled() and len(ranked) > 1:
            ranked = self._diversify_rerank(ranked, limit=limit)
        return ranked[:limit]

    def _is_diversify_enabled(self) -> bool:
        raw = os.getenv("AGENT_RECALL_DIVERSIFY_TOPK", "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _diversify_rerank(self, ranked: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """
        MMR-style reranking to improve top-k diversity.

        Objective:
          maximize  lambda * relevance(score) - (1-lambda) * similarity(candidate, selected)
        """
        try:
            lam = float(os.getenv("AGENT_RECALL_DIVERSIFY_LAMBDA", "0.88") or 0.88)
        except Exception:
            lam = 0.88
        lam = max(0.0, min(1.0, lam))

        pool_size = max(limit * 6, 20)
        pool = ranked[: min(len(ranked), pool_size)]

        selected: list[dict[str, Any]] = []
        remaining = list(pool)

        # Always start from the best-scoring item.
        selected.append(remaining.pop(0))

        while remaining and len(selected) < max(1, limit):
            best_idx = 0
            best_val = -1e9
            for i, cand in enumerate(remaining):
                rel = float(cand.get("score", 0.0) or 0.0)
                sim = max(self._hit_similarity(cand, s) for s in selected)
                val = lam * rel - (1.0 - lam) * sim
                if val > best_val:
                    best_val = val
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        # Append the rest in original score order.
        selected_ids = {str(x.get("id", "")) for x in selected}
        tail = [x for x in ranked if str(x.get("id", "")) not in selected_ids]
        return selected + tail

    def _hit_similarity(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        """
        Cheap similarity proxy for diversification:
          - dominant component match: strong similarity
          - match_reason overlap: moderate similarity
          - title/goal char overlap: weak similarity
        """
        if self._dominant_component(a) == self._dominant_component(b):
            return 0.85

        ra = self._reason_tokens(a)
        rb = self._reason_tokens(b)
        if ra and rb:
            j = len(ra & rb) / max(1, len(ra | rb))
            if j >= 0.5:
                return 0.65
            if j > 0:
                return 0.35

        ta = (str(a.get("title", "")) + " " + str(a.get("goal", ""))).strip()
        tb = (str(b.get("title", "")) + " " + str(b.get("goal", ""))).strip()
        cjk = self._cjk_overlap_ratio(ta, tb)
        return 0.20 * cjk

    def _dominant_component(self, hit: dict[str, Any]) -> str:
        bd = hit.get("score_breakdown", {}) or {}
        try:
            lexical = float(bd.get("lexical", 0.0) or 0.0)
            semantic = float(bd.get("semantic", 0.0) or 0.0)
            recency = float(bd.get("recency", 0.0) or 0.0)
        except Exception:
            return "unknown"

        if lexical >= semantic and lexical >= recency:
            return "lexical"
        if semantic >= lexical and semantic >= recency:
            return "semantic"
        return "recency"

    def _reason_tokens(self, hit: dict[str, Any]) -> set[str]:
        reason = str(hit.get("match_reason", "")).strip()
        if not reason:
            return set()
        return {t.strip() for t in reason.split(",") if t.strip()}

    def _search_session_candidates(self, query: str, candidate_limit: int) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(self._db_path))
        rows: list[tuple[Any, ...]] = []

        if self._fts_enabled:
            try:
                fts_query = self._build_fts_query(query)
                rows = conn.execute(
                    """
                    SELECT s.id, s.title, s.goal, s.updated_at, s.messages, 'fts'
                    FROM sessions_fts f
                    JOIN sessions s ON s.id = f.session_id
                    WHERE sessions_fts MATCH ?
                    ORDER BY bm25(sessions_fts) ASC, s.updated_at DESC
                    LIMIT ?
                    """,
                    (fts_query, candidate_limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            except Exception:
                rows = []

        if not rows:
            like_q = f"%{query}%"
            rows = conn.execute(
                """
                SELECT id, title, goal, updated_at, messages, 'like'
                FROM sessions
                WHERE title LIKE ? OR messages LIKE ? OR goal LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (like_q, like_q, like_q, candidate_limit),
            ).fetchall()

        if not rows:
            rows = conn.execute(
                """
                SELECT id, title, goal, updated_at, messages, 'recent'
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (candidate_limit,),
            ).fetchall()

        conn.close()

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            sid = str(row[0] or "")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            deduped.append(
                {
                    "id": sid,
                    "title": str(row[1] or ""),
                    "goal": str(row[2] or ""),
                    "updated_at": str(row[3] or ""),
                    "messages": str(row[4] or ""),
                    "source": str(row[5] or ""),
                }
            )
        return deduped

    def _compute_lexical_score(self, query: str, row: dict[str, Any]) -> float:
        title = str(row.get("title", ""))
        goal = str(row.get("goal", ""))
        messages = str(row.get("messages", ""))
        source = str(row.get("source", ""))

        score = 0.0
        q_lower = query.lower()
        if q_lower in title.lower():
            score += 1.0
        if q_lower in goal.lower():
            score += 0.85
        if q_lower in messages.lower():
            score += 0.50

        term_overlap = self._term_overlap_ratio(query, f"{title} {goal} {messages[:300]}")
        cjk_overlap = self._cjk_overlap_ratio(query, f"{title} {goal} {messages[:300]}")
        score += 0.25 * term_overlap + 0.30 * cjk_overlap

        if source == "fts":
            score += 0.15
        return max(0.0, min(1.0, score))

    def _term_overlap_ratio(self, query: str, text: str) -> float:
        q_terms = set(re.findall(r"[a-zA-Z0-9_]{2,}", query.lower()))
        if not q_terms:
            return 0.0
        t_terms = set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))
        if not t_terms:
            return 0.0
        overlap = len(q_terms & t_terms)
        return overlap / max(1, len(q_terms))

    def _build_fts_query(self, query: str) -> str:
        """
        Make FTS query safer and more expressive.

        - If query contains multiple whitespace-separated terms, use AND.
        - Strip characters that often break FTS parser.
        """
        q = (query or "").strip()
        if not q:
            return ""
        q = re.sub(r'["\'`]', " ", q)
        parts = [p.strip() for p in re.split(r"\s+", q) if p.strip()]
        if len(parts) <= 1:
            return parts[0]
        return " AND ".join(parts[:8])

    def _cjk_overlap_ratio(self, query: str, text: str) -> float:
        """
        Rough overlap for CJK queries: compare bigram sets.
        Helps Chinese queries where alnum token overlap is empty.
        """
        q = "".join(self._cjk_chars(query))
        if len(q) < 2:
            return 0.0
        t = "".join(self._cjk_chars(text))
        if len(t) < 2:
            return 0.0
        q_bi = self._cjk_bigrams(q)
        if not q_bi:
            return 0.0
        t_bi = self._cjk_bigrams(t)
        if not t_bi:
            return 0.0
        overlap = len(q_bi & t_bi)
        return overlap / max(1, len(q_bi))

    def _cjk_chars(self, s: str) -> list[str]:
        out: list[str] = []
        for ch in s:
            code = ord(ch)
            if (
                0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
                or 0x3400 <= code <= 0x4DBF  # Extension A
                or 0x3040 <= code <= 0x30FF  # JP Kana
                or 0xAC00 <= code <= 0xD7AF  # KR Hangul
            ):
                out.append(ch)
        return out

    def _cjk_bigrams(self, s: str) -> set[str]:
        if len(s) < 2:
            return set()
        return {s[i : i + 2] for i in range(len(s) - 1)}

    def _compute_semantic_scores(self, query: str, rows: list[dict[str, Any]]) -> list[float]:
        if not hasattr(self._llm, "embed"):
            return [0.0 for _ in rows]
        try:
            candidate_texts = [self._build_candidate_text(row) for row in rows]
            vectors = self._llm.embed([query, *candidate_texts])
            if not vectors or len(vectors) != len(candidate_texts) + 1:
                return [0.0 for _ in rows]
            query_vec = vectors[0]
            return [self._cosine_similarity(query_vec, vec) for vec in vectors[1:]]
        except Exception:
            return [0.0 for _ in rows]

    def _build_candidate_text(self, row: dict[str, Any]) -> str:
        title = str(row.get("title", ""))
        goal = str(row.get("goal", ""))
        messages = str(row.get("messages", ""))[:600]
        return f"title: {title}\ngoal: {goal}\nmessages: {messages}"

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def _compute_recency_scores(self, rows: list[dict[str, Any]]) -> list[float]:
        timestamps: list[datetime | None] = [self._parse_time(row.get("updated_at", "")) for row in rows]
        valid = [ts for ts in timestamps if ts is not None]
        if not valid:
            return [0.5 for _ in rows]
        newest = max(valid)
        oldest = min(valid)
        span = max(1.0, (newest - oldest).total_seconds())

        scores: list[float] = []
        for ts in timestamps:
            if ts is None:
                scores.append(0.3)
                continue
            age = (newest - ts).total_seconds()
            scores.append(max(0.0, min(1.0, 1.0 - age / span)))
        return scores

    def _parse_time(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _build_match_reasons(
        self,
        query: str,
        row: dict[str, Any],
        lexical_score: float,
        semantic_score: float,
    ) -> list[str]:
        reasons: list[str] = []
        q_lower = query.lower()
        title = str(row.get("title", ""))
        goal = str(row.get("goal", ""))
        messages = str(row.get("messages", ""))

        if q_lower in title.lower():
            reasons.append("title-match")
        if q_lower in goal.lower():
            reasons.append("goal-match")
        if q_lower in messages.lower():
            reasons.append("messages-match")
        if semantic_score >= 0.82:
            reasons.append("semantic-strong")
        elif semantic_score >= 0.65:
            reasons.append("semantic-related")
        if not reasons and lexical_score > 0:
            reasons.append("lexical-related")
        if not reasons:
            reasons.append("recency-fallback")
        return reasons

    # ==================================================================
    # 层3：语义记忆（Semantic Memory - ChromaDB）
    # ==================================================================

    def remember(
        self,
        content: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        存储语义记忆

        Args:
            content: 要记住的内容
            memory_type: 记忆类型（semantic/episodic/procedural）
            importance: 重要性 0.0-1.0（影响检索排序）
            metadata: 额外元数据

        Returns:
            str: 记忆条目ID

        面试要点：
        "存储时我立刻调用Embedding API向量化内容，
         向量和文本一起存入ChromaDB。
         查询时同样向量化查询词，用余弦相似度找最相关记忆。
         这是RAG（检索增强生成）的核心机制。"
        """
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata or {},
        )

        # 向量化内容（调用Dashscope text-embedding-v3）
        embeddings = self._llm.embed([content])
        embedding = embeddings[0]

        return self._semantic_store.add(entry, embedding)

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        检索语义相关的记忆

        Args:
            query: 查询文本
            top_k: 返回数量
            memory_type: 过滤记忆类型
            min_importance: 最低重要性阈值

        Returns:
            list[(MemoryEntry, similarity_score)]，按相似度降序

        面试要点：
        "recall()的实现是：
         1. 用同样的Embedding模型向量化查询词
         2. 在ChromaDB中做余弦相似度搜索
         3. 返回最相似的top_k条记忆
         这保证了查询和存储用同一个向量空间，结果才有意义。"
        """
        if self._semantic_store.count() == 0:
            return []

        # 向量化查询词
        query_embeddings = self._llm.embed([query])
        query_embedding = query_embeddings[0]

        results = self._semantic_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            memory_type=memory_type,
            min_importance=min_importance,
        )

        # 更新访问计数
        for entry, _ in results:
            self._semantic_store.update_access_count(entry.id)

        return results

    def format_memories_for_context(
        self,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        检索相关记忆并格式化为可注入上下文的文本

        面试要点：
        "这个方法在每次LLM调用前执行，
         把相关记忆以'[记忆]...'的格式拼接到system prompt里。
         这是RAG的'增强'那一步：
         用检索到的知识增强LLM的推理能力。"
        """
        memories = self.recall(query, top_k=top_k)
        if not memories:
            return ""

        lines = ["📚 相关历史记忆："]
        for entry, score in memories:
            lines.append(f"  [相似度:{score:.2f}] {entry.content}")

        return "\n".join(lines)

    def format_sessions_for_context(self, query: str, top_k: int = 3) -> str:
        """
        Cross-session recall: retrieve similar sessions and format as context.

        This is episodic "past tasks" recall, distinct from semantic memory recall().
        """
        hits = self.search_sessions(query, limit=max(1, top_k))
        if not hits:
            return ""

        include_excerpt = os.getenv("AGENT_RECALL_INJECT_INCLUDE_EXCERPT", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        try:
            excerpt_max = int(os.getenv("AGENT_RECALL_INJECT_EXCERPT_MAX_CHARS", "140"))
        except Exception:
            excerpt_max = 140
        excerpt_max = max(0, min(400, excerpt_max))

        lines = ["🧭 相关历史会话（past sessions, condensed）："]
        for idx, item in enumerate(hits[:top_k], start=1):
            sid = str(item.get("id", ""))
            title = str(item.get("title", "")).strip()
            goal = str(item.get("goal", "")).strip()
            reason = str(item.get("match_reason", "")).strip()
            score = float(item.get("score", 0.0) or 0.0)
            excerpt = self._get_session_excerpt(sid, max_chars=excerpt_max) if include_excerpt and excerpt_max > 0 else ""

            # Keep each item compact and action-oriented.
            header = f"{idx}) [{sid}] {title}".strip()
            meta = f"score={score:.2f} reason={reason or '-'}"
            lines.append(f"- {header} ({meta})")
            if goal:
                lines.append(f"  goal: {self._clean_inline(goal, max_chars=180)}")
            if excerpt:
                lines.append(f"  excerpt: {self._clean_inline(excerpt, max_chars=excerpt_max)}")
        return "\n".join(lines)

    def _get_session_excerpt(self, session_id: str, max_chars: int = 220) -> str:
        session_id = (session_id or "").strip()
        if not session_id:
            return ""
        try:
            conn = sqlite3.connect(str(self._db_path))
            row = conn.execute(
                "SELECT messages FROM sessions WHERE id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            conn.close()
            if not row:
                return ""
            messages = json.loads(row[0] or "[]")
            if not isinstance(messages, list):
                return ""
            parts: list[str] = []
            for m in messages[:8]:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role", "") or "")
                content = str(m.get("content", "") or "").strip()
                if not content:
                    continue
                if role and role != "system":
                    parts.append(f"{role}: {content}")
            text = " | ".join(parts)
            text = " ".join(text.split())
            if len(text) > max_chars:
                return text[: max(0, max_chars - 3)] + "..."
            return text
        except Exception:
            return ""

    def _clean_inline(self, text: str, max_chars: int = 200) -> str:
        t = " ".join(str(text or "").split())
        if max_chars <= 0:
            return ""
        if len(t) > max_chars:
            return t[: max(0, max_chars - 3)] + "..."
        return t

    # ==================================================================
    # 层4：程序性记忆（Procedural Memory - Skills）
    # ==================================================================

    def save_skill(
        self,
        skill_name: str,
        description: str,
        steps: list[str],
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        保存技能（程序性记忆）

        面试要点：
        "Skills是Hermes最有特色的设计之一。
         Agent在完成一个任务后，会把'怎么做'提炼成可复用的技能存下来。
         下次遇到类似任务，先查Skills，能直接复用就不需要重新规划。
         这是程序性记忆——'肌肉记忆'，不需要每次重新思考。"
        """
        skill_file = self._skills_dir / f"{skill_name.replace(' ', '_')}.json"
        skill_data = {
            "name": skill_name,
            "description": description,
            "steps": steps,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "use_count": 0,
            "metadata": metadata or {},
        }
        skill_file.write_text(
            json.dumps(skill_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"技能已保存: {skill_name}")

    def load_skill(self, skill_name: str) -> Optional[dict[str, Any]]:
        """加载技能"""
        skill_file = self._skills_dir / f"{skill_name.replace(' ', '_')}.json"
        if not skill_file.exists():
            return None
        return json.loads(skill_file.read_text(encoding="utf-8"))

    def list_skills(self) -> list[dict[str, Any]]:
        """列出所有技能"""
        skills = []
        for f in self._skills_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                skills.append(data)
            except Exception:
                pass
        return sorted(skills, key=lambda x: x.get("use_count", 0), reverse=True)

    def delete_skill(self, skill_name: str) -> bool:
        """删除技能文件（用于回滚自动采纳）"""
        skill_file = self._skills_dir / f"{skill_name.replace(' ', '_')}.json"
        if not skill_file.exists():
            return False
        try:
            skill_file.unlink()
            return True
        except Exception:
            return False

    def update_skill(self, skill_name: str, updates: dict[str, Any]) -> bool:
        """更新技能字段（用于治理策略与元数据维护）"""
        skill_file = self._skills_dir / f"{skill_name.replace(' ', '_')}.json"
        if not skill_file.exists():
            return False
        try:
            data = json.loads(skill_file.read_text(encoding="utf-8"))
            for key, value in updates.items():
                data[key] = value
            skill_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return True
        except Exception:
            return False

    # ==================================================================
    # 统计信息
    # ==================================================================

    def get_stats(self) -> dict[str, Any]:
        """获取记忆系统统计信息"""
        conn = sqlite3.connect(str(self._db_path))
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        conn.close()

        semantic_store = getattr(self, "_semantic_store", None)
        skills_dir = getattr(self, "_skills_dir", None)

        return {
            "working_memory_messages": len(self._working_memory),
            "episodic_sessions": session_count,
            "semantic_memories": semantic_store.count() if semantic_store else 0,
            "skills": len(list(skills_dir.glob("*.json"))) if skills_dir else 0,
            "chroma_stats": semantic_store.get_stats() if semantic_store else {},
        }

    def close(self) -> None:
        """
        关闭所有底层连接（Windows文件句柄释放）

        面试要点：
        "这是资源清理的责任链设计：
         AgentLoop.close()
           → MemoryManager.close()
             → ChromaMemoryStore.close()
               → gc.collect()（释放hnswlib文件句柄）
         每一层只负责关闭自己创建的资源，
         不向上也不跨层直接操作。"
        """
        try:
            semantic_store = getattr(self, "_semantic_store", None)
            if semantic_store:
                semantic_store.close()
            logger.info("MemoryManager：所有连接已关闭")
        except Exception as e:
            logger.warning(f"MemoryManager关闭时遇到警告: {e}")

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()