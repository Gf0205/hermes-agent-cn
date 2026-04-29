"""
src/memory/chroma_store.py - ChromaDB 向量存储

面试要点：
"我选择ChromaDB而不是FAISS的核心原因：
 1. 全功能数据库：自带持久化、元数据过滤、CRUD
 2. Windows 1.0+ 兼容性稳定，pip install 直接搞定
 3. 元数据过滤是关键：where子句一行实现复杂查询

 踩坑记录（Windows特有）：
 ChromaDB的底层是SQLite + hnswlib，在Windows上文件句柄
 不会随Python对象销毁立刻释放（GC时机不确定）。
 解决方案：
 1. 实现 close() 方法显式释放客户端引用
 2. 实现上下文管理器协议（__enter__/__exit__）
 3. 测试代码用 ignore_cleanup_errors=True 兜底

 这个问题在Linux上不存在，是Windows文件系统的特性：
 不允许删除正在被打开的文件（Linux允许，因为用inode引用计数）。"
"""

from __future__ import annotations

import gc
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb import Collection

from src.models import MemoryEntry

logger = logging.getLogger(__name__)


class ChromaMemoryStore:
    """
    ChromaDB 向量记忆存储

    职责：
    - 存储文本 + 向量 + 元数据
    - 语义相似度搜索（最近邻）
    - 元数据过滤查询（按类型、时间、重要性）
    - CRUD操作

    生命周期管理（Windows关键！）：
    - 必须调用 close() 或使用 with 语句，确保文件句柄释放
    - 否则在Windows上删除目录时会遇到 [WinError 32]

    用法：
        # 方式1：with语句（推荐，测试场景）
        with ChromaMemoryStore(persist_dir=tmp_dir) as store:
            store.add(entry, embedding)

        # 方式2：手动管理（生产场景，长期运行）
        store = ChromaMemoryStore()
        store.add(entry, embedding)
        # 程序退出前：
        store.close()
    """

    # ChromaDB元数据只支持 str/int/float/bool
    _DATETIME_FMT = "%Y-%m-%dT%H:%M:%S"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        chroma_dir = persist_dir or os.getenv(
            "AGENT_CHROMA_DIR",
            str(Path.home() / ".hermes-cn" / "chroma_db")
        )
        self._persist_dir = str(Path(chroma_dir).expanduser().resolve())
        self._collection_name = collection_name or os.getenv(
            "AGENT_CHROMA_COLLECTION", "hermes_memory"
        )
        self._closed = False  # 防止重复关闭
        self._degraded_in_memory = False
        self._fallback_vectors: dict[str, list[float]] = {}
        self._fallback_entries: dict[str, MemoryEntry] = {}

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            self._collection: Collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"ChromaDB初始化 | 目录: {self._persist_dir} "
                f"| 集合: {self._collection_name} "
                f"| 已有记忆: {self._collection.count()} 条"
            )
        except Exception as e:
            # Corporate/App Control policies may block chromadb rust bindings on Windows.
            # Degrade to in-memory semantic store so the agent can still run.
            self._degraded_in_memory = True
            self._client = None  # type: ignore[assignment]
            self._collection = None  # type: ignore[assignment]
            logger.warning(
                "ChromaDB初始化失败，降级为内存语义存储（当前进程有效，不持久化）: %s",
                e,
            )

    # ==================================================================
    # 上下文管理器协议（支持 with 语句）
    # ==================================================================

    def __enter__(self) -> "ChromaMemoryStore":
        """进入 with 块，返回自身"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        退出 with 块时自动关闭

        面试要点：
        "上下文管理器协议（__enter__/__exit__）是Python资源管理的
         最佳实践。类似Java的try-with-resources。
         无论with块内是否发生异常，__exit__都会被调用，
         保证资源一定被释放——这在Windows文件句柄管理中至关重要。"
        """
        self.close()

    def close(self) -> None:
        """
        显式释放ChromaDB连接和文件句柄

        Windows关键步骤：
        1. 置空 _collection 引用
        2. 置空 _client 引用
        3. 显式触发GC（让hnswlib的C++析构函数立即执行）

        面试要点：
        "Python的GC在CPython中基于引用计数，
         但循环引用或C扩展对象（如hnswlib）的析构时机不确定。
         显式调用 gc.collect() 强制立即回收，
         确保Windows文件句柄在close()返回前已释放。
         这是处理C扩展资源泄漏的标准手段。"
        """
        if self._closed:
            return

        self._closed = True

        # 步骤1：释放集合引用
        self._collection = None  # type: ignore[assignment]

        # 步骤2：释放客户端引用（触发chromadb内部的cleanup）
        self._client = None  # type: ignore[assignment]

        # 步骤3：强制GC，让C扩展（hnswlib）的析构函数立即执行
        # 这是Windows上释放文件句柄的关键步骤
        gc.collect()

        logger.debug(f"ChromaDB连接已关闭: {self._persist_dir}")

    def _check_not_closed(self) -> None:
        """操作前检查连接是否已关闭"""
        if self._closed or self._client is None:
            raise RuntimeError(
                "ChromaMemoryStore已关闭，无法继续操作。\n"
                "请重新创建实例或不要在close()后使用。"
            )

    # ==================================================================
    # 写操作
    # ==================================================================

    def add(self, entry: MemoryEntry, embedding: list[float]) -> str:
        """
        添加记忆条目

        Args:
            entry: 记忆条目
            embedding: 已生成的向量

        Returns:
            str: 条目ID
        """
        self._check_not_closed()
        if self._degraded_in_memory:
            self._fallback_entries[entry.id] = entry
            self._fallback_vectors[entry.id] = list(embedding or [])
            return entry.id

        metadata = {
            "memory_type":  entry.memory_type,
            "importance":   entry.importance,
            "access_count": entry.access_count,
            "created_at":   entry.created_at.strftime(self._DATETIME_FMT),
            # 只保留ChromaDB支持的基础类型
            **{
                k: v for k, v in entry.metadata.items()
                if isinstance(v, (str, int, float, bool))
            },
        }

        self._collection.add(
            ids=[entry.id],
            documents=[entry.content],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        logger.debug(f"记忆已存储: {entry.id[:8]}... | 类型: {entry.memory_type}")
        return entry.id

    def update_access_count(self, entry_id: str) -> None:
        """更新访问计数"""
        self._check_not_closed()
        if self._degraded_in_memory:
            entry = self._fallback_entries.get(entry_id)
            if entry:
                entry.access_count += 1
            return
        try:
            result = self._collection.get(ids=[entry_id], include=["metadatas"])
            if result["metadatas"]:
                meta = result["metadatas"][0]
                meta["access_count"] = meta.get("access_count", 0) + 1
                self._collection.update(ids=[entry_id], metadatas=[meta])
        except Exception as e:
            logger.warning(f"更新访问计数失败: {e}")

    def delete(self, entry_id: str) -> None:
        """删除记忆条目"""
        self._check_not_closed()
        if self._degraded_in_memory:
            self._fallback_entries.pop(entry_id, None)
            self._fallback_vectors.pop(entry_id, None)
            return
        self._collection.delete(ids=[entry_id])

    # ==================================================================
    # 读操作
    # ==================================================================

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        语义相似度搜索

        面试要点：
        "余弦距离（cosine distance）范围是0~2：
         0 = 完全相同方向，2 = 完全相反方向。
         转换公式：similarity = 1 - distance/2
         这样相似度就在0~1范围内，1表示最相似。"
        """
        self._check_not_closed()
        if self._degraded_in_memory:
            return self._search_fallback(
                query_embedding=query_embedding,
                n_results=n_results,
                memory_type=memory_type,
                min_importance=min_importance,
            )

        total = self._collection.count()
        if total == 0:
            return []

        # 构建过滤条件
        where: Optional[dict] = None
        conditions = []
        if memory_type:
            conditions.append({"memory_type": {"$eq": memory_type}})
        if min_importance > 0.0:
            conditions.append({"importance": {"$gte": min_importance}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        actual_n = min(n_results, total)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_n,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            results = self._collection.query(**query_kwargs)
        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        entries = []
        for i, entry_id in enumerate(results["ids"][0]):
            doc      = results["documents"][0][i]
            meta     = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # 余弦距离 → 相似度
            similarity = max(0.0, 1.0 - distance / 2.0)

            entry = MemoryEntry(
                id=entry_id,
                content=doc,
                memory_type=meta.get("memory_type", "semantic"),
                importance=float(meta.get("importance", 0.5)),
                access_count=int(meta.get("access_count", 0)),
                created_at=datetime.strptime(
                    meta.get("created_at", "2024-01-01T00:00:00"),
                    self._DATETIME_FMT
                ),
                metadata={
                    k: v for k, v in meta.items()
                    if k not in ("memory_type", "importance", "access_count", "created_at")
                },
            )
            entries.append((entry, similarity))

        return entries

    def get_by_type(
        self,
        memory_type: str,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """按类型获取记忆（纯元数据过滤，不需要向量）"""
        self._check_not_closed()
        if self._degraded_in_memory:
            out: list[MemoryEntry] = []
            for entry in self._fallback_entries.values():
                if entry.memory_type == memory_type:
                    out.append(entry)
            return out[: max(0, limit)]

        results = self._collection.get(
            where={"memory_type": {"$eq": memory_type}},
            limit=limit,
            include=["documents", "metadatas"],
        )

        entries = []
        for i, entry_id in enumerate(results["ids"]):
            doc  = results["documents"][i]
            meta = results["metadatas"][i]
            entries.append(MemoryEntry(
                id=entry_id,
                content=doc,
                memory_type=meta.get("memory_type", memory_type),
                importance=float(meta.get("importance", 0.5)),
            ))
        return entries

    # ==================================================================
    # 统计
    # ==================================================================

    def count(self, memory_type: Optional[str] = None) -> int:
        """统计记忆数量"""
        if self._closed or self._client is None:
            if self._degraded_in_memory and not self._closed:
                if memory_type:
                    return sum(1 for e in self._fallback_entries.values() if e.memory_type == memory_type)
                return len(self._fallback_entries)
            return 0  # 已关闭时返回0，不抛异常（方便在__del__中调用）

        if memory_type:
            results = self._collection.get(
                where={"memory_type": {"$eq": memory_type}},
                include=[],
            )
            return len(results["ids"])
        return self._collection.count()

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计信息"""
        return {
            "total":       self.count(),
            "persist_dir": self._persist_dir,
            "collection":  self._collection_name,
            "closed":      self._closed,
            "degraded_in_memory": self._degraded_in_memory,
        }

    def _search_fallback(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        items: list[tuple[MemoryEntry, float]] = []
        q = list(query_embedding or [])

        for entry_id, entry in self._fallback_entries.items():
            if memory_type and entry.memory_type != memory_type:
                continue
            if min_importance > 0.0 and float(entry.importance) < min_importance:
                continue
            vec = self._fallback_vectors.get(entry_id, [])
            sim = self._cosine_similarity(q, vec)
            items.append((entry, sim))

        items.sort(key=lambda x: x[1], reverse=True)
        return items[: max(0, n_results)]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))

    def __del__(self) -> None:
        """析构时自动关闭（最后一道防线）"""
        try:
            self.close()
        except Exception:
            pass  # 析构函数不能抛异常