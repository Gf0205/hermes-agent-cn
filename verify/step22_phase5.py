"""
Phase 5 P2 验证脚本（跨会话检索融合排序）
"""

from __future__ import annotations

import gc
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_manager import MemoryManager
from src.models import Session


class FakeLLM:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            t = text.lower()
            if "连接池" in t:
                vectors.append([1.0, 0.0, 0.0])
            elif "db pool" in t or "connection pool" in t:
                vectors.append([1.0, 0.0, 0.0])
            elif "缓存" in t or "cache" in t:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def _tmpdir_context():
    try:
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    except TypeError:
        return tempfile.TemporaryDirectory()


def _close_memory(mm: MemoryManager) -> None:
    mm.close()
    del mm
    gc.collect()


def test_hybrid_rank_prefers_title_goal_lexical_hit() -> None:
    with _tmpdir_context() as tmp:
        mm = MemoryManager(llm_client=FakeLLM(), data_dir=tmp)  # type: ignore[arg-type]
        mm.save_session(Session(
            id="s101",
            title="修复登录接口",
            messages=[{"role": "user", "content": "auth bug fix"}],
            created_at=datetime.now() - timedelta(hours=3),
            metadata={"goal": "修复登录接口并加测试"},
        ))
        mm.save_session(Session(
            id="s102",
            title="重构缓存系统",
            messages=[{"role": "user", "content": "cache rewrite"}],
            created_at=datetime.now() - timedelta(hours=1),
            metadata={"goal": "重构缓存并做压测"},
        ))

        hits = mm.search_sessions("登录", limit=5)
        assert len(hits) >= 1
        assert hits[0]["id"] == "s101"
        assert "match_reason" in hits[0]
        assert "score" in hits[0]
        _close_memory(mm)


def test_semantic_fallback_without_lexical_match() -> None:
    with _tmpdir_context() as tmp:
        mm = MemoryManager(llm_client=FakeLLM(), data_dir=tmp)  # type: ignore[arg-type]
        mm.save_session(Session(
            id="s201",
            title="系统性能优化",
            messages=[{"role": "assistant", "content": "db pool tuning and throughput benchmark"}],
            created_at=datetime.now() - timedelta(days=2),
            metadata={"goal": "提升吞吐并稳定连接"},
        ))
        mm.save_session(Session(
            id="s202",
            title="前端样式整理",
            messages=[{"role": "assistant", "content": "css theme cleanup"}],
            created_at=datetime.now(),
            metadata={"goal": "修复颜色主题"},
        ))

        # “连接池”不直接出现在 title/goal/messages（中文子串），应走 recent + semantic 排序
        hits = mm.search_sessions("连接池", limit=5)
        assert len(hits) >= 1
        assert hits[0]["id"] == "s201"
        assert "semantic" in str(hits[0].get("match_reason", ""))
        _close_memory(mm)


def main() -> None:
    test_hybrid_rank_prefers_title_goal_lexical_hit()
    print("[PASS] test_hybrid_rank_prefers_title_goal_lexical_hit")
    test_semantic_fallback_without_lexical_match()
    print("[PASS] test_semantic_fallback_without_lexical_match")
    print("[DONE] Phase 5 P2 hybrid recall ranking checks passed")


if __name__ == "__main__":
    main()
