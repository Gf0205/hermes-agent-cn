"""
Phase 4 P0 验证脚本（跨会话检索）
"""

from __future__ import annotations

import sys
import tempfile
import gc
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_manager import MemoryManager
from src.models import Session


class FakeLLM:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # 本脚本不依赖语义检索，只需要满足接口
        return [[0.0, 0.0, 0.0] for _ in texts]


def _tmpdir_context():
    """兼容不同Python版本的临时目录上下文。"""
    try:
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    except TypeError:
        return tempfile.TemporaryDirectory()


def _close_memory(mm: MemoryManager) -> None:
    mm.close()
    del mm
    gc.collect()


def test_search_sessions_match_title_and_goal() -> None:
    with _tmpdir_context() as tmp:
        mm = MemoryManager(llm_client=FakeLLM(), data_dir=tmp)  # type: ignore[arg-type]
        s1 = Session(
            id="s001",
            title="修复登录接口",
            messages=[{"role": "user", "content": "修复 auth bug"}],
            created_at=datetime.now(),
            metadata={"goal": "修复登录接口并加测试"},
        )
        s2 = Session(
            id="s002",
            title="重构缓存层",
            messages=[{"role": "user", "content": "cache refactor"}],
            created_at=datetime.now(),
            metadata={"goal": "重构缓存层并验证性能"},
        )
        mm.save_session(s1)
        mm.save_session(s2)

        hits = mm.search_sessions("登录", limit=5)
        assert len(hits) >= 1
        assert hits[0]["id"] == "s001"
        _close_memory(mm)


def test_search_sessions_like_fallback() -> None:
    with _tmpdir_context() as tmp:
        mm = MemoryManager(llm_client=FakeLLM(), data_dir=tmp)  # type: ignore[arg-type]
        # 强制走LIKE回退路径
        mm._fts_enabled = False  # type: ignore[attr-defined]
        s1 = Session(
            id="s003",
            title="数据库巡检",
            messages=[{"role": "assistant", "content": "执行 nightly db audit"}],
            created_at=datetime.now(),
            metadata={"goal": "生成数据库巡检报告"},
        )
        mm.save_session(s1)
        hits = mm.search_sessions("巡检", limit=5)
        assert len(hits) == 1
        assert hits[0]["id"] == "s003"
        _close_memory(mm)


def main() -> None:
    test_search_sessions_match_title_and_goal()
    print("[PASS] test_search_sessions_match_title_and_goal")
    test_search_sessions_like_fallback()
    print("[PASS] test_search_sessions_like_fallback")
    print("[DONE] Phase 4 P0 cross-session recall checks passed")


if __name__ == "__main__":
    main()
