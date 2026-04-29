"""
Phase 3 P0 验证脚本
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.context_compressor_v2 import ContextCompressorV2
from src.execution.parallel_executor import ParallelExecutor
from src.execution.scheduler_v2 import ExecutionBatchPlanner
from src.runtime.checkpoint_store import CheckpointRecord, CheckpointStore
from src.runtime.resume_manager import ResumeManager


def test_scheduler_conflict_detection() -> None:
    planner = ExecutionBatchPlanner()
    calls = [
        (SimpleNamespace(id="1"), "read_file", {"path": "a.txt"}),
        (SimpleNamespace(id="2"), "read_file", {"path": "a.txt"}),
        (SimpleNamespace(id="3"), "edit_file", {"path": "a.txt"}),
        (SimpleNamespace(id="4"), "read_file", {"path": "b.txt"}),
    ]
    batches = planner.plan_batches(calls)
    assert len(batches) >= 2
    assert len(batches[0]) == 2  # 前两个同路径读可并发
    assert batches[1][0][1][1] == "edit_file"


def test_parallel_executor_order_stable() -> None:
    calls = [
        (
            SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name="read_file", arguments='{"path":"a.txt"}')),
            "read_file",
            {"path": "a.txt"},
        )
        for i in range(3)
    ]
    executor = ParallelExecutor(max_workers=4)

    def execute_fn(name: str, args: dict) -> SimpleNamespace:
        return SimpleNamespace(status=SimpleNamespace(value="success"), output=args["path"])

    result = executor.execute_parallel_tools(calls, execute_fn)
    assert [item[0].id for item in result] == ["c0", "c1", "c2"]


def test_checkpoint_store_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(checkpoints_dir=Path(tmp))
        record = CheckpointRecord(
            session_id="s001",
            goal="demo goal",
            context="ctx",
            status="running",
            updated_at=datetime.now().isoformat(),
            completed_subgoals=["g1"],
            total_subgoals=3,
            total_iterations=3,
            total_tool_calls=4,
            total_tokens=123,
            final_answer="",
            metadata={
                "x": 1,
                "last_failed_subgoal": "g2:todo",
                "plan_snapshot": [
                    {"id": "g1", "description": "done", "status": "completed"},
                    {"id": "g2", "description": "todo", "status": "pending"},
                ],
            },
        )
        store.save(record)
        loaded = store.load("s001")
        assert loaded is not None
        assert loaded.goal == "demo goal"
        assert loaded.completed_subgoals == ["g1"]
        assert loaded.total_subgoals == 3
        assert loaded.metadata.get("last_failed_subgoal") == "g2:todo"
        assert len(loaded.metadata.get("plan_snapshot", [])) == 2


def test_resume_manager_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = CheckpointStore(checkpoints_dir=Path(tmp))
        done = CheckpointRecord(
            session_id="s_done",
            goal="done",
            context="",
            status="completed",
            updated_at=datetime.now().isoformat(),
        )
        runn = CheckpointRecord(
            session_id="s_running",
            goal="running",
            context="ctx",
            status="running",
            updated_at=datetime.now().isoformat(),
        )
        store.save(done)
        store.save(runn)
        manager = ResumeManager(store)
        assert ResumeManager.is_resumable_status("running") is True
        assert ResumeManager.is_resumable_status("completed") is False
        ok = manager.get_resume_record("s_running")
        assert ok.goal == "running"
        try:
            manager.get_resume_record("s_done")
            assert False, "completed 状态不应可恢复"
        except ValueError:
            pass


def test_context_compressor_v2_pinned_facts() -> None:
    class DummyLLM:
        def chat(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="- done\n- todo"))]
            )

    compressor = ContextCompressorV2(DummyLLM())  # type: ignore[arg-type]
    messages = [{"role": "system", "content": "system rules"}]
    messages.append({"role": "user", "content": "完成 src/main.py 中的恢复功能"})
    messages.extend([{"role": "assistant", "content": f"step-{i}"} for i in range(120)])
    compressed = compressor.compress(messages, max_tokens=300)
    merged = "\n".join(str(m.get("content", "")) for m in compressed)
    assert "PinnedFacts" in merged
    assert "system rules" in merged
    assert "恢复功能" in merged


def main() -> None:
    test_scheduler_conflict_detection()
    print("[PASS] test_scheduler_conflict_detection")
    test_parallel_executor_order_stable()
    print("[PASS] test_parallel_executor_order_stable")
    test_checkpoint_store_roundtrip()
    print("[PASS] test_checkpoint_store_roundtrip")
    test_resume_manager_validation()
    print("[PASS] test_resume_manager_validation")
    test_context_compressor_v2_pinned_facts()
    print("[PASS] test_context_compressor_v2_pinned_facts")
    print("[DONE] Phase 3 P0 验证通过")


if __name__ == "__main__":
    main()
