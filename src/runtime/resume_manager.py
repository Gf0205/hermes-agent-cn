"""
src/runtime/resume_manager.py - 断点恢复协调器
"""

from __future__ import annotations

from src.runtime.checkpoint_store import CheckpointRecord, CheckpointStore


class ResumeManager:
    """封装断点恢复前的状态校验和参数准备"""

    RESUMABLE_STATUSES = {"running", "failed", "interrupted"}

    def __init__(self, checkpoint_store: CheckpointStore) -> None:
        self._store = checkpoint_store

    def get_resume_record(self, session_id: str) -> CheckpointRecord:
        record = self._store.load(session_id)
        if record is None:
            raise ValueError(f"未找到 session_id={session_id} 的断点")
        if not self.is_resumable_status(record.status):
            raise ValueError(
                f"该会话状态为 {record.status}，不需要恢复。可恢复状态: {sorted(self.RESUMABLE_STATUSES)}"
            )
        return record

    @classmethod
    def is_resumable_status(cls, status: str) -> bool:
        return status in cls.RESUMABLE_STATUSES
