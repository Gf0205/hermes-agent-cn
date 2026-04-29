"""
Phase 5 P1 验证脚本（Operator UX 治理状态接口）
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.skill_distiller import SkillDistiller


class FakeMemoryManager:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self.skills: dict[str, dict] = {}
        self.updated: list[str] = []
        self.deleted: list[str] = []

    def save_skill(
        self,
        skill_name: str,
        description: str,
        steps: list[str],
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.skills[skill_name] = {
            "name": skill_name,
            "description": description,
            "steps": list(steps),
            "tags": list(tags or []),
            "created_at": datetime.now().isoformat(),
            "use_count": 0,
            "metadata": dict(metadata or {}),
        }

    def load_skill(self, skill_name: str) -> dict | None:
        return self.skills.get(skill_name)

    def list_skills(self) -> list[dict]:
        return list(self.skills.values())

    def update_skill(self, skill_name: str, updates: dict) -> bool:
        if skill_name not in self.skills:
            return False
        self.skills[skill_name].update(updates)
        self.updated.append(skill_name)
        return True

    def delete_skill(self, skill_name: str) -> bool:
        if skill_name not in self.skills:
            return False
        self.skills.pop(skill_name, None)
        self.deleted.append(skill_name)
        return True


def test_governance_status_fields_and_counts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        mm = FakeMemoryManager(base)
        mm.skills["auto_a"] = {
            "name": "auto_a",
            "description": "A",
            "steps": ["s1"],
            "tags": [],
            "created_at": (datetime.now() - timedelta(days=5)).isoformat(),
            "use_count": 0,
            "metadata": {"quality_score": 0.2, "decay_score": 3},
        }
        mm.skills["manual_b"] = {
            "name": "manual_b",
            "description": "B",
            "steps": ["s2"],
            "tags": [],
            "created_at": datetime.now().isoformat(),
            "use_count": 3,
            "metadata": {"quality_score": 0.9},
        }

        d = SkillDistiller(
            mm,
            auto_subscribe=False,
            draft_store_path=base / "drafts.json",
            adoption_log_path=base / "adoption.json",
        )
        status = d.get_governance_status()
        assert status["auto_skills"] == 1
        assert status["low_quality_auto_skills"] == 1
        assert "dedupe_similarity_threshold" in status
        assert "decay_quality_threshold" in status


def test_governance_run_returns_operational_metrics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        mm = FakeMemoryManager(base)
        mm.skills["auto_decay"] = {
            "name": "auto_decay",
            "description": "A",
            "steps": ["s1"],
            "tags": [],
            "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
            "use_count": 0,
            "metadata": {"quality_score": 0.2, "decay_score": 2},
        }

        d = SkillDistiller(
            mm,
            auto_subscribe=False,
            decay_min_age_days=0,
            decay_quality_threshold=0.65,
            decay_step=1,
            decay_remove_below=0,
            draft_store_path=base / "drafts.json",
            adoption_log_path=base / "adoption.json",
        )
        result = d.run_skill_governance()
        assert result["scanned_auto_skills"] == 1
        assert result["decayed_updated"] == 1
        assert result["decayed_removed"] == 0


def main() -> None:
    test_governance_status_fields_and_counts()
    print("[PASS] test_governance_status_fields_and_counts")
    test_governance_run_returns_operational_metrics()
    print("[PASS] test_governance_run_returns_operational_metrics")
    print("[DONE] Phase 5 P1 operator governance checks passed")


if __name__ == "__main__":
    main()
