"""
Phase 5 P1 验证脚本（技能治理 v2：语义去重 + 低价值衰减）
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.skill_distiller import SkillDistiller, SkillDraft


class FakeMemoryManager:
    def __init__(self) -> None:
        self._data_dir = Path(".")
        self.skills: dict[str, dict] = {}
        self.saved_names: list[str] = []
        self.deleted_names: list[str] = []
        self.updated_names: list[str] = []

    def save_skill(
        self,
        skill_name: str,
        description: str,
        steps: list[str],
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.saved_names.append(skill_name)
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

    def delete_skill(self, skill_name: str) -> bool:
        if skill_name not in self.skills:
            return False
        self.deleted_names.append(skill_name)
        self.skills.pop(skill_name, None)
        return True

    def update_skill(self, skill_name: str, updates: dict) -> bool:
        if skill_name not in self.skills:
            return False
        self.updated_names.append(skill_name)
        self.skills[skill_name].update(updates)
        return True


def test_semantic_deduplicate_on_adopt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        manager = FakeMemoryManager()
        manager._data_dir = base
        manager.skills["auto_db_pool_opt_0428"] = {
            "name": "auto_db_pool_opt_0428",
            "description": "数据库连接池优化流程",
            "steps": ["按顺序执行工具：read_file -> edit_file -> shell"],
            "tags": ["auto-generated", "file-ops"],
            "created_at": datetime.now().isoformat(),
            "use_count": 0,
            "metadata": {
                "quality_score": 0.82,
                "semantic_signature": "数据库 连接 池 优化 read_file edit_file shell",
                "decay_score": 3,
                "source": "auto-distilled",
            },
        }

        distiller = SkillDistiller(
            manager,
            auto_subscribe=False,
            dedupe_similarity_threshold=0.30,
            draft_store_path=base / "drafts.json",
            adoption_log_path=base / "adoptions.json",
        )
        draft = SkillDraft(
            draft_id="d1",
            name="auto_db_pool_opt_candidate",
            description="自动提炼：数据库连接池优化",
            steps=["按顺序执行工具：read_file -> edit_file -> shell"],
            tags=["auto-generated"],
            source_goal="优化数据库连接池并验证",
            quality_score=0.80,
        )
        distiller._append_draft(draft)
        adopted = distiller.adopt_draft(0, source="manual")

        assert adopted == "auto_db_pool_opt_0428"
        assert manager.saved_names == []
        records = distiller.get_recent_adoption_records(limit=5)
        assert records[-1].deduplicated is True
        assert records[-1].deduplicated_to == "auto_db_pool_opt_0428"


def test_low_value_auto_skill_decay_and_remove() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        manager = FakeMemoryManager()
        manager._data_dir = base
        manager.skills["auto_low_value"] = {
            "name": "auto_low_value",
            "description": "低价值自动技能",
            "steps": ["按顺序执行工具：shell"],
            "tags": ["auto-generated"],
            "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
            "use_count": 0,
            "metadata": {
                "source": "auto-distilled",
                "quality_score": 0.25,
                "decay_score": 1,
            },
        }
        manager.skills["auto_high_value"] = {
            "name": "auto_high_value",
            "description": "高价值自动技能",
            "steps": ["按顺序执行工具：read_file -> edit_file -> shell"],
            "tags": ["auto-generated"],
            "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
            "use_count": 2,
            "metadata": {
                "source": "auto-distilled",
                "quality_score": 0.88,
                "decay_score": 3,
            },
        }

        distiller = SkillDistiller(
            manager,
            auto_subscribe=False,
            decay_quality_threshold=0.65,
            decay_min_age_days=0,
            decay_step=1,
            decay_remove_below=0,
            draft_store_path=base / "drafts.json",
            adoption_log_path=base / "adoptions.json",
        )
        result = distiller.run_skill_governance()

        assert result["decayed_removed"] == 1
        assert "auto_low_value" not in manager.skills
        assert "auto_low_value" in manager.deleted_names
        assert "auto_high_value" in manager.skills


def test_low_value_auto_skill_decay_decrement_without_remove() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        manager = FakeMemoryManager()
        manager._data_dir = base
        manager.skills["auto_decay_only"] = {
            "name": "auto_decay_only",
            "description": "待衰减自动技能",
            "steps": ["按顺序执行工具：shell"],
            "tags": ["auto-generated"],
            "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
            "use_count": 0,
            "metadata": {
                "source": "auto-distilled",
                "quality_score": 0.20,
                "decay_score": 3,
            },
        }

        distiller = SkillDistiller(
            manager,
            auto_subscribe=False,
            decay_quality_threshold=0.65,
            decay_min_age_days=0,
            decay_step=1,
            decay_remove_below=0,
            draft_store_path=base / "drafts.json",
            adoption_log_path=base / "adoptions.json",
        )
        result = distiller.run_skill_governance()

        assert result["decayed_removed"] == 0
        assert "auto_decay_only" in manager.skills
        assert manager.skills["auto_decay_only"]["metadata"]["decay_score"] == 2
        assert "auto_decay_only" in manager.updated_names


def main() -> None:
    test_semantic_deduplicate_on_adopt()
    print("[PASS] test_semantic_deduplicate_on_adopt")
    test_low_value_auto_skill_decay_and_remove()
    print("[PASS] test_low_value_auto_skill_decay_and_remove")
    test_low_value_auto_skill_decay_decrement_without_remove()
    print("[PASS] test_low_value_auto_skill_decay_decrement_without_remove")
    print("[DONE] Phase 5 P1 skill governance checks passed")


if __name__ == "__main__":
    main()
