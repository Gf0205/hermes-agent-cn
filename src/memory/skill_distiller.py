"""
src/memory/skill_distiller.py - 自动技能蒸馏器
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.event_bus import Event, EventType, get_event_bus
from src.observability.tracer import ExecutionTracer


@dataclass
class SkillDraft:
    draft_id: str
    name: str
    description: str
    steps: list[str]
    tags: list[str] = field(default_factory=list)
    source_goal: str = ""
    quality_score: float = 0.0
    recommended: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "name": self.name,
            "description": self.description,
            "steps": list(self.steps),
            "tags": list(self.tags),
            "source_goal": self.source_goal,
            "quality_score": self.quality_score,
            "recommended": self.recommended,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillDraft":
        return cls(
            draft_id=str(data.get("draft_id", "")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            steps=[str(item) for item in (data.get("steps", []) or [])],
            tags=[str(item) for item in (data.get("tags", []) or [])],
            source_goal=str(data.get("source_goal", "")),
            quality_score=float(data.get("quality_score", 0.0) or 0.0),
            recommended=bool(data.get("recommended", False)),
            created_at=str(data.get("created_at", datetime.now().isoformat())),
        )


@dataclass
class SkillAdoptionRecord:
    record_id: str
    timestamp: str
    source: str  # auto / manual
    skill_name: str
    quality_score: float
    draft: dict[str, Any]
    deduplicated: bool = False
    deduplicated_to: str = ""
    rolled_back: bool = False
    rolled_back_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "skill_name": self.skill_name,
            "quality_score": self.quality_score,
            "draft": dict(self.draft),
            "deduplicated": self.deduplicated,
            "deduplicated_to": self.deduplicated_to,
            "rolled_back": self.rolled_back,
            "rolled_back_at": self.rolled_back_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillAdoptionRecord":
        return cls(
            record_id=str(data.get("record_id", "")),
            timestamp=str(data.get("timestamp", "")),
            source=str(data.get("source", "manual")),
            skill_name=str(data.get("skill_name", "")),
            quality_score=float(data.get("quality_score", 0.0) or 0.0),
            draft=dict(data.get("draft", {}) or {}),
            deduplicated=bool(data.get("deduplicated", False)),
            deduplicated_to=str(data.get("deduplicated_to", "")),
            rolled_back=bool(data.get("rolled_back", False)),
            rolled_back_at=str(data.get("rolled_back_at", "")),
        )


class SkillDistiller:
    """
    基于成功执行轨迹自动提炼技能草稿并落盘。
    设计为“规则优先”的首版：不额外调用LLM，先保证稳定与可追踪。
    """

    def __init__(
        self,
        memory_manager: Any,
        tracer: ExecutionTracer | None = None,
        min_tool_calls: int = 3,
        auto_subscribe: bool = True,
        draft_store_path: str | Path | None = None,
        auto_adopt_threshold: float | None = None,
        adoption_log_path: str | Path | None = None,
        dedupe_similarity_threshold: float = 0.78,
        decay_quality_threshold: float = 0.65,
        decay_min_age_days: int = 3,
        decay_step: int = 1,
        decay_remove_below: int = 0,
    ) -> None:
        self._memory = memory_manager
        self._tracer = tracer or ExecutionTracer()
        self._min_tool_calls = min_tool_calls
        self._bus = get_event_bus()
        self._auto_adopt_threshold = self._resolve_auto_adopt_threshold(auto_adopt_threshold)
        self._dedupe_similarity_threshold = max(0.0, min(1.0, dedupe_similarity_threshold))
        self._decay_quality_threshold = max(0.0, min(1.0, decay_quality_threshold))
        self._decay_min_age_days = max(0, int(decay_min_age_days))
        self._decay_step = max(1, int(decay_step))
        self._decay_remove_below = int(decay_remove_below)
        self._draft_store_path = self._resolve_draft_store_path(draft_store_path)
        self._adoption_log_path = self._resolve_adoption_log_path(adoption_log_path)
        self._recent_drafts: list[SkillDraft] = self._load_drafts()
        self._adoption_records: list[SkillAdoptionRecord] = self._load_adoption_records()

        if auto_subscribe:
            self._bus.subscribe(EventType.AGENT_COMPLETED, self._on_agent_completed)

    def _on_agent_completed(self, event: Event) -> None:
        if not bool(event.data.get("success", False)):
            return

        trace_data = self._load_latest_trace()
        if not trace_data:
            return

        goal = str(event.data.get("goal", "") or trace_data.get("goal", ""))
        draft = self.distill_from_trace_data(trace_data, goal_hint=goal)
        if draft is None:
            return

        self._append_draft(draft)
        self._try_auto_adopt()
        self.run_skill_governance()

    def distill_from_trace_data(
        self,
        trace_data: dict[str, Any],
        goal_hint: str = "",
    ) -> SkillDraft | None:
        if not bool(trace_data.get("success", False)):
            return None

        steps = trace_data.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return None

        tool_sequence: list[str] = []
        failure_hints: list[str] = []
        total_tool_calls = int(trace_data.get("total_tool_calls", 0))

        for step in steps:
            for tool_call in step.get("tool_calls", []) or []:
                tool_name = str(tool_call.get("tool_name", "")).strip()
                if tool_name and (not tool_sequence or tool_sequence[-1] != tool_name):
                    tool_sequence.append(tool_name)

            for tool_result in step.get("tool_results", []) or []:
                status = str(tool_result.get("status", ""))
                if status and status != "success":
                    tool_name = str(tool_result.get("tool_name", ""))
                    error = str(tool_result.get("error", ""))
                    failure_hints.append(f"{tool_name} 失败: {error[:100]}")

        if total_tool_calls < self._min_tool_calls and len(tool_sequence) < self._min_tool_calls:
            return None

        goal = (goal_hint or str(trace_data.get("goal", ""))).strip() or "task"
        skill_name = self._build_skill_name(goal)
        skill_steps = [f"按顺序执行工具：{' -> '.join(tool_sequence)}"]
        if failure_hints:
            skill_steps.append("常见失败与规避：")
            skill_steps.extend([f"- {item}" for item in failure_hints[:3]])

        description = (
            f"从成功任务自动提炼：{goal[:80]}。"
            f"总工具调用 {total_tool_calls} 次，核心工具链 {len(tool_sequence)} 步。"
        )
        quality_score = self._estimate_quality_score(
            total_tool_calls=total_tool_calls,
            tool_sequence=tool_sequence,
            failure_hints=failure_hints,
        )
        tags = self._infer_tags(goal, tool_sequence)
        return SkillDraft(
            draft_id=self._build_draft_id(goal, tool_sequence),
            name=skill_name,
            description=description,
            steps=skill_steps,
            tags=tags,
            source_goal=goal,
            quality_score=quality_score,
            recommended=quality_score >= 0.75 and len(failure_hints) <= 1,
        )

    def get_recent_drafts(self) -> list[SkillDraft]:
        return list(self._recent_drafts)

    def get_auto_adopt_threshold(self) -> float:
        return self._auto_adopt_threshold

    def get_recent_adoption_records(self, limit: int = 20) -> list[SkillAdoptionRecord]:
        return list(self._adoption_records[-max(1, limit):])

    def get_governance_status(self) -> dict[str, Any]:
        skills = self._safe_list_skills()
        auto_skills = [s for s in skills if str(s.get("name", "")).startswith("auto_")]
        low_quality_auto = 0
        for skill in auto_skills:
            metadata = dict(skill.get("metadata", {}) or {})
            quality = float(metadata.get("quality_score", 0.0) or 0.0)
            if quality < self._decay_quality_threshold:
                low_quality_auto += 1

        dedup_count = sum(1 for r in self._adoption_records if r.deduplicated)
        rolled_back_count = sum(1 for r in self._adoption_records if r.rolled_back)
        return {
            "drafts": len(self._recent_drafts),
            "adoption_records": len(self._adoption_records),
            "deduplicated_records": dedup_count,
            "rolled_back_records": rolled_back_count,
            "auto_skills": len(auto_skills),
            "low_quality_auto_skills": low_quality_auto,
            "dedupe_similarity_threshold": self._dedupe_similarity_threshold,
            "decay_quality_threshold": self._decay_quality_threshold,
            "decay_min_age_days": self._decay_min_age_days,
            "decay_step": self._decay_step,
            "decay_remove_below": self._decay_remove_below,
        }

    def adopt_draft(self, draft_index: int, source: str = "manual") -> str:
        if draft_index < 0 or draft_index >= len(self._recent_drafts):
            raise IndexError("draft_index 越界")

        draft = self._recent_drafts[draft_index]
        existing = self._memory.load_skill(draft.name) if hasattr(self._memory, "load_skill") else None
        if existing:
            self._recent_drafts.pop(draft_index)
            self._persist_drafts()
            return draft.name

        duplicate_skill = self._find_semantic_duplicate(draft)
        if duplicate_skill:
            self._append_adoption_record(
                SkillAdoptionRecord(
                    record_id=self._build_record_id(draft, source),
                    timestamp=datetime.now().isoformat(),
                    source=source,
                    skill_name=duplicate_skill,
                    quality_score=draft.quality_score,
                    draft=draft.to_dict(),
                    deduplicated=True,
                    deduplicated_to=duplicate_skill,
                )
            )
            self._recent_drafts.pop(draft_index)
            self._persist_drafts()
            return duplicate_skill

        self._save_skill_with_governance_metadata(draft)
        self._append_adoption_record(
            SkillAdoptionRecord(
                record_id=self._build_record_id(draft, source),
                timestamp=datetime.now().isoformat(),
                source=source,
                skill_name=draft.name,
                quality_score=draft.quality_score,
                draft=draft.to_dict(),
            )
        )
        self._recent_drafts.pop(draft_index)
        self._persist_drafts()
        return draft.name

    def run_skill_governance(self) -> dict[str, Any]:
        scanned, decayed_updated, decayed_removed = self._apply_auto_skill_decay()
        return {
            "scanned_auto_skills": scanned,
            "decayed_updated": decayed_updated,
            "decayed_removed": decayed_removed,
        }

    def rollback_last_auto_adopt(self) -> dict[str, Any] | None:
        return self.rollback_auto_adopt(record_id=None)

    def rollback_auto_adopt(self, record_id: str | None = None) -> dict[str, Any] | None:
        if record_id:
            target_idx = None
            for idx, record in enumerate(self._adoption_records):
                if record.record_id == record_id:
                    target_idx = idx
                    break
            if target_idx is None:
                return None
            candidate_indexes = [target_idx]
        else:
            candidate_indexes = list(range(len(self._adoption_records) - 1, -1, -1))

        for idx in candidate_indexes:
            record = self._adoption_records[idx]
            if record.source != "auto" or record.rolled_back:
                continue

            removed = self._delete_skill(record.skill_name)
            if not removed:
                return None

            draft_data = dict(record.draft)
            try:
                draft = SkillDraft.from_dict(draft_data)
                self._append_draft(draft)
            except Exception:
                pass

            record.rolled_back = True
            record.rolled_back_at = datetime.now().isoformat()
            self._adoption_records[idx] = record
            self._persist_adoption_records()
            return {
                "record_id": record.record_id,
                "skill_name": record.skill_name,
                "rolled_back_at": record.rolled_back_at,
            }
        return None

    def _append_draft(self, draft: SkillDraft) -> None:
        # 以 draft_id 为主键去重；若名字+目标一致则覆盖旧项，保留最新时间戳。
        filtered = [
            item
            for item in self._recent_drafts
            if item.draft_id != draft.draft_id
            and not (item.name == draft.name and item.source_goal == draft.source_goal)
        ]
        filtered.append(draft)
        self._recent_drafts = filtered[-20:]
        self._persist_drafts()

    def _resolve_draft_store_path(self, draft_store_path: str | Path | None) -> Path | None:
        if draft_store_path:
            return Path(draft_store_path)

        base_dir = getattr(self._memory, "_data_dir", None)
        if base_dir is None:
            return None

        try:
            return Path(base_dir) / "skill_drafts.json"
        except Exception:
            return None

    def _load_drafts(self) -> list[SkillDraft]:
        path = self._draft_store_path
        if path is None or not path.exists():
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("drafts", [])
            if not isinstance(rows, list):
                return []
            drafts = [SkillDraft.from_dict(item) for item in rows if isinstance(item, dict)]
            return drafts[-20:]
        except Exception:
            return []

    def _persist_drafts(self) -> None:
        path = self._draft_store_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [draft.to_dict() for draft in self._recent_drafts[-20:]]
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            # 草稿持久化失败不影响主流程
            return

    def _resolve_adoption_log_path(self, adoption_log_path: str | Path | None) -> Path | None:
        if adoption_log_path:
            return Path(adoption_log_path)

        base_dir = getattr(self._memory, "_data_dir", None)
        if base_dir is None:
            return None
        try:
            return Path(base_dir) / "skill_adoption_log.json"
        except Exception:
            return None

    def _load_adoption_records(self) -> list[SkillAdoptionRecord]:
        path = self._adoption_log_path
        if path is None or not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else payload.get("records", [])
            if not isinstance(rows, list):
                return []
            records = [
                SkillAdoptionRecord.from_dict(item)
                for item in rows
                if isinstance(item, dict)
            ]
            return records[-200:]
        except Exception:
            return []

    def _persist_adoption_records(self) -> None:
        path = self._adoption_log_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [record.to_dict() for record in self._adoption_records[-200:]]
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _append_adoption_record(self, record: SkillAdoptionRecord) -> None:
        self._adoption_records.append(record)
        self._adoption_records = self._adoption_records[-200:]
        self._persist_adoption_records()

    def _build_record_id(self, draft: SkillDraft, source: str) -> str:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        seed = f"{source}_{draft.draft_id}_{ts}"
        compact = re.sub(r"[^a-zA-Z0-9_]+", "", seed.lower())
        return compact[:32] or ts

    def _delete_skill(self, skill_name: str) -> bool:
        if hasattr(self._memory, "delete_skill"):
            try:
                return bool(self._memory.delete_skill(skill_name))
            except Exception:
                return False

        skills_dir = getattr(self._memory, "_skills_dir", None)
        if skills_dir is None:
            return False
        try:
            path = Path(skills_dir) / f"{skill_name.replace(' ', '_')}.json"
            if not path.exists():
                return False
            path.unlink()
            return True
        except Exception:
            return False

    def _resolve_auto_adopt_threshold(self, threshold: float | None) -> float:
        if threshold is None:
            raw = os.getenv("AGENT_SKILL_AUTO_ADOPT_THRESHOLD", "0")
            try:
                threshold = float(raw)
            except ValueError:
                threshold = 0.0
        return max(0.0, min(1.0, float(threshold)))

    def _find_semantic_duplicate(self, draft: SkillDraft) -> str | None:
        if not hasattr(self._memory, "list_skills"):
            return None
        try:
            skills = list(self._memory.list_skills())  # type: ignore[no-untyped-call]
        except Exception:
            return None

        target_sig = self._build_semantic_signature(draft.source_goal, draft.steps)
        if not target_sig:
            return None

        best_name = None
        best_score = 0.0
        for skill in skills:
            name = str(skill.get("name", "")).strip()
            if not name:
                continue
            metadata = skill.get("metadata", {}) or {}
            candidate_sig = str(metadata.get("semantic_signature", "")).strip()
            if not candidate_sig:
                candidate_sig = self._build_semantic_signature(
                    str(skill.get("description", "")),
                    [str(item) for item in (skill.get("steps", []) or [])],
                )
            score = self._semantic_similarity(target_sig, candidate_sig)
            if score > best_score:
                best_score = score
                best_name = name

        if best_name and best_score >= self._dedupe_similarity_threshold:
            return best_name
        return None

    def _build_semantic_signature(self, goal: str, steps: list[str]) -> str:
        text = f"{goal}\n" + "\n".join(steps)
        tokens = re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]{2,}", text.lower())
        if not tokens:
            return ""
        uniq = sorted(set(tokens))
        return " ".join(uniq[:80])

    def _semantic_similarity(self, a: str, b: str) -> float:
        a_set = set(a.split())
        b_set = set(b.split())
        if not a_set or not b_set:
            return 0.0
        overlap = len(a_set & b_set)
        union = len(a_set | b_set)
        return overlap / max(1, union)

    def _apply_auto_skill_decay(self) -> tuple[int, int, int]:
        skills = self._safe_list_skills()
        removed = 0
        updated = 0
        scanned = 0
        now = datetime.now()
        for skill in skills:
            name = str(skill.get("name", ""))
            if not name.startswith("auto_"):
                continue
            scanned += 1

            use_count = int(skill.get("use_count", 0) or 0)
            metadata = dict(skill.get("metadata", {}) or {})
            quality = float(metadata.get("quality_score", 0.0) or 0.0)
            decay_score = int(metadata.get("decay_score", 3) or 3)
            created_at = self._safe_parse_dt(str(skill.get("created_at", "")))
            if created_at is None:
                continue

            age_days = (now - created_at).days
            if age_days < self._decay_min_age_days:
                continue
            if quality >= self._decay_quality_threshold:
                continue
            if use_count > 1:
                continue

            decay_score -= self._decay_step
            if decay_score <= self._decay_remove_below:
                if self._delete_skill(name):
                    removed += 1
                continue

            metadata["decay_score"] = decay_score
            if self._update_skill_metadata(name, metadata):
                updated += 1
        return scanned, updated, removed

    def _update_skill_metadata(self, skill_name: str, metadata: dict[str, Any]) -> bool:
        if hasattr(self._memory, "update_skill"):
            try:
                return bool(self._memory.update_skill(skill_name, {"metadata": metadata}))
            except Exception:
                return False
        return False

    def _save_skill_with_governance_metadata(self, draft: SkillDraft) -> None:
        metadata = {
            "source": "auto-distilled" if draft.name.startswith("auto_") else "manual",
            "quality_score": draft.quality_score,
            "semantic_signature": self._build_semantic_signature(draft.source_goal, draft.steps),
            "decay_score": 3,
        }
        try:
            self._memory.save_skill(
                skill_name=draft.name,
                description=draft.description,
                steps=draft.steps,
                tags=draft.tags,
                metadata=metadata,
            )
            return
        except TypeError:
            # 兼容旧版 memory_manager/FakeMemoryManager，不支持 metadata 参数。
            self._memory.save_skill(
                skill_name=draft.name,
                description=draft.description,
                steps=draft.steps,
                tags=draft.tags,
            )

    def _safe_parse_dt(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    def _safe_list_skills(self) -> list[dict[str, Any]]:
        if not hasattr(self._memory, "list_skills"):
            return []
        try:
            return list(self._memory.list_skills())  # type: ignore[no-untyped-call]
        except Exception:
            return []

    def _estimate_quality_score(
        self,
        total_tool_calls: int,
        tool_sequence: list[str],
        failure_hints: list[str],
    ) -> float:
        score = 0.35
        if total_tool_calls >= self._min_tool_calls:
            score += 0.25

        unique_tools = len(set(tool_sequence))
        score += min(0.25, unique_tools * 0.05)

        has_read = any(tool in {"read_file", "grep_search", "list_dir"} for tool in tool_sequence)
        has_write = any(tool in {"edit_file", "write_file", "shell"} for tool in tool_sequence)
        if has_read and has_write:
            score += 0.15

        score -= min(0.20, len(failure_hints) * 0.07)
        return round(max(0.0, min(1.0, score)), 4)

    def _try_auto_adopt(self) -> None:
        threshold = self._auto_adopt_threshold
        if threshold <= 0:
            return

        adopted = False
        for idx in range(len(self._recent_drafts) - 1, -1, -1):
            draft = self._recent_drafts[idx]
            if draft.quality_score < threshold:
                continue
            try:
                self.adopt_draft(idx, source="auto")
                adopted = True
            except Exception:
                continue
        if adopted:
            self._persist_drafts()

    def _load_latest_trace(self) -> dict[str, Any] | None:
        traces = self._tracer.list_traces(limit=1)
        if not traces:
            return None

        trace_file = traces[0].get("file", "")
        if not trace_file:
            return None

        traces_dir = getattr(self._tracer, "_traces_dir", None)
        if traces_dir is None:
            return None

        path = Path(traces_dir) / str(trace_file)
        if not path.exists():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _build_skill_name(self, goal: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]+", "_", goal).strip("_")
        if not slug:
            slug = "task"
        slug = slug[:40]
        return f"auto_{slug}_{datetime.now().strftime('%m%d')}"

    def _build_draft_id(self, goal: str, tool_sequence: list[str]) -> str:
        payload = f"{goal}|{'-'.join(tool_sequence)}|{datetime.now().strftime('%Y%m%d%H%M%S')}"
        compact = re.sub(r"[^a-zA-Z0-9]+", "", payload.lower())
        return compact[:24] or datetime.now().strftime("%H%M%S")

    def _infer_tags(self, goal: str, tools: list[str]) -> list[str]:
        tags: list[str] = ["auto-generated"]
        goal_lower = goal.lower()
        if "api" in goal_lower:
            tags.append("api")
        if "测试" in goal or "test" in goal_lower:
            tags.append("test")
        if any(tool in {"read_file", "edit_file", "write_file"} for tool in tools):
            tags.append("file-ops")
        if "shell" in tools:
            tags.append("shell")
        return tags[:6]
