"""
Phase 6 P0 验证脚本（跨会话检索离线评估基线）
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.memory_manager import MemoryManager
from src.models import Session


class FakeLLM:
    """
    Deterministic embedding for offline evaluation.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            t = text.lower()
            if "登录" in t or "auth" in t or "jwt" in t:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "缓存" in t or "cache" in t or "redis" in t:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "连接池" in t or "db pool" in t or "throughput" in t:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            elif "巡检" in t or "audit" in t:
                vectors.append([0.0, 0.0, 0.0, 1.0])
            else:
                vectors.append([0.25, 0.25, 0.25, 0.25])
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


@dataclass
class EvalCase:
    query: str
    expected_id: str
    min_reason_token: str = ""


@dataclass
class SeedSession:
    id: str
    title: str
    messages: list[dict[str, str]]
    goal: str
    hours_ago: int


def _seed_sessions(mm: MemoryManager, sessions: list[SeedSession]) -> None:
    now = datetime.now()
    for item in sessions:
        mm.save_session(Session(
            id=item.id,
            title=item.title,
            messages=item.messages,
            created_at=now - timedelta(hours=item.hours_ago),
            metadata={"goal": item.goal},
        ))


def _compute_recall_metrics(mm: MemoryManager, cases: list[EvalCase]) -> dict[str, float]:
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_rank_sum = 0.0
    reason_coverage = 0

    for case in cases:
        hits = mm.search_sessions(case.query, limit=5)
        ids = [str(item.get("id", "")) for item in hits]

        if ids and ids[0] == case.expected_id:
            hit_at_1 += 1
        if case.expected_id in ids[:3]:
            hit_at_3 += 1

        if case.expected_id in ids:
            rank = ids.index(case.expected_id) + 1
            reciprocal_rank_sum += 1.0 / rank

        if case.min_reason_token:
            for item in hits:
                if str(item.get("id", "")) == case.expected_id:
                    reason = str(item.get("match_reason", ""))
                    if case.min_reason_token in reason:
                        reason_coverage += 1
                    break

    total = max(1, len(cases))
    return {
        "recall_at_1": hit_at_1 / total,
        "recall_at_3": hit_at_3 / total,
        "mrr": reciprocal_rank_sum / total,
        "reason_coverage": reason_coverage / total,
    }


def _load_eval_dataset(path: Path) -> tuple[list[SeedSession], list[EvalCase], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions = [
        SeedSession(
            id=str(item["id"]),
            title=str(item["title"]),
            messages=[{"role": str(m["role"]), "content": str(m["content"])} for m in item.get("messages", [])],
            goal=str(item["goal"]),
            hours_ago=int(item.get("hours_ago", 24)),
        )
        for item in payload.get("sessions", [])
    ]
    cases = [
        EvalCase(
            query=str(item["query"]),
            expected_id=str(item["expected_id"]),
            min_reason_token=str(item.get("min_reason_token", "")),
        )
        for item in payload.get("queries", [])
    ]
    thresholds = {
        "recall_at_1_min": float(payload.get("thresholds", {}).get("recall_at_1_min", 0.75)),
        "recall_at_3_min": float(payload.get("thresholds", {}).get("recall_at_3_min", 1.0)),
        "mrr_min": float(payload.get("thresholds", {}).get("mrr_min", 0.8)),
        "reason_coverage_min": float(payload.get("thresholds", {}).get("reason_coverage_min", 0.75)),
    }
    return sessions, cases, thresholds


def test_recall_offline_baseline(dataset_path: Path) -> dict[str, float]:
    with _tmpdir_context() as tmp:
        mm = MemoryManager(llm_client=FakeLLM(), data_dir=tmp)  # type: ignore[arg-type]
        sessions, cases, thresholds = _load_eval_dataset(dataset_path)
        _seed_sessions(mm, sessions)
        metrics = _compute_recall_metrics(mm, cases)

        assert metrics["recall_at_1"] >= thresholds["recall_at_1_min"]
        assert metrics["recall_at_3"] >= thresholds["recall_at_3_min"]
        assert metrics["mrr"] >= thresholds["mrr_min"]
        assert metrics["reason_coverage"] >= thresholds["reason_coverage_min"]
        _close_memory(mm)
        return metrics


def _write_report(
    report_path: Path,
    dataset_path: Path,
    metrics: dict[str, float],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(dataset_path),
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    dataset_path = Path(__file__).resolve().parent / "data_phase6_recall_cases.json"
    report_path = Path(__file__).resolve().parent / "reports" / "phase6_recall_eval_report.json"
    metrics = test_recall_offline_baseline(dataset_path=dataset_path)
    _write_report(report_path=report_path, dataset_path=dataset_path, metrics=metrics)
    print("[PASS] test_recall_offline_baseline")
    print(f"[PASS] report_written: {report_path}")
    print("[DONE] Phase 6 P0 recall evaluation baseline checks passed")


if __name__ == "__main__":
    main()
