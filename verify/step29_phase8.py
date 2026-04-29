"""
Phase 8 P0: Retrieval usability metrics (offline).

Metrics:
  - useful@1, useful@3: at least one "useful" hit appears in top-k
  - component_diversity@3: distinct dominant score components among top-3 / 3
  - stability@3: repeated runs produce identical top-3 ids

Report:
  verify/reports/phase8_recall_usability_report.json
"""

from __future__ import annotations

import gc
import json
import os
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
    thresholds_raw = payload.get("usability_thresholds", {}) or {}
    thresholds = {
        "useful_at_1_min": float(thresholds_raw.get("useful_at_1_min", 0.75)),
        "useful_at_3_min": float(thresholds_raw.get("useful_at_3_min", 1.0)),
        "component_diversity_at_3_min": float(thresholds_raw.get("component_diversity_at_3_min", 0.34)),
        "stability_at_3_min": float(thresholds_raw.get("stability_at_3_min", 1.0)),
    }
    return sessions, cases, thresholds


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


USEFUL_REASON_TOKENS = {
    "title-match",
    "goal-match",
    "messages-match",
    "semantic-strong",
    "semantic-related",
    "lexical-related",
}


def _is_useful(hit: dict) -> bool:
    reason = str(hit.get("match_reason", "")).strip()
    if not reason:
        return False
    tokens = {t.strip() for t in reason.split(",") if t.strip()}
    return any(t in USEFUL_REASON_TOKENS for t in tokens)


def _dominant_component(hit: dict) -> str:
    breakdown = hit.get("score_breakdown", {}) or {}
    lexical = float((breakdown.get("lexical", 0.0) or 0.0))
    semantic = float((breakdown.get("semantic", 0.0) or 0.0))
    recency = float((breakdown.get("recency", 0.0) or 0.0))

    # Prefer lexical/semantic when ties happen; keep deterministic.
    if lexical >= semantic and lexical >= recency:
        return "lexical"
    if semantic >= lexical and semantic >= recency:
        return "semantic"
    return "recency"


def _compute_usability_metrics(mm: MemoryManager, cases: list[EvalCase], k: int = 3, runs: int = 3) -> dict[str, float]:
    useful_at_1 = 0
    useful_at_k = 0
    diversity_sum = 0.0
    stable_count = 0

    for case in cases:
        hits = mm.search_sessions(case.query, limit=max(5, k))
        topk = hits[:k]
        top1 = hits[:1]

        if any(_is_useful(h) for h in top1):
            useful_at_1 += 1
        if any(_is_useful(h) for h in topk):
            useful_at_k += 1

        if topk:
            comps = {_dominant_component(h) for h in topk}
            diversity_sum += len(comps) / max(1, len(topk))

        # Stability: same top-k ids across repeated runs (same DB, same query)
        ids0 = [str(x.get("id", "")) for x in topk]
        stable = True
        for _ in range(max(1, runs) - 1):
            hits_r = mm.search_sessions(case.query, limit=max(5, k))
            ids_r = [str(x.get("id", "")) for x in hits_r[:k]]
            if ids_r != ids0:
                stable = False
                break
        if stable:
            stable_count += 1

    total = max(1, len(cases))
    return {
        "useful_at_1": useful_at_1 / total,
        "useful_at_3": useful_at_k / total,
        "component_diversity_at_3": diversity_sum / total,
        "stability_at_3": stable_count / total,
    }


def _write_report(report_path: Path, dataset_path: Path, metrics: dict[str, float]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_path": str(dataset_path),
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # Best weights from Phase 7 sweep (can be overridden by env).
    os.environ.setdefault("AGENT_RECALL_WEIGHT_LEXICAL", "0.50")
    os.environ.setdefault("AGENT_RECALL_WEIGHT_SEMANTIC", "0.30")
    os.environ.setdefault("AGENT_RECALL_WEIGHT_RECENCY", "0.20")
    # Enable top-k diversification by default for usability eval (can be overridden).
    os.environ.setdefault("AGENT_RECALL_DIVERSIFY_TOPK", "1")
    os.environ.setdefault("AGENT_RECALL_DIVERSIFY_LAMBDA", "0.88")

    dataset_path = Path(__file__).resolve().parent / "data_phase6_recall_cases.json"
    report_path = Path(__file__).resolve().parent / "reports" / "phase8_recall_usability_report.json"

    sessions, cases, thresholds = _load_eval_dataset(dataset_path)
    with _tmpdir_context() as tmp:
        mm = MemoryManager(llm_client=FakeLLM(), data_dir=tmp)  # type: ignore[arg-type]
        _seed_sessions(mm, sessions)
        metrics = _compute_usability_metrics(mm, cases, k=3, runs=3)
        _close_memory(mm)

    assert metrics["useful_at_1"] >= thresholds["useful_at_1_min"]
    assert metrics["useful_at_3"] >= thresholds["useful_at_3_min"]
    assert metrics["component_diversity_at_3"] >= thresholds["component_diversity_at_3_min"]
    assert metrics["stability_at_3"] >= thresholds["stability_at_3_min"]

    _write_report(report_path=report_path, dataset_path=dataset_path, metrics=metrics)
    print("[PASS] retrieval_usability_metrics")
    print(f"[PASS] report_written: {report_path}")
    print("[DONE] Phase 8 P0 retrieval usability checks passed")


if __name__ == "__main__":
    main()

