"""
Phase 7 P0: Recall weight sweep (offline).

Goal:
  - Sweep a small grid of hybrid ranking weights (lexical/semantic/recency)
  - Evaluate on Phase 6 dataset
  - Export report and print best config

Default output:
  verify/reports/phase7_recall_weight_sweep.json
"""

from __future__ import annotations

import gc
import itertools
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


def _load_eval_dataset(path: Path) -> tuple[list[SeedSession], list[EvalCase]]:
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
    return sessions, cases


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


def _score(metrics: dict[str, float]) -> float:
    # Primary: recall@1 & MRR; secondary: reason coverage.
    return (
        0.45 * float(metrics.get("recall_at_1", 0.0))
        + 0.45 * float(metrics.get("mrr", 0.0))
        + 0.10 * float(metrics.get("reason_coverage", 0.0))
    )


def _iter_weight_grid() -> list[tuple[float, float, float]]:
    # Keep search space small for fast local runs.
    lexicals = [0.50, 0.55, 0.60, 0.65, 0.70]
    semantics = [0.20, 0.25, 0.30, 0.35, 0.40]
    out: list[tuple[float, float, float]] = []
    for wl, ws in itertools.product(lexicals, semantics):
        wr = 1.0 - wl - ws
        if wr < 0.0 or wr > 0.20:
            continue
        out.append((wl, ws, wr))
    if (0.60, 0.30, 0.10) not in out:
        out.append((0.60, 0.30, 0.10))
    return sorted(set(out))


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    dataset_path = Path(__file__).resolve().parent / "data_phase6_recall_cases.json"
    report_path = Path(__file__).resolve().parent / "reports" / "phase7_recall_weight_sweep.json"

    sessions, cases = _load_eval_dataset(dataset_path)
    grid = _iter_weight_grid()

    results: list[dict] = []
    best: dict | None = None

    for (wl, ws, wr) in grid:
        # Configure weights via env vars.
        os.environ["AGENT_RECALL_WEIGHT_LEXICAL"] = str(wl)
        os.environ["AGENT_RECALL_WEIGHT_SEMANTIC"] = str(ws)
        os.environ["AGENT_RECALL_WEIGHT_RECENCY"] = str(wr)

        with _tmpdir_context() as tmp:
            mm = MemoryManager(llm_client=FakeLLM(), data_dir=tmp)  # type: ignore[arg-type]
            _seed_sessions(mm, sessions)
            metrics = _compute_recall_metrics(mm, cases)
            _close_memory(mm)

        scored = _score(metrics)
        row = {
            "weights": {"lexical": wl, "semantic": ws, "recency": wr},
            "metrics": metrics,
            "score": round(scored, 6),
        }
        results.append(row)
        if best is None or row["score"] > best["score"]:
            best = row

    results.sort(key=lambda r: float(r["score"]), reverse=True)
    payload = {
        "dataset_path": str(dataset_path),
        "generated_at": datetime.now().isoformat(),
        "grid_size": len(grid),
        "best": best,
        "top5": results[:5],
        "all": results,
    }
    _write_report(report_path, payload)

    print("[PASS] recall_weight_sweep_ran")
    print(f"[PASS] report_written: {report_path}")
    if best:
        w = best["weights"]
        m = best["metrics"]
        print(
            "[DONE] best="
            f"w=({w['lexical']:.2f},{w['semantic']:.2f},{w['recency']:.2f}) "
            f"recall@1={m['recall_at_1']:.3f} mrr={m['mrr']:.3f} reason={m['reason_coverage']:.3f}"
        )


if __name__ == "__main__":
    main()

