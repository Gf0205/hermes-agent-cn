"""
Phase 7 P1: Compare recall weight sweep report to baseline (regression gate).

Default paths:
  - current:   verify/reports/phase7_recall_weight_sweep.json
  - baseline:  verify/reports/phase7_recall_weight_sweep.baseline.json

Behavior:
  - If baseline missing: create from current and exit 0.
  - Fail (exit 1) if best-score or key metrics drop beyond thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Delta:
    name: str
    baseline: float
    current: float
    delta: float


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_best(report: dict) -> dict:
    best = report.get("best") or {}
    if isinstance(best, dict):
        return best
    return {}


def _get_metrics(best: dict) -> dict[str, float]:
    raw = best.get("metrics", {}) or {}
    out: dict[str, float] = {}
    for key in ["recall_at_1", "recall_at_3", "mrr", "reason_coverage"]:
        try:
            out[key] = float(raw.get(key, 0.0) or 0.0)
        except Exception:
            out[key] = 0.0
    return out


def _get_score(best: dict) -> float:
    try:
        return float(best.get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _compare(baseline_metrics: dict[str, float], current_metrics: dict[str, float]) -> list[Delta]:
    keys = sorted(set(baseline_metrics.keys()) | set(current_metrics.keys()))
    deltas: list[Delta] = []
    for k in keys:
        b = float(baseline_metrics.get(k, 0.0))
        c = float(current_metrics.get(k, 0.0))
        deltas.append(Delta(name=k, baseline=b, current=c, delta=c - b))
    return deltas


def _print_header(baseline_path: Path, current_path: Path) -> None:
    print("========== Phase 7 Sweep Report Diff ==========")
    print(f"baseline: {baseline_path}")
    print(f"current : {current_path}")
    print("----------------------------------------------")


def _print_deltas(score_delta: Delta, metric_deltas: list[Delta]) -> None:
    def _fmt(d: Delta) -> str:
        sign = "+" if d.delta >= 0 else "-"
        return f"{d.name:<16} baseline={d.baseline:.4f} current={d.current:.4f} diff={sign}{abs(d.delta):.4f}"

    print(_fmt(score_delta))
    for d in metric_deltas:
        print(_fmt(d))
    print("==============================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent / "reports"
    parser.add_argument("--current", type=str, default=str(base_dir / "phase7_recall_weight_sweep.json"))
    parser.add_argument("--baseline", type=str, default=str(base_dir / "phase7_recall_weight_sweep.baseline.json"))
    parser.add_argument("--allow-drop-score", type=float, default=0.01)
    parser.add_argument("--allow-drop-metric", type=float, default=0.01)
    args = parser.parse_args()

    current_path = Path(args.current).resolve()
    baseline_path = Path(args.baseline).resolve()
    allow_drop_score = max(0.0, float(args.allow_drop_score))
    allow_drop_metric = max(0.0, float(args.allow_drop_metric))

    if not current_path.exists():
        print(f"[FAIL] current report missing: {current_path}")
        sys.exit(2)

    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(current_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[PASS] baseline created: {baseline_path}")
        sys.exit(0)

    baseline_report = _read_json(baseline_path)
    current_report = _read_json(current_path)

    b_best = _get_best(baseline_report)
    c_best = _get_best(current_report)

    b_score = _get_score(b_best)
    c_score = _get_score(c_best)
    score_delta = Delta(name="best_score", baseline=b_score, current=c_score, delta=c_score - b_score)

    b_metrics = _get_metrics(b_best)
    c_metrics = _get_metrics(c_best)
    metric_deltas = _compare(b_metrics, c_metrics)

    _print_header(baseline_path, current_path)
    _print_deltas(score_delta, metric_deltas)

    degraded: list[str] = []
    if score_delta.delta < -allow_drop_score:
        degraded.append(f"best_score drop {score_delta.delta:.4f} < -{allow_drop_score:.4f}")
    for d in metric_deltas:
        if d.delta < -allow_drop_metric:
            degraded.append(f"{d.name} drop {d.delta:.4f} < -{allow_drop_metric:.4f}")

    if degraded:
        print("[FAIL] sweep regression gate triggered:")
        for item in degraded:
            print(f"  - {item}")
        sys.exit(1)

    print("[DONE] Phase 7 sweep report diff OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

