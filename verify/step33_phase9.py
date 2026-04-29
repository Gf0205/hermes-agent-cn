"""
Phase 9 P1-1: Recall health report diff / regression gate.

Default:
  - current:   verify/reports/phase9_recall_health_report.json
  - baseline:  verify/reports/phase9_recall_health_report.baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetricDelta:
    name: str
    baseline: float
    current: float
    delta: float


def _read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(report: dict) -> dict[str, float]:
    raw = report.get("metrics", {}) or {}
    keys = [
        "useful_at_3_avg",
        "useful_at_3_p50",
        "useful_at_3_p10",
        "component_diversity_at_3_avg",
        "component_diversity_at_3_p50",
        "component_diversity_at_3_p10",
    ]
    out: dict[str, float] = {}
    for k in keys:
        try:
            out[k] = float(raw.get(k, 0.0) or 0.0)
        except Exception:
            out[k] = 0.0
    return out


def _compare(baseline: dict[str, float], current: dict[str, float]) -> list[MetricDelta]:
    keys = sorted(set(baseline.keys()) | set(current.keys()))
    out: list[MetricDelta] = []
    for k in keys:
        b = float(baseline.get(k, 0.0))
        c = float(current.get(k, 0.0))
        out.append(MetricDelta(name=k, baseline=b, current=c, delta=c - b))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    base_dir = Path(__file__).resolve().parent / "reports"
    parser.add_argument("--current", type=str, default=str(base_dir / "phase9_recall_health_report.json"))
    parser.add_argument("--baseline", type=str, default=str(base_dir / "phase9_recall_health_report.baseline.json"))
    parser.add_argument("--allow-drop", type=float, default=0.02)
    args = parser.parse_args()

    current_path = Path(args.current).resolve()
    baseline_path = Path(args.baseline).resolve()
    allow_drop = max(0.0, float(args.allow_drop))

    if not current_path.exists():
        print(f"[FAIL] current report missing: {current_path}")
        sys.exit(2)

    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(current_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[PASS] baseline created: {baseline_path}")
        sys.exit(0)

    baseline_metrics = _extract_metrics(_read_report(baseline_path))
    current_metrics = _extract_metrics(_read_report(current_path))
    deltas = _compare(baseline_metrics, current_metrics)

    print("========== Phase 9 Recall Health Diff ==========")
    for d in deltas:
        sign = "+" if d.delta >= 0 else "-"
        print(f"{d.name:<28} baseline={d.baseline:.4f} current={d.current:.4f} diff={sign}{abs(d.delta):.4f}")
    print("================================================")

    degraded = [d for d in deltas if d.delta < -allow_drop]
    if degraded:
        print("[FAIL] recall health degraded beyond threshold:")
        for d in degraded:
            print(f"  - {d.name}: {d.baseline:.4f} -> {d.current:.4f} (diff {d.delta:.4f})")
        sys.exit(1)

    print("[DONE] Phase 9 recall health diff OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

