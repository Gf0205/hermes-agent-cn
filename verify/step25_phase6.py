"""
Phase 6 P2 验证脚本（评估报告对比 / 退化告警）

用法：
  python verify/step25_phase6.py --baseline verify/reports/phase6_recall_eval_report.json --current verify/reports/phase6_recall_eval_report.json

默认：
  - baseline: verify/reports/phase6_recall_eval_report.baseline.json（若不存在则复制 current 作为 baseline 并退出 0）
  - current:  verify/reports/phase6_recall_eval_report.json
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
    out: dict[str, float] = {}
    for key in ["recall_at_1", "recall_at_3", "mrr", "reason_coverage"]:
        try:
            out[key] = float(raw.get(key, 0.0) or 0.0)
        except Exception:
            out[key] = 0.0
    return out


def _compare(baseline: dict[str, float], current: dict[str, float]) -> list[MetricDelta]:
    deltas: list[MetricDelta] = []
    keys = sorted(set(baseline.keys()) | set(current.keys()))
    for key in keys:
        b = float(baseline.get(key, 0.0))
        c = float(current.get(key, 0.0))
        deltas.append(MetricDelta(name=key, baseline=b, current=c, delta=c - b))
    return deltas


def _print_summary(deltas: list[MetricDelta]) -> None:
    print("========== Phase 6 Report Diff ==========")
    for d in deltas:
        sign = "+" if d.delta >= 0 else "-"
        print(f"{d.name:<16} baseline={d.baseline:.4f} current={d.current:.4f} diff={sign}{abs(d.delta):.4f}")
    print("=========================================")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current",
        type=str,
        default=str(Path(__file__).resolve().parent / "reports" / "phase6_recall_eval_report.json"),
        help="Current report path",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(Path(__file__).resolve().parent / "reports" / "phase6_recall_eval_report.baseline.json"),
        help="Baseline report path",
    )
    parser.add_argument(
        "--allow-drop",
        type=float,
        default=0.01,
        help="Allowed metric drop before failing (default 0.01)",
    )
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

    baseline_report = _read_report(baseline_path)
    current_report = _read_report(current_path)
    base_metrics = _extract_metrics(baseline_report)
    cur_metrics = _extract_metrics(current_report)
    deltas = _compare(base_metrics, cur_metrics)
    _print_summary(deltas)

    degraded = [d for d in deltas if d.delta < -allow_drop]
    if degraded:
        print("[FAIL] metrics degraded beyond threshold:")
        for d in degraded:
            print(f"  - {d.name}: {d.baseline:.4f} -> {d.current:.4f} (diff {d.delta:.4f})")
        sys.exit(1)

    print("[DONE] Phase 6 report diff OK")
    sys.exit(0)


if __name__ == "__main__":
    main()

