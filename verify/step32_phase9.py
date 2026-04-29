"""
Phase 9 P1-1: Aggregate recall logs into a report (offline, regression-ready).

Reads JSONL from:
  - explicit --log-path, OR
  - default ~/.hermes-cn/recall/recall_logs.jsonl (respects AGENT_DATA_DIR)

Exports:
  verify/reports/phase9_recall_health_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observability.recall_logger import RecallLogger


def _f(obj: dict, path: list[str], default: float = 0.0) -> float:
    cur: Any = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except Exception:
        return default


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    q = max(0.0, min(1.0, float(q)))
    idx = int(round((len(v) - 1) * q))
    return float(v[idx])


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="How many recent recall events to aggregate")
    parser.add_argument("--log-path", type=str, default="", help="Optional explicit recall_logs.jsonl path")
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "reports" / "phase9_recall_health_report.json"),
    )
    args = parser.parse_args()

    logger = RecallLogger(log_path=args.log_path or None)
    rows = logger.read_last(n=args.n)
    if not rows:
        print(f"[FAIL] no recall logs found at: {logger.path}")
        sys.exit(2)

    useful3 = [_f(r, ["metrics", "useful_at_3"]) for r in rows]
    div3 = [_f(r, ["metrics", "component_diversity_at_3"]) for r in rows]

    report = {
        "generated_at": datetime.now().isoformat(),
        "log_path": str(logger.path),
        "window_n": len(rows),
        "metrics": {
            "useful_at_3_avg": sum(useful3) / max(1, len(useful3)),
            "useful_at_3_p50": _quantile(useful3, 0.50),
            "useful_at_3_p10": _quantile(useful3, 0.10),
            "component_diversity_at_3_avg": sum(div3) / max(1, len(div3)),
            "component_diversity_at_3_p50": _quantile(div3, 0.50),
            "component_diversity_at_3_p10": _quantile(div3, 0.10),
        },
        "thresholds": {
            "useful_at_3_min": float(os.getenv("AGENT_RECALL_HEALTH_USEFUL_AT_3_MIN", "0.80") or 0.80),
            "component_diversity_at_3_min": float(os.getenv("AGENT_RECALL_HEALTH_DIVERSITY_AT_3_MIN", "0.34") or 0.34),
        },
    }

    report_path = Path(args.report_path).resolve()
    _write_report(report_path, report)
    print("[PASS] recall_health_report_generated")
    print(f"[PASS] report_written: {report_path}")
    print(f"[DONE] window_n={report['window_n']}")


if __name__ == "__main__":
    main()

