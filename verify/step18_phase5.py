"""
Phase 5 总回归脚本

串行执行：
- step17_phase5.py (ContextCompressorV2 resilience)
- step19_phase5.py (Execution robustness)
- step20_phase5.py (Skill governance v2)
- step21_phase5.py (Operator governance UX)
- step22_phase5.py (Hybrid recall ranking)
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    name: str
    success: bool
    elapsed_s: float
    output: str
    exit_code: int


def run_script(script_path: Path) -> RunResult:
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    combined_output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return RunResult(
        name=script_path.name,
        success=proc.returncode == 0,
        elapsed_s=elapsed,
        output=combined_output.strip(),
        exit_code=proc.returncode,
    )


def print_summary(results: list[RunResult], total_elapsed_s: float) -> None:
    print("========== Phase 5 Regression Summary ==========")
    for result in results:
        icon = "[PASS]" if result.success else "[FAIL]"
        print(f"{icon} {result.name:<16} | {result.elapsed_s:>6.2f}s | exit={result.exit_code}")
    print("--------------------------------------")
    success_count = sum(1 for item in results if item.success)
    print(f"Passed: {success_count}/{len(results)}")
    print(f"Total elapsed: {total_elapsed_s:.2f}s")
    print("======================================")


def print_failures(results: list[RunResult]) -> None:
    failed = [item for item in results if not item.success]
    if not failed:
        return
    print("\n========== Failure Details ==========")
    for item in failed:
        print(f"\n[{item.name}] exit={item.exit_code}")
        print(item.output[-3000:] if item.output else "(no output)")
    print("==============================")


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    scripts = [
        current_dir / "step17_phase5.py",
        current_dir / "step19_phase5.py",
        current_dir / "step20_phase5.py",
        current_dir / "step21_phase5.py",
        current_dir / "step22_phase5.py",
    ]

    missing = [str(script) for script in scripts if not script.exists()]
    if missing:
        print("[FAIL] Missing scripts, cannot run regression:")
        for path in missing:
            print(f"  - {path}")
        sys.exit(2)

    overall_start = time.perf_counter()
    results: list[RunResult] = []

    for script in scripts:
        print(f"\n[RUN] {script.name} ...")
        result = run_script(script)
        results.append(result)
        print(result.output if result.output else "(no output)")

    total_elapsed = time.perf_counter() - overall_start
    print_summary(results, total_elapsed)
    print_failures(results)

    if all(item.success for item in results):
        print("\n[DONE] Phase 5 regression passed")
        sys.exit(0)

    print("\n[FAIL] Phase 5 regression failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
