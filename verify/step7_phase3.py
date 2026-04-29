"""
Phase 3 P1 验证脚本（Policy Engine）
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.permissions import PermissionDecision, PermissionManager
from src.security.policy_engine import PolicyDecision, PolicyEngine


def test_policy_builtin_high_risk_deny() -> None:
    engine = PolicyEngine(project_root=Path.cwd())
    result = engine.evaluate("rm -rf /")
    assert result is not None
    assert result.decision == PolicyDecision.DENY
    assert result.risk == "high"


def test_project_policy_allow_with_path_scope() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        policy_dir = root / ".hermes"
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "policies.json").write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "name": "allow-git-status",
                            "command_regex": r"^git status$",
                            "decision": "allow",
                            "risk": "low",
                            "path_scope": str(root),
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        engine = PolicyEngine(project_root=root)
        allow_hit = engine.evaluate("git status", cwd=str(root))
        deny_hit = engine.evaluate("git status", cwd=str(Path.home()))
        assert allow_hit is not None
        assert allow_hit.decision == PolicyDecision.ALLOW
        assert deny_hit is None


def test_builtin_high_risk_cannot_be_overridden() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        policy_dir = root / ".hermes"
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "policies.json").write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "name": "unsafe-allow-rm-root",
                            "command_regex": r"(^|\s)rm\s+-rf\s+(/|~)(\s|$)",
                            "decision": "allow",
                            "risk": "low",
                        }
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        engine = PolicyEngine(project_root=root)
        result = engine.evaluate("rm -rf /", cwd=str(root))
        assert result is not None
        assert result.decision == PolicyDecision.DENY
        assert result.rule_name == "deny-destroy-root"


def test_permission_manager_obeys_policy_deny() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = PolicyEngine(project_root=root)
        manager = PermissionManager(
            config_path=root / "permissions.json",
            policy_engine=engine,
        )
        decision = manager.classify("rm -rf /", cwd=str(root))
        assert decision == PermissionDecision.DENY


def main() -> None:
    test_policy_builtin_high_risk_deny()
    print("[PASS] test_policy_builtin_high_risk_deny")
    test_project_policy_allow_with_path_scope()
    print("[PASS] test_project_policy_allow_with_path_scope")
    test_builtin_high_risk_cannot_be_overridden()
    print("[PASS] test_builtin_high_risk_cannot_be_overridden")
    test_permission_manager_obeys_policy_deny()
    print("[PASS] test_permission_manager_obeys_policy_deny")
    print("[DONE] Phase 3 P1 Policy Engine 验证通过")


if __name__ == "__main__":
    main()
