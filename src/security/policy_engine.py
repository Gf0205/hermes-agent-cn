"""
src/security/policy_engine.py - 命令策略引擎
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PolicyMatch:
    decision: PolicyDecision
    risk: str = "medium"
    reason: str = ""
    rule_name: str = ""


class PolicyEngine:
    """
    规则来源：
    - 项目规则：<project>/.hermes/policies.json
    - 用户规则：~/.hermes-cn/policies.json
    """

    def __init__(
        self,
        project_root: Path | None = None,
        user_policy_path: Path | None = None,
        project_policy_path: Path | None = None,
    ) -> None:
        self._project_root = (project_root or Path.cwd()).resolve()
        self._user_policy_path = user_policy_path or (Path.home() / ".hermes-cn" / "policies.json")
        self._project_policy_path = (
            project_policy_path or (self._project_root / ".hermes" / "policies.json")
        )

        self._builtin_rules = [
            {
                "name": "deny-destroy-root",
                "command_regex": r"(^|\s)rm\s+-rf\s+(/|~)(\s|$)",
                "decision": "deny",
                "risk": "high",
                "reason": "潜在破坏性删除命令",
            },
            {
                "name": "deny-disk-overwrite",
                "command_regex": r"\bdd\s+if=",
                "decision": "deny",
                "risk": "high",
                "reason": "潜在磁盘覆盖命令",
            },
        ]

    def evaluate(self, command: str, cwd: str | None = None) -> PolicyMatch | None:
        normalized = command.strip().lower()
        current_dir = Path(cwd).resolve() if cwd else self._project_root

        rules = [
            *self._builtin_rules,
            *self._load_rules(self._user_policy_path),
            *self._load_rules(self._project_policy_path),
        ]

        for rule in rules:
            if self._rule_matches(rule, normalized, current_dir):
                decision_raw = str(rule.get("decision", "ask")).lower()
                decision = self._to_decision(decision_raw)
                if decision is None:
                    continue
                return PolicyMatch(
                    decision=decision,
                    risk=str(rule.get("risk", "medium")),
                    reason=str(rule.get("reason", "")),
                    rule_name=str(rule.get("name", "")),
                )
        return None

    def _rule_matches(self, rule: dict[str, Any], command: str, cwd: Path) -> bool:
        regex = str(rule.get("command_regex", "")).strip()
        if regex:
            try:
                if not re.search(regex, command, re.IGNORECASE):
                    return False
            except re.error:
                return False

        path_scope = str(rule.get("path_scope", "")).strip()
        if path_scope:
            target = Path(path_scope).expanduser().resolve()
            try:
                cwd.relative_to(target)
            except ValueError:
                return False
        return True

    def _load_rules(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                rules = data.get("rules", [])
            else:
                rules = data
            return [item for item in rules if isinstance(item, dict)]
        except Exception:
            return []

    def _to_decision(self, raw: str) -> PolicyDecision | None:
        if raw == "allow":
            return PolicyDecision.ALLOW
        if raw == "deny":
            return PolicyDecision.DENY
        if raw == "ask":
            return PolicyDecision.ASK
        return None
