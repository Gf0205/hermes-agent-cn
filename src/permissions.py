"""
src/permissions.py - 命令权限审批管理
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.security.policy_engine import PolicyDecision, PolicyEngine


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionState:
    always_allow: set[str]
    always_deny: set[str]


class PermissionManager:
    """维护命令审批规则，并支持交互式学习用户偏好"""

    def __init__(
        self,
        config_path: Path | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self._path = config_path or (Path.home() / ".hermes-cn" / "permissions.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        self._policy_engine = policy_engine or PolicyEngine(project_root=Path.cwd())

    def check(self, command: str, cwd: str | None = None) -> PermissionDecision:
        decision = self.classify(command, cwd=cwd)
        if decision != PermissionDecision.ASK:
            return decision
        return self._ask_user(command)

    def classify(self, command: str, cwd: str | None = None) -> PermissionDecision:
        normalized = self._normalize(command)
        policy_match = self._policy_engine.evaluate(command, cwd=cwd)
        if policy_match is not None:
            if policy_match.decision == PolicyDecision.DENY:
                return PermissionDecision.DENY
            if policy_match.decision == PolicyDecision.ALLOW:
                return PermissionDecision.ALLOW
            return PermissionDecision.ASK

        hard_deny = {
            "rm -rf /",
            "rm -rf ~",
            "mkfs",
            "dd if=",
            "chmod -r 777 /",
        }
        if self._matches_prefix(normalized, hard_deny):
            return PermissionDecision.DENY
        if self._matches_prefix(normalized, self._state.always_deny):
            return PermissionDecision.DENY
        if self._matches_prefix(normalized, self._state.always_allow):
            return PermissionDecision.ALLOW
        return PermissionDecision.ASK

    def _ask_user(self, command: str) -> PermissionDecision:
        print(f"\n[权限审批] 即将执行命令：{command}")
        try:
            allow = input("允许执行？[y/N]: ").strip().lower() in {"y", "yes"}
            remember = input("记住这次选择？[y/N]: ").strip().lower() in {"y", "yes"}
        except EOFError:
            return PermissionDecision.DENY

        normalized = self._normalize(command)
        if allow:
            if remember:
                self._state.always_allow.add(normalized)
                self._save()
            return PermissionDecision.ALLOW

        if remember:
            self._state.always_deny.add(normalized)
            self._save()
        return PermissionDecision.DENY

    def _load(self) -> PermissionState:
        if not self._path.exists():
            return PermissionState(always_allow=set(), always_deny=set())
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return PermissionState(always_allow=set(), always_deny=set())

        return PermissionState(
            always_allow=set(data.get("always_allow", [])),
            always_deny=set(data.get("always_deny", [])),
        )

    def _save(self) -> None:
        payload = {
            "always_allow": sorted(self._state.always_allow),
            "always_deny": sorted(self._state.always_deny),
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _matches_prefix(self, command: str, prefixes: set[str]) -> bool:
        return any(command.startswith(prefix) for prefix in prefixes)

    def _normalize(self, command: str) -> str:
        return command.strip().lower()
