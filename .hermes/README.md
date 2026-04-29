# Policy Engine 配置说明

项目级策略文件路径：`.hermes/policies.json`  
如果你想快速开始，可以复制模板：

```bash
cp .hermes/policies.example.json .hermes/policies.json
```

规则格式示例：

```json
{
  "rules": [
    {
      "name": "allow-git-readonly",
      "command_regex": "^git (status|diff|log).*$",
      "decision": "allow",
      "risk": "low",
      "reason": "允许常见只读 git 命令"
    }
  ]
}
```

字段说明：

- `name`: 规则名（便于日志与排查）
- `command_regex`: 命令匹配正则
- `decision`: `allow` / `deny` / `ask`
- `risk`: `low` / `medium` / `high`
- `reason`: 决策原因
- `path_scope`（可选）: 规则生效目录范围

优先级说明：

1. 内置高危规则（不可覆盖）
2. 用户级规则 `~/.hermes-cn/policies.json`
3. 项目级规则 `.hermes/policies.json`
