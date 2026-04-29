# 架构深度解析

> 面试准备文档 · Hermes Agent CN v0.1.0

---

## 1. 系统总览

---

## 2. 关键设计决策 Q&A

### Q1：为什么用分层规划而不是直接ReAct？

**面试标准答案：**

> 直接ReAct的问题是Agent容易"只见树木不见森林"。
> 遇到复杂任务（比如创建完整的Web应用），
> 没有全局规划的Agent会在第一个工具调用就迷失方向。
>
> 分层规划的好处：
> 1. 战略层（qwen-max）负责目标分解，生成4-8个子目标
> 2. 战术层（qwen-plus）为每个子目标生成具体执行路径
> 3. 执行层只需要关注当前子目标，不被全局复杂度干扰
>
> 类比：这就像项目管理中的WBS（工作分解结构）——
> 先确定里程碑，再细化每个里程碑的任务，
> 而不是直接列出几百个零散任务。

---

### Q2：ChromaDB vs FAISS，你为什么选ChromaDB？

**面试标准答案：**

> 关键区别在于ChromaDB是完整的向量**数据库**，
> FAISS只是向量**索引算法库**。
>
> 对于Agent记忆场景：
> - 需要持久化 → ChromaDB自动处理，FAISS需要手动序列化
> - 需要元数据过滤（"查找最近7天的代码相关记忆"）
>   → ChromaDB原生where子句，FAISS需要自己实现
> - 需要CRUD（更新访问次数）→ ChromaDB支持，FAISS不支持
>
> 我们的数据量级是万级，ChromaDB的HNSW索引完全够用。
> FAISS的优势在于亿级数据量的极致性能，
> 那个场景我们不需要。
>
> 踩坑：Windows上需要显式调用close()并gc.collect()，
> 否则hnswlib的C++析构函数不会立即释放文件句柄。

---

### Q3：事件总线有什么用？为什么不直接调用？

**面试标准答案：**

> 事件总线解决的是**组件间耦合**问题。
>
> 如果Executor直接调用Tracer：
> - 要测试Executor，必须Mock Tracer
> - 要新增Dashboard功能，必须修改Executor
>
> 用事件总线：
> - Executor只发布`TOOL_CALLED`事件，不知道谁在监听
> - Tracer订阅事件，完全独立于Executor
> - 新增Dashboard：只需订阅事件，不修改任何现有代码
>
> 这是Observer模式，也是微服务事件驱动架构的简化版。
> Python的logging模块用的是同样的思路（Handler机制）。

---

### Q4：4层记忆的设计依据是什么？

**面试标准答案：**

> 参考了认知科学中的人类记忆模型：
>
> | 层级 | 类比人类记忆 | 实现 | 特点 |
> |------|------------|------|------|
> | 工作记忆 | 短期记忆 | messages列表 | 快速，受Token限制 |
> | 情景记忆 | 事件记忆 | SQLite | "上次讨论了什么？" |
> | 语义记忆 | 知识记忆 | ChromaDB | "Python GIL是什么？" |
> | 程序性记忆 | 技能记忆 | JSON文件 | "如何创建Flask应用？" |
>
> 设计原则：不同时效和用途的信息用不同存储介质，
> 避免把所有记忆都塞进上下文窗口（那会超Token限制）。

---

### Q5：状态机的价值是什么？

**面试标准答案：**

> 状态机让Agent的行为变得**可预测和可调试**。
>
> 没有状态机：代码里散落着各种if/else，
> 出bug时很难知道"Agent现在处于什么阶段"。
>
> 有了状态机：
> 1. 任意时刻可以查询`state_machine.state`
> 2. 非法转换直接抛异常（比如IDLE不能跳到REFLECTING）
> 3. 状态变更自动发布事件，Observer可以追踪完整历史
>
> 实际价值：在debug时可以说
> "Agent在第3次迭代时从EXECUTING进入ERROR状态，
>  原因是shell工具超时"——这比看日志堆栈效率高10倍。

---

## 3. 数据流图

---

## 4. Phase 2 新增组件（对齐 Hermes 核心工程特性）

### 4.1 `edit_file` 行级编辑工具

- 背景：`write_file` 是整文件覆盖，大文件修改成本高且风险大。
- 方案：新增 `edit_file`，支持 `replace/insert/delete` 三种行级操作。
- 价值：让 Agent 可以像补丁一样精确修改，减少 token 与误改概率。

### 4.2 `ParallelExecutor` 并发工具执行

- 背景：原版 Hermes 通过线程池并发执行独立工具调用，明显提升吞吐。
- 方案：新增并发执行器，白名单内只读工具（`read_file`、`grep_search`、`list_dir`）可并发，其它调用强制串行。
- 关键点：执行结果按原始调用顺序回放，保证消息历史稳定，不破坏 ReAct 协议。

### 4.3 `ContextCompressor` 上下文压缩

- 背景：长任务中，简单“截最近 N 条”会丢掉中段关键信息，导致 Agent 失忆。
- 方案：当估算 token 超过上下文上限的 60% 时，保留头部 system + 最近 5 条，把中段历史交给 FAST 模型做有损摘要。
- 价值：在控制上下文体积的同时，尽可能保留执行链路中的关键决策信息。

### 4.4 权限审批与费用追踪完善

- `PermissionManager`：支持 ALLOW / DENY / ASK，用户可记住命令决策并持久化到 `~/.hermes-cn/permissions.json`。
- `CostTracker`：订阅 `LLM_RESPONSE` 事件，累计会话与按模型费用，在 CLI 中可直接查看。

### 4.5 面试表达要点（建议背诵）

- 我们保留了“分层规划 + 反思”的架构优势，同时在执行层补齐 Hermes 的工程关键能力：并发执行与上下文压缩。
- 并发策略采用“安全白名单 + 顺序回放”，优先保证正确性，再逐步扩展可并发工具集合。
- 上下文压缩采用“头尾保真、中段有损”策略，是对纯截断窗口策略的实用升级。

---

## 5. Phase 3 P1：Policy Engine（权限策略引擎）

### 5.1 为什么要从 PermissionManager 升级到 Policy Engine

- 问题：前缀记忆（always_allow/always_deny）只能表达“命令字符串偏好”，缺少规则化语义。
- 升级目标：把命令审批从“记忆偏好”升级为“可审计策略”。
- 关键能力：支持 `command_regex`、`path_scope`、`risk`、`reason`。

### 5.2 规则优先级（实现口径）

1. 内置高危兜底规则（如 `rm -rf /`、`dd if=`，不可覆盖）
2. 用户级规则：`~/.hermes-cn/policies.json`
3. 项目级规则：`.hermes/policies.json`
4. 若策略未命中，再走 PermissionManager 的学习偏好与交互确认

### 5.3 面试表达要点

- 我们把“是否允许”从交互选择升级成可复用规则，团队可以在仓库级统一策略。
- `path_scope` 让同一条命令在不同目录表现不同，降低“一刀切”策略误伤。
- 风险等级字段可作为未来 CI/非交互模式策略扩展点（例如 high-risk 默认 deny）。

---

## 6. Phase 4 P0：Closed Learning Loop（技能闭环 + 跨会话召回）

### 6.1 自动技能蒸馏（Skill Distiller）

- 背景：原有技能库依赖手工维护，复杂任务里的“有效工具链”难以沉淀。
- 方案：新增 `SkillDistiller` 订阅 `AGENT_COMPLETED` 事件，从成功轨迹中提炼技能草稿（含工具链、失败提示、标签）。
- 实现细节：
  - 默认阈值：最少工具调用数 `min_tool_calls=3`
  - 草稿池：内存保留最近 20 条，避免噪音无限增长
  - 命名策略：`auto_<goal_slug>_<mmdd>`，便于检索与去重
- 操作闭环：
  - `/skills suggest` 查看候选草稿
  - `/skills adopt <index>` 采纳草稿并落盘到程序性记忆

### 6.2 跨会话检索（Recall）

- 背景：长期使用后，会话数量增长，单靠“最近历史”无法满足复盘需求。
- 方案：在 `MemoryManager` 中加入 Session 全文检索索引。
- 检索路径：
  1. 优先 FTS5（高质量中文关键词检索）
  2. FTS 不可用时自动回退 LIKE（兼容环境）
- CLI 接口：`/recall <query>`，展示会话 ID、标题、目标、更新时间，支持快速定位旧任务。

### 6.3 平台兼容性：Windows 文件句柄释放

- 现象：`TemporaryDirectory` 清理时，`chroma.sqlite3` 可能因句柄释放延迟触发 `WinError 32`。
- 处理原则：
  - 测试用例中显式 `mm.close()` 后执行 `del + gc.collect()`
  - 对临时目录清理启用 `ignore_cleanup_errors`（可用时）
- 价值：避免“功能正确但脚本清理失败”导致的伪失败。

### 6.4 验证脚本矩阵

- `verify/step10_phase4.py`：技能蒸馏草稿生成、阈值过滤、草稿采纳落盘。
- `verify/step11_phase4.py`：会话检索 FTS 主路径 + LIKE 回退路径。
- `verify/step13_phase4.py`：草稿持久化重启恢复、采纳后草稿池清理。
- `verify/step14_phase4.py`：草稿质量评分区分与自动采纳阈值验证。
- `verify/step15_phase4.py`：自动采纳审计日志写入、最近自动采纳回滚验证。
- `verify/step16_phase4.py`：按 `record_id` 精确回滚指定自动采纳记录。
- `verify/step12_phase4.py`：Phase 4 总回归入口，统一串行执行并汇总结果。

### 6.5 面试表达要点

- Phase 4 的核心不是“再加工具”，而是把经验沉淀与历史召回打通，形成可持续学习闭环。
- 我们采用“先草稿、后采纳”的安全策略，避免自动写入低质量技能污染记忆层。
- 跨会话检索采用“双通道”（FTS + LIKE fallback），优先质量、兼顾跨平台鲁棒性。

### 6.6 P1 增量：草稿质量评分与自动采纳阈值

- 质量评分：`SkillDistiller` 对每条草稿输出 `quality_score`（0~1），并给出 `recommended` 标记。
- 打分信号：工具调用规模、工具链多样性、读写链路完整性、失败次数惩罚。
- 自动采纳：支持 `AGENT_SKILL_AUTO_ADOPT_THRESHOLD`（0~1），当草稿分数达到阈值时自动转正式技能并从草稿池移除。
- 可控性：阈值默认 `0`（关闭），避免未评估环境中出现“过度自动化”。

### 6.7 P1 增量：自动采纳审计与回滚

- 审计日志：每次草稿采纳会记录 `source(auto/manual)`、技能名、质量分、草稿快照、时间戳。
- 持久化文件：`skill_adoption_log.json`（默认位于数据目录）。
- 回滚能力：支持“最近一次自动采纳”回滚，删除技能并恢复草稿，记录 `rolled_back=true` 与回滚时间。
- Operator UX：CLI 新增 `/skills log` 与 `/skills rollback`，便于现场运维和风控处置。

### 6.8 P1 增量：指定记录精确回滚

- 问题：仅支持“回滚最近一次”在多条自动采纳并行出现时不够精确。
- 升级：`/skills rollback <record_id>` 支持按日志记录 ID 定向回滚。
- 流程：先通过 `/skills log` 查看最近记录，再指定 ID 精确撤销目标采纳，不影响其它记录。

---

## 7. Phase 5 P0：Compression Parity Upgrade（鲁棒压缩对齐）

### 7.1 目标

- 对齐原版 Hermes 在长会话压缩上的“稳定优先”策略，降低压缩抖动和失败连锁影响。

### 7.2 本阶段已落地能力

- `ContextCompressorV2` 新增 anti-thrashing 机制：
  - 连续多次压缩收益低于阈值时，暂时跳过压缩，避免“越压越抖”。
- 新增 summary failure cooldown：
  - 摘要模型失败后进入冷却窗口，窗口期走 fallback 摘要，避免重复失败风暴。
- 新增 focus-topic 压缩入口：
  - 支持传入 `focus_topic`，在 pinned facts 与摘要提示中优先保留该主题信息。
- 新增健康指标接口：
  - `get_health_metrics()` 输出压缩收益、低收益计数、冷却状态、最近错误等可观测信号。

### 7.3 验证脚本

- `verify/step17_phase5.py`：
  - 验证 anti-thrashing 跳过触发；
  - 验证 summary 失败后冷却期不重复调用 LLM；
  - 验证 focus-topic 信息注入到摘要流程。
- `verify/step19_phase5.py`：
  - 验证并发批执行异常时自动降级串行；
  - 验证超时分类重试与权限拒绝不重试；
  - 验证非标准工具返回值被规范化为可消费的 `ToolResult`。
- `verify/step20_phase5.py`：
  - 验证技能采纳前语义去重（避免同类自动技能重复入库）；
  - 验证低价值自动技能按衰减分数递减并淘汰。
- `verify/step21_phase5.py`：
  - 验证治理状态指标接口；
  - 验证手动治理运行结果指标（扫描/更新/移除）。
- `verify/step22_phase5.py`：
  - 验证跨会话检索融合排序（词法 + 语义 + 新鲜度）；
  - 验证词法未命中时的语义兜底召回。
- `verify/step18_phase5.py`：
  - Phase 5 当前总回归入口。

### 7.4 P1 增量：学习闭环治理 v2

- 语义去重：草稿采纳前先与现有技能做签名相似度比对，命中阈值则复用已有技能而不重复保存。
- 低价值衰减：对“低质量 + 低使用 + 超过最小年龄”的自动技能执行衰减分递减，归零后自动清理。
- 审计可追踪：去重采纳会记录 `deduplicated/deduplicated_to` 字段，便于后续运营追溯。

### 7.5 P1 增量：Operator UX 治理面板化

- `/skills govern status`：展示治理面状态（自动技能数量、低质量数量、阈值配置、去重/回滚计数）。
- `/skills govern run`：手动触发一次治理，并反馈本次扫描、更新、移除数量。
- `/skills log` 增加“去重”列，直接识别哪些采纳属于语义复用而非新增。

### 7.6 P2 增量：跨会话检索融合排序

- 候选召回：FTS5 优先、LIKE 回退；若词法全空则回退最近会话作为语义候选池。
- 融合重排：`0.60 * lexical + 0.30 * semantic + 0.10 * recency`。
- 可解释命中：输出 `match_reason`（如 `title-match`、`goal-match`、`semantic-strong`）。

---

## 8. Phase 6 P0：Recall Evaluation Baseline（检索评估基线）

### 8.1 目标

- 把“检索效果好不好”从主观判断升级为可重复评估指标。

### 8.2 基线脚本

- `verify/step23_phase6.py`：
  - 读取外置评估集 `verify/data_phase6_recall_cases.json`；
  - 计算 `recall@1`、`recall@3`、`MRR`；
  - 统计命中解释覆盖率（`match_reason`）；
  - 导出指标报告 `verify/reports/phase6_recall_eval_report.json`。
- `verify/step24_phase6.py`：
  - Phase 6 总回归入口。
  
### 8.4 P2 增量：指标对比与退化闸门

- `verify/step25_phase6.py`：
  - 将当前评估报告与 baseline 报告做 diff；
  - 当指标退化超过阈值（默认 0.01）时返回非 0，作为回归闸门；
  - 若 baseline 不存在，首次运行会自动用 current 生成 baseline。

### 9.1 Phase 7 P0：Recall 质量迭代闭环（权重可配置 + 扫参）

- `src/memory/memory_manager.py`：
  - `search_sessions()` 的 hybrid 权重支持环境变量配置：
    - `AGENT_RECALL_WEIGHT_LEXICAL`（默认 0.60）
    - `AGENT_RECALL_WEIGHT_SEMANTIC`（默认 0.30）
    - `AGENT_RECALL_WEIGHT_RECENCY`（默认 0.10）
  - 输出里附带 `score_breakdown`（分项分数与权重），便于离线诊断与调参。
- `verify/step26_phase7.py`：
  - 基于 `verify/data_phase6_recall_cases.json` 进行权重扫参；
  - 导出报告 `verify/reports/phase7_recall_weight_sweep.json`，输出最佳权重与指标。

### 9.2 Phase 7 P1：扫参报告退化闸门（多版本对比）

- `verify/step27_phase7.py`：
  - 对比 current 与 baseline 的扫参报告（取 `best` 项）；
  - 若 best_score 或关键指标退化超过阈值则失败（回归闸门）。
- `verify/step28_phase7.py`：
  - Phase 7 总回归入口（先扫参，再做对比闸门）。

### 10.1 Phase 8 P0：检索可用性指标（功能型指标）

- `verify/step29_phase8.py`：
  - 基于 Phase 6 数据集计算检索可用性指标：
    - `useful@1/useful@3`：top-k 内是否存在“可用命中”（基于 `match_reason` 标签）
    - `component_diversity@3`：top-3 的主导分数组件（lexical/semantic/recency）多样性
    - `stability@3`：重复检索 top-3 是否稳定一致
  - 导出报告 `verify/reports/phase8_recall_usability_report.json`
- `verify/step30_phase8.py`：
  - 对比 current 与 baseline 可用性报告，作为退化闸门。
- `verify/step31_phase8.py`：
  - Phase 8 总回归入口（指标计算 + diff 闸门）。

### 10.2 Phase 8 P2：Top-k 多样化重排（MMR rerank）

- `src/memory/memory_manager.py`：
  - 在 `search_sessions()` 的最终排序阶段支持可选的 top-k 多样化重排（MMR 风格）；
  - 目标是在不明显牺牲相关性的情况下提升 `component_diversity@k`；
  - 通过环境变量控制（默认关闭，保持历史行为）：
    - `AGENT_RECALL_DIVERSIFY_TOPK=1` 开启
    - `AGENT_RECALL_DIVERSIFY_LAMBDA`（默认 0.88，越大越偏相关性）

### 11.1 Phase 9 P0：线上可观测（/recall explain + 召回健康度）

- `src/observability/recall_logger.py`：
  - 以 JSONL 形式落盘记录 `/recall` 事件（query、top_ids、useful@k、diversity、权重等）；
  - 默认路径：`~/.hermes-cn/recall/recall_logs.jsonl`（遵循 `AGENT_DATA_DIR`）。
- `src/main.py`：
  - `/recall <query> --explain`：输出权重与可用性摘要，便于线上对齐离线口径；
  - `/recall health [N]`：汇总最近 N 次 recall 的 useful@3 与 diversity@3，并按阈值提示告警。

### 11.2 Phase 9 P1-1：闸门化（线上日志 → 离线报告 → diff 回归）

- `verify/step32_phase9.py`：
  - 读取 `recall_logs.jsonl`（支持 `--log-path`），聚合最近 N 条 recall 事件；
  - 导出 `verify/reports/phase9_recall_health_report.json`（均值 + 分位数）。
- `verify/step33_phase9.py`：
  - 对比 current 与 baseline 健康报告，退化超过阈值则失败（回归闸门）。
- `verify/step34_phase9.py`：
  - Phase 9 总回归入口（先产 report，再做 diff 闸门）。

### 12.1 Phase 10 P0：Recall 引用注入（把检索变成“可用上下文”）

- `src/agent_loop.py`：
  - 在战略规划前可选注入“跨会话检索（past sessions）”结果到 `full_context`；
  - 开关（默认关闭，避免影响历史行为）：
    - `AGENT_RECALL_INJECT_SESSIONS=1` 开启
    - `AGENT_RECALL_INJECT_SESSIONS_K` 控制注入条数（默认 3，范围 1-8）
- `src/memory/memory_manager.py`：
  - 新增 `format_sessions_for_context()`：用 `search_sessions()` 找相似会话，并带上少量 `messages` excerpt，便于规划直接复用历史经验。

### 8.3 面试表达要点

- 我们不仅实现了融合排序，还加入了离线质量基线，后续每次算法调整都能做回归对比。
- 指标体系里同时包含排序质量（MRR）与可解释性覆盖率，兼顾效果和运维可观测性。
