# Agent Memory Service

独立的长短期记忆后端，配套相邻 `memory-ui` 展示前端。开发分支为 `full-forward`。系统支持证据约束的自动抽取、跨会话记忆、事实更正与历史版本、混合检索、来源摘要和精确上下文推理复用。

当前面向单节点交付。真实课程 Engine 集成不在本轮推进范围；保留的适配器及契约测试不能替代真实宿主验收。公开基准、完整语义忠实度和生产容量也不能由本地回归测试替代。

2026-11-16 的错题复核与修复记录见 [优化审计](../../result/optimization_20261116/REPORT.md)。
该阶段保留单一答案链，修复列表证据打包、规划日期传递及精确标识符校验，并改进已有修复反馈。
当时有退步的生成提示修改已撤回；开发回归与专项修复验证分开报告，不宣称所有语义错误已解决。

2026-11-17 的机械修复及全部负结果见 [结构修复审计](../../result/structural_repairs_20261117/README.md)。
该阶段保留最终查询计划向已有检索消费者传递、窗口与状态提交修复，以及答案格式和失败反馈修复；
同一来源下的71个首次生成请求保持原样，440条历史响应回放保留全部408条原成功结果。
当时的 indexed semantics 补传与修改生成 Schema 的集成候选未通过正常规划回归，没有合入。
这些检查不代表整体语义质量提升或已达到论文验收要求。

- [架构](ARCHITECTURE.md) · [实现框架与模型调用](IMPLEMENTATION_FRAMEWORK.md) · [生命周期机制](docs/LIFECYCLE_MECHANISMS.md)
- [课程需求](docs/COURSE_REQUIREMENTS.md) · [验证记录](docs/VERIFICATION.md) · [交付入口](../DELIVERY.md)
- [技术报告](docs/技术报告.md) · [演示脚本](docs/答辩演示脚本.md) · [部署、备份与数据治理](OPERATIONS.md)

## 当前机制

| 部分 | 具体方案 |
|---|---|
| 短期记忆 | 原始事件、任务状态、摘要、结构化事实；每批最多 5 个新 turn、最多 15 个已处理上下文 turn |
| 自动准备 | answer 默认处理待抽取积压，search 保持只读；手动 extract 仍可使用 |
| 证据规则 | 原文位置和引文校验；问句不能成为肯定事实，助手猜测不能污染长期记忆 |
| 长期演化 | 自动安全晋升、同事实证据合并、现实更新、错误纠正、冲突待决、归档与恢复 |
| 一致性 | SQLite WAL、事务、请求幂等、事实版本、状态 revision、审计；演化 outbox 支持失败重试和重启恢复 |
| 检索 | 模型意图规划、短期优先、按问题类型选择词法/符号/语义通道、有条件并发、两跳显式关系证据扩展、动态预算 |
| 直接问答 | 原文证据合并后统一调用 `source_bound_synthesis`，返回带引用的 `AnswerTable`，校验、渲染并保存回执；集合检索不另开答案出口 |
| 分层维护 | Domain/Category/Trace 来源摘要、可选模型概括、缓存失效、年龄与证据保留策略 |
| 动态展示 | NDJSON 真实阶段事件，返回规划、候选、选择、来源和缓存观测；不是隐藏思维链或逐 token 流 |
| 方案生成 | 方案类请求显式形成问题拆分、所需信息、信息整合步骤和固定 Markdown 行动要素模板；按事实填充、情景假设、关键缺口需确认三层生成 |
| 推理复用 | legacy plain-answer adapter 保留精确上下文 TTL/LRU 缓存及在途调用合并；所有回答返回前复验版本和有效期 |

检索候选按记忆身份合并并保留通道排名，原文索引先还原完整阅读单元，在同一预算内优先覆盖已有请求中的显式成员或目录字段，再读未知绑定和重复覆盖的来源；每组沿用最佳上游通道排名。目录字段仅帮助定位，不作为事实真值或拒答条件。选择成本与生成端共享来源序列化方式；局部词面匹配负责原文定位，不另加最终词面评分，原文引用的角色、时间、权限和精确位置始终保留。

SQLite 保存权威事实；FTS、向量和摘要均为可重建投影。当前仍读取授权 owner 的权威快照，再使用索引筛选候选，没有实现 ANN 或分布式检索。未配置向量服务时明确使用 `lexical_baseline`。

直接问答只保留一条结构化答案链：检索与原文打包 → 一次 synthesis 请求 → 来源及输出契约校验 → 内容保留的渲染。最多允许一次同请求的格式/证据修复；错误保持可见，不切换 reader、集合执行器或第二条推理链。`AnswerSlot.derivation` 是同次调用返回的必填、至多 1000 字符建模摘要，保存供审计，不进入最终答案，也不是额外推理链。当前 concise 提示要求 `values` 给出最短完整答案，保留必要限定、单位和全部所问条目，解释放入 `derivation`；这不是发布后截断。发布器不执行数值 DSL，也不根据日期或问题硬改写模型答案。`temporal.py` 的 `resolve_time_references` 为生成输入统一解析日期、相对时间与精度；观察时间不自动等于文中事件时间。`plan.as_of` 仅约束事实有效期；给定 history 中的原文按权限和生命周期使用，允许后续话语追述早先事实。`Question date` 是相对时间参考，不是知识访问权限，发生或参考时钟都不用于裁掉已给定来源。当前不实现通用 as-known-at 查询，也不保留未使用的观测截止开关或参数。引文匹配证明来源（provenance），不证明答案语义被证据蕴含（entailment）。未实现 `synthesize` 的旧模型适配器及行动方案保留 plain-answer 接口，详见[架构边界](ARCHITECTURE.md)。

三路召回采用通用的 FTS5/BM25、主体属性精确匹配、向量余弦相似度。执行策略复用已有查询计划，不增加模型分类调用或依赖：明确单字段查值且有精确记录时省去向量检索；开放、比较、多信息项或时间问题按可用条件并发召回；字段未命中时恢复宽范围检索。纯本地通道合并到一次线程调度，短期与长期仍按证据充分性先后执行。跳过原因、执行方式和通道耗时随结果返回。具体规则和实测边界见 [条件检索说明](docs/CONDITIONAL_RETRIEVAL.md)。

答案文本的生成要求与接收校验分开：provider 保留既有的700字符简洁生成目标，使首次请求兼容；
接收端不再仅因单个 `value` 超过700字符就丢弃已有支持的回答，也不截断内容。
公开 `AnswerTable` Schema 如实反映接收范围；引文长度、来源、类型、算术和全局模型输出预算仍有效。
宿主提供的 `CallableModel` 仍需由宿主限制生成输出。普通 `operation=none` 的引用合并后按同一规则检查，
显式 `count/sum` 仍逐操作数检查。既有一次修复可以携带多个独立可计算错误及错位引用的迁移说明，
没有增加模型调用或备用答案链；它也不能证明集合完整或任意自然语言推断正确。

## 安装与启动

```powershell
# 从 SimpleMem 根目录执行，使用既有 Conda 环境。
cd memory-system
conda run --no-capture-output -n simplemem-agentmemory python -m pip install -e ".[dev]"
conda run --no-capture-output -n simplemem-agentmemory python -m agent_memory
```

后端默认地址为 `http://127.0.0.1:8088`，OpenAPI 页面为 `/docs`。在另一个终端启动前端：

```powershell
# 从 SimpleMem 根目录执行。
cd memory-ui
pnpm install --frozen-lockfile
pnpm dev
```

前端默认位于 `http://127.0.0.1:5173`，Vite 将 `/api`、`/health` 代理至后端。后端换端口时，通过前端进程环境变量 `MEMORY_UI_PROXY_TARGET` 指定实际地址。

在私有 `memory-system/.env` 中配置模型：

```dotenv
MEMORY_MODEL_BASE_URL=<OpenAI兼容服务地址>
MEMORY_MODEL_NAME=<账号可用的模型名>
MEMORY_MODEL_API_KEY=<仅保存在后端的密钥>
MEMORY_MODEL_TIMEOUT=90
MEMORY_MODEL_MAX_TOKENS=4096
MEMORY_MAINTENANCE_INTERVAL=300
```

现有进程环境优先于 `.env`，修改配置后需要重启。`GET /health` 返回实际模型、提供商和语义检索启用状态。真实 Embedding 使用独立的 `MEMORY_EMBEDDING_BASE_URL`、`MEMORY_EMBEDDING_NAME`、`MEMORY_EMBEDDING_API_KEY`。模型密钥不得放入前端、Git 或请求日志。

本地 `scripts/qwen_embedding_server.py` 显式使用 eager attention、float32 和 eval。
2026-11-17 在当前 GPU/依赖环境中复现了默认 efficient-SDPA 路径对相同输入产生不同向量，
足以改变语义候选资格；修复保持原模型权重与 query/document 提示。
更换权重、计算后端、精度或角色提示时，必须更新 `MEMORY_EMBEDDING_REVISION` 并重启客户端，
使旧派生向量失效。服务 `/health` 的 `readiness=model_loaded_only` 只表示加载成功；
开始实验前还应实际调用 embeddings，核查维度、有限数值、范数和重复输入的一致性。
完整本地复现及适用环境见 [embedding 根因记录](../../result/structural_repairs_20261117/EMBEDDING_ROOT_CAUSE.md)。

未配置模型时仍可创建会话、记录事件和执行词法查询；回答与抽取明确返回不可用状态。独立 HTTP 服务启动周期维护 worker，默认间隔 300 秒，设为 0 禁用。维护只面向当前配置 principal，不代表多租户后台调度已完成。

默认服务使用固定本地身份。独立生产模式要求 `MEMORY_DEPLOYMENT_MODE=production` 和至少 32 字符的 `MEMORY_API_KEY`，数据接口验证 Bearer；共用服务密钥不构成完整多用户身份系统。

## API 顺序与失败语义

```text
POST /api/v1/sessions
POST /api/v1/sessions/{id}/turns
POST /api/v1/answer                           默认先自动整理待处理记录
POST /api/v1/sessions/{id}/answer/stream       同一业务流程，返回NDJSON阶段事件
POST /api/v1/sessions/{id}/extract             可选手动整理
POST /api/v1/search                           只读检索
GET  /api/v1/memories
POST /api/v1/memories/{id}/evolve
GET  /api/v1/memories/{id}/history
POST /api/v1/maintenance
GET  /api/v1/groups
GET  /api/v1/hierarchy
```

调用方先保存用户输入，再请求回答；需要保留助手输出时另行追加。answer 不隐式写回用户和助手消息，避免重试生成重复事件。turn 使用稳定 `request_id`：同内容重试幂等，同 ID 不同内容返回 409。

抽取模型或证据校验失败不推进水位。抽取已提交而演化失败时，返回演化警告并保留持久化任务，后续调用可恢复；规划或生成失败不会撤销已提交抽取。不能把所有超时都解释为“系统完全未发生写入”。

演化支持 `promote/merge/supersede/correct/retract/defer/archive/restore/activate`。接口要求 `expected_version`；新调用方同时传入 `expected_revision`，目标动作传入目标版本与 revision，以拒绝状态往返后的旧页面操作。归档保留历史和证据；恢复不修改原有效期。`pending` 是未解决候选，不可默认为当前正确事实。

NDJSON 返回真实 `stage` 及最终 `result/error`，包括阶段耗时。当前回答文本生成完成后统一返回，不是逐 token 供应商流。界面展示执行轨迹和证据，不展示模型隐藏思维链。

## 加速的准确口径

历史原文读取使用 `source_session_ids` 指定同一所有者、同项目的会话。`source_read_scope` 默认 `quotes`，只读取合法记忆已经引用的内容；显式传入 `events` 才允许把这些会话内未被抽取的原始发言送入现有原文候选池。普通与流式 HTTP、SDK 的 `bind_request` 使用相同权限语义，撤回和删除仍然生效。完整发言读取可能增加证据竞争，不能仅凭能够读到原文就声称答案质量提升。

结构化重述 `content`、集合成员及合法引文共同参与候选匹配，原始事件保存完整细节。三路召回的内部候选并集传到答案阶段，公开 search/SDK 的 `results` 接口不变，内部 `candidates` 不进入公开 JSON。顶层及成员证据均可支持标量问题；先按原文边界构建段落、列表或指定成员的完整读取单元，再对序列化后的来源做一次统一预算选择，不再按摘要预截断后另行扩写，也不再设置独立 `raw_limit` / `grounding_max_raw_sources` 配额。压缩主题只帮助检索，不作为原文事实发布。`Evidence.start/end` 是原始 Unicode 字符串的半开区间，由程序定位；旧无坐标记录仍可读，重复引文不假定首次出现是唯一位置。当前离线回放检查最终可见来源及位置、角色、输入大小，不代表新答案准确率或 F1；详见[证据读取说明](docs/EVIDENCE_GROUNDING.md)。

answer 接受 `accelerate: true/false`。该生成缓存仅用于 legacy plain-answer adapter；当前结构化 `source_bound_synthesis` 不使用它。复用只避免相同完整上下文的最终生成调用，每次仍执行授权和新鲜查询规划、检索。默认缓存 TTL 为 60 秒、最多 128 项；相同在途请求共享一次生成。状态、来源、上下文变化后不能继续按原条件命中。

前端可在不追加对话事件的前提下比较基线与加速。正常聊天持续改变上下文，命中率不能由重复重跑实验推断。阶段计时、生成调用节省和保守字符预算分别报告，字符数不是模型实际计费 token。当前没有 GPU 算子优化、KV Cache 管理或投机解码。

## 验证

```powershell
# 在 memory-system 目录执行。
conda run --no-capture-output -n simplemem-agentmemory python -m pytest -q
conda run --no-capture-output -n simplemem-agentmemory python -m ruff check src tests scripts
conda run --no-capture-output -n simplemem-agentmemory python -m ruff format --check src tests scripts
```

历史 concise 阶段在固定 LoCoMo-100 上 judge 正确数保持 87/100，F1 从 0.5517 回升至 0.5883，仍低于 before 的 0.6349；这些不是本轮版本成绩。该开发集经过多版本反复观察，不作为独立确认性验证。后续 source_clock 小时截止因不符合官方给定 history 协议而撤回。各阶段结果按冻结源码和实验清单区分，不拼接最佳分数；2026-11-23 本轮离线来源回放不生成新答案，不报告新的 Accuracy/F1。

普通回归测试不调用真实模型服务。真实模型脚本单独执行，其配置、耗时、失败和结果见 [验证记录](docs/VERIFICATION.md)。历史 [真实冒烟](docs/LIVE_SMOKE.md)、[十题开发小集](docs/LIVE_QUALITY.md)和[合成性能记录](docs/EVALUATION.md)保留原实验条件，不自动代表本次版本的全面验收结论。

重点回归覆盖“zfc 是飞猪吗”无证据询问、别名陈述、临时约束、跨会话保存、明确更正、冲突拒猜、长积压、超时、演化恢复、归档恢复、权限隔离和缓存失效。公开长期对话基准、语义忠实度评审和真实生产容量仍需独立实验。
方案生成覆盖通用实施计划和应急行动计划，并向回答生成 prompt 注入“需求拆分 -> 所需信息梳理 -> 信息整合 -> 完整方案”契约。系统优先使用检索事实；明确假设可形成分案；无法由证据推出的关键条件标“需确认”，并给出追问。

## Engine 边界

可选 `agent_memory.integrations.rag_sdk` 支持课程契约的 search/generate/generate_stream，核心不依赖宿主启动。旧 SDK generate_stream 仍为缓冲兼容输出。真实宿主未在本轮集成；现有契约测试与 [SDK 说明](docs/SDK_INTEGRATION.md)仅作为后续接入依据。
