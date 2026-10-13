# MemNexus 与 SimpleMem 公平对比协议

本文档定义通用领域实验的可复现口径。它不把任何一次本地回归或小样本真实模型结果写成优于基线的证据。

## 研究问题

1. 在相同主模型、embedding、数据、问题顺序和答案裁判下，MemNexus 是否提高 LoCoMo 与 LongMemEval-S 的正确率。
2. 在相同质量约束下，MemNexus 是否降低写入调用、检索调用、最终生成调用、输入 token、延迟和费用。
3. 版本演化、作用域隔离、证据约束和缓存是否带来额外收益，代价是什么。

## 统一配置

正式主实验采用以下冻结配置；模型替换是本项目统一施加于两边的实验变量，不与 SimpleMem 论文原始数值混为一谈：

- 主模型：`deepseek-v4-flash`；结构化抽取/压缩温度 `0.0`。
- LongMemEval-S 裁判：`deepseek-v4-flash`，温度 `0.0`，只输出 `CORRECT` 或 `WRONG` JSON 标签。
- embedding：`Qwen/Qwen3-Embedding-0.6B`，1024 维。
- 写入窗口：20 turns，stride 5。SimpleMem 的 `OVERLAP_SIZE` 表示重叠量，因此必须为 15；不要把 stride 5 误写成 `OVERLAP_SIZE=5`。
- 检索深度：$k_{min}=3$，$k_{max}=20$。必须记录每题实际 `k`，不能只记录配置上限。
- SimpleMem 的 LanceDB 版本、MemNexus 的 SQLite/FTS 投影版本、Python/Node 版本、CPU/GPU、并发数全部写入结果 manifest。
- 主实验关闭并行或固定为同一并发策略；吞吐实验另行改变并发，不混入质量表。
- 两个系统使用同一 API endpoint、同一模型名、同一 token 上限、同一 timeout、同一重试次数和同一问题顺序。模型返回的原文、请求耗时、输入/输出 token、错误和费用都要落盘。

密钥只通过环境变量或本地未提交配置注入。`config.paper.py.example` 是 SimpleMem 的无密钥模板；MemNexus 使用 `MEMORY_MODEL_BASE_URL`、`MEMORY_MODEL_NAME`、`MEMORY_MODEL_API_KEY`。主模型、两套系统和裁判共享一个 8 元人民币总上限；不是每个组件各 8 元。价格未知或为 0 时禁止付费调用。

Qwen3 编码固定为方案 B：query 使用模型内置 `prompt_name="query"`，document 使用普通归一化编码。MemNexus 本地 embedding 服务通过 `input_type` 明确区分两种输入；SimpleMem 对查询和文档分别调用 `encode_query` 与 `encode_documents`。任何 query prompt 不可用的情况都必须失败，不能静默退化为 document 编码。

正式英文数据集实验固定 `MEMORY_OUTPUT_LANGUAGE=en`。原始 evidence 保持其原文；结构化自然语言字段和最终回答使用英文。这样可以直接比较 LoCoMo/LongMemEval 的英文 gold answer，并避免把语言差异误记为记忆质量差异。中文产品交互可以单独作为语言消融，不进入主结果。

完整数据集的费用估计和 ICAPS 规划实验设计见工作区的 [FULL_BUDGET_ESTIMATE](../../../result/FULL_BUDGET_ESTIMATE.json) 与 [ICAPS 2027 experiment plan](../../../result/ICAPS_2027_EXPERIMENT_PLAN.md)。估算基于真实英文 pilot usage、官方 turn/window 数量和当前峰值 cache-miss 单价；实际付费执行仍以逐请求 ledger 为准。

## 数据集和指标

### LoCoMo

使用完整公开测试集，不用当前项目的 10 题开发小集作为论文主结果。按 Single-hop、Multi-hop、Temporal、Open-domain 以及 adversarial 类别分层报告：F1、BLEU-1、类别均值、问题数、输入 token、写入/检索/回答延迟和请求费用。SimpleMem 论文报告的参考值是 GPT-4.1-mini 下平均 F1 43.24、平均约 531 input tokens；这些是论文结果，不是本地复现实验结果。

### LongMemEval-S

使用官方评分格式和完整测试集。由同一 `deepseek-v4-flash` judge 对每题输出 `CORRECT/WRONG`，保存 judge prompt、版本、temperature、原始返回和解析结果。报告 overall accuracy 及 Single-Session、Multi-Session、Temporal 等官方子类；同时报告 judge 调用费用和失败率。

### 机制与效率

单独报告以下指标，不能和 QA 正确率合成一个未经声明的“总分”：

- `write_llm_calls`、`planning_llm_calls`、`answer_llm_calls`、`maintenance_llm_calls`。
- MemNexus 每个回答响应中的 `model_calls`：`extract_calls`、`plan_calls`、`answer_calls`、`relation_calls`、`summary_calls` 和 `errors`；缓存命中只统计实际执行的底层生成调用。
- `input_tokens`、`output_tokens`、总 tokens、人民币费用和每个正确答案费用。
- P50/P95/P99 写入、检索、端到端延迟；冷启动和热缓存分开。
- 记忆条目数、压缩比、检索候选数、最终上下文 tokens、Recall@K/MRR/NDCG（仅在有 gold evidence 时）。
- 未知问题拒答、跨会话/项目泄漏、时间更正、旧版本误召回和引用忠实度。

每个系统至少执行 3 个独立重复或固定随机种子下的 bootstrap 置信区间；主表给出 paired bootstrap 或 McNemar 检验、效应量和置信区间。不得只挑选对 MemNexus 有利的样本或重试成功答案。

## 对比矩阵

1. `full-context`：不使用记忆系统，严格截断到同一生成上下文预算。
2. `SimpleMem-paper`：论文配置，保留其压缩、在线合成和意图规划。
3. `MemNexus-general`：使用通用行动方案输出；使用同一主模型和 embedding。
4. `MemNexus-no-cache` 与 `MemNexus-cache`：缓存只作为效率消融，质量问题必须使用 no-cache 主结果。
5. MemNexus 组件消融：关闭模型规划、向量路线、实体扩展、长期演化、证据校验或加速缓存，每次只关闭一个组件。

MemNexus 的行动计划输出统一使用通用字段，不包含领域专用模板；普通 action-plan 请求使用 `general_action_plan`。

## 运行顺序

1. 冻结 commit、配置文件哈希、数据集哈希和环境 lock 信息。
2. 先离线运行两个系统的单元/机制测试，再跑小样本 smoke；smoke 不写入主结果目录。
3. 对每个系统单独构建记忆，再用完全相同的问题清单和裁判配置评估；禁止共享已压缩记忆。
4. 保存每题原始输入、检索结果、最终回答、计时、token、费用、错误和 judge 输出。
5. 先运行 no-cache 主结果，再运行 cache/ablation；不得用缓存命中掩盖质量差异。
6. 生成汇总表、置信区间、错误分类和成本-质量曲线；最后才比较优劣。

## 目前能得出的结论

当前本地证据只证明工程回归：MemNexus 后端回归通过，SimpleMem cross 测试通过，前端构建通过。MemNexus 现有 `evaluate.py` 是合成检索/延迟测试，`live_quality.py` 是 10 题真实模型开发检查，不能与 SimpleMem 论文的 LoCoMo/LongMemEval-S 结果直接横比。正式比较必须补齐公开数据集适配、统一 token/cost 记录和盲 judge。

## 后续优化优先级

1. **先减少调用而不是盲目加模型**：让规则规划器处理直接字段查询，抽取按窗口批处理，维护关系和摘要改为低频/异步；记录每次 answer 的真实调用轨迹。
2. **把重复查询规划结果缓存为短 TTL**，但缓存键必须包含 owner、session revision、memory generation、模型和完整 query；答案缓存只用于效率消融。
3. **将当前 owner 全量读取改为授权过滤后的 FTS/向量候选下推**，再做实体扩展；这是 MemNexus 当前最明确的规模瓶颈。
4. **统一 embedding 与索引**：固定 Qwen3 embedding revision，批量写入向量投影，必要时引入 ANN；先测 Recall@K 再决定是否牺牲精度换延迟。
5. **建立 evidence-first reranker**：先保证 gold evidence 召回，再限制最终上下文；不要用生成答案关键词替代检索质量。
6. **做错误驱动消融**：按 single-hop、multi-hop、temporal、open-domain、unknown、correction 分类，避免用总体平均掩盖某一类退化。

这些优化比继续扩大提示词模板更可能改善与 SimpleMem 的公平竞争结果。
