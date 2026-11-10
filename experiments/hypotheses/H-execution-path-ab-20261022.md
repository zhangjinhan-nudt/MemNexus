# H-execution-path-ab — 第 3 步决策与预注册

- **落盘时间**：2026-10-22 23:18（先落盘，后执行；执行中不得修改本文任何判据）
- **性质**：第 3 步「决定 execution 路径是否纳入测量」的结论 + 该实验单元的预注册
- **依赖**：第 1 步三条缺陷修复（缺陷 3 转为路径决策）、第 2 步 3 题冒烟门禁 PASS

---

## 1. 决策：**纳入测量，但作为独立预注册 A/B，默认不进主表**

不是「启用」也不是「不启用」，是**先测再决定**。理由分三条。

### 1.1 三条路径目前零覆盖，这是必须先处置的事实

`result/bench260` 的 178 条 MemNexus 记录按 `reasoning.answer_path` 分布：

| answer_path | 条数 |
|---|---|
| `source_bound_synthesis` | 175 |
| `failed` | 3 |
| `execution`（semantic_execution） | **0** |
| `indexed` | **0** |
| `source_adjudication` | **0** |

四条产出路径，主结果只走过一条。方法文档 §2.2 已加「实测覆盖声明」，§A2 已降级为「设计意图」——但那是**止血**，不是解决。解决只有两条路：收窄主张，或者测出来。预算充足 ⇒ 选后者。

### 1.2 覆盖面足够大，且正好压在最弱的两类上

离线重跑 `classify_question(query, entities)`（`experiments/probe_execution_path_coverage.py`，零成本，不调模型），按 `runtime._answer_impl` 的
`complex_route = kind ∈ {adversarial, count, set, relation, causal, inference}` 判定：

| 题型 | 会走 execution 路径 | 总题数 | 占比 |
|---|---|---|---|
| open-domain | 8 | 9 | **88.9%** |
| multi-hop | 12 | 17 | **70.6%** |
| adversarial | 38 | 38 | 100% |
| single-hop | 20 | 77 | 26.0% |
| temporal | 2 | 37 | 5.4% |
| **合计** | **80** | **178** | **44.9%** |

机制声称起作用的场景是「集合 / 关系 / 计数 / 时间运算」，而这批题里 open-domain 和 multi-hop 恰好是模型最弱、也是覆盖率最高的两类。这不是可以对着说「反正没走到」的量级。

### 1.3 R1 强制它必须是单独一轮

开启 `MEMORY_SEMANTIC_EXECUTION` 会改变模型调用（新增 `bind_answer`）与输出形态 ⇒ 属于行为变量。R1 规定一次运行只允许一个行为变量。因此：

- **主运行**：默认配置（`--variant configured`，semantic_execution 关闭），保证与 175 题历史数据可比。
- **A/B 运行**：`--variant semantic`，同一冻结基座、同一题组，单独一轮，单独判定。

---

## 2. 假设

**H-exec**：在冻结基座 + 同题组配对条件下，启用语义执行路径（`MEMORY_SEMANTIC_EXECUTION=true`）相对默认配置，LoCoMo `Average (F1)` 的变化符号为 **正**。

- 机制：执行路径把「义务」变成可核验的绑定（`bind_answer` → `check_and_execute`），对集合 / 关系 / 计数类义务可以拒绝无证据的补全，从而抑制过度生成。
- 预期量级：若成立，预计落在 +1 ~ +4 个 Average 点（依据：`probe_length_ceiling` 给出的长度失配修复上界 +4.90 是同量级参照，非本假设的估计）。
- **可证伪**：若 Average 差值的 95% CI 完全落在 0 以下，或 `Token Cost` 增幅 > +2% 且差值不显著，假设被推翻。

## 3. 主指标

| 层级 | 指标 | 实现 |
|---|---|---|
| **主** | LoCoMo `Average (F1)`：MultiHop / Temporal / OpenDomain / SingleHop 四类类内均值再等权平均，**不含 adversarial** | `paper_protocol.paper_table()["systems"][s]["average_f1"]` |
| 副 | 四类分题型 F1 | `f1_by_type` |
| 副 | `Average (BLEU-1)` | `average_bleu1` |
| 副 | `Adversarial SR` | `adversarial_sr` |
| 副 | `Token Cost`（问答阶段每题 input+output） | `paper_table()` |
| 副 | 延迟、成本、provider repair 次数 | stage / rows |

口径纪律（协议 §9）：不得把对抗题并入 Average；不得用逐题平均代替等权平均；不得在非配对子集上比较。

## 4. 样本量

- 实验单元 = **一题**。
- 题组 = 与主运行同一套（见 `H-main-locomo-full-20261022.md` §4）。
- 本单元的**有效样本**是两配置都完成的题目的**交集**；任一侧崩溃/缺失即出组。

## 5. 重复次数

**R = 1**（用户 2026-10-22 明确：先测一遍）。

已知后果，不掩饰：冻结基座后同一配置连跑三遍，29 题中证据集三遍完全一致的只有 7/29。本单元 R=1 ⇒ **run 级误差无法从本单元估计**，只能向主运行借。

## 6. 判据（预注册，不许事后改）

### 门禁（任一不通过，不读准确率）

| 条件 | 判定 |
|---|---|
| 任一运行 `forced_abstain > 0` | **INVALID**，先修崩溃 |
| 两侧快照 sha256 不全等 | **UNPAIRED**，只记方向 |
| 配对交集 < 总题数的 90% | **INVALID**，先查崩溃/续跑 |

### 判定表

| 条件 | 判定 |
|---|---|
| 净正确数 < 0 | **否决** |
| McNemar 精确 p ≥ 0.05 | **不可判定**，不得写「提升/下降」，只记方向 |
| p < 0.05 且 Token Cost 增幅 ≤ +2% | **通过** |
| p < 0.05 且 Token Cost 增幅 > +2% | **有条件通过**，须由主张裁决并写清牺牲了什么 |

### 通过后做什么

- 通过 ⇒ 主表改用 semantic 配置重跑，并在方法文档 §2.2 把覆盖声明改为实测分布。
- 否决/不可判定 ⇒ 主表保持默认配置，方法文档把另外三条路径移入「未启用机制」附录，正文只描述实际测量的路径。

## 7. 回退条件

出现下列任一情况立即停跑并记录，不续跑：

1. 任一运行出现 `forced_abstain > 0`；
2. 单题 provider repair 次数 > 3（提示修复引入了新的输出违例）；
3. 跑完前 50 题时，两配置在任一题型上的 F1 差值绝对值 > 15（超出题间抽样噪声的合理范围，先查口径再继续）。
