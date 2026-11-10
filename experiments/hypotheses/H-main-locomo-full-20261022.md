# H-main-locomo-full — 全量 LoCoMo 主运行预注册

- **落盘时间**：2026-10-22 23:18（先落盘，后执行；执行中不得修改本文任何判据）
- **性质**：第 4 步预注册。主指标 / 样本量 / 重复次数 / 判据，全部执行前固定。
- **前置状态**：第 1 步修复完成（440 passed，新增 16 项单测）；第 2 步 3 题冒烟门禁 PASS。
- **关联**：execution 路径是否启用，由 `H-execution-path-ab-20261022.md` 单独裁定。
  **本单元使用默认配置，不启用 semantic_execution**，以保证与 175 题历史数据可比。

---

## 1. 假设

**H1（主）**：在配对条件下（同一冻结记忆库、同一题组、同一模型），MemNexus 的
LoCoMo `Average (F1)` 高于 SimpleMem。

- 方向依据：175 题配对读数为 MN 53.41 vs SM 45.40（+8.01）。
- 但其中 **+5.67 来自列表连接符**（`ExecutionResult.render()` 对 `set` 算子用 `, `，LoCoMo 多跳按逗号切条目），
  **机制净差只有 +2.34**。本单元要判定的就是这 +2.34 量级是否稳定存在。
- **可证伪**：若 Average 差值 95% CI 跨 0，H1 不成立，只记方向。

**H0（对照）**：Token Cost（问答阶段每题 input+output tokens）MemNexus 不高于 SimpleMem。

---

## 2. 主指标

| 层级 | 指标 | 唯一实现 |
|---|---|---|
| **主** | `Average (F1)` = MultiHop / Temporal / OpenDomain / SingleHop 四类**类内均值再等权平均**，**不含 adversarial** | `paper_protocol.paper_table()["systems"][s]["average_f1"]` |
| 副 | 四类分题型 F1 | `f1_by_type` |
| 副 | `Average (BLEU-1)` | `average_bleu1` |
| 副 | `Adversarial Success Rate` | `adversarial_sr` |
| 副 | `Token Cost` / 题（input+output） | `paper_table()` |
| 副 | 延迟、实际花费、provider repair 次数 | stage / rows |

**单一主指标**：只有 `Average (F1)` 可以判"通过"。分题型与 BLEU-1 只用于解释与归因，不得单独拿去做显著性主张。

口径纪律（协议 §9.2）：

- 不把对抗题并入 Average；
- 不用逐题平均代替等权平均；
- 不在非配对子集上比较；
- 不改 A.4 裁判措辞（本单元 LoCoMo 不使用裁判，裁判只用于 LongMemEval）。

---

## 3. 实验单元

**一题**。两系统在同一题上的结果构成一对。

---

## 4. 样本量

### 4.1 数据集与题组

- 数据集：`benchmark-data/official/locomo/locomo10.json`，**10 段对话 / 1986 题**。
- 官方分类分布（全量，未过滤）：multi-hop 282、temporal 321、open-domain 96、single-hop 841、adversarial 446。

### 4.2 证据视界——必须在这里定死

历史 100/150/175 题组用的是 **D1–D5 视界**：只有当一道题的金标证据全部落在其对话的
session 1..5 内才算合格（`locomo_cohort.build`）。这是为省钱做的开发口径，不是论文口径。

离线实测该规则下的合格题池：

| 视界 | multi-hop | temporal | open-domain | single-hop | adversarial | 合计 |
|---|---|---|---|---|---|---|
| D1–D5 | 31 | 69 | 20 | 142 | 75 | **337** |
| D1–D10 | 61 | 126 | 38 | 274 | 149 | 648 |
| 不限（全 sessions） | 280 | 321 | 92 | 841 | 446 | **1980** |

**本单元口径 = 不限视界（全 sessions），题量 1986。**
理由：SimpleMem 论文与 LoCoMo 惯例都是在完整对话上跑全部题目；按 D1–D5 过滤会丢掉
82% 的题，且会使结果与论文基线不可比。

### 4.3 口径差异的代价：快照必须重冻结

历史开发题组的冻结基座是 D1–D5 口径（`result/small_ab_20261020/snapshots`，
conv-26 sha256 `f3f384cf4ff6…`，92 turns / 68 memories）——**不能用于全量主运行**。

执行前须新冻结一套**全 sessions** 快照到独立目录
`result/memnexus_snapshots/full`，并在报告中记录每段对话的 sha256。

⚠️ 冻结方式：第一次传 `--memory-snapshots result/memnexus_snapshots/full`
**且不加** `--require-snapshots`（未命中 ⇒ 建库后冻结）；之后的重跑/续跑才加
`--require-snapshots`（缺失即中止，禁止静默重建）。

⚠️ 全 sessions 下 conv-26 有 419 turns（vs D1–D5 的 92），建库成本约 4.5 倍；
SimpleMem 侧同步变贵。预算估计须按新的轮数重算，不得沿用旧表。

### 4.4 有效样本

两系统都完成、且未崩溃的题目的**交集**。崩溃题出组，并在报告中单独列出。
目标：交集 = 1986（即无崩溃、无缺失）。

---

## 5. 重复次数

**R = 1**（用户 2026-10-22 明确指示：预算充足，先测一遍）。

不掩饰的后果，写在这里以免事后被重新解释：

1. **题间抽样误差可以估计，run 级抖动不能。** 由 175 题配对读数外推，
   1986 题下 Average 差值的题间标准误约 ±1.1 个 Average 点；
   但冻结基座后同一配置连跑三遍，29 题中证据集三遍完全一致的只有 7/29，
   run 级非确定性带来的误差**无法从 R=1 的运行中估计**。
2. 因此本运行的**可判读下限不是 ±1.1 点，而是此前实测的约 7 个 Average 点噪声地板**。
   差值绝对值小于该量级时，只能记方向，不得判「提升」。
3. 若要把可判读下限压回 ±1~2 点，唯一办法是加 R，或先消除 plan 抖动。本轮不做。

---

## 6. 判据（预注册，不许事后改）

### 6.1 门禁——按顺序读，任一不过立即停

| 顺序 | 条件 | 判定 |
|---|---|---|
| 1 | 任一运行 `forced_abstain > 0` | **INVALID**，不读准确率，先修崩溃 |
| 2 | 每段对话两侧快照 sha256 不全等 | **UNPAIRED**，只记方向 |
| 3 | 配对交集 < 1986 的 90% | **INVALID**，先查崩溃/续跑原因 |
| 4 | 平均单题 provider repair > 1 | **INVALID**，先查输出违例 |

### 6.2 主判定（只对 `Average (F1)`）

| 条件 | 判定 |
|---|---|
| 净正确数 < 0 | **否决** |
| 配对检验 p ≥ 0.05 | **不可判定**，只记方向，不得写「优于」 |
| p < 0.05 且差值 > 0 | **通过** |
| p ≥ 0.05 但差值 > 7（噪声地板） | **方向性通过**，须在论文中标注 R=1 |

### 6.3 成本判定（H0）

| 条件 | 判定 |
|---|---|
| MN 每题 input+output tokens ≤ SM | 通过 |
| MN > SM 且增幅 ≤ +2% | 有条件通过 |
| MN > SM 且增幅 > +2% | 未通过，须由主张裁决 |

---

## 7. 执行方式

- 解释器必须用**系统 Python 3.13.3**
  （`C:/Users/zhangjinhan/AppData/Local/Programs/Python/Python313/python.exe`）；
  两个 `.venv` 缺依赖，用它们必然 ModuleNotFoundError。
- 跑之前**先起 Qwen embedding 服务**
  （`scripts/qwen_embedding_server.py --host 127.0.0.1 --port 8099`），
  并带 `NO_PROXY="127.0.0.1,localhost"`。
- 生成题组：`python experiments/locomo_cohort.py --size 1986 --max-session 99 --out <cohort.json>`
  （实测合格 1980，取全部）。把 cohort `sha256` 写进报告。
- 冻结基座：首次
  `python experiments/benchmark100.py --config <cohort.json> --budget <B> --memory-snapshots result/memnexus_snapshots/full --no-open`；
  冻结之后的重跑/续跑才加 `--require-snapshots`。
- 串行执行（runner 固定 workers=1）。约 4000 次问答，预计 15–22 小时；
  逐题 checkpoint，可中断续跑。
- 运行期间禁止修改 `MemNexus/`、`SimpleMem/`、`experiments/` 下任何源码
  （provenance 守卫会中止；已因此损失 ¥13）。

## 8. 回退条件

出现下列任一情况立即停跑并记录，不续跑：

1. 任一运行出现 `forced_abstain > 0`；
2. 前 100 题的崩溃率 > 5%；
3. 单题 provider repair 次数 > 3；
4. 累计花费超过预注册预算的 120%。

---

## 9. 变更记录

执行后若必须修改本文，只能**追加**到本节，并写明理由；已固定的判据数值不得改写。
