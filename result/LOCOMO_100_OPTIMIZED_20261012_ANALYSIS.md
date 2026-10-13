# 优化后 MemNexus 与 SimpleMem：100 题配对复测（2026-10-12）

本次统一裁判正确率：MemNexus **75/100**，SimpleMem **81/100**，MemNexus 低于 SimpleMem **6 个百分点**。
旧版 MemNexus 为 65/100，本次净变化 +10 题；SimpleMem 历史值 82/100，本次也重新建库、生成和评分。

这是优化所用开发集上的回归复测，不是独立保留测试，不能据此宣称跨数据集泛化或 ICAPS 论文主张已成立。

## 数据与审计

- 原始 LoCoMo 文件中预先冻结的100道题，10段对话，每段仅输入D1–D2，共450 turns。两套系统独立从原始对话建库；不共享压缩记忆。两边均使用数据集已有blip_caption作为图片的文本描述，未另调用视觉模型。
- 选择条件使用官方 evidence 是否位于D1–D2，存在已知的 evidence horizon 筛选；不是完整长程LoCoMo，也不代表完整1986题的分布。
- 主模型/裁判配置均为 deepseek-v4-flash；实际供应商返回的 model 标识见下方审计。Qwen3-Embedding-0.6B，1024维，query prompt相同；英文回答、temperature=0、关闭答案缓存、顺序运行。
- SimpleMem保留原有规划与反思检索；MemNexus使用model planner、原文来源恢复、时间区间推导与最多一轮原文补检索。它们不是同等上下文长度或等token预算的消融实验。
- 20次建库均非空，200条回答与200个裁判标签完整；核对了39个源码哈希、30份ledger和请求/响应sidecar。
- 实际请求 1130 次，provider errors=0；所有token、阶段分账与总账一致，未完成预留均归零。
- 供应商返回model标识计数：`{"deepseek-flash": 1130}`。
- 首次尝试因MemNexus提示词预算溢出中止，完整保留失败记录。修复为先预留实际序列化任务/规划/规则，再进行有界证据选择；54项相关回归通过，算法随后冻结。
- 第二次尝试完成conv-26后，SimpleMem在conv-30建库连续三次输出截断。两套provider与裁判加入一致规则：仅明确finish_reason=length时允许一次8192上限恢复，普通请求仍4096；27项provider回归通过。
- 最终执行从第二次尝试保留已完整配对评分的conv-26（其调用未触发截断），只继续其余会话；没有选择分数更高的回答。继承源码哈希和恢复原因保存在resumed_from。未完成conv-30的失败建库成本单独计入本次总开销。

## 总体指标

| 指标 | SimpleMem 本次 | MemNexus 优化后 | MemNexus 相对变化 |
|---|---:|---:|---:|
| 裁判正确题数 /100 | 81 | 75 | -6个百分点 |
| 问答输入 tokens | 742,281 | 426,488 | -42.5% |
| 问答输出 tokens | 54,847 | 24,852 | -54.7% |
| 问答总 tokens | 797,128 | 451,340 | -43.4% |
| 问答模型请求 | 572 | 200 | -65.0% |
| 建库模型请求 | 66 | 92 | +39.4% |
| 建库＋问答总 tokens | 1,080,080 | 842,241 | -22.0% |
| 问答费用估算（元） | 2.3080 | 1.2622 | -45.3% |
| 建库费用估算（元） | 1.9398 | 1.4408 | -25.7% |
| 建库＋问答费用估算（元） | 4.2478 | 2.7030 | -36.4% |
| 每答对一题的建库＋问答成本（元） | 0.05244 | 0.03604 | -31.3% |
| 回答延迟P50（秒） | 6.107 | 2.692 | -55.9% |
| 回答延迟P95（秒） | 13.378 | 3.496 | -73.9% |
| 建库记忆条目数 | 1,324 | 355 | -73.2% |
| 平均最终来源数 | 31.16 | 6.53 | -79.0% |

token和请求数直接来自provider usage，问答阶段包括规划、检索反思与最终生成。费用采用与旧实验一致的保守单价：输入¥2.4/百万、输出¥9.6/百万，不抵扣cache hit或错峰折扣；它不是DeepSeek账户的实际扣款账单，也不包含本地embedding、电力与硬件成本。

完整本轮（双系统建库＋问答＋裁判）估算 **¥7.086569**；中止尝试 **¥1.857355**；本次请求合计 **¥8.943924**，低于用户提供的¥31余额。历史费用没有被当作本次账户扣款。

上表是最终成功配对流程的成本，其中包含流程内所有恢复调用；运行协议修复前废弃的部分不混入算法单题均值，但已在中止尝试总额单独报告。因此同时提供了成功流程成本和实际测试开销，不能把失败费用隐藏。

## 分层结果与统计不确定性

补充词面指标：依据官方evaluation.py的归一化、Porter词干化与类别1多答案规则，非对抗75题的原样回答F1为 SimpleMem 0.6187、MemNexus 0.1838。对两边统一删除【来源N】后的格式敏感性检查为 0.6187、0.2354。原样分数是主词面指标，清理版单独标注，未改动原始回答。

| 官方类别ID | 题数 | SimpleMem 本次 | MemNexus 本次 | MemNexus 旧版 |
|---|---:|---:|---:|---:|
| 1 | 13 | 7 | 6 | 5 |
| 2 | 25 | 24 | 21 | 13 |
| 3 | 9 | 5 | 2 | 3 |
| 4 | 28 | 23 | 25 | 23 |
| 5 | 25 | 22 | 21 | 21 |

类别名称校正：旧runner将类别1标为single-hop，但[官方evaluation.py](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)明确将1作为multi-hop。类别2是时间、5是对抗；3/4在本报告保留原始编号，不沿用旧报告未经核准的英文名称。这个修正只影响展示解释，不改变抽样、题目或评分。

逐题配对：MemNexus独对 6 题，SimpleMem独对 12 题；两者都对 69 题，两者都错 13 题。
按10段对话为cluster、重复抽样10,000次的MemNexus−SimpleMem正确率差95%区间为 **[-12.2, 0.0] 个百分点**。区间跨越0；样本只有10个cluster，结果仅作探索性估计。
假定题目独立时的McNemar双侧p=0.237885；实际题目在对话内相关，不能只用这个p值作论文显著性结论。
MemNexus旧错→新对 14 题，旧对→新错 4 题。两轮均重新建库和生成，且裁判有随机性，因此差值不是严格的单组件因果效应。

## 机制观测

相对旧版MemNexus，问答总tokens为 425,631→451,340（+6.0%），问答成本估算 ¥1.2041→¥1.2622，建库＋问答成本 ¥2.6794→¥2.7030。质量改善与成本变化必须一起阅读；相对SimpleMem节省费用，不等于相对旧版MemNexus所有开销都下降。

100题中，触发一次补检索 88 题，最终选中原文补充 81 题，出现时间推导 67 题，检测到归属冲突 0 题。最大补检索轮数=1，本地grounding平均耗时 6.52 ms。
词面missing_info槽位总数 116→83。这些是检索启发式日志，不是逻辑蕴含证明；有时间推导、槽位减少都不能直接等同回答正确。

## 可复核案例

以下按固定题目顺序各取最多3例：优化修复、仍落后于SimpleMem、以及优化退步。完整200条回答与全部新旧转换保存在JSON；不修改裁判标签。

### 旧错→新对

**conv-26 / qa_index=1**：When did Melanie paint a sunrise?

- Gold：2022
- SimpleMem：2022
- MemNexus优化后：Melanie painted a lake sunrise in 2022, since she said on May 8, 2023 that she painted it "last year"【来源1】.
- MemNexus优化前：Melanie painted a lake sunrise last year, which is special to her【来源1】. The exact date isn't recorded.
- 本次MemNexus裁判理由：The prediction states the sunrise was painted in 2022, matching the gold answer.

**conv-30 / qa_index=6**：When is Jon's group performing at a festival?

- Gold：February, 2023
- SimpleMem：February 2023
- MemNexus优化后：Jon said his group was finishing choreography to perform at a nearby festival "next month," which points to February 2023【来源5】. The exact date or time isn't recorded.
- MemNexus优化前：I don't have a specific date or time for Jon's group's festival performance. The available information only says the group is finishing choreography to perform at a nearby festival next month【来源1】, and that Jon is glad to be part of it【来源2】. The festival's name and exact timing aren't recorded.
- 本次MemNexus裁判理由：The prediction identifies February 2023 as the festival performance time, matching the gold answer.

**conv-30 / qa_index=7**：When did Gina launch an ad campaign for her store?

- Gold：29 January, 2023
- SimpleMem：29 January 2023 at 2:32 PM
- MemNexus优化后：Gina launched an ad campaign for her clothing store on 2023-01-29【来源1】.
- MemNexus优化前：The available evidence confirms Gina launched an ad campaign for her clothing store【来源1】, but it does not include the date or time when she launched it.
- 本次MemNexus裁判理由：The prediction states the same date, 2023-01-29, which is equivalent to 29 January, 2023.

### 本次SimpleMem对、MemNexus错

**conv-26 / qa_index=5**：When did Melanie run a charity race?

- Gold：The sunday before 25 May 2023
- SimpleMem：20 May 2023
- MemNexus优化后：Melanie ran a charity race for mental health on Saturday, May 20, 2023【来源1】.
- MemNexus优化前：Melanie ran a charity race for mental health last Saturday【来源1】.
- 本次MemNexus裁判理由：Gold answer specifies the Sunday before 25 May 2023 (i.e., Sunday, 21 May 2023), but the prediction says Saturday, May 20, 2023, contradicting the day.

**conv-26 / qa_index=155**：Why did Melanie choose the adoption agency?

- Gold：Not mentioned in the conversation
- SimpleMem：Not mentioned in the conversation
- MemNexus优化后：B. because of their inclusivity and support for LGBTQ+ individuals

Caroline said she chose the agency because it helps LGBTQ+ folks with adoption, and that its inclusivity and support spoke to her【来源2】.
- MemNexus优化前：B. because of their inclusivity and support for LGBTQ+ individuals 【来源1】
- 本次MemNexus裁判理由：The gold answer states the reason is not mentioned in the conversation, but the prediction provides a specific reason and attributes it to Caroline, contradicting the gold answer.

**conv-30 / qa_index=2**：How do Jon and Gina both like to destress?

- Gold：by dancing
- SimpleMem：By dancing
- MemNexus优化后：Gina suggests dancing it out to destress【来源1】. There's no evidence about Jon's own preferred way to destress, so I can't confirm whether they share this method.
- MemNexus优化前：Both Jon and Gina use dance to destress: Gina says dance is her go-to for stress relief【来源3】, and Jon says he will make time to dance and vent【来源1】.
- 本次MemNexus裁判理由：The prediction does not confirm that both Jon and Gina destress by dancing; it only states Gina's method and says Jon's is unknown, missing the core fact that both share this method.

### 旧对→新错

**conv-30 / qa_index=2**：How do Jon and Gina both like to destress?

- Gold：by dancing
- SimpleMem：By dancing
- MemNexus优化后：Gina suggests dancing it out to destress【来源1】. There's no evidence about Jon's own preferred way to destress, so I can't confirm whether they share this method.
- MemNexus优化前：Both Jon and Gina use dance to destress: Gina says dance is her go-to for stress relief【来源3】, and Jon says he will make time to dance and vent【来源1】.
- 本次MemNexus裁判理由：The prediction does not confirm that both Jon and Gina destress by dancing; it only states Gina's method and says Jon's is unknown, missing the core fact that both share this method.

**conv-42 / qa_index=206**：What genre is Joanna's first novella?

- Gold：Not mentioned in the conversation
- SimpleMem：Not mentioned in the conversation
- MemNexus优化后：B. drama and romance 【来源1】
- MemNexus优化前：A. Not mentioned in the conversation

The conversation mentions Joanna’s first full screenplay as a mix of drama and romance【来源1】, but it does not mention a first novella or its genre.
- 本次MemNexus裁判理由：The gold answer states the genre is not mentioned, but the prediction asserts a specific genre (drama and romance), which is an unsupported addition and contradicts the gold answer.

**conv-43 / qa_index=5**：Based on Tim's collections, what is a shop that he would enjoy visiting in New York city?

- Gold：House of MinaLima
- SimpleMem：MinaLima House
- MemNexus优化后：I don't have enough evidence to name a specific New York City shop for Tim. The records show he's a Harry Potter fan with a fan project and a MinaLima picture on his bookshelf【来源7】【来源8】, and he's visited Harry Potter-related places in London【来源5】—but nothing states his collection items, the types of shops he enjoys, or any NYC shop options.
- MemNexus优化前：Tim's documented collection is a picture from MinaLima, the studio that created props for the Harry Potter films【来源2】. He also went on a Harry Potter tour he found amazing【来源5】and would love to explore real Harry Potter places someday【来源6】.

A shop he would likely enjoy in New York City is **MinaLima's Harry Potter-themed store/graphic art shop** (House of MinaLima), given his MinaLima picture and Harry Potter interest【来源2】【来源5】【来源6】.

That said, there is no evidence in memory of a specific New York City shop Tim has mentioned or a stated preference for stores there, so this is an inference from his Harry Potter/MinaLima interest rather than a confirmed preference.
- 本次MemNexus裁判理由：The prediction fails to provide the gold answer 'House of MinaLima' and instead claims insufficient evidence, missing the core fact.

## 解释边界与面向ICAPS的下一步

**已核实的gold冲突和裁判不一致（不改分）**：conv-26 / qa_index=5，问题是Melanie何时参加慈善跑。官方证据D2:1的记录日期为2023-05-25，原文明确写“I ran a charity race for mental health last Saturday”，对应2023-05-20；但gold写“The sunday before 25 May 2023”。SimpleMem回答“20 May 2023”被判对，理由错误地声称这一天是周日；MemNexus回答“Saturday, May 20, 2023”被判错。两边给出同一天，原始证据也支持该日。这说明gold-only LLM裁判存在标注与一致性问题。主表保留原始评分；此处不单方面给某个系统加分，也未声称已完成全数据人工复核。

1. 同一模型生成并裁判，且裁判只看gold、问题和预测，不看完整原始对话；评分可能存在偏差。必要人工复核应单独记录，不能为了提升分数覆盖原始标签。
2. 类别5沿用此前A=Not mentioned、B=对抗答案的固定选项协议，存在位置偏差。25%的题为这类题，不能把总体分数等同官方全量LoCoMo指标。
3. runner原始字段f1/bleu1是简单词面诊断，不是官方LoCoMo评分函数。本文另计算官方F1规则下非对抗75题的原样与去引用敏感性指标；词面指标受冗长输出影响，不能与语义裁判准确率混为一谈，更不能称为完整LoCoMo成绩。
4. 这100题参与了优化诊断，因此下一阶段应冻结算法，在未用于开发的LongMemEval-S完整haystack和更长LoCoMo历史上测试，并对时间层、来源恢复和补检索做受控消融。
5. ICAPS是规划与调度会议。QA证据不足以支撑规划贡献；应把记忆误差落实到belief state、行动前置条件与执行监控，比较真实局部修复和全局重规划的plan success、replanning success、约束违反、搜索节点与修复开销。当前测试没有测这些指标。

## 原始文件

- 本次：`result\locomo_100_optimized_20261012_v5.json`
- 历史：`result\locomo_100_results_v3.json`
- 总览：`result\locomo_100_optimized_comparison.png`；另有SVG与detail图。
- 每个会话的原始请求/响应、构建记忆、逐题输出与ledger位于本次JSON同名的 `_ledgers` 目录。
- `locomo_100_optimized_source_snapshot.zip`：经哈希验证的最终源码、分析脚本和公开实验manifest；不包含.env或带API key的config.py。继承的首段会话使用截断恢复之前的provider，其差异保留在原始结果resumed_from中。
