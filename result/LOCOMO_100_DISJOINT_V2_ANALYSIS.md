# LoCoMo 第二组不重复100题配对测试

新旧 cohort 的题目键重叠 **0**；新 cohort 哈希 `5f1a4e079a76cb5ef651fcf6d6e3015c63201b9dfa04f16aec2fa63d0650cc83`。实验输入为10段对话的 D1–D5，共 1104 条原始 turn；题目官方 evidence 均落在该范围内，其中 69 题需要 D3–D5 的来源。

统一裁判正确率：SimpleMem **81/100**，MemNexus **71/100**。MemNexus−SimpleMem = **-10 个百分点**，按对话聚类重采样的95%区间为 **[-21.7, 1.0]** 个百分点。

## 实验核查

- 200 条非空回答、200 个裁判标签、20 个非空独立记忆库；provider attempts = 1398。账本中1次 error 是首段裁判阶段进程中断时未返回的请求，已按预留上界入账并补做该题裁判。
- 所有请求/响应 usage、阶段用量和费用账本已核对。供应商返回模型标识：`{'deepseek-flash': 1397}`。
- 两边使用 DeepSeek V4 Flash、Qwen3-Embedding-0.6B、英文回答，依次从同一 D1–D5 原始对话建库；答案缓存关闭。
- 类别5沿用固定 A=Not mentioned/B=干扰答案协议；问题按官方 category_id 分组。该100题因按 gold evidence horizon 选择，不代表官方全集分布。
- 旧 cohort 的 D1–D2 分数不能与本次 D1–D5 分数直接作算法前后因果对照。
- 已保存的问题路由覆盖100题；100题的回答模型调用均为1次，直接 gate 回答为 0 次。该 runner 没有保存 `AnswerResponse.reasoning` 字段，因此不能把类别5的结果归因于 typed negative gate；下轮需逐题保存 gate trace。

## 总体指标

| 指标 | SimpleMem | MemNexus | 相对变化 |
|---|---:|---:|---:|
| 裁判正确题数 /100 | 81 | 71 | -10百分点 |
| 按 LoCoMo 词面规则重算 F1，非对抗题原样 | 0.6509 | 0.1542 | -76.3% |
| 问答输入 tokens | 882,720 | 432,633 | -51.0% |
| 问答输出 tokens | 54,998 | 28,999 | -47.3% |
| 问答模型请求 | 573 | 200 | -65.1% |
| 建库模型请求 | 198 | 226 | +14.1% |
| 建库＋问答 tokens | 1,839,024 | 1,426,616 | -22.4% |
| 建库＋问答费用估算（元） | 8.8436 | 4.8794 | -44.8% |
| 每答对一题模型成本（元） | 0.10918 | 0.06872 | -37.1% |
| 回答延迟 P50（秒） | 5.623 | 2.699 | -52.0% |
| 回答延迟 P95（秒） | 11.886 | 3.832 | -67.8% |
| 记忆条目总数 | 4,036 | 856 | -78.8% |

费用用 provider 实际 token 与同一保守单价估算，不是账户扣款账单；本地 embedding 与硬件费用不计入。
本次完成的配对流程（双系统与裁判）费用估算上界为 ¥13.9016；三次未完成的 conv-48 阶段另计 ¥2.1541，本次所有调用上界约 ¥16.0557。其中中断或连接失败的未知用量按预留上界计入；供应商实际账单可能更低。

## 类别与配对结果

| 官方类别 ID | 题数 | SimpleMem 答对 | MemNexus 答对 |
|---|---:|---:|---:|
| 1 | 13 | 9 | 6 |
| 2 | 25 | 22 | 20 |
| 3 | 9 | 3 | 1 |
| 4 | 28 | 26 | 20 |
| 5 | 25 | 21 | 24 |

两边都对 62 题；仅 MemNexus 对 9 题；仅 SimpleMem 对 19 题；两边都错 10 题。问题级 McNemar 双侧 p=0.08716；该检验假设题目独立，同一对话内的相关性使其偏乐观。

## 来源覆盖与错因线索

MemNexus 在非对抗题 75 道中，建库记忆的 evidence 引用覆盖全部 gold turn 的有 74 道，最终来源全部覆盖仅 48 道；其中 26 道在建库到最终选择之间丢失至少一条引用。最终来源至少覆盖一条 gold turn 的有 57 道。

其非对抗错题 28 道中，1 道在建库记忆中没有 gold turn 引用，12 道有引用却未进入最终来源，9 道已检齐 gold turn 仍答错。记忆引用覆盖仅证明原始 turn 被索引，不证明抽取内容正确；最终 turn 命中也不等于正确推理。

SimpleMem 的返回上下文只有压缩文本，没有原始 turn ID，因此无法用同一严格方式计算其 evidence Recall。

## 逐题不一致案例

### 仅 MemNexus 答对

**conv-26/7，类别1：What is Caroline's relationship status?**

Gold：Single

SimpleMem：Not specified in the provided context

MemNexus：Caroline has said she plans to adopt as a single parent, describing it as “tough as a single parent” but saying she’s up for the challenge【来源3】. No evidence in the retrieved sources confirms a current partner, spouse, dating status, or any change in that status.

裁判理由：The prediction conveys that Caroline is single (a single parent with no confirmed partner or spouse), which matches the gold answer.

**conv-30/86，类别5：What kind of flooring is Gina looking for in her dance studio?**

Gold：Not mentioned in the conversation

SimpleMem：Marley flooring

MemNexus：A. Not mentioned in the conversation

The flooring preference discussed is for Jon's dance studio, not Gina's. Jon wants Marley flooring【来源5】, and Gina comments that Marley is a good choice【来源2】, but no flooring preference for Gina's own dance studio is stated.

裁判理由：The prediction correctly states that Gina's flooring preference is not mentioned, matching the gold answer.

**conv-41/68，类别4：What type of workout class did Maria start doing in December 2023?**

Gold：aerial yoga

SimpleMem：No information available

MemNexus：Maria started doing aerial yoga【来源1】. However, the evidence dates this to December 2022, not December 2023, so the 2023 timing in the question is not supported.

裁判理由：The prediction identifies the workout class as aerial yoga, matching the gold answer; the added timing caveat does not change the core answer.

**conv-43/183，类别5：What did Tim celebrate at a restaurant with teammates?**

Gold：Not mentioned in the conversation

SimpleMem：a tough win

MemNexus：A. Not mentioned in the conversation

The restaurant celebration with teammates was John’s, not Tim’s【来源2】.

裁判理由：The prediction selects the gold answer 'Not mentioned in the conversation' and does not contradict it.

**conv-43/184，类别5：What kind of deals did Tim sign with Nike and Gatorade?**

Gold：Not mentioned in the conversation

SimpleMem：B. basketball shoe and gear deal with Nike, potential sponsorship deal with Gatorade

MemNexus：A. Not mentioned in the conversation

The Nike basketball shoe and gear deal and the potential Gatorade sponsorship were stated by John, not Tim【来源1】【来源5】.

裁判理由：The prediction selects the gold answer 'Not mentioned in the conversation' and does not contradict it.

**conv-44/4，类别2：When did Audrey make muffins for herself?**

Gold：The week of April 3rd to 9th

SimpleMem：The week before 16 April 2023

MemNexus：Audrey said she made some of her favorite treats "last week" on 2023-04-16, which places the baking in the week of 2023-04-03 to 2023-04-09【来源1】. The treats shown were blueberry muffins, but she didn't state an exact day.

裁判理由：The prediction identifies the week of April 3rd to 9th, matching the gold answer, and notes the exact day wasn't specified.

**conv-44/23，类别1：What are some problems that Andrew faces before he adopted Toby?**

Gold：Finding the right dog and pet-friendly apartments close to open spaces

SimpleMem：The dog adoption process was annoying

MemNexus：The evidence does not mention Toby or his adoption, so I can't establish what problems Andrew faced *before* adopting Toby.

What is supported is that during his dog search Andrew found the process "both fun and annoying"【来源7】, and he noted that finding a dog-friendly place in the city can be challenging【来源9】. He also described practical considerations for choosing a dog — living space size and the breed's exercise needs — and said that as someone in an apartment, a smaller dog would suit him best【来源4】. He wanted a place near a park or woods to give a dog open space to run【来源5】.

Whether any of this occurred before Toby's adoption, or whether Toby was adopted from a shelter and when, is not in the retrieved evidence.

裁判理由：The prediction identifies the same core problems as the gold answer: finding a suitable dog and finding a dog-friendly apartment near open spaces, while noting the evidence does not confirm timing relative to Toby's adoption.

**conv-48/197，类别5：Where did Jolene meet her new friend Anna?**

Gold：Not mentioned in the conversation

SimpleMem：yoga in the park

MemNexus：A. Not mentioned in the conversation

The conversation mentions Deborah meeting her new neighbor Anna at yoga in the park【来源4】, but it does not say where Jolene met Anna.

裁判理由：The prediction correctly states that the meeting location is not mentioned in the conversation, matching the gold answer.

**conv-50/0，类别2：When did Calvin first travel to Tokyo?**

Gold：between 26 March and 20 April 2023

SimpleMem：Not specified in the context

MemNexus：Calvin's first trip to Tokyo was planned for April 2023 — he said on 2023-03-23 that he was heading there "next month"【来源4】. This was his first visit, as he had never been to Japan before【来源8】. The trip was for a music festival in Tokyo【来源9】.

裁判理由：The prediction states the first trip was planned for April 2023, which is consistent with the gold answer's range of 26 March to 20 April 2023.

### 仅 SimpleMem 答对

**conv-26/85，类别4：What are Caroline's plans for the summer?**

Gold：researching adoption agencies

SimpleMem：Researching adoption agencies, attending a transgender conference in July 2023, and pursuing counseling/mental health

MemNexus：Caroline's recorded summer plans are limited: she said she was going to a transgender conference in July 2023, where she was excited to meet others in the community and learn more about advocacy【来源7】. No other summer-specific plans or activities from Caroline appear in the evidence.

裁判理由：The prediction does not mention researching adoption agencies, which is the gold answer, and instead discusses a transgender conference, contradicting the required core fact.

**conv-26/86，类别4：What type of individuals does the adoption agency Caroline is considering support?**

Gold：LGBTQ+ individuals

SimpleMem：LGBTQ+ folks

MemNexus：The evidence doesn't specify what type of individuals the adoption agency Caroline is considering supports. It only shows she is looking into an agency and that her goal is to give kids a loving home【来源5】.

裁判理由：The prediction fails to provide the gold answer 'LGBTQ+ individuals' and instead claims the information is not specified.

**conv-41/3，类别1：What type of volunteering have John and Maria both done?**

Gold：Volunteering at a homeless shelter

SimpleMem：Volunteering at a homeless shelter

MemNexus：No shared type of volunteering is supported by the evidence. Maria has volunteered at a homeless shelter【来源1】【来源7】, including a kids’ event there【来源5】. The evidence does not show John doing any volunteering.

裁判理由：The prediction denies any shared volunteering and states John has no volunteering evidence, contradicting the gold answer that both volunteered at a homeless shelter.

**conv-41/8，类别3：What might John's financial status be?**

Gold：Middle-class or wealthy

SimpleMem：John appears financially comfortable, as his kids have so much while others don't.

MemNexus：I don't have any information about John's employment, income, or financial situation. The available evidence covers his interest in local politics, education and infrastructure, and community work, but nothing about his finances.

裁判理由：The prediction does not provide the gold answer 'Middle-class or wealthy' and instead states no information is available.

**conv-42/8，类别2：When did Nate get his first two turtles?**

Gold：2019

SimpleMem：Around 23 January 2019

MemNexus：I don't have any record of when Nate got his first two turtles. The evidence mentions Nate keeping turtles and sharing photos of them, but no date or time period for acquiring them is stated.

裁判理由：The prediction fails to provide the gold answer '2019' and instead states no date is known.

**conv-42/9，类别2：What major achievement did Joanna accomplish in January 2022?**

Gold：finished her screenplay and printed it

SimpleMem：Finished her first full screenplay on 21 January 2022

MemNexus：I don't have evidence of a major achievement by Joanna in January 2022. The recorded achievements are from later: she wrapped up her second script and had submitted a screenplay to a film festival, both mentioned in March 2022【来源7】【来源3】.

裁判理由：The prediction denies the January 2022 achievement and does not state that Joanna finished and printed her screenplay, contradicting the gold answer.

**conv-42/11，类别1：What is Joanna allergic to?**

Gold：Most reptiles,animals with fur,cockroaches, dairy

SimpleMem：Most reptiles, animals with fur, cockroaches, and dairy

MemNexus：Joanna is allergic to most reptiles and animals with fur, and she also found out she is allergic to cockroaches.【来源5】【来源6】

裁判理由：The prediction omits dairy, which is a core fact in the gold answer.

**conv-43/11，类别1：What sports does John like besides basketball?**

Gold：surfing

SimpleMem：Surfing

MemNexus：I don't have a record of any sports John likes besides basketball. The evidence shows he was considering sports brands like Nike and Under Armour for endorsement collaborations, but that doesn't establish which sports he personally likes【来源1】.

裁判理由：The prediction fails to provide the gold answer 'surfing' and instead states no record of other sports, which contradicts the gold answer.

**conv-43/74，类别4：What aspects of the Harry Potter universe will be discussed in John's fan project collaborations?**

Gold：characters, spells, magical creatures

SimpleMem：Characters, spells, and magical creatures

MemNexus：I don't have any record of John having a Harry Potter fan project or collaborations. The fan project discussing characters, spells, and magical creatures belongs to Tim, not John【来源1】. There's no evidence about aspects covered in any fan project of John's.

裁判理由：The prediction denies knowledge of John's fan project and attributes the described aspects to Tim, contradicting the gold answer that John's project covers characters, spells, and magical creatures.

**conv-43/76，类别4：What kind of picture did Tim share as part of their Harry Potter book collection?**

Gold：MinaLima's creation from the Harry Potter films

SimpleMem：A picture from MinaLima, the company that created Harry Potter film props

MemNexus：Tim shared a photo of a bookshelf with books and a picture on it, as part of his book collection【来源1】.

裁判理由：The prediction describes a generic bookshelf photo and does not identify the picture as MinaLima's creation from the Harry Potter films, missing the core fact of the gold answer.

**conv-44/5，类别2：When did Audrey see a hummingbird?**

Gold：first week of May 2023

SimpleMem：During the week before 3 May 2023

MemNexus：Audrey saw the hummingbird during a hike in the week before May 3, 2023 — she described it as "last week," placing it around April 24–30, 2023【来源1】.

裁判理由：Gold answer is the first week of May 2023, but the prediction places the sighting around April 24–30, 2023, which contradicts the gold timeframe.

**conv-47/4，类别2：When did John resume playing drums in his adulthood?**

Gold：February 2022

SimpleMem：About a month before 27 March 2022

MemNexus：I don't have evidence about when John resumed playing drums in adulthood. The only drum-related fact recorded is that John plays drums and has a drum kit set【来源4】, with no timing or reason for resuming.

裁判理由：The prediction fails to provide the gold answer 'February 2022' and instead states it lacks evidence, so it does not correctly answer the question.

**conv-47/5，类别1：What are John and James' favorite games?**

Gold：John's favorite game is CS:GO, and James's is Apex Legends.

SimpleMem：John's favorite game is CS:GO, and James's favorite game is Apex Legends.

MemNexus：John's favorite game is CS:GO【来源1】. There's no record of James's favorite game; he's mentioned getting into RPGs and strategy games, but that's not stated as a favorite【来源2】.

裁判理由：The prediction omits James's favorite game (Apex Legends) and instead says there is no record of it, contradicting the gold answer.

**conv-47/7，类别3：In which state is the shelter from which James adopted the puppy?**

Gold：Connecticut.

SimpleMem：Connecticut

MemNexus：The shelter is in Stamford, but the evidence does not specify which state it is in【来源1】.

裁判理由：The prediction fails to provide the state (Connecticut) and instead claims the state is unspecified, contradicting the gold answer.

**conv-47/8，类别1：How many pets does James have?**

Gold：Three dogs.

SimpleMem：James has three pets: dogs Max and Daisy, and a puppy named Ned.

MemNexus：I don't have any record of James owning pets, so I can't give a number. The evidence mentions James proposing a dog-walking and pet-care app【来源8】and discussing pets generally, but nothing states that James has pets of his own.

裁判理由：The prediction fails to provide the gold answer of three dogs and instead states it cannot give a number, contradicting the gold answer.

**conv-47/72，类别4：What instrument is John learning to play as of 27 March, 2022?**

Gold：Drums

SimpleMem：Drums

MemNexus：There is no evidence that John is learning to play an instrument as of 27 March 2022. The records show John plays drums and has a drum kit【来源5】, but that is not the same as learning a new instrument. The only person recorded as learning an instrument around that date is James, who started learning drums a few days before 27 March 2022【来源1】【来源7】.

裁判理由：The prediction denies that John is learning an instrument, contradicting the gold answer that John is learning drums.

**conv-48/25，类别4：What are Jolene's favorite books?**

Gold：Sapiens, Avalanche by Neal Stephenson

SimpleMem：Sapiens and Avalanche by Neal Stephenson

MemNexus：I don't have any record of Jolene's favorite books, specific titles, authors, or genres. The only book-related detail is that she shared a photo of her bookshelf【来源2】, which doesn't indicate preferences.

裁判理由：The prediction fails to provide the gold answer titles and instead claims no record of Jolene's favorite books.

**conv-49/86，类别4：What advice did Evan give Sam about finding a passion?**

Gold：keep trying new things until something sparks excitement

SimpleMem：Keep trying new things until something sparks excitement

MemNexus：Evan advised finding a fitness routine you really enjoy, making it fun, and finding little ways to make smarter diet choices—adding up over time【来源3】. He also said to find something that motivates you and makes you happy, whether large or tiny, to help conquer struggles【来源4】. This was given in earlier conversations, not the current session.

裁判理由：The prediction discusses fitness routines and diet choices, not the gold answer's advice to keep trying new things until something sparks excitement.

**conv-49/161，类别5：What habit is Evan trying to change in terms of diet?**

Gold：Not mentioned in the conversation

SimpleMem：Not mentioned in the conversation

MemNexus：B. consuming soda and candy

Evan said he was trying to eat less processed food and sugary snacks, and he specifically mentioned loving ginger snaps while cutting back on sugary snacks【来源2】.

裁判理由：The gold answer states the habit is not mentioned, but the prediction asserts a specific habit (consuming soda and candy), which contradicts the gold answer.

## 方法限制与下一步

这100题与旧题目零重叠，但使用同一10段对话，因而不是对话级独立保留集；MemNexus 的结构和错误分析曾基于旧题设计。LoCoMo 问答成绩只能支持记忆回答结论，ICAPS 的规划贡献仍须用执行、约束违反和重规划成功率检验。

建议先对错题按‘来源未召回 / 来源已齐但推理错误 / 主体类型时间错配 / 裁判或 gold 争议’作人工复核，再做事件图、问题路由、typed negative gate 的逐组件消融。将决策阈值固定在开发集后，以新的对话或 LongMemEval-S 完整历史验证。

原始逐题输出、来源、调用、用量、裁判理由和源码哈希见本次 JSON 与同名 `_ledgers` 目录。
逐题错误阶段表见 `result/LOCOMO_100_DISJOINT_V2_ERRORS.md`；原始证据与 gold 争议核查见 `result/LOCOMO_100_DISJOINT_V2_DATA_QUALITY.md`；按本轮瓶颈排序的算法和架构方案见 `result/LOCOMO_100_DISJOINT_V2_RESEARCH_PLAN.md`。
