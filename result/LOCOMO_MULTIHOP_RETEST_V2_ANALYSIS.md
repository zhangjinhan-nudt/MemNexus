# MemNexus 多跳类别13题真实模型复测

这13题来自上一轮 LoCoMo 对比的类别1，题目与旧版逐题相同，已经用于错误分析，属于开发集复测。重新从相同 D1–D5 原始对话独立构建 MemNexus 记忆，并由相同 DeepSeek 裁判评分。

旧版 MemNexus 6/13，新版 8/13，原实验 SimpleMem 9/13。旧错→新对 3 题，旧对→新错 1 题。

最终来源检齐全部官方 gold turn 的题数 5→9。问答输入 tokens 56,210→90,300；provider attempts 26→26；问答模型费用估算 ¥0.1691→¥0.2499；P50 2.722s→3.260s，P95 3.951s→3.832s。

## 逐题原始对照

| ID | 问题 / gold | 旧版→新版 | gold 来源覆盖 | 裁判理由 |
|---|---|---|---|---|
| conv-26/7 | What is Caroline's relationship status?<br>Single | ✓→✓ | 1/2→2/2 | The prediction states Caroline is a single parent, which is semantically equivalent to being single and matches the gold answer. |
| conv-26/11 | Where did Caroline move from 4 years ago?<br>Sweden | ✓→× | 2/2→1/2 | The prediction does not provide the gold answer 'Sweden' and instead claims the country is not stated. |
| conv-41/3 | What type of volunteering have John and Maria both done?<br>Volunteering at a homeless shelter | ×→✓ | 1/2→2/2 | The prediction states both John and Maria volunteered at a homeless shelter, matching the gold answer. |
| conv-41/7 | What items des John mention having as a child?<br>A doll, a film camera | ✓→✓ | 2/2→2/2 | Prediction includes both gold items: a film camera and a doll. |
| conv-42/10 | What emotions is Joanna feeling about  the screenplay she submitted?<br>Relief, excitement,worry, hope,anxiety. | ×→× | 2/2→2/2 | The prediction omits 'hope', which is a core emotion listed in the gold answer. |
| conv-42/11 | What is Joanna allergic to?<br>Most reptiles,animals with fur,cockroaches, dairy | ×→× | 2/3→2/3 | The prediction omits dairy, which is part of the gold answer, and adds unsupported details about reaction severity. |
| conv-43/11 | What sports does John like besides basketball?<br>surfing | ×→× | 0/4→1/4 | The prediction fails to provide the gold answer 'surfing' and instead states no other sport is known, contradicting the gold answer. |
| conv-43/13 | What does Tim do to escape reality?<br>Read fantasy books. | ✓→✓ | 2/2→2/2 | The prediction includes the gold answer (reading fantasy books) and adds related details without contradicting it. |
| conv-44/23 | What are some problems that Andrew faces before he adopted Toby?<br>Finding the right dog and pet-friendly apartments close to open spaces | ✓→✓ | 2/4→2/4 | The prediction captures the gold answer's core problems: searching for a dog (finding the right dog) and city living making it hard to walk dogs in the park (pet-friendly apartments close to open spaces). |
| conv-47/5 | What are John and James' favorite games?<br>John's favorite game is CS:GO, and James's is Apex Legends. | ×→✓ | 1/2→2/2 | The prediction states John's favorite game is CS:GO and James's is Apex Legends, matching the gold answer; the extra note about no shared favorite does not contradict the core facts. |
| conv-47/8 | How many pets does James have?<br>Three dogs. | ×→✓ | 1/3→3/3 | Prediction states James has three pets, matching the gold answer's count of three dogs; additional details do not contradict the core fact. |
| conv-49/15 | What kind of unhealthy snacks does Sam enjoy eating?<br>soda, candy | ✓→✓ | 1/1→1/1 | Prediction includes both gold items (soda, candy) and matches the question's intent. |
| conv-50/5 | What are Dave's dreams?<br>open a car maintenance shop, work on classic cars, build a custom car from scratch | ×→× | 0/2→2/2 | The prediction omits the gold answer's core dream of opening a car maintenance shop and adds an unsupported dream about seeing favorite artists live. |

## 净变化案例

### 退步：conv-26/11

问题：Where did Caroline move from 4 years ago?

Gold：Sweden

旧版：Caroline moved from her home country, Sweden【来源6】. The move is referenced as having happened about four years before the conversation, though the evidence does not give an exact move date【来源1】.

新版：Caroline moved from her home country; the specific country isn’t stated in the available evidence【来源1】.

裁判：The prediction does not provide the gold answer 'Sweden' and instead claims the country is not stated.

### 修复：conv-41/3

问题：What type of volunteering have John and Maria both done?

Gold：Volunteering at a homeless shelter

旧版：No shared type of volunteering is supported by the evidence. Maria has volunteered at a homeless shelter【来源1】【来源7】, including a kids’ event there【来源5】. The evidence does not show John doing any volunteering.

新版：John and Maria have both volunteered at a homeless shelter — Maria volunteered there herself【来源1】【来源4】, and John's service-focused online group went to a homeless shelter to give out food and supplies【来源8】.

裁判：The prediction states both John and Maria volunteered at a homeless shelter, matching the gold answer.

### 修复：conv-47/5

问题：What are John and James' favorite games?

Gold：John's favorite game is CS:GO, and James's is Apex Legends.

旧版：John's favorite game is CS:GO【来源1】. There's no record of James's favorite game; he's mentioned getting into RPGs and strategy games, but that's not stated as a favorite【来源2】.

新版：John's favorite game is CS:GO【来源1】, and James's favorite game is Apex Legends【来源9】. No game is recorded as a shared favorite for both.

裁判：The prediction states John's favorite game is CS:GO and James's is Apex Legends, matching the gold answer; the extra note about no shared favorite does not contradict the core facts.

### 修复：conv-47/8

问题：How many pets does James have?

Gold：Three dogs.

旧版：I don't have any record of James owning pets, so I can't give a number. The evidence mentions James proposing a dog-walking and pet-care app【来源8】and discussing pets generally, but nothing states that James has pets of his own.

新版：James has three pets: two dogs, Max and Daisy【来源1】, and a puppy named Ned, adopted from a shelter in Stamford【来源12】.

裁判：Prediction states James has three pets, matching the gold answer's count of three dogs; additional details do not contradict the core fact.


## 解释边界

离线来源回放使用旧记忆快照，仅验证选择算法；本次真实模型复测重新建库，抽取和裁判存在随机性。准确率变化不能全部归因于一个检索组件。题目已用于设计，不能报告为独立泛化结果。后续要在新的对话与任务上固定版本验证。
