# MemNexus 多跳类别13题真实模型复测

这13题来自上一轮 LoCoMo 对比的类别1，题目与旧版逐题相同，已经用于错误分析，属于开发集复测。重新从相同 D1–D5 原始对话独立构建 MemNexus 记忆，并由相同 DeepSeek 裁判评分。

旧版 MemNexus 6/13，新版 7/13，原实验 SimpleMem 9/13。旧错→新对 1 题，旧对→新错 0 题。

最终来源检齐全部官方 gold turn 的题数 5→7。问答输入 tokens 56,210→67,163；provider attempts 26→26；P50 2.722s→2.942s。

## 逐题原始对照

| ID | 问题 / gold | 旧版→新版 | gold 来源覆盖 | 裁判理由 |
|---|---|---|---|---|
| conv-26/7 | What is Caroline's relationship status?<br>Single | ✓→✓ | 1/2→2/2 | Prediction states Caroline is single, matching the gold answer; the added detail does not contradict it. |
| conv-26/11 | Where did Caroline move from 4 years ago?<br>Sweden | ✓→✓ | 2/2→2/2 | The prediction states Caroline moved from Sweden, matching the gold answer. |
| conv-41/3 | What type of volunteering have John and Maria both done?<br>Volunteering at a homeless shelter | ×→✓ | 1/2→2/2 | The prediction states both volunteered at a homeless shelter, matching the gold answer. |
| conv-41/7 | What items des John mention having as a child?<br>A doll, a film camera | ✓→✓ | 2/2→2/2 | Prediction includes both gold items: a film camera and a doll. |
| conv-42/10 | What emotions is Joanna feeling about  the screenplay she submitted?<br>Relief, excitement,worry, hope,anxiety. | ×→× | 2/2→2/2 | The prediction omits 'hope,' which is a required core emotion in the gold answer. |
| conv-42/11 | What is Joanna allergic to?<br>Most reptiles,animals with fur,cockroaches, dairy | ×→× | 2/3→2/3 | The prediction omits dairy, which is a core fact in the gold answer. |
| conv-43/11 | What sports does John like besides basketball?<br>surfing | ×→× | 0/4→0/4 | The prediction does not provide the gold answer 'surfing' and instead states no record of another sport preference, which contradicts the gold answer. |
| conv-43/13 | What does Tim do to escape reality?<br>Read fantasy books. | ✓→✓ | 2/2→2/2 | The prediction includes the gold answer (reading fantasy books) and adds related details without contradicting it. |
| conv-44/23 | What are some problems that Andrew faces before he adopted Toby?<br>Finding the right dog and pet-friendly apartments close to open spaces | ✓→✓ | 2/4→2/4 | The prediction identifies the same core problems as the gold answer: finding the right dog (search difficulties, smaller dog suited to apartment) and pet-friendly housing near open spaces (apartment, near park/woods). Although it adds caveats about lacking evidence for Toby specifically, it does not contradict the gold answer and covers the required facts. |
| conv-47/5 | What are John and James' favorite games?<br>John's favorite game is CS:GO, and James's is Apex Legends. | ×→× | 1/2→1/2 | The prediction omits James's favorite game (Apex Legends), which is a required core fact from the gold answer. |
| conv-47/8 | How many pets does James have?<br>Three dogs. | ×→× | 1/3→2/3 | Gold answer states three dogs, but prediction says two pets (two dogs), contradicting the count. |
| conv-49/15 | What kind of unhealthy snacks does Sam enjoy eating?<br>soda, candy | ✓→✓ | 1/1→1/1 | Prediction includes both gold items (soda, candy) and matches the question's intent. |
| conv-50/5 | What are Dave's dreams?<br>open a car maintenance shop, work on classic cars, build a custom car from scratch | ×→× | 0/2→1/2 | Prediction omits two of the three gold-answer dreams (open a car maintenance shop, work on classic cars) and adds an unsupported dream about seeing favorite artists live. |

## 解释边界

离线来源回放使用旧记忆快照，仅验证选择算法；本次真实模型复测重新建库，抽取和裁判存在随机性。准确率变化不能全部归因于一个检索组件。题目已用于设计，不能报告为独立泛化结果。后续要在新的对话与任务上固定版本验证。
