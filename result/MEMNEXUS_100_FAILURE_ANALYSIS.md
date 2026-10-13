# MemNexus 100题逐题错误归因与系统改进建议

本报告只分析最终运行中被统一裁判标为 WRONG 的25题；没有改写标签。完整100题状态在 JSON 的 `all_transitions` 中。至少一题存在 LoCoMo gold 与原文日期冲突，因此‘错误’是评分结果，不必然等于客观事实错误。

MemNexus 优化后 75/100，SimpleMem 81/100，优化前 MemNexus 65/100。逐题配对：两者都对69题、仅 MemNexus 对6题、仅 SimpleMem 对12题、两者都错13题。

25 道错题中，21 道在初始检索后仍有 missing_info，19 道在最终证据选择后仍有 missing_info；21 道实际选中过原文补充，16 道包含时间推导，0 道触发了现有 attribution_conflicts 检测。这说明当前瓶颈主要是覆盖和推理结构，不是已经被现有归属检查捕获的显式冲突。

## 错误模式统计

| 主归因 | 题数 | 含义 |
|---|---:|---|
| 主体/范围约束失败 | 2 | 把别的主体、类型或时间范围的事实迁移进答案。 |
| 受限地理推理拒答 | 2 | 确定的城市—国家关系没有通过知识层补全。 |
| 压缩/实体属性丢失 | 1 | 原子身份或属性没有保留到可检索记忆。 |
| gold/时间标注冲突 | 1 | 原始证据与 gold 不一致，不能用来诊断算法。 |
| 多实体关系组合 | 1 | 未完成 both/common/shared 等关系聚合。 |
| 多跳关系检索/槽位错配 | 1 | 找到邻近事实，却没有覆盖问题需要的关系链。 |
| 因果链覆盖不足 | 1 | 有目标和动机，却遗漏决策前因。 |
| 集合答案不完整且过度添加 | 1 | 列表缺项，同时加入非必要细节。 |
| 隐含蕴含过度拒答 | 1 | 证据支持受限推理，却统一判为未知。 |
| 答案集合边界错误 | 1 | 属性列表出现额外项或漏项。 |
| 标题/实体关联丢失 | 1 | 描述和标题/实体没有连成一条证据链。 |
| 对抗题假阳性 | 1 | 类型或对象不一致的事实被当作答案。 |
| 常识桥接过度拒答 | 1 | 可审计的常识桥接被统一拒答。 |
| 实体/图像关联拒答 | 1 | 图像描述、实体和问题对象没有链接。 |
| 时间区间组合拒答 | 1 | duration、event date 和 anchor date 没有做区间运算。 |
| 检索/抽取覆盖损失 | 1 | gold evidence 没有形成可回答的原子事实。 |
| 实体关系消歧失败 | 1 | 同一实体的不同购买/访问事件没有区分。 |
| 事件计数与去重不足 | 1 | 没有先规范化事件再计数。 |
| 计划状态与已发生状态混淆 | 1 | planned、started、completed 状态没有按时间升级。 |
| 事件计数与证据合并不足 | 1 |  |
| 列表覆盖/原子记忆丢失 | 1 | 多项答案召回不足，missing_info 已暴露但未恢复。 |
| 事件日期与记录日期混淆 | 1 | 观察日期替换了事件发生区间。 |
| 对抗题假阳性/类型时间错配 | 1 | 类型和时间双重不匹配仍然被引用。 |

## 逐题分析

grounding 数字来自实际回执；missing_info 是检索启发式，不是逻辑证明。完整 prediction、judge reason 和证据保存在 JSON。

| 编号 | 题目 | Gold | 归因与系统修复 |
|---|---|---|---|
| 1. conv-26/4 | What is Caroline's identity? | Transgender woman | **压缩/实体属性丢失**：身份属性未进入可用证据，输出了邻近的收养事实。（sources 5→8；missing 1→1；raw 3；temporal 0）。措施：保留实体属性卡和原子事实；身份查询强制回溯来源。 |
| 2. conv-26/5 | When did Melanie run a charity race? | The sunday before 25 May 2023 | **gold/时间标注冲突**：原文写 last Saturday，日期锚点得到 2023-05-20；gold 写成 5 月 25 日前的 Sunday。（sources 6→7；missing 0→0；raw 3；temporal 2）。措施：先修正标注或在论文中剔除冲突题，不能用错误 gold 校准算法。 |
| 3. conv-26/154 | What type of individuals does the adoption agency Melanie is considering support? | Not mentioned in the conversation | **主体/范围约束失败**：问题主体是 Melanie，证据却把 Caroline 的 LGBTQ agency 事实作为答案。（sources 3→6；missing 1→1；raw 3；temporal 1）。措施：回答前执行主体、关系、时间范围的 entailment gate。 |
| 4. conv-26/155 | Why did Melanie choose the adoption agency? | Not mentioned in the conversation | **主体/范围约束失败**：检索到了 Caroline 的 agency 原因，但问题询问 Melanie；跨主体迁移未被阻止。（sources 3→6；missing 2→1；raw 3；temporal 1）。措施：建立 subject-relation-object 三元组和会话级指代绑定。 |
| 5. conv-30/2 | How do Jon and Gina both like to destress? | by dancing | **多实体关系组合**：证据包含 Gina 的 dance 和 Jon 的相关语境，但没有完成共同方式的交集验证。（sources 6→7；missing 1→1；raw 3；temporal 0）。措施：对 both/common 查询构建实体关系图并做交集验证。 |
| 6. conv-30/3 | What do Jon and Gina both have in common? | They lost their jobs and decided to start their own businesses. | **多跳关系检索/槽位错配**：召回舞蹈和互相支持，却没有覆盖两人失业并创业的关系链。（sources 8→8；missing 3→2；raw 3；temporal 0）。措施：把 shared 查询拆成实体对、事件对和共同转折事件。 |
| 7. conv-30/4 | Why did Jon decide to start his dance studio? | He lost his job and decided to start his own business to share his passion. | **因果链覆盖不足**：回答了 passion 和 teaching，但遗漏 lost his job 这一决策前因。（sources 6→8；missing 2→2；raw 3；temporal 1）。措施：对 why/how 查询构建 antecedent→decision→goal 因果链。 |
| 8. conv-30/5 | What Jon thinks the ideal dance studio should look like? | By the water, with natural light and Marley flooring | **集合答案不完整且过度添加**：遗漏 by the water，同时加入 downtown/bounce 等非核心细节。（sources 6→6；missing 2→1；raw 3；temporal 1）。措施：先做 required-fact checklist，再做 unsupported-fact veto。 |
| 9. conv-42/4 | What pets wouldn't cause any discomfort to Joanna? | Hairless cats or pigs,since they don't have fur, which is one of the main causes of Joanna's allergy. | **隐含蕴含过度拒答**：证据说明 fur 是过敏原因，系统没有完成 hairless cats/pigs 的受限推理。（sources 6→5；missing 2→1；raw 3；temporal 0）。措施：增加可审计的属性反事实规则推理。 |
| 10. conv-42/5 | What are Joanna's hobbies? | Writing, watchingmovies, exploringnature, hanging withfriends. | **答案集合边界错误**：加入 reading，遗漏 hanging with friends；列表没有精确对齐。（sources 6→8；missing 1→0；raw 3；temporal 0）。措施：属性列表做集合合并、去重、来源支持和完整性检查。 |
| 11. conv-42/88 | What is one of Joanna's favorite movies? | "Eternal Sunshineof the Spotless Mind" | **标题/实体关联丢失**：召回 favorite romantic drama 描述，却没有把标题与描述绑定。（sources 5→5；missing 0→0；raw 0；temporal 0）。措施：保留 image/text mention 的实体别名和跨 turn 链接。 |
| 12. conv-42/206 | What genre is Joanna's first novella? | Not mentioned in the conversation | **对抗题假阳性**：把电影 genre 迁移成 first novella genre，对象类型不一致。（sources 6→7；missing 1→1；raw 3；temporal 1）。措施：增加对象类型、事件类型和谓词一致性检查。 |
| 13. conv-43/3 | Would Tim enjoy reading books by C. S. Lewis or John Greene? | C. S.Lewis | **常识桥接过度拒答**：已有 fantasy/Harry Potter 线索，却统一拒答 C. S. Lewis 的受限推断。（sources 6→8；missing 1→0；raw 3；temporal 0）。措施：区分 observed、supported-inference 和 unknown。 |
| 14. conv-43/5 | Based on Tim's collections, what is a shop that he would enjoy visiting in New York city? | House of MinaLima | **实体/图像关联拒答**：召回 MinaLima 收藏，却没有链接到 House of MinaLima。（sources 8→8；missing 2→2；raw 3；temporal 1）。措施：建立 caption→实体→地点/商店的别名图。 |
| 15. conv-44/7 | How many years passed between Audrey adopting Pixie and her other three dogs? | three years | **时间区间组合拒答**：已有三只狗 for 3 years 和 Pixie adoption date，却没有计算间隔。（sources 8→8；missing 1→1；raw 3；temporal 0）。措施：把 duration、event date、anchor date 转为区间约束。 |
| 16. conv-47/0 | What are John's suspected health problems? | Obesity | **检索/抽取覆盖损失**：gold evidence 没有形成 John's health 原子事实，最终错误地说没有记录。（sources 6→9；missing 4→4；raw 3；temporal 0）。措施：低词面重叠属性查询增加原子事件 fallback，并测 evidence recall。 |
| 17. conv-48/5 | In what country did Jolene's mother buy her the pendant? | In France | **受限地理推理拒答**：识别出 Paris，却没有完成确定的 Paris→France 实体链接。（sources 6→6；missing 0→0；raw 0；temporal 2）。措施：接入冻结、可审计的城市—国家知识表。 |
| 18. conv-48/18 | In what country did Jolene buy snake Seraphim? | In France | **实体关系消歧失败**：知道 Paris 是另一物品的购买地，因而排除了 Seraphim 的对应事件。（sources 6→6；missing 1→1；raw 3；temporal 1）。措施：按 subject-object-event 三元组区分 purchase episode。 |
| 19. conv-48/19 | How many times has Jolene been to France? | two times | **事件计数与去重不足**：只确认一次 France mention，未合并全部访问/购买事件。（sources 8→7；missing 4→3；raw 3；temporal 4）。措施：事件规范化、排序、去重后执行 count 聚合。 |
| 20. conv-49/4 | Which hobby did Sam take up in May 2023? | painting | **计划状态与已发生状态混淆**：把 painting 计划/兴趣当作未发生，未完成状态升级。（sources 6→7；missing 1→1；raw 3；temporal 1）。措施：将 planned、started、completed 建成状态机。 |
| 21. conv-49/5 | Which country was Evan visiting in May 2023? | Canada | **受限地理推理拒答**：Jasper/Rockies 线索存在，却没有完成 Canada 的确定地域映射。（sources 6→6；missing 1→1；raw 0；temporal 3）。措施：使用审计过的地理实体层，不依赖自由常识生成。 |
| 22. conv-49/6 | How many roadtrips did Evan take in May 2023? | two | **事件计数与证据合并不足**：只数出一次 Jasper road trip，遗漏另一条 May trip 证据。（sources 8→8；missing 1→1；raw 3；temporal 1）。措施：count 查询先抽取事件全集再聚合。 |
| 23. conv-50/1 | What items did Calvin buy in March 2023? | mansion in Japan, luxury car Ferrari 488 GTB | **列表覆盖/原子记忆丢失**：只保留 red sports car，遗漏 mansion；missing_info 已提示不完整但没有有效补检索。（sources 6→5；missing 3→3；raw 0；temporal 4）。措施：列表问题启用 recall-first 预算和原文扫描。 |
| 24. conv-50/2 | When did Dave see Aerosmith perform live? | on the weekend before March 26, 2023 | **事件日期与记录日期混淆**：把 March 26 对话日期当成 Aerosmith 演出日期。（sources 6→7；missing 0→0；raw 3；temporal 1）。措施：同时保存 event_time、observation_time 和 validity_interval。 |
| 25. conv-50/158 | Which DJ was Dave's favorite at the music festival in April 2023? | Not mentioned in the conversation | **对抗题假阳性/类型时间错配**：问题问 April 的 DJ，却引用 March 前一周的 Aerosmith band。（sources 3→5；missing 1→2；raw 3；temporal 1）。措施：类型、主体和时间任一不匹配时只能答未提及。 |

## 系统性结论

1. 当前 grounding 主要解决‘来源恢复’，还没有解决‘问题结构上的可计算推理’。它能找到相关句子，却没有把主体—对象—事件—时间—状态组合成可计算结构。
2. MemNexus 的拒答策略过于保守。它减少了无证据幻觉，却把地理映射、状态升级、共同属性、持续时间、计数和受限常识推理也判成 unknown。
3. 压缩在减少记忆条目的同时丢失了列表成员、身份属性、标题关联和完整事件链。conv-50/1 明确出现 missing_info，但 fallback 没有找回第二项。
4. top-k 相似度选择不适合 all、both、how many、why、between 等问题；这些问题需要 recall-first、事件 join 和完整覆盖。
5. 对抗题缺少 typed negative evidence：看到 Aerosmith 或 LGBTQ 就回答，却没有同时检查主体、对象类型和时间范围。

## 建议的系统级改造

### A. 带类型的事件图记忆
每条事实保存 `subject, predicate, object, event_type, state, event_time_interval, observation_time, source_turn, confidence, scope`。列表、计数、关系和状态变化不能只保留自然语言摘要；高风险事实保留原子事件索引。

### B. 召回—推理—验证三阶段回答器
先按问题结构选择召回目标：属性、实体对关系、时间区间、因果链、集合、计数或对抗约束；再执行确定性 join、区间算术、集合运算和计数；最后检查主体、类型、时间、状态和来源链。困难问题允许更高 evidence budget 和最多两轮定向检索，简单问题保持低预算。

### C. 分级不确定性而非统一拒答
将结论分为 `observed`、`deterministically_entailed`、`bounded_inference` 和 `unknown`。城市—国家、明确 duration 运算、属性反事实和状态升级属于可审计推理；只有来源链无法形成时才拒答。

### D. 集合和计数的 recall-first 策略
遇到 names/items/how many/both/all 时，先生成候选事件全集，再按主体、时间和事件类型过滤、去重和计数。停止条件是槽位覆盖稳定或预算耗尽，而不是 top-k 已满。

### E. typed negative evidence 与主体锁
对每个结论执行主体一致、对象类型一致、时间区间相交、谓词一致四项检查。对抗题只有确认不存在满足四项约束的事实后才能回答 Not mentioned。

### F. 以可回答性约束压缩
用 masking/recovery 测试衡量压缩质量：删除某条压缩事实后，哪些问题变得不可恢复。压缩目标应优先保留列表成员、否定、时间锚点、主体、状态和证据引用，而不是只追求条目数。

## 下一轮必须做的验证

| 实验 | 目的 | 指标 |
|---|---|---|
| 事件图 on/off | 验证表示层 | accuracy、官方F1、evidence recall、coverage、tokens |
| recall-first on/off | 验证集合/计数/多跳 | set F1、count accuracy、multi-hop accuracy、P95 |
| 分级推理 on/off | 验证拒答策略 | correct、unsupported claim rate、abstention rate |
| typed negative gate on/off | 验证对抗边界 | adversarial accuracy、false-positive rate、主体错配率 |
| 预算曲线 | 验证增 token 是否值得 | accuracy vs tokens/latency/cost |
| 未调参保留集 | 验证普适性 | LongMemEval-S 与未使用 LoCoMo |
| belief-state/planning | 对接 ICAPS | plan success、replanning success、constraint violation、search nodes |

建议门槛：在未调参保留集上总体正确率至少高于 SimpleMem 3–5 个百分点；P95 和总费用仍低于 SimpleMem；多跳、集合、计数、时间四类不得显著退化；对抗题 false-positive rate 不高于 SimpleMem。当前100题开发集尚未达到这些门槛。

## 文件

- 原始运行：`result\locomo_100_optimized_20261012_v5.json`
- 机器可读逐题分析：`result/MEMNEXUS_100_FAILURE_ANALYSIS.json`
- 完整逐题转换：`result/locomo_100_optimized_analysis.json`
