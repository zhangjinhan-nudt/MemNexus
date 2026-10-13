# LoCoMo 第二组100题：已核实的数据与裁判争议

本文件只列出对照官方 `benchmark-data/official/locomo/locomo10.json` 的原始 turn 和会话日期后，可以明确指出的冲突。它不修改实验中的 gold、预测或裁判标签，也不生成调整后的正确率。

| 题目 | 原始证据 | 争议 |
|---|---|---|
| conv-41/68 | D1:3，2022-12-17，Maria 说刚开始 aerial yoga | 题目询问“December 2023”，gold 只给“aerial yoga”。动作类型可答，题目年份与来源不一致。MemNexus 指出时间冲突仍被判对。 |
| conv-41/69 | D2:1，2022-12-22，Maria 说“yesterday”捐出旧车，即 2022-12-21 | 题目询问“December 2023”，gold 为“old car”。MemNexus 给出车与 2022 年日期，被裁判判错；原文没有 2023 年这次捐赠。 |
| conv-43/6 | D3:1，2023-07-16，John 说“Last week I scored 40 points” | gold 为“June 2023”。通常的“上周”与 7 月 16 日锚点对应 7 月，不能推出 6 月。两套回答均给 7 月而被判错。 |
| conv-43/74 | D1:16 由 Tim 发言：“We'll be discussing ... characters, spells, and magical creatures.” | 题目却问 John 的 fan project。MemNexus 指出主体是 Tim，被裁判判错；此题需要结合更宽上下文人工复核项目归属。 |
| conv-44/5 | D4:1，2023-05-03，Audrey 说“Last week ... went on a hike”并见到蜂鸟 | gold 为“first week of May 2023”。在通常的周历解释中，上周在 4 月下旬；至少 gold 不能由这条 turn 稳定推出。 |
| conv-47/72 | D3:2 James 说自己正在学乐器，配图为鼓组；D3:3 John 说“I play drums too” | 题目问 John “is learning to play”的乐器。来源区分 James 正在学习与 John 已会演奏；gold“Drums”遗漏了这个状态差异。 |

这些争议不意味着其他题目均无标注问题。统计结论沿用原始裁判结果；若要发表，应由不知道系统名称的两名人工审阅者按统一协议复核争议题，记录一致率，并同时报告原始分数与复核后的敏感性分析。
