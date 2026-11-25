# MemNexus

面向长程会话问答的记忆系统：结构化记忆定位信息，原文回读恢复语境，模型负责语义建模，确定性算法负责来源核对及显式计算。

- [2026-11-25 冻结版本与测试准备](releases/2026-11-25/README.md)
- [近期实验与可视化](reports/README.md)
- [后端使用说明](MemNexus/memory-system/README.md)
- [系统架构](MemNexus/memory-system/ARCHITECTURE.md)
- [实验工具说明](experiments/README.md)
- [SimpleMem 固定版本与本地适配](baselines/simplemem/README.md)

原始数据集、已建记忆库、API凭据和完整调用账本不纳入Git。近期公开报告单独保存在 `reports/`；本机完整实验资料仍位于 `result/`。实验协议应记录源码、模型、数据与配置版本，并使用独立输出目录。

首次拉取基线：

```bash
git submodule update --init SimpleMem
```

随后按 `baselines/simplemem/README.md` 恢复经过审计的适配。不要向SimpleMem上游推送本地修改。
