# 100题实验：断点已恢复，全部完成

2026-10-13：用户确认充值后，`locomo_100_collection_20261013_v3.json`已完成最后conv-50的6道配对题。最终100/100题，SimpleMem 73题正确、MemNexus 68题正确。审计确认200条非空回答及裁判标签、20个独立非空记忆库、30份账本、41个源码哈希，usage对账通过；有效主运行1407次请求无provider错误。无需再启动续跑或重复付费测试。

最终材料：`LOCOMO_100_COLLECTION_20261013_ANALYSIS.md`、`LOCOMO_100_COLLECTION_20261013_FINDINGS.md`、`locomo_100_collection_20261013_comparison.png`及`_detail.png`。有效100题含裁判估算¥15.5758；连同两次中断、排除复用账本重复计费后，有usage费用估算¥23.6221，含未知请求预留上界¥27.4350。均不是账户账单。

以下保留充值前的断点记录和复现命令，描述的是当时状态；当前已完成。

固定题目：`result/locomo_100_cohort_collection_20261013.json`。

修复后主运行：`result/locomo_100_collection_20261013_v2.json`，曾因 DeepSeek HTTP 402 Insufficient Balance 中断。

已完整完成 conv-26、conv-30、conv-41、conv-42、conv-43、conv-44、conv-47、conv-48、conv-49，合计94道配对题、188条裁判结果、18个非空独立记忆库。剩余 conv-50 的6题全部为此前两组未测题。失败的 conv-50 建库不计入质量指标。

首次运行 `locomo_100_collection_20261013.json` 因被拒绝候选的主体校验错误中断；其结果不与主运行混合。修复后282项回归通过，随后启动了整组重跑。

充值后从工作区根目录运行以下命令，无需重跑前94题：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUNBUFFERED='1'
SimpleMem/.venv/Scripts/python.exe experiments/run_collection_100.py --resume-from result/locomo_100_collection_20261013_v2.json --output result/locomo_100_collection_20261013_v3.json
```

runner 核验题目哈希及算法源码哈希，再复制完成对话的原始回答、裁判和账本。只从相同原始 D1–D5 重新构建 conv-50 的两套记忆库、回答6题并评分。不得把本地原型或旧试验的答案补作这6题的结果。

完成后生成整组图表（两次中断均保留，已复用的94题账本不会重复计费）：

```powershell
SimpleMem/.venv/Scripts/python.exe experiments/analyze_collection_100.py --run result/locomo_100_collection_20261013_v3.json --aborted-run result/locomo_100_collection_20261013.json --aborted-run result/locomo_100_collection_20261013_v2.json
```

本地 Qwen 服务当前已启动在127.0.0.1:8099；如果之后退出，需要恢复同一服务，再通过预检核对 query/document 编码。

注意：失败请求的预留费用上界不等于账单扣款。conv-50 的失败账本含 ¥3.8128656 的无 usage 请求预留上界；有返回 usage 的该阶段费用估算为 ¥0.3816216。完整94题及裁判估算 ¥14.2986768，首次中断试验估算 ¥7.6647；已有返回 usage 对应的合计估算约 ¥22.3450，包含预留失败上界的保守值约 ¥26.1579。实际账单可能因缓存、计价和失败请求不收费而更低。
