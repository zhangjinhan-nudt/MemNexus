# 100题对比测试一键运行

入口文件是工作区根目录的 `run_benchmark100.cmd`。它会固定题组、检查配置、启动或检查本地 Qwen embedding 服务，分别从相同的原始 LoCoMo 对话为 SimpleMem 和 MemNexus 建立独立记忆库，回答 100 道题，调用统一裁判，最后生成 HTML、Markdown、JSON 和 PNG/SVG 图表。

默认配置在 [benchmark100_config.json](C:/Users/zhangjinhan/Desktop/10-agent/experiments/benchmark100_config.json)。当前默认预算是 ¥50，可在运行前修改 `budget_rmb`；这只是上限，不是预计费用。预算包含本次运行的主模型、裁判、失败重试和格式修复；失败请求的预留上界也会计入。脚本从现有私有配置读取 API Key，不主动输出密钥，也不会将私有配置文件加入源码快照。

## 一键运行

在资源管理器中双击：

```text
run_benchmark100.cmd
```

也可以在 PowerShell 中执行：

```powershell
cd C:\Users\zhangjinhan\Desktop\10-agent
.\run_benchmark100.cmd
```

脚本会显示当前对话、阶段和累计费用。运行期间不要打开第二个同名窗口；脚本使用锁文件避免并发写同一结果目录。

## 结果位置

每次运行会创建：

```text
result\benchmark100\YYYYMMDD_HHMMSS_microseconds\
```

完成后重点查看：

```text
report.html          浏览器打开的中文报告
report.md            Markdown报告
comparison.png      主对比图
comparison.svg      矢量图
summary.json        指标、配对统计和审计结果
results.json        逐题答案和阶段数据
run.json            运行状态、配置哈希、预算和账本
attempts\           每个建库、问答和裁判阶段的调用账本
```

脚本完成后会自动打开 `report.html`。如果没有自动打开，可以双击该文件。

## 无付费预检

正式运行前可以只检查题组、API 配置、Python 依赖和报告依赖：

```powershell
SimpleMem\.venv\Scripts\python.exe experiments\benchmark100.py --dry-run
```

还可以检查并启动本地 Qwen 服务、验证 query/document 编码一致性，但不调用主模型和裁判：

```powershell
SimpleMem\.venv\Scripts\python.exe experiments\benchmark100.py --preflight-only --no-open
```

预检会自动使用 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1` 启动已有模型缓存；如果本机没有 Qwen3-Embedding-0.6B 缓存，会停止并说明原因。

## 中断后继续

再次运行同一个 `.cmd` 会自动寻找同一代码、题组和配置下未完成的结果并继续。检查点粒度为“一个系统的一段对话（建库＋该段全部问答）”；裁判逐题保存。已完成阶段不会重复调用；未完成的系统/对话阶段可能需要重新建库并回答该段问题，旧失败用量仍计费。若代码或实验配置变化，自动入口会新建实验；显式指定不兼容目录续跑会被拒绝。仅提高预算不影响续跑兼容性。强制新建可运行：

```powershell
SimpleMem\.venv\Scripts\python.exe experiments\benchmark100.py --new --no-open
```

指定某个结果目录续跑：

```powershell
SimpleMem\.venv\Scripts\python.exe experiments\benchmark100.py --resume result\benchmark100\20261015_120000_123456
```

报告重新生成而不调用 API：

```powershell
SimpleMem\.venv\Scripts\python.exe experiments\benchmark100.py --report-only result\benchmark100\20261015_120000_123456
```

## 版本选择

`benchmark100_config.json` 的 `variant` 可取：

- `configured`：使用当前 `.env` 和进程环境的实际配置。当前代码默认关闭新语义执行；运行前控制台会显示实际开关。
- `stable`：明确关闭统一语义执行，使用当前代码的既有回答路径。这不是还原历史78分版本，也不保证重现78分。
- `semantic`：开启统一语义执行和前提驱动补检索，用于 ACL 方案实验。

推荐先用 `configured` 完成一轮基线，再将配置改为 `semantic` 新建实验。`semantic`当前是实验分支，上一轮曾出现正确率退步，开关不会自动改善算法。不要在同一结果目录里改变 `variant`。

每轮都会记录真实 provider attempts、输入/输出 tokens、响应 usage、重试、错误和费用估算。空答案、失败窗口、无效裁判 JSON、账本不一致或源码在运行中改变，都会使测试停止；脚本不会用缺失题目补成 100 题。

## 解释结果时注意

固定 100 题仍是开发题组，来自此前研究过的对话，不能作为完整 LoCoMo 或 conversation-held-out 泛化结果。报告会同时提供正确率、分题型、token、费用、延迟、McNemar 和对话聚类区间。ACL 正式实验应另行封存 LongMemEval-S 或未参与开发的数据。
