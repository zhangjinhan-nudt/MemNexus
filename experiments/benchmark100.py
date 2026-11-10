"""One-command fresh paired LoCoMo development test with atomic stage checkpoints.

No paid calls on import, --dry-run, --preflight-only, or --report-only.
Default runs both systems from original dialogue; it never reuses a historical
baseline prediction. Resume requires identical code, data, and model settings.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "MemNexus/memory-system/src"))
CONFIG = Path(__file__).with_name("benchmark100_config.json")
SYSTEMS = ("SimpleMem", "MemNexus")
FIELDS = ("provider_attempts", "provider_errors", "input_tokens", "output_tokens", "cost_rmb")
# 逐字移植自 SimpleMem 原论文 Appendix A.4（Listing 4: LLM-as-a-Judge Evaluation
# Prompt）。口径是"宽松判分"：答案更长没关系，只要触及同一主题就算 CORRECT；
# 时间只要指向同一日期/时段就算 CORRECT，格式不同（"May 7th" vs "7 May"）也算对。
# 唯一改动是把三个模板占位符换成"下方 JSON 的字段名"，以及显式要求 reason 键
# （parse_judgment 需要它）。判分倾向不得再改回 strict。
# Canonical text: paper_protocol.JUDGE_INSTRUCTION (SimpleMem Appendix A.4).  This
# copy differs in exactly two ways, both deliberate: the question/gold/prediction
# are passed as a JSON payload instead of inline, and the judge must also return a
# "reason" so a label can be audited.  Judge output is a DIAGNOSTIC here — the
# paper uses this judge for LongMemEval-S only, never for LoCoMo.
JUDGE_INSTRUCTION = (
    "Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.\n"
    "You will be given the following data:\n"
    "(1) a question (posed by one user to another user),\n"
    "(2) a 'gold' (ground truth) answer,\n"
    "(3) a generated answer\n"
    "which you will score as CORRECT/WRONG.\n"
    "\n"
    "The point of the question is to ask about something one user should know about the other\n"
    "user based on their prior conversations.\n"
    "The gold answer will usually be a concise and short answer that includes the referenced\n"
    "topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading -\n"
    "as long as it touches on the same topic as the gold answer, it should be counted as\n"
    "CORRECT.\n"
    "For time related questions, the gold answer will be a specific date, month, year, etc. The\n"
    "generated answer might be much longer or use relative time references (like \"last\n"
    "Tuesday\" or \"next month\"), but you should be generous with your grading - as long as\n"
    "it refers to the same date or time period as the gold answer, it should be counted as\n"
    "CORRECT. Even if the format differs (e.g., \"May 7th\" vs \"7 May\"), consider it CORRECT\n"
    "if it's the same date.\n"
    "Now it's time for the real question. The question, gold answer and generated answer are\n"
    "given in the JSON below as \"question\", \"gold_answer\" and \"prediction\".\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with\n"
    "CORRECT or WRONG.\n"
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation\n"
    "script.\n"
    "Just return the label CORRECT or WRONG in a json format with the key as \"label\", and put\n"
    "your one-sentence explanation under the key \"reason\"."
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def progress(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def load_config(path):
    value = read(path)
    if not math.isfinite(value["budget_rmb"]) or value["budget_rmb"] <= 0:
        raise ValueError("budget_rmb 必须为正数。")
    if value["variant"] not in {"configured", "stable", "semantic"}:
        raise ValueError("variant 应为 configured、stable 或 semantic。")
    if not 1 <= value["embedding_start_timeout_seconds"] <= 600:
        raise ValueError("embedding_start_timeout_seconds 必须在1–600之间。")
    return value


def load_cohort(path):
    cohort = read(path)
    dataset_path = ROOT / cohort["dataset"]
    if digest(dataset_path) != cohort["dataset_sha256"]:
        raise ValueError("数据集哈希与固定题组不一致，停止测试。")
    dataset = read(dataset_path)
    keys = [(q["sample_id"], q["qa_index"]) for q in cohort["questions"]]
    if len(keys) != 100 or len(set(keys)) != 100 or cohort["question_count"] != 100:
        raise ValueError("题组必须恰好包含100个不同问题。")
    selection = hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()
    if selection != cohort["selection_sha256"]:
        raise ValueError("题目选择哈希不一致。")
    for q in cohort["questions"]:
        sample = dataset[q["sample_index"]]
        qa = sample["qa"][q["qa_index"]]
        if (sample["sample_id"] != q["sample_id"] or qa["category"] != q["category_id"]
                or qa["question"] != q["question"]):
            raise ValueError("题组与官方问题不匹配。")
        if not qa["question"].strip() or (qa["category"] != 5 and not str(qa.get("answer", "")).strip()):
            raise ValueError("存在空问题或空答案。")
        if qa["category"] == 5 and not qa.get("adversarial_answer"):
            raise ValueError("对抗题缺少原始干扰答案。")
    return cohort, dataset


def configure(value):
    # Import loads the existing private .env. Never print or snapshot credentials.
    from agent_memory.settings import Settings
    if value["variant"] != "configured":
        os.environ["MEMORY_SEMANTIC_EXECUTION"] = str(value["variant"] == "semantic").lower()
        if value["variant"] == "semantic":
            os.environ["MEMORY_PREMISE_RETRIEVAL"] = "true"
    os.environ["MEMORY_INPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "2.4"
    os.environ["MEMORY_OUTPUT_PRICE_RMB_PER_MILLION_TOKENS"] = "9.6"
    os.environ["EXPERIMENT_TRUNCATION_MAX_TOKENS"] = "8192"
    sys.path.insert(0, str(ROOT / "SimpleMem"))
    config = importlib.import_module("config")
    settings = Settings()
    # The model name is a configuration value, not a harness constant. Requiring
    # the three roles to agree catches the real failure (a judge on a different
    # model) without forcing a code edit whenever the model is upgraded.
    model_names = {
        "MemNexus": os.getenv("MEMORY_MODEL_NAME", "").strip(),
        "SimpleMem": str(getattr(config, "LLM_MODEL", "")).strip(),
        "judge": str(getattr(config, "JUDGE_MODEL", "")).strip(),
    }
    checks = {
        "三个模型名已填写": all(model_names.values()),
        "三个模型名一致": len(set(model_names.values())) == 1,
        "主模型及裁判Key已填写": bool(os.getenv("MEMORY_MODEL_API_KEY") and config.OPENAI_API_KEY
                                           and (config.JUDGE_API_KEY or config.OPENAI_API_KEY)),
        "温度为0": config.TEMPERATURE == 0,
        "窗口20步长5": config.WINDOW_SIZE == 20 and config.OVERLAP_SIZE == 15,
        "SimpleMem检索上限20": config.SEMANTIC_TOP_K == 20,
        "Qwen查询编码": os.getenv("MEMORY_EMBEDDING_ENCODING") == "query_prompt",
        "Qwen维度1024": config.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-0.6B" and config.EMBEDDING_DIMENSION == 1024,
        "关闭思考模式": not config.ENABLE_THINKING and not config.JUDGE_ENABLE_THINKING
                        and os.getenv("MEMORY_MODEL_DISABLE_THINKING", "true").lower() not in {"0", "false", "no", "off"},
        "输出上限4096": config.MAX_TOKENS == 4096,
    }
    if not all(checks.values()):
        failed = "、".join(k for k, v in checks.items() if not v)
        raise ValueError(f"配置预检失败：{failed}（实际模型名：{model_names}）")
    config.INPUT_PRICE_RMB_PER_MILLION_TOKENS = 2.4
    config.OUTPUT_PRICE_RMB_PER_MILLION_TOKENS = 9.6
    from run_locomo_pilot import CATEGORY_NAMES
    # Include only public settings. config.py/.env must not enter the source zip.
    public = {k:getattr(config,k) for k in (
        "LLM_MODEL", "JUDGE_MODEL", "TEMPERATURE", "MAX_TOKENS", "WINDOW_SIZE", "OVERLAP_SIZE",
        "SEMANTIC_TOP_K", "KEYWORD_TOP_K", "STRUCTURED_TOP_K", "ENABLE_PLANNING", "ENABLE_REFLECTION",
        "MAX_REFLECTION_ROUNDS", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION", "EMBEDDING_QUERY_PROMPT")}
    public.update(semantic_execution=settings.semantic_execution_enabled,
                  premise_retrieval=settings.premise_retrieval_enabled,
                  reread=settings.reread_enabled,
                  open_domain_compat=settings.open_domain_compat,
                  semantic_max_answer_calls=settings.semantic_max_answer_calls,
                  collection_queries=settings.collection_queries_enabled,
                  model=model_names["SimpleMem"], prices_rmb_per_million=[2.4,9.6], answer_cache=False,
                  planner_mode="model", output_language="en", embedding_encoding="query_prompt",
                  memnexus_pending_turns=5, memnexus_context_turns=15,
                  max_session_policy="cohort.max_session", sequential=True)
    config_paths = [ROOT/"SimpleMem/config.py", ROOT/"MemNexus/memory-system/.env"]
    public["private_configuration_hashes"] = {str(p.relative_to(ROOT)):digest(p) for p in config_paths if p.exists()}
    # Environment overrides need to invalidate resume too; only publish a digest.
    relevant_env = {k:v for k,v in os.environ.items() if k.startswith("MEMORY_")}
    public["memory_environment_sha256"] = hashlib.sha256(json.dumps(relevant_env,sort_keys=True).encode()).hexdigest()
    return public


def source_files():
    files = [*(ROOT/"MemNexus/memory-system/src/agent_memory").glob("*.py"),
             *(ROOT/"SimpleMem/simplemem").rglob("*.py"), ROOT/"SimpleMem/main.py",
             ROOT/"experiments/run_locomo_pilot.py", Path(__file__),
             ROOT/"experiments/benchmark100_report.py", ROOT/"experiments/analyze_optimized_comparison.py",
             ROOT/"experiments/analyze_locomo_disjoint.py",
             ROOT/"MemNexus/memory-system/scripts/qwen_embedding_server.py"]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in files}


def compatible(report, signature):
    return all(report["signature"].get(k) == v for k,v in signature.items())


def choose_run(output_root, signature, explicit=None, force_new=False):
    if explicit:
        path = Path(explicit).resolve()
        path = path if path.is_file() else path/"run.json"
        r = read(path)
        if not compatible(r,signature):
            raise ValueError("代码、数据或配置已变化，不能混合续跑。请新建实验。")
        return path.parent,r
    if not force_new:
        for path in sorted(output_root.glob("*/run.json"), reverse=True):
            r = read(path)
            if r["status"] != "completed" and compatible(r,signature):
                return path.parent,r
    name = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
    folder=output_root/name
    folder.mkdir(parents=True,exist_ok=False)
    return folder,None


def all_spend(folder):
    total=0.0
    for p in (folder/"attempts").glob("*/usage.json"):
        ledger=read(p)
        total += ledger.get("cost_rmb",0)+ledger.get("reserved_rmb",0)
    return total


def remaining(folder,cap):
    value=cap-all_spend(folder)
    if value <= .001:
        raise RuntimeError("本轮预算已用完；在配置中提高budget_rmb后可续跑，原调用仍累计计费。")
    return value


def new_attempt(folder,unit):
    # Each retry owns a separate immutable ledger directory.
    attempts=folder/"attempts"
    attempts.mkdir(exist_ok=True)
    index=1
    while (attempts/f"{unit}__{index:03}").exists():index+=1
    target=attempts/f"{unit}__{index:03}"
    target.mkdir()
    return target


@contextlib.contextmanager
def run_lock(root):
    import msvcrt
    root.mkdir(parents=True,exist_ok=True)
    with (root/"runner.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if not handle.read(1):handle.write(b"0");handle.flush()
        except PermissionError as exc:
            raise RuntimeError("已有100题测试正在运行，请勿重复双击。") from exc
        handle.seek(0)
        try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:raise RuntimeError("已有100题测试正在运行，请勿重复双击。") from exc
        try:yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


class EmbeddingService:
    def __init__(self, config, folder):
        self.config,self.folder,self.process,self.log=config,folder,None,None

    def start(self):
        import httpx
        def health():
            r=httpx.get("http://127.0.0.1:8099/health",timeout=5)
            r.raise_for_status()
            value=r.json()
            if value.get("model") != "Qwen/Qwen3-Embedding-0.6B" or value.get("dimension") != 1024:
                raise ValueError("8099端口的embedding服务型号/维度不匹配。")
            return value
        try:return health()
        except (httpx.ConnectError,httpx.TimeoutException):pass
        if not self.config["auto_start_embedding"]:
            raise RuntimeError("Qwen服务不可用；请启动8099端口服务。")
        progress("启动本地Qwen服务（仅使用已有模型缓存）……")
        env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONIOENCODING="utf-8")
        self.log=(self.folder/"embedding.log").open("a",encoding="utf-8")
        self.process=subprocess.Popen([sys.executable,str(ROOT/"MemNexus/memory-system/scripts/qwen_embedding_server.py"),
                                       "--port","8099"],cwd=ROOT,env=env,stdout=self.log,stderr=self.log,
                                      creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        started=time.monotonic()
        while time.monotonic()-started < self.config["embedding_start_timeout_seconds"]:
            if self.process.poll() is not None:
                raise RuntimeError("Qwen启动失败。检查embedding.log；如本地没有模型缓存，请先下载模型。")
            try:return health()
            except (httpx.ConnectError,httpx.TimeoutException):time.sleep(2)
        raise RuntimeError("Qwen启动超时，请检查embedding.log。")

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
        if self.log:self.log.close()


async def embedding_parity():
    import numpy as np
    from agent_memory.embeddings import build_embedder
    from agent_memory.settings import Settings
    from run_locomo_pilot import RemoteQwenEmbedding
    a,b=RemoteQwenEmbedding(),build_embedder(Settings())
    if b is None:raise ValueError("MemNexus embedding未配置。")
    try:
        for texts,method in [(["Which event happened first?"],"encode_query"),
                             (["An example source statement."],"encode_documents")]:
            av=getattr(a,method)(texts)
            bv=await getattr(b,method)(texts)
            if not np.allclose(av,bv,atol=1e-6):raise ValueError("两套系统的embedding编码结果不一致。")
    finally:await b.aclose()


@contextlib.contextmanager
def stage_logging(path, label):
    # Model-library chatter goes to the file, with a separate heartbeat on console.
    stop=threading.Event()
    console=sys.stdout
    started=time.monotonic()
    def beat():
        while not stop.wait(30):
            print(f"  {label}：仍在运行，已用时{int(time.monotonic()-started)}秒；详细日志：{path.name}",file=console,flush=True)
    thread=threading.Thread(target=beat,daemon=True)
    thread.start()
    try:
        with path.open("a",encoding="utf-8") as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            yield
    finally:stop.set();thread.join(timeout=1)


def parse_judgment(raw):
    # A single valid JSON object only: never mine the preferred label out of prose.
    value=json.loads(raw)
    if not isinstance(value,dict) or value.get("label") not in {"CORRECT","WRONG"}:
        raise ValueError("裁判没有返回有效label。")
    if not isinstance(value.get("reason"),str) or not value["reason"].strip():
        raise ValueError("裁判没有返回理由。")
    opposite="WRONG" if value["label"]=="CORRECT" else "CORRECT"
    if re.search(r"(?:should be|therefore(?:,? (?:it|the answer))? is|final (?:label|answer)\s*[:=])\s*"+opposite,
                 value["reason"],re.IGNORECASE):
        raise ValueError("裁判label与结论理由矛盾。")
    return value


def judge_one(row, folder, cap):
    import config
    from simplemem.core.utils.llm_client import LLMClient
    os.environ["EXPERIMENT_COMPONENT_BUDGET_RMB"]=str(remaining(folder.parents[1],cap))
    os.environ["EXPERIMENT_USAGE_LEDGER_PATH"]=str(folder/"usage.json")
    judge=LLMClient(api_key=config.JUDGE_API_KEY or config.OPENAI_API_KEY,model=config.JUDGE_MODEL,
                    base_url=config.JUDGE_BASE_URL,enable_thinking=False,use_streaming=False)
    payload=json.dumps({k:row[k] for k in ("question","gold_answer","prediction")},ensure_ascii=False)
    # The paper issues a single user message; a "strict" system prompt would
    # contradict its explicitly generous grading stance, so none is prepended.
    # Blindness still holds: the payload carries no system name.
    messages=[{"role":"user","content":JUDGE_INSTRUCTION+"\n"+payload}]
    failures=[]
    try:
        for attempt in range(2):
            raw=judge.chat_completion(messages,response_format={"type":"json_object"},max_retries=1)
            try:
                verdict=parse_judgment(raw)
                return {"judge_result":verdict["label"],"judge_reason":verdict["reason"],
                        "judge_raw_response":raw,"format_repairs":failures,"usage":dict(judge.usage)}
            except (ValueError,TypeError) as exc:
                failures.append({"raw":raw,"error":str(exc)})
                if attempt:raise ValueError("裁判在一次格式修复后仍无效，已保存账本；请续跑。") from exc
                messages=[*messages,{"role":"user","content":"Return a single valid JSON object with label and a concise reason consistent with that label. Evaluate the original question, gold and prediction above."}]
    finally:
        judge.client.close()


def annotate_stage(stage,rows,items):
    if len(rows)!=len(items) or stage.get("memory_count",0)<=0 or stage.get("failed_windows",0):
        raise ValueError("建库或答案数量不完整，不能纳入主结果。")
    for row,item in zip(rows,items,strict=True):
        if row["question"]!=item["question"] or not row["prediction"].strip():
            raise ValueError("空回答或问题不匹配。")
        row.update(qa_index=item["qa_index"],gold_evidence=item["evidence"])


async def execute(folder,report,cohort,dataset,cap,snapshots=None,require_snapshots=False):
    from run_locomo_pilot import run_simplemem,run_memnexus
    units=[]
    for index in sorted({q["sample_index"] for q in cohort["questions"]}):
        items=[q for q in cohort["questions"] if q["sample_index"]==index]
        sample=dataset[index]
        questions=[sample["qa"][q["qa_index"]] for q in items]
        for system in SYSTEMS:
            unit=f"{sample['sample_id']}_{system.lower()}"
            units.append(unit)
            if unit in report["stages"]:continue
            # Recover a committed result if the process died between two atomic writes.
            recovered=False
            for path in sorted((folder/"attempts").glob(f"{unit}__*/stage_result.json"),reverse=True):
                checkpoint=read(path)
                annotate_stage(checkpoint["stage"],checkpoint["rows"],items)
                report["stages"][unit]=path.relative_to(folder).as_posix()
                recovered=True
                break
            if recovered:save(folder/"run.json",report);continue
            allowed=remaining(folder,cap)
            attempt=new_attempt(folder,unit)
            report["active_unit"]=unit
            save(folder/"run.json",report)
            progress(f"{system} / {sample['sample_id']}：独立建库并回答{len(items)}题；剩余预算约¥{allowed:.2f}")
            with stage_logging(attempt/"console.log",unit):
                if system=="SimpleMem":
                    stage,rows=run_simplemem(sample,questions,allowed,usage_ledger=attempt/"usage.json",max_session=cohort["max_session"])
                else:
                    stage,rows=await run_memnexus(sample,questions,allowed,usage_ledger=attempt/"usage.json",max_session=cohort["max_session"],snapshot_dir=snapshots,require_snapshot=require_snapshots)
            annotate_stage(stage,rows,items)
            save(attempt/"stage_result.json",{"stage":stage,"rows":rows,"ledger":"usage.json"})
            report["stages"][unit]=(attempt/"stage_result.json").relative_to(folder).as_posix()
            save(folder/"run.json",report)
            progress(f"完成{unit}；本轮累计费用上界¥{all_spend(folder):.4f}")

    rows=[row for unit in units for row in read(folder/report["stages"][unit])["rows"]]
    random.Random(report["judge_seed"]).shuffle(rows)
    for row in rows:
        unit=f"{row['sample_id']}_{row['qa_index']}_{row['system'].lower()}"
        if unit in report["judgments"]:continue
        completed=list((folder/"attempts").glob(f"judge_{unit}__*/judgment.json"))
        if completed:
            report["judgments"][unit]=sorted(completed)[-1].relative_to(folder).as_posix()
            save(folder/"run.json",report)
            continue
        attempt=new_attempt(folder,"judge_"+unit)
        report["active_unit"]="judge_"+unit
        save(folder/"run.json",report)
        with stage_logging(attempt/"console.log","统一裁判"):
            value=judge_one(row,attempt,cap)
        value.update(sample_id=row["sample_id"],qa_index=row["qa_index"],system=row["system"],
                     prediction_sha256=hashlib.sha256(row["prediction"].encode()).hexdigest())
        save(attempt/"judgment.json",value)
        report["judgments"][unit]=(attempt/"judgment.json").relative_to(folder).as_posix()
        save(folder/"run.json",report)
        if len(report["judgments"])%10==0:progress(f"统一盲评：{len(report['judgments'])}/200；累计费用上界¥{all_spend(folder):.4f}")


async def run(args):
    os.chdir(ROOT)
    if args.report_only:
        from benchmark100_report import generate
        generate(Path(args.report_only).resolve())
        return 0
    value=load_config(args.config)
    if args.budget is not None:value["budget_rmb"]=args.budget
    if not math.isfinite(value["budget_rmb"]) or value["budget_rmb"]<=0:raise ValueError("预算必须为正数")
    if args.variant:value["variant"]=args.variant
    # A frozen base is an experimental condition: runs that replayed a snapshot
    # and runs that built from scratch must never be resumed into each other.
    snapshots=(args.memory_snapshots.resolve() if args.memory_snapshots else None)
    if args.require_snapshots and snapshots is None:raise ValueError("--require-snapshots 需要同时给出 --memory-snapshots")
    cohort,dataset=load_cohort(ROOT/value["cohort"])
    public=configure(value)
    files=source_files()
    signature={"dataset":cohort["dataset_sha256"],"selection":cohort["selection_sha256"],
               "source_sha256":files,"settings":public,"judge_seed":value["judge_seed"],
               "memory_snapshots":str(snapshots) if snapshots else None}
    progress(f"本地预检通过：100题，{cohort['conversation_count']}段对话，D1–D{cohort['max_session']}，预算¥{value['budget_rmb']:.2f}")
    progress(f"MemNexus语义执行={public['semantic_execution']}；缺口补检索={public['premise_retrieval']}")
    if args.dry_run:
        import benchmark100_report  # noqa: F401 - fail early if plotting dependencies are missing
        progress("仅检查配置/题组/依赖，未连接模型服务、未发起付费请求。")
        return 0
    output_root=(ROOT/value["output_root"]).resolve()
    with run_lock(output_root):
        if args.preflight_only:
            folder=output_root/"preflight"
            folder.mkdir(exist_ok=True)
            report=None
        else:
            folder,report=choose_run(output_root,signature,args.resume,args.new)
        embedding=EmbeddingService(value,folder)
        try:
            if report and len(report["judgments"])==200:
                from benchmark100_report import generate
                generate(folder)
                if value["open_report"] and not args.no_open:os.startfile(folder/"report.html")
                return 0
            if report is None and not args.preflight_only:
                report={"schema_version":1,"status":"running","signature":signature,"cohort":cohort,
                        "dataset_path":str(ROOT/cohort["dataset"]),"started_at":datetime.now(timezone.utc).isoformat(),
                        "environment":{"python":platform.python_version(),"platform":platform.platform()},
                        "judge_seed":value["judge_seed"],"budget_rmb":value["budget_rmb"],
                        "stages":{},"judgments":{},"budget_history":[value["budget_rmb"]]}
                save(folder/"run.json",report)
                with zipfile.ZipFile(folder/"source.zip","x",zipfile.ZIP_DEFLATED) as archive:
                    for name in files:archive.write(ROOT/name,name)
            if report:
                report["budget_rmb"]=value["budget_rmb"]
                if report["budget_history"][-1]!=value["budget_rmb"]:report["budget_history"].append(value["budget_rmb"])
                report["status"]="running"
                report.pop("failure",None)
                save(folder/"run.json",report)
                progress(f"结果目录：{folder}")
            health=embedding.start()
            await embedding_parity()
            progress("Qwen query/document编码一致性验证通过。")
            if args.preflight_only:
                progress("预检完成；未调用主模型或裁判。")
                return 0
            report["embedding_preflight"]={"health":health,"vectors_match":True}
            save(folder/"run.json",report)
            await execute(folder,report,cohort,dataset,value["budget_rmb"],snapshots,args.require_snapshots)
            if source_files()!=files:
                report["invalidated"]="source_changed_during_run"
                raise RuntimeError("测试过程中源码发生变化。结果保留，但不能合并为同一版本的完整实验。")
            report["status"]="answers_complete"
            report["finished_at"]=datetime.now(timezone.utc).isoformat()
            report["all_attempts_cost_upper_rmb"]=all_spend(folder)
            save(folder/"run.json",report)
            from benchmark100_report import generate
            generate(folder)
            progress(f"完成100题对比：{folder/'report.html'}")
            if value["open_report"] and not args.no_open:os.startfile(folder/"report.html")
            return 0
        except BaseException as exc:
            if report is not None:
                report["status"]="interrupted"
                report["failure"]={"type":type(exc).__name__,"message":str(exc)}
                report["all_attempts_cost_upper_rmb"]=all_spend(folder)
                save(folder/"run.json",report)
            with (folder/"error.log").open("a",encoding="utf-8") as log:traceback.print_exc(file=log)
            raise
        finally:embedding.close()


def cli():
    # Only the CLI owns the console; importing the runner must not mutate the
    # streams of a test harness, notebook, or host application.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=CONFIG)
    parser.add_argument("--new",action="store_true",help="忽略未完成运行，新建实验")
    parser.add_argument("--resume",help="指定结果目录；不允许跨代码/配置续跑")
    parser.add_argument("--report-only",help="仅从指定结果目录重新审计和绘图，不调用API")
    parser.add_argument("--budget",type=float,help="覆盖本轮总预算，含已失败尝试及裁判")
    parser.add_argument("--variant",choices=["configured","stable","semantic"])
    parser.add_argument("--memory-snapshots",type=Path,help="MemNexus 记忆库快照目录：命中则跳过建库，未命中则建库后冻结")
    parser.add_argument("--require-snapshots",action="store_true",help="快照缺失时中止，禁止静默重建记忆库")
    parser.add_argument("--dry-run",action="store_true",help="仅本地检查，不联网、不收费")
    parser.add_argument("--preflight-only",action="store_true",help="检查配置及本地Qwen，不调用付费模型")
    parser.add_argument("--no-open",action="store_true")
    args=parser.parse_args()
    try:return asyncio.run(run(args))
    except KeyboardInterrupt:
        progress("已中断。再次运行会续跑已完成阶段；当前未完成的系统/对话阶段可能需重建。")
        return 130
    except BaseException as exc:
        progress(f"停止：{type(exc).__name__}。{exc}")
        return 2


if __name__=="__main__":
    raise SystemExit(cli())
