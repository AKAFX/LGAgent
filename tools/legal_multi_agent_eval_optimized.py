"""
多智能体法律评测（优化版）

项目方法与创新点（与现有实现严格一致）：
1) 角色分工明确的三智能体协作：
   - 律师A（Issue Spotter）：把题干+选项转为结构化要素 JSON，不允许直接给答案倾向。
   - 法官（Process Controller）：输出流程控制 JSON（是否检索、证据需求、反例方向）。
   - 律师B（Decision Maker）：两阶段提交（B0 盲答 + B1 核验）并输出最终选项。
2) 可选多轮澄清机制：
   - 律师B与法官可进行有限轮次澄清，减少 JSON 解析失败或证据不足导致的误判。
3) 可选RAG增强：
   - 仅当法官判定 need_retrieval=True 且启用 USE_RAG 时，才触发检索，降低无效检索开销。

本脚本相对旧版的工程优化：
- 全部路径基于项目根目录（避免写到错误磁盘路径）
- 支持命令行参数（模型/数据集/并发/开关）与默认值并存
- 定期 checkpoint 落盘（防止长跑中断后结果全丢）
- 线程异常样本保底写入，确保每题都有记录
- 增加错误统计（API_ERROR/TASK_ERROR/空预测）

用法示例（项目根目录）：
  python tools/legal_multi_agent_eval_optimized.py ^
    --dataset data/Ability_merged_500.jsonl ^
    --eval-model llama-3-8b-instruct ^
    --enable-lawyer-a true ^
    --enable-judge true ^
    --enable-dialogue true ^
    --use-rag false ^
    --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root and src are importable.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _p in (ROOT_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.key_pool import load_key_pool
from tools.legal_multi_agent_prompt_demo import (  # type: ignore
    PARAM_PATH,
    build_client,
    load_generation_config,
    run_judge,
    run_lawyer_answer,
    run_lawyer_parser,
)
from ultrarag.api import ToolCall, initialize  # type: ignore


# ---- 默认配置（可被命令行覆盖）----
DEFAULT_DATASET = ROOT_DIR / "data" / "dimension_jsonl" / "4_LegalEthics_sample_200.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "output" / "legal_multi_agent_eval_optimized.json"
DEFAULT_LOG = ROOT_DIR / "output" / "legal_multi_agent_eval_optimized.log"
DEFAULT_MODEL = "glm-4-air"
DEFAULT_CONCURRENCY = 8
CHECKPOINT_EVERY = 50

# A/J 用 key 池（deepseek）
REASONING_KEYS_POOL = load_key_pool("LGAGENT_REASONING_API_KEYS")

# B 用 key 池（待测模型）
EVAL_KEYS_POOL = load_key_pool("LGAGENT_EVAL_API_KEYS")


class TeeLogger:
    def __init__(self, path: Path):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.f = path.open("w", encoding="utf-8")

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self) -> None:
        self.stdout.flush()
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def normalize_text(text: str) -> str:
    import re
    import string as _string

    text = (text or "").strip().lower()
    table = str.maketrans({c: " " for c in _string.punctuation})
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_choice(answer: str) -> str:
    import re

    text = (answer or "").strip()
    if not text:
        return ""
    if text.startswith("[API_ERROR") or text.startswith("[ERROR") or text.startswith("[TASK_ERROR]"):
        return ""
    if re.match(r"^[A-D]$", text):
        return text
    m = re.search(r"[最终答案选项]*[：:]\s*([A-D])\b", text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\b([A-D])\b", text.splitlines()[0] if text.splitlines() else text)
    if m2:
        return m2.group(1).strip()
    m3 = re.search(r"\b([A-D])\b", text)
    return m3.group(1).strip() if m3 else ""


def evaluate(gt_all: List[List[str]], pred_all: List[str]) -> Dict[str, float]:
    n = len(pred_all) or 1
    acc = 0.0
    for gts, pred in zip(gt_all, pred_all):
        p = normalize_text(pred)
        if p and any(p == normalize_text(g) for g in gts):
            acc += 1.0
    acc /= n
    # 选择题场景下 EM/F1 与 acc 一致性更强，这里保持简洁可解释
    return {"avg_acc": acc, "avg_em": acc, "avg_f1": acc}


def parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y", "on"}


def save_checkpoint(
    output_path: Path,
    dataset: Path,
    eval_conf: Dict[str, Any],
    settings: Dict[str, Any],
    detail: List[Dict[str, Any]],
    golden_all: List[List[str]],
) -> None:
    preds = [d.get("lawyerB_pred_for_eval", "") for d in detail]
    metrics = evaluate(golden_all[: len(detail)], preds)
    err_api = sum(1 for d in detail if str(d.get("lawyerB_answer", "")).startswith("[API_ERROR"))
    err_task = sum(1 for d in detail if str(d.get("lawyerB_answer", "")).startswith("[TASK_ERROR"))
    empty = sum(1 for d in detail if not d.get("lawyerB_pred_for_eval"))
    payload = {
        "dataset": str(dataset),
        "metrics": metrics,
        "model_under_test": {"model": eval_conf["model"], "base_url": eval_conf["base_url"]},
        "settings": settings,
        "stats": {
            "finished_examples": len(detail),
            "api_error_count": err_api,
            "task_error_count": err_task,
            "empty_pred_count": empty,
        },
        "examples": detail,
        "checkpoint_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--log", type=str, default=str(DEFAULT_LOG))
    ap.add_argument("--eval-model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--eval-base-url", type=str, default="")
    ap.add_argument("--max-examples", type=int, default=-1)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--enable-lawyer-a", type=str, default="true")
    ap.add_argument("--enable-judge", type=str, default="true")
    ap.add_argument("--enable-dialogue", type=str, default="true")
    ap.add_argument("--use-rag", type=str, default="false")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    log_path = Path(args.log)
    max_examples = None if args.max_examples < 0 else args.max_examples

    enable_lawyer_a = parse_bool(args.enable_lawyer_a)
    enable_judge = parse_bool(args.enable_judge)
    enable_dialogue = parse_bool(args.enable_dialogue)
    use_rag = parse_bool(args.use_rag)
    concurrency = max(1, int(args.concurrency))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(log_path)
    origin_out, origin_err = sys.stdout, sys.stderr
    sys.stdout = tee
    sys.stderr = tee

    try:
        gen_conf = load_generation_config(PARAM_PATH)
        eval_conf: Dict[str, Any] = {
            "model": args.eval_model,
            "base_url": args.eval_base_url or gen_conf["base_url"],
            "temperature": gen_conf["temperature"],
            "top_p": gen_conf["top_p"],
            "max_tokens": gen_conf["max_tokens"],
        }
        base_url_reasoning = gen_conf["base_url"]

        settings = {
            "ENABLE_LAWYER_A": enable_lawyer_a,
            "ENABLE_JUDGE": enable_judge,
            "ENABLE_DIALOGUE": enable_dialogue,
            "USE_RAG": use_rag,
            "CONCURRENCY": concurrency,
        }

        print(f"日志文件：{log_path}")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据集：{dataset_path}")
        print(f"模型（律师B）：{eval_conf['model']}")
        print(f"设置：{settings}")
        print("=" * 80)

        # RAG init
        rag_enabled = False
        if use_rag and enable_judge:
            try:
                initialize(["retriever"], server_root=str(ROOT_DIR / "servers"))
                rag_enabled = True
                print("检索模块初始化成功（RAG启用）")
            except Exception as e:
                print(f"[WARN] 检索模块初始化失败，回退纯LLM：{e}")
                rag_enabled = False
        else:
            print("RAG禁用")

        # 读取数据
        questions: List[str] = []
        golden_all: List[List[str]] = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = obj.get("question", "")
                gts = obj.get("golden_answers") or []
                if not q or not gts:
                    continue
                questions.append(q)
                golden_all.append(gts)
                if max_examples is not None and len(questions) >= max_examples:
                    break

        total = len(questions)
        print(f"读取样本数：{total}")
        if total == 0:
            raise RuntimeError("数据集为空或字段不匹配（需要 question + golden_answers）。")

        use_multi_agent = enable_lawyer_a or enable_judge
        # key 池准备
        if use_multi_agent:
            reasoning_keys = (REASONING_KEYS_POOL[:concurrency] if len(REASONING_KEYS_POOL) >= concurrency
                              else [gen_conf.get("api_key", "")] * concurrency)
        else:
            reasoning_keys = []
        eval_keys = (EVAL_KEYS_POOL[:concurrency] if len(EVAL_KEYS_POOL) >= concurrency
                     else [gen_conf.get("api_key", "")] * concurrency)
        if not any(eval_keys):
            raise RuntimeError("没有可用的律师B API key（EVAL_KEYS_POOL或parameter中的api_key为空）。")

        progress_lock = threading.Lock()
        completed = [0]

        def process_one(idx: int, q: str, gts: List[str], thread_id: int) -> Dict[str, Any]:
            reason_client = None
            if use_multi_agent:
                rk = reasoning_keys[thread_id % len(reasoning_keys)]
                reason_client = build_client(base_url_reasoning, rk)
            ek = eval_keys[thread_id % len(eval_keys)]
            eval_client = build_client(eval_conf["base_url"], ek)

            parsed = ""
            judge_res = ""
            judge_need_retrieval = False
            retrieved_docs: List[str] = []
            lawyer_a_history: List[Dict[str, str]] = []
            judge_history: List[Dict[str, str]] = []

            try:
                if enable_lawyer_a and reason_client is not None:
                    parsed, lawyer_a_history = run_lawyer_parser(reason_client, gen_conf, q)

                if enable_judge and reason_client is not None:
                    judge_res, judge_history, should_continue = run_judge(
                        reason_client, gen_conf, q, parsed, []
                    )
                    if should_continue and enable_lawyer_a:
                        parsed, lawyer_a_history = run_lawyer_parser(reason_client, gen_conf, q, lawyer_a_history)
                        judge_res, judge_history, _ = run_judge(
                            reason_client,
                            gen_conf,
                            q,
                            parsed,
                            judge_history,
                            need_clarification=True,
                            clarification_question="请补充关键缺口与证据需求。",
                        )
                    try:
                        j = json.loads((judge_res or "").strip())
                        judge_need_retrieval = bool(j.get("need_retrieval", False))
                    except Exception:
                        judge_need_retrieval = False

                if enable_judge and rag_enabled and judge_need_retrieval:
                    def _search() -> Dict[str, Any]:
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        return loop.run_until_complete(
                            ToolCall.retriever.retriever_search(query_list=[q], top_k=3)
                        )
                    try:
                        ret = _search().get("ret_psg", [])
                        if ret and isinstance(ret[0], list):
                            retrieved_docs = [str(x).strip() for x in ret[0] if x]
                    except Exception:
                        retrieved_docs = []

                full_answer, internal_json = run_lawyer_answer(
                    eval_client,
                    eval_conf,
                    q,
                    parsed,
                    judge_res,
                    retrieved_docs,
                    use_multi_agent=use_multi_agent,
                    reasoning_client=reason_client if use_multi_agent else None,
                    gen_conf=gen_conf if use_multi_agent else None,
                    enable_dialogue=enable_dialogue,
                )
            except Exception as e:
                full_answer = f"[TASK_ERROR] {e}"
                internal_json = {"error": str(e)}

            pred = extract_choice(str(full_answer))

            with progress_lock:
                completed[0] += 1
                cur = completed[0]
                pct = cur / total * 100.0
                print(f"\rEvaluating {cur}/{total} ({pct:5.1f}%)", end="", flush=True)

            return {
                "idx": idx,
                "question": q,
                "golden_answers": gts,
                "lawyerA_parsed": parsed,
                "lawyerA_history": lawyer_a_history,
                "judge_reasoning": judge_res,
                "judge_history": judge_history,
                "judge_need_retrieval": judge_need_retrieval,
                "retrieved_docs": retrieved_docs,
                "lawyerB_answer": str(full_answer),
                "lawyerB_internal_json": internal_json,
                "lawyerB_pred_for_eval": pred,
            }

        results_dict: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {
                ex.submit(process_one, idx, q, gts, idx % concurrency): idx
                for idx, (q, gts) in enumerate(zip(questions, golden_all))
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results_dict[idx] = fut.result()
                except Exception as e:
                    results_dict[idx] = {
                        "idx": idx,
                        "question": questions[idx],
                        "golden_answers": golden_all[idx],
                        "lawyerA_parsed": "",
                        "lawyerA_history": [],
                        "judge_reasoning": "",
                        "judge_history": [],
                        "judge_need_retrieval": False,
                        "retrieved_docs": [],
                        "lawyerB_answer": f"[TASK_ERROR] {e}",
                        "lawyerB_internal_json": {"error": str(e)},
                        "lawyerB_pred_for_eval": "",
                    }

                # checkpoint：每收集一定数量就落盘
                if len(results_dict) % CHECKPOINT_EVERY == 0:
                    detail_now = [results_dict[i] for i in sorted(results_dict.keys())]
                    save_checkpoint(output_path, dataset_path, eval_conf, settings, detail_now, golden_all)

        print()  # newline after progress

        detail = [results_dict[i] for i in range(total)]
        save_checkpoint(output_path, dataset_path, eval_conf, settings, detail, golden_all)
        metrics = evaluate(golden_all, [d["lawyerB_pred_for_eval"] for d in detail])
        print("评估结果：", metrics)
        print(f"详细结果已保存到：{output_path}")
        print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    finally:
        sys.stdout = origin_out
        sys.stderr = origin_err
        tee.close()


if __name__ == "__main__":
    main()

