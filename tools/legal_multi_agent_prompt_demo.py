"""
轻量级「伪多智能体」法律问答 Demo

特点：
- 不改 UltraRAG 主流水线 / servers
- 只用一次配置文件 examples/parameter/legal2_rag_parameter.yaml 中的 generation.backend_configs
- 通过三轮调用模拟三类角色：
  1）律师A：解析问题，输出结构化 JSON 要素
  2）法官：根据要素和自身知识进行法律推理（此处不接 RAG，只做纯 LLM 推理）
  3）律师B：基于法官推理，面向用户生成最终回答

用法示例：
    python tools/legal_multi_agent_prompt_demo.py "这里填入法律考试题目或法律咨询问题"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml  # 依赖 PyYAML
from dotenv import load_dotenv
from openai import OpenAI  # 依赖 openai>=1.0.0，兼容智增增 API

# ==== 用户可在这里固定律师B（待测模型）的配置 ====
EVAL_MODEL = "deepseek-v3"          # 律师B 用的模型名字（在智增增或其他平台上的 ID）
EVAL_BASE_URL = None                # 若为 None，沿用 legal2_rag_parameter.yaml 中的 base_url
EVAL_API_KEY = None                 # 若为 None，沿用 legal2_rag_parameter.yaml 中的 api_key
# ========================================
ROOT_DIR = Path(__file__).resolve().parents[1]
PARAM_PATH = ROOT_DIR / "examples" / "parameter" / "legal2_rag_parameter.yaml"
load_dotenv(ROOT_DIR / ".env")


def load_generation_config(param_path: Path) -> Dict[str, Any]:
    with open(param_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen_cfg = cfg.get("generation", {}) or {}
    backend = gen_cfg.get("backend", "openai")
    backend_cfgs = gen_cfg.get("backend_configs", {}) or {}
    backend_cfg = backend_cfgs.get(backend, {}) or {}

    sampling = gen_cfg.get("sampling_params", {}) or {}

    return {
        "backend": backend,
        "base_url": backend_cfg.get("base_url", "https://api.zhizengzeng.com/v1"),
        "api_key": backend_cfg.get("api_key") or os.environ.get("LLM_API_KEY", ""),
        "model": backend_cfg.get("model_name", "gpt-4o"),
        "temperature": sampling.get("temperature", 0.7),
        "top_p": sampling.get("top_p", 0.8),
        "max_tokens": sampling.get("max_tokens", 2048),
    }


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 1024,
) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        # 检查响应是否有错误
        if hasattr(resp, 'error') and resp.error:
            error_msg = resp.error.get('message', 'Unknown API error')
            error_code = resp.error.get('code', 'unknown')
            return f"[API_ERROR: {error_code}] {error_msg}"
        # 检查是否有choices
        if not hasattr(resp, 'choices') or not resp.choices:
            return "[API_ERROR: no_choices] API response has no choices"
        msg = resp.choices[0].message
        content = getattr(msg, "content", None)

        # Some OpenAI-compatible providers/models may return non-string content
        # (e.g., list-of-parts) or None (e.g., tool calls). Make a best-effort
        # extraction so downstream answer parsing doesn't silently become "".
        if isinstance(content, str):
            return content

        # list-of-parts: [{"type":"text","text":"..."}] or similar
        if isinstance(content, list):
            parts: List[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    # common schemas: {"text": "..."} or {"type":"text","text":"..."}
                    t = p.get("text") if "text" in p else None
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
            merged = "\n".join([s for s in parts if s is not None]).strip()
            if merged:
                return merged

        # tool-calls / other fields fallback
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            try:
                return json.dumps([tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls], ensure_ascii=False)
            except Exception:
                return str(tool_calls)

        # last resort: dump the message for debugging visibility
        try:
            if hasattr(msg, "model_dump"):
                dumped = msg.model_dump()
                return json.dumps(dumped, ensure_ascii=False)
        except Exception:
            pass
        return str(content or "")
    except Exception as e:
        # 捕获所有异常（包括内容审核失败、网络错误等）
        error_type = type(e).__name__
        error_msg = str(e)
        return f"[API_ERROR: {error_type}] {error_msg}"


def run_lawyer_parser(client: OpenAI, gen_conf: Dict[str, Any], question: str, conversation_history: List[Dict[str, str]] = None) -> Tuple[str, List[Dict[str, str]]]:
    """
    律师A（Issue Spotter / 结构化书记员）
    目标：把题目变成"检索与核验友好"的结构化要素；绝对禁止输出答案倾向。
    
    Returns:
        (parsed_json_str, updated_conversation_history)
    """
    system_prompt = (
        "你是律师A（Issue Spotter），将题目转为结构化要素。\n\n"
        "【硬约束】\n"
        "- 禁止输出答案倾向、推理链、选项排序。\n"
        "- 只输出结构化要素与关键词。\n\n"
        "【输出格式 - 严格JSON】\n"
        "{\n"
        '  "task_type": "题型（如：单选、多选、判断等）",\n'
        '  "question_focus": "题干要比较的维度（如\\"最不具有专制特色\\"=\\"专制性最低\\"）",\n'
        '  "legal_domain": "法域/学科（如：法制史、刑法、民法、行政法等）",\n'
        '  "option_claims": {\n'
        '    "A": "把选项A改写成可判真伪的命题（如\\"X行为构成Y罪\\"）",\n'
        '    "B": "把选项B改写成可判真伪的命题",\n'
        '    "C": "把选项C改写成可判真伪的命题",\n'
        '    "D": "把选项D改写成可判真伪的命题"\n'
        '  },\n'
        '  "option_keywords": {\n'
        '    "A": ["关键词1", "关键词2", ...],  // 每个选项3-5个关键词（精简）\n'
        '    "B": ["关键词1", "关键词2", ...],\n'
        '    "C": ["关键词1", "关键词2", ...],\n'
        '    "D": ["关键词1", "关键词2", ...]\n'
        '  },\n'
        '  "trap_signals": ["陷阱信号"],  // 仅列出最关键的1-2个\n'
        '  "unknowns": ["关键缺口"]  // 仅列出最关键的1-2个\n'
        "}\n\n"
        "只输出 JSON，不要添加任何多余说明、推理过程或答案倾向。"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加对话历史（如果有）
    if conversation_history:
        messages.extend(conversation_history)
    
    # 添加当前问题
    user_prompt = f"请解析下面的法律问题（题干+选项）：\n\n{question}"
    messages.append({"role": "user", "content": user_prompt})

    # 优化：律师A输出应该比较简洁，限制max_tokens
    content = chat(
        client,
        model=gen_conf["model"],
        messages=messages,
        temperature=gen_conf["temperature"],
        top_p=gen_conf["top_p"],
        max_tokens=min(gen_conf["max_tokens"], 1024),  # 限制最大token数
    )
    
    # 更新对话历史
    updated_history = messages[1:] + [{"role": "assistant", "content": content.strip()}]
    
    return content.strip(), updated_history


def run_judge(
    client: OpenAI, 
    gen_conf: Dict[str, Any], 
    question: str, 
    parsed_json: str = "",
    conversation_history: List[Dict[str, str]] = None,
    need_clarification: bool = False,
    clarification_question: str = ""
) -> Tuple[str, List[Dict[str, str]], bool]:
    """
    法官（Process Controller / 反例规划官）
    目标：做"流程与证据策略控制"，把律师A给的结构转成检索策略和证据需求。
    
    Returns:
        (judge_json_str, updated_conversation_history, should_continue_dialogue)
        should_continue_dialogue: True表示需要继续对话（最多加一轮）
    """
    system_prompt = (
        "你是法官（Process Controller），将律师A的结构转为检索策略和证据需求。\n\n"
        "【硬约束】\n"
        "- 禁止输出最终选项、选项排序或评价。\n"
        "- 只输出流程控制+证据需求+反例方向。\n\n"
        "【输出格式 - 严格JSON】\n"
        "{\n"
        '  "need_retrieval": true/false,  // 是否需要检索（先用规则版判断）\n'
        '  "global_query": "1个全局检索query（短、狠、准）",\n'
        '  "option_queries": {\n'
        '    "A": "选项A的检索query",\n'
        '    "B": "选项B的检索query",\n'
        '    "C": "选项C的检索query",\n'
        '    "D": "选项D的检索query"\n'
        '  },\n'
        '  "evidence_requirements": {\n'
        '    "A": {"support": "支持证据要点", "refute": "反驳证据要点"},  // 简短描述，1-2句话\n'
        '    "B": {"support": "...", "refute": "..."},\n'
        '    "C": {"support": "...", "refute": "..."},\n'
        '    "D": {"support": "...", "refute": "..."}\n'
        '  },\n'
        '  "counterfactual_focus": "反例方向（简短）",  // 1-2句话\n'
        '  "stop_rule": "裁决规则（简短）"  // 1-2句话\n'
        "}\n\n"
        "只输出 JSON，不要添加任何多余说明、推理过程或答案倾向。"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加对话历史（如果有）
    if conversation_history:
        messages.extend(conversation_history)
    
    # 构建用户提示
    if need_clarification and clarification_question:
        # 多轮对话：回答澄清问题
        user_prompt = (
            f"律师A的澄清问题：{clarification_question}\n\n"
            "请回答这个问题，并更新你的JSON输出。"
        )
    else:
        # 初始请求
        user_prompt = (
            "原始问题如下：\n"
            f"{question}\n\n"
        )
        if parsed_json:
            user_prompt += (
                "解析律师（律师A）给出的结构化要素：\n"
                f"{parsed_json}\n\n"
            )
        user_prompt += "请在此基础上给出你的流程控制和证据需求分析。"
    
    messages.append({"role": "user", "content": user_prompt})

    # 优化：法官输出应该比较简洁，限制max_tokens
    content = chat(
        client,
        model=gen_conf["model"],
        messages=messages,
        temperature=gen_conf["temperature"],
        top_p=gen_conf["top_p"],
        max_tokens=min(gen_conf["max_tokens"], 1024),  # 限制最大token数
    )
    
    # 更新对话历史
    updated_history = messages[1:] + [{"role": "assistant", "content": content.strip()}]
    
    # 判断是否需要继续对话（简单规则：如果JSON解析失败或内容过短，可能需要澄清）
    should_continue = False
    try:
        judge_data = json.loads(content.strip())
        # 如果某些关键字段缺失，可能需要澄清
        if not judge_data.get("need_retrieval") is not None or not judge_data.get("option_queries"):
            should_continue = True
    except:
        # JSON解析失败，可能需要澄清
        should_continue = True
    
    return content.strip(), updated_history, should_continue


def run_lawyer_answer_b0_blind(
    client: OpenAI,
    eval_conf: Dict[str, Any],
    question: str,
) -> Tuple[str, float]:
    """
    律师B阶段0：盲答（不看A/J输出、不看证据）
    返回：(initial_answer, confidence)
    """
    system_prompt = (
        "你是律师B（Decision Maker），盲答阶段。只看到题目，给出初步判断。\n\n"
        "【输出格式 - JSON】\n"
        "{\n"
        '  "initial_answer": "A/B/C/D中的一个",\n'
        '  "confidence": 0.0-1.0之间的浮点数，表示你的信心程度\n'
        "}\n\n"
        "只输出 JSON，不要添加多余说明。"
    )
    
    user_prompt = f"请分析以下法律问题并给出初步答案：\n\n{question}"
    
    # 优化：律师B B0阶段输出应该非常简洁，限制max_tokens
    content = chat(
        client,
        model=eval_conf["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=eval_conf["temperature"],
        top_p=eval_conf["top_p"],
        max_tokens=min(eval_conf["max_tokens"], 256),  # B0阶段只需要简单JSON，限制更严格
    )
    
    # 解析JSON
    try:
        result = json.loads(content.strip())
        initial_answer = result.get("initial_answer", "")
        confidence = float(result.get("confidence", 0.5))
        return initial_answer, confidence
    except:
        # 如果JSON解析失败，尝试提取选项字母
        import re
        match = re.search(r'\b([A-D])\b', content.strip())
        if match:
            return match.group(1), 0.5
        return "", 0.0


def run_lawyer_answer_b1_verification(
    client: OpenAI,
    eval_conf: Dict[str, Any],
    question: str,
    parsed_json: str = "",
    judge_json: str = "",
    retrieved_docs: List[str] | None = None,
    initial_answer: str = "",
    initial_confidence: float = 0.0,
    conversation_history: List[Dict[str, str]] = None,
    need_clarification: bool = False,
    clarification_question: str = ""
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]], bool]:
    """
    律师B阶段1：核验后裁决（看证据与A/J结构）
    逐项做 SUPPORT/REFUTE/NEI，算分后输出 final_answer
    
    Returns:
        (final_answer, internal_json, updated_conversation_history, should_continue_dialogue)
    """
    system_prompt = (
        "你是律师B（Decision Maker），基于A/J结构和证据进行核验裁决。\n\n"
        "【核验】对每个选项：SUPPORT/REFUTE/NEI，计算得分，输出最终答案（单一字母）。\n\n"
        "【硬约束】对外只输出单一字母，但必须输出内部JSON。\n\n"
        "【输出格式 - JSON】\n"
        "{\n"
        '  "final_answer": "A/B/C/D中的一个",\n'
        '  "verification": {\n'
        '    "A": {"status": "SUPPORT/REFUTE/NEI", "score": 0.0-1.0, "reason": "简短理由（1句话）"},\n'
        '    "B": {"status": "...", "score": ..., "reason": "..."},\n'
        '    "C": {"status": "...", "score": ..., "reason": "..."},\n'
        '    "D": {"status": "...", "score": ..., "reason": "..."}\n'
        '  },\n'
        '  "initial_answer": "你的盲答结果",\n'
        '  "initial_confidence": 0.0-1.0,\n'
        '  "reasoning": "最终裁决（1-2句话）"\n'
        "}\n\n"
        "只输出 JSON，不要添加多余说明。"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加对话历史（如果有）
    if conversation_history:
        messages.extend(conversation_history)
    
    # 构建用户提示
    if need_clarification and clarification_question:
        # 多轮对话：向法官提问
        user_prompt = (
            f"我的疑问：{clarification_question}\n\n"
            "请法官回答后，我重新进行核验裁决。"
        )
    else:
        # 初始核验请求
        user_prompt = (
            "原始问题：\n"
            f"{question}\n\n"
        )
        
        if parsed_json:
            user_prompt += (
                "律师A的结构化要素：\n"
                f"{parsed_json}\n\n"
            )
        
        if judge_json:
            user_prompt += (
                "法官的流程控制和证据需求：\n"
                f"{judge_json}\n\n"
            )
        
        if retrieved_docs and len(retrieved_docs) > 0:
            # 优化：减少文档数量并截断过长文档（减少token消耗）
            MAX_DOCS = 3  # 从10个减少到3个
            MAX_DOC_LENGTH = 300  # 每个文档最多300字符
            truncated_docs = []
            for i, doc in enumerate(retrieved_docs[:MAX_DOCS]):
                doc_str = str(doc).strip()
                if len(doc_str) > MAX_DOC_LENGTH:
                    doc_str = doc_str[:MAX_DOC_LENGTH] + "..."
                truncated_docs.append(f"[文档{i+1}]\n{doc_str}")
            docs_text = "\n\n".join(truncated_docs)
            user_prompt += (
                "检索到的证据：\n"
                f"{docs_text}\n\n"
            )
        
        user_prompt += (
            f"我的盲答结果：{initial_answer}（信心度：{initial_confidence:.2f}）\n\n"
            "请基于以上信息进行逐选项核验，并输出最终答案。"
        )
    
    messages.append({"role": "user", "content": user_prompt})

    # 优化：律师B B1阶段输出应该比较简洁，限制max_tokens
    content = chat(
        client,
        model=eval_conf["model"],
        messages=messages,
        temperature=eval_conf["temperature"],
        top_p=eval_conf["top_p"],
        max_tokens=min(eval_conf["max_tokens"], 1024),  # 限制最大token数
    )
    
    # 更新对话历史
    updated_history = messages[1:] + [{"role": "assistant", "content": content.strip()}]
    
    # 解析JSON
    try:
        result = json.loads(content.strip())
        final_answer = result.get("final_answer", "")
        internal_json = result
        should_continue = False
    except:
        # JSON解析失败，尝试提取选项字母
        import re
        match = re.search(r'\b([A-D])\b', content.strip())
        final_answer = match.group(1) if match else ""
        internal_json = {"final_answer": final_answer, "error": "JSON解析失败"}
        should_continue = True  # 可能需要澄清
    
    return final_answer, internal_json, updated_history, should_continue


def run_lawyer_judge_dialogue(
    reasoning_client: OpenAI,
    eval_client: OpenAI,
    gen_conf: Dict[str, Any],
    eval_conf: Dict[str, Any],
    question: str,
    parsed_json: str,
    judge_json: str,
    retrieved_docs: List[str],
    initial_answer: str,
    initial_confidence: float,
    lawyer_b_history: List[Dict[str, str]],
    max_rounds: int = 2
) -> Tuple[str, Dict[str, Any], List[Dict[str, str]]]:
    """
    律师B与法官的多轮对话机制
    
    Returns:
        (final_answer, internal_json, updated_lawyer_b_history)
    """
    final_answer = initial_answer
    internal_json = {"final_answer": initial_answer}
    current_history = lawyer_b_history.copy()
    judge_history = []
    dialogue_rounds = 0
    
    for round_num in range(max_rounds):
        dialogue_rounds = round_num + 1
        # 律师B提出疑问（如果有）
        if round_num == 0:
            # 第一轮：律师B进行核验，可能产生疑问
            final_answer, internal_json, current_history, need_clarification = run_lawyer_answer_b1_verification(
                eval_client, eval_conf, question, parsed_json, judge_json,
                retrieved_docs, initial_answer, initial_confidence, current_history
            )
            
            if not need_clarification:
                break
            
            # 提取疑问（简单规则：如果confidence低或verification中有多个NEI，可能需要澄清）
            clarification_question = ""
            if isinstance(internal_json, dict):
                verification = internal_json.get("verification", {})
                nei_count = sum(1 for v in verification.values() if isinstance(v, dict) and v.get("status") == "NEI")
                if nei_count >= 2 or internal_json.get("initial_confidence", 1.0) < 0.6:
                    clarification_question = "我对某些选项的证据支持度不确定，能否提供更具体的证据需求指导？"
        else:
            # 后续轮次：基于法官的回答继续核验
            clarification_question = "请基于之前的回答，我重新进行核验。"
        
        if not clarification_question:
            break
        
        # 法官回答律师B的疑问
        judge_response, judge_history, _ = run_judge(
            reasoning_client, gen_conf, question, parsed_json,
            judge_history, need_clarification=True, clarification_question=clarification_question
        )
        
        # 更新judge_json（合并法官的新回答）
        updated_judge_json = judge_response
        
        # 律师B基于法官的回答重新核验
        final_answer, internal_json, current_history, need_clarification = run_lawyer_answer_b1_verification(
            eval_client, eval_conf, question, parsed_json, updated_judge_json,
            retrieved_docs, initial_answer, initial_confidence, current_history,
            need_clarification=True, clarification_question=clarification_question
        )
        
        if not need_clarification:
            break
    
    # 合并B0和B1的信息到internal_json
    if isinstance(internal_json, dict):
        internal_json["b0_blind_answer"] = initial_answer
        internal_json["b0_confidence"] = initial_confidence
        internal_json["dialogue_rounds"] = dialogue_rounds
    
    return final_answer, internal_json, current_history


def run_lawyer_answer(
    client: OpenAI,
    eval_conf: Dict[str, Any],
    question: str,
    parsed_json: str = "",
    judge_json: str = "",
    retrieved_docs: List[str] | None = None,
    use_multi_agent: bool = True,
    reasoning_client: Optional[OpenAI] = None,
    gen_conf: Optional[Dict[str, Any]] = None,
    enable_dialogue: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    律师B生成最终答案，支持两阶段提交（B0盲答 + B1核验）和多智能体模式
    支持与法官的多轮对话
    
    Args:
        reasoning_client: 用于与法官对话的客户端（如果启用多轮对话）
        gen_conf: 用于与法官对话的配置（如果启用多轮对话）
        enable_dialogue: 是否启用律师B与法官的多轮对话
    
    Returns:
        (final_answer_str, internal_json_dict)
        final_answer_str: 对外输出的单一字母（A/B/C/D）
        internal_json_dict: 内部JSON用于诊断（包含B0和B1的完整信息）
    """
    # 判断是否使用多智能体模式
    has_parsed = parsed_json and parsed_json.strip()
    has_judge = judge_json and judge_json.strip()
    is_multi_agent = use_multi_agent and (has_parsed or has_judge)
    
    # B0阶段：盲答（不看A/J输出、不看证据）
    initial_answer, initial_confidence = run_lawyer_answer_b0_blind(
        client, eval_conf, question
    )
    
    # B1阶段：核验后裁决
    if is_multi_agent and enable_dialogue and reasoning_client and gen_conf:
        # 多智能体模式 + 启用多轮对话：使用对话机制
        final_answer, internal_json, _ = run_lawyer_judge_dialogue(
            reasoning_client, client, gen_conf, eval_conf, question,
            parsed_json, judge_json, retrieved_docs or [],
            initial_answer, initial_confidence, []
        )
    elif is_multi_agent:
        # 多智能体模式：使用A/J的输出和检索证据（无多轮对话）
        result = run_lawyer_answer_b1_verification(
            client, eval_conf, question, parsed_json, judge_json, 
            retrieved_docs, initial_answer, initial_confidence
        )
        if isinstance(result, tuple) and len(result) >= 2:
            final_answer, internal_json = result[0], result[1]
        else:
            final_answer = str(result) if result else ""
            internal_json = {"error": "Unexpected return type from b1_verification"}
    else:
        # 纯LLM模式：只使用检索证据（如果有）
        result = run_lawyer_answer_b1_verification(
            client, eval_conf, question, "", "", 
            retrieved_docs, initial_answer, initial_confidence
        )
        if isinstance(result, tuple) and len(result) >= 2:
            final_answer, internal_json = result[0], result[1]
        else:
            final_answer = str(result) if result else ""
            internal_json = {"error": "Unexpected return type from b1_verification"}
    
    # 合并B0和B1的信息到internal_json
    if isinstance(internal_json, dict):
        internal_json["b0_blind_answer"] = initial_answer
        internal_json["b0_confidence"] = initial_confidence
    
    return final_answer, internal_json


def main():
    parser = argparse.ArgumentParser(description="轻量级伪多智能体法律问答 Demo")
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="输入的法律问题（若为空则从标准输入读取）",
    )
    args = parser.parse_args()

    if args.question:
        question = args.question
    else:
        print("请输入法律问题，结束后按 Ctrl+D：")
        question = "".join(iter(input, ""))  # type: ignore

    gen_conf = load_generation_config(PARAM_PATH)
    if not gen_conf["api_key"]:
        raise RuntimeError(
            "generation.api_key 为空，请先在 examples/parameter/legal2_rag_parameter.yaml 中填好 API Key。"
        )

    # 律师A & 法官：使用 legal2_rag_parameter.yaml 中的 deepseek-v3（或你在 generation 中配置的模型）
    reasoning_client = build_client(gen_conf["base_url"], gen_conf["api_key"])

    # 律师B：使用待测模型，全部在文件顶部配置，不再通过命令行传参
    eval_conf: Dict[str, Any] = {
        "model": EVAL_MODEL or gen_conf["model"],
        "base_url": EVAL_BASE_URL or gen_conf["base_url"],
        "api_key": EVAL_API_KEY or os.environ.get("LGAGENT_EVAL_API_KEY") or gen_conf["api_key"],
        "temperature": gen_conf["temperature"],
        "top_p": gen_conf["top_p"],
        "max_tokens": gen_conf["max_tokens"],
    }
    eval_client = build_client(eval_conf["base_url"], eval_conf["api_key"])

    print("=== 律师A：解析问题（结构化要素） ===")
    parsed, _ = run_lawyer_parser(reasoning_client, gen_conf, question)
    print(parsed)

    print("\n=== 法官：法律推理与裁判思路（JSON） ===")
    judge_res, _, _ = run_judge(reasoning_client, gen_conf, question, parsed)
    print(judge_res)

    print("\n=== 律师B：面向用户/考生的最终回答 ===")
    final_answer, internal_json = run_lawyer_answer(
        eval_client, eval_conf, question, parsed, judge_res,
        reasoning_client=reasoning_client, gen_conf=gen_conf, enable_dialogue=True
    )
    print(f"最终答案：{final_answer}")
    print(f"\n内部JSON（用于诊断）：\n{json.dumps(internal_json, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
