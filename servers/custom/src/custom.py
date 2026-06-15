import re
from typing import List, Dict, Any

from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("custom")


@app.tool(output="ans_ls->extract_query_list")
def search_r1_query_extract(ans_ls: List[str]) -> Dict[str, List[str]]:

    def get_query(text):
        import re

        pattern = re.compile(r"<search>([^<]*)", re.DOTALL)
        matches = pattern.findall(text)

        if matches:
            query = matches[-1].strip()
            if not query.endswith("?"):
                query += "?"
            return query
        else:
            return "There is no query."

    query = [get_query(answer) for answer in ans_ls]

    return {"extract_query_list": query}


@app.tool(output="ans_ls->extract_query_list")
def r1_searcher_query_extract(ans_ls: List[str]) -> Dict[str, List[str]]:

    def get_query(text):
        import re

        pattern = re.compile(r"<|begin_of_query|>([^<]*)", re.DOTALL)
        matches = pattern.findall(text)

        if matches:
            query = matches[-1].strip()
            if not query.endswith("?"):
                query += "?"
            return query
        else:
            return "There is no query."

    query = [get_query(answer) for answer in ans_ls]

    return {"extract_query_list": query}


@app.tool(output="q_ls,ret_psg->nextq_ls")
def iterretgen_nextquery(
    q_ls: List[str],
    ans_ls: List[str | Any],
) -> Dict[str, List[str]]:
    ret = []
    for q, ans in zip(q_ls, ans_ls):
        next_query = f"{q} {ans}"
        ret.append(next_query)
    return {"nextq_ls": ret}


@app.tool(output="ans_ls->pred_ls")
def output_extract_from_boxed(ans_ls: List[str]) -> Dict[str, List[str]]:
    def extract(ans: str) -> str:
        # 优先提取 <answer></answer> 标签（用于 qa_boxed_single_choice_simple.jinja）
        if "<answer>" in ans and "</answer>" in ans:
            try:
                content = ans.split('<answer>')[1].split('</answer>')[0].strip()
                # 如果提取成功，继续处理
                if content:
                    # 强制提取选项字母（A/B/C/D/E），用于单选题精确匹配
                    option_match = re.search(r'\b([ABCDE])\b', content.upper())
                    if option_match:
                        return option_match.group(1)
                    # 如果没有找到选项字母，返回原始内容
                    return content
            except:
                pass
        
        # 其次提取 \boxed{} 格式（用于 qa_rag_boxed.jinja 等）
        start = ans.rfind(r"\boxed{")
        if start != -1:
            i = start + len(r"\boxed{")
            brace_level = 1
            end = i
            while end < len(ans) and brace_level > 0:
                if ans[end] == "{":
                    brace_level += 1
                elif ans[end] == "}":
                    brace_level -= 1
                end += 1
            content = ans[i : end - 1].strip()
            content = re.sub(r"^\$+|\$+$", "", content).strip()
            content = re.sub(r"^\\\(|\\\)$", "", content).strip()
            if content.startswith(r"\text{") and content.endswith("}"):
                content = content[len(r"\text{") : -1].strip()
            content = content.strip("()").strip()
        else:
            # 如果都没有，返回原始答案（去除首尾空白）
            content = ans.strip()

        # 强制提取选项字母（A/B/C/D/E），用于单选题精确匹配
        option_match = re.search(r'\b([ABCDE])\b', content.upper())
        if option_match:
            return option_match.group(1)

        # 清理格式
        content = content.replace("\\", " ")
        content = content.replace("  ", " ")
        return content

    return {"pred_ls": [extract(ans) for ans in ans_ls]}


@app.tool(output="ans_ls->q_ls")
def ircot_get_first_sent(
    ans_ls: List[str],
) -> Dict[str, List[str]]:
    ret = []
    for ans in ans_ls:
        match = re.search(r"(.+?[。！？.!?])", ans)
        if match:
            ret.append(match.group(1))
        else:
            ret.append(ans.strip())
    return {"q_ls": ret}


@app.tool(output="ans_ls->pred_ls")
def ircot_extract_ans(ans_ls: List[str]) -> Dict[str, List[str]]:
    ret = []
    pattern = re.compile(r"so the answer is[\s:]*([^\n]*)", re.IGNORECASE)
    for ans in ans_ls:
        match = pattern.search(ans)
        if match:
            ret.append(match.group(1).strip())
        else:
            ret.append(ans.strip())
    return {"pred_ls": ret}


@app.tool(output="ans_ls->extract_query_list")
def search_o1_query_extract(ans_ls: List[str]) -> Dict[str, List[str]]:
    import re

    BEGIN = "<|begin_search_query|>"
    END = "<|end_search_query|>"
    PATTERN = re.escape(BEGIN) + r"(.*?)" + re.escape(END)

    def get_query(text):
        matches = re.findall(PATTERN, text, flags=re.DOTALL)
        if not matches:
            return ""  
        q = matches[-1].strip()
        q = re.sub(r"\s+", " ", q).strip(' "\'')
        return q

    query = [get_query(answer) for answer in ans_ls]

    return {"extract_query_list": query}

@app.tool(output="temp_psg,ret_psg->ret_psg")
def merge_passages(
    temp_psg: List[str | Any],
    ret_psg: List[str | Any],
) -> Dict[str, List[str | Any]]:
    for t_psg, psg in zip(temp_psg, ret_psg):
        psg.extend(t_psg)

    return {"ret_psg": ret_psg}


@app.tool(output="ans_ls->pred_ls")
def evisrag_output_extract_from_special(ans_ls: List[str]) -> Dict[str, List[str]]:
    def extract(ans: str) -> str:
        try:
            content = ans.split('<answer>')[1].split('</answer>')[0].strip()
        except:
            content = ans.strip()
        return content

    return {"pred_ls": [extract(ans) for ans in ans_ls]}


@app.tool(output="paper_list_1,paper_list_2->merged_papers")
def merge_paper_lists(
    paper_list_1: List[Dict[str, Any]],
    paper_list_2: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    合并两个论文列表，去除重复项（基于标题）
    
    Args:
        paper_list_1: 第一个论文列表
        paper_list_2: 第二个论文列表
    
    Returns:
        合并后的论文列表
    """
    merged = []
    seen_titles = set()
    
    # 添加第一个列表的论文
    for paper in paper_list_1:
        title = paper.get("title", "")
        if title and title not in seen_titles:
            merged.append(paper)
            seen_titles.add(title)
    
    # 添加第二个列表的论文（去除重复）
    for paper in paper_list_2:
        title = paper.get("title", "")
        if title and title not in seen_titles:
            merged.append(paper)
            seen_titles.add(title)
    
    return {"merged_papers": merged}


if __name__ == "__main__":
    app.run(transport="stdio")
