from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List


VALID_ROUTES = {"direct", "retrieve", "knowledge", "both"}
VALID_STANCES = {"支持", "反对", "中立"}

KNOWLEDGE_MIN = 50
KNOWLEDGE_MAX = 350

KNOWLEDGE_NGRAM_N = 5
KNOWLEDGE_NGRAM_MIN_MATCHES = 3

_TURN_REF_RE = re.compile(r"[\[【]\s*轮次\s*(\d+)\s*[\]】]")


_TRAJ_RE = re.compile(
    r"<think>(?P<think1>.*?)</think>\s*"
    r"<route>\s*(?P<route>[^<\s]+)\s*</route>\s*"
    r"(?:<select_path>\s*(?P<select_raw>\[[^\]]*\])\s*</select_path>\s*)?"
    r"(?:<knowledge>(?P<knowledge>.*?)</knowledge>\s*)?"
    r"<think>(?P<think2>.*?)</think>\s*"
    r"<answer>\s*(?P<answer>[^<]+?)\s*</answer>",
    re.DOTALL,
)


@dataclass
class ParsedTraj:
    raw: str
    think1: str
    think2: str
    route: str
    select_path: Optional[List[int]]
    knowledge: Optional[str]
    answer: str


def _extract_region(text: str) -> Optional[str]:
    start = text.find("<think>")
    end = text.find("</answer>", start + len("<think>")) if start >= 0 else -1
    if start < 0:
        return None
    if end > start:
        return text[start: end + len("</answer>")]

    region = text[start:].strip()
    if re.search(r"<answer>\s*(?:支持|反对|中立)\s*$", region):
        return region + "</answer>"
    return None


def _parse_select(raw: str) -> Optional[List[int]]:
    inner = raw.strip()
    if not (inner.startswith("[") and inner.endswith("]")):
        return None
    inner = inner[1:-1].strip()
    if not inner:
        return None
    out: List[int] = []
    seen = set()
    for tok in inner.split(","):
        tok = tok.strip()
        if not tok:
            return None
        try:
            v = int(tok)
        except ValueError:
            return None
        if v < 0:
            return None
        if v in seen:
            return None
        seen.add(v)
        out.append(v)
    return out


def parse_trajectory(text: str) -> Optional[ParsedTraj]:
    region = _extract_region(text)
    if region is None:
        return None

    m = _TRAJ_RE.fullmatch(region.strip())
    if m is None:
        return None

    if region.count("<think>") != 2 or region.count("</think>") != 2:
        return None

    route = m.group("route").strip().lower()
    if route not in VALID_ROUTES:
        return None

    has_select = m.group("select_raw") is not None
    has_knowledge = m.group("knowledge") is not None

    expect_select = route in ("retrieve", "both")
    expect_knowledge = route in ("knowledge", "both")
    if has_select != expect_select or has_knowledge != expect_knowledge:
        return None

    select_path: Optional[List[int]] = None
    if has_select:
        select_path = _parse_select(m.group("select_raw"))
        if select_path is None:
            return None

    knowledge: Optional[str] = None
    if has_knowledge:
        knowledge = (m.group("knowledge") or "").strip()

    answer = m.group("answer").strip()
    if answer not in VALID_STANCES:
        return None

    return ParsedTraj(
        raw=region,
        think1=m.group("think1").strip(),
        think2=m.group("think2").strip(),
        route=route,
        select_path=select_path,
        knowledge=knowledge,
        answer=answer,
    )


def _think2_conclusion_ok(think2: str, answer: str) -> bool:
    last_pos = -1
    last_stance: Optional[str] = None
    for s in VALID_STANCES:
        p = think2.rfind(s)
        if p > last_pos:
            last_pos = p
            last_stance = s
    return last_stance is not None and last_stance == answer


def validate_for_sft(
    parsed: Optional[ParsedTraj],
    *,
    gold_stance: str,
    depth: int,
    ancestor_chain: List[int],
) -> Tuple[bool, str]:
    if parsed is None:
        return False, "轨迹解析失败：标签顺序/数量/嵌套不符合规范（必须严格输出 think→route→[select_path]→[knowledge]→think→answer 这 6-8 个标签，且只有两个 <think>）"

    if parsed.answer != gold_stance:
        return False, f"<answer> 必须等于 {gold_stance}，但生成了 {parsed.answer}"

    if not _think2_conclusion_ok(parsed.think2, parsed.answer):
        return False, f"第二个 <think> 必须显式给出立场结论且与 <answer>={parsed.answer} 一致"

    if depth <= 1 and parsed.route in ("retrieve", "both"):
        return False, "depth=1 时没有历史轮次可选，route 只能是 direct 或 knowledge"

    if parsed.select_path is not None:
        n = len(parsed.select_path)
        max_k = min(5, depth)
        if n == 0 or n > max_k:
            return False, f"select_path 长度必须在 1 到 min(5, depth)={max_k} 之间，当前为 {n}"
        if len(set(parsed.select_path)) != n:
            return False, "select_path 含重复索引"
        for idx in parsed.select_path:
            if idx not in ancestor_chain:
                return False, f"select_path 中索引 {idx} 不在祖先链 {ancestor_chain} 中"

    if parsed.knowledge is not None:
        kl = len(parsed.knowledge)
        if kl < KNOWLEDGE_MIN or kl > KNOWLEDGE_MAX:
            return False, f"<knowledge> 长度必须在 {KNOWLEDGE_MIN}-{KNOWLEDGE_MAX} 字之间，当前为 {kl} 字"

    return True, ""


def _check_route_honesty(parsed: ParsedTraj) -> bool:
    think1 = parsed.think1
    think2 = parsed.think2
    route = parsed.route

    if route in ("direct", "knowledge"):
        if _TURN_REF_RE.search(think1) is not None:
            return False
        return _TURN_REF_RE.search(think2) is None

    if not parsed.select_path:
        return False
    selected = set(parsed.select_path)
    refs = {int(m.group(1)) for m in _TURN_REF_RE.finditer(think2)}
    return refs == selected


def _knowledge_used_in_think2(knowledge: str, think2: str) -> bool:
    if len(knowledge) < KNOWLEDGE_NGRAM_N:
        return False
    kg_ngrams = {
        knowledge[i:i + KNOWLEDGE_NGRAM_N]
        for i in range(len(knowledge) - KNOWLEDGE_NGRAM_N + 1)
    }
    matches = sum(1 for g in kg_ngrams if g in think2)
    return matches >= KNOWLEDGE_NGRAM_MIN_MATCHES


def compute_grpo_rewards(
    parsed: Optional[ParsedTraj],
    *,
    gold_stance: str,
    depth: int,
    gate_open: bool = False,
) -> dict:
    if parsed is None:
        return {"R_ans": 0.0, "R_fmt": 0.0, "R_legal": 0.0}

    r_fmt = 1.0
    r_ans = 1.0 if parsed.answer == gold_stance else 0.0

    if gate_open:
        legal = _check_route_honesty(parsed)
        if legal and depth <= 1 and parsed.route in ("retrieve", "both"):
            legal = False
        if legal and not _think2_conclusion_ok(parsed.think2, parsed.answer):
            legal = False
        if legal and parsed.route in ("knowledge", "both"):
            if parsed.knowledge is None or not _knowledge_used_in_think2(
                parsed.knowledge, parsed.think2
            ):
                legal = False
        r_legal = 1.0 if legal else 0.0
    else:
        r_legal = 0.0

    return {
        "R_ans": r_ans,
        "R_fmt": r_fmt,
        "R_legal": r_legal,
    }


if __name__ == "__main__":
    cases = [
        ("<think>分析路由</think><route>direct</route><think>综合后判断说话人对目标持支持态度</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 1, "ancestor_chain": []},
         True),
        ("<think>路由</think><route>retrieve</route><select_path>[0,2]</select_path><think>综合上下文，说话人对目标持反对态度</think><answer>反对</answer>",
         {"gold_stance": "反对", "depth": 4, "ancestor_chain": [0, 1, 2]},
         True),
        ("<think>路由</think><route>both</route><select_path>[1]</select_path>"
         "<knowledge>" + "知" * 80 + "</knowledge><think>综合上下文与背景知识，说话人立场为中立</think><answer>中立</answer>",
         {"gold_stance": "中立", "depth": 3, "ancestor_chain": [0, 1]},
         True),
        ("<think>x</think><route>retrieve</route><think>y结论支持</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 3, "ancestor_chain": [0, 1]},
         False),
        ("<think>x</think><route>both</route><select_path>[0]</select_path><think>y结论支持</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 2, "ancestor_chain": [0]},
         False),
        ("<think>x</think><route>direct</route><think>结论反对</think><answer>反对</answer>",
         {"gold_stance": "支持", "depth": 1, "ancestor_chain": []},
         False),
        ("<think>x</think><route>retrieve</route><select_path>[9]</select_path><think>结论支持</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 3, "ancestor_chain": [0, 1]},
         False),
        ("<think>x</think><route>knowledge</route><knowledge>太短</knowledge><think>结论支持</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 1, "ancestor_chain": []},
         False),
        ("<think>x</think><route>retrieve</route><select_path>[0]</select_path><think>结论支持</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 1, "ancestor_chain": []},
         False),
        ("```\n<think>路由</think><route>direct</route><think>结论中立</think><answer>中立</answer>\n```",
         {"gold_stance": "中立", "depth": 1, "ancestor_chain": []},
         True),
        ("<think>路由</think><route>direct</route><think>综合判断说话人对目标持反对态度</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 1, "ancestor_chain": []},
         False),
        ("<think>路由</think><route>direct</route><think>综合推理过程的描述但没下结论</think><answer>支持</answer>",
         {"gold_stance": "支持", "depth": 1, "ancestor_chain": []},
         False),
    ]
    fails = 0
    for i, (text, ctx, expect_ok) in enumerate(cases, 1):
        parsed = parse_trajectory(text)
        ok, reason = validate_for_sft(parsed, **ctx)
        status = "✓" if ok == expect_ok else "✗"
        if ok != expect_ok:
            fails += 1
        print(f"{status} case {i}: ok={ok} expect={expect_ok}  reason={reason!r}")
    print(f"\n{len(cases) - fails}/{len(cases)} passed.")
