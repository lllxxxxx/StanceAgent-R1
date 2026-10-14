from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.data_split import split_train as _split_train
from csd.rejection_filter import parse_trajectory, validate_for_sft


LABEL_TO_STANCE = {0: "支持", 1: "反对", 2: "中立"}


SYSTEM_PROMPT = """你是对话立场检测的标注员。给定目标对象、对话历史、当前发言，生成一条推理轨迹，预测当前发言对目标的立场。任务方会在用户消息末尾告诉你正确立场，但该字段仅用于对齐最终 <answer> 标签的内容——撰写第一个 <think> 时请假装你完全看不到这个字段，仅凭语言信号决定路由。

输出格式（严格，按顺序，不要多余文字）：

<think>
分析当前发言的语言结构（如代词指代、反问、反讽、列举、专业术语等），据此决定路由。不得提前给出立场结论，不得使用任何评价性动词。
</think>

<route>direct | retrieve | knowledge | both</route>

<select_path>[索引列表]</select_path>
  仅 route=retrieve 或 both 时输出。索引为【对话上下文】中 [轮次N] 的 N 值，必须是祖先链子集，数量为 1 到 min(5, depth) 个，用 Python 列表写法，如 [0,2]。

<knowledge>
目标的相关背景知识，事实准确，100-250 字。
</knowledge>
  仅 route=knowledge 或 both 时输出。

<think>
综合发言字面含义、选中的祖先轮次（如有）、目标背景知识（如有），逐步推断说话人对目标的立场。本段结尾必须给出显式结论，例如「因此说话人对目标持支持/反对/中立态度」，以便检验推理过程与 <answer> 一致。
</think>

<answer>支持 | 反对 | 中立</answer>

路由规则：
- direct：发言立场自含 + 目标众所周知
- retrieve：发言指向历史（代词、附议、反讽前文等）+ 目标常见
- knowledge：目标冷门或专业 + 发言对话层面自含
- both：以上都成立

硬约束：
1. <answer> 必须等于给定的正确立场，但请假装你在写第一个 <think> 时看不到正确立场。
2. 第一个 <think> 只描述语言结构与路由依据，禁止使用以下评价性动词及其同义表达：附和/赞同/支持/认可/辩护/捍卫；否定/反对/批评/贬低/讥讽/反讽；中立/不表态/避而不答。也不得直接断言「立场是 X」「该发言表达 X 态度」等结论性短语。
3. 第一个 <think> 必须以可观察的语言结构为依据决定 route，例如：「含代词「这」指代前文」「以反问句回应前一轮」「列举具体应用场景」「目标为专业术语」等。
4. 选了 retrieve 或 both，第二个 <think> 必须实际引用选中的祖先轮次内容。
5. 选了 knowledge 或 both，<knowledge> 必须事实正确，且第二个 <think> 必须实际使用它。
6. 若祖先链中有当前说话人的发言、且当前发言看起来在为前文立场辩护或修正，第二个 <think> 必须显式说明当前发言相对于前文是【加强/弱化/转向中立/保持原立场】中的哪一种，不要默认归为中立。
7. depth=1 时没有历史轮次，route 只能选 direct 或 knowledge。
8. <knowledge> 中不要杜撰，不确定的事实就改选 retrieve 或 direct。
9. 严格只输出上述 6-8 个标签，第一个字符必须是 <，最后一个字符必须是 >，不要 Markdown 代码块、不要解释性文字。"""


def build_user_prompt(sample: Dict[str, Any]) -> str:
    sentences: List[str] = sample["sentences"]
    speakers: List[int] = sample["speakers"]
    target: str = sample["target"]
    depth = len(sentences)
    current_idx = depth - 1
    gold_stance = LABEL_TO_STANCE[sample["label"]]

    parts: List[str] = [
        "以下是一段关于某话题的多人对话，请按规定格式输出当前句说话人对目标的立场推理轨迹。",
        "",
        "【目标】",
        target,
        "",
        "【对话上下文】",
    ]
    if depth > 1:
        for i in range(depth - 1):
            parts.append(f"[轮次{i}] 说话人{speakers[i]}：{sentences[i]}")
    else:
        parts.append("无对话上下文")
    parts.extend([
        "",
        "【当前句】",
        f"[轮次{current_idx}] 说话人{speakers[current_idx]}：{sentences[current_idx]}",
        "",
        "【正确立场（仅供对齐 <answer>，第一个 <think> 中请假装看不到此项）】",
        gold_stance,
    ])
    return "\n".join(parts)


def build_retry_suffix(failure_reason: str, gold_stance: str) -> str:
    return (
        f"\n\n上一次生成失败，原因：{failure_reason}\n"
        f"请重新生成，注意纠正该问题。保持 <answer> 等于「{gold_stance}」。"
    )


def load_seen_ids(*paths: str) -> set:
    seen: set = set()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in obj:
                    seen.add(obj["id"])
    return seen


async def _call_teacher(client, model: str, system: str, user: str,
                        temperature: float, request_timeout: float) -> str:
    kwargs = dict(
        model=model,
        instructions=system,
        input=user,
        timeout=request_timeout,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = await client.responses.create(**kwargs)
    except TypeError:
        kwargs.pop("temperature", None)
        resp = await client.responses.create(**kwargs)
    text = getattr(resp, "output_text", None)
    if text:
        return text
    parts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


async def process_sample(
    sample: Dict[str, Any],
    *,
    client,
    model: str,
    max_retries: int,
    temperature: float,
    request_timeout: float,
    sem: asyncio.Semaphore,
    out_lock: asyncio.Lock,
    out_f,
    dropped_lock: asyncio.Lock,
    dropped_f,
    counters: Dict[str, int],
) -> str:
    depth = len(sample["sentences"])
    ancestor_chain = list(range(depth - 1))
    gold_stance = LABEL_TO_STANCE[sample["label"]]
    base_user = build_user_prompt(sample)

    last_reason: Optional[str] = None
    raw_last: str = ""
    for attempt in range(1, max_retries + 1):
        user_prompt = base_user if last_reason is None else base_user + build_retry_suffix(last_reason, gold_stance)
        async with sem:
            try:
                raw_last = await _call_teacher(
                    client, model, SYSTEM_PROMPT, user_prompt,
                    temperature, request_timeout,
                )
            except Exception as e:
                last_reason = f"API 调用异常：{type(e).__name__}: {e}"
                continue

        parsed = parse_trajectory(raw_last)
        ok, reason = validate_for_sft(
            parsed,
            gold_stance=gold_stance,
            depth=depth,
            ancestor_chain=ancestor_chain,
        )
        if ok:
            row = {
                "id": sample["id"],
                "target": sample["target"],
                "label": sample["label"],
                "gold_stance": gold_stance,
                "depth": depth,
                "ancestor_chain": ancestor_chain,
                "current_idx": depth - 1,
                "route": parsed.route,
                "select_path": parsed.select_path,
                "knowledge_len": len(parsed.knowledge) if parsed.knowledge else 0,
                "trajectory": parsed.raw,
                "attempts": attempt,
            }
            async with out_lock:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
            counters[f"ok_attempt_{attempt}"] = counters.get(f"ok_attempt_{attempt}", 0) + 1
            return "ok"
        last_reason = reason

    drop_row = {
        "id": sample["id"],
        "target": sample["target"],
        "label": sample["label"],
        "last_reason": last_reason,
        "last_raw": raw_last,
    }
    async with dropped_lock:
        dropped_f.write(json.dumps(drop_row, ensure_ascii=False) + "\n")
        dropped_f.flush()
    counters["dropped"] = counters.get("dropped", 0) + 1
    return "drop"


async def amain(args):
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        print(f"[fatal] openai package required: {e}", file=sys.stderr)
        sys.exit(2)

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.split_ratio is not None:
        sft_part, grpo_part = _split_train(data, args.split_ratio, args.split_seed)
        data = sft_part
        print(f"[builder] split_train(ratio={args.split_ratio}, seed={args.split_seed}): "
              f"SFT={len(sft_part)} GRPO={len(grpo_part)}; keeping SFT half")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.dropped) or ".", exist_ok=True)

    seen = load_seen_ids(args.out, args.dropped)
    todo = [s for s in data if s["id"] not in seen]
    if args.shuffle_seed is not None:
        import random as _random
        _random.Random(args.shuffle_seed).shuffle(todo)
    if args.max_n is not None:
        todo = todo[: args.max_n]

    print(f"[builder] src={args.src} total={len(data)} seen={len(seen)} todo={len(todo)} "
          f"model={args.model} concurrency={args.concurrency} max_retries={args.max_retries}")

    if not todo:
        print("[builder] nothing to do.")
        return

    api_key = args.api_key or os.environ.get("TEACHER_API_KEY", "")
    base_url = args.base_url or os.environ.get("TEACHER_BASE_URL", "")
    if not api_key or not base_url:
        print("[fatal] --base-url and --api-key (or TEACHER_BASE_URL / TEACHER_API_KEY env) required",
              file=sys.stderr)
        sys.exit(2)

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=args.request_timeout,
        max_retries=args.sdk_max_retries,
    )
    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    dropped_lock = asyncio.Lock()
    counters: Dict[str, int] = {}

    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out_f, \
         open(args.dropped, "a", encoding="utf-8") as dropped_f:

        tasks = [
            asyncio.create_task(process_sample(
                s,
                client=client, model=args.model,
                max_retries=args.max_retries, temperature=args.temperature,
                request_timeout=args.request_timeout,
                sem=sem, out_lock=out_lock, out_f=out_f,
                dropped_lock=dropped_lock, dropped_f=dropped_f,
                counters=counters,
            ))
            for s in todo
        ]

        done_count = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            done_count += 1
            if done_count % max(1, len(todo) // 20) == 0 or done_count == len(todo):
                elapsed = time.time() - t0
                print(f"[builder] {done_count}/{len(todo)}  elapsed={elapsed:.1f}s  counters={counters}")

    try:
        await client.close()
    except Exception:
        pass

    print(f"[builder] done in {time.time() - t0:.1f}s. counters={counters}")
    print(f"[builder] out  → {args.out}")
    print(f"[builder] drop → {args.dropped}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/base_cot/train_data.json")
    p.add_argument("--out", default="csd/data/sft_traj.jsonl")
    p.add_argument("--dropped", default="csd/data/sft_dropped.jsonl")
    p.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY", ""))
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--sdk-max-retries", type=int, default=8)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--split-ratio", type=float, default=None)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--request-timeout", type=float, default=120.0)
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
