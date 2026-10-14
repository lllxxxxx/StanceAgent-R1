from __future__ import annotations

from typing import Any, Dict, List


LABEL_TO_STANCE = {0: "支持", 1: "反对", 2: "中立"}


class CSDPromptTemplate:

    SYSTEM_PROMPT = """你是对话立场检测的标注员。给定目标对象、对话历史、当前发言，生成一条推理轨迹，预测当前发言对目标的立场。

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
1. <answer> 必须严格三选一：支持/反对/中立。
2. 第一个 <think> 只描述语言结构与路由依据，禁止使用以下评价性动词及其同义表达：附和/赞同/支持/认可/辩护/捍卫；否定/反对/批评/贬低/讥讽/反讽；中立/不表态/避而不答。也不得直接断言「立场是 X」「该发言表达 X 态度」等结论性短语。
3. 第一个 <think> 必须以可观察的语言结构为依据决定 route，例如：「含代词「这」指代前文」「以反问句回应前一轮」「列举具体应用场景」「目标为专业术语」等。
4. 选了 retrieve 或 both，第二个 <think> 必须实际引用选中的祖先轮次内容。
5. 选了 knowledge 或 both，<knowledge> 必须事实正确，且第二个 <think> 必须实际使用它。
6. 若祖先链中有当前说话人的发言、且当前发言看起来在为前文立场辩护或修正，第二个 <think> 必须显式说明当前发言相对于前文是【加强/弱化/转向中立/保持原立场】中的哪一种，不要默认归为中立。
7. depth=1 时没有历史轮次，route 只能选 direct 或 knowledge。
8. <knowledge> 中不要杜撰，不确定的事实就改选 retrieve 或 direct。
9. 严格只输出上述 6-8 个标签，第一个字符必须是 <，最后一个字符必须是 >，不要 Markdown 代码块、不要解释性文字。"""

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        sentences: List[str] = sample["sentences"]
        speakers: List[int] = sample["speakers"]
        target: str = sample["target"]
        depth = len(sentences)
        current_idx = depth - 1

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
        ])

        return "\n".join(parts)
