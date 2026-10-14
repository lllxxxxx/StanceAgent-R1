from typing import Any, Dict


class CoTPromptTemplate:

    SYSTEM_PROMPT = (
        "你是一个对话立场分析专家。你需要根据多人对话的上下文，判断当前句说话人对给定目标的立场。\n"
        "注意：说话人可能不会直接提及目标，而是通过回应他人观点、使用反语反问、或表达情感态度来间接表明立场。"
        "你需要结合完整上下文进行推理。"
    )

    def _build_user_prompt(self, sample: Dict[str, Any]) -> str:
        target = sample["target"]
        sentences = sample["sentences"]
        speakers = sample["speakers"]

        context_lines = []
        for i in range(len(sentences) - 1):
            context_lines.append(f"说话人{speakers[i]}：{sentences[i]}")

        current_speaker = speakers[-1]
        current_sentence = sentences[-1]

        parts = [
            "以下是一段关于某话题的多人对话，请判断当前句说话人对目标的立场。",
            "",
            "【目标】",
            target,
        ]

        if context_lines:
            parts.append("")
            parts.append("【对话上下文】")
            parts.extend(context_lines)

        parts.extend([
            "",
            "【当前句】",
            f"说话人{current_speaker}：{current_sentence}",
            "",
            "请先在<think>标签内逐步分析：",
            "1. 上文中各说话人分别表达了什么观点？与目标的关系是什么？",
            "2. 当前句表达了什么内容？与上文的关系是什么？（注意识别反语、反问、隐喻、表情符号等间接表达）",
            "3. 综合以上分析，当前句说话人对目标的立场是什么？",
            "然后在<answer>标签内给出立场，仅限：支持、反对、中立。",
            "",
            "<think>你的分析</think><answer>立场</answer>",
        ])

        return "\n".join(parts)
