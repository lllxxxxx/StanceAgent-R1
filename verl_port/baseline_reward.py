"""Baseline verl reward: accuracy + 0.5 * format."""
from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>")
FORMAT_RE = re.compile(r"<think>.+</think>\s*<answer>.+</answer>", re.DOTALL)
VALID_STANCES = {"支持", "反对", "中立"}


def _extract_stance(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _accuracy_score(completion: str, gt: str) -> float:
    return 1.0 if _extract_stance(completion) == _extract_stance(gt) else 0.0


def _format_score(completion: str) -> float:
    if FORMAT_RE.search(completion):
        stance = _extract_stance(completion)
        return 1.0 if stance in VALID_STANCES else 0.5
    return 0.0


W_ACC = 1.0
W_FMT = 0.5


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Accept a plain stance or an ``<answer>`` tag as ground truth."""
    acc = _accuracy_score(solution_str, ground_truth)
    fmt = _format_score(solution_str)
    return {
        "score": W_ACC * acc + W_FMT * fmt,
        "accuracy": acc,
        "format": fmt,
    }


if __name__ == "__main__":
    ok = "<think>分析</think><answer>支持</answer>"
    bad = "支持"
    print("ok :", compute_score("baseline", ok, "支持"))
    print("bad:", compute_score("baseline", bad, "支持"))
