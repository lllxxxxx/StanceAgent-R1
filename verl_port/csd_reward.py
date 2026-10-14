"""CSD verl reward: R_ans + 0.4 * R_fmt + 0.2 * R_legal."""
from __future__ import annotations

import os
import sys

# Keep project imports available when verl starts from another directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from csd.rejection_filter import parse_trajectory, compute_grpo_rewards  # noqa: E402

W_FMT = 0.4
W_LEGAL = 0.2

# Once opened, the curriculum gate stays open for this process.
_GATE_OPEN = {"v": False}


def _gate_open() -> bool:
    if _GATE_OPEN["v"]:
        return True
    p = os.environ.get("CSD_GATE_FLAG_PATH")
    if p and os.path.exists(p):
        _GATE_OPEN["v"] = True
        return True
    return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    extra_info = extra_info or {}
    depth = int(extra_info.get("depth", 1))

    parsed = parse_trajectory(solution_str)
    gate = _gate_open()
    r = compute_grpo_rewards(
        parsed, gold_stance=ground_truth, depth=depth, gate_open=gate
    )
    total = r["R_ans"] + W_FMT * r["R_fmt"] + W_LEGAL * r["R_legal"]
    return {
        "score": total,
        "R_ans": r["R_ans"],
        "R_fmt": r["R_fmt"],
        "R_legal": r["R_legal"],
        "parsed": 0.0 if parsed is None else 1.0,
        "gate_open": 1.0 if gate else 0.0,
    }


if __name__ == "__main__":
    ok = "<think>分析路由</think><route>direct</route><think>综合判断说话人对目标持支持态度</think><answer>支持</answer>"
    bad = "garbage no tags"
    print("ok :", compute_score("csd", ok, "支持", {"depth": 1}))
    print("bad:", compute_score("csd", bad, "支持", {"depth": 1}))
