#!/usr/bin/env python3
"""Convert the baseline and CSD datasets to verl parquet files."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from common.data_split import split_train  # noqa: E402
from common.prompts import CoTPromptTemplate  # noqa: E402
from csd.prompts import CSDPromptTemplate, LABEL_TO_STANCE  # noqa: E402

SPLITS = [("train", "train_data.json"), ("dev", "dev_data.json"), ("test", "test_data.json")]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_traj(path):
    traj = {}
    if not path or not os.path.exists(path):
        return traj
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in o and "trajectory" in o:
                traj[o["id"]] = o["trajectory"]
    return traj


def _csd_messages(tpl, s):
    return [
        {"role": "system", "content": tpl.SYSTEM_PROMPT},
        {"role": "user", "content": tpl._build_user_prompt(s)},
    ]


def _baseline_messages(tpl, s):
    return [
        {"role": "system", "content": tpl.SYSTEM_PROMPT},
        {"role": "user", "content": tpl._build_user_prompt(s)},
    ]


def _csd_extra(s, split, idx):
    # A JSON string avoids pyarrow inferring incompatible variable-length lists.
    depth = len(s["sentences"])
    return {
        "split": split,
        "index": idx,
        "id": s["id"],
        "label": int(s["label"]),
        "depth": depth,
        "current_idx": depth - 1,
        "ancestor_chain": json.dumps(list(range(depth - 1))),
    }


def _baseline_extra(s, split, idx):
    return {"split": split, "index": idx, "id": s["id"], "label": int(s["label"])}


def _rl_row(pipeline, messages, gold_stance, extra):
    return {
        "data_source": pipeline,
        "prompt": messages,
        "ability": "stance",
        "reward_model": {"style": "rule", "ground_truth": gold_stance},
        "extra_info": extra,
    }


def _sft_row(rl_row, messages, response):
    row = dict(rl_row)
    row["messages"] = messages + [{"role": "assistant", "content": response}]
    row["response"] = response
    return row


def _write(rows, out_path):
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(rows)} rows)")


def build(pipeline, src_dir, traj_path, out_dir, split_ratio, split_seed):
    os.makedirs(out_dir, exist_ok=True)
    if pipeline == "csd":
        tpl = CSDPromptTemplate()
        mk_msgs, mk_extra = _csd_messages, _csd_extra
    else:
        tpl = CoTPromptTemplate()
        mk_msgs, mk_extra = _baseline_messages, _baseline_extra

    traj = _load_traj(traj_path) if pipeline == "csd" else {}
    if pipeline == "csd":
        print(f"[traj] loaded {len(traj)} trajectories from {traj_path}")

    for split, fname in SPLITS:
        data = _load_json(os.path.join(src_dir, fname))

        if split == "train":
            sft_part, grpo_part = split_train(data, split_ratio, split_seed)
            print(f"[split] {pipeline} ratio={split_ratio}: "
                  f"sft={len(sft_part)} grpo={len(grpo_part)}")

            sft_rows = []
            dropped = 0
            for i, s in enumerate(sft_part):
                if pipeline == "csd":
                    if s["id"] not in traj:
                        dropped += 1
                        continue
                    response = traj[s["id"]]
                else:
                    response = s["cot_response"]
                msgs = mk_msgs(tpl, s)
                gold = LABEL_TO_STANCE[s["label"]]
                rl = _rl_row(pipeline, msgs, gold, mk_extra(s, "sft_train", i))
                sft_rows.append(_sft_row(rl, msgs, response))
            if dropped:
                print(f"[sft] {pipeline}: kept {len(sft_rows)} / dropped {dropped} (no trajectory)")
            _write(sft_rows, os.path.join(out_dir, f"{pipeline}_sft_train.parquet"))

            grpo_rows = [
                _rl_row(pipeline, mk_msgs(tpl, s), LABEL_TO_STANCE[s["label"]],
                        mk_extra(s, "grpo_train", i))
                for i, s in enumerate(grpo_part)
            ]
            _write(grpo_rows, os.path.join(out_dir, f"{pipeline}_grpo_train.parquet"))
        else:
            rows = [
                _rl_row(pipeline, mk_msgs(tpl, s), LABEL_TO_STANCE[s["label"]],
                        mk_extra(s, split, i))
                for i, s in enumerate(data)
            ]
            _write(rows, os.path.join(out_dir, f"{pipeline}_{split}.parquet"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=["baseline", "csd"], required=True)
    p.add_argument("--src_dir", default="data/base_cot")
    p.add_argument("--traj_path", default="csd/data/sft_traj.jsonl")
    p.add_argument("--out_dir", default="verl_port/data")
    p.add_argument("--split_ratio", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    args = p.parse_args()
    build(args.pipeline, args.src_dir, args.traj_path, args.out_dir,
          args.split_ratio, args.split_seed)


if __name__ == "__main__":
    main()
