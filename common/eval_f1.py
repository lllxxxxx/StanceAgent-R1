import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from common.stance import compute_stance_f1, ID_TO_STANCE


def _disable_grad_ckpt_deep(model):
    for name in ("base_model", "model"):
        sub = getattr(model, name, None)
        if sub is not None and hasattr(sub, "gradient_checkpointing_disable"):
            try:
                sub.gradient_checkpointing_disable()
            except Exception:
                pass
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass


def _enable_grad_ckpt_deep(model):
    for name in ("base_model", "model"):
        sub = getattr(model, name, None)
        if sub is not None and hasattr(sub, "gradient_checkpointing_enable"):
            try:
                sub.gradient_checkpointing_enable()
            except Exception:
                pass
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass


def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render_prompts(tokenizer, samples) -> List[str]:
    return [
        tokenizer.apply_chat_template(
            [m for m in s["messages"] if m["role"] != "assistant"],
            tokenize=False, add_generation_prompt=True,
        )
        for s in samples
    ]


def _print_eval_samples(samples, all_preds, all_labels, k: int, desc: str) -> None:
    n = len(samples)
    if n == 0 or k <= 0:
        return
    k = min(k, n)
    stride = max(1, n // k)
    indices = [min(i * stride, n - 1) for i in range(k)]
    print(f"\n========== [{desc}] sample completions (k={k}/{n}) ==========")
    for rank, idx in enumerate(indices):
        sample = samples[idx]
        user_content = next(
            (m["content"] for m in sample["messages"] if m["role"] == "user"),
            "",
        )
        completion = all_preds[idx] if idx < len(all_preds) else "(no pred)"
        gold = ID_TO_STANCE.get(all_labels[idx], f"?{all_labels[idx]}")
        import re as _re
        m = _re.search(r"<answer>(.*?)</answer>", completion)
        pred = m.group(1).strip() if m else "(no answer tag)"
        match = "✓" if pred == gold else "✗"
        prompt_tail = user_content[-350:] if len(user_content) > 350 else user_content
        comp_show = completion[:600] + (" …[truncated]" if len(completion) > 600 else "")
        print(f"\n--- sample {rank+1}/{k} (idx={idx}) [{match}] gold={gold} pred={pred} ---")
        print(f"[prompt tail]\n{prompt_tail}")
        print(f"[completion]\n{comp_show}")
    print(f"========== end [{desc}] samples ==========\n", flush=True)


@torch.no_grad()
def generate_and_score(
    model,
    tokenizer,
    samples,
    *,
    batch_size: int,
    max_new: int,
    max_length: int,
    handle_grad_ckpt: bool = False,
    desc: str = "F1 eval",
    prompts: Optional[List[str]] = None,
    sample_log_count: int = 0,
    return_preds_labels: bool = False,
):
    was_training = model.training
    model.eval()
    orig_use_cache = getattr(model.config, "use_cache", False)
    if handle_grad_ckpt:
        _disable_grad_ckpt_deep(model)
    model.config.use_cache = True

    orig_pad = tokenizer.padding_side
    orig_trunc = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tid for tid in [im_end_id, tokenizer.eos_token_id] if tid is not None]

    device = next(model.parameters()).device
    max_prompt_len = max_length - max_new

    if prompts is None:
        prompts_all = render_prompts(tokenizer, samples)
    else:
        if len(prompts) != len(samples):
            raise ValueError(
                f"generate_and_score: prompts len {len(prompts)} != samples len {len(samples)}"
            )
        prompts_all = prompts

    all_preds = []
    all_labels = []

    pbar = tqdm(
        range(0, len(samples), batch_size),
        desc=f"[{desc}] bs={batch_size} n={len(samples)}",
        leave=True,
    )
    try:
        for i in pbar:
            j = min(i + batch_size, len(samples))
            batch_prompts = prompts_all[i:j]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
                add_special_tokens=False,
            )
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                stop_strings=["</answer>"],
                tokenizer=tokenizer,
            )
            new_tokens = out[:, prompt_len:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            all_preds.extend(texts)
            all_labels.extend(s["label"] for s in samples[i:j])
    finally:
        tokenizer.padding_side = orig_pad
        tokenizer.truncation_side = orig_trunc
        model.config.use_cache = orig_use_cache
        if handle_grad_ckpt:
            _enable_grad_ckpt_deep(model)
        if was_training:
            model.train()

    if sample_log_count > 0:
        _print_eval_samples(samples, all_preds, all_labels, sample_log_count, desc)

    metrics = compute_stance_f1(all_preds, all_labels)
    if return_preds_labels:
        return metrics, all_preds, all_labels
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new", type=int, default=512)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading base model: {args.model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter: {args.ckpt}")
    model = PeftModel.from_pretrained(base, args.ckpt)

    print(f"Loading samples: {args.data}")
    samples = load_samples(args.data)
    print(f"  {len(samples)} samples")

    metrics = generate_and_score(
        model, tokenizer, samples,
        batch_size=args.batch_size,
        max_new=args.max_new,
        max_length=args.max_length,
        handle_grad_ckpt=False,
        desc="offline eval",
    )
    print("\n===== F1 Results =====")
    for k, v in metrics.items():
        print(f"  {k:<20} = {v}")


if __name__ == "__main__":
    main()
