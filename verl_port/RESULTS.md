# verl migration — baseline results (Qwen2.5-3B, single RTX 5090)

Baseline comparison on a single RTX 5090 (32 GB).

## Setup

- Model: Qwen2.5-3B-Instruct, LoRA r32/a64, SFT cold-start → merged → GRPO.
- SFT: 36 steps, val/loss 1.83→1.33. Merged model dev accuracy = **29.5%**.
- GRPO/DAPO: 50 steps each, n=8, batch=64, micro=4, gpu_mem=0.4,
  **model_dtype=bf16** (the key fix for 32GB colocate OOM), ~3 min/step, 0 OOM.
- Both arms start from the SAME merged SFT model; only the algorithm differs.
- Reward = 1.0·accuracy + 0.5·format. Dev set = 2560 samples.

## Result — dev accuracy

| step | Vanilla GRPO | DAPO (clip-higher 0.28 + no KL) |
|------|--------------|---------------------------------|
| 0 (SFT) | 29.5% | 29.5% |
| 25   | 36.7% | 37.2% |
| 50   | **37.3%** | **38.5%** |

- Both GRPO variants lift baseline stance accuracy by ~+8 points over the SFT
  cold-start (29.5% → 37–38.5%). The verl pipeline works end-to-end.
- **DAPO edges vanilla by ~1.2 pts** (38.5% vs 37.3%) on this baseline / 50 steps.
- **Both are numerically stable**: grad_norm ~0.14–0.16 throughout, no collapse,
  0 OOM. Notably DAPO — the exact configuration that collapsed (grad_norm >10) in
  the swift pipeline — runs cleanly in verl. The migration fixed the instability.

## Caveats

- 50 steps is a short run (capped for overnight time); neither is converged. The
  ranking could shift with more steps / a full epoch.
- This is the **baseline** pipeline (accuracy+format reward). CSD (adaptive
  routing + F1-gated R_legal) is the next migration target.
- GPU was run at the stable ~50–70% utilization (bf16 config); a more aggressive
  config could push higher util now that the fp32 memory wall is removed.
