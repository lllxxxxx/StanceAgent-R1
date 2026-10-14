# verl training notes

`verl_port/` adapts the baseline and CSD pipelines to verl without modifying
verl itself. Prompts and deterministic data splits come from `common/` and
`csd/`; generated parquet files are written to `verl_port/data/`.

## Compatibility

The validated stack uses:

- PyTorch 2.8 with a matching CUDA build
- vLLM 0.11
- `transformers>=4.55.2,<5.0`
- `tensordict==0.10.0`
- verl installed from source with `--no-deps`

The scripts select SDPA and disable remove-padding for GPUs without compatible
flash-attn kernels. verl may still import `flash_attn.bert_padding`, so install a
wheel matching the local Python, PyTorch, CUDA, and C++ ABI.

Do not enable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for GRPO when
vLLM shares the training process; the colocated allocator is incompatible with
that option.

## Data conversion

From the repository root:

```bash
python verl_port/make_parquet.py --pipeline csd
python verl_port/make_parquet.py --pipeline baseline
```

Expected row counts with the included split are:

| Pipeline | SFT train | GRPO train | Dev | Test |
|---|---:|---:|---:|---:|
| Baseline | 1,167 | 10,511 | 2,560 | 2,542 |
| CSD | 1,155 | 10,511 | 2,560 | 2,542 |

Twelve CSD SFT samples are omitted because no accepted teacher trajectory is
available.

## Rewards

- `baseline_reward.py`: `accuracy + 0.5 × format`
- `csd_reward.py`: `R_ans + 0.4 × R_fmt + 0.2 × R_legal`

For CSD, `R_legal` becomes active after the path in `CSD_GATE_FLAG_PATH` exists.
The launcher does not create the file automatically. Set the path before
training and create the file only after an independent dev evaluation reaches
macro-F1 0.6. The in-process latch does not close again.

## Training

The launchers accept additional Hydra overrides after their positional arguments:

```bash
bash verl_port/configs/sft_3b.sh csd outputs/csd_sft \
  data.train_batch_size=32

MODEL_PATH=outputs/csd_sft_merged \
  bash verl_port/configs/grpo_3b.sh csd \
  trainer.total_training_steps=50
```

The default GRPO configuration uses one GPU, eight rollouts, a batch size of 64,
LoRA, KL regularization, eager vLLM execution, and no prefix cache. The DAPO
launcher reuses the same configuration while enabling asymmetric clipping and
removing the KL term.

The measured baseline comparison is recorded in [RESULTS.md](RESULTS.md).
