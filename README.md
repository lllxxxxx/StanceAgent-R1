# StanceAgent-R1

StanceAgent-R1 is a Chinese multi-party dialogue stance-detection project built on
[verl](https://github.com/volcengine/verl). Given a target, a dialogue history,
and the current utterance, it predicts one of three labels: `支持`, `反对`, or
`中立`.

The repository contains two training pipelines:

- **Baseline CoT** generates a reasoning trace followed by
  `<answer>支持|反对|中立</answer>`. Its reward is
  `accuracy + 0.5 × format`.
- **CSD adaptive routing** chooses `direct`, `retrieve`, `knowledge`, or `both`
  before producing the final stance. Its reward is
  `R_ans + 0.4 × R_fmt + 0.2 × R_legal`.

`R_legal` uses a one-way curriculum gate. It stays disabled until the file named
by `CSD_GATE_FLAG_PATH` exists, then checks route honesty, dialogue depth, answer
consistency, and use of generated knowledge.

## Repository layout

```text
common/                Shared prompts, deterministic splits, and F1 evaluation
csd/                   CSD prompts, trajectory parser, reward logic, teacher-data builder
csd/data/              Filtered teacher trajectories for CSD SFT
data/default/          Stance data without CoT responses
data/base_cot/         Stance data with baseline CoT responses
verl_port/             Parquet conversion, verl rewards, and training launchers
```

The JSON datasets use the fields `id`, `target`, `sentences`, `speakers`, and
`label`; `base_cot` also contains `cot_response`. Account mentions and literal IP
addresses in the published data are anonymized.

The source data is [ZS-CSD](https://github.com/whu-yzding/ZS-CSD), introduced in
[Ding et al. (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.168/).
`base_cot` adds generated chain-of-thought responses, and `csd/data` adds filtered
teacher trajectories. Consult the upstream project for the dataset's original
terms and citation requirements.

## Setup

The tested training stack uses Python 3.12, PyTorch 2.8, vLLM 0.11, and a
source installation of verl. Install verl separately:

```bash
git clone https://github.com/volcengine/verl
python -m pip install --no-deps -e verl
python -m pip install pandas pyarrow tqdm openai peft \
  "transformers>=4.55.2,<5.0" "tensordict==0.10.0"
```

GPU-specific PyTorch, vLLM, and flash-attn builds depend on the local CUDA
environment. The launchers default to `Qwen/Qwen2.5-3B-Instruct`; set
`MODEL_PATH` to use a local model.

## Prepare verl data

```bash
python verl_port/make_parquet.py --pipeline csd
python verl_port/make_parquet.py --pipeline baseline
```

The commands create `{pipeline}_{sft_train,grpo_train,dev,test}.parquet` under
`verl_port/data/`. The generated files are ignored by Git.

## Run checks

```bash
python csd/rejection_filter.py
python verl_port/baseline_reward.py
python verl_port/csd_reward.py
```

## Train

```bash
bash verl_port/configs/sft_3b.sh baseline outputs/baseline_sft

python -m verl.model_merger merge --backend fsdp \
  --local_dir outputs/baseline_sft/global_step_<N> \
  --target_dir outputs/baseline_sft_merged

MODEL_PATH=outputs/baseline_sft_merged \
  bash verl_port/configs/grpo_3b.sh baseline
```

Use `csd` instead of `baseline` for the adaptive-routing pipeline. The DAPO arm
is available through `verl_port/configs/dapo_3b.sh`.

The CSD gate is manual in the current launcher:

```bash
export CSD_GATE_FLAG_PATH=outputs/csd_gate_open
mkdir -p outputs
# After an independent dev evaluation reaches macro-F1 >= 0.6:
touch "$CSD_GATE_FLAG_PATH"
```

Do not create the flag before reaching the threshold. Once a worker observes the
file, its in-process gate remains open.

See [verl_port/README.md](verl_port/README.md) for the tested runtime constraints
and [verl_port/RESULTS.md](verl_port/RESULTS.md) for the baseline comparison.
