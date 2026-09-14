# Fine-tune, step by step

The smoke test exists to prove the plumbing before any GPU hours go into
the real run: train a few steps, save, load the saved weights back, ask
one question. If all four work, the setup is good.

Answers only in run 1. Reasoning is run 2, so the difference between the
two runs is a measurement and not a guess.

## One-time setup on Cannon (login node, it needs the internet)

```bash
python -m venv ~/envs/disco
source ~/envs/disco/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate peft datasets bitsandbytes

export HF_HOME=$SCRATCH/hf
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct     # smoke, ~15GB
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct    # real, ~65GB
```

Copy the repo over, or clone it, so that `~/graded_oracle` exists.

## Smoke test

Fill in the partition, account and module names at the top of
`training/smoke.sbatch`, then:

```bash
sbatch training/smoke.sbatch
squeue -u $USER
tail -f runs/smoke_<jobid>.out
```

What good looks like in that log:

1. `149 pairs -> 134 train, 14 eval, N over 8192 tokens and dropped`
2. `trainable params: ...` a small number next to the total
3. a loss that prints every step and goes down
4. `saved adapter -> runs/smoke/adapter`
5. a model answer that looks like a list of invariants

The answer does not have to be correct. This is a 7B model after 20
steps. It has to load and produce the right shape of answer.

## Real run

Same script, bigger model, whole file:

```bash
python training/train_lora.py \
    --pairs extender/sft_train.jsonl \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --out runs/v1 --epochs 3 --load-4bit
```

`--load-4bit` keeps 32B on one 80GB card. Drop it if you have two.

## Files

- `sft_data.py` reads the pairs, makes the chat turns, splits, and hides
  the question from the loss. No torch, so it is tested on the laptop.
- `train_lora.py` the run itself. Rows over the length budget are
  dropped and counted, never cut in half.
- `check_adapter.py` loads the saved adapter and answers one question.
- `smoke.sbatch` the job file.
