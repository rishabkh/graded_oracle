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

## Making designs on the cluster (16 Nov 2026)

The initiator, the extender and their checks run on the cluster through
`training/make_designs.sbatch`: one 48-core machine of the `shared`
partition, 44 checks at once. On the laptop 150 growth steps took 4.5
hours (15 Nov); the 5,000 of a big run would take about six days.

One-time setup, on a login node (it needs the internet):

```bash
cd ~
curl -L -O https://github.com/YosysHQ/oss-cad-suite-build/releases/download/2026-07-10/oss-cad-suite-linux-x64-20260710.tgz
tar -xzf oss-cad-suite-linux-x64-20260710.tgz     # makes ~/oss-cad-suite
~/oss-cad-suite/bin/yosys -V                       # Yosys 0.67+4, as on the laptop
cd ~/graded_oracle && git pull
```

That build has the laptop's date (`oss-cad-suite/VERSION` there says
20260710). If it is no longer offered, take the nearest date from the
project's releases page; the replay below says whether it judges the same.

Then prove the cluster's checker judges like the laptop's, and that a job
reaches the model. The key is exported in the shell that calls `sbatch`,
never written in a file:

```bash
cd ~/graded_oracle
sbatch --cpus-per-task=12 --mem=32G --time=01:00:00 training/make_designs.sbatch initiator/checker_calibrate.py --slots CHECK_SLOTS --ask-model
tail -f runs/make_designs_<jobid>.out
```

What good looks like in that log: `40 of 40 designs judged the same` and
`the model answered: 'ready'`. The 40 designs and the laptop's verdicts
are in `results/checker_calibration_pilot_designs_2026-11-16.jsonl`
(`initiator/checker_calibrate.py --make` wrote it on the laptop). The one
question costs under a cent. No web proxy is set: FASRC's proxy page says
it is for FASSE and must not be set on Cannon.

A run's steps are in the notes at the top of the job file. The word
`CHECK_SLOTS` in a command stands for the number of checks the machine
runs at once. Kinds lists for a run that may use a kind again are drawn
on the laptop (`initiator/catalog_kinds.py --draw-reusable 0 --name ...`),
where the held-out test kinds are, and reach the cluster with the code.
