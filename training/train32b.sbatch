#!/bin/bash
# The real run: LoRA on Qwen2.5-Coder-32B over the corpus pairs.
# Settings copied from formal-disco/config/distill.yaml so the hardware
# and software sides differ in their data, not their method.
#SBATCH --job-name=disco-train
#SBATCH --partition=gpu
#SBATCH --account=amin_lab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=runs/train_%j.out

set -euo pipefail

module load python/3.12.8-fasrc01 cuda/12.4.1-fasrc01
source "$HOME/envs/disco/bin/activate"

export HF_HOME="/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/models"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

cd "$HOME/graded_oracle"
mkdir -p runs

# All three are required (8 Oct 2026): the old defaults were v1's file and
# runs/v1 under home, and a MODEL left in the shell would have trained on
# top of an old model. Training always starts from the base model:
#   --export=ALL,MODEL=Qwen/Qwen2.5-Coder-32B-Instruct,PAIRS=<file>,OUT=/n/netscratch/...
MODEL="${MODEL:?set MODEL, e.g. MODEL=Qwen/Qwen2.5-Coder-32B-Instruct}"
PAIRS="${PAIRS:?set PAIRS to the training file}"
OUT="${OUT:?set OUT to an absolute path under /n/netscratch}"
# --- guard ---
case "$OUT" in
    /*) ;;
    *) echo "OUT must be an absolute scratch path, not $OUT"; exit 1 ;;
esac
case "$OUT" in
    "$HOME"/*|*'$HOME'*) echo "OUT is in the home folder: refusing ($OUT)"; exit 1 ;;
esac
# a finished model is never overwritten (a slip like OUT=.../runs/v2 would
# replace the model the new one is compared with); a folder a preempted
# run left without a merged model is fine, so a requeued job carries on
if [ -e "$OUT/merged" ]; then
    echo "$OUT already holds a finished model: refusing"; exit 1
fi
# every run in this project trains from the untrained base, never on top
# of an earlier fine-tune
case "$MODEL" in
    Qwen/Qwen2.5-Coder-32B-Instruct) ;;
    *) echo "MODEL must be the untrained base Qwen/Qwen2.5-Coder-32B-Instruct, not $MODEL"; exit 1 ;;
esac
# --- end guard ---

echo "training $MODEL on $PAIRS -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

python training/train_lora.py \
    --pairs "$PAIRS" \
    --model "$MODEL" \
    --out "$OUT" \
    --epochs 3 \
    --merge

echo "=== loading the adapter back and answering one question ==="
python training/check_adapter.py \
    --adapter "$OUT/adapter" \
    --model "$MODEL" \
    --pairs "$PAIRS"

echo "done: $OUT/merged is ready to serve with vllm"
