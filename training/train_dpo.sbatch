#!/bin/bash
# Run 4b: preference training on top of run 4a (training/train_dpo.py).
# The starting model is also the reference, so name it explicitly; the
# merged result is about 65 GB, so OUT must be in scratch, never home.
#
#   sbatch --export=ALL,MODEL=<scratch>/runs/v4a/merged,OUT=<scratch>/runs/v4b training/train_dpo.sbatch
#SBATCH --job-name=disco-dpo
#SBATCH --partition=gpu
#SBATCH --account=amin_lab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=runs/dpo_%j.out

set -euo pipefail

module load python/3.12.8-fasrc01 cuda/12.4.1-fasrc01
source "$HOME/envs/disco/bin/activate"

export HF_HOME="/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/models"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

cd "$HOME/graded_oracle"
mkdir -p runs

MODEL="${MODEL:?set MODEL to the run-4a merged model in scratch}"
PAIRS="${PAIRS:-extender/pref_v4b.jsonl}"
OUT="${OUT:?set OUT to a folder in scratch}"
# --- guard ---
case "$OUT" in
    /*) ;;
    *) echo "OUT must be an absolute scratch path, not $OUT"; exit 1 ;;
esac
case "$OUT" in
    "$HOME"/*|*'$HOME'*) echo "OUT is in the home folder: refusing ($OUT)"; exit 1 ;;
esac
# --- end guard ---

echo "preference training $MODEL on $PAIRS -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

python training/train_dpo.py --pairs "$PAIRS" --model "$MODEL" --out "$OUT" --merge

echo "done: $OUT/merged is ready to serve with vllm"
