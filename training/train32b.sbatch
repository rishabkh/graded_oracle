#!/bin/bash
# The real run: LoRA on Qwen2.5-Coder-32B over the corpus pairs.
# Settings copied from formal-disco/config/distill.yaml so the hardware
# and software sides differ in their data, not their method.
#SBATCH --job-name=disco-train
#SBATCH --partition=gpu
#SBATCH --account=amin_lab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
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

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
PAIRS="${PAIRS:-extender/sft_train.jsonl}"
OUT="${OUT:-runs/v1}"

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
