#!/bin/bash
#SBATCH --job-name=disco-smoke
#SBATCH --partition=gpu_test         # 12h cap, short queue; use gpu for the real run
#SBATCH --account=amin_lab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/smoke_%j.out

set -euo pipefail

module load python/3.12.8-fasrc01 cuda/12.4.1-fasrc01
source "$HOME/envs/disco/bin/activate"

# models are large, keep them off the small home folder
export HF_HOME="/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/models"
export TOKENIZERS_PARALLELISM=false

cd "$HOME/graded_oracle"

# override with: sbatch --export=ALL,MODEL=Qwen/Qwen2.5-Coder-7B-Instruct ...
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
# any training file (9 Oct 2026), e.g. the chat-record sample
# extender/sft_smoke_with_design_writing.jsonl:
#   sbatch --export=ALL,PAIRS=extender/sft_smoke_with_design_writing.jsonl training/smoke.sbatch
PAIRS="${PAIRS:-extender/sft_smoke.jsonl}"
echo "practice run on $MODEL with $PAIRS"

python training/train_lora.py \
    --pairs "$PAIRS" \
    --model "$MODEL" \
    --out runs/smoke \
    --steps 20

python training/check_adapter.py \
    --adapter runs/smoke/adapter \
    --model "$MODEL" \
    --pairs "$PAIRS"
