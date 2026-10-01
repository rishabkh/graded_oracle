#!/bin/bash
#SBATCH --job-name=disco-merge
#SBATCH --partition=shared
#SBATCH --account=amin_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=runs/merge_%j.out
# Bake a trained adapter into its base model and write the result to
# scratch. No GPU: merging is arithmetic, so this never waits in the GPU
# queue. About 65 GB for the 32B model, which is why it must not go home.
#
# Submit from ~/graded_oracle:
#   sbatch --export=ALL,ADAPTER=runs/v2/adapter,OUT=/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/runs/v2/merged training/merge.sbatch
set -euo pipefail
module load python/3.12.8-fasrc01
source "$HOME/envs/disco/bin/activate"
export HF_HOME="/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/models"
cd "$HOME/graded_oracle"
mkdir -p runs

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
ADAPTER="${ADAPTER:?set ADAPTER to the folder training saved the adapter into}"
OUT="${OUT:?set OUT to a folder in scratch, never in home}"
echo "merging $ADAPTER into $MODEL -> $OUT"
python training/merge_adapter.py --adapter "$ADAPTER" --model "$MODEL" --out "$OUT"
echo "done: $OUT is ready to serve with vllm"
