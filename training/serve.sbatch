#!/bin/bash
# Serve a model so it survives a dropped laptop connection.
# An interactive `srun --pty` server dies with the ssh session; this does
# not. Which model to serve comes from MODEL, so the same file serves the
# base model and the fine-tune:
#
#   sbatch training/serve.sbatch                       # base model
#   sbatch --export=ALL,MODEL=$HOME/graded_oracle/runs/v1/merged \
#          training/serve.sbatch                       # the fine-tune
#
# The log's first lines print the node name, which is what the tunnel
# needs:  ssh -N -L 8000:<node>.rc.fas.harvard.edu:8000 <you>@login...
#SBATCH --job-name=disco-serve
#SBATCH --partition=gpu
#SBATCH --account=amin_lab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=runs/serve_%j.out

set -euo pipefail

module load python/3.12.8-fasrc01 cuda/12.4.1-fasrc01
source "$HOME/envs/vllm/bin/activate"

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME="/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/models"

# Some nodes (especially in gpu_requeue, which mixes hardware) refuse the
# direct card-to-card link and vLLM dies with "NCCL error: invalid usage".
# Falling back to host memory is slower to start and fine for serving.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN

echo "GPUs visible to this job:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
echo "=================================================="
echo " serving : $MODEL"
echo " node    : $(hostname)"
echo " tunnel  : ssh -N -L 8000:$(hostname):8000 $USER@login.rc.fas.harvard.edu"
echo "=================================================="

vllm serve "$MODEL" --served-model-name llm \
    --max-model-len 32768 --tensor-parallel-size 2 --port 8000
