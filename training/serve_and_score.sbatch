#!/bin/bash
#SBATCH --job-name=disco-score
#SBATCH --partition=gpu
#SBATCH --account=amin_lab
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=runs/score_%j.out
# Serve a model and score it on the Large Lemma Miners sets in one job,
# so a run needs no laptop and leaves no idle GPUs: the server stops the
# moment the passes end, or the moment anything fails.
#
# The server starts exactly as serve.sbatch starts it (same settings,
# environment and GPUs), and answers are judged by the cluster EBMC that
# build_ebmc.sbatch proved identical to the laptop's, at the 300 s limit
# the earlier scores used. So these scores compare with the old ones.
#
# From ~/graded_oracle:
#   sbatch --export=ALL,MODEL=/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/runs/v2/merged training/serve_and_score.sbatch
#   sbatch --export=ALL,PASSES=1 training/serve_and_score.sbatch     # untrained model, one pass of each set
#   repair-loop runs check many lemmas at once with EBMC, which needs cores
#   and memory the one-shot runs do not; ask for them on the sbatch line:
#   sbatch --cpus-per-task=8 --mem=256G --export=ALL,ROUNDS=5,SHOW_CEX=1,... training/serve_and_score.sbatch
#   the repair loop with and without its feedback, one after the other in
#   one job, hard set only (about 2.5 hours each for v2, so more time):
#   sbatch --cpus-per-task=8 --mem=256G --time=08:00:00 --export=ALL,PASSES=1,ROUNDS=5,SHOW_CEX=1,FEEDBACK=both,SETS=hard,MODEL=... training/serve_and_score.sbatch
set -euo pipefail

# gcc pinned to the version build_ebmc.sbatch compiled EBMC with
module load python/3.12.8-fasrc01 cuda/12.4.1-fasrc01 gcc/13.2.0-fasrc01
source "$HOME/envs/vllm/bin/activate"

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME="/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/models"
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN

cd "$HOME/graded_oracle"
mkdir -p runs
# which code this job ran: a git pull during a job changes the scorer
# between passes (v4b's last pass, 6 Oct 2026)
echo "code $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
PASSES="${PASSES:-5}"
# the benchmark authors' repair loop (initiator/repair_loop.py); 0 = one answer
ROUNDS="${ROUNDS:-0}"
SHOW_CEX="${SHOW_CEX:-0}"
# on: the authors' feedback after each answer; off: the control, told only
# that the design is not solved yet; both: the two, in this one job
FEEDBACK="${FEEDBACK:-on}"
SETS="${SETS:-hard main_experiment}"
case "$FEEDBACK" in
    on) ARMS="on" ;;
    off) ARMS="off" ;;
    both) ARMS="on off" ;;
    *) echo "FEEDBACK must be on, off or both, not $FEEDBACK"; exit 1 ;;
esac
if [ "$FEEDBACK" != "on" ] && [ "$ROUNDS" = "0" ]; then
    echo "FEEDBACK=$FEEDBACK needs ROUNDS: there is no repair loop to change"; exit 1
fi
for s in $SETS; do
    case "$s" in hard|main_experiment) ;;
        *) echo "unknown set $s (hard, main_experiment)"; exit 1 ;;
    esac
done
EBMC_BIN="$HOME/bin/ebmc"
if [ ! -x "$EBMC_BIN" ]; then
    echo "no $EBMC_BIN - run build_ebmc.sbatch first"; exit 1
fi
echo "serving $MODEL on $(hostname); $PASSES pass(es) of $SETS; ROUNDS=$ROUNDS SHOW_CEX=$SHOW_CEX FEEDBACK=$FEEDBACK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

VLOG="runs/score_${SLURM_JOB_ID}_vllm.log"
vllm serve "$MODEL" --served-model-name llm \
    --max-model-len 32768 --tensor-parallel-size 2 --port 8000 > "$VLOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null || true' EXIT

# give the model up to 45 minutes to load; stop early if the server dies
for _ in $(seq 1 270); do
    curl -sf http://localhost:8000/v1/models > /dev/null && break
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "server died while loading - see $VLOG"; exit 1
    fi
    sleep 10
done
if ! curl -sf http://localhost:8000/v1/models > /dev/null; then
    echo "server never became ready - see $VLOG"; exit 1
fi
echo "server ready"

export QWEN_BASE_URL=http://localhost:8000/v1 QWEN_API_KEY=none QWEN_MODEL=llm
export EBMC_PATH="$EBMC_BIN" EBMC_TIMEOUT_S=300
REPAIR=""
if [ "$ROUNDS" != "0" ]; then REPAIR="--rounds $ROUNDS --ebmc-workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))"; fi
for i in $(seq 1 "$PASSES"); do
    echo "=== pass $i of $PASSES ==="
    for s in $SETS; do
        case "$s" in hard) n=31 ;; main_experiment) n=78 ;; esac
        for arm in $ARMS; do
            ARM=""
            if [ -n "$REPAIR" ] && [ "$arm" = "off" ]; then ARM="--no-feedback"
            elif [ -n "$REPAIR" ] && [ "$SHOW_CEX" = "1" ]; then ARM="--show-cex"; fi
            [ -n "$REPAIR" ] && echo "--- $s, feedback $arm ---"
            python initiator/benchmark_solve.py --solver qwen --set "$s" --n "$n" $REPAIR $ARM
        done
    done
done
echo "done: $PASSES pass(es) of $SETS scored; stopping the server"
