#!/bin/bash
#SBATCH --job-name=disco-designs
#SBATCH --partition=shared
#SBATCH --account=amin_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=runs/make_designs_%j.out
# Make and check designs on the cluster, so a big run needs no laptop
# (16 Nov 2026). On 15 Nov, 150 growth steps took 4.5 hours of the laptop
# at 6 checks at once; the 5,000 of the big run would take about six days.
# This job runs one step of making designs on a whole 48-core machine: the
# initiator, the corpus build, the labels, the extender, or the checker
# replay. Whatever follows the job file's name is the step: a script of
# this repo and its options. The word CHECK_SLOTS in the options stands
# for the number of checks this machine runs at once (every core but 4).
#
# Once, on a login node (it needs the internet): the tool bundle the
# laptop uses, in its Linux build, unpacked as ~/oss-cad-suite (see
# training/README.md). Then prove this machine judges like the laptop,
# and that a job reaches the model (one question, under a cent):
#   sbatch --cpus-per-task=12 --mem=32G --time=01:00:00 training/make_designs.sbatch initiator/checker_calibrate.py --slots CHECK_SLOTS --ask-model
#
# The steps of a run, each from ~/graded_oracle with the key exported in
# that shell first (never written in a file):
#   sbatch training/make_designs.sbatch initiator/run.py pilot --n 1000 --repairs 2 --workers 60 --grade-slots CHECK_SLOTS --freeze-check --recheck-timeout-s 900 --even-seeds --constructs <kinds list> --scales initiator/scales_real_hardware.txt --log <log file>
#   sbatch training/make_designs.sbatch extender/build_corpus.py <its options>
#   sbatch --cpus-per-task=8 --mem=16G training/make_designs.sbatch extender/label_designs.py --corpus <corpus> --out <labels>
#   sbatch training/make_designs.sbatch extender/batch.py --corpus <corpus> --labels <labels> --max-calls 1000 --workers CHECK_SLOTS --freeze-check --log-dir <folder>
# Direct calls, not --batch: a half-price batch can stay out for hours
# (6.5 hours and counting on 15 Nov), and the cores would sit idle
# meanwhile, which FASRC's Job Defense Shield flags.
#
# No web proxy is set: FASRC's proxy page says the proxy is for FASSE and
# must not be set on Cannon, where it breaks the internet.
set -euo pipefail

: "${OPENAI_API_KEY:?export OPENAI_API_KEY in the shell you call sbatch from (never in a file)}"
[ "$#" -gt 0 ] || { echo "nothing to run: give a script and its options after the job file"; exit 1; }
TOOLS="${OSS_CAD_SUITE:-$HOME/oss-cad-suite}"
[ -f "$TOOLS/environment" ] || { echo "no tool bundle at $TOOLS - see training/README.md"; exit 1; }

module load python/3.12.8-fasrc01
# yosys, sby, yices and iverilog, the same builds as the laptop's
source "$TOOLS/environment"
# the bundle puts its own python3 first on the path, and that one has no
# openai package: always this environment's python, by its full name
PY="$HOME/envs/vllm/bin/python"

cd "$HOME/graded_oracle"
mkdir -p runs
echo "code $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
export LLM_PROVIDER=openai
unset CLAUDE_PROVIDER
# 4 cores are left for the run itself, its model calls and the tools'
# own helpers; a check is one or two busy processes
CHECK_SLOTS=$(( ${SLURM_CPUS_PER_TASK:-8} - 4 ))
echo "checks at once: $CHECK_SLOTS"
args=("${@//CHECK_SLOTS/$CHECK_SLOTS}")
"$PY" "${args[@]}"
