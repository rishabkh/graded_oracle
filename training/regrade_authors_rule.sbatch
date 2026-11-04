#!/bin/bash
#SBATCH --job-name=disco-regrade
#SBATCH --partition=shared
#SBATCH --account=amin_lab
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/regrade_%j.out
# Re-grade the saved one-try answers with the benchmark authors' rule
# (initiator/regrade_authors_rule.py): EBMC only, no GPU. Every finished
# answer is written at once, so a job that runs out of time is simply
# sent again with the same --out and carries on.
#
# From ~/graded_oracle (hard first: it is the set that decides):
#   sbatch training/regrade_authors_rule.sbatch --sets hard --out initiator/logs/regraded_seven_sets_authors_rule_2026-11-03.jsonl
#   sbatch training/regrade_authors_rule.sbatch --sets main_experiment --out initiator/logs/regraded_seven_sets_authors_rule_2026-11-03.jsonl
set -euo pipefail

# gcc pinned to the version build_ebmc.sbatch compiled EBMC with: the
# binary needs that compiler's C++ library
module load python/3.12.8-fasrc01 gcc/13.2.0-fasrc01
source "$HOME/envs/vllm/bin/activate"

cd "$HOME/graded_oracle"
mkdir -p runs
echo "code $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
export EBMC_PATH="$HOME/bin/ebmc"
# each answer runs up to 4 EBMC checks at once
python initiator/regrade_authors_rule.py \
    --workers $(( ${SLURM_CPUS_PER_TASK:-8} / 4 )) "$@"
