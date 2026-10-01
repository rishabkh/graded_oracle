#!/bin/bash
#SBATCH --job-name=disco-ebmc
#SBATCH --partition=shared
#SBATCH --account=amin_lab
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=runs/build_ebmc_%j.out
# Build the same EBMC the laptop scored with (6.0, commit 0308d417), copy
# it to ~/bin/ebmc, then prove it judges identically by replaying 40
# stored answers. No score from it counts unless all 40 match.
#
# Fetch the source first, on a login node (it needs the internet):
#   cd /n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco
#   git clone https://github.com/diffblue/hw-cbmc.git && cd hw-cbmc
#   git checkout 0308d417 && git submodule update --init --recursive
#   make -C lib/cbmc/src minisat2-download
# Then, from ~/graded_oracle:   sbatch training/build_ebmc.sbatch
set -euo pipefail

COMMIT=0308d417
SRC="${SRC:-/n/netscratch/amin_lab/Lab/$USER/hardware-formal-disco/hw-cbmc}"
# GCC pinned: an unversioned load gave GCC 16.2 on 1 Oct 2026, whose new
# checks break this CBMC version (it treats every warning as an error).
# serve_and_score.sbatch loads the same version to run the binary.
module load python/3.12.8-fasrc01 gcc/13.2.0-fasrc01
echo "compiler: $(g++ --version | head -1)"

cd "$SRC"
if [ "$(git rev-parse --short=8 HEAD)" != "$COMMIT" ]; then
    echo "source in $SRC is not at $COMMIT - check it out first"; exit 1
fi
make -C src -j"${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$HOME/bin"
cp src/ebmc/ebmc "$HOME/bin/ebmc"
"$HOME/bin/ebmc" --version

cd "$HOME/graded_oracle"
EBMC_PATH="$HOME/bin/ebmc" EBMC_TIMEOUT_S=300 python initiator/ebmc_calibrate.py
echo "done: $HOME/bin/ebmc judges exactly as the laptop's EBMC did"
