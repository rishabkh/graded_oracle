"""The cluster job for run 4b: same environment as the training job that
made v1 to v3, the starting model named explicitly, and never a model
written into the home folder (the 30 Sep quota incident)."""
import subprocess
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
JOB = (TRAINING / "train_dpo.sbatch").read_text()
TRAIN = (TRAINING / "train32b.sbatch").read_text()


def test_same_environment_as_the_training_job():
    for line in ("module load python/3.12.8-fasrc01 cuda/12.4.1-fasrc01",
                 'source "$HOME/envs/disco/bin/activate"',
                 "#SBATCH --gres=gpu:2", "#SBATCH --mem=256G"):
        assert line in JOB and line in TRAIN


def test_the_starting_model_and_output_must_be_named():
    assert 'MODEL="${MODEL:?' in JOB and 'OUT="${OUT:?' in JOB


def test_it_refuses_to_write_into_home(tmp_path):
    guard = JOB.split("# --- guard ---")[1].split("# --- end guard ---")[0]
    for out, ok in (("/n/netscratch/x/runs/v4b", True),
                    ("runs/v4b", False), ("$HOME/runs/v4b", False)):
        r = subprocess.run(["bash", "-c", f'HOME=/home/me; OUT="{out}"\n{guard}'],
                           capture_output=True, text=True)
        assert (r.returncode == 0) is ok, (out, r.stdout, r.stderr)


def test_it_runs_the_preference_trainer_and_merges():
    assert 'python training/train_dpo.py --pairs "$PAIRS" --model "$MODEL" --out "$OUT" --merge' in JOB
