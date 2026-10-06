"""Ask the cluster only for what each job uses. On 6 Oct 2026 FASRC's Job
Defense Shield flagged job 50445664 (v3 training): 8 cores allocated,
10.8% used, about one core. Wasted allocation lowers the lab's standing
and lengthens queues. Measured: training and serving use one to a few
cores and about 44 GB of memory; EBMC build and the CPU merge really run
in parallel. The repair-loop scoring runs many EBMC checks at once, so it
asks for more on the sbatch line, not in the file."""
import re
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"


def ask(name, key):
    m = re.search(rf"^#SBATCH --{key}=(\S+)", (TRAINING / name).read_text(), re.M)
    return m.group(1) if m else None


def test_training_jobs_ask_for_two_cores_and_128g():
    for name in ("train32b.sbatch", "train_dpo.sbatch"):
        assert ask(name, "cpus-per-task") == "2", name
        assert ask(name, "mem") == "128G", name


def test_serving_jobs_ask_for_four_cores_and_128g():
    for name in ("serve.sbatch", "serve_and_score.sbatch"):
        assert ask(name, "cpus-per-task") == "4", name
        assert ask(name, "mem") == "128G", name


def test_the_practice_run_asks_for_two_cores():
    assert ask("smoke.sbatch", "cpus-per-task") == "2"


def test_repair_scoring_says_to_ask_for_more_on_the_command_line():
    text = (TRAINING / "serve_and_score.sbatch").read_text()
    assert "--cpus-per-task=8 --mem=256G" in text
