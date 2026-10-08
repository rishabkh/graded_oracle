"""The training job, made safe before the catalog run (8 Oct 2026): it used
to fall back to v1's training file and an output folder under home, and
its MODEL default could be overridden by a leftover shell variable, so a
run could quietly train on top of an old model or fill the home quota
(the 30 Sep incident). Model, training file and output folder are now
required, and the output must be an absolute path outside home."""
import subprocess
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
JOB = (TRAINING / "train32b.sbatch").read_text()
DPO = (TRAINING / "train_dpo.sbatch").read_text()


def test_model_pairs_and_output_must_be_named():
    for var in ("MODEL", "PAIRS", "OUT"):
        assert f'{var}="${{{var}:?' in JOB, var
    for old in ("${MODEL:-", "${PAIRS:-", "${OUT:-"):
        assert old not in JOB, old                   # no silent fallback


def test_the_guard_starts_with_the_preference_jobs_checks():
    cut = lambda t: t.split("# --- guard ---")[1].split("# --- end guard ---")[0]
    assert cut(JOB).startswith(cut(DPO).rstrip("\n"))


def test_it_refuses_to_write_into_home_or_a_relative_path():
    guard = JOB.split("# --- guard ---")[1].split("# --- end guard ---")[0]
    for out, ok in (("/n/netscratch/x/runs/v5", True),
                    ("runs/v5", False), ("$HOME/runs/v5", False)):
        r = subprocess.run(["bash", "-c", 'HOME=/home/me; MODEL="Qwen/Qwen2.5-'
                            f'Coder-32B-Instruct"; OUT="{out}"\n{guard}'],
                           capture_output=True, text=True)
        assert (r.returncode == 0) is ok, (out, r.stdout, r.stderr)


def test_training_and_the_adapter_check_are_unchanged():
    assert "--epochs 3" in JOB and "--merge" in JOB
    assert "python training/check_adapter.py" in JOB


def _guard_run(env):
    guard = JOB.split("# --- guard ---")[1].split("# --- end guard ---")[0]
    sets = "; ".join(f'{k}="{v}"' for k, v in env.items())
    return subprocess.run(["bash", "-c", f"HOME=/home/me; {sets}\n{guard}"],
                          capture_output=True, text=True)


def test_a_finished_model_is_never_overwritten(tmp_path):
    """A slip like OUT=.../runs/v2 would overwrite the model the new one
    is compared with. A folder left by a preempted run (no merged model)
    is fine, so a requeued job can carry on."""
    done = tmp_path / "v2"
    (done / "merged").mkdir(parents=True)
    partial = tmp_path / "v5"
    partial.mkdir()
    base = "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert _guard_run({"OUT": done, "MODEL": base}).returncode != 0
    assert _guard_run({"OUT": partial, "MODEL": base}).returncode == 0


def test_training_starts_from_the_untrained_base_only(tmp_path):
    out = tmp_path / "v5"
    assert _guard_run({"OUT": out, "MODEL": "/n/netscratch/x/runs/v2/merged"}
                      ).returncode != 0
    assert _guard_run({"OUT": out, "MODEL": "Qwen/Qwen2.5-Coder-32B-Instruct"}
                      ).returncode == 0


def test_the_job_passes_the_raised_length_limit():
    assert '--max-len "${MAXLEN:-16384}"' in JOB
