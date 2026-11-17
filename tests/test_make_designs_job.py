"""The cluster job that makes and checks designs (16 Nov 2026, Rishab:
"alright let's do it", change 7 of 7).

150 growth steps took 4.5 hours of the laptop on 15 Nov; the 5,000 of
the big run would take about six days of it, lid open. The job runs any
step of making designs (the initiator, the corpus build, the labels, the
extender, the checker replay) on a whole 48-core machine of the shared
partition, with the tool bundle and the model switch set up:
  - one node, one task, 48 cores, and it says how many checks to run at
    once (every core but 4), so no core sits idle and none is fought over;
  - the key comes from the shell that calls sbatch, never from a file,
    and a job without it stops at once;
  - no web proxy: FASRC's own page says the proxy is for FASSE and must
    not be set on Cannon;
  - the word CHECK_SLOTS in the options stands for the number of checks
    at once, so a command needs no number that depends on the machine."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOB = ROOT / "training" / "make_designs.sbatch"
TEXT = JOB.read_text() if JOB.exists() else ""
# what the shell runs, without the notes
CODE = "\n".join(l for l in TEXT.splitlines() if not l.lstrip().startswith("#"))


def ask(key):
    m = re.search(rf"^#SBATCH --{key}=(\S+)", TEXT, re.M)
    return m.group(1) if m else None


def test_it_asks_for_one_whole_48_core_machine_of_the_shared_partition():
    assert ask("partition") == "shared"
    assert ask("account") == "amin_lab"
    assert ask("nodes") == "1" and ask("ntasks") == "1"
    assert ask("cpus-per-task") == "48"
    assert ask("mem") == "128G"
    assert ask("time") == "48:00:00"
    assert ask("output") == "runs/make_designs_%j.out"


def test_the_shell_reads_it_without_error():
    done = subprocess.run(["bash", "-n", str(JOB)], capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr


def test_the_key_comes_from_the_calling_shell_and_never_from_a_file():
    assert '"${OPENAI_API_KEY:?' in CODE
    assert not re.search(r"OPENAI_API_KEY\s*=", TEXT)
    assert not re.search(r"sk-[A-Za-z0-9_-]{8,}", TEXT)   # nothing key-shaped


def test_it_runs_gpt_through_the_same_switch_as_the_laptop():
    assert "export LLM_PROVIDER=openai" in CODE
    assert "unset CLAUDE_PROVIDER" in CODE


def test_it_sets_no_web_proxy():
    assert "proxy" not in CODE.lower()
    assert "must not be set on Cannon" in TEXT          # and says why


def test_it_loads_the_tool_bundle_and_stops_without_one():
    assert 'TOOLS="${OSS_CAD_SUITE:-$HOME/oss-cad-suite}"' in CODE
    assert 'source "$TOOLS/environment"' in CODE
    assert CODE.index('[ -f "$TOOLS/environment" ]') < \
        CODE.index('source "$TOOLS/environment"')


def test_it_uses_the_environments_own_python_not_the_bundles():
    # the tool bundle puts its own python3 first on the path, and that one
    # has no openai package (10 Nov 2026: 20 steps failed that way)
    assert 'PY="$HOME/envs/vllm/bin/python"' in CODE
    assert '"$PY" "${args[@]}"' in CODE
    assert not re.search(r"^\s*python3?\s", CODE, re.M)


def test_it_says_how_many_checks_run_at_once():
    assert "CHECK_SLOTS=$(( ${SLURM_CPUS_PER_TASK:-8} - 4 ))" in CODE
    assert 'args=("${@//CHECK_SLOTS/$CHECK_SLOTS}")' in CODE


def test_it_stops_when_given_nothing_to_run():
    assert '[ "$#" -gt 0 ]' in CODE


def test_it_names_the_code_it_ran():
    assert "git rev-parse --short HEAD" in CODE


def run_job(tmp_path, *args, key="k"):
    """The job's own shell lines with a stand-in home: a fake tool bundle,
    a fake python that prints its arguments, and no cluster commands."""
    home = tmp_path / "home"
    (home / "oss-cad-suite").mkdir(parents=True)
    (home / "oss-cad-suite" / "environment").write_text("")
    (home / "graded_oracle").mkdir()
    py = home / "envs" / "vllm" / "bin"
    py.mkdir(parents=True)
    (py / "python").write_text('#!/bin/bash\necho "ran: $@"\n')
    (py / "python").chmod(0o755)
    script = tmp_path / "job.sh"
    script.write_text("module() { :; }\n" + TEXT)
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin",
           "SLURM_CPUS_PER_TASK": "48"}
    if key:
        env["OPENAI_API_KEY"] = key
    return subprocess.run(["bash", str(script), *args], capture_output=True,
                          text=True, env=env)


def test_a_command_is_run_with_the_number_of_checks_filled_in(tmp_path):
    done = run_job(tmp_path, "extender/batch.py", "--workers", "CHECK_SLOTS",
                   "--max-calls", "5000")
    assert done.returncode == 0, done.stderr
    assert "checks at once: 44" in done.stdout
    assert "ran: extender/batch.py --workers 44 --max-calls 5000" in \
        done.stdout


def test_a_job_without_the_key_stops_before_running_anything(tmp_path):
    done = run_job(tmp_path, "extender/batch.py", key=None)
    assert done.returncode != 0
    assert "ran:" not in done.stdout
    assert "export OPENAI_API_KEY" in done.stderr


def test_a_job_with_nothing_to_run_says_so(tmp_path):
    done = run_job(tmp_path)
    assert done.returncode != 0
    assert "nothing to run" in done.stdout + done.stderr
