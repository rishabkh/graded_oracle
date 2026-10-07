"""One cluster job that serves a model, scores it, and stops the server.

Scoring used to need the laptop: a tunnel, a terminal kept open, and a
server left running afterwards (26 idle GPU-hours flagged on 1 Oct 2026).
The combined job runs the passes where the model is served and stops the
server the moment they end. Its scores only compare with the old ones if
the server starts exactly as serve.sbatch started it, and the judge is
the EBMC proved identical to the laptop's, at the old 300 s limit."""
import re
import subprocess
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
SERVE = (TRAINING / "serve.sbatch").read_text()
SCORE = (TRAINING / "serve_and_score.sbatch").read_text()
BUILD = (TRAINING / "build_ebmc.sbatch").read_text()


# The job run for real, with every outside program (module, vllm, curl,
# nvidia-smi, the scorer itself) replaced by a stub; returns the scorer
# commands it would have run. A wrong loop costs GPU hours on the cluster.
STUBS = {"module": "", "nvidia-smi": "", "nvcc": "", "curl": "",
         "vllm": "sleep 30",
         "python": 'echo "$@" >> "$CALLS"'}


def run_job(tmp_path, **env):
    home = tmp_path / "home"
    (home / "graded_oracle").mkdir(parents=True)
    (home / "envs" / "vllm" / "bin").mkdir(parents=True)
    (home / "envs" / "vllm" / "bin" / "activate").write_text("")
    (home / "bin").mkdir()
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for path, body in [(stubs / n, b) for n, b in STUBS.items()] + \
            [(home / "bin" / "ebmc", "")]:
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    r = subprocess.run(
        ["bash", str(TRAINING / "serve_and_score.sbatch")], capture_output=True,
        text=True, timeout=60,
        env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(home),
             "CALLS": str(calls), "USER": "tester", "SLURM_JOB_ID": "1",
             "SLURM_CPUS_PER_TASK": "8", **env})
    return r, [c.removeprefix("initiator/benchmark_solve.py ")
               for c in calls.read_text().splitlines()]


def joined(text):
    return text.replace("\\\n", " ")


def vllm_args(text):
    """Everything after `vllm serve`, up to any redirection or `&`."""
    line = next(l for l in joined(text).splitlines() if "vllm serve" in l)
    out = []
    for tok in line.split("vllm serve", 1)[1].split():
        if tok in ("&", "2>&1") or tok.startswith(">"):
            break
        out.append(tok)
    return out


def sbatch(text, key):
    m = re.search(rf"^#SBATCH --{key}=(\S+)", text, re.M)
    return m.group(1) if m else None


def exports(text):
    return {l.strip() for l in text.splitlines()
            if l.strip().startswith("export ")}


def test_the_server_starts_with_exactly_the_settings_of_serve_sbatch():
    assert vllm_args(SCORE) == vllm_args(SERVE)


def test_the_server_runs_on_the_same_gpus_and_environment():
    assert sbatch(SCORE, "gres") == sbatch(SERVE, "gres")
    assert sbatch(SCORE, "partition") == sbatch(SERVE, "partition")
    assert exports(SERVE) <= exports(SCORE)


def test_answers_are_judged_like_the_earlier_scores():
    assert "EBMC_TIMEOUT_S=300" in SCORE
    assert '$HOME/bin/ebmc' in SCORE


def test_every_pass_covers_all_files_of_both_sets(tmp_path):
    r, calls = run_job(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["--solver qwen --set hard --n 31",
                     "--solver qwen --set main_experiment --n 78"] * 5


def test_the_server_is_stopped_however_the_job_ends():
    assert re.search(r"trap .*kill .*EXIT", SCORE)


def test_the_build_pins_the_laptop_commit_and_proves_the_judge():
    assert "0308d417" in BUILD
    assert "ebmc_calibrate.py" in BUILD and "EBMC_TIMEOUT_S=300" in BUILD
    assert sbatch(BUILD, "gres") is None


def gcc_module(text):
    m = re.search(r"^module load .*?\b(gcc\S*)", text, re.M)
    return m.group(1) if m else None


def test_the_judge_is_built_and_run_with_one_named_compiler():
    """1 Oct 2026: an unversioned `module load gcc` gave GCC 16.2, whose
    stricter new checks broke this CBMC version's warnings-are-errors
    build. Name the version, and run the binary with the same one, since
    it needs the C++ library of the compiler that built it."""
    assert gcc_module(BUILD) == gcc_module(SCORE) == "gcc/13.2.0-fasrc01"


def test_repair_is_off_unless_asked_for(tmp_path):
    """ROUNDS and SHOW_CEX unset must give exactly the one-shot command
    the earlier scores were made with. Checked by running the job, since
    7 Oct 2026, not by reading its text."""
    assert 'ROUNDS="${ROUNDS:-0}"' in SCORE
    assert 'SHOW_CEX="${SHOW_CEX:-0}"' in SCORE
    r, calls = run_job(tmp_path, PASSES="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["--solver qwen --set hard --n 31",
                     "--solver qwen --set main_experiment --n 78"]


def test_a_repair_run_gives_the_command_it_always_has(tmp_path):
    r, calls = run_job(tmp_path, PASSES="1", ROUNDS="5", SHOW_CEX="1")
    assert r.returncode == 0, r.stdout + r.stderr
    repair = "--rounds 5 --ebmc-workers 6 --show-cex"
    assert calls == [f"--solver qwen --set hard --n 31 {repair}",
                     f"--solver qwen --set main_experiment --n 78 {repair}"]


def test_both_versions_on_the_hard_set_in_one_job(tmp_path):
    """The repair loop with and without its feedback (7 Oct 2026): one
    job, so both versions run with the same code, server and day."""
    r, calls = run_job(tmp_path, PASSES="1", ROUNDS="5", SHOW_CEX="1",
                       FEEDBACK="both", SETS="hard")
    assert r.returncode == 0, r.stdout + r.stderr
    same = "--solver qwen --set hard --n 31 --rounds 5 --ebmc-workers 6"
    # the control shows no counterexamples: it has no feedback to put them in
    assert calls == [f"{same} --show-cex", f"{same} --no-feedback"]


def test_the_control_alone(tmp_path):
    r, calls = run_job(tmp_path, PASSES="2", ROUNDS="5", FEEDBACK="off",
                       SETS="hard")
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["--solver qwen --set hard --n 31 --rounds 5 "
                     "--ebmc-workers 6 --no-feedback"] * 2


def test_bad_settings_stop_before_the_server_starts(tmp_path):
    for i, env in enumerate(({"FEEDBACK": "off"},     # no repair loop to control
                             {"FEEDBACK": "maybe", "ROUNDS": "5"},
                             {"SETS": "hardd", "ROUNDS": "5"})):
        r, calls = run_job(tmp_path / str(i), **env)
        assert r.returncode != 0 and calls == [], env
        assert "serving" not in r.stdout, env


def test_the_job_says_what_it_is_running():
    assert "ROUNDS=$ROUNDS SHOW_CEX=$SHOW_CEX FEEDBACK=$FEEDBACK" in SCORE


def test_repair_checks_use_the_cores_the_job_was_given():
    """6 Oct 2026: 4 checks at once left most of the allocation idle while
    EBMC ran; use all but two cores (the server keeps two)."""
    assert 'REPAIR="--rounds $ROUNDS --ebmc-workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))"' in SCORE


def test_every_scoring_log_names_the_code_it_ran(tmp_path):
    """v4b's scoring changed grader halfway (a git pull during the job
    brought in the enum fix before its last pass); every log now says
    which commit it ran."""
    r, calls = run_job(tmp_path, PASSES="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert any(l.startswith("code ") for l in r.stdout.splitlines())
