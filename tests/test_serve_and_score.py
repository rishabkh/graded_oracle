"""One cluster job that serves a model, scores it, and stops the server.

Scoring used to need the laptop: a tunnel, a terminal kept open, and a
server left running afterwards (26 idle GPU-hours flagged on 1 Oct 2026).
The combined job runs the passes where the model is served and stops the
server the moment they end. Its scores only compare with the old ones if
the server starts exactly as serve.sbatch started it, and the judge is
the EBMC proved identical to the laptop's, at the old 300 s limit."""
import re
from pathlib import Path

TRAINING = Path(__file__).resolve().parent.parent / "training"
SERVE = (TRAINING / "serve.sbatch").read_text()
SCORE = (TRAINING / "serve_and_score.sbatch").read_text()
BUILD = (TRAINING / "build_ebmc.sbatch").read_text()


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


def test_every_pass_covers_all_files_of_both_sets():
    assert "--set hard --n 31" in SCORE
    assert "--set main_experiment --n 78" in SCORE


def test_the_server_is_stopped_however_the_job_ends():
    assert re.search(r"trap .*kill .*EXIT", SCORE)


def test_the_build_pins_the_laptop_commit_and_proves_the_judge():
    assert "0308d417" in BUILD
    assert "ebmc_calibrate.py" in BUILD and "EBMC_TIMEOUT_S=300" in BUILD
    assert sbatch(BUILD, "gres") is None
