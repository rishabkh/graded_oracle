"""Baking a trained adapter into its base model, as a step of its own.

On 1 Oct 2026 the merged save at the end of training ran into the 100 GB
home quota (the first merged model already sat there at about 65 GB),
so the run was stopped once the adapter, the actual training result,
had been written. Merging has to be possible on its own, and must not
be able to repeat that mistake. These checks run before any large load,
and on a machine without the training libraries."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

import merge_adapter as M                                      # noqa: E402


def adapter_dir(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    return d


def test_refuses_to_write_the_merged_model_into_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(SystemExit) as e:
        M.check_paths(adapter_dir(tmp_path), home / "graded_oracle/runs/v2/merged")
    assert "home" in str(e.value)


def test_accepts_an_output_outside_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    M.check_paths(adapter_dir(tmp_path), tmp_path / "scratch/runs/v2/merged")


def test_refuses_a_folder_that_is_not_an_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(SystemExit) as e:
        M.check_paths(empty, tmp_path / "scratch")
    assert "adapter_config.json" in str(e.value)


def test_the_checks_load_without_the_training_libraries():
    """This laptop has no torch; the heavy imports live inside merge()."""
    assert callable(M.merge) and callable(M.check_paths)


JOB = Path(__file__).resolve().parent.parent / "training" / "merge.sbatch"


def test_the_merge_job_asks_for_no_gpu():
    """Merging is arithmetic. A GPU request would put it back in the
    queue that took 20 hours to clear on 1 Oct 2026."""
    text = JOB.read_text()
    assert "--gres" not in text and "gpu" not in text.split("#SBATCH --partition=")[1].split()[0]


def test_the_merge_job_will_not_start_without_its_two_paths():
    text = JOB.read_text()
    assert "${ADAPTER:?" in text and "${OUT:?" in text
    assert "merge_adapter.py" in text
