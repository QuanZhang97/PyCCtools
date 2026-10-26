import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "package_workflow"
)


@pytest.fixture
def example_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    names = [
        "settings",
        "_workflow_state",
        "step00_phase_pair_curve",
        "step01_select_events",
        "step02_select_stations",
        "step03_download_coda",
        "step04_correlate_coda",
        "step05_plot_results",
        "run_all",
    ]
    for name in names:
        sys.modules.pop(name, None)
    modules = {name: importlib.import_module(name) for name in names}
    yield modules
    for name in names:
        sys.modules.pop(name, None)


def test_package_examples_compile():
    for path in EXAMPLE_DIR.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_example_settings_are_consistent(example_modules):
    example_modules["settings"].validate_consistency()


def test_example_state_reports_missing_prerequisite(
    example_modules, monkeypatch, tmp_path
):
    state_module = example_modules["_workflow_state"]
    monkeypatch.setattr(state_module, "STATE_FILE", tmp_path / "state.json")
    with pytest.raises(RuntimeError, match="catalogue_file"):
        state_module.load_state("catalogue_file")


def test_phase_pair_example_runs_offline(example_modules, monkeypatch, tmp_path):
    state_module = example_modules["_workflow_state"]
    phase_module = example_modules["step00_phase_pair_curve"]
    state_file = tmp_path / "state.json"
    output_file = tmp_path / "S_minus_P.txt"
    monkeypatch.setattr(state_module, "STATE_FILE", state_file)
    monkeypatch.setattr(
        phase_module,
        "PHASE_PAIR_SETTINGS",
        {
            "phase_a": "P",
            "phase_b": "S",
            "model": "prem",
            "depth_km": 10.0,
            "distance_min_deg": 30.0,
            "distance_max_deg": 32.0,
            "distance_step_deg": 1.0,
            "output_file": output_file,
        },
    )

    result = phase_module.run()

    assert result["valid_points"] == 3
    values = np.loadtxt(output_file)
    assert values.shape == (3, 2)
    assert np.all(values[:, 1] > 0)
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["theoretical_arrival_file"] == str(output_file.resolve())
