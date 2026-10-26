"""Compute directed pair correlations and distance-bin stacks."""
from pycctools import correlation

from _workflow_state import load_state, print_outputs, update_state
from settings import CORRELATION_SETTINGS, validate_consistency


def run():
    validate_consistency()
    state = load_state(
        "catalogue_file",
        "station_run_dir",
        "waveform_run_dir",
    )
    settings = dict(CORRELATION_SETTINGS)
    settings.update(
        CATALOGUE_FILE=state["catalogue_file"],
        STATION_RUN_DIR=state["station_run_dir"],
        WAVEFORM_RUN_DIR=state["waveform_run_dir"],
    )
    result = correlation.run(**settings)
    update_state(correlation_dir=result["correlation_dir"])
    return result


def main():
    print_outputs(run())


if __name__ == "__main__":
    main()
