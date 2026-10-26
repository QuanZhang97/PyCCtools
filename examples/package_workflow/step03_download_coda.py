"""Download the selected coda windows and write ordered-pair tables."""
from pycctools import download

from _workflow_state import load_state, print_outputs, update_state
from settings import DOWNLOAD_SETTINGS, validate_consistency


def run():
    validate_consistency()
    state = load_state("catalogue_file", "station_run_dir")
    settings = dict(DOWNLOAD_SETTINGS)
    settings.update(
        CATALOGUE_FILE=state["catalogue_file"],
        STATION_RUN_DIR=state["station_run_dir"],
    )
    result = download.run(**settings)
    update_state(waveform_run_dir=result["run_dir"])
    return result


def main():
    print_outputs(run())


if __name__ == "__main__":
    main()
