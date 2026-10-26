"""Select station metadata around each event epicenter."""
from pycctools import stations

from _workflow_state import load_state, print_outputs, update_state
from settings import STATION_SETTINGS, validate_consistency


def run():
    validate_consistency()
    state = load_state("catalogue_file")
    settings = dict(STATION_SETTINGS)
    settings["CATALOGUE_FILE"] = state["catalogue_file"]
    result = stations.run(**settings)
    update_state(station_run_dir=result["run_dir"])
    return result


def main():
    print_outputs(run())


if __name__ == "__main__":
    main()
