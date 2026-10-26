"""Plot stacked correlograms and wiggles with the phase-pair curve."""
from pycctools import plotting

from _workflow_state import load_state, print_outputs, update_state
from settings import PLOT_SETTINGS


def run():
    state = load_state("correlation_dir", "theoretical_arrival_file")
    settings = dict(PLOT_SETTINGS)
    settings.update(
        CORRELATION_DIR=state["correlation_dir"],
        THEORETICAL_ARRIVAL_FILE=state["theoretical_arrival_file"],
    )
    result = plotting.run(**settings)
    update_state(figure_dir=result["figure_dir"])
    return result


def main():
    print_outputs(run())


if __name__ == "__main__":
    main()
