"""Run all package-based examples with automatic output-path handoff."""
from _workflow_state import print_outputs
from step00_phase_pair_curve import run as run_phase_pair_curve
from step01_select_events import run as run_event_selection
from step02_select_stations import run as run_station_selection
from step03_download_coda import run as run_download
from step04_correlate_coda import run as run_correlation
from step05_plot_results import run as run_plotting
from settings import validate_consistency


def run():
    validate_consistency()
    return {
        "phase_pair": run_phase_pair_curve(),
        "events": run_event_selection(),
        "stations": run_station_selection(),
        "download": run_download(),
        "correlation": run_correlation(),
        "plotting": run_plotting(),
    }


def main():
    print_outputs(run())


if __name__ == "__main__":
    main()
