"""Edit these settings before running the package-based examples.

Paths are resolved relative to the directory from which the scripts are run.
The defaults are deliberately capped for a small first run. Remove the caps
only after inspecting the station and waveform audit files.
"""
from pathlib import Path


RUN_ROOT = Path("example_run")
STATE_FILE = RUN_ROOT / "workflow_state.json"


PHASE_PAIR_SETTINGS = {
    "phase_a": "PKIKP",
    "phase_b": "PKJKP",
    "model": "prem",
    "depth_km": 0.0,
    "distance_min_deg": 90.0,
    "distance_max_deg": 180.0,
    "distance_step_deg": 1.0,
    "output_file": RUN_ROOT / "travel_times" / "PKJKP_minus_PKIKP.txt",
}


# Replace this path with your local NDK or QuakeML catalogue. Set
# LOCAL_CATALOGUE to None to use the online monthly GCMT catalogue instead.
EVENT_SETTINGS = {
    "LOCAL_CATALOGUE": Path("data/source.ndk"),
    "LOCAL_FORMAT": "NDK",
    "START_TIME": "2010-01-01T00:00:00",
    "END_TIME": "2011-01-01T00:00:00",
    "LAT_MIN": -5.0,
    "LAT_MAX": 5.0,
    "LON_MIN": 120.0,
    "LON_MAX": 140.0,
    "MAG_MIN": 7.0,
    "MAG_MAX": 8.0,
    "MECHANISMS": {"normal", "reverse"},
    "REQUIRE_PLOTTABLE_STF": True,
    "CACHE_DIR": RUN_ROOT / "cache" / "gcmt",
    "CATALOGUE_DIR": RUN_ROOT / "catalogue",
    "FIG_DIR": RUN_ROOT / "fig" / "events",
    "PREVIEW_COUNT": 0,
}


STATION_SETTINGS = {
    "WINDOW_START_OFFSET_S": 3.0 * 3600.0,
    "WINDOW_END_OFFSET_S": 10.0 * 3600.0,
    "AZIMUTH_START_DEG": 0.0,
    "AZIMUTH_STOP_DEG": 180.0,
    "AZIMUTH_INCREMENT_DEG": 10.0,
    "RADIAL_INCREMENT_DEG": 20.0,
    # Start with two global networks. Use "*" for a full archive search.
    "NETWORK": "IU,II",
    "CHANNELS": ("BHZ", "HHZ"),
    "MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN": 5,
    "CACHE_DIR": RUN_ROOT / "cache" / "stations",
    "OUTPUT_DIR": RUN_ROOT / "stations",
    "FIG_DIR": RUN_ROOT / "fig" / "stations",
    "PREVIEW_EVENT_COUNT": 0,
    "SAVE_PDF": False,
}


DOWNLOAD_SETTINGS = {
    "START_HOURS": 3.0,
    "END_HOURS": 10.0,
    "AZIMUTH_START_DEG": 0.0,
    "AZIMUTH_STOP_DEG": 180.0,
    "AZIMUTH_BIN_WIDTH_DEG": 10.0,
    # These caps make the first run manageable. Set them to None for all data.
    "MAX_EVENTS": 1,
    "MAX_STATIONS_PER_EVENT": 50,
    "WAVEFORM_DIR": RUN_ROOT / "waveforms",
    "CACHE_DIR": RUN_ROOT / "cache" / "waveforms",
    "FIG_DIR": RUN_ROOT / "fig" / "station_pairs",
}


CORRELATION_SETTINGS = {
    "START_HOURS": 3.0,
    "END_HOURS": 10.0,
    "AZIMUTH_START_DEG": 0.0,
    "AZIMUTH_STOP_DEG": 180.0,
    "AZIMUTH_BIN_WIDTH_DEG": 10.0,
    "PAIR_ANGULAR_OFFSET_RANGE_DEG": (-15.0, 15.0),
    "STACK_DISTANCE_RANGE_DEG": (0.0, 180.0),
    "STACK_DISTANCE_BIN_WIDTH_DEG": 3.0,
    "TARGET_SAMPLING_RATE_HZ": 20.0,
    "FINAL_CORRELATION_PERIOD_MIN_S": 15.0,
    "FINAL_CORRELATION_PERIOD_MAX_S": 50.0,
    "MAX_EVENTS": 1,
    "N_PREPROCESS_WORKERS": 4,
    "N_STACK_WORKERS": 4,
    "CORRELATION_DIR": RUN_ROOT / "coda_processed",
}


PLOT_SETTINGS = {
    "DISPLAY_DISTANCE_RANGE_DEG": (0.0, 180.0),
    # Include both signed-lag branches.
    "DISPLAY_LAG_RANGE_SEC": (-3600.0, 3600.0),
    "FIG_DIR": RUN_ROOT / "fig" / "correlations",
    "SAVE_PNG": True,
    "SAVE_PDF": False,
}


def validate_consistency():
    """Fail before network work when linked stage geometry/settings disagree."""
    station_window_hours = (
        STATION_SETTINGS["WINDOW_START_OFFSET_S"] / 3600.0,
        STATION_SETTINGS["WINDOW_END_OFFSET_S"] / 3600.0,
    )
    download_window_hours = (
        DOWNLOAD_SETTINGS["START_HOURS"],
        DOWNLOAD_SETTINGS["END_HOURS"],
    )
    correlation_window_hours = (
        CORRELATION_SETTINGS["START_HOURS"],
        CORRELATION_SETTINGS["END_HOURS"],
    )
    if not station_window_hours == download_window_hours == correlation_window_hours:
        raise ValueError(
            "Station, download, and correlation time windows must match."
        )

    station_azimuth = (
        STATION_SETTINGS["AZIMUTH_START_DEG"],
        STATION_SETTINGS["AZIMUTH_STOP_DEG"],
        STATION_SETTINGS["AZIMUTH_INCREMENT_DEG"],
    )
    download_azimuth = (
        DOWNLOAD_SETTINGS["AZIMUTH_START_DEG"],
        DOWNLOAD_SETTINGS["AZIMUTH_STOP_DEG"],
        DOWNLOAD_SETTINGS["AZIMUTH_BIN_WIDTH_DEG"],
    )
    correlation_azimuth = (
        CORRELATION_SETTINGS["AZIMUTH_START_DEG"],
        CORRELATION_SETTINGS["AZIMUTH_STOP_DEG"],
        CORRELATION_SETTINGS["AZIMUTH_BIN_WIDTH_DEG"],
    )
    if not station_azimuth == download_azimuth == correlation_azimuth:
        raise ValueError(
            "Station, download, and correlation azimuth bins must match."
        )
