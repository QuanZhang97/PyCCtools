"""Plotting stage, adapted from user-supplied 05_plot_coda_correlation_and_wiggles.py.

No processing occurs at import. Settings use the original uppercase names.
Each run has independent settings, helpers and FFT worker state.
"""
from ._config import setting, validate_settings

SETTINGS = ('CORRELATION_DIR', 'FIG_DIR', 'EVENT_FILTER', 'CORRELATION_FILES', 'DISPLAY_DISTANCE_RANGE_DEG', 'DISPLAY_LAG_RANGE_SEC', 'CORRELOGRAM_BIN_WIDTH_DEG', 'HISTOGRAM_BIN_WIDTH_DEG', 'CORRELOGRAM_GRID_DISTANCE_DEG', 'CORRELOGRAM_MAX_LAG_TICKS', 'CORRELATION_DISPLAY_VMAX', 'CORRELOGRAM_ALPHA', 'COLORMAP_NAME', 'WIGGLE_AMPLITUDE_DEG', 'WIGGLE_ALPHA', 'WIGGLE_LINE_WIDTH', 'WIGGLE_MAX_PLOT_SAMPLES', 'THEORETICAL_ARRIVAL_FILE', 'THEORETICAL_LINE_WIDTH', 'THEORETICAL_LINE_ALPHA', 'FIGURE_SIZE', 'PLOT_DPI', 'SAVE_PNG', 'SAVE_PDF', 'COLLECT_GARBAGE_AFTER_FILE', 'STACKED_FILENAME_RE')

def run(**overrides):
    """Execute this stage and return its output paths. See SETTINGS for options."""
    validate_settings(overrides, SETTINGS)
    #!/usr/bin/env python3
    # -*- coding: utf-8 -*-
    """Plot the distance-binned, frequency-domain stacks written by script 04.

    Script 04 writes one compact ``*_stacked.npz`` file for each event and
    azimuth group.  Each file already contains one averaged correlation per
    distance bin, so this script opens one file, makes its figures, closes it,
    and only then moves to the next file.  It never reconstructs or concatenates
    station-pair correlations in memory.

    The wiggle traces are therefore stacked-distance-bin traces.  Individual
    station-pair wiggles are intentionally not available because script 04 no
    longer saves every pair's time-domain correlation.
    """

    import gc
    import json
    import re
    from pathlib import Path
    import warnings

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator


    # ---------------------------------------------------------------------------
    # Input/output and plotting settings
    # ---------------------------------------------------------------------------
    CORRELATION_DIR = Path("./coda_processed")
    CORRELATION_DIR = setting(overrides, "CORRELATION_DIR", CORRELATION_DIR)
    FIG_DIR = Path("./fig")
    FIG_DIR = setting(overrides, "FIG_DIR", FIG_DIR)

    # None processes every stacked event/group file below CORRELATION_DIR.  To
    # process a subset, use e.g. EVENT_FILTER = {"event_name_1234"}.
    EVENT_FILTER = None
    EVENT_FILTER = setting(overrides, "EVENT_FILTER", EVENT_FILTER)
    CORRELATION_FILES = None  # Optional explicit list of *_stacked.npz files.
    CORRELATION_FILES = setting(overrides, "CORRELATION_FILES", CORRELATION_FILES)

    # Keep the requested display range.  Script 04 can save another range, but
    # only the part inside this window is plotted here.
    DISPLAY_DISTANCE_RANGE_DEG = (0.0, 180.0)
    DISPLAY_DISTANCE_RANGE_DEG = setting(overrides, "DISPLAY_DISTANCE_RANGE_DEG", DISPLAY_DISTANCE_RANGE_DEG)
    DISPLAY_LAG_RANGE_SEC = (000.0, 3600.0)
    DISPLAY_LAG_RANGE_SEC = setting(overrides, "DISPLAY_LAG_RANGE_SEC", DISPLAY_LAG_RANGE_SEC)

    CORRELOGRAM_BIN_WIDTH_DEG = 1.0
    CORRELOGRAM_BIN_WIDTH_DEG = setting(overrides, "CORRELOGRAM_BIN_WIDTH_DEG", CORRELOGRAM_BIN_WIDTH_DEG)
    HISTOGRAM_BIN_WIDTH_DEG = 1.0
    HISTOGRAM_BIN_WIDTH_DEG = setting(overrides, "HISTOGRAM_BIN_WIDTH_DEG", HISTOGRAM_BIN_WIDTH_DEG)
    CORRELOGRAM_GRID_DISTANCE_DEG = 15.0
    CORRELOGRAM_GRID_DISTANCE_DEG = setting(overrides, "CORRELOGRAM_GRID_DISTANCE_DEG", CORRELOGRAM_GRID_DISTANCE_DEG)
    CORRELOGRAM_MAX_LAG_TICKS = 12
    CORRELOGRAM_MAX_LAG_TICKS = setting(overrides, "CORRELOGRAM_MAX_LAG_TICKS", CORRELOGRAM_MAX_LAG_TICKS)
    CORRELATION_DISPLAY_VMAX = 1.0
    CORRELATION_DISPLAY_VMAX = setting(overrides, "CORRELATION_DISPLAY_VMAX", CORRELATION_DISPLAY_VMAX)
    CORRELOGRAM_ALPHA = 1.0
    CORRELOGRAM_ALPHA = setting(overrides, "CORRELOGRAM_ALPHA", CORRELOGRAM_ALPHA)
    COLORMAP_NAME = "Greys"
    COLORMAP_NAME = setting(overrides, "COLORMAP_NAME", COLORMAP_NAME)

    # Black line, positive side filled, negative side line-only.
    WIGGLE_AMPLITUDE_DEG = 1.60
    WIGGLE_AMPLITUDE_DEG = setting(overrides, "WIGGLE_AMPLITUDE_DEG", WIGGLE_AMPLITUDE_DEG)
    WIGGLE_ALPHA = 0.8
    WIGGLE_ALPHA = setting(overrides, "WIGGLE_ALPHA", WIGGLE_ALPHA)
    WIGGLE_LINE_WIDTH = 0.35
    WIGGLE_LINE_WIDTH = setting(overrides, "WIGGLE_LINE_WIDTH", WIGGLE_LINE_WIDTH)
    WIGGLE_MAX_PLOT_SAMPLES = 4000
    WIGGLE_MAX_PLOT_SAMPLES = setting(overrides, "WIGGLE_MAX_PLOT_SAMPLES", WIGGLE_MAX_PLOT_SAMPLES)

    # The attached theoretical curve has two columns: distance (degrees), time
    # (seconds).  Set this to another file if the curve is stored elsewhere.
    THEORETICAL_ARRIVAL_FILE = None
    THEORETICAL_ARRIVAL_FILE = setting(overrides, "THEORETICAL_ARRIVAL_FILE", THEORETICAL_ARRIVAL_FILE)
    THEORETICAL_LINE_WIDTH = 1.2
    THEORETICAL_LINE_WIDTH = setting(overrides, "THEORETICAL_LINE_WIDTH", THEORETICAL_LINE_WIDTH)
    THEORETICAL_LINE_ALPHA = 1.0
    THEORETICAL_LINE_ALPHA = setting(overrides, "THEORETICAL_LINE_ALPHA", THEORETICAL_LINE_ALPHA)

    FIGURE_SIZE = (7.0, 10.0)
    FIGURE_SIZE = setting(overrides, "FIGURE_SIZE", FIGURE_SIZE)
    PLOT_DPI = 220
    PLOT_DPI = setting(overrides, "PLOT_DPI", PLOT_DPI)
    SAVE_PNG = True
    SAVE_PNG = setting(overrides, "SAVE_PNG", SAVE_PNG)
    SAVE_PDF = True
    SAVE_PDF = setting(overrides, "SAVE_PDF", SAVE_PDF)
    COLLECT_GARBAGE_AFTER_FILE = True
    COLLECT_GARBAGE_AFTER_FILE = setting(overrides, "COLLECT_GARBAGE_AFTER_FILE", COLLECT_GARBAGE_AFTER_FILE)


    STACKED_FILENAME_RE = re.compile(
        r"^az_(?P<lo>[-+0-9.]+)_(?P<hi>[-+0-9.]+)_stacked\.npz$"
    )
    STACKED_FILENAME_RE = setting(overrides, "STACKED_FILENAME_RE", STACKED_FILENAME_RE)


    from .common import regular_edges as regular_edges


    def load_stacked_files():
        """Find only the compact outputs produced by the new script 04."""
        if CORRELATION_FILES is not None:
            files = [Path(path) for path in CORRELATION_FILES]
        else:
            files = sorted(
                CORRELATION_DIR.glob("*/correlations/az_*_stacked.npz")
            )

        if EVENT_FILTER is not None:
            files = [
                path for path in files
                if path.parent.parent.name in EVENT_FILTER
            ]

        if not files:
            raise FileNotFoundError(
                "No *_stacked.npz files found below {}. Run the revised script "
                "04 first.".format(CORRELATION_DIR)
            )
        return files


    def parse_group_filename(npz_path):
        match = STACKED_FILENAME_RE.match(npz_path.name)
        if match is None:
            raise ValueError("Unrecognized stacked filename: {}".format(npz_path))
        return float(match.group("lo")), float(match.group("hi"))


    def read_group_limits(npz_path, data):
        """Read azimuth limits from metadata, checking the filename fallback."""
        if (
            "azimuth_bin_start_deg" in data.files
            and "azimuth_bin_end_deg" in data.files
        ):
            return (
                float(data["azimuth_bin_start_deg"]),
                float(data["azimuth_bin_end_deg"]),
            )
        return parse_group_filename(npz_path)


    def load_theoretical_arrival(path):
        """Load finite ``distance_deg, time_sec`` pairs from a text file."""
        if path is None:
            return None

        path = Path(path)
        if not path.exists():
            warnings.warn(
                "Theoretical arrival file not found; no dashed line will be "
                "drawn: {}".format(path)
            )
            return None

        try:
            values = np.loadtxt(
                path,
                comments="#",
                usecols=(0, 1),
                ndmin=2,
            )
        except Exception as exc:
            warnings.warn(
                "Could not read theoretical arrival file {}; no dashed line "
                "will be drawn: {}".format(path, exc)
            )
            return None

        if values.shape[1] != 2:
            warnings.warn(
                "Theoretical arrival file must have two columns: {}".format(path)
            )
            return None

        values = values[np.isfinite(values[:, 0])]
        values[~np.isfinite(values[:, 1]), 1] = np.nan
        if len(values) == 0:
            warnings.warn(
                "Theoretical arrival file contains no finite points: {}".format(
                    path
                )
            )
            return None

        values = values[np.argsort(values[:, 0])]
        return values[:, 0], values[:, 1]


    def prepare_lag_indices(lag_sec):
        """Return lag indices and an increasing lag axis for the display window."""
        lag_sec = np.asarray(lag_sec, dtype=float)
        if lag_sec.ndim != 1 or not np.all(np.isfinite(lag_sec)):
            raise ValueError("The saved lag axis must be a finite 1-D array.")

        order = np.argsort(lag_sec)
        sorted_lag = lag_sec[order]
        t0, t1 = DISPLAY_LAG_RANGE_SEC
        keep = (sorted_lag >= t0) & (sorted_lag <= t1)
        if not np.any(keep):
            raise ValueError(
                "The requested lag window {} is absent from the saved lag axis."
                .format(DISPLAY_LAG_RANGE_SEC)
            )
        return order[keep], sorted_lag[keep]


    def normalize_stacked_rows(correlation, pair_count):
        """Normalize each saved bin trace for display, preserving empty bins."""
        correlation = np.asarray(correlation, dtype=np.float32)
        pair_count = np.asarray(pair_count, dtype=np.int64)
        correlation = np.array(correlation, dtype=np.float32, copy=True)
        correlation[~np.isfinite(correlation)] = 0.0

        scale = np.max(np.abs(correlation), axis=1)
        invalid = (pair_count <= 0) | (
            scale <= np.finfo(np.float32).eps
        )
        scale[invalid] = 1.0
        correlation /= scale[:, None]
        correlation[pair_count <= 0, :] = np.nan
        return correlation


    def distance_ticks(d0, d1):
        first = np.ceil(d0 / CORRELOGRAM_GRID_DISTANCE_DEG) \
            * CORRELOGRAM_GRID_DISTANCE_DEG
        return np.arange(first, d1 + 1e-9, CORRELOGRAM_GRID_DISTANCE_DEG)


    def save_figure(fig, stem):
        if SAVE_PNG:
            fig.savefig(stem.with_suffix(".png"), dpi=PLOT_DPI, bbox_inches="tight")
        if SAVE_PDF:
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


    def add_theoretical_line(axis, theoretical_arrival):
        if theoretical_arrival is None:
            return
        distance_deg, arrival_sec = theoretical_arrival
        axis.plot(
            distance_deg,
            arrival_sec,
            color="black",
            linestyle="--",
            linewidth=THEORETICAL_LINE_WIDTH,
            alpha=THEORETICAL_LINE_ALPHA,
            zorder=20,
        )


    def read_stacked_file(npz_path):
        """Read one stacked file and close its zip handle before plotting."""
        with np.load(str(npz_path), allow_pickle=False) as data:
            required = {
                "correlation",
                "distance_deg",
                "pair_count",
                "lag_sec",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(
                    "{} is missing required arrays: {}".format(
                        npz_path, ", ".join(missing)
                    )
                )

            correlation = np.asarray(data["correlation"], dtype=np.float32)
            distance_deg = np.asarray(data["distance_deg"], dtype=float)
            pair_count = np.asarray(data["pair_count"], dtype=np.int64)
            lag_sec = np.asarray(data["lag_sec"], dtype=float)
            if "distance_bin_edges_deg" in data.files:
                distance_edges = np.asarray(
                    data["distance_bin_edges_deg"], dtype=float
                )
            else:
                distance_edges = regular_edges(
                    float(np.min(distance_deg - CORRELOGRAM_BIN_WIDTH_DEG / 2.0)),
                    float(np.max(distance_deg + CORRELOGRAM_BIN_WIDTH_DEG / 2.0)),
                    CORRELOGRAM_BIN_WIDTH_DEG,
                )
            azimuth_start, azimuth_end = read_group_limits(npz_path, data)

        if correlation.ndim != 2:
            raise ValueError("correlation must be a 2-D bin-by-lag array.")
        if correlation.shape != (len(distance_deg), len(lag_sec)):
            raise ValueError(
                "Array dimensions disagree in {}: correlation={}, distance={}, "
                "lag={}".format(
                    npz_path,
                    correlation.shape,
                    len(distance_deg),
                    len(lag_sec),
                )
            )
        if len(pair_count) != len(distance_deg):
            raise ValueError(
                "pair_count does not match distance_deg in {}".format(npz_path)
            )
        if len(distance_edges) != len(distance_deg) + 1:
            raise ValueError(
                "distance_bin_edges_deg must have one more element than "
                "distance_deg in {}".format(npz_path)
            )

        return {
            "correlation": correlation,
            "distance_deg": distance_deg,
            "distance_edges": distance_edges,
            "pair_count": pair_count,
            "lag_sec": lag_sec,
            "azimuth_start": azimuth_start,
            "azimuth_end": azimuth_end,
        }


    def select_display_window(data):
        """Select requested distance and lag windows without concatenating files."""
        original_distances = data["distance_deg"]
        distance_order = np.argsort(original_distances)
        distances = original_distances[distance_order]
        correlation = data["correlation"][distance_order]
        pair_count = data["pair_count"][distance_order]
        distance_edges = data["distance_edges"]

        if not np.all(np.diff(distance_edges) > 0):
            raise ValueError("Distance-bin edges are not strictly increasing.")
        if not np.all(np.diff(distances) > 0):
            raise ValueError("Distance-bin centers are not strictly increasing.")
        if not np.all(
            (distance_edges[:-1] < distances)
            & (distances < distance_edges[1:])
        ):
            raise ValueError("Distance-bin metadata are inconsistent.")

        d0, d1 = DISPLAY_DISTANCE_RANGE_DEG
        distance_keep = (distances >= d0) & (distances <= d1)
        if not np.any(distance_keep):
            raise ValueError(
                "The requested distance window {} is absent from the saved "
                "bins.".format(DISPLAY_DISTANCE_RANGE_DEG)
            )
        selected_rows = np.flatnonzero(distance_keep)
        if len(selected_rows) > 1 and not np.all(np.diff(selected_rows) == 1):
            raise ValueError("Selected distance bins are not contiguous.")

        row_start = selected_rows[0]
        row_end = selected_rows[-1] + 1
        distance_edges = distance_edges[row_start:row_end + 1]
        distances = distances[selected_rows]
        correlation = correlation[selected_rows]
        pair_count = pair_count[selected_rows]

        lag_indices, lag_sec = prepare_lag_indices(data["lag_sec"])
        correlation = correlation[:, lag_indices]
        correlation = normalize_stacked_rows(correlation, pair_count)
        return correlation, distances, distance_edges, pair_count, lag_sec


    def create_correlogram(
        event_name,
        azimuth_start,
        azimuth_end,
        correlation,
        distances,
        distance_edges,
        pair_count,
        lag_sec,
        theoretical_arrival,
    ):
        d0, d1 = DISPLAY_DISTANCE_RANGE_DEG
        t0, t1 = DISPLAY_LAG_RANGE_SEC
        cmap = plt.get_cmap(COLORMAP_NAME).with_extremes(bad="white")

        title = (
            "{} | {:g}–{:g}° + {:g}–{:g}° | frequency-domain stacked bins"
            .format(
                event_name,
                azimuth_start,
                azimuth_end,
                azimuth_start + 180.0,
                azimuth_end + 180.0,
            )
        )
        figure = plt.figure(figsize=FIGURE_SIZE)
        grid = figure.add_gridspec(2, 1, height_ratios=(1, 6), hspace=0.06)
        histogram_axis = figure.add_subplot(grid[0])
        image_axis = figure.add_subplot(grid[1], sharex=histogram_axis)

        histogram_centers = 0.5 * (distance_edges[:-1] + distance_edges[1:])
        histogram_axis.bar(
            histogram_centers,
            pair_count,
            width=np.diff(distance_edges) * 0.95,
            color="0.55",
            edgecolor="0.35",
            linewidth=0.2,
            align="center",
        )
        histogram_axis.set_xlim(d0, d1)
        histogram_axis.set_ylim(0, max(1.0, 1.10 * np.max(pair_count)))
        histogram_axis.set_ylabel("Number of\ncorrelation pairs")
        histogram_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        histogram_axis.tick_params(axis="x", labelbottom=False)
        histogram_axis.spines["right"].set_visible(False)
        histogram_axis.spines["top"].set_visible(False)
        histogram_axis.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

        image_axis.imshow(
            correlation.T,
            extent=(distance_edges[0], distance_edges[-1], lag_sec[0], lag_sec[-1]),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=-CORRELATION_DISPLAY_VMAX,
            vmax=CORRELATION_DISPLAY_VMAX,
            alpha=CORRELOGRAM_ALPHA,
        )
        image_axis.set_xlim(d0, d1)
        image_axis.set_ylim(t0, t1)
        image_axis.set_xticks(distance_ticks(d0, d1))
        image_axis.yaxis.set_major_locator(MaxNLocator(
            nbins=CORRELOGRAM_MAX_LAG_TICKS,
            steps=[1, 2, 2.5, 5, 10],
        ))
        image_axis.set_xlabel("Interstation distance (°)")
        image_axis.set_ylabel("Time lag (s)")
        image_axis.grid(
            True, linestyle="--", linewidth=0.8, color="0.25", alpha=0.65
        )
        histogram_axis.set_title(title, pad=8)
        add_theoretical_line(image_axis, theoretical_arrival)
        figure.subplots_adjust(
            left=0.16, right=0.98, bottom=0.08, top=0.96, hspace=0.06
        )
        return figure


    def create_wiggles(
        event_name,
        azimuth_start,
        azimuth_end,
        correlation,
        distances,
        lag_sec,
        theoretical_arrival,
    ):
        d0, d1 = DISPLAY_DISTANCE_RANGE_DEG
        t0, t1 = DISPLAY_LAG_RANGE_SEC
        title = (
            "{} | {:g}–{:g}° + {:g}–{:g}° | stacked-distance-bin wiggles"
            .format(
                event_name,
                azimuth_start,
                azimuth_end,
                azimuth_start + 180.0,
                azimuth_end + 180.0,
            )
        )

        figure, axis = plt.subplots(figsize=FIGURE_SIZE)
        axis.set_xlim(d0, d1)
        axis.set_ylim(t0, t1)
        axis.set_xticks(distance_ticks(d0, d1))
        axis.yaxis.set_major_locator(MaxNLocator(
            nbins=CORRELOGRAM_MAX_LAG_TICKS,
            steps=[1, 2, 3, 5, 6, 10],
        ))
        axis.set_xlabel("Interstation distance (°)")
        axis.set_ylabel("Time lag (s)")
        axis.grid(True, linestyle="--", linewidth=0.8, color="0.25", alpha=0.65)
        axis.set_title(title, pad=8)

        plot_indices = np.arange(len(lag_sec))
        if len(plot_indices) > WIGGLE_MAX_PLOT_SAMPLES:
            plot_indices = np.linspace(
                0, len(plot_indices) - 1, WIGGLE_MAX_PLOT_SAMPLES
            ).astype(int)
        plot_lag_sec = lag_sec[plot_indices]

        for distance, trace in zip(distances, correlation):
            if not np.any(np.isfinite(trace)):
                continue
            plot_trace = np.nan_to_num(trace[plot_indices], nan=0.0)
            x = distance + WIGGLE_AMPLITUDE_DEG * plot_trace
            axis.plot(
                x,
                plot_lag_sec,
                color="black",
                alpha=WIGGLE_ALPHA,
                linewidth=WIGGLE_LINE_WIDTH,
                solid_capstyle="round",
                zorder=3,
            )
            axis.fill_betweenx(
                plot_lag_sec,
                distance,
                x,
                where=(plot_trace > 0),
                color="black",
                alpha=WIGGLE_ALPHA,
                linewidth=0,
                zorder=2,
            )

        # Draw this last so the theoretical arrival remains visible over the
        # wiggles, just as it does over the correlogram image.
        add_theoretical_line(axis, theoretical_arrival)
        figure.subplots_adjust(left=0.16, right=0.98, bottom=0.08, top=0.96)
        return figure


    def plot_stacked_file(npz_path, theoretical_arrival):
        """Load, plot, save, and release one stacked file."""
        event_name = npz_path.parent.parent.name
        data = read_stacked_file(npz_path)
        correlation, distances, distance_edges, pair_count, lag_sec = \
            select_display_window(data)

        correlogram_figure = create_correlogram(
            event_name,
            data["azimuth_start"],
            data["azimuth_end"],
            correlation,
            distances,
            distance_edges,
            pair_count,
            lag_sec,
            theoretical_arrival,
        )
        wiggle_figure = create_wiggles(
            event_name,
            data["azimuth_start"],
            data["azimuth_end"],
            correlation,
            distances,
            lag_sec,
            theoretical_arrival,
        )

        output_dir = FIG_DIR / ("coda_correlation_" + event_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = "az_{:g}_{:g}".format(
            data["azimuth_start"], data["azimuth_end"]
        )
        try:
            save_figure(
                correlogram_figure,
                output_dir / (stem + "_correlogram"),
            )
            save_figure(
                wiggle_figure,
                output_dir / (stem + "_wiggles"),
            )
        finally:
            plt.close(correlogram_figure)
            plt.close(wiggle_figure)

        print(
            "Plotted {}: {} | {} distance bins, {} pairs, theoretical line {}"
            .format(
                event_name,
                npz_path.name,
                len(distances),
                int(np.sum(pair_count)),
                "enabled" if theoretical_arrival is not None else "not available",
            )
        )


    def save_plot_parameters():
        settings = {
            "display_distance_range_deg": DISPLAY_DISTANCE_RANGE_DEG,
            "display_lag_range_sec": DISPLAY_LAG_RANGE_SEC,
            "correlogram_bin_width_deg": CORRELOGRAM_BIN_WIDTH_DEG,
            "histogram_bin_width_deg": HISTOGRAM_BIN_WIDTH_DEG,
            "correlogram_alpha": CORRELOGRAM_ALPHA,
            "wiggle_amplitude_deg": WIGGLE_AMPLITUDE_DEG,
            "wiggle_alpha": WIGGLE_ALPHA,
            "wiggle_line_width": WIGGLE_LINE_WIDTH,
            "wiggle_max_plot_samples": WIGGLE_MAX_PLOT_SAMPLES,
            "input_type": "frequency-domain distance-bin stacked NPZ",
            "load_one_stacked_file_at_a_time": True,
            "collect_garbage_after_file": COLLECT_GARBAGE_AFTER_FILE,
            "colormap": COLORMAP_NAME,
            "theoretical_arrival_file": str(THEORETICAL_ARRIVAL_FILE),
            "theoretical_line": "black dashed",
        }
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(FIG_DIR / "plot_parameters.json", "w") as handle:
            json.dump(settings, handle, indent=2, sort_keys=True)


    def main():
        save_plot_parameters()
        files = load_stacked_files()
        theoretical_arrival = load_theoretical_arrival(
            THEORETICAL_ARRIVAL_FILE
        )

        for file_number, npz_path in enumerate(files, start=1):
            print("Loading stacked file {}/{}: {}".format(
                file_number, len(files), npz_path
            ))
            plot_stacked_file(npz_path, theoretical_arrival)
            if COLLECT_GARBAGE_AFTER_FILE:
                gc.collect()

        print("Figures:", FIG_DIR.resolve())


    main()
    return {"figure_dir": FIG_DIR}
