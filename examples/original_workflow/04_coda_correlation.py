#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process coda windows and save distance-binned stacked correlations.

The stacking order follows the ESR notebook:

1. preprocess each station window into a zero-padded spectrum;
2. define interstation-distance bins before correlation;
3. accumulate ``spec_a * conj(spec_b)`` into those bins in the frequency
   domain, using parallel workers, after applying the pair angular-offset
   filter;
4. inverse-transform only the finished bin-level spectra; and
5. save the stacked time-domain correlations and pair histogram.

The output is therefore one compact stacked file per event/azimuth group,
not one full correlation trace for every station pair.  The angular-offset
filter uses the folded forward bearing of the alphabetically ordered station
pair relative to the azimuth-bin center.
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import hashlib
import json
import re
import threading

import numpy as np
import pandas as pd
from obspy import read, read_events, Stream

try:
    import libgcc
except ImportError as exc:
    raise ImportError(
        "This script follows the ESR libgcc.GCC workflow. Put libgcc.py "
        "beside this script or add its directory to PYTHONPATH."
    ) from exc


# ---------------------------------------------------------------------------
# Input and processing settings
# ---------------------------------------------------------------------------
CATALOGUE_FILE = Path("./catalogue/09eacf4ce01d/selected_events.xml")
WAVEFORM_RUN_DIR = Path("./waveforms/dce0637980d8")
STATION_RUN_DIR = Path("./stations/4777cb801a0d")

START_HOURS, END_HOURS = 3.0, 10.0
MIN_COVERAGE_FRACTION = 0.9999
TARGET_SAMPLING_RATE_HZ = 20.0

CENTER_LON, CENTER_LAT = 130.0, 0.0
AZIMUTH_START_DEG, AZIMUTH_STOP_DEG = 0.0, 180.0
AZIMUTH_BIN_WIDTH_DEG = 10.0

# Keep only station pairs whose forward-bearing offset from the centre of
# their azimuth bin is within this signed axial range.  This is the same
# ``local_bearing_offset_deg`` / ``bearing_offsets`` convention used by the
# station-pair geometry notebook, not the signed great-circle plane angle.
PAIR_ANGULAR_OFFSET_RANGE_DEG = (-15.0, 15.0)

# The current plotting requirement is 90–175 degrees.  Change this to
# (0.0, 180.0) if a full-distance stack is needed for a later analysis.
STACK_DISTANCE_RANGE_DEG = (0.0, 180.0)
STACK_DISTANCE_BIN_WIDTH_DEG = 3.0

# These parameters describe the signal that is correlated.  They are saved in
# processing_parameters.json for reproducibility.
TEMP_NORMALIZATION_WIDTH_S = 128.0
SPECTRAL_NORMALIZATION_WIDTH_HZ = 2e-3
# In libgcc this band is used to calculate the running-absolute-mean weight;
# it is not a final band-pass applied to the whitened spectrum.
RAM_FREQUENCY_BAND_HZ = (0.02, 0.067)

# This is the actual final correlation band in the ESR notebook.  The
# NormalizeFilter() call below converts the period limits to frequency limits
# (1/50 to 1/15 Hz), divides by pair_count, and applies a zero-phase
# four-corner bandpass after the bin stack has been inverse transformed.
FINAL_CORRELATION_PERIOD_MIN_S = 15.0
FINAL_CORRELATION_PERIOD_MAX_S = 50.0

CORRELATION_DIR = Path("./coda_processed")
OVERWRITE_CORRELATION_FILES = True
MAX_EVENTS = None             # Set to 1 for a quick test run.

# Parallelism.  The pair-stack worker returns one cross-spectrum at a time,
# avoiding the very large per-thread (nbins, nfreq) local buffers used by the
# original notebook implementation.
N_PREPROCESS_WORKERS = 4
N_STACK_WORKERS = 4

# The ESR notebook uses a doubled FFT length before frequency-domain stacking.
FFT_PADDING_FACTOR = 2
STACKED_SAVE_DTYPE = np.float32
LAG_SAVE_DTYPE = np.float32


# GCC owns reusable FFTW buffers, so one preprocessor is kept per worker
# thread.  The returned spectrum is copied before that buffer can be reused.
_PREPROCESS_LOCAL = threading.local()


def event_origin(event):
    hypocenters = [
        origin for origin in event.origins
        if getattr(origin, "origin_type", None) == "hypocenter"
    ]
    origin = hypocenters[0] if hypocenters else event.preferred_origin()
    if origin is None or origin.time is None:
        raise ValueError("Missing origin time: {}".format(event.resource_id))
    return origin


def event_slug(event):
    resource_id = str(event.resource_id)
    parts = resource_id.rstrip("/").split("/")
    label = parts[-2] if parts[-1] == "event" and len(parts) > 1 else parts[-1]
    label = re.sub(r"[^A-Za-z0-9_-]", "_", label)
    return label + "_" + hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:8]


def vector(lat, lon):
    latitude, longitude = np.deg2rad([lat, lon])
    return np.array([
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude),
    ])


def distance_bearing(lat0, lon0, lat1, lon1):
    first, second = vector(lat0, lon0), vector(lat1, lon1)
    distance = np.rad2deg(np.arctan2(
        np.linalg.norm(np.cross(first, second)), np.dot(first, second)
    ))

    latitude0, latitude1 = np.deg2rad([lat0, lat1])
    delta_lon = np.deg2rad(lon1 - lon0)
    y = np.sin(delta_lon) * np.cos(latitude1)
    x = (
        np.cos(latitude0) * np.sin(latitude1)
        - np.sin(latitude0) * np.cos(latitude1) * np.cos(delta_lon)
    )
    bearing = (
        np.rad2deg(np.arctan2(y, x)) % 360.0
        if np.hypot(x, y) > 1e-12 else np.nan
    )
    return float(distance), float(bearing)


def axial_angle(value):
    """Fold an angle into the established axial interval [-90, 90)."""
    return float((value + 90.0) % 180.0 - 90.0)


def pair_angular_offset(bearing, bin_center):
    """Return pair bearing minus azimuth-bin center, folded axially."""
    if not np.isfinite(bearing):
        return np.nan
    return axial_angle(bearing - bin_center)


def coverage_fraction(stream, start, end):
    spans = sorted(
        (
            max(start, trace.stats.starttime),
            min(end, trace.stats.endtime + trace.stats.delta),
        )
        for trace in stream if trace.stats.npts
    )
    total = 0.0
    cursor = start
    for lo, hi in spans:
        if hi > max(cursor, lo):
            total += float(hi - max(cursor, lo))
            cursor = max(cursor, hi)
    return min(1.0, total / float(end - start))


def number_of_samples():
    return int(round(
        (END_HOURS - START_HOURS)
        * 3600.0
        * TARGET_SAMPLING_RATE_HZ
    ))


def make_bin_edges(start, stop, width):
    if width <= 0:
        raise ValueError("STACK_DISTANCE_BIN_WIDTH_DEG must be positive.")
    if stop <= start:
        raise ValueError("The stack distance range must have stop > start.")
    count = int(np.ceil((stop - start) / float(width)))
    edges = start + np.arange(count + 1, dtype=float) * width
    edges[-1] = stop
    return edges


def distance_bin_index(distance, edges):
    index = int(np.searchsorted(edges, distance, side="right") - 1)
    if index == len(edges) - 1:
        index -= 1
    if index < 0 or index >= len(edges) - 1:
        return None
    return index


def gcc_for_current_worker():
    """Return the libgcc preprocessor assigned to the current worker."""
    gcc = getattr(_PREPROCESS_LOCAL, "gcc", None)
    if gcc is None:
        gcc = libgcc.GCC(
            FFT_PADDING_FACTOR * number_of_samples(),
            TEMP_NORMALIZATION_WIDTH_S,
            SPECTRAL_NORMALIZATION_WIDTH_HZ,
            RAM_FREQUENCY_BAND_HZ,
        )
        _PREPROCESS_LOCAL.gcc = gcc
    return gcc


def preprocess(stream, start, end, gcc):
    """Return a libgcc-preprocessed, zero-padded station spectrum and status.

    ``libgcc.GCC.pre_proc`` performs the same running-absolute-mean
    normalization and spectral whitening used by the ESR notebook.  Its FFT
    length is twice the time-window length, so the later inverse transform is
    also done with that same doubled length.
    """
    stream = stream.copy()
    stream.merge(method=-1)
    if len(stream) != 1:
        return None, 0.0, "multiple_segments_or_overlap"

    trace = stream[0].copy()
    trace.trim(start, end, pad=False, nearest_sample=False)
    coverage = coverage_fraction(Stream([trace]), start, end)
    if coverage < MIN_COVERAGE_FRACTION:
        return None, coverage, "coverage_below_threshold"

    if abs(trace.stats.sampling_rate - TARGET_SAMPLING_RATE_HZ) > 1e-5:
        trace.resample(TARGET_SAMPLING_RATE_HZ)

    n_samples = number_of_samples()
    trace.trim(
        trace.stats.starttime,
        trace.stats.starttime + (n_samples - 1) / TARGET_SAMPLING_RATE_HZ,
        pad=False,
    )
    if trace.stats.npts != n_samples or not np.isfinite(trace.data).all():
        return None, coverage, "nonfinite_or_incomplete_after_resample"

    # libgcc.ReadData applies both linear detrending and demeaning before
    # GCC.pre_proc.  Keep the same order for the MiniSEED input workflow.
    trace.detrend("linear")
    trace.detrend("demean")

    try:
        spectrum = gcc.pre_proc(trace)
    except Exception as exc:
        return None, coverage, "libgcc_preprocessing_error: " + str(exc)

    if spectrum is None or not np.all(np.isfinite(spectrum)):
        return None, coverage, "nonfinite_spectrum"

    # GCC returns a view backed by its reusable FFTW buffer.  Copy it before
    # the same worker handles another station.
    spectrum = np.asarray(spectrum, dtype=np.complex64).copy()
    return spectrum, coverage, "complete"


def pair_bin(azimuth, azimuth_edges):
    if not np.isfinite(azimuth):
        return None
    axial_azimuth = azimuth % 180.0
    if not (AZIMUTH_START_DEG <= axial_azimuth < AZIMUTH_STOP_DEG):
        return None
    return int(np.searchsorted(azimuth_edges, axial_azimuth, side="right") - 1)


def preprocess_station_task(task):
    """Thread worker for one station waveform."""
    row, start, end = task
    try:
        spectrum, coverage, status = preprocess(
            read(row["file"]), start, end, gcc_for_current_worker()
        )
    except Exception as exc:
        spectrum, coverage = None, 0.0
        status = "read_or_processing_error: " + str(exc)
    return row, spectrum, coverage, status


def cross_spectrum_task(pair_record):
    """Return one libgcc-style frequency-domain cross-spectrum.

    This is exactly the operation used by
    ``GCC.xcorr(spec_a, spec_b, freq_domain=True)``: multiply one spectrum by
    the complex conjugate of the other, but do not inverse-transform here.
    """
    spectrum_a, spectrum_b, bin_index = pair_record
    cross_spectrum = spectrum_a * np.conj(spectrum_b)
    return bin_index, cross_spectrum


def save_processing_parameters():
    parameters = {
        "catalogue_file": str(CATALOGUE_FILE),
        "waveform_run_dir": str(WAVEFORM_RUN_DIR),
        "station_run_dir": str(STATION_RUN_DIR),
        "start_hours": START_HOURS,
        "end_hours": END_HOURS,
        "minimum_coverage_fraction": MIN_COVERAGE_FRACTION,
        "target_sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
        "center_lon": CENTER_LON,
        "center_lat": CENTER_LAT,
        "azimuth_start_deg": AZIMUTH_START_DEG,
        "azimuth_stop_deg": AZIMUTH_STOP_DEG,
        "azimuth_bin_width_deg": AZIMUTH_BIN_WIDTH_DEG,
        "pair_angular_offset_range_deg": PAIR_ANGULAR_OFFSET_RANGE_DEG,
        "pair_angular_offset_definition":
            "folded pair forward bearing minus azimuth-bin center",
        "stack_distance_range_deg": STACK_DISTANCE_RANGE_DEG,
        "stack_distance_bin_width_deg": STACK_DISTANCE_BIN_WIDTH_DEG,
        "temporal_normalization_width_s": TEMP_NORMALIZATION_WIDTH_S,
        "spectral_normalization_width_hz": SPECTRAL_NORMALIZATION_WIDTH_HZ,
        "ram_frequency_band_hz": RAM_FREQUENCY_BAND_HZ,
        "final_correlation_period_band_s": (
            FINAL_CORRELATION_PERIOD_MIN_S,
            FINAL_CORRELATION_PERIOD_MAX_S,
        ),
        "final_correlation_frequency_band_hz": (
            1.0 / FINAL_CORRELATION_PERIOD_MAX_S,
            1.0 / FINAL_CORRELATION_PERIOD_MIN_S,
        ),
        "preprocess_workers": N_PREPROCESS_WORKERS,
        "stack_workers": N_STACK_WORKERS,
        "fft_padding_factor": FFT_PADDING_FACTOR,
        "preprocessing_implementation": "libgcc.GCC.pre_proc",
        "frequency_domain_pair_operation":
            "spectrum_a * conjugate(spectrum_b)",
        "final_correlation_filter":
            "libgcc.NormalizeFilter after inverse FFT",
        "stacked_save_dtype": np.dtype(STACKED_SAVE_DTYPE).name,
        "lag_save_dtype": np.dtype(LAG_SAVE_DTYPE).name,
        "output_type": "distance-binned frequency-domain stack converted to time domain",
    }
    CORRELATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORRELATION_DIR / "processing_parameters.json", "w") as handle:
        json.dump(parameters, handle, indent=2, sort_keys=True)


def load_events():
    events = sorted(
        read_events(str(CATALOGUE_FILE)),
        key=lambda event: event_origin(event).time,
    )
    if MAX_EVENTS is not None:
        events = events[:MAX_EVENTS]
    if not events:
        raise ValueError("The selected catalogue is empty.")
    return events


def process_event(event, azimuth_edges, processing_log):
    name = event_slug(event)
    origin = event_origin(event)
    start = origin.time + START_HOURS * 3600.0
    end = origin.time + END_HOURS * 3600.0
    waveform_dir = WAVEFORM_RUN_DIR / name
    station_dir = STATION_RUN_DIR / name

    stations = pd.read_csv(
        station_dir / "stations.csv", dtype=str, keep_default_na=False
    )
    files = sorted(waveform_dir.glob("*.mseed"))
    candidates = []

    for waveform_file in files:
        match = re.match(
            r"(?P<sta>[^_]+)_(?P<net>[^_]+)_(?P<loc>[^_]+)_"
            r"(?P<cha>BH?Z|HHZ)\.mseed$",
            waveform_file.name,
        )
        if match is None:
            continue

        row = match.groupdict()
        row["loc"] = "" if row["loc"] == "--" else row["loc"]
        row["file"] = str(waveform_file)
        row["station_id"] = row["net"] + "." + row["sta"]

        station = stations[stations.station_id == row["station_id"]]
        if station.empty:
            continue
        station = station.iloc[0]
        row["latitude"] = float(station.latitude)
        row["longitude"] = float(station.longitude)
        _, azimuth = distance_bearing(
            CENTER_LAT, CENTER_LON, row["latitude"], row["longitude"]
        )
        row["center_azimuth_deg"] = azimuth
        row["pair_bin"] = pair_bin(azimuth, azimuth_edges)
        if row["pair_bin"] is not None:
            candidates.append((row, start, end))

    rows = []
    if candidates:
        worker_count = min(N_PREPROCESS_WORKERS, len(candidates))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(preprocess_station_task, candidates)
            for row, spectrum, coverage, status in results:
                row["coverage_fraction"] = coverage
                row["status"] = status
                processing_log.append({
                    "event": name,
                    "station_id": row["station_id"],
                    "file": row["file"],
                    "coverage_fraction": coverage,
                    "status": status,
                })
                if spectrum is not None:
                    row["spec"] = spectrum
                    rows.append(row)

    event_output = CORRELATION_DIR / name
    event_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "spec"}
        for row in rows
    ]).to_csv(event_output / "usable_stations.csv", index=False)

    save_event_stacks(name, rows, azimuth_edges)
    return name, len(rows)


def save_event_stacks(name, rows, azimuth_edges):
    """Stack all pair cross-spectra in distance bins for each azimuth group."""
    correlation_output = CORRELATION_DIR / name / "correlations"
    correlation_output.mkdir(parents=True, exist_ok=True)

    stack_edges = make_bin_edges(
        STACK_DISTANCE_RANGE_DEG[0],
        STACK_DISTANCE_RANGE_DEG[1],
        STACK_DISTANCE_BIN_WIDTH_DEG,
    )
    stack_centers = 0.5 * (stack_edges[:-1] + stack_edges[1:])
    n_bins = len(stack_centers)
    n_samples = number_of_samples()
    fft_n_samples = FFT_PADDING_FACTOR * n_samples
    lag_sec = np.arange(n_samples, dtype=float) / TARGET_SAMPLING_RATE_HZ

    for group in range(len(azimuth_edges) - 1):
        members = [row for row in rows if row["pair_bin"] == group]
        lo, hi = azimuth_edges[group:group + 2]
        azimuth_bin_center = 0.5 * (lo + hi)
        stem = "az_{:g}_{:g}".format(lo, hi)
        output_path = correlation_output / (stem + "_stacked.npz")

        # Remove only stale stacked output for this exact group.
        if OVERWRITE_CORRELATION_FILES and output_path.exists():
            output_path.unlink()

        pair_records = []
        pair_count = np.zeros(n_bins, dtype=np.int64)
        candidate_pair_count = 0
        rejected_offset_pair_count = 0

        for first, second in combinations(
            sorted(members, key=lambda row: row.get("station_id", "")), 2
        ):
            distance, forward_bearing = distance_bearing(
                first["latitude"], first["longitude"],
                second["latitude"], second["longitude"],
            )
            candidate_pair_count += 1
            angular_offset = pair_angular_offset(
                forward_bearing, azimuth_bin_center
            )
            offset_min, offset_max = PAIR_ANGULAR_OFFSET_RANGE_DEG
            if (
                not np.isfinite(angular_offset)
                or angular_offset < offset_min
                or angular_offset > offset_max
            ):
                rejected_offset_pair_count += 1
                continue

            bin_index = distance_bin_index(distance, stack_edges)
            if bin_index is None:
                continue
            pair_records.append((first["spec"], second["spec"], bin_index))
            pair_count[bin_index] += 1

        if not pair_records:
            continue

        nfreq = len(pair_records[0][0])
        spec_stack = np.zeros((n_bins, nfreq), dtype=np.complex64)
        worker_count = min(N_STACK_WORKERS, len(pair_records))

        print(
            "Stacking {} | {}–{}° + {}–{}°: {} accepted pairs "
            "({} candidates, {} rejected by angular offset), "
            "{} bins, {} workers"
            .format(
                name, lo, hi, lo + 180.0, hi + 180.0,
                len(pair_records), candidate_pair_count,
                rejected_offset_pair_count, n_bins, worker_count,
            )
        )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for bin_index, cross_spectrum in executor.map(
                cross_spectrum_task, pair_records
            ):
                spec_stack[bin_index] += cross_spectrum

        # Inverse-transform the completed, frequency-domain SUM for each bin.
        # This is deliberately after all pair spectra have been accumulated.
        # Do not divide by pair_count here: NormalizeFilter() performs that
        # division together with the final ESR 15–50 s bandpass below.
        inverse_gcc = libgcc.GCC(
            fft_n_samples,
            TEMP_NORMALIZATION_WIDTH_S,
            SPECTRAL_NORMALIZATION_WIDTH_HZ,
            RAM_FREQUENCY_BAND_HZ,
        )
        stacked_correlation = np.zeros(
            (n_bins, n_samples), dtype=STACKED_SAVE_DTYPE
        )
        for bin_index, count in enumerate(pair_count):
            if count <= 0:
                continue
            correlation = inverse_gcc.inv_rfft(spec_stack[bin_index])
            correlation = 0.5 * (correlation + correlation[::-1])
            stacked_correlation[bin_index] = correlation[:n_samples]

        # This is the step used by the ESR notebook after the inverse FFT:
        # divide every bin by its number of contributing pairs and apply the
        # requested zero-phase bandpass.  With periods 15–50 s, libgcc uses
        # frequency limits 0.02–0.0666667 Hz.
        stacked_correlation = libgcc.NormalizeFilter(
            stacked_correlation,
            pair_count,
            1.0 / TARGET_SAMPLING_RATE_HZ,
            FINAL_CORRELATION_PERIOD_MIN_S,
            FINAL_CORRELATION_PERIOD_MAX_S,
        )
        stacked_correlation = np.asarray(
            stacked_correlation, dtype=STACKED_SAVE_DTYPE
        )

        np.savez(
            output_path,
            correlation=stacked_correlation,
            distance_deg=stack_centers,
            distance_bin_edges_deg=stack_edges,
            pair_count=pair_count,
            lag_sec=np.asarray(lag_sec, dtype=LAG_SAVE_DTYPE),
            sampling_rate_hz=TARGET_SAMPLING_RATE_HZ,
            start_hours=START_HOURS,
            end_hours=END_HOURS,
            center_lon=CENTER_LON,
            center_lat=CENTER_LAT,
            azimuth_bin_start_deg=lo,
            azimuth_bin_end_deg=hi,
            azimuth_bin_center_deg=azimuth_bin_center,
            pair_angular_offset_min_deg=PAIR_ANGULAR_OFFSET_RANGE_DEG[0],
            pair_angular_offset_max_deg=PAIR_ANGULAR_OFFSET_RANGE_DEG[1],
            stack_distance_start_deg=STACK_DISTANCE_RANGE_DEG[0],
            stack_distance_end_deg=STACK_DISTANCE_RANGE_DEG[1],
            stack_distance_bin_width_deg=STACK_DISTANCE_BIN_WIDTH_DEG,
            final_correlation_period_min_s=FINAL_CORRELATION_PERIOD_MIN_S,
            final_correlation_period_max_s=FINAL_CORRELATION_PERIOD_MAX_S,
            final_correlation_frequency_min_hz=(
                1.0 / FINAL_CORRELATION_PERIOD_MAX_S
            ),
            final_correlation_frequency_max_hz=(
                1.0 / FINAL_CORRELATION_PERIOD_MIN_S
            ),
            fft_length_samples=fft_n_samples,
            total_pair_count=int(pair_count.sum()),
        )
        print(
            "Saved stacked result: {} ({} pairs)".format(
                output_path, int(pair_count.sum())
            )
        )

        del inverse_gcc, spec_stack, stacked_correlation, pair_records


def main():
    if not CATALOGUE_FILE.exists():
        raise FileNotFoundError("Catalogue not found: {}".format(CATALOGUE_FILE))
    if not WAVEFORM_RUN_DIR.exists():
        raise FileNotFoundError(
            "Waveform directory not found: {}".format(WAVEFORM_RUN_DIR)
        )
    if not STATION_RUN_DIR.exists():
        raise FileNotFoundError(
            "Station directory not found: {}".format(STATION_RUN_DIR)
        )

    azimuth_edges = np.r_[
        np.arange(
            AZIMUTH_START_DEG,
            AZIMUTH_STOP_DEG,
            AZIMUTH_BIN_WIDTH_DEG,
        ),
        AZIMUTH_STOP_DEG,
    ]
    save_processing_parameters()
    processing_log = []
    event_counts = {}

    for event in load_events():
        name, count = process_event(event, azimuth_edges, processing_log)
        event_counts[name] = count

    pd.DataFrame(processing_log).to_csv(
        CORRELATION_DIR / "processing_log.csv", index=False
    )
    print("Usable stations:", event_counts)
    print("Stacked correlations:", CORRELATION_DIR.resolve())


if __name__ == "__main__":
    main()
