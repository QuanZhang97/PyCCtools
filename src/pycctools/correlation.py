"""Correlation stage, adapted from user-supplied 04_coda_correlation.py.

No processing occurs at import. Settings use the original uppercase names.
Each run has independent settings, helpers and FFT worker state.
"""
from ._config import setting, validate_settings
from .common import bounded_map

SETTINGS = ('CATALOGUE_FILE', 'WAVEFORM_RUN_DIR', 'STATION_RUN_DIR', 'START_HOURS', 'END_HOURS', 'MIN_COVERAGE_FRACTION', 'TARGET_SAMPLING_RATE_HZ', 'AZIMUTH_START_DEG', 'AZIMUTH_STOP_DEG', 'AZIMUTH_BIN_WIDTH_DEG', 'PAIR_ANGULAR_OFFSET_RANGE_DEG', 'STACK_DISTANCE_RANGE_DEG', 'STACK_DISTANCE_BIN_WIDTH_DEG', 'TEMP_NORMALIZATION_WIDTH_S', 'SPECTRAL_NORMALIZATION_WIDTH_HZ', 'RAM_FREQUENCY_BAND_HZ', 'FINAL_CORRELATION_PERIOD_MIN_S', 'FINAL_CORRELATION_PERIOD_MAX_S', 'CORRELATION_DIR', 'OVERWRITE_CORRELATION_FILES', 'MAX_EVENTS', 'N_PREPROCESS_WORKERS', 'N_STACK_WORKERS', 'FFT_PADDING_FACTOR', 'STACKED_SAVE_DTYPE', 'LAG_SAVE_DTYPE')

def run(**overrides):
    """Execute this stage and return its output paths. See SETTINGS for options."""
    validate_settings(overrides, SETTINGS)
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

    The output includes one stacked file per event/azimuth group,
    and a separate signed correlation for every accepted ordered station pair.  The angular-offset
    filter uses the folded forward bearing of the alphabetically ordered station
    pair relative to the azimuth-bin center.
    """

    from concurrent.futures import ThreadPoolExecutor
    from itertools import permutations
    from pathlib import Path
    import hashlib
    import json
    import re
    import threading

    import numpy as np
    import pandas as pd
    from obspy import read, read_events, Stream

    from . import signal as libgcc


    # ---------------------------------------------------------------------------
    # Input and processing settings
    # ---------------------------------------------------------------------------
    CATALOGUE_FILE = None
    CATALOGUE_FILE = setting(overrides, "CATALOGUE_FILE", CATALOGUE_FILE)
    WAVEFORM_RUN_DIR = None
    WAVEFORM_RUN_DIR = setting(overrides, "WAVEFORM_RUN_DIR", WAVEFORM_RUN_DIR)
    STATION_RUN_DIR = None
    STATION_RUN_DIR = setting(overrides, "STATION_RUN_DIR", STATION_RUN_DIR)

    START_HOURS, END_HOURS = 3.0, 10.0
    START_HOURS = setting(overrides, "START_HOURS", START_HOURS)
    END_HOURS = setting(overrides, "END_HOURS", END_HOURS)
    MIN_COVERAGE_FRACTION = 0.9999
    MIN_COVERAGE_FRACTION = setting(overrides, "MIN_COVERAGE_FRACTION", MIN_COVERAGE_FRACTION)
    TARGET_SAMPLING_RATE_HZ = 20.0
    TARGET_SAMPLING_RATE_HZ = setting(overrides, "TARGET_SAMPLING_RATE_HZ", TARGET_SAMPLING_RATE_HZ)

    AZIMUTH_START_DEG, AZIMUTH_STOP_DEG = 0.0, 180.0
    AZIMUTH_START_DEG = setting(overrides, "AZIMUTH_START_DEG", AZIMUTH_START_DEG)
    AZIMUTH_STOP_DEG = setting(overrides, "AZIMUTH_STOP_DEG", AZIMUTH_STOP_DEG)
    AZIMUTH_BIN_WIDTH_DEG = 10.0
    AZIMUTH_BIN_WIDTH_DEG = setting(overrides, "AZIMUTH_BIN_WIDTH_DEG", AZIMUTH_BIN_WIDTH_DEG)

    # Keep only station pairs whose forward-bearing offset from the centre of
    # their azimuth bin is within this signed axial range.  This is the same
    # ``local_bearing_offset_deg`` / ``bearing_offsets`` convention used by the
    # station-pair geometry notebook, not the signed great-circle plane angle.
    PAIR_ANGULAR_OFFSET_RANGE_DEG = (-15.0, 15.0)
    PAIR_ANGULAR_OFFSET_RANGE_DEG = setting(overrides, "PAIR_ANGULAR_OFFSET_RANGE_DEG", PAIR_ANGULAR_OFFSET_RANGE_DEG)

    # The current plotting requirement is 90–175 degrees.  Change this to
    # (0.0, 180.0) if a full-distance stack is needed for a later analysis.
    STACK_DISTANCE_RANGE_DEG = (0.0, 180.0)
    STACK_DISTANCE_RANGE_DEG = setting(overrides, "STACK_DISTANCE_RANGE_DEG", STACK_DISTANCE_RANGE_DEG)
    STACK_DISTANCE_BIN_WIDTH_DEG = 3.0
    STACK_DISTANCE_BIN_WIDTH_DEG = setting(overrides, "STACK_DISTANCE_BIN_WIDTH_DEG", STACK_DISTANCE_BIN_WIDTH_DEG)

    # These parameters describe the signal that is correlated.  They are saved in
    # processing_parameters.json for reproducibility.
    TEMP_NORMALIZATION_WIDTH_S = 128.0
    TEMP_NORMALIZATION_WIDTH_S = setting(overrides, "TEMP_NORMALIZATION_WIDTH_S", TEMP_NORMALIZATION_WIDTH_S)
    SPECTRAL_NORMALIZATION_WIDTH_HZ = 2e-3
    SPECTRAL_NORMALIZATION_WIDTH_HZ = setting(overrides, "SPECTRAL_NORMALIZATION_WIDTH_HZ", SPECTRAL_NORMALIZATION_WIDTH_HZ)
    # In libgcc this band is used to calculate the running-absolute-mean weight;
    # it is not a final band-pass applied to the whitened spectrum.
    RAM_FREQUENCY_BAND_HZ = (0.02, 0.067)
    RAM_FREQUENCY_BAND_HZ = setting(overrides, "RAM_FREQUENCY_BAND_HZ", RAM_FREQUENCY_BAND_HZ)

    # This is the actual final correlation band in the ESR notebook.  The
    # NormalizeFilter() call below converts the period limits to frequency limits
    # (1/50 to 1/15 Hz), divides by pair_count, and applies a zero-phase
    # four-corner bandpass after the bin stack has been inverse transformed.
    FINAL_CORRELATION_PERIOD_MIN_S = 15.0
    FINAL_CORRELATION_PERIOD_MIN_S = setting(overrides, "FINAL_CORRELATION_PERIOD_MIN_S", FINAL_CORRELATION_PERIOD_MIN_S)
    FINAL_CORRELATION_PERIOD_MAX_S = 50.0
    FINAL_CORRELATION_PERIOD_MAX_S = setting(overrides, "FINAL_CORRELATION_PERIOD_MAX_S", FINAL_CORRELATION_PERIOD_MAX_S)

    CORRELATION_DIR = Path("./coda_processed")
    CORRELATION_DIR = setting(overrides, "CORRELATION_DIR", CORRELATION_DIR)
    OVERWRITE_CORRELATION_FILES = True
    OVERWRITE_CORRELATION_FILES = setting(overrides, "OVERWRITE_CORRELATION_FILES", OVERWRITE_CORRELATION_FILES)
    MAX_EVENTS = None             # Set to 1 for a quick test run.
    MAX_EVENTS = setting(overrides, "MAX_EVENTS", MAX_EVENTS)

    # Parallelism.  The pair-stack worker returns one cross-spectrum at a time,
    # avoiding the very large per-thread (nbins, nfreq) local buffers used by the
    # original notebook implementation.
    N_PREPROCESS_WORKERS = 4
    N_PREPROCESS_WORKERS = setting(overrides, "N_PREPROCESS_WORKERS", N_PREPROCESS_WORKERS)
    N_STACK_WORKERS = 4
    N_STACK_WORKERS = setting(overrides, "N_STACK_WORKERS", N_STACK_WORKERS)

    # The ESR notebook uses a doubled FFT length before frequency-domain stacking.
    FFT_PADDING_FACTOR = 2
    FFT_PADDING_FACTOR = setting(overrides, "FFT_PADDING_FACTOR", FFT_PADDING_FACTOR)
    STACKED_SAVE_DTYPE = np.float32
    STACKED_SAVE_DTYPE = setting(overrides, "STACKED_SAVE_DTYPE", STACKED_SAVE_DTYPE)
    LAG_SAVE_DTYPE = np.float32
    LAG_SAVE_DTYPE = setting(overrides, "LAG_SAVE_DTYPE", LAG_SAVE_DTYPE)


    # GCC owns reusable FFTW buffers, so one preprocessor is kept per worker
    # thread.  The returned spectrum is copied before that buffer can be reused.
    if CATALOGUE_FILE is None:
        raise ValueError("CATALOGUE_FILE must be supplied explicitly.")
    CATALOGUE_FILE = Path(CATALOGUE_FILE)
    if STATION_RUN_DIR is None:
        raise ValueError("STATION_RUN_DIR must be supplied explicitly.")
    STATION_RUN_DIR = Path(STATION_RUN_DIR)
    if WAVEFORM_RUN_DIR is None:
        raise ValueError("WAVEFORM_RUN_DIR must be supplied explicitly.")
    WAVEFORM_RUN_DIR = Path(WAVEFORM_RUN_DIR)

    if not 0 <= START_HOURS < END_HOURS:
        raise ValueError("Require 0 <= START_HOURS < END_HOURS")
    if TARGET_SAMPLING_RATE_HZ <= 0 or N_PREPROCESS_WORKERS < 1 or N_STACK_WORKERS < 1 or FFT_PADDING_FACTOR < 2:
        raise ValueError("Invalid sampling rate, worker count or FFT padding")
    if not 0 < FINAL_CORRELATION_PERIOD_MIN_S < FINAL_CORRELATION_PERIOD_MAX_S:
        raise ValueError("Require 0 < minimum period < maximum period")
    if 1. / FINAL_CORRELATION_PERIOD_MIN_S >= TARGET_SAMPLING_RATE_HZ / 2.:
        raise ValueError("Final bandpass must lie below the Nyquist frequency")
    _PREPROCESS_LOCAL = threading.local()


    from .common import located_origin as event_origin


    from .common import event_slug as event_slug


    from .common import vector as vector


    from .common import distance_bearing as distance_bearing


    from .common import axial_angle as axial_angle


    from .common import pair_angular_offset as pair_angular_offset


    from .common import coverage_fraction as coverage_fraction


    def number_of_samples():
        return int(round(
            (END_HOURS - START_HOURS)
            * 3600.0
            * TARGET_SAMPLING_RATE_HZ
        ))


    from .common import regular_edges as make_bin_edges


    from .common import distance_bin_index as distance_bin_index


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


    from .common import azimuth_bin as pair_bin


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
        spectrum_a, spectrum_b, bin_index, metadata = pair_record
        cross_spectrum = spectrum_a * np.conj(spectrum_b)
        return bin_index, cross_spectrum, metadata


    def save_processing_parameters():
        parameters = {
            "catalogue_file": str(CATALOGUE_FILE),
            "waveform_run_dir": str(WAVEFORM_RUN_DIR),
            "station_run_dir": str(STATION_RUN_DIR),
            "start_hours": START_HOURS,
            "end_hours": END_HOURS,
            "minimum_coverage_fraction": MIN_COVERAGE_FRACTION,
            "target_sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
            "center_definition": "event_epicenter from hypocentral origin",
            "azimuth_start_deg": AZIMUTH_START_DEG,
            "azimuth_stop_deg": AZIMUTH_STOP_DEG,
            "azimuth_bin_width_deg": AZIMUTH_BIN_WIDTH_DEG,
            "pair_angular_offset_range_deg": PAIR_ANGULAR_OFFSET_RANGE_DEG,
            "pair_angular_offset_definition":
                "folded A-to-B bearing of alphabetically sorted physical pair; shared eligibility for both directions",
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
            "pair_order": "directed: A->B and B->A are separate contributions",
            "lag_convention": "A * conjugate(B); full signed lag axis",
            "symmetrized": False,
            "pair_output": "one unfiltered, preprocessed signed correlation per ordered pair",
            "pair_count_definition": "ordered correlations; 2 per accepted physical pair",
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
        CENTER_LAT, CENTER_LON = origin.latitude, origin.longitude
        start = origin.time + START_HOURS * 3600.0
        end = origin.time + END_HOURS * 3600.0
        waveform_dir = WAVEFORM_RUN_DIR / name
        station_dir = STATION_RUN_DIR / name

        stations = pd.read_csv(
            station_dir / "stations.csv", dtype=str, keep_default_na=False
        )
        files = sorted(waveform_dir.glob("*.mseed"))
        usable_path = waveform_dir / "usable_stations.csv"
        if usable_path.exists():
            usable = pd.read_csv(usable_path, dtype=str, keep_default_na=False)
            allowed = {Path(value).resolve() for value in usable["file"]}
            files = [path for path in files if path.resolve() in allowed]
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

        # Keep one accepted vertical recording per NET.STA; alternative
        # channels must never introduce self-pairs or duplicate ordered pairs.
        rows.sort(key=lambda row: (row["station_id"], row.get("cha") != "BHZ", row.get("loc", "")))
        unique_rows = {}
        for row in rows:
            unique_rows.setdefault(row["station_id"], row)
        rows = list(unique_rows.values())
        event_output = CORRELATION_DIR / name
        event_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            {key: value for key, value in row.items() if key != "spec"}
            for row in rows
        ]).to_csv(event_output / "usable_stations.csv", index=False)

        save_event_stacks(name, rows, azimuth_edges, origin)
        return name, len(rows)


    def save_event_stacks(name, rows, azimuth_edges, origin):
        """Stack ordered pair cross-spectra in distance bins for one event."""
        CENTER_LAT, CENTER_LON = origin.latitude, origin.longitude
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
        lag_sec = np.arange(-(n_samples - 1), n_samples, dtype=float) / TARGET_SAMPLING_RATE_HZ

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

            for first, second in permutations(
                sorted(members, key=lambda row: row.get("station_id", "")), 2
            ):
                distance, forward_bearing = distance_bearing(
                    first["latitude"], first["longitude"],
                    second["latitude"], second["longitude"],
                )
                candidate_pair_count += 1
                # Apply geometric eligibility once per physical station pair.
                # Both ordered correlations then enter the same distance bin.
                reference_bearing = forward_bearing
                if first["station_id"] > second["station_id"]:
                    _, reference_bearing = distance_bearing(
                        second["latitude"], second["longitude"],
                        first["latitude"], first["longitude"],
                    )
                angular_offset = pair_angular_offset(
                    reference_bearing, azimuth_bin_center
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
                pair_records.append((first["spec"], second["spec"], bin_index, {
                    "station_a": first["station_id"], "station_b": second["station_id"],
                    "epicenter_latitude": CENTER_LAT, "epicenter_longitude": CENTER_LON,
                    "distance_deg": distance, "forward_bearing_deg": forward_bearing,
                    "selection_angular_offset_deg": angular_offset,
                }))
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

            pair_output = CORRELATION_DIR / name / "pairs" / stem
            pair_output.mkdir(parents=True, exist_ok=True)
            pair_manifest = []
            pair_gcc = libgcc.GCC(fft_n_samples, None, None, None)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for bin_index, cross_spectrum, metadata in bounded_map(
                    executor, cross_spectrum_task, pair_records, max(1, 2 * worker_count)
                ):
                    spec_stack[bin_index] += cross_spectrum
                    pair_id = hashlib.sha256(json.dumps(
                        [metadata["station_a"], metadata["station_b"]]
                    ).encode()).hexdigest()[:20]
                    pair_path = pair_output / (pair_id + ".npz")
                    circular = pair_gcc.inv_rfft(cross_spectrum)
                    values = libgcc.linear_lags(circular, n_samples)
                    np.savez_compressed(
                        pair_path, correlation=values.astype(STACKED_SAVE_DTYPE),
                        lag_sec=lag_sec.astype(LAG_SAVE_DTYPE), **metadata,
                        pair_order="directed", convention="A * conjugate(B)",
                        final_bandpass_applied=False,
                    )
                    pair_manifest.append(dict(metadata, file=str(pair_path)))
            del pair_gcc
            pd.DataFrame(pair_manifest).to_csv(pair_output / "manifest.csv", index=False)

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
                (n_bins, 2 * n_samples - 1), dtype=STACKED_SAVE_DTYPE
            )
            for bin_index, count in enumerate(pair_count):
                if count <= 0:
                    continue
                correlation = inverse_gcc.inv_rfft(spec_stack[bin_index])
                stacked_correlation[bin_index] = libgcc.linear_lags(correlation, n_samples)

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
                pair_order="directed",
                symmetrized=False,
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


    main()
    return {"correlation_dir": CORRELATION_DIR}
