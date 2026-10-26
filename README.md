# PyCCtools

PyCCtools is a Python package for seismic coda cross-correlation and theoretical phase-pair travel-time analysis. It supports the full workflow from event selection to station selection, coda download, azimuth/distance-bin stacking, and correlogram or wiggle plotting.

The travel-time API accepts any phase supported by ObsPy TauP. It is not limited to the I–J phase pair. The default Earth model is PREM, and the default example compares `PKIKP` and `PKJKP`.

## Installation

PyCCtools requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[maps,dev]'
pytest -q
```

Use `python -m pip install -e .` if you only need travel-time calculations or want to process existing waveforms without station maps. The `maps` extra installs Cartopy for station selection and mapping. The `dev` extra installs the test and build tools.

Importing PyCCtools does not make network requests, create output directories, or start a processing run.

## Travel times for arbitrary phase pairs

The default model is PREM:

```python
import numpy as np

from pycctools import phase_pair_curve, travel_time

arrival = travel_time(
    distance_deg=150,
    depth_km=20,
    phase="PKIKP",
)
print(arrival.time_sec, arrival.ray_parameter_sec_degree)

curve = phase_pair_curve(
    np.arange(90.0, 181.0),
    depth_km=20,
    phase_a="PKIKP",
    phase_b="PKJKP",
    model="prem",
)
curve.save("results/PKJKP_minus_PKIKP.txt")

# Other phases and models are supported as well.
other = phase_pair_curve(
    [30, 40, 50, 60],
    phase_a="P",
    phase_b="S",
    model="ak135",
)
```

The same calculation is available from the command line:

```bash
pycctools traveltime \
  --phase-a P \
  --phase-b S \
  --model prem \
  --depth-km 20 \
  --distance-min 30 \
  --distance-max 90 \
  --step 1 \
  --output results/S_minus_P.txt
```

The signed-lag convention is:

```text
lag = t(phase_b) - t(phase_a)
```

Times and lags are in seconds. For each phase, PyCCtools selects the earliest arrival at the requested epicentral distance and source depth. A point with no valid ray path is stored as `NaN`; the package does not interpolate across a shadow zone. Invalid phase names, model names, or numeric inputs raise an error. The saved text file records the model, phases, source depth, lag convention, and branch-selection rule in its header.

This function produces a same-distance travel-time-difference reference curve. It is not a matched-ray-parameter correlation-phase prediction, automatic detector, or phase picker. Before interpreting a line on a correlation plot, confirm that the phase geometry and branch selection are appropriate for the correlation wavefield under study.

Negative theoretical lags are retained. The plotting stage defaults to `DISPLAY_LAG_RANGE_SEC=(0, 3600)`, so set a negative lower bound when you need to display the negative branch.

## Stage-by-stage workflow

The five processing stages use uppercase setting names passed as keyword arguments. Unknown settings raise an error. Each invocation has its own settings and FFT worker buffers.

Inspect the supported settings for any stage with:

```python
from pycctools import correlation, download, events, plotting, stations

print(correlation.SETTINGS)
```

The following example runs every stage explicitly:

```python
from pathlib import Path

from pycctools import correlation, download, events, plotting, stations

# Use a local NDK or QuakeML catalogue to avoid downloading monthly GCMT files.
event_result = events.run(
    LOCAL_CATALOGUE="data/source.ndk",
    LOCAL_FORMAT="NDK",
    START_TIME="2010-01-01",
    END_TIME="2011-01-01",
    LAT_MIN=-5.0,
    LAT_MAX=5.0,
    LON_MIN=120.0,
    LON_MAX=140.0,
    MAG_MIN=7.0,
    MAG_MAX=8.0,
    MECHANISMS={"normal", "reverse"},
)

station_result = stations.run(
    CATALOGUE_FILE=event_result["catalogue_file"],
    WINDOW_START_OFFSET_S=3 * 3600.0,
    WINDOW_END_OFFSET_S=10 * 3600.0,
)

download_result = download.run(
    CATALOGUE_FILE=event_result["catalogue_file"],
    STATION_RUN_DIR=station_result["run_dir"],
    START_HOURS=3.0,
    END_HOURS=10.0,
)

correlation_result = correlation.run(
    CATALOGUE_FILE=event_result["catalogue_file"],
    STATION_RUN_DIR=station_result["run_dir"],
    WAVEFORM_RUN_DIR=download_result["run_dir"],
    START_HOURS=3.0,
    END_HOURS=10.0,
    N_PREPROCESS_WORKERS=4,
    N_STACK_WORKERS=4,
)

plotting.run(
    CORRELATION_DIR=correlation_result["correlation_dir"],
    THEORETICAL_ARRIVAL_FILE=Path(
        "results/PKJKP_minus_PKIKP.txt"
    ),
)
```

`stations.run(CATALOGUE_FILE=None)` performs automatic discovery only when exactly one `./catalogue/*/selected_events.xml` file exists. The download and correlation stages require explicit input paths and do not rely on hard-coded run IDs. Output-directory settings accept either strings or `Path` objects. Relative paths are resolved from the current working directory.

The metadata-selection, waveform-download, and correlation windows must be consistent. Their package defaults are all 3–10 hours after the event origin. Path handoff does not verify that scientific settings such as time windows and azimuth bins agree across stages; the caller remains responsible for that consistency.

### Automatic path handoff

The workflow helper can run all five stages and pass the actual output paths downstream:

```python
from pycctools.workflow import run_workflow

result = run_workflow(
    event_settings={
        "LOCAL_CATALOGUE": "data/source.ndk",
        "LOCAL_FORMAT": "NDK",
        "START_TIME": "2010-01-01",
        "END_TIME": "2011-01-01",
    },
    station_settings={
        "WINDOW_START_OFFSET_S": 3 * 3600.0,
        "WINDOW_END_OFFSET_S": 10 * 3600.0,
    },
    download_settings={
        "START_HOURS": 3.0,
        "END_HOURS": 10.0,
    },
    correlation_settings={
        "START_HOURS": 3.0,
        "END_HOURS": 10.0,
    },
    plot_settings={},
)
```

Do not put stage-to-stage path settings such as `CATALOGUE_FILE` in downstream dictionaries; `run_workflow` supplies them and rejects conflicting values before making any network requests.

### JSON configuration and CLI stages

Every stage also has a CLI subcommand that accepts a JSON settings object:

```bash
pycctools correlation --config examples/correlation.json
pycctools plotting --config examples/plotting.json
```

JSON configurations use arrays where Python examples use tuples or sets. Paths in the example files are placeholders and must be replaced with paths from your run. Online GCMT selection can request many monthly catalogue files, so set a deliberate time interval or supply a local catalogue.

## Event epicenter as the station-selection center

Every event uses its own epicenter from the hypocentral origin as the station-selection and azimuth-binning center. A global fixed center is not supported. Passing legacy `CENTER_LAT`, `CENTER_LON`, or map-center settings raises an unknown-setting error.

Station metadata are queried for the full time span of all events. Radius and azimuth filters are then applied locally and independently around each event epicenter. Each event map uses the event longitude as its central longitude.

Because the station query is global before local per-event filtering, a wide catalogue time span may produce a large FDSN metadata request. Prefer a narrow event/time selection or use local/cached metadata when possible.

The interstation distance is calculated directly from the two station coordinates and therefore does not depend on the event epicenter.

## Ordered station pairs and lag direction

Station pairs are ordered. For every accepted physical pair, PyCCtools computes and records both directions:

```text
A -> B: xcorr(A, B)
B -> A: xcorr(B, A)
```

Self-correlations are excluded. Before geometric filtering, `N` stations produce `N(N-1)` ordered pairs. `A -> B` and `B -> A` have the same interstation distance, enter the same distance bin, and each contribute one count to `pair_count`.

Geometric eligibility is evaluated once for the physical station pair using the station-ID-sorted orientation. If the pair is accepted, both ordered directions are retained.

Each ordered pair is saved separately at:

```text
<CORRELATION_DIR>/<event>/pairs/az_<start>_<end>/<pair_hash>.npz
```

Each pair file contains the full signed `lag_sec` axis, `correlation`, `station_a`, `station_b`, interstation distance, forward bearing, and event epicenter. The `manifest.csv` in the same directory maps each ordered direction to its file. No per-pair positive/negative-lag folding is applied, and swapping the input traces reverses the lag axis.

The per-pair values are correlations of running-absolute-mean-normalized and spectrally whitened traces, before the final pair-count normalization and 15–50 s band-pass filter. The distance-bin product sums all accepted directed correlations in the bin, divides by the directed `pair_count`, and applies the final zero-phase filter. Consequently, when both `A -> B` and `B -> A` are present in a bin, their contributions are both included in the aggregate stack even though their individual files remain separate.

A long coda window and many stations can produce a large number of pair files; use `MAX_EVENTS` or process events in batches when necessary.

## Correlating two existing traces

```python
from obspy import read

from pycctools.signal import correlate_pair

trace_a = read("A.mseed")[0]
trace_b = read("B.mseed")[0]

trace_a.detrend("linear").detrend("demean")
trace_b.detrend("linear").detrend("demean")

lag_sec, correlation_values = correlate_pair(trace_a, trace_b)
```

The two-trace interface requires identical start times, sample intervals, and sample counts, with finite and unmasked samples. It processes copies and does not modify the input traces.

By default it applies running-absolute-mean normalization and spectral whitening, then returns lags from `-(N-1) * delta` to `+(N-1) * delta`. The FFT convention is `spectrum_a * conjugate(spectrum_b)`. One invocation represents one ordered pair; positive and negative directions are not folded. This convenience function does not apply the final band-pass filter used by the distance-bin pipeline.

## Modules and outputs

| Module | Purpose and main outputs |
| --- | --- |
| `traveltime` | Generic single-phase and phase-pair TauP calculations; two-column reference curves with metadata |
| `common` | Event origin/slug handling, spherical geometry, axial angles, coverage, and binning |
| `events` | GCMT or local-catalogue filtering; QuakeML, audit CSV files, and source-time-function plots |
| `stations` | Metadata-epoch coverage, station selection, StationXML, station/channel tables, and maps |
| `download` | Chunked download, caching, completeness checks, miniSEED files, and ordered-pair tables |
| `signal` | FFTW preprocessing, directed cross-correlation, full signed-lag extraction, stacking, filtering, and NetCDF I/O |
| `correlation` | Per-event, azimuth, and interstation-distance frequency-domain stacking; ordered-pair NPZ files, stacked NPZ files, parameters, and logs |
| `plotting` | Stacked NPZ to correlogram and wiggle PNG/PDF figures, with an optional phase-pair curve |
| `workflow` | Path handoff across all five processing stages |

The stacked NPZ files include:

- `correlation`: distance-bin by signed-lag array
- `distance_deg`: distance-bin centers
- `distance_bin_edges_deg`: distance-bin edges
- `pair_count`: number of contributing ordered correlations per bin
- `lag_sec`: full signed-lag axis
- sampling rate, event epicenter, azimuth range, and processing parameters

Processing logs distinguish rejected waveforms. Download logs distinguish complete, partial, and failed requests. The waveform pipeline does not remove instrument response automatically; verify the response metadata, channel selection, and waveform quality required by your analysis.

## Recommended package examples

[`examples/package_workflow`](examples/package_workflow) contains a refined `00`–`05` workflow that calls the package functions directly. Edit its shared `settings.py`, then either run each numbered script or execute the complete sequence:

```bash
python examples/package_workflow/run_all.py
```

The numbered scripts write actual output paths to `example_run/workflow_state.json`, so catalogue, station, waveform, and correlation run hashes do not need to be hard-coded. See the [package-workflow guide](examples/package_workflow/README.md) for the individual commands and initial smoke-run limits.

## Validation

```bash
MPLBACKEND=Agg pytest -q
python -m build
```

The test suite covers PREM and alternative models/phases, absent arrivals, invalid parameters, lag signs, FFT buffer ownership, short and zero-valued signals, spherical geometry, binning, import side effects, local event selection, multi-epicenter station selection, mocked FDSN station and download services, ordered-direction files and counts, duplicate vertical-channel removal, synthetic miniSEED stacking, and figure output.

The mocked-service tests validate the local processing logic but do not prove that a live remote service is currently available.

## Implementation notes

`signal.py` provides `GCC`, `NormalizeFilter`, and related names together with snake-case aliases. The generic `travel_time` and `phase_pair_curve` APIs support `PKIKP`/`PKJKP` and other TauP phase pairs.

The package also fixes several issues found during migration:

- The hard-coded `events.pop(2)` no longer skips the third event or fails when fewer than three events are present.
- Ordered station directions and the full signed-lag axis are preserved in separate pair files.
- FFT results are copied so a later FFTW call cannot overwrite an earlier returned array.
- Zero-valued signals no longer create `NaN` values during phase-weighted stacking.
- A smoothing window longer than a short signal no longer causes an array-length mismatch.
- Missing theoretical arrivals remain `NaN`, so the plotting code does not draw a line across an absent branch or shadow zone.

The scientific selection criteria and output/log structure are retained where practical.

## Network destinations

Online operation may require access to:

- `www.ldeo.columbia.edu` for GCMT catalogues
- `service.earthscope.org` and `geofon.gfz.de` for FDSN station metadata and waveforms
- `naturalearth.s3.amazonaws.com` when Cartopy downloads Natural Earth data for the first map

Local catalogue files, existing waveforms, and populated caches can be used offline. These destinations are not automatically added to a restricted cloud-environment network policy.
