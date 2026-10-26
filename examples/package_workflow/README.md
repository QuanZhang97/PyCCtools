# Package-based workflow examples

These examples mirror the original `00`–`05` workflow while calling the reusable `pycctools` APIs. The scripts record actual output paths in `example_run/workflow_state.json`, so a later step does not need a hard-coded run hash.

Run all commands from the repository root after installing the package:

```bash
python -m pip install -e '.[maps]'
```

First edit `examples/package_workflow/settings.py`. In particular, replace `data/source.ndk` with your catalogue, or set `LOCAL_CATALOGUE` to `None` to download monthly GCMT data. The default station and event caps are intended for a small first run.

Run the stages one at a time:

```bash
python examples/package_workflow/step00_phase_pair_curve.py
python examples/package_workflow/step01_select_events.py
python examples/package_workflow/step02_select_stations.py
python examples/package_workflow/step03_download_coda.py
python examples/package_workflow/step04_correlate_coda.py
python examples/package_workflow/step05_plot_results.py
```

Or run the complete sequence:

```bash
python examples/package_workflow/run_all.py
```

The output handoff is:

```text
selected_events.xml
  -> station run directory
  -> waveform run directory
  -> correlation directory
  -> figures
```

Every event uses its own hypocentral epicenter for station geometry. Correlation outputs contain separate `A -> B` and `B -> A` pair files with full signed lags. Both directions have the same interstation distance and contribute separately to the distance-bin stack.

The station metadata query is global over the selected event time span before per-event filtering. Keep the initial time range and network selection narrow, inspect all audit files, and remove the example caps only when you are ready for a full scientific run.
