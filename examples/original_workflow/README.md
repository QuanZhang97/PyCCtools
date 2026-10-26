# Original workflow examples

This directory contains unchanged copies of the files supplied when PyCCtools was packaged. They are retained as scientific and migration examples:

1. `00_I-J_traveltime_diff_calc.ipynb` calculates the original I–J travel-time curve.
2. `01_select_seismic_events.py` selects catalogue events.
3. `02_select_stations_and_great_circles.py` selects stations and constructs great-circle geometry.
4. `03_download_coda_and_station_pairs.py` downloads coda waveforms and writes station-pair tables.
5. `04_coda_correlation.py` preprocesses and stacks coda correlations.
6. `05_plot_coda_correlation_and_wiggles.py` creates correlogram and wiggle plots.
7. `libgcc.py` contains the original signal-processing helpers.

These files retain their original hard-coded paths and behavior. They are examples, not the recommended package entry points. See the repository-level `README.md` for the corrected and configurable API.
