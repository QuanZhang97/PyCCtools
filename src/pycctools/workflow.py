"""Explicit path handoff between stages; no hard-coded run identifiers."""


def run_workflow(*, event_settings, station_settings=None, download_settings=None,
                 correlation_settings=None, plot_settings=None):
    """Run the complete online workflow and return all stage output paths.

    Settings dictionaries use uppercase names. Input paths are
    supplied by the preceding stage and cannot be overridden by a later stage.
    Online stages can issue many requests; call individual modules for partial
    runs or work with previously downloaded data.
    """
    from . import events, stations, download, correlation, plotting

    def downstream(settings, **inputs):
        settings = dict(settings or {})
        overlap = settings.keys() & inputs.keys()
        if overlap:
            raise ValueError(f"Workflow supplies these paths: {sorted(overlap)}")
        return dict(settings, **inputs)

    # Check conflicting paths before any network requests.
    downstream(station_settings, CATALOGUE_FILE=None)
    downstream(download_settings, CATALOGUE_FILE=None, STATION_RUN_DIR=None)
    downstream(correlation_settings, CATALOGUE_FILE=None, STATION_RUN_DIR=None, WAVEFORM_RUN_DIR=None)
    downstream(plot_settings, CORRELATION_DIR=None)
    for module, settings in [(events, event_settings), (stations, station_settings),
                             (download, download_settings), (correlation, correlation_settings),
                             (plotting, plot_settings)]:
        from ._config import validate_settings
        validate_settings(settings or {}, module.SETTINGS)
    result = {}
    result["events"] = events.run(**event_settings)
    catalogue = result["events"]["catalogue_file"]
    result["stations"] = stations.run(**downstream(station_settings, CATALOGUE_FILE=catalogue))
    station_dir = result["stations"]["run_dir"]
    result["download"] = download.run(**downstream(
        download_settings, CATALOGUE_FILE=catalogue, STATION_RUN_DIR=station_dir))
    result["correlation"] = correlation.run(**downstream(
        correlation_settings, CATALOGUE_FILE=catalogue, STATION_RUN_DIR=station_dir,
        WAVEFORM_RUN_DIR=result["download"]["run_dir"]))
    result["plotting"] = plotting.run(**downstream(
        plot_settings, CORRELATION_DIR=result["correlation"]["correlation_dir"]))
    return result
