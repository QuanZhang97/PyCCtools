"""Events stage, adapted from user-supplied 01_select_seismic_events.py.

No processing occurs at import. Settings use the original uppercase names.
Each run has independent settings, helpers and FFT worker state.
"""
from ._config import setting, validate_settings

SETTINGS = ('START_TIME', 'END_TIME', 'LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX', 'MAG_MIN', 'MAG_MAX', 'MECHANISMS', 'AXIS_PLUNGE_THRESHOLD_DEG', 'LOCATION_ORIGIN', 'DEPTH_MIN_KM', 'DEPTH_MAX_KM', 'REQUIRE_PLOTTABLE_STF', 'LOCAL_CATALOGUE', 'LOCAL_FORMAT', 'CACHE_DIR', 'CATALOGUE_DIR', 'FIG_DIR', 'REFRESH_CACHE', 'PREVIEW_COUNT', 'GCMT_BASE')

def run(**overrides):
    """Execute this stage and return its output paths. See SETTINGS for options."""
    validate_settings(overrides, SETTINGS)
    #!/usr/bin/env python
    # coding: utf-8

    # # 01 · Select earthquakes for coda correlation
    #
    # Build an **ObsPy Catalog**, filter earthquakes, plot individual beachballs alongside catalogue source-time functions, and write **QuakeML** for the waveform-download notebook. Edit the configuration cell and run top to bottom.
    #
    # **Data source:** Global Centroid Moment Tensor (GCMT). Automatic download uses standard monthly NDK files from 2005 onward; for earlier dates, supply a local GCMT NDK catalogue. A local QuakeML file retaining GCMT origins, moment tensors, principal axes and source-time metadata is also supported. This is a moment-tensor catalogue, not a complete catalogue of small earthquakes. Recent standard monthly solutions may not yet be published.
    #
    # **Terminology:** “transverse” is interpreted as **strike-slip**. Selection uses the best-double-couple principal-axis geometry, not an assumed identification of the actual fault plane.
    #
    # **Source-time limitation:** GCMT's triangular/boxcar moment-rate shape and half-duration are assumed in the inversion; they are not independently recovered rupture histories. The plots reconstruct those assumed models. They cannot establish rupture complexity or identify genuinely impulsive events. No waveforms are downloaded here.
    #
    # Outputs are relative to the notebook working directory: `./catalogue/` and `./fig/`. Each configuration gets its own subdirectory, so different selections do not mix figures.

    from pathlib import Path
    from io import BytesIO
    import hashlib
    import json
    import re
    import platform
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import obspy
    from obspy import Catalog, UTCDateTime, read_events
    from obspy.core.event import Comment
    from obspy.imaging.beachball import beach
    from ._config import display

    print("ObsPy:", obspy.__version__)
    print("Working directory:", Path.cwd())


    START_TIME = "2005-01-01T00:00:00"
    START_TIME = setting(overrides, "START_TIME", START_TIME)
    END_TIME   = "2026-02-01T00:00:00"
    END_TIME = setting(overrides, "END_TIME", END_TIME)
    LAT_MIN, LAT_MAX = -5.0, 5.0
    LAT_MIN = setting(overrides, "LAT_MIN", LAT_MIN)
    LAT_MAX = setting(overrides, "LAT_MAX", LAT_MAX)
    LON_MIN, LON_MAX = 120.0, 140.0
    LON_MIN = setting(overrides, "LON_MIN", LON_MIN)
    LON_MAX = setting(overrides, "LON_MAX", LON_MAX)
    MAG_MIN, MAG_MAX = 7.0, 8.0
    MAG_MIN = setting(overrides, "MAG_MIN", MAG_MIN)
    MAG_MAX = setting(overrides, "MAG_MAX", MAG_MAX)
    MECHANISMS = {"normal", "reverse"} #, "strike-slip"}
    MECHANISMS = setting(overrides, "MECHANISMS", MECHANISMS)
    AXIS_PLUNGE_THRESHOLD_DEG = 50.0
    AXIS_PLUNGE_THRESHOLD_DEG = setting(overrides, "AXIS_PLUNGE_THRESHOLD_DEG", AXIS_PLUNGE_THRESHOLD_DEG)
    LOCATION_ORIGIN = "hypocenter"  # "hypocenter" or "centroid"
    LOCATION_ORIGIN = setting(overrides, "LOCATION_ORIGIN", LOCATION_ORIGIN)
    DEPTH_MIN_KM, DEPTH_MAX_KM = None, None  # optional additional filter
    DEPTH_MIN_KM = setting(overrides, "DEPTH_MIN_KM", DEPTH_MIN_KM)
    DEPTH_MAX_KM = setting(overrides, "DEPTH_MAX_KM", DEPTH_MAX_KM)
    REQUIRE_PLOTTABLE_STF = True
    REQUIRE_PLOTTABLE_STF = setting(overrides, "REQUIRE_PLOTTABLE_STF", REQUIRE_PLOTTABLE_STF)

    # None: download monthly GCMT files. Otherwise supply a local .ndk or .xml path.
    LOCAL_CATALOGUE = None
    LOCAL_CATALOGUE = setting(overrides, "LOCAL_CATALOGUE", LOCAL_CATALOGUE)
    LOCAL_FORMAT = "NDK"  # "NDK" or "QUAKEML"
    LOCAL_FORMAT = setting(overrides, "LOCAL_FORMAT", LOCAL_FORMAT)
    CACHE_DIR = Path("./data/gcmt")
    CACHE_DIR = setting(overrides, "CACHE_DIR", CACHE_DIR)
    CATALOGUE_DIR = Path("./catalogue")
    CATALOGUE_DIR = setting(overrides, "CATALOGUE_DIR", CATALOGUE_DIR)
    FIG_DIR = Path("./fig")
    FIG_DIR = setting(overrides, "FIG_DIR", FIG_DIR)
    REFRESH_CACHE = False
    REFRESH_CACHE = setting(overrides, "REFRESH_CACHE", REFRESH_CACHE)
    PREVIEW_COUNT = 3  # all selected STFs are saved, only this many are displayed
    PREVIEW_COUNT = setting(overrides, "PREVIEW_COUNT", PREVIEW_COUNT)
    GCMT_BASE = "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY"
    GCMT_BASE = setting(overrides, "GCMT_BASE", GCMT_BASE)


    start, end = UTCDateTime(START_TIME), UTCDateTime(END_TIME)
    assert start < end, "END_TIME must be later than START_TIME."
    assert -90 <= LAT_MIN <= LAT_MAX <= 90
    assert all(-180 <= x <= 180 for x in (LON_MIN, LON_MAX))
    assert np.isfinite([MAG_MIN, MAG_MAX]).all() and MAG_MIN <= MAG_MAX
    assert 45 < AXIS_PLUNGE_THRESHOLD_DEG <= 90
    assert LOCATION_ORIGIN in {"hypocenter", "centroid"}
    assert PREVIEW_COUNT >= 0
    assert DEPTH_MIN_KM is None or np.isfinite(DEPTH_MIN_KM)
    assert DEPTH_MAX_KM is None or np.isfinite(DEPTH_MAX_KM)
    assert DEPTH_MIN_KM is None or DEPTH_MAX_KM is None or DEPTH_MIN_KM <= DEPTH_MAX_KM
    allowed_mechanisms = {x.lower().replace("_", "-") for x in MECHANISMS}
    allowed_mechanisms = {"strike-slip" if x == "transverse" else x for x in allowed_mechanisms}
    assert allowed_mechanisms and allowed_mechanisms <= {"normal", "reverse", "strike-slip", "oblique"}
    if LOCAL_CATALOGUE is None and start.year < 2005:
        raise ValueError("For dates before 2005, download the full GCMT NDK catalogue and set LOCAL_CATALOGUE.")

    config = dict(start=str(start), end_exclusive=str(end), latitude=[LAT_MIN, LAT_MAX],
                  longitude=[LON_MIN, LON_MAX], magnitude=[MAG_MIN, MAG_MAX],
                  mechanisms=sorted(allowed_mechanisms), axis_threshold=AXIS_PLUNGE_THRESHOLD_DEG,
                  location_origin=LOCATION_ORIGIN, depth_km=[DEPTH_MIN_KM, DEPTH_MAX_KM],
                  require_stf=REQUIRE_PLOTTABLE_STF, local_catalogue=str(LOCAL_CATALOGUE),
                  local_format=LOCAL_FORMAT, gcmt_base=GCMT_BASE)
    print(json.dumps(config, indent=2))


    def parse_catalogue(path, fmt):
        # Strict parsing: malformed/skipped NDK events must be investigated.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cat = read_events(str(path), format=fmt)
        if caught:
            raise RuntimeError(f"Parsing warnings for {path}: " + "; ".join(str(w.message) for w in caught))
        if not cat:
            raise RuntimeError(f"Empty source catalogue: {path}")
        return cat

    def fetch_gcmt_month(year, month):
        # Historical files use june/july/sept; newer files often use jun/jul/sep.
        month_aliases = [("jan",), ("feb",), ("mar",), ("apr",), ("may",),
                         ("jun", "june"), ("jul", "july"), ("aug",),
                         ("sep", "sept"), ("oct",), ("nov",), ("dec",)]
        candidates = []
        for alias in month_aliases[month - 1]:
            name = f"{alias}{year % 100:02d}.ndk"
            candidates.append((CACHE_DIR / str(year) / name, f"{GCMT_BASE}/{year}/{name}"))
        if not REFRESH_CACHE:
            for path, url in candidates:
                if path.exists():
                    return path, url
        for path, url in candidates:
            print("Downloading", url)
            response = session.get(url, timeout=(15, 120))
            if response.status_code == 404:
                continue  # Try a historical filename, not a partial catalogue.
            response.raise_for_status()  # Do not hide network, permission or server errors.
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".download")
            try:
                temporary.write_bytes(response.content)
                parse_catalogue(temporary, "NDK")
                temporary.replace(path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            return path, url
        tried = ", ".join(url for _, url in candidates)
        raise RuntimeError(f"GCMT month unavailable after checking filename variants: {tried}. No partial selection will be saved.")

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    source_records = []
    raw_catalogue = Catalog()

    if LOCAL_CATALOGUE is not None:
        source_paths = [(Path(LOCAL_CATALOGUE).expanduser(), LOCAL_FORMAT, None)]
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        source_paths = []
        year, month = start.year, start.month
        while UTCDateTime(year, month, 1) < end:
            path, url = fetch_gcmt_month(year, month)
            source_paths.append((path, "NDK", url))
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)

    for path, fmt, url in source_paths:
        part = parse_catalogue(path, fmt)
        raw_catalogue += part
        source_records.append(dict(path=str(path), url=url, event_count=len(part),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    # Do not silently choose between conflicting revisions with identical IDs.
    ids = [str(ev.resource_id) for ev in raw_catalogue]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event IDs in source files. Resolve duplicate/revised solutions before selecting.")
    run_id = hashlib.sha256(json.dumps(dict(config=config, sources=source_records), sort_keys=True).encode()).hexdigest()[:12]
    run_dir = CATALOGUE_DIR / run_id
    fig_dir = FIG_DIR / run_id
    print(f"Loaded {len(raw_catalogue)} events; run {run_id}.")


    from .common import finite as finite

    def find_origin(event, kind):
        return next((o for o in event.origins if o.origin_type == kind), None)

    def focal_mechanism(event):
        return event.preferred_focal_mechanism() or (event.focal_mechanisms[0] if event.focal_mechanisms else None)

    def moment_magnitude(event):
        mags = [m for m in event.magnitudes if (m.magnitude_type or "").lower().startswith("mw") and finite(m.mag)]
        preferred = event.preferred_magnitude()
        return preferred if preferred in mags else (mags[0] if mags else None)

    def mechanism_class(fm):
        axes = fm.principal_axes if fm else None
        if axes is None:
            return "unknown"
        plunges = {name: getattr(axes, name) for name in ("p_axis", "t_axis", "n_axis")}
        if any(axis is None or not finite(axis.plunge) or not 0 <= axis.plunge <= 90 for axis in plunges.values()):
            return "unknown"
        labels = {"p_axis": "reverse", "t_axis": "normal", "n_axis": "strike-slip"}
        hits = [labels[name] for name, axis in plunges.items() if axis.plunge >= AXIS_PLUNGE_THRESHOLD_DEG]
        return hits[0] if len(hits) == 1 else ("oblique" if not hits else "unknown")

    def longitude_inside(lon):
        if LON_MIN == -180 and LON_MAX == 180:
            return True
        lon = (lon + 180) % 360 - 180
        candidates = [lon, 180.0] if lon == -180 else [lon]
        return any((LON_MIN <= x <= LON_MAX) if LON_MIN <= LON_MAX
                   else (x >= LON_MIN or x <= LON_MAX) for x in candidates)

    def stf_metadata(event, fm):
        mt = fm.moment_tensor if fm else None
        stf = mt.source_time_function if mt else None
        centroid = None
        if mt and mt.derived_origin_id:
            centroid = next((o for o in event.origins if o.resource_id == mt.derived_origin_id), None)
        if centroid is None:
            centroid = find_origin(event, "centroid")
        ok = bool(stf and stf.type in {"triangle", "box car"} and finite(stf.duration)
                  and stf.duration > 0 and finite(mt.scalar_moment) and mt.scalar_moment > 0
                  and centroid and centroid.time is not None)
        return mt, stf, centroid, ok

    selected_catalogue = Catalog()
    audit_rows = []
    for event in raw_catalogue:
        hypo = find_origin(event, "hypocenter")
        loc = find_origin(event, LOCATION_ORIGIN)
        mag = moment_magnitude(event)
        fm = focal_mechanism(event)
        style = mechanism_class(fm)
        mt, stf, centroid, stf_ok = stf_metadata(event, fm)
        reasons = []
        if hypo is None or hypo.time is None:
            reasons.append("missing_hypocentral_time")
        elif not start <= hypo.time < end:
            reasons.append("outside_time")
        if loc is None or not finite(loc.latitude) or not finite(loc.longitude):
            reasons.append("missing_location")
        elif not (LAT_MIN <= loc.latitude <= LAT_MAX and longitude_inside(loc.longitude)):
            reasons.append("outside_box")
        if mag is None:
            reasons.append("missing_Mw")
        elif not MAG_MIN <= mag.mag <= MAG_MAX:
            reasons.append("outside_magnitude")
        if style not in allowed_mechanisms:
            reasons.append("mechanism_" + style)
        depth = loc.depth / 1000 if loc and finite(loc.depth) else None
        if DEPTH_MIN_KM is not None or DEPTH_MAX_KM is not None:
            if depth is None:
                reasons.append("missing_depth")
            elif (DEPTH_MIN_KM is not None and depth < DEPTH_MIN_KM) or (DEPTH_MAX_KM is not None and depth > DEPTH_MAX_KM):
                reasons.append("outside_depth")
        if REQUIRE_PLOTTABLE_STF and not stf_ok:
            reasons.append("missing_or_unsupported_STF")
        row = dict(event_id=str(event.resource_id), selected=not reasons, rejection_reasons=";".join(reasons),
                   hypocentral_time=str(hypo.time) if hypo and hypo.time is not None else None,
                   centroid_time=str(centroid.time) if centroid and centroid.time is not None else None,
                   latitude=loc.latitude if loc else None, longitude=loc.longitude if loc else None,
                   depth_km=depth, magnitude=mag.mag if mag else None,
                   magnitude_type=mag.magnitude_type if mag else None, mechanism=style,
                   stf_type=stf.type if stf else None, stf_full_duration_s=stf.duration if stf else None,
                   scalar_moment_Nm=mt.scalar_moment if mt else None, stf_plottable=stf_ok,
                   stf_figure=None)
        audit_rows.append(row)
        if not reasons:
            saved = event.copy()
            saved.preferred_origin_id = find_origin(saved, "hypocenter").resource_id
            saved.comments.append(Comment(text=f"Selection {run_id}: {style}; axis plunge threshold {AXIS_PLUNGE_THRESHOLD_DEG} deg; preferred origin set to hypocenter for waveform timing. STF is catalogue assumed model."))
            selected_catalogue.append(saved)
    selected_catalogue.events.sort(key=lambda ev: ev.preferred_origin().time)
    audit = pd.DataFrame(audit_rows)
    print(f"Selected {len(selected_catalogue)} / {len(raw_catalogue)} events.")
    display(audit.loc[audit.selected].drop(columns=["rejection_reasons", "selected"]))
    print("Rejection reasons (an event may have more than one):")
    display(audit.loc[~audit.selected, "rejection_reasons"].str.split(";").explode().value_counts())


    def stf_curve(event):
        mt, stf, centroid, ok = stf_metadata(event, focal_mechanism(event))
        if not ok:
            return None
        h = float(stf.duration) / 2
        center = float(centroid.time - find_origin(event, "hypocenter").time)
        if stf.type == "triangle":
            relative = np.linspace(-1.2*h, 1.2*h, 1201)
            rate = mt.scalar_moment / h * np.maximum(1 - np.abs(relative)/h, 0)
        else:
            relative = np.array([-1.2*h, -h, -h, h, h, 1.2*h])
            rate = np.array([0, 0, 1, 1, 0, 0]) * mt.scalar_moment / (2*h)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        assert np.isclose(integrate(rate, relative), mt.scalar_moment, rtol=1e-5), "STF integral differs from scalar moment."
        return relative + center, rate, center

    def draw_beachball(ax, event):
        """GCMT deviatoric tensor in Up/South/East order; DC fallback if necessary."""
        fm = focal_mechanism(event)
        mt = fm.moment_tensor if fm else None
        tensor = mt.tensor if mt else None
        values = [getattr(tensor, k, None) for k in
                  ("m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp")]
        planes = fm.nodal_planes if fm else None
        np1 = planes.nodal_plane_1 if planes else None
        if all(finite(v) for v in values):
            values = np.asarray(values, dtype=float)
            values[:3] -= np.mean(values[:3])
            scale = np.max(np.abs(values))
            if scale > 0:
                mechanism = values / scale  # shape only; no change to catalogue tensor
                label = "GCMT deviatoric tensor"
            else:
                mechanism = None
        else:
            mechanism = None
        if mechanism is None and np1 and all(finite(getattr(np1, k)) for k in ("strike", "dip", "rake")):
            mechanism = [np1.strike, np1.dip, np1.rake]
            label = "Best double-couple (fallback)"
        ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.axis("off")
        if mechanism is None:
            ax.text(0, 0, "Focal mechanism unavailable", ha="center", va="center", fontsize=9)
            label = "No beachball metadata"
        else:
            ax.add_collection(beach(mechanism, xy=(0, 0), width=2, size=200,
                                    facecolor="#176B87", bgcolor="white",
                                    edgecolor="#203040", linewidth=1.2))
            for x, y, direction in [(0, 1.17, "N"), (1.17, 0, "E"), (0, -1.17, "S"), (-1.17, 0, "W")]:
                ax.text(x, y, direction, ha="center", va="center", fontsize=10)
        ax.set_title("Focal mechanism", fontsize=12, pad=12)
        ax.text(.5, -.08, label + "\nLower-hemisphere projection", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#465360")
        return label

    fig_dir.mkdir(parents=True, exist_ok=True)
    # Existing stf_figure column now points to the combined beachball + STF image.
    audit["beachball_model"] = None
    for index, event in enumerate(selected_catalogue):
        curve = stf_curve(event)
        event_id = str(event.resource_id)
        match = audit.event_id == event_id
        row = audit.loc[match].iloc[0]
        short_id = re.sub(r"[^A-Za-z0-9_-]", "_", event_id.rstrip("/").split("/")[-2] if event_id.endswith("/event") else event_id.rstrip("/").split("/")[-1])
        suffix = hashlib.sha256(event_id.encode()).hexdigest()[:8]
        out = fig_dir / f"{short_id}_{suffix}_beachball_stf.png"
        fig, (ax_beach, ax) = plt.subplots(1, 2, figsize=(11, 4.8),
                                         gridspec_kw={"width_ratios": [1, 2.5]})
        fig.subplots_adjust(left=.04, right=.98, top=.76, bottom=.22, wspace=.32)
        model = draw_beachball(ax_beach, event)
        fig.suptitle(f"{short_id} | {row.magnitude_type} {row.magnitude:.2f} | {row.mechanism}",
                     fontsize=14, y=.97)
        fig.text(.5, .88, f"Hypocentral origin: {row.hypocentral_time}", ha="center", fontsize=10, color="#465360")
        if curve is not None:
            t, rate, center = curve
            ax.plot(t, rate, color="#176B87", lw=2)
            ax.fill_between(t, rate, color="#64CCC5", alpha=.35)
            ax.axvline(center, color="gray", ls="--", lw=1, label="Centroid time")
            ax.set(xlabel="Time relative to hypocentral origin (s)", ylabel="Moment rate (N m / s)",
                   title=f"Assumed {row.stf_type} STF | full duration {row.stf_full_duration_s:g} s")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(.5, .5, "Supported source-time function unavailable", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title("Source-time function")
            ax.set_xticks([]); ax.set_yticks([])
        fig.text(.5, .035, "STF: assumed GCMT inversion model, not an independently recovered rupture history.",
                 ha="center", fontsize=9, color="#465360")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        if index < PREVIEW_COUNT:
            display(fig)
        plt.close(fig)
        audit.loc[match, "stf_figure"] = str(out)
        audit.loc[match, "beachball_model"] = model
    print(f"Saved {audit.loc[audit.selected, 'stf_figure'].notna().sum()} combined beachball + STF figures to {fig_dir}.")


    run_dir.mkdir(parents=True, exist_ok=True)
    catalogue_path = run_dir / "selected_events.xml"
    selected_catalogue.comments.append(Comment(text="GCMT selection settings: " + json.dumps(config, sort_keys=True)))
    selected_catalogue.write(str(catalogue_path), format="QUAKEML", validate=True)
    reloaded = read_events(str(catalogue_path), format="QUAKEML")
    assert len(reloaded) == len(selected_catalogue)
    assert [str(e.resource_id) for e in reloaded] == [str(e.resource_id) for e in selected_catalogue]
    for original, restored in zip(selected_catalogue, reloaded):
        assert restored.preferred_origin().origin_type == "hypocenter"
        assert restored.preferred_origin().time == original.preferred_origin().time
        assert len(restored.origins) == len(original.origins)
        assert mechanism_class(focal_mechanism(restored)) == mechanism_class(focal_mechanism(original))
        a = stf_metadata(original, focal_mechanism(original))
        b = stf_metadata(restored, focal_mechanism(restored))
        if a[3]:
            assert b[3] and a[1].duration == b[1].duration and a[0].scalar_moment == b[0].scalar_moment

    audit.to_csv(run_dir / "selection_audit.csv", index=False)
    audit.loc[audit.selected].to_csv(run_dir / "selected_events.csv", index=False)
    manifest = dict(run_id=run_id, generated_utc=str(UTCDateTime()), config=config,
                    source_files=source_records, loaded_count=len(raw_catalogue), selected_count=len(reloaded),
                    catalogue_file=str(catalogue_path), figures_directory=str(fig_dir),
                    catalogue_sha256=hashlib.sha256(catalogue_path.read_bytes()).hexdigest(),
                    versions=dict(python=platform.python_version(), obspy=obspy.__version__,
                                  numpy=np.__version__, pandas=pd.__version__),
                    stf_note="Assumed GCMT inversion models, not independently recovered source-time functions.")
    (run_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("QuakeML saved and read-back verified:", catalogue_path.resolve())
    print("Selected-event summary:", (run_dir / "selected_events.csv").resolve())
    print("All-event audit:", (run_dir / "selection_audit.csv").resolve())
    print("Reproducibility manifest:", (run_dir / "selection_manifest.json").resolve())


    return {"run_dir": run_dir, "catalogue_file": catalogue_path}
