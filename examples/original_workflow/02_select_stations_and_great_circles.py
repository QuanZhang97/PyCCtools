#!/usr/bin/env python
# coding: utf-8

# # 02 · Station coverage and full source–station great circles
# 
# Read the **selected QuakeML catalogue from notebook 01**, search **BHZ/HHZ channel metadata**, bin stations around a fixed centre, and export **one `stations.xml` per event** with global maps.
# 
# ## Definitions
# - Default centre: **130°E, 0°N**. Azimuth is spherical forward bearing **clockwise from north**, normalized to `[0, 360)`.
# - Pair bin `i` combines `[a, a+Δa)` and `[a+180, a+180+Δa)`. Generate primary sectors only within `[0, 180)` to avoid duplicates. The default 10° increment yields 18 pairs, ending at `(170–180°, 350–360°)`.
# - Radius means **angular distance in degrees from the fixed centre**, not from the individual event. Radius bounds and radial-band increment are configurable.
# - “Best” is a **geometric coverage ranking**: maximize the less-populated side of each opposite pair, then total distinct stations. This is not a waveform-quality, noise, or signal-to-noise ranking. All qualifying stations are retained by default; no arbitrary station cap is imposed.
# - Each full great circle is calculated from the **actual event and station coordinates**, not from the fixed binning centre. It includes both the minor and major arcs. These surface rings are geometric guides, not 3-D seismic ray paths.
# 
# **Availability limitation:** channel operating epochs must cover each requested event window. This does not prove continuous recorded waveforms. Coverage is limited to the configured archives; it is not a claim of exhaustive worldwide coverage. No waveform downloads occur here.

# In[2]:


# Install once in your notebook environment, then restart the kernel if needed.
# %pip install obspy numpy pandas matplotlib requests cartopy


# In[3]:


from pathlib import Path
from io import BytesIO
from copy import deepcopy
import hashlib
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import obspy
from obspy import UTCDateTime, read_events, read_inventory
from obspy.core.inventory import Inventory
from IPython.display import display
print("ObsPy", obspy.__version__, "| working directory:", Path.cwd())


# ## 1 · Parameters
# 
# Set `CATALOGUE_FILE` to your exported `selected_events.xml`. With `None`, the notebook proceeds only if exactly one `./catalogue/*/selected_events.xml` exists; it never silently chooses among multiple selections.
# 
# `WINDOW_START_OFFSET_S` and `WINDOW_END_OFFSET_S` are relative to each hypocentral origin. The initial 0–3 hour window is only an editable starting point for a later coda workflow. The overall station query spans the earliest window start through the latest window end, split into cached requests.
# 
# `KEEP_ALL_MATCHING_CHANNELS=True` retains eligible BHZ and HHZ alternatives and locations; station counts and rings count each `NET.STA` once. Station geometry uses the deterministic representative channel (BHZ preferred, then location priority). Channels at materially different coordinates are not silently mixed.
# 
# `REQUIRE_BOTH_SIDES=True` optionally excludes a bin/radius-band cell unless both opposite sides have stations. `MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN` optionally caps each side; station persistence across the catalogue is prioritized before proximity to the sector centre. Neither is enabled by default.

# In[4]:


CATALOGUE_FILE = './catalogue/09eacf4ce01d/selected_events.xml'  # e.g. Path("./catalogue/8597dbfb2f46/selected_events.xml")
CENTER_LON, CENTER_LAT = 130.0, 0.0
AZIMUTH_START_DEG, AZIMUTH_STOP_DEG = 0.0, 180.0
AZIMUTH_INCREMENT_DEG = 10.0
RADIUS_MIN_DEG, RADIUS_MAX_DEG = 0.0, 180.0
RADIAL_INCREMENT_DEG = 20.0
WINDOW_START_OFFSET_S = 0.0
WINDOW_END_OFFSET_S = 3 * 3600.0
CHANNELS = ("BHZ", "HHZ")  # order is also the representative-channel preference
LOCATION_PREFERENCE = ("00", "", "10", "01")
KEEP_ALL_MATCHING_CHANNELS = True
REQUIRE_BOTH_SIDES = False
MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN = None
COLOCATION_TOLERANCE_DEG = 0.01  # coordinate consistency within a station
MIN_SAMPLE_RATE_HZ = 1.0
MAX_VERTICAL_DIP_ERROR_DEG = 5.0  # accept either up/down vertical polarity

# First listed archive wins when it duplicates the same NSLC at the same event.
PROVIDERS = {
    "EARTHSCOPE": "https://service.earthscope.org/fdsnws/station/1/query",
    "GEOFON": "https://geofon.gfz.de/fdsnws/station/1/query",
}
NETWORK = "*"  # e.g. "IU,II" for a small initial test
STATION = "*"
METADATA_LEVEL = "channel"  # set "response" to include complete instrument responses
QUERY_CHUNK_DAYS = 366
REQUEST_TIMEOUT_S = 180
ALLOW_PARTIAL_PROVIDERS = False  # failures otherwise stop export; no silent omissions
REFRESH_CACHE = False
CACHE_DIR = Path("./data/station_cache")
OUTPUT_DIR = Path("./stations")
FIG_DIR = Path("./fig")
GREAT_CIRCLE_STEP_DEG = 1.0
LABEL_STATIONS = False  # useful for small selections
PREVIEW_EVENT_COUNT = 2
SAVE_PDF = False
MAP_CENTRAL_LONGITUDE = CENTER_LON


# In[5]:


if CATALOGUE_FILE is None:
    matches = sorted(Path("./catalogue").glob("*/selected_events.xml"))
    if len(matches) != 1:
        raise ValueError("Set CATALOGUE_FILE explicitly. Found: " + str([str(p) for p in matches]))
    CATALOGUE_FILE = matches[0]
CATALOGUE_FILE = Path(CATALOGUE_FILE).expanduser()
catalogue = read_events(str(CATALOGUE_FILE), format="QUAKEML")
if not catalogue:
    raise ValueError("The selected catalogue is empty; there are no event windows to search.")
assert -90 < CENTER_LAT < 90 and -180 <= CENTER_LON <= 180
assert 0 <= AZIMUTH_START_DEG < AZIMUTH_STOP_DEG <= 180
assert 0 < AZIMUTH_INCREMENT_DEG <= 180
assert 0 <= RADIUS_MIN_DEG < RADIUS_MAX_DEG <= 180
assert RADIAL_INCREMENT_DEG > 0 and QUERY_CHUNK_DAYS > 0
assert WINDOW_START_OFFSET_S < WINDOW_END_OFFSET_S
assert 0 < GREAT_CIRCLE_STEP_DEG <= 10
assert METADATA_LEVEL in {"channel", "response"}
assert CHANNELS and set(CHANNELS) <= {"BHZ", "HHZ"}
assert PROVIDERS and 0 <= MAX_VERTICAL_DIP_ERROR_DEG <= 90
assert MIN_SAMPLE_RATE_HZ > 0 and COLOCATION_TOLERANCE_DEG >= 0
assert MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN is None or (isinstance(MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN, int) and MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN > 0)
assert PREVIEW_EVENT_COUNT >= 0

def make_edges(start, stop, step):
    edges = np.arange(start, stop, step, dtype=float)
    return np.r_[edges[edges < stop - 1e-9], stop]

az_edges = make_edges(AZIMUTH_START_DEG, AZIMUTH_STOP_DEG, AZIMUTH_INCREMENT_DEG)
radial_edges = make_edges(RADIUS_MIN_DEG, RADIUS_MAX_DEG, RADIAL_INCREMENT_DEG)
N_AZ = len(az_edges) - 1
N_RADIAL = len(radial_edges) - 1

def origin_for(event):
    candidates = [o for o in event.origins if o.origin_type == "hypocenter"]
    origin = candidates[0] if candidates else event.preferred_origin()
    if origin is None or origin.time is None or origin.latitude is None or origin.longitude is None:
        raise ValueError("Missing timed, located origin: " + str(event.resource_id))
    if not np.isfinite([origin.latitude, origin.longitude]).all():
        raise ValueError("Non-finite source coordinates: " + str(event.resource_id))
    return origin

def event_slug(event):
    ident = str(event.resource_id)
    parts = ident.rstrip("/").split("/")
    label = parts[-2] if parts[-1] == "event" and len(parts) > 1 else parts[-1]
    return re.sub(r"[^A-Za-z0-9_-]", "_", label) + "_" + hashlib.sha256(ident.encode()).hexdigest()[:8]

events = sorted(catalogue, key=lambda e: origin_for(e).time)
assert len(set(str(e.resource_id) for e in events)) == len(events), "Duplicate event IDs."
query_start = min(origin_for(e).time + WINDOW_START_OFFSET_S for e in events)
query_end = max(origin_for(e).time + WINDOW_END_OFFSET_S for e in events)
settings = dict(catalogue_sha256=hashlib.sha256(CATALOGUE_FILE.read_bytes()).hexdigest(),
                center=[CENTER_LON, CENTER_LAT], az_edges=az_edges.tolist(), radial_edges=radial_edges.tolist(),
                window_offsets=[WINDOW_START_OFFSET_S, WINDOW_END_OFFSET_S], providers=list(PROVIDERS.items()),
                channels=list(CHANNELS), network=NETWORK, station=STATION, metadata_level=METADATA_LEVEL,
                keep_all_channels=KEEP_ALL_MATCHING_CHANNELS, both_sides=REQUIRE_BOTH_SIDES,
                cap=MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN, min_sample_rate=MIN_SAMPLE_RATE_HZ,
                dip_tolerance=MAX_VERTICAL_DIP_ERROR_DEG, colocation_tolerance=COLOCATION_TOLERANCE_DEG,
                location_preference=list(LOCATION_PREFERENCE), allow_partial=ALLOW_PARTIAL_PROVIDERS,
                circle_step=GREAT_CIRCLE_STEP_DEG, map_longitude=MAP_CENTRAL_LONGITUDE,
                label_stations=LABEL_STATIONS, save_pdf=SAVE_PDF)
print("Events:", len(events), "| station-query interval:", query_start, "to", query_end)
print("Paired azimuth sectors:", N_AZ, "| radial bands:", N_RADIAL)
print("Example:", tuple(az_edges[:2]), "and", tuple(az_edges[:2] + 180))


# ## 2 · Query and cache channel epochs
# 
# Requests use current HTTPS station-service URLs, standard FDSN parameters, and no retired `matchtimeseries` or `includeavailability` parameters. Each request is independently cached with its query and a SHA-256 checksum. A 204 response means no matching metadata; other errors are recorded and normally stop processing. To recover from an oversized query, reduce the geographic range, use a network subset, or shorten `QUERY_CHUNK_DAYS`.
# 
# **Channel-level StationXML is sufficient to identify waveform requests, but does not contain full response stages.** Set `METADATA_LEVEL="response"` if you want response-ready output, or fetch response metadata during waveform download. Querying response-level metadata for decades of global stations can be large.

# In[6]:


session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
              status_forcelist=[429, 500, 502, 503, 504])))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
query_log = []
channel_records = []
seen_epochs = set()

def epoch_key(net, sta, cha):
    return (net.code, sta.code, cha.location_code, cha.code,
            str(net.start_date), str(net.end_date), str(sta.start_date), str(sta.end_date),
            str(cha.start_date), str(cha.end_date), float(cha.latitude), float(cha.longitude),
            float(cha.elevation), float(cha.depth), str(cha.sample_rate), str(cha.dip))

for provider, endpoint in PROVIDERS.items():
    chunk_start = query_start
    while chunk_start < query_end:
        chunk_end = min(chunk_start + QUERY_CHUNK_DAYS * 86400, query_end)
        params = dict(network=NETWORK, station=STATION, location="*", channel=",".join(CHANNELS),
                      starttime=str(chunk_start), endtime=str(chunk_end), level=METADATA_LEVEL,
                      latitude=CENTER_LAT, longitude=CENTER_LON, minradius=RADIUS_MIN_DEG,
                      maxradius=RADIUS_MAX_DEG, includerestricted="false", format="xml", nodata=204)
        key = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
        cache = CACHE_DIR / (key + ".xml")
        empty = CACHE_DIR / (key + ".empty.json")
        entry = dict(provider=provider, endpoint=endpoint, parameters=params, cache=str(cache))
        try:
            if not REFRESH_CACHE and cache.exists():
                content = cache.read_bytes()
                entry["status"] = "cached"
            elif not REFRESH_CACHE and empty.exists():
                entry["status"] = "cached_no_metadata"
                query_log.append(entry)
                chunk_start = chunk_end
                continue
            else:
                print(provider, str(chunk_start), "→", str(chunk_end), flush=True)
                response = session.get(endpoint, params=params, timeout=(20, REQUEST_TIMEOUT_S))
                entry["resolved_url"] = response.url
                if response.status_code == 204:
                    if cache.exists(): cache.unlink()
                    empty.write_text(json.dumps(entry, indent=2), encoding="utf-8")
                    entry["status"] = "no_metadata"
                    query_log.append(entry)
                    chunk_start = chunk_end
                    continue
                response.raise_for_status()
                content = response.content
                read_inventory(BytesIO(content), format="STATIONXML")  # validate before caching
                cache.write_bytes(content)
                if empty.exists(): empty.unlink()
                entry["status"] = "downloaded"
            inventory = read_inventory(BytesIO(content), format="STATIONXML")
            entry["sha256"] = hashlib.sha256(content).hexdigest()
            entry["channel_epochs"] = 0
            for net in inventory:
                for sta in net:
                    for cha in sta:
                        if cha.code not in CHANNELS:
                            continue
                        ek = (provider,) + epoch_key(net, sta, cha)
                        if ek in seen_epochs:
                            continue
                        seen_epochs.add(ek)
                        channel_records.append(dict(provider=provider, net=net, sta=sta, cha=cha))
                        entry["channel_epochs"] += 1
        except Exception as exc:
            entry.update(status="failed", error=str(exc))
            query_log.append(entry)
            failure_path = CACHE_DIR / "last_query_failure.json"
            failure_path.write_text(json.dumps(query_log, indent=2), encoding="utf-8")
            if not ALLOW_PARTIAL_PROVIDERS:
                raise RuntimeError(f"Station search incomplete at {provider}. See {failure_path}. Cached successful requests will be reused.") from exc
            warnings.warn(f"Incomplete coverage: {provider}: {exc}")
            chunk_start = chunk_end
            continue
        query_log.append(entry)
        chunk_start = chunk_end

if not channel_records:
    raise RuntimeError("No BHZ/HHZ channel metadata found. Review query parameters and query_log.")
source_signature = [(x["provider"], x["parameters"]["starttime"], x["parameters"]["endtime"],
                     x.get("sha256"), x["status"] == "failed") for x in query_log]
run_id = hashlib.sha256(json.dumps([settings, source_signature], sort_keys=True).encode()).hexdigest()[:12]
run_dir = OUTPUT_DIR / run_id
figure_dir = FIG_DIR / ("station_coverage_" + run_id)
run_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "query_log.json").write_text(json.dumps(query_log, indent=2), encoding="utf-8")
print("Unique channel epochs:", len(channel_records), "| run:", run_id)


# ## 3 · Geometry, active epochs, and unique station candidates
# 
# All bearings, angular distances and circles use a consistent **spherical Earth**. Azimuth sectors are left-inclusive/right-exclusive. The outermost radius includes its upper boundary. Bearings at coincident or exactly antipodal points are undefined and are rejected explicitly.
# 
# Each channel must cover the **entire event window**, considering channel, station and network epochs together. Adjacent channel epochs can jointly cover the window; all relevant epochs are preserved. Metadata with a known restricted status, insufficient sampling rate, or a nonvertical dip is excluded. Unknown access status is retained and reported. Coordinate changes during a window are excluded rather than combined into a fictitious stationary receiver.

# In[7]:


def spherical_geometry(lat0, lon0, lat1, lon1):
    p0, p1 = np.deg2rad([lat0, lat1])
    dl = np.deg2rad(lon1 - lon0)
    x = np.cos(p0)*np.sin(p1) - np.sin(p0)*np.cos(p1)*np.cos(dl)
    y = np.sin(dl)*np.cos(p1)
    cross_norm = np.hypot(x, y)
    dot = np.sin(p0)*np.sin(p1) + np.cos(p0)*np.cos(p1)*np.cos(dl)
    distance = float(np.rad2deg(np.arctan2(cross_norm, dot)))
    azimuth = float(np.rad2deg(np.arctan2(y, x)) % 360) if cross_norm > 1e-12 else np.nan
    return distance, azimuth

def classify_geometry(latitude, longitude):
    radius, azimuth = spherical_geometry(CENTER_LAT, CENTER_LON, latitude, longitude)
    if not np.isfinite(azimuth): return None
    primary = azimuth % 180
    if not AZIMUTH_START_DEG <= primary < AZIMUTH_STOP_DEG: return None
    if not RADIUS_MIN_DEG <= radius <= RADIUS_MAX_DEG: return None
    pair = int(np.searchsorted(az_edges, primary, side="right") - 1)
    radial = min(int(np.searchsorted(radial_edges, radius, side="right") - 1), N_RADIAL - 1)
    return dict(center_distance_deg=radius, center_azimuth_deg=azimuth,
                pair_bin=pair, side=int(azimuth >= 180), radial_bin=radial)

def interval(record, start, end):
    nodes = [record["net"], record["sta"], record["cha"]]
    lower = max([start] + [n.start_date for n in nodes if n.start_date is not None])
    upper = min([end] + [n.end_date for n in nodes if n.end_date is not None])
    return (lower, upper) if upper > lower else None

def covers_window(records, start, end):
    spans = sorted([v for r in records for v in [interval(r, start, end)] if v is not None])
    cursor = start
    for lower, upper in spans:
        if lower > cursor + 1e-6: return False
        cursor = max(cursor, upper)
        if cursor >= end: return True
    return False

def nslc(record):
    return (record["net"].code, record["sta"].code,
            record["cha"].location_code, record["cha"].code)

def finite(value):
    return value is not None and np.isfinite(value)

def channel_preference(candidate):
    loc = candidate["location"]
    return (CHANNELS.index(candidate["channel"]),
            LOCATION_PREFERENCE.index(loc) if loc in LOCATION_PREFERENCE else len(LOCATION_PREFERENCE),
            loc, list(PROVIDERS).index(candidate["provider"]))

groups = {}
for record in channel_records:
    groups.setdefault((record["provider"],) + nslc(record), []).append(record)

candidate_rows = []
candidates_by_event = {}
station_records_by_event = {}
for event in events:
    origin = origin_for(event)
    slug = event_slug(event)
    start, end = origin.time + WINDOW_START_OFFSET_S, origin.time + WINDOW_END_OFFSET_S
    stream_candidates = {}
    for (provider, net, sta, loc, channel), records in groups.items():
        active = [r for r in records if interval(r, start, end) is not None]
        if not active: continue
        reason = None
        usable = []
        for r in active:
            cha = r["cha"]
            if any(str(node.restricted_status).lower() in {"closed", "partial"} for node in [r["net"], r["sta"], cha]): continue
            if not finite(cha.sample_rate) or cha.sample_rate < MIN_SAMPLE_RATE_HZ: continue
            if not finite(cha.dip) or abs(abs(cha.dip) - 90) > MAX_VERTICAL_DIP_ERROR_DEG: continue
            if not finite(cha.latitude) or not finite(cha.longitude): continue
            usable.append(r)
        if not covers_window(usable, start, end):
            reason = "no_full_window_of_usable_metadata"
        reference = usable[0]["cha"] if usable else active[0]["cha"]
        if reason is None and any(spherical_geometry(reference.latitude, reference.longitude,
                     r["cha"].latitude, r["cha"].longitude)[0] > COLOCATION_TOLERANCE_DEG for r in usable):
            reason = "coordinates_change_within_window"
        geometry = classify_geometry(reference.latitude, reference.longitude) if finite(reference.latitude) and finite(reference.longitude) else None
        if reason is None and geometry is None: reason = "outside_geometry_or_undefined_bearing"
        row = dict(event=slug, provider=provider, network=net, station=sta, location=loc, channel=channel,
                   station_id=net + "." + sta, latitude=float(reference.latitude), longitude=float(reference.longitude),
                   window_start=str(start), window_end=str(end), reason=reason or "eligible",
                   waveform_availability="not_verified", access_status=str(reference.restricted_status))
        if geometry: row.update(geometry)
        candidate_rows.append(row)
        if reason: continue
        key = (net, sta, loc, channel)
        if key in stream_candidates:
            row["reason"] = "duplicate_archive_lower_priority"
            continue
        stream_candidates[key] = dict(row, records=usable)
    by_station = {}
    for candidate in stream_candidates.values():
        by_station.setdefault(candidate["station_id"], []).append(candidate)
    station_rows, channel_lookup = [], {}
    for station_id, streams in by_station.items():
        streams.sort(key=channel_preference)
        representative = streams[0]
        # Keep alternative channels only if they share the representative geometry cell and site.
        compatible = [c for c in streams if
            (c["pair_bin"], c["side"], c["radial_bin"]) == (representative["pair_bin"], representative["side"], representative["radial_bin"])
            and spherical_geometry(representative["latitude"], representative["longitude"], c["latitude"], c["longitude"])[0] <= COLOCATION_TOLERANCE_DEG]
        kept_streams = compatible if KEEP_ALL_MATCHING_CHANNELS else [representative]
        channel_lookup[station_id] = [r for c in kept_streams for r in c["records"]]
        row = {k:v for k,v in representative.items() if k != "records"}
        row["channels"] = ";".join(".".join([c["network"], c["station"], c["location"], c["channel"]]) for c in kept_streams)
        row["event_distance_deg"], row["event_azimuth_deg"] = spherical_geometry(origin.latitude, origin.longitude, row["latitude"], row["longitude"])
        station_rows.append(row)
    candidates_by_event[slug] = station_rows
    station_records_by_event[slug] = channel_lookup

persistence = {}
for rows in candidates_by_event.values():
    for row in rows: persistence[row["station_id"]] = persistence.get(row["station_id"], 0) + 1
pd.DataFrame(candidate_rows).to_csv(run_dir / "channel_candidate_audit.csv", index=False)
print("Distinct stations eligible in at least one event:", len(persistence))


# ## 4 · Rank coverage and export event-specific StationXML
# 
# Each event directory contains:
# - `stations.xml`: selected channel epochs, preserving the original metadata hierarchy.
# - `stations.csv`: one row per selected station with centre and event geometry.
# - `channels.csv`: exact selected channel epochs and their metadata provider; use this to route later requests.
# - `coverage.csv`: both side counts, the smaller count, and the total in every azimuth/radius cell, before and after optional thinning.
# - `waveform_requests.csv`: unique NSLC requests and event-window limits for the next notebook.
# 
# The paired count is the **minimum of the two side counts**, not proof of exact station antipodality or a constructed pairing. Stations inside the opposite sectors need not be 180° apart on the globe. Any cross-correlation pairing requirements remain a separate step.
# 
# **Empty selections:** every event gets `status.json`. An event with no qualifying stations gets header-only tables and an empty-coverage map, but no invalid StationXML file. Its `stationxml` entry in the summary is blank, so the downloader can skip it explicitly.

# In[8]:


def coverage_table(rows):
    output = []
    for pair in range(N_AZ):
        for radial in range(N_RADIAL):
            counts = [sum(r["pair_bin"] == pair and r["radial_bin"] == radial and r["side"] == side for r in rows) for side in (0, 1)]
            output.append(dict(pair_bin=pair, radial_bin=radial,
                azimuth_start=az_edges[pair], azimuth_end=az_edges[pair+1],
                opposite_start=az_edges[pair]+180, opposite_end=az_edges[pair+1]+180,
                radius_start=radial_edges[radial], radius_end=radial_edges[radial+1],
                side_0=counts[0], side_1=counts[1], balanced_count=min(counts), total=sum(counts)))
    return pd.DataFrame(output)

def make_inventory(records):
    # Exact original epoch objects: no select() calls that accidentally broaden dates/channels.
    nets, stations, seen = {}, {}, set()
    for r in records:
        net, sta, cha = r["net"], r["sta"], r["cha"]
        key = epoch_key(net, sta, cha)
        if key in seen: continue
        seen.add(key)
        nk = (net.code, str(net.start_date), str(net.end_date))
        sk = (nk, sta.code, str(sta.start_date), str(sta.end_date), float(sta.latitude), float(sta.longitude))
        if nk not in nets:
            nets[nk] = deepcopy(net); nets[nk].stations = []
            nets[nk].selected_number_of_stations = None
        if sk not in stations:
            stations[sk] = deepcopy(sta); stations[sk].channels = []
            stations[sk].selected_number_of_channels = None
            nets[nk].stations.append(stations[sk])
        channel_copy = deepcopy(cha)
        # ObsPy 1.3 validators reject incomplete response values such as
        # Value=None or Frequency=None. Channel metadata are sufficient
        # for waveform requests, so omit incomplete response objects.
        if channel_copy.response is not None:
            response_text = str(channel_copy.response)
            if "None" in response_text or METADATA_LEVEL != "response":
                channel_copy.response = None
        if channel_copy.azimuth is not None and channel_copy.azimuth >= 360:
            channel_copy.azimuth = channel_copy.azimuth % 360.0
        stations[sk].channels.append(channel_copy)
    return Inventory(networks=list(nets.values()), source="Event station selection from configured FDSN archives")

selected_by_event = {}
coverage_by_event = {}
event_summary = []
for event in events:
    slug = event_slug(event)
    rows = candidates_by_event[slug]
    candidate_coverage = coverage_table(rows)
    allowed = {(int(r.pair_bin), int(r.radial_bin)) for r in candidate_coverage.itertuples() if r.balanced_count > 0}
    kept = [dict(r) for r in rows if not REQUIRE_BOTH_SIDES or (r["pair_bin"], r["radial_bin"]) in allowed]
    for r in kept:
        r["eligible_event_count"] = persistence[r["station_id"]]
        midpoint = (az_edges[r["pair_bin"]] + az_edges[r["pair_bin"]+1])/2
        r["sector_center_offset_deg"] = abs(r["center_azimuth_deg"] % 180 - midpoint)
    kept.sort(key=lambda r: (-r["eligible_event_count"], r["sector_center_offset_deg"], r["station_id"]))
    if MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN is not None:
        selected, counts = [], {}
        for r in kept:
            key = (r["pair_bin"], r["radial_bin"], r["side"])
            if counts.get(key, 0) < MAX_STATIONS_PER_SIDE_PER_RADIAL_BIN:
                selected.append(r); counts[key] = counts.get(key, 0) + 1
        kept = selected
    selected_by_event[slug] = kept
    coverage = coverage_table(kept)
    coverage["candidate_side_0"] = candidate_coverage.side_0
    coverage["candidate_side_1"] = candidate_coverage.side_1
    coverage["rank"] = 0
    order = coverage.sort_values(["balanced_count", "total", "pair_bin", "radial_bin"], ascending=[False, False, True, True]).index
    coverage.loc[order, "rank"] = np.arange(1, len(order)+1)
    coverage_by_event[slug] = coverage
    folder = run_dir / slug
    folder.mkdir(parents=True, exist_ok=True)
    records = [r for station in kept for r in station_records_by_event[slug][station["station_id"]]]
    inv = make_inventory(records)
    xml_path = folder / "stations.xml"
    if inv.networks:
        inv.write(str(xml_path), format="STATIONXML", validate=True)
        check = read_inventory(str(xml_path))
        original_keys = {epoch_key(n,s,c) for n in inv for s in n for c in s}
        restored_keys = {epoch_key(n,s,c) for n in check for s in n for c in s}
        assert original_keys == restored_keys, "StationXML round-trip mismatch."
    elif xml_path.exists():
        xml_path.unlink()  # remove a stale generated file for this exact run/event
    (folder / "status.json").write_text(json.dumps(dict(
        event=slug, status="selected" if kept else "no_qualifying_stations",
        station_count=len(kept), stationxml=str(xml_path) if kept else None,
        note="No StationXML is written for an empty selection because it would fail schema validation."
             if not kept else "Waveform availability is not verified."), indent=2), encoding="utf-8")
    pd.DataFrame(kept, columns=list(kept[0]) if kept else ["event", "station_id", "latitude", "longitude", "pair_bin", "side", "radial_bin"]).to_csv(folder / "stations.csv", index=False)
    channel_rows = []
    unique = set()
    for r in records:
        key = (r["provider"],) + epoch_key(r["net"],r["sta"],r["cha"])
        if key in unique: continue
        unique.add(key)
        net,sta,loc,cha = nslc(r)
        channel_rows.append(dict(provider=r["provider"], network=net, station=sta, location=loc, channel=cha,
             epoch_start=str(r["cha"].start_date), epoch_end=str(r["cha"].end_date),
             latitude=float(r["cha"].latitude), longitude=float(r["cha"].longitude),
             sample_rate=float(r["cha"].sample_rate), metadata_level=METADATA_LEVEL))
    channel_frame = pd.DataFrame(channel_rows, columns=["provider","network","station","location","channel","epoch_start","epoch_end","latitude","longitude","sample_rate","metadata_level"])
    channel_frame.to_csv(folder / "channels.csv", index=False)
    requests_frame = channel_frame[["provider","network","station","location","channel"]].drop_duplicates().copy()
    origin = origin_for(event)
    requests_frame["starttime"] = str(origin.time + WINDOW_START_OFFSET_S)
    requests_frame["endtime"] = str(origin.time + WINDOW_END_OFFSET_S)
    requests_frame.to_csv(folder / "waveform_requests.csv", index=False)
    coverage.to_csv(folder / "coverage.csv", index=False)
    event_summary.append(dict(event=slug, event_id=str(event.resource_id), origin_time=str(origin.time),
         origin_type=str(origin.origin_type), station_count=len(kept), channel_count=len(requests_frame),
         occupied_pair_bins=int(coverage.loc[coverage.total>0, "pair_bin"].nunique()),
         cells_with_both_sides=int((coverage.balanced_count>0).sum()),
         stationxml=str(folder / "stations.xml") if kept else None, waveform_availability="not_verified"))
summary = pd.DataFrame(event_summary)
summary.to_csv(run_dir / "event_summary.csv", index=False)
print("Event-specific StationXML files saved and read-back verified.")
display(summary[["event", "station_count", "channel_count", "occupied_pair_bins", "cells_with_both_sides"]])


# ## 5 · Full-ring global maps and paired-sector coverage
# 
# The map shows one complete great-circle ring per selected station. The station markers and rings share the colour of the **fixed-centre azimuth pair**. A star marks the actual source; a black cross marks the binning centre. Dense maps remain dense by design: no rings are silently omitted.
# 
# A separate coverage figure shows station counts for each side versus azimuth pair and radius. Natural Earth coastlines are downloaded by Cartopy on first use and then cached; the first map requires access to its data server. All images go into `./fig/station_coverage_<run_id>/`.

# In[ ]:


def xyz(latitude, longitude):
    lat, lon = np.deg2rad([latitude, longitude])
    return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def full_great_circle(source_lat, source_lon, station_lat, station_lon):
    a, b = xyz(source_lat, source_lon), xyz(station_lat, station_lon)
    tangent = b - np.dot(a, b)*a
    length = np.linalg.norm(tangent)
    if length < 1e-10:
        return None  # coincident/antipodal points do not determine a unique plane
    tangent /= length
    theta = np.deg2rad(np.linspace(0, 360, int(np.ceil(360/GREAT_CIRCLE_STEP_DEG))+1))
    points = np.cos(theta)[:,None]*a + np.sin(theta)[:,None]*tangent
    lon = np.rad2deg(np.arctan2(points[:,1], points[:,0]))
    lat = np.rad2deg(np.arcsin(np.clip(points[:,2], -1, 1)))
    return lon, lat

def ring_segments(lon, lat):
    # Split at the map seam before drawing; avoid spurious horizontal connectors.
    relative = (lon - MAP_CENTRAL_LONGITUDE + 180) % 360 - 180
    cuts = np.flatnonzero(np.abs(np.diff(relative)) > 180) + 1
    return [np.column_stack((x, y)) for x,y in zip(np.split(relative,cuts), np.split(lat,cuts)) if len(x)>1]

pair_cmap = plt.get_cmap("BrBG", N_AZ)
pair_norm = BoundaryNorm(np.arange(N_AZ+1)-.5, N_AZ)
plot_audit = []
for index, event in enumerate(events):
    origin, slug = origin_for(event), event_slug(event)
    rows = selected_by_event[slug]
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=MAP_CENTRAL_LONGITUDE))
    ax.set_global()
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f1efe8", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#eaf2f7", zorder=0)
    ax.coastlines(resolution="110m", linewidth=.5, color="#687078", zorder=1)
    ax.gridlines(linewidth=.4, color="gray", alpha=.4, linestyle=":")
    segments, colors, degenerate = [], [], []
    for row in rows:
        ring = full_great_circle(origin.latitude, origin.longitude, row["latitude"], row["longitude"])
        if ring is None:
            degenerate.append(row["station_id"])
            continue
        parts = ring_segments(*ring)
        segments.extend(parts)
        colors.extend([pair_cmap(row["pair_bin"])]*len(parts))
    # Relative longitudes match a PlateCarree frame centred on the map's meridian.
    if segments:
        collection = LineCollection(segments, colors=colors, linewidths=.55,
                                    alpha=.45, transform=ccrs.PlateCarree(central_longitude=MAP_CENTRAL_LONGITUDE))
        ax.add_collection(collection)
    if rows:
        ax.scatter([r["longitude"] for r in rows], [r["latitude"] for r in rows],
                   c=[r["pair_bin"] for r in rows], cmap=pair_cmap, norm=pair_norm, marker="^", s=24,
                   edgecolors="#26333d", linewidths=.3, transform=ccrs.PlateCarree(), zorder=4, label="Selected stations")
        if LABEL_STATIONS:
            for r in rows:
                ax.text(r["longitude"], r["latitude"], " " + r["station_id"], fontsize=6,
                        transform=ccrs.PlateCarree(), zorder=6)
    ax.scatter([CENTER_LON], [CENTER_LAT], marker="x", c="black", s=65, linewidths=1.8,
               transform=ccrs.PlateCarree(), zorder=7, label="Binning centre")
    ax.scatter([origin.longitude], [origin.latitude], marker="*", c="#ed3b35", s=150,
               edgecolors="black", linewidths=.6, transform=ccrs.PlateCarree(), zorder=8, label="Event")
    ax.set_title(f"{slug} | {len(rows)} stations | full source–station great circles\n"
                 f"Origin: {origin.time} | binning centre {CENTER_LON:g}°E, {CENTER_LAT:g}°N", fontsize=12, pad=15)
    ax.legend(loc="lower left", fontsize=8)
    bar = fig.colorbar(ScalarMappable(norm=pair_norm, cmap=pair_cmap), ax=ax,
                      orientation="horizontal", fraction=.04, pad=.07, ticks=np.arange(N_AZ))
    bar.ax.set_xticklabels([f"{az_edges[i]:g}–{az_edges[i+1]:g}" for i in range(N_AZ)], rotation=45, ha="right", fontsize=7)
    bar.set_label("Azimuth sector (° clockwise from north); each colour also includes the sector +180°", fontsize=9)
    fig.text(.5, .01, f"Metadata epoch coverage only; recordings unverified. Undefined rings omitted: {len(degenerate)}.", ha="center", fontsize=9)
    path = figure_dir / (slug + "_great_circles.png")
    fig.savefig(str(path), dpi=180, bbox_inches="tight")
    if SAVE_PDF: fig.savefig(str(path.with_suffix(".pdf")), bbox_inches="tight")
    if index < PREVIEW_EVENT_COUNT: display(fig)
    plt.close(fig)
    plot_audit.append(dict(event=slug, map=str(path), undefined_rings=degenerate))

    cov = coverage_by_event[slug]
    maximum = max(1, int(max(cov.side_0.max(),cov.side_1.max())))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.subplots_adjust(left=.09, right=.86, bottom=.22, top=.82, wspace=.10)
    for side, ax in enumerate(axes):
        matrix = cov.pivot(index="radial_bin", columns="pair_bin", values="side_"+str(side)).reindex(index=range(N_RADIAL), columns=range(N_AZ)).values
        mesh = ax.pcolormesh(np.arange(N_AZ+1), np.arange(N_RADIAL+1), matrix, cmap="viridis", vmin=0, vmax=maximum)
        ax.set_xticks(np.arange(N_AZ)+.5)
        ax.set_xticklabels([f"{az_edges[i]+180*side:g}–{az_edges[i+1]+180*side:g}" for i in range(N_AZ)], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(np.arange(N_RADIAL)+.5)
        ax.set_yticklabels([f"{radial_edges[j]:g}–{radial_edges[j+1]:g}" for j in range(N_RADIAL)])
        ax.set_title("Primary sectors" if side==0 else "Opposite sectors")
        ax.set_xlabel("Azimuth from fixed centre (°)")
        if N_AZ*N_RADIAL <= 240:
            for j in range(N_RADIAL):
                for i in range(N_AZ):
                    ax.text(i+.5,j+.5,str(int(matrix[j,i])),ha="center",va="center",fontsize=7,
                            color="white" if matrix[j,i]<maximum*.55 else "black")
    axes[0].set_ylabel("Radius from fixed centre (°)")
    fig.colorbar(mesh, cax=fig.add_axes([.89,.25,.02,.5]), label="Distinct stations")
    fig.suptitle(slug + " | paired-sector station coverage", fontsize=13)
    fig.savefig(str(figure_dir / (slug + "_coverage.png")), dpi=160, bbox_inches="tight")
    if index < PREVIEW_EVENT_COUNT: display(fig)
    plt.close(fig)
print("Maps and coverage figures saved:", figure_dir.resolve())


# ## 6 · Catalogue-wide ranking and handoff
# 
# `coverage_ranking.csv` pools **event–station incidences**, not unique worldwide stations: the same station can contribute to multiple events. Ranking uses the summed within-event minimum of the opposite side counts, followed by the total incidences. `station_persistence.csv` reports how many event windows each station could cover before optional thinning.
# 
# For downloading, use each event's `waveform_requests.csv` together with `channels.csv` and `stations.xml`. Preserve location codes as strings (`00` is not zero); an empty location becomes `--` in an FDSN text request. Verify waveform completeness and obtain instrument responses before response removal.

# In[ ]:


all_coverage = pd.concat([v.assign(event=k) for k,v in coverage_by_event.items()], ignore_index=True)
ranking = all_coverage.groupby(["pair_bin","radial_bin","azimuth_start","azimuth_end","opposite_start","opposite_end","radius_start","radius_end"], as_index=False)[["side_0","side_1","balanced_count","total"]].sum()
ranking = ranking.rename(columns={"balanced_count":"sum_within_event_balanced_count", "total":"event_station_incidences"})
ranking = ranking.sort_values(["sum_within_event_balanced_count","event_station_incidences"], ascending=False)
ranking.to_csv(run_dir / "coverage_ranking.csv", index=False)
pd.DataFrame([dict(station_id=k,eligible_event_count=v,event_fraction=v/len(events)) for k,v in persistence.items()],
             columns=["station_id","eligible_event_count","event_fraction"]).to_csv(run_dir / "station_persistence.csv", index=False)
manifest = dict(run_id=run_id, created_utc=str(UTCDateTime()), configuration=settings,
                catalogue_file=str(CATALOGUE_FILE), event_count=len(events),
                query_start=str(query_start), query_end=str(query_end),
                partial_provider_coverage=any(q["status"]=="failed" for q in query_log),
                obspy_version=obspy.__version__, query_log="query_log.json", figures=plot_audit,
                waveform_availability="not_verified", metadata_level=METADATA_LEVEL)
(run_dir / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
print("Station outputs:",run_dir.resolve())
print("Figures:",figure_dir.resolve())
display(ranking.head(12))

# Example for the next notebook:
# requests_table = pd.read_csv("./stations/<run_id>/<event>/waveform_requests.csv",
#                             dtype=str, keep_default_na=False)
# inventory = read_inventory("./stations/<run_id>/<event>/stations.xml")
# Route each request to its provider's FDSN dataselect service and verify actual traces.


# ## References
# 
# - [EarthScope station service](https://service.earthscope.org/fdsnws/station/1/): supported parameters, channel epochs, metadata levels and current endpoint.
# - [EarthScope station-service migration](https://www.earthscope.org/news/the-fdsnws-station-web-service-is-moving/): retired availability parameters.
# - [GEOFON FDSN services](https://geofon.gfz.de/waveform/webservices/): alternative archive access.
# - [FDSN StationXML](https://www.fdsn.org/xml/station/): metadata format used for per-event station files.
# - [Cartopy documentation](https://cartopy.readthedocs.io/stable/): global map projections and Natural Earth features.
# 
# The binning, balanced-count ranking, epoch-union policy and optional station cap are explicit workflow choices, not an established optimality criterion for coda correlation.
