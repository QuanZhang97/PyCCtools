"""Shared event identity, geometry, coverage and binning utilities."""
import hashlib
import re
from itertools import islice
import numpy as np

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

def distance_bin_index(distance, edges):
    if not np.isfinite(distance) or distance < edges[0] or distance > edges[-1]:
        return None
    index = int(np.searchsorted(edges, distance, side="right") - 1)
    if index == len(edges) - 1:
        index -= 1
    if index < 0 or index >= len(edges) - 1:
        return None
    return index

def finite(value):
    return value is not None and np.isfinite(value)

def regular_edges(start, stop, width):
    if width <= 0:
        raise ValueError("STACK_DISTANCE_BIN_WIDTH_DEG must be positive.")
    if stop <= start:
        raise ValueError("The stack distance range must have stop > start.")
    count = int(np.ceil((stop - start) / float(width)))
    edges = start + np.arange(count + 1, dtype=float) * width
    edges[-1] = stop
    return edges


def located_origin(event):
    """Hypocentral timed origin, with valid geographic coordinates required."""
    origin = event_origin(event)
    if origin.latitude is None or origin.longitude is None:
        raise ValueError(f"Missing source coordinates: {event.resource_id}")
    if not np.isfinite([origin.latitude, origin.longitude]).all():
        raise ValueError(f"Non-finite source coordinates: {event.resource_id}")
    if not -90 <= origin.latitude <= 90 or not -180 <= origin.longitude <= 180:
        raise ValueError(f"Out-of-range source coordinates: {event.resource_id}")
    return origin


def bounded_map(executor, function, iterable, batch_size):
    """Limit pending results so long FFT spectra cannot accumulate unboundedly."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield from executor.map(function, batch)


def azimuth_bin(azimuth, edges):
    """Index an axial bearing using half-open sectors; opposite bearings share a bin."""
    if not np.isfinite(azimuth):
        return None
    axial = azimuth % 180.0
    if not edges[0] <= axial < edges[-1]:
        return None
    return int(np.searchsorted(edges, axial, side="right") - 1)
