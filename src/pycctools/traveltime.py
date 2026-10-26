"""TauP predictions for arbitrary phases, defaulting to PREM.

The curve is a difference of earliest arrivals at the same epicentral distance
and source depth. It is a theoretical reference, not an automatic phase picker
or a matched-ray-parameter correlation-phase calculation.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from obspy.taup import TauPyModel


@lru_cache(maxsize=8)
def _model(name):
    return TauPyModel(model=name)


@dataclass(frozen=True)
class Arrival:
    phase: str
    time_sec: float
    ray_parameter_sec_degree: float


def travel_time(distance_deg, depth_km=0.0, phase="PKIKP", model="prem"):
    """Return the earliest arrival; absent physical paths have NaN values.

    Invalid phase names/model names raise TauP errors instead of being silently
    converted to missing arrivals. Distances are in [0, 180], depths in km.
    """
    if not np.isfinite(distance_deg) or not 0 <= distance_deg <= 180:
        raise ValueError("distance_deg must be finite and within [0, 180].")
    if not np.isfinite(depth_km) or depth_km < 0:
        raise ValueError("depth_km must be finite and nonnegative.")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("phase must be a nonempty TauP phase name.")
    arrivals = _model(str(model)).get_travel_times(
        source_depth_in_km=float(depth_km), distance_in_degree=float(distance_deg),
        phase_list=[phase],
    )
    if not arrivals:
        return Arrival(phase, float("nan"), float("nan"))
    first = min(arrivals, key=lambda arrival: arrival.time)
    return Arrival(phase, float(first.time), float(first.ray_param_sec_degree))


@dataclass(frozen=True)
class PhasePairCurve:
    distance_deg: np.ndarray
    time_a_sec: np.ndarray
    time_b_sec: np.ndarray
    phase_a: str
    phase_b: str
    depth_km: float
    model: str

    @property
    def lag_sec(self):
        """Signed lag: t(phase_b) - t(phase_a)."""
        return self.time_b_sec - self.time_a_sec

    def save(self, path):
        """Save plot-compatible distance/lag columns and JSON metadata in header."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(model=self.model, phase_a=self.phase_a, phase_b=self.phase_b,
                        depth_km=self.depth_km, convention="t_b - t_a",
                        arrival_selection="earliest at same distance and depth")
        np.savetxt(path, np.column_stack((self.distance_deg, self.lag_sec)),
                   header=json.dumps(metadata) + "\ndistance_deg lag_sec", fmt="%.8f")
        return path


def phase_pair_curve(distances, depth_km=0.0, phase_a="PKIKP", phase_b="PKJKP", model="prem"):
    """Compute directly on the requested grid; preserve gaps with NaN.

    No interpolation is used across absent branches or shadow zones. TauP can
    return multiple branches; this convenience API selects the earliest one.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("distances must be a nonempty one-dimensional sequence.")
    a = [travel_time(d, depth_km, phase_a, model).time_sec for d in distances]
    b = [travel_time(d, depth_km, phase_b, model).time_sec for d in distances]
    return PhasePairCurve(distances.copy(), np.asarray(a), np.asarray(b),
                          phase_a, phase_b, float(depth_km), str(model))
