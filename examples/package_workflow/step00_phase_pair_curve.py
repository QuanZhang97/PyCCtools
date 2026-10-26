"""Generate a theoretical curve for any TauP phase pair."""
import numpy as np

from pycctools import phase_pair_curve

from _workflow_state import print_outputs, update_state
from settings import PHASE_PAIR_SETTINGS


def run():
    settings = dict(PHASE_PAIR_SETTINGS)
    start = float(settings.pop("distance_min_deg"))
    stop = float(settings.pop("distance_max_deg"))
    step = float(settings.pop("distance_step_deg"))
    output = settings.pop("output_file")
    if not 0.0 <= start < stop <= 180.0 or step <= 0.0:
        raise ValueError("Require 0 <= distance_min < distance_max <= 180 and step > 0")

    distances = np.arange(start, stop, step, dtype=float)
    if not len(distances) or distances[-1] < stop:
        distances = np.r_[distances, stop]
    curve = phase_pair_curve(distances, **settings)
    path = curve.save(output).resolve()
    update_state(theoretical_arrival_file=path)
    return {
        "theoretical_arrival_file": path,
        "valid_points": int(np.count_nonzero(np.isfinite(curve.lag_sec))),
        "total_points": len(curve.distance_deg),
        "lag_convention": "t(phase_b) - t(phase_a)",
    }


def main():
    print_outputs(run())


if __name__ == "__main__":
    main()
