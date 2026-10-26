"""Command line entry points; online stages require explicit invocation."""
import argparse
import importlib
import json
from pathlib import Path
import numpy as np
from .traveltime import phase_pair_curve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Seismic coda correlation tools")
    commands = parser.add_subparsers(dest="command", required=True)
    times = commands.add_parser("traveltime", help="Generate a phase-pair reference curve")
    times.add_argument("--phase-a", default="PKIKP")
    times.add_argument("--phase-b", default="PKJKP")
    times.add_argument("--model", default="prem")
    times.add_argument("--depth-km", type=float, default=0.0)
    times.add_argument("--distance-min", type=float, default=90.0)
    times.add_argument("--distance-max", type=float, default=180.0)
    times.add_argument("--step", type=float, default=1.0)
    times.add_argument("--output", type=Path, default=Path("phase_pair_time.txt"))
    for stage in ("events", "stations", "download", "correlation", "plotting"):
        sub = commands.add_parser(stage, help=f"Run {stage} using a JSON settings object")
        sub.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "traveltime":
        if (not np.isfinite([args.distance_min, args.distance_max, args.step]).all()
                or not 0 <= args.distance_min < args.distance_max <= 180 or args.step <= 0):
            parser.error("Require 0 <= distance-min < distance-max <= 180 and finite step > 0")
        count = int(np.ceil((args.distance_max - args.distance_min) / args.step))
        distances = np.r_[args.distance_min + np.arange(count) * args.step, args.distance_max]
        curve = phase_pair_curve(distances, args.depth_km, args.phase_a, args.phase_b, args.model)
        curve.save(args.output)
        print(f"Saved {args.output}: {np.isfinite(curve.lag_sec).sum()}/{len(distances)} valid points")
    else:
        settings = json.loads(args.config.read_text())
        if not isinstance(settings, dict):
            parser.error("The config file must contain a JSON object.")
        result = importlib.import_module(f"pycctools.{args.command}").run(**settings)
        print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
