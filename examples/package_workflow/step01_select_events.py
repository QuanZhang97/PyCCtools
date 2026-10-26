"""Select events with :func:`pycctools.events.run`."""
from pathlib import Path

from pycctools import events

from _workflow_state import print_outputs, update_state
from settings import EVENT_SETTINGS


def run():
    settings = dict(EVENT_SETTINGS)
    source = settings.get("LOCAL_CATALOGUE")
    if source is not None and not Path(source).expanduser().is_file():
        raise FileNotFoundError(
            f"Local catalogue not found: {source}. Edit EVENT_SETTINGS in "
            "settings.py, or set LOCAL_CATALOGUE=None for online GCMT data."
        )
    result = events.run(**settings)
    update_state(
        event_run_dir=result["run_dir"],
        catalogue_file=result["catalogue_file"],
    )
    return result


def main():
    print_outputs(run())


if __name__ == "__main__":
    main()
