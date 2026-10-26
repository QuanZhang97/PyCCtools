"""Small JSON handoff used by the numbered example scripts."""
import json
from pathlib import Path

from settings import STATE_FILE


def load_state(*required_keys):
    """Load state and report which earlier example step is missing."""
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = {}
    missing = [key for key in required_keys if not state.get(key)]
    if missing:
        raise RuntimeError(
            "Missing workflow state: {}. Run the preceding numbered step(s) "
            "first, or run run_all.py.".format(", ".join(missing))
        )
    return state


def update_state(**paths):
    """Record absolute output paths without discarding earlier stage outputs."""
    state = load_state()
    state.update({
        key: str(Path(value).expanduser().resolve())
        for key, value in paths.items()
    })
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(STATE_FILE)
    return state


def print_outputs(result):
    print(json.dumps(result, indent=2, default=str, sort_keys=True))
