"""Settings shared by invocation-local workflow stages."""
from pathlib import Path


def validate_settings(overrides, allowed):
    unknown = set(overrides) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")


def setting(overrides, name, default):
    value = overrides.get(name, default)
    if isinstance(default, Path) and value is not None:
        return Path(value).expanduser()
    return value


def display(value):
    """Text summaries work in terminals as well as notebooks."""
    print(value.to_string() if hasattr(value, "to_string") else value)
