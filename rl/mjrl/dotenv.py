"""A minimal `.env` reader with no third-party dependency.

It does one thing: fill the file's keys into `os.environ` **where nothing is set
yet**. Existing variables are never overwritten, which makes the precedence fall
out for free:

    command line > shell environment > .env.local > .env > argument default

The entry points under `scripts/` call this **before** parsing arguments, and
argparse defaults then read from `os.environ` -- so "the command line wins if
given" needs no special-casing; that is already argparse's semantics.

`python-dotenv` is not used: twenty lines is all that is needed, and every extra
dependency is one more thing that may or may not be installed on a given machine.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

__all__ = ["parse_dotenv", "load_dotenv", "env_default", "env_flag", "env_choice"]

#: What counts as a switch in a `.env` file, in any case. `bool()` is no use
#: here -- `bool("off")` is True, so every "off" would read as an "on".
_TRUE = frozenset({"on", "true", "yes", "1"})
_FALSE = frozenset({"off", "false", "no", "0"})


def env_default(name: str, fallback: str | None = None) -> str | None:
    """Read an environment variable as an argument default; **an empty string
    counts as unset**.

    That way `MJRL_MODEL=` in `.env` means "leave it empty, use the built-in
    default" rather than passing an empty string down as a real value.
    """
    return os.environ.get(name, "").strip() or fallback


def env_flag(name: str, fallback: bool = False) -> bool:
    """Read an environment variable as an on/off switch.

    `on/off`, `true/false`, `yes/no` and `1/0` are all accepted, in any case:
    those are the spellings people write in a config file, and a switch that
    honours exactly one of them is a switch that looks broken. Unset or empty
    gives `fallback`.

    Raises:
        ValueError: for anything else, **rather than falling back**. Guessing
            would mean `MJRL_TENSORBOARD=of` quietly meaning "on" -- the setting
            has no visible effect, nothing says why, and the natural conclusion is
            that the feature ignores its own configuration.
    """
    raw = env_default(name)
    if raw is None:
        return fallback
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not on or off "
        f"(true/false, yes/no and 1/0 are accepted too, in any case)"
    )


def env_choice(
    name: str, choices: Sequence[str], fallback: str | None = None
) -> str | None:
    """Read an environment variable that has to be one of `choices`.

    Matching ignores case and the value comes back in the spelling `choices` uses,
    so a switch behaves the same way whichever key it is set through.

    This exists because **argparse does not check a default against `choices`** --
    only values actually typed on the command line. A `choices=[...]` argument
    whose default comes from `.env` therefore validates nothing, and the bad value
    travels on to whatever finally consumes it: `MJRL_STRIP_VISUAL=OFF` used to
    surface as a bare `KeyError: 'OFF'` from a lookup table three call levels away.

    Raises:
        ValueError: naming the value and the choices, for the same reason
            `env_flag` does rather than falling back.
    """
    raw = env_default(name)
    if raw is None:
        return fallback
    value = raw.strip().lower()
    for choice in choices:
        if choice.lower() == value:
            return choice
    raise ValueError(f"{name}={raw!r} is not one of: {', '.join(choices)}")


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse `.env` text.

    Supports `KEY=VALUE`, `export KEY=VALUE`, `#` comments, blank lines, and
    quoting a value that contains spaces or `#`. Variable interpolation is not
    supported -- a config file should not be a script.
    """
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue  # not an assignment; ignore rather than error, to tolerate notes
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]  # keep quoted content verbatim, including # and spaces
        else:
            value = value.split(" #", 1)[0].rstrip()  # strip a trailing comment
        result[key] = value
    return result


def load_dotenv(*paths: Path) -> dict[str, str]:
    """Read several `.env` files in order, filling keys not already in `os.environ`.

    Later files take precedence (`.env.local` should come after `.env`).
    **Existing environment variables are never overwritten** -- a value exported
    explicitly in the shell should beat one from a file.

    Returns:
        The keys and values actually written to `os.environ`, excluding those
        skipped. A missing file is not an error.
    """
    merged: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            merged.update(parse_dotenv(path.read_text(encoding="utf-8")))

    applied = {k: v for k, v in merged.items() if k not in os.environ}
    os.environ.update(applied)
    return applied
