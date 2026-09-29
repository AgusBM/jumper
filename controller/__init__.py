"""Game controllers, as their own axis.

A fourth thing alongside the framework, the tasks and the scenes, and separate for
the same reason a scene is: **it knows nothing about robots.** Nothing here has an
opinion about what a stick means. It reports that the left stick is at (0.4, -0.9)
and that A is down; deciding that this is 0.2 m/s forward belongs to the task, in
`tasks/<family>/mdp/`, the same way a key's meaning does.

    controller/          this package: find a pad, read it, name it
      vocabulary.json      what a pad can say -- the dictionary, and the only
                           copy of that list anywhere
    tasks/.../mdp/       what a stick and a button *mean* for this robot
    deploy/fsm/          the same dictionary, for the robot and the browser
    scripts/             unchanged -- no entry point mentions a controller

`python -m controller --vocabulary` prints the dictionary. Three programs read
it and none of them writes it: this package, a task validating its
`controls.yaml`, and the deployed controller refusing a `[[fsm.button]]`.

## What it is for

`play.py` samples random velocity commands. That is right for seeing a policy
across its training distribution and useless for "walk it into the corner and see
what happens". The keyboard already covers that (`mjrl.viewer.keys` plus a task's
teleop term); a stick covers it better, because walking is a continuous thing and
a key press is not.

## Using it

    import controller

    pad = controller.open()          # None if nothing is plugged in
    if pad is not None:
        s = pad.state()              # a snapshot; never blocks
        print(s.lx, s.ly, s.a)

`open()` returns None rather than raising when there is no pad, because "no
controller today" is the normal case and not an error. It *does* print why when
there is a device it cannot use -- a pad that is plugged in but unreadable is a
problem to be told about, and the cause is almost always permissions rather than
anything to do with this code.

## Permissions, which is where this actually fails on Linux

`/dev/input/event*` is `crw-rw---- root input`, so reading it needs membership of
the `input` group:

    sudo usermod -aG input "$USER"     # then log out and back in

Until that is done a paired, working, blinking controller reads as absent. This
package says so explicitly rather than looking like it simply found nothing --
`describe()` reports what it saw and why each candidate was rejected.

## A new top-level package needs the editable install refreshed

`pip install -e .` writes a **strict name-to-path table**, not a path entry, so a
package added after the install is invisible to anything that does not happen to
be run from the repository root. The symptom is confusing: `python -c "import
controller"` works (the current directory is on `sys.path`) while
`python scripts/play.py` raises `ModuleNotFoundError`, because a script's
`sys.path[0]` is `scripts/`. `pip install -e .` again, once, fixes it -- the same
thing was true of `scenes` when it arrived.

## Platform

Linux through evdev (`device.py`), Windows through XInput (`xinput.py`), both with
no dependency beyond the standard library. The Windows reader presents each
reading as the evdev table Linux's `xpad` driver would have produced, so both
end in the same `xbox.normalise` and a stick means the same number on either.
macOS exposes nothing equivalent without a package, and adding one for a
convenience feature is the wrong trade; `open()` returns None there, `describe()`
says it is the platform, and everything that uses it carries on without a pad.
"""

from __future__ import annotations

import sys

from . import device, xinput
from .device import Gamepad, GamepadState, PadInfo
from .xinput import XInputPad

__all__ = [
    "Gamepad", "GamepadState", "XInputPad", "available", "describe", "find_pads", "open",
]


def _backend():
    """The module that reads pads on this platform, or None where nothing does."""
    if sys.platform == "linux":
        return device
    if sys.platform == "win32":
        return xinput
    return None


def find_pads(readable_only: bool = False) -> list[PadInfo]:
    """Every pad this platform's reader can see, in a stable order."""
    backend = _backend()
    return backend.find_pads(readable_only) if backend is not None else []


def describe() -> str:
    """What was found and, for anything rejected, why. For diagnosing "the pad is
    connected but nothing happens", which is otherwise indistinguishable from
    "no pad"."""
    backend = _backend()
    if backend is None:
        return (
            f"pads are read on Linux (evdev) and Windows (XInput), not on "
            f"{sys.platform}; the keyboard drives."
        )
    return backend.describe()


def available() -> bool:
    """Is there a controller this process can actually read?

    Cheap: it stats the device nodes and checks readability (on Windows, asks each
    XInput slot once), and never opens a reader thread. Safe to call while building
    a config.
    """
    return bool(find_pads(readable_only=True))


def open(name: str | None = None) -> Gamepad | XInputPad | None:
    """Open the first usable controller, or None.

    Args:
        name: substring of the device name to require, for a machine with more
            than one pad. None takes the first.
    """
    backend = _backend()
    if backend is None:
        return None
    pads = backend.find_pads(readable_only=True)
    if name is not None:
        pads = [p for p in pads if name.lower() in p.name.lower()]
    if not pads:
        return None
    pad = (XInputPad if backend is xinput else Gamepad)(pads[0])
    pad.start()
    return pad
