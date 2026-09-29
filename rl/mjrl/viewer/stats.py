"""How fast the replay is running, and how fast the robot is going.

A live viewer answers "does it look right" and says nothing about *how quickly the
world is passing*. That matters more than it sounds: a gait watched at a third of
real time looks composed and careful, and the same gait at real time can be a
scramble. Without a number on screen there is nothing to tell the two apart, and
"it looks smooth" quietly means "the machine is too slow to show it honestly".

## The three numbers

**fps** -- control steps executed per wall-clock second. What the eye sees.

**speed factor** -- simulated seconds per wall-clock second, i.e. `fps * step_dt`.
**1.0x is real time**; below that is slow motion and above it is fast-forward. The
two are proportional and both are worth showing: fps says whether the picture is
smooth, the factor says whether what you are judging is what the robot would
actually do.

**speed** -- how fast the robot is travelling, beside the speed it was asked for,
so "is it tracking" is readable without the tensorboard.

## Why the numbers are averaged over a window

Per-step timing on a GPU is mostly noise -- one step can stall on a kernel launch
and read as 4 fps. Averaging over the reporting interval gives a number that
changes slowly enough to read, and the interval is wall-clock rather than a step
count so that a very slow run still reports rather than appearing to hang.
"""

from __future__ import annotations

import time

__all__ = ["Pacer", "RunStats"]


class Pacer:
    """Hold a replay to a chosen multiple of real time.

    Without this, a replay runs at whatever rate the machine manages, which is a
    different speed on every machine and on the same machine at two different
    environment counts. That makes the thing on screen unjudgeable: a gait cannot
    be called quick or laboured if the clock it is being watched on is unknown.
    So the default is **1.0, real time**, and anything else is asked for.

    ## Falling behind is not made up for

    If a step takes longer than its slot -- a big scene, a slow machine -- the
    schedule is reset to now rather than the deficit being carried. Carrying it
    produces the worst possible picture: a stall, then a burst of steps run
    flat out to catch up, then another stall. Resetting means the replay simply
    runs as fast as it can, and `RunStats` reports a factor below the target,
    which is the honest signal that the machine cannot keep up.

    A `speed` of 0 disables pacing entirely.
    """

    def __init__(self, step_dt: float, speed: float = 1.0) -> None:
        self.speed = speed
        self.interval = step_dt / speed if speed > 0 else 0.0
        self._next: float | None = None

    def wait(self) -> None:
        """Sleep until this step's slot. Call once per control step."""
        if self.interval <= 0.0:
            return
        now = time.perf_counter()
        if self._next is None:
            self._next = now + self.interval
            return
        delay = self._next - now
        if delay > 0:
            time.sleep(delay)
            self._next += self.interval
        else:
            self._next = now + self.interval


class RunStats:
    """Accumulates steps and reports a line at most every `interval` seconds."""

    def __init__(self, step_dt: float, interval: float = 0.5) -> None:
        self.step_dt = step_dt
        self.interval = interval
        self._steps = 0
        self._last = time.perf_counter()

    def update(self, extra: str = "") -> str | None:
        """Count one control step. Returns a line when the interval has elapsed,
        `None` otherwise, so the caller can print unconditionally."""
        self._steps += 1
        now = time.perf_counter()
        elapsed = now - self._last
        if elapsed < self.interval:
            return None

        fps = self._steps / elapsed
        factor = fps * self.step_dt
        self._steps = 0
        self._last = now

        line = f"{fps:5.1f} fps   {factor:4.2f}x real time"
        return f"{line}   {extra}" if extra else line
