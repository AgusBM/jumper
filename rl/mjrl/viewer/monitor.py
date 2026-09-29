"""Live telemetry for `play.py --measure`: position, velocity, torque, foot force.

What `--measure` is for: watching a replay tells you whether the robot walks, and
almost nothing about what the joints are doing to achieve it. A servo sitting on
its torque limit, one joint doing all the work, a velocity spike at every
touchdown -- all of it is invisible in the picture and obvious in three plots.

## Where the numbers come from

The entity the camera is following (`play._watched`), one environment, straight
off the simulation:

    position   `data.joint_pos`      rad, absolute (not relative to home)
    velocity   `data.joint_vel`      rad/s
    torque     `data.actuator_force` N*m, the actuator's applied force

and, when the scene has a contact sensor that tracks air time, one more:

    foot force `sensor.data.force`   N, the contact force magnitude per foot

**The sensor is chosen by `track_air_time`, not by name.** A contact sensor that
accumulates air time is one somebody built to reason about footfalls; one that
does not -- `self_collision`, say -- is watching for something else, and plotting
its force beside the feet would be noise in the panel and a column of zeros in the
csv. If several qualify the first is taken and the rest are named in the startup
line, so a scene with two foot sensors says so rather than picking silently.

Air time is also what makes the landing impact available: `compute_first_contact`
marks the primaries that touched down within the last step, and the force on that
step is the impact. That number is the one `soft_landing` is scored on and the one
`Metrics/landing_force_mean` reports, so `--measure` and the reward curves are
looking at the same quantity.

**Torque is per actuator and the other two are per joint**, and those are
different lists in general -- this robot drives all 22 of its joints today, but
it once had 20 actuators on 22 joints, with the grippers held by their PD. They
are plotted as separate panels with
their own names rather than zipped together, because pairing them by index is
wrong the moment a model has an unactuated joint, and wrong silently.

Nothing here is task-specific or robot-specific: the names come from the entity.

## The display, and why there are two of them

A live window is the right answer and is not always available: matplotlib needs
an interactive backend, which needs a GUI toolkit, and a training box reached
over SSH often has none. So this picks in order:

1. **a matplotlib window**, if an interactive backend imports -- three stacked
   panels, one line per joint, a rolling window;
2. **a terminal table** otherwise, rewritten in place at the same rate, carrying
   the current value and the window's range for every joint.

The fallback is not a consolation prize: for "is anything saturating", a table of
maxima is easier to read than twenty overlapping lines. What it cannot show is
*shape* -- a velocity spike at touchdown against a smooth ramp -- which is what
the window is for.

Either way the run is recorded and written out at the end as `measure.csv` plus
`measure.png`, so a headless session still produces the plot; it just produces it
afterwards.

## Cost

One `.cpu()` per step on three small tensors. Measured on this repository's
hexapod (22 joints, 20 actuators) at 50 Hz control: under 0.2 ms per step,
against a 20 ms budget. The redraw is decimated to `refresh_hz` (10 by default)
because a matplotlib redraw is 10-30 ms and doing it every step would halve the
replay rate -- the pacer would then quietly stop keeping real time, which is the
one way a measurement tool could change what it is measuring.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

#: Seconds of history kept and drawn. 5 s is about 15 gait cycles at this
#: robot's cadence -- enough to see a rhythm and short enough that a transient
#: does not stay on screen after it has gone.
WINDOW_S = 5.0

#: How often the display is redrawn, in hertz. See the note on cost above.
REFRESH_HZ = 10.0


#: The GUI toolkits worth trying, most preferred first, as
#: `(matplotlib backend, the module that has to import, environment it needs)`.
#:
#: **Each Qt binding is a separate candidate**, and that is the whole reason this
#: is a list of three near-identical rows rather than one `QtAgg`. matplotlib
#: picks a binding for itself and prefers the newest installed, so on a machine
#: with both PySide6 and PyQt5 it takes PySide6 -- and if that one cannot load
#: its platform plugin, "QtAgg does not work here" is the wrong conclusion: the
#: other binding ships its own Qt and works. Measured on this machine: PySide6
#: 6.11 needs `libxcb-cursor0`, which is a system package, and PyQt5 opens a
#: window with nothing installed beyond the venv.
#:
#: `QT_API` is how the choice is made to stick; matplotlib's `qt_compat` reads it.
_BACKENDS = (
    ("QtAgg", "PySide6", {"QT_API": "pyside6"}),
    ("QtAgg", "PyQt6", {"QT_API": "pyqt6"}),
    ("QtAgg", "PyQt5", {"QT_API": "pyqt5"}),
    ("TkAgg", "tkinter", {}),
    ("GTK4Agg", "gi", {}),
)

#: What the probe below runs. Creating the figure is what instantiates the
#: canvas, which is what loads the platform plugin -- importing the toolkit is
#: not enough to find out whether a window can open.
_PROBE = (
    "import matplotlib;matplotlib.use({backend!r});"
    "import matplotlib.pyplot as p;p.figure();p.close('all')"
)


def _interactive_backend() -> "tuple[str, dict[str, str]] | None":
    """The first matplotlib backend that can actually open a window here.

    **Probed in a subprocess, and that is not caution for its own sake.** The
    obvious version of this asks whether the toolkit imports, and the obvious
    version is what shipped first: PySide6 imported perfectly on a machine whose
    Qt could not load its `xcb` platform plugin (`libxcb-cursor0` missing), and
    the failure is `qFatal` -> `abort()`. Not an exception -- the whole replay
    went down with SIGABRT, several seconds after it had started, taking the
    environment with it.

    There is no in-process way to catch that, so the question is asked somewhere
    it is allowed to crash. The cost is one interpreter start per candidate, once,
    only when `--measure` is given.
    """
    import importlib.util
    import os
    import subprocess
    import sys

    for backend, module, env in _BACKENDS:
        if importlib.util.find_spec(module) is None:
            continue
        probe = subprocess.run(
            [sys.executable, "-c", _PROBE.format(backend=backend)],
            capture_output=True,
            timeout=60,
            check=False,
            env={**os.environ, **env},
        )
        if probe.returncode == 0:
            return backend, env
        why = (probe.stderr.decode(errors="replace").strip().splitlines() or [""])[-1]
        print(f"[measure] {module} is installed but cannot open a window: {why}")
    return None


class JointMonitor:
    """Collects and displays joint telemetry for one environment.

    `update` is called once per control step from `play._loop`; `close` writes the
    artifacts. Construction decides the display, so a failure to open a window
    happens before the replay starts rather than a minute into it.
    """

    def __init__(
        self,
        entity: Any,
        index: int,
        step_dt: float,
        out_dir: Path,
        sensor: Any = None,
        foot_names: "tuple[str, ...] | None" = None,
        window_s: float = WINDOW_S,
        refresh_hz: float = REFRESH_HZ,
    ) -> None:
        self.entity = entity
        self.index = index
        self.step_dt = step_dt
        self.out_dir = out_dir
        self.joint_names = tuple(getattr(entity, "joint_names", ()) or ())
        self.actuator_names = tuple(getattr(entity, "actuator_names", ()) or ())
        self.sensor = sensor
        self.foot_names = tuple(foot_names or ())
        self.capacity = max(2, int(round(window_s / max(step_dt, 1e-6))))
        self.every = max(1, int(round(1.0 / (refresh_hz * max(step_dt, 1e-6)))))

        self._t: list[float] = []
        self._pos: list[list[float]] = []
        self._vel: list[list[float]] = []
        self._tau: list[list[float]] = []
        self._force: list[list[float]] = []
        self._step = 0
        self._peak_tau = [0.0] * len(self.actuator_names)
        self._peak_force = [0.0] * len(self.foot_names)
        #: Every landing's impact force, in the order they happened. Kept in full
        #: rather than reduced on the fly: the distribution is the interesting
        #: part -- a mean of 17 N made of steady 17s and one made of 5s with an
        #: occasional 60 are different robots -- and a replay produces a few
        #: thousand of them at most.
        self._impacts: list[float] = []

        self._fig = None
        self._lines: dict[str, list] = {}
        chosen = _interactive_backend()
        self._backend = None if chosen is None else chosen[0]
        if chosen is not None:
            self._open_window(*chosen)
        else:
            print(
                "[measure] no toolkit here can open a window, so the live view is "
                "a table (`pip install PyQt5` in the venv is usually enough). The "
                "plot is still written at the end"
            )

    # ── The window ───────────────────────────────────────────────────────

    def _open_window(self, backend: str, env: dict) -> None:
        import os

        import matplotlib

        # The probe chose a Qt binding; `QT_API` is what makes this process use
        # the same one. Without it matplotlib re-picks, which on a machine with
        # two bindings installed means picking the one the probe rejected.
        os.environ.update(env)
        matplotlib.use(backend)
        import matplotlib.pyplot as plt

        plt.ion()
        rows = 4 if self.foot_names else 3
        self._fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(10, 2.7 * rows))
        panels = [
            ("pos", axes[0], "position [rad]", self.joint_names),
            ("vel", axes[1], "velocity [rad/s]", self.joint_names),
            ("tau", axes[2], "torque [N*m]", self.actuator_names),
        ]
        if self.foot_names:
            panels.append(("force", axes[3], "foot force [N]", self.foot_names))
        for key, ax, label, names in panels:
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            # One line per joint and **no legend**: twenty entries would cover
            # the panel they describe. Hovering is not available either, so the
            # window answers "is anything an outlier" and the table answers
            # "which one".
            self._lines[key] = [ax.plot([], [], lw=0.8)[0] for _ in names]
        axes[-1].set_xlabel("time [s]")
        feet = f" / {len(self.foot_names)} feet" if self.foot_names else ""
        self._fig.suptitle(
            f"{len(self.joint_names)} joints / {len(self.actuator_names)} actuators"
            f"{feet}  --  env {self.index}"
        )
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        print(f"[measure] live plot on {self._backend}")

    # ── Per step ─────────────────────────────────────────────────────────

    def update(self) -> None:
        """Record this step and, every `every` steps, redraw."""
        data = self.entity.data
        pos = data.joint_pos[self.index].detach().cpu().tolist()
        vel = data.joint_vel[self.index].detach().cpu().tolist()
        tau = data.actuator_force[self.index].detach().cpu().tolist()

        self._t.append(self._step * self.step_dt)
        self._pos.append(pos)
        self._vel.append(vel)
        self._tau.append(tau)
        if self.foot_names:
            self._record_force()
        for i, value in enumerate(tau):
            if i < len(self._peak_tau):
                self._peak_tau[i] = max(self._peak_tau[i], abs(value))
        # The window is what is *drawn*; everything is kept for the csv, which is
        # the point of running the thing at all. A 10 minute replay at 50 Hz is
        # 30000 rows of 62 floats -- 15 MB of text, which is not worth a ring
        # buffer and a second code path.
        self._step += 1

        if self._step % self.every:
            return
        if self._fig is not None:
            self._redraw()
        else:
            self._print_table()

    def _record_force(self) -> None:
        """This step's per-foot contact force, and any landing among them.

        `compute_first_contact` is read-only -- it compares the sensor's own
        contact-time accumulator against `dt` -- so calling it here does not
        disturb the reward term that calls it too.
        """
        import torch

        force = self.sensor.data.force
        if force is None:
            return
        mag = torch.linalg.norm(force[self.index], dim=-1).detach().cpu().tolist()
        self._force.append(mag)
        for i, value in enumerate(mag):
            if i < len(self._peak_force):
                self._peak_force[i] = max(self._peak_force[i], value)
        try:
            landed = self.sensor.compute_first_contact(dt=self.step_dt)
        except RuntimeError:  # the sensor does not track air time after all
            return
        for i, hit in enumerate(landed[self.index].tolist()):
            if hit and i < len(mag):
                self._impacts.append(mag[i])

    def _window(self):
        lo = max(0, len(self._t) - self.capacity)
        return lo, self._t[lo:]

    def _redraw(self) -> None:
        lo, t = self._window()
        panels = [("pos", self._pos), ("vel", self._vel), ("tau", self._tau)]
        if self.foot_names:
            panels.append(("force", self._force))
        for key, rows in panels:
            series = rows[lo:]
            lines = self._lines[key]
            for i, line in enumerate(lines):
                line.set_data(t, [row[i] for row in series])
            ax = line.axes
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(t[0], max(t[-1], t[0] + 1e-3))
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def _print_table(self) -> None:
        """The fallback view: one rewritten block, current value and window range.

        Written with an explicit cursor-up escape rather than by clearing the
        screen, so the lines `play.py` prints around it stay where they are.
        """
        lo, _ = self._window()
        rows = []
        for i, name in enumerate(self.joint_names):
            pos = [r[i] for r in self._pos[lo:]]
            vel = [r[i] for r in self._vel[lo:]]
            rows.append((name, pos[-1], min(vel), max(vel)))
        width = shutil.get_terminal_size((100, 24)).columns
        limit = max(4, min(len(rows), 24))

        lines = [
            f"{'joint':<24}{'pos rad':>9}{'vel min':>9}{'vel max':>9}"
            f"{'torque':>9}{'peak':>8}"
        ]
        tau_of = {n: i for i, n in enumerate(self.actuator_names)}
        for name, pos, vmin, vmax in rows[:limit]:
            j = tau_of.get(name)
            tau = self._tau[-1][j] if j is not None else math.nan
            peak = self._peak_tau[j] if j is not None else math.nan
            tail = (
                f"{tau:>9.3f}{peak:>8.3f}"
                if j is not None
                else f"{'--':>9}{'--':>8}"  # a joint with no actuator
            )
            lines.append(f"{name:<24}{pos:>9.3f}{vmin:>9.2f}{vmax:>9.2f}{tail}")
        if self.foot_names and self._force:
            now = self._force[-1]
            lines.append("")
            lines.append(
                f"{'foot':<24}{'force N':>9}{'peak':>9}{'down':>9}"
                f"{'landings':>10}{'impact N':>10}"
            )
            down = [
                sum(1 for r in self._force[lo:] if r[i] > 1.0) / max(1, len(self._force[lo:]))
                for i in range(len(self.foot_names))
            ]
            for i, name in enumerate(self.foot_names):
                lines.append(
                    f"{name:<24}{now[i]:>9.2f}{self._peak_force[i]:>9.2f}"
                    f"{down[i]:>8.0%} {'':>9}{'':>10}"
                )
            mean = sum(self._impacts) / len(self._impacts) if self._impacts else 0.0
            peak = max(self._impacts) if self._impacts else 0.0
            lines.append(
                f"{'  landing impact':<24}{'':>9}{'':>9}{'':>9}"
                f"{len(self._impacts):>10}{mean:>7.1f}/{peak:<.0f}"
            )
        block = "\n".join(line[:width] for line in lines)
        if self._printed:
            print(f"\033[{len(lines)}A", end="")
        print(block, flush=True)
        self._printed = True

    _printed = False

    # ── The end ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Write `measure.csv` and `measure.png`, and say where they went."""
        if not self._t:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        csv = self.out_dir / "measure.csv"
        header = (
            ["t"]
            + [f"pos/{n}" for n in self.joint_names]
            + [f"vel/{n}" for n in self.joint_names]
            + [f"tau/{n}" for n in self.actuator_names]
            + [f"force/{n}" for n in self.foot_names]
        )
        with csv.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for k, t in enumerate(self._t):
                force = self._force[k] if k < len(self._force) else []
                values = [t, *self._pos[k], *self._vel[k], *self._tau[k], *force]
                f.write(",".join(f"{v:.6g}" for v in values) + "\n")

        png = self.out_dir / "measure.png"
        try:
            self._write_png(png)
        except Exception as exc:  # a missing font must not lose the csv
            print(f"\n[measure] the plot could not be written ({exc}); csv is intact")
            png = None

        if self._impacts:
            impacts = sorted(self._impacts)
            mean = sum(impacts) / len(impacts)
            p95 = impacts[min(len(impacts) - 1, int(0.95 * len(impacts)))]
            print(
                f"\n[measure] {len(impacts)} landings  impact mean {mean:.1f} N  "
                f"p95 {p95:.1f} N  peak {impacts[-1]:.1f} N"
                f"  (soft_landing is scored on this)"
            )
        elif self.foot_names:
            print(
                "\n[measure] no landing was recorded -- the feet never left the "
                "ground, or the sensor does not track air time"
            )
        print(f"\n[measure] {len(self._t)} steps -> {csv}" + (f"  {png}" if png else ""))
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)

    def _write_png(self, path: Path) -> None:
        """The whole run, not the rolling window -- this is the artifact."""
        import matplotlib

        if self._fig is None:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        count = 4 if self.foot_names else 3
        fig, axes = plt.subplots(count, 1, sharex=True, figsize=(12, 3 * count))
        panels = [
            (axes[0], self._pos, "position [rad]", self.joint_names),
            (axes[1], self._vel, "velocity [rad/s]", self.joint_names),
            (axes[2], self._tau, "torque [N*m]", self.actuator_names),
        ]
        if self.foot_names:
            panels.append((axes[3], self._force, "foot force [N]", self.foot_names))
        for ax, rows, label, names in panels:
            for i in range(len(names)):
                ax.plot(self._t, [r[i] for r in rows], lw=0.7)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("time [s]")
        peak = max(self._peak_tau) if self._peak_tau else 0.0
        title = f"{len(self._t)} steps, peak |torque| {peak:.3f} N*m"
        if self._impacts:
            mean = sum(self._impacts) / len(self._impacts)
            title += (
                f", {len(self._impacts)} landings at {mean:.1f} N mean / "
                f"{max(self._impacts):.1f} peak"
            )
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)


def make_monitor(entity: Any, index: int, env, out_dir: Path) -> "JointMonitor | None":
    """Build a monitor for an entity, or None if it has no joints to measure.

    The entity is passed in rather than found here: `play.py` already picks the
    one the camera follows, and a second rule for "which entity" would be a
    second answer -- the plots would describe a robot the readout above them does
    not.

    Returns None rather than raising when there is nothing with joints. A
    measurement switch that stops a replay from running would be worse than one
    that has nothing to measure, and the caller says which happened.
    """
    if entity is None or not getattr(entity, "joint_names", ()):
        return None
    unwrapped = getattr(env, "unwrapped", env)
    sensor, names = _find_foot_sensor(unwrapped)
    return JointMonitor(entity, index, unwrapped.step_dt, out_dir, sensor, names)


def _find_foot_sensor(env) -> "tuple[Any, tuple[str, ...]]":
    """The scene's footfall sensor and a name per primary, or `(None, ())`.

    **Chosen by `track_air_time`, not by name.** A contact sensor that accumulates
    air time was built to reason about footfalls; one that does not is watching
    for something else -- `self_collision` is the example -- and its force belongs
    in neither the panel nor the csv. Air time is also what makes
    `compute_first_contact` work, which is where the landing impact comes from.

    Names come from the sensor's own `primary.pattern` when that is an explicit
    tuple, which is how a foot sensor is usually written. A regex would resolve to
    a count this cannot recover without the model, so those get indices -- a
    number is a worse label than a name and a better one than a wrong name.
    """
    sensors = getattr(getattr(env, "scene", None), "sensors", None) or {}
    found = []
    for name, sensor in (
        sensors.items() if hasattr(sensors, "items") else enumerate(sensors)
    ):
        cfg = getattr(sensor, "cfg", None)
        if cfg is None or "force" not in getattr(cfg, "fields", ()):
            continue
        if getattr(cfg, "track_air_time", False):
            found.append((name, sensor, cfg))
    if not found:
        return None, ()
    name, sensor, cfg = found[0]
    if len(found) > 1:
        others = ", ".join(str(n) for n, _, _ in found[1:])
        print(f"[measure] foot force from {name!r}; also available: {others}")
    pattern = getattr(getattr(cfg, "primary", None), "pattern", None)
    if isinstance(pattern, (tuple, list)) and pattern:
        labels = tuple(str(p) for p in pattern)
    else:
        force = getattr(sensor.data, "force", None)
        count = 0 if force is None else int(force.shape[1])
        labels = tuple(f"{name}[{i}]" for i in range(count))
    return sensor, labels


__all__ = ["JointMonitor", "REFRESH_HZ", "WINDOW_S", "make_monitor"]
