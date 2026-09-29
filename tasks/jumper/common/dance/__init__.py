"""The dance tasks' shared machinery: every `jumper.dance*` task is this plus a clip
and its own numbers.

It lived in `tasks/jumper/dance/mdp/` while `jumper.dance` was the only dance, by the
rule in `docs/USAGE.md` that a term used by one task stays with that task. The
second dance moved it here; nothing in it changed on the way except that the
material directory is now the task's to pass (`motion.find_material`), since with
several dances a default is one of them chosen for the rest.

- `motion`       -- reading the choreography from a task's `media/` and converting it
  for mjlab's `MotionLoader`, with the checks that make a wrong clip an error
- `observations` -- the reference, under names `scripts/export.py` can export
- `rewards`      -- joint-space tracking, and charging for torque peaks
- `env`          -- `dance_env_cfg`, the environment every dance task builds. It holds
  mechanism only: every weight, threshold and range is a required argument, so a
  dance task states all of them rather than inheriting one nobody chose for it.

The tracking rewards, terminations and the `MotionCommand` itself are **not** here.
They come from mjlab's `mjlab.tasks.tracking.mdp`, a complete re-implementation of
BeyondMimic that ships in the vendored tree. What this package adds is what is
specific to this robot: the clip format, exportable observation names, and the
torque-peak term.
"""

from __future__ import annotations

from .motion import (
    LEG_JOINTS,
    REFERENCE,
    SUPPORT_LEGS,
    ensure_motion_npz,
    find_material,
    load_source,
)
from .observations import (
    FUTURE_HORIZONS,
    clip_phase,
    ref_future,
    ref_joint_pos,
    ref_joint_vel,
    ref_tilt_error,
)
from .rewards import (
    actuator_power_cost,
    peak_actuator_force,
    torque_headroom_cost,
)

__all__ = [
    "FUTURE_HORIZONS",
    "LEG_JOINTS",
    "REFERENCE",
    "SUPPORT_LEGS",
    "actuator_power_cost",
    "clip_phase",
    "ensure_motion_npz",
    "find_material",
    "load_source",
    "peak_actuator_force",
    "ref_future",
    "ref_joint_pos",
    "ref_joint_vel",
    "ref_tilt_error",
    "torque_headroom_cost",
]
