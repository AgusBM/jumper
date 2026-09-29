"""jumper.flat -- no gait prior; the one-variable control for tetrapod and ripple.

It adds no gait prior at all, so against `jumper.tetrapod` and `jumper.ripple` it is a
genuine control: those two add exactly one reward term each and change nothing
else. **Not against `jumper.tripod`**, which adds two reward terms (`tripod_gait`
and `stance_load`) and a `gait_phase` observation -- three variables at once, so
an outcome there cannot be attributed to the gait prior among them. See the
measured table in `tasks/jumper/common/ppo.py`.

It already walks -- tracking rates of 94/91/99% (see docs/DESIGN.md). The value of
the gait tasks is in regularising the shape of the gait, not in making walking
work at all.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`) and its hyper-parameters
(`rl_cfg.py`). It adds no gait prior, so it has no `mdp/`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.flat",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-ground velocity tracking; no gait prior (the one-variable control for tetrapod and ripple)",
    tags=("locomotion", "jumper", "flat", "base"),
)
