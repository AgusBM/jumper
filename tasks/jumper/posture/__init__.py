"""jumper.posture -- the tripod gait with the body's posture under command.

Walks like `jumper.tripod` and is additionally told how to hold its body while it
does: how far to twist it over its feet, how far to pitch and roll it, and how
high to carry it. Four more commanded numbers, three more reward terms, and a
faster clock -- 5 Hz rather than 3.125 -- because this task's 0.8 m/s ceiling
cannot be reached at tripod's cadence with the stride the legs have.

`jumper.tripod` is this task's control and is deliberately untouched by it.

The robot definition and the environment skeleton live in `tasks/jumper/common/`;
the gait reward and the phase clock are imported from `tasks/jumper/tripod/mdp/`
rather than copied. What is here is the posture: its command (`mdp/commands.py`),
what it is measured against (`mdp/state.py`) and what it is paid for
(`mdp/rewards.py`).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.posture",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod velocity tracking with a commanded body posture "
        "(twist / pitch / roll / height); tripod gait at 5 Hz, 0.8 m/s"
    ),
    tags=("locomotion", "jumper", "flat", "tripod", "posture"),
)
