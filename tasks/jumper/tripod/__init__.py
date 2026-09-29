"""jumper.tripod -- tripod gait: 3 legs swinging / 3 in support, the fastest.

Tripod gait: two groups of legs alternate in antiphase, three swinging and
three in support at a time. The fastest of the three gaits, with the smallest
and least stable support polygon.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`), and the MDP terms only it uses (`mdp/`, here its gait reward).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.tripod",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-ground velocity tracking; tripod gait, 3 swinging / 3 in support, fastest",
    tags=("locomotion", "jumper", "flat", "tripod"),
)
