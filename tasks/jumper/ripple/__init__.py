"""jumper.ripple -- ripple gait: 1 leg swinging / 5 in support, the most stable.

Ripple gait: one leg swings at a time and five stay in support. The slowest of
the three and the most stable, with the largest support polygon.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`), and the MDP terms only it uses (`mdp/`, here its gait reward).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.ripple",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-ground velocity tracking; ripple gait, 1 swinging / 5 in support, most stable",
    tags=("locomotion", "jumper", "flat", "ripple"),
)
