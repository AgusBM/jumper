"""jumper.tetrapod -- tetrapod gait: 2 legs swinging / 4 in support, the middle ground.

Tetrapod gait: two legs swing at a time and four stay in support. Between
tripod and ripple in both speed and stability.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`), and the MDP terms only it uses (`mdp/`, here its gait reward).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.tetrapod",
    assets=JUMPER_ASSETS,
    description="jumper hexapod flat-ground velocity tracking; tetrapod gait, 2 swinging / 4 in support, a middle ground",
    tags=("locomotion", "jumper", "flat", "tetrapod"),
)
