"""jumper.ref_free_jump -- a high jump whose policy needs no recording.

The policy commands absolute joint targets around HOME, observes its body and
no clock, and jumps when it starts to run; an episode is that jump and the way
back to HOME. `high_jump_flat.npz` is used only while training: fed forward into
the action and as a prior on the joints, both fading to nothing, as spawn
states, and in the critic. See `env_cfg.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.ref_free_jump",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod high jump, reference-free on the robot "
        "(high_jump_flat.npz only while training: a fading feedforward and joint prior)"
    ),
    tags=("locomotion", "jumper", "jump", "action-prior"),
)
