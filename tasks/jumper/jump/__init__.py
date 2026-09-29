"""jumper.jump -- reference-guided high jump.

The robot performs one in-place high jump following
`tasks/jumper/jump/ref/high_jump_flat.npz`. The reward design is ported from the
reference-guided high jump task in kk-rl-lab (tuned on the same robot).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.jump",
    assets=JUMPER_ASSETS,
    description="jumper hexapod reference-guided high jump (high_jump_flat.npz crouch-jump-tuck-land)",
    tags=("locomotion", "jumper", "jump", "reference-tracking"),
)
