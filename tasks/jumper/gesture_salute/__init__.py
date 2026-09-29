"""jumper.gesture_salute -- a salute, from rl-wbc-fsm, learned by imitation.

A one-shot gesture on the dance machinery: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/salute.npz`, written by `tools/import_wbc_gestures.py` from the `q_cmd` table
of rl-wbc-fsm's `policy/hexa/action_salute/action_salute.reference_clip.json`. On
rl-wbc-fsm the d-pad's right plays it as the FSM state `action_salute`, open loop,
twice as fast as it was recorded; the clip here is timed as the board plays it.

The robot crouches onto its left paw and rear legs, rises 50 mm onto four legs with
the right front arm raised in salute, holds it, and settles back, in 4.0 s. At the
top the body is 48 mm above its standing height, where none of the other three
gestures goes more than 11 mm above it. The tool adds a lead-in from `HOME` and a lead-out back to it, and half
a second standing there, so the whole clip is 6.0 s and begins and ends where the
walking policy stands. Its docstring has why, and how the base pose is solved from
the joints -- which for this clip is where the dances' method is furthest off.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_salute",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's salute gesture; the clip is "
                "committed in tasks/jumper/gesture_salute/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
