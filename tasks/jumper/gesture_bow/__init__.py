"""jumper.gesture_bow -- a bow with the paws pressed together, from rl-wbc-fsm,
learned by imitation.

A one-shot gesture on the dance machinery: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/bow.npz`, written by `tools/import_wbc_gestures.py` from the `q_cmd` table
of rl-wbc-fsm's `policy/hexa/action_bow/action_bow.reference_clip.json`. On
rl-wbc-fsm the d-pad's down plays it as the FSM state `action_bow` (作揖, the
cupped-hands bow), open loop, twice as fast as it was recorded; the clip here is
timed as the board plays it.

Both front arms rise and meet, hold, and come down, in 6.4 s, while all four legs
move under them -- it is the only one of the four gestures that moves all 22 joints. The tool adds a lead-in from
`HOME` and a lead-out back to it, and half a second standing there, so the whole
clip is 8.6 s and begins and ends where the walking policy stands. Its docstring
has why, how the base pose is solved from the joints, and the one thing about this
clip a policy cannot do exactly: the two claws meet before the table expects them
to, and are asked to press on into each other for 2 s.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_bow",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's bow gesture (paws pressed "
                "together); the clip is committed in tasks/jumper/gesture_bow/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
