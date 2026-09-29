"""jumper.gesture_hello -- waving hello, from rl-wbc-fsm, learned by imitation.

A one-shot gesture on the dance machinery: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/hello.npz`, written by `tools/import_wbc_gestures.py` from the `q_cmd` table
of rl-wbc-fsm's `policy/hexa/action_hello/action_hello.reference_clip.json`. On
rl-wbc-fsm the d-pad's up plays it as the FSM state `action_hello`, open loop, three
times faster than it was recorded; the clip here is timed as the board plays it.

The robot leans back 8 degrees on its four legs and waves the left front arm
overhead, six times in 5.3 s. The tool adds a lead-in from `HOME` and a lead-out back
to it, and half a second standing there, so the whole clip is 7.3 s and begins and
ends where the walking policy stands. Its docstring has why, and how the base pose,
which the table does not have, is solved from the joints.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_hello",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's hello gesture (a wave); the "
                "clip is committed in tasks/jumper/gesture_hello/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
