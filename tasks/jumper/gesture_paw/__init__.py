"""jumper.gesture_paw -- offering a paw, from rl-wbc-fsm, learned by imitation.

A one-shot gesture on the dance machinery: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/paw.npz`, written by `tools/import_wbc_gestures.py` from the `q_cmd` table
of rl-wbc-fsm's `policy/hexa/action_shake/action_shake.reference_clip.json`. On
rl-wbc-fsm the d-pad's left plays it as the FSM state `action_shake`, open loop, at
the speed it was written. The state is named for a shake; its config notes that
what it plays for now is offering a paw, and that is what the table does, so this
task is named for what it does.

The table was not recorded but interpolated from hand-placed keyframes
(rl-wbc-fsm's `tools/synth_action_clip.py`, which solved the poses with MuJoCo): the
robot sits back on its rear legs, curls the left front arm, reaches it forward,
lowers the paw as if onto a hand, holds it there for two seconds, takes it back and
stands, in 5.1 s. The right front arm stays down and carries weight throughout. The
tool adds short lead-ins, since the table starts and ends within 0.15 rad of `HOME`
already, and half a second standing there: 6.4 s in all.

There is no music, and nothing to export a performance video with: export with
`--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_paw",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's paw gesture (sits and offers "
                "its left paw); the clip is committed in tasks/jumper/gesture_paw/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
