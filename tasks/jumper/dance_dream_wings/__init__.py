"""jumper.dance_dream_wings -- the dream_wings dance, from rl-wbc-fsm, learned by imitation.

The same task as `jumper.dance` with a different clip: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/dream_wings.npz`, written by `tools/import_wbc_dances.py` from rl-wbc-fsm's
`dance_choreo5.npz`, 55.58 s at 500 Hz. On rl-wbc-fsm the board plays it as
open-loop playback of these rows, as the FSM state `dance_dream_wings`.

The joint angles are the commanded ones taken as the state, and the base pose is
solved from them: the source's own is the planner's body target, not a pose the
robot takes. The tool's docstring has the measurements behind both.

rl-wbc-fsm also carries an older, 69.0 s take of it (`dance_final.npz`). It is not
imported: `dance_choreo5.npz` is the one the board plays.

No music ships with it, and training does not need any. The performance video
`scripts/export.py` renders does, so export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_dream_wings",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's dream_wings dance; the clip is "
                "committed in tasks/jumper/dance_dream_wings/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
