"""jumper.dance_brazilian -- the Brazilian dance, from rl-wbc-fsm, learned by imitation.

The same task as `jumper.dance` with a different clip: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/brazilian.npz`, written by `tools/import_wbc_dances.py` from rl-wbc-fsm's
`dance_choreo4.npz`, 66.07 s at 500 Hz. On rl-wbc-fsm the board plays it as
open-loop playback of these rows, as the FSM state `dance_baxi` -- *baxi* being
the pinyin for Brazil; this repository names its dances in English.

The joint angles are the commanded ones taken as the state, and the base pose is
solved from them: the source's own is the planner's body target, not a pose the
robot takes. The tool's docstring has the measurements behind both.

Two things about this clip are not true of the others, and both are in the rows the
board sends:

- **The left wrist is asked past its stop.** `LF_J3_joint` is commanded to -2.618
  against a limit of -2.100 on 16.7% of the track. The import clamps it to the
  limit, since a state cannot lie past a mechanical stop.
- **The support legs straighten to exactly 0.000** wherever the planner's IK ran out
  of reach, a 0.30 rad step in one 2 ms tick -- 22.0 rad/s at the control rate,
  inside the servo's 30.7 rad/s corner speed, so the loader accepts it.

No music ships with it, and training does not need any. The performance video
`scripts/export.py` renders does, so export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_brazilian",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's Brazilian dance; the clip is "
                "committed in tasks/jumper/dance_brazilian/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
