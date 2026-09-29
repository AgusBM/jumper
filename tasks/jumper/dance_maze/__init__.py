"""jumper.dance_maze -- the maze dance, from rl-wbc-fsm, learned by imitation.

The same task as `jumper.dance` with a different clip: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/maze.npz`, written by `tools/import_wbc_dances.py` from the `joint_pos` table
of rl-wbc-fsm's `policy/hexa/dance/dance_maze.reference_clip.json`, 10.00 s at 50
Hz. On rl-wbc-fsm the board runs it as a policy, `dance_maze`, trained elsewhere on
this table; there is no playback track of it.

The joint angles are the table's, taken as the state, and the base pose, which the
table does not have, is solved from them. The tool's docstring has how, and the
control it was checked against.

The table is joint angles only and no npz of it exists anywhere; its note names
`dance_final_new_2s-12s.npz`. It is the shortest of the dances by far, and the board
loops it.

No music ships with it, and training does not need any. The performance video
`scripts/export.py` renders does, so export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_maze",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's maze dance; the clip is "
                "committed in tasks/jumper/dance_maze/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
