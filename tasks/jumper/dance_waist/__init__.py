"""jumper.dance_waist -- the waist (waist-twist) dance, from rl-wbc-fsm, learned by imitation.

The same task as `jumper.dance` with a different clip: the environment is
`tasks/jumper/common/dance/env.py`, and what this directory adds is the clip and
the numbers.

## The clip

`media/waist.npz`, written by `tools/import_wbc_dances.py` from the `q_cmd` table of
rl-wbc-fsm's `policy/hexa/dance/dance_waist.reference_clip.json`, 56.57 s at 200 Hz.
On rl-wbc-fsm the board plays it as open-loop playback of these rows, as the FSM
state `dance_waist`.

The joint angles are the table's, taken as the state, and the base pose, which the
table does not have, is solved from them. The tool's docstring has how, and the
control it was checked against.

The table is joint angles only, and the npz it was cut from is lost: its note names
`dance_choreo6_record.npz`, and the file of that name in motion-planner does not
contain it (0.1-0.9 rad apart wherever the two are aligned). So this clip is the
rows the board plays and nothing more.

No music ships with it, and training does not need any. The performance video
`scripts/export.py` renders does, so export with `--no-video`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance_waist",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating rl-wbc-fsm's waist dance; the clip is "
                "committed in tasks/jumper/dance_waist/media/",
    tags=("imitation", "jumper", "dance", "flat"),
)
