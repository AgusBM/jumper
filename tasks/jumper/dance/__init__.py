"""jumper.dance -- learn one fixed choreography by imitation.

Unlike the four velocity tasks, this one has no command to follow and no gait
prior. It has a **clip**: a recorded dance the policy is scored against frame by
frame, plus four things asked of it on top of following the steps -- stay upright,
draw little power, move smoothly, and keep torque peaks below what the servo can
hold. `env_cfg.py` states each of those as its own reward term.

## It needs material, and a demonstration clip is committed

A clip is the one thing this task cannot be configured into existence without, so
`media/demo.{npz,mp4}` ships with the repository and `--task jumper.dance` runs
from a fresh clone like every other task. The music the demo was made for does
not ship -- its source could not be established -- and training does not need it;
only the performance video `scripts/export.py` renders does. Your own choreography
replaces the demo: drop it in `media/` and take the demo out -- see
`media/README.md`.

Missing or ambiguous material is an error naming the directory and the file kind,
raised while the environment is being built, rather than a run that starts and
trains on nothing.

## Where the parts come from

The imitation machinery is mjlab's `mjlab.tasks.tracking`, a re-implementation of
BeyondMimic that ships in the vendored tree and that nothing here had used. It is
imported, not copied and not modified: `rl/` stays word-for-word with upstream.
What this repository adds -- reading the clip, naming the observations so the
policy stays exportable, the torque-peak term and the environment built from them
-- is in `tasks/jumper/common/dance/`, shared with the other dance tasks since
there were several. This directory holds the clip, the numbers and the
performance video.
## Not deployable, and the export says so

`jumper.dance` has no `controls.yaml` and should not: nobody drives a dance with
a stick. Its command is a recorded clip, and the five terms that make it one --
`clip_phase`, `ref_joint_pos`, `ref_joint_vel`, `ref_tilt_error`, `ref_future` --
are terms the on-robot observation builder has no case for. `scripts/export.py`
refuses the export for that reason, by name.

That refusal is new. `_DEPLOY_TERMS` used to claim all five, so the export would
have succeeded and the robot would have refused the bundle on the bench.

Selecting a dance on the robot, when there is one to select, is a **mode
switch** and belongs in a deploy manifest as a `[[fsm.button]]` -- `menu` plus a
direction, which is what the C++ used and what the `with =` modifier exists for.
Not here: which button enters a mode is a property of a bundle, not of a task.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.dance",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating one recorded choreography; needs material "
                "in tasks/jumper/dance/media/ (see its README)",
    tags=("imitation", "jumper", "dance", "flat"),
)
