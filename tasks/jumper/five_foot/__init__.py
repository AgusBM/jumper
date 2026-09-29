"""jumper.five_foot -- walking on five legs, with the left-front arm carried as a claw.

The other four jumper tasks differ from each other by a gait prior and nothing
else. This one differs by a **leg**: the left-front 5-DoF arm is taken out of the
action and held stowed against the trunk, so the policy has to walk, turn and
stand on the remaining five feet. On the robot the d-pad swings the arm out to a
preset and the trigger closes the claw (`deploy/lib.rs`).

Ported from kk-rl-lab's `hexa_5d/locomotion_5foot`, which trains the same idea on
Isaac Lab. The two share the asset exactly -- every mesh is byte-identical and
every joint has the same axis and range -- so the arm's hold pose crosses over
unchanged; what does not cross over is the reward set, which was written against
a skeleton that had far fewer terms of its own. `mdp/rewards.py` has the table.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds what belongs to this task: its fixed name (the register call
below), the claw's geometry and joint sets (`claw.py`), its environment config
(`env_cfg.py`), its hyper-parameters (`rl_cfg.py`), and the MDP terms only it uses
(`mdp/`).

**Not a control group for the other four**, and not comparable to them: the
action is 16 joints rather than 20, the observation is 21 joints rather than 20,
the contact sensor has five columns rather than six, and symmetry augmentation is
off because the robot is no longer left-right symmetric. Nothing about a number
measured here transfers to `jumper.flat` or back.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.five_foot",
    assets=JUMPER_ASSETS,
    description="jumper hexapod velocity tracking on five feet; the left-front arm is carried stowed, as a claw",
    tags=("locomotion", "jumper", "flat", "five_foot", "manipulation"),
)


def cli_args(group):
    """This task's own command-line arguments; see `tasks.load_cli_args`.

    Both are replay's, and both used to live elsewhere: `--objects` was a scene
    (`--scene objects`) and `--hold` a flag in `scripts/play.py`. They moved here
    when the task moved everything it needs inside its own directory -- a flag
    that names this claw's props has no business in the parser train, play and
    export share (`tests/test_log_layout.py`), and the scene registry would have
    needed five fields that only this row used.

    Lives here rather than in `env_cfg.py` because `__init__.py` is the module of a
    task that is always imported and must stay free of simulation dependencies
    (`tests/test_registry.py` checks that in a subprocess). Declaring an argument
    needs `argparse` and nothing else.

    Returns the `dest` names it added, so the entry point knows which parsed values
    belong to this task.
    """
    group.add_argument(
        "--objects",
        action="store_true",
        help="put the pick-place row in front of the robot: a bar of soap, a "
        "notebook, a can and a bin, under the friction model they are held with. "
        "Physical, so replay only, and every environment gets its own row -- pass "
        "--num_envs 1 to look at one robot",
    )
    group.add_argument(
        "--hold",
        default=None,
        metavar="PROP",
        help="start the episode with this prop already in the claw, placed on the "
        "point the claw could actually have closed on. Needs --objects",
    )
    return ("objects", "hold")
