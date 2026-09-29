"""jumper.swing -- stand on a swing's plank seat and pump it.

The robot starts standing on the seat of the swing built by `scenes/swing.py` and
is rewarded for how hard the swing is swinging. It has to stay on a deck that
tilts, accelerates and drops away underneath it, and it has to move its weight in
time with the swing to put energy in.

**A different swing every episode.** The rope length is drawn per environment from
the whole range the frame can be rigged at, so the period runs 1.50 to 2.65 s
across a batch and a policy cannot learn one rhythm and call it swinging. The
only thing that generalises across that is closing the loop on what the swing is
actually doing -- which the robot can feel, since the deck hangs perpendicular to
the rope and its own IMU therefore carries the swing angle and rate. Nothing was
added to the actor's observations to make this work; see `ROPE_LENGTH_RANGE`.

**This task builds the swing itself** rather than requiring `--scene swing`. A
task that needs a prop and does not build it would train perfectly happily on an
empty field the moment someone forgot the flag. The consequence is the other way
round: `--task jumper.swing --scene swing` is an **error**, because the scene would
be adding a second swing over the top of this one. Any look-only scene is fine.

The robot definition and the environment skeleton live in `tasks/jumper/common/`.
This directory holds only what belongs to this task: its fixed name (the register
call below), its environment config (`env_cfg.py`), its hyper-parameters
(`rl_cfg.py`), and the MDP terms only it uses (`mdp/`).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.swing",
    assets=JUMPER_ASSETS,
    description="jumper hexapod stands on a rope swing's plank seat and pumps it",
    tags=("jumper", "swing", "balance", "prop"),
)


def cli_args(group):
    """This task's own command-line arguments; see `tasks.load_cli_args`.

    `--swing-angle` is a task's knob, so it is declared by the task. Putting it in
    `scripts/_cli.py` -- one parser shared by train, play and export -- would be a
    task name in a file that is not allowed any, which is the layering
    `tests/test_log_layout.py` pins and the one `play.py` was quietly breaking to
    print a commanded speed.

    Lives here rather than in `env_cfg.py` because `__init__.py` is the module of a
    task that is always imported, and it must stay free of simulation
    dependencies -- `tests/test_registry.py` checks that in a subprocess. Declaring
    an argument needs `argparse` and nothing else, so nothing is imported here that
    was not already.

    The default is `MJRL_SWING_ANGLE`, which is how every other option in this
    repository reaches `.env`: the command line wins, then the shell, then
    `.env.local`, then `.env`.

    Returns the `dest` names it added, so the entry point knows which parsed values
    belong to this task without introspecting argparse's private structures.
    """
    import os

    group.add_argument(
        "--swing-angle",
        default=os.environ.get("MJRL_SWING_ANGLE", "").strip() or None,
        metavar="DEG[,DEG]",
        help="release the swing from this angle, in degrees; a pair is a range "
        "drawn from uniformly per episode, with the sign randomised either way. "
        "Falls back to MJRL_SWING_ANGLE in .env, then to 0,45 when training and 0 "
        "when replaying",
    )
    group.add_argument(
        "--swing-length",
        default=os.environ.get("MJRL_SWING_LENGTH", "").strip() or None,
        metavar="M[,M]",
        help="rope length in metres, beam to the plank; a pair is a range drawn "
        "from per episode. Falls back to MJRL_SWING_LENGTH in .env, then to the "
        "whole 0.6-1.8 m the frame can be rigged at. One value pins every swing "
        "to it, which is how to see whether a policy really generalises",
    )
    return ("swing_angle", "swing_length")
