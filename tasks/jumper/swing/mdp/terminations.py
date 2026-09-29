"""When the episode is over: off the plank, or over on the deck."""

from __future__ import annotations

from typing import Any

import torch

from . import state


def off_the_plank(env: Any, margin: float, drop: float,
                  robot_name: str = "robot",
                  entity_name: str = "swing") -> torch.Tensor:
    """The robot's base has left the plank, measured in the plank's own frame.

    Two ways out and both are needed. Sideways: the base is further from the
    plank's centre than half the plank plus `margin` -- generous, because the base
    can lean well past an edge while the feet still hold. Downwards: the base has
    dropped `drop` below the deck, which catches the robot that has slid off and
    is on its way to the ground while still nominally over the plank.

    **In deck coordinates, not world.** The plank is 0.60 m up at rest, swings
    through half a metre of travel and tilts as it goes, so a world-frame box
    around the robot's start is a box the plank leaves and the robot with it.
    """
    offset = state.robot_on_deck(env, robot_name, entity_name)
    outside = (offset[:, 0].abs() > state.BOARD_X / 2 + margin) | (
        offset[:, 1].abs() > state.BOARD_Y / 2 + margin
    )
    return outside | (offset[:, 2] < -drop)


def tipped_on_the_deck(env: Any, limit_angle: float, robot_name: str = "robot",
                       entity_name: str = "swing") -> torch.Tensor:
    """The robot has gone over **relative to the deck**.

    `mdp.bad_orientation`, which the velocity tasks use at 50 degrees, measures
    against gravity. Here the deck itself reaches 24 degrees at the amplitudes
    this task is aiming for and 37 under an off-centre load, so half the budget is
    spent by the swing doing what it is supposed to and the robot would be
    terminated for standing correctly. Measuring against the deck's own up-axis
    restores the meaning the 50 degrees had.
    """
    tilt = state.deck_up_in_body(env, robot_name, entity_name)[:, 2].clamp(-1.0, 1.0)
    return torch.arccos(tilt).abs() > limit_angle


def swing_over(env: Any, limit_angle: float, entity_name: str = "swing") -> torch.Tensor:
    """The seat itself has gone over, whatever the robot on it is doing.

    **Not redundant with the other two, and the gap it fills is exactly the one
    that crashed a run.** Both of those are measured in the deck's own frame: a
    robot glued squarely to a plank that is 75 degrees past level reads as
    perfectly upright and perfectly centred, so neither fires, and the episode
    carries on into a region the model is not built for. The seat's three hinges
    are Euler angles and one of them is singular at 90 degrees -- roll now, and it
    was pitch, which is what produced a NaN in `qpos` and a training run that died
    reporting only "the observation contains NaN".

    Reordering the hinges moved the singularity 72 degrees away from anything the
    model does. This is the second line: a swing this far over is not a swing, the
    robot on it has lost whatever it was doing, and there is nothing left in the
    episode worth sampling.
    """
    return state.deck_tilt(env, entity_name) > limit_angle
