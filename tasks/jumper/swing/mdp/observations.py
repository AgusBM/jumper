"""What the **critic** sees of the swing. The actor does not get any of it.

The actor runs on proprioception alone -- see the block in `env_cfg.py` that
mounts these -- because the swing turns out to be in the IMU already: the deck
hangs perpendicular to the rope, the robot stands square on the deck, so
`projected_gravity` carries the swing angle and the gyro carries its rate. What
is left over is genuinely privileged: where the plank is under the robot's feet,
and which way its surface points, neither of which a robot can feel.

**All four terms are three-vectors, and that is not a coincidence.** Symmetry
augmentation is on by default for this robot family (`common/ppo.py`), and
`common/mdp/symmetry.py` raises at startup for any observation term with no
mirror rule -- so a new term either declares a `mirror_kind` that already exists
or the shared table has to grow an entry. Every quantity here is a genuine vector
in a frame that mirrors with the robot (the swing's frame, the deck's frame, the
body's frame), so `"vec3"` -- negate `y` -- is the correct rule for all of them and
nothing shared has to change.

`mirror_kind` is accepted and discarded by every function here, which is the
convention `common/mdp/phase.py` sets: `symmetry.py` reads it off the term's
`params`, and `params` are also handed to the function as keywords, so a term that
declares one and does not accept it fails while the observation manager is
measuring the term's width.

That is also *why* the swing state is given as a vector rather than as the pair
(angle, angular rate) it is really about: two fore-aft scalars have no mirror kind
in the table, and adding one for them would have been a change to a file four
other tasks depend on, to express less than the vector does. The vector carries
the lateral sway as well, which the scalar pair drops.
"""

from __future__ import annotations

from typing import Any

import torch

from . import state


def swing_offset(env: Any, entity_name: str = "swing",
                 mirror_kind: str | None = None) -> torch.Tensor:
    """Where the seat is in its arc, as a fraction of the rope's length.

    `(0, 0, -1)` is hanging still at the bottom, **on a swing of any length**, and
    that is the point of dividing: the critic reads one shape across a batch whose
    ropes differ by a factor of 2.6, rather than a raw offset that means "at the
    bottom" at -0.71 in one environment and -1.81 in the next.

    Per environment (`state.rope_length`), not by the nominal. Dividing a batch of
    different swings by one constant would leave exactly the length information
    this normalisation exists to take out -- and would leave it in the *sign* of
    the residual, which is the most learnable form it could take.
    """
    del mirror_kind  # read off params by symmetry.py; see common/mdp/phase.py
    length = state.rope_length(env, entity_name).unsqueeze(-1)
    return state.swing_offset(env, entity_name) / length


def swing_velocity(env: Any, entity_name: str = "swing",
                   mirror_kind: str | None = None) -> torch.Tensor:
    """How fast the seat is going, and which way.

    **Half of the phase, and the half that position cannot supply.** Pumping is
    phase-locked -- measured on a servoed rider, leaning in time with the seat's
    velocity holds 12 degrees on 20 mm of travel while the same travel out of
    phase does nothing -- and at the bottom of the arc, where the drive matters
    most, position is zero and velocity is everything.

    Scaled by `sqrt(g * L)`, the speed a pendulum has at the bottom of a
    quarter-circle swing, so the number is order one across the useful range --
    and with `L` per environment, across a batch of different swings too. A long
    swing is genuinely faster at the bottom for the same angle, and that is the
    part of the difference this scaling removes; what is left is the phase, which
    is what the term is for.
    """
    del mirror_kind
    scale = torch.sqrt(9.81 * state.rope_length(env, entity_name)).unsqueeze(-1)
    return state.swing_velocity(env, entity_name) / scale


def deck_up(env: Any, robot_name: str = "robot", entity_name: str = "swing",
            mirror_kind: str | None = None) -> torch.Tensor:
    """The deck's up-axis in the robot's own frame: `(0, 0, 1)` is square on it.

    The observation `projected_gravity` is already in the actor group and measures
    the body against gravity, which on a swing is a different question -- the deck
    tilts with the arc, so the two disagree by the swing angle. The policy needs
    both: gravity tells it where down is, this tells it where the floor is.
    """
    del mirror_kind
    return state.deck_up_in_body(env, robot_name, entity_name)


def deck_offset(env: Any, robot_name: str = "robot", entity_name: str = "swing",
                mirror_kind: str | None = None) -> torch.Tensor:
    """Where the robot is standing on the plank, in the plank's frame, in metres.

    Not normalised: the plank is 0.442 x 0.500 m, so these are already tenths, and
    the quantity the robot cares about is its absolute distance from an edge it
    can fall off rather than a fraction of a plank it cannot measure.
    """
    del mirror_kind
    return state.robot_on_deck(env, robot_name, entity_name)
