"""Where the swing is, and where the robot is on it.

Every term in this task -- reward, observation, termination and the reset event --
needs some part of the same small piece of geometry, so it is computed in one
place and read from there. The alternative is four files each deriving the swing
angle slightly differently, and a disagreement between them that shows up as a
policy optimising something other than what the termination measures.

## The swing has no single angle, so this measures energy instead

`scenes/swing.py` says it outright: the seat hangs on six ropes with twelve
degrees of freedom, and no `qpos` is "the swing angle". Worse, `seat_pitch` keeps
counting past 90 degrees, so a plank that has been round once reads 400.

What is well defined is the seat's **specific energy above rest** -- how high it
would get if you let go:

    h = |rel| - drop      how far the seat is above the bottom of its own arc
    E = g h + v^2 / 2     per unit mass, so no mass appears anywhere
    A = arccos(1 - E / (g |rel|))          the amplitude that energy is worth

`rel` is the seat's offset from the beam and `drop` is how far below the beam it
hangs, so `h` is exactly zero whenever the seat is directly under the pivot --
**including when the ropes have stretched**, which is why `h` is written as a
difference of two lengths that both stretch rather than as `L(1 - cos theta)`
against a nominal `L`. The rig sags 2.1 mm empty and 4.8 mm loaded
(`scenes/swing.py`), and a nominal length would read that sag as swing.

`A` folds position and velocity into one number, which is what makes it usable as
a reward: at the bottom of the arc the seat is level but moving fastest, and an
amplitude read from position alone would score that as zero.

## Every environment's swing is a different length

`jumper.swing` draws a rope length per episode, so `REST_L` is the *nominal* and not
the pendulum. Almost nothing here has to care, and that is not luck: `length` is
taken from `|rel|` every time it is needed, because the ropes stretch and a
nominal length would read the sag as swing. That measurement was already
length-free before there was more than one length, and the terms built on it --
`swing_amplitude`, `sway_amplitude`, `swing_rate`, and every deck-frame quantity
-- needed no change at all.

What does care is anything that has to *put* the swing somewhere or express it as
a fraction: the reset pose, and the two observation terms that normalise. Those
call `rope_length`.

## The deck frame

The robot must be scored against the **deck**, not against gravity. A swing at 20
degrees carries a deck at 20 degrees, and a robot standing correctly on it is 20
degrees off vertical -- so `mdp.upright` and `mdp.bad_orientation`, which the
velocity tasks use, both read a robot doing exactly the right thing as a robot
about to fall over. `deck_axes` and `robot_on_deck` are what replace them.
"""

from __future__ import annotations

from typing import Any

import torch

# The swing's geometry is read from the scene module rather than copied, so that
# moving the swing cannot leave this task measuring against the old height. The
# rig has already been rescaled once, by a factor of three.
from scenes.swing import (
    BEAM_Z,
    BOARD_T,
    BOARD_TOP_Z,
    BOARD_X,
    BOARD_Y,
    HANG_L,
    ROPES,
    STANCE_CENTRE_X,
    hang_for,
    seat_drop,
)

from ...common.constants import STAND_Z

#: The seat body's rest distance below the beam **at the nominal length**.
#:
#: Every environment hangs its own seat at its own length (`rope_length`), so this
#: is not the pendulum -- it is what the scene compiles, and what a task that does
#: not randomise gets. It survives as the scale for the viewer and as the value the
#: model's own `qpos0` corresponds to, which is what the reset writes its offsets
#: against.
REST_L = seat_drop(HANG_L)

#: Where the robot stands on the deck, in the deck's own frame.
#:
#: `STANCE_CENTRE_X` is **negated**: the robot's footprint is centred 0.036 m
#: ahead of its base, so the base goes that far behind the plank's centre to put
#: the six feet in the middle of it. Placing the base at the centre instead leaves
#: the front feet 0.036 m nearer the edge than the rear ones, on a plank whose
#: whole margin is 0.05 m.
DECK_OFFSET = (-STANCE_CENTRE_X, 0.0, BOARD_T / 2 + STAND_Z)

_GRAVITY = 9.81


def swing_entity(env: Any, name: str = "swing"):
    return env.scene[name]


def seat_row(env: Any, name: str = "swing") -> int:
    """The seat's index in the entity's body rows.

    **Resolved by name, never hardcoded.** mjlab wraps a fixed-base entity in a
    `mocap_base` body, so the rows are `('mocap_base', 'frame', 'seat', 'knot_l',
    'knot_r')` and the seat is row 2 -- but that ordering is an artefact of the
    wrapper, and `root_link_pos_w` describes the wrapper rather than the seat,
    which is the trap this exists to keep anyone from walking into.
    """
    return env.scene[name].body_names.index("seat")


def rope_ids(env: Any, name: str = "swing") -> list[int]:
    """The two upper ropes' rows in the **simulation's** tendon arrays.

    `entity.tendon_names` is local to the entity and `indexing.tendon_ids` maps
    those to the whole scene's model, which is what `env.sim.model` is indexed by.
    Going through the entity rather than counting is the same rule `seat_row`
    states for bodies: the numbers are an artefact of what else is in the scene.
    """
    entity = env.scene[name]
    names = entity.tendon_names
    return [int(entity.indexing.tendon_ids[names.index(r)]) for r in ROPES]


def rope_length(env: Any, name: str = "swing") -> torch.Tensor:
    """How long this environment's swing is: beam to the plank's centre. (N,)

    **Read back from the model rather than remembered.** The length is a per-world
    model field that `reset_on_the_swing` writes, and reading it here means the
    number the reward normalises by and the number the physics integrates are the
    same object -- there is no shadow copy to drift out of step with a reset that
    was skipped, re-ordered, or run on a subset of environments.

    Only `rope_lf` is read. All four ropes are written together and a swing whose
    ropes disagree is not a length at all, it is a twisted rig; `tests/` pins that
    they agree rather than this function averaging over a state that should not
    exist.

    Broadcast to `num_envs` when the field has not been expanded per world, which
    is what a config with no length randomisation leaves it as -- mjlab only
    allocates real per-world memory for fields some event declares.

    **Returned as a plain tensor, and that is not tidying.** The native backend
    hands out model fields as `torch.Tensor` subclasses (`_ExpandedView`, and
    `_SharedView` for the fields it has not expanded, which raises on any in-place
    write). Arithmetic keeps the subclass, so a length read from here and carried
    into a reward stays one -- and the first thing that fails is `float()` or an
    f-string, deep in a `play` readout, complaining about `__format__` on a class
    nobody downstream has heard of. Dropping it here costs a view and keeps the
    backend's business at the backend.
    """
    field = env.sim.model.tendon_range
    rope = field[:, rope_ids(env, name)[0], 1]
    if rope.shape[0] != env.num_envs:
        rope = rope.expand(env.num_envs)
    return (hang_for(rope) + BOARD_T / 2).as_subclass(torch.Tensor)


def pivot_w(env: Any, name: str = "swing") -> torch.Tensor:
    """The beam, in world coordinates, per environment. Shape (N, 3).

    The prop's root is its mocap base, which `registry._place_props` puts at each
    environment's own origin plus `SWING_X` -- so this is the one place the
    per-environment offset enters, and everything downstream is a difference
    against it and therefore origin-free.
    """
    root = env.scene[name].data.root_link_pos_w
    return root + torch.tensor([0.0, 0.0, BEAM_Z], device=root.device)


def swing_offset(env: Any, name: str = "swing") -> torch.Tensor:
    """The seat's offset from the beam, in the frame's coordinates. Shape (N, 3).

    The mocap base is never rotated, so subtracting it gives frame coordinates
    directly: `x` is along the swing, `y` is along the beam, `z` is down-negative.
    """
    entity = env.scene[name]
    return entity.data.body_link_pos_w[:, seat_row(env, name)] - pivot_w(env, name)


def swing_velocity(env: Any, name: str = "swing") -> torch.Tensor:
    """The seat's linear velocity, world frame. Shape (N, 3).

    The frame is welded, so world and frame velocities are the same thing.
    """
    entity = env.scene[name]
    return entity.data.body_link_lin_vel_w[:, seat_row(env, name)]


def _plane_energy(rel: torch.Tensor, vel: torch.Tensor,
                  axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The seat's specific energy above the bottom of its own arc, in one vertical
    plane, and the radius it was measured on. Shapes (N,) and (N,).

    This is the quantity the module docstring describes, and it is factored out
    because **two terms read it and they must not derive it twice**: the amplitude
    it is worth, and the speed it is worth at the bottom. Those are the same
    measurement in two currencies -- `v = sqrt(2 E)` and `A = arccos(1 - E / gL)`
    -- so a second derivation is a second set of clamps and a second chance for
    the reward and the termination to disagree about what the swing is doing.

    **The plane is the point.** Written against the full 3-vector -- which is how
    this started -- the measure cannot tell a swing from a sideways sway, and the
    two score identically. The reward would then be indifferent between the thing
    the task is named after and a mode that is easier to excite and useless, and a
    policy optimising an indifferent reward picks whichever is cheaper.

    Splitting by plane is exact for a pure mode: with `rel_y = 0`, the lateral
    length is `-rel_z` and its height comes out exactly zero, and the same the
    other way round. The one quantity genuinely shared is `v_z` -- both modes lift
    the seat -- and it is counted in both, which over-states each slightly when
    both are running. That errs towards reporting *more* sway, which is the safe
    direction for something used as a penalty.
    """
    length = torch.hypot(rel[:, axis], rel[:, 2]).clamp(min=1e-4)
    height = length + rel[:, 2]  # rel[:, 2] is negative: this is L - drop
    speed_sq = vel[:, axis].pow(2) + vel[:, 2].pow(2)
    return _GRAVITY * height + 0.5 * speed_sq, length


def _plane_amplitude(rel: torch.Tensor, vel: torch.Tensor, axis: int) -> torch.Tensor:
    """Amplitude in one vertical plane: `axis` against z. Shape (N,).

    Zero when the seat hangs still; `pi/2` when it has enough energy to reach the
    height of the beam. Saturated there rather than allowed to go complex, so a
    swing that does go over the top reports its maximum instead of a NaN that
    would propagate into the reward and quietly kill a run.
    """
    energy, length = _plane_energy(rel, vel, axis)
    return torch.arccos((1.0 - energy / (_GRAVITY * length)).clamp(-1.0, 1.0))


def swing_amplitude(env: Any, name: str = "swing") -> torch.Tensor:
    """How hard the swing is swinging **fore and aft**, in radians. Shape (N,).

    The x-z plane: along the swing, which is the axis the A-frames splay along and
    the only one the rig is built to travel in.
    """
    return _plane_amplitude(swing_offset(env, name), swing_velocity(env, name), 0)


def swing_bottom_speed(env: Any, name: str = "swing") -> torch.Tensor:
    """How fast the deck goes through the bottom of its arc, m/s. Shape (N,).

    The same fore-aft energy `swing_amplitude` reads, in metres per second instead
    of radians: `v = sqrt(2 E)`. At the bottom that is the deck's actual speed --
    the height term is zero there and only `v^2 / 2` is left -- and everywhere else
    on the arc it is the speed the deck *will* pass the bottom at, which is what
    makes it usable every step rather than once a half-cycle. A term that only paid
    out in the frames near the bottom would be a reward the policy collects a
    handful of times per episode, on the frames it has least control over.

    **It is length-aware, and unlike everything else in this module that is on
    purpose.** `v = sqrt(2 g L (1 - cos A))`, so the same amplitude is worth a lot
    more speed on a long rope: 45 degrees is 1.87 m/s at the short end of
    `HANG_L_RANGE` and 3.22 m/s at the long end, a factor of 1.72. Every other
    measure here was deliberately written to mean the same thing at every length.
    Whoever reads this term into a reward owes a decision about that, and
    `env_cfg.py::TARGET_SPEED` is where it is made.

    `clamp(min=0)` before the root guards a negative energy, which the geometry
    cannot produce but a stretched rope and a fast integrator can: `height` is a
    difference of two lengths that are equal at the bottom, and it lands a few
    microns either side of zero there.
    """
    energy, _ = _plane_energy(swing_offset(env, name), swing_velocity(env, name), 0)
    return (2.0 * energy).clamp(min=0.0).sqrt()


def bottom_speed_for(amplitude: float, hang_l: float) -> float:
    """What `swing_bottom_speed` reads for a free swing of `amplitude` on `hang_l`.

    The inverse of the term above, and it lives here rather than in `env_cfg.py`
    so that the constant a reward normalises by and the quantity it normalises
    cannot be derived from two different pictures of the geometry. `seat_drop` is
    the radius the seat actually hangs at, which is the rope plus half the plank's
    thickness -- that half-centimetre is 0.4% of the shortest rope, and leaving it
    out is the sort of thing that goes unnoticed for good.
    """
    import math

    return math.sqrt(
        2.0 * _GRAVITY * seat_drop(hang_l) * (1.0 - math.cos(amplitude))
    )


def sway_amplitude(env: Any, name: str = "swing") -> torch.Tensor:
    """How hard the seat is swinging **sideways**, in radians. Shape (N,).

    The y-z plane: along the beam. A real swing does this when you get on it badly
    and it is a nuisance rather than a feature -- it puts the ropes out of plane,
    it rolls the deck under the robot's feet, and it is not what "swinging" means.

    It is a mode this rig genuinely has: the seat hangs from two points 0.5 m
    apart on a 1.95 m beam, so sideways is a pendulum of very nearly the same
    length as fore-aft and therefore of very nearly the same frequency -- which is
    exactly why the reward has to name the plane rather than hope to tell them
    apart by their period.
    """
    return _plane_amplitude(swing_offset(env, name), swing_velocity(env, name), 1)


def swing_rate(env: Any, name: str = "swing") -> torch.Tensor:
    """The swing's angular rate about the beam, rad/s, signed. Shape (N,).

    Computed from the seat's position and velocity rather than from `seat_pitch`,
    and **the two now agree**: four ropes straight to the beam hold the deck square
    to the rope, so the deck's pitch rate *is* the rope's angular rate. Measured
    over 120 steps at 35 degrees, correlation 1.00000 and a worst-case difference
    of 0.007 rad/s. Under the knotted rigging this replaced they did not agree --
    the deck was a second pendulum hanging off the first and overshot by 18% -- and
    that disagreement is why this was written geometrically in the first place.

    It stays geometric even so. `seat_pitch` is an Euler angle, so it is singular
    at 90 degrees and keeps counting past it; the cross product of the offset with
    the velocity is the same number with neither problem, and it goes on meaning
    the rope's rate if the rigging changes again. `tests/` pins the agreement
    rather than this function assuming it.
    """
    rel, vel = swing_offset(env, name), swing_velocity(env, name)
    denominator = (rel[:, 0].pow(2) + rel[:, 2].pow(2)).clamp(min=1e-6)
    return (rel[:, 2] * vel[:, 0] - rel[:, 0] * vel[:, 2]) / denominator


def deck_axes(env: Any, name: str = "swing") -> torch.Tensor:
    """The deck's rotation matrix, world frame. Shape (N, 3, 3), columns x/y/z."""
    from mjlab.utils.lab_api.math import matrix_from_quat

    entity = env.scene[name]
    return matrix_from_quat(entity.data.body_link_quat_w[:, seat_row(env, name)])


def body_rate_on_deck(env: Any, robot_name: str = "robot",
                      name: str = "swing") -> torch.Tensor:
    """The robot's angular velocity **with the deck's taken out**, in the robot's
    own frame. Shape (N, 3).

    A robot riding a swing rotates because the swing does. Subtracting the deck's
    own rate is what separates "the seat is turning" from "the robot is wobbling",
    and the two are not close: measured on a robot doing nothing at all, the world-
    frame `wx^2 + wy^2` runs 0.757 at rest and **2.111** on a 35 degree swing, while
    the same quantity relative to the deck runs 0.465 and 0.604. Nearly all of the
    growth is the deck.
    """
    from mjlab.utils.lab_api.math import matrix_from_quat

    robot = env.scene[robot_name]
    deck_w = env.scene[name].data.body_link_ang_vel_w[:, seat_row(env, name)]
    body = matrix_from_quat(robot.data.root_link_quat_w)
    deck_b = torch.einsum("nij,nj->ni", body.transpose(1, 2), deck_w)
    return robot.data.root_link_ang_vel_b - deck_b


def deck_tilt(env: Any, name: str = "swing") -> torch.Tensor:
    """How far the deck is from horizontal, in radians. Shape (N,).

    Against **gravity**, unlike everything else here, because what this is for is
    the one question the deck-relative measures cannot answer: has the seat itself
    gone over. A robot riding a deck that is 75 degrees past level reads perfectly
    square on it, so `tipped_on_the_deck` sees nothing and `robot_on_deck` sees
    nothing -- the whole assembly has left the world the task is about together.
    """
    up = deck_axes(env, name)[:, 2, 2]
    return torch.arccos(up.clamp(-1.0, 1.0))


def robot_on_deck(env: Any, robot_name: str = "robot",
                  name: str = "swing") -> torch.Tensor:
    """The robot's base relative to the plank's centre, in **deck** coordinates.

    Shape (N, 3): `x` fore-aft along the plank, `y` across it, `z` above its top
    face. This is what "on the plank" means -- the world-frame offset is useless,
    because the plank is somewhere else and pointing somewhere else every step.
    """
    entity = env.scene[name]
    seat = seat_row(env, name)
    delta = (env.scene[robot_name].data.root_link_pos_w
             - entity.data.body_link_pos_w[:, seat])
    local = torch.einsum("nij,nj->ni", deck_axes(env, name).transpose(1, 2), delta)
    return local - torch.tensor([0.0, 0.0, BOARD_T / 2], device=local.device)


def deck_up_in_body(env: Any, robot_name: str = "robot",
                    name: str = "swing") -> torch.Tensor:
    """The deck's up-axis, expressed in the robot's own frame. Shape (N, 3).

    `(0, 0, 1)` means the robot is square on the deck whatever the swing is doing;
    it is the deck-relative replacement for `projected_gravity`, and the quantity
    both `deck_upright` and the tip-over termination are built from.
    """
    from mjlab.utils.lab_api.math import matrix_from_quat

    body = matrix_from_quat(env.scene[robot_name].data.root_link_quat_w)
    return torch.einsum("nij,nj->ni", body.transpose(1, 2), deck_axes(env, name)[:, :, 2])


__all__ = [
    "BOARD_TOP_Z",
    "BOARD_X",
    "BOARD_Y",
    "DECK_OFFSET",
    "REST_L",
    "body_rate_on_deck",
    "bottom_speed_for",
    "deck_axes",
    "deck_tilt",
    "deck_up_in_body",
    "pivot_w",
    "robot_on_deck",
    "rope_ids",
    "rope_length",
    "seat_row",
    "sway_amplitude",
    "swing_amplitude",
    "swing_bottom_speed",
    "swing_entity",
    "swing_offset",
    "swing_rate",
    "swing_velocity",
]
