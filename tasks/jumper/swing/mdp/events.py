"""Starting an episode: the swing somewhere in its arc, the robot standing on it.

**The two have to be placed together or neither is placed at all.** mjlab's
`reset_base` puts the robot at a fixed offset from its environment origin, which
is a point in mid-air once the plank has swung out from under it -- and the
default range scatters it half a metre in x and y and a full turn in yaw, which on
a 0.44 x 0.50 m plank is a robot thrown off before the first step. So this term
replaces `reset_base` outright rather than running beside it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from mjlab.managers.event_manager import requires_model_fields

from scenes.swing import BEAM_Z, SWING_X, rope_for, seat_drop

from . import state


@requires_model_fields("tendon_range")
def reset_on_the_swing(
    env: Any,
    env_ids: torch.Tensor | None,
    angle_range: tuple[float, float],
    length_range: tuple[float, float],
    robot_name: str = "robot",
    entity_name: str = "swing",
) -> None:
    """Rig the swing, put it at a random point of its arc, stand the robot on it.

    ## A different swing every episode

    `length_range` is drawn from uniformly, per environment, and written into the
    two upper ropes' `tendon_range` -- a per-world model field, so one compiled
    model hangs a different pendulum in every environment of the batch. The bridle
    is left alone: lengthening a swing means paying out rope at the beam, not
    re-rigging the seat (`scenes/swing.py`).

    **Length and pose are drawn in the same function on purpose.** They are one
    initial condition -- where the seat is released from is `L` times a direction
    -- and splitting them across two event terms would put the whole thing at the
    mercy of the order the event manager happens to iterate its dict in. Nothing
    would raise: the pose would simply be computed for the length the *previous*
    episode had, every rope would start over-stretched, and what comes out is a
    seat that snaps on the first step and reads as a swing that was pushed.

    The reset is what makes `rope_length` true, so the two must not disagree; this
    writes the model and every downstream term reads it back.

    ## A random point of the arc

    `angle_range` is a range of release magnitudes, drawn from **uniformly** with a
    random sign; `env_cfg.py`'s `RESET_ANGLE_RANGE_DEG` has the reasoning and the
    measurements. Every amplitude between the ends is reachable, which is what
    separates this from the three-point table it replaced -- the critic is fitting a
    value function over a continuum and a draw that only ever visits three points of
    it has to interpolate the rest.

    The angle is an angle and not a height, so it means the same thing on every
    length: 30 degrees is 30 degrees on a 0.7 m rope and on a 1.8 m one, and the
    energy it corresponds to is what differs. That is the right way round -- the
    task's objective is an angle too.

    ## Released at the turning point, on purpose

    The seat is placed at angle `theta` with **zero velocity** -- the far end of
    the arc, where a real swing is let go from.

    That is not only realism, it is the one placement that does not fight the
    ropes. Six one-sided rope constraints have a configuration manifold, and a
    state off it is a state with every rope over-stretched: measured while
    calibrating the swing's damping, displacing the seat along `x` alone cost 21%
    of the amplitude in the first half-swing as the constraints unloaded, which
    reads as heavy damping rather than as a bad initial condition. Rotating the
    whole assembly about the beam keeps every rope at exactly its own length.

    ## Everything is computed, nothing is read

    The seat's world pose is **not** read back from `data` to place the robot on
    it, and that is deliberate: this runs inside the reset, before the forward
    pass that would make those buffers describe the state being written, so
    reading them would place the robot on the *previous* episode's deck. The whole
    pose comes from `env_origins` and the geometry constants instead, exactly as
    `reset_root_state_uniform` builds the prop's own mocap pose.
    """
    from mjlab.envs.mdp.events import resolve_env_ids

    env_ids = resolve_env_ids(env, env_ids)
    swing = env.scene[entity_name]
    robot = env.scene[robot_name]
    device = env.device
    n = len(env_ids)

    # Uniform in the **magnitude**, with the sign drawn separately: a swing released
    # forward and one released back are the same episode mirrored, so this covers
    # `+/-high` from a range that only has to be written once.
    #
    # With the low end at zero it is the same distribution as a uniform draw over
    # `(-high, high)` -- the magnitude of that is uniform too -- and the difference
    # only appears when the low end is not zero, which is exactly the case the
    # `--swing-angle` flag exists for. `--swing-angle 20` has to mean 20 degrees
    # every episode, and a signed draw would quietly turn it into "somewhere
    # between -20 and 20", which is a pin that does not pin.
    #
    # `low_a` is added rather than dropped for the same reason. `high_a * rand` is
    # right for the default range and silently wrong for every pinned one.
    low_a, high_a = angle_range
    magnitude = low_a + (high_a - low_a) * torch.rand(n, device=device)
    sign = torch.randint(2, (n,), device=device, dtype=torch.float) * 2.0 - 1.0
    theta = magnitude * sign
    sin, cos = torch.sin(theta), torch.cos(theta)

    # ── The rigging ──────────────────────────────────────────────────────
    # Uniform in length rather than in period. The two are within a square root of
    # each other over this range, so the distinction is small; length is what the
    # rig is built in and what `HANG_L_RANGE`'s bounds are stated in.
    low, high = length_range
    hang = low + (high - low) * torch.rand(n, device=device)
    field = env.sim.model.tendon_range
    for tendon in state.rope_ids(env, entity_name):
        field[env_ids, tendon, 1] = rope_for(hang)
    # Beam to the plank's **centre**, which is what every pose below is measured
    # against; `hang` is to its top face, which is what a rope is cut to.
    drop = seat_drop(hang)

    # ── The swing ────────────────────────────────────────────────────────
    # A rigid rotation of the seat about the beam is, in the seat's own six
    # coordinates, a slide and a pitch: rotating (0, 0, -drop) by theta about y
    # lands at (-drop sin, 0, -drop cos), and the slides are measured from the
    # model's own rest pose, which hangs the seat at the nominal `REST_L`. The
    # pitch has to come with them or the deck stays level while its ropes tilt,
    # which over-stretches the bridle -- and looks, in a single frame, exactly
    # like a swing.
    #
    # **The four ropes need no correction of their own, and that is a property of
    # the rig rather than an omission.** Each tie point rides the seat's pitch, so
    # rotating the whole assembly about the beam leaves every rope at exactly the
    # `rope_for(hang)` just written -- at any length and any release angle. A rig
    # with a fitting at a fixed height would not have that property, which is what
    # the knotted version this replaced had to work around.
    names = swing.joint_names
    q = torch.zeros((n, len(names)), device=device)
    q[:, names.index("seat_slide_x")] = -drop * sin
    q[:, names.index("seat_slide_z")] = state.REST_L - drop * cos
    q[:, names.index("seat_pitch")] = theta
    swing.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)

    # ── The robot ────────────────────────────────────────────────────────
    origin = env.scene.env_origins[env_ids]
    pivot = origin + torch.tensor([SWING_X, 0.0, BEAM_Z], device=device)

    # The deck's rotation is R_y(theta); its columns are where the robot's own
    # axes end up, so the standing offset goes through it too.
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    rot = torch.stack(
        [
            torch.stack([cos, zero, sin], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sin, zero, cos], dim=-1),
        ],
        dim=1,
    )
    down = torch.zeros((n, 3), device=device)
    down[:, 2] = -drop
    seat = pivot + torch.einsum("nij,nj->ni", rot, down)
    pos = seat + torch.einsum(
        "nij,j->ni", rot, torch.tensor(state.DECK_OFFSET, device=device)
    )
    quat = torch.stack(
        [torch.cos(theta / 2), zero, torch.sin(theta / 2), zero], dim=-1
    )

    root = torch.zeros((n, 13), device=device)
    root[:, 0:3] = pos
    root[:, 3:7] = quat
    robot.write_root_state_to_sim(root, env_ids=env_ids)


def radians(degrees: tuple[float, float]) -> tuple[float, float]:
    """Convert a release range, so `env_cfg.py` can write the degrees it was chosen
    in and this file can take the radians it needs."""
    low, high = degrees
    return math.radians(low), math.radians(high)
