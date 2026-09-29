"""What this task pays for: a swing that is swinging, and a robot still on it."""

from __future__ import annotations

from typing import Any

import torch

from . import state


def swing_amplitude(env: Any, target_angle: float, over_std: float,
                    entity_name: str = "swing"):
    """The objective. Amplitude as a fraction of the target, peaking there.

    **Amplitude, not amplitude gained.** Rewarding the increase would be the
    textbook shaping -- it is potential-based, so it cannot change which policy is
    optimal, and it pays out the instant the robot does anything useful. It is
    also a derivative of a quantity that a contact-rich simulation makes noisy,
    and it pays a robot that pumps hard and then falls off exactly as much as one
    that pumps hard and stays. Paying for the level instead makes holding a high
    swing worth as much as reaching it, which is what "swinging" means, and the
    bootstrap problem it creates is solved on the other side: `reset_on_the_swing`
    starts a fresh episode at a random amplitude, so there is signal from the
    first step whether or not the policy has learnt to pump yet.

    Saturating rather than linear because the swing is not a thing to maximise
    without limit -- past about 90 degrees the ropes go slack and the seat is in
    free fall, which is not a swing and is certainly not something to stand on.

    **And falling off above the target rather than merely flat, because overshoot
    turns out to be the easy failure and not the hard one.** Measured on the
    open-loop rig: a rider that keeps the phase drives every length in
    `HANG_L_RANGE` past 85 degrees on as little as +/-20 mm of travel, a fifth of
    what this robot has. So a policy that learns to pump at all learns something
    that, left alone, ends the episode on `swing_over`.

    A plateau with a cliff behind it is a bad shape to learn: the gradient is zero
    from the target up to the termination, so nothing tells the policy to stop
    until the episode does, and what the run shows is a reward that climbs and then
    collapses without a term to blame. `over_std` puts a Gaussian shoulder there
    instead -- full marks at the target, and a slope back down towards it -- which
    is the same deadband-then-Gaussian shape `deck_upright` uses, for the same
    reason.

    **How far the shoulder has to reach has changed twice and the parameter has
    not had to move either time**, which is a coincidence rather than a property.
    `swing_over` fires on the deck's tilt at 70 degrees. The deck used to overshoot
    the rope by 18%, so that meant an amplitude of about 59 and `over_std = 0.30`
    put it 1.4 shoulder-widths above a 35 degree peak. Four ropes straight to the
    beam hold the deck square to the rope -- measured, slope 1.000 -- so 70 degrees
    of deck became 70 of amplitude and the gap stretched to 2.0 widths. Raising the
    peak to 45 spends that back to 1.45. The term now reads 0.71 at 55 degrees,
    0.47 at 60 and 0.12 at the termination: the reward turns the policy back a
    good 15 degrees before the episode does, which is the right way round. Move
    either the peak or the limit again and this has to be recomputed rather than
    assumed.

    Regulating rather than maximising is also what the task is: on a short rope the
    policy has to reach the target and **hold**, and on a long one it has to work
    for it. Those are the same skill only if the reward peaks.
    """
    amplitude = state.swing_amplitude(env, entity_name)
    below = (amplitude / target_angle).clamp(0.0, 1.0)
    over = ((amplitude - target_angle).clamp(min=0.0) / over_std).pow(2)
    return below * torch.exp(-over)


def deck_speed_through_the_bottom(env: Any, target_speed: float,
                                  entity_name: str = "swing"):
    """The deck's speed through the bottom of the arc, as a fraction of a target.

    `clamp(v / target_speed, 0, 1)`. Paid every step: `state.swing_bottom_speed`
    reads the swing's energy, so off the bottom this is the speed the deck is
    *going* to pass the bottom at rather than a number that is only meaningful for
    the few frames it is actually down there.

    **This is the objective again in a second currency, and the currency is the
    whole of what it adds.** `v = sqrt(2 g L (1 - cos A))` -- a deterministic
    function of the amplitude `swing_amplitude` already pays for and the radius the
    seat hangs at. Normalised per rope it would be a reparametrisation and worth
    nothing. In absolute metres per second it is **length-aware**, and that is the
    point: the same 20 degrees is 0.81 m/s on a 0.6 m rope and 1.40 m/s on a 1.8 m
    one, so a long slow swing gets paid for the energy it is carrying at an angle
    where the amplitude term still reads it as barely started.

    That is aimed at a specific weakness. The long ropes are the hard half of
    `ROPE_LENGTH_RANGE` -- a period of 2.65 s is 133 control steps, and `rl_cfg.py`
    says under `gamma` that the discount bites hardest exactly there. Dense early
    credit is worth most where the payoff is furthest away.

    **The saturation is what keeps it from fighting the peak.** `swing_amplitude`
    falls off above `TARGET_ANGLE`, so a term that went on paying for speed above
    it would be pulling the other way in the one band where the reward is supposed
    to be turning the policy back. Clamped at `TARGET_SPEED` it cannot: past
    saturation it is flat and contributes no gradient in either direction. See
    `env_cfg.py::TARGET_SPEED` for which rope the constant is taken from and why
    it has to be that one.

    Read against the fore-aft plane only, through `swing_bottom_speed`. A sideways
    sway carries real speed and this must not pay for it -- that is the same trap
    `_plane_energy` exists to close, and it is closed once for both currencies
    rather than here as well.
    """
    return (state.swing_bottom_speed(env, entity_name) / target_speed).clamp(0.0, 1.0)


class posture:
    """Keep the joints near `HOME`, loosely. `exp(-mean(err^2 / std^2))`.

    **The same formula as mjlab's `variable_posture`, which is what the four
    velocity tasks mount under the key `pose`.** The single difference is the
    width: that one picks between `std_standing`, `std_walking` and `std_running`
    by the magnitude of the twist command, and this task has no command to pick
    with, so there is one map. The term is mounted under `pose` too, so the
    tensorboard key means the same thing across all five tasks.

    The name here is `posture` rather than `variable_posture` because it is not
    variable -- the same way mjlab's key and function names differ.

    **The width is the whole decision, and the standing width would forbid the
    task.** Pumping means moving the body's centre of mass over planted feet, and
    for this robot that is joint motion: the hip-to-foot arm is about 0.226 m, so
    swinging the leg about a planted foot far enough to shift the body 60 mm takes
    0.265 rad, which lands as 0.09 to 0.13 rad on each of the hip, knee and ankle.
    Against the velocity tasks' `std_standing` of 0.05 on a leg joint that is
    `exp(-3.24) = 0.04` -- the term collapses, and it collapses precisely when the
    robot does the thing it is here to do. At the walking width, 0.30, the same
    deviation keeps 0.91.

    So this is a regulariser and not a posture target: what it is for is the
    contorted stance and the flailing arm, not the lean. It is also the only term
    that watches the **arms** at all -- the front pair are 5-DoF arms and swinging
    one moves a lot of mass, while `feet_slip_on_the_deck` sees only the six foot
    bodies.
    """

    def __init__(self, cfg, env) -> None:
        from mjlab.utils.lab_api.string import resolve_matching_names_values

        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default = asset.data.default_joint_pos
        _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, _, std = resolve_matching_names_values(
            data=cfg.params["std"], list_of_strings=names
        )
        self.std = torch.tensor(std, device=env.device, dtype=torch.float32)

    def __call__(self, env: Any, std, asset_cfg) -> torch.Tensor:
        del std  # resolved per joint in __init__
        asset = env.scene[asset_cfg.name]
        error = asset.data.joint_pos[:, asset_cfg.joint_ids] - self.default[
            :, asset_cfg.joint_ids
        ]
        return torch.exp(-torch.mean(error.pow(2) / self.std.pow(2), dim=1))


def body_tilt_rate_on_the_deck(env: Any, deadband: float, robot_name: str = "robot",
                              entity_name: str = "swing") -> torch.Tensor:
    """The robot's own roll and pitch rate above a deadband, `(rad/s)^2`.

    **The shared `body_tilt_rate` charges for riding the swing.** It penalises
    `wx^2 + wy^2` in the world, and on a swing nearly all of that is the deck:
    measured on a robot doing nothing at all, the world-frame value runs 0.002 at
    rest and **1.450** on a 35 degree swing. No weight makes that mean the right
    thing -- it would simply charge less for succeeding.

    Taking the deck's own rate out leaves **0.001 at both**. Not "nearly flat":
    flat to three decimal places, because the deck rides square to the rope and a
    robot standing still on it does not rotate relative to it at all. Under the
    knotted rigging this was 0.465 rising to 0.604 -- the residue there was the
    second pendulum mode shaking the robot, and that mode is gone.

    Pitch is not excluded, and it should not be: leaning fore and aft is how the
    robot pumps, but leaning is a *displacement*, and `pump_power` is what pays for
    it. What is charged here is the **rate**, relative to the deck, above a
    deadband -- a lean that is smooth costs nothing and one that is a twitch costs.

    **The deadband is the shared term's 0.5, and what it buys changed completely
    with the rigging.** It used to sit just above what a passive robot produced
    (0.465), so almost the whole allowance was spent on riding the swing. A passive
    robot now produces 0.001, so the entire 0.5 is available for what the term is
    actually for. Measured under training conditions -- noise, pushes, domain
    randomisation -- at an action sigma of 0.4 on a 35 degree swing: mean 0.622,
    p95 1.781, and 34.5% of steps charged. That is a term that bites on thrashing
    and not on the task.

    Left at 0.5 rather than lowered to match the new floor: charging a policy for
    small deliberate motion early on is how exploration gets suppressed, and 34.5%
    of steps is already a real signal.
    """
    rate = state.body_rate_on_deck(env, robot_name, entity_name)
    return (rate[:, 0].pow(2) + rate[:, 1].pow(2) - deadband).clamp(min=0.0)


def sideways_sway(env: Any, entity_name: str = "swing") -> torch.Tensor:
    """Penalise the seat swinging along the beam instead of along the swing.

    **Fixing `swing_amplitude` to name its plane stops the reward paying for sway;
    this is what makes it worth suppressing.** Those are different things: a mode
    nobody pays for is still a mode a disturbance excites, and sideways here is a
    pendulum of nearly the same length as fore-aft -- the seat hangs from two
    points 0.5 m apart on a 1.95 m beam -- so it rings at nearly the same frequency
    and does not decay away on its own any faster.

    Squared, so a small sway is nearly free and a large one is not: the robot
    should not be paying to stamp out the millimetre of sway that getting on
    unevenly produces.

    **Measured, there is more of it than that, and it does not come from the
    policy.** On a 35 degree swing the sway runs 6.2 degrees mean and 10.5 at p95,
    and the figure is the *same* at an action sigma of 0 as at 0.4 -- it is the
    startup randomisation (`base_com`, `encoder_bias`, `foot_friction`) making each
    robot slightly asymmetric, not anything the policy did. At weight -2 that is
    0.023 per step, which is small, and it is very nearly a constant early on: the
    critic absorbs it. What the term is for is the second half of training, where a
    policy that can shift its weight sideways can damp it out and is paid to.
    """
    return state.sway_amplitude(env, entity_name).pow(2)


def deck_upright(env: Any, deadband: float, std: float, robot_name: str = "robot",
                 entity_name: str = "swing") -> torch.Tensor:
    """Full marks for anything within `deadband` of square to the deck; falls off
    beyond it.

    **This replaces `mdp.upright`, which would be actively wrong here.** That term
    measures the body against gravity, and a swing at 20 degrees carries its deck
    at 20 degrees -- so a robot standing perfectly on the plank reads as 20 degrees
    of tilt, and the reward would pay it to lean out of the deck at exactly the
    moments the swing needs it most. The failure is silent and it points the wrong
    way: the policy would learn to stay vertical, which on a tilting deck means
    falling off the back.

    Measured on a robot merely standing through a **35** degree swing: tilt against
    the deck stays at 0.5 degrees, well inside the 15 degree deadband, so this term
    reads exactly 1. Riding the swing costs nothing at all, including at the top of
    the arc, which is the whole reason it is written against the deck. Under
    training conditions at an action sigma of 0.4 it runs 2.1 degrees mean and 6.6
    at p95 -- still inside the band, so the band is being spent on deliberate
    leaning rather than on noise.

    **The deadband is not a softening, it is the difference between a safety rail
    and a posture demand.** Pumping means moving the body's centre of mass over
    planted feet, and leaning is one of the two ways to do it -- the other is
    shearing the legs to translate while staying level. A bare Gaussian taxes the
    first: at `std = 0.35` a 10 degree lean costs 22% of the term, which against a
    weight of 2 is 0.44 out of an objective worth at most 4. That is the reward
    charging the robot for using the mechanism the task is about.

    Inside the band the value is exactly 1, so this is also the flat part of the
    "still on the plank" bonus that keeps the episode worth continuing; outside it
    the Gaussian still makes being on the way over expensive, and the termination
    at 50 degrees is behind that.
    """
    tilt = torch.arccos(
        state.deck_up_in_body(env, robot_name, entity_name)[:, 2].clamp(-1.0, 1.0)
    )
    return torch.exp(-((tilt - deadband).clamp(min=0.0) / std) ** 2)


def pump_power(env: Any, robot_name: str = "robot",
               entity_name: str = "swing") -> torch.Tensor:
    """Lean the way the swing is already going. Shape (N,), signed.

    **The only term that pays for the act rather than the result.** Everything else
    scores a state -- how high the swing is, how square the robot is standing --
    and at a dead stop every one of them is flat, so nothing tells a policy which
    way to move on the first step of the first episode.

    This is the rider's mechanical power on the pendulum, up to constants:

        P = torque * rate = (m g d) * theta_dot

    with `d` the body's fore-aft offset from the plank's centre and `theta_dot` the
    rope's angular rate. So it is positive exactly when the robot is feeding the
    swing and **negative when it is damping it**, which is the half that makes it a
    pumping term rather than a shaking term.

    **Offset times rate, not velocity times velocity**, and the difference is the
    whole thing. Rewarding "the body is moving fore and aft" pays for wobble at any
    frequency and any phase, including the phase that kills the swing. Measured on
    a servoed rider, the drive that works has its *displacement* in phase with the
    seat's velocity -- the rider is furthest forward as the seat sweeps through the
    bottom going forward -- and a velocity-velocity product would score exactly the
    quarter-cycle-shifted motion that does nothing.

    Signed and unclipped. Clipping the negative half would make damping free, and a
    policy that pumps on the way out and stalls on the way back would collect the
    same as one that pumps throughout.

    The feet staying put is not this term's job -- `feet_slip_on_the_deck` has it.
    Together they are the mechanism the task was asked for: move the body, not the
    feet.
    """
    # Measured from where the robot **stands**, not from the plank's geometric
    # centre: `DECK_OFFSET` puts the base 18 mm behind centre so the six feet come
    # out centred, and leaving that in would give a robot that never moves a
    # standing bias of -0.018 times the rate. It averages to zero over a whole
    # cycle, but only over a whole cycle, and a decaying swing is not symmetric.
    #
    # (This comment said 36 mm until it was checked against the model.
    # `scenes/swing.py::STANCE_CENTRE_X` is 0.018 and always was; the code was
    # right and the number written beside it was not.)
    offset = (state.robot_on_deck(env, robot_name, entity_name)[:, 0]
              - state.DECK_OFFSET[0])
    return offset * state.swing_rate(env, entity_name)


def centred_on_the_plank(env: Any, std: float, robot_name: str = "robot",
                         entity_name: str = "swing") -> torch.Tensor:
    """Stay near the middle of the plank, measured in the plank's own frame.

    A gentle pull rather than a wall: the robot **has to** move fore and aft to
    pump -- that is the whole mechanism, and a rider that keeps the phase drives
    every length in `HANG_L_RANGE` past 85 degrees on as little as +/-20 mm of
    travel. So `std` has to be wide enough to leave that free, and at 0.15 it is:
    averaged over a cycle, +/-20 mm of lean costs 0.009 of this term per step and
    +/-100 mm costs 0.189, against an objective worth 5.0 at the target. What this
    term is for is the slow drift towards an edge that ends an episode, which the
    termination alone only punishes once it is too late to recover.

    **Measured from where the robot stands, not from the plank's geometric
    centre.** `DECK_OFFSET` puts the base 18 mm behind centre, because the front
    legs reach 0.190 m forward and the rear ones only 0.154 m back and it is the
    six feet that have to be centred (`scenes/swing.py::STANCE_CENTRE_X`). Left
    out, this term peaks 18 mm ahead of where the robot naturally stands, and
    measured on a passive robot that is the difference between:

        standing where the reset put it   front foot 47.6 mm from the edge, rear 45.4
        at this term's maximum            front foot 29.5 mm from the edge, rear 63.4

    -- so the maximum of a term whose whole job is "do not drift towards an edge"
    would be the least safe place on the plank, 38% of the front margin gone.
    Nothing raises; a robot standing correctly simply collects 0.985 instead of 1.0
    for ever, and a constant one-way pull of that size is exactly the kind that
    accumulates. `pump_power` subtracts the same offset for the same reason, and
    the two terms disagreeing about where the middle of the plank is was the whole
    of the bug.
    """
    local = state.robot_on_deck(env, robot_name, entity_name)
    fore_aft = local[:, 0] - state.DECK_OFFSET[0]
    lateral = local[:, 1]
    return torch.exp(-(fore_aft.pow(2) + lateral.pow(2)) / std ** 2)


def feet_off_the_plank(env: Any, robot_name: str = "robot",
                       entity_name: str = "swing") -> torch.Tensor:
    """Penalise a foot that has left the plank's outline, one count per foot.

    Reads geometry rather than contact, and deliberately so: the task's contact
    sensor watches `terrain`, so a foot **on the plank** and a foot in mid-air are
    the same reading to it (`scenes/swing.py` records this). Rebuilding the sensor
    against the plank geom is the better fix and is not done yet; until then, a
    foot outside the plank's rectangle is the observable that does not lie.
    """
    from ...common.constants import FEET

    robot = env.scene[robot_name]
    entity = env.scene[entity_name]
    seat = state.seat_row(env, entity_name)
    ids = [robot.body_names.index(f) for f in FEET]

    delta = robot.data.body_link_pos_w[:, ids] - entity.data.body_link_pos_w[
        :, seat
    ].unsqueeze(1)
    axes = state.deck_axes(env, entity_name)
    local = torch.einsum("nij,nkj->nki", axes.transpose(1, 2), delta)
    outside = (local[..., 0].abs() > state.BOARD_X / 2) | (
        local[..., 1].abs() > state.BOARD_Y / 2
    )
    return outside.float().sum(-1)


def feet_slip_on_the_deck(env: Any, robot_name: str = "robot",
                          entity_name: str = "swing") -> torch.Tensor:
    """Penalise a foot travelling across the plank, summed over the six feet.

    **The robot is supposed to pump by moving its own centre of mass over planted
    feet, not by walking about on the seat.** A rider changes the pendulum's centre
    of mass by changing its own, and it can do that with its feet where they are:
    lean, and the whole assembly follows. Stepping across the deck is a different
    and much less useful thing -- it moves the robot relative to a plank whose
    margin is 0.05 m, and it puts the feet somewhere the reset did not choose.

    Measured **in the deck's frame and against the deck's own motion**, which is
    the whole content of the term. The plank is translating along an arc and
    rotating as it goes, so a foot perfectly stuck to it has a large world-frame
    velocity; subtracting the velocity of the deck's material point under each foot
    (`v_seat + omega x r`) is what turns that into slip. A world-frame foot-speed
    penalty would score a robot riding the swing correctly as a robot skating
    across it, and would be worst exactly at the bottom of the arc where the deck
    is fastest.

    Tangential only. Lifting a foot straight up and putting it back in the same
    place does not change where it is on the plank, which is what was asked for;
    it is travel across the deck that this is against.
    """
    from ...common.constants import FEET

    robot = env.scene[robot_name]
    entity = env.scene[entity_name]
    seat = state.seat_row(env, entity_name)
    ids = [robot.body_names.index(f) for f in FEET]

    arm = (robot.data.body_link_pos_w[:, ids]
           - entity.data.body_link_pos_w[:, seat].unsqueeze(1))
    deck_at_foot = entity.data.body_link_lin_vel_w[:, seat].unsqueeze(1) + torch.cross(
        entity.data.body_link_ang_vel_w[:, seat].unsqueeze(1).expand_as(arm), arm, dim=-1
    )
    relative = robot.data.body_link_lin_vel_w[:, ids] - deck_at_foot
    axes = state.deck_axes(env, entity_name)
    local = torch.einsum("nij,nkj->nki", axes.transpose(1, 2), relative)
    return local[..., :2].pow(2).sum(-1).sum(-1)
