"""The gait reward for jumper.tripod: the tripod gait, driven by a fixed-frequency
phase clock.

Only this task uses it, so it lives in this task's directory. The shared pieces
(`phase_match` / `cycling_gate` / `moving_gate`) are in
`tasks/jumper/common/mdp/rewards.py`, which also explains why gating on both ends is
necessary. The clock itself is in `phase.py` next door.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from ...common.mdp.rewards import (
    contact_state,
    cycling_gate,
    moving_gate,
    phase_match,
)
from .phase import GAIT_FREQ_HZ, gait_phase

# tripod: the 0.0 group and the 0.5 group in the topology phase table
TRIPOD_A = (0, 3, 4)  # LF, RM, LR
TRIPOD_B = (1, 2, 5)  # RF, LM, RR


def tripod_phase_score(
    contact: torch.Tensor, phase: torch.Tensor, sigma: float = 0.7
) -> torch.Tensor:
    """Score the contact state against **the group the clock says should be
    swinging**, in (0, 1].

        phase in [0.0, 0.5)  ->  group A airborne, group B in support
        phase in [0.5, 1.0)  ->  group B airborne, group A in support

    Duty factor 0.5 is what makes the split even: tripod is 3 legs swinging against
    3 in support, so each group owns exactly half the cycle.

    This is a hard switch at the half-cycle, not a blend. The alternative -- a
    tolerance window that scores both patterns near the boundary -- was not taken
    because the boundary is already soft in the only way that matters: `phase_match`
    scores by *how many* legs are wrong, so a swap caught one step late costs the
    "one leg off" score (~0.13) rather than falling to zero, and the policy has 10
    control steps per half-cycle to get there.

    Kept separate from `tripod_gait` so it can be tested on plain tensors, with no
    environment, sensors or command manager to stand up.
    """
    swinging_a = phase < 0.5
    return torch.where(
        swinging_a,
        phase_match(contact, TRIPOD_A, sigma),
        phase_match(contact, TRIPOD_B, sigma),
    )


def tripod_gait(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
    sigma: float = 0.7,
    max_air_time: float = 0.5,
    max_contact_time: float = 1.0,
    freq_hz: float = GAIT_FREQ_HZ,
) -> torch.Tensor:
    """Reward the tripod gait at a fixed step frequency: **the group the clock
    currently calls for is entirely airborne while the other is entirely in
    support**.

        reward = match(the group phase(t) selects)               in (0, 1]

    See `phase_match` for `match`: it scores by how many legs deviate from that
    phase, so the grouping is binding.

    **Which group lifts first is now specified**, unlike the earlier version of this
    reward. That version took `max` over both groups, so both phases scored full
    marks and the cadence was left entirely to the policy; its docstring noted that
    enforcing an order "would need a phase clock in the observation". That clock now
    exists (`phase.py`) and is in the observation, so the order is enforced here and
    the step frequency is a constraint rather than an outcome.

    The trade is deliberate and worth stating plainly: the policy is no longer free
    to discover its own cadence, and 3.125 Hz is an assumption imposed on it. If
    training plateaus with the gait term stuck low while velocity tracking is fine,
    the frequency is the first thing to question -- see `GAIT_FREQ_HZ`.

    An earlier implementation used `|mean(contact_A) - mean(contact_B)|`, which
    suffers the mean-dilution problem `phase_match` was written to fix: lifting two
    legs in A and one in B still scores 1/3, and that is not a tripod gait at all.

    Active only when the command asks for motion -- a standing command should not
    force the feet up. Note that the clock itself keeps running while standing; only
    this reward is gated.

    `cycling_gate` is kept, and under a fixed clock it is **mostly redundant**: at
    3.125 Hz a correct swing lasts 0.16 s and a correct stance 0.16 s, so neither
    `max_air_time` (0.5 s) nor `max_contact_time` (1.0 s) can fire during a gait
    that scores well in the first place. Both exploits it was written to close --
    holding a group up, and marching in place with the other half planted as props
    -- are now scored down directly by the clock, since a leg that does not swap on
    time is simply in the wrong state. It stays as insurance and to keep this task
    comparable with the other two gait tasks, which still need it; dropping it would
    be a second, uncontrolled change.
    """
    contact = contact_state(env, sensor_name)
    phase = gait_phase(env, freq_hz)
    return (
        tripod_phase_score(contact, phase, sigma)
        * moving_gate(env, command_name, command_threshold)
        * cycling_gate(env, sensor_name, max_air_time, max_contact_time)
    )


def stance_foot_load(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    force_threshold: float = 5.0,
    command_threshold: float = 0.05,
    max_air_time: float = 0.5,
    max_contact_time: float = 1.0,
    freq_hz: float = GAIT_FREQ_HZ,
) -> torch.Tensor:
    """Reward the legs the clock puts in **support** for actually carrying load.

        reward = sum over stance legs of min(force / `force_threshold`, 1) / 3

    in [0, 1], per step. Being a per-step term, the episode return is proportional
    to **how long** the load was held -- integrating it is what turns "above 5 N" into
    "above 5 N for longer pays more", with no duration state to keep. Note the
    consequence: 10 scattered steps of load and 10 consecutive ones score the same.
    If continuity itself turns out to matter (a foot chattering on and off the
    ground at 5 N is not the same as one planted), this term has to grow within a
    stance interval, and that does need per-leg state.

    **Why load and not contact.** `tripod_gait` already scores which legs are
    touching, and touching is cheap: a foot can rest on the ground carrying almost
    nothing while the body's weight goes through the other legs, or through a leg
    that should be swinging. The robot weighs about 2 kg (29.5 N), so a tripod
    stance puts roughly 9.8 N through each supporting foot. **5 N is about half of
    that fair share** -- comfortably above a foot merely grazing the ground, and
    well below what a genuinely loaded leg carries, so it does not demand a
    perfectly even split.

    Force is the net contact force's magnitude. On flat ground the vertical
    component dominates; if lateral load ever needs excluding, this is the place to
    switch to `force[..., 2]`.

    **The two gates are not optional here.** Standing on all six feet loads the
    three the clock designates as stance just as well as walking does, so
    ungated this term would pay full marks for standing still -- the local optimum
    this robot falls into by default, and one that reward terms *add* into: the
    policy could bank this term while eating a zero from `tripod_gait`. Gating on
    the same `cycling_gate` closes it exactly as it is closed there: standing puts
    `current_contact_time` past `max_contact_time` and the term goes to zero.
    """
    sensor = env.scene.sensors[sensor_name]
    force = sensor.data.force
    assert force is not None, f"sensor {sensor_name!r} has no force field"
    # **A ramp, not a step.** `loaded` used to be `(force > threshold)`, and that
    # made the term flat across everything short of the threshold: a foot in the
    # air, a foot 1 mm off the ground and a foot pressing 4.9 N all scored zero.
    #
    # Measured on the 25927-iteration tripod run of 2026-09-06 (87 MB of
    # tensorboard under locomotion-training-master), averaged over the last 30%:
    #
    #     stance_load raw 0.673  ->  2.02 of 3 stance feet above 5 N
    #     tripod_gait raw 0.591  ->  exp(-n^2/0.49) gives n_wrong = 0.51 legs in
    #                                the wrong contact state
    #     so of the ~0.98 under-loaded stance legs, ~0.51 are not touching at all
    #     and **~0.47 are touching and carrying too little**
    #
    # That second half is exactly what a step function cannot help: the foot is
    # already down, and going from 2 N to 4 N earns nothing. The two curves are
    # also locked together -- stance_load / tripod_gait held 0.568 to 0.575 with a
    # standard deviation of 0.008 over the whole run, and stance_load did not
    # improve at all between iteration 2000 and 25000. It was not being optimised
    # on its own; it was riding on the gait term.
    #
    # The ramp saturates at the threshold rather than continuing to pay, so a
    # genuinely loaded foot still scores exactly 1 and nothing is gained by
    # stamping harder. What changes is only that the 0 -> threshold stretch is a
    # slope instead of a cliff.
    loaded = (torch.linalg.norm(force, dim=-1) / force_threshold).clamp(0.0, 1.0)

    # The stance group is whichever group is not swinging (see tripod_phase_score).
    stance = torch.zeros_like(loaded)
    phase = gait_phase(env, freq_hz)
    stance[:, list(TRIPOD_B)] = (phase < 0.5).float().unsqueeze(1)
    stance[:, list(TRIPOD_A)] = (phase >= 0.5).float().unsqueeze(1)

    # Both gates stay. **Standing is not this term's job** -- see
    # `standing_foot_load` in `common/mdp/rewards.py`, which carries its own
    # threshold because a fair share over six legs is not one over three.
    return (
        (loaded * stance).sum(dim=1)
        / len(TRIPOD_A)
        * moving_gate(env, command_name, command_threshold)
        * cycling_gate(env, sensor_name, max_air_time, max_contact_time)
    )


__all__ = [
    "TRIPOD_A",
    "TRIPOD_B",
    "stance_foot_load",
    "tripod_gait",
    "tripod_phase_score",
]
