"""The gait reward for jumper.ripple: the ripple gait.

Only this task uses it, so it lives in this task's directory. The shared pieces
(`phase_match` / `cycling_gate` / `moving_gate`) are in
`tasks/jumper/common/mdp/rewards.py`, which also explains why gating on both ends is
necessary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from ...common.mdp.phase import clocked_phase_match, gait_phase
from ...common.mdp.rewards import (
    contact_state,
    cycling_gate,
    moving_gate,
)

# ripple: the six legs are offset by 1/6 of a cycle in turn, duty 5/6 -> exactly
# one leg airborne at any moment.
#
# The order **alternates sides**: LF -> RM -> LR -> RF -> LM -> RR, i.e.
# L R L R L R. This differs from the LF->LR->RM->RF->LM->RR (L L R R L R) recorded
# in the topology file, which lifts two same-side legs consecutively twice. The
# alternating version is better in two ways:
#   1. the swinging leg alternates between sides, so lateral shifts of the centre
#      of mass cancel evenly
#   2. consecutively lifted legs are never adjacent, so local support during a
#      transition is fuller
# It is also the textbook hexapod ripple order (L1, R2, L3, R1, L2, R3).
RIPPLE_ORDER = (0, 3, 4, 1, 2, 5)  # LF, RM, LR, RF, LM, RR
#: The expected number of simultaneously airborne legs. Since moving to
#: `_phase_match` this is implied by the "one leg airborne" phase and is no longer
#: referenced in code; kept as documentation.
RIPPLE_AIRBORNE = 1

#: The swing groups in phase order: one leg per slot, in RIPPLE_ORDER. The clock
#: and this reward read the same constant so they cannot disagree about the
#: sequence -- a disagreement would not raise, it would just score the policy
#: against a leg it was never told to lift.
SWING_GROUPS = tuple((i,) for i in RIPPLE_ORDER)


def ripple_gait(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
    std: float = 0.7,
    freq_hz: float = 25.0 / 6.0,
    max_air_time: float = 0.5,
    max_contact_time: float = 1.5,
) -> torch.Tensor:
    """Reward the ripple gait: **exactly one leg airborne at any moment, and
    alternating sides**.

        duty  = max_i match(leg i airborne)
        alt   = 1 if the current swing leg is on the opposite side to the previous
                one, else 0
        reward = duty * (1 - w + w*alt)                     in (0, 1]

    See `phase_match` for `match`. An earlier implementation counted only the number
    of airborne legs (``exp(-(n_air - 1)^2 / std^2)``) without caring **which** leg
    was airborne -- the same flaw as in tripod and tetrapod, though mildest here
    (with a single swinging leg, "the right count" nearly implies "the right
    grouping"). Per-phase matching makes all three consistent.

    `alternation_weight` (w) controls how much alternation counts: w=0 reduces to a
    pure duty constraint, w=1 scores nothing without alternation. The default of
    0.5 keeps half the duty score when sides do not alternate, so early exploration
    still gets gradient rather than nothing for having the order wrong.

    This corresponds to the topology's ripple duty_factor of 5/6: the six legs are
    offset by 1/6 of a cycle in turn, so exactly one swings and five support at any
    instant.

    The trade against tripod: ripple keeps **five-point support** at every step,
    with a far larger static stability margin than tripod's three-point, suiting
    low speed, rough terrain or carrying a load. The cost is a lower top speed (one
    swinging leg means a small propulsive fraction). tripod is fast with a small
    margin.

    **How the previous swing leg is identified without a phase clock**: among the
    legs in contact, the one with the smallest `current_contact_time` landed most
    recently, and is therefore the leg from the previous swing phase. The current
    swing leg is the airborne one with the largest `current_air_time` (lifted
    earliest). Different sides means alternation. No extra state or observation term
    is needed.

    **The full six-step order is still not constrained** (`RIPPLE_ORDER` is
    documentation only): only "consecutive legs are on opposite sides" is. Rewarding
    the full order would need a phase clock, and alternating sides already rules out
    the main defect of lifting two same-side legs in a row; the rest is left for the
    policy to discover.

    std defaults to 0.7: with two legs airborne the score is still
    exp(-1/0.49) ~= 0.13, so a normal transient with two feet briefly off the ground
    is not crushed.

    The cost is that **zero airborne (standing still) also collects a floor of
    0.13**. The command gate helps, but when motion is asked for and none happens
    that 13% is still free score -- exactly the kind of floor that caused trouble in
    the tracking rewards here. If the policy is seen loitering, std=0.4 brings it
    down to 0.002.
    """
    contact = contact_state(env, sensor_name)
    # **Scored against the leg the clock names.** This used to be `max` over all
    # six legs -- any single leg airborne paid full marks -- plus a separate term
    # rewarding the swing leg for being on the opposite side to the previous one,
    # with "the previous one" inferred from contact timings because there was no
    # clock to ask.
    #
    # The clock removes both. Which leg is due is now stated rather than inferred,
    # and side alternation is a property of `RIPPLE_ORDER` itself (LF, RM, LR, RF,
    # LM, RR alternates by construction), so it cannot be violated by a policy
    # following the clock and there is nothing left to reward separately.
    #
    # That deletion also removes a real bug: the inference referred to `t_contact`
    # and `leg_side`, neither of which was defined in this module -- ruff F821 --
    # so `ripple_gait` raised NameError the moment it was called. It is not clear
    # this reward had ever run.
    score = clocked_phase_match(
        contact, gait_phase(env, freq_hz), SWING_GROUPS, std
    )
    return (
        score
        * moving_gate(env, command_name, command_threshold)
        * cycling_gate(env, sensor_name, max_air_time, max_contact_time)
    )


__all__ = ["RIPPLE_AIRBORNE", "RIPPLE_ORDER", "SWING_GROUPS", "ripple_gait"]
