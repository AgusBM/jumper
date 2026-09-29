"""The gait reward for jumper.tetrapod: the tetrapod gait.

Only this task uses it, so it lives in this task's directory. The shared pieces
(`phase_match` / `cycling_gate` / `moving_gate`) are in
`tasks/jumper/common/mdp/rewards.py`, which also explains why gating on both ends is
necessary.

## Measured after the fix (tetrapod, weight 1.0 @2000)

|                       | both gates, w1.0 | air_time only, w2.0 | baseline (no gait) |
|---|---|---|---|
| vx / vy tracking      | 91% / 93%        | 6% / 6%             | 93% / 91%          |
| displacement over 12 s| 2.43 m           | 0.065 m             | 1.69 m             |
| mean feet in contact  | 2.54             | 3.89                | 2.16               |

The reward hack is closed: displacement went from 0.065 m back to 2.43 m, in fact
past the control group with no gait reward at all.

**But the gait itself is only mildly shaped**: 2.54 feet in contact, far from a
tetrapod gait's 4.0 (the control group is at 2.16). The reason is that velocity
tracking carries a combined weight of 4.0 against the gait's 1.0, so once the gates
close off standing still as a way to collect gait reward, the policy prioritises
velocity instead. Expressing the gait more strongly needs a higher weight -- which
only becomes a meaningful experiment now the gating is fixed; before that, weight
2.0 collapsed into marching in place.
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

# tetrapod: three pairs lift in turn, duty 2/3 -> two airborne and four in support
# at any moment. The topology phase table has group 0.0 = {id0,id3}, group 1/3 =
# {id1,id4} and group 2/3 = {id2,id5}, which translate to the three LEGS index
# pairs below. Each of the three leaves a stable four-point support.
TETRAPOD_PAIRS = (
    (0, 5),  # LF + RR (diagonal)
    (2, 3),  # LM + RM (the two middles)
    (4, 1),  # LR + RF (diagonal)
)


def tetrapod_gait(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.05,
    sigma: float = 0.7,
    max_air_time: float = 0.5,
    max_contact_time: float = 1.0,
    freq_hz: float = 25.0 / 6.0,
) -> torch.Tensor:
    """Reward the tetrapod gait: **the pair the clock names airborne, with the
    other four all in support**.

        reward = match(TETRAPOD_PAIRS[k] airborne),  k = the clock's current slot

    See `clocked_phase_match`: it scores by how many legs deviate from that phase,
    so the grouping is binding. This corresponds to the topology's tetrapod
    duty_factor of 2/3, and the clock gives each pair a third of the cycle.

    **Why the pairing has to be binding rather than just counting "two airborne"**:
    lifting any two legs is destabilising. Lifting LF and LM together leaves the
    front-left with no support at all, whereas each of the three pairs in
    `TETRAPOD_PAIRS` (diagonal, both middles, diagonal) leaves an evenly
    distributed four-point support polygon.

    An earlier implementation did iterate over pairs but used a product of means,
    which does not actually bind them -- the worst grouping, LF+LM, still scored
    0.375. See `phase_match` for the detail.

    **This used to be `max` over the three pairs**, on the reasoning that any one
    of them airborne is a valid tetrapod phase and which is due now would need a
    phase clock. It now has one. What `max` gave up is what a clock is for: with
    every grouping paying equally, nothing set the cadence and nothing forbade a
    swap that never completed, since the instant after a swap scored exactly as
    well as the instant before it.

    It sits between tripod (three-point support, fast) and ripple (five-point,
    stable), trading speed against margin.
    """
    contact = contact_state(env, sensor_name)
    # **Scored against the pair the clock names, not against whichever pair the
    # policy happens to have lifted.** This used to be `max` over the three, which
    # paid full marks for any valid grouping and so said nothing about *when* to
    # swap: the cadence was unconstrained, and the instant after a swap scored
    # exactly as well as the instant before it, so a swap that never completed was
    # free. See `clocked_phase_match`.
    score = clocked_phase_match(
        contact, gait_phase(env, freq_hz), TETRAPOD_PAIRS, sigma
    )
    return (
        score
        * moving_gate(env, command_name, command_threshold)
        * cycling_gate(env, sensor_name, max_air_time, max_contact_time)
    )


__all__ = ["TETRAPOD_PAIRS", "tetrapod_gait"]
