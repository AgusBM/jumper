"""The gait phase clock, shared by the three gait tasks.

A fixed-frequency clock, fed to the policy as an observation and used by the gait
and stance rewards to decide **which** legs should be swinging right now. It is
what turns a gait reward from "some valid grouping is airborne" into "*this*
grouping, *now*".

This was `tasks/jumper/tripod/mdp/phase.py` and served one task. The mechanism is
the same for all three; only the frequency and the grouping differ, and both are
arguments. Each task keeps its own constants in its own `mdp/phase.py`.

## Where the phase comes from

    phase = frac(episode_length_buf * step_dt * freq_hz)     in [0, 1)

A pure function of episode time -- **no new buffer, no reset hook**.
`episode_length_buf` is reset to 0 by `_reset_idx`, so the clock restarts with the
episode for free and cannot drift out of sync with anything.

## Choosing a frequency

Three constraints, and they fight:

**1. Top speed.** A leg is in support for `duty * T`, so the foot's fore-aft
travel over its own stance is `v * duty / freq` and

    v_max = stride_max * freq / duty

with `stride_max` about 0.08 m on this robot (IK-limited). A frequency too low
makes the command curriculum's top level silently unreachable -- not an error, just
a tracking reward that can never be collected, and a terrain curriculum that waits
on a command curriculum that never finishes.

**2. An integer number of control steps per cycle.** `gait_phase` takes an exact
integer-modulo path when `1/(freq*step_dt)` is a whole number. Off it, the float
path leaves the step that ends a cycle at 0.99999994 rather than 0.0, which lands
on the wrong side of every `phase < x` group test -- one control step per cycle
scored against the group that just finished instead of the one about to start, at
precisely the transition.

**3. Divisible by the number of groups.** Otherwise one group is called for more
often than another, which is a standing left-right bias, which is exactly what the
mirror augmentation in `symmetry.py` assumes is absent.

At 50 Hz control (`step_dt` 0.02), what those give:

    gait       duty   groups   steps/cycle   freq Hz   steps/swing   v_max m/s
    tripod     1/2      2          16         3.125         8          0.50
    tetrapod   2/3      3          12         4.1667        4          0.50
    ripple     5/6      6          12         4.1667        2          0.40

**tripod and tetrapod come out exact; ripple does not.** 0.50 m/s would need
5.2083 Hz, i.e. 9.6 steps, which is not an integer at all. The divisible-by-six
neighbours are 12 steps (4.1667 Hz, 0.40 m/s) and 6 steps (8.33 Hz, one control
step per swing, unusable). So ripple takes 12 and **its command ceiling is lowered
to match** rather than being left unreachable -- see `tasks/jumper/ripple/env_cfg.py`.

Two control steps per swing is coarse and is the thing to revisit first if ripple
underperforms. The alternatives, all of which cost top speed: 18 steps gives
2.778 Hz, 3 steps per swing and 0.267 m/s; 24 steps gives 2.083 Hz, 4 steps and
0.20 m/s. Raising the control rate for that task alone is the other direction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .rewards import moving_gate

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def gait_phase(env: "ManagerBasedRlEnv", freq_hz: float) -> torch.Tensor:
    """The gait phase in [0, 1), as a [num_envs] tensor.

    **The wrap has to be exact, and in float32 it is not.** Written the obvious way
    as `remainder(buf.float() * step_dt * freq_hz, 1.0)`, the step that ends a
    cycle comes out as **0.99999994 rather than 0.0**, because neither 0.02 nor the
    product is representable. The phase error is negligible; the consequence is
    not, because the group is selected by comparing against fractions of a cycle
    and that value lands on the wrong side of every one of them. It was measured,
    not reasoned about.

    So when a cycle is a whole number of control steps the modulo is done in
    **integers**, where the wrap cannot be anything but exact. The float64 fallback
    covers a frequency that does not divide evenly; there the wrap is quantised by
    the control rate anyway, and float64 keeps the residual error ~1e-10 instead of
    ~1e-7.
    """
    # **A task that installs a clock of its own is read here too**, exactly as
    # `tripod/mdp/phase.py::gait_phase` reads it, and `freq_hz` is then not
    # consulted. Nothing installs one except `jumper.posture`, whose
    # `mdp/cadence.py` hangs its `CadenceClock` on the env under this name --
    # everywhere else the attribute is absent and this function is what it was.
    #
    # Without it `stance_ground_gap`, which reaches the phase through this
    # function and not tripod's, ran posture's fixed clock while the gait reward
    # and the observation ran the variable one: the other stance group on 49.8%
    # of walking steps at the first command rung, and nothing to say so.
    clock = getattr(env, "gait_clock", None)
    if clock is not None:
        return clock.phase(env)

    steps_per_cycle = 1.0 / (freq_hz * env.step_dt)
    whole = round(steps_per_cycle)
    if whole >= 1 and abs(steps_per_cycle - whole) < 1e-9:
        return (env.episode_length_buf % whole).to(torch.float32) / whole
    t = env.episode_length_buf.to(torch.float64) * env.step_dt
    return torch.remainder(t * freq_hz, 1.0).to(torch.float32)


def swing_index(phase: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Which group is swinging, as a [num_envs] integer in [0, n_groups).

    The cycle is divided into `n_groups` equal slots and the groups take them in
    order, so group `k` swings while `k/n <= phase < (k+1)/n`.

    `clamp` guards the boundary: a phase of exactly 1.0 cannot arise from
    `gait_phase` (it is in [0, 1) by construction on both paths) but a caller
    passing one would otherwise index off the end, and silently -- Python's
    negative-index wrap would score the last group as the first.
    """
    return (phase * n_groups).long().clamp(0, n_groups - 1)


def stance_mask(
    phase: torch.Tensor, groups: tuple[tuple[int, ...], ...], n_legs: int = 6
) -> torch.Tensor:
    """[num_envs, n_legs], 1.0 for every leg the clock puts in **support**.

    `groups` is the swing groups in phase order: `groups[k]` holds the LEGS indices
    that swing during slot `k`. Support is the complement -- every leg not in the
    currently swinging group -- which is what makes this work unchanged for a
    tripod (3 swing, 3 support), a tetrapod (2 and 4) and a ripple (1 and 5).

    Built by indexing a per-group table rather than by comparing phases per leg, so
    the group boundaries are evaluated once and every leg of a group is guaranteed
    to agree about which slot it is in.
    """
    table = torch.ones(len(groups), n_legs, device=phase.device)
    for k, g in enumerate(groups):
        table[k, list(g)] = 0.0
    return table[swing_index(phase, len(groups))]


def swing_mask(
    phase: torch.Tensor, groups: tuple[tuple[int, ...], ...], n_legs: int = 6
) -> torch.Tensor:
    """[num_envs, n_legs], 1.0 for every leg the clock puts in **swing**."""
    return 1.0 - stance_mask(phase, groups, n_legs)


def phase_clock(
    env: "ManagerBasedRlEnv",
    freq_hz: float,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """The clock as an observation: [num_envs, 2] holding (sin, cos), or
    **(0, 0) for any environment whose command asks it to stand**.

    The standing case is the whole reason this takes a command. A free-running
    clock measurably made the policy step in place while commanded to hold still:
    autocorrelating the roll rate over 300 standing steps peaked at +0.765 at a lag
    of 20 steps -- exactly the clock period at the 2.5 Hz this was found at -- and
    the mean number of feet on the ground was 5.17 of 6. Nothing rewarded that;
    nothing suppressed it either, and the network went on responding to a ticking
    input the way it had learned to while walking.

    **Zero, not frozen.** A frozen clock still reads as "some moment in the cycle"
    -- its (sin, cos) has magnitude 1 like any other phase. Zero has magnitude 0, a
    value no phase can produce, so "standing" becomes unambiguous rather than
    ambiguous. The discontinuity at the threshold is real and is a clean, learnable
    signal: clock off means hold still.

    The gate applies to **the observation only**. `gait_phase` keeps running,
    because the phase is still well defined and the gait rewards multiply by
    `moving_gate` anyway; what changes is only what the policy is shown.

    **Why sin/cos rather than the raw phase**: the phase is circular, and a raw
    value in [0, 1) has a discontinuity at the wrap where 0.999 and 0.001 are
    adjacent in time but maximally far apart as numbers. (sin, cos) is continuous
    across the wrap and, while walking, has constant magnitude, so it normalises
    cleanly too.

    No noise is attached. The other observations are corrupted to model sensor
    error, but the clock is not a measurement -- on hardware the controller
    generates it and it is known exactly. Noise here would model nothing real.

    **Mirroring is not the identity and is not the same for every gait.** See
    `symmetry.py`: tripod and ripple mirror onto themselves half a cycle later
    (negate both components), tetrapod onto itself *reflected* (negate sin only).
    Getting it wrong quietly feeds the policy samples whose phase and contact
    pattern contradict each other. Either way the off-value survives: negating
    (0, 0) gives (0, 0), so a standing environment mirrors onto a standing one.

    `mirror_kind` is **not used here**. It is read off this term's params by
    `symmetry.py`, which needs to know which of those two rules applies and cannot
    tell from the term's name -- all three gaits call it `gait_phase`, because that
    name is in `scripts/export.py`'s deployable whitelist and the on-robot
    observation builder constructs terms by it. Accepted and discarded so the
    declaration lives next to the grouping that determines it.
    """
    del mirror_kind
    angle = gait_phase(env, freq_hz) * (2.0 * math.pi)
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    moving = moving_gate(env, command_name, command_threshold)
    return clock * moving.unsqueeze(-1)


def clocked_phase_match(
    contact: torch.Tensor,
    phase: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
    sigma: float,
) -> torch.Tensor:
    """`phase_match`, but against **the group the clock says**, per environment.

        n_wrong = legs in the wrong state for this moment in the cycle
        score   = exp(-n_wrong^2 / sigma^2)                        in (0, 1]

    `rewards.phase_match` takes one fixed group, which is what the gait rewards
    used before a clock: they scored `max` over every valid grouping, so any
    grouping paid and nothing said *when* to swap. That is under-specified in two
    ways -- nothing sets the cadence, and a swap that never completes scores as
    well as one that does -- and measured, the policy settles into a slow shuffle.

    Here the desired contact pattern is `stance_mask`, which varies per
    environment because the phase does. Same scoring, same sigma semantics; only
    the target moves.
    """
    desired = stance_mask(phase, groups, n_legs=contact.shape[1])
    n_wrong = (contact - desired).abs().sum(dim=1)
    return torch.exp(-(n_wrong**2) / sigma**2)


def steps_per_cycle(freq_hz: float, step_dt: float) -> float:
    """Control steps in one gait cycle. For the checks in `tests/test_phase.py`
    and for anything reporting a frequency's exactness."""
    return 1.0 / (freq_hz * step_dt)


__all__ = [
    "clocked_phase_match",
    "gait_phase",
    "phase_clock",
    "stance_mask",
    "steps_per_cycle",
    "swing_index",
    "swing_mask",
]
