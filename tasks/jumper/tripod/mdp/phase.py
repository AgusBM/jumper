"""The gait phase clock for jumper.tripod.

A **fixed 3.125 Hz** clock, fed to the policy as an observation and used by the gait
reward to decide which group of legs should be swinging right now. It is the one
piece of state that turns the tripod reward from "either group airborne is fine"
into "*this* group, *now*".

## Why a clock at all

The gait reward used to be `max(match(A airborne), match(B airborne))`: both phases
scored full marks, so the reward said nothing about **when** to swap. That is
under-specified in two ways --

- nothing sets the step frequency, so the policy is free to settle at whatever
  cadence happens to be cheapest, and measured it prefers a slow shuffle;
- nothing forbids a swap that never completes, since the instant after a swap
  scores exactly as well as the instant before it.

Putting the phase in the observation and driving the reward from the same clock
fixes both: the cadence is now a constraint, not an outcome, and the policy can
actually see which half of the cycle it is in, so "swap now" is a learnable
function of the observation rather than a guess.

## Where the phase comes from

    phase = frac(episode_length_buf * step_dt * freq_hz)     in [0, 1)

It is a pure function of episode time -- **no new buffer, no reset hook**. That
matters for correctness as much as for brevity: `episode_length_buf` is reset to 0
by `_reset_idx`, so the clock restarts with the episode for free, and it cannot
drift out of sync with anything.

`step_dt` is 0.02 s here (timestep 0.005 x decimation 4), so a 3.125 Hz cycle is 16
control steps, **8 per half-cycle**. That is the resolution the swap is quantised
to, and it is the practical floor on how fast this clock can usefully run: at 5 Hz
a swing would get 5 steps.

**Pick frequencies that divide the control rate into an even number of steps.**
A cycle that is not a whole number of steps does not just lose the exact wrap
below -- it splits the two half-cycles unevenly, so one tripod group is called for
more often than the other. Measured over 2000 steps as the share of steps
assigning each group the swing:

    3.125 Hz   16.000 steps/cycle   1000 / 1000    0.00% apart
    3.2 Hz     15.625 steps/cycle   1008 /  992    0.80% apart
    3.571 Hz   14.000 nominally     878 / 1122    12.20% apart

The last one is the warning: 25/7 Hz is a whole 14 steps in exact arithmetic, but
written as a truncated decimal it misses, takes the float path, and the residue
accumulates into a systematic left-right bias. A standing bias between the groups
is exactly what the mirror augmentation in `symmetry.py` assumes is not there.

Observation and reward read the clock at the same point in the step and therefore
always agree -- `ManagerBasedRlEnv.step` increments `episode_length_buf` once, then
computes rewards and observations from it (see `manager_based_rl_env.py`, the
`episode_length_buf += 1` at the top of that block).

## Two properties worth knowing

**The clock is switched off in the observation while standing** -- both components
go to zero once the command drops below `command_threshold`.

This reverses the original choice, which was to let it free-run on the argument
that freezing it "would make the observation ambiguous (a frozen phase and a
running one look identical at that instant)". **Measured, the free-running clock
made the robot rock in place.** Standing the trained policy for 300 steps and
autocorrelating the roll/pitch rate:

    lag 10 steps  +0.738        lag 4-6, 14-16   about -0.5
    lag 20 steps  +0.765   <-   exactly 2.50 Hz, the clock period

and the mean number of feet on the ground while standing was 5.17 of 6 -- the
robot was not planted, it was **stepping in place**. Nothing rewarded that:
`tripod_gait` and `stance_load` are both gated off by `moving_gate` when the
command is ~0. But nothing suppressed it either, and the network went on
responding to a ticking input the way it had learned to while walking.

Zeroing rather than freezing is what answers the original ambiguity objection.
A frozen clock still reads as "some moment in the cycle" -- its (sin, cos) has
magnitude 1 like any other phase. Zero has magnitude 0, a value no phase can
produce, so "standing" becomes unambiguous rather than ambiguous. The
discontinuity at the threshold is real and remains, but it is a clean, learnable
signal: clock off means hold still.

The gate applies to **the observation only**. `gait_phase` keeps running, because
the phase is still perfectly well defined and the gait reward multiplies by
`moving_gate` anyway; what changes is only what the policy is shown. And zero
survives the mirror transform unchanged (`phase_half_shift` negates both
components, and -0 is 0), so symmetry augmentation needs nothing.

**Every episode starts at phase 0, i.e. group A swings first.** There is no
per-env random phase offset, which would need a buffer and a reset event. The
left-right mirror augmentation covers the resulting asymmetry: mirroring maps group
A onto group B (see `phase_clock`'s note below), so every sample the policy trains
on is also seen with the groups swapped. If phase diversity within a batch ever
looks like the binding constraint, a random offset is the thing to add.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ...common.mdp.rewards import moving_gate

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: The gait's step frequency, in hertz. One full cycle is both tripod groups
#: swinging once, so at 3.125 Hz each group swings for 0.16 s.
#:
#: **This is the single source of truth for the frequency.** The observation term
#: and the reward term are both wired from it in `env_cfg.py`; if they ever
#: disagree, the policy is shown one clock and scored against another, which fails
#: silently -- the reward simply stops being learnable.
#:
#: ## Why 3.125 and not the 2.5 this started at
#:
#: A fixed frequency makes speed a function of stride alone. A leg is in support
#: for half a cycle, so the foot's fore-aft travel over its own stance is
#: `v * T/2`, and the top speed the gait can reach is
#:
#:     v_max = stride_max * 2 * freq
#:
#: The IK limits this robot's usable stride to about 0.08 m, which put 2.5 Hz at
#: **0.40 m/s** -- below the 0.50 m/s top level of the command curriculum in
#: `mdp/curriculum.py`. That failure is entirely silent: an unreachable command
#: does not raise, it just becomes a tracking reward that can never be collected,
#: and the terrain curriculum gates on tracking *and* on the command curriculum
#: reaching its maximum, so an unreachable top level can stall terrain progression
#: too. 3.125 Hz makes it exactly 0.50 m/s.
#:
#: ## Why not widen `sigma` and let the policy find its own cadence
#:
#: Because the gait reward gives it no direction to find one in. `phase_match`
#: scores `exp(-n_wrong^2 / sigma^2)` on the *number of legs in the wrong state*,
#: not on how far the cadence is from the clock -- and a frequency mismatch is not
#: a small timing error, it drifts until the two are anti-phase and all six legs
#: are wrong at once. Measured on `tripod_phase_score` with the clock at 3 Hz and
#: the policy running a *clean* tripod at another frequency, mean score over 20 s:
#:
#:     sigma      3.0 Hz   3.2 Hz   3.5 Hz   4.0 Hz   2.5 Hz
#:     0.7 - 2.0   1.000    0.500    0.500    0.500    0.500
#:     3.0         1.000    0.509    0.509    0.509    0.509
#:     6.0         1.000    0.684    0.684    0.684    0.684
#:
#: Two things kill it. Widening sigma barely moves the number -- 0.7 to 3.0 buys
#: 0.009 -- because the 0.500 is "half the time in phase, half the time entirely
#: out of it", which sigma does not govern. And 3.2 Hz scores identically to
#: 4.0 Hz, so the landscape is a spike at the clock frequency with no slope
#: anywhere else: nothing points toward "step slightly faster". Meanwhile sigma=3
#: pays 0.368 for having *three legs wrong*, which is the diluted-reward failure
#: `phase_match`'s docstring already records costing the tetrapod gait its
#: grouping. Adaptive cadence needs the frequency to be a function of the command
#: or an output of the policy; it cannot be a slackened constraint.
GAIT_FREQ_HZ = 3.125


def gait_phase(
    env: "ManagerBasedRlEnv", freq_hz: float = GAIT_FREQ_HZ
) -> torch.Tensor:
    """The gait phase in [0, 1), as a [num_envs] tensor.

    0 <= phase < 0.5 is the half-cycle in which group A swings, 0.5 <= phase < 1 is
    group B's. See `tripod_phase_score` for the other half of that convention.

    **The wrap has to be exact, and in float32 it is not.** Written as the obvious
    `remainder(buf.float() * step_dt * freq_hz, 1.0)`, the step ending a cycle --
    step 20 at the 2.5 Hz this was found at, step 16 now -- comes out as
    **0.99999994 rather than 0.0**,
    because neither 0.02 nor the product is representable. The phase error is
    negligible; the *consequence* is not, because the group is selected by
    `phase < 0.5` and that value lands on the wrong side of the wrap. One control
    step per cycle, 5% of them, would score the policy against the group that just
    finished swinging instead of the one about to start -- and at precisely the
    transition, the most informative moment in the cycle. It was measured, not
    reasoned about; see the note at the top of this module about the clock being a
    pure function of episode time.

    So when a cycle is a whole number of control steps -- which 3.125 Hz at 50 Hz
    control is, exactly 16 -- the modulo is done in **integers**, where the wrap
    cannot be anything but exact. The float64 fallback covers a frequency that does
    not divide evenly (3 Hz would be 16.67 steps); there the wrap is quantised by
    the control rate anyway, and float64 keeps the residual error ~1e-10 instead of
    ~1e-7.
    """
    # **A task may install a clock of its own**, and if it has, that one is the
    # clock for everything: this function is how `tripod_gait` and `stance_load`
    # reach the phase, and `common/mdp/phase.py::gait_phase` -- which is how
    # `stance_ground_gap` reaches it -- defers the same way, so every reward is on
    # the clock the observation shows without any of them knowing which kind it
    # is. `freq_hz` is then not consulted -- the installed clock sets its own rate.
    #
    # Nothing installs one except `jumper.posture`, whose `mdp/cadence.py` has the
    # argument for a cadence that follows the command. The attribute is absent
    # everywhere else, so the four locomotion tasks read exactly what they always
    # did -- which is the only reason this hook is acceptable in a file they
    # share.
    clock = getattr(env, "gait_clock", None)
    if clock is not None:
        return clock.phase(env)

    steps_per_cycle = 1.0 / (freq_hz * env.step_dt)
    whole = round(steps_per_cycle)
    if whole >= 1 and abs(steps_per_cycle - whole) < 1e-9:
        return (env.episode_length_buf % whole).to(torch.float32) / whole
    t = env.episode_length_buf.to(torch.float64) * env.step_dt
    return torch.remainder(t * freq_hz, 1.0).to(torch.float32)


def phase_clock(
    env: "ManagerBasedRlEnv",
    freq_hz: float = GAIT_FREQ_HZ,
    command_name: str = "twist",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """The phase clock as an observation: [num_envs, 2] holding (sin, cos), or
    **(0, 0) for any environment whose command asks it to stand**.

    The standing case is the whole reason this function takes a command: a
    free-running clock measurably made the policy step in place while commanded to
    hold still (2.50 Hz in the roll rate's autocorrelation, 5.17 of 6 feet on the
    ground). See the module docstring for the measurement and for why zero is the
    right off-value rather than a frozen phase.

    **Why sin/cos rather than the raw phase**: the phase is circular, and a raw
    value in [0, 1) has a discontinuity at the wrap where 0.999 and 0.001 are
    adjacent in time but maximally far apart as numbers. An MLP would have to spend
    capacity learning to glue that seam. (sin, cos) is continuous across the wrap
    and, while the robot is walking, has constant magnitude, so it also normalises
    cleanly.

    No noise is attached to this term. The other observations are corrupted to model
    sensor error, but the clock is not a measurement -- on hardware the controller
    generates it, and it is known exactly. Adding noise would model nothing real.

    **Mirroring this term is not the identity.** The left-right mirror maps
    LF<->RF, LM<->RM, LR<->RR, and group A = {LF, RM, LR} therefore maps onto
    {RF, LM, RR} = group B exactly. So a mirrored robot at phase `p` is in the same
    situation as the original robot at phase `p + 0.5`, and the mirrored clock must
    read sin/cos of `p + 0.5`, which is **both components negated**. That is
    registered as the `phase_half_shift` kind in
    `tasks/jumper/common/mdp/symmetry.py`; get it wrong and symmetry augmentation
    quietly feeds the policy samples whose phase and contact pattern contradict each
    other. The off-value is unaffected: negating (0, 0) gives (0, 0) back, so a
    standing environment mirrors onto a standing environment.
    """
    angle = gait_phase(env, freq_hz) * (2.0 * math.pi)
    clock = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    moving = moving_gate(env, command_name, command_threshold)
    return clock * moving.unsqueeze(-1)


__all__ = ["GAIT_FREQ_HZ", "gait_phase", "phase_clock"]
