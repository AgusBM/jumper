"""jumper.ripple's gait clock: which single leg swings, and when.

The mechanism is `tasks/jumper/common/mdp/phase.py`; this file holds only what is
this gait's own -- the frequency, the grouping, the stance-load threshold and the
mirror rule.

**This is the gait the three constraints cannot all be satisfied for**, and the
resolution is recorded here rather than buried: the command ceiling comes down.
"""

from __future__ import annotations

from .rewards import SWING_GROUPS as _SWING_GROUPS

#: The step frequency, in hertz. One cycle is all six legs swinging once.
#:
#: **25/6, and it does not reach the command range's top.** At 50 Hz control:
#:
#:     top speed        v_max = 0.08 * freq / duty, duty 5/6   ->  0.096 * freq
#:                      0.50 m/s would need                    ->  freq = 5.2083
#:     integer steps    1 / (5.2083 * 0.02)                    ->  9.6, not whole
#:     divisible by 6   six legs, evenly                       ->  6, 12, 18, ...
#:
#: 9.6 is not an integer and the divisible-by-six neighbours are 12 steps
#: (25/6 Hz, 0.40 m/s, 2 control steps of swing per leg) and 6 steps (25/3 Hz,
#: 0.80 m/s, **one** control step per swing, which is not a gait). There is no
#: value that reaches 0.50.
#:
#: So 12, and `env_cfg.py` lowers this task's command ceiling to 0.40 to match.
#: The alternative -- leaving the ceiling at 0.50 -- is the failure that cost
#: tripod 37500 iterations: a top command level that no gait can reach is not an
#: error, it is a tracking reward that can never be collected and a terrain
#: curriculum waiting on a command curriculum that never finishes.
#:
#: **2 control steps per swing is coarse and is the first thing to revisit** if
#: ripple underperforms. Every alternative costs top speed: 18 steps gives
#: 25/9 Hz, 3 steps per swing and 0.267 m/s; 24 steps gives 25/12 Hz, 4 steps and
#: 0.20 m/s. Raising this one task's control rate (decimation 4 -> 2) is the other
#: direction and buys resolution without the speed, at twice the sim cost and with
#: a control rate unlike the other three tasks'.
GAIT_FREQ_HZ = 25.0 / 6.0

#: The swing groups in phase order: one leg each, in `RIPPLE_ORDER`.
#:
#: LF -> RM -> LR -> RF -> LM -> RR, which alternates sides -- the textbook
#: hexapod ripple order (L1, R2, L3, R1, L2, R3). Derived from `rewards`' own
#: constant so the clock and the reward cannot disagree about the sequence.
SWING_GROUPS = _SWING_GROUPS

#: Force above which a supporting foot counts as loaded, in newtons.
#:
#: Five legs support, so of the robot's 29.5 N a fair share is 5.9 N and this is
#: about half. **Not tripod's 5.0**: that was half of 29.5/3, and asking five legs
#: for it would demand 85% of a fair share from each -- unreachable by any even
#: split, so the term would read as failure for a correct gait.
STANCE_FORCE_THRESHOLD = 3.0

#: How the clock mirrors, read by `common/mdp/symmetry.py` off the term's params.
#:
#: **A half-cycle shift, the same as tripod's -- derived, not assumed.** The legs
#: sit at sequence positions LF=0, RM=1, LR=2, RF=3, LM=4, RR=5. The sagittal
#: mirror sends LF<->RF, RM<->LM, LR<->RR, i.e. 0<->3, 1<->4, 2<->5: position
#: i -> i+3, and with six slots that is exactly half a cycle. Tetrapod's is a
#: reflection instead; the two are different operations and each gait's had to be
#: worked out on its own grouping.
MIRROR_KIND = "phase_half_shift"

__all__ = [
    "GAIT_FREQ_HZ",
    "MIRROR_KIND",
    "STANCE_FORCE_THRESHOLD",
    "SWING_GROUPS",
]
