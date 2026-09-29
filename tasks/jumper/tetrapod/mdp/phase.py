"""jumper.tetrapod's gait clock: which pair swings, and when.

The mechanism is `tasks/jumper/common/mdp/phase.py`; this file holds only what is
this gait's own -- the frequency, the grouping, the stance-load threshold and the
mirror rule.
"""

from __future__ import annotations

from .rewards import TETRAPOD_PAIRS

#: The step frequency, in hertz. One cycle is all three pairs swinging once.
#:
#: **25/6 exactly, and every digit of it matters.** Three constraints have to hold
#: at once at 50 Hz control (see the derivation in `common/mdp/phase.py`):
#:
#:     top speed        v_max = 0.08 * freq / duty, duty 2/3   ->  0.12 * freq
#:                      the command curriculum tops at 0.50    ->  freq >= 4.1667
#:     integer steps    1 / (freq * 0.02) whole                ->  12 steps
#:     divisible by 3   three pairs, evenly                    ->  12 / 3 = 4
#:
#: 12 control steps is exactly 25/6 Hz, which is exactly 0.50 m/s at the 0.08 m
#: stride the IK allows, with exactly 4 steps of swing per pair. Tetrapod is the
#: one gait where all three constraints land on the same number.
#:
#: Written as the fraction rather than as 4.1667: the integer-modulo path in
#: `gait_phase` is taken only when `1/(freq*step_dt)` is within 1e-9 of a whole
#: number, and a truncated decimal misses it. 4.1667 gives 11.99992 steps, takes
#: the float path, and accumulates a residue into a systematic pair-to-pair bias
#: -- which is precisely what the mirror augmentation assumes is not there.
GAIT_FREQ_HZ = 25.0 / 6.0

#: The swing groups in phase order. Slot k is `SWING_GROUPS[k]`, and the clock
#: gives each a third of the cycle.
#:
#: Taken from `rewards.TETRAPOD_PAIRS` rather than written out again: the reward
#: and the clock disagreeing about the grouping does not raise, it just scores the
#: policy against a pattern it was never shown.
SWING_GROUPS = TETRAPOD_PAIRS

#: Force above which a supporting foot counts as loaded, in newtons.
#:
#: Four legs support, so of the robot's 29.5 N a fair share is 7.4 N and this is
#: about half. **Not tripod's 5.0**: that was half of 29.5/3, and asking four legs
#: for it would demand 68% of a fair share from each, which no even split provides
#: -- the term would read as failure for a correct gait.
STANCE_FORCE_THRESHOLD = 3.7

#: How the clock mirrors, read by `common/mdp/symmetry.py` off the term's params.
#:
#: **A reflection, not a shift.** The pairs are {LF,RR} / {LM,RM} / {LR,RF} at
#: phases 0, 1/3, 2/3. The sagittal mirror sends LF<->RF, LM<->RM, LR<->RR, so
#: {LF,RR} -> {RF,LR} = the third pair, {LM,RM} -> itself, {LR,RF} -> the first.
#: The slots map 0->0, 1/3->2/3, 2/3->1/3, which is p -> -p. The fixed middle pair
#: is what rules out a shift: a shift has no fixed point unless it is the identity.
MIRROR_KIND = "phase_reflect"

__all__ = [
    "GAIT_FREQ_HZ",
    "MIRROR_KIND",
    "STANCE_FORCE_THRESHOLD",
    "SWING_GROUPS",
]
