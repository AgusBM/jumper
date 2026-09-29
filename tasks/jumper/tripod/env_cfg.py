"""Environment config for jumper.tripod.

The environment skeleton is shared by the four jumper tasks
(`tasks/jumper/common/velocity_env.py`), and the only difference between this task
and the other three is the `tripod` gait prior. To tune environment parameters for
this task alone, add arguments to the call below -- nothing else is affected.

What this task adds on top of the skeleton is the **fixed-frequency phase clock**:
one observation term, and the same frequency handed to the gait reward. Both are
wired here rather than in the skeleton, so the other three tasks are provably
untouched -- their configs never see this code.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..common.mdp.curriculum import (
    STD_ANG_RATIO,
    STD_LIN_RATIO,
)
from ..common.mdp.rewards import stance_foot_ground_gap, standing_foot_load
from ..common.tof import tof_sensor
from ..common.velocity_env import velocity_env_cfg
from .mdp.phase import GAIT_FREQ_HZ, phase_clock
from .mdp.rewards import TRIPOD_A, TRIPOD_B, stance_foot_load, tripod_gait

#: The observation term's name. It is also the key the left-right mirror transform
#: looks up in `tasks/jumper/common/mdp/symmetry.py`: adding an observation term
#: whose name is not registered there raises at startup, deliberately, because
#: symmetry augmentation would otherwise mirror it wrongly without saying so.
PHASE_OBS_NAME = "gait_phase"


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
    """
    cfg = velocity_env_cfg(
        # This task's own operator controls. Named here rather than inherited:
        # which way a stick counts as positive is a decision, and a decision
        # four tasks share by default is one none of them made.
        controls=Path(__file__).parent / "controls.yaml",
        asset=asset,
        play=play,
        gait_reward=tripod_gait,
        gait_name="tripod",
        # The two halves of the "stand still" barrier, named rather than
        # inherited. Measured on a policy that genuinely never moved: `pose` paid
        # +0.75 for joints staying at home while `action_rate_l2` charged -0.95
        # for moving at all -- about 1.7 of reward arguing against walking, on a
        # statically stable robot that pays nothing to stand.
        #
        # These are mjlab's own values, kept deliberately. They were once weakened
        # to -0.02 / 0.5 to dismantle that barrier, but the judgement predated the
        # collision-box fix (the robot was sitting on a body box sunk into the
        # ground and could not have walked whatever the weights were), so it did
        # not hold. Rolled back to find out whether weakening is needed at all --
        # and this gait walks with them, which is the answer for this gait.
        #
        # `action_rate_weight` is also the reference the `action_scale`
        # compensation scales from: the penalty goes as 1/scale^2, so changing one
        # without the other changes two variables at once.
        action_rate_weight=-0.1,
        pose_weight=1.0,
        # How high this gait lifts a foot, in metres. Per task because it is a
        # choice about the gait, and this is the task that measured it.
        #
        # **0.03 -> 0.025, because 0.03 was not being reached and asking harder
        # did not help.** On `logs/jumper/jumper.tripod/2026-09-08_19-57-33` at
        # iteration 4766 `Metrics/peak_height_mean` was 0.0159, 53% of the target.
        # Both weights were then doubled and re-measured 15697 iterations later:
        # **0.0151**. The penalties were being paid rather than avoided, which is
        # a target out of reach and not a weight too small.
        #
        #     target   of standing height   asks vs today   clears terrain to
        #     0.020    18.7%                +32%            level 3.7
        #     0.025    23.4%                +66%            level 4.7   <- here
        #     0.030    28.0%                +99%            level 5.6
        #
        # 0.025 is a stretch the policy can plausibly close -- Go1's target is 36%
        # of its standing height -- and it keeps a terrain level in hand over 0.02.
        #
        # **It is a ceiling as well as a target**, because both terms punish
        # overshoot. The terrain is scaled to this robot, so the ceiling is
        # legible: steps are 0.016 m at level 3, 0.021 at 4, 0.032 at 6, 0.048 at
        # 9. A gait peaking at 0.025 clears level 4 and no further. That is not
        # what is holding terrain back today -- it has been sitting at 3.6 behind
        # the yaw gate -- but it will be, and raising this is the first thing to
        # try once that gate stops binding.
        foot_target_height=0.025,
        # How strictly this gait is marked, as a fraction of the range. Named
        # rather than inherited: holding it constant across the rungs is what
        # makes widening the range legitimate, and that is a claim each task has
        # to make for itself. These are `curriculum.py`'s calibrated pair -- 0.25
        # and 0.20 at range 0.5, chosen so that standing still collects the same
        # 0.14 it does on a Go1 -- and this task has no reason to differ.
        # The fastest this gait may be commanded. **The only curriculum number
        # this task sets** -- the ladder is one shared shape scaled to it, so
        # level 1 means the same thing in all four. Here: 3.125 Hz x a 0.08 m stroke over one stance caps the gait at 0.50 m/s
        # exactly, which is where this comes from.
        command_lin_ceiling=0.5,
        command_ang_ceiling=0.75,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
        # `terrain_scan` is left off, and the reason is measured rather than
        # architectural. Asymmetric actor-critic is sound -- the critic is thrown
        # away at export, so privileged input costs the deployed policy nothing --
        # but the scan as configured was actively harmful early on.
        #
        # Terrain promotion waits for the command curriculum, so the robot spends
        # the whole first phase on level 0, which is flat. Measured there, the
        # scan's 117 columns had a standard deviation of 2.99e-3 against 0.43 to
        # 2.2 for every other term in the group: a third of the critic's width
        # carrying no signal, with `EmpiricalNormalization` then scaling the
        # robot's own pitch wobble back up to unit variance and feeding that to
        # the value function.
        #
        # What is given up is look-ahead, not terrain sense: `foot_height` stays,
        # and it is the ground height under each foot. For a *value* function that
        # is the larger half -- a state's value depends mostly on the state.
        #
        # Worth revisiting once the terrain curriculum is actually climbing, when
        # the scan would have content from the first step. The mistake to avoid
        # repeating is adding it while it is constant.
    )

    # ── The phase clock ───────────────────────────────────────────────────
    # The observation and the reward are handed the frequency **from the same
    # constant, at the same place**, which is the only thing keeping them in sync.
    # A mismatch does not raise: the policy would simply be shown one clock and
    # scored against another, and the gait term would stop being learnable while
    # everything else looked healthy. That is why both assignments live here rather
    # than relying on the default argument at each end.
    #
    # The term goes into both groups. They are separate dicts (the skeleton's critic
    # group is built as `{**actor_terms, ...}`, a copy), so adding to one does
    # nothing to the other -- and the critic needs it as much as the actor, since the
    # value of a state depends on where in the cycle it sits.
    #
    # A fresh ObservationTermCfg per group, not one shared instance: the observation
    # manager mutates term configs in place while resolving them (it converts
    # `scale` to a tensor and collects terms that carry a `reset`), so sharing one
    # object across two groups would have them resolve over each other.
    # The clock is switched off (both components zero) while the command asks the
    # robot to stand, which is why the term needs the command name and the same
    # threshold the gait reward uses. A free-running clock measurably made the
    # policy step in place when it should have been still -- the roll rate's
    # autocorrelation peaked at 2.50 Hz, the clock's own frequency, and only 5.17
    # of 6 feet were on the ground. See `mdp/phase.py`.
    # `history_length=0` is a decision, not the default landing here by accident.
    # The skeleton gives every other term five frames, and it does so *before*
    # this function adds the clock -- so a term mounted here gets no history
    # unless it asks, which is a quiet trap for whoever adds the next one.
    #
    # For this term the answer happens to be none: the clock is a deterministic
    # function of episode time, so its five frames are completely determined by
    # the current one. History on it is exactly zero information and eight wasted
    # inputs.
    for group in ("actor", "critic"):
        cfg.observations[group].terms[PHASE_OBS_NAME] = ObservationTermCfg(
            func=phase_clock,
            history_length=0,
            params={
                "freq_hz": GAIT_FREQ_HZ,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        )

    cfg.rewards["tripod_gait"].params["freq_hz"] = GAIT_FREQ_HZ

    # ── Load on the supporting legs ───────────────────────────────────────
    # `tripod_gait` scores *which* legs are touching; this scores whether the ones
    # that should be supporting are actually carrying the robot. Touching is cheap
    # -- a foot can rest on the ground with the weight going elsewhere -- and the
    # two terms together say "these three legs, and really on them".
    #
    # The weight is deliberately below the gait term's 1.0. This is a shaping term
    # for the gait, not a competitor to it, and every positive term added to this
    # task is also an addition to what a policy might collect without walking (the
    # gates inside the term are what keep that at zero; the weight decides what it
    # would be worth if a gap were ever found). Raise it if trained policies keep
    # the right legs down but skate along barely loading them.
    #
    # The 5 N threshold is set against this robot: ~29.5 N all up, so a tripod
    # stance is ~9.8 N per foot and 5 N is about half a fair share.
    cfg.rewards["stance_load"] = RewardTermCfg(
        func=stance_foot_load,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "force_threshold": 5.0,
            "command_threshold": 0.05,
            "freq_hz": GAIT_FREQ_HZ,
        },
    )

    # ── Lifting the feet ──────────────────────────────────────────────────
    # Measured on `logs/jumper/jumper.tripod/2026-09-08_19-57-33` at iteration 4766:
    # `Metrics/peak_height_mean` is **0.0159 against a target of 0.03** -- the feet
    # are clearing 53% of what they are asked to. The term meant to be paying for
    # that was collecting -0.0597 per step against `track_linear_velocity`'s
    # +1.8133, so it was worth 3% of the term it competes with and the policy was
    # right to ignore it. Doubled, from mjlab's -2.0.
    #
    # `feet_clearance` is linear in the weight (`sum |h - target| * |v_xy|`), so
    # this doubles the pressure exactly. **It is also weighted by foot speed, which
    # is an escape hatch**: the cost falls just as well by moving the feet slower
    # as by lifting them higher, and moving them slower is what velocity tracking
    # is paying for. Watch `Metrics/peak_height_mean` and
    # `Metrics/twist/error_vel_xy` together -- the first rising while the second
    # rises too is the term being satisfied the wrong way.
    #
    # `foot_swing_height` is raised with it, and it is the cleaner of the two: it
    # scores `(peak / target - 1)^2` once per landing, with no velocity factor, so
    # there is no slow-the-feet-down way to satisfy it. It is also the one that can
    # be driven to zero -- a foot that peaks at the target pays nothing, whereas
    # `feet_clearance` charges for every height the foot passes through on the way
    # up and so keeps a floor no gait can clear.
    #
    # At the measured peak the per-landing error is (0.53 - 1)^2 = 0.22 per foot,
    # which -0.25 was diluting to -0.0370 per step. Both terms together now come to
    # about -0.27 against the two tracking terms' +3.36, so they can be felt and
    # cannot outvote what the robot is actually for.
    # **Raised again, now that the target is reachable.** The doubling above bought
    # nothing measurable -- peak height 0.0159 -> 0.0151 across 15697 iterations --
    # and the skeleton's note records why: at a target the policy cannot reach,
    # weight buys penalty and not behaviour. With the target at 0.025, asking 66%
    # more than the robot delivers rather than 99%, the gradient points somewhere
    # it can go and the weight decides how much that is worth.
    #
    # `foot_swing_height` takes the larger share and is now the heavier of the two,
    # reversing the order they started in. It is the term that scores peak height
    # directly, at landing, with no velocity factor and no floor -- a foot that
    # peaks at the target pays exactly zero. `foot_clearance` stays where it is
    # because its escape hatch has not gone away: the cost falls just as well by
    # slowing the feet as by lifting them, and more weight on it is more pressure
    # on that shortcut too.
    #
    # If the peak still does not move, the next question is not the weight. It is
    # whether the leg can lift 0.025 m at 3.125 Hz at all, which is a stride
    # measurement nobody has taken.
    cfg.rewards["foot_clearance"].weight = -4.0
    cfg.rewards["foot_swing_height"].weight = -3.0

    # ── Loosening the standing posture, and only the standing one ─────────
    # `pose` rewards joints for staying near the home posture, and it does so with
    # two tolerances: `std_standing` (0.10 arm / 0.05 leg) below
    # `walking_threshold`, `std_walking` (0.60 / 0.30) above it. The standing one
    # is **six times tighter**, so what the robot is held to when it stops is a far
    # more exact posture than anything it is held to while moving.
    #
    # The weight is the wrong lever here and was tried first: it multiplies the
    # whole term, so halving it relaxes the walking half by exactly as much as the
    # standing half. Widening `std_standing` alone is the change that is actually
    # being asked for, and `std_walking` is left at the skeleton's values, so what
    # a moving leg may do is untouched.
    #
    # Doubled rather than measured -- there is no run that isolates this yet.
    # Standing is a small slice of any episode (`rel_standing_envs` is about 10%),
    # so the change will show up in how the robot *holds* a stop rather than in the
    # episode return: watch a replay at zero command, and `Episode_Reward/pose`
    # only as a sanity check that the walking half really did not move.
    #
    # Note the switch is on measured speed, not on the command, so a robot still
    # decelerating from a walk is scored by `std_walking` for as long as it is
    # actually moving.
    _pose_std = cfg.rewards["pose"].params["std_standing"]
    cfg.rewards["pose"].params["std_standing"] = {k: 2.0 * v for k, v in _pose_std.items()}

    # ── How much of training is spent standing ────────────────────────────
    # Doubled, from mjlab's 0.1. The flag is redrawn at every resample, so the
    # share of environments is also the share of time: standing goes from 10.02%
    # of training to 20.02%, the 0.02 being sampled commands that happen to land
    # under the gate (measured, 0.0188% -- about one resample in 5300).
    #
    # **A weight cannot create samples.** Everything gated on the command being
    # zero -- `stance_load_stand`, `standing_sway`, `pose`'s tight `std_standing`
    # -- was learning from a tenth of the data, and raising weights only rescales
    # the gradient that tenth produces. This is the other half of that fix and the
    # more fundamental one.
    #
    # It is paid for out of walking. Everything the task is actually for -- both
    # tracking terms, `tripod_gait`, `stance_load`, the foot terms -- gates the
    # other way, so this is 10% of their exposure handed to the standing case.
    # Twice seemed the most that could be justified for a task whose name is a
    # gait; further, and it stops being a locomotion task with a standing case.
    #
    # **It also loosens both curricula, slightly.** `error_vel_xy` accumulates over
    # an episode and a standing stretch contributes almost nothing to it, so the
    # measured error is diluted by the standing share: about 0.9x of the walking
    # error before, about 0.8x now, which makes every promotion bar effectively
    # 11% looser without any bar moving. There is room for it -- the command
    # curriculum's last measurement was 0.0646 against a bar of 0.18 -- but the
    # terrain curriculum runs on much thinner margins and this is worth
    # remembering if it starts promoting faster than the robot deserves.
    cfg.commands["twist"].rel_standing_envs = 0.2

    # ── Load while standing ───────────────────────────────────────────────
    # The standing half of `stance_load`, which is gated off there on purpose --
    # it is a positive term, and paying it while the command says "move" is the
    # stand-still local optimum. Separate terms rather than one with a branch
    # because the threshold differs: 5 N is half a fair share over three
    # supporting legs, while standing spreads ~27 N over six for about 4.5 N
    # each, and 3 N is two thirds of that. Measured standing, the six feet carry
    # 3.67 to 5.32 N, so all six clear it with room for an uneven split.
    # **The weight carries a duty cycle, and 0.5 did not.** This term is gated on
    # the command, so it only pays in the envs commanded to stand -- the share set
    # by `rel_standing_envs` just above, and since the flag is redrawn at every
    # resample, the share of environments is also the share of time.
    #
    # At mjlab's 0.5 with mjlab's 10% standing the term was worth 0.5 * 0.10 = 0.05
    # of an episode return of about 114, while `stance_load` -- the walking half of
    # the same idea, at the same 0.5 -- was worth 0.5 * 0.90 = 0.45. Nine times
    # more, for no reason other than how often each was switched on.
    #
    # **This weight and `rel_standing_envs` are the same lever**, and the arithmetic
    # has to be redone whenever either moves or they multiply:
    #
    #     standing   parity weight   stance_load_stand   stance_load
    #     10%            4.5              0.45              0.45
    #     20%            2.0              0.40              0.40   <- here
    #     20% at 4.5      --              0.90              0.40   (2.25x, tried)
    #
    # 2.0 is `0.5 * 0.8 / 0.2`: the two halves of one idea worth the same per
    # episode, which is what they were meant to be, with the extra standing data
    # coming from the duty cycle rather than from stacking a second multiplier on
    # top of it.
    #
    # Raising it is safe in the way raising a positive term usually is not. The gate
    # is `moving_gate`, which reads the **command**, not the robot: when the command
    # says move, this pays zero however the robot behaves, so no amount of weight
    # makes refusing to walk profitable. That is exactly why `stance_load` is gated
    # off while standing and why these are two terms rather than one.
    #
    # If the gradient still looks thin, the next lever is the threshold, not the
    # weight. Measured standing, the six feet carry 3.67 to 5.32 N and the ramp
    # saturates at 3 N, so a stance that is already correct scores a flat 1.0 and
    # the weight only scales a constant there -- what is left to learn is the wrong
    # stances, where a foot off the ground reads 0 N. A term pinned at its ceiling
    # from early on wants `force_threshold` raised towards the 4.5 N fair share.
    cfg.rewards["stance_load_stand"] = RewardTermCfg(
        func=standing_foot_load,
        weight=2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "force_threshold": 3.0,
            "command_threshold": 0.05,
        },
    )

    # ── Distance to the ground for feet that should be down ───────────────
    # The dense counterpart to the two load terms. Contact force is exactly zero
    # until the foot arrives, so neither says anything about *approaching*: a
    # foot 5 cm up, 1 mm up and pressing 4.9 N all score the same. This one reads
    # the gap to the terrain and falls the whole way to contact. Measured on the
    # 25927-iteration run of 2026-09-06, about half the under-loaded stance legs
    # were not touching at all, and that half is what this reaches.
    #
    # **It pulls against `foot_clearance`, which was just doubled to -4.0 above.**
    # That term costs |h - 0.1| * |v_xy| for every foot and this one costs height
    # for the feet that should be down, so they disagree exactly on a support foot
    # that is high. The overlap is narrower than it looks -- `foot_clearance` is
    # weighted by foot speed and a support foot is nearly stationary -- but at
    # -4.0 it is twice as strong as when this weight was chosen, so start low:
    # -0.5 makes a foot at the 5 cm clamp cost 0.025.
    #
    # The two are measured by different signs of the same statistic. If
    # `Metrics/peak_height_mean` stops climbing after this goes in, that is the
    # tension biting, and the thing to reach for is `max_gap` -- shrinking the
    # band where this term has any opinion -- rather than its weight.
    cfg.rewards["stance_ground_gap"] = RewardTermCfg(
        func=stance_foot_ground_gap,
        weight=-0.5,
        params={
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "groups": (TRIPOD_A, TRIPOD_B),
            "freq_hz": GAIT_FREQ_HZ,
            "max_gap": 0.05,
            "command_threshold": 0.05,
        },
    )

    # The critic already sees per-foot contact forces: mjlab puts
    # `foot_contact_forces` (sign(f)*log1p(|f|), 6x3) in the critic group only, as
    # privileged information. Nothing to add -- a second force term would just be
    # the same signal twice.

    # A note on the command range, which the skeleton pins at +/-0.5 on all three
    # axes (and where it also drops mjlab's Go1-sized command curriculum, without
    # which that pin does not survive to training):
    #
    # It binds harder here than for the other three tasks, because a **fixed** step
    # frequency turns speed into a derived quantity rather than a free one --
    #
    #     speed = stroke per cycle x frequency
    #
    # -- so the policy can no longer buy speed by lengthening its stride at a slower
    # cadence. Measured on a trained policy, the pairing works out comfortably: it
    # reaches **0.427 m/s at a command of 0.5** (85% tracking, the same ratio it
    # holds at 0.2), which at 2.5 Hz is a 0.17 m stroke per cycle. The gait is
    # genuinely a tripod while it does that -- 2.77 feet on the ground against the
    # ideal 3.0, and 5.75 of 6 legs in the state the clock calls for.
    #
    # An earlier version of this note asserted that 0.5 m/s "asks for 0.2 m per
    # cycle against a measured capability near 0.14 m/s" and was therefore out of
    # reach. That 0.14 came from an old audit of `jumper.flat`, a different task and a
    # different policy, and it does not describe this one.
    #
    # **The clock is 3.125 Hz now, and that reading is why.** A leg supports for
    # half a cycle, so 0.427 m/s at 2.5 Hz is a 0.085 m stroke over one stance --
    # which is the top of the roughly 0.06-0.08 m the IK allows. The robot was
    # against its stride limit, and 2.5 Hz caps the gait at 0.08 * 2 * 2.5 =
    # 0.40 m/s, under the command range's own maximum. 3.125 Hz lifts the cap to
    # 0.50 m/s exactly, and to 0.53 with the 0.085 m actually observed.
    #
    # Do not read the whole 85% as the stride limit, though. The same 85% held at a
    # command of 0.2, where the stroke is only 0.034 m and nothing is saturating --
    # so most of that gap is a uniform tracking bias that a taller clock will not
    # fix. What the frequency buys is headroom at the top of the range, not the
    # missing 15%.

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg
