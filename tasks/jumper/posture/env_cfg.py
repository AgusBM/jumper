"""Environment config for jumper.posture.

`jumper.tripod` with the body's posture made commandable: the policy is told what
twist, pitch, roll and height to hold, on top of where to go, and is scored on
both.

## What it is and is not

**It is the tripod gait.** The gait reward, the phase clock and the two stance
terms are imported from `tasks/jumper/tripod/mdp/` rather than copied -- there is
one tripod gait in this repository and a second copy would be a second thing to
keep in step. What this task changes about it is the cadence, and only because
the speed ceiling forces it; see `GAIT_FREQ_HZ` below.

**It is not one of the four controls.** `jumper.flat`, `jumper.tripod`, `jumper.ripple`
and `jumper.tetrapod` exist to differ in exactly one variable, and this task
differs in a command, three rewards, two observations, a termination bound and
both ceilings. `tests/test_task_parity.py` therefore lists it under
`NOT_LOCOMOTION`, with the same reasoning `jumper.swing` and `jumper.jump` carry:
there is no number it could agree with the four on that would mean the same
thing. Its control is `jumper.tripod`, one task, and that comparison is the point
of keeping tripod untouched.

## The four commands

    twist    the body yawed over its planted feet, +/- 30 degrees parked, 15 walking
    pitch    nose down positive, +/- 20 degrees parked, 15 walking
    roll     left side up positive, +/- 15 degrees
    height   the base above the ground, 0.07 to 0.15 m (it stands at 0.107)

Sampled independently and held until the next resample, **including while
walking** -- the robot is asked to walk at a commanded lean, not to choose
between the two. `mdp/commands.py` has what that costs and what to watch.

## What had to be taken out of the skeleton for this to mean anything

Two terms in `velocity_env_cfg` measure the body against a frame this task now
commands away from, and both would have gone on producing a number every step
while paying the policy to ignore its command:

    upright   scores alignment with gravity, i.e. pitch = roll = 0. Replaced by
              `track_tilt`, which is the same function at a zero command.
    pose      scores joints near HOME, tightly while standing. Holding a
              commanded posture *is* leaving HOME, and the standing width is
              six times the walking one. Widened to the walking value -- except
              at the neutral posture, where HOME is still the answer and the
              width ramps back to `jumper.tripod`'s standing one.

This is `jumper.swing`'s lesson arriving on a second task: a term named for what
you want, measured against a frame the robot is no longer in. There it was a
tilting deck; here it is a command.

## The ceilings, and the cadence they force

0.8 m/s and 4.0 rad/s, against `jumper.tripod`'s 0.5 and 0.75. Neither is a free
number; both follow from the stride and the cadence below.

**The stride is fixed and the cadence is what varies**, which is the other way
round from every other task here. A leg supports for half a cycle, so
`v_max = stride x 2 x freq` ties the three together; tripod fixes the frequency
and lets the stride fall out of the command, and this task fixes the stride and
lets the frequency follow it.

`mdp/cadence.py` has why: at a fixed cadence the robot takes 8.8 mm steps at
0.1 m/s and 59.3 mm at the ceiling, so it lifts its feet 7.5 times as often per
metre at the slow end -- and the slow end is where most of training happens, the
command ladder's level 0 being +/-0.4 m/s.

    stride 0.072 m    ->  f at the ceiling = 0.8 / (2 x 0.072) = 5.5556 Hz
                          18 control steps a cycle, 9 of swing

**72 mm is chosen against the parity of the control rate**, not freely: the cycle
is `100/f` control steps, `phase.py` wants that whole and even, and at a fixed
ceiling that leaves the stride only the values `v * N / 200` -- 56, 64, 72, 80 mm.
`GAIT_STRIDE` has the trade between them and what 72 spends: it is 21% beyond the
59.3 mm any policy has been measured using.

That is the risk to keep in view because it is silent. An unreachable stride does
not raise; it becomes a top speed the robot never reaches, a tracking reward that
cannot be collected, and a terrain curriculum that stalls behind it because it
gates on tracking *and* on the command ladder topping out.

**The angular ceiling is bound by the same law**, through the foot the turn has
to swing: `wz x 0.20 m` against the `2 x stride x freq` the gait can deliver. So
it is derived rather than chosen too -- 4.0 rad/s, where a pure turn uses the
whole gait exactly as 0.8 m/s does. That is 3.2x the 1.25 it was, and the command
ladder is five rungs rather than three because of it; `LADDER_ANG_FRACTIONS` has
the spacing and what the first two rungs are for.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..common.actuator import CONTINUOUS_TORQUE
from ..common.constants import EFFORT_LIMIT, GAIT_JOINTS, LEGS, STAND_Z
from ..common.mdp.controls import load_controls
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.mdp.rewards import (
    actuator_headroom,
    actuator_power,
    stance_foot_ground_gap,
    standing_foot_load,
)
from ..common.tof import tof_sensor
from ..common.velocity_env import velocity_env_cfg
from ..tripod.mdp.phase import phase_clock
from ..tripod.mdp.rewards import TRIPOD_A, TRIPOD_B, stance_foot_load, tripod_gait
from .mdp.cadence import ADVANCE, VariableGaitClock
from .mdp.commands import PostureCommandCfg
from .mdp.curriculum import (
    POSTURE_STD_RATIO,
    POSTURE_STD_SCALES,
    WARMUP_STEPS,
    PostureRangeCurriculum,
    angle_std,
    band_levels,
    height_std,
    posture_levels,
)
from .mdp.history import StridedHistory
from .mdp.metrics import landing_force_max
from .mdp.observations import posture_command, posture_state_obs
from .mdp.rewards import (
    foot_clearance_shortfall,
    hip_yaw_center,
    pose_widened_by_posture,
    soft_touchdown,
    track_height,
    track_tilt,
    track_twist,
)
from .mdp.teleop import CONTROLS, TeleopPostureCommandCfg
from .rl_cfg import NUM_STEPS_PER_ENV

#: The command term's key, and therefore the prefix its metrics are logged under
#: (`Metrics/posture/error_twist` and the other three).
POSTURE_COMMAND = "posture"

#: The physics timestep and the control decimation: **500 Hz and 100 Hz**.
#:
#: The skeleton's are 0.005 and 4, i.e. 200 Hz and 50. See the block in
#: `env_cfg` for what moves with them and what does not.
#:
#: `DECIMATION` is 5 rather than 4 because the pair has to give a whole control
#: period: 0.001 x 5 = 0.005 s exactly. `play.py --physics-hz` also requires the
#: physics rate to divide the control rate into a whole number of steps, and at
#: 1000 Hz physics that division is exactly `DECIMATION`.
SIM_DT = 0.001
DECIMATION = 5

#: Control steps between consecutive frames of the observation history.
#:
#: `OBS_HISTORY` is 5 frames and was chosen at 50 Hz, where consecutive frames
#: are 20 ms apart and the oldest reaches back 80 ms. At 200 Hz they are 5 ms
#: apart and reach back 20 ms, so the count survived two rate changes that its
#: own justification did not. 4 restores **exactly** the 50 Hz spacing -- 4 x
#: 5 ms = 20 ms -- without changing the actor's input width. `mdp/history.py`
#: has the trade it makes against resolution at the recent end, and why reach is
#: quoted rather than `velocity_env.py`'s frame-count convention.
OBS_HISTORY_STRIDE = 4

#: The fastest the robot is ever commanded, in m/s. A module constant because
#: `GAIT_FREQ_HZ` is derived from it and from `GAIT_STRIDE`: a leg supports for
#: half a cycle, so `v_max = stride x 2 x freq` ties the three together and any
#: two of them fix the third. Written once so the pair cannot drift apart.
COMMAND_LIN_CEILING = 0.8



#: The stride the cadence law assumes, in metres.
#:
#: **Measured rather than assumed.** At the 0.8 m/s ceiling a trained policy uses
#: 59.3 mm; `mdp/cadence.py` has the table. `jumper.tripod`'s notes read the IK as
#: allowing roughly 0.06-0.08 m and this is the bottom of that range, which is
#: also what the measurement landed on.
#:
#: With the cadence variable this number does a different job than it did when the
#: cadence was fixed: it is no longer a guess about what the legs can reach at the
#: ceiling, it is **the stride the robot is asked to take at every speed**.
#:
#: **72 mm rather than 60, and the parity of the control rate is why.** At 100 Hz
#: the cycle is `100/f` control steps and `phase.py` wants that whole and even, so
#: at a fixed ceiling the stride can only take the values `v * N / 200` for even
#: `N`: 56, 64, 72, 80 mm. 60 is not among them -- it gives 15 steps, odd, and
#: `mdp/cadence.py` measures the 6.7% left-right bias that costs whenever the
#: command is *held* at the ceiling.
#:
#: Of the legal values 72 mm buys the most: 9 control steps of swing against 8 at
#: 64 and 7.5 at 60, and 6.94 foot lifts per metre at the ceiling against 7.81.
#: What it spends is margin. The only measured stride is **59.3 mm** -- what a
#: trained policy actually used -- and 72 is 21% beyond it, against an IK range of
#: 0.06-0.08 that `jumper.tripod`'s notes assert and nobody has measured.
#:
#: That is the risk worth naming because it is silent: a stride the legs cannot
#: reach does not raise, it becomes a top speed the robot never achieves and a
#: tracking reward that cannot be collected. If `track_linear_velocity` plateaus
#: with `Metrics/twist/error_vel_xy` stuck near the command, this number is the
#: first suspect and 64 mm is the fallback.
GAIT_STRIDE = 0.072

#: The cadence follows the command rather than being fixed at `GAIT_FREQ_HZ`.
#:
#: `False` restores the fixed clock exactly -- the four locomotion tasks keep it,
#: and so does every run before this. Left as a switch because this is a change to
#: the plant and the comparison against those runs is worth one config line.
VARIABLE_CADENCE = True

#: The floor the cadence is clamped to, in Hz. **This is the guess in the law.**
#:
#: Below it the stride shrinks instead of the cadence, which is the standard
#: arrangement -- vary stride at low speed, cadence at high -- and something has to
#: set the changeover. Two bounds hold it rather than a measurement: at 1 Hz a
#: swing is half a second and the robot creeps through commands it could walk, and
#: at 3 Hz the law is already clamped for every command below 0.36 m/s, which is
#: most of level 0 and most of the benefit.
#:
#: 2 Hz puts the changeover at 0.24 m/s and gives a 26 mm stride at 0.1 m/s, three
#: times what a fixed clock produced there. At this control rate it is 50 steps a
#: cycle, so resolution is not a consideration at the bottom.
#:
#: The measurement that would replace this is lift energy per metre against
#: cadence at a held low speed, and it needs a policy trained with the law first.
GAIT_FREQ_MIN_HZ = 2.0

#: The moment arm turning is charged at, in metres: base to outermost foot, so
#: that `|wz| * r` is the foot speed a spin asks for. The module docstring's
#: ceiling arithmetic uses the same 0.20 m.
GAIT_TURN_RADIUS = 0.20

#: The cadence at the top of the command range, and the fixed cadence when
#: `VARIABLE_CADENCE` is off. **Derived, not chosen**: a leg supports for half a
#: cycle, so `v_max = stride x 2 x freq` and two of the three are decisions made
#: elsewhere.
#:
#: 0.8 / (2 x 0.072) = 100/18 Hz = 5.5556, which is **18 control steps a cycle at
#: 100 Hz -- whole and even**, so `phase.py`'s requirement is met rather than
#: worked around. It is written as the quotient of the two numbers it comes from
#: so that changing either cannot leave it stale; `tripod/mdp/phase.py` records a
#: 12.2% left-right split from a frequency that was written as a truncated
#: decimal instead.
#:
#: 9 control steps of swing, against the 5 `phase.py` names as the practical floor.
GAIT_FREQ_HZ = COMMAND_LIN_CEILING / (2.0 * GAIT_STRIDE)

#: The fastest it is ever asked to spin, in rad/s. **Derived, like the cadence.**
#:
#:     wz_max = 2 * GAIT_STRIDE * GAIT_FREQ_HZ / GAIT_TURN_RADIUS = 4.0
#:
#: A turn asks the outermost foot for `wz * r`, and the gait can deliver at most
#: `2 * stride * freq` of foot travel per support. So the same relation that fixes
#: the cadence from the linear ceiling fixes the angular ceiling from the cadence:
#: 4.0 rad/s is where a pure turn uses the whole gait, exactly as 0.8 m/s is.
#:
#: **1.25 was the old value and it was inert.** It asks for 0.25 m/s of foot speed
#: while the cadence floor already delivers `2 * freq_min * stride` = 0.288, so a
#: pure turn ran clamped at `GAIT_FREQ_MIN_HZ` however fast it was commanded --
#: the yaw term in `mdp/cadence.py`'s law did nothing for it. It had cleared the
#: floor by 4% at the 0.06 m stride this task briefly used; the stride moved and
#: this did not, which is how a paired number goes stale.
#:
#: **What it costs is the yaw ruler at the top of the ladder.**
#: `track_angular_velocity`'s std is `ang_std_ratio` times *the rung's* range, so
#: it is 14 deg/s at rung 0 and 92 at the top -- and 61 on the precision rung. A
#: 92 deg/s ruler is blind to anything slower than a brisk turn, which is already
#: a known problem at a zero command: a robot drifting at 1.6 deg/s scores 0.993
#: of 1.0. Widening the range widens the ruler with it; that is the scaled-ruler
#: design working as intended, and it is also why the drift is invisible.
#:
#: **229 deg/s is not known to be reachable.** Nothing has measured whether this
#: robot can spin that fast without slipping, and the usual silent failure applies
#: -- an unreachable command becomes a tracking reward that cannot be collected.
#: The ladder is the protection: it only arrives at a rung by clearing the one
#: below, so an unreachable top shows up as `Curriculum/command/level` parked at
#: the rung where turning stopped working, which is a fact rather than a guess.
COMMAND_ANG_CEILING = 2.0 * GAIT_STRIDE * GAIT_FREQ_HZ / GAIT_TURN_RADIUS

#: The command ladder's rungs, as fractions of the two ceilings. **This task's
#: own**, where every other task takes `common/mdp/curriculum.ladder`'s.
#:
#: The shared ladder is three rungs, `(0.5, 0.7, 1.0)` linear against
#: `(0.4, 2/3, 1.0)` angular, and it was written for an angular ceiling of 0.75 to
#: 1.25 rad/s. Against 4.0 its first rung is 1.6 rad/s -- above this task's entire
#: previous ceiling -- so a run would start by being asked to turn faster than any
#: policy here has ever turned. Three rungs over a range that tripled is not the
#: same curriculum; it is the same word for a coarser one.
#:
#: Five rungs, and the angular fractions are spaced to keep the *first* one where
#: the robot is known to cope:
#:
#:     rung   linear    angular            a pure turn needs
#:       0    0.24 m/s  0.60 rad/s   34 deg/s    2.00 Hz  (clamped at the floor)
#:       1    0.40      1.20         69          2.00 Hz  (clamped)
#:       2    0.56      2.00        115          2.78 Hz
#:       3    0.68      3.00        172          4.17 Hz
#:       4    0.80      4.00        229          5.56 Hz  (the whole gait)
#:       5    0.80      4.00        229          5.56 Hz  (precision rung)
#:
#: The linear fractions keep the shared ladder's 0.5, 0.7 and 1.0 and insert a
#: gentler start and a finer approach; one level is one rung on both axes, so the
#: two lists have to be the same length whether or not the linear axis needed the
#: extra rungs.
#:
#: Turning is free of the gait for the first two rungs and gait-limited after --
#: the cadence only starts responding to `wz` at rung 2. That is not a flaw in the
#: spacing; it is where the floor sits, and it means the first two rungs test
#: whether the robot can turn at all before they test whether it can turn fast.
LADDER_LIN_FRACTIONS = (0.3, 0.5, 0.7, 0.85, 1.0)
LADDER_ANG_FRACTIONS = (0.15, 0.3, 0.5, 0.75, 1.0)

#: The lean the robot is asked for **while walking**, in radians: 15 degrees on
#: all three axes, the command's `moving` band.
#:
#: **One number for three axes is a simplification, not a measurement.** Twisting
#: the body over planted feet, pitching it and rolling it are three different
#: demands on the legs and there is no reason they should run out at the same
#: angle. 15 degrees is what the task was asked for; whether any of the three is
#: the binding one is a question for `Metrics/posture/error_*` after a run.
#:
#: It is also the ruler: the curriculum's stds and promotion bars are taken from
#: this band's ladder, `POSTURE_ANGLES`, whichever band a command was drawn from.
MOVE_LEAN = math.radians(15.0)

#: The lean the robot is asked for **while parked**, in radians -- the command's
#: `ranges`, and what a stick at full deflection reaches.
#:
#: **These are `jumper.five_foot`'s bands, and they are here because they are
#: its**: the two tasks' pose commands reach the robot on one set of channels and
#: one stick layout, so a full deflection has to mean the same lean in both. Five
#: foot has them from kk-rl-lab, which measured and trains +-20 degrees of pitch,
#: +-15 of roll and +-30 of twist standing and +-15 on every axis walking, on the
#: same legs. Standing, nothing is swinging and the body can be asked for more;
#: a leg in mid-swing needs its amplitude back, which is why walking stays at
#: `MOVE_LEAN`. Written out rather than imported, so that a change to either
#: task is a diff here too -- and `tests/test_posture.py` fails when they part.
#:
#: **Not measured on six legs.** Five feet with a claw carried is the harder
#: stance, so these should be within reach, but "should" is the word:
#: `Metrics/posture/error_*` from the first run on these bands is what settles it.
STAND_TWIST = math.radians(30.0)
STAND_PITCH = math.radians(20.0)
STAND_ROLL = math.radians(15.0)

#: Below this norm of the velocity command's `[vx, vy, wz]` a posture is drawn
#: from the standing band. `jumper.five_foot`'s `stand_threshold`, for the reason
#: above: the controller applies one rule, from the contract, to both.
STAND_THRESHOLD = 0.06

#: Height above the ground the base may be asked to hold, in metres.
#:
#: The robot stands at `STAND_Z` = 0.107, so this is 37 mm of squat and 43 mm of
#: extension -- very nearly symmetric, which it was not designed to be and is.
#:
#: **The top was 0.12 on a claim that turned out to be wrong**, and the claim is
#: worth recording because it was the kind that never fails loudly: "the legs are
#: near straight at HOME and there is very little extension left in them". Nobody
#: had measured it. Measured two ways --
#:
#: *Reach*, by damped-least-squares IK per leg with the joint limits on, feet
#: pinned at their HOME xy: every height from 0.107 to 0.16 solves to a residual
#: under 1 um. There is no kinematic ceiling anywhere near here.
#:
#: *Holding it*, which is the question reach does not answer: the six legs IK'd
#: onto the ground, then the task's own PD (kp 10 / kd 0.5, torque capped at
#: `EFFORT_LIMIT`) left to hold the pose against gravity for 8 s on a plane.
#:
#:     target   settles at   sag     peak torque   of peak / continuous   limit slack
#:     0.070      0.0647     5.3 mm     0.350        20.0% / 29.1%          2.5 deg
#:     0.107      0.1050     2.0        0.253        14.5% / 21.1%         30.1
#:     0.120      0.1192     0.8        0.285        16.3% / 23.8%         41.8
#:     0.140      0.1417    -1.7        0.293        16.8% / 24.4%         43.1
#:     0.150      0.1539    -3.9        0.242        13.8% / 20.2%         43.1
#:
#: **The binding end is the squat, not the extension.** At 0.07 the tightest joint
#: has 2.5 degrees of travel left and the robot sags 5.3 mm; at 0.14 it has 43
#: degrees and holds the height to within 2 mm. Standing tall costs *less* torque
#: than standing at HOME, because a straighter leg carries the load closer to its
#: own axes.
#:
#: **The top is the last height that was actually held, not the last that solves.**
#: IK reaches 0.16 and beyond; the table stops at 0.15 because that is where the
#: standing experiment stops, and a ceiling taken from the reach column would be a
#: command backed by kinematics alone. The two questions have different answers and
#: only one of them is about whether the robot can do it.
#:
#: At 0.15 the robot settles 3.9 mm *above* the target rather than below it -- the
#: PD overshooting a pose it barely has to work for, since that row also has the
#: lowest torque in the table. Harmless, and the reason the number to watch on a
#: trained policy is a *signed* height error rather than a magnitude.
#:
#: What this does **not** measure is holding a tall posture while walking at
#: 0.8 m/s -- the table is a static stand. `Metrics/posture/error_height` bucketed
#: by commanded height is what would, and it needs a trained policy first.
HEIGHT_RANGE = (0.07, 0.15)

#: The command's ladder: `+/- 7.5, 10, 15` degrees on the angles and the same
#: fractions of the height's excursion, then 15 again as the precision rung,
#: derived rather than written out so that moving `MOVE_LEAN` or `HEIGHT_RANGE`
#: moves the rungs with it. See `mdp/curriculum.py` for the shape, the
#: measurement behind its bars, and `POSTURE_STD_SCALES` for the precision rung's.
POSTURE_ANGLES, POSTURE_HEIGHTS = posture_levels(MOVE_LEAN, HEIGHT_RANGE, STAND_Z)

#: The standing band's ladder: the same fractions of `STAND_TWIST`, `STAND_PITCH`
#: and `STAND_ROLL`, one tuple per axis. It moves what a parked robot is asked
#: for and nothing else -- the rulers stay on `POSTURE_ANGLES`.
POSTURE_STAND_ANGLES = band_levels(
    {"twist": STAND_TWIST, "pitch": STAND_PITCH, "roll": STAND_ROLL}
)


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
    """
    cfg = velocity_env_cfg(
        asset=asset,
        play=play,
        gait_reward=tripod_gait,
        gait_name="tripod",
        # **`jumper.tripod`'s -0.1, doubled, and the doubling is not a retune.**
        # This term costs the squared step-to-step change of the raw action, and
        # this task runs the control loop at 100 Hz rather than the skeleton's
        # 50 (see the rates block below). The same trajectory then has half the
        # per-step change, so the per-step cost falls 4x and the steps double:
        # the cost per second of identical motion halves. -0.2 restores it.
        #
        # So this is the same weight as tripod's in the only units that mean
        # anything across two control rates, and the comparison with tripod is
        # preserved rather than broken by it. `pose` and every other per-step
        # term needs no such correction -- they price a state, not a difference,
        # so their per-second value is already unchanged.
        action_rate_weight=-0.4,
        pose_weight=1.0,
        # **20 mm.** Both foot terms take this. `foot_clearance` charges
        # `max(0, target - height) x |horizontal foot speed|` every step -- one
        # sided since a foot carried higher than it needs to be is not what that
        # term is about -- and `foot_swing_height` charges `|peak - target|` once
        # per swing, symmetric.
        #
        # 25 mm was a target the robot never reached. Measured at iteration 38310,
        # `Metrics/peak_height_mean` is **0.0167**, and it has sat near there
        # through every run at this control rate -- 9 control steps of swing at the
        # top cadence is not much time to lift a foot and put it back. A target
        # nothing gets near is a near-constant penalty whose gradient always says
        # "higher", which is a term that cannot discriminate: it pays the same
        # whether the swing is improving or not.
        #
        # **20 mm is just above what the robot does**, where 15 was just below.
        # That matters for `foot_swing_height`, which is symmetric and therefore
        # changes sign across the target: at 15 it pushed the lift *down* from
        # 16.7 mm, and at 20 it pushes up again, by 3.3 mm rather than the 8.3 it
        # was asking for at 25. A target the robot is close to on either side is a
        # term that can say "a little more" and be believed; one it is nowhere
        # near is a constant.
        #
        # `foot_clearance` is one-sided and does not change sign, so for it this
        # is simply a 5 mm higher floor under the height at which a foot may
        # translate.
        #
        # 20 mm of clearance is judged enough for the flat ground this task trains
        # on. It stops being enough on `--scene rough`, where `random_rough`'s
        # peak-to-peak is 0.10 m, and this is the first number to raise if the
        # feet start catching there.
        foot_target_height=0.020,
        # See the module docstring: the linear ceiling and `GAIT_FREQ_HZ` are one
        # decision, not two.
        command_lin_ceiling=COMMAND_LIN_CEILING,
        command_ang_ceiling=COMMAND_ANG_CEILING,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
        # ── The operator's controls ──────────────────────────────────────
        # This task's own `controls.yaml`, schema 2: both commands on one pad
        # and on the keyboard beside it, and the height moved rather than
        # placed. The replay block at the end installs the posture half.
        controls=CONTROLS,
    )

    # ── The rates ─────────────────────────────────────────────────────────
    # **200 Hz control on 1000 Hz physics**, against the skeleton's 50 on 200.
    # Task-local: the four locomotion tasks are each other's controls and share
    # the skeleton's rates, so changing them there would silently re-scale every
    # per-step number in all four. `SIM_DT` and `DECIMATION` are the only two
    # values set here; everything below that moves is derived from them and is
    # marked as such.
    #
    # **What it buys is the swing's resolution, and this time that is the point
    # rather than a side effect.** At the 5.5556 Hz cadence ceiling a cycle is 36
    # control steps and a swing is 18, against 9 at 100 Hz. `mdp/cadence.py` ends
    # its note on the stride by saying so in advance: 9 steps of swing "is the
    # number to watch if feet start clipping the ground ... and it is a reason to
    # raise the control rate rather than to lower the cadence".
    #
    # Feet did not start clipping the ground; they started arriving at 0.63 m/s,
    # driven down at 1.94 g over the last four steps of the swing. Four steps is
    # what a policy had to shape the approach with. Eight is not obviously
    # enough either, but it is the resolution `soft_touchdown` is asking the
    # policy to spend and it did not have it.
    #
    # Both ends of the cadence clamp stay whole and **even**, which
    # `mdp/cadence.py` requires or the two tripod groups get unequal time: the
    # ceiling is 36 steps a cycle and the 2.0 Hz floor is 100.
    #
    # **What it costs is throughput**: twice the physics steps per second of
    # simulated time and twice the inferences, so the same simulated seconds take
    # about twice the wall clock. A run that took a night takes two.
    #
    # ── Three numbers had to move with it, and each would have been silent ──
    #
    # `action_rate_l2` is the one number this rate change got wrong twice, in
    # opposite directions, and both arguments are kept because the second one is
    # the instructive failure.
    #
    # It penalises the step-to-step change of the raw action. **The same
    # trajectory** at twice the control rate has half the per-step change, so the
    # squared cost falls 4x against only 2x the steps and the cost per second
    # halves. That reasoning doubled the weight at the 50 -> 100 change and
    # doubled it again here, to -0.4.
    #
    # It looked refuted. The per-step quantity had not fallen at all:
    #
    #     100 Hz  08-23-12   -1.9216 at -0.2   raw 9.61
    #     200 Hz  10-57-55   -3.7938 at -0.4   raw 9.43
    #
    # -- explained by exploration noise, which is injected per step and does not
    # shrink when the step does (`Policy/mean_std` 0.58 at 6900 iterations). So
    # the weight went back to -0.2, to restore the 19.9% share of the reward
    # budget the 100 Hz run learned its gait under, against the 32.1% it had
    # grown to while `soft_touchdown` sat at 1.0% and would not move.
    #
    # **That was wrong, and the error is the shape of it: the raw quantity was
    # measured under one weight and treated as a property of the rate.** It is
    # not. It is an equilibrium the policy chooses, and the policy moved it.
    # Resumed from `10-57-55/model_6950` at -0.2, 1200 iterations:
    #
    #                            at -0.4      at -0.2
    #     Episode_Reward          -3.79        -3.51      -7%
    #     raw quantity             9.43        17.54      +86%
    #     Policy/mean_std          0.58         0.79
    #     soft_touchdown          -0.118       -0.116     unchanged
    #
    # **Halving the weight bought a 7% reduction in cost and a doubling of the
    # motion.** The share of the budget barely moved, 32% to 30%. The robot began
    # stepping in place under a zero command, which it had not done, and the
    # operator saw it before any of these numbers said so.
    #
    # Back to -0.4. What this closes off is the manoeuvre, not just the value:
    # **budget cannot be freed for one term by lightening another.** A term that
    # is not biting has to be raised on its own, which is what `soft_touchdown`
    # needs and is a separate change with a separate run.
    #
    # One caveat on the attribution, since the comparison is not clean: a resume
    # restarts the curriculum at level 0, so the easier commands are also pushing
    # `mean_std` up. They do not explain the raw quantity doubling -- easier
    # commands should mean *less* action change, not 86% more.
    #
    # `gamma` and `lam` are per step, so the discount horizon is in steps and
    # quarters in seconds: 0.99 is a 1.99 s horizon at 50 Hz and 0.50 s at 200.
    # `rl_cfg.py` carries the corrected pair and the rollout length with it.
    #
    # The self-collision sensor buffers one entry per physics substep and the
    # skeleton sizes it from `decimation` **while building**, before this runs.
    # Left alone it would hold 4 of the 5 substeps and the collision count would
    # be short by a fifth, with nothing to show for it.
    #
    # ── What did not move, and is worth knowing ──
    #
    # The observation history **is** restored, and by striding rather than by
    # widening -- the fourth number that had to move, and the one the last rate
    # change left undone. `mdp/history.py` has the argument; the short form is
    # that `OBS_HISTORY = 5` was chosen at 50 Hz, where consecutive frames are
    # 20 ms apart and the oldest reaches back 80 ms; at 200 Hz they are 5 ms
    # apart and reach back 20 ms. Taking the same five frames 4 control steps
    # apart makes them 20 ms apart again -- **the same history the 5 was chosen
    # for**, not merely a similar one -- for an actor input that does not change.
    #
    # PPO's rollout is 48 steps rather than 24 and so covers the 0.24 s it did.
    # Every curriculum dwell is in iterations and so is unchanged in iterations
    # and quartered in seconds against 50 Hz.
    #
    # **The deployment takes the rate from the contract, and this was checked
    # rather than assumed.** `layout.json` carries `control_hz` and
    # `rl-wbc-fsm/src/app.cpp` sets the inference period from it on every model
    # entry, in so many words: "policies do not agree on their rate
    # (locomotion/carry 50 Hz, the reference-residual jump 200 Hz)". The loop
    # itself ticks at `output_rate_hz` = 1000 and holds the last PD target
    # between inferences, so 100 Hz is inference every 10 ticks rather than 20.
    # No C++ change, and the jump already runs at twice this.
    cfg.sim.mujoco.timestep = SIM_DT
    cfg.decimation = DECIMATION
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "self_collision":
            sensor.history_length = DECIMATION

    # ── The history spans a duration, not a step count ────────────────────
    # `mdp/history.py` has the whole argument. What has to be right here is the
    # handover, and it has three halves that fail silently on their own:
    #
    #   * the manager's `history_length` goes to **0**, or the frames are stacked
    #     twice and the actor's input is five times what the layout says;
    #   * the term's `noise` moves **into** the wrapper, or a buffered frame is
    #     re-noised on every step and the policy can average away an error no
    #     robot can;
    #   * `common/mdp/symmetry.py` learns the count from `history_frames`, or it
    #     mirrors five frames as one wide observation and the mirrored sample is
    #     a robot whose left legs lag its right by four control steps.
    #
    # Applied to every group, because the critic is mirrored too and a critic
    # whose history is a different age than the actor's is a value function
    # fitted against a different observation.
    #
    # **And each group wraps its own copy of the term.** mjlab's skeleton builds
    # the critic as `{**actor_terms, ...}`, so `joint_vel` and `actions` are one
    # object in both groups. Unwrapped, that is harmless -- the manager clears
    # the noise per group, on its own copy. Wrapped in place, it was not: the
    # actor's pass put its noise into the shared params, the critic's pass found
    # `history_length` already 0 and skipped the term, and the critic --
    # uncorrupted by design -- read the actor's +/-1.5 rad/s into its `joint_vel`
    # history. No width changed and the buffers were never shared, since the
    # manager deep-copies each group's terms; only the noise was, riding in the
    # params. `tests/test_posture.py` holds that noise to the training actor.
    for _group in cfg.observations.values():
        if _group.history_length:
            raise ValueError(
                "a group-level history_length overwrites every term's during "
                "_prepare_terms, so the per-term rewiring below would be undone "
                "without changing any shape; set it per term or not at all"
            )
        for _name, _term in list(_group.terms.items()):
            if _term is None or not _term.history_length:
                continue
            # Shallow is enough: every field below is reassigned, `params`
            # included, never mutated.
            _term = _group.terms[_name] = copy.copy(_term)
            p = StridedHistory.PREFIX
            _term.params = {
                # The wrapped term's own params stay where every consumer of a
                # term config expects to find them -- `export.py` sizes the
                # deployment's per-joint blocks from `params["asset_cfg"]`, and
                # the manager resolves `SceneEntityCfg`s at this level only.
                # `mdp/history.py` has the three things that broke when they were
                # nested one dict down.
                **(_term.params or {}),
                p + "func": _term.func,
                p + "frames": _term.history_length,
                p + "stride": OBS_HISTORY_STRIDE,
                # `enable_corruption` is the group's switch and the manager
                # applies it by clearing the term's noise; do the same here or a
                # replay -- which turns corruption off -- keeps the noise this
                # wrapper owns.
                p + "noise": _term.noise if _group.enable_corruption else None,
            }
            assert not any(
                k.startswith(p) for k in (_term.params.keys() - {
                    p + "func", p + "frames", p + "stride", p + "noise"
                })
            ), f"{_name} has a param colliding with {p!r}"
            _term.func = StridedHistory
            _term.history_length = 0
            _term.noise = None

    #: The six foot sites, in `LEGS` order, which the twist estimate requires.
    #: `preserve_order` is what keeps it in that order; `state._check_foot_order`
    #: raises if anything still reorders it.
    foot_cfg = SceneEntityCfg("robot", site_names=LEGS, preserve_order=True)

    #: The 20 actuators the policy drives. The two grippers are held closed by
    #: their PD and are not the policy's to spend: including them would add a
    #: near-constant to both actuator terms below, which is a bias on a penalty
    #: rather than a signal.
    actuator_cfg = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))

    # ── The posture command ───────────────────────────────────────────────
    cfg.commands[POSTURE_COMMAND] = PostureCommandCfg(
        # The same 3 to 8 s the velocity command resamples on. Deliberately the
        # same clock rather than a slower one: a posture that outlived several
        # velocity commands would correlate the two, and the policy would learn
        # "this lean goes with that speed" from the sampler rather than from the
        # physics.
        resampling_time_range=(3.0, 8.0),
        foot_sites=LEGS,
        # A fifth of the **moving** environments hold the neutral posture
        # exactly. See `mdp/commands.py`: a uniform draw over four axes never
        # produces "level, square and standing", and that is the posture the
        # tripod gait this task inherits was learned at.
        rel_neutral_envs=0.2,
        # **And a fifth of the parked ones**, which was zero and is what left a
        # robot told nothing standing somewhere other than HOME.
        #
        # Zero bought the "hold a pose while parked" case all of the standing
        # time, on the argument that idle-at-home would still arrive from the
        # *moving* draw. It does, and measured it is thin: the two commands
        # resample on their own 3-8 s clocks, so a posture drawn neutral while
        # moving outlives the walk on about **2%** of training time (a Monte Carlo
        # of the two clocks). `2026-09-28_17-32-26/model_47000` was trained on
        # that 2% and, told nothing at all, holds its middle legs 0.37 rad off
        # HOME and its nose 3.2 degrees up -- a stance the reward itself scores
        # 0.39 a step *below* HOME. `mdp/rewards.py::pose_widened_by_posture` has
        # the measurement. The same Monte Carlo, with `rel_standing_envs` at 0.3:
        #
        #     neutral while parked   parked and neutral   parked and posed
        #           0.0                     2.1%               28.0%   (before)
        #           0.2                     6.0%               24.0%   <- here
        #           0.3                     8.0%               22.1%
        #
        # 0.2 because it is the moving draw's rate, so a fifth of every command is
        # neutral whether the robot is walking or not, and it costs the posed case
        # a seventh of its time rather than a quarter. The pull towards HOME is
        # not this number's job -- `pose`'s parked width below is what makes being
        # at HOME worth something here -- this is only what gives that width
        # samples to act on.
        rel_neutral_standing_envs=0.2,
        neutral_height=STAND_Z,
        # The standing band, which is what `ranges` means for the angles: the
        # widest the command goes and what a full stick reaches. Walking draws
        # from `moving`; `STAND_THRESHOLD` decides which.
        ranges=PostureCommandCfg.Ranges(
            twist=STAND_TWIST,
            pitch=STAND_PITCH,
            roll=STAND_ROLL,
            height=HEIGHT_RANGE,
        ),
        moving=PostureCommandCfg.Moving(
            twist=MOVE_LEAN,
            pitch=MOVE_LEAN,
            roll=MOVE_LEAN,
        ),
        stand_threshold=STAND_THRESHOLD,
    )

    # ── What it is paid for ───────────────────────────────────────────────
    # **1.0 each, against the velocity terms' 2.0 each.** Three new positive terms
    # worth 3.0 on a task whose reward budget was 4.0 of tracking, and the size is
    # the thing to watch rather than the shape.
    #
    # The risk is the one this robot keeps falling into: a positive term that a
    # standing robot can collect makes standing more attractive. All three of
    # these are collectable while standing -- a stationary robot can hold any
    # commanded posture perfectly -- and they are *harder* to collect while
    # walking, so they tilt the trade towards standing exactly the way `pose` and
    # `action_rate_l2` were measured to.
    #
    # What keeps that bounded is that they do not scale with the tilt: half of the
    # reward budget is still the two velocity terms, and they pay nothing to a
    # robot that does not move. If the policy does settle into standing, the tell
    # is `Metrics/twist/error_vel_xy` flat near the command magnitude while
    # `Episode_Reward/track_tilt` and `track_height` climb, and these three weights
    # are the first thing to halve -- not `action_rate_weight`, which is holding
    # the same job it holds in `jumper.tripod`.
    #
    # They are equal to each other because nothing yet says they should not be.
    # Twist, lean and height are three axes of one command and no measurement
    # distinguishes their difficulty; the first run is what would.
    #
    # **The stds here are the ladder's top rung, not the range the run starts
    # at** -- the precision rung, so they carry its scale: 3.75 degrees and
    # 10 mm, half of level 2's. The curriculum below writes level 0's narrower
    # pair in before the first command is drawn, so these are what training
    # *ends* on -- and they
    # are also what `play.py` replays with, since it builds with `curriculum =
    # {}` and never runs the curriculum at all. Taken from `angle_std` /
    # `height_std` rather than written out, so that the two cannot disagree; the
    # velocity terms take theirs from `lin_std(-1)` for exactly that reason.
    posture_std = angle_std(-1, POSTURE_ANGLES, POSTURE_STD_SCALES)
    cfg.rewards["track_twist"] = RewardTermCfg(
        func=track_twist,
        weight=1.0,
        params={
            "std": posture_std,
            "command_name": POSTURE_COMMAND,
            "asset_cfg": foot_cfg,
        },
    )
    cfg.rewards["track_tilt"] = RewardTermCfg(
        func=track_tilt,
        weight=1.0,
        params={
            "std": posture_std,
            "command_name": POSTURE_COMMAND,
            "asset_cfg": foot_cfg,
        },
    )
    cfg.rewards["track_height"] = RewardTermCfg(
        func=track_height,
        weight=1.0,
        params={
            # The same ratio applied to the height's own half-range, at the
            # ladder's top rung, with the precision rung's scale. 0.010 m.
            "std": height_std(-1, POSTURE_HEIGHTS, POSTURE_STD_SCALES),
            "command_name": POSTURE_COMMAND,
            "asset_cfg": foot_cfg,
        },
    )

    # ── What the actuators are spending ───────────────────────────────────
    # Two terms about the servos rather than about the gait, and they measure
    # different halves of the same worry. `actuator_power` charges for moving a
    # load; `actuator_headroom` charges for being near the torque the servo can
    # currently deliver, which a robot holding a heavy pose at zero speed is,
    # while spending no mechanical power at all.
    #
    # **Both are sized against a measurement of the running policy**, taken with
    # `play.py --measure` on `model_10550` of the run of 2026-09-17, 400 control
    # steps of the deterministic policy at the full command range:
    #
    #     mechanical power   11.51 W summed over 22 actuators, 0.523 W each
    #                        8.55 W peak in one actuator
    #     |tau| / capability mean 0.141, p90 0.463, above 0.687 for 5.2% of
    #                        (step x actuator) samples
    #     saturation         some actuator pinned at EFFORT_LIMIT on 40.8% of
    #                        steps; the two front shoulder_pitch joints carry it,
    #                        at 18-21% duty each and 20-28% of the time above the
    #                        1.2 N*m the servo can hold indefinitely
    #
    # That last line is why the headroom term is not cosmetic, and the reason is
    # **not** the one written here first, which was wrong and is worth recording
    # as such: "nothing in simulation models that, so nothing else in this reward
    # can see it". Simulation models it exactly.
    # `common/actuator.py::ServoCurveActuator.compute` carries a normalised
    # winding temperature per joint and drops that joint's ceiling to
    # `continuous_torque` once it trips -- measured on this config, 600 steps of
    # hard driving takes theta to 1.011 and derates on 172 of them. The hardware
    # does the same thing in the servo itself, so neither side needs to be told.
    #
    # What the term adds is **where the gradient is**. The thermal model prices
    # the *consequence*: nothing at all until theta crosses 1, then a lost
    # ceiling that shows up as worse tracking, with no indication of what caused
    # it. This prices the *approach*, continuously, before anything trips. It is
    # the same relation `stance_foot_ground_gap` has to contact force -- force is
    # exactly zero until the foot arrives, so it says nothing about getting
    # closer -- and the same argument for a deadband: below it, torque is free.
    #
    # The second half is not thermal at all. An actuator at its curve limit has
    # no torque left to reject a disturbance with, whatever its temperature, and
    # that is control authority rather than heat.
    #
    # **Measured, and it works**: two checkpoints 750 iterations apart in the run
    # of 2026-09-17 08:38, replayed on the same scene at the same command range,
    # 400 steps x 16 envs each. Saturation duty 1.9% -> 1.1%, time above the
    # continuous rating 6.1% -> 4.3%, mechanical power 0.673 -> 0.303 W. The peak
    # torque did not move and cannot: it is pinned at the clamp for as long as
    # any saturation remains, which is why duty is the number to read.
    #
    # What did not improve is the pair of front `shoulder_pitch` joints -- four
    # knees left saturation entirely while those two stayed put, one of them
    # doubling its time above the continuous rating. A penalty redistributes load
    # it cannot remove; that pair looks structural rather than wasteful.
    cfg.rewards["energy"] = RewardTermCfg(
        func=actuator_power,
        # **Weak on purpose, and the arithmetic is the whole justification.** At
        # the measured 0.523 W per actuator this is -0.0105 per step, against
        # `action_rate_l2`'s -1.4 and a tracking term's +1.0. About 1% of the
        # budget: enough to break a tie between two gaits that track equally
        # well, not enough to make a slower gait worth more than tracking.
        #
        # That is the same size `standing_sway` was calibrated to and for the same
        # reason -- it puts the quantity in the reward breakdown where it can be
        # watched before anyone decides it should bite. If `Episode_Reward/energy`
        # stays flat while the gait changes, it is inert and can be raised; if
        # walking speed drops when it goes in, it is already too strong.
        weight=-0.02,
        params={"asset_cfg": actuator_cfg},
    )
    cfg.rewards["torque_headroom"] = RewardTermCfg(
        func=actuator_headroom,
        # At the measured distribution this costs -0.027 per step, which puts it
        # beside `foot_clearance` (-0.11), `body_tilt_rate` (-0.03) and
        # `foot_slip` (-0.02) rather than beside the objectives. It is meant to
        # bite on the 5% of samples that are above the deadband and to cost
        # nothing on the other 95%.
        #
        # **The number to watch is not this term's own value.** It is
        # `Metrics/...` -- there is none -- so it is a `--measure` run: the
        # saturation duty on the two front shoulder_pitch joints, which is what
        # this term exists to bring down. If it falls and tracking holds, raise
        # the weight; if tracking falls first, the robot needs that torque and
        # the answer is a lower speed ceiling rather than a heavier penalty.
        weight=-1.0,
        params={
            # 1.2 / 1.7464, the share of peak torque the servo holds
            # indefinitely. See `actuator_headroom` for why that identity is a
            # starting point rather than a derivation.
            "deadband": CONTINUOUS_TORQUE / EFFORT_LIMIT,
            "asset_cfg": actuator_cfg,
        },
    )

    # ── And the two terms that would have argued with them ────────────────
    # `upright` pays for pitch = roll = 0, which is half this task's command range
    # asking for the opposite. `track_tilt` is the command-relative version of it
    # and is the same function at a zero command, so this is a replacement rather
    # than a deletion. See the module docstring.
    cfg.rewards.pop("upright", None)

    # `pose`'s standing width is six times its walking one, and holding a
    # commanded posture is precisely leaving HOME -- at 15 degrees of twist every
    # leg joint is displaced, and the standing width (0.10 arm / 0.05 leg) reads
    # that as failure. So a parked robot at a commanded posture is held to the
    # walking width, what a walking one is held to and no more.
    #
    # **But not at the neutral posture**, where HOME is the answer and the walking
    # width let the policy settle 0.37 rad away from it on the middle hip yaws
    # (`mdp/rewards.py::pose_widened_by_posture` has the measurement). The parked
    # width now ramps with how far the posture command is from neutral:
    #
    #   `std_neutral`    **`jumper.tripod`'s standing width**, the skeleton's
    #                    doubled by the same expression tripod uses. Tripod is
    #                    this task's control and its parked robot is only ever
    #                    at the neutral posture, so at that posture this one is
    #                    held to exactly what tripod holds its own to. At the
    #                    measured stance it scores 0.13 where the walking width
    #                    gave 0.80 -- the pull back to HOME goes from 0.2 a step
    #                    to 0.87.
    #   `std_standing`   the walking width, as before, reached at `full_width_at`.
    #   `full_width_at`  **0.1 of the reach**: 3 degrees of twist, 2 of pitch,
    #                    1.5 of roll, 4 mm of height. Where a posture's own
    #                    displacement has already grown to the neutral width:
    #                    parked on model_47000, +/-0.1 rad of twist moved the
    #                    middle and rear hip yaws 0.07-0.19 rad from where they
    #                    stood at neutral, so 3 degrees moves them 0.04-0.10 --
    #                    the leg's 0.10. Past that the tight width would be
    #                    charging the command, not the stance, and a policy paid
    #                    to under-track small postures to stay near HOME.
    #   `posture_reach`  the standing band per angle, and the height's
    #                    half-range -- the same half-range `height_std` measures
    #                    by, so the squat and the extension are one ruler here too.
    #
    # `std_walking` and `std_running` are untouched: what a moving leg may do is
    # the same question it was.
    _pose = cfg.rewards["pose"]
    cfg.rewards["pose"] = RewardTermCfg(
        func=pose_widened_by_posture,
        weight=_pose.weight,
        params={
            **_pose.params,
            "std_neutral": {k: 2.0 * v for k, v in _pose.params["std_standing"].items()},
            "std_standing": dict(_pose.params["std_walking"]),
            "posture_command_name": POSTURE_COMMAND,
            "neutral_height": STAND_Z,
            "posture_reach": (
                STAND_TWIST,
                STAND_PITCH,
                STAND_ROLL,
                (HEIGHT_RANGE[1] - HEIGHT_RANGE[0]) / 2.0,
            ),
            "full_width_at": 0.1,
        },
    )

    # ── The middle hip yaws, centred ──────────────────────────────────────
    # `pose` above holds the *parked* stance at neutral; it cannot hold the
    # walking one without charging the swing, which is as large as the drift it
    # would be charging (`mdp/rewards.py::hip_yaw_center` has both numbers). This
    # scores the centre of the middle hip yaws -- a cycle's average -- against the
    # angle the commanded posture needs, so the swing is free and the drift is not.
    #
    #   `std`     0.10 rad, `jumper.tripod`'s standing width on a leg joint, the
    #             same number `pose` holds a parked neutral stance to. At the
    #             drift `model_47000` reached walking (0.17) it scores 0.06; at the
    #             +/-0.02-0.09 `model_76400` held for 70000 iterations, 0.45-0.96.
    #   `tau`     0.5 s, one cycle at `GAIT_FREQ_MIN_HZ`: the slowest swing is
    #             averaged to a sixth of its amplitude, the fastest to a
    #             seventeenth.
    #   weight    **0.5**. At the measured drift that is a 0.47 a step pull back,
    #             against the 0.035 `pose`'s walking width managed -- and the drift
    #             bought nothing measurable, so the pull only has to outweigh noise.
    #             Half of `pose`'s weight, because it watches two joints to
    #             `pose`'s twenty. If `track_linear_velocity` drops when it goes in,
    #             the drift was buying speed on this model after all, and that is
    #             what the weight should be read against.
    cfg.rewards["middle_hip_yaw_center"] = RewardTermCfg(
        func=hip_yaw_center,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "legs": ("LM", "RM"),
            "footprint": LEGS,
            "posture_command_name": POSTURE_COMMAND,
            "neutral_height": STAND_Z,
            "std": 0.10,
            "tau": 1.0 / GAIT_FREQ_MIN_HZ,
        },
    )

    # `fell_over` terminates at 50 degrees of tilt and the command asks for up to
    # 15, on both axes at once -- so a robot doing exactly what it was told can be
    # 21 degrees off vertical, leaving 29 of margin. Left where it is: that is
    # still more than half the budget, and raising it would spend the margin on a
    # robot that is genuinely going over. Watch the termination breakdown if runs
    # start ending early at large tilt commands.

    # ── The phase clock, at this task's frequency ─────────────────────────
    # The same wiring `jumper.tripod` does, and for the same reason: the
    # observation and the gait reward must be handed the frequency from one
    # place, because a mismatch does not raise -- the policy is simply shown one
    # clock and scored against another, and the gait term quietly stops being
    # learnable while everything else looks healthy.
    #
    # `history_length=0` because the clock is a deterministic function of episode
    # time, so five frames of it are exactly determined by the current one. The
    # skeleton sets every term's history before this runs, so a term mounted here
    # gets none unless it asks.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
            func=phase_clock,
            history_length=0,
            params={
                "freq_hz": GAIT_FREQ_HZ,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        )
    cfg.rewards["tripod_gait"].params["freq_hz"] = GAIT_FREQ_HZ

    # ── The cadence follows the command ───────────────────────────────────
    # `VARIABLE_CADENCE` and `GAIT_STRIDE` have why; `mdp/cadence.py` has the law
    # and the clock.
    #
    # **The swap is on the observation term only, and that is the mechanism
    # rather than a shortcut.** mjlab builds the term once per group, so the
    # actor and the critic each get one; whichever it builds first hangs a
    # `CadenceClock` on the env and the other finds it there, so both groups
    # are shown one phase. The rewards reach that same object without being
    # rewired: `tripod_gait` and `stance_load` through
    # `tripod/mdp/phase.py::gait_phase`, `stance_ground_gap` through
    # `common/mdp/phase.py::gait_phase` -- both defer to an installed clock --
    # and the swing gate on `foot_clearance` by reading it directly.
    # (`stance_load_stand` reads no phase.) One object answers all six, which is
    # the property that matters: a second integrator would be a second phase,
    # and the reward would score the policy against a clock it was never shown.
    #
    # **That is what this had at first, twice over**, and nothing said so: each
    # group's term kept its own phase, the rewards read the critic's, and the
    # actor's ran a control step ahead of it after every reset; and
    # `stance_ground_gap` read neither, because `common`'s `gait_phase` did not
    # look for a clock yet. `mdp/cadence.py` has the numbers, and
    # `tests/test_posture.py` holds all six readers to one value through a
    # command redraw and a reset.
    #
    # `freq_hz` is left on the three reward terms and is **not read** while this
    # is on. Deleting it would look tidier and be worse: it is what they fall
    # back to the moment `VARIABLE_CADENCE` goes False, and a task that had
    # deleted it would silently run them at mjlab's default.
    if VARIABLE_CADENCE:
        for group in ("actor", "critic"):
            cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
                func=VariableGaitClock,
                history_length=0,
                params={
                    "command_name": "twist",
                    "stride": GAIT_STRIDE,
                    "freq_min": GAIT_FREQ_MIN_HZ,
                    "freq_max": GAIT_FREQ_HZ,
                    "turn_radius": GAIT_TURN_RADIUS,
                    # No `mirror_kind`: the term keeps the name `gait_phase` and
                    # `common/mdp/symmetry.py`'s table maps that name to
                    # `phase_half_shift` already. Declaring it here as well would
                    # be a second place for one fact to be right or wrong.
                    "command_threshold": 0.05,
                    # When a host advances the clock. Not a choice made here --
                    # the term refuses any other value -- but a fact the contract
                    # has to carry: policies exported before it existed were
                    # trained in the other order, and a host tells the two apart
                    # by this and nothing else. `mdp/cadence.py::ADVANCE`.
                    "advance": ADVANCE,
                },
            )

    # ── The stance terms, exactly as jumper.tripod wires them ───────────────
    # Imported rather than copied: they are the tripod gait's, and the numbers in
    # them are about how many legs carry the robot, which this task does not
    # change. The one thing that does change is `freq_hz`, which `stance_load`
    # and `stance_ground_gap` take, and it comes from the constant above --
    # though neither reads it while `VARIABLE_CADENCE` is on.
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
    cfg.rewards["stance_load_stand"] = RewardTermCfg(
        func=standing_foot_load,
        # **This weight and `rel_standing_envs` are the same lever**, and tripod's
        # note says the arithmetic has to be redone whenever either moves. It
        # moved: 0.2 -> 0.3 below, so the parity weight -- the one that makes the
        # standing half of this idea worth the same per episode as the walking
        # half, which is gated the other way -- moves with it.
        #
        #     standing   parity weight   stance_load_stand   stance_load
        #       10%          4.5              0.45              0.45
        #       20%          2.0              0.40              0.40   (tripod)
        #       30%          1.167            0.35              0.35   <- here
        #
        # 1.167 is `0.5 * 0.7 / 0.3`. Left at tripod's 2.0 it would have been
        # worth 0.60 against `stance_load`'s 0.35 -- 1.7x more for the standing
        # case, bought purely by how often each is switched on. That is the exact
        # imbalance the tripod note exists to prevent, arriving through the other
        # variable.
        weight=0.5 * 0.7 / 0.3,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "force_threshold": 3.0,
            "command_threshold": 0.05,
        },
    )
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

    # tripod's foot-lift weights, unchanged, and its standing share.
    # ── This task's own command ladder ────────────────────────────────────
    # `LADDER_*_FRACTIONS` has the rungs and why there are five of them. The
    # skeleton has already built the shared three-rung ladder from the two
    # ceilings; this replaces the lists rather than the shape, so the precision
    # rung, the std ratios and the gate all keep working as they do everywhere.
    #
    # **Written here rather than in `common/mdp/curriculum.py`** because that file
    # is four other tasks' ladder, and a rung added there is a rung they climb
    # without having chosen it -- which is the failure `jumper.tetrapod` is the
    # standing example of.
    if "command" in cfg.curriculum:
        from ..common.mdp.curriculum import PRECISION_ANG_SCALE, PRECISION_LIN_SCALE

        lin = tuple(round(COMMAND_LIN_CEILING * f, 6) for f in LADDER_LIN_FRACTIONS)
        ang = tuple(round(COMMAND_ANG_CEILING * f, 6) for f in LADDER_ANG_FRACTIONS)
        params = cfg.curriculum["command"].params
        # **The dwell is this task's to own**, and it was not. `CommandRangeCurriculum`
        # defaults to a literal `24 * 100` environment steps, which is 100
        # iterations only at a 24-step rollout -- this task's is 48, so the
        # inherited default paced the command ladder at 50 and nothing said so.
        # The same trap `mdp/curriculum.py::DWELL_STEPS` fell into, from the same
        # cause, one file away.
        #
        # Four tasks share that default and three of them still run a 24-step
        # rollout, so it is corrected here rather than there -- and writing it
        # here is what CLAUDE.md asks for anyway: a pacing number a task inherits
        # without choosing is the failure `jumper.tetrapod` lost months to.
        params["dwell_steps"] = NUM_STEPS_PER_ENV * 100
        # **And a floor under the first promotion**, which the dwell is not.
        # `CommandRangeCurriculum` defaults it to 0 because the four locomotion
        # tasks do not need one: at their 24-step rollout the same 2400-step
        # dwell already puts the first promotion at iteration 100. This task's
        # rollout is 48, so the dwell alone put it at 50 -- and at 200 Hz an
        # episode is 83 iterations, so that decision was made on the environments
        # that had ended *early*, which are the only ones that had ended.
        # `mdp/curriculum.py::WARMUP_STEPS` has the measurement.
        params["warmup_steps"] = WARMUP_STEPS
        params["levels"] = (*lin, lin[-1])
        params["ang_levels"] = (*ang, ang[-1])
        params["lin_std_scales"] = (*(1.0,) * len(lin), PRECISION_LIN_SCALE)
        params["ang_std_scales"] = (*(1.0,) * len(ang), PRECISION_ANG_SCALE)

    # ── A weak penalty on joint acceleration ──────────────────────────────
    # For smoothness that `action_rate_l2` does not buy. That term charges the
    # *command's* second difference; this one charges what the joint actually
    # does, `qacc`, which includes everything the command did not ask for --
    # impacts, the leg being dragged by the body, a servo hitting its limit.
    #
    # **The weight is measured, not guessed, because the quantity is not
    # order-one.** `sum(joint_acc^2)` over the 20 driven joints, model_9999 held
    # at a speed for 400 control steps, 64 environments:
    #
    #     vx = 0.0    mean 8.7e-03    p95 4.0e-02    peak |acc|   1.6 rad/s^2
    #     vx = 0.3    mean 6.2e+04    p95 4.5e+05    peak |acc| 920.8
    #     vx = 0.6    mean 1.5e+05    p95 8.1e+05    peak |acc| 992.8
    #
    # Seven orders of magnitude between standing and walking, so a weight picked
    # by eye is either inert or the whole objective. 4e-7 puts it at **0.061 a
    # step at 0.6 m/s** and 0.025 at 0.3 -- level with `foot_clearance` as it
    # measures out (0.067), well above `energy` (0.0015), and 1/60 of
    # `action_rate_l2` (3.86).
    #
    # It started at 2e-7 and half those values, described there as "weak enough to
    # be a preference and large enough to be one". At 4e-7 the first half of that
    # has gone: this is a shaping term the size of the foot-clearance one, and a
    # gait that comes out slower or lower is now a candidate rather than a
    # rounding error.
    #
    # It costs nothing while standing (8.7e-03 x 4e-7), which is the right shape:
    # the term is about how the gait moves, and a robot holding still has no
    # acceleration to charge for.
    #
    # **Unlike `action_rate_l2`, this weight does not move with the control
    # rate.** That term is a difference of successive actions, so the same
    # trajectory sampled twice as often has half the difference and a quarter the
    # square -- hence the 4x correction this file carries for it. `joint_acc` is
    # MuJoCo's `qacc`: a physical acceleration, the same number whatever rate
    # observes it. Rewards are scaled by `dt` before they are summed, so the
    # per-second cost is already rate-invariant and there is nothing to correct.
    # This is the counter-example worth keeping next to the rule.
    cfg.rewards["joint_acc"] = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-4.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))},
    )

    # ── Landing: charge the cause, measure the effect ─────────────────────
    # `soft_landing` charged `|force|` at first contact and was retuned six times
    # on this task without settling. It would not settle because it was not the
    # term on the other end of the rope. Measured on `08-23-12/model_47350` at the
    # cadence ceiling, 4758 swings aligned on touchdown:
    #
    #     steps before touchdown   T-4    T-3    T-2    T-1     T
    #     foot height (mm)         22.0   21.4   18.4   12.9   6.6
    #     vz (mm/s)                +113    -61   -299   -544  -633
    #
    # The last three steps average **1.94 g downward**: the foot is driven into
    # the ground at nearly twice what gravity would do. A free fall from that
    # 22 mm peak still arrives at 0.55 m/s, 13% softer -- so the impact is not a
    # consequence of a short descent, it is a consequence of nothing asking the
    # leg to brake. It has 2 g of authority to brake with and no reason to.
    #
    # Force is the wrong handle on that: it is an output of the contact solver,
    # resolved inside one control step, so its gradient with respect to the action
    # is short and noisy. Approach velocity is a continuous function of the joint
    # trajectory the policy is choosing. The force is still reported --
    # `Episode_Metrics/landing_force_max` below -- as a diagnostic rather than a
    # lever, which is the arrangement this replaces the term to get.
    #
    # `mdp/rewards.py::soft_touchdown` has how -4.5 was arrived at, which is
    # mostly a record of two ways of choosing it that did not work: sizing it
    # against a replay, and sizing it against the share of the reward budget it
    # ought to have. It also has the reason the velocity is read one step early
    # rather than at contact, and the number that would falsify this weight --
    # the raw quantity, not the term's own curve.
    #
    # The failure to watch for is the one every landing penalty has, and the one
    # `soft_landing` was measured producing: a robot that stops landing hard by
    # not putting its feet down. `Metrics/peak_height_mean` recovering towards
    # 0.025 while the impact falls is soft landing; both falling in lockstep is
    # skimming, and the answer then is a lighter weight or a heavier
    # `foot_swing_height`, not more of this.
    del cfg.rewards["soft_landing"]
    cfg.rewards["soft_touchdown"] = RewardTermCfg(
        func=soft_touchdown,
        weight=-4.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": foot_cfg,
        },
    )

    # **One-sided: only a foot that is too low while translating is charged.**
    # `mdp/rewards.py::foot_clearance_shortfall` has the argument; the short form
    # is that clearance is a floor rather than a set point, and the cost of
    # lifting higher than necessary is already priced three other ways.
    # ── The feet are not held while parked, and that is a choice ──────────
    # A twist command means the feet stay where they are in the world and the
    # body turns. `track_twist` scores the body against the footprint -- the right
    # frame, and `motion-planner`'s heading frame -- and **cannot tell which of
    # the two rotated**. Measured on model_38000, commanded 10 degrees while
    # parked: it scored 10.05 against a base heading change of 0.00.
    #
    # `standing_foot_anchor` used to charge the other half, each foot's world
    # displacement past a 10 mm deadzone, and it worked: at 1500 iterations
    # against a 47350-iteration policy without it, the body's share of a 15 degree
    # twist went from 48% to 88% and the footprint's drift from 8.21 to 1.55
    # degrees. It was removed on request.
    #
    # **So nothing scores which end turns.** What the policy does with that is
    # known rather than hypothetical -- it shears the front pair apart, 66 and
    # 77 mm at a 15 degree command where a rigid rotation of the whole footprint
    # would be 5.1, dragging rather than stepping. `mdp/commands.py` keeps the
    # anchor and reports `Metrics/posture/foot_displacement` and
    # `footprint_rotation` so the behaviour is visible when it returns; neither is
    # paid for. They are the numbers to read before concluding a twist is tracked.

    # ── One metric, no weight ─────────────────────────────────────────────
    # `soft_landing` logs the *mean* landing impact, and a mean is the wrong
    # statistic for an impact: measured over a replay, 23 landings averaged 6.3 N
    # with a peak of 21.2. `reduce="max"` makes this the episode peak, which is
    # what a leg actually has to survive.
    #
    # It appears as **`Episode_Metrics/landing_force_max`**, not under `Metrics/`
    # where its mean counterpart is: that one is pushed into the log by the
    # `soft_landing` reward every step, this one goes through the metrics manager
    # and is reported at episode end. `mdp/metrics.py` has the rest.
    cfg.metrics["landing_force_max"] = MetricsTermCfg(
        func=landing_force_max,
        reduce="max",
        params={"sensor_name": "feet_ground_contact"},
    )

    cfg.rewards["foot_clearance"].func = foot_clearance_shortfall
    cfg.rewards["foot_clearance"].weight = -4.0
    # **And only over the first 60% of the swing.** This term's cost is the time a
    # foot spends low while translating, so the cheapest way to pay it is to cross
    # the low band fast -- it was, by construction, paying for a hard landing.
    # Measured on `08-23-12/model_47350` over 4758 swings aligned on touchdown, it
    # charged 0.0185 over the descent against 0.0079 over the rise in the same
    # eight-step window, with its single largest sample **at touchdown itself**.
    #
    # 0.6 is where the lift peaks: the peak sits four steps before touchdown of a
    # 11.4-step mean swing, so about 65% of the way through. Tolling the rise and
    # the transit and freeing the approach is what the term was wanted for --
    # `jumper.posture` runs on flat ground, and a foot 13 mm up and 10 ms from its
    # foothold is not what trips a robot.
    #
    # `mdp/rewards.py::_swing_gate` has why the gate reads the clock rather than
    # the foot's own descent, which is the spelling this was nearly written in and
    # would have paid for the exemption in the currency being overcharged.
    cfg.rewards["foot_clearance"].params["swing_gate"] = 0.6
    cfg.rewards["foot_swing_height"].weight = -3.0

    # **Five times mjlab's -0.1, and the -0.1 was never this task's choice.** It
    # is the upstream velocity-task default; the skeleton only retargets the term
    # at this robot's six sites and leaves the weight alone, so four tasks and
    # this one have been carrying a number nobody picked -- the pattern CLAUDE.md
    # records `jumper.tetrapod` losing months to. Writing it here is the point even
    # at an unchanged value; changing it is the occasion.
    #
    # What the size means, from the measurement in `torque_headroom` above: at
    # -0.1 this term was costing about -0.02 a step, next to `body_tilt_rate`
    # (-0.03) and an order below `foot_clearance` (-0.11). At -0.5 it lands at
    # roughly -0.10, i.e. alongside `foot_clearance` rather than beneath it --
    # a term that bites rather than one that is merely present. That is the
    # intent, and it is also the risk: a slip penalty this size is a reason not
    # to put the foot down, and the failure it would produce is a shorter stance
    # with a longer swing, which is not obviously wrong from the reward curves.
    #
    # So the number to watch is not `Episode_Reward/foot_slip` going down -- it
    # will. It is `Metrics/peak_height_mean` and the standing foot count: if the
    # feet stop slipping by spending less time on the ground, this made the gait
    # worse while looking like it worked.
    #
    # **Two things about this note have moved since it was written.** The -0.02 it
    # calibrates against was measured at 200 Hz control and still applies: this
    # term is `|v_xy|^2`, a physical quantity, so its per-step value does not
    # scale with the rate the way `action_rate_l2`'s does. And the
    # `foot_clearance (-0.11)` it compares itself to was the symmetric form at a
    # 25 mm target; that term is now one-sided at 15 mm and measures smaller, so
    # this one sits closer to it than the text says.
    #
    # **It has no standing counterpart, so below the 0.05 threshold nothing
    # charges a foot for moving at all.** It once had two, in succession:
    # `standing_foot_slip`, this quantity with the opposite gate, and then
    # `standing_foot_anchor`, which replaced it because a parked robot delivers a
    # twist by *stepping* rather than sliding -- 9.3 of 19.6 mm of foot path
    # walked in the air, where a contact-gated term charges nothing. The second
    # was removed on request.
    #
    # So `foot_slip` is a walking term and the parked robot is unpoliced. The
    # note above `track_twist` in the rewards module has what that costs.
    cfg.rewards["foot_slip"].weight = -0.5
    # **0.3, against `jumper.tripod`'s 0.2**, and the argument tripod makes for
    # its number is the argument for changing it here. That note says 0.2 was
    # "the most that could be justified for a task whose name is a gait" --
    # further, "and it stops being a locomotion task with a standing case". This
    # task's name is not a gait. Holding a commanded posture while parked is the
    # case the whole command exists for, and it is the one a deployment spends
    # most of its time in.
    #
    # What it costs is walking exposure, and the cost is exact: everything gated
    # on the command being non-zero -- both tracking terms, `tripod_gait`,
    # `stance_load`, the two foot terms -- goes from 80% of the time to 70%, a
    # relative loss of 12.5%. With `rel_neutral_standing_envs = 0.2` above,
    # four fifths of the standing time bought is spent on postures the robot has
    # to work for and one fifth on standing at home -- which it was once all
    # spent on the first, and the robot then stood at home badly.
    #
    # It also loosens both curricula slightly, for the reason tripod records: a
    # standing stretch contributes almost nothing to `error_vel_xy`, so the
    # measured error is diluted by the standing share and every promotion bar is
    # effectively that much looser. Tripod measured 0.9x -> 0.8x going from 10%
    # to 20%; this is 0.8x -> 0.7x.
    cfg.commands["twist"].rel_standing_envs = 0.3

    # ── What it sees ──────────────────────────────────────────────────────
    # The command to both, the measurement to the critic only. `mdp/observations.py`
    # has the argument; the short form is that the command arrives down a wire on
    # hardware and the measurement does not exist there.
    #
    # `mirror_kind` is declared in the params rather than added to the table in
    # `common/mdp/symmetry.py`, following `jumper.swing`: the kind is this task's
    # decision and the table is four other tasks' file. The kind itself
    # (`posture4`, negate twist and roll) does live there, because it is the
    # mirror algebra rather than a choice.
    #
    # No history on either. The command is resampled every 3 to 8 s, so its
    # history is the same number copied -- measured at +0.0001 of the fit
    # `HISTORY_TERMS` was chosen by, the smallest contribution of any term. The
    # measurement is a different case and could plausibly use frames, but it goes
    # to the critic beside `joint_pos`, which already has five.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["posture_command"] = ObservationTermCfg(
            func=posture_command,
            history_length=0,
            params={
                "command_name": POSTURE_COMMAND,
                # In the params rather than read off the command term, so that it
                # reaches `layout.json`; see the note in `mdp/observations.py`.
                "neutral_height": STAND_Z,
                "mirror_kind": "posture4",
            },
        )
    cfg.observations["critic"].terms["posture_state"] = ObservationTermCfg(
        func=posture_state_obs,
        history_length=0,
        params={
            "asset_cfg": foot_cfg,
            "neutral_height": STAND_Z,
            "mirror_kind": "posture4",
        },
    )

    # ── The posture climbs its own ladder, in training only ───────────────
    # `mdp/curriculum.py` carries the shape, the measured promotion bars and the
    # reason the ruler has to scale with the range.
    #
    # **`not play` is load-bearing and was missing.** `velocity_env_cfg` ends
    # with `cfg.curriculum = {}` under `play`, which is what drops the velocity
    # and terrain ladders for a replay -- and this term is added *after* that
    # call, so it survived the clearing and went on running in replay. What that
    # looked like from the outside: the curriculum's first call wrote level 0's
    # ranges into the command term, the teleop clamps the operator to
    # `cfg.ranges`, and the keyboard would not go past 7.5 degrees. Nothing
    # raised; the robot simply refused to lean any further, which reads as the
    # policy's limit rather than as a config that had narrowed the command
    # underneath it.
    #
    # A comment three lines below this one used to assert the opposite -- "note
    # `cfg.curriculum = {}` in `play`, so the ranges here are the config's" --
    # which is exactly the belief that made the bug invisible while writing it.
    # `tests/test_posture.py` now pins it.
    #
    # **Two curricula now run side by side**, this one and the velocity ladder,
    # and they are independent on purpose: they gate on different errors and
    # promote on their own dwell clocks. What that costs is the joint corner --
    # nothing coordinates "widest speed" with "widest lean", so a run can arrive
    # at the hardest combination of the two without ever having been asked for it
    # gradually. What coupling them would cost is worse: one gate, and a policy
    # that stalls on either axis stops climbing on both, with the log unable to
    # say which.
    #
    # They are logged as `Curriculum/posture/*` against `Curriculum/command/*`,
    # and the key names line up (`level`, `settled`, `promoted`, `*_err` against
    # `*_err_bar`) so the three curricula in this task read as one dashboard.
    if not play:
        cfg.curriculum[POSTURE_COMMAND] = CurriculumTermCfg(
            func=PostureRangeCurriculum,
            params={
                "command_name": POSTURE_COMMAND,
                "angles": POSTURE_ANGLES,
                "heights": POSTURE_HEIGHTS,
                "stand_angles": POSTURE_STAND_ANGLES,
                "std_ratio": POSTURE_STD_RATIO,
                # 1.0 on every rung but the last, the precision rung, which marks
                # the full range by half the std. `POSTURE_STD_SCALES` has the
                # measurement.
                "std_scales": POSTURE_STD_SCALES,
                "rewards": ("track_twist", "track_tilt", "track_height"),
                # Written here even though the class defaults to it, so that both
                # ladders' floors are visible in one place and neither can drift
                # from the other without the diff showing it.
                "warmup_steps": WARMUP_STEPS,
            },
        )

    # ── And in replay, the posture is the operator's too ──────────────────
    # `velocity_env_cfg` swaps the *velocity* command for its teleop subclass
    # when `play` is set; this is the same swap for the other command, and it has
    # to be here because the skeleton knows nothing about it.
    #
    # A drop-in subclass built from the sampling term's own fields, so every
    # range, the neutral posture and the resampling clock carry over untouched --
    # and until a key is pressed it *is* the sampling term, which is what keeps
    # `play.py --headless` and the scripted audits behaving as before.
    #
    # The operator is clamped to `cfg.ranges`, which in replay is the config's
    # own -- the ladder's top rung, which is what training ends on. That holds
    # only because the block above does not install the curriculum here; it is
    # the curriculum writing a narrower range in that produced the 7.5 degree
    # ceiling on the keyboard. See the note there.
    #
    # Both halves read the same file and share one operator (`mdp/operator.py`),
    # which refuses to run if a command the file describes has no term here --
    # so leaving this block out would stop `play` rather than leave the right
    # stick's pitch and the triggers bound to nothing.
    if play:
        from dataclasses import fields as _dc_fields

        posture = cfg.commands[POSTURE_COMMAND]
        cfg.commands[POSTURE_COMMAND] = TeleopPostureCommandCfg(
            **{f.name: getattr(posture, f.name) for f in _dc_fields(posture)},
            controls=load_controls(CONTROLS),
            term=POSTURE_COMMAND,
        )

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg
