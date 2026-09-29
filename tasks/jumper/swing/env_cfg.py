"""Environment config for jumper.swing.

This starts from `velocity_env_cfg` and takes the velocity out, rather than
building a config from mjlab's skeleton directly. The shared function is about 250
lines and almost none of it is about tracking a twist: the solver budget a hexapod
needs, the 20 of 22 actuators the policy drives, the action scale, the contact
sensors, the hull collision, the domain randomisation and the observation groups
are all "this robot in this simulator", and a task that rebuilt them would be a
copy that stops matching the robot the first time the robot changes.

What comes out is therefore a subtraction, and the list below is the whole of it.

## Nothing that reads the twist command survives

The command is removed, so every term that gates on it has to go with it or raise:

    commands      twist                       the command itself
    observations  command                     reads generated_commands
    rewards       track_linear_velocity       tracking, both axes
                  track_angular_velocity
                  pose                        variable_posture switches std by
                                              command magnitude
                  air_time, foot_clearance,   all gated on |twist| > threshold
                  foot_swing_height,
                  foot_slip, soft_landing,
                  standing_sway
    curriculum    terrain, command            both take command_name="twist"

**`foot_slip` and `soft_landing` are losses, not tidying.** A foot sliding on a
plank and a foot slammed onto one are both worth penalising here -- more than on
the ground, since the plank is what the robot is standing on and there is 0.05 m
of it outside each foot. They are gated on a command that no longer exists, so
they would need rewriting against this task rather than deleting; that is the
first thing to add.

## Two terms would have been actively wrong, not merely absent

`upright` and `fell_over` both measure the robot against **gravity**, and the deck
tilts with the arc -- a swing at 20 degrees carries the robot 20 degrees off
vertical while it stands perfectly. Left in, `upright` pays the policy to lean out
of the deck exactly when the swing is doing most, and `fell_over` spends half of
its 50 degree budget on the task succeeding. Both are replaced by deck-relative
versions in `mdp/`. This is the failure this task most easily could have shipped
with: both terms are named for what you want, both produce a number every step,
and nothing anywhere would have said the number was about the wrong floor.

## The swing is built here, not selected with --scene

`--scene` is optional by construction, so a task that needed `--scene swing` would
train on an empty field the moment someone forgot it. The prop and its placement
event are therefore installed below, exactly as `scenes/registry.py` would have.
The cost is that `--scene swing` on top of this task **raises** -- two swings, one
name. Any look-only scene is fine.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from scenes.swing import HANG_L_RANGE

from ..common.constants import GAIT_JOINTS, STAND_Z
from ..common.mdp.curriculum import STD_ANG_RATIO, STD_LIN_RATIO
from ..common.tof import tof_sensor
from ..common.velocity_env import _ARM_JOINTS, _LEG_JOINTS, velocity_env_cfg
from .mdp import events as swing_events
from .mdp import observations as swing_obs
from .mdp import rewards as swing_rewards
from .mdp import state as swing_state
from .mdp import terminations as swing_terms

#: Terms that only make sense with a twist command to gate on. Removed as a named
#: list rather than one `pop` per line so that the docstring above and the code
#: cannot disagree about what was taken out.
_VELOCITY_REWARDS = (
    "track_linear_velocity",
    "track_angular_velocity",
    "pose",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
    "standing_sway",
)

#: The amplitude the reward peaks at, in radians. 45 degrees.
#:
#: **Measured against what a rider can actually do, not chosen for ambition.** A
#: servoed 3 kg point mass on this deck, leaning fore and aft in time with the
#: swing, settles at a steady amplitude that depends only on how far it travels:
#:
#:     lean +/-20 mm -> 12.3 deg      lean +/-40 mm -> 17.7 deg
#:     lean +/-60 mm -> 24.2 deg      squat only, any amplitude -> it dies
#:
#: The robot's footprint is 0.344 m on a 0.444 m plank, so it has about 50 mm of
#: fore-aft travel without moving a foot, and it can step.
#:
#: **Raised from 20 degrees, because 20 was being collected without doing
#: anything.** Measured on the running task: the reward term read 9.4 degrees of
#: amplitude with the policy outputting nothing at all and 11.0 with a sigma of
#: 0.8 of action noise -- against a reset that draws uniformly from +/-20, so the
#: mean gift is 10. At a 20 degree target that is 55% of the objective arriving
#: from the reset draw, and the largest term in the reward (weight 4) was very
#: nearly a constant. A constant contributes no policy gradient, which is what left
#: `action_rate_l2` as the only thing opposing the entropy bonus and let sigma
#: settle at 0.8.
#:
#: **Then raised again from 35, because 35 was left over from a measurement that
#: had since been withdrawn.** The table above was taken with the rider driven open
#: loop at the swing's small-amplitude period; that detunes as the amplitude grows,
#: so what it recorded was the cost of pumping at a fixed slightly-wrong frequency
#: and it reads exactly like a strength limit. 35 was picked against it -- half as
#: far again as the best row. When the rig was re-run with the rider keeping the
#: *phase* instead, every length in `ROPE_LENGTH_RANGE` went past 85 degrees on
#: +/-20 mm of travel, and the ceiling those numbers implied stopped existing. The
#: target did not move with them at the time; this is that following through.
#:
#: 45 sits far above anything the reset or the noise supplies, so the objective is
#: something the policy has to earn.
#:
#: **What does get harder at 45 is the ride, and it is worth watching rather than
#: assuming.** The rider's apparent gravity is `g (1 + 2(1 - cos A))` at the bottom
#: of the arc and `g cos A` at the ends, so 35 degrees is 1.36 g down to 0.82 g and
#: 45 is **1.59 g down to 0.71 g**. The legs have to hold half again their own
#: weight through every pass, and the swing goes light enough at the ends that a
#: foot can be unloaded when the robot least wants it. If the run learns to pump and
#: then starts shedding episodes to `tipped_on_the_deck` at high amplitude, this is
#: where to look first -- not at the objective.
#:
#: **45 is not a ceiling either**, and that is why the reward has to fall off above
#: it (`over_std` on the term) rather than plateau: what stops a competent policy
#: here is the objective peaking, not the robot running out of lean.
#:
#: **Squatting is not the mechanism here** and it is worth knowing before reading a
#: training curve. Parametric pumping needs to change the pendulum's length by
#: enough to beat the loss, and this rider is only 68% of the swinging mass, so
#: +/-60 mm of squat moves the centre of mass 2.5% of a 1.35 m rope against a 5.2%
#: per-swing loss. Measured open loop, squatting alone does not merely fail to
#: pump, it damps: 0.001 m of amplitude left after 60 s against 0.005 m for doing
#: nothing at all. It does help *once there is amplitude to work with* -- 40 mm of
#: squat on top of 40 mm of lean is worth 2.7 degrees.
TARGET_ANGLE = math.radians(45.0)

#: The release angles a fresh episode is drawn from, in degrees: **uniform over
#: this range**, per environment, redrawn every reset. The sign is drawn separately
#: and either way, so these are magnitudes and the arc covered is `+/-45`.
#:
#: **The top is `TARGET_ANGLE` and moves with it.** A range left behind when the
#: target moves changes how much of the objective arrives free, which is a failure
#: this file already has a record of, two paragraphs down.
#:
#: **This replaced a three-point table -- 45 / 20 / 0, a third of the batch each --
#: and the reason to prefer a continuum is the critic.** The value function it is
#: fitting is defined over every amplitude between rest and the target; a reset that
#: only ever visits three of them gives it three points to interpolate a curve
#: through, and the states in between are reached only as an episode decays or
#: builds its way past them. Uniform, every amplitude in the range is a state the
#: critic has seen as a *start*, with a full episode of return behind it. It also
#: removes a cliff that was an artefact of the table rather than of the task: under
#: RSI a third of the batch faced the hardest possible start and the rest faced one
#: of two easy ones, with nothing graded in between.
#:
#: **What it costs is the size of that hard third.** Building a swing from near
#: rest is the half of the skill that has to be learnt -- sustaining one is the
#: easier half -- and a uniform draw puts only about 11% of episodes below 5
#: degrees where RSI put 33% at exactly zero. If the run learns to sustain and
#: never learns to build, that is the number to look at, and the fix is the
#: curriculum at the end of this comment rather than a return to the table.
#:
#: The draw does not corrupt the advantage, and the reason is the same one that
#: made RSI safe: the critic can see the swing -- `swing_offset` and
#: `swing_velocity` are in its group -- so "this episode was handed 40 degrees" is
#: absorbed by the value baseline rather than showing up as reward the policy is
#: credited with.
#:
#: Measured, releasing from high up is no harder to survive than from rest: at
#: zero actions every release angle keeps environments alive at the same rate, and
#: at sigma 0.8 all of them sit at 98% of steps. The deck hangs perpendicular to
#: the apparent gravity, so a steeper swing does not push the robot sideways.
#:
#: **A uniform reset failed here once, and this is not that.** It was `+/-20`
#: against a 20 degree target, and it gave away more than it was worth: a mean
#: magnitude of 10 against a term that *saturated* at 20 meant the largest reward
#: in the task was very nearly a constant, and a constant contributes no policy
#: gradient. What has changed is not the shape of the draw but where the objective
#: sits relative to it. `swing_amplitude` now rises linearly across the whole range
#: and peaks at the top of it, so a draw of 22 degrees is worth 0.5 of the term
#: rather than 1.0, and the reward a fresh episode starts on spans the full 0 to 1
#: instead of clustering at the ceiling. The failure was saturation, not uniformity.
#:
#: And a dead start is not the dead
#: signal it was once assumed to be: from rest, with no policy at all, action noise
#: alone sustains a real swing over a 20 s episode --
#:
#:     action sigma   mean amplitude at 2 s / 5 s / 10 s / 20 s   peak reached
#:     0.0             1.5   2.6   2.3   2.2 deg                   9.0 deg
#:     0.3             2.7   3.7   3.1   2.9                      9.8
#:     0.8             4.7   4.3   4.5   3.4                     11.6
#:     1.2             4.2   4.4   4.0   4.6                     12.7
#:
#: -- which against a 45 degree target is 5-10% of the objective, and it **varies
#: with what the actions do**: twice the amplitude at sigma 0.8 as at 0. That is a
#: gradient, and it points the right way. The peak column matters as much: single
#: episodes reach 9-15 degrees, so the buffer contains examples of a swing that got
#: going for the critic to learn a value from.
#:
#: Even at sigma 0 the swing finds 2.2 degrees. That is the robot's weight arriving
#: on the plank plus the startup randomisation -- `encoder_bias`, `base_com`,
#: `foot_friction` -- making each robot slightly asymmetric, so a completely
#: passive one still rocks a little.
#:
#: **If it stalls, the fix is a curriculum, not a constant.** Start the range wide
#: so the policy learns to *sustain* -- which is the easier half -- and narrow the
#: **top** of it towards zero as it succeeds, so more and more of the batch has to
#: *build*. That is the idiom the four velocity tasks use for their command ranges,
#: and now that this is a range rather than a table it is the same shape of object
#: their `CommandRangeCurriculum` already moves. `cfg.curriculum` is empty here and
#: waiting.
RESET_ANGLE_RANGE_DEG = (0.0, 45.0)

#: The rope lengths an episode is drawn from, in metres, beam to the plank's top
#: face. Uniform, per environment, redrawn every reset.
#:
#: **This is what makes the policy a swinger rather than a swinger-on-this-swing.**
#: A single length lets a policy learn the one drive frequency that works and
#: nothing about swinging; across this range the period runs 1.50 to 2.65 s, a
#: factor of 1.77, so a fixed rhythm is wrong nearly everywhere and the only thing
#: that generalises is closing the loop on what the swing is doing.
#:
#: It can, from proprioception alone, which is the reason this costs no new
#: observation: the deck hangs perpendicular to the rope, so `projected_gravity`
#: carries the angle and the gyro carries the rate, and the drive that pumps is
#: `lean towards the sign of the rate`. That law has **no period in it** -- it is
#: the same controller on every swing here, and it is the one the open-loop rig
#: below is a caricature of.
#:
#: The range is the whole of what this frame can be rigged at
#: (`scenes/swing.py::HANG_L_RANGE`): at 0.60 m the robot on the deck has 0.40 m of
#: headroom under the beam, and at 1.80 m the deck is 0.15 m off the ground. The
#: deck therefore sits anywhere from 1.35 m to 0.15 m up, so these are visibly
#: different swings and not only numerically different ones.
#:
#: **It was 1.00 m for one round, for a reason that no longer exists.** The
#: rigging then put a knot at a fixed 0.25 m above the deck, so a short rope left a
#: stub above the knot and the rig read as a triangle rather than as a rope holding
#: a plank. Four ropes straight to the beam have no fixed-height fitting to run out
#: of, and the short end is limited by headroom instead -- which is 0.40 m away
#: rather than 0.
#:
#: **What is *not* different is how hard 45 degrees is**, and that took measuring
#: to establish, because the obvious argument says otherwise: a rider's static
#: authority over the pendulum is roughly `mu a / L`, which over this range varies
#: by a factor of 3.8 (`scenes/swing.py` has the column). A target that a short
#: swing reaches trivially and a long one cannot reach at all would make the
#: objective mostly a report of which swing the reset handed out.
#:
#: It does not, because authority is not the binding constraint. A rider that
#: keeps the phase drives **every length here past 85 degrees on +/-20 mm of
#: travel** -- a fifth of what the robot has. What limits the amplitude is timing,
#: not strength, and timing is exactly as hard on a 0.7 m rope as on a 1.8 m one.
#: So `TARGET_ANGLE` stays one number and goes on meaning the same thing.
#:
#: The measurements it replaced said the opposite and were wrong in a way worth
#: recording: the rider was driven open loop at the small-amplitude period, which
#: detunes as the amplitude grows, so what was measured was the cost of pumping at
#: a fixed slightly-wrong frequency. That reads exactly like a strength limit.
ROPE_LENGTH_RANGE = HANG_L_RANGE

#: The deck speed through the bottom of the arc that `deck_speed_through_the_bottom`
#: saturates at, in m/s. 1.87 -- what `TARGET_ANGLE` is worth on the **shortest**
#: rope in the range.
#:
#: **Which rope this is taken from is the whole decision, and there is only one
#: answer that does not put two reward terms in opposition.** Speed and amplitude
#: are the same measurement -- `v = sqrt(2 g L (1 - cos A))` -- so this term is the
#: objective again in a length-aware currency, and the constant decides at what
#: angle each rope stops being paid more for going higher:
#:
#:     rope   v at 45 deg   saturates at
#:     0.60      1.87 m/s      45.0 deg
#:     0.90      2.29          36.5
#:     1.35      2.80          29.7
#:     1.80      3.22          25.7
#:
#: Take it from the shortest rope and every swing in the range can reach full marks
#: **without being asked past `TARGET_ANGLE`**, which is required: `swing_amplitude`
#: peaks at 45 and falls off above it, so a term still paying for more speed up
#: there would be pulling against the reward in exactly the band where the reward
#: is supposed to be turning the policy back. Take it from the longest rope instead
#: -- the other obvious choice -- and a 0.6 m swing tops out at 0.58 of the term
#: however well it is ridden, which is a tax on a rope for being short.
#:
#: What it costs: on a 1.8 m rope this term is flat from 26 degrees up, so over the
#: last 19 degrees to the target the amplitude term is on its own. That is the
#: intended division rather than a leak. The long ropes are where `gamma` bites
#: hardest (`rl_cfg.py` has the arithmetic: 133 control steps to a period against
#: 75), so dense credit is worth most early there and worth least at the top, where
#: the 4.0-weight objective is already steep.
#:
#: On the shortest rope it is very nearly a duplicate of the amplitude ramp --
#: 0.228 against 0.222 at 10 degrees -- because `sin(A/2) / sin(22.5 deg)` is
#: linear for small `A`. The two terms separate only as the rope gets longer, which
#: is the only place the second currency was buying anything.
TARGET_SPEED = swing_state.bottom_speed_for(TARGET_ANGLE, min(ROPE_LENGTH_RANGE))

#: How many frames of the IMU the policy sees. 20, which is 0.4 s.
#:
#: **The two IMU terms are this task's signal and the shared table gives them no
#: history at all.** `velocity_env.py::HISTORY_TERMS` grants history to the
#: per-joint and per-foot terms and withholds it from `projected_gravity` and
#: `base_ang_vel`, on a measurement: their history contributes +0.0020 and +0.0016
#: to the fit that table was chosen by. That measurement is right and it was taken
#: **on a walking task**, where the IMU reports the body's bob. Here the deck hangs
#: perpendicular to the rope, so those same two terms *are* the swing, and a task
#: that inherited the table would run with one frame of the only thing it is about.
#: This is the failure `CLAUDE.md` names -- a number four tasks inherit without
#: choosing -- arriving on the fifth.
#:
#: Measured, as the held-out R^2 of a small network reading `k` frames of the
#: actor's own noisy observation. A policy early in training, action sigma 0.4,
#: 900 steps x 64 environments across the whole length range:
#:
#:     frames       1      3      5     10     20     40
#:     rope length  0.00   0.01   0.00  -0.01  -0.09  -0.07
#:     swing rate   0.50   0.68   0.78   0.88   0.95   0.92
#:     swing angle  0.81   0.84   0.84   0.85   0.89   0.95
#:
#: Two things in that table, and they point opposite ways.
#:
#: **The rope length is not recoverable at any history length**, so there is no
#: system identification to be had and a recurrent policy would have nothing to
#: remember. Even handed the exact angle and rate, 40 frames only reaches 0.54.
#: That is fine, because the drive that pumps -- lean towards the sign of the
#: rope's angular rate -- has no period in it; the policy never needs to know which
#: swing it is on, only what this one is doing now.
#:
#: **The rate it does need is half-buried in one frame**, 0.50, and 20 frames take
#: it to 0.95. That is not identification, it is filtering: the gyro carries
#: +/-0.35 rad/s of noise and the robot's own motion on top, and the sign of the
#: rate is what the drive switches on -- precisely at the bottom of the arc, where
#: the rate passes through zero and the signal is smallest. One frame there is a
#: coin toss.
#:
#: 40 frames is worse than 20, which is the other half of the choice: 0.4 s is
#: under a quarter of the shortest period here and a window much longer than that
#: starts averaging away the change it is meant to report.
#:
#: The cost is 114 numbers on a 406-wide observation. Measured at sigma 0.4, so a
#: converged policy -- moving smoothly, polluting its own IMU less -- will need
#: less than this; it is sized for the part of training that has to get off the
#: ground.
IMU_HISTORY = 20

#: The environment key that overrides the rope length, in metres.
#:
#: `MJRL_SWING_LENGTH=1.8` hangs every environment's seat at 1.8 m;
#: `MJRL_SWING_LENGTH=0.9,1.8` restores a range. Empty or unset draws from
#: `ROPE_LENGTH_RANGE`, **in replay as well as in training** -- unlike the release
#: angle, which replay pins at rest. There is no neutral length the way a dead stop
#: is a neutral angle, and a replay that silently showed one swing would be the
#: least informative possible view of a policy whose whole claim is that it does
#: not need one.
SWING_LENGTH_ENV = "MJRL_SWING_LENGTH"

#: The environment key that overrides the release angle, in degrees.
#:
#: `MJRL_SWING_ANGLE=20` releases every episode from 20 degrees;
#: `MJRL_SWING_ANGLE=0,45` restores the training range. Empty or unset leaves the
#: defaults -- the range when training, a dead stop when replaying.
#:
#: **Read by the task, not by the parser.** `scripts/_cli.py` holds one parser
#: shared by all three entry points, and a `--swing-angle` flag on it would be a
#: task's vocabulary in a file that is not allowed any -- the layering
#: `tests/test_log_layout.py` pins, and the one `play.py` was quietly breaking to
#: print a commanded speed. `mjrl.dotenv.load_dotenv` has already filled
#: `os.environ` from `.env` and `.env.local` by the time `env_cfg` is called, so
#: reading it here gets the project default, the personal override and the shell,
#: in that order, for free.
SWING_ANGLE_ENV = "MJRL_SWING_ANGLE"


def _release_range(play: bool, given: str | None = None) -> tuple[float, float]:
    """The release angles this run should draw from, in degrees.

    One value releases every episode from it; two give a range drawn from
    uniformly. The same grammar as `--swing-length` below, deliberately: the two
    flags sit next to each other in `.env` and a reader who has learnt one should
    not have to discover that the other takes a list.

    Raises on anything unparseable rather than falling back. A setting that is
    silently ignored reads as a feature that does not work, which is the rule
    `.env` states for every other key in it.
    """
    import os

    raw = (given or os.environ.get(SWING_ANGLE_ENV, "")).strip()
    if not raw:
        return (0.0, 0.0) if play else RESET_ANGLE_RANGE_DEG
    try:
        parts = tuple(float(piece) for piece in raw.split(",") if piece.strip())
    except ValueError:
        raise ValueError(
            f"--swing-angle / {SWING_ANGLE_ENV} = {raw!r} is not one angle or two, "
            f"in degrees. Write '20' to release every episode from 20 degrees, or "
            f"'0,45' to draw from that range."
        ) from None
    if len(parts) == 1:
        parts = (parts[0], parts[0])
    if len(parts) != 2 or parts[0] > parts[1] or parts[0] < 0.0:
        raise ValueError(
            f"--swing-angle / {SWING_ANGLE_ENV} = {raw!r} is not one angle or an "
            f"ascending pair of non-negative magnitudes. The sign is randomised per "
            f"environment, so '0,45' already covers releases either way."
        )
    if parts[1] > 90.0:
        raise ValueError(
            f"--swing-angle / {SWING_ANGLE_ENV} = {raw!r} asks for more than 90 "
            f"degrees. The seat's "
            f"hinges are Euler angles and 90 is where they are singular; the task "
            f"terminates at 70 (`swing_over`) for the same reason."
        )
    return parts


def _length_range(given: str | None = None) -> tuple[float, float]:
    """The rope lengths this run should draw from, in metres.

    One value pins every environment to it; two give a range. Raises on anything
    outside what the frame can be rigged at rather than clamping -- a swing whose
    upper rope came out negative is not a shorter swing, and a seat below the
    ground is not a longer one.
    """
    import os

    raw = (given or os.environ.get(SWING_LENGTH_ENV, "")).strip()
    if not raw:
        return ROPE_LENGTH_RANGE
    try:
        parts = tuple(float(piece) for piece in raw.split(",") if piece.strip())
    except ValueError:
        raise ValueError(
            f"--swing-length / {SWING_LENGTH_ENV} = {raw!r} is not one length or "
            f"two, in metres. Write '1.8' to hang every swing at 1.8 m, or "
            f"'0.9,1.8' to draw from that range."
        ) from None
    if len(parts) == 1:
        parts = (parts[0], parts[0])
    if len(parts) != 2 or parts[0] > parts[1]:
        raise ValueError(
            f"--swing-length / {SWING_LENGTH_ENV} = {raw!r} is not one length or "
            f"an ascending pair"
        )
    low, high = HANG_L_RANGE
    if parts[0] < low or parts[1] > high:
        raise ValueError(
            f"--swing-length / {SWING_LENGTH_ENV} = {raw!r} is outside what this "
            f"frame can be rigged at, {low}-{high} m. Shorter and the robot on the "
            f"deck runs out of headroom under the beam; longer and the deck reaches "
            f"the ground."
        )
    return parts


def env_cfg(
    asset: Path | None = None,
    play: bool = False,
    swing_angle: str | None = None,
    swing_length: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode -- observation noise and external disturbances off,
            longer episodes.
        swing_angle: release angles in degrees, comma separated, from
            `--swing-angle`. None falls back to `MJRL_SWING_ANGLE` and then to the
            mode's own default. The task declares the flag itself; see
            `cli_args` in this task's `__init__.py`.
        swing_length: rope length in metres, or two for a range, from
            `--swing-length`. None falls back to `MJRL_SWING_LENGTH` and then to
            `ROPE_LENGTH_RANGE`.
    """
    from mjlab.envs.mdp import events as mdp_events

    from scenes.swing import BOARD_TOP_Z, SWING_X, swing_entity

    cfg = velocity_env_cfg(
        asset=asset,
        play=play,
        # Nothing to teleoperate: there is no command, and the pad and the keymap
        # both exist to drive one. Left on, `play` would swap the twist term for a
        # gamepad term that this task has just deleted.
        teleop=False,
        gamepad=False,
        # Nothing here reads a twist command -- it is removed below, with every
        # term gated on it -- so there is no operator input for a `controls.yaml`
        # to describe and none is required.
        operator_command=False,
        # The standing-sway penalty is one of the terms removed below, and this is
        # the argument that would otherwise re-weight it after the fact.
        standing_sway_weight=0.0,
        # ── Required, and almost all of it inert here ────────────────────
        #
        # `velocity_env_cfg` requires these rather than defaulting them, on the
        # argument that a parameter belongs to the task and four tasks agreeing on
        # a number should be four decisions rather than one default. That argument
        # is right and it lands awkwardly on this task, because the skeleton is
        # built and *then* subtracted from: every value below except the first
        # configures a term this task deletes a few lines later.
        #
        # They are passed as the shared constants rather than as zeros or
        # placeholders. A zero would read as a decision -- "this task marks the
        # command infinitely strictly" -- about a command that does not exist by
        # the time anything reads it.
        #
        # `action_rate_weight` is the exception and is real: the term survives, and
        # it is the only thing opposing the entropy bonus in this task's reward
        # (measured: it is exactly quadratic in the action sigma, and carries the
        # whole downward force on it).
        action_rate_weight=-0.1,
        pose_weight=1.0,  # replaced below by a fixed-width `pose`
        foot_target_height=0.025,  # the foot terms are removed below
        command_lin_ceiling=0.8,  # the command is removed below
        command_ang_ceiling=1.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )

    # ── The swing ────────────────────────────────────────────────────────
    cfg.scene.entities["swing"] = swing_entity()
    # The same event `scenes/registry._place_props` installs, and for the same
    # reason: mjlab applies `env_origins` inside this function and nowhere else,
    # so without it every swing in the batch sits at the world origin while the
    # robots spread out. The pose range stays **empty** -- the entity's own
    # `init_state.pos` already carries the offset and passing it again doubles it.
    cfg.events["reset_prop_swing"] = EventTermCfg(
        func=mdp_events.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("swing")},
    )

    # ── Take the velocity out ────────────────────────────────────────────
    cfg.commands.pop("twist", None)
    cfg.curriculum = {}
    for name in _VELOCITY_REWARDS:
        cfg.rewards.pop(name, None)
    # **And `upright`, which is not velocity-gated and is the more dangerous of
    # the two.** It measures the body against gravity and would have survived
    # every "does it still build?" check while paying the policy to lean out of a
    # tilting deck. It was still here after the docstring above was written saying
    # it had been removed, which is exactly how this class of bug travels.
    cfg.rewards.pop("upright", None)
    for group in cfg.observations.values():
        group.terms.pop("command", None)

    # **`body_tilt_rate` measures against the world, so it charges for riding the
    # swing.** Replaced rather than re-weighted, which is what was tried first: the
    # world-frame value nearly triples from 0.757 to 2.111 as the swing goes from
    # rest to 35 degrees, on a robot doing nothing at all, so no weight makes it
    # mean the right thing -- it would simply charge less for succeeding. The
    # deck-relative version runs 0.465 to 0.604 over the same range.
    #
    # This is the third term in this task with that fault, after `upright` and
    # `fell_over`. All three are named for something the robot does and measured
    # against a frame the robot is no longer in.
    cfg.rewards.pop("body_tilt_rate", None)
    cfg.rewards["body_tilt_rate_on_the_deck"] = RewardTermCfg(
        func=swing_rewards.body_tilt_rate_on_the_deck,
        # Back to the shared term's own weight: the tenth it was cut to was
        # compensating for the frame being wrong, and the frame is right now.
        weight=-0.05,
        params={"deadband": 0.5},
    )

    # ── The robot starts on the plank ────────────────────────────────────
    robot = cfg.scene.entities["robot"]
    # The **nominal** deck height, because that is the one the model compiles with
    # and this is only where the robot stands before the first reset runs. Every
    # episode thereafter is placed by `reset_on_the_swing`, on the deck of whatever
    # length that episode drew; a value here cannot know the length and does not
    # need to.
    robot.init_state.pos = (
        SWING_X + swing_state.DECK_OFFSET[0], 0.0, BOARD_TOP_Z + STAND_Z
    )
    # Replaces `reset_base` rather than joining it: mjlab's version puts the robot
    # at a fixed offset from the environment origin with half a metre of scatter
    # and a full turn of yaw, which is a robot in mid-air beside a 0.44 m plank.
    cfg.events.pop("reset_base", None)
    cfg.events["reset_on_the_swing"] = EventTermCfg(
        func=swing_events.reset_on_the_swing,
        mode="reset",
        # **Replay always starts at rest, whatever training used.**
        #
        # Handing an episode a swing it did not build is a training device: it is
        # how the critic learns what a large swing is worth before the policy can
        # produce one. Watching a replay, that is the one thing you do not want --
        # a robot dropped onto a 45 degree swing looks like a robot that has learnt
        # to pump, for the first ten seconds, whether or not it has. From rest
        # there is nowhere for the amplitude to come from except the policy.
        params={
            "angle_range": swing_events.radians(_release_range(play, swing_angle)),
            "length_range": _length_range(swing_length),
        },
    )
    # The push happens through the plank the robot is standing on, so a shove that
    # is fine on the ground is a shove off the edge here. Halved rather than
    # removed -- recovering from a disturbance is most of what this task is.
    if "push_robot" in cfg.events:
        ranges = cfg.events["push_robot"].params["velocity_range"]
        for key, (lo, hi) in list(ranges.items()):
            ranges[key] = (lo / 2.0, hi / 2.0)

    # ── What it is paid for ──────────────────────────────────────────────
    # ── The weights, and why they are this big ───────────────────────────
    #
    # **A task whose reward rate is negative pays the policy to end the episode.**
    # There is no alive bonus here and there does not need to be -- `deck_upright`
    # and `centred_on_the_plank` are both near 1 for a robot simply standing on the
    # plank, so together they *are* the alive bonus -- but they have to outweigh
    # the penalties or falling off is the cheapest thing available.
    #
    # Measured, at the first weights tried (2.0 / 1.0 / 0.5): mean episode length
    # 10 steps, 0.2 s, ending 66% `off_the_plank` and 34% `tipped_on_the_deck`,
    # with `action_rate_l2` at -0.031 per episode against +0.011 for the objective.
    # The policy was not failing to learn to swing, it was learning to stop.
    #
    # So standing still and square on a level deck is worth 2.0 + 1.0 = 3.0 before
    # the swing pays anything at all, and a swing at the target adds 5.0 on top --
    # 4.0 for the amplitude and 1.0 for the speed it carries through the bottom.
    cfg.rewards["swing_amplitude"] = RewardTermCfg(
        func=swing_rewards.swing_amplitude,
        weight=4.0,
        # The shoulder above the target is 0.30 rad, 17 degrees: against a 45 degree
        # peak the term reads 92% at 50, 71% at 55, 47% at 60 and 12% at the 70
        # degree termination. The falloff is well under way a full 15 degrees before
        # the episode ends, which is the point -- the policy should be turned back
        # by the reward, not by the termination.
        #
        # **0.30 did not have to move when the target went 35 -> 45, and the reason
        # is what the number was originally chosen for.** It was picked to put the
        # effective termination about 1.4 shoulder-widths above the peak, measured
        # when the knotted rigging made a 70 degree deck tilt an amplitude of 59:
        # (59 - 35) / 17.2 = 1.4. Four ropes to the beam made the deck ride square
        # (measured, slope 1.000) and that stretched the gap to 2.0 widths, which
        # read as the parameter having got *better*. Raising the target spends that
        # back: (70 - 45) / 17.2 = 1.45, the shape it was calibrated as. A wider
        # target and the same shoulder is a coincidence worth stating, because the
        # next time either moves it will not hold.
        params={"target_angle": TARGET_ANGLE, "over_std": 0.30},
    )
    cfg.rewards["deck_speed_at_the_bottom"] = RewardTermCfg(
        func=swing_rewards.deck_speed_through_the_bottom,
        # **A quarter of the objective's weight, because it is a second view of the
        # objective and not a second objective.** `v = sqrt(2 g L (1 - cos A))`, so
        # this and `swing_amplitude` are functions of one another at a fixed rope;
        # weighting them equally would be paying 8.0 for one thing while every
        # regulariser in the reward stays sized against 4.0.
        #
        # What the 1.0 buys is where the two disagree, which is across lengths. At
        # 20 degrees on a 1.8 m rope the amplitude term reads 0.44 and pays 1.78,
        # and this reads 0.78 and pays 0.78 -- a 44% bonus for the energy a long
        # slow swing is carrying at an angle the amplitude term still calls barely
        # started. On a 0.6 m rope the same 20 degrees pays 0.45 here, so the bonus
        # is 25%. The gradient it adds points the same way as the objective
        # everywhere; it is only steeper where the discount is worst.
        weight=1.0,
        params={"target_speed": TARGET_SPEED},
    )
    # **Keyed `pose`, like the four velocity tasks, and re-added after the pop
    # above removed theirs.** The order matters and is not incidental:
    # `_VELOCITY_REWARDS` takes mjlab's `pose` out because `variable_posture`
    # switches its width on the twist command, and this puts a fixed-width one back
    # under the same name. Same key and the same formula -- `exp(-mean(err^2 /
    # std^2))` -- so `Episode_Reward/pose` still means the same thing across the
    # five tasks and can be read side by side. What differs is the width, which is
    # in the params below where it is visible.
    cfg.rewards["pose"] = RewardTermCfg(
        func=swing_rewards.posture,
        # **An eighth of the velocity tasks' weight for their `pose`**, cut from a
        # half. It is a regulariser here and not an objective, and it is the one
        # term that pays the robot for *not* moving -- on a task whose mechanism is
        # moving the body over planted feet.
        #
        # What it costs to cut: a robot standing square and still on a level plank
        # now collects 3.125 rather than 3.5 before the swing pays anything, against
        # an objective worth up to 4. That margin over the penalties is what keeps
        # the episode worth continuing (see the weights block above), so it is the
        # number to watch if episodes start ending early again.
        #
        # What it buys: the width is already the walking width rather than the
        # standing one, because at `std_standing` the term collapses exactly when
        # the robot leans far enough to pump (see `posture`). Cutting the weight as
        # well means a policy that finds a contorted but effective stance is not
        # argued out of it. The term is still there for the flailing arm, which
        # nothing else watches.
        weight=0.125,
        params={
            # **The walking widths, not the standing ones.** See `posture`: at the
            # standing width of 0.05 on a leg joint, shifting the body far enough
            # to pump reads as `exp(-3.24) = 0.04` and the term collapses exactly
            # when the robot succeeds.
            "std": {_ARM_JOINTS: 0.60, _LEG_JOINTS: 0.30},
            "asset_cfg": SceneEntityCfg("robot", joint_names=GAIT_JOINTS),
        },
    )
    cfg.rewards["sideways_sway"] = RewardTermCfg(
        func=swing_rewards.sideways_sway,
        # Squared radians, so 10 degrees of sway costs 0.03 and 30 costs 0.27.
        # **Untuned**: it wants to be big enough that the policy damps a sway out
        # rather than living with it, and small enough that it never competes with
        # the objective. Watch `Episode_Reward/sideways_sway` against
        # `Episode_Reward/swing_amplitude`.
        weight=-2.0,
        params={},
    )
    cfg.rewards["pump_power"] = RewardTermCfg(
        func=swing_rewards.pump_power,
        # Metres times rad/s. A 50 mm lean while the rope sweeps at 1 rad/s is
        # 0.05, so this weight makes a well-timed lean worth about 0.5 -- an
        # eighth of the objective at its ceiling, and available from the first
        # step, which the objective is not.
        #
        # **Untuned, and the one to watch.** Too big and the policy farms the
        # shaping instead of the swing, which is the classic failure of a term
        # that pays for the mechanism; the tell is `Episode_Reward/pump_power`
        # rising while `Episode_Reward/swing_amplitude` does not.
        weight=10.0,
        params={},
    )
    cfg.rewards["deck_upright"] = RewardTermCfg(
        func=swing_rewards.deck_upright,
        weight=2.0,
        # 15 degrees of lean off the deck is **free**, and beyond that the falloff
        # is a Gaussian with a 0.35 rad width -- so 30 degrees keeps 84% and 45
        # keeps 44%, with the termination at 50. The band exists because leaning is
        # how a rider moves its centre of mass; see `deck_upright`.
        params={"deadband": 0.26, "std": 0.35},
    )
    cfg.rewards["centred_on_the_plank"] = RewardTermCfg(
        func=swing_rewards.centred_on_the_plank,
        # **1.0, and it was checked rather than left alone.** The worry is the one
        # `pose` was cut for: a positive term that is near 1 for a robot doing
        # nothing, on a task whose mechanism is moving the body over planted feet.
        # Measured, it does not bite. Averaged over a cycle this term costs
        # **0.009 per step** for the +/-20 mm of lean that drives every rope in the
        # range past 85 degrees, 0.075 for +/-60 mm and 0.189 for +/-100, against an
        # objective worth 5.0 at the target. Even a robot swinging its body the full
        # width of its stance keeps 96% of it.
        #
        # The gradient is also in phase with the mechanism rather than against it.
        # Resonant pumping wants the rider's *displacement* in phase with the seat's
        # velocity, so at the ends of the arc -- where the rate passes through zero
        # -- the rider should be crossing the middle. `pump_power`'s gradient goes to
        # zero exactly there, being proportional to the rate; this one does not, and
        # it points the way the drive wants to go.
        weight=1.0,
        # 0.15 against a plank whose half-length is 0.222 m. **One number for both
        # axes, on a task that is not isotropic**, which is the thing here most
        # worth revisiting: fore-aft is the axis the robot must move on, across is
        # the axis it must not, and 60 mm of lateral drift is charged exactly what
        # 60 mm of pumping lean is. A per-axis pair is the right shape. What is
        # missing is a measurement of how narrow the lateral one should be, and a
        # guessed number in a task config is the failure `CLAUDE.md` is about.
        params={"std": 0.15},
    )
    cfg.rewards["feet_slip_on_the_deck"] = RewardTermCfg(
        func=swing_rewards.feet_slip_on_the_deck,
        # **Untuned.** It is the only term asking the robot to keep its feet where
        # the reset put them, and the scale it needs depends on how much travel the
        # policy turns out to need to pump, which nothing has measured yet. Too
        # high and the robot freezes and cannot lean; too low and it walks around
        # the plank instead of riding it. Read `Episode_Reward/feet_slip_on_the_deck`
        # against `Episode_Reward/swing_amplitude` and move it.
        weight=-0.2,
        params={},
    )
    cfg.rewards["feet_off_the_plank"] = RewardTermCfg(
        func=swing_rewards.feet_off_the_plank,
        # Small: it counts up to six, so a robot mid-scramble can collect -3.0 from
        # this term alone and that is the same trap in miniature.
        weight=-0.2,
        params={},
    )

    # ── When it is over ──────────────────────────────────────────────────
    cfg.terminations.pop("fell_over", None)
    cfg.terminations["tipped_on_the_deck"] = TerminationTermCfg(
        func=swing_terms.tipped_on_the_deck,
        params={"limit_angle": math.radians(50.0)},
    )
    cfg.terminations["swing_over"] = TerminationTermCfg(
        func=swing_terms.swing_over,
        # 70 degrees. The objective peaks at 45 and the deck rides square to the
        # rope (four ropes to the beam, measured slope 1.000), so a legitimate swing
        # at the target puts the deck at 45 -- this is 25 clear of that, and 20
        # short of the Euler singularity at 90. It used to be 35 clear, against a 35
        # degree target; `swing_amplitude`'s shoulder is what has to cover the gap
        # and the note there says how much of it is left.
        params={"limit_angle": math.radians(70.0)},
    )
    cfg.terminations["off_the_plank"] = TerminationTermCfg(
        func=swing_terms.off_the_plank,
        # The base may lean 0.10 m past an edge before this fires, because it can:
        # the feet reach 0.05 m further out than the base does on every side.
        # `drop` catches the robot that is over the plank and below it.
        params={"margin": 0.10, "drop": 0.05},
    )

    # ── What it sees ─────────────────────────────────────────────────────
    #
    # **The swing terms go to the critic only.** The actor runs on proprioception:
    # the IMU, the joints, its own last actions and the actuator forces. Nothing
    # that a robot standing on a real swing could not measure about itself.
    #
    # That is not a handicap, because **the swing is already in the IMU**. The deck
    # hangs perpendicular to the rope and the robot stands square on the deck, so
    # the body is tilted by the swing angle and `projected_gravity`'s x component
    # *is* its sine -- measured, +0.342 at -20.0 degrees, which is sin(20) to three
    # figures, and -0.445 at +20.5. The gyro carries the other half: -1.05 rad/s
    # through the bottom of the arc, near zero at both ends, in the right phase. So
    # position and rate are both there, and `swing_offset` / `swing_velocity` were
    # only ever a more convenient spelling of them.
    #
    # `deck_up` and `deck_offset` are the two that are genuinely privileged -- where
    # the plank is under its feet is not something the robot can feel -- and they
    # are exactly what an asymmetric critic is for.
    #
    # **One sim-to-real caveat, and it is not small.** `projected_gravity` here is
    # derived from the body's orientation, not from an accelerometer. A real IMU on
    # a swing measures *apparent* gravity, which points along the rope, so a naive
    # attitude filter reports the robot as level and hides the swing angle
    # completely. The gyro survives -- a rate gyro really does see the swing -- but
    # a policy trained against this term and deployed against a fused attitude
    # estimate is reading a different signal.
    #
    # A fresh cfg object per group even so: the observation manager resolves term
    # configs in place, so one object shared between groups resolves twice over
    # itself. And `history_length` has to be stated -- the skeleton sets it for
    # every term it knows about before this runs, so anything mounted afterwards
    # silently gets 0.
    # **The IMU gets history, which the shared table denies it.** See
    # `IMU_HISTORY`: those two terms are the only ones carrying the swing to the
    # actor, and one frame of the gyro is a coin toss on the sign of the rate at
    # exactly the moment the drive has to switch. Set after `velocity_env_cfg`
    # returns, which is the only order that works -- it assigns every term's
    # `history_length` from `HISTORY_TERMS` on its way out.
    for group in cfg.observations.values():
        for name in ("projected_gravity", "base_ang_vel"):
            if name in group.terms:
                group.terms[name].history_length = IMU_HISTORY

    for group in ("critic",):
        for name, func in (
            ("swing_offset", swing_obs.swing_offset),
            ("swing_velocity", swing_obs.swing_velocity),
            ("deck_up", swing_obs.deck_up),
            ("deck_offset", swing_obs.deck_offset),
        ):
            cfg.observations[group].terms[name] = ObservationTermCfg(
                func=func,
                history_length=5,
                # Every one of these is a true three-vector in a frame that
                # mirrors with the robot, so the shared table's `vec3` rule --
                # negate y -- is correct and `common/mdp/symmetry.py` needs no
                # entry of its own. Declared here rather than left to the table
                # because the table is four other tasks' file.
                params={"mirror_kind": "vec3"},
            )

    # ── Every environment at the same origin ─────────────────────────────
    #
    # **This is a workaround for a viewer bug, not a physics decision, and it is
    # free only because of what this task is.** Environments are batched worlds,
    # not neighbours in one scene -- measured, kicking one environment's seat moves
    # it 599 mm and moves every other environment's by 0.0000 mm, on both backends.
    # `env_spacing` therefore only decides where the picture puts them, and this
    # task runs on an infinite plane with no terrain tiles to line up.
    #
    # What it buys: `mjrl/viewer/live.py` copies `mocap_pos` for the environment
    # the camera follows, but the ones it draws beside it get `qpos` and `qvel`
    # only (`LiveViewer._draw_other_envs`), on both backends. The swing is
    # fixed-base, so mjlab wraps it in a mocap body and its pose lives nowhere else
    # -- every other swing is drawn at the model's default while its robot rides its
    # real `env_origin`, and what you see is robots hanging in mid-air beside empty
    # swings. Stacking the origins makes `mocap_pos` equal the model default, so the
    # stale value is the right one and the picture is honest again.
    #
    # It is a workaround: the fix belongs in the viewer, and this line should go
    # when it lands. Nothing about the physics changes either way -- the numbers in
    # a headless run are identical.
    #
    # ── The ropes draw slack in the viewer, and that one has no workaround ──
    #
    # **Same root cause, different symptom, and this one has to be read as a
    # rendering artefact because it looks exactly like a modelling mistake.** The
    # viewer draws from a *separately compiled* model (`render_model` on native,
    # `sim.mj_model` on warp), and no per-world model field reaches it -- `mocap_pos`
    # above is one, and `tendon_range` is the other. The ropes' limits there are
    # therefore always the nominal `rope_for(HANG_L)` = 1.362 m, whatever this
    # episode drew.
    #
    # MuJoCo draws a spatial tendon whose length is under its limit as a **hanging
    # slack line**, so on a short rope the viewer shows the rope running to the
    # plank and then looping a long way below it. Measured at 0.60 m: the physics
    # has `ten_length` 0.6268 against a limit of 0.6264 -- 0.4 mm of stretch, all
    # four ropes in tension -- while the viewer is drawing 0.74 m of slack. Copying
    # `tendon_range` into the drawing model makes it straight and changes nothing
    # else.
    #
    # **Left unfixed on purpose, and the alternative was worse.** The fix is a few
    # lines in `mjrl/viewer/live.py` beside the `mocap_pos` one, and `rl/` is held
    # read-only here -- not the vendored argument, which does not apply to `mjrl`
    # (only `rl/mjlab/` and `rl/rsl_rl/` are upstream copies), but the standing
    # convention that the framework is a base and tasks are where work happens.
    #
    # Nothing can be done from *this* side either, and it is worth writing down why
    # rather than leaving it to be rediscovered. Compiling the ropes at the bottom
    # of the range instead of at the nominal would make every episode's rope *over*
    # its drawn limit rather than under it, and MuJoCo draws those straight -- but
    # the seat's rest pose is compiled from `HANG_L` too, so the rig would start
    # 0.76 m over-stretched and snap on the first step. Moving `HANG_L` itself is
    # worse: `BEAM_Z` is derived from it and the whole frame goes with it. Bending
    # the model to flatter a renderer is the wrong trade in any case.
    #
    # **Watching a replay, `--swing-length 1.35` draws straight ropes** -- that is
    # the length the drawing model was compiled at. Any other length is honest
    # physics and a misleading picture, and `play_status` prints the rope length on
    # every line so which one you are looking at is never a guess.
    # ── The episode ──────────────────────────────────────────────────────
    #
    # **30 s rather than mjlab's 20.** The swing's period is 2.35 s, so 20 s is 8.5
    # swings -- and this task starts every episode at a dead stop and asks the
    # policy to build 35 degrees, which is not something 8.5 swings is generous
    # for. 30 s is 12.8.
    #
    # It also puts the episode on the right side of the swing's own decay: the
    # loaded half-life is 24 s (`scenes/swing.py`), so over 20 s a swing left alone
    # keeps 56% of its amplitude and over 30 s it keeps 42%. Coasting was always
    # meant to cost something and now it costs more than half.
    #
    # **Not in `play`.** `velocity_env_cfg` sets `episode_length_s` to 1e9 there so
    # a replay runs continuously, and writing 30 over the top of it -- which this
    # did -- gives a replay that silently resets every thirty seconds. What that
    # looks like from the outside is the robot teleporting back onto the plank and
    # falling off again, which reads as a spawn bug rather than as an episode
    # boundary.
    if not play:
        cfg.episode_length_s = 30.0

    cfg.scene.env_spacing = 0.0

    cfg.viewer.distance = 2.2
    cfg.viewer.elevation = -10.0

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg
