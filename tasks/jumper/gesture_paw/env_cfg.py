"""Environment config for jumper.gesture_paw: offering a paw, `media/paw.npz`.

The environment is `tasks/jumper/common/dance/env.py`; this file holds every
number. Where a number follows a rule `jumper.dance` wrote down for its own clip,
the rule is applied to this clip's measurements and the result is quoted -- read
`tasks/jumper/dance/env_cfg.py` for the reasoning behind each rule.

## This clip, measured

At the 50 Hz control rate, after conversion (6.4 s, lead-in and lead-out included):
per-joint position std 0.191 rad at the median and 0.661 at the most active, RMS
amplitude 0.291 rad; joint velocity RMS 0.566 rad/s, per-joint std 0.317 at the
median and 1.496 at the most active, peaking at 6.9 rad/s; the base between 80 and
107 mm; tilt at most 9.9 degrees.

**The reward weights are starting points**, as they are for jumper.dance: no policy
has been trained on this clip in this repository yet. One of them departs from the
rule the others follow, and says why: `joint_limit_weight`.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.dance.env import dance_env_cfg

#: This task's material, written by `tools/import_wbc_gestures.py`.
MEDIA = Path(__file__).resolve().parent / "media"

#: Episode length, seconds. Half the 6.4 s clip, jumper.dance_maze's rule for a
#: short clip: the command teleports the robot back onto the reference wherever an
#: episode runs off the end, and at half the clip an episode sampled in its first
#: half runs clean.
EPISODE_S = 3.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config. See `jumper.dance`'s for the arguments."""
    return dance_env_cfg(
        media=MEDIA,
        asset=asset,
        play=play,
        episode_s=EPISODE_S,
        # ── Reference-state initialisation ────────────────────────────────
        # jumper.dance's: fractions of this robot (10 mm is a tenth of its standing
        # height), not of the clip, so they carry over unchanged.
        rsi_pose_range={
            "x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05),
        },
        rsi_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
        rsi_joint_range=(-0.05, 0.05),
        # ── Observation noise ─────────────────────────────────────────────
        # The tracking skeleton's, as jumper.dance has them; not measured.
        joint_pos_noise=0.01,
        joint_vel_noise=0.5,
        # ── Rewards ───────────────────────────────────────────────────────
        # Positive budget 6.5 a step, as jumper.dance: mjlab's six tracking terms
        # unchanged (5.0) plus the two joint-space terms below (1.5).
        #
        # Joint position: jumper.dance's rule is a std at 0.44 of the clip's RMS
        # amplitude, which scores a robot standing still through the clip at
        # 0.006. Here: 0.44 x 0.291 -> 0.13, standing still 0.007.
        joint_pos_std=0.13,
        joint_pos_weight=1.0,
        # Joint velocity: two thirds of the clip's velocity RMS, standing still
        # about 0.10. Here: 0.566 -> 0.38, standing still 0.11. Half the position
        # weight, so when they disagree the pose wins.
        joint_vel_std=0.38,
        joint_vel_weight=0.5,
        # Smoothness: -0.1 is calibrated at this action scale (0.25) and the
        # second difference at half of it -- properties of the robot and the
        # action space, not of the clip.
        action_rate_weight=-0.1,
        action_acc_weight=-0.05,
        # Power: 22 joints at ~0.25 N*m and this clip's ~0.57 rad/s is ~3.1 W, so
        # -0.02 is about -0.06 a step, 1.0% of the budget -- jumper.dance's weight,
        # which a gentler clip leaves smaller in proportion.
        power_weight=-0.02,
        # Torque above the continuous rating: anchored to the servo, not the clip
        # -- one joint at the plateau costs 0.6, 9% of the budget.
        torque_headroom_weight=-2.0,
        # Joint limits. **Not the dances' -10, because this reference does not stay
        # inside the soft limits.** rl-wbc-fsm's keyframes lowered the rear as far
        # as it would go: through the 3.9 s sit both rear knees are on their
        # mechanical stop (1.60 rad) and both rear ankles on theirs (-2.40), and the
        # middle ankles are 0.14 rad past their soft limit. The soft limit is 0.9 of
        # the range about its centre, so at the deepest of the sit a faithful policy
        # is 1.11 rad past it summed over the six joints, and at -10 that is 11.1 a
        # step -- 170% of the budget, which no tracking term could outbid: the
        # policy would learn not to sit. At -0.5 the same faithful sit costs 0.55,
        # 8.5%, about what one saturated actuator costs under the torque term above.
        # The price is that the term is twenty times weaker than the dances'
        # everywhere else as well.
        joint_limit_weight=-0.5,
        # Leg-on-leg contact above 1.0 N, ~8% of the robot's worst-case push.
        self_collision_weight=-1.0,
        self_collision_force=1.0,
        # ── Terminations ──────────────────────────────────────────────────
        # Root height: this clip moves the body through 80-107 mm, a 27 mm band,
        # so 40 mm exceeds all of it and cannot fire on a policy merely tracking
        # badly -- jumper.dance's rule, and its number.
        anchor_height_error=0.04,
        # 1 - cos(tilt) = 0.3 is 45.6 degrees, "has fallen over"; the clip leans 9.9.
        anchor_tilt_error=0.3,
        # A support foot 50 mm out vertically is half the standing height.
        support_foot_error=0.05,
        # ── Disturbance ───────────────────────────────────────────────────
        # jumper.dance's: gentler and rarer than mjlab's, sized to a 2 kg robot.
        push_interval_s=(3.0, 8.0),
        push_velocity_range={
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
            "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2),
        },
    )
