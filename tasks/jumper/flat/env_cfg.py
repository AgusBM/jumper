"""Environment config for jumper.flat.

The environment skeleton is shared by the four jumper tasks
(`tasks/jumper/common/velocity_env.py`), and the only difference between this task
and the other three is the gait prior, which this one omits. To tune environment
parameters for this task alone, add arguments to the call below -- nothing else is
affected.

**This task's command range goes further than the other three's**, and the reason
is the same absence: with no gait prior there is no phase clock, so nothing fixes
the cadence. See the block at the end of `env_cfg`.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..common.mdp.curriculum import (
    STD_ANG_RATIO,
    STD_LIN_RATIO,
)
from ..common.tof import tof_sensor
from ..common.velocity_env import velocity_env_cfg

#: The rung this task adds on top of the shared ones, in m/s and rad/s.
#:
#: **Why this task and not the other three.** A fixed-frequency clock makes speed
#: a derived quantity: `v_max = stride * freq / duty`, and with the IK allowing
#: about 0.08 m of stride the three gait tasks top out at 0.50 (tripod, tetrapod)
#: and 0.40 (ripple). Asking any of them for 0.80 would be the failure that cost
#: tripod 37500 iterations -- a command level no gait can reach, which is not an
#: error but a tracking reward that can never be collected. `jumper.flat` has no
#: clock at all, so its cadence is free and that ceiling does not apply.
#:
#: **The angular figure has measured support; the linear one does not.** On a
#: policy trained to 30000 iterations, yaw tracking held 65-70% of the command and
#: did not saturate anywhere tested:
#:
#:     commanded   0.20   0.35   0.50   0.75   1.00  rad/s
#:     achieved    0.138  0.227  0.350  0.486  0.660
#:
#: -- so 1.00 rad/s is reachable, and `ANG_LEVELS` already named it as the obvious
#: next rung. For 0.80 m/s there is no equivalent measurement: nothing has been
#: asked to walk that fast on this robot, and 0.80 is 7.6 body heights per second.
#: If the level proves unreachable the symptom is a curriculum that stops at rung
#: 2 -- read `Curriculum/command/lin_err` against `lin_err_bar`, which is what
#: those two are logged side by side for.
FLAT_TOP_LIN = 0.80
FLAT_TOP_ANG = 1.00


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
        # How high this gait lifts a foot, in metres. Per task on purpose: the
        # skeleton requires it, because it is a choice about the gait and not a
        # fact about the robot. The measurement behind 0.025 is in
        # `jumper.tripod`'s copy; nothing here has been measured separately, and
        # matching tripod is the assumption rather than the finding.
        foot_target_height=0.025,
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
        # How strictly this gait is marked, as a fraction of the range. Named
        # rather than inherited: holding it constant across the rungs is what
        # makes widening the range legitimate, and that is a claim each task has
        # to make for itself. These are `curriculum.py`'s calibrated pair -- 0.25
        # and 0.20 at range 0.5, chosen so that standing still collects the same
        # 0.14 it does on a Go1 -- and this task has no reason to differ.
        # The fastest this gait may be commanded. **The only curriculum number
        # this task sets** -- the ladder is one shared shape scaled to it, so
        # level 1 means the same thing in all four. Here: no clock fixes the cadence here, so speed is free and this is the
        # fastest a trained policy has been measured holding.
        command_lin_ceiling=0.8,
        command_ang_ceiling=1.0,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )

    # ── Shared with the other locomotion tasks ────────────────────────────
    # Identical in all four on purpose: none of these is a property of the gait,
    # so a difference between tasks would be an accident rather than a finding.
    # The measurements behind them are in `jumper.tripod`, which is where they were
    # taken; changing one here without changing it there breaks the comparison
    # the four tasks exist to support.
    #
    # `foot_clearance` / `foot_swing_height`: at the old -2.0 / -0.25 the penalties
    # were paid rather than avoided -- peak height 0.0159 against a 0.03 target,
    # unmoved by 15697 iterations of doubled weights. Raised once the target came
    # down to something reachable.
    #
    # `rel_standing_envs` 0.1 -> 0.2, because a weight cannot create samples:
    # everything gated on a zero command was learning from a tenth of the data.
    # `stance_load_stand` 2.0 is the duty-cycle parity that follows from it
    # (0.5 * 0.8 / 0.2), which is why the two must move together.
    cfg.rewards["foot_clearance"].weight = -4.0
    cfg.rewards["foot_swing_height"].weight = -3.0
    cfg.commands["twist"].rel_standing_envs = 0.2

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg
