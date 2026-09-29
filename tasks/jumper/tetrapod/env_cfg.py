"""Environment config for jumper.tetrapod.

The environment skeleton is shared by the four jumper tasks
(`tasks/jumper/common/velocity_env.py`). What this task adds is the `tetrapod` gait
prior and the **fixed-frequency phase clock** that drives it: one observation
term, and the same frequency handed to the gait reward and to `stance_load`.

All three constraints on the frequency land on one number for this gait: 12
control steps, 25/6 Hz, 4 steps of swing per pair, and exactly 0.50 m/s at the
0.08 m stride the IK allows. See `mdp/phase.py`.
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
from ..common.mdp.phase import phase_clock
from ..common.mdp.rewards import (
    stance_foot_ground_gap,
    stance_foot_load,
    standing_foot_load,
)
from ..common.tof import tof_sensor
from ..common.velocity_env import velocity_env_cfg
from .mdp.phase import (
    GAIT_FREQ_HZ,
    MIRROR_KIND,
    STANCE_FORCE_THRESHOLD,
    SWING_GROUPS,
)
from .mdp.rewards import tetrapod_gait

#: The observation term's name. **Not gait-specific on purpose**: it is in
#: `scripts/export.py`'s deployable whitelist and the on-robot observation builder
#: constructs terms by it, so all three gait tasks use the same name. What differs
#: between them is the mirror rule, which travels in the term's params -- see
#: `mdp/phase.py::MIRROR_KIND` and `common/mdp/symmetry.py`.
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
        gait_reward=tetrapod_gait,
        gait_name="tetrapod",
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
        # level 1 means the same thing in all four. Here: the same 4.167 Hz clock as ripple but four legs down, so the stroke per
        # stance is longer and the cap lands where tripod's does.
        command_lin_ceiling=0.5,
        command_ang_ceiling=0.75,
        command_lin_std_ratio=STD_LIN_RATIO,
        command_ang_std_ratio=STD_ANG_RATIO,
    )

    # ── The phase clock ───────────────────────────────────────────────────
    # The observation, the gait reward and `stance_load` are handed the frequency
    # **from the same constant, at the same place**, which is the only thing
    # keeping them in sync. A mismatch does not raise: the policy would be shown
    # one clock and scored against another, and the gait term would stop being
    # learnable while everything else looked healthy.
    #
    # A fresh ObservationTermCfg per group, not one shared instance: the
    # observation manager mutates term configs in place while resolving them, so
    # sharing one object across two groups would have them resolve over each other.
    #
    # `history_length=0` is a decision. The skeleton gives every other term five
    # frames *before* this function runs, so a term mounted here gets none unless
    # it asks -- and for this term the answer is none: the clock is a deterministic
    # function of episode time, so five frames are completely determined by the
    # current one. History on it is exactly zero information and eight wasted
    # inputs.
    for group in ("actor", "critic"):
        cfg.observations[group].terms[PHASE_OBS_NAME] = ObservationTermCfg(
            func=phase_clock,
            history_length=0,
            params={
                "freq_hz": GAIT_FREQ_HZ,
                "command_name": "twist",
                "command_threshold": 0.05,
                # Read by `common/mdp/symmetry.py`, ignored by `phase_clock`.
                "mirror_kind": MIRROR_KIND,
            },
        )

    cfg.rewards["tetrapod_gait"].params["freq_hz"] = GAIT_FREQ_HZ

    # ── Load on the supporting legs ───────────────────────────────────────
    # The gait term scores *which* legs are touching; this scores whether the ones
    # that should be supporting are actually carrying the robot. Touching is cheap
    # -- a foot can rest on the ground with the weight going elsewhere -- and the
    # two together say "these legs, and really on them".
    #
    # The weight is deliberately below the gait term's 1.0: this is shaping for the
    # gait, not a competitor to it, and every positive term is also an addition to
    # what a policy might collect without walking (the gates inside the term are
    # what keep that at zero).
    #
    # `STANCE_FORCE_THRESHOLD` is this gait's own -- four legs support, so a fair share of the robot's 29.5 N is
    # 7.4 N and the threshold is about half of it. Tripod's 5.0 would ask 68% of a
    # fair share from each of four legs, which no even split provides.
    cfg.rewards["stance_load"] = RewardTermCfg(
        func=stance_foot_load,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "groups": SWING_GROUPS,
            "freq_hz": GAIT_FREQ_HZ,
            "force_threshold": STANCE_FORCE_THRESHOLD,
            "command_threshold": 0.05,
        },
    )

    # ── Load while standing ───────────────────────────────────────────────
    # The standing half of `stance_load`, which is gated off here on purpose.
    # Separate terms because the threshold differs: 5 N is half a fair share over
    # three supporting legs, while standing spreads ~27 N over six for about
    # 4.5 N each, and 3 N is two thirds of that. See `standing_foot_load`.
    cfg.rewards["stance_load_stand"] = RewardTermCfg(
        func=standing_foot_load,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "force_threshold": 3.0,
            "command_threshold": 0.05,
        },
    )

    # ── Distance to the ground for feet that should be down ───────────────
    # The dense counterpart to the two load terms. Force is zero until the foot
    # arrives, so those two say nothing about *approaching*; this one is the gap
    # to the terrain and falls the whole way to contact. Measured on the
    # 2026-09-06 run, about half the under-loaded stance legs were not touching at
    # all, and that half is what this reaches.
    #
    # Small to begin with. It pulls against `foot_clearance` (weight -2.0), which
    # costs |height - 0.1| * |v_xy| for every foot -- narrowly, because that term
    # is weighted by foot speed and a support foot is nearly stationary, but a
    # support foot that is high *and* moving is contested. At -0.5 a foot at the
    # 5 cm clamp costs 0.025, an order below the gait term. If a retrain shows
    # swing height collapsing, that is the two fighting: reach for `max_gap`
    # before this weight.
    cfg.rewards["stance_ground_gap"] = RewardTermCfg(
        func=stance_foot_ground_gap,
        weight=-0.5,
        params={
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "groups": SWING_GROUPS,
            "freq_hz": GAIT_FREQ_HZ,
            "max_gap": 0.05,
            "command_threshold": 0.05,
        },
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
    cfg.rewards["stance_load_stand"].weight = 2.0

    # The dToF, which the model carries (`common/tof.py`). Replay only, where the
    # live viewer shows it in the window's corner; nothing in training reads it,
    # and there it would be 2268 rays per environment per step spent on nothing.
    if play:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    return cfg
