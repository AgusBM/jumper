"""PPO hyperparameters for jumper.ref_free_jump."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update.
#:
#: **A rollout is a duration**, and this task runs at 200 Hz: the baseline's 24
#: steps would be 0.12 s, a tenth of the recording's span. 48 is 0.24 s, the
#: correction `jumper.posture` made for its own move to 200 Hz.
#:
#: A module constant because `env_cfg.py` needs it: the prior's schedule is
#: written in iterations and applied by a curriculum that counts environment
#: steps, and this is the factor between the two. A literal in both places would
#: let one move without the other and change the schedule's pace in silence.
NUM_STEPS_PER_ENV = 48


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.ref_free_jump",
        # jumper.posture's, which are the baseline's: the fourth, 64-unit layer
        # came with the walking tasks' five frames of history and actuator force
        # (`HIDDEN_DIMS` in common/ppo.py), and this actor now carries the same --
        # 407 wide, the critic 435. It had jumper.jump's three while it saw one
        # frame, 67 wide.
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # Exploration is set in radians, not in action units. At this task's
        # action scale of 0.5 a std of 0.3 is 0.15 rad, which is what jumper.jump
        # explores with (0.6 at 0.25). At kp=20 that is already 1.7x the 0.087 rad
        # of error that saturates the torque plateau.
        init_std=0.3,
        # A fifth of jumper.jump's 0.005, which ran away here. With the
        # feedforward carrying the motion, the rewards hardly notice action
        # noise, so the entropy bonus was nearly all the gradient std had: over
        # 848 iterations of `2026-09-26_18-59-59` (4096 envs, RTX 5090 D,
        # warp:cuda) std went 0.30 -> 1.83 -- 0.9 rad of target noise at this
        # action scale -- `action_rate_l2` -0.0005 -> -0.56, and the mean reward
        # 7.8 -> 5.1. The bonus, `entropy_coef * Loss/entropy`, was 0.005 x 5 to
        # 7 early against a surrogate of 0.002 to 0.005, and 0.005 x 40 = 0.2
        # against 0.015 at the end. At 0.001 it starts at 0.001 x 4.4, the
        # surrogate's own order. If std still climbs, the next step is 0.
        #
        # Sized while `action_rate_l2` charged 0.0144 per squared radian of
        # target. It now charges 16x that (`ACTION_RATE_PER_RAD_SQ`, env_cfg.py),
        # and that term taxes the same noise: jumper.posture measured the two as
        # one loop, so a change to either wants a look at the other.
        entropy_coef=0.001,
        # Corrected for 200 Hz, as jumper.posture's are: the discount buys a
        # horizon in steps, and 0.99 at 200 Hz is 0.5 s -- about the gap between
        # the push and the landing that pays for it. The fourth root keeps the
        # 2 s the baseline has at 50 Hz.
        gamma=0.99**0.25,
        lam=0.95**0.25,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # The prior is gone by `PRIOR_END_ITERATIONS` (3000, env_cfg.py); the
        # other half is the objective on its own.
        max_iterations=6000,
    )
