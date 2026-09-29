"""Hyper-parameters for jumper.swing.

**Only this file changes this task's training hyper-parameters**; no other task is
affected, and nothing this task does not state is inherited silently.

The numbers start from the four velocity tasks' shared baseline, and that is a
starting point rather than a controlled comparison -- this task has a different
reward, a different termination and a different observation, so nothing about it
is a control group for them. Two departures from that baseline are deliberate and
are noted where they appear: the discount and the rollout length both have to
cover a **cycle** rather than a step, and there is no `runner_cls` because the
velocity runner logs tracking metrics this task does not have.

**Every timing number here depends on the swing's period, and the period is no
longer one number.** Each episode draws a rope length from `ROPE_LENGTH_RANGE`, so
the period is 1.50 to 2.65 s -- 75 to 133 control steps at 50 Hz -- and a 30 s
episode is 11 to 20 swings. Anything below that quotes a range for that reason.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config.

    `experiment_name` is the middle segment of the log path: the full structure is
    `logs/<model>/jumper.swing/<date-time>`, with the outermost segment decided by
    `--model`.
    """
    return jumper_ppo_baseline(
        experiment_name="jumper.swing",
        # ── Network ──
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # **Lowered from the velocity tasks' 1.0, and this one is about the
        # plank rather than about learning rate.** The action is a joint offset
        # scaled by 0.25 rad, so a policy that starts at sigma = 1 samples
        # +/-0.25 rad on twenty joints at once -- a robot thrashing hard enough to
        # walk itself off a 0.44 m plank. Measured at iteration 0 on native cpu,
        # 128 environments: mean episode length **17 steps**, a third of a second,
        # ending almost entirely `off_the_plank`. On flat ground that same noise
        # costs nothing, because there is nowhere to fall off to, which is why the
        # four velocity tasks can leave it at 1.0.
        #
        # 0.3 is +/-0.075 rad, which is enough to explore a stance and not enough
        # to leave the deck on the first step. The cost is slower exploration of
        # large postures; if the policy stalls having learnt only to stand still,
        # this is the first number to raise.
        init_std=0.3,
        # ── PPO ──
        # **0.005, and the number has been wrong once already in an instructive
        # way.**
        #
        # Two forces act on sigma and nothing else does: the entropy bonus pushes
        # it up with a gradient of `entropy_coef / sigma`, and the return pushes it
        # down. Setting those equal against a return whose cost grows as sigma^2
        # gives `sigma_eq ~ sqrt(entropy_coef)`, so the coefficient is a factor of
        # four for every halving of sigma.
        #
        # The scaling held. The **constant** did not, and the reason is worth
        # keeping: it was fitted at 0.01 -> sigma 0.8, measured on a reward where
        # the only thing opposing noise was `action_rate_l2` and the objective was
        # actually *helped* by it -- random thrashing excites a pendulum. That fit
        # predicted 0.0025 -> 0.4. What happened was 0.25 and still falling.
        #
        # In between, `pump_power` was added. It is signed and unclipped, so a lean
        # at the wrong phase is charged rather than ignored, and random leaning
        # averages to zero power while costing variance. That is a second and much
        # stronger downward force on sigma, and it did not exist when the constant
        # was measured. **A scaling law fitted against one reward does not survive
        # a change to the reward**, which is the general form of the mistake.
        #
        # So this is re-anchored on the only point measured against the reward as
        # it now stands -- 0.0025 -> 0.25 -- and the same square-root scaling puts
        # 0.005 at about **0.35**, near where `init_std` starts it.
        #
        # If sigma keeps falling rather than settling, the thing to check is not
        # this number: a policy whose noise collapses while its reward and episode
        # length climb has found something and is exploiting it, which is fine. A
        # collapse while both are flat is premature convergence, and then this goes
        # up again.
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        # **Raised from the velocity tasks' 0.99.** The payoff for a pump arrives
        # most of a swing after the action that earned it, and a swing here is 75
        # to 133 control steps depending on the rope this episode drew. At
        # gamma = 0.99 a reward a full period out is discounted to 0.47 on the
        # shortest rope and 0.26 on the longest; at 0.995 it is 0.69 and 0.51. The
        # horizon has to cover the thing being learnt, and here the thing being
        # learnt is a cycle rather than a step -- and the longest ropes, which are
        # the hardest, are the ones the discount bites hardest on.
        gamma=0.995,
        # **The baseline's 0.95, and it is worth knowing what it does and does not
        # reach.** GAE weights the residuals by `(gamma * lam)^k`, so the advantage
        # estimate's own horizon is `1 / (1 - gamma * lam)` = **18 steps, 0.37 s**.
        # That is a fifth of a period. Nothing about raising `gamma` changes this:
        # credit for an action does not travel across a swing through the advantage
        # estimate, it travels through the **critic**, which bootstraps at the end
        # of the segment.
        #
        # That is the right division of labour here rather than a compromise --
        # this critic is privileged. It reads `swing_offset` and `swing_velocity`
        # with five frames of history, so the value of a state is a nearly
        # observable function of what it is given and the bootstrap should be
        # accurate. Raising `lam` to reach a period (0.99 gives 67 steps) would buy
        # less bias at a lot more variance, against a value function that does not
        # need the help.
        #
        # Watch it if the run stalls: a critic that cannot predict the swing makes
        # this the wrong trade, and the tell is the explained variance.
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        # ── Runner ──
        # **Doubled from the velocity tasks' 24, and the reason written here first
        # was wrong.** It said the segment had to contain a whole half-cycle, so
        # that the advantage for "lean forward now" would include the part of the
        # arc it paid off in. It does not: see `lam` -- the advantage estimate's
        # horizon is 18 steps whatever this is set to, and a half-cycle is 38 to 66.
        # That argument also had a number in it (a 2.35 s period) that stopped being
        # a single number when the rope length started varying, which is how it came
        # to be re-read.
        #
        # What the segment length actually decides is how far the rollout is
        # truncated relative to that 18-step horizon, since the cut is covered by
        # the critic's bootstrap. 24 steps is only 1.3 horizons and leaves real
        # truncation bias; 48 is 2.7 and does not. The conclusion survives its own
        # justification being replaced, which is the only reason the number did not
        # move with it.
        num_steps_per_env=48,
        max_iterations=10_000,
    )


# No `runner_cls`. `VelocityOnPolicyRunner` exists to log velocity-tracking
# metrics, and this task has no twist command to track -- `registry.load_runner_cls`
# treats a missing function as "use rsl_rl's default", which is what is wanted.


def play_status():
    """The line `scripts/play.py` prints while replaying.

    Not the velocity tasks' line: this task has no command, and the robot's own
    speed is close to meaningless here -- it is carried along an arc by a plank,
    so it reads fastest at the bottom of the swing whatever the policy is doing.
    What is worth watching is the swing.

    **The rope length is on the line because it is redrawn every episode.** Without
    it a replay is a policy doing visibly different things for no visible reason,
    and the one question worth asking of this policy -- does it cope with a swing
    it has not been tuned to -- is the one you cannot ask. Pin it with
    `--swing-length` to watch one.
    """

    def status(entity, index: int, env) -> str:
        import torch

        from .mdp import state

        unwrapped = getattr(env, "unwrapped", env)
        with torch.inference_mode():
            amplitude = float(state.swing_amplitude(unwrapped)[index])
            speed = float(state.swing_bottom_speed(unwrapped)[index])
            sway = float(state.sway_amplitude(unwrapped)[index])
            tilt = float(state.deck_up_in_body(unwrapped)[index, 2].clamp(-1.0, 1.0))
            length = float(state.rope_length(unwrapped)[index])
        deg = 57.29578
        # The speed is on the line as well as the amplitude, and they are the same
        # measurement -- `v = sqrt(2 g L (1 - cos A))`. Both, because the reward
        # pays for both and they part company across rope lengths: 30 degrees is
        # 1.27 m/s on a short swing and 2.18 on a long one, so a replay that shows
        # only the angle cannot tell you why two runs with the same number on screen
        # are scoring differently.
        return (
            f"rope {length:4.2f} m   "
            f"swing {amplitude * deg:5.1f} deg   "
            f"bottom {speed:4.2f} m/s   "
            f"sway {sway * deg:4.1f}   "
            f"off deck {torch.arccos(torch.tensor(tilt)).item() * deg:4.1f} deg"
        )

    return status
