"""The PPO constructor the four jumper tasks build their configs from.

**This file owns the plumbing, not the numbers.** Every hyper-parameter and every
network dimension is an argument, and each task states its own in its own
`tasks/jumper/<task>/rl_cfg.py`. Editing a default here changes only what a task
that does not state that value would get -- and `tests/test_ppo_cfg.py` requires
all four to state the ones that matter, so in practice a default is what a *new*
task starts from rather than something an existing one silently inherits.

This used to hand every task one identical config, justified by the four tasks
being "each other's control groups", so that holding the algorithm side fixed
would attribute any difference in outcome to the gait prior.

**That justification was never true, and it is worth being specific about why,
because it is the kind of claim that survives by sounding right.** Measured on
the built configs:

    task            extra actor obs   extra rewards                positive weight
    jumper.flat       --                --                                     6.00
    jumper.tripod     gait_phase        tripod_gait 1.0, stance_load 0.5       7.50
    jumper.tetrapod   --                tetrapod_gait 1.0                      7.00
    jumper.ripple     --                ripple_gait 1.0                        7.00

Pairwise, only two of the six comparisons differ by a single variable:

    flat vs tetrapod    +tetrapod_gait 1.0                          one variable
    flat vs ripple      +ripple_gait 1.0                            one variable
    flat vs tripod      +tripod_gait 1.0, +stance_load 0.5, +obs    three
    tripod vs tetrapod  as above, in reverse                        three
    tripod vs ripple    as above, in reverse                        three
    tetrapod vs ripple  same budget, structurally different terms   not a control

So `jumper.flat` is a genuine control for two of the three gait tasks, and for
`jumper.tripod` it is not: three things change at once there, and an outcome cannot
be attributed to the gait prior among them. `tetrapod vs ripple` is not a
control in either direction -- the gait terms are mutually exclusive by
construction (a perfect tripod pattern scores 0 under ripple), so they are two
different objectives rather than two settings of one.

None of that is on the algorithm side, which is the point: identical
hyper-parameters were never what made any of these comparisons valid, and
differing ones would not break the two that are.

So the real reason the four carried the same numbers is the honest and much
smaller one: **it was a reasonable starting point and nobody had a measured
reason to diverge.** That remains true, and it is why all four still state the
same values today. What has changed is only that diverging no longer requires
breaking the other three.

The cost is that four copies of a number drift, and nothing here can tell an
intentional divergence from a typo. When changing the starting point for **all**
jumper tasks, change all four files.

## What is not a per-task argument

`symmetry_cfg`'s `data_augmentation_func` points at `mdp/symmetry.py::mirror_jumper`,
which is a property of the robot's topology rather than of a task. All four jumper
tasks are the same robot, so it stays fixed here; a task that wanted a different
one is really a different robot.
"""

from __future__ import annotations

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

#: The default hidden layers, for a task that does not state its own.
#:
#: A fourth layer was added with the 64-unit tail when the observation grew from
#: 74 to 470 -- five frames of history, actuator force included. The first layer
#: was already doing most of the compression (74 -> 512 widened, 470 -> 512 does
#: not), so the depth is there to give the extra input somewhere to be reduced
#: rather than to add raw capacity.
#:
#: Actor and critic used to be pinned to this single constant, described as
#: "shared so the two cannot drift apart". They are now separate arguments, so
#: that protection is gone by choice: the critic sees 13 observation terms to the
#: actor's 8 and there is no reason its width has to match. If a task sets them
#: differently, the reason belongs in a comment next to the values.
HIDDEN_DIMS = (512, 256, 128, 64)


def jumper_ppo_baseline(
    experiment_name: str,
    *,
    # ── Network ──
    actor_hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
    critic_hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
    activation: str = "elu",
    obs_normalization: bool = True,
    init_std: float = 1.0,
    std_type: str = "scalar",
    # ── PPO ──
    entropy_coef: float = 0.01,
    learning_rate: float = 1.0e-3,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_param: float = 0.2,
    value_loss_coef: float = 1.0,
    use_clipped_value_loss: bool = True,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    max_grad_norm: float = 1.0,
    # ── Runner ──
    num_steps_per_env: int = 24,
    max_iterations: int = 10_000,
    save_interval: int = 50,
    # ── Symmetry ──
    symmetry: bool = True,
    use_data_augmentation: bool = True,
    use_mirror_loss: bool = True,
    mirror_loss_coeff: float = 1.0,
) -> RslRlOnPolicyRunnerCfg:
    """Build one task's rsl_rl config.

    Everything after `experiment_name` is keyword-only. These are twenty-odd
    scalars of the same few types, and a positional call that silently put
    `gamma` where `lam` belongs would train, converge to something, and never
    say what happened.

    Args:
        experiment_name: the log directory name. Each task passes its own id so
            that experiment logs land in separate directories.
        actor_hidden_dims: actor MLP hidden layers.
        critic_hidden_dims: critic MLP hidden layers. Free to differ from the
            actor's -- the critic sees more observation terms.
        activation: hidden activation for both networks.
        obs_normalization: `EmpiricalNormalization` on the inputs of both.
        init_std: initial action standard deviation.
        std_type: "scalar" for one std across all action dimensions, "log" for
            per-dimension.
        entropy_coef: the entropy bonus. **The one to reach for first when a run
            shows a rising `Policy/mean_std` with falling precision-sensitive
            rewards** -- measured on a rough-terrain run, sigma climbed 0.24 ->
            0.43 while `tripod_gait` fell 21% and `action_rate_l2`'s penalty grew
            77%, with the coarse objectives (tracking, upright, episode length)
            flat or better. That signature is action noise, not a locomotion
            failure, and this is its coefficient.
        learning_rate: the initial rate; `schedule="adaptive"` then moves it to
            hold `desired_kl`.
        schedule: "adaptive" or "fixed".
        desired_kl: the KL the adaptive schedule targets.
        gamma: discount.
        lam: GAE lambda.
        clip_param: PPO ratio clip.
        value_loss_coef: weight on the value loss.
        use_clipped_value_loss: clip the value loss as well as the policy loss.
        num_learning_epochs: passes over each batch.
        num_mini_batches: mini-batches per pass.
        max_grad_norm: gradient clipping.
        num_steps_per_env: rollout length per environment per iteration.
        max_iterations: the default iteration count; `--max-iterations` overrides.
        save_interval: iterations between checkpoints.
        symmetry: whether to enable left-right symmetry. The hexapod has strict
            sagittal mirror symmetry (mirror_pairs in the topology); the mirror
            transform is in `mdp/symmetry.py`.
        use_data_augmentation: append mirrored samples to every mini-batch,
            doubling the effective sample count.
        use_mirror_loss: an extra MSE term penalising "the policy's output on a
            mirrored observation != the mirror of its original output".
        mirror_loss_coeff: the coefficient of that mirror loss.

    With both symmetry switches off, rsl_rl still computes and logs the symmetry
    loss but detaches it from the graph -- useful for observing how asymmetric the
    policy is before deciding whether to enable it for real.

    Note: rsl_rl's PPO **does not support symmetry** with an RNN actor or critic
    and raises outright. The current networks are MLPs and are unaffected; moving
    to an LSTM later (for partially observable rough terrain, say) would mean
    choosing between the two.

    Every dict this builds is constructed here rather than at module level, so two
    tasks' configs never alias one. `tests/test_ppo_cfg.py` pins that: a shared
    `symmetry_cfg` would let a task tuning its own mirror loss silently retune the
    other three, which is the failure this whole file is arranged to prevent.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=tuple(actor_hidden_dims),
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": std_type,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=tuple(critic_hidden_dims),
            activation=activation,
            obs_normalization=obs_normalization,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=value_loss_coef,
            use_clipped_value_loss=use_clipped_value_loss,
            clip_param=clip_param,
            entropy_coef=entropy_coef,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            learning_rate=learning_rate,
            schedule=schedule,
            gamma=gamma,
            lam=lam,
            desired_kl=desired_kl,
            max_grad_norm=max_grad_norm,
            symmetry_cfg=(
                {
                    "data_augmentation_func": "tasks.jumper.common.mdp.symmetry:mirror_jumper",
                    "use_data_augmentation": use_data_augmentation,
                    "use_mirror_loss": use_mirror_loss,
                    "mirror_loss_coeff": mirror_loss_coeff,
                }
                if symmetry
                else None
            ),
        ),
        experiment_name=experiment_name,
        save_interval=save_interval,
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
    )


__all__ = ["HIDDEN_DIMS", "jumper_ppo_baseline"]
