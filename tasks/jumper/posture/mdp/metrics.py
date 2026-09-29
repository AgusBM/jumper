"""Metrics this task logs but does not pay for.

A metric is a number on the dashboard with no weight behind it. That makes it the
right home for a quantity you want to *watch* while deciding what it is worth --
and the wrong home for anything the policy should optimise, which belongs in a
reward where its weight can be argued about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def landing_force_max(
    env: "ManagerBasedRlEnv", sensor_name: str
) -> torch.Tensor:
    """The hardest footfall this step, in newtons, [num_envs].

    `soft_landing` already logs `Metrics/landing_force_mean`, and a mean is the
    wrong statistic for an impact. Measured over a replay: 23 landings averaging
    6.3 N with a p95 of 20.3 and a peak of 21.2 -- a factor of three between the
    typical landing and the worst one. What breaks a leg is the tail, and the mean
    moves hardly at all when the tail does.

    Declared with `reduce="max"`, so the episode metric is the **peak over the
    episode** rather than the mean of per-step peaks. That is the number to read
    beside `landing_force_mean`: the mean says how the gait lands, the max says
    what the worst one was, and a change that improves the first while the second
    holds is a gait that has stopped landing rather than started landing softly.

    **The two live under different prefixes, which is worth knowing before
    looking for them.** `landing_force_mean` is pushed straight into
    `env.extras["log"]` by the `soft_landing` reward and appears as
    `Metrics/landing_force_mean`, every step. This one goes through the metrics
    manager, which reports at episode end, so it is
    **`Episode_Metrics/landing_force_max`**. Same subject, two different places in
    the dashboard, because one is a reward function logging on the way past and
    the other is a metric.

    Zero on a step where nothing landed, which is most of them. That is correct
    under `max` and would be wrong under the default `mean`, where the zeros would
    dilute the number by the duty cycle and turn it into a quantity about cadence.
    """
    sensor = env.scene[sensor_name]
    force = sensor.data.force
    assert force is not None, f"sensor {sensor_name!r} carries no force field"
    landed = sensor.compute_first_contact(dt=env.step_dt).float()
    return torch.max(torch.linalg.norm(force, dim=-1) * landed, dim=1).values


__all__ = ["landing_force_max"]
