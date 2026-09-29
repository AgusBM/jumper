"""`actuator_power` and `actuator_headroom`: the two ways they can be wrong quietly.

Both terms multiply a torque by something that belongs to a joint, and both
produce a perfectly plausible number when the something belongs to the **wrong**
joint. A power computed from a mismatched pair is still watts; a headroom ratio
computed against another joint's speed is still a fraction. Nothing downstream
can tell, and the policy is trained to minimise a quantity nobody named.

So these tests run the real functions against a stub entity, which is what makes
the pairing testable at all: in a real environment the robot's actuator and joint
lists are identical and in the same order, so index-pairing and name-pairing agree
and neither test would fail against the broken version.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="the terms are torch code")

from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402

from tasks.jumper.common.actuator import CURVE  # noqa: E402
from tasks.jumper.common.mdp.rewards import (  # noqa: E402
    actuator_headroom,
    actuator_power,
)


class _Data:
    def __init__(self, force, vel):
        self.actuator_force = force
        self.joint_vel = vel


class _Entity:
    """Just enough entity for the two terms: names, forces and joint speeds."""

    def __init__(self, joint_names, actuator_names, force, vel):
        self.joint_names = tuple(joint_names)
        self.actuator_names = tuple(actuator_names)
        self.data = _Data(force, vel)


class _Env:
    def __init__(self, entity):
        self.scene = {"robot": entity}


def _cfg(entity, actuator_names=None) -> SceneEntityCfg:
    """A resolved `SceneEntityCfg` without a scene to resolve against.

    `resolve` needs a real entity, so the ids are filled in by hand -- which is
    also what keeps each test's config object distinct, since the terms cache
    their name lookup by `id(asset_cfg)`.
    """
    cfg = SceneEntityCfg("robot", actuator_names=actuator_names)
    if actuator_names is None:
        cfg.actuator_ids = slice(None)
    else:
        order = {n: i for i, n in enumerate(entity.actuator_names)}
        cfg.actuator_ids = [order[n] for n in actuator_names]
    return cfg


def test_torque_is_paired_with_its_own_joints_speed() -> None:
    """The pairing is by name, and pairing by index would be silent.

    The stub's two lists are **deliberately in different orders**, which is the
    only arrangement that can tell the two implementations apart. A real hexa
    entity has them identical, so this failure could not be reproduced there --
    which is exactly why it would survive.
    """
    entity = _Entity(
        joint_names=("a", "b"),
        actuator_names=("b", "a"),  # reversed
        force=torch.tensor([[2.0, 0.0]]),  # actuator "b" pushes 2 N*m, "a" none
        vel=torch.tensor([[10.0, 1.0]]),  # joint "a" spins at 10, "b" at 1
    )
    env = _Env(entity)

    # Correct: 2 N*m on actuator "b" x joint "b"'s 1 rad/s = 2 W, over 2 actuators.
    got = actuator_power(env, _cfg(entity))
    assert got.item() == pytest.approx(1.0)

    # The control: index pairing would have used joint "a"'s 10 rad/s and read
    # 20 W, i.e. ten times the truth.
    naive = (entity.data.actuator_force * entity.data.joint_vel).abs().mean()
    assert naive.item() == pytest.approx(10.0)


def test_power_is_a_magnitude_so_braking_is_not_paid_for() -> None:
    """Signed power would pay a policy for falling into its joint limits."""
    entity = _Entity(("a",), ("a",), torch.tensor([[-1.5]]), torch.tensor([[2.0]]))
    assert actuator_power(_Env(entity), _cfg(entity)).item() == pytest.approx(3.0)


def test_an_actuator_with_no_joint_of_its_name_raises() -> None:
    """Rather than pairing it with whatever is at that index."""
    entity = _Entity(("a",), ("gearbox",), torch.zeros(1, 1), torch.zeros(1, 1))
    with pytest.raises(KeyError, match="drive no joint"):
        actuator_power(_Env(entity), _cfg(entity))


@pytest.mark.parametrize(
    "fraction, expected",
    [(0.0, 0.0), (0.5, 0.0), (0.687, 0.0), (1.0, 1.0)],
)
def test_headroom_is_zero_below_the_deadband_and_one_at_the_limit(
    fraction: float, expected: float
) -> None:
    """The term's whole shape, at a speed where the curve is at its plateau.

    `fraction` is of what the servo can produce at that speed, so 1.0 is an
    actuator pinned against the clamp -- which is the state the term is named
    for.
    """
    deadband = CURVE.continuous_torque / CURVE.plateau_torque
    slow = 0.5 * CURVE.corner_speed  # on the plateau, so the limit is the plateau
    entity = _Entity(
        ("a",), ("a",),
        torch.tensor([[fraction * CURVE.plateau_torque]]),
        torch.tensor([[slow]]),
    )
    got = actuator_headroom(_Env(entity), deadband, _cfg(entity))
    assert got.item() == pytest.approx(expected, abs=2e-3)


def test_headroom_measures_against_the_speed_the_joint_is_actually_at() -> None:
    """The same torque is more of the capability when the joint is moving fast.

    This is the property that makes the term worth its complexity: dividing by the
    flat `EFFORT_LIMIT` would score these two identically, and they are not the
    same situation -- the fast one has no torque left to reject a disturbance.
    """
    tau = 0.9 * CURVE.continuous_torque
    fast = CURVE.corner_speed + 2.0 * CURVE.decay_speed  # well past the corner
    slow = 0.0
    deadband = CURVE.continuous_torque / CURVE.plateau_torque

    costs = []
    for speed in (slow, fast):
        entity = _Entity(("a",), ("a",), torch.tensor([[tau]]), torch.tensor([[speed]]))
        costs.append(actuator_headroom(_Env(entity), deadband, _cfg(entity)).item())

    assert costs[0] == pytest.approx(0.0), "below the deadband while slow"
    assert costs[1] > 0.5, (
        "the same torque past the corner speed is most of what the servo can "
        f"still produce, and the term did not see it: {costs[1]:.3f}"
    )


def test_past_the_cutoff_speed_the_ratio_stays_finite() -> None:
    """The curve is exactly zero up there, and a division by it is not a big
    number but a meaningless one. The clamp keeps the gradient finite."""
    entity = _Entity(
        ("a",), ("a",),
        torch.tensor([[0.1]]),
        torch.tensor([[CURVE.cutoff_speed * 1.5]]),
    )
    got = actuator_headroom(_Env(entity), 0.687, _cfg(entity)).item()
    # Bounded by 1 by construction (the ratio is clamped there), and measured at
    # 1.0000002 in float32 -- the division and the square each round up. Worth a
    # tolerance rather than a tighter clamp: two parts in ten million on a reward
    # term is not a number anything can notice, and chasing it would add an
    # operation to every step of every environment.
    assert math.isfinite(got) and 0.0 <= got <= 1.0 + 1e-5
