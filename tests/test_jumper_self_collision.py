"""jumper leg-vs-leg collision masks.

The self-collision penalty is only as good as the collision masks underneath it:
the reward term counts contacts, so anything the masks let through is scored,
and anything they block is invisible. Neither failure reports itself. A hull
that starts intersecting its own leg would show up as a constant penalty the
policy can only escape by standing still; a leg pair that stops colliding would
silently switch the penalty off for that pair.

What is guarded here:

1. Geoms on the **same** leg never collide. The collision geometry is a convex
   hull and a hull fills in concavities, so links that are merely near each
   other at rest (the finger against its own wrist link, `J2` against `J0`)
   would otherwise sit in permanent contact.
2. Geoms on **different** legs do collide -- that is the thing being penalised.
3. Every leg geom still collides with the ground.
4. The home pose has no self-contact at all, so a standing robot is scored zero.
"""

from __future__ import annotations

import itertools

import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjlab")

from tasks.jumper.common.constants import (
    HOME,
    HYBRID_COLLISION,
    LEGS,
    STAND_Z,
    get_spec,
)


def _leg_of(geom_name: str) -> str | None:
    """Leg a collision geom belongs to, or None for base_link/terrain."""
    return next((leg for leg in LEGS if geom_name.startswith(leg + "_")), None)


@pytest.fixture(scope="module")
def model():
    """Compiled jumper model with the hybrid collision policy and a ground plane."""
    spec = get_spec(None)
    HYBRID_COLLISION.edit_spec(spec)
    # Stand in for the terrain, which is contype=1/conaffinity=1.
    ground = spec.worldbody.add_geom()
    ground.name = "ground"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [0, 0, 1]
    ground.contype, ground.conaffinity, ground.condim = 1, 1, 3
    return spec.compile()


def _names(model) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        for i in range(model.ngeom)
    ]


def _active(model) -> list[int]:
    return [
        i
        for i in range(model.ngeom)
        if model.geom_contype[i] or model.geom_conaffinity[i]
    ]


def _can_collide(model, i: int, j: int) -> bool:
    """MuJoCo's pair test: (contype1 & conaffinity2) || (contype2 & conaffinity1)."""
    return bool(
        (model.geom_contype[i] & model.geom_conaffinity[j])
        or (model.geom_contype[j] & model.geom_conaffinity[i])
    )


def test_same_leg_geoms_never_collide(model):
    """Links on one leg are adjacent by construction and must not be scored."""
    names = _names(model)
    offenders = [
        f"{names[i]} x {names[j]}"
        for i, j in itertools.combinations(_active(model), 2)
        if _can_collide(model, i, j)
        and _leg_of(names[i]) is not None
        and _leg_of(names[i]) == _leg_of(names[j])
    ]
    assert not offenders, f"same-leg pairs are mask-allowed: {offenders}"


def test_different_legs_can_collide(model):
    """Every pair of distinct legs must be able to register a collision."""
    names = _names(model)
    colliding_pairs = {
        frozenset((_leg_of(names[i]), _leg_of(names[j])))
        for i, j in itertools.combinations(_active(model), 2)
        if _can_collide(model, i, j)
        and _leg_of(names[i]) is not None
        and _leg_of(names[j]) is not None
    }
    expected = {frozenset(p) for p in itertools.combinations(LEGS, 2)}
    assert colliding_pairs == expected, f"missing leg pairs: {expected - colliding_pairs}"


def test_every_collision_geom_still_hits_the_ground(model):
    """Enabling self-collision must not cost any geom its ground contact."""
    names = _names(model)
    ground = names.index("ground")
    missing = [
        names[i]
        for i in _active(model)
        if i != ground and not _can_collide(model, i, ground)
    ]
    assert not missing, f"geoms that no longer collide with the ground: {missing}"


def test_home_pose_has_no_self_contact(model):
    """A robot standing at HOME must score zero on the collision penalty."""
    data = mujoco.MjData(model)
    names = _names(model)
    data.qpos[:3] = [0.0, 0.0, STAND_Z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for joint_name, value in HOME.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)

    self_contacts = [
        f"{names[c.geom1]} x {names[c.geom2]} (dist={c.dist:.5f})"
        for c in (data.contact[i] for i in range(data.ncon))
        if "ground" not in (names[c.geom1], names[c.geom2])
    ]
    assert not self_contacts, f"self-contact while standing at HOME: {self_contacts}"
