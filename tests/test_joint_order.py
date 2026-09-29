"""The joint order the deployment contract is written against.

This pins the failure that actually happened. The V1.6 CAD export declares the
right leg of each pair before the left; the previous model declared left first.
Importing the export as-is renumbered all six legs, and **nothing in the suite
noticed**, because:

- `GAIT_JOINTS` is derived from `HOME`'s key order and stayed left-first, so
  `test_symmetry.py`, the only test that reads a joint order, still passed
- the action vector does **not** follow `GAIT_JOINTS`. `Entity.find_joints`
  defaults to `preserve_order=False` and returns model order, so the MJCF's
  declaration order is what fixes each slot in the 20-wide action and in
  `layout.json`'s `action_joint_order`

The result would have been a checkpoint that loads, an environment that steps,
and every left leg driven by its right leg's action -- forever, silently. The
robot on the bench does not have a way to tell you either: the gait would look
plausible and merely be wrong.

`build_jumper.order_legs` puts the legs back in `LEG_ORDER`. These tests are the
thing that makes a future export flipping them again a red test rather than a
week of debugging a limp.

Both tests were run against a deliberately reordered model before being trusted,
and both fail on it -- see `test_the_order_test_can_fail`, which builds that
model rather than asserting it from memory.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco", reason="compiling the model needs mujoco")

import mujoco

from tasks.jumper.common.constants import (
    GAIT_JOINTS,
    GRIPPER_JOINTS,
    JUMPER_XML,
    LEGS,
)

#: The 22 joints in the order every deployed `layout.json` and every trained
#: checkpoint was built against. Written out rather than derived, because a
#: derivation would follow the model and this has to be independent of it.
CONTRACT_ORDER = (
    "LF_J0_joint", "LF_J1_joint", "LF_J2_joint",
    "LF_J3_joint", "LF_J4_joint",
    "RF_J0_joint", "RF_J1_joint", "RF_J2_joint",
    "RF_J3_joint", "RF_J4_joint",
    "LM_J0_joint", "LM_J1_joint", "LM_J2_joint",
    "RM_J0_joint", "RM_J1_joint", "RM_J2_joint",
    "LR_J0_joint", "LR_J1_joint", "LR_J2_joint",
    "RR_J0_joint", "RR_J1_joint", "RR_J2_joint",
)


def model_joint_order(model=None) -> tuple[str, ...]:
    """The hinge joints of the compiled model, in model order.

    Takes a compiled `MjModel` so a caller can hand over one it built in memory.
    The control group below needs that: the asset's mesh paths reach into two
    directories now (the vendored export and `meshes/`), and a copy written
    anywhere else resolves neither.
    """
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(JUMPER_XML))
    names = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return tuple(names)


def test_model_joint_order_matches_the_deployment_contract():
    """The MJCF's declaration order *is* the contract; a CAD reshuffle breaks it."""
    assert model_joint_order() == CONTRACT_ORDER


def test_gait_joints_agrees_with_the_model_order():
    """The two must not drift apart.

    They are computed from different sources -- `GAIT_JOINTS` from `HOME`'s key
    order, the action from the model -- and agreeing today is not the same as
    being kept in agreement. When the export flipped the legs, this pair
    disagreed and was the only available signal.
    """
    from_model = tuple(j for j in model_joint_order()
                       if j not in GRIPPER_JOINTS)
    assert GAIT_JOINTS == from_model


def test_legs_constant_matches_the_contract_order():
    """`LEGS` names the tripod groups; a leg absent here is a silent no-op."""
    seen: list[str] = []
    for joint in CONTRACT_ORDER:
        leg = joint.split("_")[0]
        if leg not in seen:
            seen.append(leg)
    assert tuple(seen) == LEGS


def test_the_order_test_can_fail():
    """The control group.

    An assertion over a name tuple is exactly the kind that passes vacuously if
    the helper silently returns something empty or sorted. So build the model
    the broken export would have produced -- right leg first -- and require the
    check to reject it.
    """
    spec = mujoco.MjSpec.from_file(str(JUMPER_XML))
    original = model_joint_order()

    swapped = list(original)
    for a, b in (("LF", "RF"), ("LM", "RM"), ("LR", "RR")):
        left = [i for i, j in enumerate(swapped) if j.startswith(f"{a}_")]
        right = [i for i, j in enumerate(swapped) if j.startswith(f"{b}_")]
        assert len(left) == len(right), (a, b)
        for i, k in zip(left, right):
            swapped[i], swapped[k] = swapped[k], swapped[i]
    assert tuple(swapped) != original, "the swap did nothing; the control is vacuous"

    # Rename the joints in place to fake the flipped model: the check reads
    # names in model order, so a rename is enough to reproduce the failure
    # without rebuilding the tree. Two passes via a temporary prefix, because a
    # swap renames A to B while B still holds the name and MjSpec rejects that.
    rename = dict(zip(original, swapped))
    for joint in spec.joints:
        if joint.name in rename:
            joint.name = f"__tmp__{joint.name}"
    for joint in spec.joints:
        if joint.name.startswith("__tmp__"):
            joint.name = rename[joint.name.removeprefix("__tmp__")]
    assert model_joint_order(spec.compile()) != CONTRACT_ORDER
