"""Correctness of loading and querying the jump reference motion (npz).

Does not depend on the simulation environment: it validates permutation,
interpolation and time semantics directly against `JumpReference`.
How to run: `.venv/Scripts/python -m pytest tests/test_jump_reference.py -q`
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tasks.jumper.jump.mdp.reference import JUMP_REF_NPZ, JumpReference

JOINT_NAMES = [
    "LF_J0_joint", "LF_J1_joint", "LF_J2_joint",
    "LF_J3_joint", "LF_J4_joint",
    "RF_J0_joint", "RF_J1_joint", "RF_J2_joint",
    "RF_J3_joint", "RF_J4_joint",
    "LM_J0_joint", "LM_J1_joint", "LM_J2_joint",
    "RM_J0_joint", "RM_J1_joint", "RM_J2_joint",
    "LR_J0_joint", "LR_J1_joint", "LR_J2_joint",
    "RR_J0_joint", "RR_J1_joint", "RR_J2_joint",
]
LEG_ORDER = ["LF", "RF", "LM", "RM", "LR", "RR"]


@pytest.fixture(scope="module")
def ref() -> JumpReference:
    return JumpReference(JUMP_REF_NPZ, "cpu", JOINT_NAMES)


def test_meta_matches_npz(ref: JumpReference) -> None:
    d = np.load(JUMP_REF_NPZ, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    assert ref.n == len(d["t"])
    assert ref.t_go == meta["t_go"]
    assert ref.duration == pytest.approx(float(d["t"][-1]))
    assert ref.span == pytest.approx(ref.duration - ref.t_go)


def test_permutation_identity_when_orders_match(ref: JumpReference) -> None:
    """When joint_order matches the asset (as it currently does), q columns match the raw npz."""
    d = np.load(JUMP_REF_NPZ, allow_pickle=True)
    q0 = torch.as_tensor(d["q"], dtype=torch.float32)
    assert torch.allclose(ref.q, q0)


def test_every_joint_name_in_the_recording_is_this_robots() -> None:
    """Every joint name in the npz is one of this robot's, including where nothing reads it.

    `joint_order` fails loudly when stale -- the loader refuses a missing joint.
    `meta["stand_pose"]` does not: no code reads it, so a relabelling that reached
    one and missed the other would leave the file naming joints no model has while
    every load succeeded. The next reference taken from the generator arrives on
    the pre-V1.6.1 names, so this is the relabelling most likely to be half done.
    """
    meta = json.loads(str(np.load(JUMP_REF_NPZ, allow_pickle=True)["meta"]))
    assert sorted(meta["joint_order"]) == sorted(JOINT_NAMES), meta["joint_order"]
    assert sorted(meta["stand_pose"]) == sorted(JOINT_NAMES), list(meta["stand_pose"])


def test_sample_at_exact_frames_is_interpolation_free(ref: JumpReference) -> None:
    """At integer frame indices, sample should return that frame exactly."""
    for k in (0, ref.go_frame, ref.n - 1):
        idx = torch.tensor([float(k)])
        s = ref.sample(idx)
        assert torch.allclose(s["q"][0], ref.q[k])
        assert torch.allclose(s["foot_b"][0], ref.foot_b[k])
        assert torch.equal(s["contact"][0], ref.contact[k])


def test_time_semantics_before_go_clamps_to_stance(ref: JumpReference) -> None:
    """The waiting period (t_since_go < 0) clamps to the go frame — the stance pose."""
    tsg = torch.tensor([-0.5, 0.0])
    idx = ref.index_of(tsg)
    assert idx[0] == float(ref.go_frame)
    assert ref.phase(tsg)[0] == 0.0
    assert ref.phase(tsg)[1] == 0.0


def test_phase_one_at_end(ref: JumpReference) -> None:
    tsg = torch.tensor([ref.span + 5.0])
    assert ref.phase(tsg)[0] == 1.0
    idx = ref.index_of(tsg)
    assert idx[0] == float(ref.n - 1)


def test_foot_order_matches_legs(ref: JumpReference) -> None:
    """The column order of `foot_b` is `meta["ik_ee"]`, one leg each, in LEGS order.

    Checked by **leg prefix**, which is the part that has to hold: column 3 being
    `LM`'s is what makes `foot_b[:, 2]` the left middle foot, and a reordered
    recording would put the imitation reward on the wrong leg with nothing saying
    so.

    Not checked by body name, and that is deliberate. The recording's joint names
    were relabelled to this robot's, because every joint has exactly one
    successor; its bodies were not, so `ik_ee` still says `LF_F_Link` and
    `LM_toe_tip_link` -- the bodies that era's model measured feet at.
    `tasks/jumper/common/constants.py::FEET` now names `LF_palm_pad_b_link` and
    `LM_foot_tip_link`, and asserting those two lists are the same list would claim
    an identity nobody has established: the front foot went `LF_F_Link` ->
    `LF_cehou_tip_link` -> `LF_palm_pad_b_link` across three exports, and measured
    against today's model these feet sit 8 to 17 mm apart in the base frame. See
    `reference.py::JUMP_REF_NPZ` on what that does and does not invalidate.
    """
    ik_ee = ref.meta["ik_ee"]
    assert len(ik_ee) == len(LEG_ORDER)
    assert [name[:2] for name in ik_ee] == LEG_ORDER
    # One body per leg, so no leg is measured twice.
    assert len(set(ik_ee)) == len(ik_ee), ik_ee


def test_midpoint_interpolation(ref: JumpReference) -> None:
    """A fractional index k+0.5 should equal the mean of the two neighboring frames."""
    k = ref.go_frame
    s = ref.sample(torch.tensor([k + 0.5]))
    expected = 0.5 * (ref.q[k] + ref.q[k + 1])
    assert torch.allclose(s["q"][0], expected, atol=1e-6)


def test_a_reference_term_is_admitted_only_by_a_contract_that_carries_the_recording() -> None:
    """The gate must read the `reference` block, not just the term's name.

    The silent failure this pins, which a name-only gate had already let
    through once: `jumper.dance` and `jumper.jump` both have a term called
    `ref_future`, and they are not the same term. Dance indexes its preview in
    frames of the clip; jump indexes it in seconds off a 250 Hz table. A board
    that read one as the other would preview fifty times too far ahead -- a
    policy shown a motion it is not about to make, at exactly the right vector
    width, with nothing raised.

    So a name in `_DEPLOY_TERMS` is necessary and the block is what makes it
    sufficient. Both directions are asserted: without the block the export is
    refused, with it the same contract passes.
    """
    import importlib.util
    import sys
    from pathlib import Path

    # `export.py` does `from _cli import ...`, its sibling. Registering `_cli`
    # under that name first is how `tests/test_dotenv.py` reaches the same
    # directory -- rewriting `sys.path` is what this repository does not do, and
    # `test_layout.py::test_no_sys_path_mutation` enforces it.
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    cs = importlib.util.spec_from_file_location("_cli", scripts / "_cli.py")
    cli = importlib.util.module_from_spec(cs)
    sys.modules["_cli"] = cli
    cs.loader.exec_module(cli)
    spec = importlib.util.spec_from_file_location("_export_for_test", scripts / "export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def contract(**extra):
        c = {
            "obs_joint_order": ["a", "b"],
            "observation": {"terms": [{"name": "ref_future", "dim": 4}]},
        }
        c.update(extra)
        return c

    block = {
        "lookahead_s": [0.01, 0.02],
        "file": "t.trajectory.json",
        "baseline": "q_cmd",
    }

    with pytest.raises(SystemExit) as e:
        mod._validate_deployable(contract())
    assert "ref_future" in str(e.value), "the term a dance would carry must be named"

    # The control: the identical contract, with the recording declared.
    mod._validate_deployable(contract(reference=block))

    # And a width that disagrees with the block is refused, because the
    # controller builds the block's version and the two would differ only in
    # what the robot did.
    wrong = contract(reference=block)
    wrong["observation"]["terms"][0]["dim"] = 6
    with pytest.raises(SystemExit) as e:
        mod._validate_deployable(wrong)
    assert "4" in str(e.value) and "6" in str(e.value), str(e.value)
