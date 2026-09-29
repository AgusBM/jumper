"""The mirror transform, and what history does to it.

The observation groups carry five frames each, flattened into one row. That makes
a term's slice five consecutive copies of itself, and a mirror written for a
single frame -- which is what this was, before history existed -- goes wrong in
the one way that leaves the shapes intact: the joint permutation shuffles values
*across* frames, and the mirrored sample becomes a robot whose left legs are
several control steps out of step with its right ones.

Nothing raises. The batch is the right size, the loss is finite, and the mirror
loss simply teaches the policy something untrue. So the tests below are aimed at
that specific corruption rather than at "does mirroring run".
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the mirror transform is torch")

from tasks.jumper.common.constants import GAIT_JOINTS, JUMPER_XML
from tasks.jumper.common.mdp.symmetry import (
    JOINT_PERM,
    JOINT_SIGN,
    _mirror_group,
    _mirror_slice,
)

NJ = len(GAIT_JOINTS)


def test_the_single_frame_joint_mirror_is_an_involution() -> None:
    """Mirroring twice is the identity -- necessary, and famously not sufficient:
    a mirror that shuffled values between frames would pass this too. It is here
    as the floor, with the frame tests below as the actual check."""
    x = torch.randn(4, NJ)
    assert torch.allclose(_mirror_slice(_mirror_slice(x, "joint"), "joint"), x, atol=1e-6)


def test_the_joint_mirror_swaps_left_and_right() -> None:
    """The permutation is the left-right pairing, and the signs come from the joint
    axes **measured on the model**.

    This test used to write the axis rule out as a name test -- `1.0 if
    "shoulder_pitch" in name else -1.0` -- which is the same sentence
    `symmetry.py` had, so the two agreed and the pair of them was one claim, not a
    check. The V1.6 rename deleted the name both of them matched on. Every joint
    then fell through to -1.0, the front arms' shoulder pitch was negated along
    with the rest, and this test passed: it had stopped matching in exactly the way
    the code had.

    So the expected sign is read off `jnt_axis` instead. Under the x-z mirror
    M = diag(1, -1, 1) a joint rotation is a pseudovector, w -> -M.w, so a y axis
    keeps its sign and x or z negates. A rename cannot reach that.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(JUMPER_XML))
    for i, name in enumerate(GAIT_JOINTS):
        partner = GAIT_JOINTS[JOINT_PERM[i]]
        assert partner[:2] != name[:2], f"{name} mirrors onto its own leg"
        assert partner[2:] == name[2:], f"{name} mirrors onto a different joint"

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"{name} is not in the model"
        axis = model.jnt_axis[jid]
        dominant = max(range(3), key=lambda k: abs(axis[k]))
        # A blended axis would make "is it y" a matter of degree, and the sign a
        # judgement call. Every joint on this robot is cardinal to 1e-9.
        assert abs(axis[dominant]) > 1 - 1e-9, f"{name} axis {axis} is not cardinal"
        expected = 1.0 if dominant == 1 else -1.0
        assert JOINT_SIGN[i] == expected, (
            f"{name} has a {'xyz'[dominant]} axis, so it should mirror with sign "
            f"{expected:+.0f}, but the table says {JOINT_SIGN[i]:+.0f}"
        )

    # The rule is not uniform, which is the whole reason it needs a table: J1 is a
    # y axis on the front arms and an x axis on the walking legs, so a test that
    # only ever saw one sign would pass against a table of all -1.0.
    assert set(JOINT_SIGN) == {1.0, -1.0}, JOINT_SIGN
    assert sum(1 for v in JOINT_SIGN if v == 1.0) == 2, (
        "exactly LF_J1_joint and RF_J1_joint keep their sign"
    )


FRAMES = 5


class _StubTermCfg:
    """What `_history_length` reads off a resolved term."""

    def __init__(self, history_length: int) -> None:
        self.history_length = history_length
        self.flatten_history_dim = True


class _StubObsManager:
    """Enough observation manager for `_mirror_group`, with history."""

    def __init__(self, terms: list[tuple[str, int]], frames: int) -> None:
        self.active_terms = {"actor": [n for n, _ in terms]}
        self.group_obs_term_dim = {"actor": [(d * frames,) for _, d in terms]}
        self._group_obs_term_cfgs = {"actor": [_StubTermCfg(frames) for _ in terms]}


class _StubEnv:
    def __init__(self, om: _StubObsManager) -> None:
        self.observation_manager = om


def _group(terms, frames):
    return _StubEnv(_StubObsManager(terms, frames))


def test_history_is_mirrored_frame_by_frame() -> None:
    """Five frames mirrored must equal each of the five mirrored on its own.

    Drives the real `_mirror_group` -- the earlier version of this test compared a
    reference against itself, which proves nothing about the code that ships.
    """
    env = _group([("joint_pos", NJ)], FRAMES)
    x = torch.randn(3, FRAMES * NJ)
    got = _mirror_group(env, "actor", x)
    want = torch.cat(
        [_mirror_slice(x[:, f * NJ : (f + 1) * NJ], "joint") for f in range(FRAMES)],
        dim=1,
    )
    assert torch.allclose(got, want, atol=1e-6)


def test_a_value_never_moves_between_frames() -> None:
    """The decisive one: put a signal in a single frame and check it stays there.

    If the permutation is applied across the whole flattened slice, a value from
    frame 0 lands in another frame and this fails. With per-frame mirroring the
    other four stay **exactly** zero -- exactly, because the corruption is a
    reordering rather than a numerical error.

    Verified against a deliberately broken mirror before being trusted: replacing
    the reshape in `_mirror_group` with a single whole-slice `_mirror_slice` makes
    this fail and leaves every other test in this file passing.
    """
    env = _group([("joint_pos", NJ)], FRAMES)
    for live in range(FRAMES):
        x = torch.zeros(2, FRAMES * NJ)
        x[:, live * NJ : (live + 1) * NJ] = torch.randn(2, NJ) + 1.0
        out = _mirror_group(env, "actor", x)
        for f in range(FRAMES):
            block = out[:, f * NJ : (f + 1) * NJ]
            if f == live:
                assert block.abs().sum() > 0, "the live frame was emptied"
            else:
                assert torch.count_nonzero(block) == 0, (
                    f"frame {live} leaked into frame {f}: the mirror is being "
                    f"applied across the flattened history"
                )


def test_a_group_without_history_still_works() -> None:
    """frames = 1 must take the plain path, so the fix cannot regress the tasks
    that carry no history."""
    env = _group([("joint_pos", NJ), ("command", 3)], 1)
    x = torch.randn(2, NJ + 3)
    got = _mirror_group(env, "actor", x)
    assert torch.allclose(got[:, :NJ], _mirror_slice(x[:, :NJ], "joint"), atol=1e-6)
    assert torch.allclose(got[:, NJ:], _mirror_slice(x[:, NJ:], "twist"), atol=1e-6)


@pytest.mark.parametrize("kind", ["vec3", "pseudo3", "twist", "joint", "leg"])
def test_every_kind_is_its_own_inverse(kind: str) -> None:
    """A mirror is a reflection, so applying it twice must return the original --
    for every kind, not just the joint one. `phase_half_shift` is included
    implicitly: negating twice is the identity."""
    width = {"vec3": 3, "pseudo3": 3, "twist": 3, "joint": NJ, "leg": 6}[kind]
    x = torch.randn(4, width)
    assert torch.allclose(_mirror_slice(_mirror_slice(x, kind), kind), x, atol=1e-6)


# ── The height scan, whose mirror is derived rather than written ─────────


class _StubSensor:
    def __init__(self, offsets) -> None:
        self._local_offsets = offsets


class _StubScene:
    def __init__(self, offsets) -> None:
        self.sensors = {"terrain_scan": _StubSensor(offsets)}


class _StubScanEnv:
    def __init__(self, offsets) -> None:
        self.scene = _StubScene(offsets)


def _grid(nx: int, ny: int, res: float = 0.05) -> torch.Tensor:
    """The same order `GridPatternCfg` produces: meshgrid(x, y, indexing='xy')
    flattened, i.e. y major."""
    x = torch.arange(nx, dtype=torch.float32) * res - (nx - 1) * res / 2
    y = torch.arange(ny, dtype=torch.float32) * res - (ny - 1) * res / 2
    gx, gy = torch.meshgrid(x, y, indexing="xy")
    return torch.stack([gx.flatten(), gy.flatten(), torch.zeros(nx * ny)], dim=1)


def test_the_scan_mirror_is_read_off_the_rays() -> None:
    """Every ray must land on the one at mirrored y, with x untouched.

    Checked against the ray coordinates rather than against an index formula,
    because the formula is what the implementation deliberately avoids: a second
    copy of the pattern's ordering, correct until the pattern changes and silent
    afterwards.
    """
    from tasks.jumper.common.mdp.symmetry import _SCAN_PERM, _scan_perm

    _SCAN_PERM.clear()
    nx, ny = 13, 9
    offsets = _grid(nx, ny)
    perm = _scan_perm(_StubScanEnv(offsets), nx * ny)

    xy = offsets[:, :2]
    assert torch.allclose(xy[perm][:, 0], xy[:, 0], atol=1e-6), "x moved"
    assert torch.allclose(xy[perm][:, 1], -xy[:, 1], atol=1e-6), "y is not mirrored"
    assert torch.equal(perm[perm], torch.arange(nx * ny)), "not an involution"
    _SCAN_PERM.clear()


def test_a_scan_that_is_not_symmetric_raises() -> None:
    """A pattern with no mirror partner for some ray has no correct permutation,
    and the failure has to be loud: silently pairing a ray with the nearest wrong
    one gives a mirrored sample describing ground that is not there."""
    from tasks.jumper.common.mdp.symmetry import _SCAN_PERM, _scan_perm

    _SCAN_PERM.clear()
    offsets = _grid(4, 3)
    offsets[0, 1] += 0.031  # nudge one ray off its mirror partner
    with pytest.raises(ValueError, match="not symmetric about y"):
        _scan_perm(_StubScanEnv(offsets), offsets.shape[0])
    _SCAN_PERM.clear()


def test_a_missing_scan_sensor_raises() -> None:
    """`height_scan` in the observation with no sensor on the scene is a config
    error, not something to mirror as identity."""
    from tasks.jumper.common.mdp.symmetry import _SCAN_PERM, _scan_perm

    _SCAN_PERM.clear()

    class _Bare:
        pass

    with pytest.raises(AttributeError, match="terrain_scan"):
        _scan_perm(_Bare(), 117)
    _SCAN_PERM.clear()
