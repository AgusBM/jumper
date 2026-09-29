"""`play.py --measure`: the foot-force channel, and the two ways it goes quiet.

The monitor writes a csv nobody reads until something looks wrong, and a plot
somebody glances at. Both fail silently in the same way -- a column of zeros, or
a column that is not the quantity its header names -- so what is pinned here is
which sensor the force comes from and that the csv's header still describes its
rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the monitor is torch code")

from mjrl.viewer.monitor import _find_foot_sensor  # noqa: E402


class _Match:
    def __init__(self, pattern):
        self.pattern = pattern


class _Cfg:
    def __init__(self, name, fields, air, pattern):
        self.name = name
        self.fields = fields
        self.track_air_time = air
        self.primary = _Match(pattern)


class _Sensor:
    def __init__(self, cfg, count=6):
        self.cfg = cfg
        self.data = type("D", (), {"force": torch.zeros(2, count, 3)})()


class _Scene:
    def __init__(self, sensors):
        self.sensors = sensors


class _Env:
    def __init__(self, sensors):
        self.scene = _Scene(sensors)


def test_the_force_comes_from_the_sensor_that_tracks_air_time() -> None:
    """Not from the first contact sensor, and not from one matched by name.

    A scene has more than one contact sensor and only one of them is about
    footfalls. `self_collision` on this robot requests `force` exactly like the
    foot sensor does, so "the first sensor with a force field" picks the wrong one
    -- and the failure is a panel of near-zeros that looks like a robot with very
    soft feet.

    `track_air_time` is the discriminator because it is what the *author* of the
    sensor set when they meant footfalls, and because it is what
    `compute_first_contact` needs, so a sensor that qualifies is exactly a sensor
    the landing impact can be read from.
    """
    feet = _Sensor(_Cfg("feet_ground_contact", ("found", "force"), True, ("LF", "RF")))
    collision = _Sensor(_Cfg("self_collision", ("found", "force"), False, "base_link"))

    # Declared first, so an implementation that takes the first force sensor picks
    # it -- which is the mistake this is here to catch.
    sensor, names = _find_foot_sensor(_Env({"self_collision": collision, "feet": feet}))
    assert sensor is feet, "the collision sensor was taken for the feet"
    assert names == ("LF", "RF"), "the names did not come from the primary pattern"

    # No qualifying sensor is not an error: the monitor drops the channel.
    sensor, names = _find_foot_sensor(_Env({"self_collision": collision}))
    assert sensor is None and names == ()

    # A regex primary cannot be turned into names, so it gets indices rather than
    # a guess. A wrong name is worse than a number.
    regex = _Sensor(_Cfg("feet", ("found", "force"), True, ".*_toe"))
    sensor, names = _find_foot_sensor(_Env({"feet": regex}))
    assert sensor is regex and names == tuple(f"feet[{i}]" for i in range(6))


def test_the_csv_header_still_describes_its_rows(tmp_path: Path) -> None:
    """A header one column short shifts every reading after it, silently.

    The force columns were appended to both the header and the row, in two
    different places. That is the shape of edit where they come apart: the file
    still parses, every column still holds plausible numbers, and `tau/LF_hip` is
    now somebody's foot.
    """
    from mjrl.viewer.monitor import JointMonitor

    joints = ("a", "b", "c")
    actuators = ("a", "b")
    feet = ("LF", "RF")

    class _Entity:
        joint_names = joints
        actuator_names = actuators

        class data:
            joint_pos = torch.zeros(1, len(joints))
            joint_vel = torch.zeros(1, len(joints))
            actuator_force = torch.zeros(1, len(actuators))

    class _Feet:
        cfg = _Cfg("feet", ("found", "force"), True, feet)
        data = type("D", (), {"force": torch.ones(1, len(feet), 3)})()

        @staticmethod
        def compute_first_contact(dt):
            del dt
            return torch.ones(1, len(feet), dtype=torch.bool)

    monitor = JointMonitor(
        _Entity(), 0, 0.01, tmp_path, sensor=_Feet(), foot_names=feet
    )
    # The window may or may not have opened depending on the machine; the csv is
    # what this is about.
    monitor._fig = None
    for _ in range(4):
        monitor.update()
    monitor.close()

    rows = (tmp_path / "measure.csv").read_text().splitlines()
    header = rows[0].split(",")
    assert header[-len(feet):] == [f"force/{n}" for n in feet]
    for row in rows[1:]:
        assert len(row.split(",")) == len(header), (
            "a data row is a different width than the header, so every column "
            "after the shortest one is mislabelled"
        )
    assert len(rows) - 1 == 4

    # And the landing impacts were counted rather than the force sampled: every
    # step reported a first contact on both feet.
    assert len(monitor._impacts) == 4 * len(feet)


def test_the_posture_command_says_where_its_height_is_centred() -> None:
    """The contract has to carry the offset, not just the companion markdown.

    `posture_command` centres channel 3 on the neutral height, because a raw
    0.07-0.15 m reading is one near-constant input beside three signed ones. That
    subtraction used to read `term.cfg.neutral_height` -- across into the command
    manager's config, where `export.py` cannot see it -- so `layout.json`
    described a four-channel term and said nothing about the offset in one of
    them. A builder working from the contract alone fed the raw height and was a
    constant 0.107 m out, for ever, on an input whose whole range is 0.08 m.

    It was found by `--bundle`: eight terms matched the simulator's own
    observation to the last bit and this one was out by exactly `STAND_Z`.

    The bundle player refuses a contract without it rather than assuming a
    default, which is the same choice: a wrong number that looks right is worse
    than a stop.
    """
    import tasks
    from tasks.jumper.common.constants import STAND_Z

    cfg = tasks.load_env_cfg("jumper.posture")
    for group in ("actor", "critic"):
        params = cfg.observations[group].terms["posture_command"].params
        assert params.get("neutral_height") == pytest.approx(STAND_Z), (
            f"{group}'s posture_command does not carry neutral_height in its "
            f"params, so `export.py` cannot write it into the contract"
        )

    # `export.py` keeps scalar params, which is what puts it in layout.json.
    assert isinstance(params["neutral_height"], float)


def test_a_bundle_is_a_directory_checkpoint_can_be_pointed_at(tmp_path) -> None:
    """The copied checkpoint keeps its own name, and that is the point of it.

    `export.py` copies the checkpoint into the bundle so the artifact carries its
    own weights -- otherwise the two things anyone does with a bundle later, a
    comparison against the policy it claims to be and a re-export after the
    contract gains a field, both need a path that a month of training has since
    deleted or moved.

    Keeping `model_<N>.pt` rather than renaming it to something fixed means the
    repository's existing resolver finds it, so the bundle directory works
    everywhere a run directory does -- `play --checkpoint <bundle>` and
    `train --resume --checkpoint <bundle>`. A fixed name would have needed every
    one of those to learn a second convention.
    """
    from mjrl.checkpoint import find_checkpoint

    bundle = tmp_path / "model_38000"
    bundle.mkdir()
    (bundle / "layout.json").write_text("{}")
    (bundle / "actor.onnx").write_bytes(b"")
    (bundle / "model_38000.pt").write_bytes(b"")

    assert find_checkpoint(str(bundle), tmp_path) == bundle / "model_38000.pt"

    # And a bundle without it is a bundle from before this, which has to fail
    # loudly rather than resolve to something else in the tree.
    bare = tmp_path / "model_1"
    bare.mkdir()
    (bare / "actor.onnx").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        find_checkpoint(str(bare), tmp_path)
