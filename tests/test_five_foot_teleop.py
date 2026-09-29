"""`jumper.five_foot`'s walk in replay: the family's operator, from this task's file.

This task used to drive its walk from a term of its own -- the arrows and `A` /
`D` / `R`, a command that lapsed 0.6 s after the last key -- because the family's
keyboard was the numeric keypad then, which this task's attitude command owned.
The family's keyboard is a path of its own on letters now
(`tasks/jumper/common/mdp/operator.py`; a copy of the pad until 2026-09-29), and
the attitude has moved onto the same operator, laid out as jumper.posture's: the
walk on the left stick, the twist and then the turn on the right stick's two
halves, and on the keyboard W A S D and J L. So the keyboard in replay is the one
the exported contract describes, for both commands.

What is pinned is what a replay gets wrong without a word:

1. **The walk read from some other file.** A sign is the one value whose wrong
   answer looks exactly like the right one: the robot walks backwards and every
   screen is normal.
2. **Keys that reach nothing.** A term built and never driven is a replay that
   looks like a policy standing still. The presses go through the operator the
   replay installs, into this task's own ranges, and come back out as numbers.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("torch", reason="the task config imports the command terms")

from tasks.jumper.common.mdp.controls import load_controls
from tasks.jumper.common.mdp.operator import Operator, OperatorVelocityCommandCfg, spans

CONTROLS = pathlib.Path(__file__).parents[1] / "tasks" / "jumper" / "five_foot" / "controls.yaml"


def _twist():
    import tasks

    return tasks.load_env_cfg("jumper.five_foot", play=True).commands["twist"]


class _Clock:
    """Seconds, moved by the test: what a key's hold is measured by."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _key(op: Operator, key: str, down: bool) -> None:
    """Put `key` down, or let it up, by the dictionary's name."""
    assert op.key(key, down) or not down, f"{key} is not a key this file reads"


def test_replay_drives_the_walk_from_this_task_s_own_file() -> None:
    twist = _twist()
    assert isinstance(twist, OperatorVelocityCommandCfg), (
        f"replay drives the walk with {type(twist).__name__}, not the family's operator"
    )
    assert twist.controls.source_sha256 == load_controls(CONTROLS).source_sha256, (
        f"replay reads {twist.controls.source}, not this task's controls.yaml"
    )
    walk = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    pad = {
        axis: [(b.source, b.sign, tuple(b.travel)) for b in twist.controls.bindings[axis]]
        for axis in walk
    }
    # Pushing a stick forward is `Ly = -1` in evdev's convention, and forward
    # is +x in the body frame, so the sign is -1; the same holds for left and
    # for a counter-clockwise turn. The turn is the right stick's second half:
    # its first half twists the body, laid out as jumper.posture lays it out.
    assert pad == {
        "lin_vel_x": [("Ly", -1.0, (0.0, 1.0))],
        "lin_vel_y": [("Lx", -1.0, (0.0, 1.0))],
        "ang_vel_z": [("Rx", -1.0, (0.5, 1.0))],
    }, pad


def test_two_seconds_held_are_the_edge_of_this_task_s_range_and_letting_go_stops_it() -> None:
    import tasks

    replay = tasks.load_env_cfg("jumper.five_foot", play=True)
    twist = replay.commands["twist"]
    controls = load_controls(CONTROLS)
    clock = _Clock()
    op = Operator(controls, listen=False, clock=clock)
    op.attach("twist", spans(controls.command("twist"), twist))
    # The file describes the attitude too, and the operator refuses to run while
    # a command it describes has no term -- so it is attached, at its own ranges.
    op.attach("body_pose", spans(controls.command("body_pose"), replay.commands["body_pose"]))
    lo, hi = twist.ranges.ang_vel_z

    stamp = 0

    def command() -> list[float]:
        nonlocal stamp
        stamp += 1
        values = op.command("twist", stamp)
        assert values is not None, "a key was pressed and the operator did not take over"
        return values

    # `L` turns clockwise, the negative end, and only turns: on the keyboard the
    # twist is a keystroke of its own (none, in this task), not the first half of
    # a stick.
    _key(op, "key_l", True)
    clock.t = 1.0
    assert command()[2] == pytest.approx(0.5 * lo, rel=1e-6), "a second is half of L's travel"
    clock.t = 1.8
    short = command()[2]
    clock.t = 2.0
    full = command()[2]
    # The control group is 1.8 s: short of the edge by a tenth of the hold, so
    # two seconds landing on it is the hold and not a clamp that was always there.
    assert short == pytest.approx(0.9 * lo, rel=1e-6), (short, lo)
    assert full == pytest.approx(lo, rel=1e-6), (full, lo)
    clock.t = 3.5
    assert command()[2] == pytest.approx(lo, rel=1e-6), "held past two seconds, it left the range"
    _key(op, "key_l", False)
    assert command()[2] == 0.0, "let go, the turn did not stop"

    _key(op, "key_w", True)
    clock.t = 4.5
    walking = command()
    assert walking[0] == pytest.approx(0.5 * twist.ranges.lin_vel_x[1], rel=1e-6), walking

    _key(op, "key_w", False)
    assert command() == [0.0, 0.0, 0.0], "let go, the robot kept walking"
    assert hi > 0 > lo, "the yaw range does not straddle zero; the test reads the wrong end"
