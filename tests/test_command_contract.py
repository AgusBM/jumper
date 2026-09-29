"""The command range a policy is trained on, and the one it is replayed on.

They have to be the same number, and for a while they were not. The curriculum's
angular levels were extended to 0.75 while `velocity_env.py` still wrote 0.5 by
hand, and nothing failed: training used the curriculum's value, `play` builds with
`curriculum = {}` and used the config's, so a pad could ask for two thirds of what
the policy had been trained to do and the replay under-reported the robot.

That is the shape of failure this suite exists for -- no exception, no warning, a
number quietly meaning two things. These tests pin the agreement rather than the
values, so raising a level stays a one-line change.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab", reason="the env config needs mjlab installed")

from tasks.jumper.common.mdp.curriculum import (  # noqa: E402
    ANG_LEVELS,
    LEVELS,
    ang_std,
    lin_std,
)
from tasks.registry import load_env_cfg  # noqa: E402

JUMPER_TASKS = ("jumper.flat", "jumper.tripod", "jumper.tetrapod", "jumper.ripple")


@pytest.mark.parametrize("task", JUMPER_TASKS)
def test_the_config_range_is_the_curriculum_s_last_level(task: str) -> None:
    """What `play` commands is what training finished on.

    `play` drops the curriculum, so the config's own ranges are what a replay --
    and therefore the keyboard and the pad, which clamp to them -- can ask for.
    If those are below the curriculum's endpoint, every replay is quietly capped
    below the policy's actual training envelope.

    Read from **the task's own** curriculum rather than from the module
    constants. A task may narrow its levels -- `jumper.ripple` does, because no
    admissible clock frequency reaches the shared 0.50 m/s -- and the property
    being pinned here is "play asks for what training finished on", which is per
    task. Asserting the shared endpoint instead would have made a task that
    correctly lowered both its range and its levels look broken.
    """
    # The levels come from the **training** config: `play` drops the curriculum
    # entirely, which is the whole reason this test exists.
    train = load_env_cfg(task, play=False)
    params = getattr(train.curriculum.get("command"), "params", None) or {}
    lin = params.get("levels", LEVELS)
    ang = params.get("ang_levels", ANG_LEVELS)
    ranges = load_env_cfg(task, play=True).commands["twist"].ranges
    assert ranges.lin_vel_x == (-lin[-1], lin[-1])
    assert ranges.lin_vel_y == (-lin[-1], lin[-1])
    assert ranges.ang_vel_z == (-ang[-1], ang[-1])


def _ladder(task: str) -> tuple[tuple, tuple, tuple | None, tuple | None]:
    """A task's own `(levels, ang_levels, lin_scales, ang_scales)`."""
    params = getattr(load_env_cfg(task).curriculum.get("command"), "params", None) or {}
    return (
        params.get("levels", LEVELS),
        params.get("ang_levels", ANG_LEVELS),
        params.get("lin_std_scales"),
        params.get("ang_std_scales"),
    )


@pytest.mark.parametrize("task", JUMPER_TASKS)
def test_the_config_std_is_the_curriculum_s_last_level(task: str) -> None:
    """What `play` marks with is what training finished on.

    The sibling of the range test above and the same failure: `play` drops the
    curriculum, so the config's std is what a replay scores against. If it is not
    the std the last rung sets, every replay is marked on a different ruler than
    the policy was trained on -- and being a reward, nothing about the replay looks
    wrong, the numbers are simply not comparable to training's.

    Read from **the task's own** ladder, because a task may append rungs:
    `jumper.flat` a wider one, `jumper.tripod` a precision rung that repeats the top
    range with a tighter std. Asserting `ratio * range` instead would pin the
    widening rule and call the precision rung a bug, which it is not -- but see
    the next test, which is what stops that becoming a licence to tighten anywhere.
    """
    cfg = load_env_cfg(task)
    lin_levels, ang_levels, lin_scales, ang_scales = _ladder(task)
    lin = cfg.rewards["track_linear_velocity"].params["std"]
    ang = cfg.rewards["track_angular_velocity"].params["std"]
    assert lin == pytest.approx(lin_std(-1, lin_levels, lin_scales))
    assert ang == pytest.approx(ang_std(-1, ang_levels, ang_scales))


@pytest.mark.parametrize("task", JUMPER_TASKS)
def test_only_a_rung_that_widens_nothing_may_tighten_the_ruler(task: str) -> None:
    """The ruler may change only where the range does not.

    This is the whole invariance argument, as an assertion. Measured, with std held
    fixed a robot that never moves collects 1.35 of the tracking terms' 4.0 at range
    0.5 and 3.11 at 0.15 -- so a rung that widens the range *and* moves the std is
    two difficulties at once and a policy that fails it cannot tell you which. A
    rung that repeats the range has only the one variable, which is what makes
    `jumper.tripod`'s precision rung legitimate and would make the same tightening
    one rung earlier a mistake.
    """
    lin_levels, ang_levels, lin_scales, ang_scales = _ladder(task)
    for axis, levels, scales in (
        ("linear", lin_levels, lin_scales),
        ("angular", ang_levels, ang_scales),
    ):
        if scales is None:
            continue
        for i in range(1, len(levels)):
            if scales[i] != scales[i - 1]:
                assert levels[i] == pytest.approx(levels[i - 1]), (
                    f"{task} {axis} rung {i} changes both the range "
                    f"({levels[i - 1]} -> {levels[i]}) and the ruler "
                    f"({scales[i - 1]} -> {scales[i]})"
                )


def test_the_ruler_check_rejects_a_rung_that_moves_both() -> None:
    """The control group: the loop above can fail.

    Written because the version of it that ran over every task passed on the day it
    was added -- every task's scales were `None` or flat except one, so an assertion
    that never executed looked exactly like an assertion that held.
    """
    levels, scales = (0.25, 0.35, 0.50), (1.0, 1.0, 0.5)
    offenders = [
        i
        for i in range(1, len(levels))
        if scales[i] != scales[i - 1] and levels[i] != levels[i - 1]
    ]
    assert offenders == [2], "a rung that widens and tightens at once must be caught"


def test_training_and_replay_agree_on_the_range() -> None:
    """`play=True` changes noise, episode length and the command *term*; it must
    not change what can be commanded."""
    train = load_env_cfg("jumper.tripod").commands["twist"].ranges
    replay = load_env_cfg("jumper.tripod", play=True).commands["twist"].ranges
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        assert getattr(train, axis) == getattr(replay, axis), axis


def test_the_operator_can_ask_for_the_whole_trained_envelope() -> None:
    """Full stick reaches the edge of the range, on every axis.

    The pad clamps to the trained range on purpose -- commanding off-distribution
    tells you nothing about the policy. The other half of that bargain is that it
    can reach the whole of it, which is what broke when the two numbers drifted.
    """
    from tasks.jumper.common.mdp.operator import deflections, scale, spans

    twist = load_env_cfg("jumper.tripod", play=True).commands["twist"]
    ranges = twist.ranges
    # The stick signs come from the task's controls.yaml; the command config
    # carries the parsed file, so this asks the object play uses.
    controls = twist.controls
    edges = spans(controls.command("twist"), twist)
    stick = {"Lx": -1.0, "Ly": -1.0, "Rx": -1.0, "Ry": 0.0, "LT": 0.0, "RT": 0.0}
    x = deflections(stick, False, controls)
    vx, vy, wz = (scale(x[a], edges[a]) for a in ("lin_vel_x", "lin_vel_y", "ang_vel_z"))
    assert vx == pytest.approx(ranges.lin_vel_x[1])
    assert vy == pytest.approx(ranges.lin_vel_y[1])
    assert wz == pytest.approx(ranges.ang_vel_z[1])
    assert wz == pytest.approx(ANG_LEVELS[-1]), "full stick must reach the top level"
