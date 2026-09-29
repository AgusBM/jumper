"""The curriculum terms, where they read a number from the wrong place.

Both failures pinned here are ones nothing reports. A terrain curriculum that
starts climbing a rung early logs `commands_maxed` as 1.0 and trains on; a
constructor that honours a param `__call__` cannot take works in every unit test
of the constructor and raises only in the one run that sets it.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="the curriculum terms are torch code")
pytest.importorskip("mjlab", reason="building the configs needs mjlab")

import tasks
from tasks.jumper.common.mdp.curriculum import (
    LEVELS,
    CommandRangeCurriculum,
    TerrainLevels,
)
from tasks.jumper.five_foot.mdp.curriculum import PayloadCurriculum
from tasks.jumper.posture.mdp.curriculum import PostureRangeCurriculum

#: Every task that climbs a command ladder. Each one appends a rung to the shared
#: shape (`ladder()` adds a precision rung; `jumper.posture` climbs six), so every
#: one of them tops out above `len(LEVELS) - 1`.
LADDERED = (
    "jumper.flat",
    "jumper.tripod",
    "jumper.ripple",
    "jumper.tetrapod",
    "jumper.five_foot",
    "jumper.posture",
)


def _command_term(task: str | None):
    if task is None:
        # A term that passes no `levels` of its own climbs the shared ladder.
        return SimpleNamespace(params={"command_name": "twist"})
    return tasks.load_env_cfg(task).curriculum["command"]


def _terrain_decision(command_term, level: int) -> dict:
    """`TerrainLevels`' output for one reset batch with the commands at `level`.

    Every environment tracked perfectly, ran a full episode and has served its
    dwell, so the command curriculum is the only thing left that can hold the
    terrain back.
    """
    n = 8
    # What the manager does to a class term: `func` becomes the instance, and the
    # params stay on the cfg beside it.
    command = SimpleNamespace(
        func=CommandRangeCurriculum(command_term, None), params=command_term.params
    )
    command.func.level = level

    class _Terrain:
        cfg = SimpleNamespace(terrain_generator=object())
        terrain_levels = torch.zeros(n, dtype=torch.long)

        def update_env_origins(self, env_ids, move_up, move_down) -> None:
            del env_ids, move_up, move_down

    twist = SimpleNamespace(
        cfg=SimpleNamespace(
            ranges=SimpleNamespace(lin_vel_x=(-0.5, 0.5), ang_vel_z=(-0.75, 0.75)),
            resampling_time_range=(3.0, 8.0),
        ),
        metrics={"error_vel_xy": torch.zeros(n), "error_vel_yaw": torch.zeros(n)},
    )
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        step_dt=0.02,
        common_step_counter=1000,
        episode_length_buf=torch.full((n,), 1000),
        scene=SimpleNamespace(terrain=_Terrain()),
        command_manager=SimpleNamespace(get_term=lambda name: twist),
        curriculum_manager=SimpleNamespace(_term_cfgs=[command]),
    )
    return TerrainLevels(None, env)(
        env, torch.arange(n), command_name="twist", dwell_episodes=1
    )


@pytest.mark.parametrize("task", [*LADDERED, pytest.param(None, id="shared-ladder")])
def test_terrain_waits_for_the_last_rung_of_the_tasks_own_ladder(
    task: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terrain is held until the command curriculum is on **its own** top rung.

    **This pins two curricula climbing together**, which `TerrainLevels` exists
    to prevent. It judged "the commands are maxed" against the shared `LEVELS`,
    whose top is 2, while every task that climbs a ladder passes a longer one --
    so on rough ground the terrain started moving at level 2, with the precision
    rung (or, for `jumper.posture`, three rungs) still to come. `commands_maxed`
    logged 1.0 and nothing else changed.

    The control group is the top rung itself: there, every environment in the
    stub promotes, so a refusal below it is the gate and not a stub that cannot
    promote at all. The shared-ladder case keeps the fallback honest for a term
    that passes no `levels`.
    """
    monkeypatch.delenv("MJRL_COMMAND_LEVEL", raising=False)
    term = _command_term(task)
    top = len(term.params.get("levels", LEVELS)) - 1

    at_top = _terrain_decision(term, top)
    assert at_top["commands_maxed"] == 1.0 and at_top["promoted"] == 1.0, (
        f"{task}: nothing promoted with the commands at their top rung {top}, so "
        f"the stub cannot promote and the refusals below would mean nothing"
    )

    for level in range(top):
        out = _terrain_decision(term, level)
        assert out["commands_maxed"] == 0.0 and out["promoted"] == 0.0, (
            f"{task}: terrain promoted with the command curriculum at level "
            f"{level} of {top} -- the top was read from somewhere other than this "
            f"task's own ladder"
        )


class _Reads(dict):
    """A params dict that remembers which keys were asked for."""

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.asked: set[str] = set()

    def get(self, key, default=None):
        self.asked.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.asked.add(key)
        return super().__getitem__(key)

    def __contains__(self, key) -> bool:
        self.asked.add(key)
        return super().__contains__(key)


@pytest.mark.parametrize(
    ("task", "term_name", "cls", "always_read"),
    [
        ("jumper.tripod", "command", CommandRangeCurriculum, "levels"),
        ("jumper.posture", "posture", PostureRangeCurriculum, "angles"),
        ("jumper.five_foot", "payload", PayloadCurriculum, "levels"),
    ],
)
def test_every_param_the_constructor_reads_is_one_the_call_accepts(
    task: str,
    term_name: str,
    cls: type,
    always_read: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A class term's params reach it twice, and both doors have to open.

    mjlab passes `cfg` to the constructor once and then splats `cfg.params` into
    every call as keywords (`CurriculumManager.compute`), with no check between
    the two. So a key the constructor honours and `__call__` does not accept is a
    knob that constructs fine and raises TypeError at the first reset -- and since
    no configuration sets it, nothing ever runs the path that would say so.
    The command and posture ladders both read `start_level` exactly that way,
    until this test; the payload ladder was written without it.

    The control group is `always_read`, a key each constructor cannot work
    without: if the recorder missed it, an empty result below would only mean the
    recorder sees nothing.
    """
    monkeypatch.delenv("MJRL_COMMAND_LEVEL", raising=False)
    monkeypatch.delenv("MJRL_POSTURE_LEVEL", raising=False)
    monkeypatch.delenv("MJRL_PAYLOAD_LEVEL", raising=False)
    params = _Reads(tasks.load_env_cfg(task).curriculum[term_name].params)
    cls(SimpleNamespace(params=params), None)

    assert always_read in params.asked, (
        f"the recorder did not see {cls.__name__} read {always_read!r}, so it is "
        f"not seeing the constructor's reads at all"
    )
    call = inspect.signature(cls.__call__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in call.values()):
        return
    refused = sorted(params.asked - call.keys())
    assert not refused, (
        f"{cls.__name__}.__init__ reads {refused} from params, which "
        f"__call__ does not accept: setting one raises TypeError on the first reset"
    )
