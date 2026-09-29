"""Curriculum levels across `train.py --resume`: the ways a resume restarts a ladder
without saying so.

A ladder earns its level (`CheckpointedLadder`); mjlab's runner checkpoints the
policy and the step counter and nothing else. A resume through it trains on and
logs plausible curves, from level 0, with the step counter already past every
dwell -- a policy that had outgrown the narrow ranges is trained back onto them.
Each test names the way it goes wrong:

1. **The level is not in the checkpoint**, or is in it and never reaches the
   managers: the term says level 2 while the command ranges and the reward std
   are still rung 0's, because the wrapper's first reset already applied those.
2. **A task climbs a ladder through a runner that does not save it.** The default
   runner is what `runner_cls()` returned for every task before this, and what a
   task copied from an older template still would.
3. **A level-keeping curriculum that is not a `CheckpointedLadder`**, which the
   runner cannot find and so silently leaves at its constructor's level.
4. **Evidence carried onto the wrong rung.** An error average measured against
   one range, restored beside a level `MJRL_*_LEVEL` moved, decides a promotion
   on a different test.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="the curriculum terms are torch code")
pytest.importorskip("mjlab", reason="building the configs needs mjlab")

import tasks
from tasks.jumper.common.mdp.curriculum import (
    CheckpointedLadder,
    CommandRangeCurriculum,
    TerrainLevels,
    lin_std,
)
from tasks.jumper.common.runner import CHECKPOINT_KEY, CurriculumRunner
from tasks.jumper.five_foot.mdp.curriculum import PayloadCurriculum
from tasks.jumper.posture.mdp.curriculum import PostureRangeCurriculum

LEVEL_ENVS = ("MJRL_COMMAND_LEVEL", "MJRL_POSTURE_LEVEL", "MJRL_PAYLOAD_LEVEL")

#: Curriculum classes that keep state and deliberately do not checkpoint it.
#: `TerrainLevels`' state is one terrain row per environment, and the environment
#: count may differ on a resume; drawn afresh, as on any start (decided 2026-09-28).
NOT_CHECKPOINTED = {TerrainLevels}


@pytest.fixture(autouse=True)
def _no_level_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shell's own `MJRL_*_LEVEL` is not this test's."""
    for name in LEVEL_ENVS:
        monkeypatch.delenv(name, raising=False)


def _command(levels=(0.25, 0.35, 0.50, 0.50)) -> CommandRangeCurriculum:
    return CommandRangeCurriculum(SimpleNamespace(params={"levels": levels}), None)


# ── Which source wins (failure 4) ─────────────────────────────────────────


def test_the_checkpoint_restores_the_level_and_its_evidence() -> None:
    saved = _command()
    saved.level, saved.err_ema, saved.last_promotion_step = 2, 0.0421, 1234
    state = saved.state_dict()

    resumed = _command()
    assert resumed.level == 0, "control: a fresh term starts at 0"
    line = resumed.load_state_dict(state, step=5000)

    assert (resumed.level, resumed.err_ema, resumed.last_promotion_step) == (2, 0.0421, 1234)
    assert "from the checkpoint" in line


def test_the_environment_overrides_and_drops_evidence_from_the_other_rung(monkeypatch) -> None:
    """Pins failure 4. The error average was measured at the checkpoint's rung;
    beside the rung the variable names it means nothing, and the dwell counts from
    the resume as it would after a promotion."""
    saved = _command()
    saved.level, saved.err_ema, saved.last_promotion_step = 2, 0.0421, 1234
    state = saved.state_dict()

    monkeypatch.setenv("MJRL_COMMAND_LEVEL", "1")
    resumed = _command()
    line = resumed.load_state_dict(state, step=5000)

    assert resumed.level == 1
    assert resumed.err_ema is None, "an error average from rung 2 was kept at rung 1"
    assert resumed.last_promotion_step == 5000
    assert "$MJRL_COMMAND_LEVEL" in line and "2" in line


def test_an_environment_that_agrees_keeps_the_evidence(monkeypatch) -> None:
    """The control for the test above: the evidence is dropped because the rung
    moved, not because the variable was set."""
    saved = _command()
    saved.level, saved.err_ema, saved.last_promotion_step = 2, 0.0421, 1234

    monkeypatch.setenv("MJRL_COMMAND_LEVEL", "2")
    resumed = _command()
    resumed.load_state_dict(saved.state_dict(), step=5000)

    assert (resumed.level, resumed.err_ema, resumed.last_promotion_step) == (2, 0.0421, 1234)


def test_a_level_past_a_shortened_ladder_is_clamped_without_its_evidence() -> None:
    saved = _command()
    saved.level, saved.err_ema = 3, 0.0421

    resumed = _command(levels=(0.25, 0.35, 0.50))
    line = resumed.load_state_dict(saved.state_dict(), step=5000)

    assert (resumed.level, resumed.err_ema, resumed.last_promotion_step) == (2, None, 5000)
    assert "top" in line


def test_a_checkpoint_without_the_ladder_leaves_the_constructors_level(monkeypatch) -> None:
    monkeypatch.setenv("MJRL_COMMAND_LEVEL", "1")
    resumed = _command()
    line = resumed.load_state_dict(None, step=5000)

    assert resumed.level == 1
    assert "does not carry" in line and "$MJRL_COMMAND_LEVEL" in line


def test_every_ladder_saves_what_its_promotion_reads() -> None:
    """The posture ladder's average is per axis, and the payload ladder's dwell
    runs from `_maxed_since` as well as the last promotion."""
    posture = PostureRangeCurriculum(SimpleNamespace(params={"angles": (0.1, 0.2)}), None)
    posture.level, posture.err_ema["height"] = 1, 0.007
    back = PostureRangeCurriculum(SimpleNamespace(params={"angles": (0.1, 0.2)}), None)
    back.load_state_dict(posture.state_dict(), step=0)
    assert back.level == 1 and back.err_ema == posture.err_ema

    payload = PayloadCurriculum(SimpleNamespace(params={"levels": ((0.0, 0.0), (0.0, 0.6))}), None)
    payload.level, payload._maxed_since = 1, 4321
    back = PayloadCurriculum(SimpleNamespace(params={"levels": ((0.0, 0.0), (0.0, 0.6))}), None)
    back.load_state_dict(payload.state_dict(), step=0)
    assert back.level == 1 and back._maxed_since == 4321


# ── Which tasks are covered (failures 2 and 3) ────────────────────────────


def _training_curricula() -> dict[str, dict]:
    return {
        spec.id: {name: term.func for name, term in tasks.load_env_cfg(spec.id).curriculum.items()}
        for spec in tasks.all_specs()
    }


def test_every_task_that_climbs_a_ladder_saves_it() -> None:
    """Pins failure 2.

    The control is the set itself: if the detector saw no ladders, "every task
    that has one" would be true of none.
    """
    laddered = {
        task
        for task, terms in _training_curricula().items()
        if any(inspect.isclass(f) and issubclass(f, CheckpointedLadder) for f in terms.values())
    }
    assert {"jumper.tripod", "jumper.posture", "jumper.five_foot"} <= laddered

    wrong = {
        task: getattr(tasks.load_runner_cls(task), "__name__", None)
        for task in sorted(laddered)
        if not issubclass(tasks.load_runner_cls(task) or object, CurriculumRunner)
    }
    assert not wrong, (
        f"{wrong}: these tasks climb a ladder through a runner that does not "
        f"checkpoint it, so --resume starts it again at level 0. Return "
        f"`tasks.jumper.common.runner.CurriculumRunner` from `runner_cls()`."
    )


def test_every_curriculum_that_keeps_a_level_is_checkpointed() -> None:
    """Pins failure 3: a class term with a `self.level` that the runner cannot find.

    The control is `CommandRangeCurriculum`: the detector has to flag it, or an
    empty result means only that the detector sees nothing.
    """

    def keeps_a_level(func) -> bool:
        return inspect.isclass(func) and "self.level" in inspect.getsource(func)

    assert keeps_a_level(CommandRangeCurriculum), "control: the detector misses a known ladder"
    missed = {
        f"{task}:{name}": func.__name__
        for task, terms in _training_curricula().items()
        for name, func in terms.items()
        if keeps_a_level(func)
        and not issubclass(func, CheckpointedLadder)
        and func not in NOT_CHECKPOINTED
    }
    assert not missed, (
        f"{missed} keep a level the checkpoint does not carry. Make them "
        f"`CheckpointedLadder`s, or add them to NOT_CHECKPOINTED with the reason."
    )


# ── Through a real runner and environment (failure 1) ─────────────────────

TASK = "jumper.tripod"
N_ENVS = 4


@pytest.fixture(scope="module")
def built():
    """One `jumper.tripod` environment on native CPU, and its training config."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjrl.backend.select import use_backend

    resolve = importlib.import_module("mjrl.backend.resolve")
    use_backend(resolve.resolve(backend="native", device="cpu", num_envs=N_ENVS))
    with pytest.MonkeyPatch.context() as mp:
        for name in LEVEL_ENVS:
            mp.delenv(name, raising=False)
        cfg = tasks.load_env_cfg(TASK)
        cfg.scene.num_envs = N_ENVS
        env = ManagerBasedRlEnv(cfg, device="cpu")
        wrapped = RslRlVecEnvWrapper(env)
    yield env, wrapped
    env.close()


def _runner(built, cls):
    _, wrapped = built
    return cls(wrapped, asdict(tasks.load_agent_cfg(TASK)), None, "cpu")


def _fresh_ladder(env) -> CommandRangeCurriculum:
    """Put the command ladder back where a new process has it before the load:
    constructed, and applied once by the wrapper's reset."""
    term_cfg = env.curriculum_manager.get_term_cfg("command")
    term_cfg.func = CommandRangeCurriculum(term_cfg, env)
    env.common_step_counter = 0
    term_cfg.func(env, torch.arange(env.num_envs), **term_cfg.params)
    return term_cfg.func


def _live(env) -> tuple[float, float]:
    """What training actually runs on: the command range and the tracking std."""
    lin_x = env.command_manager.get_term("twist").cfg.ranges.lin_vel_x[1]
    std = env.reward_manager.get_term_cfg("track_linear_velocity").params["std"]
    return lin_x, std


def _rung(env, level: int) -> tuple[float, float]:
    """What `_live` reads at `level`, from the task's own ladder and ratio."""
    p = env.curriculum_manager.get_term_cfg("command").params
    return p["levels"][level], lin_std(level, p["levels"], p["lin_std_scales"], p["lin_std_ratio"])


@pytest.fixture(scope="module")
def checkpoint(built, tmp_path_factory):
    """A run that had reached level 2, saved through `CurriculumRunner`.

    The error average is far above any rung's bar, so the load does not promote
    on it -- a checkpoint that was due a promotion takes it on the load, as it
    would have at the next reset, and that would move what this pins.
    """
    env, _ = built
    ladder = _fresh_ladder(env)
    ladder.level, ladder.err_ema, ladder.last_promotion_step = 2, 1.0, 1234
    env.common_step_counter = 5000
    path = tmp_path_factory.mktemp("run") / "model_7.pt"
    _runner(built, CurriculumRunner).save(str(path))
    return path


def test_the_checkpoint_carries_the_level(checkpoint) -> None:
    infos = torch.load(checkpoint, weights_only=False)["infos"]
    assert infos[CHECKPOINT_KEY]["command"]["level"] == 2
    assert infos["env_state"]["common_step_counter"] == 5000, (
        "mjlab's own entry is gone -- the save no longer goes through its runner"
    )


def test_a_resume_climbs_on_from_the_saved_rung(built, checkpoint) -> None:
    """Pins failure 1, in the managers as well as on the term."""
    env, _ = built
    ladder = _fresh_ladder(env)
    assert _live(env) == _rung(env, 0), "control: rung 0 is live"
    assert _rung(env, 0) != _rung(env, 2), "control: the two rungs are told apart"

    _runner(built, CurriculumRunner).load(str(checkpoint), map_location="cpu")

    assert (ladder.level, ladder.err_ema, ladder.last_promotion_step) == (2, 1.0, 1234)
    assert _live(env) == _rung(env, 2), (
        "the term is at level 2 and the managers are still on rung 0's range and std"
    )


def test_control_the_runner_the_tasks_had_restarts_the_ladder(built, checkpoint) -> None:
    """The same checkpoint through `VelocityOnPolicyRunner`: level 0, silently.
    If this passed as level 2, the test above could not fail."""
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

    env, _ = built
    ladder = _fresh_ladder(env)
    _runner(built, VelocityOnPolicyRunner).load(str(checkpoint), map_location="cpu")

    assert env.common_step_counter == 5000, "the step counter is restored either way"
    assert ladder.level == 0


def test_an_older_checkpoint_says_so_and_takes_the_environment(
    built, tmp_path, monkeypatch, capsys
) -> None:
    """A checkpoint written before levels were saved: `MJRL_COMMAND_LEVEL` is the
    only source, and the log has to say that it was."""
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

    env, _ = built
    _fresh_ladder(env)
    old = tmp_path / "model_3.pt"
    _runner(built, VelocityOnPolicyRunner).save(str(old))
    assert CHECKPOINT_KEY not in torch.load(old, weights_only=False)["infos"]

    monkeypatch.setenv("MJRL_COMMAND_LEVEL", "1")
    ladder = _fresh_ladder(env)
    _runner(built, CurriculumRunner).load(str(old), map_location="cpu")

    assert ladder.level == 1
    out = capsys.readouterr().out
    assert "predates checkpointed curriculum levels" in out
    assert "$MJRL_COMMAND_LEVEL" in out
