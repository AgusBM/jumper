"""jumper.five_foot's payload curriculum: the ways it goes wrong without raising.

The claw's payload used to be a second stage, switched on by editing a constant
and resuming. It is now a curriculum term (`tasks/jumper/five_foot/mdp/
curriculum.py`), and each of these is a way for that to run, log plausible
numbers and train the wrong thing:

1. **A fresh run starting loaded.** The failure the curriculum replaces: master
   sat on the second stage's range, so a cold start carried 0-600 g from the first
   iteration -- the configuration measured, twice, to stand still.
2. **The gate opening on a robot that does not walk**, or before the commands
   have finished climbing.
3. **A promotion reaching into an episode.** Every environment is supposed to
   take its load at its *next reset*; loading them all at the promotion changes
   the mass under a robot mid-stride.
4. **A load that reaches the model and not the physics.** The event manager
   recomputes the derived constants after an event fires; nothing does after a
   curriculum term, so the term has to, or `body_subtreemass` and the invweights
   stay at their empty-claw values.
5. **One mass for every environment.** The fields are per-world only because the
   startup event declares them; a curriculum declares none.
6. **A resume that cannot say where it was.** The level is not checkpointed.

Each test names the one it pins. The running-environment tests are on the native
CPU backend, and read the per-environment `MjModel`s the physics steps with,
never `env.sim.model`: on native a read marks the model dirty and the next step
scatters the buffers back, derived constants included, which would overwrite the
very recomputation 4 is about.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="the curriculum is torch")
mujoco = pytest.importorskip("mujoco", reason="the payload body is looked up by name")

import tasks
from tasks.jumper.five_foot import env_cfg as ff_env_cfg
from tasks.jumper.five_foot.claw import PAYLOAD_BODY, PAYLOAD_RANGE, PAYLOAD_UNIT_INERTIA
from tasks.jumper.five_foot.mdp.curriculum import PAYLOAD_LEVEL_ENV, PayloadCurriculum

N_ENVS = 8


# ── The configuration ─────────────────────────────────────────────────────


def test_a_fresh_run_starts_empty_and_the_curriculum_owns_the_load() -> None:
    """Pins failure 1, and the half of 5 a config can show.

    The startup event is still there -- it is what makes the fields per-world --
    but it samples an empty claw, and the range the policy ends up under is the
    curriculum's top rung. The control is that the top rung is the measured range
    and not a second zero, or "starts empty" would be true of a curriculum that
    never loads anything.
    """
    cfg = tasks.load_env_cfg("jumper.five_foot")
    assert "claw_payload" in cfg.events, (
        "the startup event is gone; without it body_mass is not per-world and the "
        "curriculum's writes land in one shared field"
    )
    event = cfg.events["claw_payload"]
    assert event.mode == "startup"
    assert tuple(event.params["ranges"]) == (0.0, 0.0), (
        f"a fresh run carries {event.params['ranges']} kg from the first iteration: "
        f"the cold start that was measured to stand still"
    )

    assert "payload" in cfg.curriculum, "nothing ever loads the claw"
    levels = cfg.curriculum["payload"].params["levels"]
    assert tuple(levels[0]) == tuple(event.params["ranges"])
    assert tuple(levels[-1]) == tuple(PAYLOAD_RANGE)
    assert PAYLOAD_RANGE[1] > 0.0, "the top rung is empty too; nothing is ever carried"

    # It reads the command curriculum's level, so it has to run after it: the
    # manager calls terms in insertion order.
    names = list(cfg.curriculum)
    assert names.index("payload") > names.index("command")

    play = tasks.load_env_cfg("jumper.five_foot", play=True)
    assert "payload" not in play.curriculum and "claw_payload" not in play.events


def test_the_dwell_is_the_iterations_it_was_chosen_in() -> None:
    """The dwell is in environment steps and was chosen in iterations.

    `NUM_STEPS_PER_ENV` is what converts one into the other. The control is the
    agent config: if the rollout length moves and the dwell is still computed from
    a literal, this fails rather than silently halving or doubling the wait.
    """
    cfg = tasks.load_env_cfg("jumper.five_foot")
    agent = tasks.load_agent_cfg("jumper.five_foot")
    assert cfg.curriculum["payload"].params["dwell_steps"] == (
        ff_env_cfg.PAYLOAD_DWELL_ITERATIONS * agent.num_steps_per_env
    )


# ── Where a run starts ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "level"),
    [(None, 0), ("", 0), (" ", 0), ("1", 1), ("7", 1), ("-3", 0)],
)
def test_the_resume_level_comes_from_the_environment(monkeypatch, value, level) -> None:
    """Pins failure 6.

    Unset and empty are the same intent -- `MJRL_PAYLOAD_LEVEL=` is what a shell
    writes when the value is a variable that happens to be empty -- and out of
    range is clamped to this ladder rather than raising or indexing past it.
    """
    if value is None:
        monkeypatch.delenv(PAYLOAD_LEVEL_ENV, raising=False)
    else:
        monkeypatch.setenv(PAYLOAD_LEVEL_ENV, value)
    term = PayloadCurriculum(SimpleNamespace(params={"levels": ((0.0, 0.0), PAYLOAD_RANGE)}), None)
    assert term.level == level


def test_rung_zero_has_to_be_what_the_startup_event_samples() -> None:
    """Every environment carries the event's range until the curriculum loads it.

    A ladder whose first rung disagrees would log "level 0: 0.0-0.3 kg" over a
    population that is carrying whatever the event put there, and would never
    load rung 0 at all, because every environment already counts as carrying it.
    """
    env = SimpleNamespace(
        event_manager=SimpleNamespace(
            get_term_cfg=lambda name: SimpleNamespace(params={"ranges": (0.0, 0.0)})
        )
    )
    term = PayloadCurriculum(SimpleNamespace(params={"levels": ((0.0, 0.3), PAYLOAD_RANGE)}), env)
    with pytest.raises(ValueError, match="rung 0"):
        term(env, None, levels=((0.0, 0.3), PAYLOAD_RANGE), lin_err_bar=0.12, dwell_steps=0)


# ── In a running environment ──────────────────────────────────────────────


def _build(monkeypatch, drop_event: bool = False):
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.select import use_backend

    resolve = importlib.import_module("mjrl.backend.resolve")
    use_backend(resolve.resolve(backend="native", device="cpu", num_envs=N_ENVS))

    # The commands start on their top rung, with its ranges applied, so a robot
    # that stands still is judged against a 0.5 m/s range: measured in this
    # fixture, a mean error of 0.46, nearly four times the payload's bar. The
    # shell's own settings are not this test's.
    monkeypatch.setenv("MJRL_COMMAND_LEVEL", "3")
    monkeypatch.delenv(PAYLOAD_LEVEL_ENV, raising=False)

    cfg = tasks.load_env_cfg("jumper.five_foot")
    cfg.scene.num_envs = N_ENVS
    # Short enough to run in seconds: the dwell and the sample size are what a
    # test cannot afford at their training values. The bar is the task's own.
    cfg.curriculum["payload"].params.update(dwell_steps=20, min_steps=10, min_samples=1)
    if drop_event:
        del cfg.events["claw_payload"]
    return ManagerBasedRlEnv(cfg, device="cpu")


@pytest.fixture(scope="module")
def run():
    """One environment taken through the gate, with a snapshot after each phase.

    Module-scoped because building the environment is the expensive part, and the
    tests below each read a different phase of the same run.
    """
    from tasks.jumper.common.mdp.curriculum import CommandRangeCurriculum

    with pytest.MonkeyPatch.context() as mp:
        env = _build(mp)
        try:
            env.reset()
            cm = env.curriculum_manager
            payload = next(c.func for c in cm._term_cfgs if isinstance(c.func, PayloadCurriculum))
            command = next(
                c.func for c in cm._term_cfgs if isinstance(c.func, CommandRangeCurriculum)
            )
            body = mujoco.mj_name2id(
                env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{PAYLOAD_BODY}"
            )
            zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
            first, second = [0, 1, 2, 3], [4, 5, 6, 7]

            def step(n: int) -> None:
                for _ in range(n):
                    env.step(zero)

            def reset(ids: list[int]) -> None:
                env.reset(env_ids=torch.tensor(ids))

            def snap() -> dict:
                # The models the physics steps with, not `env.sim.model` (see the
                # module docstring for why a read there would spoil the reading).
                models = env.sim._models
                return {
                    **cm._curriculum_state["payload"],
                    "mass": [float(m.body_mass[body]) for m in models],
                    "inertia": [m.body_inertia[body].copy() for m in models],
                    "subtree": [float(m.body_subtreemass[0]) for m in models],
                    "episode": env.episode_length_buf.tolist(),
                }

            out = {"start": snap()}

            # A: one rung short of the top -- the top *range*, before the precision
            # rung this task appends, which is where judging against the shared
            # ladder would call the commands maxed. A permissive bar, so that only
            # the command gate can be what holds the level.
            cm.get_term_cfg("payload").params["lin_err_bar"] = 10.0
            command.level = 2
            step(30)
            reset(first)
            out["not_maxed"] = snap()
            command.level = 3

            # B: maxed from here. The first call to see it is still inside the
            # dwell, so the permissive bar cannot promote it yet.
            step(30)
            reset(first)
            out["dwelling"] = snap()

            # B': the dwell has passed; the task's real bar, and a robot that stands.
            cm.get_term_cfg("payload").params["lin_err_bar"] = ff_env_cfg.PAYLOAD_LIN_ERR_BAR
            step(30)
            reset(second)
            out["standing"] = snap()

            # C: the same, with a bar a standing robot clears -- the control that
            # shows B was held by the error and not by something else.
            cm.get_term_cfg("payload").params["lin_err_bar"] = 10.0
            step(30)
            out["before_promotion"] = snap()
            reset(first)
            out["promoted"] = snap()

            # D: the other half resets and takes its load.
            step(5)
            reset(second)
            out["all_loaded"] = snap()
            out["level"] = payload.level
            yield out
        finally:
            env.close()


def test_the_claw_is_empty_until_the_gate_opens(run) -> None:
    """Pins failure 2: not un-maxed commands, not the dwell, not a robot that stands.

    Each phase holds the level with exactly one condition false. The first two
    run with a bar anything clears, so only the commands and the dwell can be what
    holds them; the standing phase is judged against the task's real bar, and the
    promotion phase is the same run with the bar opened again -- so the level
    staying at 0 is each condition's doing and not a gate that cannot open.
    """
    assert run["start"]["level"] == 0.0 and max(run["start"]["mass"]) == 0.0

    held = run["not_maxed"]
    assert held["commands_maxed"] == 0.0 and held["level"] == 0.0, (
        "the claw was loaded before the command curriculum reached its top rung"
    )
    assert max(held["mass"]) == 0.0

    # Maxed, an error under the (permissive) bar, and held by the dwell alone.
    dwelling = run["dwelling"]
    assert dwelling["commands_maxed"] == 1.0 and dwelling["lin_err"] < dwelling["lin_err_bar"]
    assert dwelling["settled"] == 0.0 and dwelling["level"] == 0.0, (
        "the claw was loaded on the first reset after the commands maxed out, "
        "without the dwell"
    )

    standing = run["standing"]
    assert standing["commands_maxed"] == 1.0 and standing["settled"] == 1.0
    assert standing["lin_err"] > ff_env_cfg.PAYLOAD_LIN_ERR_BAR, (
        f"a robot that stands still tracks to {standing['lin_err']:.3f}; the bar "
        f"{ff_env_cfg.PAYLOAD_LIN_ERR_BAR} would let it carry"
    )
    assert standing["level"] == 0.0 and max(standing["mass"]) == 0.0

    # Control: the same run promotes as soon as the bar is one it can clear.
    assert run["promoted"]["promoted"] == 1.0 and run["promoted"]["level"] == 1.0


def test_a_promotion_loads_each_environment_at_its_own_reset(run) -> None:
    """Pins failure 3.

    At the promotion only the environments that were resetting take a load; the
    four mid-episode keep an empty claw until their own reset, and then take one.
    The episode counters are the control that the four really were mid-episode.
    """
    promoted = run["promoted"]
    first, second = slice(0, 4), slice(4, 8)
    assert min(run["before_promotion"]["episode"][second]) > 0, (
        "the second half was not mid-episode when the promotion landed; the check "
        "below would pass for the wrong reason"
    )
    assert all(m > 0.0 for m in promoted["mass"][first]), "the resetting half was not loaded"
    assert promoted["mass"][second] == [0.0] * 4, (
        f"a promotion loaded environments mid-episode: {promoted['mass'][second]}"
    )
    assert promoted["loaded"] == pytest.approx(0.5)

    loaded = run["all_loaded"]
    assert all(m > 0.0 for m in loaded["mass"]) and loaded["loaded"] == 1.0
    # Sampled once: the first half kept what it was given.
    assert loaded["mass"][first] == promoted["mass"][first]


def test_every_environment_carries_its_own_load(run) -> None:
    """Pins failure 5 in a running environment: one sample per environment.

    Eight draws from the top rung, all distinct and all inside it. A shared field
    would show one value eight times.
    """
    low, high = PAYLOAD_RANGE
    masses = run["all_loaded"]["mass"]
    assert all(low <= m <= high for m in masses), masses
    assert len({round(m, 6) for m in masses}) == N_ENVS, f"one mass for many environments: {masses}"


def test_the_load_reaches_the_physics_with_its_inertia(run) -> None:
    """Pins failure 4: the derived constants move with the mass.

    `body_subtreemass` of the world body is the robot's total mass, and it is
    only right if the constants were recomputed after the write. Compared against
    the same model before anything was loaded, per environment, so the check does
    not depend on knowing the robot's own mass. The inertia is the event's -- the
    mass times the cylinder's inertia per kilogram.
    """
    empty = run["start"]["subtree"]
    loaded = run["all_loaded"]
    for i in range(N_ENVS):
        m = loaded["mass"][i]
        assert loaded["subtree"][i] - empty[i] == pytest.approx(m, abs=1e-5), (
            f"env {i} carries {m:.3f} kg and its total mass moved by "
            f"{loaded['subtree'][i] - empty[i]:.3f}: the constants were not recomputed"
        )
        assert loaded["inertia"][i] == pytest.approx([m * u for u in PAYLOAD_UNIT_INERTIA], rel=1e-5)


def test_the_curriculum_refuses_to_run_without_the_event(monkeypatch) -> None:
    """Pins failure 5 at its cause: without the event nothing is per-world.

    So the curriculum looks the event up and raises, instead of writing into a
    field every environment shares. The control is the fixture's run, which has
    the event and gets through the same reset.
    """
    env = _build(monkeypatch, drop_event=True)
    try:
        with pytest.raises(ValueError, match="claw_payload"):
            env.reset()
    finally:
        env.close()
