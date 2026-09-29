"""A replay does not evaluate the reward terms, and the policy must not be able to tell.

`mjrl/replay.py` takes the reward manager out of a replay step. Nothing in a replay
reads a reward, and evaluating them was the largest single cost of the step -- 4.7
of 11.3 ms on jumper.posture with one environment on warp.

**What could go wrong is silent.** A reward term that is the first thing in a step
to advance some state an observation reads -- a clock advanced lazily for whoever
asks first, a cache keyed on the step counter -- would, once skipped, leave that
advance to the observation. The observation runs after the reset and after the
command update, so the policy would be shown a phase one step off after every
reset, or a reference one step late, and every replay would look fine. Both kinds
of state exist in this repository today: jumper.posture's gait clock and
jumper.jump's reference cache.

So this runs every registered task twice from the same seed, with and without the
rewards, through time-outs and command resamples, and compares what the actor is
shown bit for bit. jumper.posture's actor and rewards do share one gait clock, and
a reward term is the first to advance it; it passes here because the clock
integrates the tempo the observation recorded and counts a reset as asking
(`CadenceClock` in `tasks/jumper/posture/mdp/cadence.py`). Made to integrate the
command live at whoever asks first, it fails at step 0; made to skip that catch-up
at a reset, at step 19, the first time-out.
"""

from __future__ import annotations

import pytest

import tasks

torch = pytest.importorskip("torch")

#: Steps per rollout, an episode length and a command resampling interval, all in
#: control steps. The episode is short so that the run crosses time-out resets --
#: where a lazily advanced clock goes wrong -- and the resampling so that it
#: crosses command changes, where one integrating a command-dependent rate does.
STEPS = 48
EPISODE_STEPS = 20
RESAMPLE_STEPS = 7

#: **Two environments, so a replay is compared batched, the way training runs** --
#: rollout on two threads, every environment stepped in per-thread scratch and put
#: back into its own MjData -- not the one-environment special case of one thread
#: and one history. It ran one until native was deterministic across environments:
#: each one's warm start and control leaked through rollout's per-thread scratch,
#: two identical unskipped runs of jumper.posture diverged at step 2 and
#: jumper.swing at step 5, and this test failed for a reason that had nothing to do
#: with rewards. `tests/test_native_determinism.py` holds that now.
#:
#: Both environments still time out on the same step, in every task, so a reset
#: never lands on one alone.
NUM_ENVS = 2


class _Counted:
    """A reward term that counts its calls and is otherwise the term.

    Attribute access falls through, so a class-based term keeps its `reset` --
    which the manager calls through `term_cfg.func` on every reset.
    """

    def __init__(self, func) -> None:
        self._func = func
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._func(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._func, name)


def _clone(obs):
    if isinstance(obs, dict):
        return {k: _clone(v) for k, v in obs.items()}
    return obs.clone()


def _equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return torch.equal(a, b)


def _rollout(task_id: str, skip: bool):
    """The actor's observation after every step, and how many reward terms ran.

    **The training config, not `play=True`.** Replay opens a connected gamepad, and
    a pad touched between the two rollouts would make them differ for a reason that
    has nothing to do with rewards. What is under test -- the order the managers
    run in and whatever state a reward term leaves behind -- is the same in both.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.replay import skip_rewards

    cfg = tasks.load_env_cfg(task_id)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.episode_length_s = EPISODE_STEPS * step_dt
    for term in cfg.commands.values():
        if hasattr(term, "resampling_time_range"):
            term.resampling_time_range = (RESAMPLE_STEPS * step_dt,) * 2

    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        counted = []
        for term_cfg in env.reward_manager._term_cfgs:
            term_cfg.func = _Counted(term_cfg.func)
            counted.append(term_cfg.func)
        if skip:
            skip_rewards(env)

        actions = torch.Generator().manual_seed(1)
        seen, resets = [], 0
        for _ in range(STEPS):
            action = torch.rand(env.action_space.shape, generator=actions) - 0.5
            obs, _, terminated, truncated, _ = env.step(action)
            assert "actor" in obs, f"{task_id} has no 'actor' observation group"
            seen.append(_clone(obs["actor"]))
            resets += int((terminated | truncated).sum())
        return seen, sum(c.calls for c in counted), resets
    finally:
        env.close()


@pytest.fixture
def native_backend():
    """Build on native cpu, and put back whatever backend was registered before."""
    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    use_backend(resolve(backend="native", device="cpu", num_envs=NUM_ENVS))
    try:
        yield
    finally:
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


@pytest.mark.parametrize("task_id", tasks.list_ids())
def test_skipping_rewards_does_not_change_what_the_policy_sees(
    task_id: str, native_backend
) -> None:
    trained, ran, resets = _rollout(task_id, skip=False)
    replayed, ran_skipped, resets_skipped = _rollout(task_id, skip=True)

    # The controls: the counting works, and the run crossed the resets it exists
    # to cross. Without the first, "no reward term ran" below would pass vacuously.
    assert ran > 0, f"{task_id}: no reward term was ever evaluated, even unskipped"
    assert resets > 0, f"{task_id}: the rollout never reset; shorten EPISODE_STEPS"

    assert ran_skipped == 0, f"{task_id}: {ran_skipped} reward term calls in a replay"
    assert resets_skipped == resets, (
        f"{task_id}: {resets} resets with rewards, {resets_skipped} without"
    )
    for step, (a, b) in enumerate(zip(trained, replayed, strict=True)):
        if _equal(a, b):
            continue
        # Only now is it worth a third rollout: to tell a reward term that matters
        # from a simulation that is not deterministic, which would make any
        # difference here meaningless -- see `NUM_ENVS`.
        again, _, _ = _rollout(task_id, skip=False)
        if not all(_equal(x, y) for x, y in zip(trained, again, strict=True)):
            pytest.fail(
                f"{task_id}: two identical rollouts with the rewards on already "
                f"differ, so the difference at step {step} says nothing about "
                f"skipping them. See NUM_ENVS."
            )
        pytest.fail(
            f"{task_id}: the actor's observation differs at step {step} once rewards "
            f"are skipped -- a reward term is the first to advance something the "
            f"observation reads. See mjrl/replay.py."
        )
