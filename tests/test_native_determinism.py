"""Two native rollouts from one seed agree bit for bit, with more than one environment.

`mujoco.rollout` hands its rolls to threads from a work queue, and the `MjData` it is
given are per-thread scratch: when it returns, `NativeSimulation._datas[t]` for
`t < nthread` holds the warm start and the control of whichever roll thread `t` ran
last. The backend read each environment's next warm start back from `_datas[i]`, so
environment `i` started its solver from another environment's accelerations, and
recomputed its derived quantities -- the actuator and contact forces the rewards read
-- from another environment's `ctrl`. Which one was thread scheduling.
`_refresh_derived` now puts `i`'s own back.

**What goes wrong is silent.** The solver converges from any warm start, and `_scatter`
rounds the state to float32 at every physics step, which erases most of the error
before an observation can show it: with two environments, most runs agree in every
float32 observation, and now and then one does not and grows from there --
jumper.posture once diverged at step 2 and jumper.swing at step 5, which is why
`tests/test_replay.py` ran one environment. So this compares the float64 state after
every physics step, before the next step rounds it, and the forces the rewards read,
as well as what the actor is shown.

Measured against the backend before the fix (i9-14900KF, 2026-09-26): both tests
failed in each of five runs. Two runs parted in the float64 state from physics step 1
to 10 of 48 (max |difference| 2e-15 to 4e-8) and, in the two runs that recorded them,
in the forces the rewards read from step 0 to 8 (max |difference| 18); one thread and
two parted from step 1 in the state (9.4e-7) and from step 0 in the forces (15 to 18).
Put back only the warm start and both tests still fail, on the forces; only the
control, and both fail on the float64 state. The control passed throughout.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

import tasks

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

#: In contact from its first step, four to eight contacts per environment, so the
#: solver and its warm start are in play from the start. jumper.flat and
#: jumper.posture drop onto the floor first; their float64 state diverged 11 and 51
#: physics steps in, against 1-6 for this one.
TASK = "jumper.swing"
NUM_ENVS = 2
#: Control steps per rollout -- four physics steps each for this task.
STEPS = 12

_FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class Run(NamedTuple):
    #: [physics step, env, FULLPHYSICS] -- float64, before the next step rounds it
    states: np.ndarray
    #: [physics step, env, nu + nsensordata] -- the actuator forces and sensor
    #: readings the reward and termination terms read, before `forward()` renews them
    forces: torch.Tensor
    #: [control step, env, obs] -- what the actor was shown
    actor: torch.Tensor


def _rollout(nthread: int, before_step=None) -> Run:
    """The training config from seed 0, driven by a fixed action sequence.

    `before_step(sim)` runs ahead of every physics step -- the control below uses it
    to hand environment 0 a warm start that is not its own.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    use_backend(
        resolve(backend="native", device="cpu", num_envs=NUM_ENVS, cpu_threads=nthread)
    )
    cfg = tasks.load_env_cfg(TASK)
    cfg.scene.num_envs = NUM_ENVS
    cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        sim = env.sim
        # Were this quietly the default, the thread-count test would compare a run
        # with itself.
        assert sim._nthread == nthread, f"asked for {nthread} threads, got {sim._nthread}"

        states, forces = [], []
        step = sim.step
        size = mujoco.mj_stateSize(sim.mj_model, _FULL)

        def recording_step(*args, **kwargs):
            if before_step is not None:
                before_step(sim)
            step(*args, **kwargs)
            now = np.zeros((NUM_ENVS, size))
            for i in range(NUM_ENVS):
                mujoco.mj_getState(*sim.env_mjdata(i), now[i], _FULL)
            states.append(now)
            forces.append(torch.cat([sim.data.actuator_force, sim.data.sensordata], 1))

        sim.step = recording_step

        actions = torch.Generator().manual_seed(1)
        actor = []
        for _ in range(STEPS):
            action = torch.rand(env.action_space.shape, generator=actions) - 0.5
            obs, *_ = env.step(action)
            actor.append(obs["actor"].clone())
        return Run(np.stack(states), torch.stack(forces), torch.stack(actor))
    finally:
        env.close()


def _differences(a: Run, b: Run) -> list[str]:
    """Where two runs part, one line per record that differs."""
    found = []
    for name, x, y, unit in (
        ("the float64 state", torch.from_numpy(a.states), torch.from_numpy(b.states),
         "physics step"),
        ("the forces the rewards read", a.forces, b.forces, "physics step"),
        ("the actor's observation", a.actor, b.actor, "control step"),
    ):
        differ = (x != y).flatten(1).any(dim=1)
        if differ.any():
            k = int(differ.float().argmax())
            gap = float((x - y).abs().max())
            found.append(
                f"{name} from {unit} {k} of {len(differ)} (max |difference| {gap:.1e})"
            )
    return found


@pytest.fixture(scope="module")
def baseline():
    """One rollout on NUM_ENVS threads, and whatever backend was registered before
    put back afterwards."""
    from mjlab.sim import get_simulation_cls, set_simulation_cls
    from mjrl.backend import native_sim

    saved_cls = get_simulation_cls()
    saved_nthread = native_sim._DEFAULT_NTHREAD
    try:
        yield _rollout(NUM_ENVS)
    finally:
        set_simulation_cls(saved_cls)
        native_sim._DEFAULT_NTHREAD = saved_nthread


def _cold_start_env0(sim) -> None:
    sim.env_mjdata(0)[1].qacc_warmstart[:] = 0.0


def test_a_warm_start_that_is_not_its_own_shows(baseline: Run) -> None:
    """The control: without it, the two tests below could pass on a scenario the
    warm start never reaches -- no contact, a solver that converges to the same bits
    from anywhere.

    Environment 0 is handed a warm start that is not the one it earned -- zero, at
    every physics step, which is what the old code handed every environment past
    the first nthread -- and the float64 state has to show it.
    """
    cold = _rollout(NUM_ENVS, before_step=_cold_start_env0)
    assert any("float64 state" in d for d in _differences(baseline, cold)), (
        f"{TASK}: starting environment 0 cold changed nothing, so agreement below "
        f"proves nothing -- choose a task and length that load the contact solver"
    )


def test_two_runs_from_one_seed_agree(baseline: Run) -> None:
    differences = _differences(baseline, _rollout(NUM_ENVS))
    assert not differences, (
        f"{TASK} with {NUM_ENVS} environments: two identical runs from one seed "
        f"disagree in {'; '.join(differences)}. An environment is being stepped from "
        f"another one's warm start or control; see `_refresh_derived` in "
        f"rl/mjrl/backend/native_sim.py."
    )


def test_the_thread_count_does_not_change_the_result(baseline: Run) -> None:
    """`MJRL_CPU_THREADS` is a performance knob; USAGE.md says it changes throughput
    and nothing else, and `resolve.DEFAULT_MAX_THREADS` was chosen on that premise.

    Unlike the test above, this one does not rely on two runs happening to be
    scheduled differently: on one thread there is one schedule, and in it
    environment 0's scratch ends every rollout holding the last environment's roll
    -- which a run on more threads would have to reproduce at every physics step to
    agree with it.
    """
    differences = _differences(baseline, _rollout(1))
    assert not differences, (
        f"{TASK}: {NUM_ENVS} threads and 1 thread disagree in "
        f"{'; '.join(differences)}. The thread count must not be a hyper-parameter; "
        f"see `_refresh_derived` in rl/mjrl/backend/native_sim.py."
    )
