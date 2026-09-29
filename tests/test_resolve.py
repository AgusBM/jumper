"""Backend and device resolution.

The emphasis is entirely on **not falling back**: explicitly asking for an
unavailable combination must fail rather than quietly become something else and
let someone believe they are training on a GPU. No test depends on whether this
machine has CUDA -- the detection functions are stubbed wherever a result is
needed.
"""

from __future__ import annotations

import sys

import pytest

from mjrl.backend.resolve import BackendUnavailable, resolve

# `from .resolve import resolve` in `rl/mjrl/backend/__init__.py` binds the
# **function** to the package attribute `mjrl.backend.resolve`, shadowing the
# submodule of the same name -- so `import mjrl.backend.resolve as m` yields the
# function, not the module. Stubbing the module's detection functions means going
# through sys.modules.
resolve_mod = sys.modules["mjrl.backend.resolve"]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """External environment variables change the result, so they are cleared."""
    monkeypatch.delenv("MJRL_BACKEND", raising=False)
    monkeypatch.delenv("MJRL_DEVICE", raising=False)


def _stub_probes(monkeypatch: pytest.MonkeyPatch, *, gpu: bool) -> None:
    for name in ("_mujoco_warp_importable", "_torch_cuda", "_warp_sees_cuda"):
        monkeypatch.setattr(resolve_mod, name, lambda gpu=gpu: gpu)


# ── No fallback ───────────────────────────────────────────────────────────


def test_native_rejects_gpu_device() -> None:
    with pytest.raises(BackendUnavailable, match="native backend only supports device=cpu"):
        resolve(backend="native", device="cuda:0")


def test_warp_without_cuda_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The critical one: asking explicitly for a GPU the machine does not have
    must fail, not silently switch to CPU."""
    monkeypatch.setattr(resolve_mod, "_mujoco_warp_importable", lambda: True)
    monkeypatch.setattr(resolve_mod, "_torch_cuda", lambda: False)
    monkeypatch.setattr(resolve_mod, "_warp_sees_cuda", lambda: False)
    with pytest.raises(BackendUnavailable, match="no usable\\s+CUDA device"):
        resolve(backend="warp", device="cuda:0")


def test_warp_without_mujoco_warp_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_probes(monkeypatch, gpu=False)
    with pytest.raises(BackendUnavailable, match="mujoco_warp is not installed"):
        resolve(backend="warp", device="cuda:0")


def test_unknown_backend() -> None:
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        resolve(backend="bogus")


# ── auto detection ────────────────────────────────────────────────────────


def test_auto_picks_native_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_probes(monkeypatch, gpu=False)
    r = resolve()
    assert (r.backend, r.device) == ("native", "cpu")
    assert r.forced is False
    assert r.num_envs == 64


def test_auto_picks_warp_with_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_probes(monkeypatch, gpu=True)
    r = resolve()
    assert (r.backend, r.device) == ("warp", "cuda:0")
    assert r.num_envs == 4096


# ── Precedence: command line > environment ────────────────────────────────


def test_env_var_used_when_cli_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_probes(monkeypatch, gpu=True)
    monkeypatch.setenv("MJRL_BACKEND", "native")
    r = resolve()
    assert r.backend == "native", "the environment wins even with a GPU present"
    assert r.forced is True


def test_cli_beats_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_probes(monkeypatch, gpu=False)
    monkeypatch.setenv("MJRL_BACKEND", "warp")
    r = resolve(backend="native", device="cpu")
    assert r.backend == "native", "an explicit command-line value must beat the environment"


# ── Warnings ──────────────────────────────────────────────────────────────


def test_non_default_num_envs_warns_about_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing num_envs changes PPO's effective batch, which has to be said or
    nobody thinks to re-tune the hyper-parameters."""
    _stub_probes(monkeypatch, gpu=False)
    r = resolve(backend="native", device="cpu", num_envs=512)
    assert any("num_envs=512" in n for n in r.notes)


def test_a_replay_count_is_not_a_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay's `num_envs` is how many robots are on screen, so the batch-size
    warning does not apply to it.

    It used to print on every `play` -- "the hyper-parameters need re-tuning"
    about a run that trains nothing -- and a warning that is always there is one
    people learn to skip, including where it means something. The test above,
    with the same count and `training` left at its default, is the control.
    """
    _stub_probes(monkeypatch, gpu=False)
    r = resolve(backend="native", device="cpu", num_envs=512, training=False)
    assert r.num_envs == 512
    assert not any("batch size" in n for n in r.notes), r.notes


def test_banner_reports_threads_for_native(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_probes(monkeypatch, gpu=False)
    banner = resolve(backend="native", device="cpu", cpu_threads=8).banner()
    assert "backend=native" in banner
    assert "threads=8" in banner
    assert "(forced)" in banner


def test_the_automatic_thread_count_is_capped_below_the_core_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Every core" is the obvious default and it is measurably wrong.

    About half of a native step is serial (the gather and scatter between the
    batch buffers and the per-environment MjData is Python-side), so by Amdahl
    the parallel part is spent by around eight workers and every thread after
    that is pure overhead. Measured on an i9-14900KF (32 logical cores), ms per
    policy step at 64 environments: 8 threads 15.27, 16 threads 21.18, 24
    (the physical core count) 22.39, 32 (every core) 22.36 -- the old default
    was 46% off the optimum, and at 256 environments 16% off.

    Nothing about that failure is visible: training runs, results are identical
    (the thread count is not a hyper-parameter -- see
    tests/test_native_determinism.py), it is simply slower than it should be on
    exactly the machines bought for being fast.

    The core count is faked rather than read: on a host with fewer cores than the
    ceiling the assertion would hold no matter what the code did, and this test
    has to fail on the machines the change is *for*. (Written the honest way
    first, it passed against the old "every core" default on a 6-core laptop.)
    """
    import importlib

    # `import mjrl.backend.resolve as R` would bind the **function** `resolve`:
    # the package's __init__ re-exports it, shadowing the submodule name.
    R = importlib.import_module("mjrl.backend.resolve")

    many = R.DEFAULT_MAX_THREADS * 4
    monkeypatch.setattr(R, "_available_cores", lambda: many)
    assert R._default_threads() == R.DEFAULT_MAX_THREADS, (
        f"a {many}-core host must not default to every core"
    )
    r = resolve(backend="native", device="cpu", num_envs=1024, cpu_threads=0)
    assert r.cpu_threads == R.DEFAULT_MAX_THREADS

    # And the other direction: a small host uses what it has, never the ceiling.
    few = 3
    monkeypatch.setattr(R, "_available_cores", lambda: few)
    assert R._default_threads() == few


def test_an_explicit_thread_count_beats_the_ceiling() -> None:
    """The ceiling is a default, not a limit.

    It comes from measurements on two machines; someone on a third whose
    workload scales further has to be able to say so, and `MJRL_CPU_THREADS`
    is how.
    """
    from mjrl.backend.resolve import DEFAULT_MAX_THREADS

    asked = DEFAULT_MAX_THREADS * 4
    r = resolve(backend="native", device="cpu", num_envs=1024, cpu_threads=asked)
    assert r.cpu_threads == asked, "an explicit thread count must be honoured as given"


def test_backend_and_resolve_agree_on_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One source for the number, or the two drift apart in silence.

    `NativeSimulation` built directly -- in tests, or by anything not going
    through `resolve` -- resolves its own thread count. Were that path to keep
    its own copy of "every core", the measured default would apply to training
    and not to the benchmarks written to check it.
    """
    import importlib

    from mjrl.backend.native_sim import _resolve_nthread

    R = importlib.import_module("mjrl.backend.resolve")  # see the note above

    monkeypatch.setattr(R, "_available_cores", lambda: R.DEFAULT_MAX_THREADS * 4)
    assert _resolve_nthread(None, 1024) == R._default_threads() == R.DEFAULT_MAX_THREADS
    # And the environment-count cap still applies on that path
    assert _resolve_nthread(None, 2) == 2


def test_threads_are_capped_by_env_count_so_the_banner_is_honest() -> None:
    """The printed thread count must be the one **actually used**.

    `mujoco.rollout` requires the scratch list to be exactly nthread long and the
    backend has only num_envs MjData copies, so the thread count is necessarily
    capped by the environment count. Without capping in `resolve()`, the banner
    would print "threads=32" while 16 actually ran.

    This configuration's earlier defect was exactly "printing a value that never
    took effect"; do not reintroduce it in a different form.
    """
    r = resolve(backend="native", device="cpu", num_envs=4, cpu_threads=32)
    assert r.cpu_threads == 4, f"threads not capped by environment count: {r.cpu_threads}"
    assert "threads=4" in r.banner() if hasattr(r, "banner") else True

    r = resolve(backend="native", device="cpu", num_envs=1024, cpu_threads=8)
    assert r.cpu_threads == 8, "an explicit thread count must not be raised by the env count"
