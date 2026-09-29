"""Backend and device resolution.

Precedence: **command line > environment variables > task cfg defaults**.

The governing principle is **no silent fallback**: asking explicitly for `warp` on
a machine without CUDA must fail, not quietly switch to CPU and let someone
believe they are training on a GPU. That is the easiest trap for a framework like
this to set.

Of the three combinations only two are training paths:

    warp   · cuda   primary training
    native · cpu    real training without a GPU
    warp   · cpu    kernel-logic debugging and cross-checking only (very slow --
                    measured at roughly 1/40 of multi-threaded native; do not
                    train with it)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

__all__ = ["Resolution", "resolve", "BackendUnavailable"]

Backend = Literal["warp", "native"]

_ENV_BACKEND = "MJRL_BACKEND"
_ENV_DEVICE = "MJRL_DEVICE"


class BackendUnavailable(RuntimeError):
    """An explicitly requested backend/device is unavailable here. Fail, do not
    fall back."""


@dataclass(frozen=True)
class Resolution:
    backend: Backend
    device: str
    num_envs: int
    cpu_threads: int
    forced: bool
    """True when this came from an explicit choice rather than detection."""
    strip_visual: bool | None = None
    """Whether native strips visual-only meshes. None = decide from the memory the
    per-environment models would take."""
    notes: tuple[str, ...] = ()
    """Things the user should be told, printed before training starts."""

    def banner(self) -> str:
        how = "forced" if self.forced else "auto"
        head = (
            f"[mjrl] backend={self.backend} device={self.device} "
            f"num_envs={self.num_envs}"
        )
        if self.backend == "native":
            head += f" threads={self.cpu_threads}"
        return "\n".join(
            [f"{head} ({how})", *(f"[mjrl] warning: {n}" for n in self.notes)]
        )


# ── Detection ─────────────────────────────────────────────────────────────


def _torch_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _mujoco_warp_importable() -> bool:
    import importlib.util

    return importlib.util.find_spec("mujoco_warp") is not None


def _warp_sees_cuda() -> bool:
    try:
        import warp as wp
    except ImportError:
        return False
    try:
        wp.init()
        return any(d.is_cuda for d in wp.get_devices())
    except Exception:  # noqa: BLE001 - a failed warp init means unavailable
        return False


#: Ceiling on the automatic thread count.
#:
#: "Every core" is the obvious default and it is **measurably wrong**. Roughly
#: half of a native step is serial -- the gather and scatter between the batch
#: buffers and the per-environment MjData is Python-side -- so by Amdahl the
#: parallel part is already spent by about eight workers, and past that each
#: extra thread only adds synchronisation and scheduling overhead.
#:
#: Measured on an i9-14900KF (8 performance + 16 efficiency cores, 32 logical),
#: ms per policy step, native:cpu, jumper.tetrapod:
#:
#:     envs    2      4      8     16     24     32 (= every core)
#:       16  10.63   8.10   7.17   9.70    --    9.85
#:       64  24.14  16.31  15.27  21.18  22.39  22.36
#:      256  92.84  69.79  59.09  62.84  67.39  68.82
#:
#: Eight is the optimum at every environment count, and "every core" costs
#: 16-46%. Pinning does not rescue it -- eight threads on eight dedicated
#: performance cores measured 21.44 ms against 15.71 unpinned, because the main
#: thread needs somewhere to run too. Nor is the physical core count the answer
#: (24 here): it measured no better than 32.
#:
#: A machine with fewer cores than this simply uses what it has; the ceiling
#: only bites on many-core hosts, which is exactly where the old default hurt.
#: `MJRL_CPU_THREADS` overrides it in either direction.
DEFAULT_MAX_THREADS = 8


def _available_cores() -> int:
    """Cores this process may actually use.

    `sched_getaffinity` respects cgroup / taskset limits, while `os.cpu_count()`
    inside a container reports the host's cores.
    """
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def _default_threads() -> int:
    return min(_available_cores(), DEFAULT_MAX_THREADS)


# ── Resolution ────────────────────────────────────────────────────────────


def resolve(
    backend: str = "auto",
    device: str = "auto",
    num_envs: int | None = None,
    cpu_threads: int = 0,
    strip_visual: bool | None = None,
    num_envs_gpu: int = 4096,
    num_envs_cpu: int = 64,
    training: bool = True,
) -> Resolution:
    """Resolve the final backend / device / num_envs / thread count.

    When `backend` or `device` is ``"auto"``, environment variables are consulted
    first and capabilities detected second. Explicit values (from the command line
    or the environment) never fall back.

    `training` says `num_envs` is PPO's batch size, which is what makes a count
    other than the backend's default worth a warning. A replay's count is how many
    robots are on screen and has no batch to change, so there the warning is
    noise -- printed on every replay, it teaches people to skip the one that means
    something.
    """
    env_backend = os.environ.get(_ENV_BACKEND)
    env_device = os.environ.get(_ENV_DEVICE)

    if backend == "auto" and env_backend:
        backend = env_backend
    if device == "auto" and env_device:
        device = env_device

    forced = backend != "auto" or device != "auto"

    if backend == "auto":
        if _mujoco_warp_importable() and _torch_cuda() and _warp_sees_cuda():
            backend, auto_device = "warp", "cuda:0"
        else:
            backend, auto_device = "native", "cpu"
        if device == "auto":
            device = auto_device

    if device == "auto":
        device = "cuda:0" if backend == "warp" else "cpu"

    # ── Availability check for the resolved combination: fail, never fall back ──
    if backend == "warp":
        if not _mujoco_warp_importable():
            raise BackendUnavailable(
                "backend=warp was requested but mujoco_warp is not installed.\n"
                "Install it with: pip install \"mjlab[cu128]\", or use "
                "--backend native."
            )
        if device.startswith("cuda") and not (_torch_cuda() and _warp_sees_cuda()):
            raise BackendUnavailable(
                f"device={device} was requested but this machine has no usable "
                f"CUDA device.\n"
                "Diagnose with `python -c \"import warp as wp; wp.init(); "
                "print(wp.get_devices())\"`, or use --backend native --device cpu."
            )
    elif backend == "native":
        if device != "cpu":
            raise BackendUnavailable(
                f"the native backend only supports device=cpu, got {device!r}.\n"
                "For a GPU use --backend warp --device cuda:0."
            )
    else:
        raise BackendUnavailable(
            f"unknown backend {backend!r}; choose warp / native / auto"
        )

    default_envs = num_envs_gpu if backend == "warp" else num_envs_cpu
    resolved_envs = num_envs if num_envs is not None else default_envs
    # The thread count is capped by the environment count here, **so that the
    # banner tells the truth**. `mujoco.rollout` requires the scratch list to be
    # exactly nthread long and the backend only has num_envs MjData copies, so a
    # cap is inevitable; capping only inside the backend would leave the banner
    # printing "threads=32" while 16 actually ran.
    threads = min(cpu_threads or _default_threads(), resolved_envs)

    notes: list[str] = []
    if backend == "warp" and device == "cpu":
        notes.append(
            "warp's cpu device is a serial debugging path, measured at roughly "
            "1/40 of multi-threaded native. Do not train with it; for CPU training "
            "use --backend native."
        )
    if training and resolved_envs != default_envs:
        notes.append(
            f"num_envs={resolved_envs} differs from this backend's default of "
            f"{default_envs}. PPO's effective batch size changes with it and the "
            "hyper-parameters need re-tuning -- changing backend is not the same as "
            "changing machine and reproducing the same policy."
        )

    return Resolution(
        backend=backend,  # type: ignore[arg-type]
        device=device,
        num_envs=resolved_envs,
        cpu_threads=threads,
        strip_visual=strip_visual,
        forced=forced,
        notes=tuple(notes),
    )
