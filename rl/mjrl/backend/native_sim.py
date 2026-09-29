"""A batched simulation backend built on native MuJoCo.

Implements the same interface as `mjlab.sim.Simulation` (MuJoCo Warp / GPU) while
running on **native MuJoCo plus a `mujoco.rollout` thread pool** across many CPU
cores.

## Why not Warp's cpu device

Warp's CPU backend compiles the same kernels to CPU code and runs them
**serially**; upstream positions it as a development and debugging device.
Measured on an i9-14900KF (32 threads) with the hexapod's hybrid collision:

| Backend | Throughput |
|---|---|
| `native:cpu` (32 threads) | 184k env-steps/s |
| `warp:cpu` | 6.2k env-steps/s |

About 30x apart, and it shows at the training level too: `warp:cpu` needs roughly
20 hours for 2000 iterations at 1024 environments, and its throughput barely grows
with environment count (37% from 64 to 1024) because the cores sit idle.

## Design

What `mujoco.rollout` guarantees (established by measurement, not read off the
documentation):

- the `data` list must be **exactly `nthread` long**; `MjData` is per-thread
  scratch
- `model` may be a single instance or a list of length nbatch (the latter is
  per-environment domain randomisation)
- batched state lives in the `initial_state` / `state` numpy arrays, which can be
  preallocated and reused
- gather and scatter happen inside rollout's C++, not in a Python loop over
  `MjData`

Thread count and environment count are **two different things** here; see
`_resolve_nthread` and the comment in `__init__`.

## Domain randomisation

MuJoCo has no batched model, but rollout's `model` parameter **accepts a list of
length nbatch**, which is what makes everything below possible.

**Not every randomisation needs model copies.** Half of them never touch
`mjModel`:

| Randomisation | Writes | Needs a model copy |
|---|---|---|
| `push_robot` | qvel | no |
| `encoder_bias` | the entity's own fields | no |
| initial-state randomisation | qpos/qvel | no |
| `foot_friction` | `geom_friction` | **yes** |
| `base_com` | `body_ipos` | **yes** |
| `pd_gains` / `effort_limits` | `actuator_*` | **yes** |

For the ones that do, `expand_model_fields(names)` creates genuinely
per-environment buffers for those fields and scatters them back into each
`MjModel` before stepping.

The overhead is negligible: `foot_friction` / `base_com` are **startup** events
(once), `pd_gains` / `effort_limits` are **reset** events (once per episode), and
**none of them is per-step**.

Memory is the only real cost. A single hexapod `MjModel` with its 31 meshes is
41.8 MiB, so 4096 copies would be 167 GiB. Almost all of that is **visual meshes**
that training never uses, and stripping them takes a copy to 3.3 MiB (13.3 GiB at
4096). **So confirm the training model carries no visual meshes before enabling
domain randisation** -- see `_maybe_strip_visuals`, which decides automatically by
memory.

The N model copies are created lazily on the first `expand_model_fields`; not
using domain randomisation costs nothing.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import mujoco
from mujoco import rollout

# ── Derived quantities gathered from MjData per environment ───────────────
# These are not in the FULLPHYSICS state rollout returns (which holds only
# time/qpos/qvel/act) but the entity/data.py layer above reads them. The list
# comes from actual usage; see docs/DESIGN.md.
_DERIVED_FIELDS: tuple[str, ...] = (
    "xpos", "xquat", "xmat", "xipos",  # link poses (xmat is the rotation matrix,
                                       # which ray casting needs)
    "cvel", "subtree_com",          # link velocities, subtree centre of mass
    "site_xpos", "site_xmat",       # site poses
    "geom_xpos", "geom_xmat",       # geom poses
    "actuator_force",               # actual actuator torque
    "qacc",                         # joint accelerations
    "ten_length", "ten_velocity",   # tendons
)

# Fields written by the layer above that must be scattered back into MjData
# before stepping
_WRITABLE_FIELDS: tuple[str, ...] = (
    "ctrl", "xfrc_applied", "mocap_pos", "mocap_quat",
)

#: These four fields must be handed to rollout **per roll**, not written into
#: MjData in advance.
#:
#: For each roll `mujoco.rollout` performs
#: `mj_setState(model, data, initial_state, mjSTATE_FULLPHYSICS)` followed by
#: `mj_setState(model, data, control, control_spec)`, and FULLPHYSICS holds only
#: time/qpos/qvel/act/history/plugin -- **`xfrc_applied` and mocap are not in it**.
#:
#: Passing only `ctrl` (rollout's default `control_spec`) and writing the other
#: three into `MjData` through `_scatter()` **has no effect at all**: measured,
#: applying 500 N to one MjData and rolling out gives displacements of **exactly
#: 0** for all 8 rolls (0.002 m when passed correctly). So domain randomisation
#: based on `xfrc_applied` -- mjlab's push-by-wrench event -- would silently never
#: happen: no error, no NaN, just an absent randomisation.
#:
#: The hexapod happens not to hit it: its `push_robot` uses the variant that writes
#: qvel (which is in FULLPHYSICS) and it has `nmocap=0`. Another task would.
#:
#: This is also a **precondition for decoupling the thread count**: with fewer
#: threads than environments the scratch MjData no longer correspond one-to-one
#: with environments, and anything "written into MjData in advance" is necessarily
#: misplaced.
_CONTROL_SPEC: int = (
    int(mujoco.mjtState.mjSTATE_CTRL)
    | int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
    | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
    | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
)


#: Which domain-randomisation events need N MjModel copies on the native backend.
#: A checklist for the configuration side: events not in NEEDS_MODEL_COPIES work
#: out of the box, events in it must go through `expand_model_fields` first (which
#: lazily creates the N copies).
DR_NO_MODEL_COPY: tuple[str, ...] = (
    "push_robot",          # writes qvel
    "encoder_bias",        # the entity's own fields
    "reset_base",          # writes qpos/qvel
    "reset_robot_joints",  # writes qpos/qvel
)

DR_NEEDS_MODEL_COPIES: tuple[str, ...] = (
    "foot_friction",   # geom_friction
    "base_com",        # body_ipos
    "pd_gains",        # actuator_gainprm / actuator_biasprm
    "effort_limits",   # actuator_forcerange
    "body_mass",       # body_mass (needs recompute_constants afterwards)
)


#: The float precision exposed to the layer above.
#:
#: **Native MuJoCo and mjlab disagree here**: MuJoCo's `mjtNum` is float64
#: (`MjData.qpos` and friends are all double), while mjlab and mjwarp are float32
#: throughout. Handing float64 upwards raises
#: `RuntimeError: expected scalar type Double but found Float` the moment it meets
#: a float32 constant -- for example in
#: `quat_apply_inverse(root_link_quat_w, gravity_vec_w)`.
#:
#: The policy is to **convert at the boundary**: expose float32 upwards (matching
#: warp bit for bit in semantics) and use float64 when talking to MuJoCo. There are
#: only three conversion points:
#:   - `_scatter` / `_gather` -- numpy slice assignment converts, no extra code
#:   - the control passed to rollout in `step()` -- an explicit astype, since
#:     rollout wants mjtNum
#:   - `_NativeModel.__getattr__` -- an explicit cast of model fields
UPCAST = np.float32

#: mjwarp gives these fields a leading dimension of length nworld; every other
#: field gets **none**.
#:
#: **Why this has to be a measured list**: the layer above indexes model fields in
#: mjwarp's layout, mixing `model.body_ipos[:, root_body_id]` (dimension 0 is
#: world) with `model.geom_bodyid[geom_id]` (no world dimension). A native
#: implementation that adds the leading dimension uniformly makes the second form
#: silently return an entire row instead of a scalar -- no error, all data wrong.
#:
#: It cannot be guessed: among the fields without a world dimension there are 29
#: float64 and 4 float32, so "integers are topology" is wrong -- `geom_matid` and
#: `geom_rgba` carry a world dimension, `geom_friction` does, and `geom_type` does
#: not.
#:
#: Regenerate with `tools/checks/warp_dims.py` on a machine with mjlab+mjwarp, and
#: rerun it after upgrading mjlab or mujoco. Measured against mjlab 1.6.0.
WORLD_BATCHED_FIELDS: frozenset[str] = frozenset((
    "actuator_acc0", "actuator_actrange", "actuator_biasprm",
    "actuator_cranklength", "actuator_ctrlrange", "actuator_dynprm",
    "actuator_forcerange", "actuator_gainprm", "actuator_gear",
    "actuator_lengthrange", "body_gravcomp", "body_inertia", "body_invweight0",
    "body_ipos", "body_iquat", "body_mass", "body_pos", "body_quat",
    "body_subtreemass", "cam_fovy", "cam_intrinsic", "cam_pos", "cam_pos0",
    "cam_poscom0", "cam_quat", "dof_armature", "dof_damping", "dof_dampingpoly",
    "dof_frictionloss", "dof_invweight0", "dof_solimp", "dof_solref", "eq_data",
    "eq_solimp", "eq_solref", "geom_dataid", "geom_friction", "geom_gap",
    "geom_margin", "geom_matid", "geom_pos", "geom_quat", "geom_rbound",
    "geom_rgba", "geom_size", "geom_solimp", "geom_solmix", "geom_solref",
    "jnt_actfrcrange", "jnt_axis", "jnt_margin", "jnt_pos", "jnt_range",
    "jnt_solimp", "jnt_solref", "jnt_stiffness", "jnt_stiffnesspoly",
    "light_active", "light_ambient", "light_attenuation", "light_castshadow",
    "light_cutoff", "light_diffuse", "light_dir", "light_dir0", "light_exponent",
    "light_pos", "light_pos0", "light_poscom0", "light_specular", "light_type",
    "mat_emission", "mat_rgba", "mat_shininess", "mat_specular", "mat_texid",
    "mat_texrepeat", "pair_friction", "pair_gap", "pair_margin", "pair_solimp",
    "pair_solref", "pair_solreffriction", "qpos0", "qpos_spring", "site_pos",
    "site_quat", "tendon_actfrcrange", "tendon_armature", "tendon_damping",
    "tendon_dampingpoly", "tendon_frictionloss", "tendon_invweight0",
    "tendon_length0", "tendon_lengthspring", "tendon_margin", "tendon_range",
    "tendon_solimp_fri", "tendon_solimp_lim", "tendon_solref_fri",
    "tendon_solref_lim", "tendon_stiffness", "tendon_stiffnesspoly",
))


class _NativeData:
    """Expose contiguous numpy buffers to the layer above as `[N, ...]` torch
    tensors.

    That layer (`mjlab/entity/data.py`) accesses them purely by attribute plus
    tensor indexing, e.g. ``self.data.qpos[env_ids, adr] = pose``, so as long as
    the attribute names and shapes line up it need not know whether the memory
    behind them is warp device memory or numpy host memory.

    The torch tensors are built with `torch.from_numpy` and **share memory** with
    the numpy buffers: writes from above land in the buffers directly, and
    `NativeSimulation` scatters them back into MjData before stepping.
    """

    def __init__(self, buffers: dict[str, np.ndarray], num_envs: int) -> None:
        object.__setattr__(self, "_np", buffers)
        object.__setattr__(
            self, "_t", {k: torch.from_numpy(v) for k, v in buffers.items()}
        )
        # mjwarp's Data has nworld (the world count) and mjlab's entity, sensor
        # and offscreen_renderer all read it. Native MjData has no such concept, so
        # it is supplied here -- without it, building an env raises AttributeError.
        object.__setattr__(self, "nworld", num_envs)

    def __getattr__(self, name: str) -> torch.Tensor:
        t = self.__dict__["_t"]
        if name not in t:
            raise AttributeError(
                f"the native backend does not provide the field {name!r}. If the "
                f"layer above genuinely needs it, add it to _DERIVED_FIELDS or "
                f"_WRITABLE_FIELDS in native_sim.py."
            )
        return t[name]

    def __setattr__(self, name: str, value) -> None:
        t = self.__dict__["_t"]
        if name in t:
            t[name].copy_(torch.as_tensor(value))
            return
        object.__setattr__(self, name, value)

    def numpy(self, name: str) -> np.ndarray:
        return self.__dict__["_np"][name]

    def keys(self):
        return self.__dict__["_t"].keys()


class _ExpandedView(torch.Tensor):
    """An expanded per-environment model field that **marks dirty on write**.

    `_scatter_model` uses the dirty flag to decide whether the buffers need moving
    into the N `MjModel` copies -- that is 8-10% of `sim.step`, and domain
    randomisation does not write those buffers during ordinary stepping at all.

    `_NativeModel.__getattr__` already marks dirty on *getting* the tensor, which
    covers how mjlab actually writes (`sim.model.<field>[env_ids] = ...`). This
    subclass catches one more case: **holding the tensor and writing to it several
    steps later**, where `__getattr__` is no longer called and only `__setitem__`
    can intercept.

    What still slips through are in-place operators like `t.copy_()` / `t.mul_()`;
    catching those too would mean `__torch_function__`, at a cost out of proportion
    to the benefit. mjlab's domain randomisation does not write that way, and
    `tests/test_native_dr.py` checks end to end that randomised values really reach
    each environment's model.
    """

    #: Class-level default: instances derived by operators do not carry this
    #: attribute and must not blow up
    _sim = None

    def __setitem__(self, key, value):
        sim = self._sim
        if sim is not None:
            sim._model_dirty = True
        return super().__setitem__(key, value)


class _SharedView(torch.Tensor):
    """A broadcast view whose environments share one buffer, **refusing every
    in-place write**.

    In a view produced by `expand`, all N environments point at the same memory, so
    writing one writes all of them. torch does **not** prevent this -- measured,
    `t[0, 0, 0] = 9.9` simply succeeds. So domain randomisation that forgets to
    call `expand_model_fields` first would silently set every environment to one
    value: you would believe each environment had a different friction coefficient
    while they were all identical, and the resulting policy would be far less
    robust than expected, with no sign of anything wrong.

    Intercepting here turns a silent error into an immediate one. Indexing and
    slicing still point at the same shared memory and stay read-only; arithmetic
    produces new memory and returns an ordinary tensor.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        inplace = (
            name == "__setitem__"
            or (name.startswith("__i") and name.endswith("__"))
            or (name.endswith("_") and not name.endswith("__"))
        )
        if inplace:
            raise RuntimeError(
                f"{name} on a shared broadcast view: this memory is shared by every "
                f"environment, so writing one writes all of them. For "
                f"per-environment domain randomisation, call "
                f"sim.expand_model_fields((field_name,)) first, after which the "
                f"field becomes a genuinely [N, ...] writable tensor."
            )
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
        if name == "__getitem__" and isinstance(ret, torch.Tensor):
            return ret.as_subclass(cls)  # still shared memory, still read-only
        if isinstance(ret, cls):
            return ret.as_subclass(torch.Tensor)
        return ret


class _NativeModel:
    """Expose `MjModel`'s fields to the layer above in mjwarp's layout.

    **The dimension rule is measured, not guessed** (see
    `WORLD_BATCHED_FIELDS`): mjwarp gives a leading dimension of length nworld to
    the 103 fields that can vary per world and **none** to the other 192 pure
    topology/index fields. The layer above therefore mixes two indexing forms:

        model.body_ipos[:, root_body_id]   # world dimension, dimension 0 is world
        model.geom_bodyid[geom_id]         # none; indexed by geom directly

    An implementation that adds the leading dimension uniformly makes the second
    form silently return an entire row instead of a scalar -- no error, all data
    wrong. Three such fields were caught in measurement: site_bodyid, geom_bodyid
    and geom_type.

    Fields with a world dimension come in two forms:

    - **After `expand_model_fields`**: a genuinely `[N, ...]` writable tensor,
      independent per environment, scattered into each `MjModel` before stepping.
      This is the path for per-environment domain randomisation.
    - **Not expanded**: an `expand` broadcast view -- zero copy, shared by every
      environment, and forced read-only by `_SharedView`; see that class.
    """

    def __init__(self, mj_model: mujoco.MjModel, num_envs: int, sim) -> None:
        object.__setattr__(self, "_m", mj_model)
        object.__setattr__(self, "_n", num_envs)
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_cache", {})

    def __getattr__(self, name: str):
        cache = self.__dict__["_cache"]
        m, n, sim = self.__dict__["_m"], self.__dict__["_n"], self.__dict__["_sim"]
        if name in cache:
            # Mark dirty on a cache hit too: what is handed out may be the tensor
            # that gets written
            if name in sim._expanded_buf:
                sim._model_dirty = True
            return cache[name]

        # Expanded: a genuinely per-environment writable tensor
        buf = sim._expanded_buf.get(name)
        if buf is not None:
            # Getting it marks dirty -- better an unnecessary scatter than a
            # missed one. `_ExpandedView` additionally catches "hold the reference
            # and write several steps later"; see that class.
            sim._model_dirty = True
            t = torch.from_numpy(buf).as_subclass(_ExpandedView)
            t._sim = sim
            cache[name] = t
            return t

        if name == "nworld":
            return n  # mjwarp's Model has this field; native MjModel does not

        val = getattr(m, name)
        if not isinstance(val, np.ndarray):
            return val  # scalars (nbody, nq, ...) are returned as-is

        if val.dtype == np.float64:
            val = val.astype(UPCAST)  # see the UPCAST note at the top of the file
        t = torch.from_numpy(np.ascontiguousarray(val))
        if name in WORLD_BATCHED_FIELDS:
            # Mirror mjwarp: add the world dimension as a shared broadcast view,
            # forced read-only
            t = t.unsqueeze(0).expand((n,) + tuple(t.shape)).as_subclass(_SharedView)
        else:
            # Pure topology/index: mjwarp adds no world dimension, so neither can this
            t = t.as_subclass(_SharedView)
        cache[name] = t
        return t

    def __setattr__(self, name, value):
        raise AttributeError(
            f"_NativeModel is read-only. For per-environment domain randomisation, "
            f"call sim.expand_model_fields(({name!r},)) first, then write "
            f"model.{name}[env_ids] = ..."
        )

    def clear_cache(self) -> None:
        self.__dict__["_cache"].clear()


#: Memory ceiling for one `MjModel` per environment. Above it, visual meshes are
#: stripped; see `_maybe_strip_visuals`. 2 GiB matches the warning threshold in
#: `_materialize_models`.
_MODEL_BUDGET_BYTES: int = 2 << 30

#: The `MJRL_STRIP_VISUAL` override: None decides by memory, True/False forces it.
_STRIP_VISUALS: bool | None = None


def set_strip_visuals(mode: bool | None) -> None:
    """Force visual-mesh stripping on or off; `None` decides by memory."""
    global _STRIP_VISUALS
    _STRIP_VISUALS = mode


def _maybe_strip_visuals(spec, num_envs: int) -> tuple:
    """Domain randomisation copies one `MjModel` per environment, and almost all
    of that is visual meshes that are never used.

    Measured on the hexapod: 44.41 MiB per copy, most of it in 31 mesh assets while
    the hybrid collision scheme uses only the 6 foot meshes. Stripped, that becomes
    **2.29 MiB (-94.8%)**:

        4096 envs   177 GiB (does not fit)  ->  9.17 GiB (fits)
         256 envs    10.3 GiB               ->  0.57 GiB

    Stripping has to happen after `CollisionCfg` (which zeroes contype on the geoms
    not selected). The spec arriving here has already been through the entity's
    spec edits, which makes this the right moment.

    ## Why it is decided by memory rather than always on

    Stripping takes `ngeom` from 83 to 27, and **what the camera sensor sees
    changes with it**: it renders the N per-environment models and therefore sees
    collision geometry rather than appearance meshes. At small scale that is a
    pointless downgrade; at large scale the alternative is being killed by the OOM
    killer, which is worse.

    The live viewer is **unaffected**: it needs only one model, so the pre-strip
    compiled model is kept here as `render_model` and the window still draws the
    real appearance. See `viewer/live.py`.

    So the test is whether the per-environment models fit. `MJRL_STRIP_VISUAL`
    overrides it.

    > native only. On warp, mjwarp manages model memory itself and never comes
    > through here.
    """
    from .model_slim import (
        model_nbytes,
        strip_noncolliding_geoms,
        strip_visual_meshes,
    )

    want = _STRIP_VISUALS
    # This compile has to happen anyway, to measure the pre-strip size. **Keep the
    # result**: it carries the full appearance meshes, which is exactly what the
    # live viewer wants to draw. Stripping modifies the spec in place and there is
    # no way back afterwards -- one 41.8 MiB model buys a viewer that draws the
    # real appearance.
    full_model = spec.compile()
    before = model_nbytes(full_model)
    if want is None:
        want = before * num_envs > _MODEL_BUDGET_BYTES
    if not want:
        return spec, False, None

    try:
        spec = strip_visual_meshes(spec)
        # Then delete the non-colliding **primitives**. Once visual meshes are
        # stripped, what the window shows is the collision geometry, and these
        # leftovers (six spheres at the hexapod's feet) get drawn at the feet and
        # suggest sphere contact -- while contact is actually made by meshes and
        # the spheres take no part in the simulation. Afterwards, what is drawn is
        # exactly what is simulated.
        spec = strip_noncolliding_geoms(spec)
    except RuntimeError as e:
        warnings.warn(
            f"stripping visual meshes failed; continuing with the full model: "
            f"{e}\n"
            f"{num_envs} MjModel copies need about "
            f"{before * num_envs / 2**30:.1f} GiB.",
            RuntimeWarning,
            stacklevel=3,
        )
        return spec, False, None
    after = model_nbytes(spec.compile())
    print(
        f"[mjrl] stripped visual meshes: MjModel {before / 2**20:.1f} -> "
        f"{after / 2**20:.1f} MiB per copy, {num_envs} copies "
        f"{before * num_envs / 2**30:.1f} -> {after * num_envs / 2**30:.2f} GiB."
        f"\n[mjrl] cost: **camera sensors** now see collision geometry rather "
        f"than appearance meshes (the live viewer is unaffected; it has its own "
        f"full model, see render_model)."
        f"\n[mjrl] Use MJRL_STRIP_VISUAL=off to disable stripping if memory allows."
    )
    return spec, True, full_model


#: The default thread count written here by `use_backend()`
#: (`MJRL_CPU_THREADS` / `--cpu_threads`).
#:
#: A global rather than a constructor argument because `NativeSimulation` is built
#: by the vendored `manager_based_rl_env.py` through `get_simulation_cls()(...)`,
#: whose signature is fixed and leaves no room for extra arguments. Same pattern
#: and same lifetime as `set_simulation_cls`.
_DEFAULT_NTHREAD: int | None = None


def set_default_nthread(n: int | None) -> None:
    """Set how many threads the next `NativeSimulation` uses. 0 or None means
    automatic, from the core count.

    Called by `mjrl.backend.select.use_backend()` **before** the env is built.
    """
    global _DEFAULT_NTHREAD
    _DEFAULT_NTHREAD = n if n else None


def _resolve_nthread(explicit: int | None, num_envs: int) -> int:
    """Resolve the thread count: explicit argument > global default > core count,
    capped by the environment count.

    The cap is required: `mujoco.rollout` needs the scratch list to be **exactly**
    nthread long and there are only num_envs MjData copies. More threads than
    environments would be pointless anyway.

    The measured optimum is **independent of the environment count** and sits well
    below the core count -- eight, on hosts with 6 and with 32 logical cores alike
    -- so the default is a ceiling rather than "every core". The measurements and
    the reason are on `resolve.DEFAULT_MAX_THREADS`, which is the single source of
    that number: this path is only reached when a `NativeSimulation` is built
    without going through `resolve` (tests, and direct use), and the two must not
    be allowed to drift apart.
    """
    n = explicit if explicit else _DEFAULT_NTHREAD
    if not n:
        from .resolve import _default_threads

        n = _default_threads()
    return max(1, min(int(n), num_envs))


class NativeSimulation:
    """Batched simulation on native MuJoCo, interface-compatible with
    `mjlab.sim.Simulation`."""

    def __init__(
        self,
        num_envs: int,
        cfg=None,
        model: mujoco.MjModel | None = None,
        device: str = "cpu",
        *,
        spec: mujoco.MjSpec | None = None,
        variant_info=None,
        nthread: int | None = None,
    ) -> None:
        """The signature matches `mjlab.sim.Simulation.__init__` **word for word**.

        That is what makes the seam work: the one construction in
        `ManagerBasedRlEnv` need not know which backend it is building, and
        swapping the class is enough (see section 3 of docs/DESIGN.md).
        """
        if device != "cpu":
            raise ValueError(
                f"NativeSimulation only supports device='cpu', got {device!r}. "
                f"For a GPU use mjlab's Simulation (MuJoCo Warp)."
            )
        if variant_info:
            raise NotImplementedError(
                "the native backend does not support per-world mesh variants. "
                "That path relies on mjwarp's per-world geom_dataid, and one "
                "MjModel per environment would be far too expensive on native. "
                "Remove the variant configuration from the scene, or use "
                "--backend warp."
            )

        stripped = False
        full_model = None
        if spec is not None:
            spec, stripped, full_model = _maybe_strip_visuals(spec, num_envs)
            mj_model = spec.compile()
        elif model is not None:
            mj_model = model
        else:
            raise ValueError("either model or spec must be given.")

        # Apply SimulationCfg's mujoco section to the model, as mjlab does.
        # **Not optional**: timestep, solver and gravity are set here, and omitting
        # it means the two backends run different physics -- which a consistency
        # comparison would then attribute to the wrong cause.
        if cfg is not None and getattr(cfg, "mujoco", None) is not None:
            cfg.mujoco.apply(mj_model)
            if full_model is not None:
                cfg.mujoco.apply(full_model)

        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._mj_model = mj_model
        self._sensor_context = None
        self._live_viewer = None
        #: Were visual meshes stripped? The live viewer uses this to decide whether
        #: to turn on the collision-geometry group -- without it, all that is left
        #: after stripping is the six foot meshes and the view is nothing but toes.
        self.visuals_stripped = stripped
        #: The full model with appearance meshes, for drawing only. **One copy.**
        #: See render_model.
        self._render_model = full_model
        #: Whether the expanded model buffers may have been touched (see
        #: _scatter_model). True initially: the expanded values have to reach each
        #: environment's model before the first step.
        self._model_dirty = True
        #: The camera off-screen renderer, built only when the scene has camera
        #: sensors (see set_sensor_context)
        self._camera_renderer = None

        # Thread count and environment count are **two different things**, and are
        # deliberately no longer tied together.
        #
        # `self._datas` in fact serves two roles at once:
        #
        #   | Role | How many | Who uses it |
        #   |---|---|---|
        #   | rollout's **per-thread** scratch | nthread | worker threads |
        #   | **per-environment** storage for derived quantities | num_envs | main thread |
        #
        # rollout's own documentation says `data` is nthread long and `model` is
        # nbatch long; it was this one list that bound the two together. Only the
        # first nthread entries are handed over as scratch now.
        #
        # Deriving quantities no longer depends on what the rollout threads leave
        # behind: they are recomputed by the main thread's own mj_forward loop at
        # the end of step().
        #
        # The cost of binding them is measured (i9-14900KF, 24 cores / 32 threads):
        # 4096 environments would mean 4096 threads, and a single rollout at 256
        # threads takes 2.6x what 24 threads do; at 256 environments it is 19.2 ms
        # against 2.2 ms, **slower than a single thread**. It also turns "use all
        # the cores" into "change num_envs" -- and num_envs is PPO's batch size, so
        # tuning performance would mean changing the learning problem.
        self._nthread = _resolve_nthread(nthread, num_envs)

        self._datas = [mujoco.MjData(mj_model) for _ in range(num_envs)]
        for d in self._datas:
            mujoco.mj_resetData(mj_model, d)
            mujoco.mj_forward(mj_model, d)

        self._pool = rollout.Rollout(nthread=self._nthread)
        # The thread pool for recomputing derived quantities. **This is native's
        # dominant cost**: on the real hexapod model, 64 environments take 13-17 ms
        # serially while rollout takes 1-2 ms -- the serial part is over 85%. The
        # work is independent per environment with no shared writes, and MuJoCo's
        # bindings **release the GIL** during mj_forward (measured 10.4x at 24
        # threads), so Python threads really are parallel here. The pool is
        # persistent rather than rebuilt each step -- building one per step costs
        # more than it saves.
        self._fwd_pool = (
            ThreadPoolExecutor(max_workers=self._nthread, thread_name_prefix="mjrl-fwd")
            if self._nthread > 1
            else None
        )
        # Each thread takes a contiguous span, reducing dispatch overhead
        _b = np.linspace(0, num_envs, self._nthread + 1).astype(int)
        self._fwd_chunks = [
            range(int(_b[i]), int(_b[i + 1]))
            for i in range(self._nthread)
            if _b[i] < _b[i + 1]
        ]
        self._nstate = mujoco.mj_stateSize(
            mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        # ── Preallocated buffers ──────────────────────────────────────────
        m, N = mj_model, num_envs
        buffers: dict[str, np.ndarray] = {
            "qpos": np.zeros((N, m.nq), dtype=UPCAST),
            "qvel": np.zeros((N, m.nv), dtype=UPCAST),
            "qacc": np.zeros((N, m.nv), dtype=UPCAST),
            "ctrl": np.zeros((N, m.nu), dtype=UPCAST),
            "time": np.zeros(N, dtype=UPCAST),
            "sensordata": np.zeros((N, m.nsensordata), dtype=UPCAST),
            "xpos": np.zeros((N, m.nbody, 3), dtype=UPCAST),
            "xquat": np.zeros((N, m.nbody, 4), dtype=UPCAST),
            # Body rotation matrices. The ray sensor's prepare_rays reads
            # `data.xmat` for body-frame poses (sites and geoms have their own
            # *_xmat), and native previously provided only xquat, so a
            # body-framed ray sensor raised AttributeError outright.
            "xmat": np.zeros((N, m.nbody, 9), dtype=UPCAST),
            "xipos": np.zeros((N, m.nbody, 3), dtype=UPCAST),
            "cvel": np.zeros((N, m.nbody, 6), dtype=UPCAST),
            "subtree_com": np.zeros((N, m.nbody, 3), dtype=UPCAST),
            "xfrc_applied": np.zeros((N, m.nbody, 6), dtype=UPCAST),
            "site_xpos": np.zeros((N, m.nsite, 3), dtype=UPCAST),
            "site_xmat": np.zeros((N, m.nsite, 9), dtype=UPCAST),
            "geom_xpos": np.zeros((N, m.ngeom, 3), dtype=UPCAST),
            "geom_xmat": np.zeros((N, m.ngeom, 9), dtype=UPCAST),
            "actuator_force": np.zeros((N, m.nu), dtype=UPCAST),
            "mocap_pos": np.zeros((N, max(m.nmocap, 1), 3), dtype=UPCAST),
            "mocap_quat": np.zeros((N, max(m.nmocap, 1), 4), dtype=UPCAST),
            "ten_length": np.zeros((N, max(m.ntendon, 1)), dtype=UPCAST),
            "ten_velocity": np.zeros((N, max(m.ntendon, 1)), dtype=UPCAST),
        }
        # Domain-randomisation state: without DR, _models stays None and the N
        # model copies are never paid for
        self._models: list[mujoco.MjModel] | None = None
        self._expanded: set[str] = set()
        self._expanded_buf: dict[str, np.ndarray] = {}
        self._default_fields: dict[str, torch.Tensor] = {}
        self._needs_setconst = False
        #: A scratch MjData dedicated to mj_setConst -- it wipes the qpos of
        #: whatever is passed to it, so it must never be one of self._datas.
        #: Allocated on demand; without DR it is never built.
        self._const_scratch: mujoco.MjData | None = None

        self.data = _NativeData(buffers, num_envs)
        self.model = _NativeModel(mj_model, num_envs, self)
        self._build_copy_plans()

        # rollout's input/output buffers, reused for the lifetime of the sim.
        # These are **fed to rollout** and must be MuJoCo's mjtNum (float64).
        # `np.zeros` defaults to float64, but that is incidental rather than
        # guaranteed -- stating it explicitly also lets
        # test_native_exposes_float32 pin it down. The buffers exposed upwards are
        # float32 (UPCAST); see the UPCAST note for the boundary between them.
        self._state_in = np.zeros((N, self._nstate), dtype=np.float64)
        # The solver's warm start, per environment. `mjSTATE_FULLPHYSICS` (8223)
        # does not include `WARMSTART` (32), and rollout **zeroes** `qacc_warmstart`
        # for every roll unless `initial_warmstart` is given (measured on 3.11: the
        # result is bitwise independent of what the scratch MjData held, and equal
        # to passing zeros). Passed per roll, each environment's solver starts from
        # its own last solution, as it would stepping its own MjData.
        #
        # **It has to be environment i's own.** It is read from `_datas[i]` in
        # step(), and the first nthread of those are rollout's scratch: they come
        # back holding the warm start and the ctrl of whichever roll their thread
        # ran last -- scheduling, not i. Read back as they stood, results depended
        # on the thread schedule and the thread count, and the first nthread
        # environments had their actuator and contact forces recomputed from
        # another environment's ctrl. `_refresh_derived` puts i's own back; what it
        # was measured to do before is in docs/DESIGN.md 5.4, and
        # tests/test_native_determinism.py holds it.
        #
        # The field is `qacc_warmstart`. rollout's docstring calls it
        # `qfrc_warmstart`, which is an upstream typo -- MjData has no such
        # attribute.
        self._warmstart_in = np.zeros((N, m.nv), dtype=np.float64)
        self._state_out = np.zeros((N, 1, self._nstate), dtype=np.float64)
        self._sensor_out = np.zeros((N, 1, m.nsensordata), dtype=np.float64)
        # The control width is determined by _CONTROL_SPEC (it is more than nu).
        # Compute it with mj_stateSize rather than by hand, and leave the packing
        # order to mj_getState, so the layout cannot be written wrongly.
        self._ncontrol = mujoco.mj_stateSize(m, _CONTROL_SPEC)
        self._control = np.zeros((N, 1, self._ncontrol), dtype=np.float64)

        self._gather()

    # ── Internals: buffers <-> MjData ──────────────────────────────────────

    def _build_copy_plans(self) -> None:
        """Resolve the fields `_scatter` / `_gather` touch, once.

        Both are **pure Python loops that never release the GIL**, so parallelising
        them buys nothing; the only saving available is the Python work itself. And
        they are the largest single item in `sim.step()` -- measured, `_gather` is
        32% at 64 environments and 36% at 256.

        The original code did three lookups per field per environment in the
        innermost loop (two for the shape, one for the assignment): about 2880 for
        64 environments and 14 fields, and 14400 for the five gathers in one
        control step (cProfile measured `numpy()` called 294400 times over 20
        steps). Resolving buffer references and target shapes up front leaves one
        `getattr` and one assignment in the inner loop.

        Buffers are never reallocated after construction (`expand_model_fields`
        only touches the model side), so caching the references is safe.
        """
        np_ = self.data.numpy
        d0 = self._datas[0]
        # gather: the four fixed fields first, then the derived quantities
        self._gather_head = tuple(
            (f, np_(f)) for f in ("qpos", "qvel", "time", "sensordata")
        )
        plan = []
        for f in _DERIVED_FIELDS:
            buf = np_(f)
            src = getattr(d0, f, None)
            # When the model has none of a given element (no tendons, say), the
            # MjData array is empty -- tested once here rather than in the inner
            # loop
            if src is None or (buf.shape[1:] == (1,) and src.size == 0):
                continue
            plan.append((f, buf, buf.shape[1:]))
        self._gather_plan = tuple(plan)
        # scatter: mocap is written only when the model has mocap bodies
        fields = ["qpos", "qvel", "ctrl", "xfrc_applied"]
        if self._mj_model.nmocap:
            fields += ["mocap_pos", "mocap_quat"]
        self._scatter_plan = tuple((f, np_(f)) for f in fields)

    def _gather_range(self, idx) -> None:
        """Gather state and derived quantities for the environments in `idx`.

        Factored out so that the **parallel `_refresh_derived` could fold it in**
        -- see the note there. The serial `_gather()` goes through the same path,
        so there is only one implementation.
        """
        datas, head, plan = self._datas, self._gather_head, self._gather_plan
        for i in idx:
            d = datas[i]
            for f, buf in head:
                buf[i] = getattr(d, f)
            for f, buf, shp in plan:
                buf[i] = getattr(d, f).reshape(shp)

    def _scatter(self) -> None:
        """Scatter what the layer above wrote into the buffers back into MjData."""
        for i, d in enumerate(self._datas):
            for f, buf in self._scatter_plan:
                getattr(d, f)[:] = buf[i]

    def _gather(self) -> None:
        """Gather state and derived quantities from every MjData into the
        contiguous buffers (all environments, serial)."""
        self._gather_range(range(self.num_envs))

    # ── Public interface (matching mjlab.sim.Simulation) ──────────────────

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    def step(self, nstep: int = 1) -> None:
        """Advance nstep physics steps. Whatever the layer above wrote to ctrl and
        friends is scattered into effect here."""
        self._scatter()
        # Both reads come from the **same** MjData, with the packing order left to
        # mj_getState:
        #   - state_in -- the initial state (FULLPHYSICS)
        #   - control  -- ctrl + xfrc_applied + mocap (see _CONTROL_SPEC)
        # The latter cannot be left in MjData in advance: there they would follow
        # the thread rather than the environment.
        ctrl_buf = np.zeros(self._ncontrol)
        for i, d in enumerate(self._datas):
            mujoco.mj_getState(
                self._mj_model, d, self._state_in[i], mujoco.mjtState.mjSTATE_FULLPHYSICS
            )
            mujoco.mj_getState(self._mj_model, d, ctrl_buf, _CONTROL_SPEC)
            self._control[i, 0] = ctrl_buf
            self._warmstart_in[i] = d.qacc_warmstart
        # rollout works in MuJoCo's mjtNum (float64)
        control = np.repeat(self._control, nstep, axis=1)
        state_out = np.zeros((self.num_envs, nstep, self._nstate), dtype=np.float64)
        sensor_out = np.zeros(
            (self.num_envs, nstep, self._mj_model.nsensordata), dtype=np.float64
        )
        self._scatter_model()
        self._pool.rollout(
            self._models if self._models is not None else self._mj_model,
            # Only nthread scratch copies are needed (rollout's convention), not
            # one per environment. They are overwritten from state_out immediately
            # afterwards, so rollout dirtying them does not matter.
            self._datas[: self._nthread],
            self._state_in,
            control,
            control_spec=_CONTROL_SPEC,
            initial_warmstart=self._warmstart_in,
            nstep=nstep,
            state=state_out,
            sensordata=sensor_out,
        )

        # **Environment i's result must not be read from `self._datas[i]`.**
        #
        # `mujoco.rollout` dispatches rolls to threads from a work queue, and
        # `data` is only **per-thread scratch** -- `_datas[i]` holds whatever roll
        # that thread processed last, which has nothing to do with i. Only the
        # `state` and `sensordata` output arrays are indexed by roll (that is, by
        # environment).
        #
        # Gathering directly from `_datas` **reshuffles the environments every
        # step**: the physics is computed correctly and stored in the wrong slots.
        # In training that is catastrophic (observations from environment A paired
        # with actions meant for B), and it **raises nothing and produces no NaN**
        # -- the policy simply does not learn. It was caught by comparing against
        # warp per environment: for four environments the result was
        # native[0]=warp[3], [1]=[1], [2]=[0], [3]=[2] -- a permutation.
        #
        # So state is always read back from `state_out` and derived quantities are
        # recomputed by a per-environment `mj_forward` (xpos, site_xpos, cvel and
        # friends exist only inside MjData and are not in the output arrays). The
        # cost is one extra forward per environment per step; correctness is not
        # negotiable.
        self._refresh_derived(state_out[:, -1])
        self._gather()
        if self._live_viewer is not None:
            self._live_viewer.maybe_sync()

    def _refresh_derived(self, final_state: np.ndarray) -> None:
        """Write each environment's final state back into **its own** MjData and
        recompute the derived quantities.

        `xpos`, `site_xpos`, `cvel` and friends exist only inside `MjData` and are
        not in rollout's output arrays, so they have to be recomputed. The work is
        independent per environment with no shared writes -- distinct
        `(MjModel, MjData)` pairs share no mutable state and `mj_forward` is safe
        to run concurrently on them.

        `final_state` is indexed by **roll (that is, by environment)**, which is
        the trustworthy source; it must never be read from `self._datas[i]`, which
        is indexed by thread (see the note in step()).

        **Gathering does not belong here.** Folding `_gather` into this parallel
        loop was tried, on the reasoning that `mj_forward` is C and releases the
        GIL while gathering is Python and does not, so interleaving should take the
        total from `forward/threads + gather` to `max(...)`. **It measured 17-19%
        slower** (at both 64 and 256 environments) and was reverted.

        The plausible explanation is GIL handoff: a worker doing Python gathering
        holds the GIL, and 32 threads contending for it costs more than the overlap
        saves. Split into two passes, the parallel section is nearly pure C with
        minimal GIL traffic and gathering is one clean serial pass.

        Recorded here so it is not tried a second time.

        **Putting each environment's control and warm start back costs** one
        `mj_setState` and two small copies per environment, under the GIL. Measured
        on the i9-14900KF (2026-09-26), jumper.tripod on 8 threads, medians of 5
        runs interleaved with the code before it, before -> after:

        | envs | `sim.step` ms | `env.step` ms | this function ms |
        |---|---|---|---|
        | 64 | 2.79 -> 2.94 | 23.6 -> 24.4 | 0.85 -> 1.02 |
        | 128 | 6.30 -> 6.43 | 46.1 -> 45.9 | 1.97 -> 2.13 |

        Two more batches, taken beside a GPU training, put `sim.step` at +4-10%
        for 64 environments and +1-3% for 128, and `env.step` within +5%. The
        rollout's time did not move. Carrying the warm start barely changes the
        solver's work on this model (Newton: 3.8-3.9 iterations per rollout
        forward, warm or cold; 3.8 against 3.3 for this function's own). Starting
        every forward cold instead is as deterministic and measured between 1%
        dearer and 4% cheaper in `sim.step` -- inside the spread between batches.
        The warm start is kept because it is what `mj_step` on one MjData does.
        """
        models = (
            self._models
            if self._models is not None
            else [self._mj_model] * self.num_envs
        )
        datas, full = self._datas, mujoco.mjtState.mjSTATE_FULLPHYSICS
        control, warmstart = self._control, self._warmstart_in

        def work(idx: range) -> None:
            for i in idx:
                d = datas[i]
                mujoco.mj_setState(models[i], d, final_state[i], full)
                # **The state is not all rollout left behind.** The first nthread
                # MjData were its scratch and still hold the control and the warm
                # start of whichever roll their thread ran last, so the forward
                # below would compute environment i's actuator forces from another
                # environment's ctrl, and the next step would start environment i's
                # solver from another environment's accelerations. Put back i's own,
                # as step() read them before the rollout.
                mujoco.mj_setState(models[i], d, control[i, 0], _CONTROL_SPEC)
                d.qacc_warmstart[:] = warmstart[i]
                mujoco.mj_forward(models[i], d)
                # The warm start carried into the next step: the solution just
                # found at the state that step starts from. mj_forward does not
                # store it (mj_step does, in its integrator) -- see `_warmstart_in`.
                d.qacc_warmstart[:] = d.qacc

        if self._fwd_pool is None:
            work(range(self.num_envs))
        else:
            # list() is required: map is lazy and would not wait for completion
            list(self._fwd_pool.map(work, self._fwd_chunks))

    def forward(self) -> None:
        """Refresh derived quantities after a state write, without advancing time."""
        self._scatter()
        self._scatter_model()
        models = self._models if self._models is not None else [self._mj_model] * self.num_envs
        datas = self._datas

        def work(idx: range) -> None:
            for i in idx:
                mujoco.mj_forward(models[i], datas[i])

        if self._fwd_pool is None:
            work(range(self.num_envs))
        else:
            list(self._fwd_pool.map(work, self._fwd_chunks))
        self._gather()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = range(self.num_envs) if env_ids is None else env_ids.tolist()
        for i in ids:
            mujoco.mj_resetData(self._mj_model, self._datas[int(i)])
        self._gather()

    def sense(self) -> None:
        """Run the sensing pipeline: prepare -> intersect -> finalize.

        Structurally identical to mjlab's `Simulation.sense()`; only the two
        computations in the middle have different implementations. Ray casting goes
        from mjwarp's BVH kernel to `mujoco.mj_ray` (dispatched in
        `RayCastSensor.raycast_kernel`; see the [mjrl] marker there), and cameras
        go from `mjwarp.render()` to off-screen `mujoco.Renderer` (see
        `mjrl/sensor/camera.py`). The order matches mjlab -- cameras then rays --
        and the two do not depend on each other.

        Called once per control step, before observations are computed.
        """
        ctx = self._sensor_context
        if ctx is None:
            return
        ctx.prepare()  # pure PyTorch: transform the rays into world coordinates
        if self._camera_renderer is not None:
            models = self._models if self._models is not None else [self._mj_model] * self.num_envs
            self._camera_renderer.render_into(models, self._datas)
        for sensor in ctx.raycast_sensors:
            sensor.raycast_kernel()
        ctx.finalize()  # pure PyTorch: turn distances into hit points


    @property
    def render_model(self) -> mujoco.MjModel:
        """The model used for drawing: **with the appearance meshes intact**, kept
        separate from the one physics uses.

        Physics runs on `mj_model` with its visual meshes stripped (13.26 GiB for
        4096 copies); this is **one copy** (41.8 MiB on the hexapod, 0.3% of that)
        used only by the live viewer for `mjv_addGeoms`, and it takes no part in
        any simulation.

        qpos and qvel are interchangeable between the two bit for bit: stripping
        deletes only mesh geoms whose contype and conaffinity are both zero, along
        with the mesh assets they orphan -- **not one body, joint or degree of
        freedom is touched** (and `strip_visual_meshes` verifies per-body mass and
        inertia are unchanged, raising if they are not).

        Without stripping this is `mj_model` itself, so callers need no special
        case.
        """
        return self._render_model if self._render_model is not None else self._mj_model

    def env_mjdata(self, index: int = 0) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Return environment `index`'s **live** `MjModel` / `MjData`, zero-copy.

        Taking a frame is **cheap** on native: the state is already in host memory
        and already an `MjData`, so the live viewer copies qpos/qvel out of it. The
        warp path can render live too (mjlab's play does), but every frame has to
        bring qpos/qvel and friends back from the device. So the difference is
        **cost**, not capability -- see the comparison table at the top of
        `mjrl/viewer/live.py`.

        Note that this returns **a reference, not a snapshot**: the next `step()`
        overwrites it in place. Copy out of it to keep a frame -- which is exactly
        what the live viewer does, on the simulation thread, because it draws on
        another.

        Once domain randomisation has expanded the model, this returns that
        environment's own `MjModel`.
        """
        if not 0 <= index < self.num_envs:
            raise IndexError(
                f"environment index {index} out of range (of {self.num_envs})"
            )
        model = self._models[index] if self._models is not None else self._mj_model
        return model, self._datas[index]

    def attach_viewer(self, viewer) -> None:
        """Register a live viewer; `step()` calls its `maybe_sync()` at the end.

        The frame is **taken** here, on the simulation's own thread: `step()` has
        just finished `_gather()`, `MjData` is in a consistent state, and a copy
        taken now cannot tear. A thread reading live state on a timer could catch
        rollout halfway through writing it. What the viewer does with the copy --
        `mj_forward`, drawing, `handle.sync()` -- happens on a thread of its own
        and costs this one nothing; see `mjrl/viewer/live.py`.
        """
        self._live_viewer = viewer

    def set_sensor_context(self, ctx) -> None:
        """Register the sensor context and wire each kind of sensor to its native
        implementation.

        **This must never be a no-op.** Accepting the context here without wiring
        it means `sense()` never actually computes sensors: a height scan keeps
        reading its **initial values**, nothing raises, nothing goes NaN, and the
        policy trains to convergence on garbage.

        The wiring itself is shallow:

        - **Rays**: hand each sensor the single `MjModel` and the per-environment
          `MjData` that its `raycast_kernel` needs on the native branch (see
          `mjrl/sensor/raycast.py`).
        - **Cameras**: build a `NativeCameraRenderer`. The pixel buffers have
          already been allocated in mjwarp's layout by the native branch of
          `SensorContext` (see `mjrl/sensor/camera.py`).
        """
        self._sensor_context = ctx
        for sensor in getattr(ctx, "raycast_sensors", ()) or ():
            sensor._native_mj_model = self._mj_model
            sensor._native_datas = self._datas

        if getattr(ctx, "camera_sensors", None):
            from mjrl.sensor.camera import NativeCameraRenderer

            self._camera_renderer = NativeCameraRenderer(ctx, self._mj_model)

    def close(self) -> None:
        # The camera's GL context must be released explicitly rather than left to
        # __del__: EGL raises EGLError on its cleanup path at interpreter exit,
        # which looks like a crash. See NativeCameraRenderer.close.
        if self._camera_renderer is not None:
            self._camera_renderer.close()
            self._camera_renderer = None
        self._pool.close()
        if self._fwd_pool is not None:
            self._fwd_pool.shutdown(wait=True)
            self._fwd_pool = None

    # ── Domain randomisation ──────────────────────────────────────────────

    @property
    def expanded_fields(self) -> set[str]:
        return self._expanded

    @property
    def per_world_default_fields(self) -> set[str]:
        return set()

    @property
    def default_model_fields(self) -> dict[str, torch.Tensor]:
        return self._default_fields

    def get_default_field(self, field: str) -> torch.Tensor:
        """Return the field's original value, **before randomisation**.

        Domain randomisation's `_select_default_values` uses it as the baseline for
        scaling and offsets, so it must come from the unmodified shared model
        rather than from the current (possibly already randomised) value.
        """
        if field not in self._default_fields:
            if not hasattr(self._mj_model, field):
                raise ValueError(f"the model has no field {field!r}")
            self._default_fields[field] = torch.as_tensor(
                np.array(getattr(self._mj_model, field)), dtype=torch.float32
            ).clone()
        return self._default_fields[field]

    def expand_model_fields(self, fields: tuple[str, ...]) -> None:
        """Expand the given model fields into per-environment writable buffers.

        The N `MjModel` copies are created lazily on the first call, so not using
        domain randomisation costs no memory. Afterwards these fields are scattered
        back into their models before every step.
        """
        if not fields:
            return
        missing = [f for f in fields if not hasattr(self._mj_model, f)]
        if missing:
            raise ValueError(f"the model has no such fields: {missing}")
        # Pure topology/index fields have no world dimension in mjwarp either, and
        # changing one per environment would mean a different model structure
        # (degree-of-freedom count and tree shape would change), invalidating the
        # batched layout above immediately. Rejected here rather than allowed to
        # produce an error that is hard to locate.
        not_batched = [f for f in fields if f not in WORLD_BATCHED_FIELDS]
        if not_batched:
            raise ValueError(
                f"these fields cannot be expanded per environment: {not_batched}. "
                f"They are pure topology/index fields, and mjwarp gives them no "
                f"world dimension either (see WORLD_BATCHED_FIELDS)."
            )

        if self._models is None:
            self._materialize_models()

        for f in fields:
            if f in self._expanded:
                continue
            base = np.array(getattr(self._mj_model, f))
            if base.dtype == np.float64:
                base = base.astype(UPCAST)
            buf = np.repeat(base[None, ...], self.num_envs, axis=0).copy()
            self._expanded_buf[f] = buf
            self._expanded.add(f)
        self._model_dirty = True
        self.model.clear_cache()

    def _materialize_models(self) -> None:
        """Create N independent MjModel copies.

        **Memory is the main constraint on this path, and almost all of it goes on
        things that should not be copied.** Measured on the hexapod (mjlab 1.6.0):

            total per copy      43.02 MB
              bvh_*             29.88 MB  (69.4%)   bvh_aabb alone is 22.41 MB
              mesh_*            12.80 MB  (29.7%)
              real physics      0.08 MB  ( 0.2%)

        Meshes and BVHs are identical across environments and copying them is pure
        waste, but `deepcopy` does not know that. Only 6 of the 31 meshes (the
        feet) take part in collision, accounting for 4% of the faces; the other 25
        are visual-only and headless CPU training never uses them.

            before stripping  43.02 MB per copy -> 22.03 GB at N=512 (infeasible)
            after stripping   ~1.79 MB per copy ->  0.92 GB at N=512 (feasible)

        `_maybe_strip_visuals` handles this automatically when a spec is given; the
        warning below covers the case where a model was passed in with its visual
        meshes intact.
        """
        est = sum(
            v.nbytes for k in dir(self._mj_model)
            if not k.startswith("_")
            and isinstance((v := getattr(self._mj_model, k, None)), np.ndarray)
        )
        if est * self.num_envs > 2 << 30:  # > 2 GiB
            warnings.warn(
                f"materialising {self.num_envs} MjModel copies needs about "
                f"{est * self.num_envs / 2**30:.1f} GiB "
                f"({est / 2**20:.1f} MiB each). This model most likely still "
                f"carries visual-only meshes, which are identical across "
                f"environments and pure waste to copy. Pass a training model with "
                f"them stripped.",
                RuntimeWarning,
                stacklevel=2,
            )
        import copy as _copy

        self._models = [_copy.deepcopy(self._mj_model) for _ in range(self.num_envs)]

    def _scatter_model(self) -> None:
        """Write the expanded fields' per-environment values back into their models.

        **Only done when the buffers may have been touched.** This step is a
        `getattr` plus slice assignment for 8 fields across N models, measured at
        8.7% of `sim.step` at 64 environments and 10.4% at 256 -- while domain
        randomisation writes those buffers only during startup / reset / interval
        events and **never touches them** during ordinary stepping (measured: zero
        accesses over 100 control steps). Doing it every step means moving the same
        values 512 times.

        The dirty flag hangs off `_NativeModel.__getattr__`: getting that tensor is
        the **only entry point** for reading *or* writing, so "mark dirty on get"
        cannot miss a write. The cost is that reads mark dirty too and cause one
        unnecessary scatter -- an error in the safe direction. The one case it does
        miss is holding a tensor reference and writing several steps later, which
        `tests/test_native_dr.py` guards with an end-to-end check of the randomised
        values.
        """
        if not self._expanded or self._models is None:
            return
        if not self._model_dirty and not self._needs_setconst:
            return
        self._model_dirty = False
        for f, buf in self._expanded_buf.items():
            for i, mi in enumerate(self._models):
                getattr(mi, f)[:] = buf[i]
        if self._needs_setconst:
            # mj_setConst recomputes stat along with everything else, and three of
            # those -- extent / center / meansize -- are **purely visual** and often
            # set explicitly by the user (mjlab's SceneCfg.extent writes into
            # spec.stat.extent; the hexapod sets 2.0). Letting them be recomputed
            # from geometry discards that setting silently: measured, 2.0 becomes
            # 0.6322 on the hexapod, while mjwarp does not recompute stat per world
            # at all and keeps the nominal value.
            #
            # The consequence is not a slightly uglier picture: MuJoCo derives
            # znear and zfar from `vis.map.* * stat.extent`, so each
            # per-environment model's **frustum** differs from the nominal one by
            # 3.16x and camera depth is scaled by the same factor -- with no error,
            # no NaN, and a depth map that looks entirely normal. LiveViewer takes
            # these models through env_mjdata and is affected too.
            #
            # meanmass / meaninertia are not restored: they participate in the
            # solver's scale normalisation and *should* follow randomised masses,
            # which is the whole point of calling mj_setConst.
            #
            # **The second argument must be scratch, never a live MjData.**
            # `mj_setConst` treats `d` as a workspace: it sets `qpos` to `qpos0`,
            # runs kinematics to compute constants like `body_invweight0`, and
            # **restores nothing afterwards**. Passing a live one wipes every
            # environment's pose:
            #
            #     d.qpos[:3] = [1.43, -1.23, 0.14] → mj_setConst → [0, 0, 0]
            #
            # Nothing about this raises. What it looks like is **every one of 4096
            # robots piled at the world origin the moment training starts**, then
            # migrating to its grid cell one at a time as episodes time out and
            # reset -- which reads as "the rendering gradually fixed itself" but is
            # physics state that was erased once. native only: on warp the model
            # constants are computed in put_model and never touch live data.
            stat = self._mj_model.stat
            keep = (float(stat.extent), np.array(stat.center), float(stat.meansize))
            if self._const_scratch is None:
                self._const_scratch = mujoco.MjData(self._mj_model)
            scratch = self._const_scratch
            for mi in self._models:
                mujoco.mj_setConst(mi, scratch)
                mi.stat.extent, mi.stat.center[:], mi.stat.meansize = (
                    keep[0], keep[1], keep[2],
                )
            self._needs_setconst = False

    def recompute_constants(self, level=None) -> None:
        """Derived constants need recomputing after mass or inertia fields change."""
        del level
        self._needs_setconst = True
