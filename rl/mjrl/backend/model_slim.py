"""Strip visual-only meshes down to a model that only physics needs.

**Why this exists**: per-environment domain randomisation needs one independent
`MjModel` per environment (see
`native_sim.NativeSimulation.expand_model_fields`), and almost all of a single
model's footprint goes on things that are identical across environments. Measured
on the hexapod:

    total per copy      43.02 MB
      bvh_*             29.88 MB  (69.4%)   bvh_aabb alone is 22.41 MB
      mesh_*            12.80 MB  (29.7%)
      real physics      0.08 MB  ( 0.2%)

Only 6 of the 31 meshes (the feet) take part in collision, and they account for 4%
of the faces. The other 25 are visual-only, are never used by headless CPU
training, and take the model copies for N=512 from 0.92 GB to 22.03 GB.

**Why deleting geoms is safe, and why it is still checked**: when a body has no
explicit `<inertial>`, MuJoCo **derives** its mass and inertia from its geoms, so
on such a model deleting visual geoms **silently changes the dynamics**. The
hexapod's 31 bodies all carry an explicit `<inertial>` (they come from URDF) and
are unaffected -- but that is a property of this model, not a general rule. So
this module **measures every time** whether per-body mass, centre of mass and
inertia are unchanged, and raises when they are not, rather than relying on
someone remembering that a model has inertials.
"""

from __future__ import annotations

import numpy as np

import mujoco

__all__ = ["strip_visual_meshes", "model_nbytes"]


def model_nbytes(model: mujoco.MjModel) -> int:
    """Total bytes of every numpy array in the model."""
    total = 0
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            val = getattr(model, name)
        except Exception:
            continue
        if isinstance(val, np.ndarray):
            total += val.nbytes
    return total


def _collides(model: mujoco.MjModel, i: int) -> bool:
    return bool(model.geom_contype[i] or model.geom_conaffinity[i])


def strip_noncolliding_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Delete **every** non-colliding geom, not just meshes.

    `strip_visual_meshes` deliberately touches only meshes -- its goal was memory,
    and the memory is all in mesh assets. But it leaves behind a set of
    **non-colliding primitives**: `build_jumper.py` emits several collision schemes
    at once and `CollisionCfg(disable_other_geoms=True)` zeroes contype and
    conaffinity on the ones not selected, leaving the geometry itself in place. On
    the hexapod that is six spheres at the feet.

    **Why they have to go**: once visual meshes are stripped, what the live viewer
    shows *is* the collision geometry. Those six spheres get drawn at the feet and
    suggest the feet make sphere contact -- while the hybrid scheme makes mesh
    contact and the spheres take no part in the simulation at all. **Present in the
    picture, absent from the physics** is worse than not being drawn.

    Afterwards the geoms in the model are exactly the geoms being simulated, and
    what the viewer draws is what is being computed.

    As in `strip_visual_meshes`, per-body mass, centre of mass and inertia are
    checked afterwards: if some body has no explicit `<inertial>` and its inertia
    was derived from these geoms, deleting them would change the dynamics, and
    that raises rather than silently altering the simulation.

    Raises:
        RuntimeError: stripping changed the per-body dynamics, or the number of
            colliding geoms changed.
    """
    before = spec.compile()
    ref = {
        f: np.array(getattr(before, f))
        for f in ("body_mass", "body_ipos", "body_iquat", "body_inertia")
    }
    n_col_before = sum(_collides(before, i) for i in range(before.ngeom))

    doomed = {
        before.geom(i).name
        for i in range(before.ngeom)
        if not _collides(before, i)
    }
    if not doomed:
        return spec
    for body in spec.bodies:
        for g in list(body.geoms):
            if g.name in doomed:
                spec.delete(g)

    after = spec.compile()
    for field, base in ref.items():
        got = np.asarray(getattr(after, field))
        if got.shape != base.shape or np.abs(got - base).max() > 1e-9:
            raise RuntimeError(
                f"deleting non-colliding geoms changed {field} -- some body in "
                f"this model has no explicit <inertial> and its inertia is derived "
                f"from geoms. Add <inertial> first."
            )
    n_col_after = sum(_collides(after, i) for i in range(after.ngeom))
    if n_col_after != n_col_before:
        raise RuntimeError(
            f"the number of colliding geoms went from {n_col_before} to "
            f"{n_col_after} -- the wrong things were deleted."
        )
    return spec


def strip_visual_meshes(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Delete non-colliding mesh geoms in place, along with the mesh assets they
    orphan.

    Only geoms of **mesh type** whose contype and conaffinity are both zero are
    touched -- capsules, spheres and boxes are left alone, as is anything that
    collides. That captures 99.8% of the benefit with the smallest possible
    footprint of change.

    **Must be called after the collision scheme has been chosen.**
    `assets/jumper/jumper.xml` carries the geometry of several schemes at once (31
    `*_meshcol`, 31 `*_visual`, 14 capsules + 6 spheres + 1 box), and mjlab's
    `CollisionCfg` (`disable_other_geoms=True`) selects between them at the spec
    stage by zeroing contype and conaffinity on the ones not chosen. Called before
    that, every `*_meshcol` still "collides" and not one of the 31 meshes can be
    deleted -- the function returns normally having done nothing useful (measured:
    geoms 83 -> 52 while nmesh stays at 31 and the model size does not move).

    Returns the same `spec`, modified in place, so it can be chained.

    Raises:
        RuntimeError: per-body mass, centre of mass or inertia changed, meaning the
            model derives inertia from geoms and cannot be stripped this way.
    """
    before = spec.compile()
    mesh_t = int(mujoco.mjtGeom.mjGEOM_MESH)

    # Record the baseline first, then compare body by body
    ref = {
        "body_mass": np.array(before.body_mass),
        "body_ipos": np.array(before.body_ipos),
        "body_iquat": np.array(before.body_iquat),
        "body_inertia": np.array(before.body_inertia),
    }
    n_col_before = sum(_collides(before, i) for i in range(before.ngeom))

    # Geoms to delete: mesh type and non-colliding. Located by name, because
    # deletion invalidates indices.
    doomed = {
        before.geom(i).name
        for i in range(before.ngeom)
        if before.geom_type[i] == mesh_t and not _collides(before, i)
    }
    if not doomed:
        return spec

    kept_meshes: set[str] = set()
    for i in range(before.ngeom):
        if before.geom_type[i] == mesh_t and before.geom(i).name not in doomed:
            did = int(before.geom_dataid[i])
            if did >= 0:
                kept_meshes.add(before.mesh(did).name)

    for body in spec.bodies:
        for g in list(body.geoms):
            if g.name in doomed:
                spec.delete(g)

    for m in list(spec.meshes):
        if m.name not in kept_meshes:
            spec.delete(m)

    after = spec.compile()

    # ── Check: the dynamics must be unchanged body by body ────────────────
    for field, base in ref.items():
        got = np.asarray(getattr(after, field))
        if got.shape != base.shape:
            raise RuntimeError(
                f"stripping visual meshes changed the shape of {field} "
                f"{base.shape} -> {got.shape}; the model structure is broken."
            )
        d = np.abs(got - base).max()
        if d > 1e-9:
            bad = int(np.abs(got - base).reshape(len(base), -1).max(axis=1).argmax())
            raise RuntimeError(
                f"stripping visual meshes changed {field} (max difference "
                f"{d:.3e}, body[{bad}] {after.body(bad).name!r}). Some body in this "
                f"model has no explicit <inertial> and its inertia is derived from "
                f"geoms, so deleting visual geoms changed the dynamics. Add "
                f"<inertial> to those bodies first."
            )

    n_col_after = sum(_collides(after, i) for i in range(after.ngeom))
    if n_col_after != n_col_before:
        raise RuntimeError(
            f"the number of colliding geoms went from {n_col_before} to "
            f"{n_col_after} -- the wrong things were deleted."
        )
    return spec
