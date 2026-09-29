"""Ray casting for the native backend: `mujoco.mj_multiRay` / `mj_ray` in place of
mjwarp's BVH kernel.

## Why only this one step has to be replaced

Almost all of `RayCastSensor` is **backend-agnostic**:

- `prepare_rays()` computes world-frame ray origins and directions from
  `data.xpos` / `site_xpos` / `geom_xpos` and the matching rotation matrices, all
  in pure PyTorch -- and those tensors are exactly what native provides.
- `postprocess_rays()` turns distances into hit points and masks misses by
  `max_distance`, also pure PyTorch.
- The `wp.zeros(..., device=...)` ray buffers **allocate fine on a CPU device**
  (warp's CPU backend works), and `wp.to_torch` handles CPU arrays too.

The only part that cannot work is `raycast_kernel()`: it needs `model.struct` /
`data.struct`, which are mjwarp structures that native does not have. So the seam
is cut at that single method.

## Two differences from the warp version

1. **No normals.** This module zeroes `normal`. Nothing in the repository reads
   `RayCastData.normals`, so existing tasks are unaffected. MuJoCo 3.11's
   `mj_ray` and `mj_multiRay` can both return normals, so filling them is a
   matter of passing a buffer -- not done yet.
2. **Performance.** Rays that share an origin go to `mj_multiRay`, one C call per
   environment per frame; the rest are a Python loop of `mj_ray`, one call per
   ray. See "Shared origins" below.

## Shared origins

A pattern whose local offsets are all equal -- a pinhole, and the jumper's dToF,
whose 54 x 42 zones all leave the eye -- casts every ray of a frame from one
point, which is exactly what `mj_multiRay` takes: one origin, one excluded body,
an array of directions. Patterns that give each ray its own origin (grids, rings:
the height scans) cannot use it and keep the loop.

Measured on the jumper standing on a plane with a box ahead, the dToF's 2268 rays
(2026-09-29, the 5090 D training box): **10.47 ms per frame as a loop, 0.94 ms as
one `mj_multiRay`**, with distances and geom ids bit-identical ray for ray.
`cutoff` is `mjMAXVAL`, i.e. none, so the nearest hit is reported however far
away and `postprocess_rays` masks by `max_distance`, as with warp and as with
`mj_ray`.
"""

from __future__ import annotations

import mujoco
import numpy as np

__all__ = ["raycast_into"]


def raycast_into(sensor, mj_model: mujoco.MjModel, datas: list) -> None:
    """Fill the sensor's ray output buffers in place.

    Called at exactly the same point as mjwarp's `raycast_kernel()`: after
    `prepare_rays()` and before `postprocess_rays()`. The former has already
    written world-frame origins and directions into `_ray_pnt` / `_ray_vec`; this
    function reads those and fills `_ray_dist` / `_ray_geomid`.

    Frames whose rays share an origin go through `mj_multiRay`; see the module
    docstring.

    Args:
        sensor: an `mjlab.sensor.RayCastSensor`.
        mj_model: a single `MjModel` (topology is the same for every environment).
        datas: one `MjData` per environment, ordered by environment index.
    """
    import warp as wp

    dist = wp.to_torch(sensor._ray_dist)
    geomid = wp.to_torch(sensor._ray_geomid)
    normal = wp.to_torch(sensor._ray_normal)
    n_env, n_ray = dist.shape

    # Cache on the sensor whatever does not change between calls.
    # `_geomgroup_array` used to recompute the mask every time and `exclude` was
    # converted from a warp array every time -- both wasted.
    cache = getattr(sensor, "_native_cache", None)
    if cache is None or cache[0] != (n_env, n_ray):
        exclude = wp.to_torch(sensor._ray_bodyexclude).numpy()
        cache = (
            (n_env, n_ray),
            _geomgroup_array(sensor._geomgroup),
            [int(x) for x in exclude],          # also avoids a per-ray int()
            np.empty((n_env, n_ray), dtype=np.float64),
            np.empty((n_env, n_ray), dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            _shared_origin_frames(sensor, n_ray),
        )
        sensor._native_cache = cache
    _, group, excl, out_dist, out_gid, gid_buf, shared = cache

    # **Convert to contiguous float64 once, rather than per ray.**
    # `mj_ray` wants mjtNum (float64) and contiguous memory while the ray buffers
    # are float32. The inner loop used to call `np.ascontiguousarray` twice per
    # ray -- 3840 small allocations for 64 environments x 30 rays, measured at
    # 1.125 ms on its own, 38% of the whole loop. Converting once takes it from
    # 2.972 ms to 1.696 ms (-43%) with bit-identical results.
    pnt = np.ascontiguousarray(
        wp.to_torch(sensor._ray_pnt).view(n_env, -1, 3).numpy(), dtype=np.float64
    )
    vec = np.ascontiguousarray(
        wp.to_torch(sensor._ray_vec).view(n_env, -1, 3).numpy(), dtype=np.float64
    )

    if shared is not None:
        multi = mujoco.mj_multiRay
        for b, d in enumerate(datas):
            for start, stop in shared:
                multi(
                    mj_model,
                    d,
                    pnt[b, start],                      # the frame's one origin
                    vec[b, start:stop].reshape(-1),
                    group,
                    1,  # flg_static: as in the warp version, static geoms are hit
                    excl[start],                        # one body per frame
                    out_gid[b, start:stop],             # contiguous views: written in place
                    out_dist[b, start:stop],
                    None,
                    stop - start,
                    mujoco.mjMAXVAL,
                )
    else:
        ray = mujoco.mj_ray  # local name, saves an attribute lookup per inner iteration
        for b, d in enumerate(datas):
            pb, vb = pnt[b], vec[b]
            ob, og = out_dist[b], out_gid[b]
            for r in range(n_ray):
                ob[r] = ray(
                    mj_model,
                    d,
                    pb[r],
                    vb[r],
                    group,
                    1,  # flg_static: as in the warp version, static geoms are hit
                    excl[r],
                    gid_buf,
                )
                og[r] = gid_buf[0]

    # Both return -1 on a miss, the same convention the warp version uses and the
    # one postprocess_rays expects
    dist.copy_(_as_tensor(out_dist, dist))
    geomid.copy_(_as_tensor(out_gid, geomid))
    normal.zero_()  # not filled; see the module docstring


def _shared_origin_frames(sensor, n_ray: int) -> list[tuple[int, int]] | None:
    """The `(start, stop)` ray span of every frame, if each frame's rays share an origin.

    Decided from the pattern rather than from the world-frame origins: offsets that
    are equal in the frame are the same point by construction, where world origins
    computed ray by ray are only equal up to rounding. None means the loop.
    """
    offsets = getattr(sensor, "_local_offsets", None)
    per = int(getattr(sensor, "_num_rays_per_frame", 0))
    frames = int(getattr(sensor, "_num_frames", 0))
    if offsets is None or per < 2 or frames * per != n_ray:
        return None
    if not bool((offsets == offsets[0]).all()):
        return None
    return [(f * per, (f + 1) * per) for f in range(frames)]


def _as_tensor(arr: np.ndarray, like):
    import torch

    return torch.from_numpy(arr).to(dtype=like.dtype)


def _geomgroup_array(geomgroup) -> np.ndarray | None:
    """Convert mjwarp's vec6 mask into the uint8 mask `mj_ray` expects.

    **The two conventions differ, so a direct cast is wrong:**

    | | Group included | Group excluded |
    |---|---|---|
    | mjwarp's vec6 (`_geom_groups_to_vec6`) | **-1** | 0 |
    | MuJoCo's `mjtByte geomgroup[mjNGROUP]` | non-zero (1) | 0 |

    So it is re-encoded as "non-zero means included" rather than cast as-is; a
    plain cast gives
    `OverflowError: Python integer -1 out of bounds for uint8`.

    `None` means no filtering, and both sides spell that the same way.
    """
    if geomgroup is None:
        return None
    flags = [1 if int(v) != 0 else 0 for v in geomgroup]
    flags += [0] * max(0, mujoco.mjNGROUP - len(flags))
    return np.ascontiguousarray(flags[: mujoco.mjNGROUP], dtype=np.uint8)
