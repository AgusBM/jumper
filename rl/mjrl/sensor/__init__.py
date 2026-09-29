"""Sensor implementations for the native backend.

## What lives here

Every sensor in mjlab's `mjlab/sensor/` splits into two halves:

- **The backend-agnostic half** -- configuration, coordinate transforms,
  post-processing; pure PyTorch. Both backends share it and it stays in mjlab
  untouched.
- **The half that does the actual computation** -- an mjwarp kernel. The native
  backend has no `model.struct`, so this half has to be rewritten.

This package is that rewritten half, one module per sensor:

| Module | mjwarp call it replaces | Replaced by |
|---|---|---|
| `raycast` | `mujoco_warp.rays()` BVH kernel | `mujoco.mj_ray` |
| `camera` | `mujoco_warp.render()` | off-screen `mujoco.Renderer` |

Dispatch happens in the mjlab half (each site carries an `[mjrl]` marker), and the
test is always `hasattr(model, "struct")` -- present means mjwarp, absent means
native.

## Why a separate package rather than more files under backend/

`backend/` is about **the simulation itself**: `native_sim` is the second
implementation of `Simulation`, and `resolve` / `select` decide which backend to
use. Sensors are **another layer** attached to the simulation -- whether their
implementation changes is a separate question from how the backend is chosen.
With only ray casting the two could share a directory; adding the camera made the
split worth making.
"""
