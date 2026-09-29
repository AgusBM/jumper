#!/usr/bin/env python3
"""Check that the hexapod's standing pose actually stands.

Reads `HOME` and `STAND_Z` straight from constants and measures where the six
contact meshes end up. Two numbers matter:

- **coplanarity**, the spread of the six contact points in z. A leg left even a
  millimetre high starts every episode carrying no load, and nothing in training
  reports it -- the robot just settles over the first few steps.
- **the suggested base_z**, what `STAND_Z` would have to be for the lowest foot to
  touch the ground exactly. If it disagrees with `STAND_Z`, the robot is spawning
  hovering or sunk.

This is the quick check. `calibrate_home.py` is the one that *solves* the pose,
and it also verifies limit margins and whether the stance is holdable at all.

An earlier version of this script compared two collision schemes, mesh against
capsule. There is only one scheme now, so the comparison went with it.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from tasks.jumper.common.constants import FEET, HOME, STAND_Z


def set_home(m: mujoco.MjModel, d: mujoco.MjData, base_z: float) -> None:
    mujoco.mj_resetData(m, d)
    d.qpos[0:3] = [0.0, 0.0, base_z]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, val in HOME.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[j]] = val
    mujoco.mj_forward(m, d)


def lowest_z(m: mujoco.MjModel, d: mujoco.MjData, body: str, suffix: str) -> float:
    """The lowest world-frame point of this body's geoms whose names end in suffix."""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    best = np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.endswith(suffix):
            continue
        z = float(d.geom_xpos[g][2])
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            adr, num = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = m.mesh_vert[adr : adr + num]
            rot = d.geom_xmat[g].reshape(3, 3)
            best = min(best, float((rot @ verts.T).T[:, 2].min()) + z)
        else:  # sphere / capsule: centre minus radius
            best = min(best, z - float(m.geom_size[g][0]))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="assets/jumper/jumper.xml")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    set_home(m, d, STAND_Z)

    print(f"foot contact heights at STAND_Z = {STAND_Z} (m)\n")
    print(f"{'body':<24}{'lowest z':>12}{'above lowest':>16}")
    zs = [lowest_z(m, d, f, "_meshcol") for f in FEET]
    lo = min(zs)
    for f, z in zip(FEET, zs):
        print(f"{f:<24}{z:>12.5f}{(z - lo) * 1000:>13.3f} mm")

    suggested = STAND_Z - lo
    print()
    print(f"coplanarity  {(max(zs) - lo) * 1000:.3f} mm")
    print(f"lowest foot  {lo * 1000:+.3f} mm relative to the ground")
    print(f"suggested STAND_Z {suggested:.5f}  "
          f"({(suggested - STAND_Z) * 1000:+.3f} mm from the current value)")


if __name__ == "__main__":
    main()
