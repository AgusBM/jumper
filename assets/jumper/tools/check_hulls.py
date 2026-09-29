#!/usr/bin/env python3
"""Render what the collision geometry actually is, after `tools/hull_collision.py`.

The swap is invisible in every other check. The tests pass, the stance is
identical, and what moved was 100 MB of memory -- so if a geom were repointed
at the wrong hull, or a mesh that touches the ground were hulled by accident,
nothing would fail. It would just walk slightly wrong forever.

The hulls are decimated to 64 directions (`hull_collision.py -k 64`), so unlike
the exact hull they *are* an approximation: up to 7.81 mm inward on
`upper_shell_link`. Inward is the safe direction -- contact triggers later, never
earlier -- but it is a reason to look rather than to assume.

So this draws three views:

  visual      the mesh geoms, group 2 -- what the robot looks like
  collision   the hulled geoms, group 1 -- what the physics touches
  gap         hull depth minus shell depth per pixel, masked to pixels where
              both show the SAME body -- how far the hull bulges outward

**Green means the mesh was kept, not hulled.** Those are `constants.FEET`, the
six geoms excluded from hulling because they are the ones that touch the
ground, where the hull's flat facets would change contact timing. If a green
part is missing from the overlay, the exclusion list and `FEET` have drifted
apart.

**Most of the gap panel is not damage this change did.** The upper shell reads
tens of millimetres, and nearly all of that was already true before any hull was
written: MuJoCo has always collided a mesh by its hull, so the hollow dome has
always collided as a solid block. What decimation adds on top is bounded by the
7.81 mm of inward shrink, and it goes the other way -- inward. The panel is
mostly measuring what convex collision costs in general.

It is here because that cost is otherwise invisible, and because a part whose
hull bridges something it should not -- a gripper that collides across its own
open jaws -- is worth seeing before a policy learns to exploit it.

Usage:
    python assets/jumper/tools/check_hulls.py                 # writes a PNG
    python assets/jumper/tools/check_hulls.py --out /tmp/h.png --azimuth 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from tasks.jumper.common.constants import FEET, JUMPER_XML, HOME, STAND_Z

VISUAL_GROUP = 2
COLLISION_GROUP = 1

KEPT_RGBA = (0.10, 0.80, 0.35, 1.0)  # feet: mesh kept as-is
HULL_RGBA = (0.20, 0.55, 0.95, 1.0)  # everything else: replaced by its hull


def posed_model(xml: str):
    spec = mujoco.MjSpec.from_file(xml)
    spec.worldbody.add_light(
        pos=[0.5, -0.5, 1.2], dir=[-0.35, 0.35, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, STAND_Z]
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, value in HOME.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)
    return model, data


def colour_collision_geoms(model, alpha: float) -> tuple[int, int]:
    """Paint hulled geoms blue and kept-mesh geoms green. Returns (hulled, kept)."""
    hulled = kept = 0
    for gid in range(model.ngeom):
        if model.geom_group[gid] != COLLISION_GROUP:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        rgba = KEPT_RGBA if body in FEET else HULL_RGBA
        model.geom_rgba[gid] = (*rgba[:3], alpha)
        if body in FEET:
            kept += 1
        else:
            hulled += 1
    return hulled, kept


def render(model, data, renderer, groups, camera, wireframe: bool = False) -> np.ndarray:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    renderer.update_scene(data, camera, opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = int(wireframe)
    return renderer.render()


def depth_gap(model, data, renderer, camera) -> tuple[np.ndarray, dict]:
    """How far the collision hull sits in front of the visual shell, per pixel.

    This is the third panel because it is the one number the swap can get
    wrong in a way that matters. A hull fills in every concavity, so between
    the gripper jaws and under the belly the robot now collides with a volume
    it does not visually occupy. Rendering both depth buffers from one camera
    and subtracting shows exactly where, and by how much.
    """
    renderer.enable_depth_rendering()
    try:
        shell = render(model, data, renderer, [VISUAL_GROUP], camera).copy()
        hull = render(model, data, renderer, [COLLISION_GROUP], camera).copy()
    finally:
        renderer.disable_depth_rendering()

    # Depth alone is not enough. At a silhouette the nearest hull and the
    # nearest visual mesh at one pixel belong to **different bodies** -- a near
    # leg's hull in front of a far leg's shell -- and subtracting those reported
    # a 307 mm bulge on a 200 mm robot. Segmentation ids restrict the comparison
    # to pixels where both buffers show the same body.
    renderer.enable_segmentation_rendering()
    try:
        seg_shell = render(model, data, renderer, [VISUAL_GROUP], camera)[..., 0].copy()
        seg_hull = render(model, data, renderer, [COLLISION_GROUP], camera)[..., 0].copy()
    finally:
        renderer.disable_segmentation_rendering()
    body_of = np.concatenate([model.geom_bodyid, [-1]])  # -1 for the background
    b_shell = body_of[np.where(seg_shell < 0, -1, seg_shell)]
    b_hull = body_of[np.where(seg_hull < 0, -1, seg_hull)]

    both = (b_shell >= 0) & (b_shell == b_hull)
    gap = np.where(both, shell - hull, 0.0)  # hull nearer to camera => positive
    gap = np.clip(gap, 0.0, None)
    per_body = {}
    for bid in np.unique(b_shell[both]):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
        per_body[name] = float(gap[both & (b_shell == bid)].max())
    stats = {
        "max": float(gap.max()),
        "mean": float(gap[both].mean()) if both.any() else 0.0,
        "pixels": int(both.sum()),
        "per_body": per_body,
    }

    # A three-stop ramp, done in numpy: matplotlib is not a declared dependency.
    hi = max(stats["max"], 1e-9)
    t = np.clip(gap / hi, 0, 1)[..., None]
    stops = np.array([[16, 24, 48], [40, 110, 200], [255, 232, 64]], float)
    lo_half = t * 2
    img = np.where(
        t < 0.5,
        stops[0] + (stops[1] - stops[0]) * lo_half,
        stops[1] + (stops[2] - stops[1]) * (lo_half - 1),
    )
    img[~both] = 0
    return img.astype(np.uint8), stats


def launch_viewer(model, data) -> None:
    """Hold the robot at HOME in the interactive viewer, collision group only.

    `mj_forward` rather than `mj_step`: nothing here is a simulation, and a
    stepped robot at HOME with no controller collapses within a second, which
    is not what anyone opened this to look at.
    """
    import time

    import mujoco.viewer

    print("\n  viewer keys:  1 = collision (hulls)   2 = visual (meshes)")
    print("                both on shows the hull z-fighting through the shell")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[COLLISION_GROUP] = 1
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 60)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, default=Path(JUMPER_XML))
    ap.add_argument("--out", type=Path, default=Path("hulls.png"))
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=0.62)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--viewer", action="store_true",
                    help="open the interactive viewer instead of writing a PNG")
    args = ap.parse_args(argv)

    model, data = posed_model(str(args.model))
    hulled, kept = colour_collision_geoms(model, 1.0)
    print(f"model {args.model}")
    print(f"  {hulled} collision geoms hulled (blue), {kept} meshes kept (green)")
    print(f"  kept = {', '.join(sorted(FEET))}")
    if kept != len(FEET):
        print(f"  ** {len(FEET) - kept} of constants.FEET have no collision geom **")

    if args.viewer:
        launch_viewer(model, data)
        return 0

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-0.02, 0, 0.06]
    camera.azimuth, camera.elevation = args.azimuth, args.elevation
    camera.distance = args.distance

    renderer = mujoco.Renderer(model, args.height, args.width)
    visual = render(model, data, renderer, [VISUAL_GROUP], camera)
    collision = render(model, data, renderer, [COLLISION_GROUP], camera)
    gap, stats = depth_gap(model, data, renderer, camera)
    print(f"  hull stands proud of the shell by {stats['max']*1e3:.2f} mm at worst, "
          f"{stats['mean']*1e3:.2f} mm mean over {stats['pixels']} pixels")
    print("  worst bodies (this view only -- a concavity facing away reads 0):")
    for name, mm in sorted(stats["per_body"].items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {name:26s} {mm*1e3:7.2f} mm")

    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("this needs Pillow: pip install pillow") from None
    Image.fromarray(np.hstack([visual, collision, gap])).save(args.out)
    print(f"  wrote {args.out}   (visual | collision | hull-minus-shell gap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
