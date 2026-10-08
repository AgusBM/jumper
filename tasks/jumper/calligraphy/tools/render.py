#!/usr/bin/env python3
"""Step 4: the two shots of a `write.py` run, and the ink for the compositor.

    MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py logs/calligraphy/u65e0/<run>
    MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py <run> --check
    MUJOCO_GL=osmesa python tasks/jumper/calligraphy/tools/render.py <run> --no-ink --shots low

Writes into the run's directory:

    top.mp4, low.mp4    the overhead and the low shot, real time, ink drawn as it is laid
                        (--no-ink for clean plates to composite onto)
    wu.gif              the overhead shot sped up, small, for the README
    ink.json            every mark: time, floor xy, force, planned press, width, and its
                        pixels in each shot; the shots' intrinsics and poses
    ink.svg             the ink from above at the character's scale

**Plain MuJoCo, no mjlab, no torch.** The run's `model.mjb` and the `qpos` in its
`log.npz` are the whole state, replayed with `mj_forward`; nothing is simulated
again. That is also what makes it run on a machine with no GPU: OSMesa renders on
the CPU, and it crashes the process when torch is loaded beside it (measured on
this container, 2026-10-08: `import mujoco; import torch; import tensordict` under
MUJOCO_GL=osmesa dies; under egl it does not), so the renderer must not share a
process with the simulation.

`--check` puts markers at known floor points, renders them from each shot and
measures where they land against `cameras.Shot.project` -- the export's pixels
are only worth anything if that agrees.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco

from tasks.jumper.calligraphy import cameras, hanzi, ink

#: 地书 is water on stone: the ink is the stone darkened, not black.
INK_RGBA = (0.22, 0.21, 0.21, 1.0)
#: The paving: light stone slabs with darker joints.
STONE = np.array([0.60, 0.58, 0.55])
JOINT = np.array([0.42, 0.40, 0.38])
HAZE = (0.86, 0.87, 0.88, 1.0)


def dress(m: mujoco.MjModel) -> None:
    """The training floor -- blue, chequered, reflective -- as paving stones.

    Only what is drawn changes: the texture's pixels, the material's reflectance and
    the fog. Contact, friction and geometry are the run's, untouched.
    """
    g = m.geom("terrain").id
    mat = m.geom_matid[g]
    tex = m.mat_texid[mat][1]  # mjTEXROLE_RGB
    w, h, c = m.tex_width[tex], m.tex_height[tex], m.tex_nchannel[tex]
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w]
    slabs = 2  # per texture repeat
    sx, sy = (xx * slabs) // w, (yy * slabs) // h
    tint = 1.0 + 0.05 * rng.standard_normal((slabs, slabs))
    img = STONE[None, None, :] * tint[sy, sx][..., None]
    img *= 1.0 + 0.025 * rng.standard_normal((h, w))[..., None]
    joint = ((xx % (w // slabs)) < 3) | ((yy % (h // slabs)) < 3)
    img[joint] = JOINT
    data = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., :c]
    adr = m.tex_adr[tex]
    m.tex_data[adr: adr + w * h * c] = data.ravel()
    m.mat_rgba[mat] = (1.0, 1.0, 1.0, 1.0)
    m.mat_reflectance[mat] = 0.0
    m.vis.rgba.fog = HAZE
    m.vis.map.fogstart = 1.0
    m.vis.map.fogend = 3.5
    # The scene's light is a spot 1.5 m up; from the low camera the edge of its
    # shadow map lies on the floor as dark wedges along the horizon. A low sun
    # from the side shades the same and casts shadows the whole floor agrees on.
    sun = np.array([0.35, -0.25, -1.0])
    m.light_type[0] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_dir[0] = sun / np.linalg.norm(sun)
    m.vis.headlight.ambient = (0.35, 0.35, 0.35)
    m.vis.headlight.diffuse = (0.40, 0.40, 0.40)


def _add_disc(scene, x: float, y: float, r: float, rgba=INK_RGBA) -> bool:
    """A flat wet patch: an ellipsoid r x r x 0.3 mm, so it has no rim."""
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ELLIPSOID, np.array([r, r, 0.0003]),
                        np.array([x, y, ink.LIFT]), np.eye(3).ravel(),
                        np.asarray(rgba, dtype=np.float32))
    g.specular = 0.0
    scene.ngeom += 1
    return True


def _backdrop(scene, shot: cameras.Shot) -> None:
    """A wall of haze behind everything, square to the camera: the model has no
    sky, and the low shot would otherwise put black above the horizon."""
    f = shot.forward()
    fh = np.array([f[0], f[1], 0.0]) / np.linalg.norm(f[:2])
    centre = shot.position() + 9.0 * fh
    right = np.cross(fh, (0.0, 0.0, 1.0))
    rot = np.column_stack([right, fh, (0.0, 0.0, 1.0)])
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([30.0, 0.01, 10.0]),
                        centre, rot.ravel(), np.asarray(HAZE, dtype=np.float32))
    g.emission = 1.0
    g.specular = 0.0
    # Decor casts no shadow; as a plain geom it laid dark bands along the horizon.
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom += 1


def _add_ball(scene, p, r: float, rgba) -> None:
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]),
                        np.asarray(p, dtype=float), np.eye(3).ravel(),
                        np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


class Replay:
    def __init__(self, run: Path):
        self.m = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
        dress(self.m)
        self.d = mujoco.MjData(self.m)
        self.log = ink.Log(run / "log.npz")
        if self.log.qpos is None:
            raise SystemExit(f"{run}/log.npz has no qpos; rerun write.py")
        self.plan = hanzi.Plan.from_json(json.loads((run / "plan.json").read_text()))
        self.t = self.log["t"]

    def at(self, t: float) -> None:
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        self.d.qpos[:] = self.log.qpos[i]
        mujoco.mj_forward(self.m, self.d)

    def renderer(self, shot: cameras.Shot) -> mujoco.Renderer:
        self.m.vis.global_.fovy = shot.fovy
        self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, shot.width)
        self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, shot.height)
        return mujoco.Renderer(self.m, shot.height, shot.width, max_geom=20000)


def ink_points(marks) -> np.ndarray:
    """(n, 4) rows of t, x, y, radius: every inked point once, in time order, so
    a frame at time t draws a prefix."""
    if not marks:
        return np.zeros((0, 4))
    pts = np.concatenate([np.column_stack([m.t, m.xyz[:, :2], m.width / 2]) for m in marks])
    return pts[np.argsort(pts[:, 0])]


def frame(rp: Replay, r: mujoco.Renderer, shot: cameras.Shot, cam, pts, t: float) -> np.ndarray:
    """One picture of the run at time t; the ink laid by then, unless `pts` is None."""
    rp.at(t)
    r.update_scene(rp.d, camera=cam)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
    r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    if shot.elevation > -60:
        _backdrop(r.scene, shot)
    if pts is not None:
        n = np.searchsorted(pts[:, 0], t, side="right")
        for x, y, rad in pts[:n, 1:]:
            if not _add_disc(r.scene, x, y, rad):
                break
    return r.render()


def render_shot(rp: Replay, shot: cameras.Shot, marks, out: Path, fps: float, speed: float,
                with_ink: bool, hold: float = 2.0, scale: float = 1.0) -> list[np.ndarray]:
    """Frames of one shot; written to `out` as mp4 (or returned for a gif)."""
    r = rp.renderer(shot)
    cam = shot.mjv_camera()
    # Every inked point once, in time order, so a frame draws a prefix.
    pts = ink_points(marks)
    t_end = rp.t[-1]
    times = np.arange(0.0, t_end, speed / fps)
    times = np.concatenate([times, np.full(int(hold * fps), t_end)])
    frames = []
    writer = None
    if out.suffix == ".mp4":
        import imageio.v2 as imageio

        writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=8)
    for k, t in enumerate(times):
        img = frame(rp, r, shot, cam, pts if with_ink else None, t)
        if scale != 1.0:
            from PIL import Image

            img = np.asarray(Image.fromarray(img).resize(
                (int(img.shape[1] * scale), int(img.shape[0] * scale)), Image.LANCZOS))
        if writer is not None:
            writer.append_data(img)
        else:
            frames.append(img)
        if k % 200 == 0:
            print(f"  {shot.name}: frame {k}/{len(times)}", flush=True)
    if writer is not None:
        writer.close()
    r.close()
    return frames


def write_gif(frames: list[np.ndarray], path: Path, fps: float) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    # One palette for the whole clip, from a frame with all the ink on it: per-frame
    # palettes make the floor flicker between neighbouring greys.
    palette = imgs[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0,
              optimize=True)


def check(rp: Replay, shots) -> float:
    """Markers at known floor points: rendered pixel vs `Shot.project`."""
    cx, cy = rp.plan.origin
    h = rp.plan.size / 2
    # Resting on the floor rather than centred in it: half a ball sunk in the floor
    # shows only its top from a low camera, and the visible part's centroid sits
    # 2-3 px above the centre -- an error in the check, not in the projection.
    rad = 0.006
    probe = np.array([[cx, cy, rad], [cx + h, cy + h, rad], [cx - h, cy - h, rad],
                      [cx + h, cy - h, rad], [cx - h, cy + h, rad]])
    rp.at(0.0)
    worst = 0.0
    for shot in shots:
        r = rp.renderer(shot)
        for p in probe:
            r.update_scene(rp.d, camera=shot.mjv_camera())
            _add_ball(r.scene, p, rad, (1.0, 0.0, 1.0, 1.0))
            img = r.render().astype(int)
            magenta = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] > 200)
            if not magenta.any():
                print(f"  {shot.name}: marker at {p[:2]} not visible")
                continue
            ys, xs = np.nonzero(magenta)
            got = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
            want = shot.project(p[None])[0]
            err = float(np.linalg.norm(got - want))
            worst = max(worst, err)
            print(f"  {shot.name}: marker {p[:2]} rendered at {got.round(1)}, "
                  f"projected {want.round(1)}: {err:.1f} px")
        r.close()
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a write.py output directory")
    ap.add_argument("--shots", nargs="+", default=["top", "low"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-ink", action="store_true", help="clean plates, no ink drawn")
    ap.add_argument("--gif-speed", type=float, default=6.0)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--force", type=float, default=1.0,
                    help="the --force write.py ran with, N (the width's reference)")
    ap.add_argument("--still", type=float, nargs="+", metavar="T",
                    help="write <shot>_<T>.png at these times (s) and stop")
    ap.add_argument("--check", action="store_true",
                    help="check the projection against the renderer and stop")
    args = ap.parse_args()

    rp = Replay(args.run)
    shots = {s.name: s for s in cameras.shots(rp.plan.origin, rp.plan.size)}
    if args.check:
        worst = check(rp, list(shots.values()))
        print(f"worst marker error: {worst:.1f} px")
        # Measured on run7 (2026-10-08): top <= 0.5 px, low 1.4-2.1 px, a residual
        # upward bias of the low shot that is under 1 mm on the floor.
        return 0 if worst < 2.5 else 1

    ms = ink.marks(rp.log, rp.plan, args.force)
    if args.still:
        from PIL import Image

        pts = None if args.no_ink else ink_points(ms)
        for name in args.shots:
            shot = shots[name]
            r = rp.renderer(shot)
            for t in args.still:
                t = min(t, float(rp.t[-1]))
                path = args.run / f"{name}_{t:05.1f}.png"
                Image.fromarray(frame(rp, r, shot, shot.mjv_camera(), pts, t)).save(path)
                print(f"[render] wrote {path}")
            r.close()
        return 0
    data = ink.to_json(ms, rp.plan, list(shots.values()), args.force)
    (args.run / "ink.json").write_text(json.dumps(data))
    (args.run / "ink.svg").write_text(ink.to_svg(ms, rp.plan))
    print(f"[render] {len(ms)} marks, {sum(len(m.t) for m in ms)} points -> ink.json, ink.svg")

    suffix = "" if not args.no_ink else "_plate"
    for name in args.shots:
        path = args.run / f"{name}{suffix}.mp4"
        render_shot(rp, shots[name], ms, path, args.fps, 1.0, not args.no_ink)
        print(f"[render] wrote {path}")
    if not args.no_gif and not args.no_ink:
        fps = 15.0
        frames = render_shot(rp, shots["top"], ms, args.run / "wu.gif", fps, args.gif_speed,
                             True, hold=2.5, scale=0.5)
        write_gif(frames, args.run / "wu.gif", fps)
        size = (args.run / "wu.gif").stat().st_size / 1e6
        print(f"[render] wrote {args.run / 'wu.gif'} ({len(frames)} frames, {size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
