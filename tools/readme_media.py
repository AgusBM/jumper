"""Draw the pictures in README.md: trained policies, in simulation, off-screen.

    python tools/readme_media.py                    # every clip, and the still
    python tools/readme_media.py --only walk jump   # some of them
    python tools/readme_media.py --list

Each clip is one task's committed export (`tasks/<task>/out/example/`) run in a
replay environment, so a picture in the README is of a policy somebody can load,
not of one that lived in a `logs/` directory and is gone. Re-run it when a model,
a scene or an export changes; it writes `docs/media/`.

## No window, and one environment

The environment is built in-process on the native backend with one environment,
and the frames are drawn afterwards by a plain `mujoco.Renderer` over EGL. So it
runs over ssh and beside a desktop, and it never reads pixels out of a live
simulation -- `tasks/jumper/dance/export_media.py` has why that split matters.

## The command is scripted, the way the operator's is written

A replay hands the velocity and posture commands to the operator -- the pad and
the keyboard. Nobody is holding either here, so each clip's command is a function
of time, written into the command term after the term's own `compute`, which is
where the operator writes its own. The pad is never opened: one that is plugged
in and nudged would drive the clip instead.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "media"

#: What is drawn, and what the README shows. Drawn at twice the size and scaled
#: down, which is the anti-aliasing: a 360-pixel render of a robot whose legs are
#: 20 mm across is mostly stair-steps.
DRAW = (720, 540)
SHOW = (360, 270)

#: Frames a second. A GIF counts a frame's delay in hundredths of a second, and
#: 12.5 is 8 of them exactly -- 15 or 25 would be rounded, and the clip would play
#: at a speed the policy did not. It also divides both control rates there are:
#: one step in 4 of a 50 Hz task, one in 16 of jumper.posture's and jumper.jump's
#: 200 Hz. Taking "one in 4" for granted drew those two at 50 frames a second,
#: four times the file for the same picture.
FPS = 12.5

#: The still at the top of the README, drawn at twice this for the same reason.
STILL = (1600, 640)

SCENE = "studio"


@dataclass(frozen=True)
class Camera:
    distance: float = 0.72
    #: Degrees, **from the way the robot faces in the clip's first frame** and not
    #: from the world's x: a reset turns the robot to a heading of its own, and a
    #: camera placed in the world showed the front of one clip and the back of the
    #: next. 0 looks along the robot from behind it, 180 straight at its front.
    azimuth: float = 140.0
    elevation: float = -16.0
    #: Height of the point looked at, metres. The body rides at about 0.11.
    height: float = 0.06
    #: Follow the body over the ground. Off for a motion that stays where it is,
    #: where a camera that moved would add motion the policy did not make.
    follow: bool = True
    #: Seconds the camera takes to catch up. The body sways with every step, and
    #: a camera locked to it turns the sway into the whole picture shaking; one
    #: that is slow loses a robot that leaps. A quarter of a second suits a walk.
    lag: float = 0.25


#: `(seconds since the clip began) -> command`, one per command term. A posture's
#: height may be NaN, the term's own neutral height, or `(height, share)`, that
#: share of the way there from it.
Script = Callable[[float], tuple]


@dataclass(frozen=True)
class Clip:
    task: str
    #: Seconds of rollout before the first frame kept -- the settling after a
    #: reset, or the part of a recording before the part worth showing.
    skip: float
    #: How long the clip is. `None` is to the end of the recording the task
    #: tracks: a gesture is drawn whole, from HOME and back to it, so the loop
    #: closes on the pose it opened with.
    seconds: float | None
    camera: Camera = field(default_factory=Camera)
    #: Command term -> its script. A term not named keeps what it samples.
    commands: dict[str, Script] = field(default_factory=dict)


def _ramp(t: float, t0: float, t1: float) -> float:
    """0 before `t0`, 1 after `t1`, smooth in between: a stick is not a switch."""
    x = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return x * x * (3.0 - 2.0 * x)


def _walk(speed: float) -> Script:
    return lambda t: (speed * _ramp(t, 0.3, 1.3), 0.0, 0.0)


def _stand(_t: float) -> tuple[float, ...]:
    return (0.0, 0.0, 0.0)


def _neutral(_t: float) -> tuple[float, ...]:
    # (twist, pitch, roll, height); a height that is NaN is the term's own
    # `neutral_height`.
    return (0.0, 0.0, 0.0, float("nan"))


#: jumper.posture's standing band is 30, 20 and 15 degrees and 0.07 to 0.15 m
#: (`tasks/jumper/posture/env_cfg.py`); the clip asks for most of each, not all.
_TWIST, _PITCH, _ROLL, _HIGH = 0.45, 0.30, 0.22, 0.14


def _pose(t: float) -> tuple:
    """Pitch, roll, twist, then height: one after another, each back to rest."""
    def bump(t0: float) -> float:
        return _ramp(t, t0, t0 + 0.4) - _ramp(t, t0 + 1.0, t0 + 1.4)

    lift = bump(4.4)
    return (_TWIST * bump(3.0), _PITCH * bump(0.2), _ROLL * bump(1.6),
            (_HIGH, lift))


CLIPS: dict[str, Clip] = {
    "walk": Clip("jumper.posture", skip=1.0, seconds=4.0,
                 commands={"twist": _walk(0.5), "posture": _neutral}),
    "posture": Clip("jumper.posture", skip=1.0, seconds=6.0,
                    camera=Camera(follow=False),
                    commands={"twist": _stand, "posture": _pose}),
    "claw": Clip("jumper.five_foot", skip=1.0, seconds=4.0,
                 commands={"twist": _walk(0.35)}),
    # Followed over the ground and not in height, so the jump is the robot
    # leaving the floor and not the floor leaving the picture.
    "jump": Clip("jumper.jump", skip=0.0, seconds=1.76,
                 camera=Camera(distance=0.85, height=0.17, lag=0.06)),
    "dance": Clip("jumper.dance_brazilian", skip=6.0, seconds=5.0,
                  camera=Camera(follow=False)),
    # From the side of the arm that waves; from the other, the body is in the way.
    "gesture": Clip("jumper.gesture_hello", skip=0.0, seconds=None,
                    camera=Camera(azimuth=205.0, follow=False)),
}

#: The clip the still is a frame of, and how far into it.
STILL_FROM, STILL_AT = "posture", 0.2
STILL_CAMERA = Camera(distance=0.62, azimuth=145.0, elevation=-12.0, height=0.07, follow=False)


def _checkpoint(task: str) -> Path:
    out = REPO / "tasks" / Path(*task.split(".")) / "out" / "example"
    found = sorted(out.glob("model_*.pt"))
    if len(found) != 1:
        raise SystemExit(
            f"error: {out.relative_to(REPO)} holds {len(found)} checkpoints, and a clip "
            f"is of exactly one. `scripts/export.py --task {task}` writes that directory."
        )
    return found[0]


def _script(term, script: Script, step_dt: float, start: list[int], env) -> None:
    """Have `term` hold what `script` says, from its own `compute` on."""
    import torch

    compute = term.compute
    columns = term.command.shape[1]

    def scripted(dt, env_ids=None):
        compute(dt, env_ids)
        t = (env.common_step_counter - start[0]) * step_dt
        values = list(script(t))
        if hasattr(term, "posture_command"):
            rest = term.cfg.neutral_height
            height = values[3]
            if isinstance(height, tuple):
                values[3] = rest + (height[0] - rest) * height[1]
            elif math.isnan(height):
                values[3] = rest
            term.posture_command[:] = torch.tensor(values, device=term.device)
            term.hold_to_band()
            return
        if len(values) != columns:
            raise SystemExit(f"error: a script of {len(values)} for a command of {columns}")
        term.vel_command_b[:] = torch.tensor(values, device=term.device)
        if hasattr(term, "vel_command_w"):
            term.vel_command_w[:] = term.vel_command_b

    term.compute = scripted


def rollout(clip: Clip):
    """Run `clip`'s policy. Returns `(qpos per control step, model, control rate)`."""
    from dataclasses import asdict

    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjrl.replay import skip_rewards

    import scenes
    import tasks
    from tasks.jumper.dance.export_media import _qpos_of, _verify_qpos_roundtrip

    env_cfg = tasks.load_env_cfg(clip.task, play=True)
    for name, term in env_cfg.commands.items():
        if hasattr(term, "pad"):
            term.pad = False
            print(f"[media] {clip.task}: command {name!r} opens no pad")
    scenes.apply(env_cfg, SCENE)
    env_cfg.scene.num_envs = 1
    agent_cfg = tasks.load_agent_cfg(clip.task)
    step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
    try:
        skip_rewards(env)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = tasks.load_runner_cls(clip.task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device="cpu")
        runner.load(str(_checkpoint(clip.task)), load_cfg={"actor": True}, strict=True,
                    map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")

        start = [0]
        for name, script in clip.commands.items():
            _script(env.command_manager.get_term(name), script, step_dt, start, env)

        robot = env.scene["robot"]
        obs, _ = wrapped.reset()
        start[0] = env.common_step_counter
        _verify_qpos_roundtrip(env, robot, mujoco)
        model = env.sim.mj_model
        if model.nq != 7 + robot.data.joint_pos.shape[1]:
            raise SystemExit(
                f"error: the model has {model.nq} coordinates and the robot accounts for "
                f"{7 + robot.data.joint_pos.shape[1]}; a scene with a prop of its own "
                f"needs its coordinates recorded too"
            )

        if clip.seconds is None:
            # One short of the total: on reaching it the command resamples, which
            # puts the robot back at the recording's first frame.
            motion = env.command_manager.get_term("motion").motion
            steps = int(motion.time_step_total) - 1
        else:
            steps = round((clip.skip + clip.seconds) / step_dt)
        qpos = np.zeros((steps, model.nq))
        with torch.inference_mode():
            for t in range(steps):
                qpos[t] = _qpos_of(env, robot)
                obs, _, dones, _ = wrapped.step(policy(obs))
                if bool(dones[0]):
                    # Said rather than hidden: the frames after it are a second
                    # attempt, which is what a replay of a one-shot motion does.
                    print(f"[media] {clip.task}: the episode ended at {t * step_dt:.2f} s")
        return qpos[round(clip.skip / step_dt):], model, 1.0 / step_dt
    finally:
        env.close()


def _option(model):
    """Appearance only: the collision hulls drawn over the meshes hide the robot."""
    import mujoco
    from mjrl.viewer.live import LiveViewer

    option = mujoco.MjvOption()
    for group in LiveViewer._collision_only_groups(model):
        option.geomgroup[group] = 0
    return option


def _frames(model, qpos: np.ndarray, camera: Camera, size: tuple[int, int]):
    """Yield one drawn frame per row of `qpos`."""
    import mujoco

    os.environ.setdefault("MUJOCO_GL", "egl")
    # The off-screen buffer is the model's, and the still is wider than its default.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, size[1])

    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = camera.distance, camera.elevation
    w, x, y, z = qpos[0, 3:7]
    heading = np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    cam.azimuth = heading + camera.azimuth
    option = _option(model)
    gain = 1.0 - np.exp(-(1.0 / FPS) / camera.lag)
    lookat = np.array([*qpos[0, :2], camera.height])
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    try:
        for row in qpos:
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            if camera.follow:
                lookat[:2] += gain * (row[:2] - lookat[:2])
            cam.lookat[:] = lookat
            renderer.update_scene(data, camera=cam, scene_option=option)
            yield renderer.render()
    finally:
        renderer.close()


def _gif(frames, fps: float, path: Path) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(raw), size=DRAW, fps=fps, macro_block_size=1, codec="libx264",
            pix_fmt_out="yuv444p", output_params=["-crf", "4"],
        )
        writer.send(None)
        for frame in frames:
            writer.send(np.ascontiguousarray(frame).tobytes())
        writer.close()
        # One palette for the whole clip, from the clip: a GIF has 256 colours,
        # and the generic ones spend most of them on hues a grey studio lacks.
        # `fps` first: without it the muxer fills the clip out to 50 frames a
        # second with copies, which plays the same and is four times the frames.
        graph = (
            f"fps={fps:g},scale={SHOW[0]}:{SHOW[1]}:flags=lanczos,split[a][b];"
            "[a]palettegen=stats_mode=diff[p];"
            "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        )
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(raw), "-filter_complex", graph,
             "-loop", "0", str(path)],
            check=True,
        )


def _png(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(frame).resize(STILL, Image.Resampling.LANCZOS).save(path, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(CLIPS), metavar="CLIP",
                    help="draw these and leave the rest as they are")
    ap.add_argument("--list", action="store_true", help="name the clips and exit")
    args = ap.parse_args()

    if args.list:
        for name, clip in CLIPS.items():
            length = "the whole recording" if clip.seconds is None else f"{clip.seconds:g} s"
            print(f"{name:<8}  {clip.task:<24} {length}")
        return 0

    import torch
    from mjrl.backend import resolve
    from mjrl.backend.select import use_backend

    # One environment: the step is serial, and torch spread over every core
    # spends longer handing the work out than doing it.
    torch.set_num_threads(1)
    # Before the first environment is built, which reads the backend once.
    use_backend(resolve(backend="native", device="cpu", num_envs=1, training=False))

    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(CLIPS):
        clip = CLIPS[name]
        qpos, model, rate = rollout(clip)
        path = OUT / f"{name}.gif"
        every = rate / FPS
        if abs(every - round(every)) > 1e-6:
            raise SystemExit(f"error: {clip.task} steps at {rate:g} Hz, which {FPS:g} "
                             f"frames a second does not divide")
        _gif(_frames(model, qpos[::round(every)], clip.camera, DRAW), FPS, path)
        print(f"[media] {path.relative_to(REPO)}  {path.stat().st_size / 1e6:.2f} MB")
        if name == STILL_FROM:
            row = qpos[round(STILL_AT * rate)][None]
            still = OUT / "jumper.png"
            twice = (2 * STILL[0], 2 * STILL[1])
            _png(next(iter(_frames(model, row, STILL_CAMERA, twice))), still)
            print(f"[media] {still.relative_to(REPO)}  {still.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
