"""The performance, as opposed to the policy: what `scripts/export.py` writes here
beyond the policy and its contract.

`actor.onnx` + `layout.json` + `README.md` are what the robot needs. They are also
completely unwatchable, and this task's whole point is a dance. So this module
renders the trained policy dancing the full choreography, puts the music back on
it, and ships the face-screen animation alongside.

## Why the artifacts are in a subdirectory

`out/<date-time>/` **is** the exported policy: `deploy/` copies that
directory to the board. A 30 MB video and a 5 MB mp3 sitting in it would be copied
too, over the network, to a device with an SD card. They go in `out/<checkpoint>/
media/` instead, which the importer does not walk into.

## Why the rendering is decoupled from the rollout

The rollout records `qpos` per control step and nothing else; rendering happens
afterwards from that array, through a plain `mujoco.Renderer`. Two reasons, both
learned the hard way elsewhere in this repository:

- **It is backend-independent.** Reading pixels out of a live env would mean one
  path for mjwarp (device buffers, `mjwarp.render()`) and another for native
  (`mujoco.Renderer` over `_datas[i]`, which is not environment `i` -- see
  `native_sim.py`). Recording `qpos` uses the `Entity` API both backends already
  implement, and the renderer then never sees a backend at all.
- **It separates two failures that look alike.** A policy that dances badly and a
  renderer pointed the wrong way both produce a bad video. With the trajectory on
  one side and the picture on the other, the rollout can be checked numerically
  (it is -- see `_verify_qpos_roundtrip`) before a single frame is drawn.

## The audio offset is real and is read, not assumed

The choreography opens with a silent lead-in: `audio_start_in_sim` is **2.0 s**,
and in the crab clip that figure is exact rather than approximate -- across all 496
beats, `beat_times_sim - beat_times_audio` is 2.0 at the first beat and 2.0 at the
last. Muxing the music at t=0 would put the whole performance two and a half beats
early, which looks *almost* right and is the kind of error nobody catches by
watching once.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..common.dance.motion import AUDIO_SUFFIXES, find_material

__all__ = ["export_media"]

#: Render size. 16:9 at a size that survives being watched full-screen without
#: making the file large enough to be annoying to move around: the crab clip comes
#: out around 25 MB at 235 s.
WIDTH, HEIGHT = 960, 540

def _fps_from(env_cfg) -> float:
    """Frames per second of the output video.

    **Not a free parameter**: it is the control rate, one rendered frame per policy
    step, so the video's time axis is the simulation's with no resampling and no
    accumulating drift. 50 Hz for this task (5 ms physics, decimation 4); changing
    the env's decimation changes this with no edit here.
    """
    return 1.0 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)


#: Camera. A fixed shot, because `body_x` and `body_y` are **identically zero**
#: through the entire clip -- the crab dances in place, and a tracking camera would
#: add motion the performance does not have. Aimed at the mid-height of the body's
#: 105-130 mm range.
#:
#: The elevation is set by a measurement rather than by taste. **The scene has no
#: skybox**, so everything above the horizon renders pure black, and a shallow
#: angle frames a band of it: at -18 degrees the top 66 rows of a 540-row frame
#: were black, which reads as an accidental letterbox. -26 degrees puts the horizon
#: off the top edge entirely (measured on the reference pose at frame 3000: 66 rows
#: at -18, 0 at -26, 0 at -32), and the distance comes in to 0.72 m to keep the
#: robot -- 105 mm tall, against mjlab's default framing for a 1.3 m humanoid --
#: filling the frame at the steeper angle.
CAM_LOOKAT = (0.0, 0.0, 0.115)
CAM_DISTANCE = 0.72
CAM_AZIMUTH = 138.0
CAM_ELEVATION = -26.0

#: How far the reconstructed `qpos` may put a body from where the live environment
#: says it is, in metres, before the rollout is rejected. This is a
#: same-quantity-two-ways comparison within one timestep, so the only thing between
#: them is float32/float64 rounding through `mj_forward`; the tolerance is set well
#: above that and far below any real mistake, which would be a swapped quaternion
#: convention (centimetres) or a wrong joint order (the whole robot).
_QPOS_ROUNDTRIP_TOL = 1e-4


def _ffmpeg() -> str:
    """The ffmpeg binary imageio ships, so this does not depend on a system one."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _rollout(task_id: str, asset: Path | None, checkpoint: Path, res):
    """Run the policy through the whole clip, returning `qpos` per control step.

    Returns `(qpos, model, fps, frames_total, stopped_early)`.

    The environment is built in **play mode**, which this task already defines as
    "the whole dance from its first frame, unperturbed": `episode_length_s` is
    effectively unbounded, observation noise and pushes are off, and the motion
    command's `sampling_mode` is `"start"` rather than the adaptive
    reference-state initialisation training uses. So no special-casing is needed
    here to get a performance rather than a 10-second excerpt.

    **Failure terminations stay on.** If the policy falls over 40 seconds in, the
    recording stops at 40 seconds and says so. Disabling them would produce a video
    of a robot teleporting back to the reference and carrying on, which would
    misrepresent the policy exactly where it matters most.
    """
    from dataclasses import asdict

    import mujoco
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    import tasks

    env_cfg = tasks.load_env_cfg(task_id, asset=asset, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = tasks.load_agent_cfg(task_id)
    fps = _fps_from(env_cfg)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = tasks.load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=res.device)
        runner.load(
            str(checkpoint), load_cfg={"actor": True}, strict=True,
            map_location=res.device,
        )
        policy = runner.get_inference_policy(device=res.device)

        robot = env.scene["robot"]
        command = env.command_manager.get_term("motion")
        # One short of the total: `MotionCommand._update_command` resamples on
        # reaching `time_step_total`, which teleports the robot back to the
        # reference. Stopping before that keeps the last frame part of the dance.
        frames_total = int(command.motion.time_step_total) - 1

        # Step through the rsl_rl wrapper rather than the raw env: it is what
        # supplies the policy's observation TensorDict, and it folds `terminated |
        # truncated` into one `dones`. Truncation cannot fire here -- play mode
        # sets `episode_length_s` effectively unbounded -- so a done is always a
        # real failure.
        obs, _ = wrapped.reset()
        _verify_qpos_roundtrip(env, robot, mujoco)

        qpos = np.zeros((frames_total, env.sim.mj_model.nq), dtype=np.float64)
        stopped_early = None
        with torch.inference_mode():
            for t in range(frames_total):
                qpos[t] = _qpos_of(env, robot)
                obs, _, dones, _ = wrapped.step(policy(obs))
                if bool(dones[0]):
                    stopped_early = t
                    qpos = qpos[: t + 1]
                    break

        model = env.sim.mj_model
        return qpos, model, fps, frames_total, stopped_early
    finally:
        env.close()


def _qpos_of(env, robot) -> np.ndarray:
    """The single environment's `qpos`, rebuilt from the `Entity` API.

    MuJoCo's free-joint layout is `[xyz, quat wxyz]` followed by the hinge joints
    in model order, which is what `root_link_pose_w` and `joint_pos` supply --
    minus the scene origin, since `Entity` reports world coordinates and the model
    expects the environment's own frame.
    """
    pose = robot.data.root_link_pose_w[0].detach().cpu().numpy()
    origin = env.scene.env_origins[0].detach().cpu().numpy()
    joints = robot.data.joint_pos[0].detach().cpu().numpy()
    return np.concatenate([pose[:3] - origin, pose[3:7], joints])


def _body_id(model, name: str, mujoco) -> int:
    """A body's id, tolerating the prefix `MjSpec.attach` adds.

    mjlab builds the scene by attaching the robot spec into a world spec, and
    attachment renames every body to `<entity>/<name>`. `Entity.body_names` reports
    the unprefixed names, so a lookup has to try both.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        return bid
    for i in range(model.nbody):
        full = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if full is not None and full.rsplit("/", 1)[-1] == name:
            return i
    return -1


def _verify_qpos_roundtrip(env, robot, mujoco) -> None:
    """Check the reconstruction against the live environment before recording.

    This pins a failure that produces no error and a perfectly watchable video:
    if the free joint's quaternion were read `xyzw` where the model wants `wxyz`,
    or the joints came back in a different order than the model's, every frame
    still renders -- a robot in a plausible pose, dancing a dance that is not the
    one the policy performed. Nothing downstream would complain, and the video
    would be believed.

    So: take `qpos` the way `_qpos_of` builds it, push it through `mj_forward` on a
    scratch `MjData`, and compare the resulting body positions against what the
    environment itself reports for the same instant. Agreement to a tenth of a
    millimetre means the layout is right; a convention error moves bodies by
    centimetres and an order error by the size of the robot.
    """
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = _qpos_of(env, robot)
    mujoco.mj_forward(model, data)

    origin = env.scene.env_origins[0].detach().cpu().numpy()
    # `body_link_pos_w`, not `body_com_pos_w`: MuJoCo's `xpos` is the body frame's
    # origin, and comparing it against the centre of mass would report an offset
    # for every body with a non-zero `ipos` -- a real difference, but not the one
    # being checked.
    live = robot.data.body_link_pos_w[0].detach().cpu().numpy() - origin
    names = list(robot.body_names)

    worst_name, worst, matched = "", 0.0, 0
    for i, name in enumerate(names):
        bid = _body_id(model, name, mujoco)
        if bid < 0:
            continue
        matched += 1
        err = float(np.linalg.norm(data.xpos[bid] - live[i]))
        if err > worst:
            worst_name, worst = name, err

    # Without this the check passes vacuously. `Entity.body_names` are the names
    # as the robot spec declares them, while the compiled scene prefixes every
    # body with the entity it was attached from -- so a plain `mj_name2id` can
    # return -1 for *all* of them, leave `worst` at 0.0, and report success having
    # compared nothing. `_body_id` handles the prefix; this makes sure it did.
    if matched == 0:
        raise RuntimeError(
            f"none of the robot's {len(names)} body names resolved in the compiled "
            f"model, so the qpos check compared nothing. First few looked for: "
            f"{names[:3]}. Model has: "
            f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(min(4, model.nbody))]}."
        )

    if worst > _QPOS_ROUNDTRIP_TOL:
        raise RuntimeError(
            f"the recorded qpos does not reproduce the live pose: body "
            f"{worst_name!r} is {worst * 1000:.1f} mm away after mj_forward "
            f"(tolerance {_QPOS_ROUNDTRIP_TOL * 1000:.2f} mm). The video would "
            f"render a pose the policy never took. Check the free-joint quaternion "
            f"convention and the joint order in _qpos_of."
        )


def _render(model, qpos: np.ndarray, fps: float, path: Path) -> None:
    """Draw every recorded frame to an mp4."""
    import imageio_ffmpeg
    import mujoco

    # EGL unless the caller has already chosen: rendering must not need a window,
    # and export is routinely run over ssh. Set before the first Renderer is built,
    # which is what binds the GL backend.
    os.environ.setdefault("MUJOCO_GL", "egl")

    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION

    writer = imageio_ffmpeg.write_frames(
        str(path), size=(WIDTH, HEIGHT), fps=fps, macro_block_size=1,
        codec="libx264", pix_fmt_out="yuv420p", quality=7,
    )
    writer.send(None)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    try:
        for t in range(len(qpos)):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            writer.send(renderer.render().tobytes())
    finally:
        renderer.close()
        writer.close()


def _mux(video: Path, audio: Path, offset_s: float, out: Path) -> None:
    """Put the music on the video, starting `offset_s` in.

    `-shortest` cuts at whichever runs out first. That is the video in practice:
    the soundtrack is a few seconds longer than the choreography, and its tail is
    past the last beat.
    """
    cmd = [
        _ffmpeg(), "-y", "-loglevel", "error",
        "-i", str(video),
        "-itsoffset", f"{offset_s:.6f}", "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def export_media(out: Path, *, task_id: str, checkpoint: Path, asset, res) -> None:
    """Render the performance and ship it beside the bundle.

    Called by `scripts/export.py` through `tasks.load_export_media`; see that
    function for why the seam is shaped this way.
    """
    from ..common.dance.motion import load_source
    from .env_cfg import MEDIA

    material = find_material(MEDIA)
    # Checked before the rollout, which takes minutes: the music is optional for
    # training, so this is the first place its absence matters, and the policy and
    # its contract are already written by the time this runs.
    if material.audio is None:
        raise FileNotFoundError(
            f"no music in {material.motion.parent} (looked for "
            f"{', '.join(AUDIO_SUFFIXES)}). The policy and its contract are written; "
            f"the performance video is the dance with its music on it, so it needs "
            f"the track the choreography was made for. Put it there and export "
            f"again, or pass --no-video."
        )
    media = Path(out) / "media"
    media.mkdir(parents=True, exist_ok=True)

    print(f"\n[export] rendering the performance -> {media}")
    qpos, model, fps, frames_total, stopped_early = _rollout(
        task_id, asset, Path(checkpoint), res
    )

    if stopped_early is not None:
        print(
            f"[export] WARNING: the policy terminated at frame {stopped_early} of "
            f"{frames_total} ({stopped_early / fps:.1f} s of "
            f"{frames_total / fps:.1f} s). The video is the performance up to that "
            f"point, not the whole choreography."
        )

    raw = media / "dance.silent.mp4"
    _render(model, qpos, fps, raw)

    # The offset belongs to the choreography, so it is read from the clip rather
    # than written down here. See the module docstring.
    clip = load_source(material.motion)
    final = media / "dance.mp4"
    _mux(raw, material.audio, clip.audio_start_s, final)
    raw.unlink()

    shutil.copy2(material.audio, media / f"music{material.audio.suffix.lower()}")
    if material.eyes is not None:
        # Named, because the face animation is identified by being the one video in
        # `media/` -- so a reference recording left there is shipped as the robot's
        # face without anything else noticing. The name in the log is what makes
        # that visible.
        print(f"[export] face animation: {material.eyes.name}")
        shutil.copy2(material.eyes, media / f"eyes{material.eyes.suffix.lower()}")
    else:
        print(
            f"[export] no face animation found; put one in "
            f"{material.motion.parent}/ to ship it with the performance."
        )

    for f in sorted(media.iterdir()):
        print(f"    media/{f.name:<12}{f.stat().st_size / 1024 / 1024:8.1f} MB")
