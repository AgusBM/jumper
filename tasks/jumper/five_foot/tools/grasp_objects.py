#!/usr/bin/env python3
"""End-to-end: can the claw actually pick up each object in the row?

Nothing is welded. The claw holds what it squeezes, by friction, under the row's
own friction settings (`objects.py::PROP_CONE`), and every part of that can be
right on its own while the robot still holds nothing -- the jaws' collision
geometry, where the prop sits in the mouth, whether the row's cone reached the
model, how hard the finger's servo squeezes. None of those fails loudly. This
drives the whole path per object and reports what happened, so "the claw picks
things up" is a measurement rather than a recollection.

    python tasks/jumper/five_foot/tools/grasp_objects.py
    python tasks/jumper/five_foot/tools/grasp_objects.py --backend native
    python tasks/jumper/five_foot/tools/grasp_objects.py --checkpoint <path>.pt

The task is built the way `play.py --objects` builds it, so what this checks is the
row an operator gets.

For each graspable prop it:

  1. opens the claw, puts the prop `HOLD_DEPTH` into the mouth and holds it
     there, then **lets go of it and closes the claw on it** -- never both at
     once: a prop held in place while the finger squeezes cannot yield, and the
     overlap throws it out when it is let go (`mdp/grasp.py::PLACE_STEPS`);
  2. raises the claw with the shoulder: **grasped** is a prop that came up with
     it, at least `FOLLOWED` of the way;
  3. swings the shoulder for two seconds and measures how far the prop moved
     **relative to the claw**: **carried** is under `CARRY_SLIP_MM`;
  4. opens the jaws: **released** is a prop back on the floor. A claw that cannot
     put anything down is broken too.

Then one negative control: a prop parked beside the claw, out of the mouth, let go
of and the claw closed, must **not** come up with the claw. Without
it every "grasped" above could be a prop resting on a jaw rather than squeezed
between them.

Last, a reset: a prop squeezed in the claw, the rest of the row out of the way,
and then `env.reset()` -- which has to bring every prop home, or restarting the
episode leaves half of the old one behind.

Before any of that it checks two things that fail silently in the least helpful
way. **Where the episode starts**: the robot on its spawn facing the row,
commanding nothing (`objects.py::START`) -- otherwise the replay opens with the
robot walking off on a sampled command. And **which friction cone the model runs**:
a row whose `PROP_CONE` never reached the model builds, runs, and drops everything
the claw picks up.

Runs on either backend. Run it from the repository root, like everything else here
-- the editable install is what puts `tasks` on the path, and nothing
in this repository manipulates `sys.path` to paper over a missing one.
"""

from __future__ import annotations

import argparse
import math

#: How far a prop has to rise, as a share of how far the claw's anvil rose, to
#: have come up with the claw. A held prop sits further out along the claw than
#: the anvil, so it rises by more -- measured 94 to 120% -- and one left behind
#: rises by nothing.
FOLLOWED = 0.8

#: The negative control's ceiling on the same share: a prop beside the claw that
#: rises by a fifth of the claw has been carried by something.
NOT_FOLLOWED = 0.2

#: How far a carried prop may move relative to the claw over the swing, in mm.
#: Measured walking a carried prop back 0.6 m and turning it: 4.5 to 10.6 mm
#: (`tasks/jumper/five_foot/objects.py::PROP_CONE`).
CARRY_SLIP_MM = 20.0

#: How far above where it stood before the lift a released prop may still be, in
#: mm: back on the floor, upright or on its side.
RELEASED_MM = 5.0

#: Shoulder travel, in radians: the lift, and the swing about the lifted angle.
#: The swing is smaller than the lift so the claw never comes back down past where
#: it closed and puts the prop on the floor, which would read as slip.
LIFT_RAD = 0.30
SWING_RAD = 0.20


def _let_go(settle: int, step, grip: float) -> None:
    """Open the jaws, which is what drops an object.

    A helper rather than a bare `step(...)` because the release path is the one
    thing here with no second witness: nothing else in the run commands the claw
    open, so a claw that never let go would otherwise be reported as one that was
    never asked to.
    """
    step(settle, grip)


def _stash(env: object, name: str, device: object, index: int) -> None:
    """Put a finished prop out of reach, standing on the floor.

    On the floor rather than in the air, and each at its own spot: a prop left
    hanging above the claw falls back into the mouth, which is how the negative
    control below first reported a grab it had set up itself.
    """
    import torch

    angle = 2.0 * math.pi * index / 8.0 + 0.4
    entity = env.scene[name]
    pos = torch.tensor(
        [[3.0 * math.cos(angle), 3.0 * math.sin(angle), 0.1]], device=device
    )
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    entity.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=1))
    entity.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=device))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="jumper.five_foot")
    p.add_argument("--backend", choices=("warp", "native"), default="warp")
    p.add_argument("--checkpoint", default=None,
                   help="a policy to hold the robot up; without one it stands on "
                        "its reset pose, which is enough to test the claw")
    p.add_argument("--settle", type=int, default=60)
    p.add_argument("--close", type=int, default=80)
    p.add_argument("--lift", type=int, default=50)
    p.add_argument("--sweep", type=int, default=100)
    args = p.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import mujoco
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend=args.backend,
                  device="cuda" if args.backend == "warp" else "cpu", num_envs=1)
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv

    from tasks.jumper.five_foot.claw import (
        FINGER_JOINT,
        GRIPPER_CLOSED,
        GRIPPER_OPEN,
        SQUEEZE_LEAD,
    )
    from tasks.jumper.five_foot.mdp.grasp import HOLD_DEPTH, Claw, _rotate
    from tasks.jumper.five_foot.objects import GRASPABLE, PROP_CONE, PROP_IMPRATIO

    cfg = tasks.load_env_cfg(args.task, play=True, task_args={"objects": True})
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)

    policy = None
    if args.checkpoint:
        from dataclasses import asdict

        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.rl.runner import MjlabOnPolicyRunner

        agent = tasks.load_agent_cfg(args.task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = (tasks.load_runner_cls(args.task) or MjlabOnPolicyRunner)(
            wrapped, asdict(agent), device=res.device
        )
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                    map_location=res.device)
        policy = runner.get_inference_policy(device=res.device)

    props = GRASPABLE
    if not props:
        print("no graspable props in the row; nothing to check")
        return 1

    robot = env.scene["robot"]
    finger = robot.find_joints([FINGER_JOINT])[0][0]
    shoulder = robot.find_joints(["LF_J1_joint"])[0][0]
    claw = Claw(env)
    zero = torch.zeros(env.action_space.shape, device=res.device)
    one = torch.zeros(1, 1, device=res.device)
    obs = None

    def step(n: int, grip: float, park: str | None = None,
             offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
             shoulder_at: float | None = None) -> None:
        nonlocal obs
        with torch.inference_mode():
            for _ in range(n):
                if policy is None:
                    env.step(zero)
                else:
                    # The whole dict, not one group: the model concatenates the
                    # groups its own config names (`obs_groups`), so handing it a
                    # single tensor indexes a 2-d tensor with a string.
                    obs = env.observation_manager.compute()
                    env.step(policy(obs))
                # Squeezed as the operator squeezes and no harder
                # (`claw.py::SQUEEZE_LEAD`), so a claw that only holds at the
                # servo's full torque fails here rather than in replay.
                one.fill_(min(grip, float(robot.data.joint_pos[0, finger]) + SQUEEZE_LEAD))
                robot.set_joint_position_target(one, joint_ids=[finger])
                if shoulder_at is not None:
                    one_s = torch.full((1, 1), shoulder_at, device=res.device)
                    robot.set_joint_position_target(one_s, joint_ids=[shoulder])
                if park is not None:
                    # Held in place while the claw is open, so it is where the
                    # seat says when the jaws start to close -- and only then:
                    # `mdp/grasp.py::PLACE_STEPS` has what holding it through
                    # the squeeze does.
                    entity = env.scene[park]
                    pos = claw.seat(env, park) + torch.tensor([offset], device=res.device)
                    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=res.device)
                    entity.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=1))
                    entity.write_root_com_velocity_to_sim(
                        torch.zeros(1, 6, device=res.device)
                    )

    def height(name: str) -> float:
        return float(env.scene[name].data.root_link_pos_w[0, 2])

    def anvil_height() -> float:
        return float(claw.mouth()[0, 2])

    def in_claw(name: str) -> torch.Tensor:
        """Where `name` is relative to the anvil, in the palm's frame."""
        inverse = robot.data.body_link_quat_w[:, claw.palm].clone()
        inverse[:, 1:] = -inverse[:, 1:]
        return _rotate(inverse, env.scene[name].data.root_link_pos_w - claw.mouth())[0]

    def lift(name: str, rest: float, grip: float) -> tuple[float, float]:
        """Raise the claw by `LIFT_RAD`: (how far `name` rose, how far the anvil did)."""
        z0, a0 = height(name), anvil_height()
        half = max(1, args.lift // 2)
        for k in range(half):
            step(1, grip, shoulder_at=rest + raise_by * (k + 1) / half)
        step(args.lift - half, grip, shoulder_at=rest + raise_by)
        return height(name) - z0, anvil_height() - a0

    print(f"--objects  backend={args.backend}  props={', '.join(props)}\n")
    failures = 0

    # ── The friction the props are held under ────────────────────────────
    model = env.sim.mj_model
    cone = "elliptic" if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else "pyramidal"
    wanted = (PROP_CONE, float(PROP_IMPRATIO))
    same = (cone, float(model.opt.impratio)) == wanted
    failures += not same
    print(f"friction       the model runs the {cone} cone at impratio "
          f"{model.opt.impratio:g}; the row asks for {wanted[0]} at {wanted[1]:g}   "
          f"{'ok' if same else '<- NOT what the row asked for'}")

    # ── Where the episode starts ─────────────────────────────────────────
    # Read straight after a reset, which is the only moment the spawn is visible:
    # a few steps later the robot has settled, slumped or walked, and none of
    # those is distinguishable from having started somewhere else.
    env.reset()
    origin = env.scene.env_origins[0]
    start = (robot.data.root_link_pos_w[0] - origin).clone()
    yaw = float(torch.atan2(
        2.0 * (robot.data.root_link_quat_w[0, 0] * robot.data.root_link_quat_w[0, 3]
               + robot.data.root_link_quat_w[0, 1] * robot.data.root_link_quat_w[0, 2]),
        1.0 - 2.0 * (robot.data.root_link_quat_w[0, 2] ** 2
                     + robot.data.root_link_quat_w[0, 3] ** 2),
    ))
    placed = abs(float(start[0])) < 0.01 and abs(float(start[1])) < 0.01 \
        and abs(yaw) < 0.02
    failures += not placed
    print(f"start          robot at ({float(start[0]):+.3f}, {float(start[1]):+.3f}) "
          f"m, yaw {math.degrees(yaw):+.1f} deg   "
          f"{'on its spawn, facing the row' if placed else '<- NOT the spawn'}")

    # The command, over a couple of seconds of stepping rather than at t=0: the
    # sampler resamples on its own schedule, so one reading proves nothing about
    # whether the robot is going to stay put.
    worst = 0.0
    for _ in range(args.settle):
        step(1, GRIPPER_OPEN)
        for cmd_name in env.command_manager.active_terms:
            worst = max(worst, float(
                env.command_manager.get_command(cmd_name)[0].abs().max()
            ))
    still = worst < 1e-6
    failures += not still
    drift = float((robot.data.root_link_pos_w[0, :2] - origin[:2] - start[:2]).norm())
    print(f"command        largest commanded value over {args.settle} steps: "
          f"{worst:.3f}   {'nothing commanded' if still else '<- NOT still'}")
    print(f"drift          {drift * 1000:.0f} mm in that time "
          f"{'(no policy: the robot is holding its reset pose)' if policy is None else ''}")

    # Home, for the reset check at the end. Read **after** a reset rather than at
    # construction: mjlab applies `env_origins` inside the prop reset event, so
    # until one has run every prop reads as being at the world origin.
    home = {n: env.scene[n].data.root_link_pos_w[0].clone() for n in props}

    # And where the row is from here, which is what "a little in front of the
    # objects" has to mean in metres.
    here = claw.mouth()[0]
    for name in props:
        to = env.scene[name].data.root_link_pos_w[0] - here
        print(f"  {name:<11} {float(to[0]) * 1000:+5.0f} mm ahead of the mouth, "
              f"{float(to[1]) * 1000:+5.0f} mm across, {float(to[2]) * 1000:+5.0f} mm up")

    # Which way the shoulder raises the claw, asked of the robot rather than
    # assumed: the sign depends on how the joint is mounted.
    rest = float(robot.data.joint_pos[0, shoulder])
    before = anvil_height()
    step(40, GRIPPER_OPEN, shoulder_at=rest - LIFT_RAD)
    raise_by = -LIFT_RAD if anvil_height() > before else LIFT_RAD
    step(40, GRIPPER_OPEN, shoulder_at=rest)
    print(f"  the shoulder raises the claw at {raise_by:+.2f} rad; props go "
          f"{HOLD_DEPTH * 1000:.0f} mm into the mouth\n")

    print(f"{'object':<12}{'grasped':>26}{'carried':>12}{'released':>10}   verdict")
    for i, name in enumerate(props):
        step(args.settle, GRIPPER_OPEN, park=name, shoulder_at=rest)
        step(args.close, GRIPPER_CLOSED, shoulder_at=rest)
        z_before = height(name)
        rose, claw_rose = lift(name, rest, GRIPPER_CLOSED)
        share = rose / claw_rose if claw_rose > 1e-4 else 0.0
        grasped = claw_rose > 0.005 and share >= FOLLOWED

        # Swing about the lifted angle, and watch the prop in the claw's frame.
        top = rest + raise_by
        was = in_claw(name).clone()
        for k in range(args.sweep):
            step(1, GRIPPER_CLOSED, shoulder_at=top + SWING_RAD * math.sin(k / 16.0))
        slip_mm = float((in_claw(name) - was).norm()) * 1000.0
        carried = grasped and slip_mm < CARRY_SLIP_MM

        _let_go(args.settle, step, GRIPPER_OPEN)
        released = (height(name) - z_before) * 1000.0 < RELEASED_MM

        ok = grasped and carried and released
        failures += not ok
        rise = f" ({share:.0%} of {claw_rose * 1000:.0f} mm)"
        print(f"{name:<12}{('yes' if grasped else 'NO') + rise:>26}"
              f"{(f'{slip_mm:.1f} mm' if grasped else '-'):>12}"
              f"{'yes' if released else 'NO':>10}   {'ok' if ok else 'FAILED'}")
        step(args.settle, GRIPPER_OPEN, shoulder_at=rest)
        # Out of the way and **on the floor**: parked in mid-air it falls, and
        # objects raining down around the robot is how the control below first
        # "failed" -- one of them landed in the mouth.
        _stash(env, name, res.device, i)

    # Negative control: beside the mouth, held while the claw closes, never in it.
    decoy = props[0]
    aside = (0.12, 0.0, 0.0)
    step(args.settle, GRIPPER_OPEN, park=decoy, offset=aside, shoulder_at=rest)
    step(args.close, GRIPPER_CLOSED, shoulder_at=rest)
    rose, claw_rose = lift(decoy, rest, GRIPPER_CLOSED)
    share = rose / claw_rose if claw_rose > 1e-4 else 0.0
    caught = share >= NOT_FOLLOWED
    failures += caught
    print(f"\n{'(control)':<12}{decoy} parked 120 mm aside, claw shut and raised: it rose "
          f"{share:.0%} of the claw"
          f"{'  <- the checks above prove nothing' if caught else ', as it should'}")
    _let_go(args.settle, step, GRIPPER_OPEN)
    step(args.settle, GRIPPER_OPEN, shoulder_at=rest)

    # ── A reset: the row comes back ──────────────────────────────────────
    # Every prop is 3 m away on the floor by now and one is about to be squeezed
    # in the claw, so a reset has both halves to undo. Run **inside inference
    # mode**, the way a replay loop steps: the managers' buffers are then inference
    # tensors, and resetting one from outside raises `Inplace update to inference
    # tensor outside InferenceMode` instead of restarting.
    step(args.settle, GRIPPER_OPEN, park=decoy, shoulder_at=rest)
    step(args.close, GRIPPER_CLOSED, shoulder_at=rest)
    holding = float(in_claw(decoy).norm()) < 0.10
    with torch.inference_mode():
        env.reset()
    step(1, GRIPPER_OPEN)
    worst, where = 0.0, ""
    for name in props:
        moved = float((env.scene[name].data.root_link_pos_w[0] - home[name]).norm())
        if moved > worst:
            worst, where = moved, name
    reset_ok = holding and worst < 0.001
    failures += not reset_ok
    print(f"\n(reset)     holding {'yes' if holding else 'NO'}, then reset: "
          f"props back to within {worst * 1000:.1f} mm ({where})   "
          f"{'ok' if reset_ok else 'FAILED'}")

    env.close()
    print(f"\n{'FAILED' if failures else 'all objects pick up, carry and release'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
