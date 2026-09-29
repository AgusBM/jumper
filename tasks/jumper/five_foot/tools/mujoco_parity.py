#!/usr/bin/env python3
"""Does `tasks/jumper/five_foot/tools/pick_place.py` build the policy's input the way the trainer does?

    python tasks/jumper/five_foot/tools/mujoco_parity.py

The standalone MuJoCo build assembles the observation itself, from
`layout.json`, because that is what a deployment does. So it can be wrong in the
one way that produces no error: a vector of the right length holding the wrong
numbers, fed to a policy that will happily act on it. The robot then stands
there, or walks badly, and nothing says why.

This puts the **same physical state** through both assemblers and compares them
term by term. The state comes from a live mjlab environment; it is written into a
plain `MjData` of the standalone model, and the two vectors are lined up against
the offsets the contract declares.

It does not check the physics -- two MuJoCo models with the same state are the
same model -- only the arithmetic between state and policy input, which is where
a deployment goes wrong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="jumper.five_foot")
    ap.add_argument("--bundle", default="tasks/jumper/five_foot/out/model_6999")
    ap.add_argument("--checkpoint",
                    default="logs/hexa/hexa.five_foot/2026-09-11_21-16-42/model_6999.pt")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--tol", type=float, default=2e-5)
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import mujoco
    import torch
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    import tasks

    res = resolve(backend="warp", device="cuda", num_envs=1)
    use_backend(res)
    # By path, because this directory is not a package and this repository does
    # not manipulate `sys.path` to pretend otherwise (see `scripts/_cli.py`).
    import importlib.util
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    spec = importlib.util.spec_from_file_location(
        "pick_place", Path(__file__).resolve().parent / "pick_place.py"
    )
    pick_place = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pick_place)
    Observer, build_model = pick_place.Observer, pick_place.build_model

    layout = json.loads((REPO / args.bundle / "layout.json").read_text())
    cfg = tasks.load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    # **The encoder bias comes off.** It is a simulation-only term -- the export
    # README says in so many words not to reproduce it on hardware -- and the
    # observation uses the biased joint position, so leaving it on would compare
    # a deployment that is right against a simulation that is deliberately
    # lying to itself. Measured with it on: joint_pos differs by 1.5e-02, which
    # is exactly the bias.
    for name in list(cfg.events):
        if "encoder" in name or "bias" in name:
            cfg.events.pop(name)
    env = ManagerBasedRlEnv(cfg=cfg, device=res.device)
    agent = tasks.load_agent_cfg(args.task)
    runner = (tasks.load_runner_cls(args.task) or MjlabOnPolicyRunner)(
        RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions), asdict(agent),
        device=res.device)
    runner.load(str(REPO / args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=res.device)
    policy = runner.get_inference_policy(device=res.device)

    # The standalone side: its own model, its own data, its own assembler.
    model, _ = build_model(with_objects=False)
    data = mujoco.MjData(model)
    obs = Observer(layout, model)
    robot = env.scene["robot"]
    wire = layout["wire_joint_order"]
    ids = robot.find_joints(wire, preserve_order=True)[0]
    qadr = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in wire
    ])
    dadr = np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in wire
    ])
    obs_cols = np.array([wire.index(n) for n in layout["obs_joint_order"]])

    def mirror():
        """Put the environment's state into the standalone MjData."""
        data.qpos[0:3] = (robot.data.root_link_pos_w[0]
                          - env.scene.env_origins[0]).cpu().numpy()
        data.qpos[3:7] = robot.data.root_link_quat_w[0].cpu().numpy()
        data.qpos[qadr] = robot.data.joint_pos[0, ids].cpu().numpy()
        # **The COM velocity, not the link's.** A free joint's linear qvel is the
        # velocity of the body's centre of mass in world coordinates, and mjlab
        # reports both -- `root_link_lin_vel_w` is the frame origin's, which
        # differs by omega x (com - origin). Mirroring the wrong one injects a
        # velocity error the size of the CoM offset times the spin rate.
        data.qvel[0:3] = robot.data.root_com_lin_vel_w[0].cpu().numpy()
        # A free joint's rotational velocity is in the **body** frame, and the
        # entity reports it in the world's. Mirroring the world vector straight
        # in made the gyro read 4.16 rad/s of difference -- a harness bug that
        # looked exactly like a broken observation.
        quat = robot.data.root_link_quat_w[0].cpu().numpy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        data.qvel[3:6] = rot.reshape(3, 3).T @ (
            robot.data.root_com_ang_vel_w[0].cpu().numpy()
        )
        data.qvel[dadr] = robot.data.joint_vel[0, ids].cpu().numpy()
        mujoco.mj_forward(model, data)

    worst: dict[str, float] = {}
    action = np.zeros(len(layout["action_joint_order"]), dtype=np.float32)
    with torch.inference_mode():
        for step in range(args.steps):
            mine = env.observation_manager.compute()["actor"][0].cpu().numpy()
            mirror()
            torque = robot.data.actuator_force[0, ids].cpu().numpy()[obs_cols]
            twist = env.command_manager.get_command("twist")[0, :3].cpu().numpy()
            pose = env.command_manager.get_command("body_pose")[0].cpu().numpy()
            theirs = obs(data, action, twist, pose, torque, reset=(step == 0))
            for term in layout["observation"]["terms"]:
                lo, hi = term["offset"], term["offset"] + term["dim"]
                d = float(np.abs(mine[lo:hi] - theirs[lo:hi]).max())
                worst[term["name"]] = max(worst.get(term["name"], 0.0), d)
            act = policy(env.observation_manager.compute())
            action = act[0].cpu().numpy()
            env.step(act)

    # ── Second half: the same state and the same action, one step ────────
    # The observation is only half of what a standalone build has to reproduce.
    # This mirrors the environment's state into the plain model, applies the
    # action both sides, steps both one control step, and compares where each
    # ended up -- which isolates the dynamics from the policy and from drift.
    Servo = pick_place.Servo
    servo = Servo(model, wire)
    default = np.array([layout["default_joint_pos"][n] for n in wire])
    act_cols = np.array([wire.index(n) for n in layout["action_joint_order"]])
    decimation = layout["control"]["decimation"]
    worst_q = worst_v = worst_driven = worst_locked = 0.0
    worst_name = ""
    with torch.inference_mode():
        for _ in range(args.steps):
            mirror()
            act = policy(env.observation_manager.compute())
            target = default.copy()
            target[act_cols] += act[0].cpu().numpy() * layout["action_scale"]
            for _ in range(decimation):
                servo.apply(data, target)
                mujoco.mj_step(model, data)
            env.step(act)
            theirs_q = data.qpos[qadr].copy()
            theirs_v = data.qvel[dadr].copy()
            mine_q = robot.data.joint_pos[0, ids].cpu().numpy()
            mine_v = robot.data.joint_vel[0, ids].cpu().numpy()
            dq, dv = np.abs(theirs_q - mine_q), np.abs(theirs_v - mine_v)
            if dq.max() > worst_q:
                worst_q, worst_name = float(dq.max()), wire[int(dq.argmax())]
            worst_v = max(worst_v, float(dv.max()))
            driven = np.array([wire.index(n) for n in layout["action_joint_order"]])
            locked = np.array([i for i in range(len(wire)) if i not in set(driven)])
            worst_driven = max(worst_driven, float(dq[driven].max()))
            worst_locked = max(worst_locked, float(dq[locked].max()))
    print("\none control step from the same state and action:")
    print(f"  worst joint position {worst_q * 1000:7.3f} mm-equivalent rad*1000 "
          f"({worst_name})")
    print(f"  worst joint velocity {worst_v:7.3f} rad/s")
    print(f"  worst of the 16 driven joints {worst_driven:7.4f} rad")
    print(f"  worst of the 6 locked joints  {worst_locked:7.4f} rad "
          f"(the carried arm: this environment samples its hold, the standalone "
          f"holds the nominal)")

    bad = 0
    # ── The physics settings ─────────────────────────────────────────────
    # Written out in the standalone file rather than imported, so this is what
    # keeps the two copies together.
    live = cfg.sim.mujoco
    off = [
        f"{key}: standalone {value} against the task's {getattr(live, key)}"
        for key, value in pick_place.SIM.items()
        if abs(float(getattr(live, key)) - float(value)) > 1e-12
    ]
    bad += len(off)
    print("physics settings match the task" if not off
          else "physics settings differ: " + "; ".join(off))

    # ── The torque curve, exactly ────────────────────────────────────────
    # Not the applied torque: that is a comparison between two different
    # moments, because the force a step reports is the last substep's and the
    # state read after it is later still. The *law* can be compared exactly, and
    # a law that agrees plus a state that does not is a solver difference rather
    # than an actuator one.
    from tasks.jumper.common.actuator import servo_torque_limit
    from tasks.jumper.common.constants import JUMPER_ACTUATOR_CFG as A

    speeds = np.linspace(-80.0, 80.0, 401)
    theirs = servo.available(speeds)
    mine = servo_torque_limit(
        torch.tensor(speeds), A.plateau_torque, A.corner_speed, A.decay_speed,
        A.cutoff_speed,
    ).numpy()
    curve = float(np.abs(theirs - mine).max())
    print(f"\ntorque curve over -80..80 rad/s: worst {curve:.3e} N*m "
          f"{'ok' if curve < 1e-6 else 'DIFFERS'}")

    print(f"{'term':<22}{'worst difference':>18}")
    for name, d in worst.items():
        ok = d <= args.tol
        bad += not ok
        print(f"  {name:<20}{d:18.3e}   {'ok' if ok else 'DIFFERS'}")
    env.close()
    print("\n" + ("the two assemblers agree" if not bad else
                  f"{bad} term(s) differ: the standalone build is feeding the "
                  f"policy something else"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
