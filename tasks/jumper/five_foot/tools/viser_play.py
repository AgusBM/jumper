#!/usr/bin/env python3
"""Replay a checkpoint in a **browser**, for a machine you are not sitting at.

    python tasks/jumper/five_foot/tools/viser_play.py --task jumper.five_foot --objects
    # then open the URL it prints (VS Code forwards the port automatically)

`scripts/play.py` opens MuJoCo's native window, which needs a display: on a
remote box there is none, and X11 forwarding a 3D scene is slow enough to be
useless for judging a gait. mjlab ships a Viser viewer -- the same scene served
over WebSockets and drawn by the browser -- and this is the thin wrapper that
points it at a task from this repository's registry.

It is not in `scripts/` deliberately. `scripts/` holds the three entry points whose
arguments are a stable contract (`tests/test_log_layout.py` pins the list), and this
is a development convenience: a second way to look at the same thing, not a fourth
thing to do. It came with this task and lives beside it, but nothing in it is this
task's: any `--task` works, and the task's own flags (`--objects`, `--hold` here)
are asked of the task the way `scripts/_cli.py::parse_with_task_args` asks.

The heavy lifting is all mjlab's; what is here is the registry lookup, the scene,
and the same checkpoint resolution `play.py` and `train.py --resume` share, so
"the newest checkpoint" cannot mean two different files depending on which command
is asking.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, help="task id; see train.py --list")
    parser.add_argument("--model", default=None, help="asset name or .xml path")
    parser.add_argument("--scene", default=None, help="scene id; see train.py --list")
    parser.add_argument("--checkpoint", default=None, help="a .pt; default is the newest")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--backend", default="warp", choices=["warp", "native"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--port", type=int, default=8080)

    # The selected task's own flags, in two phases because the task id is itself
    # an argument -- the same shape as `scripts/_cli.py::parse_with_task_args`,
    # which is not importable from here (`scripts/` holds entry points, not a
    # package). `add_help=False` so `--help` waits for the real parse and lists
    # the task's flags too.
    import tasks

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task")
    task_id = pre.parse_known_args()[0].task
    add = tasks.load_cli_args(task_id) if task_id else None
    names = tuple(add(parser.add_argument_group(f"{task_id} options"))) if add else ()
    args = parser.parse_args()
    task_args = {name: getattr(args, name) for name in names}

    import importlib

    from mjrl.backend.select import use_backend

    # **warp, not native, and this is not a preference.** mjlab's Viser viewer
    # opens with `assert isinstance(sim, Simulation)`, and the native backend is
    # exactly the thing that replaces that class (`mjrl.backend.native_sim`, the
    # seam CLAUDE.md describes). Under `--backend native` the assertion fires
    # inside `setup()` with no message, seconds after the checkpoint has loaded
    # and the port is already open, which reads like the viewer failing rather
    # than like a backend mismatch. The flag stays so that is reachable
    # deliberately rather than by accident.
    resolve = importlib.import_module("mjrl.backend.resolve")
    res = resolve.resolve(backend=args.backend, device=args.device,
                          num_envs=args.num_envs)
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.viewer import ViserPlayViewer

    # The same search `play.py` and `train.py --resume` use. Taken from
    # `mjrl.checkpoint` rather than from `scripts/_cli.py`, which is not
    # importable -- `scripts/` holds entry points, not a package -- and rather
    # than re-implemented here, because "the newest checkpoint" meaning two
    # different files depending on which command asked is exactly the failure
    # `_cli.resolve_checkpoint`'s docstring is about.
    from mjrl.checkpoint import find_checkpoint

    from tasks.paths import log_root_for

    spec = tasks.get(args.task)
    asset = spec.resolve_asset(args.model)
    ckpt = find_checkpoint(args.checkpoint, Path(log_root_for(asset)) / spec.id)
    print(f"[viser] checkpoint {ckpt}")

    env_cfg = tasks.load_env_cfg(spec.id, asset=asset, play=True, task_args=task_args)
    if args.scene:
        import scenes

        scenes.apply(env_cfg, args.scene)
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = tasks.load_runner_cls(spec.id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=res.device)
        runner.load(str(ckpt), load_cfg={"actor": True}, strict=True,
                    map_location=res.device)
        policy = runner.get_inference_policy(device=res.device)

        import viser

        server = viser.ViserServer(port=args.port, label="mjrl")
        print(f"[viser] open http://127.0.0.1:{args.port}/  "
              f"(VS Code forwards it; otherwise ssh -L {args.port}:127.0.0.1:{args.port})")
        ViserPlayViewer(wrapped, policy, viser_server=server).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
