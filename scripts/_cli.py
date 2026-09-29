"""The command-line skeleton shared by the entry points under scripts/.

train / play / export differ only in what they do *after* obtaining a TaskSpec, a
Resolution and a model path. Argument parsing, registry lookup, asset resolution
and backend resolution are identical, so they live here.

**Nothing here rewrites sys.path.** `mjrl`, `tasks`, `mjlab` and `rsl_rl` all come
from `pip install -e .` (see the two package roots in pyproject.toml), so imports
depend on the environment rather than on happening to run from the repository
root. Inserting the repository root at the top of every script would make "where
you ran it from" an implicit dependency, and that is exactly the source of the
shadowing trap documented in docs/VENDOR.md.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

from mjrl.dotenv import env_default as _env
from mjrl.dotenv import env_choice, env_flag, load_dotenv
from tasks.paths import REPO_ROOT

# Load .env into os.environ *before* parsing (without overwriting existing values);
# the defaults below then read from os.environ. Precedence follows for free:
#     command line > shell environment > .env.local > .env > built-in default
load_dotenv(REPO_ROOT / ".env", REPO_ROOT / ".env.local")


def _int_env(name: str, fallback: int | None = None) -> int | None:
    """Read an integer environment variable, erroring rather than silently
    falling back to the default when the value is not an integer."""
    raw = _env(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"error: {name}={raw!r} is not an integer (check .env)") from None


def _bool_env(name: str, fallback: bool) -> bool:
    """Read a switch from the environment, erroring rather than silently falling
    back when the value is not one -- the same rule as `_int_env`, and for the same
    reason: a setting that is ignored looks like a feature that does not work."""
    try:
        return env_flag(name, fallback)
    except ValueError as e:
        raise SystemExit(f"error: {e} (check .env)") from None


def _choice_env(name: str, choices: Sequence[str], fallback: str) -> str:
    """Read one of `choices` from the environment, erroring rather than passing an
    unrecognised value on.

    **argparse checks `choices` only for values typed on the command line**, never
    for a default -- so `default=_env(...)` with `choices=[...]` validates nothing,
    and the bad value goes wherever the resolved argument goes.
    """
    try:
        chosen = env_choice(name, choices, fallback)
    except ValueError as e:
        raise SystemExit(f"error: {e} (check .env)") from None
    # `fallback` is a str, so `chosen` never is None; coalesce rather than assert,
    # which `python -O` would strip.
    return chosen if chosen is not None else fallback


#: `--strip-visual`'s three states and what each means to `mjrl.backend.resolve`.
#: **One definition**, so what argparse accepts and what the backend is handed
#: cannot drift apart: they used to be two separate literals, and a value in only
#: one of them was a `KeyError` from inside `resolve_all`.
STRIP_VISUAL: dict[str, bool | None] = {"auto": None, "on": True, "off": False}

#: `--backend`'s three values. `auto` detects; the other two never fall back.
BACKENDS = ("auto", "warp", "native")


def build_parser(
    prog: str, description: str, *, replay: bool = False
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument(
        "--task",
        default=_env("MJRL_TASK"),
        help="task id; see --list. Falls back to MJRL_TASK in .env",
    )
    p.add_argument(
        "--model",
        default=_env("MJRL_MODEL"),
        metavar="NAME|PATH",
        help="which model asset to use: an asset name registered by the task, or a "
        "path to an .xml file (MuJoCo picks its decoder by extension and has none "
        "for .mjcf). Falls back to MJRL_MODEL in .env, then to the task's "
        "default asset",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list every registered task with its available assets, then exit",
    )

    g = p.add_argument_group("backend and device (defaults come from .env)")
    g.add_argument(
        "--backend", default=_choice_env("MJRL_BACKEND", BACKENDS, "auto"),
        choices=BACKENDS,
    )
    g.add_argument(
        "--device", default=_env("MJRL_DEVICE", "auto"),
        help="auto | cuda:0 | cpu. With --backend auto on a machine with CUDA, cpu "
        "still gets warp, whose cpu device is a serial debugging path; pair it "
        "with --backend native for the CPU backend",
    )
    # `--num_envs` is two quantities under one flag, and each has its own `.env`
    # key. Training's is PPO's batch size: every hyper-parameter is tuned against
    # it. A replay's is how many robots are on screen, all running one policy.
    # They shared MJRL_NUM_ENVS once, and the repository's 4096 -- right for
    # training on warp -- built 4096 native CPU environments to replay an app's
    # controller that reads one, and never got past building them.
    if replay:
        g.add_argument(
            "--num_envs", type=int, default=_int_env("MJRL_PLAY_NUM_ENVS", 1),
            help="how many environments to replay, every one running the same "
            "policy. Falls back to MJRL_PLAY_NUM_ENVS in .env, then to 1. "
            "MJRL_NUM_ENVS is training's batch size and is not read",
        )
    else:
        g.add_argument(
            "--num_envs", type=int, default=_int_env("MJRL_NUM_ENVS"),
            help="parallel environments, which is PPO's batch size: changing it "
            "means re-tuning. Falls back to MJRL_NUM_ENVS in .env, then to the "
            "backend's default (warp 4096 / native 64)",
        )
    g.add_argument(
        "--cpu_threads", type=int, default=_int_env("MJRL_CPU_THREADS", 0),
        help="0 means the available cores, at most 8, then capped by num_envs; "
        "past about 8 the step is serial-bound (DEFAULT_MAX_THREADS in "
        "mjrl/backend/resolve.py)",
    )
    g.add_argument(
        "--strip-visual", dest="strip_visual",
        default=_choice_env("MJRL_STRIP_VISUAL", tuple(STRIP_VISUAL), "auto"),
        choices=tuple(STRIP_VISUAL),
        help=(
            "native: strip visual-only meshes. Domain randomisation copies one "
            "MjModel per environment, and most of jumper's 76 MiB per copy is "
            "visual meshes that are never used (5 MiB once stripped). "
            "auto = strip once the per-environment models exceed 2 GiB. "
            "Cost: camera sensors then see collision geometry rather than "
            "appearance meshes (the live viewer is unaffected)"
        ),
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve task, asset and backend, print the result, and build nothing",
    )
    return p


def print_task_table() -> None:
    """List every task. Assets are listed too, since that is where `--model`'s
    valid values come from."""
    import tasks

    specs = tasks.all_specs()
    if not specs:
        print("The registry is empty.")
        return
    width = max(len(s.id) for s in specs)
    print(f"{'task id':<{width}}  description")
    print(f"{'-' * width}  {'-' * 52}")
    for spec in specs:
        print(f"{spec.id:<{width}}  {spec.description}")
        for i, asset in enumerate(spec.assets):
            mark = "default" if i == 0 else "       "
            print(f"{'':<{width}}    --model {asset.name}  [{mark}] {asset.description}")
    print("\n--model also accepts a path to an .xml file.")

    import scenes

    specs = scenes.all_specs()
    if specs:
        width = max(len(s.id) for s in specs)
        print(f"\n{'scene':<{width}}  description  (--scene)")
        print(f"{'-' * width}  {'-' * 52}")
        for spec in specs:
            print(f"{spec.id:<{width}}  {spec.description}")
        print("\nWithout --scene the task's own configuration is left untouched.")


def parse_with_task_args(parser: argparse.ArgumentParser, task: str | None = None):
    """Parse, after letting the selected task add arguments of its own.

    Returns the namespace, with the task's own values collected on it as
    `args.task_args` -- a dict keyed by `dest`, ready to hand straight to
    `tasks.load_env_cfg`. On the namespace rather than beside it so that nothing
    between here and there has to carry a second value it does not use; every
    entry point already passes `args` down whole.

    **Two phases, because the task id is itself an argument.** A throwaway parser
    finds `--task` (falling back to `MJRL_TASK` exactly as the real one does), the
    task adds whatever it takes, and then the real parse runs. The first parser has
    `add_help=False` on purpose: with help enabled, `--help` would be handled in
    phase one and print a usage message missing the very arguments this exists to
    add.

    An unknown task is not an error here -- it is `resolve_all`'s, which says so
    properly and lists the alternatives. This just declines to ask it anything.

    `task` replaces `MJRL_TASK` as the default, for both phases: an entry point
    that knows the task before the command line names one (`play --app`, whose
    app does) and would otherwise add the options of the wrong task's.
    """
    import tasks

    if task is not None:
        parser.set_defaults(task=task)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", default=task or _env("MJRL_TASK"))
    task = pre.parse_known_args()[0].task
    names: tuple[str, ...] = ()
    if task:
        try:
            add = tasks.load_cli_args(task)
        except KeyError:
            add = None
        if add is not None:
            names = tuple(add(parser.add_argument_group(f"{task} options")))
    # The same two phases again for the **scene**, which can also take options of
    # its own (`scenes/rough.py` picks a terrain row and column). Separate from
    # the task's: a run names a task and a scene independently, and collapsing
    # them into one dict would mean a scene and a task could not both use the
    # name `--row`.
    #
    # Guarded on the entry point having `--scene` at all. `export.py` does not --
    # nothing a scene changes reaches an ONNX file -- and offering its options
    # there would be the "accepted and then ignored" flag `add_scene_args`
    # refuses to be.
    scene_names: tuple[str, ...] = ()
    if any(a.dest == "scene" for a in parser._actions):  # noqa: SLF001
        import scenes

        pre_scene = argparse.ArgumentParser(add_help=False)
        pre_scene.add_argument("--scene", default=_env("MJRL_SCENE"))
        scene_id = pre_scene.parse_known_args()[0].scene
        if scene_id:
            try:
                add_scene = scenes.load_cli_args(scene_id)
            except (KeyError, ModuleNotFoundError):
                add_scene = None
            if add_scene is not None:
                scene_names = tuple(
                    add_scene(parser.add_argument_group(f"{scene_id} scene options"))
                )

    args = parser.parse_args()
    args.task_args = {name: getattr(args, name) for name in names}
    args.scene_args = {name: getattr(args, name) for name in scene_names}
    return args


def resolve_all(args: argparse.Namespace, *, for_training: bool = False):
    """Return (TaskSpec, Resolution, asset).

    Any failure exits with an actionable message. `asset` is the resolved model
    path, or None when the task registers no assets. `for_training` is the entry
    point saying which it is, so that an argument only one of them accepts is
    refused here -- where `--dry-run` can still see it.
    """
    import tasks
    from mjrl.backend import BackendUnavailable, resolve
    from tasks.registry import error_text

    if not args.task:
        print(
            "error: no task. Pass --task, or set MJRL_TASK in the repository's .env.\n"
            "Use --list to see the available tasks.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        spec = tasks.get(args.task)
    except KeyError as e:
        print(f"error: {error_text(e)}", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        asset: Path | None = spec.resolve_asset(args.model)
    except (KeyError, FileNotFoundError) as e:
        print(f"error: {error_text(e)}", file=sys.stderr)
        raise SystemExit(2) from None

    try:
        res = resolve(
            backend=args.backend,
            device=args.device,
            num_envs=args.num_envs,
            cpu_threads=args.cpu_threads,
            strip_visual=STRIP_VISUAL[getattr(args, "strip_visual", "auto")],
            training=for_training,
        )
    except BackendUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1) from None

    # Validate --scene here, though it is applied much later, so that --dry-run
    # catches a typo. Checking arguments is the whole purpose of --dry-run, and a
    # misspelled scene that only surfaces after the environment is built would
    # get past exactly the check meant to find it. Same reason --model is
    # resolved here rather than at use.
    if getattr(args, "scene", None):
        import scenes
        from scenes.package import PackageError, is_package_path, read_package

        try:
            if is_package_path(args.scene):
                # Read it now rather than at use, for the same reason as the
                # registry lookup: --dry-run exists to catch a bad argument, and
                # a package whose hashes do not match is one. So is a package
                # handed to training -- and that refusal has to be here rather
                # than where scenes are applied, because --dry-run stops first.
                refuse_package_for_training(read_package(args.scene), for_training)
            else:
                scenes.get(args.scene)
        except KeyError as e:
            print(f"error: {error_text(e)}", file=sys.stderr)
            raise SystemExit(2) from None
        except PackageError as e:
            print(f"error: {e}", file=sys.stderr)
            raise SystemExit(2) from None

    print(res.banner())
    if asset is not None:
        print(f"[mjrl] task={spec.id} model={asset}")
    return spec, res, asset


def add_scene_args(parser: argparse.ArgumentParser) -> None:
    """Add `--scene`. Only the entry points that put a picture on screen take it.

    Not in the shared parser, for the same reason `--headless` is not: export
    produces an ONNX file and a contract, and nothing a scene changes can reach
    either. A flag that is accepted and then ignored is worse than a missing one
    -- it reads as supported.
    """
    parser.add_argument(
        "--scene",
        default=_env("MJRL_SCENE"),
        metavar="NAME",
        help="how the world looks: ground, sky and lights; see --list. Also takes a "
        "path to an imported kk-scene-package/1 (a directory or .zip), which "
        "scripts/play.py accepts and scripts/train.py refuses. Falls back "
        "to MJRL_SCENE in .env, and to the task's own configuration when neither "
        "is given",
    )


def apply_scene(env_cfg, args, *, for_training: bool = False) -> None:
    """Apply `--scene` to a freshly built environment config.

    After the task rather than before: the task configures the ground it was
    designed for, and a scene is the user overruling that look for this run.
    Passing no scene leaves the config alone, so the default behaviour is
    unchanged rather than merely equivalent.

    `for_training` is the caller saying which entry point it is, because an
    imported scene package is refused by one of them -- see
    `refuse_package_for_training`.
    """
    scene_id = getattr(args, "scene", None)
    if not scene_id:
        return

    import scenes
    from scenes.package import PackageError, is_package_path
    from tasks.registry import error_text

    if is_package_path(scene_id):
        try:
            _apply_package(env_cfg, scene_id, for_training=for_training)
        except PackageError as e:
            print(f"error: {e}", file=sys.stderr)
            raise SystemExit(2) from None
        return

    try:
        scene = scenes.apply(env_cfg, scene_id, **getattr(args, "scene_args", {}))
    except (KeyError, ModuleNotFoundError) as e:
        print(f"error: {error_text(e)}", file=sys.stderr)
        raise SystemExit(2) from None
    # Say what it changed, in the order that matters. "look only" beside a
    # warning that the scene just added a ball is worse than saying nothing:
    # the banner is the line people skim, and it was contradicting the warning
    # directly above it.
    changed = []
    if scene.props:
        changed.append(f"{len(scene.props)} physical prop(s)")
    if scene.terrain_type:
        changed.append("the ground")
    changed.append("the look")
    print(f"[mjrl] scene={scene_id} (changes {', '.join(changed)})")


SCHEMA_LABEL = "kk-scene-package/1"


def refuse_package_for_training(package, for_training: bool) -> None:
    """Refuse an imported package on the training entry point.

    The refusal is the feature. The package is one room at the world origin and
    mjlab replicates nothing that is not an entity, so at any real `--num_envs`
    exactly one robot is indoors and every other one trains on empty ground.
    Training runs, converges to something, and nothing anywhere reports a
    problem -- the same failure class as rough terrain with no height scan.
    Refusing beats warning, because a warning here would be about a run that is
    already meaningless.
    """
    if not for_training:
        return
    print(
        f"error: scene package {package.id!r} cannot be used for training. It is one "
        f"room at the world origin, and mjlab places environments on a grid -- only the "
        f"robot at the origin would be inside it, while every other environment trained "
        f"on empty ground. Nothing would report that. Use it with scripts/play.py.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _apply_package(env_cfg, source: str, *, for_training: bool) -> None:
    """Apply an imported `kk-scene-package/1`.

    Routed through `scenes.registry.apply_scene`, the same function a registered
    scene goes through, so an imported one cannot quietly skip a check: the prop
    name clash, the per-environment placement, the contact budget and the warning
    that physical props change the run are all in there.

    Unlike a registered scene, this one's geometry **collides**: a room the robot
    walks through is not a room. `Scene.decorate` may normally only add geometry
    the robot cannot touch, because a look should not change the task -- and the
    refusal above, rather than a weaker room, is what keeps that protection where
    it matters.
    """
    import scenes
    from scenes.package import load_package, read_package

    package = read_package(source)
    refuse_package_for_training(package, for_training)
    scene = scenes.apply_scene(env_cfg, load_package(source), package.id)

    counts = package.manifest.get("world", {}).get("counts", {})
    print(
        f"[mjrl] scene={package.id} (imported {SCHEMA_LABEL}: {counts.get('geoms', '?')} geoms, "
        f"{len(scene.props)} entities, {counts.get('lights', '?')} lights; "
        f"nconmax={scene.nconmax} njmax={scene.njmax} measured by settling the world)"
    )
    # Said out loud because it is the one way this differs from every registered
    # scene: those may only add geometry the robot cannot touch.
    print(
        "[mjrl] the imported world collides -- walls and furniture are solid, so contacts "
        "and solver load are not comparable to a run on a bare scene"
    )
    # A package's own `rules` asks a backend that cannot do flex to say so rather
    # than drop it silently. This backend does not drop it, which is worse: it
    # runs, and it runs at a speed that reads as a slow machine. Measured here,
    # 4 environments on an RTX 5090: a package with one 195-vertex cloth ran at
    # 0.4 fps / 0.01x real time against 6.7 fps / 0.13x for one without.
    flex = package.manifest.get("flex") or []
    if flex:
        print(
            f"[mjrl] warning: this package carries {len(flex)} flex (soft body). It "
            f"simulates, but roughly an order of magnitude slower -- 0.01x real time "
            f"against 0.13x for a package without one, measured at 4 environments. "
            f"Nothing else reports this; it reads as a slow machine."
        )

# ── Checkpoints, shared by train.py and play.py ───────────────────────────


def resolve_checkpoint(wanted, task_id: str, asset: Path | None) -> Path:
    """Turn a `--checkpoint` value into a checkpoint that exists.

    `wanted` is None for "the newest one under `logs/<model>/<task>/`", which is
    what both callers mean by the default: for play, the run that just finished;
    for `train --resume`, the training to be continued.

    The search itself is `mjrl.checkpoint`; what this adds is the one thing that
    module deliberately does not know -- that the tree to search is the same
    `logs/<model>/<task>` train writes into -- plus turning the failure into an
    exit, since every caller is a command-line entry point.
    """
    from mjrl.checkpoint import find_checkpoint
    from tasks.paths import log_root_for

    try:
        return find_checkpoint(wanted, Path(log_root_for(asset)) / task_id)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from None


# ── The live viewer, shared by train.py and play.py ───────────────────────


def add_viewer_args(
    parser: argparse.ArgumentParser, ui_default: bool = False, fps_default: float = 30.0
) -> None:
    """Add the live-viewer switches. Both train and play take the same ones.

    Args:
        ui_default: whether MuJoCo's own panels start shown. One of **the two
            switches whose default differs between the two entry points**: `play`
            exists to look at a policy, so the controls are wanted; `train` keeps
            a window open for hours and nobody reaches for them.
        fps_default: frames a second, the other one. `play` is watched closely
            and asks for 60; `train` is glanced at, and 30 halves the copies taken
            on its simulation thread. Either way the drawing itself is on the
            viewer's own thread (see `mjrl/viewer/live.py`), so a higher rate no
            longer slows the world down the way it did.
    """
    g = parser.add_argument_group("live viewer")
    g.add_argument(
        "--headless",
        action="store_true",
        help="do not open the live viewer. **It is on by default**, drawing at most "
        "128 environments (all of them below 128; above that it prints how many it "
        "drew). With no usable display this happens automatically",
    )
    g.add_argument(
        "--viewer-env", type=int, default=None,
        help="which environment the camera follows. Defaults to the one nearest the "
        "**centre of the scene** -- environment 0 sits at a corner of the grid, so "
        "following it leaves two sides empty",
    )
    g.add_argument(
        "--viewer-fps", type=float, default=fps_default,
        help=f"frames a second the window shows; {fps_default:g} by default here. "
        f"Drawing runs on a thread of its own, so this costs the simulation one "
        f"copy of the state per frame and nothing else",
    )
    g.add_argument(
        "--viewer-env-num", type=int, default=None,
        help="how many environments to draw (including the followed one); 128 if not "
        "given. **Not the same as --viewer-env**: that one is which, this one is "
        "how many",
    )
    # `BooleanOptionalAction` rather than `store_true`, because the default
    # differs by entry point: a flag that is on by default needs a way to be
    # turned off, and `--no-viewer-ui` is that way.
    g.add_argument(
        "--viewer-ui", action=argparse.BooleanOptionalAction, default=ui_default,
        help=f"show MuJoCo's own panels -- rendering flags, visualisation "
        f"options, the model tree. **{'On' if ui_default else 'Off'} by default "
        f"here.** Watching a policy is when those controls get used; a training "
        f"run is hours long and nobody touches them, so they only cost frame "
        f"time. Note they edit the *followed* environment only; the rest of the "
        f"drawn field is not theirs to change",
    )


@contextlib.contextmanager
def maybe_viewer(env, args):
    """Attach the live viewer, unless `--headless` or there is no usable display.

    **It is on by default.** With no display it falls back to headless rather than
    crashing: otherwise running over ssh, or from cron, would fail outright because
    the viewer defaults to on, which is far worse than not seeing the picture.
    """
    if args.headless:
        yield None
        return
    from mjrl.viewer.live import no_display_reason

    reason = no_display_reason()
    if reason:
        print(f"[mjrl] going headless: {reason}. Use --headless to say so explicitly")
        yield None
        return

    from mjrl.viewer.live import LiveViewer

    try:
        viewer = LiveViewer(
            env.sim,
            env_index=args.viewer_env,
            fps=args.viewer_fps,
            # Draw the environments nearest the camera. Without origins it would
            # take the first few by index, and 4096 environments form a 64x64 grid
            # -- the first 128 are exactly the two outermost rows.
            env_origins=getattr(env.scene, "env_origins", None),
            max_draw_envs=args.viewer_env_num,
            show_ui=getattr(args, "viewer_ui", False),
        ).start()
    except Exception as e:  # noqa: BLE001
        # A viewer that fails to open must never take the run down with it -- the
        # window is there to look at, it is not part of the run.
        print(f"[mjrl] live viewer failed to open, continuing headless: "
              f"{type(e).__name__}: {e}")
        yield None
        return
    try:
        yield viewer
    finally:
        viewer.stop()


# ── TensorBoard, started alongside training ───────────────────────────────


def add_tensorboard_args(parser: argparse.ArgumentParser) -> None:
    """Add the TensorBoard switches. Only train.py has anything to serve."""
    g = parser.add_argument_group("tensorboard")
    # A pair rather than a lone `--no-tensorboard`: the precedence the whole
    # configuration rests on is "command line > .env", and with only the negative
    # form there is no way back **on** for one run once `.env` has turned it off.
    g.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=_bool_env("MJRL_TENSORBOARD", True),
        help="start TensorBoard on this run's log directory, on the first free "
        "port at or after --tb-port. **It is on by default**: --no-tensorboard "
        "turns it off for one run, MJRL_TENSORBOARD=off in .env turns it off for "
        "good, and --tensorboard turns it back on over that",
    )
    g.add_argument(
        "--tb-port", type=int, default=_int_env("MJRL_TB_PORT", 6006),
        help="preferred port. Taken ports are skipped, so a second run gets the "
        "next one rather than failing",
    )
    g.add_argument(
        "--tb-scope", choices=["run", "task", "all"], default="task",
        help="how much to serve: this run only, every run of this task and model "
        "(the default, so the previous run is there to compare against), or the "
        "whole logs/ tree",
    )


@contextlib.contextmanager
def maybe_tensorboard(log_dir: Path, args):
    """Serve `log_dir` for the duration of the run, unless switched off.

    Failure is never fatal, for the same reason it is not for the live viewer:
    the server is there to look at, it is not part of the run. A run that dies
    because a port was taken would be a far worse trade than not seeing curves.
    """
    if not getattr(args, "tensorboard", False):
        yield None
        return
    from mjrl.viewer.tensorboard import sigterm_runs_finally, start_tensorboard

    try:
        tb = start_tensorboard(log_dir, scope=args.tb_scope, port=args.tb_port)
    except Exception as e:  # noqa: BLE001
        print(f"[mjrl] tensorboard did not start, continuing without it: "
              f"{type(e).__name__}: {e}")
        yield None
        return

    print(f"[mjrl] tensorboard {tb.url} serving {tb.logdir}"
          + (f" (port {args.tb_port} was taken)" if tb.port != args.tb_port else ""))
    tb.open_browser()
    try:
        with sigterm_runs_finally():
            yield tb
    finally:
        tb.stop()
