#!/usr/bin/env python3
"""Export a trained policy from a **live** mjlab environment.

Produces actor.onnx + layout.json + README.md. One principle governs everything
here:

    **Every value is read from the live env / cfg / checkpoint. Nothing is copied
    by hand.**

Hand-copied values drift. A real incident this guards against: training changed a
command range from +/-1.0 m/s to +/-0.8, the hardcoded copies in the replay and
evaluation scripts did not follow, and their comments still said "training
ranges" -- so the policy was fed commands from outside its training
distribution.

Outputs:

    <out>/actor.onnx      [1, obs_dim] -> [1, act_dim]; feed observations, get the
                          mean action
    <out>/layout.json     deployment contract: joint order, action transform, PD
                          gains, control rate, observation layout
    <out>/README.md       how to assemble observations and apply actions
    <out>/model_<N>.pt    the checkpoint this was built from, copied in -- see the
                          note beside the copy for why a bundle carries its own
                          weights. Named so that `--checkpoint <out>` finds it

Usage:

    python scripts/export.py --task jumper.tetrapod \
        --checkpoint logs/jumper/jumper.tetrapod/<date-time>/model_1999.pt

## Why the ONNX exported during training is not enough

`VelocityOnPolicyRunner.save()` exports an ONNX with metadata every time it saves
a checkpoint. That metadata has 12 fields but **lacks the action-to-joint
mapping**, and on this robot `joint_names` has 22 entries while the action is 20
wide (the two gripper joints are not policy-controlled). A deployment that pairs
them by index is misaligned from the fifth joint onwards -- with no error, and a
robot that moves wrongly.

This script adds `action_joint_order` and ten hard validations (see
`_validate_*`). Four of them are about the far side rather than this one: that
every observation term is a term the on-robot controller can build, that none of
them is a quantity the robot cannot measure, that the home pose covers every
joint on the wire, and that each per-joint block is exactly as wide as
`obs_joint_order`. Those failures are invisible here and expensive there.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from _cli import (
    build_parser,
    parse_with_task_args,
    print_task_table,
    resolve_all,
)



# ──────────────────────────────────────────────────────────────────────
# The deployment vocabulary
# ──────────────────────────────────────────────────────────────────────
#
# The contract declares `_schema: isaac_layout v1`, and what defines that schema
# is the consumer: the on-robot controller builds each observation term **by
# name**, and a name it does not know is a hard failure at load. mjlab's names
# are not that vocabulary -- it calls the velocity command `command` and the
# actuator torque `actuator_force` -- so the export translates, and refuses to
# write a contract whose terms the deployment could not build.
#
# The table lives here rather than in the task configs because the names come
# from vendored mjlab (`rl/mjlab/tasks/velocity/velocity_env_cfg.py`); a task
# that renamed them would drift from upstream at every rebase, and the rename
# belongs to the export, not to training.

#: mjlab's term name -> what the deployment calls the same quantity.
_TERM_RENAME = {
    "command": "velocity_commands",
    "actuator_force": "joint_torque",
}

#: Observation terms no robot can measure, whatever the controller can build.
#:
#: `base_lin_vel` is the whole set today. An IMU gives angular velocity straight
#: from the gyro, so `base_ang_vel` is fine; linear velocity comes only from a
#: state estimator integrating that gyro against foot contacts, and this robot
#: has none. In simulation the term is free and exact, which is what makes it
#: dangerous -- the policy leans on it, trains well, replays well, and has no
#: counterpart on hardware.
#:
#: These belong in the critic, which is thrown away at export. The jumper skeleton
#: puts them there (`tasks/jumper/common/velocity_env.py`); this is the check that
#: keeps it that way.
_UNMEASURABLE_TERMS = {
    "base_lin_vel": "no state estimator on this robot; the IMU gives angular "
                    "velocity only",
}

#: Every term name the deployment's observation builder can construct, including
#: the three aliases it maps itself. A term outside this set makes the contract
#: undeployable, which `_validate_deployable` turns into a failed export rather
#: than a runtime throw on the robot.
#:
#: **A copy of `deploy/fsm/src/obs.rs`'s `SUPPORTED`**, plus the aliases
#: `layout.rs` rewrites onto `commands`. Copied because this runs on a training
#: machine with no Rust toolchain; `tests/test_layout.py` reads that file and
#: requires the two to agree, because they had drifted and the drift was the
#: permissive direction -- this list claimed ten terms the controller cannot
#: build, five of them `jumper.dance`'s. A gate that waves a task through is worse
#: than no gate: the export looks fine and the robot refuses it on the bench,
#: which is the exact failure `_validate_deployable` exists to prevent.
_DEPLOY_TERMS = frozenset({
    # obs.rs SUPPORTED
    "base_lin_vel", "base_ang_vel", "projected_gravity", "base_pose",
    "commands", "commands_diff",
    "joint_pos", "joint_vel", "joint_torque",
    "actions", "gait_phase",
    # Buildable, but only from a recording: see `_REFERENCE_TERMS`.
    "jump_phase", "ref_future",
    "clip_phase", "ref_joint_pos", "ref_joint_vel", "ref_tilt_error",
    # layout.rs rewrites these three onto `commands` before the builder sees them
    "velocity_commands", "jump_command", "jump_go",
    # `jumper.posture`'s second command. It was listed here once as a promise,
    # before the crate could build it, and removed when `tests/test_layout.py`
    # held this set to exactly what `obs.rs` builds -- a promise in the gate is
    # a bundle that exports cleanly and throws on the board. It is back because
    # the builder is: `obs.rs` builds it from the operator's command and the
    # term's own `neutral_height`.
    "posture_command",
})

#: The terms above that the controller can build **only from a recording**.
#:
#: Being in `_DEPLOY_TERMS` says `obs.rs` has a case for the name. It does not
#: say the case can run: both of these read the contract's `reference` block and
#: the trajectory beside it, and `ObservationBuilder::new` refuses them when
#: that is missing.
#:
#: The distinction is not academic, and a test found it rather than a reader.
#: `jumper.dance` has a term named `ref_future` too, and it is **not the same
#: term**: its preview is indexed in frames of the clip where `jumper.jump`'s is
#: in seconds off a 250 Hz table. Admitting the name unconditionally waved
#: dance's version through a gate that exists precisely to stop that, and the
#: robot would have read frames as seconds -- a preview 50x too far ahead,
#: which is a policy shown a motion it is not about to make.
#:
#: So the name is necessary and the block is what makes it sufficient.
_REFERENCE_TERMS = frozenset({
    "jump_phase", "ref_future",
    "clip_phase", "ref_joint_pos", "ref_joint_vel", "ref_tilt_error",
})


# ──────────────────────────────────────────────────────────────────────
# Building the contract
# ──────────────────────────────────────────────────────────────────────


def _uniform_or_map(values, names):
    """Return a scalar when every value is equal, otherwise a {name: value} map.

    The map form means the deployment side never has to trust array order.
    """
    vals = [float(v) for v in values]
    if len(set(round(v, 12) for v in vals)) == 1:
        return vals[0]
    return {n: v for n, v in zip(names, vals)}


def _actuator_scalars(env, action_joint_order):
    """Read each joint's kp / kd / torque limit in **action order**.

    From the live actuator objects, which is where the numbers are. This used to
    read `mj_model.actuator_gainprm` / `actuator_biasprm`, on the reasoning that
    the compiled model is what the simulation actually uses -- true of a MuJoCo
    position actuator, where those fields *are* the gains, and **false of this
    robot**. `ServoCurveActuatorCfg` compiles to a plain torque actuator and runs
    its PD in Python, so the model carries MuJoCo's defaults for the fields the
    old code read:

        gaintype=FIXED  gainprm[0] = 1.0   ->  exported as kp = 1.0
        biastype=NONE   biasprm[2] = 0.0   ->  exported as kd = -0.0

    against the 10.0 and 0.5 the simulation is really running. It exported, the
    contract was well-formed, and the numbers were placeholders -- caught only
    because the deployment side read them and said so. `effort_limit` was right
    either way, since `forcerange` does carry the plateau.

    Worse than a wrong number: `rl-wbc-fsm`'s `contract.cpp` reads these with
    `ctrl.value("kp", 10.0f)`, whose fallback happens to be the correct value. A
    *missing* field would have deployed correctly; a present and wrong one runs a
    tenth of the intended stiffness with no damping at all.

    Still the live object and not cfg -- cfg may hold regexes and defaults, and
    domain randomisation writes gains onto the actuator, not back into cfg. Env 0
    is the reference, for the same reason the rest of this file uses it.
    """
    robot = env.scene["robot"]
    found: dict[str, tuple[float, float, float]] = {}
    for act in robot.actuators:
        # `param_names` says which of these an actuator carries; every PD-shaped
        # one has the three below, and an actuator without them cannot describe a
        # deployable joint anyway.
        for i, joint in enumerate(act.target_names):
            found[joint] = (
                float(act.stiffness[0, i]),
                float(act.damping[0, i]),
                float(act.force_limit[0, i]),
            )

    missing = [j for j in action_joint_order if j not in found]
    if missing:
        raise SystemExit(f"[export] no actuator found for these controlled joints: {missing}")

    kp, kd, eff = (
        np.array([found[j][k] for j in action_joint_order]) for k in range(3)
    )
    # The placeholders the old code produced, named rather than described: a
    # torque actuator read as a position one gives exactly this pair, and it is
    # the one failure this function has actually had.
    if np.allclose(kp, 1.0) and np.allclose(kd, 0.0):
        raise SystemExit(
            "[export] kp is 1.0 and kd is 0.0 for every joint, which is what "
            "MuJoCo's defaults look like rather than a gain. The actuators are "
            "not reporting their PD parameters -- refusing to write a contract "
            "the robot would run at a tenth of the intended stiffness."
        )
    return (
        _uniform_or_map(kp, action_joint_order),
        _uniform_or_map(kd, action_joint_order),
        _uniform_or_map(eff, action_joint_order),
    )


#: The trajectory file a reference-guided policy ships beside its ONNX.
#:
#: `trajectory`, not `reference`: a bundle already has a `reference.json` and it
#: is a different thing -- the recorded frames three hosts replay to check they
#: agree. Two meanings for one word is what `bundle` was doing a week ago.
TRAJECTORY_SUFFIX = ".trajectory.json"


def _reference(env_cfg, out: Path, wire_joint_order: list[str]) -> dict:
    """The recorded trajectory a residual policy corrects, and how to read it.

    A velocity policy needs nothing like this: its command is three numbers and
    every observation is measurable now. A reference-guided one is not runnable
    without the recording it was trained against -- `jumper.jump`'s action *is* a
    residual on it -- so the recording is part of the contract and travels with
    the policy.

    JSON rather than the npz, following `rl-wbc-fsm`, which put the reason in
    its converter: parsing an npz on the target means a zip reader, raw deflate
    and a `.npy` parser for one array. This one is read by the same serde the
    contract already uses, and it diffs.

    **Both tables ship.** `q` is the state the recording reached and feeds the
    `ref_future` observation; `q_cmd` is the position command the recorded
    controller actually sent, and is the residual's baseline. They are not
    interchangeable and they are not even the same length -- 368 frames at 250
    Hz against 295 at 200 Hz here. `mdp/actions.py` measured the difference at
    0.3 rad over the push-off and says why using the wrong one makes the jump
    weak; `rl-wbc-fsm` ships only `q` and compensates with a lead, which is a
    different answer to the same problem and not the one this policy trained
    against.
    """
    cmd = getattr(env_cfg, "reference_contract", None)
    if callable(cmd):
        cmd = cmd()
    if not isinstance(cmd, dict):
        return {}

    ref = cmd["reference"]
    src = list(cmd["joint_order"])
    missing = [j for j in wire_joint_order if j not in src]
    if missing:
        raise SystemExit(
            f"[export] the reference trajectory is missing joints {missing}.\n"
            f"       Its own order is {src}\n"
            f"       A permuted trajectory is a robot tracking the wrong joints, "
            f"and nothing downstream can see it."
        )
    perm = [src.index(j) for j in wire_joint_order]

    #: Decimal places the tables are written with.
    #:
    #: 1e-5 rad is 0.0006 degrees, two orders below anything this robot's servos
    #: resolve, and it is the difference between a 4.5 MB file and a 10.7 MB one
    #: for `jumper.dance`'s 11783 frames. `repr(float32)` is what json writes
    #: otherwise, and most of those digits are the binary representation
    #: showing through rather than anything measured.
    places = 5

    def rows(table, columns=None) -> list[list[float]]:
        out = []
        for row in table:
            picked = row if columns is None else [row[i] for i in columns]
            out.append([round(float(v), places) for v in picked])
        return out

    tables = {"joint_order": wire_joint_order, "q": rows(ref["q"], perm)}
    for name in ("q_cmd", "qd"):
        if name in ref:
            tables[name] = rows(ref[name], perm)
    if "root_quat" in ref:
        # Not per-joint: one orientation per frame, `(w, x, y, z)`.
        tables["root_quat"] = rows(ref["root_quat"])

    path = out / (cmd["name"] + TRAJECTORY_SUFFIX)
    path.write_text(
        json.dumps({
            "_source": f"scripts/export.py, from {cmd['npz']}",
            "_note": "Every per-joint table is in **wire order**, already permuted. "
                     "`q` is the state reached; `q_cmd`, where present, is what the "
                     "recorded controller commanded and is the residual baseline -- "
                     "a different rate and a different length. `qd` and `root_quat` "
                     "are per-frame companions of `q`.",
            **tables,
        }, separators=(",", ":")) + "\n",
        "utf-8",
    )

    carried = ["residual_action", "rec_hz", "duration", "lookahead_s"]
    optional = ["starts_on_entry", "control_hz", "t_go", "go_frame", "span",
                "baseline", "baseline_lead_s", "go_event", "n_cmd"]
    block = {k: cmd[k] for k in carried}
    block.update({k: cmd[k] for k in optional if cmd.get(k) is not None})
    block["file"] = path.name
    block["n_frames"] = len(tables["q"])
    if "q_cmd" in tables:
        block["n_cmd"] = len(tables["q_cmd"])

    shapes = "  ".join(f"{k} {len(v)}x{len(v[0])}"
                       for k, v in tables.items() if k != "joint_order")
    kind = "residual on " + str(block.get("baseline")) if block["residual_action"] else "played"
    print(f"[export] reference  {cmd['name']}: {shapes}  ({kind}, "
          f"{block['duration']:.2f} s at {block['rec_hz']:.0f} Hz, "
          f"{path.stat().st_size / 1e6:.2f} MB)")
    return {"reference": block}


def _controller(env_cfg):
    """How an operator's input becomes the command, when the task describes it.

    `command_ranges` says how far an axis may be pushed; it does not say which
    way is positive, nor what a stick at full deflection means. Both were being
    decided a second time by every consumer, from the comments in the task's
    teleop modules -- and a consumer that reads a sign wrong drives the robot
    sideways while everything else looks correct.

    Read off the env config rather than assembled here: this file is generic and
    the bindings are the robot's. A task that does not describe its controller
    simply exports no `controller` block, which is honest -- better than a
    default that a reader would take for a measurement.
    """
    contract = getattr(env_cfg, "controller_contract", None)
    if callable(contract):
        contract = contract()
    return {"controller": contract} if isinstance(contract, dict) else {}


def _command_ranges(env, env_cfg):
    """Command ranges are part of the contract too.

    A deployment whose joystick spans more than the training range is pushing the
    policy out of distribution; one that spans less quietly hides capability.
    Both have happened, so this reads from the live cfg.
    """
    out = {}
    for name in env.command_manager.active_terms:
        cfg = env_cfg.commands.get(name) if hasattr(env_cfg.commands, "get") else None
        if cfg is None:
            cfg = getattr(env_cfg.commands, name, None)
        rng = getattr(cfg, "ranges", None)
        if rng is None:
            continue
        fields = {}
        for f, v in vars(rng).items():
            if f.startswith("_"):
                continue
            if isinstance(v, (tuple, list)) and len(v) == 2:
                fields[f] = [float(v[0]), float(v[1])]
            # A half-range is a range too: `jumper.posture`'s angles are one
            # number each, symmetric by construction. Skipped, a stick on that
            # axis has nothing to scale to, and the controller refuses the
            # contract -- which is how it was found.
            elif isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                fields[f] = [-float(v), float(v)]
        if fields:
            out[name] = fields
    return out


def _json_params(params) -> dict:
    """A term's params, restricted to what JSON can carry.

    Params hold live objects -- a resolved `SceneEntityCfg`, a sensor handle --
    beside the scalars that describe the term's behaviour. Only the scalars mean
    anything to a reader on the other side, and the rest are not serialisable.
    """
    return {
        k: v
        for k, v in (params or {}).items()
        if isinstance(v, (bool, int, float, str))
    }


def _obs_terms(env):
    om = env.observation_manager
    names = list(om.active_terms["actor"])
    dims = [int(d[0]) for d in om.group_obs_term_dim["actor"]]
    terms, off = [], 0
    for name, dim in zip(names, dims):
        cfg = om.get_term_cfg("actor", name)
        deploy_name = _TERM_RENAME.get(name, name)
        entry = {"name": deploy_name, "dim": dim, "offset": off}
        if deploy_name != name:
            # Kept so the contract can still be diffed against the env it came
            # from; the deployment ignores it.
            entry["source_name"] = name
        params = _json_params(getattr(cfg, "params", None))
        # The deployment reads a gait clock's period **in seconds** from
        # `params.period`; mjlab's term carries the frequency instead. Derived
        # here rather than left to the far side, where the missing key does not
        # raise -- it defaults to a 1.0 s period, which for this robot's 3.125 Hz
        # clock is 3.1x wrong: a policy stepping to the wrong tempo, with nothing
        # anywhere saying so.
        if deploy_name == "gait_phase" and params.get("freq_hz"):
            params["period"] = 1.0 / float(params["freq_hz"])
        if params:
            entry["params"] = params
        scale = getattr(cfg, "scale", None)
        if scale is not None:
            entry["scale"] = (
                [float(s) for s in scale]
                if isinstance(scale, (list, tuple))
                else float(scale.cpu().tolist())
                if isinstance(scale, torch.Tensor)
                else float(scale)
            )
        clip = getattr(cfg, "clip", None)
        if clip is not None:
            entry["clip"] = [float(c) for c in clip]
        # **Two ways a term can carry history, and only one of them is the
        # manager's.** `history_length` on the config is mjlab's own stacking. A
        # term may instead stack its own frames and report the count as
        # `history_frames` on the callable -- `hexa.posture`'s `StridedHistory`
        # does, because at 200 Hz five *consecutive* frames span 25 ms and the
        # five were chosen at 50 Hz to span 100 ms.
        #
        # Reading only the config would write `history_length` absent for a
        # 100-wide `joint_pos`, and the deployment sizes that block from
        # `obs_joint_order`: it would read one frame of 100 against a 20-joint
        # order and take every later term at the wrong offset. `_validate_shapes`
        # catches exactly that, which is how this was found.
        hist = int(getattr(cfg, "history_length", 0) or 0)
        stride = 1
        own = int(getattr(getattr(cfg, "func", None), "history_frames", 0) or 0)
        if own:
            if hist:
                raise SystemExit(
                    f"[export] term {name} stacks {own} frames itself and the "
                    f"manager stacks {hist} more; the contract cannot describe "
                    f"the result"
                )
            hist = own
            stride = int(getattr(cfg.func, "history_stride", 1) or 1)
        if hist > 0:
            entry["history_length"] = hist
            entry["flatten_history_dim"] = bool(getattr(cfg, "flatten_history_dim", True))
            entry["base_dim"] = dim // hist
            # **Present only when it is not 1**, so a builder that has never seen
            # it cannot quietly default: an unstrided contract looks exactly as it
            # always did, and a strided one carries a field an old reader will
            # fail on rather than ignore.
            if stride != 1:
                entry["history_stride"] = stride
        terms.append(entry)
        off += dim
    return terms, off


def _observed_joint_order(env, robot):
    """The joints the observation's per-joint blocks cover, in their real order.

    **Not the robot's full joint list.** The deployment derives every per-joint
    term's width from `len(obs_joint_order)`, so a list longer than the
    observation really carries shifts every term after `joint_pos` and reads the
    whole vector at the wrong offsets.

    On this robot the two differ: the grippers are in `robot.joint_names` but an
    `asset_cfg` keeps them out of `joint_pos` / `joint_vel`, so the full list
    claims a 22-wide block against a real one of 20. That was this field's value
    until now, and the bundle README had to carry a note saying joint quantities
    are ordered by `action_joint_order` instead -- a documented workaround for a
    field that was simply wrong.

    Read from the resolved `asset_cfg` on the live term, so it follows whatever
    the task restricted the observation to.
    """
    om = env.observation_manager
    active = list(om.active_terms["actor"])
    orders = {}
    for name in ("joint_pos", "joint_vel"):
        if name not in active:
            continue
        asset_cfg = (getattr(om.get_term_cfg("actor", name), "params", None) or {}).get(
            "asset_cfg"
        )
        ids = getattr(asset_cfg, "joint_ids", None)
        # An unrestricted term leaves joint_ids as slice(None): every joint.
        orders[name] = (
            list(robot.joint_names)
            if ids is None or isinstance(ids, slice)
            else [robot.joint_names[i] for i in ids]
        )
    if not orders:
        return list(robot.joint_names)
    distinct = {tuple(v) for v in orders.values()}
    if len(distinct) > 1:
        raise SystemExit(
            "[export] joint_pos and joint_vel observe different joint sets "
            f"({ {k: len(v) for k, v in orders.items()} }); the contract has one "
            f"obs_joint_order and cannot describe both."
        )
    return list(next(iter(distinct)))


def _checkpoint_ref(ckpt: Path) -> dict:
    """Where a checkpoint is, said in a way another machine can use.

    Repository-relative when it is inside one, because an absolute path is a
    fact about the machine that exported rather than about the policy. `run` is
    the training directory -- `logs/<model>/<task>/<date-time>` -- which is the
    part a person recognises; `iteration` is the file.
    """
    from tasks.paths import REPO_ROOT

    ckpt = ckpt.resolve()
    where = ckpt.relative_to(REPO_ROOT) if ckpt.is_relative_to(REPO_ROOT) else ckpt
    return {"path": str(where), "run": str(where.parent), "iteration": ckpt.stem}


def build_contract(env, env_cfg, agent_cfg, checkpoint: Path | None = None) -> dict:
    robot = env.scene["robot"]

    act_terms = list(env.action_manager.active_terms)
    if len(act_terms) != 1:
        raise SystemExit(f"[export] expected exactly 1 action term, got {act_terms}")
    term = env.action_manager.get_term(act_terms[0])

    # Every joint the robot has, in the order the entity holds them. This keys
    # the home pose, which the deployment requires to cover every joint on the
    # wire -- including the ones the policy never sees.
    wire_joint_order = list(robot.joint_names)
    # The joints the observation carries, which is a subset here. See
    # `_observed_joint_order`.
    obs_joint_order = _observed_joint_order(env, robot)
    # target_names holds the **joint names** returned by
    # find_joints_by_actuator_names, in the order of the action vector. This is
    # the single most important entry in the contract.
    action_joint_order = list(term.target_names)

    default_jp = robot.data.default_joint_pos[0].detach().cpu().tolist()
    default_joint_pos = {n: float(v) for n, v in zip(wire_joint_order, default_jp)}

    # The hard stops, per joint, in the same radians as the home pose.
    #
    # Exported because the *deployment* is the one host that cannot measure
    # them. Both simulators read this straight off the model they loaded -- mjlab
    # from this same tensor, the browser from the MJCF -- and clamp every
    # commanded target to it. The robot has no model, so the board fell back to
    # one hand-written pair in its own config, and the pair it held was
    # `[-3.30, 3.30]`: the placeholder out of this crate's unit tests. Twenty of
    # this robot's twenty-two joints have both stops inside that, so the last
    # clamp between a policy and the motors never fired on any of them.
    lim = robot.data.joint_pos_limits[0].detach().cpu().tolist()
    joint_limits = {n: [float(lo), float(hi)] for n, (lo, hi) in zip(wire_joint_order, lim)}

    # Joints the policy does not control: they must be locked at these positions
    # on hardware, or the contact geometry will not match training
    unactuated = {
        n: default_joint_pos[n] for n in wire_joint_order if n not in action_joint_order
    }

    scale = term.scale
    action_scale = (
        _uniform_or_map(scale[0].cpu().tolist(), action_joint_order)
        if isinstance(scale, torch.Tensor)
        else float(scale)
    )

    raw_clip = None
    if getattr(term, "_clip", None) is not None:
        lo = term._clip[0, :, 0].cpu().tolist()
        hi = term._clip[0, :, 1].cpu().tolist()
        raw_clip = {n: [float(a), float(b)] for n, a, b in zip(action_joint_order, lo, hi)}

    kp, kd, eff = _actuator_scalars(env, action_joint_order)
    obs_terms, obs_dim = _obs_terms(env)

    init_pos = env_cfg.scene.entities["robot"].init_state.pos
    actor_cfg = agent_cfg.actor

    return {
        "_source": "scripts/export.py -- re-export whenever the env / robot / agent config changes",
        # Which checkpoint this came from, so the directory's name does not have
        # to carry it -- a name cannot hold the run as well as the iteration, and
        # nothing can check a name. `deploy/manifests.json` finds an export by
        # matching this rather than by rebuilding a path.
        **({"_checkpoint": _checkpoint_ref(checkpoint)} if checkpoint else {}),
        "_schema": "isaac_layout v1",
        # The joints the observation carries, in order. Every per-joint term
        # (joint_pos / joint_vel / joint_torque) is this wide and in this order.
        "obs_joint_order": obs_joint_order,
        # Every joint the robot has. Not read by the deployment -- it is here so
        # that `default_joint_pos` and `unactuated_joints` can be checked against
        # the robot rather than against the observation.
        "wire_joint_order": wire_joint_order,
        "action_joint_order": action_joint_order,
        # The policy does not control these joints; lock them at the given positions
        "unactuated_joints": unactuated,
        "action_scale": action_scale,
        **({"raw_action_clip": raw_clip} if raw_clip is not None else {}),
        "use_default_offset": True,
        "default_joint_pos": default_joint_pos,
        # [lo, hi] per joint, keyed like default_joint_pos and covering every
        # joint on the wire. A consumer that clamps targets should clamp to this
        # rather than to a number in its own config -- see the note above.
        "joint_limits": joint_limits,
        "base_height": float(init_pos[2]),
        "base_init_pos": [float(v) for v in init_pos],
        "control": {
            "kp": kp,
            "kd": kd,
            "effort_limit": eff,
            "decimation": int(env_cfg.decimation),
            "sim_dt": float(env.physics_dt),
            "control_dt": float(env.step_dt),
            "control_hz": round(1.0 / float(env.step_dt), 6),
        },
        "observation": {
            "dim": obs_dim,
            "history_length": 1,
            "history_order": "oldest_first",
            # **Spelled out here as well as in the README**, because a contract
            # that carries a number a reader does not know to look for is how a
            # robot ends up with a correct-shaped observation built from the
            # wrong instants. Present only when some term is strided, so an
            # unstrided bundle is byte-identical to what it always was.
            **(
                {
                    "history_stride_note": (
                        "One or more terms carry `history_stride` > 1. Those "
                        "terms' frames are NOT consecutive control steps: frame "
                        "i counts back i*stride steps from now, oldest first. "
                        "Build them by keeping a ring of stride*(length-1)+1 "
                        "frames and emitting every stride-th, newest included. "
                        "A builder that shifts one frame per inference -- which "
                        "is what rl-wbc-fsm's ObservationBuilder does today -- "
                        "produces the right SHAPE from a window stride times too "
                        "short, and nothing will report it."
                    )
                }
                if any(t.get("history_stride", 1) != 1 for t in obs_terms)
                else {}
            ),
            # The actor carries an EmpiricalNormalization and it is **baked into
            # the ONNX graph** (see rsl_rl `_OnnxMLPModel.forward`: obs_normalizer
            # first, then the mlp). Feed raw observations on the deployment side
            # and **do not normalise again**.
            "normalization": "baked_into_onnx",
            "terms": obs_terms,
        },
        "action": {"dim": int(env.action_manager.total_action_dim)},
        "command_ranges": _command_ranges(env, env_cfg),
        **_controller(env_cfg),
        "network": {
            "actor_hidden_dims": [int(h) for h in actor_cfg.hidden_dims],
            "activation": str(actor_cfg.activation),
            "obs_normalization": bool(actor_cfg.obs_normalization),
        },
        "deploy_notes": _deploy_notes(obs_terms),
    }


def _deploy_notes(obs_terms) -> list[str]:
    """Known sim2real gaps, shipped with the contract so they are not discovered
    during deployment."""
    notes = [
        "Action transform: target = action * action_scale + default_joint_pos[joint], "
        "paired joint by joint in action_joint_order.",
        "In simulation, apply_actions also subtracts encoder_bias (a domain "
        "randomisation term modelling encoder offset). Do not reproduce it on hardware.",
        "Joints in unactuated_joints are not policy-controlled and must be locked at "
        "the given positions: they take part in collision, and freeing them changes "
        "the foot contact geometry.",
        "The ONNX already contains observation normalisation; feed raw observations.",
    ]
    # What `posture_command` is, for a consumer that is not this repository's
    # controller. `deploy/fsm` builds it from the term's own params and needs
    # none of this; the note is for whoever builds it a second time.
    if any(t["name"] == "posture_command" for t in obs_terms):
        notes += [
            "posture_command is 4 floats ordered [twist, pitch, roll, height - "
            "params.neutral_height]: the command term's own order, the height "
            "centred on standing. Not the builder's `base_pose`, which is three of "
            "the same quantities ordered [pitch, roll, twist].",
        ]
    # There used to be a `base_lin_vel` warning here. It is gone because a note is
    # the wrong instrument: the bundle exported anyway, and the note was read by
    # whoever already knew. `_validate_measurable` now refuses the export instead.
    return notes


# ──────────────────────────────────────────────────────────────────────
# Validation: better to fail the export than to ship a wrong one
# ──────────────────────────────────────────────────────────────────────


def _validate_shapes(contract):
    n_act = contract["action"]["dim"]
    n_joint = len(contract["action_joint_order"])
    if n_act != n_joint:
        raise SystemExit(f"[export] action dim {n_act} != controlled joint count {n_joint}")
    terms_sum = sum(t["dim"] for t in contract["observation"]["terms"])
    if terms_sum != contract["observation"]["dim"]:
        raise SystemExit(
            f"[export] observation terms sum to {terms_sum} != observation dim "
            f"{contract['observation']['dim']}"
        )
    n_wire = len(contract["wire_joint_order"])
    n_un = len(contract["unactuated_joints"])
    if n_wire != n_joint + n_un:
        raise SystemExit(
            f"[export] joint counts disagree: robot has {n_wire} != controlled "
            f"{n_joint} + unactuated {n_un}"
        )
    # The deployment refuses to start unless the home pose covers every joint on
    # the wire, including the ones the policy never sees -- they still have to be
    # held somewhere.
    missing = [j for j in contract["wire_joint_order"] if j not in contract["default_joint_pos"]]
    if missing:
        raise SystemExit(f"[export] default_joint_pos does not cover these joints: {missing}")
    # Same rule for the stops, and for the same reason: a joint the clamp does
    # not cover is a joint with no clamp, which is indistinguishable from a
    # working one until it reaches its stop.
    missing = [j for j in contract["wire_joint_order"] if j not in contract["joint_limits"]]
    if missing:
        raise SystemExit(f"[export] joint_limits does not cover these joints: {missing}")
    inverted = {j: v for j, v in contract["joint_limits"].items() if not v[0] < v[1]}
    if inverted:
        raise SystemExit(f"[export] joint_limits are not [lo, hi]: {inverted}")
    outside = {
        j: (contract["default_joint_pos"][j], contract["joint_limits"][j])
        for j in contract["wire_joint_order"]
        if not (contract["joint_limits"][j][0] - 1e-6
                <= contract["default_joint_pos"][j]
                <= contract["joint_limits"][j][1] + 1e-6)
    }
    if outside:
        # The home pose is where every mode switch ramps to. Outside the clamp it
        # is a pose the robot is asked to reach and then prevented from holding.
        raise SystemExit(f"[export] the home pose is outside the joint limits: {outside}")
    stray = [j for j in contract["obs_joint_order"] if j not in contract["default_joint_pos"]]
    if stray:
        raise SystemExit(f"[export] observed joints absent from the robot: {stray}")

    # Every per-joint term must be exactly `obs_joint_order` wide per frame. This
    # is the check that would have caught obs_joint_order carrying all 22 joints
    # while joint_pos was 20 wide: the deployment sizes those blocks from the
    # list's length, so the two disagreeing shifts every later term's offset.
    n_obs_j = len(contract["obs_joint_order"])
    for t in contract["observation"]["terms"]:
        if t["name"] not in ("joint_pos", "joint_vel", "joint_torque"):
            continue
        per_frame = t["dim"] // int(t.get("history_length", 1) or 1)
        if per_frame != n_obs_j:
            raise SystemExit(
                f"[export] term {t['name']} is {per_frame} wide per frame but "
                f"obs_joint_order has {n_obs_j} joints; the deployment sizes that "
                f"block from obs_joint_order and would read every later term at "
                f"the wrong offset"
            )
    print(f"[export] dimensions consistent  obs={contract['observation']['dim']} act={n_act} "
          f"(robot {n_wire} joints = controlled {n_joint} + locked {n_un}; "
          f"observation carries {n_obs_j})")


def _validate_deployable(contract):
    """Every observation term must be one the deployment can build.

    The on-robot observation builder constructs terms **by name** and throws on a
    name it does not know, so an unknown term is a robot that will not start --
    discovered on the bench, after the export looked fine. A new observation term
    in a task is the way this happens: it costs nothing in training and nothing in
    simulation, and only the deployment cares.

    Failing here is the point. Either the term has a deployment name and belongs
    in `_TERM_RENAME`, or the policy genuinely cannot be deployed until the
    controller learns to build it.
    """
    allowed = set(_DEPLOY_TERMS)
    spec = contract.get("reference")
    if spec:
        # `ref_future` is one joint vector per declared offset. A width that
        # disagrees means the term and the block describe different previews,
        # and the controller would build the block's -- silently, at the right
        # total width only by coincidence.
        n_obs = len(contract["observation"].get("joint_order", contract["obs_joint_order"]))
        want = len(spec["lookahead_s"]) * n_obs
        for t in contract["observation"]["terms"]:
            if t["name"] == "ref_future" and t["dim"] != want:
                raise SystemExit(
                    f"[export] ref_future is {t['dim']} wide but the reference block "
                    f"declares {len(spec['lookahead_s'])} offsets over {n_obs} observed "
                    f"joints, which is {want}.\n"
                    f"       The deployment builds the block's version, so these have to "
                    f"agree or the robot previews something the policy never saw."
                )
    else:
        allowed -= _REFERENCE_TERMS

    # Terms the controller builds from parameters the contract has to carry.
    # Missing, each would be built from a default the policy never saw: a gait
    # clock at the host's 0.32 s while this policy's tempo follows its speed, or
    # a posture height 0.107 m out for ever. The controller refuses both too;
    # this is the same refusal before anything is packed.
    for t in contract["observation"]["terms"]:
        params = t.get("params") or {}
        if t["name"] == "gait_phase":
            law = {"stride", "freq_min", "freq_max", "turn_radius"}
            if "period" not in params and not law <= set(params):
                raise SystemExit(
                    f"[export] gait_phase carries neither `period` nor the cadence law "
                    f"({', '.join(sorted(law))}); its params are {sorted(params)}.\n"
                    f"       The controller would run it at a default tempo."
                )
        if t["name"] == "posture_command" and "neutral_height" not in params:
            raise SystemExit(
                "[export] posture_command carries no `neutral_height`, so the controller "
                "cannot centre its height channel the way training did."
            )

    unknown = [t["name"] for t in contract["observation"]["terms"] if t["name"] not in allowed]
    if unknown:
        raise SystemExit(
            f"[export] these observation terms have no deployment equivalent: {unknown}\n"
            f"       The on-robot observation builder constructs terms by name and "
            f"throws on an unknown one, so this contract would not load.\n"
            f"       Add a mapping to _TERM_RENAME in this file if the deployment "
            f"already builds the same quantity under another name; otherwise the "
            f"controller has to gain the term first."
        )
    named = [f"{t['name']}({t['dim']})" for t in contract["observation"]["terms"]]
    print(f"[export] deployable terms  {' '.join(named)}")


def _validate_measurable(contract):
    """The actor must not depend on anything the robot cannot measure.

    This one fails the export rather than warning, and the reason is that a
    warning did not work. The contract carried a `deploy_notes` line saying
    `base_lin_vel` needs a state estimator, and the bundle exported anyway --
    a note in a JSON file is read once, by whoever already knew.

    What makes the failure the right severity is that there is no fix on the
    deployment side. A missing term can be added to the controller; an
    unmeasurable one cannot be added to the robot. The policy has to be retrained
    without it, so the earlier that is known the cheaper it is -- and the export
    is the last point before someone starts building hardware around it.
    """
    bad = [t["name"] for t in contract["observation"]["terms"] if t["name"] in _UNMEASURABLE_TERMS]
    if bad:
        why = "\n".join(f"         {n}: {_UNMEASURABLE_TERMS[n]}" for n in bad)
        raise SystemExit(
            f"[export] the actor observes quantities the robot cannot measure: {bad}\n"
            f"{why}\n"
            f"       These belong in the critic, which is discarded at export. Move "
            f"them there and retrain -- there is nothing the deployment side can do "
            f"about it, and a policy that depends on them cannot run on hardware."
        )
    print(f"[export] every observed term is measurable on hardware ({len(contract['observation']['terms'])} terms)")


def _validate_home_pose(contract, env_cfg, tol=1e-4):
    """The exported home pose must match `init_state.joint_pos` in the cfg.

    They should share a source, but default_joint_pos can be altered by
    post-processing such as soft_joint_pos_limit. A mismatch means something
    changed the pose along the way and has to be understood first.
    """
    authored = env_cfg.scene.entities["robot"].init_state.joint_pos or {}
    bad, unspec = {}, []
    for j, dumped in contract["default_joint_pos"].items():
        if j in authored:
            a = float(authored[j])
            if abs(dumped - a) > tol:
                bad[j] = (dumped, a)
        else:
            unspec.append(j)
    if bad:
        lines = "\n  ".join(
            f"{j}: env={d:.5f} cfg={a:.5f} (delta {d - a:+.2e})"
            for j, (d, a) in sorted(bad.items())
        )
        raise SystemExit(f"[export] home pose disagrees with init_state ({len(bad)} joints, "
                         f"tolerance {tol}):\n  {lines}")
    note = f"; {len(unspec)} not specified in the cfg" if unspec else ""
    print(f"[export] home pose  {len(contract['default_joint_pos']) - len(unspec)} "
          f"joints match init_state (tolerance {tol}){note}")


def _validate_checkpoint(contract, ckpt_path):
    """The checkpoint's dimensions must match the current task config.

    Pairing weights trained on a different task with this contract is the easiest
    mistake to make and the hardest to find.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = sd.get("actor_state_dict")
    if model is None:  # other rsl_rl versions name it differently
        model = sd.get("model_state_dict") or sd.get("model") or sd
        if isinstance(model, dict) and isinstance(model.get("actor"), dict):
            model = model["actor"]
    # mlp.<i>.weight, sorted by layer number -- lexicographic order would put
    # mlp.10 before mlp.2
    ws = sorted(
        ((k, tuple(v.shape)) for k, v in model.items()
         if k.startswith("mlp.") and k.endswith(".weight")
         and hasattr(v, "shape") and v.ndim == 2),
        key=lambda kv: int(kv[0].split(".")[1]),
    )
    if not ws:
        raise SystemExit(
            f"[export] cannot infer the network shape from the checkpoint "
            f"(top-level keys: {list(sd)[:6]}). This validation exists to stop "
            f"weights from another task being paired with this contract and must "
            f"not be skipped silently."
        )
    obs_dim = ws[0][1][1]
    act_dim = ws[-1][1][0]
    hidden = [shape[0] for _, shape in ws[:-1]]
    mism = []
    if obs_dim != contract["observation"]["dim"]:
        mism.append(f"obs_dim: checkpoint={obs_dim} contract={contract['observation']['dim']}")
    if act_dim != contract["action"]["dim"]:
        mism.append(f"act_dim: checkpoint={act_dim} contract={contract['action']['dim']}")
    if hidden != contract["network"]["actor_hidden_dims"]:
        mism.append(f"hidden: checkpoint={hidden} contract={contract['network']['actor_hidden_dims']}")
    if mism:
        raise SystemExit(
            "[export] checkpoint does not match the current task config:\n  "
            + "\n  ".join(mism)
            + "\nExport with the task this checkpoint was trained on."
        )
    print(f"[export] checkpoint dims  obs={obs_dim} act={act_dim} hidden={hidden}")


def _export_onnx(runner, out: Path) -> None:
    """Write the ONNX a robot runs: one input, one output.

    Some runners export something else. mjlab's motion-tracking runner bundles
    the whole reference clip into the graph as constant buffers and exports
    `(obs, time_step) -> (actions, joint_pos, joint_vel, body_pos_w, ...)`, so a
    host can read the reference rows back out instead of carrying the clip
    beside the model. That is a reasonable answer to the same problem this
    repository answers with a companion trajectory file, and it is the wrong one
    **here**, for reasons about the target rather than taste:

      * the board runs `.rknn`, and `deploy/fsm/src/rknn.rs` feeds one tensor
        and reads one -- a second input and six extra outputs are not a
        conversion detail, they are a different interface;
      * the buffers are the clip. `jumper.dance`'s is 11784 frames across six
        arrays, which is a large constant blob to push through the NPU converter
        for data the controller can read as text.

    **Decided by what the graph turns out to be, not by who wrote the method.**
    This first reached past *any* override to the base class, which was wrong
    for every task: `MjlabOnPolicyRunner` also overrides this, deliberately, to
    pass `dynamo=False` and stay off torch>=2.9's new export path -- so every
    ordinary export silently took rsl_rl's route instead of mjlab's. The only
    override this needs to refuse is one that changes the *interface*, and that
    is visible in the file it just wrote.
    """
    runner.export_policy_to_onnx(str(out), "actor.onnx")
    path = out / "actor.onnx"

    import onnx

    inputs = [i.name for i in onnx.load(str(path)).graph.input]
    if len(inputs) == 1:
        return

    base = next(
        (c.export_policy_to_onnx for c in type(runner).__mro__[1:]
         if "export_policy_to_onnx" in c.__dict__),
        None,
    )
    if base is None:
        raise SystemExit(
            f"[export] {type(runner).__name__} exported a graph taking {inputs}, and "
            f"the deployment feeds exactly one tensor.\n"
            f"       No base class defines another export to fall back to."
        )
    print(f"[export] {type(runner).__name__} exported a graph taking {inputs}; the "
          f"deployment feeds one tensor, so re-exporting with "
          f"{base.__qualname__.split('.')[0]}'s. The reference travels as a "
          f"trajectory file instead of inside the graph.")
    base(runner, str(out), "actor.onnx")


def _validate_onnx(onnx_path, contract, policy, device):
    """The ONNX must match the torch policy in both shape **and** value.

    Matching shapes do not imply matching values (whether the normaliser was baked
    in, whether the mean rather than a sample was taken), so the same observations
    are fed to both and compared element-wise. Without onnxruntime only shapes can
    be checked, and that is stated explicitly.
    """
    import onnx

    m = onnx.load(str(onnx_path))
    ishape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    oshape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    if ishape[-1] != contract["observation"]["dim"]:
        raise SystemExit(f"[export] ONNX input {ishape} does not match observation "
                         f"dim {contract['observation']['dim']}")
    if oshape[-1] != contract["action"]["dim"]:
        raise SystemExit(f"[export] ONNX output {oshape} does not match action "
                         f"dim {contract['action']['dim']}")
    print(f"[export] ONNX shape  {ishape} -> {oshape}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("[export] WARNING: onnxruntime is not installed, so the numerical "
              "check was SKIPPED. Shapes can match while values do not (for "
              "example when the normaliser was not baked into the graph). "
              "It is a dependency: pip install -e . and rerun to check")
        return

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        x = rng.normal(0, 1, size=(1, contract["observation"]["dim"])).astype(np.float32)
        y_onnx = sess.run(None, {iname: x})[0]
        with torch.inference_mode():
            td = _as_policy_input(policy, torch.from_numpy(x).to(device))
            y_torch = policy(td).cpu().numpy()
        worst = max(worst, float(np.abs(y_onnx - y_torch).max()))
    if worst > 1e-4:
        raise SystemExit(f"[export] ONNX and torch policy outputs disagree, "
                         f"max difference {worst:.3e}")
    print(f"[export] ONNX values  max difference from the torch policy {worst:.2e} "
          f"over 8 random observations")


def _as_policy_input(policy, x):
    """Wrap a flat observation tensor into the form the policy expects."""
    from tensordict import TensorDict

    groups = getattr(policy, "obs_groups", None) or ["actor"]
    return TensorDict({g: x for g in groups}, batch_size=[x.shape[0]], device=x.device)


# ──────────────────────────────────────────────────────────────────────
# README
# ──────────────────────────────────────────────────────────────────────


def write_readme(path: Path, contract: dict, task: str, ckpt: str) -> None:
    o = contract["observation"]
    lines = [
        f"# Deployment bundle: {task}",
        "",
        f"Source checkpoint: `{ckpt}`",
        "",
        "Generated from a live environment by `scripts/export.py`.",
        "**Do not edit by hand** -- re-export whenever the config changes; values",
        "edited by hand always drift.",
        "",
        "## Files",
        "",
        "| File | Contents |",
        "|---|---|",
        f"| `actor.onnx` | `[1, {o['dim']}] -> [1, {contract['action']['dim']}]`; "
        f"takes raw observations, returns the mean action |",
        "| `layout.json` | The deployment contract. Read every number below from "
        "it rather than copying them into code |",
        "",
        "## Assembling the observation",
        "",
        f"Length {o['dim']}, concatenated in the order below. Joint quantities are "
        f"always ordered by `action_joint_order` ({contract['action']['dim']} of "
        f"them), **not** by `obs_joint_order`.",
        "",
        "| Range | Term | Dim |",
        "|---|---|---|",
    ]
    for t in o["terms"]:
        lines.append(f"| `[{t['offset']}:{t['offset'] + t['dim']}]` | {t['name']} | {t['dim']} |")
    strided = [t for t in o["terms"] if t.get("history_stride", 1) != 1]
    if strided:
        lines += [
            "",
            "### The history is strided -- read this before building it",
            "",
            "These terms stack several frames, and **the frames are not "
            "consecutive control steps**:",
            "",
            "| Term | Frames | Stride | Spans |",
            "|---|---|---|---|",
        ]
        hz = contract["control"]["control_hz"]
        for t in strided:
            n, k = t["history_length"], t["history_stride"]
            lines.append(
                f"| {t['name']} | {n} | {k} | {(n - 1) * k / hz * 1000:.0f} ms "
                f"({(n - 1) * k} control steps) |"
            )
        lines += [
            "",
            "Frame `i` of a term is the value **`i * stride` control steps "
            "before now**, oldest first, newest last. Build it by keeping a ring "
            "of `stride * (frames - 1) + 1` frames and emitting every "
            "`stride`-th, with the newest always included.",
            "",
            "**Why it matters.** The frame count was chosen at 50 Hz to span "
            "100 ms. The policy now runs at "
            f"{hz:.0f} Hz, where the same count of *consecutive* frames spans "
            f"{(strided[0]['history_length'] - 1) / hz * 1000:.0f} ms. Striding "
            "restores the window without widening the network's input.",
            "",
            "**What goes wrong if this is ignored.** A builder that shifts one "
            "frame per inference emits a tensor of exactly the right length, "
            "the ONNX accepts it, the robot walks -- on a window "
            f"{strided[0]['history_stride']}x shorter than the one the policy "
            "was trained against. There is no error and no log line. "
            "`rl-wbc-fsm`'s `ObservationBuilder` shifts one frame per "
            "inference as of this writing and needs the change above.",
        ]
    lines += [
        "",
        f"Observation normalisation is already inside the ONNX "
        f"(`normalization = {o['normalization']}`), so **feed raw values and do not "
        f"normalise again**.",
        "",
        "## Applying the action",
        "",
        "```",
        "for i, joint in enumerate(contract['action_joint_order']):",
        "    target[joint] = action[i] * action_scale + default_joint_pos[joint]",
        "```",
        "",
        f"Then track with PD: kp = {contract['control']['kp']}, "
        f"kd = {contract['control']['kd']}, "
        f"torque limit {contract['control']['effort_limit']} N*m, "
        f"control rate {contract['control']['control_hz']} Hz.",
        "",
        "Joints in `unactuated_joints` are not policy-controlled and **must be locked "
        "at the given positions**: they take part in collision, and freeing them "
        "changes the foot contact geometry.",
        "",
        "## Read before deploying",
        "",
    ]
    for n in contract["deploy_notes"]:
        lines.append(f"- {n}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser("export.py", __doc__ or "")
    g = parser.add_argument_group("export")
    g.add_argument("--checkpoint", help="model_*.pt to export")
    g.add_argument(
        "--out", default=None,
        help="output directory; defaults to tasks/<task path>/out/<date-time>/",
    )
    # Only some tasks have extra artifacts at all (see tasks.load_export_media);
    # for the rest this switch does nothing either way. It exists because the ones
    # that do are slow -- jumper.dance renders its whole 235 s choreography -- and
    # export is also run just to re-check the contract, where waiting minutes for a
    # video nobody asked for is the wrong default.
    g.add_argument(
        "--video", action=argparse.BooleanOptionalAction, default=True,
        help="write the task's extra artifacts (rendered video, audio) when it has "
             "any; --no-video writes only the policy and its contract",
    )
    # The selected task may take arguments of its own; see
    # `tasks.load_cli_args`. Nothing here knows which, or how many.
    args = parse_with_task_args(parser)

    if args.list:
        print_task_table()
        return

    spec, res, asset = resolve_all(args)
    if args.dry_run:
        print(f"[mjrl] resolved: task={spec.id} -- --dry-run, stopping here")
        return
    if not args.checkpoint:
        raise SystemExit("[export] --checkpoint is required")

    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"[export] checkpoint does not exist: {ckpt}")

    from mjrl.backend.select import use_backend

    import tasks

    # The backend has to be registered before the environment is built; see the
    # note in scripts/train.py.
    use_backend(res)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    from tasks.paths import out_dir_for

    # `tasks/<task>/out/<date-time>/` -- beside the task it came from, and
    # tracked, unlike `logs/`. See `tasks.paths.out_dir_for`.
    #
    # Timestamped rather than named after the checkpoint. Two exports of one
    # checkpoint are two different things whenever anything between the config
    # and this script has changed, and the old name meant the second silently
    # replaced the first. The same `%Y-%m-%d_%H-%M-%S` as `logs/` and
    # `out/bundle_*`, so there is one way to read a directory name here.
    #
    # Which checkpoint it came from is recorded *inside*, in `layout.json`,
    # because a name is not a place to keep provenance: it cannot hold the run
    # as well as the iteration, and nothing can check it.
    out = Path(args.out) if args.out else (
        out_dir_for(spec.id) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    out.mkdir(parents=True, exist_ok=True)

    # One environment is enough: the contract describes the robot, not a batch.
    env_cfg = tasks.load_env_cfg(
        spec.id, asset=asset, play=True, task_args=args.task_args
    )
    env_cfg.scene.num_envs = 1
    agent_cfg = tasks.load_agent_cfg(spec.id)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = tasks.load_runner_cls(spec.id) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=res.device)
        runner.load(
            str(ckpt), load_cfg={"actor": True}, strict=True, map_location=res.device
        )
        policy = runner.get_inference_policy(device=res.device)

        print()
        contract = build_contract(env, env_cfg, agent_cfg, checkpoint=ckpt)
        # After the contract, because it needs the wire order the contract
        # resolved; before validation, because the block is part of what is
        # validated.
        contract.update(_reference(env_cfg, out, contract["wire_joint_order"]))

        _validate_shapes(contract)
        _validate_deployable(contract)
        _validate_measurable(contract)
        _validate_home_pose(contract, env_cfg)
        _validate_checkpoint(contract, ckpt)

        _export_onnx(runner, out)
        _validate_onnx(out / "actor.onnx", contract, policy, res.device)
    finally:
        env.close()

    # `layout.json`. The on-robot importer (rl-wbc-fsm `tools/import_deploy.py`)
    # takes this name first and falls back to `isaac_layout.json` for older
    # bundles, so both generations import unchanged. `_schema` still says
    # `isaac_layout v1` -- that names the format, which kk-rl-lab and the robot
    # controller also speak, and is not this repository's to rename.
    (out / "layout.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_readme(out / "README.md", contract, spec.id, str(ckpt))

    # **The checkpoint travels with the bundle.**
    #
    # A bundle is otherwise a one-way artifact: `layout.json` records which file
    # it came from, and a month later that path is a run somebody deleted or a
    # config that has moved underneath it. Both of the things anyone wants to do
    # with a bundle afterwards need the weights -- comparing it against the policy
    # it claims to be (`play.py --checkpoint <this directory>` replays the copy), and exporting
    # it again after the contract gains a field, which is not hypothetical: this
    # week a term was missing `neutral_height` and every bundle written before
    # then had to be redone.
    #
    # Re-exporting from the *repository* is the trap that makes this worth the
    # 10 MB. The observation width does not change when a gait constant does, so
    # an old checkpoint re-exported against today's config produces a contract
    # that loads, runs, and describes a cadence the policy never trained under.
    # Keeping the weights beside the contract at least makes "which policy is
    # this" answerable without git archaeology.
    #
    # Copied rather than linked: a bundle is a thing to send somewhere. Kept
    # under its own name rather than a fixed one, so `CHECKPOINT_GLOB` finds
    # it -- which makes the bundle directory something `--checkpoint` and
    # `train --resume` can be pointed at directly.
    shutil.copy2(ckpt, out / ckpt.name)

    print(f"\n[export] done -> {out}")
    for f in sorted(out.iterdir()):
        print(f"    {f.name:<16}{f.stat().st_size / 1024:8.1f} KB")

    # Extra artifacts, if this task has any. Deliberately after the bundle is
    # complete and outside the env's lifetime: this hook builds its own
    # environment (it needs a full-length episode, which the contract env is not),
    # and a failure here must not leave a half-written actor.onnx behind.
    #
    # This file knows only that the hook may exist -- never what it writes. The
    # rule that `scripts/` does not change when a task is added still holds; what
    # changed once, here, is that there is now a place for a task to put artifacts
    # that are not part of the deployment contract. See tasks.load_export_media.
    if args.video:
        extra = tasks.load_export_media(spec.id)
        if extra is not None:
            extra(out, task_id=spec.id, checkpoint=ckpt, asset=asset, res=res)


if __name__ == "__main__":
    main()
