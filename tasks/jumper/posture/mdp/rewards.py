"""What jumper.posture is paid for on top of walking: holding the commanded posture.

Three terms, all in the shape the four velocity tasks' tracking terms use --
`exp(-err^2 / std^2)`, larger is better, weight positive -- so that
`Episode_Reward/track_*` can be read side by side with them without a conversion.

## The std is what decides whether these are learnable

The formula makes `std` "how much error still counts as tracking", and the
calibration this repository uses for it is a ratio of the command's own range:
`STD_LIN_RATIO` is 0.5 and `STD_ANG_RATIO` is 0.4, chosen in
`common/mdp/curriculum.py` so that a robot which does nothing at all collects
about 0.14 -- the same free score a Go1 gets on mjlab's velocity task.

The same ratio is applied here, and the reason to reuse it rather than pick a
number is the one that note records: a std that does not move with its range is a
ruler that changes length, and a term calibrated against one range silently
becomes either free or unreachable when the range moves. The arithmetic at this
task's ranges, for a policy that holds the neutral posture whatever it is asked
for:

    term     range       std        free score at the far end / at half of it
    twist    +/-15 deg   0.131 rad  0.018 / 0.37
    tilt     +/-15 deg   0.131 rad  0.0003 / 0.14  (both axes at once)
    height   0.07-0.15   0.020 m    0.010 / 0.42   (0.033 at the bottom end;
                                                    neutral is 0.107, which is
                                                    not the middle of the range)

-- so ignoring the command is worth about a third of each term on a middling draw
and essentially nothing at the extremes, which is the incentive gap
`track_yaw_velocity`'s docstring argues is the number that actually matters.

Two asymmetries in that table are deliberate and worth reading rather than
smoothing out. `track_tilt` carries two errors in one exponent, so the corner of
its box is `sqrt(2)` further out than either axis alone and the free score there
is much lower than the twist's -- a robot asked for 15 degrees of pitch *and* 15
of roll and delivering neither collects nothing at all. And the height's neutral
sits 0.037 m above the bottom of its range and 0.043 m below the top, because the
robot stands at 0.107 m and the range was chosen around what the legs were
measured to hold rather than around where they rest.

## Why pitch and roll are one term and the twist is not

`track_tilt` sums both errors into one exponent, which makes a robot that gets
pitch right and roll wrong score like one that misses both by a bit. That is the
intended reading: pitch and roll are one lean, the command is a direction to lean
in, and half of it is not half a posture. The twist is a different actuation
entirely -- it is the legs yawing the body over planted feet rather than the body
tilting on them -- so it is scored on its own and can be learned on its own.

Height is separate for the same reason and one more: it is the only one of the
four in metres, and folding it into an exponent with radians would make the std a
number with no physical reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import variable_posture
from mjlab.utils.lab_api.string import resolve_matching_names_values

from ...tripod.mdp.rewards import TRIPOD_A, TRIPOD_B
from . import state as posture_state
from .cadence import ENV_ATTR

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _command(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    assert cmd is not None, f"command {command_name!r} not found"
    assert cmd.shape[1] == 4, (
        f"command {command_name!r} is {cmd.shape[1]} wide; these terms read "
        f"(twist, pitch, roll, height)"
    )
    return cmd


def track_twist(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """`exp(-(twist_cmd - twist)^2 / std^2)`, the body yawed over its feet.

    **The measurement ripples with the gait and the std has to absorb it.** Three
    of the six feet are airborne for half of every cycle and they are included in
    the footprint (see `state.py` for why), so the estimate carries a periodic
    component at the step frequency that no policy can remove -- it is what
    swinging a leg forward looks like to a footprint fit. A std tight enough to
    resolve a degree of twist would spend most of its range on that ripple.

    Nothing has measured the ripple's amplitude yet, which is the honest state of
    this term: the std below comes from the range-ratio calibration rather than
    from a measurement of how still this quantity can be held while walking. The
    number to look at once a policy exists is
    `Metrics/posture/error_twist` for a run commanded to zero twist throughout --
    that is the ripple plus the tracking error, and if it sits near the std then
    the term is measuring the gait rather than the posture.

    ## The footprint is the heading frame, walking and standing alike

    `body_twist` is the body against **its own footprint**, and it is tempting to
    call that the wrong frame -- a commanded twist means the body turning in the
    world, and this measures it against something that moves. It briefly had a
    second formula for standing, anchored to the world heading, for that reason.

    That was wrong, and the planner this robot's C++ stack descends from says why.
    `motion-planner` on `develop` carries the same command:

        heading_yaw_ += omega * dt;                  // travel yaw, integral of the
                                                     // commanded rate
        body_twist_yaw_ = rate_limited(target, ...)  // the twist, an absolute target
        world_yaw = heading_yaw_ + body_twist_yaw_;  // the body's attitude in the world

    and plans the feet in the heading frame with `omega` alone --
    `foot_planner.cpp:36` takes `body_.omega()`, which `body_planner.cpp:79` sets
    from `omega` and **not** from `omega + twist_rate`. So the feet do not know the
    twist exists and the body turns over them, by construction, in one code path
    that covers standing as the `omega = 0` case.

    **Their heading frame is the footprint**, made physical: it is the frame the
    feet are placed in. So `body_twist` is the measured form of their
    `body_twist_yaw_`, the definition is already the world one, and it is already
    the same walking and standing.

    ## What RL is missing that their stack has for free

    In that planner the feet can only move when `omega` moves them. Here a policy
    can move them for any reason, so the same measurement admits a solution their
    structure forbids: rotate the footprint under a still body. Measured on
    model_38000, commanded 10 degrees while parked, this term scored 10.05 against
    a base heading change of 0.00.

    The answer is not a second measurement of the twist -- it is the constraint
    their planner gets structurally, and **this task no longer has one**. The term
    that held each foot where it was while parked was removed on request, and
    `feet_slip` says only that a foot in contact does not slide while *walking*.
    So the fixed frame this term assumes it is scored against is an assumption:
    `mdp/commands.py::foot_displacement` is what says whether it held.
    """
    cmd = _command(env, command_name)
    err = cmd[:, 0] - posture_state.body_twist(env, asset_cfg)
    return torch.exp(-err.square() / std**2)


def track_tilt(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """`exp(-(pitch_err^2 + roll_err^2) / std^2)`, the commanded lean.

    **This is also this task's `upright`, and it replaces it.** mjlab's `upright`
    rewards the body for aligning with gravity, i.e. for pitch and roll being
    zero -- which is precisely what this task spends half its command range asking
    the robot not to do. Left in alongside this term, the two would disagree on
    every non-zero tilt command, and `upright` would win at small commands because
    it is flat near zero where this one is steepest.

    The command-relative version subsumes it rather than merely replacing it: at a
    zero tilt command the two are the same function up to their widths, so a task
    that never commanded a lean would behave as before.

    This is the same failure `jumper.swing` records for its own `upright` -- a term
    named for what you want, measured against a frame the robot is no longer being
    asked to hold. There it was the deck that tilted; here it is the command.
    """
    cmd = _command(env, command_name)
    err = cmd[:, 1:3] - posture_state.body_tilt(env, asset_cfg)
    return torch.exp(-err.square().sum(dim=1) / std**2)


def track_height(
    env: "ManagerBasedRlEnv",
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """`exp(-(height_cmd - height)^2 / std^2)`, the base above the ground.

    **The end of this range that binds is the squat, and it is not the end anyone
    expects.** The robot stands at `STAND_Z` = 0.107 m and the range asks for 0.07
    to 0.15 -- 37 mm down, 43 mm up. Measured standing (the table in
    `env_cfg.HEIGHT_RANGE`), holding 0.07 leaves the tightest joint **2.5 degrees**
    of travel and the robot still sags 5.3 mm short; holding 0.15 leaves 43 degrees
    and costs less torque than standing at HOME does, because a straighter leg
    carries the load closer to its own axes.

    So a bias towards the tall end is the expected shape, and is not by itself
    evidence of anything. What would be: a **signed** height error that sits
    positive at the bottom of the range -- a robot that cannot get down. Before
    reaching for the weight there, remember that weight buys penalty and not
    behaviour against a target the leg cannot reach, which is what
    `jumper.tripod`'s `foot_target_height` note records learning over 15697
    iterations of doubling one.

    That table is a **static stand**. Squatting to 0.07 while swinging three legs
    at 5 Hz is a different demand and nothing has measured it.
    """
    cmd = _command(env, command_name)
    err = cmd[:, 3] - posture_state.body_height(env, asset_cfg)
    return torch.exp(-err.square() / std**2)


def posture_reach_fraction(
    command: torch.Tensor, neutral_height: float, reach: torch.Tensor
) -> torch.Tensor:
    """How far a posture command is from neutral, as a share of its reach, [num_envs].

    The largest of the four axes, each divided by its own reach: `|twist|`,
    `|pitch|` and `|roll|` against the standing band, and `|height -
    neutral_height|` against the height's half-range. The largest rather than a
    norm, because a stance has to make room for its most demanding axis and a
    posture asking for one thing is not a smaller posture for asking for nothing
    else.

    **The height is taken from `neutral_height`, and that subtraction is the part
    that can be silently wrong.** The command carries the height absolute, so
    reading column 3 as if it were centred makes the neutral posture 2.7 reaches
    out -- every command then reads as a far one and nothing ever looks neutral.
    Checked on the built env when this was written: the neutral command reads 0.0
    with the subtraction and 2.66 without it.
    """
    offset = command.clone()
    offset[:, 3] = offset[:, 3] - neutral_height
    return (offset.abs() / reach).amax(dim=1)


class pose_widened_by_posture(variable_posture):
    """mjlab's `variable_posture`, whose **parked** width follows the posture command.

        std = std_walking / std_running            moving, exactly as before
        std = lerp(std_neutral, std_standing, s)   parked
        s   = min(1, posture_reach_fraction / full_width_at)

    Mounted under `pose`, so `Episode_Reward/pose` still reads beside the other
    velocity tasks'.

    ## Why the parked width cannot be one number here

    It was one number, the walking width, and for a good reason: holding a
    commanded posture *is* leaving HOME, and the skeleton's standing width reads
    15 degrees of twist as failure. But that width applied to the one parked
    posture where HOME is the right answer too, and measured on
    `2026-09-28_17-32-26/model_47000` it is what the policy learned there. Parked
    at the neutral posture, it walks away from HOME within 0.8 s and holds

        LM_J0 +0.375   RM_J0 -0.379   (both middle feet 33 mm further back)
        LF_J2 -0.243   RF_J2 +0.250   pitch 3.2 degrees nose up

    and returns to it after any walk or any posture. At the walking width that
    stance still scores 0.80, and the whole reward prefers HOME by only 0.39 a step
    (9.13 held at HOME by a zero action, 8.74 as the policy stands) -- a margin
    the policy was never shown, since `rel_neutral_standing_envs` was 0.0 and the
    parked-and-neutral case came only from the two commands' clocks slipping, about
    2% of training.

    The middle hip yaws are where it shows because nothing else can see them:
    swinging both middle legs back symmetrically moves two feet along the ground
    and changes neither the height, the tilt nor the footprint's twist, and no
    term charges a parked foot for moving. This term is the only one watching.

    ## The ramp

    `std_neutral` at the neutral posture, reaching `std_standing` at
    `full_width_at` of the reach. That point is where a posture's own joint
    displacement has grown to the size of the neutral width, which is where a
    tight width stops describing the stance and starts charging for the command.
    `env_cfg.py` has the number and the measurement it comes from.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        super().__init__(cfg, env)
        asset = env.scene[cfg.params["asset_cfg"].name]
        _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, _, std_neutral = resolve_matching_names_values(
            data=cfg.params["std_neutral"], list_of_strings=names
        )
        self.std_neutral = torch.tensor(std_neutral, device=env.device, dtype=torch.float32)
        self.reach = torch.tensor(
            cfg.params["posture_reach"], device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        std_standing,
        std_walking,
        std_running,
        std_neutral,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        posture_command_name: str,
        neutral_height: float,
        posture_reach,
        full_width_at: float,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running, std_neutral, posture_reach
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"command {command_name!r} not found"
        speed = command[:, :2].norm(dim=1) + command[:, 2].abs()
        standing = (speed < walking_threshold).float().unsqueeze(1)
        walking = ((speed >= walking_threshold) & (speed < running_threshold))
        walking = walking.float().unsqueeze(1)
        running = (speed >= running_threshold).float().unsqueeze(1)

        posture = _command(env, posture_command_name)
        s = posture_reach_fraction(posture, neutral_height, self.reach) / full_width_at
        s = s.clamp(0.0, 1.0).unsqueeze(1)
        std_parked = self.std_neutral + (self.std_standing - self.std_neutral) * s
        std = std_parked * standing + self.std_walking * walking + self.std_running * running

        asset = env.scene[asset_cfg.name]
        error = (
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - self.default_joint_pos[:, asset_cfg.joint_ids]
        )
        return torch.exp(-torch.mean(error.square() / std.square(), dim=1))


def _body_rotation(posture: torch.Tensor) -> torch.Tensor:
    """`Rz(twist) Ry(pitch) Rx(roll)`, [num_envs, 3, 3], from a posture command.

    The same axes and signs `state.body_tilt` and `state.body_twist` measure in:
    right-hand about the body's own x forward, y left, z up, so +pitch is nose
    down and +roll is left side up.
    """
    tw, pi, ro = posture[:, 0], posture[:, 1], posture[:, 2]
    cz, sz = tw.cos(), tw.sin()
    cy, sy = pi.cos(), pi.sin()
    cx, sx = ro.cos(), ro.sin()
    zero, one = torch.zeros_like(tw), torch.ones_like(tw)
    rz = torch.stack((cz, -sz, zero, sz, cz, zero, zero, zero, one), -1).view(-1, 3, 3)
    ry = torch.stack((cy, zero, sy, zero, one, zero, -sy, zero, cy), -1).view(-1, 3, 3)
    rx = torch.stack((one, zero, zero, zero, cx, -sx, zero, sx, cx), -1).view(-1, 3, 3)
    return rz @ ry @ rx


class hip_yaw_center:
    """Keep the middle legs' hip yaw **centred** where the commanded posture puts
    it, walking and parked alike. `exp(-mean((centre - target)^2) / std^2)`.

        centre = EMA of the joint angle, time constant `tau`
        target = the hip yaw that puts the foot on its HOME footprint with the
                 body at the commanded posture -- HOME itself at neutral

    ## Why a term of its own

    Measured over one training run, `2026-09-27_18-06-25`: at `model_28950` the
    middle hip yaws stood at HOME (+/-0.03) and swung about +/-0.08 of it at 0.3
    m/s; by `model_47000` they stood at +/-0.37 and swung about +/-0.18, with the
    configuration unchanged. `2026-09-24_19-35-43` sat at the top of the same
    curriculum for 70000 iterations without moving (+/-0.02 standing, +/-0.05 to
    0.09 walking). So it is a drift, and along the one direction nothing else
    watches: swinging both middle legs back symmetrically moves two feet along the
    ground and changes no height, tilt or twist, and `pose`'s walking width charges
    0.035 a step for the 0.17 it reached.

    `pose` itself cannot be the lever, because it charges the *instantaneous*
    angle: measured on the resumed run at 0.3-0.8 m/s, the swing (0.145 rad RMS)
    and the drift (0.15-0.17) are the same size, so a width tight enough for the
    drift charges walking itself 0.3 a step. The mean over a gait cycle keeps the
    drift and drops the swing, so this term can be tight and walking costs nothing.

    ## The centre

    An exponential average with `tau` at one cycle of the slowest cadence (2 Hz,
    0.5 s): a swing of amplitude A leaves `A / sqrt(1 + (2 pi f tau)^2)` of ripple,
    at most A/6.4 and A/17 at the top cadence. Parked, it is the joint angle a
    `tau` late. Reset to HOME with the episode.

    ## The target

    A twist is the body yawed over planted feet, and the middle hip yaws have to
    follow it by about 1.2 times the angle -- measured on `model_76400`, parked at
    +/-0.3 rad of twist they moved by -/+0.37. A target fixed at HOME would argue
    with `track_twist` exactly as `pose`'s standing width did. So the target is
    the angle the leg needs with its foot left on the HOME footprint and the body
    at the command: rotated by `Rz(twist) Ry(pitch) Rx(roll)` about the
    footprint's centre -- the point `state.body_twist` measures the twist about --
    and raised by the height offset. For a yaw-pitch-pitch leg that angle is
    closed-form: the foot's direction from the hip, in the plane normal to the
    hip axis. The two pitch joints set the reach and not the direction.

    The geometry is taken from the environment's own model at HOME, so a new
    robot revision moves it with the XML.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        import mujoco
        import numpy as np

        from ...common.constants import STAND_Z

        legs = tuple(cfg.params["legs"])
        robot = env.scene[cfg.params["asset_cfg"].name]
        joint_ids, _ = robot.find_joints(
            tuple(f"{leg}_J0_joint" for leg in legs), preserve_order=True
        )
        foot_ids, _ = robot.find_sites(legs, preserve_order=True)
        all_ids, _ = robot.find_sites(tuple(cfg.params["footprint"]), preserve_order=True)

        model = env.sim.mj_model
        data = mujoco.MjData(model)
        ix = robot.indexing
        free = ix.free_joint_q_adr.cpu().numpy()
        data.qpos[free[:3]] = (0.0, 0.0, STAND_Z)
        data.qpos[free[3:7]] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[ix.joint_q_adr.cpu().numpy()] = (
            robot.data.default_joint_pos[0].cpu().numpy()
        )
        mujoco.mj_forward(model, data)
        base = ix.root_body_id
        p0, r0 = data.xpos[base], data.xmat[base].reshape(3, 3)

        def local(x: np.ndarray) -> np.ndarray:
            return r0.T @ (x - p0)

        jid = ix.joint_ids.cpu().numpy()[joint_ids]
        sid = ix.site_ids.cpu().numpy()
        as_t = lambda a: torch.tensor(np.asarray(a), device=env.device, dtype=torch.float32)  # noqa: E731
        self.anchor = as_t([local(data.xanchor[j]) for j in jid])            # [L, 3]
        self.axis = as_t([r0.T @ data.xaxis[j] for j in jid])                 # [L, 3]
        self.foot = as_t([local(data.site_xpos[sid[i]]) for i in foot_ids])  # [L, 3]
        centre = np.mean([local(data.site_xpos[sid[i]]) for i in all_ids], axis=0)
        self.pivot = as_t([centre[0], centre[1], 0.0])                        # [3]
        self.home = robot.data.default_joint_pos[0, joint_ids].clone()        # [L]
        self.joint_ids = joint_ids
        self.alpha = env.step_dt / float(cfg.params["tau"])
        self.centre = self.home.expand(env.num_envs, -1).clone()             # [B, L]

    def target(self, posture: torch.Tensor, neutral_height: float) -> torch.Tensor:
        """The hip yaw the command asks for, [num_envs, L]. HOME at neutral."""
        rot = _body_rotation(posture)                                          # [B, 3, 3]
        lift = torch.zeros(posture.shape[0], 3, device=posture.device)
        lift[:, 2] = posture[:, 3] - neutral_height
        # Foot in the commanded body frame: p + R^T (f - p - lift).
        rel = self.foot.unsqueeze(0) - self.pivot - lift.unsqueeze(1)          # [B, L, 3]
        foot_b = self.pivot + torch.einsum("bji,blj->bli", rot, rel)
        u = self.axis.unsqueeze(0)

        def flat(v: torch.Tensor) -> torch.Tensor:
            return v - (v * u).sum(-1, keepdim=True) * u

        now = flat(foot_b - self.anchor)
        home = flat((self.foot - self.anchor).unsqueeze(0).expand_as(now))
        turn = torch.atan2((u * torch.cross(home, now, dim=-1)).sum(-1), (home * now).sum(-1))
        return self.home + turn

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.centre[env_ids] = self.home

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        asset_cfg: SceneEntityCfg,
        legs,
        footprint,
        posture_command_name: str,
        neutral_height: float,
        std: float,
        tau: float,
    ) -> torch.Tensor:
        del legs, footprint, tau  # resolved in __init__
        q = env.scene[asset_cfg.name].data.joint_pos[:, self.joint_ids]
        self.centre += self.alpha * (q - self.centre)
        error = self.centre - self.target(_command(env, posture_command_name), neutral_height)
        return torch.exp(-error.square().mean(dim=1) / std**2)


__all__ = [
    "hip_yaw_center",
    "pose_widened_by_posture",
    "posture_reach_fraction",
    "track_height",
    "track_tilt",
    "track_twist",
]


def foot_clearance_shortfall(
    env: "ManagerBasedRlEnv",
    target_height: float,
    height_sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    swing_gate: float | None = None,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """`feet_clearance`, charging only the feet that are **too low**.

        shortfall = max(0, target_height - foot_height)
        cost      = sum over feet of shortfall * |horizontal foot speed|

    mjlab's version uses `abs(foot_height - target_height)`, so a foot carried
    higher than the target is charged exactly as much as one dragged the same
    distance below it. That symmetry is the right default for a term whose job is
    to hold a trajectory *at* a height. It is the wrong one here, for a reason
    this task has measured rather than assumed.

    ## Why one-sided, on this task

    `target_height` is 15 mm and `Metrics/peak_height_mean` is 16.7. The robot
    already clears the bar, so under the symmetric form every millimetre of extra
    lift is a cost -- and the other terms that touch the swing all pull the same
    way: `soft_landing` (-0.06 a step), `joint_acc` (-0.075) and this term's own
    velocity factor, which gets *cheaper* the less the foot is moved. After the
    target came down from 25 mm nothing pushed the lift up at all.

    Clearance is a floor, not a set point. What it exists to prevent is a foot
    catching the ground while it translates; a foot carried higher than it needs
    to be is wasteful, and the waste is already priced -- by `joint_acc`, by
    `energy`, and by the swing time a taller arc costs at 9 control steps of
    swing. Charging it twice, in a term named for clearance, is what made the
    pressure one-directional in the wrong direction.

    ## What this does not change

    `foot_swing_height` is still symmetric and still charges overshoot, once per
    swing at the peak. So "do not lift higher than necessary" is still expressed
    -- by the term that measures the peak, which is where it belongs -- and what
    is no longer expressed is "do not *move* higher than necessary", which was
    charging the same behaviour a second time and per step.

    The velocity weighting and the command gate are mjlab's and are kept: a foot
    standing still is not translating and is not charged whatever its height, and
    the whole term is off while the command asks the robot to stand.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg as _Cfg

    asset = env.scene[(asset_cfg or _Cfg("robot")).name]
    sensor = env.scene[height_sensor_name]
    foot_height = sensor.data.heights
    # `slice(None)` and not `None`: indexing with `None` inserts an axis rather
    # than selecting every site, which broadcasts into a shape that is wrong by a
    # dimension. Every call from `env_cfg` passes an `asset_cfg`, so this branch
    # is the one nothing exercises -- it was wrong until a test with no
    # `asset_cfg` reached it.
    sites = slice(None) if asset_cfg is None else asset_cfg.site_ids
    foot_vel_xy = asset.data.site_lin_vel_w[:, sites, :2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)

    shortfall = (target_height - foot_height).clamp_min(0.0)
    if swing_gate is not None:
        shortfall = shortfall * _swing_gate(env, swing_gate)
    cost = torch.sum(shortfall * vel_norm, dim=1)

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total = command[:, :2].norm(dim=1) + command[:, 2].abs()
            cost = cost * (total > command_threshold).float()
    return cost


def _swing_gate(env: "ManagerBasedRlEnv", gate: float) -> torch.Tensor:
    """1 for a foot in the first `gate` of its own swing, 0 after, [num_envs, 6].

    The tripod clock runs one phase in [0, 1) for the whole robot: group A swings
    over [0, 0.5) and group B over [0.5, 1), so a foot's position *within its own
    swing* is the phase folded into that half and doubled. A foot whose group is
    in stance is outside the gate entirely.

    **The gate is on the clock and not on the foot's own motion**, and that is the
    whole design. The obvious spelling -- exempt a foot that is descending -- pays
    for the exemption in exactly the currency being overcharged: descend faster,
    qualify sooner, pay less. The clock is exogenous, so there is nothing to game.
    """
    clock = getattr(env, ENV_ATTR, None)
    if clock is None:
        raise RuntimeError(
            "foot_clearance_shortfall was given a swing_gate but the environment "
            "has no gait clock to read it from; the gate cannot fall back on the "
            "foot's own velocity without rewarding a faster descent"
        )
    phase = clock.phase(env)  # [B], in [0, 1)
    in_a = phase < 0.5
    # How far through its own swing each group is, or 1.0 (past the gate) when
    # that group is in stance.
    frac_a = torch.where(in_a, phase * 2.0, torch.ones_like(phase))
    frac_b = torch.where(in_a, torch.ones_like(phase), (phase - 0.5) * 2.0)
    gates = torch.ones(phase.shape[0], 6, device=phase.device, dtype=phase.dtype)
    gates[:, list(TRIPOD_A)] = (frac_a < gate).to(phase.dtype).unsqueeze(1)
    gates[:, list(TRIPOD_B)] = (frac_b < gate).to(phase.dtype).unsqueeze(1)
    return gates


class soft_touchdown:
    """Charge the speed a foot was **falling at** when it landed, [num_envs].

        cost = sum over feet of max(0, -vz one step before touchdown)

    Metres per second, larger is worse, the weight carries the minus sign. Gated
    off while the command asks the robot to stand.

    ## Why velocity and not the force this replaced

    `soft_landing` charges `|force|` at first contact, and it was retuned six
    times on this task without settling. The reason it would not settle is that it
    was not the term holding the other end of the rope. Measured on
    `2026-09-18_08-23-12/model_47350` at the cadence ceiling, 4758 swings aligned
    on touchdown:

        steps before touchdown   T-4    T-3    T-2    T-1     T
        foot height (mm)         22.0   21.4   18.4   12.9   6.6
        vz (mm/s)                +113    -61   -299   -544  -633

    The last three steps average **19.0 m/s^2 downward, 1.94 g**. The foot is not
    falling, it is being **driven** into the ground at nearly twice what gravity
    would do -- and against that, a free fall from the 22 mm peak would still
    arrive at 0.55 m/s, only 13% softer. So the impact is not a consequence of a
    short descent. It is a consequence of nothing asking the leg to brake, and the
    leg demonstrably has 2 g of authority to brake with.

    Force is the wrong handle on that. It is an output of the contact solver,
    resolved inside one control step, so its gradient with respect to the action
    is short and noisy. Approach velocity is a continuous function of the joint
    trajectory the policy is actually choosing. **Charge the cause, measure the
    effect**: `Episode_Metrics/landing_force_max` still reports the force, and is
    now a diagnostic rather than a lever.

    ## The one-step delay is not an approximation

    **Read at first contact, `vz` is already post-impact.** Contact happens inside
    the five physics substeps of a control step, so by the time the sensor reports
    it the foot has been stopped by it -- the term would charge close to zero for
    every landing however hard, and would look alive while measuring nothing.

    So the class keeps one step of history and charges the *previous* step's
    velocity when the contact is first seen: -544 mm/s in the table above, 10 ms
    and about 6 mm before touchdown. `tests/test_posture.py` pins the difference
    against the instantaneous reading, which is the form this was almost written
    in.

    ## The weight

    **-4.5, and the honest part of that number is how it was not chosen.**

    The first version was sized against a replay -- 23 landings averaging 6.3 N --
    and came out 24% *lighter* than the `soft_landing` it replaced while the note
    claimed it doubled it. The training log says what that term was really
    charging, at a command distribution a replay does not reproduce:

        soft_landing      -0.1190   at -0.02, so 5.95 N a step of raw quantity
        joint_acc         -0.1527
        foot_swing_height -0.1207
        foot_clearance    -0.0961
        action_rate_l2    -1.9216   the largest negative by a factor of twelve

    So the second version was sized against the budget instead: put this at the
    share the landing ought to have. **That reasoning has since been measured and
    is wrong.** `action_rate_l2` was halved to free room for exactly this term,
    and the policy expanded its action changes to fill the slack -- the raw
    quantity rose 86%, the cost fell 7%, the share barely moved and nothing
    reached here. `env_cfg.py`'s note on that term has the table. A term's
    realised cost is an equilibrium the policy chooses, not a number its weight
    sets, so **the settled value of this term cannot be predicted from its weight
    and is not predicted here.**

    What can be said is the direction. Unlike an action-rate penalty, approach
    speed has no upside: the policy gains nothing from landing harder except the
    braking it does not have to do. So raising the weight should move the
    equilibrium down rather than be absorbed.

    3x rather than 2x because at -1.5 this sat at -0.118 a step and about 1% of
    the budget for 8000 iterations without moving at all, and a factor of two has
    a real chance of being invisible again. Not more than 3x because
    `Metrics/peak_height_mean` is 0.0162 against a 0.025 target, already 35% low,
    and the cheapest answer to any landing penalty is to stop landing.

    **What would falsify this.** Not the term's own curve, which can fall simply
    because the weight rose. The raw quantity -- `Episode_Reward/soft_touchdown`
    divided by 4.5 -- has to fall below the 0.079 a step it holds now. If it does
    not, the approach speed is not what this term is reaching, and the next move
    is a different measurement rather than a larger number.

    The budget it has to stay small against is the tracking, and it does:
    `track_angular_velocity` alone pays 1.83 and the positive terms total 7.08.

    The failure to watch for is the one every landing penalty has: a robot that
    stops landing hard by not putting its feet down. `Metrics/peak_height_mean`
    and the standing foot count are what say so, not this term's own curve.
    """

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
        sensor = env.scene[cfg.params["sensor_name"]]
        num_feet = sensor.data.found.shape[1]
        self.prev_vz = torch.zeros(env.num_envs, num_feet, device=env.device)

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        sensor_name: str,
        command_name: str,
        command_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        from mjlab.managers.scene_entity_config import SceneEntityCfg as _Cfg

        from ...common.mdp.rewards import moving_gate

        asset = env.scene[(asset_cfg or _Cfg("robot")).name]
        sensor = env.scene[sensor_name]
        sites = slice(None) if asset_cfg is None else asset_cfg.site_ids

        first_contact = sensor.compute_first_contact(dt=env.step_dt).float()
        approach = self.prev_vz.clamp(max=0.0).neg()
        cost = torch.sum(approach * first_contact, dim=1)

        self.prev_vz = asset.data.site_lin_vel_w[:, sites, 2].clone()
        return cost * moving_gate(env, command_name, command_threshold)

    def reset(self, env_ids=None) -> None:
        """Drop the banked velocity, which belongs to an episode that is over.

        **Required for this term to be reset at all.** `RewardManager` collects
        its class terms with `hasattr(func, "reset")`, so a term without the
        method is silently absent from the list rather than reset with an empty
        body -- and this one carries a step of state.

        Left alone, the first landing of a new episode is charged the speed a
        foot was falling at on the last step of the *previous* one. A reset
        teleports the base and redraws the joints, so the banked number has no
        relationship to the landing it is charged against; it can be the terminal
        velocity of a fall.
        """
        if env_ids is None or isinstance(env_ids, slice):
            self.prev_vz.zero_()
        else:
            self.prev_vz[env_ids] = 0.0
