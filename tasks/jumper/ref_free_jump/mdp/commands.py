"""The go command, with the bookkeeping a height objective needs.

`jumper.jump`'s `JumpMotionCommand` supplies the go instant -- a delay for an
episode spawned at HOME, a negative go step for one spawned partway along the
recording -- and is used as it is. What this adds is **whose flight it is**.

A height reward that pays for any time in the air pays reference-state
initialisation for its own work: an episode spawned mid-flight rises to the
recording's apex on the recording's push and collects the jump without having
made one. kk-rl-lab's reference-free jump found exactly that, with the apex
reward farmed by its airborne spawns. So a flight counts here only if the
episode pushed it:

- it leaves the ground after the go, and
- the episode did not spawn past the bottom of the recording's crouch -- the
  lowest point of the countermovement, after which the push, and the momentum
  RSI hands the spawn, belong to the recording.

One jump per episode, and a jump is a flight at whose peak the robot's lowest
point has risen `min_rise` above HOME. Its touchdown ends it.

**A second jump forfeits the first.** Once the jump has landed, the first
moment the robot is in the air again with its lowest point `min_rise` up -- the
jump's own gate -- the reward that follows takes back everything `apex_height`
has paid this episode, and it pays nothing more. Paying nothing for the second
flight was not enough: `model_3200` of `2026-09-28_08-28-59`, from 64 HOME
spawns (native:cpu, DR and observation noise on, no pushes, its mean), jumped
again in 44, 10 to 15 cm, and still collected 9.4 of the first jump's apex after
the second began -- nearly all of it. Five of the 44 had spent the jump on a 1
to 5 cm hop first.

**Height is the robot's lowest point, against HOME**: the least, over the six
foot sites (`LEGS`) and the base_link origin, of each one's height less its own
at HOME. Neither alone is the jump. The base alone pays for a base pushed up on
legs that hang below it; the feet alone pay for feet flicked up over a base that
never rose -- all six go up 34 to 49 mm on one joint each, the base where it
was. Taking the lowest of the seven, the robot is as high as the part it keeps
lowest. At the recording's apex that is the base: fed forward, the jump lifts it
0.115 to 0.123 m, the lowest foot 0.167 to 0.176 m, legs tucked (64 HOME spawns,
no DR, native:cpu). Against HOME because the foot sites sit 8.7 to 9.0 mm above
the soles; the seven HOME heights come from the compiled model, so a new
revision of the robot brings its own.

**A flight that does not lift the robot is not the jump.** Counting the first
flight after the go, whatever it was, lost almost every jump: six feet all under
1 N for a step or two happens without anyone jumping. With the training config
(DR on) and the feedforward alone, 28 of 64 HOME spawns "flew" 0 to 5 ms after
the go -- the targets step by 0.2 rad there -- with the lowest point "rising"
-3.9 to -1.8 mm; with exploration noise of 0.3 there were 155 such hops in 64
episodes, the feet under 3 mm, the lowest point at most 1.3 mm. Such a hop now
leaves the episode's jump where it was. So does a push that gets the feet off
the ground and the base no higher than HOME -- 10 in the same run, feet 22 to
47 mm up, the base -4.5 to 9.2 mm: by this measure it did not go up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from ...common.constants import GAIT_JOINTS, LEGS
from ...jump.mdp.commands import JumpMotionCommand, JumpMotionCommandCfg
from ...jump.mdp.reference import JumpReference

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def push_start_phase(ref: JumpReference) -> float:
    """The phase of the bottom of the recording's crouch.

    Read off the recording rather than its generator's parameters: the lowest
    base height between the go and the first frame with every foot in the air.
    """
    airborne = (ref.contact.sum(dim=1) == 0).nonzero().flatten()
    takeoff = int(airborne[airborne >= ref.go_frame][0])
    z = ref.base_pos[ref.go_frame : takeoff + 1, 2]
    bottom = ref.go_frame + int(torch.argmin(z))
    return (bottom * ref.dt - ref.t_go) / ref.span


@dataclass(kw_only=True)
class RefFreeJumpCommandCfg(JumpMotionCommandCfg):
    contact_sensor_name: str = "feet_ground_contact"
    """The feet's contact sensor. A flight is every foot at or under
    `force_threshold`."""

    min_rise: float = 0.01
    """Metres the robot's lowest point must rise above HOME, at a flight's peak,
    for the flight to be the episode's jump. Contact chatter reaches 1.3 mm, and
    a push that clears the feet but not the base 9.2; see the module docstring."""

    home_tolerance: float
    """Back at HOME: the 20 driven joints' RMS distance from it, in radians, under
    this. `env_cfg.py` sets it."""

    home_speed_sq: float
    """And their mean squared speed, in (rad/s)^2, under this."""

    def __post_init__(self):
        self.class_type = RefFreeJumpCommand

    def build(self, env: ManagerBasedRlEnv) -> RefFreeJumpCommand:
        return RefFreeJumpCommand(self, env)


class RefFreeJumpCommand(JumpMotionCommand):
    cfg: RefFreeJumpCommandCfg

    def __init__(self, cfg: RefFreeJumpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        #: May this episode still make a jump of its own.
        self.can_jump = torch.ones(n, dtype=torch.bool, device=self.device)
        self.in_flight = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: Its own flight is over and it is down again.
        self.landed = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: Peak of `lowest_rise` during its own flight, latched.
        self.apex_rise = torch.zeros(n, device=self.device)
        #: It flew again after its jump had landed: the jump pays nothing more.
        self.second_jump = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: Consecutive steps it has been back at HOME since its jump was done.
        self.home_steps = torch.zeros(n, device=self.device)
        self._gait_ids, _ = self.robot.find_joints(list(GAIT_JOINTS), preserve_order=True)
        #: Steps `apex_height` has paid this episode, and whether the reward about
        #: to be computed is the one that takes them back -- see `_update_command`.
        self.paid_steps = torch.zeros(n, device=self.device)
        self.clawback_pending = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: The six foot sites, in `LEGS` order; then the height at HOME of each of
        #: them and of the base, in that order.
        self._foot_sites, _ = self.robot.find_sites(list(LEGS), preserve_order=True)
        self._home_z = self._heights_at_home()
        self._push_start = push_start_phase(self.reference)
        self._own_rise = torch.zeros(n, device=self.device)
        self._jumped = torch.zeros(n, device=self.device)
        self._second = torch.zeros(n, device=self.device)
        self.metrics["own_rise"] = self._own_rise
        self.metrics["jumped"] = self._jumped
        self.metrics["second_jump"] = self._second

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Read the spawn phase before the parent consumes it -- it zeroes the
        # record once it has turned it into a go step.
        rsi = getattr(self._env, "_jump_rsi_phase", None)
        phase = (
            rsi[env_ids].clone()
            if rsi is not None
            else torch.zeros(len(env_ids), device=self.device)
        )
        super()._resample_command(env_ids)
        self.can_jump[env_ids] = phase <= self._push_start
        self.in_flight[env_ids] = False
        self.landed[env_ids] = False
        self.apex_rise[env_ids] = 0.0
        self.second_jump[env_ids] = False
        self.home_steps[env_ids] = 0.0
        self.paid_steps[env_ids] = 0.0
        self.clawback_pending[env_ids] = False

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self._own_rise[:] = self.apex_rise.clamp(min=0.0) * self.landed
        self._jumped[:] = self.landed.float()
        self._second[:] = self.second_jump.float()
        self.metrics["own_rise"] = self._own_rise
        self.metrics["jumped"] = self._jumped
        self.metrics["second_jump"] = self._second

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        if env_ids is not None:
            # Reset path: the spawn state was just written and nothing has flown.
            return
        # The step's rewards were computed before this update, on the state the
        # last one left: count the apex payment they made, and the clawback, if
        # it was due, is paid now.
        self.paid_steps += (self.landed & ~self.second_jump).float()
        self.clawback_pending.zero_()

        forces = self._env.scene.sensors[self.cfg.contact_sensor_name].data.force
        assert forces is not None
        airborne = (forces.norm(dim=-1) <= self.cfg.force_threshold).all(dim=1)
        rise = self.lowest_rise()

        takeoff = (
            self.can_jump & ~self.in_flight & ~self.landed
            & (self.time_since_go >= 0.0) & airborne
        )
        self.in_flight |= takeoff
        # In place, like every update here. Rebinding the attribute to a fresh
        # tensor makes it an inference tensor whenever the step runs under
        # `torch.inference_mode()`, as rollouts do, and the next in-place write
        # outside that mode -- a reset from play or an evaluation script -- then
        # raises.
        self.apex_rise.copy_(
            torch.where(self.in_flight, torch.maximum(self.apex_rise, rise), self.apex_rise)
        )
        touchdown = self.in_flight & ~airborne
        jumped = touchdown & (self.apex_rise > self.cfg.min_rise)
        self.in_flight &= ~touchdown
        self.landed |= jumped
        self.can_jump &= ~jumped
        # A hop that never lifted the feet gives the jump back.
        self.apex_rise.masked_fill_(touchdown & ~jumped, 0.0)

        # Up again after the jump, by the jump's own gate: the next reward takes
        # back everything the apex has paid, and none follows.
        again = self.landed & ~self.second_jump & airborne & (rise > self.cfg.min_rise)
        self.second_jump |= again
        self.clawback_pending |= again

        q = self.robot.data.joint_pos[:, self._gait_ids]
        home = self.robot.data.default_joint_pos[:, self._gait_ids]
        qd = self.robot.data.joint_vel[:, self._gait_ids]
        back = (
            self.jump_done
            & (((q - home) ** 2).mean(dim=1).sqrt() < self.cfg.home_tolerance)
            & ((qd**2).mean(dim=1) < self.cfg.home_speed_sq)
        )
        self.home_steps.copy_(torch.where(back, self.home_steps + 1.0, torch.zeros_like(self.home_steps)))

    @property
    def jump_done(self) -> torch.Tensor:
        """The episode's jump is over: it has landed its own, or the recording's
        span has run out without one -- a spawn past the push, or a policy that
        did not jump."""
        return self.landed | (self.time_since_go > self.reference.span)

    def lowest_rise(self) -> torch.Tensor:
        """How far the robot's lowest point is above where it is at HOME, now: the
        least, over the six foot sites and the base_link origin, of each one's
        height less its own at HOME.

        `site_xpos` directly rather than `EntityData.site_pos_w`, which converts
        `site_xmat` to quaternions on the way and raises on the native backend --
        see `posture/mdp/state.py::foot_pos_w`.
        """
        data = self.robot.data
        feet = data.data.site_xpos[:, data.indexing.site_ids][:, self._foot_sites, 2]
        z = torch.cat([feet, data.root_link_pos_w[:, 2:3]], dim=1)
        z = z - self._env.scene.env_origins[:, 2:3]
        return (z - self._home_z).min(dim=1).values

    def _heights_at_home(self) -> torch.Tensor:
        """The six foot sites' heights above the ground, then the base_link
        origin's, with the robot at its default root pose and HOME: forward
        kinematics on a scratch copy of the compiled model, so nothing the
        simulation holds is touched and a new revision of the robot brings its
        own numbers."""
        m = self._env.sim.mj_model
        d = mujoco.MjData(m)
        ix = self.robot.indexing
        d.qpos[ix.free_joint_q_adr.cpu().numpy()] = (
            self.robot.data.default_root_state[0, :7].cpu().numpy()
        )
        d.qpos[ix.joint_q_adr.cpu().numpy()] = self.robot.data.default_joint_pos[0].cpu().numpy()
        mujoco.mj_kinematics(m, d)
        sites = ix.site_ids[self._foot_sites].cpu().numpy()
        z = [*d.site_xpos[sites, 2], d.xpos[ix.root_body_id, 2]]
        return torch.as_tensor(z, dtype=torch.float32, device=self.device)


__all__ = ["RefFreeJumpCommand", "RefFreeJumpCommandCfg", "push_start_phase"]
