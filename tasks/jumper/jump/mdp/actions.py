"""Residual joint-position action: target = current reference joint angle + scale × policy output.

During the jump's push-off the target deviates from HOME by about 1 rad (knees
and ankles fully extended). If actions were based at HOME as in walking tasks,
the policy would have to explore the whole crouch-jump trajectory before it
could earn any tracking score -- under reference guidance that is unlearnable.
With the reference as the baseline the policy only has to learn the
**deviation**, and a near-zero residual at cold start already tracks the
reference.

Same as the lab project (`ReferenceResidualJointPositionAction`), with the
scale this project uses, 0.25 (`JUMPER_ACTION_SCALE`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
    command_name: str = "jump"
    """Name of the command term that provides the reference motion."""

    def __post_init__(self):
        super().__post_init__()
        # The residual scheme's baseline is the reference joint angle, not HOME -- the
        # offset must be 0.
        self.use_default_offset = False

    def build(self, env: "ManagerBasedRlEnv") -> "ReferenceResidualJointPositionAction":
        return ReferenceResidualJointPositionAction(self, env)


class ReferenceResidualJointPositionAction(JointPositionAction):
    cfg: ReferenceResidualJointPositionActionCfg

    def __init__(self, cfg: ReferenceResidualJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.command_name
        self._cmd_term = None

    def _get_ref_q(self) -> torch.Tensor:
        """Controller command q_cmd (E, num_targets) at the end of this control interval
        (t + step_dt).

        The baseline uses q_cmd (the target the recorded controller actually sent)
        rather than q (the state actually reached): zero action then exactly
        reproduces the recorded controller's PD behavior. Over the push-off the two
        differ by 0.3 rad on average, and using q as the baseline amounts to a
        double lag, making the push-off weak -- one of the physical roots of the
        earlier policy being forced to invent its own motion (crouch and lift the
        middle legs). The command term is built before the action manager (see the
        ordering in ManagerBasedRlEnv.load_managers), so it is fetched lazily once
        here.
        """
        if self._cmd_term is None:
            self._cmd_term = self._env.command_manager.get_term(self._command_name)
        cmd = self._cmd_term
        # Absolute recording time = time elapsed since go + the recording's own lead-in;
        # one step ahead (the ZOH target should be the interval's end value).
        t_abs = cmd.time_since_go + cmd.reference.t_go + self._env.step_dt
        return cmd.reference.sample_cmd(t_abs)

    def apply_actions(self) -> None:
        ref_q = self._get_ref_q()
        target = ref_q[:, self._target_ids] + self._processed_actions
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(
            target - encoder_bias, joint_ids=self._target_ids
        )


__all__ = [
    "ReferenceResidualJointPositionAction",
    "ReferenceResidualJointPositionActionCfg",
]
