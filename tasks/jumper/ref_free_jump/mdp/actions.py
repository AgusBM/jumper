"""Absolute joint targets, with the recording's command fed forward while training.

The target this term applies is

    HOME + scale * action + feedforward * (q_cmd(t) - HOME)      inside the span
    HOME + scale * action                                        outside it

`feedforward` is 1 until `hold_step`, falls linearly to 0 by `end_step`, and is
0 from then on -- which is the only value the robot will ever run. Early on a
zero action is therefore exactly the recording's command, the one `jumper.jump`'s
residual starts from, and the policy is never asked to *discover* a 50 ms push it
has never felt. As the feedforward comes down the policy takes the motion over a
little at a time, with the joint-tracking prior asking it to keep the body where
the recording's was.

**Why it had to exist.** With the recording as a reward only, the first run
learnt to crouch and stay crouched: `model_300` of `2026-09-26_18-21-37` matched
the prior at 0.91 in the crouch and 0.99 in the landing, where holding a crouch
is nearly the right pose, and at 0.39 in the push and 0.33 in the flight, no
better than standing at HOME (0.43 and 0.28). It never left the ground. The push
and the flight are a third of the span and need the mid and rear knees 1.8 rad
from HOME within about 50 ms; far from the recording an exponential kernel has
no gradient to follow, and a half-made push costs the landing, so not jumping was
the better deal. A prior on the joints' state has the same shape: the same
checkpoint, measured on this task's `prior_joint_pos`, scores 0.78 in the crouch,
0.63 in the push, 0.24 in the flight and 0.87 in the landing. What breaks the
trap is starting from the jump, not which prior measures it.

**`feedforward` is read off `common_step_counter` on every step**, not set by the
curriculum at resets. `mjlab/rl/runner.py` restores that counter when it loads a
checkpoint, so a replay runs a checkpoint with the feedforward it was trained
with from its first step. One from before `end_step` still leans on the
recording: replaying it without would show a policy that cannot jump, which it
could not on the robot either.

Inside the span only: before the go the robot waits at HOME, and after the span
the recording's command is its own stand, not HOME.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class PriorFeedforwardJointPositionActionCfg(JointPositionActionCfg):
    command_name: str = "jump"
    """The command term that holds the recording and the go."""

    hold_step: int
    """Environment step until which the feedforward is 1."""

    end_step: int
    """Environment step from which it is 0."""

    def __post_init__(self):
        super().__post_init__()
        # The feedforward is written relative to HOME, which is this offset.
        if not self.use_default_offset:
            raise ValueError("the feedforward is relative to HOME: use_default_offset must be True")
        if not 0 <= self.hold_step < self.end_step:
            raise ValueError(
                f"need 0 <= hold_step < end_step, got {self.hold_step} and {self.end_step}"
            )

    def build(self, env: ManagerBasedRlEnv) -> PriorFeedforwardJointPositionAction:
        return PriorFeedforwardJointPositionAction(self, env)


class PriorFeedforwardJointPositionAction(JointPositionAction):
    cfg: PriorFeedforwardJointPositionActionCfg

    @property
    def feedforward(self) -> float:
        """1 until `hold_step`, 0 from `end_step`, linear between."""
        s = self._env.common_step_counter
        frac = (s - self.cfg.hold_step) / (self.cfg.end_step - self.cfg.hold_step)
        return 1.0 - min(max(frac, 0.0), 1.0)

    def apply_actions(self) -> None:
        target = self._processed_actions
        ff = self.feedforward
        if ff > 0.0:
            cmd = self._env.command_manager.get_term(self.cfg.command_name)
            ref = cmd.reference
            # Held over the interval, so read at its end -- `jumper.jump`'s
            # baseline convention.
            t_end = cmd.time_since_go + self._env.step_dt
            q_cmd = ref.sample_cmd(t_end + ref.t_go)[:, self._target_ids]
            window = ((t_end >= 0.0) & (t_end <= ref.span)).unsqueeze(1)
            target = target + ff * (q_cmd - self._offset) * window
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)


__all__ = ["PriorFeedforwardJointPositionAction", "PriorFeedforwardJointPositionActionCfg"]
