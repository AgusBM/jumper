"""Operator control of the carried claw's aperture, from its task control.

`LF_J4_joint` is not in the action -- the policy walks on five legs and does
not drive the claw -- so nothing in the normal step loop moves it. What holds it
is `events.py::hold_carried_arm`, which samples an aperture per episode and writes
the position target once at reset. That is right for training, where the point is
that the policy sees the whole open-to-shut travel across episodes, and useless
for an operator watching a replay who wants the claw to close *now*.

This is the other half: a `mode="step"` event that writes the finger's position
target every control step, from the claw's control. It is **replay-only** --
`env_cfg.py` adds it under `play` -- for the same reason the velocity teleop is:
during training there is no viewer and no pad, and a term that could be driven
from outside would make a run unreproducible.

## The control is the claw

`claw_left`, the task control at the carried claw's side, and its travel is the
claw's: let go is `GRIPPER_OPEN`, all the way is `GRIPPER_CLOSED`, and halfway
is halfway. The task's `controls.yaml` declares it (`task:`) and puts it on `LT`
on the pad and Space on the keyboard, and it arrives through the family's
operator (`common/mdp/operator.py::Operator.task_control`) by that name, from
whichever device was touched last: squeezed as far as the trigger is, or closing
for as long as Space is held and opening when it comes up. The robot answers the
same file the same way, in `deploy/lib.rs`, where a mode carrying the claw on the
right answers `claw_right` -- `RT`, and Space again.

It replaced `[` / `]` / `G`, kk-rl-lab's keys, which stepped the target a press
at a time and left it there. Those were a mapping of the operator's input beside
the one `controls.yaml` describes -- on keys MuJoCo's viewer already binds to
its cameras -- and the robot had no counterpart to either.

## Why a step event and not an action

Putting the finger back in the action would change the policy's input and output
widths, invalidate every checkpoint trained so far, and hand a joint to a network
that was never rewarded for anything it does with it. The claw is not part of the
locomotion problem; it is a thing the operator aims. A step event is the smallest
mechanism that says exactly that.

`ManagerBasedRlEnv.step` applies `mode="step"` events **after** the decimation
loop, so a squeeze takes effect on the next control step -- 20 ms at 50 Hz, which
no operator can see. The events come before the commands in that step, so this
term is usually the one that reads the pad for it; the commands then find it read.

## Handing over

Nothing is written until the operator has taken over -- the first touch of either
device, the moment the commands leave the sampler -- so an untouched replay is
the replay it was: `hold_carried_arm` put the claw somewhere and it stays. From
then on the claw follows its control every step, and a release (`B`) hands it
back the way the commands go back: the term stops writing, and the claw keeps
its last target until a reset.

The thread discipline is the operator's: its key handler runs on the viewer's
UI thread and touches only Python numbers, and this term reads one of them and
does every device read and write on the stepping thread.

## A reset

`hold_carried_arm` pins the finger wide open on reset (`env_cfg.py`, block 4).
With the operator driving, the control decides again on the next step: a claw
held shut through a reset shuts again, because the control is the claw. What a
reset does forget is a hold.

## `play --hold`

`mdp/grasp.py::ClawHold` starts an episode with a prop squeezed in the claw, and
a control at rest would open it on the next step. So the hold is handed over
(`hold_at`): the claw stays shut until its control is held as far as the hold
is shut, and follows the control from there -- let go, and it opens. Taking the
claw back at the position it is already in is what keeps the hand-over from
snapping it open.

## How hard it squeezes

Squeezed all the way, the target is `GRIPPER_CLOSED`, but the target written is
never more than `claw.py::SQUEEZE_LEAD` past where the finger actually is
(`squeeze_limited`), plus `claw.py::LEAD_PER_SPEED` of its closing speed. A
finger closing on nothing is led the further the faster it goes, and closes at
the servo's own pace; a finger stopped by an object squeezes with the same
0.6 N*m however hard the control is held. The policy observes the finger's torque and never trained with
one, and squeezed shut the servo's 1.2 N*m halved how well it turns -- `claw.py`
has the measurement. The robot applies the same limit (`deploy/lib.rs`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ...common.mdp.operator import Operator
from ..claw import (
    FINGER_JOINT,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LEAD_PER_SPEED,
    SQUEEZE_LEAD,
    aperture_mm,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["CONTROL", "GripperTeleop", "gripper_teleop_event", "squeeze_limited"]

#: The task control at the carried claw's side, by `controls.yaml`'s name for
#: it. The claw is on the left in training and so in every replay. Until
#: 2026-09-29 this was the pad's `LT` itself, and the keyboard reached it by
#: pressing a virtual pad's trigger.
CONTROL = "claw_left"


def squeeze_limited(
    target: torch.Tensor,
    angle: torch.Tensor,
    lead: float = SQUEEZE_LEAD,
    speed: torch.Tensor | None = None,
    per_speed: float = LEAD_PER_SPEED,
) -> torch.Tensor:
    """The finger target actually sent: never more than `lead` shut of the finger,
    plus `per_speed` seconds of its closing `speed` when that is given.

    A target below the angle -- an opening -- passes through untouched, and a
    finger closing on nothing is led all the way to its target, the further the
    faster it closes (`claw.py::LEAD_PER_SPEED`). What changes is a finger
    stopped by an object: its speed is zero, and its servo squeezes with
    `STIFFNESS * lead`, however far past the object the operator has squeezed.
    `claw.py::SQUEEZE_LEAD` has why that matters.
    """
    if speed is not None:
        return torch.minimum(target, angle + lead + per_speed * speed.clamp(min=0.0))
    return torch.minimum(target, angle + lead)


class GripperTeleop:
    """Drive `LF_J4_joint`'s position target from the claw's task control.

    A `mode="step"` event term, so mjlab constructs it once with `(cfg, env)` and
    then calls it with `(env, env_ids)` every control step.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        params = cfg.params
        asset_cfg: SceneEntityCfg = params["asset_cfg"]
        self._asset = env.scene[asset_cfg.name]

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            joint_ids = list(range(*joint_ids.indices(self._asset.num_joints)))
        if len(joint_ids) != 1:
            raise ValueError(
                f"GripperTeleop drives exactly one joint, got {len(joint_ids)}: "
                f"{asset_cfg.joint_names}"
            )
        self._joint_ids = torch.as_tensor(list(joint_ids), device=env.device)

        self._open = float(params.get("open_at", GRIPPER_OPEN))
        self._closed = float(params.get("closed_at", GRIPPER_CLOSED))
        # (open, closed) rather than (low, high): on this joint more-open is the
        # *lower* number, which is the sign error `hold_carried_arm` guards
        # against in its own ranges and the reason the two ends are named for what
        # they do rather than for which is larger.
        if not self._open < self._closed:
            raise ValueError(
                f"GripperTeleop expects open_at < closed_at on this joint, got "
                f"open_at={self._open} closed_at={self._closed}"
            )
        self._lead = float(params.get("squeeze_lead", SQUEEZE_LEAD))
        self._per_speed = float(params.get("lead_per_speed", LEAD_PER_SPEED))
        self._control = str(params.get("control", CONTROL))

        # Found on the first step: the event manager builds its terms before the
        # command manager, whose terms are what make the operator.
        self._operator: Operator | None = None
        # What the operator's claw control last asked for, and a hold `play --hold`
        # handed over; `None` for neither. Plain floats, read by the tests.
        self._target: float | None = None
        self._hold: float | None = None
        # Filled and written once per step; allocated here so the hot path does
        # not allocate.
        self._buf = torch.zeros((env.num_envs, 1), device=env.device)

        print(f"[gripper] {self._control} closes the claw as far as it is held, "
              f"{aperture_mm(self._open):.0f} mm down to {aperture_mm(self._closed):.0f}; "
              "let go and it opens")

    def hold_at(self, target: float) -> None:
        """Hold the claw at `target` until its control is held as far.

        For `play --hold` (`mdp/grasp.py::ClawHold`), which shuts the claw on a
        prop at the start of an episode: untold, a control at rest would open
        the jaws on the prop the next step. From the stepping thread.
        """
        self._hold = max(self._open, min(self._closed, float(target)))

    def _asked(self, squeeze: float) -> float:
        """The control's travel laid onto the claw's, clamped to the working range.

        Both ends are a place where squeezing further stops meaning what it says:
        past `closed` the jaws are shut on each other, and just past `open` the
        moving jaw closes on an edge rather than its grip face. And both are
        landed on exactly: `-0.65 + 1.0 * 0.75` is 0.09999999999999998, and shut
        being a value that varies with arithmetic is the kind of thing something
        downstream compares for equality one day.
        """
        if squeeze >= 1.0:
            return self._closed
        return self._open + max(0.0, squeeze) * (self._closed - self._open)

    def __call__(self, env: Any, env_ids: Any = None, **kw: Any) -> None:
        del env_ids, kw
        if self._operator is None:
            self._operator = Operator.of_env(env)
            if self._operator is None:
                raise RuntimeError(
                    "the claw follows the operator's claw control, and this environment has no "
                    "operator -- its command terms are not the operator's. The claw "
                    "would sit wherever the reset left it with nothing to say why."
                )

        # A reset forgets a hold. Read from the episode counter rather than from
        # a reset hook because this is a `mode="step"` term and there is no reset
        # callback to hang it on; `mdp/grasp.py` finds a reset the same way.
        if self._hold is not None and int(env.episode_length_buf.min()) <= 1:
            self._hold = None

        squeeze = self._operator.task_control(self._control, env.common_step_counter)
        self._target = None if squeeze is None else self._asked(squeeze)
        held = self._hold is not None and self._target is not None
        if held and self._target >= self._hold - 1e-6:
            self._hold = None  # held as far as the hold: the control has it now
        target = self._hold if self._hold is not None else self._target
        if target is None:
            return  # nobody has taken over, and nothing is held

        # Every environment gets the same aperture, for the reason the family's
        # operator gives for the walk (`common/mdp/operator.py`): a field where one
        # robot obeys and fifteen do something else is harder to read, not richer.
        self._buf.fill_(target)
        angle = self._asset.data.joint_pos[:, self._joint_ids]
        speed = self._asset.data.joint_vel[:, self._joint_ids]
        self._asset.set_joint_position_target(
            squeeze_limited(self._buf, angle, self._lead, speed, self._per_speed),
            joint_ids=self._joint_ids,
        )


def gripper_teleop_event(
    *, squeeze_lead: float = SQUEEZE_LEAD, lead_per_speed: float = LEAD_PER_SPEED,
    control: str = CONTROL,
) -> EventTermCfg:
    """The event term config, so `env_cfg.py` states the wiring in one line.

    Args:
        squeeze_lead: radians the finger may be told past where it is.
        lead_per_speed: seconds of its closing speed added to that lead.
        control: the task control that closes the claw, by the name the
            task's `controls.yaml` declares it under.
    """
    return EventTermCfg(
        func=GripperTeleop,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[FINGER_JOINT]),
            "open_at": GRIPPER_OPEN,
            "closed_at": GRIPPER_CLOSED,
            "squeeze_lead": squeeze_lead,
            "lead_per_speed": lead_per_speed,
            "control": control,
        },
    )
