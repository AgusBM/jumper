"""Driving jumper.posture in `play`: one person, both commands, pad or keyboard.

This task has **two** commands, the walk and the posture, and one operator for
both. The controls are the task's `controls.yaml`, read by
`common/mdp/operator.py`; this module is only the posture command's end of it.

    pad                                               keyboard
    left stick      walk: forward/back, left/right    W S A D, or the arrows
    right stick     up/down: nose down/up             I K
                    left/right: twist, then turn      Shift + J L twist, J L turn
    R3 held         left/right rolls instead,         U O roll
                    up/down raises and lowers         N M raise and lower
    R3 tapped       back to standing height           (B)
    B               hand both commands back           B
    (hands off)     stand still, level and square     (let go)

**The right stick's left-right is two things in order.** The first half of its
travel twists the body over its planted feet; past half-travel the turn climbs
from zero and the twist unwinds, back to zero at three quarters, where the turn
is at half its top rate -- so the body leads into the turn and is square again
before the turn is fast. **Held down it is the other layer**: with `R3` held the
left-right travel rolls and the up-down travel moves the height, up taller, with
twist, turn and pitch at zero, and letting go of `R3` is the turn and the pitch
again at once. **The height is moved, not placed** (`integrate_s` in the file,
since 2026-09-29): the stick is a speed, and let go the body stays at the height
it got to; `R3` tapped -- pressed and let go without the stick moving -- puts it
back at standing height, and so does the release. The height was on the triggers
until the morning of 2026-09-29, and placed by the stick until that afternoon.

**Full deflection is the standing edge, and walking narrows it.** A stick pushed
all the way asks for the edge of `ranges` -- the band the command draws from
while parked -- and while the velocity command walks the three angles are held to
`moving` (`PostureCommand.hold_to_band`). So pushing the right stick past half
travel into a turn brings the twist back from its standing edge to its walking
one at once, a turning robot being a moving one, and the unwinding comes under
that edge at 0.625 of the travel. The controller does the same on the robot and
in a browser, from the band the contract carries.

**The keyboard is a path of its own** (since 2026-09-29; it was a virtual pad
read through the pad's mapping until then). Each keystroke pushes one direction
of one axis -- full after the file's `full_after_s`, two seconds, and back at
rest the moment it comes up -- laid out as the robot's operator guide,
Control-agent 3.1, lays it out: `J` and `L` turn, `Shift` with them twists, `U`
and `O` roll, `N` and `M` move the height. Either device drives every channel
alone; the one touched last drives.

## Every letter also toggles something in the viewer

MuJoCo's built-in shortcuts are handled in C++ and the user callback runs **in
addition to** them, so each letter bound here also flips a render flag:

    W wireframe     A auto-connect   S shadow        D static body
    I inertia       K skybox         J joint         L additive
    U actuator      O perturb object N island        M centre of mass
    B perturb force

There is no way to intercept that from Python -- `mjrl.viewer.keys` says so. The
cost is cosmetic, the robot obeys the key, and it was accepted for a keyboard
shaped like the pad. Pressing a letter again puts its flag back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ...common.mdp.controls import Controls
from ...common.mdp.operator import connect
from .commands import PostureCommand, PostureCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: The task's controls: both commands, both devices. Read by `play` through
#: `velocity_env_cfg` and through `env_cfg`'s replay block, which both parse it.
CONTROLS = Path(__file__).resolve().parents[1] / "controls.yaml"


@dataclass(kw_only=True)
class TeleopPostureCommandCfg(PostureCommandCfg):
    """A posture command the operator can take over.

    A drop-in subclass of the sampling term, so every range, the neutral posture
    and the resampling clock carry over untouched -- and until somebody touches
    a control, this *is* the sampling term.
    """

    controls: Controls
    """The task's parsed `controls.yaml`, which describes this command."""

    term: str = "posture"
    """This term's key in `cfg.commands`, which is how the file names it."""

    pad: bool = True
    """Open a pad if one is connected. Must match the velocity term's: one
    person, one device."""

    pad_name: str | None = None

    def build(self, env: ManagerBasedRlEnv) -> TeleopPostureCommand:
        return TeleopPostureCommand(self, env)


class TeleopPostureCommand(PostureCommand):
    """`PostureCommand` with the environment's operator on top of it."""

    cfg: TeleopPostureCommandCfg

    #: `posture_command`'s columns.
    LAYOUT = ("twist", "pitch", "roll", "height")

    def __init__(self, cfg: TeleopPostureCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._operator = connect(cfg, env, self.LAYOUT)

    def compute(
        self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        super().compute(dt, env_ids)
        values = self._operator.command(self.cfg.term, self._env.common_step_counter)
        if values is None:
            return
        # Every environment gets the same posture, for the reason the velocity
        # term gives: a field where one robot obeys and fifteen wander is
        # harder to read, not richer.
        self.posture_command[:] = torch.tensor(values, device=self.device, dtype=torch.float32)
        # Full deflection is the standing edge; a walking robot is held to the
        # moving band, the same clamp the controller applies on the robot.
        self.hold_to_band()


__all__ = ["CONTROLS", "TeleopPostureCommand", "TeleopPostureCommandCfg"]
