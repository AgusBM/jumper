"""Observation terms for the reference-guided high jump: phase + future reference preview.

- `jump_phase`  -- scalar 0->1, marking how far the motion has progressed. Without
  it, the same joint state appears in both the crouch phase and the recovery phase,
  and the policy cannot tell whether to push or to tuck.
- `ref_future`  -- reference joint angles at several future moments (relative to
  HOME, in GAIT_JOINTS order). The push-off phase is only 40 ms (8 steps at 200 Hz),
  so watching the current phase alone means knowing to push only once it is nearly
  over; the preview gives the policy reaction time.
"""

from __future__ import annotations

import torch

from ...common.constants import GAIT_JOINTS, HOME
from .reference import ref_state

#: Offset moments (s) of the future preview. Covers 0.01-0.16 s: push-off 0.04 s
#: and flight 0.34 s both fall inside. Uniform sampling at 50 ms steps would miss
#: the push-off, so use roughly 2x tighter spacing.
FUTURE_OFFSETS = (0.01, 0.02, 0.04, 0.08, 0.16)

# The reference's joint order matches the MJCF (it is permuted to the entity joint
# names on load, see reference.py), so statically compute the columns GAIT_JOINTS
# occupies in the 22-dim q.
_GAIT_COL = [i for i, j in enumerate(HOME) if j in set(GAIT_JOINTS)]
assert len(_GAIT_COL) == len(GAIT_JOINTS)

# HOME is ordered by the 22-dim entity joint order; taking the GAIT columns
# gives the 20-dim baseline.
_HOME_VEC = torch.tensor([list(HOME.values())[i] for i in _GAIT_COL])


def jump_phase(env, command_name: str = "jump") -> torch.Tensor:
    _, _, phase, _ = ref_state(env, command_name)
    return phase.unsqueeze(1)


def ref_future(
    env,
    command_name: str = "jump",
    offsets: tuple[float, ...] = FUTURE_OFFSETS,
) -> torch.Tensor:
    r, _, _, tsg = ref_state(env, command_name)
    home = _HOME_VEC.to(tsg.device)
    cols = []
    for dt in offsets:
        s = r.sample(r.index_of(tsg + dt))
        cols.append(s["q"][:, _GAIT_COL] - home)
    return torch.cat(cols, dim=1)


__all__ = ["FUTURE_OFFSETS", "jump_phase", "ref_future"]
