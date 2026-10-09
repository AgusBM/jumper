"""jumper.calligraphy -- jumper.five_foot, fine-tuned to stand still while the arm writes.

## Built on five_foot's config, on purpose

Every other jumper task builds its config from the shared skeleton and owns each
number in it. This one starts from `five_foot.env_cfg` instead, and that is the
point rather than a shortcut: it is trained **from five_foot's checkpoint**
(`--checkpoint tasks/jumper/five_foot/out/example/model_86600.pt`), and a
checkpoint loads only into a network with the same observations and actions, in
the same order, at the same sizes. Restating five_foot's ~30 reward terms, its
observation set and its commands here would be a second copy that has to stay
identical to the first for the warm start to load at all. So everything is
five_foot's, and what this file changes is listed below and nothing else:

| | five_foot | here |
|---|---|---|
| the brush | none | `brush.apply`: a 15 g brush in the shut claw (visual hair, no contact) |
| the carried arm | held in `LF_GRASP_BOX` all episode | out about half the time: unfolded, moved over the writing band at writing speed, folded back (`mdp/writing.py`) |
| the commands while the arm is out | sampled | velocity zero, body level -- what `tools/write.py` sends |
| reward | -- | `hold_position`: stay where the trunk stood when it became still |
| `feet_still` weight | -2.0 | -10.0: a foot slid along the floor while standing costs five times as much |
| `feet_planted` gate | command under 0.10 m/s | under 0.05, the gate every other standing term uses |

Observations are untouched -- the arm's joints were already observed (five_foot
observes the carried joints), and the writing command is not -- so the shipped
checkpoint loads strictly. The operator controls are five_foot's file, read by
five_foot's config; a copy here would be read by nothing.

## The numbers, and where they come from

- `rel_writing_envs` 0.5 and segments of 4-10 s: the arm out about half of
  every episode, so standing with it out is learnt without the walking that
  five_foot already does being forgotten.
- unfold 2-4 s, fold 1.5-3 s: `tools/write.py` unfolds in 3 s (1.5 s staggered
  the shipped policy, run 9); either side of it.
- moves 2-8 cm at 2-6 cm/s: a stroke stretch of 无 at 10-30 cm, written at 3 cm/s.
- heights 20% hover, 30% on the floor, 50% sunk: most of writing is with the hair
  down.
- `hold_position` std 10 mm and 3 deg: the tolerance writing has -- the 10 mm
  planning margin and 3 deg, which at the band's 0.25 m is 13 mm. Weight 4.0, the
  same as `track_linear_velocity`: standing where you were is as much the task
  as going where you are told. A guess at the right balance, to be read off
  `Episode_Reward/hold_position` against `track_linear_velocity` in the first
  run.
- grace 0.5 s: the last step of a walk lands within it (run 8, the trunk settled
  5-7 mm over about that long).

## The second round: feet that slide, and a walk that will not start

`out/model_89599.pt`, the first round (3000 iterations from five_foot's), holds the
trunk still while the arm writes -- 0-2 mm -- and two habits came with it, both
measured in replay (2026-10-09, native:cpu):

- **It moves its feet by sliding them.** As the arm unfolds every foot slides
  along the floor without lifting: 4-68 mm per foot over a 4 s unfold (five_foot's
  own policy: 4-124 mm, the trunk rising up to 45 mm). Nothing charged it enough:
  `feet_still` prices summed foot speed while standing at -2.0, so that slide --
  ~0.045 m/s summed -- cost 0.09 per second against `hold_position`'s 4.0. At
  -10.0 it costs 0.45, and stepping or holding the feet becomes the cheaper way.
- **It stands for slow commands.** A push of 0.07 m/s for 0.12-1.5 s moved it
  0.3-2.1 mm, 0.10 m/s 0.6-5 mm (five_foot's: 4-100 mm), so `tools/write.py` walks
  with five_foot's policy -- and the switch between the two stances at the unfold
  is itself a slide of 26-95 mm per foot with the trunk rising 22 mm. five_foot's
  `feet_planted` pays every planted foot while the command is under 0.10 m/s, and
  every other standing term switches at 0.05: between the two, standing was paid
  and walking only half-scored, and with `hold_position` added the first round
  settled on standing. At 0.05 that band is gone.

Both are guesses at the cause, checked by replaying the result: the slide per
unfold (`feet moved`) and a push of 0.07-0.10 m/s that moves the robot.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from . import brush
from .mdp.writing import WritingArmCommandCfg, hold_position

WRITING = "writing"


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task's environment config.

    Args:
        asset: model XML path, from `--model`. None uses the default jumper.xml.
        play: replay mode. The writing command is left out: in replay the arm is
            driven by `tools/write.py`, and a term moving it too would fight it.
    """
    from ..five_foot.env_cfg import env_cfg as five_foot_env_cfg

    cfg = five_foot_env_cfg(asset=asset, play=play)
    brush.apply(cfg)
    if play:
        return cfg

    # Registered after `twist` and `body_pose`, so its overrides come last each step.
    cfg.commands[WRITING] = WritingArmCommandCfg(
        resampling_time_range=(4.0, 10.0),
        rel_writing_envs=0.5,
        unfold_s=(2.0, 4.0),
        fold_s=(1.5, 3.0),
        move_reach=(0.02, 0.08),
        move_speed=(0.02, 0.06),
        height_probs=(0.2, 0.3, 0.5),
        finger_hold=brush.FINGER_HOLD,
        stow_noise=0.05,
        grace_s=0.5,
    )
    cfg.rewards["feet_still"].weight = -10.0
    cfg.rewards["feet_planted"].params["command_threshold"] = 0.05
    cfg.rewards["hold_position"] = RewardTermCfg(
        func=hold_position,
        weight=4.0,
        params={"command_name": WRITING, "std_xy": 0.010, "std_yaw": math.radians(3.0)},
    )
    return cfg
