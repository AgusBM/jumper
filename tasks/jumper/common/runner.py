"""The runner the laddered jumper tasks train with: mjlab's velocity runner, with
the curriculum levels in its checkpoints.

mjlab's runner checkpoints `common_step_counter` "to preserve curricula state",
which covers a curriculum that is a function of the step -- an annealed weight, a
schedule. It does not cover one that *earns* its level. `CommandRangeCurriculum`,
`PostureRangeCurriculum` and `PayloadCurriculum` keep theirs on the term object, so
a resume used to start them at 0 (or wherever `MJRL_*_LEVEL` said) with the step
counter already past every dwell: a policy that had outgrown the narrow ranges was
trained back down onto them, and the first promotion was free to land as soon as
the error average had its first sample.

This saves every `CheckpointedLadder`'s `state_dict()` under `infos["curriculum"]`,
keyed by term name, and restores it on load. Which source wins when a level is set
twice -- the checkpoint or `MJRL_*_LEVEL` -- is `CheckpointedLadder.load_state_dict`'s
decision, not this file's.

**Per-environment state is not carried.** `TerrainLevels`' rows are drawn afresh, as
on any start. `PayloadCurriculum`'s record of which environment carries which rung
starts empty, so every environment takes the restored rung's load at its first
reset -- what `MJRL_PAYLOAD_LEVEL` always did.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .mdp.curriculum import CheckpointedLadder

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

__all__ = ["CHECKPOINT_KEY", "CurriculumRunner", "curriculum_state", "ladders", "restore_curriculum"]

#: Where in a checkpoint's `infos` the ladders are kept.
CHECKPOINT_KEY = "curriculum"


def ladders(env: ManagerBasedRlEnv) -> list[tuple[str, CurriculumTermCfg]]:
    """`(term name, term cfg)` of every checkpointed ladder, in the manager's order.

    Found by type, as `TerrainLevels` finds the command ladder: the manager
    replaces a class term's `func` with its instance, so the object is on the cfg,
    and renaming a term in a config cannot hide it. A replay builds with
    `curriculum = {}` and has none.
    """
    cm = getattr(env, "curriculum_manager", None)
    names = getattr(cm, "_term_names", None) or ()
    cfgs = getattr(cm, "_term_cfgs", None) or ()
    return [
        (name, cfg)
        for name, cfg in zip(names, cfgs, strict=True)
        if isinstance(cfg.func, CheckpointedLadder)
    ]


def curriculum_state(env: ManagerBasedRlEnv) -> dict[str, dict]:
    """Every ladder's `state_dict()`, by term name."""
    return {name: cfg.func.state_dict() for name, cfg in ladders(env)}


def restore_curriculum(env: ManagerBasedRlEnv, saved: dict[str, dict] | None) -> list[str]:
    """Load `saved` into the live ladders; return one line per ladder for the log.

    `saved` is None for a checkpoint written before levels were saved; every
    ladder then keeps the level its constructor chose.
    """
    found = ladders(env)
    step = env.common_step_counter
    lines = [
        f"{name}: {cfg.func.load_state_dict((saved or {}).get(name), step)}"
        for name, cfg in found
    ]
    # **Written into the managers now, not at the first reset.** The wrapper reset
    # every environment when it was built, so each ladder has already applied the
    # constructor's level to the command ranges and the reward stds; left alone,
    # the restored level would take over only at the next reset batch. A call with
    # no environments applies it and samples nothing. After every ladder is loaded,
    # and in the manager's order, because the payload ladder reads the command
    # ladder's level. Commands already drawn keep their range until they resample.
    none = torch.empty(0, dtype=torch.long, device=env.device)
    for name, cfg in found:
        env.curriculum_manager._curriculum_state[name] = cfg.func(env, none, **cfg.params)
    return lines


class CurriculumRunner(VelocityOnPolicyRunner):
    """`VelocityOnPolicyRunner`, whose checkpoints also carry the curriculum levels.

    Every task with a `CheckpointedLadder` in its curriculum has to return this from
    `runner_cls()`; `tests/test_curriculum_resume.py` fails on one that does not,
    because the default runner saves nothing and says nothing.
    """

    def save(self, path: str, infos: dict | None = None) -> None:
        state = curriculum_state(self.env.unwrapped)
        super().save(path, {**(infos or {}), CHECKPOINT_KEY: state})

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        env = self.env.unwrapped
        if ladders(env):
            saved = (infos or {}).get(CHECKPOINT_KEY)
            if saved is None:
                print(
                    f"[mjrl] {path} predates checkpointed curriculum levels; "
                    f"they come from MJRL_*_LEVEL, or start at 0"
                )
            for line in restore_curriculum(env, saved):
                print(f"[mjrl] curriculum {line}")
        return infos
