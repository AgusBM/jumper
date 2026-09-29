"""Events only `jumper.five_foot` uses: holding the carried arm, and what it carries.

Both are reset/startup events rather than anything the policy touches -- the arm
is out of the action, and this is what decides where "out of the action" leaves
it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

__all__ = ["claw_payload", "hold_carried_arm"]


def hold_carried_arm(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: tuple[tuple[float, float], ...],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Lock the carried leg's joints at a per-episode random pose inside `ranges`.

    `ranges` is one (low, high) per joint, aligned with `asset_cfg.joint_ids` --
    so the `SceneEntityCfg` must be built with `preserve_order=True`, or the
    ranges land on whichever joints the model happens to list first. Passing the
    two together, in one order, is the only thing keeping them paired; nothing
    downstream can tell a shuffled pairing from a correct one.

    **Writing the position target is the crux, not the state.** mjlab clears
    `joint_pos_target` to 0.0 on every reset and the action term then writes only
    the joints it drives, so a joint taken out of the action is commanded to
    0.0 rad -- not held where it was put. Measured on `jumper.flat`, whose grippers
    are already out of the action: `LF_J4_joint` leaves HOME's -1.0 and
    reaches -0.0001 in 25 control steps (on the model before V1.6, whose HOME
    had the fingers at -1.0 rather than 0.0). Writing only the state here would give a
    claw that starts at its hold and folds shut over the first half second of
    every episode, with no error anywhere and every other signal healthy.

    Both writes are scoped to `env_ids`, so a partial reset does not disturb the
    arms of environments that are mid-episode.
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)

    low = torch.tensor([r[0] for r in ranges], device=env.device)
    high = torch.tensor([r[1] for r in ranges], device=env.device)
    names = list(asset_cfg.joint_names or ())
    if low.shape[0] != len(names):
        raise ValueError(
            f"hold_carried_arm got {low.shape[0]} ranges for {len(names)} joints"
        )
    # An inverted range still samples the same interval -- `low + (high - low) * u`
    # walks it backwards and lands in the same place -- so nothing downstream
    # would ever notice, and the next reader would take the pair to mean what it
    # says. The claw's aperture is where this is easy to get wrong: on that joint
    # "open" is the more negative number.
    if bool((high < low).any()):
        wrong = [n for n, lo, hi in zip(names, low.tolist(), high.tolist()) if hi < lo]
        raise ValueError(f"hold_carried_arm ranges are (low, high); inverted: {wrong}")
    hold = low + (high - low) * torch.rand(
        (len(env_ids), low.shape[0]), device=env.device
    )

    asset.write_joint_state_to_sim(
        hold, torch.zeros_like(hold), joint_ids=joint_ids, env_ids=env_ids
    )
    asset.set_joint_position_target(hold, joint_ids=joint_ids, env_ids=env_ids)


_PAYLOAD_ID: dict[int, int] = {}


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def claw_payload(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float],
    unit_inertia: tuple[float, float, float],
    body_name: str,
) -> None:
    """Load the claw with something, per environment: a mass **and its inertia**.

    `dr.body_mass` is the obvious way and its own warning rules it out: it writes
    mass and leaves the inertia tensor alone, which is a point mass at whatever
    body it is pointed at. A carried object is neither a point nor at a link's
    centre of mass -- see `claw.py::PAYLOAD_BODY` for what that cost -- so this
    writes both onto the body `claw.py::add_payload_body` parks on the mouth, and
    scales the inertia with the mass so the two stay a possible object.

    Sampled once per environment and held from then on: an episode that changed
    what the claw is holding halfway through would be teaching the policy to
    expect something no operator can do. As a startup event it loads the first
    rung of `mdp/curriculum.py::PayloadCurriculum`, an empty claw; the curriculum
    calls this function again, with the next rung's `ranges`, for each
    environment at the reset after its promotion -- never mid-episode.
    """
    asset_ids = resolve_env_ids(env, env_ids)
    key = id(env)
    if key not in _PAYLOAD_ID:
        import mujoco

        found = mujoco.mj_name2id(
            env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{body_name}"
        )
        if found < 0:
            raise ValueError(
                f"no body named 'robot/{body_name}' in the compiled model -- the "
                f"payload body is added by claw.add_payload_body, which the task "
                f"config has to wire onto the robot's spec_fn"
            )
        _PAYLOAD_ID[key] = found
    body = _PAYLOAD_ID[key]

    lo, hi = ranges
    mass = torch.empty(len(asset_ids), device=env.device).uniform_(lo, hi)
    unit = torch.tensor(unit_inertia, device=env.device)
    env.sim.model.body_mass[asset_ids, body] = mass
    env.sim.model.body_inertia[asset_ids, body] = mass.unsqueeze(1) * unit
