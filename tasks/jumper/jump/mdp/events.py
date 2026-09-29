"""RSI (Reference State Initialization) reset events.

An env spawns at a random phase of the reference trajectory: joint
positions/velocities and base pose/velocity all take the reference values.
Every env starts "somewhere on the trajectory", so the policy always learns to
"finish the motion from an intermediate state".

Ordering dependency: in mjlab's `_reset_idx` the reset events run **before**
`command_manager.reset()`, so this event records the spawn phase in
`env._jump_rsi_phase` and the command term consumes it (setting it to 0) in
`_resample_command`, so the delayed sampling is not overwritten.

The spawn velocity writes only linear velocity: the npz's angular-velocity
channel is stored under the MuJoCo free-joint convention (linear velocity in
the world frame, angular velocity in the body frame), and mixing the two risks
a silent frame mismatch; the body tilts < 2° throughout the jump, so the error
from zeroing angular velocity is negligible.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .reference import get_reference

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_from_reference_phase(
    env,
    env_ids: torch.Tensor | None,
    command_name: str = "jump",
    rsi_fraction: float = 1.0,
    phase_range: tuple[float, float] = (0.0, 0.9),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Spawn a fraction of envs at a reference phase; the others fall back to the
    default pose (standing).

    Joint/base state not covered by this event is handled by the
    `reset_scene_to_default` event in the config (which runs before this one).
    """
    del command_name, asset_cfg
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    robot: Entity = env.scene["robot"]
    ref = get_reference(env)

    r = torch.empty(n, device=env.device)
    is_rsi = r.uniform_(0.0, 1.0) < rsi_fraction
    phase = r.uniform_(*phase_range)

    # Record the spawn phase for the command term to consume at reset. 0 means non-RSI.
    if not hasattr(env, "_jump_rsi_phase"):
        env._jump_rsi_phase = torch.zeros(env.num_envs, device=env.device)
    env._jump_rsi_phase[env_ids] = torch.where(is_rsi, phase, torch.zeros_like(phase))

    rsi_ids = env_ids[is_rsi]
    if len(rsi_ids) == 0:
        return

    # Note: sampling uses positions local to this batch of envs; the global ids are
    # used only when writing back to sim. auto-reset passes in the subset to reset,
    # so indexing this batch's samples with global ids would go out of bounds.
    rsi_local = is_rsi.nonzero(as_tuple=False).flatten()

    # phase -> continuous frame index (the phase axis starts at go; index_of adds t_go).
    tsg = phase[rsi_local] * ref.span
    s = ref.sample(ref.index_of(tsg))

    # Base: xy and z both use the reference's absolute trajectory + the env
    # origin. The jump is vertical, so the xy offset ≈ 0.
    origin = env.scene.env_origins[rsi_ids]
    pos = origin + s["base_pos"]
    quat = s["base_quat"]
    lin_vel = s["base_lin_vel"]
    ang_vel = torch.zeros_like(lin_vel)

    root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)
    robot.write_root_state_to_sim(root_state, env_ids=rsi_ids)
    robot.write_joint_state_to_sim(s["q"], s["qd"], env_ids=rsi_ids)


__all__ = [
    "reset_from_reference_phase",
]
