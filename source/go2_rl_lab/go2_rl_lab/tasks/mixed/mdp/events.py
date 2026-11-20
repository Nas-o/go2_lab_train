"""Custom domain-randomization events for the Go2 mixed-terrain task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_joints_by_scale(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
    """Reset joint positions and velocities by scaling their default values."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    joint_pos = asset.data.default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()

    position_scale = sample_uniform(
        position_range[0],
        position_range[1],
        joint_pos.shape,
        device=env.device,
    )
    velocity_scale = sample_uniform(
        velocity_range[0],
        velocity_range[1],
        joint_vel.shape,
        device=env.device,
    )
    joint_pos *= position_scale
    joint_vel *= velocity_scale

    asset.write_joint_state_to_sim(
        position=joint_pos,
        velocity=joint_vel,
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
    )


def randomize_action_joint_pos_offset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    action_term_name: str,
    offset_range: tuple[float, float],
):
    """Randomize the motor zero offset of a joint-position action term."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    action_term = env.action_manager.get_term(action_term_name)
    if not hasattr(action_term, "_offset") or not isinstance(action_term._offset, torch.Tensor):
        raise TypeError(
            f"Action term '{action_term_name}' does not expose a tensor '_offset', "
            "so it cannot be used for motor zero-offset randomization."
        )

    cache_name = f"_default_action_offset_{action_term_name}"
    default_offset = getattr(env, cache_name, None)
    if default_offset is None or default_offset.shape != action_term._offset.shape:
        default_offset = action_term._offset.clone()
        setattr(env, cache_name, default_offset)

    offset_noise = sample_uniform(
        offset_range[0],
        offset_range[1],
        (len(env_ids), action_term._offset.shape[1]),
        device=env.device,
    )
    action_term._offset[env_ids] = default_offset[env_ids] + offset_noise


__all__ = ["randomize_action_joint_pos_offset", "reset_joints_by_scale"]
