"""Privileged proprioceptive observations for velocity locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_acc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return joint accelerations for the selected joints."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def joint_effort(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return applied actuator efforts for the selected joints."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.joint_ids]


def foot_contact_force_norm(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Return flattened current and historical foot-contact force norms."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    contact_forces = contact_sensor.data.force_history
    if contact_forces is None:
        raise ValueError(
            f"Contact sensor '{sensor_name}' must be configured with history_length > 0."
        )
    contact_force_norm = torch.norm(contact_forces, dim=-1)
    contact_force_norm = contact_force_norm.permute(0, 2, 1)
    max_contact_force_norm = torch.max(contact_force_norm, dim=1).values
    contact_force_norm = torch.cat(
        [max_contact_force_norm.unsqueeze(1), contact_force_norm],
        dim=1,
    )
    return contact_force_norm.flatten(start_dim=-2)


__all__ = ["foot_contact_force_norm", "joint_acc", "joint_effort"]
