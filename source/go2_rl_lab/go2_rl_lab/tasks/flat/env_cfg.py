"""Flat-ground velocity locomotion using the mixed-task PPO baseline."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from go2_rl_lab.tasks.mixed import mdp
from go2_rl_lab.tasks.mixed.env_cfg import go2_flat_env_cfg


def go2_flat_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-plane PPO environment with fixed velocity commands."""
    cfg = go2_flat_env_cfg(play=play)
    if play:
        return cfg

    # Keep the mixed-task sampler and limit-command behavior, but remove all
    # command-range and zero-command schedules for a controlled ablation.
    cfg.commands["base_velocity"] = mdp.RLGymVelCommandCfg(
        entity_name="robot",
        zero_command_curriculum=None,
        command_range_curriculum=[],
        terrain_max_command_ranges={},
        ranges=mdp.RLGymVelCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_yaw=(-2.0, 2.0),
        ),
    )
    # A plane has no generated terrain, so no terrain curriculum applies.
    cfg.curriculum.pop("terrain_levels", None)
    return cfg


__all__ = ["go2_flat_velocity_env_cfg"]
