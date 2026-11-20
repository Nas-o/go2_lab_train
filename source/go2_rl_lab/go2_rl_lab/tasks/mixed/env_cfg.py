"""Go2 locomotion environment configuration for mjlab."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from go2_rl_lab.assets import get_go2_robot_cfg
from go2_rl_lab.tasks.mixed import mdp
from go2_rl_lab.tasks.mixed.terrain import MIXED_TERRAIN_CFG
from go2_rl_lab.utils.terrains import configure_play_sub_terrains


JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
BASE_LINK_NAME = "base_link"
FOOT_SITE_NAMES = ("FL", "FR", "RL", "RR")
FOOT_GEOM_NAME = ".*_foot_collision"
BASE_HEIGHT_TARGET = 0.38
BASE_HEIGHT_SCAN_X_RANGE = (-0.2, 0.2)
BASE_HEIGHT_SCAN_Y_RANGE = (-0.15, 0.15)


def make_contact_sensors() -> tuple[ContactSensorCfg, ...]:
    """Create the foot, non-foot, and base contact sensors used by the MDP."""
    feet_ground_cfg = ContactSensorCfg(
        name="contact_forces",
        primary=ContactMatch(mode="geom", pattern=(FOOT_GEOM_NAME,), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=3,
        track_air_time=True,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_contact_forces",
        primary=ContactMatch(
            mode="body",
            pattern=r".*(thigh|calf)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=3,
    )
    base_ground_cfg = ContactSensorCfg(
        name="base_contact_forces",
        primary=ContactMatch(mode="body", pattern=BASE_LINK_NAME, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=3,
    )
    return (feet_ground_cfg, nonfoot_ground_cfg, base_ground_cfg)


def make_terrain_scan() -> RayCastSensorCfg:
    """Create the terrain-only ray cast sensor used for height observations."""
    return RayCastSensorCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name=BASE_LINK_NAME, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=100.0,
        exclude_parent_body=True,
        debug_vis=False,
    )


def make_scene(num_envs: int, terrain_type: str = "generator", env_spacing: float = 0.5) -> SceneCfg:
    """Create the terrain scene with the Go2 robot and its sensors."""
    if terrain_type == "generator":
        terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=MIXED_TERRAIN_CFG,
            max_init_terrain_level=5,
            env_spacing=env_spacing,
        )
    else:
        terrain = TerrainEntityCfg(terrain_type="plane", env_spacing=env_spacing)
    return SceneCfg(
        terrain=terrain,
        entities={"robot": get_go2_robot_cfg()},
        sensors=(make_terrain_scan(), *make_contact_sensors()),
        num_envs=num_envs,
        env_spacing=env_spacing,
    )


def make_observations() -> dict[str, ObservationGroupCfg]:
    """Create the policy, critic, and deployment observation groups."""
    joint_pos_params = {"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
    joint_vel_params = {"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}

    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(0.5, 1.0, 1.0),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params=joint_pos_params,
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-100.0, 100.0),
            scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params=joint_vel_params,
            noise=Unoise(n_min=-2.0, n_max=2.0),
            clip=(-100.0, 100.0),
            scale=0.05,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        ),
    }

    critic_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=(0.5, 1.0, 1.0),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=0.25,
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        ),
        "velocity_commands": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(0.5, 1.0, 1.0),
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params=joint_pos_params,
            clip=(-100.0, 100.0),
            scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params=joint_vel_params,
            clip=(-100.0, 100.0),
            scale=0.05,
        ),
        "actions": ObservationTermCfg(
            func=envs_mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        ),
        "joint_acc": ObservationTermCfg(
            func=mdp.joint_acc,
            params=joint_pos_params,
            clip=(-100.0, 100.0),
            scale=1.0e-4,
        ),
        "joint_torque": ObservationTermCfg(
            func=mdp.joint_effort,
            params=joint_pos_params,
            clip=(-100.0, 100.0),
            scale=0.01,
        ),
        "contact_force": ObservationTermCfg(
            func=mdp.foot_contact_force_norm,
            params={"sensor_name": "contact_forces"},
            clip=(-100.0, 100.0),
            scale=1.0e-3,
        ),
        "height_scan": ObservationTermCfg(
            func=mdp.ray_caster_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
            scale=2.5,
        ),
    }

    def make_group(terms: dict[str, ObservationTermCfg], history_length: int) -> ObservationGroupCfg:
        return ObservationGroupCfg(
            terms=terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=history_length,
            flatten_history_dim=True,
        )

    policy_terms = dict(policy_terms)
    single_terms = dict(policy_terms)
    return {
        "policy": make_group(policy_terms, history_length=10),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "single_obs": make_group(single_terms, history_length=1),
    }


def make_actions() -> dict[str, JointPositionActionCfg]:
    """Create the joint-position action term."""
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale={
                ".*_hip_joint": 0.25,
                "^(?!.*_hip_joint).*": 0.25,
            },
            use_default_offset=True,
            preserve_order=True,
        ),
    }


def make_commands(play: bool = False) -> dict:
    """Create the velocity command term."""
    if play:
        return {
            "base_velocity": mdp.RLGymVelCommandCfg(
                entity_name="robot",
                ranges=mdp.RLGymVelCommandCfg.Ranges(
                    lin_vel_x=(3.0, 3.0),
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_yaw=(0.0, 0.0),
                ),
                dynamic_resample_commands=False,
                command_range_curriculum=[],
                # The hold combo (1, 0, 0) requires a zero direction code for lin_vel_y.
                limit_vel_curriculum=[
                    {
                        "iter": 0,
                        "lin_vel_x": [-1, 1],
                        "lin_vel_y": [-1, 0, 1],
                        "ang_vel_yaw": [-1, 0, 1],
                    },
                ],
                limit_vel_prob=0.0,
                limit_vel_hold_prob=1.0,
                limit_vel_hold_combo=(1, 0, 0),
                zero_command_curriculum=None,
                resampling_time=30.0,
                debug_vis=True,
            )
        }
    return {
        "base_velocity": mdp.RLGymVelCommandCfg(
            entity_name="robot",
            limit_vel_curriculum=[
                {
                    "iter": 0,
                    "lin_vel_x": [-1, 1],
                    "lin_vel_y": [-1, 1],
                    "ang_vel_yaw": [-1, 0, 1],
                },
                {
                    "iter": 25000,
                    "lin_vel_x": [-1, 1],
                    "lin_vel_y": [0],
                    "ang_vel_yaw": [-1, 0, 1],
                },
            ],
        )
    }


def make_events(play: bool = False) -> dict[str, EventTermCfg]:
    """Create the domain-randomization and reset events."""
    events = {
        # Keep the source's absolute mass distribution (add U(-1, 1) kg on the
        # base). mjlab's body_mass leaves the inertia tensor unchanged; the
        # source recomputed it from mass, so a uniform-density variant is
        # available through dr.pseudo_inertia if that coupling is preferred.
        "randomize_rigid_body_mass_base": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
                "ranges": (-1.0, 1.0),
                "operation": "add",
            },
        ),
        "randomize_rigid_body_mass_others": EventTermCfg(
            func=dr.body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=r"^(?!.*base).*"
                ),
                "ranges": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "randomize_com_positions": EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
                "operation": "add",
                "ranges": {
                    0: (-0.03, 0.03),
                    1: (-0.03, 0.03),
                    2: (-0.03, 0.03),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        ),
        "randomize_actuator_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.9, 1.1),
                "kd_range": (0.9, 1.1),
                "operation": "scale",
            },
        ),
        "randomize_motor_zero_offset": EventTermCfg(
            func=mdp.randomize_action_joint_pos_offset,
            mode="reset",
            params={
                "action_term_name": "joint_pos",
                "offset_range": (-0.035, 0.035),
            },
        ),
        "randomize_push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 4.0),
            params={
                "velocity_range": {
                    "x": (-0.4, 0.4),
                    "y": (-0.4, 0.4),
                    "roll": (-0.6, 0.6),
                    "pitch": (-0.6, 0.6),
                    "yaw": (-0.6, 0.6),
                }
            },
        ),
        "randomize_rigid_body_material": EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_collision",)),
                "ranges": (0.0, 2.0),
                "operation": "abs",
            },
        ),
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.2),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
    }
    return events


def make_rewards() -> dict[str, RewardTermCfg]:
    """Create the locomotion reward terms."""
    return {
        # Target-tracking rewards
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=mdp.track_lin_vel_xy_exp,
            weight=2.0,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=mdp.track_ang_vel_z_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
        # Regularization rewards
        "lin_vel_z_l2": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
        "joint_acc_l2": RewardTermCfg(
            func=mdp.clipped_joint_acc_l2,
            weight=-3.0e-7,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
                "max_penalty": 1.0e8,
            },
        ),
        "joint_power": RewardTermCfg(
            func=mdp.joint_power,
            weight=-1.5e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        ),
        "joint_torques_l2": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-6e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        ),
        "base_height_l2": RewardTermCfg(
            func=mdp.base_height_l2,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
                "target_height": BASE_HEIGHT_TARGET,
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "x_range": BASE_HEIGHT_SCAN_X_RANGE,
                "y_range": BASE_HEIGHT_SCAN_Y_RANGE,
            },
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.clipped_action_rate_l2,
            weight=-0.01,
            params={"max_penalty": 20.0},
        ),
        "action_smoothness_l2": RewardTermCfg(
            func=mdp.clipped_action_smoothness_l2,
            weight=-0.01,
            params={"max_penalty": 50.0},
        ),
        "undesired_contacts": RewardTermCfg(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_name": "nonfoot_contact_forces",
                "threshold": 110.0,
            },
        ),
        "joint_pos_limits": RewardTermCfg(
            func=envs_mdp.joint_pos_limits,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
        ),
        "feet_regulation": RewardTermCfg(
            func=mdp.feet_regulation,
            weight=-0.05,
            params={
                "base_height_target": BASE_HEIGHT_TARGET,
                "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "x_range": BASE_HEIGHT_SCAN_X_RANGE,
                "y_range": BASE_HEIGHT_SCAN_Y_RANGE,
            },
        ),
        "hip_pos_penalty_l1": RewardTermCfg(
            func=mdp.joint_pos_penalty_l1,
            weight=-0.05,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
                "stand_still_scale": 10.0,
                "stand_cmd_idxs": [1, 2],
            },
        ),
        "joint_pos_penalty_l1": RewardTermCfg(
            func=mdp.joint_pos_penalty_l1,
            weight=-0.01,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=".*_(thigh|calf)_joint"
                ),
                "stand_still_scale": 10.0,
            },
        ),
    }


def make_terminations() -> dict[str, TerminationTermCfg]:
    """Create the episode termination terms."""
    return {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={
                "sensor_name": "base_contact_forces",
                "threshold": 1.0,
            },
        ),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "limit_angle": 2.0,
            },
        ),
    }


def make_curriculum() -> dict[str, CurriculumTermCfg]:
    """Create the terrain and reward-weight curriculum terms."""
    return {
        "terrain_levels": CurriculumTermCfg(func=mdp.TerrainLevelsVelGym),
        "base_linear_velocity_z": CurriculumTermCfg(
            func=mdp.gradual_reward_weight_modification,
            params={
                "term_name": "lin_vel_z_l2",
                "initial_weight": -2.0,
                "final_weight": 0.0,
                "start_it": 0,
                "end_it": 1500,
            },
        ),
        "base_height": CurriculumTermCfg(
            func=mdp.gradual_reward_weight_modification,
            params={
                "term_name": "base_height_l2",
                "initial_weight": -1.0,
                "final_weight": -10.0,
                "start_it": 0,
                "end_it": 5000,
            },
        ),
    }


def _base_env_cfg(num_envs: int, terrain_type: str, play: bool) -> ManagerBasedRlEnvCfg:
    """Assemble the shared Go2 velocity environment configuration."""
    return ManagerBasedRlEnvCfg(
        scene=make_scene(num_envs, terrain_type=terrain_type),
        observations=make_observations(),
        actions=make_actions(),
        commands=make_commands(play=play),
        rewards=make_rewards(),
        terminations=make_terminations(),
        events=make_events(play=play),
        curriculum={} if play else make_curriculum(),
        sim=SimulationCfg(
            nconmax=128,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                integrator="implicitfast",
                cone="pyramidal",
                impratio=1.0,
                iterations=100,
                ls_iterations=50,
            ),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=BASE_LINK_NAME,
            distance=1.5,
            elevation=-10.0,
        ),
        decimation=4,
        episode_length_s=25.0,
        # The reward weights were tuned without an explicit dt factor.
        scale_rewards_by_dt=False,
    )


def go2_mixed_env_cfg(
    play: bool = False,
    play_terrain_proportions: dict[str, float] | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create the Go2 mixed-terrain velocity environment."""
    cfg = _base_env_cfg(num_envs=32 if play else 16384, terrain_type="generator", play=play)
    if play:
        cfg.episode_length_s = 25.0
        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            # Isaac Lab assigns difficulty across rows and terrain types across columns.
            # Playback assigns robots deterministically from the hardest row to the easiest.
            terrain_generator = cfg.scene.terrain.terrain_generator
            terrain_generator.num_rows = 10
            configure_play_sub_terrains(terrain_generator, play_terrain_proportions)
            cfg.scene.terrain.max_init_terrain_level = terrain_generator.num_rows - 1
    return cfg


def go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the flat-plane Go2 velocity environment."""
    cfg = _base_env_cfg(num_envs=32 if play else 16384, terrain_type="plane", play=play)
    if play:
        cfg.commands["base_velocity"] = UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(30.0, 30.0),
            rel_standing_envs=0.0,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(1.0, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
            debug_vis=True,
        )
    return cfg


__all__ = [
    "BASE_LINK_NAME",
    "FOOT_GEOM_NAME",
    "FOOT_SITE_NAMES",
    "JOINT_NAMES",
    "go2_flat_env_cfg",
    "go2_mixed_env_cfg",
    "make_actions",
    "make_commands",
    "make_contact_sensors",
    "make_curriculum",
    "make_events",
    "make_observations",
    "make_rewards",
    "make_terminations",
]
