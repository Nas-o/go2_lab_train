"""mjlab entity configuration for the Unitree Go2."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from go2_rl_lab.assets.go2_actuator import Go2ActuatorCfg

##
# MJCF and assets.
##

GO2_XML_PATH: Path = Path(__file__).resolve().parent / "xmls" / "go2.xml"
"""Path to the Go2 MJCF asset shipped with the task package."""


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, GO2_XML_PATH.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML_PATH))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

# Go2 motor curves, delays and friction require the Python actuator path instead
# of a MuJoCo builtin position actuator.
GO2_ACTUATOR_CFG = Go2ActuatorCfg(
    target_names_expr=(".*",),
    stiffness=25.0,
    damping=0.5,
    frictionloss=0.01,
    armature=0.01,
    X1=13.5,
    X2=30.0,
    Y1=20.2,
    Y2=23.4,
    Fs=0.0,
    Fd=0.0,
    Va=0.01,
)

GO2_DELAYED_ACTUATOR_CFG = DelayedActuatorCfg(
    base_cfg=GO2_ACTUATOR_CFG,
    delay_target="position",
    delay_min_lag=0,
    delay_max_lag=4,
)

##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.4),
    joint_pos={
        ".*R_hip_joint": -0.1,
        ".*L_hip_joint": 0.1,
        "F[L,R]_thigh_joint": 0.8,
        "R[L,R]_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# Keep articulation self-collisions enabled and collide every collision geom
# with the terrain, matching the Isaac Lab articulation properties.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    contype=1,
    conaffinity=1,
    condim=3,
    friction=(0.6,),
    solimp=(0.9, 0.95, 0.023),
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(GO2_DELAYED_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
    """Get a fresh Go2 robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO2_ARTICULATION,
    )


__all__ = ["GO2_XML_PATH", "get_assets", "get_go2_robot_cfg", "get_spec"]
