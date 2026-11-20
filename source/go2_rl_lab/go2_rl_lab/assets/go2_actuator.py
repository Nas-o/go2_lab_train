"""Go2 actuator model with motor torque-speed and friction limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import Actuator, ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


class Go2Actuator(IdealPdActuator):
    """Apply the Go2 motor torque-speed curve and joint friction."""

    cfg: Go2ActuatorCfg

    def __init__(
        self,
        cfg: Go2ActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._joint_vel: torch.Tensor | None = None

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """Compute and limit joint effort for one simulation step."""
        self._joint_vel = cmd.vel
        effort = super().compute(cmd)

        effort = effort - (
            self.cfg.Fs * torch.tanh(cmd.vel / self.cfg.Va) + self.cfg.Fd * cmd.vel
        )
        return effort

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        assert self._joint_vel is not None
        joint_vel = self._joint_vel
        same_direction = joint_vel * effort > 0
        max_effort = torch.where(
            same_direction,
            torch.full_like(joint_vel, self.cfg.Y1),
            torch.full_like(joint_vel, self.cfg.Y2),
        )
        max_effort = torch.where(
            joint_vel.abs() < self.cfg.X1,
            max_effort,
            self._compute_effort_limit(max_effort),
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort: torch.Tensor) -> torch.Tensor:
        assert self._joint_vel is not None
        slope = -max_effort / (self.cfg.X2 - self.cfg.X1)
        limit = slope * (self._joint_vel.abs() - self.cfg.X1) + max_effort
        return limit.clip(min=0.0)


@dataclass(kw_only=True)
class Go2ActuatorCfg(IdealPdActuatorCfg):
    """Configuration for the Unitree Go2 high-voltage motor."""

    X1: float = 13.5
    """Maximum speed at full torque, in rad/s."""

    X2: float = 30.0
    """No-load speed, in rad/s."""

    Y1: float = 20.2
    """Peak torque when torque and speed have the same direction, in N m."""

    Y2: float = 23.4
    """Peak torque when torque and speed have opposite directions, in N m."""

    Fs: float = 0.0
    """Static friction coefficient."""

    Fd: float = 0.0
    """Dynamic friction coefficient."""

    Va: float = 0.01
    """Velocity at which friction is fully activated."""

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> Go2Actuator:
        return Go2Actuator(self, entity, target_ids, target_names)


__all__ = ["Go2Actuator", "Go2ActuatorCfg"]
