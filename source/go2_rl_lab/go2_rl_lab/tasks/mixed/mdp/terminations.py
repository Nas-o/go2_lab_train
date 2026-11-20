"""Termination terms used by the Go2 mixed-terrain task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor


def illegal_contact(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_name: str,
) -> torch.Tensor:
    """Terminate when the tracked body touches the terrain above ``threshold``."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    contact_forces = contact_sensor.data.force_history
    if contact_forces is None:
        raise ValueError(
            f"Contact sensor '{sensor_name}' must be configured with history_length > 0."
        )
    is_contact = (
        torch.max(torch.norm(contact_forces, dim=-1), dim=-1).values > threshold
    )
    return torch.any(is_contact, dim=1)


__all__ = ["illegal_contact"]
