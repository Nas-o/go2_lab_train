# Copyright (c) 2026 Tianyang Wu
#
# SPDX-License-Identifier: MIT

"""Export Go2 policies and deployment metadata for MuJoCo playback.

The exported TorchScript module owns the observation history: callers pass one
observation frame per control step and receive one deterministic action. That is
the interface ``deploy/deploy_mujoco`` expects.
"""

from __future__ import annotations

import copy
import hashlib
import os
from datetime import datetime
from typing import Any

import torch
import yaml


class TorchPolicyHistoryExporter(torch.nn.Module):
    """Export an actor with internal observation history."""

    def __init__(
        self,
        policy: object,
        actor_obs_normalizer: object | None,
        num_single_obs: int,
        num_actions: int,
    ) -> None:
        super().__init__()

        if not hasattr(policy, "mlp"):
            raise ValueError("RSL-RL policy does not have an MLP actor module.")

        self.actor = copy.deepcopy(policy.mlp)
        self.actor_obs_normalizer = (
            copy.deepcopy(actor_obs_normalizer) if actor_obs_normalizer is not None else torch.nn.Identity()
        )
        self.num_single_obs = int(num_single_obs)
        self.num_actor_obs = int(self.actor[0].in_features)

        if self.num_actor_obs % self.num_single_obs != 0:
            raise ValueError(
                f"num_actor_obs ({self.num_actor_obs}) must be divisible by "
                f"num_single_obs ({self.num_single_obs})."
            )

        self.history_len = self.num_actor_obs // self.num_single_obs

        # Velocity policies store [vx, vy, yaw_rate] as the third vector block:
        # [base_ang_vel, projected_gravity, velocity_command, joints...].
        num_actions = int(num_actions)
        expected_single_obs = 3 + 3 + 3 + 3 * num_actions
        if self.num_single_obs != expected_single_obs:
            raise ValueError(
                "Unsupported single_obs layout: expected "
                f"3+3+3+3*num_actions={expected_single_obs}, "
                f"got {self.num_single_obs}."
            )
        self.feature_dims = [3, 3, 3, num_actions, num_actions, num_actions]

        self.register_buffer("obs_history", torch.zeros(1, self.num_actor_obs, dtype=torch.float32))

    def forward(self, single_obs: torch.Tensor) -> torch.Tensor:
        """Append one observation frame and return the deterministic action."""
        if single_obs.dim() == 1:
            single_obs = single_obs.unsqueeze(0)
        if single_obs.shape[-1] != self.num_single_obs:
            raise ValueError(f"Expected single_obs last dimension {self.num_single_obs}, got {single_obs.shape[-1]}.")
        if single_obs.shape[0] != 1:
            raise ValueError("TorchScript PPO deployment supports batch size 1 only.")

        next_history = self.obs_history.clone()
        history_offset = 0
        single_offset = 0
        for dim in self.feature_dims:
            block_size = dim * self.history_len
            block_end = history_offset + block_size
            single_end = single_offset + dim
            block = self.obs_history[:, history_offset:block_end]
            shifted_block = torch.cat([block[:, dim:], single_obs[:, single_offset:single_end]], dim=-1)
            next_history[:, history_offset:block_end] = shifted_block
            history_offset = block_end
            single_offset = single_end
        self.obs_history.copy_(next_history)

        return self.actor(self.actor_obs_normalizer(self.obs_history))

    @torch.jit.export
    def reset(self) -> None:
        """Clear the internal observation history."""
        self.obs_history.zero_()

    def export(self, path: str, filename: str) -> None:
        """Write the stateful deployment module to disk."""
        os.makedirs(path, exist_ok=True)
        self.eval()
        self.to("cpu")
        torch.jit.script(self).save(os.path.join(path, filename))


def export_policy_as_jit(
    policy: object,
    path: str,
    num_single_obs: int,
    num_actions: int,
    filename: str = "policy.pt",
) -> str:
    """Export the runner policy as a history-aware TorchScript module."""
    normalizer = getattr(policy, "obs_normalizer", None)
    exporter = TorchPolicyHistoryExporter(
        policy=policy,
        actor_obs_normalizer=normalizer,
        num_single_obs=num_single_obs,
        num_actions=num_actions,
    )
    exporter.export(path, filename)
    return os.path.join(path, filename)


def export_model_config(
    path: str,
    checkpoint_path: str,
    checkpoint_iteration: int,
    task: str,
    experiment_name: str,
    *,
    num_single_obs: int,
    num_actor_obs: int,
    num_actions: int,
    control_dt: float,
    filename: str = "config.yaml",
) -> str:
    """Write deployment metadata and the original training configs to YAML."""
    os.makedirs(path, exist_ok=True)
    checkpoint_path = os.path.abspath(checkpoint_path)
    run_dir = os.path.dirname(checkpoint_path)
    run_directory = os.path.basename(run_dir)
    params_dir = os.path.join(run_dir, "params")
    agent_params = _load_yaml_metadata(os.path.join(params_dir, "agent.yaml"))
    env_params = _load_yaml_metadata(os.path.join(params_dir, "env.yaml"))

    num_envs = _as_int(_nested_value(env_params, "scene", "num_envs"))
    num_steps_per_env = _as_int(_nested_value(agent_params, "num_steps_per_env"))
    max_iterations = _as_int(_nested_value(agent_params, "max_iterations"))
    seed = _as_int(_nested_value(agent_params, "seed"))

    completed_learning_iterations = max(int(checkpoint_iteration) + 1, 0)
    environment_steps_per_iteration = None
    estimated_environment_steps = None
    if num_envs is not None and num_steps_per_env is not None:
        environment_steps_per_iteration = num_envs * num_steps_per_env
        estimated_environment_steps = completed_learning_iterations * environment_steps_per_iteration

    history_length = None
    if num_single_obs > 0 and num_actor_obs % num_single_obs == 0:
        history_length = num_actor_obs // num_single_obs

    command_cfg = _nested_value(env_params, "commands", "base_velocity")
    deployment_model = {
        "onnx_file": "policy.onnx",
        "torchscript_file": "policy.pt",
        "single_observation_size": int(num_single_obs),
        "actor_observation_size": int(num_actor_obs),
        "observation_history_length": history_length,
        "action_size": int(num_actions),
        "control_dt": float(control_dt),
        "command_kind": _nested_value(command_cfg or {}, "cmd_kind") or "command/body/velocity",
        "command_elements": _nested_value(command_cfg or {}, "element_names")
        or ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
    }
    command_scale = _nested_value(
        env_params, "observations", "single_obs", "terms", "velocity_commands", "scale"
    )
    if command_scale is None:
        command_scale = _nested_value(
            env_params, "observations", "policy", "terms", "velocity_commands", "scale"
        )
    if command_scale is not None:
        deployment_model["command_observation_scale"] = command_scale

    metadata = {
        "metadata": {
            "format_version": 1,
            "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "task": task,
            "experiment_name": _nested_value(agent_params, "experiment_name") or experiment_name,
            "run_directory": run_directory,
            "configured_run_name": _nested_value(agent_params, "run_name") or None,
        },
        "checkpoint": {
            "file": os.path.relpath(checkpoint_path, path),
            "sha256": _sha256(checkpoint_path),
            "iteration": int(checkpoint_iteration),
        },
        "training_progress": {
            "checkpoint_iteration": int(checkpoint_iteration),
            "completed_learning_iterations": completed_learning_iterations,
            "num_environments": num_envs,
            "rollout_steps_per_environment": num_steps_per_env,
            "environment_steps_per_iteration": environment_steps_per_iteration,
            "estimated_environment_steps": estimated_environment_steps,
            "configured_max_iterations": max_iterations,
        },
        "deployment_model": deployment_model,
        "training_summary": {
            "seed": seed,
            "physics_backend": "mujoco_warp",
        },
        "training_parameters": {
            "agent": agent_params or None,
            "environment": env_params or None,
        },
    }

    config_path = os.path.join(path, filename)
    temporary_path = f"{config_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(metadata, config_file, sort_keys=False, allow_unicode=True)
    os.replace(temporary_path, config_path)
    return config_path


class _TrainingConfigLoader(yaml.SafeLoader):
    """Safely normalize Python-specific tags emitted by the training YAML dumper."""


def _construct_python_tuple(loader: _TrainingConfigLoader, node: yaml.Node) -> list[Any]:
    return loader.construct_sequence(node, deep=True)


def _construct_python_slice(loader: _TrainingConfigLoader, node: yaml.Node) -> dict[str, Any]:
    values = loader.construct_sequence(node, deep=True)
    values.extend([None] * (3 - len(values)))
    return {"start": values[0], "stop": values[1], "step": values[2]}


def _construct_python_object(loader: _TrainingConfigLoader, suffix: str, node: yaml.Node) -> str:
    """Represent arbitrary Python objects (for example enums) by their qualified name."""
    del loader, node
    return suffix


_TrainingConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
_TrainingConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_python_slice
)
for _python_tag in (
    "python/object/apply:",
    "python/object:",
    "python/object/new:",
    "python/name:",
):
    _TrainingConfigLoader.add_multi_constructor(
        f"tag:yaml.org,2002:{_python_tag}", _construct_python_object
    )


def _load_yaml_metadata(path: str) -> dict[str, Any]:
    """Load a dumped mjlab training config without constructing arbitrary objects."""
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as config_file:
        data = yaml.load(config_file, Loader=_TrainingConfigLoader)
    return data if isinstance(data, dict) else {}


def _nested_value(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _sha256(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["TorchPolicyHistoryExporter", "export_model_config", "export_policy_as_jit"]
