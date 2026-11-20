# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Runner specialization for MoE-CTS."""

from __future__ import annotations

import os

import torch
from rsl_rl.runners import OnPolicyRunner

from .algorithm import MoECTS
from .exporter import export_policy_as_jit as export_moe_policy_as_jit
from .exporter import export_policy_as_onnx as export_moe_policy_as_onnx


class MoeCtsRunner(OnPolicyRunner):
    """Train, checkpoint, and export a MoE-CTS policy with the rsl-rl runner."""

    alg: MoECTS

    def get_inference_policy(self, device: str | None = None):
        """Return the deterministic student inference callable."""
        self.alg.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def export_policy_as_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the stateful student policy for deployment."""
        policy = self.alg.policy
        export_moe_policy_as_jit(
            policy,
            actor_obs_normalizer=policy.actor_obs_normalizer,
            single_obs_normalizer=policy.single_obs_normalizer,
            path=path,
            filename=filename,
        )

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the stateless student policy, consuming the full observation history."""
        policy = self.alg.policy
        export_moe_policy_as_onnx(
            policy,
            path,
            actor_obs_normalizer=policy.actor_obs_normalizer,
            single_obs_normalizer=policy.single_obs_normalizer,
            filename=filename,
            verbose=verbose,
        )

    def save(self, path: str, infos=None) -> None:
        """Save policy and both optimizers, plus the environment curriculum state."""
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        if self.cfg.get("upload_model"):
            self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Load a MoE-CTS checkpoint and restore the curriculum step counter.

        ``load_cfg={"actor": True}`` restores the policy only (playback); the
        student encoder optimizer is restored together with the PPO optimizer.
        """
        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        infos = loaded_dict.get("infos")
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos


__all__ = ["MoeCtsRunner"]
