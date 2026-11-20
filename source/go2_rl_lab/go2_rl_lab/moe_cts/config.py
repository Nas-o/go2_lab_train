# Copyright (c) 2026 Tianyang Wu
# SPDX-License-Identifier: MIT

"""Configuration schema for the MoE-CTS algorithm package."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class MoeCtsSymmetryCfg:
    """Optional symmetry augmentation for segmented teacher/student batches."""

    use_symmetric_augmentation: bool = True
    symmetry_class: str = "go2_rl_lab.moe_cts.symmetry:Go2Symmetry"


@dataclass
class RslRlMoeCtsActorCriticCfg:
    """Teacher/student policy-network configuration."""

    # Required fields first: ``dataclasses.MISSING`` is treated as "no default".
    init_noise_std: float = MISSING
    actor_hidden_dims: tuple[int, ...] = MISSING
    critic_hidden_dims: tuple[int, ...] = MISSING
    teacher_encoder_hidden_dims: tuple[int, ...] = MISSING
    student_encoder_hidden_dims: tuple[int, ...] = MISSING
    expert_num: int = MISSING
    latent_dim: int = MISSING
    activation: str = MISSING

    class_name: str = "go2_rl_lab.moe_cts.modules:ActorCriticMoECTS"
    noise_std_type: Literal["scalar", "log"] = "scalar"
    state_dependent_std: bool = False
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    norm_type: Literal["l2norm", "simnorm"] = "l2norm"


@dataclass
class RslRlMoeCtsAlgorithmCfg:
    """Concurrent teacher/student PPO and latent-distillation configuration."""

    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    learning_rate: float = MISSING
    student_encoder_learning_rate: float = MISSING
    schedule: str = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    entropy_coef: float = MISSING
    desired_kl: float = MISSING
    max_grad_norm: float = MISSING
    value_loss_coef: float = MISSING
    load_balance_coef: float = MISSING
    use_clipped_value_loss: bool = MISSING
    clip_param: float = MISSING
    teacher_env_ratio: float = MISSING

    class_name: str = "go2_rl_lab.moe_cts.algorithm:MoECTS"
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    normalize_advantage_per_mini_batch: bool = False
    symmetry_cfg: MoeCtsSymmetryCfg | None = None
    rnd_cfg: None = None
    """Unused; the shared runner queries this key to skip intrinsic-reward logging."""


@dataclass
class RslRlMoeCtsRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for MoE-CTS."""

    class_name: str = "go2_rl_lab.moe_cts.runner:MoeCtsRunner"
    policy: RslRlMoeCtsActorCriticCfg = field(default_factory=RslRlMoeCtsActorCriticCfg)
    algorithm: RslRlMoeCtsAlgorithmCfg = field(default_factory=RslRlMoeCtsAlgorithmCfg)


__all__ = [
    "MoeCtsSymmetryCfg",
    "RslRlMoeCtsActorCriticCfg",
    "RslRlMoeCtsAlgorithmCfg",
    "RslRlMoeCtsRunnerCfg",
]
