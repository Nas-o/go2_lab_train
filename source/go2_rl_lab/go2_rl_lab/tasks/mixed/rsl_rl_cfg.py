"""RSL-RL PPO configuration for the Go2 locomotion task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from go2_rl_lab.moe_cts import (
    MoeCtsSymmetryCfg,
    RslRlMoeCtsActorCriticCfg,
    RslRlMoeCtsAlgorithmCfg,
    RslRlMoeCtsRunnerCfg,
)


def go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create the PPO runner configuration shared by the Go2 tasks."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=100000,
        save_interval=500,
        experiment_name="",  # same as task name
        logger="tensorboard",
        obs_groups={"actor": ("policy",), "critic": ("critic",)},
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )




def go2_moe_cts_runner_cfg() -> RslRlMoeCtsRunnerCfg:
    """Create the MoE-CTS teacher/student runner configuration."""
    return RslRlMoeCtsRunnerCfg(
        num_steps_per_env=24,
        max_iterations=10000,
        save_interval=500,
        experiment_name="",  # same as task name
        logger="tensorboard",
        obs_groups={"policy": ("policy",), "critic": ("critic",)},
        policy=RslRlMoeCtsActorCriticCfg(
            init_noise_std=1.0,
            noise_std_type="scalar",
            actor_hidden_dims=(512, 256, 128),
            critic_hidden_dims=(512, 256, 128),
            teacher_encoder_hidden_dims=(512, 256),
            student_encoder_hidden_dims=(512, 256, 256),
            expert_num=8,
            latent_dim=32,
            norm_type="l2norm",
            activation="elu",
        ),
        algorithm=RslRlMoeCtsAlgorithmCfg(
            value_loss_coef=1.0,
            load_balance_coef=0.01,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            student_encoder_learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            desired_kl=0.01,
            max_grad_norm=1.0,
            teacher_env_ratio=0.75,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=MoeCtsSymmetryCfg(),
        ),
    )


__all__ = ["go2_moe_cts_runner_cfg", "go2_ppo_runner_cfg"]
