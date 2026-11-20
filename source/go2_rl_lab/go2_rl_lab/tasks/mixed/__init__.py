"""Go2 mixed-terrain environment, MDP, and agent configuration."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from go2_rl_lab.moe_cts import MoeCtsRunner
from go2_rl_lab.tasks.mixed.env_cfg import go2_mixed_env_cfg
from go2_rl_lab.tasks.mixed.rsl_rl_cfg import go2_moe_cts_runner_cfg, go2_ppo_runner_cfg


register_mjlab_task(
    task_id="go2-mixed",
    env_cfg=go2_mixed_env_cfg(),
    play_env_cfg=go2_mixed_env_cfg(play=True),
    rl_cfg=go2_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)

register_mjlab_task(
    task_id="go2-mixed-moe-cts",
    env_cfg=go2_mixed_env_cfg(),
    play_env_cfg=go2_mixed_env_cfg(play=True),
    rl_cfg=go2_moe_cts_runner_cfg(),
    runner_cls=MoeCtsRunner,
)
