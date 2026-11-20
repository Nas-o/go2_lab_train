"""Go2 flat-ground velocity locomotion task."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from go2_rl_lab.tasks.flat.env_cfg import go2_flat_velocity_env_cfg
from go2_rl_lab.tasks.flat.rsl_rl_cfg import go2_ppo_runner_cfg


register_mjlab_task(
    task_id="go2-flat",
    env_cfg=go2_flat_velocity_env_cfg(),
    play_env_cfg=go2_flat_velocity_env_cfg(play=True),
    rl_cfg=go2_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
