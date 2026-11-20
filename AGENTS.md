# Repository Guide

Go2 RL Lab trains and evaluates Unitree Go2 velocity locomotion policies on **mjlab 1.2.0**
(MuJoCo Warp) with `rsl-rl-lib` 5.0.1. The project was ported from an Isaac Lab 3.0 EA
implementation; observation layouts, reward formulas, command sampling, and deployment
interfaces are preserved. Run commands from the repository root.

## Local validation files

- Treat the entire `tests/` directory as local-only, including Python scripts, Markdown reports, logs, images, videos, converted assets, and checkpoints.
- Never stage, commit, force-add, push, upload, or attach any file under `tests/` to a remote service, pull request, or release.
- Place new local validation scripts and reports under `tests/`; do not move them elsewhere to bypass this policy.
- Keep `/tests/` ignored without exceptions. `.gitignore` does not exclude files already tracked: remove such files from the Git index while preserving their local copies.
- Before a commit or push, verify that `git ls-files tests` is empty and that no added or modified `tests/` file appears in the index. Staged deletions of previously tracked test files are allowed to enforce this policy.
- Keep `logs/`, exported policies (`policy.pt`, `policy.onnx`, `config.yaml`), and training checkpoints out of commits.
- Keep the bundled Go2 MJCF meshes under `source/go2_rl_lab/go2_rl_lab/assets/xmls/assets/` and their Git LFS configuration.

## Framework and repository layout

| Path | Responsibility |
| --- | --- |
| `go2_rl_lab.sh` | Project CLI for editable installation, task discovery, training, playback, and MuJoCo deployment |
| `scripts/rsl_rl/` | mjlab training launcher, playback/export entry point, and TorchScript + metadata export |
| `scripts/utils/` | Task listing and optional dependency checks |
| `scripts/tools/` | GPU utilization tool |
| `source/go2_rl_lab/go2_rl_lab/tasks/` | mjlab task configs, MDP terms, terrain generation, and task registration |
| `source/go2_rl_lab/go2_rl_lab/assets/` | Go2 MJCF, `EntityCfg`, and the custom Go2 motor actuator |
| `source/go2_rl_lab/go2_rl_lab/cmd_utils/` | Go2 Gym body-frame velocity command and curricula |
| `source/go2_rl_lab/go2_rl_lab/utils/` | Terrain lookup helpers and finite height-scan queries |
| `source/go2_rl_lab/go2_rl_lab/moe_cts/` | MoE-CTS algorithm, teacher/student networks, symmetry augmentation, runner, and exporter |
| `deploy/deploy_mujoco/` | Standalone MuJoCo inference and simulation-to-simulation deployment |
| `docs/` | Project documentation and upstream license notices |

mjlab owns simulation, assets, sensors, scene assembly, and manager-based environments. Tasks are
registered through `register_mjlab_task` and exported through the `mjlab.tasks` entry point declared
in `source/go2_rl_lab/pyproject.toml`. Every manager config is a dict of term configs
(`dict[str, XxxTermCfg]`); keep that style.

### Task families and training pipeline

- `go2-flat`: flat-plane PPO baseline with fixed velocity ranges and the mixed-task reward set.
- `go2-mixed`: mixed-terrain PPO with the RobotLab terrain generator, velocity/limit-command
  curricula, terrain-level curriculum, and reward-weight curricula.
- `go2-mixed-moe-cts`: mixed-terrain MoE-CTS teacher/student training with a mixture-of-experts
  student encoder, latent distillation, and Go2 symmetry augmentation; runner and algorithms live
  in `go2_rl_lab/moe_cts` on top of `rsl-rl-lib` 5.0.1.

All tasks use body-frame commands `[v_x, v_y, yaw_rate]` in `m/s`, `m/s`, and `rad/s`. Policy
observations contain a 45-dimensional frame and ten-frame history; critic observations and the MoE
teacher include privileged state, contact forces, and terrain heights. `deploy/deploy_mujoco`
receives raw velocity commands, and exported TorchScript policies maintain observation history
internally. Preserve observation layouts, command semantics, checkpoint formats, task-specific
rewards, and the teacher/student split when updating compatibility code.

Discover the registered task set with `./go2_rl_lab.sh --list`.

## Runtime and installation

The validated runtime is Python 3.13 with `mjlab==1.2.0`, `mujoco==3.5.0`,
`mujoco-warp==3.5.0`, `warp-lang==1.12.0`, `rsl-rl-lib==5.0.1`, and PyTorch 2.13 (CUDA).
`mujoco` and `mujoco-warp` versions must match: newer `mujoco` releases removed
`mjENBL_MULTICCD`, which `mujoco-warp` 3.5.0 requires. `warp-lang` must still expose
`warp.context` (mjlab 1.2.0 reads `wp.context.runtime.driver_version`).

```bash
# Install the task package (registers the mjlab.tasks entry point)
./go2_rl_lab.sh --install

# List current tasks
./go2_rl_lab.sh --list
```

Run `--install` inside every Python environment that will train or play. mjlab discovers tasks
through the `mjlab.tasks` entry point, so an environment without the editable install fails with
`KeyError: '<task>'` even though `mjlab` itself is importable.

## mjlab contracts

- Register tasks with `register_mjlab_task(task_id=..., env_cfg=..., play_env_cfg=..., rl_cfg=..., runner_cls=...)`; `mjlab` imports the entry point at startup.
- Manager configs are plain dicts; `ObservationGroupCfg` groups hold `terms` dicts. Do not reintroduce `@configclass` manager trees.
- `EnvCfg.scale_rewards_by_dt` defaults to `True`; these tasks set it to `False` because the reward weights were tuned without a `dt` factor.
- Episode time limits require an explicit `time_out` termination with `time_out=True`.
- Scenes are `SceneCfg(terrain=..., entities={"robot": ...}, sensors=(...))`; there is no `prim_path`, USD, or scene cloning.
- Entities expose `root_link_pos_w`, `root_link_lin_vel_b`, `joint_acc`, `actuator_force`, `default_joint_pos`, and `write_*_to_sim` helpers.
- Contact sensors are pattern based (`ContactMatch`, `reduce`, `num_slots`, `history_length`) and expose `data.force_history` shaped `[num_envs, num_primaries, history, 3]`.
- `RayCastSensorCfg` height scans return `data.hit_pos_w`/`data.distances`; ray misses report `distances < 0`.
- The Go2 motor model is a project actuator (`Go2ActuatorCfg`) that applies the X1/X2/Y1/Y2 torque-speed envelope and wraps a `DelayedActuatorCfg` for 0-4 physics-step command delay; keep the Python actuator path instead of MuJoCo builtin position actuators.
- Terrain sub-terrains implement `SubTerrainCfg.function(difficulty, spec, rng)` and add MuJoCo height fields or mesh geoms directly.
- Playback configs are factory arguments (`play=True`), not separate registrations.

## Development and validation

Inspect the working tree before editing. Use four-space Python indentation and concise docstrings.
Keep `README.md` and the existing Chinese `README.zh-CN.md` synchronized when changing documented
behavior or commands. Do not create commits or publish changes unless the user requests it.

```bash
# Static validation
git diff --check
bash -n go2_rl_lab.sh
python -m compileall -q scripts source/go2_rl_lab

# Focused training smoke
./go2_rl_lab.sh --train --task go2-flat --num_envs 16 --max_iterations 2 --logger tensorboard
```

A passing train smoke requires actual rollout and optimizer updates, finite losses, changed
trainable weights, saved checkpoints, and exit code 0. Validate both `go2-flat` and `go2-mixed`
(terrain generation is the slow part of mixed-task startup). Validate export and deployment when
those paths change: load `policy.pt` for finite inference with a 45-dimensional input, check
`policy.onnx`, and confirm that `config.yaml` reports `command/body/velocity` and the 45/450/12
observation/action sizes. State any untested interactive visualization, large-scale, or multi-GPU
paths explicitly in the local report.
