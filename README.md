<div align="center">
  <h1>Go2 RL Lab</h1>
  <p><strong>High-speed, agile velocity locomotion control for Unitree Go2 with mjlab (MuJoCo Warp)<br><a href="https://github.com/wty-yy/go2_rl_lab/tree/gym">(original go2_rl_gym branch)</a></strong></p>
  <p>
    <a href="https://github.com/mujocolab/mjlab"><img height="20" src="https://img.shields.io/badge/mjlab-1.2.0-silver.svg" alt="mjlab 1.2.0"></a>&nbsp;
    <a href="https://swanlab.cn/@wty-yy/go2-rl-lab/v1/lxqfax/overview"><img height="20" src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>&nbsp;
    <a href="LICENSE"><img height="20" src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
  <p><strong>🌎 English</strong>&nbsp;&nbsp;·&nbsp;&nbsp;<a href="README.zh-CN.md">🇨🇳 中文</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://www.roboticsproceedings.org/rss22/p156.pdf">📄 Paper [RSS 2026]</a></p>
</div>

📢 The project now runs on **[mjlab](https://github.com/mujocolab/mjlab) 1.2.0 / MuJoCo Warp**: the task layer, terrain generator, velocity-command curriculum, rewards, motor model, and training entry points were ported from the Isaac Lab 3.0 EA implementation while keeping observation, action, and reward semantics for deployment

<table width="100%">
  <thead>
    <tr>
      <th width="33%" align="center"><div align="center">IsaacSim</div></th>
      <th width="33%" align="center"><div align="center">MuJoCo</div></th>
      <th width="33%" align="center"><div align="center">Real Robot</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Kit demo" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_isaaclab_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="MuJoCo demo" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/python_mujoco_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Real robot stair demo" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_stairs_DJI_20261017144009_0111_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">3.5 m/s Forward and 4.4 m/s Backward</div></th>
      <th width="50%" align="center"><div align="center">Running (Top View)</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Hardware forward and backward walking" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_backward_1x_then_0.5x.gif" /></td>
      <td align="center"><img width="100%" alt="Hardware fast running" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/overlook_fastrun_DJI_20261017143527_0110_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">Slope</div></th>
      <th width="50%" align="center"><div align="center">Stairwell</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Hardware slope walking" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/slope_DJI_20261017142100_0108_D_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="Hardware stairwell walking" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/stairwell_DJI_20261017145051_0117_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

## 📦 Installation

The project runs on **mjlab 1.2.0** (MuJoCo Warp) with `rsl-rl-lib` 5.0.1. Python 3.12+ and an NVIDIA GPU are required.

```bash
# Create an environment with the mjlab runtime (MuJoCo Warp + Warp + PyTorch)
conda create -n go2_mjlab python=3.12
conda activate go2_mjlab
pip install "mjlab==1.2.0" "mujoco-warp==3.5.0" "mujoco==3.5.0"

# Install the task package in editable mode, list the registered tasks
./go2_rl_lab.sh --install
./go2_rl_lab.sh --list
```

`./go2_rl_lab.sh --install` installs `source/go2_rl_lab` in editable mode and registers the
`mjlab.tasks` entry point, so `mjlab` discovers `go2-flat`, `go2-mixed`, and `go2-mixed-moe-cts`
automatically.

## 🛠️ Command Reference

### 0. Environment setup (once per Python environment)

```bash
conda activate <your-env>          # any env with mjlab 1.2.0 installed
./go2_rl_lab.sh --install          # editable install + mjlab.tasks entry point
./go2_rl_lab.sh --list             # go2-flat / go2-mixed / go2-mixed-moe-cts
```

> mjlab discovers tasks through the `mjlab.tasks` entry point. After creating or switching a
> Python environment, run `--install` again in that environment, otherwise
> `load_env_cfg()` raises `KeyError: '<task>'`.

### 1. Train

| Task ID | Terrain | Algorithm |
| --- | --- | --- |
| `go2-flat` | Plane | PPO |
| `go2-mixed` | Mixed generator (10 sub-terrains) | PPO |
| `go2-mixed-moe-cts` | Mixed generator (10 sub-terrains) | MoE-CTS (concurrent teacher/student, 0.75 teacher ratio) |

```bash
# Flat-ground PPO
./go2_rl_lab.sh --train --task go2-flat --num_envs 2048

# Mixed-terrain PPO
./go2_rl_lab.sh --train --task go2-mixed --num_envs 1024

# Mixed-terrain MoE-CTS (teacher/student)
./go2_rl_lab.sh --train --task go2-mixed-moe-cts --num_envs 2048
```

| Flag | Meaning |
| --- | --- |
| `--task` | Task id: `go2-flat`, `go2-mixed`, `go2-mixed-moe-cts` |
| `--num_envs` | Parallel environments (memory bound; see notes) |
| `--max_iterations` | Number of PPO iterations (default 100000) |
| `--seed` | Environment/agent seed |
| `--experiment_name` | Log group under `logs/rsl_rl/` |
| `--run_name` | Suffix of the run directory (use it to tell runs apart) |
| `--logger` | `tensorboard` (default) or `wandb` |
| `--resume` | Resume the latest checkpoint of the experiment |
| `--video`, `--video_length`, `--video_interval` | Record training videos (`--video_interval` must be > 0) |
| `--gpu_ids 0 1 ...` | GPU selection; more than one GPU goes through mjlab's torchrunx |

Any mjlab/tyro config field can be overridden on top, for example:

```bash
./go2_rl_lab.sh --train --task go2-mixed-moe-cts --num_envs 2048 \
  --env.scene.num-envs 1536 --agent.algorithm.learning-rate 5e-4
```

Notes:

- Mixed tasks generate a 10 x 20 terrain grid on first start: **about 7 minutes** before iteration 0.
- GPU/host memory: 2048 flat environments use ~3 GB of GPU memory; 1024 mixed-terrain
  environments use ~1.7 GB on top of the desktop. Mixed tasks also need several GB of **host**
  RAM, so close heavy desktop apps before a 2048-environment run. Do not run two trainings on
  one GPU.
- Checkpoints are written every `save_interval` (500) iterations as `model_<iter>.pt`, plus a final
  checkpoint; directory `logs/rsl_rl/<timestamp>_<run_name>/`.
- Resume continues the curriculum step counter, so terrain/command curricula do not restart.

### 2. Monitor with TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --port 6006        # open http://127.0.0.1:6006
```

Pick the run in the top-right run selector (`<timestamp>_<run_name>`). Useful scalars:

| Scalar | What it tells you |
| --- | --- |
| `Episode_Reward/*` | Per-term reward contributions |
| `Train/mean_reward`, `Train/mean_episode_length` | Overall learning progress |
| `Curriculum/terrain_levels/*/mean_level` | Terrain difficulty progression per sub-terrain |
| `Episode_Termination/*` | illegal contacts / time-outs |
| `Loss/mean_latent_loss`, `Loss/mean_load_balance_loss` | MoE-CTS student distillation and expert balance |
| `Policy/mean_std`, `Loss/learning_rate` | Exploration and adaptive-KL behaviour |

### 3. Play and export

```bash
# Native MuJoCo window (needs a display)
./go2_rl_lab.sh --play --task go2-flat --checkpoint logs/rsl_rl/<run>/model_12000.pt

# Browser viewer with velocity sliders (no display needed)
./go2_rl_lab.sh --play --task go2-flat --checkpoint <ckpt> --viewer viser

# Record a video instead of opening a window
./go2_rl_lab.sh --play --task go2-flat --checkpoint <ckpt> --video --video_length 300
```

| Flag | Meaning |
| --- | --- |
| `--checkpoint` | Checkpoint path; omit to pick the latest under `logs/rsl_rl/<experiment>` |
| `--num_envs` | Playback environments (default 32) |
| `--viewer` | `auto` (native if a display exists, else viser), `native`, `viser` |
| `--video`, `--video_length` | Record to `<run>/videos/play/rl-video-step-0.mp4` (use >= 5) |
| `--no-export` | Skip the policy export step |

Playback writes `<run>/exported/{policy.pt,policy.onnx,config.yaml}` by default. `policy.pt` is a
TorchScript module that takes one 45-dimensional observation frame and keeps the 10-frame history
internally, which is exactly what `deploy/deploy_mujoco` consumes. Mixed tasks still pay the
terrain-generation cost on the first playback.

### 4. Sim2Sim (MuJoCo)

```bash
./go2_rl_lab.sh --deploy-mujoco \
  --policy logs/rsl_rl/<run>/exported/policy.pt --no-joystick --headless --duration 10
```

### 5. Sim2Real

Hardware deployment uses C++. See [unitree_cpp_deploy - go2 branch](https://github.com/wty-yy/unitree_cpp_deploy/tree/go2) for code and instructions

### Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `KeyError: 'go2-flat'` | Package not installed in the active environment: run `./go2_rl_lab.sh --install` there |
| First mixed run hangs for minutes | Terrain generation (10 x 20 tiles); it happens once per process |
| No video file | `--video_length` too small (use >= 5); `--video_interval 0` divides by zero |
| CUDA OOM | Lower `--num_envs` (2048 for flat, 1024 if the GPU also drives a display) |
| Playback window does not open | No `DISPLAY`; use `--viewer viser` or `--video` |

## 🧩 Framework

```text
.
├── go2_rl_lab.sh                # Unified CLI
├── scripts/
│   ├── rsl_rl/                  # Train, play/export, deployment metadata
│   ├── tools/                   # GPU utilization helper
│   └── utils/                   # Task listing and optional dependency checks
├── source/
│   └── go2_rl_lab/go2_rl_lab/
│       ├── tasks/               # flat/mixed env configs and MDP terms
│       ├── cmd_utils/           # Go2 Gym velocity command and curricula
│       ├── assets/              # Go2 MJCF, entity config, and motor model
│       ├── moe_cts/             # MoE-CTS algorithm, symmetry, exporter
│       └── utils/               # Height scans and terrain lookups
└── deploy/deploy_mujoco/        # MuJoCo inference and scenes
```

## 🎉 Acknowledgements

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab): project foundation and robot training structure
- [mjlab](https://github.com/mujocolab/mjlab): manager-based RL environment framework on MuJoCo Warp
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): original simulation and environment framework
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): reinforcement learning infrastructure
- [Go2 RL Gym](https://github.com/wty-yy/go2_rl_lab/tree/gym): velocity baseline and MuJoCo deployment reference
- [Go2 RL RobotLab](https://github.com/wertyuilife2/go2_rl_robotlab): locomotion configuration reference
- [RoboGauge](https://github.com/wty-yy/RoboGauge): policy evaluation
- [MuJoCo](https://github.com/google-deepmind/mujoco): standalone simulation

## 📄 License

Project-owned code uses the [MIT License](LICENSE), copyright 2026 Tianyang Wu. Third-party code retains its original terms:

- Unitree RL Lab-derived code uses Apache 2.0
- Isaac Lab and original RSL-RL code use BSD 3-Clause
- MuJoCo deployment retains the [Go2 RL Gym notice](docs/licenses/go2-rl-gym-license.txt)
- Go2 USD assets come from the [Unitree model dataset](https://huggingface.co/datasets/unitreerobotics/unitree_model)

See the consolidated [LICENSE](LICENSE) and [license notices](docs/licenses/)

## 📚 Citation

If you find our work helpful, please cite:

```bibtex
@INPROCEEDINGS{WuT-RSS-26,
    AUTHOR    = {Tianyang Wu AND Hanwei Guo AND Yuhang Wang AND Junshu Yang AND Xinyang Sui AND Jiayi Xie AND Xingyu Chen AND Zeyang Liu AND Xuguang Lan},
    TITLE     = {{Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion}},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2026},
    ADDRESS   = {Sydney, Australia},
    MONTH     = {July},
    DOI       = {10.15607/RSS.2026.XXII.156}
}
```
