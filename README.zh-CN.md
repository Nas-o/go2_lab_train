<div align="center">
  <h1>Go2 RL Lab</h1>
  <p><strong>基于 mjlab（MuJoCo Warp）的 Unitree Go2 高速灵巧速度运动控制<br><a href="https://github.com/wty-yy/go2_rl_lab/tree/gym">（原 go2_rl_gym 分支）</a></strong></p>
  <p>
    <a href="https://github.com/mujocolab/mjlab"><img height="20" src="https://img.shields.io/badge/mjlab-1.2.0-silver.svg" alt="mjlab 1.2.0"></a>&nbsp;
    <a href="https://swanlab.cn/@wty-yy/go2-rl-lab/v1/lxqfax/overview"><img height="20" src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" alt="SwanLab"></a>&nbsp;
    <a href="LICENSE"><img height="20" src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  </p>
  <p><a href="README.md">🌎 English</a>&nbsp;&nbsp;·&nbsp;&nbsp;<strong>🇨🇳 中文</strong>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://www.roboticsproceedings.org/rss22/p156.pdf">📄 Paper [RSS 2026]</a></p>
</div>

📢 本工程已迁移到 **[mjlab](https://github.com/mujocolab/mjlab) 1.2.0 / MuJoCo Warp**：任务层、地形生成器、速度指令课程、奖励项、电机模型与训练入口均由 Isaac Lab 3.0 EA 实现逐项移植，并保持部署所需的观测、动作与奖励语义不变

<table width="100%">
  <thead>
    <tr>
      <th width="33%" align="center"><div align="center">IsaacSim</div></th>
      <th width="33%" align="center"><div align="center">MuJoCo</div></th>
      <th width="33%" align="center"><div align="center">真机</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="Kit 演示" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/play_isaaclab_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="MuJoCo 演示" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/sim/python_mujoco_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="真机楼梯演示" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_stairs_DJI_20261017144009_0111_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">3.5m/s前进与4.4m/s后退</div></th>
      <th width="50%" align="center"><div align="center">奔跑俯视</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="真机前进与后退" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/forward_backward_1x_then_0.5x.gif" /></td>
      <td align="center"><img width="100%" alt="真机快速奔跑" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/overlook_fastrun_DJI_20261017143527_0110_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

<table width="100%">
  <thead>
    <tr>
      <th width="50%" align="center"><div align="center">斜坡</div></th>
      <th width="50%" align="center"><div align="center">楼梯间</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img width="100%" alt="真机斜坡行走" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/slope_DJI_20261017142100_0108_D_encoded.gif" /></td>
      <td align="center"><img width="100%" alt="真机楼梯间行走" src="https://raw.githubusercontent.com/wty-yy/picture-bed/master/go2_rl_lab/real/stairwell_DJI_20261017145051_0117_D_encoded.gif" /></td>
    </tr>
  </tbody>
</table>

## 📦 安装

本工程运行在 **mjlab 1.2.0**（MuJoCo Warp）与 `rsl-rl-lib` 5.0.1 之上，需要 Python 3.12+ 和 NVIDIA GPU。

```bash
# 创建包含 mjlab 运行时的环境（MuJoCo Warp + Warp + PyTorch）
conda create -n go2_mjlab python=3.12
conda activate go2_mjlab
pip install "mjlab==1.2.0" "mujoco-warp==3.5.0" "mujoco==3.5.0"

# 以可编辑模式安装任务包，并列出已注册任务
./go2_rl_lab.sh --install
./go2_rl_lab.sh --list
```

`./go2_rl_lab.sh --install` 以可编辑模式安装 `source/go2_rl_lab` 并注册 `mjlab.tasks` entry point，`mjlab` 因此会自动发现 `go2-flat`、`go2-mixed`、`go2-mixed-moe-cts`。

## 🛠️ 命令速查

### 0. 环境准备（每个 Python 环境执行一次）

```bash
conda activate <你的环境>          # 装有 mjlab 1.2.0 的环境
./go2_rl_lab.sh --install          # 可编辑安装 + mjlab.tasks entry point
./go2_rl_lab.sh --list             # go2-flat / go2-mixed / go2-mixed-moe-cts
```

> mjlab 通过 `mjlab.tasks` entry point 自动发现任务。**新建或切换 Python 环境后要在该环境里重新执行
> `--install`**，否则会报 `KeyError: '<task>'`。

### 1. 训练

| 任务 ID | 地形 | 算法 |
| --- | --- | --- |
| `go2-flat` | 平面 | PPO |
| `go2-mixed` | 混合地形生成器（10 种子地形） | PPO |
| `go2-mixed-moe-cts` | 混合地形生成器（10 种子地形） | MoE-CTS（教师/学生并发，教师比例 0.75） |

```bash
# 平地 PPO
./go2_rl_lab.sh --train --task go2-flat --num_envs 2048

# 混合地形 PPO
./go2_rl_lab.sh --train --task go2-mixed --num_envs 1024

# 混合地形 MoE-CTS（教师/学生）
./go2_rl_lab.sh --train --task go2-mixed-moe-cts --num_envs 2048
```

| 参数 | 含义 |
| --- | --- |
| `--task` | 任务 id：`go2-flat`、`go2-mixed`、`go2-mixed-moe-cts` |
| `--num_envs` | 并行环境数（受显存限制，见下） |
| `--max_iterations` | PPO 迭代次数（默认 100000） |
| `--seed` | 环境/智能体随机种子 |
| `--experiment_name` | `logs/rsl_rl/` 下的实验分组 |
| `--run_name` | 运行目录后缀（区分多次实验） |
| `--logger` | `tensorboard`（默认）或 `wandb` |
| `--resume` | 从该实验最近的检查点继续 |
| `--video`、`--video_length`、`--video_interval` | 录制训练视频（`--video_interval` 必须 > 0） |
| `--gpu_ids 0 1 ...` | 选择 GPU；多卡会走 mjlab 的 torchrunx |

还可直接覆盖任意 mjlab/tyro 配置项，例如：

```bash
./go2_rl_lab.sh --train --task go2-mixed-moe-cts --num_envs 2048 \
  --env.scene.num-envs 1536 --agent.algorithm.learning-rate 5e-4
```

说明：

- 混合地形任务首次启动会生成 10 × 20 的地形网格，**约 7 分钟**后才开始第 0 次迭代。
- 8 GB 显卡建议 2048 环境（实测平地约 3 GB，MoE-CTS 2048 环境整卡约 6.2 GB）；不要在同一张卡上并行两个训练。
- 每 `save_interval`（默认 500）次迭代保存 `model_<iter>.pt`，训练结束再存一个最终检查点，目录为
  `logs/rsl_rl/<时间戳>_<run_name>/`。
- `--resume` 会恢复课程步数，地形/指令课程不会从头开始。

### 2. 用 TensorBoard 看训练

```bash
tensorboard --logdir logs/rsl_rl --port 6006        # 浏览器打开 http://127.0.0.1:6006
```

在右上角的 run 选择器里选 `<时间戳>_<run_name>`。重点曲线：

| 曲线 | 含义 |
| --- | --- |
| `Episode_Reward/*` | 各奖励项贡献 |
| `Train/mean_reward`、`Train/mean_episode_length` | 整体学习进度 |
| `Curriculum/terrain_levels/*/mean_level` | 各地形难度推进情况 |
| `Episode_Termination/*` | 非法接触 / 超时 |
| `Loss/mean_latent_loss`、`Loss/mean_load_balance_loss` | MoE-CTS 学生蒸馏与专家负载均衡 |
| `Policy/mean_std`、`Loss/learning_rate` | 探索强度与自适应 KL 行为 |

### 3. 回放与导出

```bash
# 原生 MuJoCo 窗口（需要显示环境）
./go2_rl_lab.sh --play --task go2-flat --checkpoint logs/rsl_rl/<run>/model_12000.pt

# 浏览器可视化，可拖速度滑条（无需显示器）
./go2_rl_lab.sh --play --task go2-flat --checkpoint <ckpt> --viewer viser

# 不开窗口，直接录视频
./go2_rl_lab.sh --play --task go2-flat --checkpoint <ckpt> --video --video_length 300
```

| 参数 | 含义 |
| --- | --- |
| `--checkpoint` | 检查点路径；省略则取 `logs/rsl_rl/<experiment>` 下最新 |
| `--num_envs` | 回放环境数（默认 32） |
| `--viewer` | `auto`（有显示则 native，否则 viser）、`native`、`viser` |
| `--video`、`--video_length` | 录到 `<run>/videos/play/rl-video-step-0.mp4`（建议 ≥ 5） |
| `--no-export` | 跳过策略导出 |

回放默认会把策略导出到 `<run>/exported/{policy.pt,policy.onnx,config.yaml}`。`policy.pt` 是
TorchScript 模块，输入一帧 45 维观测、内部维护 10 帧历史，正是 `deploy/deploy_mujoco` 使用的接口。
混合地形任务首次回放同样要等约 7 分钟生成地形。

### 4. Sim2Sim（MuJoCo）

```bash
./go2_rl_lab.sh --deploy-mujoco \
  --policy logs/rsl_rl/<run>/exported/policy.pt --no-joystick --headless --duration 10
```

### 5. Sim2Real

实机部署使用 C++，代码与说明见 [unitree_cpp_deploy - go2 分支](https://github.com/wty-yy/unitree_cpp_deploy/tree/go2)

### 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `KeyError: 'go2-flat'` | 当前环境没装包：在该环境执行 `./go2_rl_lab.sh --install` |
| 混合地形首次启动卡几分钟 | 正在生成 10 × 20 地形，每个进程只发生一次 |
| 没有生成视频文件 | `--video_length` 太小（用 ≥ 5）；`--video_interval 0` 会除零 |
| CUDA 显存不足 | 降低 `--num_envs`（平地 2048，显卡还要带显示时用 1024） |
| 回放窗口没弹出 | 没有 `DISPLAY`：改用 `--viewer viser` 或 `--video` |

## 🧩 框架

```text
.
├── go2_rl_lab.sh                # 统一命令行入口
├── scripts/
│   ├── rsl_rl/                  # 训练、回放/导出、部署元信息
│   ├── tools/                   # GPU 占用辅助工具
│   └── utils/                   # 任务列表与可选依赖检查
├── source/
│   └── go2_rl_lab/go2_rl_lab/
│       ├── tasks/               # flat/mixed 环境配置与 MDP 项
│       ├── cmd_utils/           # Go2 Gym 速度指令与课程
│       ├── assets/              # Go2 MJCF、实体配置与电机模型
│       ├── moe_cts/             # MoE-CTS 算法、对称性与导出
│       └── utils/               # 高度扫描与地形查询
└── deploy/deploy_mujoco/        # MuJoCo 推理与场景
```

## 🎉 致谢

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)：项目基础和机器人训练结构
- [mjlab](https://github.com/mujocolab/mjlab)：基于 MuJoCo Warp 的 manager 式 RL 环境框架
- [mjlab](https://github.com/mujocolab/mjlab)：基于 MuJoCo Warp 的 manager 式 RL 环境框架
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)：原始仿真与环境框架
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)：强化学习基础设施
- [Go2 RL Gym](https://github.com/wty-yy/go2_rl_lab/tree/gym)：速度基线和 MuJoCo 部署参考
- [Go2 RL RobotLab](https://github.com/wertyuilife2/go2_rl_robotlab)：运动控制配置参考
- [RoboGauge](https://github.com/wty-yy/RoboGauge)：策略评估
- [MuJoCo](https://github.com/google-deepmind/mujoco)：独立仿真

## 📄 许可证

项目自有代码采用 [MIT License](LICENSE)，版权归 2026 Tianyang Wu。第三方代码保留原有条款：

- Unitree RL Lab 衍生代码采用 Apache 2.0
- Isaac Lab 及原始 RSL-RL 代码采用 BSD 3-Clause
- MuJoCo 部署保留 [Go2 RL Gym 声明](docs/licenses/go2-rl-gym-license.txt)
- Go2 USD 资源来自 [Unitree 模型数据集](https://huggingface.co/datasets/unitreerobotics/unitree_model)

完整条款见合并后的 [LICENSE](LICENSE) 和 [许可证声明](docs/licenses/)

## 📚 引用

如果我们的工作对你有所帮助，请引用：

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
