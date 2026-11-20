# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified for the Go2-only Go2 RL Lab project.

"""Train Go2 policies with mjlab and the RSL-RL PPO runner.

Quick Start:
    ./go2_rl_lab.sh --train --task go2-flat

Full Command:
    ./go2_rl_lab.sh --train --task go2-mixed --num_envs 4096 \
        --max_iterations 50000 --logger tensorboard

Notes:
    Requires the mjlab runtime (MuJoCo Warp). Extra arguments use mjlab's tyro
    syntax, for example ``--env.scene.num-envs 4096``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import tyro

from mjlab.scripts.train import TrainConfig, launch_training


def parse_args(argv: list[str]) -> tuple[str, TrainConfig]:
    """Parse the project CLI flags, then mjlab/tyro overrides."""
    parser = argparse.ArgumentParser(
        description="Train a Go2 policy with mjlab and RSL-RL.",
        add_help=False,
    )
    parser.add_argument("--task", type=str, required=True, help="Task ID to train.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Environment and agent seed.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run-name suffix.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Log group name.")
    parser.add_argument("--logger", type=str, default=None, choices=("tensorboard", "wandb"))
    parser.add_argument("--resume", action="store_true", default=None, help="Resume the latest run.")
    parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
    parser.add_argument("--video_length", type=int, default=None, help="Recorded video length (steps).")
    parser.add_argument("--video_interval", type=int, default=None, help="Steps between recordings.")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=None, help="GPU ids used for training.")
    known, remaining = parser.parse_known_args(argv)

    cfg = TrainConfig.from_task(known.task)
    if known.num_envs is not None:
        cfg.env.scene.num_envs = known.num_envs
    if known.max_iterations is not None:
        cfg.agent.max_iterations = known.max_iterations
    if known.seed is not None:
        cfg.agent.seed = known.seed
    if known.run_name is not None:
        cfg.agent.run_name = known.run_name
    if known.experiment_name is not None:
        cfg.agent.experiment_name = known.experiment_name
    if known.logger is not None:
        cfg.agent.logger = known.logger
    if known.resume:
        cfg.agent.resume = True

    # TrainConfig is a frozen dataclass; replace() keeps the overrides immutable.
    frozen_updates = {}
    if known.video:
        frozen_updates["video"] = True
    if known.video_length is not None:
        frozen_updates["video_length"] = known.video_length
    if known.video_interval is not None:
        frozen_updates["video_interval"] = known.video_interval
    if known.gpu_ids is not None:
        frozen_updates["gpu_ids"] = known.gpu_ids
    if frozen_updates:
        cfg = replace(cfg, **frozen_updates)

    cfg = tyro.cli(
        TrainConfig,
        args=remaining,
        default=cfg,
        prog=sys.argv[0] + f" {known.task}",
    )
    return known.task, cfg


def main() -> None:
    """Launch training for the requested task."""
    import mjlab.tasks  # noqa: F401  (populates the task registry)

    task_id, cfg = parse_args(sys.argv[1:])
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
