# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified for the Go2-only Go2 RL Lab project.

"""Play and export a trained Go2 RSL-RL policy.

Quick Start:
    ./go2_rl_lab.sh --play --task go2-flat --checkpoint path/to/model.pt

Full Command:
    ./go2_rl_lab.sh --play --task go2-mixed --checkpoint path/to/model.pt \
        --num_envs 4 --video --video_length 200
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

import export
from go2_rl_lab.utils.terrains import configure_high_to_low_play_origins


def parse_args(argv: list[str]) -> tuple[str, argparse.Namespace, list[str]]:
    """Parse the project CLI flags, leaving tyro overrides for mjlab."""
    parser = argparse.ArgumentParser(
        description="Play and export a trained Go2 RSL-RL policy.",
        add_help=False,
    )
    parser.add_argument("--task", type=str, required=True, help="Task ID to play.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint or exported policy path.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Accepted for CLI compatibility; mjlab viewers already pace themselves in real time.",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="auto",
        choices=("auto", "native", "viser"),
        help="Playback viewer backend.",
    )
    parser.add_argument("--no-export", action="store_false", dest="export", help="Skip policy export.")
    return parser.parse_known_args(argv)


def resolve_checkpoint(task: str, checkpoint: str | None) -> Path:
    """Return the checkpoint path, defaulting to the latest run of the task."""
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    agent_cfg = load_rl_cfg(task)
    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


def export_policy(
    task: str,
    runner: MjlabOnPolicyRunner,
    env: RslRlVecEnvWrapper,
    checkpoint_path: Path,
) -> None:
    """Export the TorchScript policy, ONNX model, and deployment metadata."""
    current_obs = env.get_observations()
    num_single_obs = int(current_obs["single_obs"].shape[-1])
    num_actor_obs = int(current_obs["policy"].shape[-1])
    num_actions = env.num_actions
    export_dir = str(checkpoint_path.parent / "exported")

    if hasattr(runner, "export_policy_as_jit"):
        # MoE-CTS runners own a student-specific exporter.
        runner.export_policy_as_jit(path=export_dir, filename="policy.pt")
    else:
        export.export_policy_as_jit(
            policy=runner.alg.get_policy(),
            path=export_dir,
            num_single_obs=num_single_obs,
            num_actions=num_actions,
            filename="policy.pt",
        )
    try:
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    except Exception as error:  # noqa: BLE001
        print(f"[WARN] ONNX export failed: {error}")

    config_path = export.export_model_config(
        path=export_dir,
        checkpoint_path=str(checkpoint_path),
        checkpoint_iteration=runner.current_learning_iteration,
        task=task,
        experiment_name=load_rl_cfg(task).experiment_name,
        num_single_obs=num_single_obs,
        num_actor_obs=num_actor_obs,
        num_actions=num_actions,
        control_dt=env.unwrapped.step_dt,
    )
    print(f"[INFO]: Exported policies to: {export_dir}")
    print(f"[INFO]: Exported model configuration to: {config_path}")


def main() -> None:
    """Load, export, and play the policy."""
    import mjlab.tasks  # noqa: F401  (populates the task registry)

    args, _remaining = parse_args(sys.argv[1:])
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    checkpoint_path = resolve_checkpoint(args.task, args.checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    device = "cuda:0"

    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    configure_high_to_low_play_origins(env)

    video_recorder = None
    if args.video:
        from mjlab.utils.wrappers import VideoRecorder

        print("[INFO]: Recording playback video")
        video_recorder = VideoRecorder(
            env,
            video_folder=checkpoint_path.parent / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        env = video_recorder

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), str(checkpoint_path.parent), device)
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)

    if args.export:
        export_policy(args.task, runner, wrapped, checkpoint_path)

    if args.video:
        policy = runner.get_inference_policy(device=device)
        obs = wrapped.get_observations()
        for _ in range(args.video_length):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = wrapped.step(actions)
        env.close()
        return

    if args.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer = "native" if has_display else "viser"
    else:
        viewer = args.viewer

    policy = runner.get_inference_policy(device=device)
    if viewer == "native":
        NativeMujocoViewer(wrapped, policy).run()
    else:
        ViserPlayViewer(wrapped, policy).run()
    env.close()


if __name__ == "__main__":
    main()
