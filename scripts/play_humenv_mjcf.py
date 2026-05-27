"""Minimal random-action smoke test for the HumEnv MJCF IsaacLab task."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play the minimal HumEnv MJCF task with random actions.")
parser.add_argument("--task", type=str, default="HumEnv-MJCF-Flat-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=10000, help="Maximum number of simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import bfm.tasks  # noqa: F401
from bfm.tasks.humenv.humenv_env_cfg import HumEnvMjcfEnvCfg


def _describe_tree(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, dict):
        return {key: _describe_tree(item) for key, item in value.items()}
    return type(value).__name__


def main():
    env_cfg = HumEnvMjcfEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()

    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"[INFO] task={args_cli.task}")
    print(f"[INFO] num_envs={env.unwrapped.num_envs}")
    print(f"[INFO] action_dim={action_dim}")
    print(f"[INFO] observation={_describe_tree(obs)}")

    step = 0
    while simulation_app.is_running() and step < args_cli.num_steps:
        with torch.inference_mode():
            action = torch.rand(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device) * 2.0 - 1.0
            obs, reward, terminated, truncated, info = env.step(action)
        if step == 0:
            print(f"[INFO] first_step_reward_shape={_describe_tree(reward)}")
            print(f"[INFO] first_step_done_shape={_describe_tree(terminated | truncated)}")
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
