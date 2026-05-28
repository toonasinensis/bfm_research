"""Debug HumEnv MJCF stage contents and short-horizon dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug HumEnv MJCF import and dynamics.")
parser.add_argument("--task", type=str, default="HumEnv-MJCF-Flat-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_steps", type=int, default=120, help="Maximum number of simulation steps.")
parser.add_argument("--zero_actions", action="store_true", help="Use zero actions instead of random actions.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from pxr import Usd, UsdPhysics

import bfm.tasks  # noqa: F401
from bfm.tasks.humenv.humenv_env_cfg import HumEnvMjcfEnvCfg


def _stage_report(env) -> None:
    stage = env.unwrapped.scene.stage
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    worldbody = stage.GetPrimAtPath("/World/envs/env_0/Robot/worldBody")
    floor = stage.GetPrimAtPath("/World/envs/env_0/Robot/worldBody/floor")
    roots = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(robot_prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    print(f"[DEBUG] robot_prim_valid={robot_prim.IsValid()}", flush=True)
    print(f"[DEBUG] xml_worldBody_exists={worldbody.IsValid()}", flush=True)
    print(f"[DEBUG] xml_floor_exists={floor.IsValid()}", flush=True)
    print(f"[DEBUG] articulation_roots={roots}", flush=True)


def _tensor_row(tensor: torch.Tensor) -> list[float]:
    return [round(float(value), 5) for value in tensor[0].detach().cpu()]


def main():
    env_cfg = HumEnvMjcfEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    action_dim = env.unwrapped.action_manager.total_action_dim
    _stage_report(env)
    print(f"[DEBUG] body_names={robot.body_names}", flush=True)
    print(f"[DEBUG] joint_names={robot.joint_names}", flush=True)
    print(f"[DEBUG] action_dim={action_dim}", flush=True)
    print(f"[DEBUG] default_root_pos={_tensor_row(robot.data.default_root_state[:, :3])}", flush=True)
    print(f"[DEBUG] default_root_quat={_tensor_row(robot.data.default_root_state[:, 3:7])}", flush=True)

    step = 0
    while simulation_app.is_running() and step < args_cli.num_steps:
        with torch.inference_mode():
            if args_cli.zero_actions:
                action = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
            else:
                action = torch.rand(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device) * 2.0 - 1.0
            env.step(action)

        if step < 10 or step % 10 == 0:
            root_pos = robot.data.root_pos_w
            root_quat = robot.data.root_quat_w
            root_lin_vel = robot.data.root_lin_vel_w
            joint_abs_max = robot.data.joint_pos.abs().max()
            finite = torch.isfinite(root_pos).all() and torch.isfinite(root_quat).all()
            print(
                "[DEBUG] "
                f"step={step:04d} "
                f"root_pos={_tensor_row(root_pos)} "
                f"root_quat={_tensor_row(root_quat)} "
                f"lin_vel={_tensor_row(root_lin_vel)} "
                f"joint_abs_max={float(joint_abs_max):.5f} "
                f"finite={bool(finite)}",
                flush=True,
            )
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
