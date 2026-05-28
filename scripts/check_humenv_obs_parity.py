"""Compare IsaacLab HumEnv observations against HumEnv h5 reference observations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_DIR = PROJECT_DIR / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check HumEnv observation parity for qpos/qvel reference frames.")
    parser.add_argument("--motions", type=str, required=True)
    parser.add_argument("--motions-root", type=str, required=True)
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--stride", type=int, default=10)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _extract_obs(obs: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    if isinstance(obs, dict):
        value = obs.get("policy", obs.get("actor", obs))
        if isinstance(value, torch.Tensor):
            return value.reshape(value.shape[0], -1)
        return torch.cat([_extract_obs(item) for item in value.values()], dim=-1)
    raise TypeError(f"Unsupported observation type {type(obs)!r}.")


def _set_motion_frame(env, episode: dict, frame: int) -> None:
    from bfm.tasks.humenv.mdp.commands import motion_joint_indices_for_robot, root_ang_vel_from_humenv_qpos_qvel

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = unwrapped.device
    env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=device)
    motion_joint_indices = motion_joint_indices_for_robot(robot.joint_names, device=device)

    qpos = torch.as_tensor(episode["qpos"][frame], dtype=torch.float32, device=device)
    qvel = torch.as_tensor(episode["qvel"][frame], dtype=torch.float32, device=device)

    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :3] = qpos[:3] + unwrapped.scene.env_origins
    root_state[:, 3:7] = qpos[3:7]
    root_state[:, 7:10] = qvel[:3]
    root_state[:, 10:13] = root_ang_vel_from_humenv_qpos_qvel(qpos, qvel).expand(unwrapped.num_envs, -1)

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[:] = qpos[7 + motion_joint_indices]
    joint_vel[:] = qvel[6 + motion_joint_indices]

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    unwrapped.scene.update(dt=0.0)


def _segment_stats(diff: torch.Tensor) -> tuple[float, float, float]:
    return (
        float(torch.mean(diff * diff).item()),
        float(torch.mean(torch.abs(diff)).item()),
        float(torch.max(torch.abs(diff)).item()),
    )


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import bfm.tasks  # noqa: F401
    from bfm.tasks.humenv.humenv_env_cfg import HumEnvMjcfEnvCfg
    from bfm.tasks.humenv.mdp.commands import load_motion_episode

    motion_path, episode = load_motion_episode(
        args.motions,
        args.motions_root,
        motion_index=args.motion_index,
        episode_index=args.episode_index,
    )

    env_cfg = HumEnvMjcfEnvCfg()
    env_cfg.seed = 0
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    env_cfg.commands.motion.motions = ""
    env_cfg.commands.motion.motions_root = ""
    env_cfg.episode_length_s = 1.0e6

    env = gym.make("HumEnv-MJCF-Flat-v0", cfg=env_cfg)
    try:
        env.reset()
        reference_obs = torch.as_tensor(episode["observation"], dtype=torch.float32, device=env.unwrapped.device)
        total_frames = reference_obs.shape[0]
        frames = list(range(max(args.start_step, 0), total_frames, max(args.stride, 1)))[: args.num_frames]
        if not frames:
            raise ValueError("No frames selected for parity check.")

        segment_slices = {
            "root_h": slice(0, 1),
            "local_body_pos": slice(1, 70),
            "local_body_rot_obs": slice(70, 214),
            "local_body_vel": slice(214, 286),
            "local_body_ang_vel": slice(286, 358),
        }
        totals = {name: [] for name in ["all", *segment_slices.keys()]}

        print(
            f"[INFO] motion={motion_path} episode_index={args.episode_index} "
            f"frames={frames[0]}..{frames[-1]} count={len(frames)}",
            flush=True,
        )
        for frame in frames:
            _set_motion_frame(env, episode, frame)
            obs = _extract_obs(env.unwrapped.observation_manager.compute())[0]
            ref = reference_obs[frame]
            diff = obs - ref
            totals["all"].append(_segment_stats(diff))
            for name, seg in segment_slices.items():
                totals[name].append(_segment_stats(diff[seg]))

        print("name mse mae max_abs", flush=True)
        for name, values in totals.items():
            stacked = torch.tensor(values)
            mean_values = stacked.mean(dim=0)
            print(
                name,
                f"{mean_values[0].item():.8f}",
                f"{mean_values[1].item():.8f}",
                f"{mean_values[2].item():.8f}",
                flush=True,
            )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
