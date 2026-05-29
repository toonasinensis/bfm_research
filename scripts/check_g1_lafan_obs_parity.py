from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_DIR = PROJECT_DIR / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check G1 LAFAN expert/body-index/observation parity.")
    parser.add_argument(
        "--motions",
        type=str,
        default=str(PROJECT_SOURCE_DIR / "bfm" / "data" / "lafan"),
    )
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=0,
        help="Optional steps after the parity check. Use 0 with --hold to keep the viewer open indefinitely.",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep the simulation window open after the parity check. GUI + --debug-vis enables this by default.",
    )
    parser.add_argument(
        "--play-motion",
        action="store_true",
        help="Advance the robot and expert markers through LAFAN frames instead of holding one frame.",
    )
    parser.add_argument("--print-every", type=int, default=60, help="Print debug errors every N viewer steps.")
    parser.add_argument("--debug-vis", action="store_true", help="Show robot/expert body frame markers.")
    parser.add_argument("--output", type=str, default="", help="Optional npz output with robot/expert body arrays.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _extract_obs(obs: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    if "policy" in obs:
        value = obs["policy"]
    else:
        value = next(iter(obs.values()))
    if isinstance(value, torch.Tensor):
        return value.reshape(value.shape[0], -1)
    leaves = [item.reshape(item.shape[0], -1) for item in value.values()]
    return torch.cat(leaves, dim=-1)


def _quat_angle_error(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _set_motion_frame(command, unwrapped, motion_index: int, frame: int, env_ids: torch.Tensor) -> None:
    motion_ids = torch.full((unwrapped.num_envs,), motion_index, dtype=torch.long, device=unwrapped.device)
    time_steps = torch.full((unwrapped.num_envs,), frame, dtype=torch.long, device=unwrapped.device)
    command.motion_ids[:] = motion_ids
    command.time_steps[:] = time_steps
    command._reset_robot_to_motion(env_ids, motion_ids, time_steps)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()


def _compute_errors(command, obs: torch.Tensor | dict, motion_index: int, frame: int) -> dict[str, torch.Tensor]:
    obs_tensor = _extract_obs(obs)
    expert_obs = command.motion.observation_episode(motion_index)[frame].to(obs_tensor.device)
    obs_error = obs_tensor - expert_obs.reshape(1, -1)
    body_pos_error = torch.norm(command.robot_body_pos_w - command.body_pos_w, dim=-1)
    body_rot_error = _quat_angle_error(command.robot_body_quat_w, command.body_quat_w)
    return {
        "obs": obs_tensor,
        "expert_obs": expert_obs,
        "obs_error": obs_error,
        "body_pos_error": body_pos_error,
        "body_rot_error": body_rot_error,
    }


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import bfm.tasks  # noqa: F401
    from bfm.tasks.g1.g1_env_cfg import G1LafanEnvCfg

    env_cfg = G1LafanEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.commands.motion.motions = args.motions
    env_cfg.commands.motion.debug_vis = args.debug_vis
    env_cfg.commands.motion.reset_robot_state = False

    env = gym.make("G1-LAFAN-Flat-v0", cfg=env_cfg)
    try:
        obs, _ = env.reset()
        unwrapped = env.unwrapped
        command = unwrapped.command_manager.get_term("motion")
        if args.motion_index < 0 or args.motion_index >= len(command.motion):
            raise IndexError(f"motion_index={args.motion_index} out of range for {len(command.motion)} motions.")
        frame = max(0, min(args.frame, command.motion.length(args.motion_index) - 1))

        env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
        _set_motion_frame(command, unwrapped, args.motion_index, frame, env_ids)
        obs = unwrapped.observation_manager.compute()
        errors = _compute_errors(command, obs, args.motion_index, frame)
        obs_tensor = errors["obs"]
        expert_obs = errors["expert_obs"]
        obs_error = errors["obs_error"]
        body_pos_error = errors["body_pos_error"]
        body_rot_error = errors["body_rot_error"]

        print(
            f"[INFO] motion={command.motion.files[args.motion_index]} frame={frame} "
            f"obs_shape={tuple(obs_tensor.shape)} expert_obs_shape={tuple(expert_obs.shape)}",
            flush=True,
        )
        print(
            f"[INFO] obs mse={torch.mean(obs_error.square()).item():.8f} "
            f"max_abs={obs_error.abs().max().item():.8f}",
            flush=True,
        )
        print(
            f"[INFO] body_pos mean={body_pos_error.mean().item():.8f} max={body_pos_error.max().item():.8f} "
            f"body_rot mean={body_rot_error.mean().item():.8f} max={body_rot_error.max().item():.8f}",
            flush=True,
        )

        print("[INFO] body index mapping and frame-0 errors:", flush=True)
        for local_idx, body_name in enumerate(command.cfg.body_names):
            robot_idx = int(command.body_indexes[local_idx].item())
            print(
                f"  local={local_idx:02d} robot_idx={robot_idx:02d} "
                f"robot_name={command.robot.body_names[robot_idx]} cfg_name={body_name} "
                f"pos_err={body_pos_error[0, local_idx].item():.8f} "
                f"rot_err={body_rot_error[0, local_idx].item():.8f}",
                flush=True,
            )

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                robot_body_pos=command.robot_body_pos_w.detach().cpu().numpy(),
                expert_body_pos=command.body_pos_w.detach().cpu().numpy(),
                robot_body_quat=command.robot_body_quat_w.detach().cpu().numpy(),
                expert_body_quat=command.body_quat_w.detach().cpu().numpy(),
                obs=obs_tensor.detach().cpu().numpy(),
                expert_obs=expert_obs.detach().cpu().numpy(),
                body_names=np.asarray(command.cfg.body_names),
                body_indexes=command.body_indexes.detach().cpu().numpy(),
            )
            print(f"[INFO] saved parity arrays to {output_path}", flush=True)

        if args.debug_vis and hasattr(command, "set_debug_vis"):
            command.set_debug_vis(True)

        zero_actions = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device)
        hold = args.hold or (args.debug_vis and not args.headless)
        max_steps = args.num_steps if args.num_steps > 0 else (2**60 if hold else 0)
        for step in range(max_steps):
            if not simulation_app.is_running():
                break
            if args.play_motion:
                frame = (frame + 1) % command.motion.length(args.motion_index)
                _set_motion_frame(command, unwrapped, args.motion_index, frame, env_ids)
                obs = unwrapped.observation_manager.compute()
            else:
                env.step(zero_actions)
                obs = unwrapped.observation_manager.compute()

            if step == 0 or (args.print_every > 0 and (step + 1) % args.print_every == 0):
                errors = _compute_errors(command, obs, args.motion_index, frame)
                print(
                    f"[INFO] debug step={step + 1} frame={frame} "
                    f"obs_mse={torch.mean(errors['obs_error'].square()).item():.8f} "
                    f"body_pos_mean={errors['body_pos_error'].mean().item():.8f} "
                    f"body_pos_max={errors['body_pos_error'].max().item():.8f}",
                    flush=True,
                )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
