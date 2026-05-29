from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_DIR = PROJECT_DIR / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FB-CPR tracking inference on the G1 LAFAN IsaacLab task.")
    parser.add_argument("--checkpoint", type=str, required=True, help="FB-CPR checkpoint folder or model folder.")
    parser.add_argument("--motions", type=str, default=str(PROJECT_SOURCE_DIR / "bfm" / "data" / "lafan"))
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--mean", action="store_true", help="Use mean action instead of sampling.")
    parser.add_argument("--debug-vis", action="store_true", help="Show expert and robot body frame markers.")
    parser.add_argument("--loop-motion", action="store_true", help="Loop the selected motion when it reaches the end.")
    parser.add_argument("--print-every", type=int, default=60)
    parser.add_argument("--continue-on-nan", action="store_true")
    parser.add_argument(
        "--trace-output",
        type=str,
        default=None,
        help="Write an IsaacLab alignment trace that can be replayed by the MuJoCo G1 script.",
    )
    parser.add_argument("--trace-env-index", type=int, default=0, help="Environment row to save in --trace-output.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _resolve_model_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint path {path!s}.")
    return path / "model" if (path / "model").is_dir() else path


def _load_fbcpr_model(model_dir: Path, device: str):
    from agents.metamotivo.fb_cpr.model import FBcprModel

    with (model_dir / "config.json").open() as f:
        model_cfg = json.load(f)
    model_cfg["device"] = device
    model = FBcprModel(**model_cfg)
    missing, unexpected = safetensors.torch.load_model(
        model,
        model_dir / "model.safetensors",
        strict=False,
        device=device,
    )
    if missing or unexpected:
        print(
            f"[INFO] loaded model with missing_keys={len(missing)} unexpected_keys={len(unexpected)}",
            flush=True,
        )
    model.train(False)
    return model


def _extract_obs(obs: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    if "policy" in obs:
        value = obs["policy"]
    elif "actor" in obs:
        value = obs["actor"]
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


def _row_cpu(value: torch.Tensor, env_index: int) -> np.ndarray:
    return value[env_index].detach().cpu().numpy().astype(np.float32)


def _action_term_vector(value, env_index: int, action_dim: int, device: torch.device) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.ndim >= 2:
            out = value[env_index]
        else:
            out = value
        return out.detach().cpu().numpy().astype(np.float32)
    return torch.full((action_dim,), float(value), device=device).detach().cpu().numpy().astype(np.float32)


def _snapshot_robot(command, env_index: int) -> dict[str, np.ndarray]:
    robot = command.robot
    return {
        "root_pos": _row_cpu(robot.data.root_pos_w, env_index),
        "root_quat": _row_cpu(robot.data.root_quat_w, env_index),
        "root_lin_vel": _row_cpu(robot.data.root_lin_vel_w, env_index),
        "root_ang_vel": _row_cpu(robot.data.root_ang_vel_w, env_index),
        "joint_pos": _row_cpu(robot.data.joint_pos, env_index),
        "joint_vel": _row_cpu(robot.data.joint_vel, env_index),
        "body_pos": _row_cpu(robot.data.body_pos_w[:, command.body_indexes], env_index),
        "body_quat": _row_cpu(robot.data.body_quat_w[:, command.body_indexes], env_index),
        "body_lin_vel": _row_cpu(robot.data.body_lin_vel_w[:, command.body_indexes], env_index),
        "body_ang_vel": _row_cpu(robot.data.body_ang_vel_w[:, command.body_indexes], env_index),
    }


def _trace_append(trace: dict[str, list[np.ndarray]], prefix: str, snapshot: dict[str, np.ndarray]) -> None:
    for key, value in snapshot.items():
        trace[f"{prefix}_{key}"].append(value)


def _write_trace(
    path: str | Path,
    *,
    trace: dict[str, list[np.ndarray]],
    metadata: dict,
    joint_names: list[str],
    action_joint_names: list[str],
    body_names: list[str],
    action_scale: np.ndarray,
    action_offset: np.ndarray,
) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(values, dtype=np.float32) for key, values in trace.items()}
    arrays["metadata"] = np.asarray(json.dumps(metadata, indent=2))
    arrays["joint_names"] = np.asarray(joint_names)
    arrays["action_joint_names"] = np.asarray(action_joint_names)
    arrays["body_names"] = np.asarray(body_names)
    arrays["action_scale"] = action_scale.astype(np.float32)
    arrays["action_offset"] = action_offset.astype(np.float32)
    np.savez_compressed(output, **arrays)
    print(f"[INFO] wrote IsaacLab alignment trace: {output}", flush=True)


def _finite(name: str, value: torch.Tensor, step: int) -> bool:
    if not torch.is_floating_point(value):
        return True
    bad = ~torch.isfinite(value)
    if not bool(bad.any().item()):
        return True
    flat = value.detach().reshape(-1)
    bad_flat = bad.reshape(-1)
    print(
        f"[WARN] non-finite {name} step={step} count={int(bad.sum().item())} "
        f"first_values={flat[bad_flat][:5].detach().cpu().tolist()}",
        flush=True,
    )
    return False


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import bfm.tasks  # noqa: F401
    from bfm.tasks.g1.g1_env_cfg import G1LafanEnvCfg

    model_dir = _resolve_model_dir(args.checkpoint)
    print(f"[INFO] loading FB-CPR model from {model_dir}", flush=True)
    model = _load_fbcpr_model(model_dir, device=args.device)

    env_cfg = G1LafanEnvCfg()
    env_cfg.seed = 0
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.commands.motion.motions = args.motions
    env_cfg.commands.motion.debug_vis = args.debug_vis
    env_cfg.commands.motion.reset_robot_state = False
    env_cfg.episode_length_s = 1.0e6

    env = gym.make("G1-LAFAN-Flat-v0", cfg=env_cfg)
    try:
        obs, _ = env.reset()
        unwrapped = env.unwrapped
        command = unwrapped.command_manager.get_term("motion")
        if args.motion_index < 0 or args.motion_index >= len(command.motion):
            raise IndexError(f"motion_index={args.motion_index} out of range for {len(command.motion)} motions.")
        if args.trace_env_index < 0 or args.trace_env_index >= unwrapped.num_envs:
            raise IndexError(f"trace_env_index={args.trace_env_index} out of range for num_envs={unwrapped.num_envs}.")

        env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device)
        motion_length = command.motion.length(args.motion_index)
        start_frame = max(0, min(args.start_frame, motion_length - 2))
        end_frame = motion_length if args.num_steps <= 0 else min(start_frame + args.num_steps + 1, motion_length)
        if end_frame <= start_frame + 1:
            raise ValueError(f"Motion {args.motion_index} is too short for start_frame={start_frame}.")

        reference_obs = command.motion.observation_episode(args.motion_index).to(model.cfg.device, dtype=torch.float32)
        if reference_obs.shape[1] != model.cfg.obs_dim:
            raise ValueError(f"Reference obs dim {reference_obs.shape[1]} does not match model obs dim {model.cfg.obs_dim}.")
        with torch.inference_mode():
            z_reference = model.tracking_inference(reference_obs[start_frame + 1 : end_frame])

        print(
            f"[INFO] motion={command.motion.files[args.motion_index]} frames=[{start_frame}, {end_frame}) "
            f"obs={tuple(reference_obs.shape)} z={tuple(z_reference.shape)} mean={args.mean}",
            flush=True,
        )

        if args.debug_vis and hasattr(command, "set_debug_vis"):
            command.set_debug_vis(True)

        _set_motion_frame(command, unwrapped, args.motion_index, start_frame, env_ids)
        obs = unwrapped.observation_manager.compute()
        obs_tensor = _extract_obs(obs).to(model.cfg.device, dtype=torch.float32)
        action_term = unwrapped.action_manager.get_term("joint_pos")
        action_dim = unwrapped.action_manager.total_action_dim
        trace: dict[str, list[np.ndarray]] | None = None
        if args.trace_output:
            trace = {
                "frame": [],
                "obs_before": [],
                "obs_after": [],
                "reference_obs": [],
                "z": [],
                "action": [],
                "processed_action": [],
                "reward": [],
                "terminated": [],
                "truncated": [],
                "done": [],
                "expert_body_pos": [],
                "expert_body_quat": [],
                "expert_body_lin_vel": [],
                "expert_body_ang_vel": [],
                "before_root_pos": [],
                "before_root_quat": [],
                "before_root_lin_vel": [],
                "before_root_ang_vel": [],
                "before_joint_pos": [],
                "before_joint_vel": [],
                "before_body_pos": [],
                "before_body_quat": [],
                "before_body_lin_vel": [],
                "before_body_ang_vel": [],
                "after_root_pos": [],
                "after_root_quat": [],
                "after_root_lin_vel": [],
                "after_root_ang_vel": [],
                "after_joint_pos": [],
                "after_joint_vel": [],
                "after_body_pos": [],
                "after_body_quat": [],
                "after_body_lin_vel": [],
                "after_body_ang_vel": [],
            }

        step = 0
        while simulation_app.is_running():
            local = step % z_reference.shape[0] if args.loop_motion else step
            if local >= z_reference.shape[0]:
                break

            frame = min(start_frame + local + 1, motion_length - 1)
            command.motion_ids[:] = args.motion_index
            command.time_steps[:] = frame
            z = z_reference[local].reshape(1, -1).expand(args.num_envs, -1)
            if not _finite("obs", obs_tensor, step) and not args.continue_on_nan:
                break
            if not _finite("z", z, step) and not args.continue_on_nan:
                break

            with torch.inference_mode():
                action = model.act(obs=obs_tensor, z=z, mean=args.mean).to(unwrapped.device)
            if not _finite("action", action, step) and not args.continue_on_nan:
                break

            if trace is not None:
                trace["frame"].append(np.asarray(frame, dtype=np.float32))
                trace["obs_before"].append(_row_cpu(obs_tensor, args.trace_env_index))
                trace["reference_obs"].append(
                    reference_obs[frame].detach().cpu().numpy().astype(np.float32)
                )
                trace["z"].append(_row_cpu(z, args.trace_env_index))
                trace["action"].append(_row_cpu(action, args.trace_env_index))
                trace["expert_body_pos"].append(_row_cpu(command.body_pos_w, args.trace_env_index))
                trace["expert_body_quat"].append(_row_cpu(command.body_quat_w, args.trace_env_index))
                trace["expert_body_lin_vel"].append(_row_cpu(command.body_lin_vel_w, args.trace_env_index))
                trace["expert_body_ang_vel"].append(_row_cpu(command.body_ang_vel_w, args.trace_env_index))
                _trace_append(trace, "before", _snapshot_robot(command, args.trace_env_index))

            obs, reward, terminated, truncated, info = env.step(action)
            del info
            done = torch.logical_or(terminated, truncated)
            obs_tensor = _extract_obs(obs).to(model.cfg.device, dtype=torch.float32)
            if trace is not None:
                trace["obs_after"].append(_row_cpu(obs_tensor, args.trace_env_index))
                trace["processed_action"].append(_row_cpu(action_term.processed_actions, args.trace_env_index))
                trace["reward"].append(_row_cpu(reward.reshape(-1, 1), args.trace_env_index))
                trace["terminated"].append(_row_cpu(terminated.float().reshape(-1, 1), args.trace_env_index))
                trace["truncated"].append(_row_cpu(truncated.float().reshape(-1, 1), args.trace_env_index))
                trace["done"].append(_row_cpu(done.float().reshape(-1, 1), args.trace_env_index))
                _trace_append(trace, "after", _snapshot_robot(command, args.trace_env_index))

            body_pos_error = torch.norm(command.robot_body_pos_w - command.body_pos_w, dim=-1)
            body_rot_error = _quat_angle_error(command.robot_body_quat_w, command.body_quat_w)
            obs_target = reference_obs[frame].reshape(1, -1).expand_as(obs_tensor)
            obs_mse = (obs_tensor - obs_target).square().mean()

            if not _finite("obs_after_step", obs_tensor, step) and not args.continue_on_nan:
                break
            if bool(done.any().item()):
                print(
                    f"[WARN] done step={step + 1} frame={frame} "
                    f"terminated={terminated.detach().cpu().tolist()} truncated={truncated.detach().cpu().tolist()}",
                    flush=True,
                )

            if step == 0 or (args.print_every > 0 and (step + 1) % args.print_every == 0):
                print(
                    f"[INFO] step={step + 1} frame={frame} reward={reward.float().mean().item():.6f} "
                    f"obs_mse={obs_mse.item():.6f} body_pos_mean={body_pos_error.mean().item():.6f} "
                    f"body_pos_max={body_pos_error.max().item():.6f} body_rot_mean={body_rot_error.mean().item():.6f}",
                    flush=True,
                )
            step += 1

        if trace is not None:
            metadata = {
                "version": 1,
                "source": "scripts/tracking_inference_fbcpr_g1.py",
                "motion_file": str(command.motion.files[args.motion_index]),
                "motion_index": args.motion_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "saved_steps": step,
                "trace_env_index": args.trace_env_index,
                "num_envs": args.num_envs,
                "mean": args.mean,
                "device": args.device,
                "dt": float(unwrapped.cfg.sim.dt),
                "decimation": int(unwrapped.cfg.decimation),
                "obs_dim": int(model.cfg.obs_dim),
                "action_dim": int(action_dim),
            }
            _write_trace(
                args.trace_output,
                trace=trace,
                metadata=metadata,
                joint_names=list(command.robot.joint_names),
                action_joint_names=list(action_term._joint_names),
                body_names=list(command.cfg.body_names),
                action_scale=_action_term_vector(action_term._scale, args.trace_env_index, action_dim, unwrapped.device),
                action_offset=_action_term_vector(action_term._offset, args.trace_env_index, action_dim, unwrapped.device),
            )

        print(f"[INFO] tracking inference completed steps={step}", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
