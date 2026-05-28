from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
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
    parser = argparse.ArgumentParser(description="Run FB-CPR tracking inference on a HumEnv reference motion.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to an FB-CPR checkpoint/model folder, or a HuggingFace repo id such as facebook/metamotivo-S-1.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=str(PROJECT_DIR / "pretrained"),
        help="Local directory used when --checkpoint is a HuggingFace repo id.",
    )
    parser.add_argument("--motions", type=str, required=True, help="Motion txt file used to select the reference h5.")
    parser.add_argument("--motions-root", type=str, required=True, help="Root directory for relative h5 names.")
    parser.add_argument("--motion-index", type=int, default=0, help="Line index inside --motions.")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index inside the selected h5.")
    parser.add_argument("--start-step", type=int, default=0, help="First reference frame used for tracking inference.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=300,
        help="Maximum rollout steps per episode. Use <=0 to play each selected episode to the end.",
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of IsaacLab envs to roll out.")
    parser.add_argument("--mean", action="store_true", help="Use the policy mean action instead of sampling.")
    parser.add_argument("--no-init-state", action="store_true", help="Do not reset robot state from motion qpos/qvel.")
    parser.add_argument(
        "--iterate-motions",
        action="store_true",
        help="Play every motion/episode from --motion-index/--episode-index onward.",
    )
    parser.add_argument(
        "--loop-motions",
        action="store_true",
        help="When --iterate-motions is enabled, wrap back to the first motion after the list ends.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Stop after this many episodes in iterate mode. Use 0 for no explicit episode limit.",
    )
    parser.add_argument(
        "--continue-on-nan",
        action="store_true",
        help="Keep rolling out after NaN/Inf is detected. By default rollout stops at the first non-finite value.",
    )
    parser.add_argument("--output", type=str, default="", help="Optional npz path for rollout obs/action/z traces.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _extract_obs(obs: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs.reshape(obs.shape[0], -1)
    if isinstance(obs, dict):
        if "policy" in obs:
            value = obs["policy"]
        elif "actor" in obs:
            value = obs["actor"]
        else:
            value = obs
        if isinstance(value, torch.Tensor):
            return value.reshape(value.shape[0], -1)
        leaves = []
        for item in value.values():
            leaves.append(_extract_obs(item))
        return torch.cat(leaves, dim=-1)
    raise TypeError(f"Unsupported observation type {type(obs)!r}.")


def _format_nan_context(context: Mapping[str, object]) -> str:
    return " ".join(f"{key}={value}" for key, value in context.items())


def _tensor_nonfinite_record(path: str, value: torch.Tensor) -> dict[str, object] | None:
    if not (torch.is_floating_point(value) or torch.is_complex(value)):
        return None

    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    bad_mask = ~finite_mask
    if not bool(bad_mask.any().item()):
        return None

    flat_bad = detached.reshape(-1)[bad_mask.reshape(-1)]
    first_indices = bad_mask.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
    first_values = flat_bad[:5].detach().cpu().tolist()
    nan_count = int(torch.isnan(detached).sum().item())
    inf_mask = torch.isinf(detached)
    posinf_count = int(torch.logical_and(inf_mask, detached.real > 0).sum().item())
    neginf_count = int(torch.logical_and(inf_mask, detached.real < 0).sum().item())
    return {
        "path": path,
        "shape": tuple(detached.shape),
        "dtype": str(detached.dtype),
        "count": int(bad_mask.sum().item()),
        "nan": nan_count,
        "posinf": posinf_count,
        "neginf": neginf_count,
        "first_indices": first_indices,
        "first_values": first_values,
    }


def _array_nonfinite_record(path: str, value: np.ndarray) -> dict[str, object] | None:
    if not np.issubdtype(value.dtype, np.number):
        return None

    bad_mask = ~np.isfinite(value)
    if not bool(bad_mask.any()):
        return None

    flat_bad = value.reshape(-1)[bad_mask.reshape(-1)]
    first_indices = np.argwhere(bad_mask)[:5].tolist()
    first_values = flat_bad[:5].tolist()
    return {
        "path": path,
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "count": int(bad_mask.sum()),
        "nan": int(np.isnan(value).sum()),
        "posinf": int(np.logical_and(np.isinf(value), np.real(value) > 0).sum()),
        "neginf": int(np.logical_and(np.isinf(value), np.real(value) < 0).sum()),
        "first_indices": first_indices,
        "first_values": first_values,
    }


def _iter_nonfinite_records(value: object, path: str):
    if isinstance(value, torch.Tensor):
        record = _tensor_nonfinite_record(path, value)
        if record is not None:
            yield record
        return

    if isinstance(value, np.ndarray):
        record = _array_nonfinite_record(path, value)
        if record is not None:
            yield record
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_nonfinite_records(item, f"{path}.{key}")
        return

    if isinstance(value, tuple | list):
        for index, item in enumerate(value):
            yield from _iter_nonfinite_records(item, f"{path}[{index}]")


def _truthy_info_keys(info: object) -> list[str]:
    if not isinstance(info, Mapping):
        return []

    keys = []
    for key, value in info.items():
        if isinstance(value, torch.Tensor) and value.dtype == torch.bool and bool(value.any().item()):
            keys.append(str(key))
        elif isinstance(value, np.ndarray) and value.dtype == np.bool_ and bool(value.any()):
            keys.append(str(key))
        elif isinstance(value, Mapping):
            for child_key in _truthy_info_keys(value):
                keys.append(f"{key}.{child_key}")
    return keys


class NonFiniteMonitor:
    def __init__(self, max_reports: int = 32) -> None:
        self.max_reports = max_reports
        self.report_count = 0
        self.checks_with_nonfinite = 0
        self.total_nonfinite_values = 0
        self.field_stats: dict[str, dict[str, int]] = {}

    def check(self, value: object, path: str, context: Mapping[str, object]) -> bool:
        records = list(_iter_nonfinite_records(value, path))
        if not records:
            return False

        self.checks_with_nonfinite += 1
        context_text = _format_nan_context(context)
        for record in records:
            field = str(record["path"])
            stats = self.field_stats.setdefault(
                field,
                {"events": 0, "values": 0, "nan": 0, "posinf": 0, "neginf": 0},
            )
            stats["events"] += 1
            stats["values"] += int(record["count"])
            stats["nan"] += int(record["nan"])
            stats["posinf"] += int(record["posinf"])
            stats["neginf"] += int(record["neginf"])
            self.total_nonfinite_values += int(record["count"])

            if self.report_count < self.max_reports:
                print(
                    "[WARN] non-finite detected "
                    f"{context_text} field={field} shape={record['shape']} dtype={record['dtype']} "
                    f"count={record['count']} nan={record['nan']} +inf={record['posinf']} "
                    f"-inf={record['neginf']} first_indices={record['first_indices']} "
                    f"first_values={record['first_values']}",
                    flush=True,
                )
            elif self.report_count == self.max_reports:
                print("[WARN] non-finite report limit reached; suppressing further per-field reports.", flush=True)
            self.report_count += 1

        return True

    def print_summary(self) -> None:
        if self.checks_with_nonfinite == 0:
            print("[INFO] nan_check: no NaN/Inf values detected during rollout.", flush=True)
            return

        print(
            f"[WARN] nan_check: detected NaN/Inf in {self.checks_with_nonfinite} checks; "
            f"total_nonfinite_values={self.total_nonfinite_values}",
            flush=True,
        )
        for field, stats in sorted(self.field_stats.items()):
            print(
                "[WARN] nan_check field "
                f"{field}: events={stats['events']} values={stats['values']} nan={stats['nan']} "
                f"+inf={stats['posinf']} -inf={stats['neginf']}",
                flush=True,
            )


def _resolve_model_dir(checkpoint: str, hf_cache_dir: str | Path) -> Path:
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path / "model" if (checkpoint_path / "model").is_dir() else checkpoint_path

    if "/" not in checkpoint:
        raise FileNotFoundError(
            f"Cannot find checkpoint path {checkpoint!r}. Pass a local folder or a HuggingFace repo id."
        )

    from huggingface_hub import snapshot_download

    local_dir = Path(hf_cache_dir).expanduser() / checkpoint.split("/")[-1]
    model_dir = snapshot_download(
        repo_id=checkpoint,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=["config.json", "model.safetensors"],
    )
    return Path(model_dir)


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
            f"[INFO] loaded model with missing_keys={len(missing)} unexpected_keys={len(unexpected)} "
            "(target-network keys are expected for training checkpoints).",
            flush=True,
        )
    model.train(False)
    return model


def _iter_motion_episodes(
    motions: str | Path,
    motions_root: str | Path,
    start_motion_index: int,
    start_episode_index: int,
    loop: bool,
):
    from bfm.tasks.humenv.mdp.commands import canonicalize, load_episode_based_h5, read_motion_list

    motion_files = read_motion_list(motions)
    if not motion_files:
        raise ValueError(f"No motion files listed in {motions!s}.")
    if start_motion_index < 0 or start_motion_index >= len(motion_files):
        raise IndexError(f"motion_index={start_motion_index} is out of range for {len(motion_files)} motion files.")

    motion_index = start_motion_index
    episode_index = start_episode_index
    while True:
        for current_motion_index in range(motion_index, len(motion_files)):
            motion_path = canonicalize(motion_files[current_motion_index], base_path=motions_root)
            episodes = load_episode_based_h5(motion_path)
            if not episodes:
                continue
            current_episode_index = episode_index if current_motion_index == motion_index else 0
            if current_episode_index < 0 or current_episode_index >= len(episodes):
                raise IndexError(
                    f"episode_index={current_episode_index} is out of range for "
                    f"{len(episodes)} episodes in {motion_path!s}."
                )
            for ep_idx in range(current_episode_index, len(episodes)):
                yield current_motion_index, motion_path, ep_idx, episodes[ep_idx]

        if not loop:
            break
        motion_index = 0
        episode_index = 0


def _set_motion_state(env, episode: dict, start_step: int) -> None:
    """Best-effort IsaacLab state reset from HumEnv qpos/qvel arrays."""

    if "qpos" not in episode or "qvel" not in episode:
        print("[WARN] motion episode has no qpos/qvel; skipping reference state initialization.", flush=True)
        return

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = unwrapped.device
    env_ids = torch.arange(unwrapped.num_envs, dtype=torch.long, device=device)
    from bfm.tasks.humenv.mdp.commands import motion_joint_indices_for_robot, root_ang_vel_from_humenv_qpos_qvel

    motion_joint_indices = motion_joint_indices_for_robot(robot.joint_names, device=device)

    qpos = torch.as_tensor(episode["qpos"][start_step], dtype=torch.float32, device=device)
    qvel = torch.as_tensor(episode["qvel"][start_step], dtype=torch.float32, device=device)

    root_state = robot.data.default_root_state[env_ids].clone()
    if qpos.numel() >= 7:
        root_state[:, :3] = qpos[:3]
        root_state[:, :3] += unwrapped.scene.env_origins
        root_state[:, 3:7] = qpos[3:7]
    if qvel.numel() >= 6:
        root_state[:, 7:10] = qvel[:3]
        root_state[:, 10:13] = root_ang_vel_from_humenv_qpos_qvel(qpos, qvel).expand(unwrapped.num_envs, -1)

    joint_pos_count = robot.data.joint_pos.shape[1]
    joint_vel_count = robot.data.joint_vel.shape[1]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    if qpos.numel() >= 7 + joint_pos_count:
        joint_pos[:] = qpos[7 + motion_joint_indices]
    else:
        print(
            f"[WARN] qpos has {qpos.numel()} values, expected at least {7 + joint_pos_count}; "
            "using default joint positions.",
            flush=True,
        )
    if qvel.numel() >= 6 + joint_vel_count:
        joint_vel[:] = qvel[6 + motion_joint_indices]
    else:
        print(
            f"[WARN] qvel has {qvel.numel()} values, expected at least {6 + joint_vel_count}; "
            "using default joint velocities.",
            flush=True,
        )

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()


def main() -> None:
    args = parse_args()
    if args.loop_motions:
        args.iterate_motions = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import bfm.tasks  # noqa: F401
    from bfm.tasks.humenv.humenv_env_cfg import HumEnvMjcfEnvCfg

    model_dir = _resolve_model_dir(args.checkpoint, args.hf_cache_dir)
    print(f"[INFO] loading FB-CPR model from {model_dir}", flush=True)
    model = _load_fbcpr_model(model_dir, device=args.device)

    env_cfg = HumEnvMjcfEnvCfg()
    env_cfg.seed = 0
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.commands.motion.motions = args.motions
    env_cfg.commands.motion.motions_root = args.motions_root
    # Inference sets the selected h5 frame explicitly below. Leaving the command
    # term state reset on lets its independent random motion clock teleport the
    # robot when that command reaches the end of its sampled clip.
    env_cfg.commands.motion.reset_robot_state = False
    env_cfg.episode_length_s = 1.0e6

    env = gym.make("HumEnv-MJCF-Flat-v0", cfg=env_cfg)
    nan_monitor = NonFiniteMonitor()
    traces: dict[str, list[np.ndarray]] = {
        "obs": [],
        "action": [],
        "z": [],
        "done": [],
        "reference_observation": [],
        "motion_index": [],
        "episode_index": [],
        "reference_step": [],
    }

    try:
        episodes_played = 0
        global_steps = 0
        stop_due_to_nonfinite = False
        for motion_index, motion_path, episode_index, episode in _iter_motion_episodes(
            args.motions,
            args.motions_root,
            start_motion_index=args.motion_index,
            start_episode_index=args.episode_index,
            loop=args.loop_motions,
        ):
            if stop_due_to_nonfinite:
                break
            if not args.iterate_motions and episodes_played > 0:
                break
            if args.max_episodes > 0 and episodes_played >= args.max_episodes:
                break
            if not simulation_app.is_running():
                break

            reference_obs = torch.as_tensor(episode["observation"], dtype=torch.float32, device=model.cfg.device)
            if reference_obs.ndim != 2:
                raise ValueError(f"Expected reference observations to be rank-2, got {tuple(reference_obs.shape)}.")
            if reference_obs.shape[1] != model.cfg.obs_dim:
                raise ValueError(
                    f"Reference obs dim {reference_obs.shape[1]} does not match checkpoint obs dim {model.cfg.obs_dim}."
                )

            first_selected_episode = episodes_played == 0
            start_step = args.start_step if first_selected_episode else 0
            start_step = max(0, min(start_step, reference_obs.shape[0] - 1))
            end_step = reference_obs.shape[0] if args.num_steps <= 0 else min(start_step + args.num_steps, reference_obs.shape[0])
            if end_step <= start_step:
                print(
                    f"[WARN] skipping motion_index={motion_index} episode_index={episode_index}: "
                    f"invalid frames=[{start_step}, {end_step}).",
                    flush=True,
                )
                continue

            with torch.inference_mode():
                z_reference = model.tracking_inference(reference_obs[start_step:end_step])
            episode_context = {
                "global_step": global_steps,
                "motion_index": motion_index,
                "episode_index": episode_index,
                "reference_step": f"{start_step}:{end_step}",
            }
            found_nonfinite = False
            found_nonfinite |= nan_monitor.check(reference_obs[start_step:end_step], "reference_observation", episode_context)
            found_nonfinite |= nan_monitor.check(z_reference, "z_reference", episode_context)
            if found_nonfinite and not args.continue_on_nan:
                print("[ERROR] stopping rollout because NaN/Inf was detected before stepping the environment.", flush=True)
                stop_due_to_nonfinite = True
                break
            rollout_steps = end_step - start_step
            print(
                f"[INFO] motion_index={motion_index} episode_index={episode_index} motion={motion_path} "
                f"frames=[{start_step}, {end_step}) obs={tuple(reference_obs.shape)} z={tuple(z_reference.shape)}",
                flush=True,
            )

            obs, _ = env.reset()
            if not args.no_init_state:
                _set_motion_state(env, episode, start_step)
                obs = env.unwrapped.observation_manager.compute()
            obs_tensor = _extract_obs(obs).to(model.cfg.device, dtype=torch.float32)
            init_context = {
                "global_step": global_steps,
                "motion_index": motion_index,
                "episode_index": episode_index,
                "reference_step": start_step,
            }
            if nan_monitor.check(obs_tensor, "obs_after_reset", init_context) and not args.continue_on_nan:
                print("[ERROR] stopping rollout because NaN/Inf was detected after reset.", flush=True)
                stop_due_to_nonfinite = True
                break

            for local_step in range(rollout_steps):
                if not simulation_app.is_running():
                    break
                reference_step = start_step + local_step
                step_context = {
                    "global_step": global_steps + 1,
                    "motion_index": motion_index,
                    "episode_index": episode_index,
                    "local_step": local_step,
                    "reference_step": reference_step,
                }
                z = z_reference[local_step].reshape(1, -1).expand(args.num_envs, -1)
                found_nonfinite = False
                found_nonfinite |= nan_monitor.check(obs_tensor, "obs_before_act", step_context)
                found_nonfinite |= nan_monitor.check(reference_obs[reference_step], "reference_observation", step_context)
                found_nonfinite |= nan_monitor.check(z, "z", step_context)
                if found_nonfinite and not args.continue_on_nan:
                    print("[ERROR] stopping rollout because NaN/Inf was detected before action.", flush=True)
                    stop_due_to_nonfinite = True
                    break

                with torch.inference_mode():
                    action = model.act(obs=obs_tensor, z=z, mean=args.mean).to(env.unwrapped.device)
                if nan_monitor.check(action, "action", step_context) and not args.continue_on_nan:
                    print("[ERROR] stopping rollout because NaN/Inf action was produced.", flush=True)
                    stop_due_to_nonfinite = True
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                done = torch.logical_or(terminated, truncated)
                obs_tensor = _extract_obs(obs).to(model.cfg.device, dtype=torch.float32)
                found_nonfinite = False
                found_nonfinite |= nan_monitor.check(obs_tensor, "obs_after_step", step_context)
                found_nonfinite |= nan_monitor.check(reward, "reward", step_context)
                found_nonfinite |= nan_monitor.check(terminated, "terminated", step_context)
                found_nonfinite |= nan_monitor.check(truncated, "truncated", step_context)
                found_nonfinite |= nan_monitor.check(done, "done", step_context)
                found_nonfinite |= nan_monitor.check(info, "info", step_context)
                if found_nonfinite and not args.continue_on_nan:
                    print("[ERROR] stopping rollout because NaN/Inf was detected after environment step.", flush=True)
                    stop_due_to_nonfinite = True
                    break
                if bool(done.any().item()):
                    print(
                        f"[WARN] env done global_step={global_steps + 1} motion_index={motion_index} "
                        f"episode_index={episode_index} local_step={local_step} reference_step={reference_step} "
                        f"terminated={terminated.detach().cpu().tolist()} "
                        f"truncated={truncated.detach().cpu().tolist()} "
                        f"info_true_keys={_truthy_info_keys(info)}",
                        flush=True,
                    )

                traces["obs"].append(_to_numpy(obs_tensor))
                traces["action"].append(_to_numpy(action))
                traces["z"].append(_to_numpy(z))
                traces["done"].append(_to_numpy(done.reshape(args.num_envs, 1)))
                traces["reference_observation"].append(_to_numpy(reference_obs[reference_step]))
                traces["motion_index"].append(np.asarray(motion_index, dtype=np.int64))
                traces["episode_index"].append(np.asarray(episode_index, dtype=np.int64))
                traces["reference_step"].append(np.asarray(reference_step, dtype=np.int64))
                global_steps += 1

                if local_step == 0 or (local_step + 1) % 100 == 0 or local_step + 1 == rollout_steps:
                    print(
                        f"[INFO] global_step={global_steps} motion_index={motion_index} "
                        f"episode_index={episode_index} step={local_step + 1}/{rollout_steps} "
                        f"done={int(done.sum().item())} reward_mean={reward.float().mean().item():.6f}",
                        flush=True,
                    )

            episodes_played += 1
            if stop_due_to_nonfinite:
                break
            if not simulation_app.is_running():
                break

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stacked = {key: np.stack(value, axis=0) for key, value in traces.items() if value}
            np.savez_compressed(output_path, **stacked)
            print(f"[INFO] saved rollout trace to {output_path}", flush=True)
    finally:
        nan_monitor.print_summary()
        env.close()
        simulation_app.close()
        print("[INFO] tracking inference completed", flush=True)


if __name__ == "__main__":
    main()
