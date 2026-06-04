from __future__ import annotations

import dataclasses
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from agents.runner.env import unpack_observations


def _quat_angle_error(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _finite_count(*values: object) -> int:
    count = 0
    for value in values:
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            count += int((~torch.isfinite(value)).sum().item())
    return count


def _distance_matrix(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_norm = lhs.pow(2).sum(dim=1).reshape(-1, 1)
    rhs_norm = rhs.pow(2).sum(dim=1).reshape(1, -1)
    value = lhs_norm + rhs_norm - 2.0 * torch.matmul(lhs, rhs.T)
    return torch.sqrt(torch.clamp(value, min=0.0))


def _greedy_emd(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.numel() == 0 or rhs.numel() == 0:
        return torch.zeros((), device=lhs.device)
    distances = _distance_matrix(lhs, rhs)
    return 0.5 * (distances.min(dim=0).values.mean() + distances.min(dim=1).values.mean())


def _set_motion_frames(
    command,
    unwrapped,
    motion_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    motion_ids = motion_ids.to(dtype=torch.long, device=unwrapped.device)
    frame_ids = frame_ids.to(dtype=torch.long, device=unwrapped.device)
    env_ids = env_ids.to(dtype=torch.long, device=unwrapped.device)
    command.motion_ids[env_ids] = motion_ids
    command.time_steps[env_ids] = frame_ids
    command._reset_robot_to_motion(env_ids, motion_ids, frame_ids)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    return env_ids


def _set_motion_command_frames(
    command,
    motion_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    env_ids = env_ids.to(dtype=torch.long, device=command.device)
    command.motion_ids[env_ids] = motion_ids.to(dtype=torch.long, device=command.device)
    command.time_steps[env_ids] = frame_ids.to(dtype=torch.long, device=command.device)


def _assign_motions_to_envs(motion_ids: list[int], env_count: int, device: torch.device) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    if not motion_ids:
        raise ValueError("Cannot assign an empty motion list.")
    shuffled_env_ids = list(range(env_count))
    random.shuffle(shuffled_env_ids)
    shuffled_env_ids_t = torch.as_tensor(shuffled_env_ids, dtype=torch.long, device=device)
    motion_ids_t = torch.as_tensor(motion_ids, dtype=torch.long, device=device)
    assigned = motion_ids_t[shuffled_env_ids_t % len(motion_ids)]
    motion_to_envs: dict[int, list[int]] = defaultdict(list)
    for env_id, motion_id in enumerate(assigned.tolist()):
        motion_to_envs[int(motion_id)].append(env_id)
    return assigned, {
        motion_id: torch.as_tensor(env_ids, dtype=torch.long, device=device)
        for motion_id, env_ids in motion_to_envs.items()
    }


def _old_set_motion_frame(command, unwrapped, motion_index: int, frame: int, env_count: int) -> torch.Tensor:
    env_ids = torch.arange(env_count, dtype=torch.long, device=unwrapped.device)
    motion_ids = torch.full((env_count,), motion_index, dtype=torch.long, device=unwrapped.device)
    time_steps = torch.full((env_count,), frame, dtype=torch.long, device=unwrapped.device)
    return _set_motion_frames(command, unwrapped, motion_ids, time_steps, env_ids)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _motion_label(command, motion_index: int) -> str:
    files = getattr(getattr(command, "motion", None), "files", None)
    if files is None or motion_index < 0 or motion_index >= len(files):
        return str(motion_index)
    return Path(files[motion_index]).name


def _print_assignment(command, chunk_index: int, assigned_motions: torch.Tensor, limit: int) -> None:
    assigned = assigned_motions.detach().cpu().tolist()
    counts: dict[int, int] = defaultdict(int)
    for motion_index in assigned:
        counts[int(motion_index)] += 1
    summary = ", ".join(
        f"{motion_id}:{count} envs/{_motion_label(command, motion_id)}" for motion_id, count in sorted(counts.items())
    )
    print(
        f"[G1TrackingEvaluator] chunk={chunk_index} parallel assignment "
        f"envs={len(assigned)} motions={len(counts)} {summary}",
        flush=True,
    )
    if limit == 0:
        return
    max_rows = len(assigned) if limit < 0 else min(len(assigned), limit)
    for env_id, motion_index in enumerate(assigned[:max_rows]):
        print(
            f"  env={env_id:04d} motion={int(motion_index):04d} file={_motion_label(command, int(motion_index))}",
            flush=True,
        )
    if max_rows < len(assigned):
        print(f"  ... {len(assigned) - max_rows} more env assignments omitted", flush=True)


def _extend_episode_length_for_rollout(unwrapped, rollout_steps: int) -> float:
    previous_episode_length_s = float(unwrapped.cfg.episode_length_s)
    required_episode_length_s = float((rollout_steps + 2) * unwrapped.step_dt)
    if required_episode_length_s > previous_episode_length_s:
        unwrapped.cfg.episode_length_s = required_episode_length_s
        print(
            "[G1TrackingEvaluator] temporarily extended eval episode length "
            f"{previous_episode_length_s:.2f}s -> {required_episode_length_s:.2f}s "
            f"for {rollout_steps} rollout steps",
            flush=True,
        )
    return previous_episode_length_s


@dataclasses.dataclass
class G1TrackingEvaluator:
    cfg: Any
    agent: Any
    env: Any

    def __call__(self, runner) -> dict[str, float]:
        unwrapped = self.env.unwrapped
        command = unwrapped.command_manager.get_term("motion")
        model = self.agent._model
        previous_training = model.training
        device = torch.device(model.cfg.device)
        motions = self.cfg.tracking_eval_motions or self.cfg.motions
        if motions and str(motions) != str(command.cfg.motions):
            print(
                "[G1TrackingEvaluator] using train env motion command; "
                f"requested motions={motions} active={command.cfg.motions}",
                flush=True,
            )

        max_motions = self.cfg.tracking_eval_max_motions
        motion_count = len(command.motion)
        if max_motions and max_motions > 0:
            motion_count = min(motion_count, max_motions)
        motion_ids = list(range(motion_count))
        if not motion_ids:
            raise ValueError("G1 tracking eval found no motions.")
        eval_env_count = min(max(int(self.cfg.tracking_eval_num_envs), 1), unwrapped.num_envs)
        chunk_size = eval_env_count

        print(
            "[G1TrackingEvaluator] "
            f"motions={motion_count}/{len(command.motion)} eval_envs={eval_env_count}/{unwrapped.num_envs} "
            f"mean_action={self.cfg.tracking_eval_mean_action}",
            flush=True,
        )
        start_time = time.time()
        metrics: dict[str, list[float]] = defaultdict(list)
        nan_count = 0
        total_steps = 0
        done_steps = 0
        original_episode_length_s = float(unwrapped.cfg.episode_length_s)
        iterator = range(0, len(motion_ids), chunk_size)
        if self.cfg.tracking_eval_progress and tqdm is not None:
            total_chunks = (len(motion_ids) + chunk_size - 1) // chunk_size
            iterator = tqdm(iterator, desc="g1 tracking eval", total=total_chunks, unit="chunk", dynamic_ncols=True, leave=False)

        try:
            model.train(False)
            for chunk_start in iterator:
                chunk_motion_ids = motion_ids[chunk_start : chunk_start + chunk_size]
                assigned_motions, motion_to_envs = _assign_motions_to_envs(
                    chunk_motion_ids,
                    env_count=eval_env_count,
                    device=unwrapped.device,
                )
                if getattr(self.cfg, "tracking_eval_print_assignments", False):
                    _print_assignment(
                        command=command,
                        chunk_index=chunk_start // max(chunk_size, 1),
                        assigned_motions=assigned_motions,
                        limit=int(getattr(self.cfg, "tracking_eval_print_assignment_limit", 64)),
                    )
                reference_obs_by_motion = {}
                z_by_motion = {}
                rollout_steps_by_motion = {}
                for motion_index in chunk_motion_ids:
                    with torch.no_grad():
                        reference_obs = command.motion.observation_episode(motion_index).to(device=device, dtype=torch.float32)
                        if reference_obs.ndim != 2:
                            raise ValueError(f"Expected rank-2 reference obs, got {tuple(reference_obs.shape)}.")
                        if reference_obs.shape[1] != model.cfg.obs_dim:
                            raise ValueError(
                                f"G1 eval obs dim {reference_obs.shape[1]} does not match model obs dim {model.cfg.obs_dim}."
                            )
                        if reference_obs.shape[0] < 2:
                            continue
                        reference_obs_by_motion[motion_index] = reference_obs
                        z_by_motion[motion_index] = model.tracking_inference(reference_obs[1:])
                    rollout_steps_by_motion[motion_index] = min(reference_obs.shape[0] - 1, z_by_motion[motion_index].shape[0])
                if not reference_obs_by_motion:
                    continue

                rollout_steps = max(rollout_steps_by_motion.values())
                if self.cfg.tracking_eval_max_steps and self.cfg.tracking_eval_max_steps > 0:
                    rollout_steps = min(rollout_steps, self.cfg.tracking_eval_max_steps)
                print(
                    f"[G1TrackingEvaluator] chunk={chunk_start // max(chunk_size, 1)} "
                    f"rollout_steps={rollout_steps} "
                    f"motion_steps=[{min(rollout_steps_by_motion.values())}, {max(rollout_steps_by_motion.values())}]",
                    flush=True,
                )
                _extend_episode_length_for_rollout(unwrapped, rollout_steps)

                self.env.reset()
                eval_env_ids = torch.arange(eval_env_count, dtype=torch.long, device=unwrapped.device)
                _set_motion_frames(
                    command=command,
                    unwrapped=unwrapped,
                    motion_ids=assigned_motions,
                    frame_ids=torch.zeros(eval_env_count, dtype=torch.long, device=unwrapped.device),
                    env_ids=eval_env_ids,
                )
                obs, _ = unpack_observations(unwrapped.observation_manager.compute())
                obs = obs[eval_env_ids].to(device=device, dtype=torch.float32)

                active = torch.ones(eval_env_count, dtype=torch.bool, device=device)
                z_dim = next(iter(z_by_motion.values())).shape[1]
                for local_step in range(rollout_steps):
                    motion_frame_ids = torch.zeros(eval_env_count, dtype=torch.long, device=unwrapped.device)
                    z = torch.empty(eval_env_count, z_dim, dtype=obs.dtype, device=device)
                    target = torch.empty(eval_env_count, model.cfg.obs_dim, dtype=obs.dtype, device=device)
                    step_active = active.clone()
                    for motion_index, env_ids in motion_to_envs.items():
                        agent_env_ids = env_ids.to(device=device)
                        if motion_index not in z_by_motion:
                            step_active[agent_env_ids] = False
                            continue
                        motion_step = local_step % rollout_steps_by_motion[motion_index]
                        z[agent_env_ids] = z_by_motion[motion_index][motion_step]
                        target[agent_env_ids] = reference_obs_by_motion[motion_index][motion_step + 1]
                        motion_frame_ids[env_ids] = motion_step
                    if not bool(step_active.any().item()):
                        break
                    _set_motion_command_frames(command, assigned_motions, motion_frame_ids, eval_env_ids)
                    with torch.no_grad():
                        eval_action = model.act(obs=obs, z=z, mean=self.cfg.tracking_eval_mean_action).to(unwrapped.device)
                    eval_action[torch.logical_not(step_active).to(device=eval_action.device)] = 0.0
                    action = torch.zeros(
                        unwrapped.num_envs,
                        unwrapped.action_manager.total_action_dim,
                        dtype=eval_action.dtype,
                        device=unwrapped.device,
                    )
                    action[eval_env_ids] = eval_action
                    next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
                    del reward, info
                    done = torch.logical_or(terminated, truncated)[eval_env_ids]
                    next_obs, _ = unpack_observations(next_obs_raw)
                    obs = next_obs[eval_env_ids].to(device=device, dtype=torch.float32)

                    obs_error = obs - target
                    body_pos_error = torch.norm(
                        command.robot_body_pos_w[eval_env_ids] - command.body_pos_w[eval_env_ids],
                        dim=-1,
                    )
                    body_rot_error = _quat_angle_error(
                        command.robot_body_quat_w[eval_env_ids],
                        command.body_quat_w[eval_env_ids],
                    )

                    metric_active = step_active
                    body_metric_active = metric_active.to(device=body_pos_error.device)
                    active_obs_error = obs_error[metric_active]
                    active_obs = obs[metric_active]
                    active_target = target[metric_active]
                    active_body_pos_error = body_pos_error[body_metric_active]
                    active_body_rot_error = body_rot_error[body_metric_active]

                    metrics["obs_distance"].append(float(torch.norm(active_obs_error, dim=-1).mean().item()))
                    metrics["obs_mse"].append(float(active_obs_error.square().mean().item()))
                    metrics["obs_emd"].append(float(_greedy_emd(active_obs.detach().cpu(), active_target.detach().cpu()).item()))
                    metrics["body_pos_mean"].append(float(active_body_pos_error.mean().item()))
                    metrics["body_pos_max"].append(float(active_body_pos_error.max().item()))
                    metrics["body_rot_mean"].append(float(active_body_rot_error.mean().item()))
                    metrics["body_rot_max"].append(float(active_body_rot_error.max().item()))
                    nan_count += _finite_count(obs, action, body_pos_error, body_rot_error)
                    done_steps += int(done[metric_active.to(device=done.device)].float().sum().item())
                    total_steps += int(metric_active.float().sum().item())
                    active = metric_active & torch.logical_not(done.to(device=device))
                    if not bool(active.any().item()):
                        break
        finally:
            unwrapped.cfg.episode_length_s = original_episode_length_s
            model.train(previous_training)

        output = {f"tracking/{key}": _mean(value) for key, value in sorted(metrics.items())}
        output["tracking/done_rate"] = float(done_steps / max(total_steps, 1))
        output["tracking/nan_count"] = float(nan_count)
        output["tracking/num_motions"] = float(len(motion_ids))
        output["tracking/parallel_envs"] = float(eval_env_count)
        output["tracking/chunks"] = float((len(motion_ids) + chunk_size - 1) // chunk_size)
        output["tracking/num_steps"] = float(total_steps)
        output["tracking/time"] = time.time() - start_time
        runner.request_observation_refresh()
        print(f"[G1TrackingEvaluator] done metrics={output}", flush=True)
        return output


def make_tracking_evaluator(cfg, agent, env=None):
    if env is None:
        raise ValueError("G1 tracking evaluator requires the active IsaacLab env.")
    return G1TrackingEvaluator(cfg=cfg, agent=agent, env=env)
