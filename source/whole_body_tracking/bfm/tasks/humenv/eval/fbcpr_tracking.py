from __future__ import annotations

import dataclasses
import time
from collections import defaultdict
from typing import Any

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from agents.runner.env import unpack_observations


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


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


def _pose_obs(obs: torch.Tensor) -> torch.Tensor:
    return obs[:, : min(obs.shape[1], 214)]


def _distance_proximity(
    next_obs: torch.Tensor,
    tracking_target: torch.Tensor,
    bound: float = 2.0,
    margin: float = 2.0,
) -> dict[str, torch.Tensor]:
    distance = torch.norm(_pose_obs(next_obs) - _pose_obs(tracking_target), dim=-1)
    in_bounds = distance <= bound
    out_bounds = distance > bound + margin
    proximity = in_bounds.float() + ((bound + margin - distance) / margin) * (~in_bounds) * (~out_bounds)
    return {
        "distance": distance.mean(),
        "proximity": proximity.mean(),
        "success": in_bounds.all().float(),
    }


def _greedy_emd(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.numel() == 0 or rhs.numel() == 0:
        return torch.zeros((), device=lhs.device)
    distances = _distance_matrix(_pose_obs(lhs), _pose_obs(rhs))
    return 0.5 * (distances.min(dim=0).values.mean() + distances.min(dim=1).values.mean())


def _phc_metrics(next_obs: torch.Tensor, tracking_target: torch.Tensor) -> dict[str, torch.Tensor]:
    root_h_obs = 1
    local_body_pos = 69
    if next_obs.shape[1] < root_h_obs + local_body_pos or tracking_target.shape[1] < root_h_obs + local_body_pos:
        return {}

    pred = next_obs[:, root_h_obs : root_h_obs + local_body_pos]
    target = tracking_target[:, root_h_obs : root_h_obs + local_body_pos]
    output = {
        "mpjpe_g": torch.norm(target - pred, dim=1).mean() * 1000.0,
    }
    if pred.shape[0] >= 2:
        vel_target = target[1:] - target[:-1]
        vel_pred = pred[1:] - pred[:-1]
        output["vel_dist"] = torch.norm(vel_pred - vel_target, dim=1).mean() * 1000.0
    if pred.shape[0] >= 3:
        accel_target = target[:-2] - 2.0 * target[1:-1] + target[2:]
        accel_pred = pred[:-2] - 2.0 * pred[1:-1] + pred[2:]
        output["accel_dist"] = torch.norm(accel_pred - accel_target, dim=1).mean() * 1000.0

    pred_joints = pred.reshape(pred.shape[0], -1, 3)
    target_joints = target.reshape(target.shape[0], -1, 3)
    joint_error = torch.norm(pred_joints - target_joints, dim=-1)
    output["success_phc_linf"] = torch.all(joint_error <= 0.5).float()
    output["success_phc_mean"] = torch.all(joint_error.mean(dim=-1) <= 0.5).float()
    return output


def _append_metric(metrics: dict[str, list[float]], key: str, value: object) -> None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return
        metric = float(value.detach().float().mean().cpu().item())
    else:
        metric = float(value)
    if torch.isfinite(torch.tensor(metric)):
        metrics[key].append(metric)


def _append_humenv_bench_metrics(
    metrics: dict[str, list[float]],
    rollout_obs: list[torch.Tensor],
    rollout_target: list[torch.Tensor],
) -> None:
    if not rollout_obs or not rollout_target:
        return

    next_obs = torch.cat([value.detach().cpu() for value in rollout_obs], dim=0).float()
    tracking_target = torch.cat([value.detach().cpu() for value in rollout_target], dim=0).float()

    bench_metrics = {}
    bench_metrics.update(_distance_proximity(next_obs=next_obs, tracking_target=tracking_target))
    bench_metrics["emd"] = _greedy_emd(next_obs, tracking_target)
    bench_metrics.update(_phc_metrics(next_obs=next_obs, tracking_target=tracking_target))

    for key, value in bench_metrics.items():
        _append_metric(metrics, key, value)


def _set_motion_frame(command, unwrapped, motion_index: int, frame: int, env_count: int) -> torch.Tensor:
    env_ids = torch.arange(env_count, dtype=torch.long, device=unwrapped.device)
    motion_ids = torch.full((env_count,), motion_index, dtype=torch.long, device=unwrapped.device)
    time_steps = torch.full((env_count,), frame, dtype=torch.long, device=unwrapped.device)
    command.motion_ids[env_ids] = motion_ids
    command.time_steps[env_ids] = time_steps
    command._reset_robot_to_motion(env_ids, motion_ids, time_steps)
    unwrapped.scene.write_data_to_sim()
    unwrapped.sim.forward()
    return env_ids


@dataclasses.dataclass
class HumEnvTrackingEvaluator:
    cfg: Any
    agent: Any
    env: Any

    def __call__(self, runner) -> dict[str, float]:
        unwrapped = self.env.unwrapped
        command = unwrapped.command_manager.get_term("motion")
        if command.motion is None:
            raise ValueError("HumEnv tracking eval requires the active env motion loader.")

        model = self.agent._model
        previous_training = model.training
        device = torch.device(model.cfg.device)
        motion_count = len(command.motion.episodes)
        max_motions = self.cfg.tracking_eval_max_motions
        if max_motions and max_motions > 0:
            motion_count = min(motion_count, max_motions)
        motion_ids = list(range(motion_count))
        if not motion_ids:
            raise ValueError("HumEnv tracking eval found no loaded motion episodes.")
        eval_env_count = min(max(int(self.cfg.tracking_eval_num_envs), 1), unwrapped.num_envs)

        print(
            "[HumEnvTrackingEvaluator] "
            f"motions={motion_count}/{len(command.motion.episodes)} eval_envs={eval_env_count}/{unwrapped.num_envs} "
            f"mean_action={self.cfg.tracking_eval_mean_action}",
            flush=True,
        )

        start_time = time.time()
        metrics: dict[str, list[float]] = defaultdict(list)
        nan_count = 0
        total_steps = 0
        done_steps = 0
        iterator = motion_ids
        if self.cfg.tracking_eval_progress and tqdm is not None:
            iterator = tqdm(iterator, desc="humenv tracking eval", unit="motion", dynamic_ncols=True, leave=False)

        try:
            model.train(False)
            with torch.inference_mode():
                for motion_index in iterator:
                    episode = command.motion.episodes[motion_index]
                    reference_obs = episode["observation"].to(device=device, dtype=torch.float32)
                    if reference_obs.ndim != 2:
                        raise ValueError(f"Expected rank-2 reference obs, got {tuple(reference_obs.shape)}.")
                    if reference_obs.shape[1] != model.cfg.obs_dim:
                        raise ValueError(
                            f"HumEnv eval obs dim {reference_obs.shape[1]} does not match model obs dim {model.cfg.obs_dim}."
                        )
                    if reference_obs.shape[0] < 2:
                        continue

                    z_reference = model.tracking_inference(reference_obs[1:])
                    self.env.reset()
                    eval_env_ids = _set_motion_frame(command, unwrapped, motion_index, frame=0, env_count=eval_env_count)
                    obs, _ = unpack_observations(unwrapped.observation_manager.compute())
                    obs = obs[eval_env_ids].to(device=device, dtype=torch.float32)

                    rollout_steps = min(reference_obs.shape[0] - 1, z_reference.shape[0])
                    if self.cfg.tracking_eval_max_steps and self.cfg.tracking_eval_max_steps > 0:
                        rollout_steps = min(rollout_steps, self.cfg.tracking_eval_max_steps)

                    rollout_obs = []
                    rollout_target = []
                    for local_step in range(rollout_steps):
                        z = z_reference[local_step].reshape(1, -1).expand(eval_env_count, -1)
                        eval_action = model.act(obs=obs, z=z, mean=self.cfg.tracking_eval_mean_action).to(unwrapped.device)
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

                        reference_step = min(local_step + 1, reference_obs.shape[0] - 1)
                        target = reference_obs[reference_step].reshape(1, -1).expand_as(obs)
                        obs_error = obs - target
                        rollout_obs.append(obs.detach().cpu())
                        rollout_target.append(target.detach().cpu())
                        metrics["obs_distance"].append(float(torch.norm(obs_error, dim=-1).mean().item()))
                        metrics["obs_mse"].append(float(obs_error.square().mean().item()))
                        metrics["obs_emd"].append(float(_greedy_emd(obs.detach().cpu(), target.detach().cpu()).item()))
                        nan_count += _finite_count(obs, eval_action)
                        done_steps += int(done.any().item())
                        total_steps += 1
                        if bool(done.any().item()):
                            break
                    _append_humenv_bench_metrics(metrics, rollout_obs, rollout_target)
        finally:
            model.train(previous_training)

        output = {f"tracking/{key}": _mean(value) for key, value in sorted(metrics.items())}
        output["tracking/done_rate"] = float(done_steps / max(total_steps, 1))
        output["tracking/nan_count"] = float(nan_count)
        output["tracking/num_motions"] = float(len(motion_ids))
        output["tracking/num_steps"] = float(total_steps)
        output["tracking/time"] = time.time() - start_time
        runner.request_observation_refresh()
        print(f"[HumEnvTrackingEvaluator] done metrics={output}", flush=True)
        return output


def make_tracking_evaluator(cfg, agent, env=None):
    if env is None:
        raise ValueError("HumEnv tracking evaluator requires the active IsaacLab env.")
    return HumEnvTrackingEvaluator(cfg=cfg, agent=agent, env=env)
