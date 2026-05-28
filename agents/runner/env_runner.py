from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from .env import VecEnv


@dataclasses.dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    extras: dict
    next_obs: torch.Tensor
    step_count: torch.Tensor
    valid: torch.Tensor


class AlgorithmAdapter(Protocol):
    def act(self, obs: torch.Tensor, extras: dict, step: int) -> tuple[torch.Tensor, dict]:
        ...

    def record_transition(self, transition: Transition, rollout_context: dict) -> None:
        ...

    def ready_to_update(self, step: int) -> bool:
        ...

    def update(self, step: int) -> dict[str, torch.Tensor]:
        ...

    def save(self, path: str | Path) -> None:
        ...


class RunnerHook:
    def on_train_start(self, runner: "EnvRunner") -> None:
        pass

    def on_step_end(self, runner: "EnvRunner", transition: Transition) -> None:
        pass

    def on_update_end(self, runner: "EnvRunner", metrics: dict[str, torch.Tensor]) -> None:
        pass

    def on_train_end(self, runner: "EnvRunner") -> None:
        pass


class FBcprAdapter:
    """FB-CPR adapter for the generic environment runner."""

    def __init__(
        self,
        agent,
        replay_buffer: dict[str, Any],
        num_seed_steps: int,
        update_agent_every: int,
        num_agent_updates: int,
        action_dim: int,
        action_device: torch.device | str,
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.num_seed_steps = num_seed_steps
        self.update_agent_every = update_agent_every
        self.num_agent_updates = num_agent_updates
        self.action_dim = action_dim
        self.action_device = action_device
        self._context: torch.Tensor | None = None

    def act(self, obs: torch.Tensor, extras: dict, step: int) -> tuple[torch.Tensor, dict]:
        step_count = extras["step_count"].to(self.agent.device)
        with torch.no_grad():
            agent_obs = obs.to(self.agent.device, dtype=torch.float32)
            self._context = self.agent.maybe_update_rollout_context(z=self._context, step_count=step_count)
            if step < self.num_seed_steps:
                action = torch.rand(obs.shape[0], self.action_dim, device=self.action_device) * 2.0 - 1.0
            else:
                action = self.agent.act(obs=agent_obs, z=self._context, mean=False).detach().to(self.action_device)
        return action.float(), {"z": self._context.detach(), "step_count": step_count.detach()}

    def record_transition(self, transition: Transition, rollout_context: dict) -> None:
        valid = transition.valid.to(transition.obs.device)
        if not torch.any(valid):
            return
        z = rollout_context["z"].to(self.agent.device)
        step_count = rollout_context["step_count"].to(self.agent.device)
        agent_valid = transition.valid.to(self.agent.device)
        data = {
            "observation": transition.obs[valid].to(torch.float32),
            "action": transition.action[valid].to(torch.float32),
            "z": z[agent_valid].to(torch.float32),
            "step_count": step_count[agent_valid],
            "next": {
                "observation": transition.next_obs[valid].to(torch.float32),
                "terminated": transition.terminated[valid].reshape(-1, 1).bool(),
                "truncated": transition.truncated[valid].reshape(-1, 1).bool(),
                "reward": transition.reward[valid].reshape(-1, 1).to(torch.float32),
            },
        }
        self.replay_buffer["train"].extend(data)

    def ready_to_update(self, step: int) -> bool:
        return (
            len(self.replay_buffer["train"]) > 0
            and step > self.num_seed_steps
            and step % self.update_agent_every == 0
        )

    def update(self, step: int) -> dict[str, torch.Tensor]:
        total_metrics: dict[str, torch.Tensor] | None = None
        for _ in range(self.num_agent_updates):
            metrics = self.agent.update(self.replay_buffer, step)
            if total_metrics is None:
                total_metrics = {key: value.detach().clone() for key, value in metrics.items()}
            else:
                total_metrics = {key: total_metrics[key] + value.detach() for key, value in metrics.items()}
        if total_metrics is None:
            return {}
        return {key: value / self.num_agent_updates for key, value in total_metrics.items()}

    def save(self, path: str | Path) -> None:
        self.agent.save(str(path))


class LogHook(RunnerHook):
    def __init__(self, every_steps: int, use_wandb: bool = False):
        self.every_steps = every_steps
        self.use_wandb = use_wandb
        self._metrics: dict[str, torch.Tensor] = {}
        self._updates = 0
        self._last_log_step = 0
        self._last_log_time = time.time()
        self._start_time = time.time()

    def on_train_start(self, runner: "EnvRunner") -> None:
        self._last_log_step = runner.step
        self._last_log_time = time.time()
        self._start_time = time.time()

    def on_update_end(self, runner: "EnvRunner", metrics: dict[str, torch.Tensor]) -> None:
        if not metrics:
            return
        self._updates += 1
        for key, value in metrics.items():
            self._metrics[key] = self._metrics.get(key, torch.zeros_like(value.detach())) + value.detach()

    def on_step_end(self, runner: "EnvRunner", transition: Transition) -> None:
        del transition
        if self._updates == 0 or runner.step % self.every_steps != 0:
            return
        now = time.time()
        elapsed_steps = max(runner.step - self._last_log_step, runner.env.num_envs)
        fps = elapsed_steps / max(now - self._last_log_time, 1e-6)
        log_dict = {
            key: np.round((value / self._updates).mean().item(), 6)
            for key, value in sorted(self._metrics.items())
        }
        log_dict["duration [minutes]"] = (now - self._start_time) / 60.0
        log_dict["FPS"] = fps
        print(log_dict, flush=True)
        if self.use_wandb and wandb is not None:
            wandb.log({f"train/{key}": value for key, value in log_dict.items()}, step=runner.step)
        self._metrics.clear()
        self._updates = 0
        self._last_log_step = runner.step
        self._last_log_time = now


class CheckpointHook(RunnerHook):
    def __init__(self, every_steps: int, output_dir: str | Path, name: str = "checkpoint"):
        self.every_steps = every_steps
        self.output_dir = Path(output_dir)
        self.name = name

    def on_step_end(self, runner: "EnvRunner", transition: Transition) -> None:
        del transition
        if runner.step % self.every_steps == 0:
            runner.algorithm.save(self.output_dir / self.name)

    def on_train_end(self, runner: "EnvRunner") -> None:
        runner.algorithm.save(self.output_dir / self.name)


class EvalHook(RunnerHook):
    """Placeholder hook for future HumEnv/IsaacLab evaluation."""

    def __init__(self, every_steps: int, enabled: bool = False):
        self.every_steps = every_steps
        self.enabled = enabled

    def on_step_end(self, runner: "EnvRunner", transition: Transition) -> None:
        del transition
        if self.enabled and runner.step % self.every_steps == 0:
            print(f"[EvalHook] step={runner.step}: evaluation is not implemented in this backend yet.", flush=True)


class EnvRunner:
    def __init__(self, env: VecEnv, algorithm: AlgorithmAdapter, hooks: list[RunnerHook] | None = None):
        self.env = env
        self.algorithm = algorithm
        self.hooks = hooks or []
        self.step = 0

    def train(self, num_env_steps: int) -> None:
        obs, extras = self.env.reset()
        done = torch.zeros(self.env.num_envs, dtype=torch.bool, device=obs.device)
        for hook in self.hooks:
            hook.on_train_start(self)

        while self.step < num_env_steps:
            step_count = self.env.episode_length_buf.clone().to(obs.device).reshape(-1, 1)
            policy_extras = dict(extras)
            policy_extras["step_count"] = step_count
            action, rollout_context = self.algorithm.act(obs, policy_extras, self.step)
            next_obs, reward, next_done, next_extras = self.env.step(action)
            terminated = next_extras.get("terminated", next_done)
            truncated = next_extras.get("truncated", torch.zeros_like(next_done))
            transition = Transition(
                obs=obs,
                action=action,
                reward=reward,
                done=next_done,
                terminated=terminated,
                truncated=truncated,
                extras=extras,
                next_obs=next_obs,
                step_count=step_count,
                valid=torch.logical_not(done).reshape(-1),
            )
            self.algorithm.record_transition(transition, rollout_context)

            self.step += self.env.num_envs
            if self.algorithm.ready_to_update(self.step):
                metrics = self.algorithm.update(self.step)
                for hook in self.hooks:
                    hook.on_update_end(self, metrics)

            for hook in self.hooks:
                hook.on_step_end(self, transition)
            obs, extras, done = next_obs, next_extras, next_done

        for hook in self.hooks:
            hook.on_train_end(self)
