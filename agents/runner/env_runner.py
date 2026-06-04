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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

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
        aux_rewards = transition.extras.get("aux_rewards")
        if aux_rewards is not None:
            data["aux_rewards"] = {
                key: value[valid].reshape(-1, 1).to(torch.float32)
                for key, value in aux_rewards.items()
                if not key.startswith("_")
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
        print(
            f"[LogHook] enabled every_steps={self.every_steps} use_wandb={self.use_wandb and wandb is not None}",
            flush=True,
        )

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
        print(f"[LogHook] step={runner.step} updates={self._updates} metrics={log_dict}", flush=True)
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

    def on_train_start(self, runner: "EnvRunner") -> None:
        del runner
        print(
            f"[CheckpointHook] enabled every_steps={self.every_steps} output={self.output_dir / self.name}",
            flush=True,
        )

    def on_step_end(self, runner: "EnvRunner", transition: Transition) -> None:
        del transition
        if runner.step % self.every_steps == 0:
            path = self.output_dir / self.name
            start_time = time.time()
            print(f"[CheckpointHook] step={runner.step} saving={path}", flush=True)
            runner.algorithm.save(path)
            print(f"[CheckpointHook] step={runner.step} save_done duration={time.time() - start_time:.2f}s", flush=True)

    def on_train_end(self, runner: "EnvRunner") -> None:
        path = self.output_dir / self.name
        start_time = time.time()
        print(f"[CheckpointHook] train_end saving={path}", flush=True)
        runner.algorithm.save(path)
        print(f"[CheckpointHook] train_end save_done duration={time.time() - start_time:.2f}s", flush=True)


class EvalHook(RunnerHook):
    """Run an optional evaluator and report flat metrics."""

    def __init__(self, every_steps: int, enabled: bool = False, use_wandb: bool = False, evaluator=None):
        self.every_steps = every_steps
        self.enabled = enabled
        self.use_wandb = use_wandb
        self.evaluator = evaluator

    def on_train_start(self, runner: "EnvRunner") -> None:
        del runner
        print(
            f"[EvalHook] enabled={self.enabled} every_steps={self.every_steps} "
            f"use_wandb={self.use_wandb and wandb is not None} evaluator={self.evaluator is not None}",
            flush=True,
        )

    def on_train_end(self, runner: "EnvRunner") -> None:
        del runner
        close = getattr(self.evaluator, "close", None)
        if callable(close):
            close()
            print("[EvalHook] evaluator closed", flush=True)

    def on_step_end(self, runner: "EnvRunner", transition: Transition) -> None:
        del transition
        if not self.enabled or self.every_steps <= 0 or runner.step % self.every_steps != 0:
            return

        if self.evaluator is None:
            metrics = {"trigger": 1.0, "not_implemented": 1.0}
            print(
                f"[EvalHook] step={runner.step} triggered, but no evaluator is attached for this backend yet.",
                flush=True,
            )
        else:
            start_time = time.time()
            metrics = _to_float_metrics(self.evaluator(runner))
            metrics.setdefault("time", time.time() - start_time)
            print(f"[EvalHook] step={runner.step} duration={metrics['time']:.2f}s metrics={metrics}", flush=True)

        if self.use_wandb and wandb is not None:
            wandb.log({f"eval/{key}": value for key, value in metrics.items()}, step=runner.step)


def _to_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            flat[key] = float(value.detach().float().mean().cpu().item())
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                continue
            flat[key] = float(np.asarray(value, dtype=np.float64).mean())
        elif isinstance(value, np.number):
            flat[key] = float(value)
        elif isinstance(value, bool):
            flat[key] = float(value)
        elif isinstance(value, int | float):
            flat[key] = float(value)
    return flat


class ProgressHook(RunnerHook):
    """Tqdm progress bar for total rollout/env steps."""

    def __init__(self, enabled: bool = True, desc: str = "rollout"):
        self.enabled = enabled
        self.desc = desc
        self._bar = None
        self._shown_step = 0

    def on_train_start(self, runner: "EnvRunner") -> None:
        total = getattr(runner, "num_env_steps", None)
        self._shown_step = min(runner.step, total or runner.step)
        print(
            f"[ProgressHook] enabled={self.enabled} total_steps={total} num_envs={runner.env.num_envs}",
            flush=True,
        )
        if not self.enabled:
            return
        if tqdm is None:
            print("[ProgressHook] tqdm is not installed; progress bar disabled.", flush=True)
            return
        self._bar = tqdm(
            total=total,
            initial=self._shown_step,
            desc=self.desc,
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )
        self._bar.set_postfix(num_envs=runner.env.num_envs)

    def on_step_end(self, runner: "EnvRunner", transition: Transition) -> None:
        del transition
        if self._bar is None:
            return
        total = getattr(runner, "num_env_steps", None)
        shown_step = min(runner.step, total) if total is not None else runner.step
        delta = shown_step - self._shown_step
        if delta > 0:
            self._bar.update(delta)
            self._shown_step = shown_step

    def on_update_end(self, runner: "EnvRunner", metrics: dict[str, torch.Tensor]) -> None:
        if self._bar is None or not metrics:
            return
        postfix = {"num_envs": runner.env.num_envs}
        for key in ("actor_loss", "critic_loss", "fb_loss", "disc_loss"):
            value = metrics.get(key)
            if value is not None:
                postfix[key] = round(value.detach().mean().item(), 4)
        self._bar.set_postfix(postfix)

    def on_train_end(self, runner: "EnvRunner") -> None:
        if self._bar is not None:
            total = getattr(runner, "num_env_steps", None)
            shown_step = min(runner.step, total) if total is not None else runner.step
            delta = shown_step - self._shown_step
            if delta > 0:
                self._bar.update(delta)
            self._bar.close()
        print(f"[ProgressHook] train_end step={runner.step}", flush=True)


class EnvRunner:
    def __init__(self, env: VecEnv, algorithm: AlgorithmAdapter, hooks: list[RunnerHook] | None = None):
        self.env = env
        self.algorithm = algorithm
        self.hooks = hooks or []
        self.step = 0
        self.num_env_steps = 0
        self._observation_refresh_requested = False

    def request_observation_refresh(self) -> None:
        self._observation_refresh_requested = True

    def train(self, num_env_steps: int) -> None:
        self.num_env_steps = num_env_steps
        obs, extras = self.env.reset()
        done = torch.zeros(self.env.num_envs, dtype=torch.bool, device=obs.device)
        for hook in self.hooks:
            hook.on_train_start(self)

        try:
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
                    extras=next_extras,
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
                if self._observation_refresh_requested:
                    obs, extras = self.env.get_observations()
                    done = torch.zeros(self.env.num_envs, dtype=torch.bool, device=obs.device)
                    self._observation_refresh_requested = False
                else:
                    obs, extras, done = next_obs, next_extras, next_done
        except Exception as exc:
            print(
                f"[EnvRunner] train_loop_exception type={type(exc).__name__} repr={exc!r} "
                f"step={self.step} target={num_env_steps}",
                flush=True,
            )
            raise

        for hook in self.hooks:
            hook.on_train_end(self)
