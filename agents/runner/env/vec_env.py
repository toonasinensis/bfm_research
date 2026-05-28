from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch


class VecEnv(ABC):
    """Thin vectorized environment interface used by runners."""

    num_envs: int
    num_actions: int
    max_episode_length: int | torch.Tensor
    episode_length_buf: torch.Tensor
    device: torch.device | str
    cfg: dict | object

    @abstractmethod
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError


def _flatten_leaf(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor observation leaf, got {type(value)!r}.")
    return value.reshape(value.shape[0], -1)


def flatten_observation(value: Any) -> torch.Tensor:
    """Flatten TensorDict-like observations into one actor observation tensor."""

    if isinstance(value, torch.Tensor):
        return _flatten_leaf(value)
    if isinstance(value, Mapping):
        leaves = [flatten_observation(item) for item in value.values()]
        if not leaves:
            raise ValueError("Cannot flatten an empty observation mapping.")
        return torch.cat(leaves, dim=-1)
    raise TypeError(f"Unsupported observation type {type(value)!r}.")


def unpack_observations(obs: Any, extras: dict | None = None) -> tuple[torch.Tensor, dict]:
    """Extract actor observations and preserve all non-actor groups in extras."""

    extras = {} if extras is None else dict(extras)
    extras.setdefault("observations", {})

    if isinstance(obs, Mapping):
        actor_key = "actor" if "actor" in obs else "policy" if "policy" in obs else None
        if actor_key is not None:
            actor_obs = flatten_observation(obs[actor_key])
            for key, value in obs.items():
                if key != actor_key:
                    extras["observations"][key] = flatten_observation(value)
            return actor_obs, extras

    return flatten_observation(obs), extras


class IsaacLabVecEnvAdapter(VecEnv):
    """Adapt an IsaacLab/Gymnasium vectorized env to the local runner interface."""

    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped
        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self.unwrapped.device
        self.cfg = self.unwrapped.cfg
        self.episode_length_buf = self.unwrapped.episode_length_buf
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = int(self.unwrapped.action_manager.total_action_dim)

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        if hasattr(self.unwrapped, "get_observations"):
            return unpack_observations(self.unwrapped.get_observations())
        obs, extras = self.env.reset()
        return unpack_observations(obs, extras)

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, extras = self.env.reset()
        self.episode_length_buf = self.unwrapped.episode_length_buf
        return unpack_observations(obs, extras)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, rewards, terminated, truncated, extras = self.env.step(actions)
        done = torch.logical_or(terminated, truncated)
        extras = dict(extras)
        extras["terminated"] = terminated
        extras["truncated"] = truncated
        self.episode_length_buf = self.unwrapped.episode_length_buf
        next_obs, extras = unpack_observations(obs, extras)
        return next_obs, rewards, done, extras

    def close(self) -> None:
        self.env.close()
