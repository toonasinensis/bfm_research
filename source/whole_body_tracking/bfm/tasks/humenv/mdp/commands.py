from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import h5py

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def read_motion_list(motions: str | Path) -> list[str]:
    with open(motions, "r") as txtf:
        return [line.strip().replace(" ", "") for line in txtf.readlines() if line.strip()]


def canonicalize(motion: str | Path, base_path: str | Path) -> Path:
    motion = Path(motion)
    if motion.is_file():
        return motion
    candidate = Path(base_path) / motion
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot find motion file {motion!s} under root {base_path!s}.")


def load_episode_based_h5(path: str | Path, keys: Sequence[str] | None = None) -> list[dict]:
    episodes = []
    with h5py.File(path, "r") as h5f:
        for episode_key in sorted(h5f.keys()):
            group = h5f[episode_key]
            if not isinstance(group, h5py.Group):
                continue
            episode = {}
            for key, value in group.items():
                if keys is not None and key not in keys:
                    continue
                if isinstance(value, h5py.Dataset):
                    episode[key] = value[()]
            episodes.append(episode)
    return episodes


def load_motion_episode(
    motions: str | Path,
    motions_root: str | Path,
    motion_index: int = 0,
    episode_index: int = 0,
) -> tuple[Path, dict]:
    motion_files = read_motion_list(motions)
    if not motion_files:
        raise ValueError(f"No motion files listed in {motions!s}.")
    if motion_index < 0 or motion_index >= len(motion_files):
        raise IndexError(f"motion_index={motion_index} is out of range for {len(motion_files)} motion files.")

    motion_path = canonicalize(motion_files[motion_index], base_path=motions_root)
    episodes = load_episode_based_h5(motion_path)
    if not episodes:
        raise ValueError(f"No episodes found in {motion_path!s}.")
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode_index={episode_index} is out of range for {len(episodes)} episodes.")
    return motion_path, episodes[episode_index]


def motion_joint_indices_for_robot(robot_joint_names: Sequence[str], device: str | torch.device) -> torch.Tensor:
    """Return motion-qpos joint indexes ordered like the IsaacLab articulation joints."""

    from bfm.robots.humenv_smpl import HUMENV_ACTUATOR_JOINT_NAMES

    motion_joint_lookup = {name: idx for idx, name in enumerate(HUMENV_ACTUATOR_JOINT_NAMES)}
    missing = [name for name in robot_joint_names if name not in motion_joint_lookup]
    if missing:
        raise ValueError(f"Robot joints missing from HumEnv motion qpos order: {missing}")
    return torch.tensor([motion_joint_lookup[name] for name in robot_joint_names], dtype=torch.long, device=device)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_w = quat[..., :1]
    quat_vec = quat[..., 1:]
    return (
        vec * (2.0 * quat_w * quat_w - 1.0)
        + torch.cross(quat_vec, vec, dim=-1) * quat_w * 2.0
        + quat_vec * torch.sum(quat_vec * vec, dim=-1, keepdim=True) * 2.0
    )


def root_ang_vel_from_humenv_qpos_qvel(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
    """Convert HumEnv/MuJoCo freejoint angular qvel into IsaacLab world angular velocity."""

    return _quat_rotate(qpos[..., 3:7], qvel[..., 3:6])


def load_expert_trajectories(motions: str | Path, motions_root: str | Path, device: str, sequence_length: int):
    from tqdm import tqdm

    from agents.metamotivo.buffers.buffers import TrajectoryBuffer

    episodes = []
    for h5 in tqdm(read_motion_list(motions), leave=False):
        h5 = canonicalize(h5, base_path=motions_root)
        loaded_episodes = load_episode_based_h5(h5)
        for episode in loaded_episodes:
            episode["observation"] = episode["observation"].astype(np.float32)
            episode.pop("file_name", None)
        episodes.extend(loaded_episodes)

    buffer = TrajectoryBuffer(capacity=len(episodes), seq_length=sequence_length, device=device)
    buffer.extend(episodes)
    return buffer


class HumEnvMotionLoader:
    """Small h5-backed loader for HumEnv reference episodes."""

    def __init__(self, motions: str | Path, motions_root: str | Path, device: str):
        self.device = device
        self.episodes: list[dict[str, torch.Tensor]] = []
        for h5 in read_motion_list(motions):
            h5 = canonicalize(h5, base_path=motions_root)
            for episode in load_episode_based_h5(h5):
                converted = {}
                for key, value in episode.items():
                    if key == "file_name":
                        continue
                    if isinstance(value, np.ndarray):
                        converted[key] = torch.tensor(value, dtype=torch.float32, device=device)
                if "observation" in converted:
                    self.episodes.append(converted)
        if not self.episodes:
            raise ValueError(f"No HumEnv motion episodes loaded from {motions!s}.")

    def sample_starts(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_ids = torch.randint(0, len(self.episodes), (count,), device=self.device)
        time_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        for idx, episode_id in enumerate(episode_ids.tolist()):
            length = self.episodes[episode_id]["observation"].shape[0]
            time_ids[idx] = torch.randint(0, max(length - 1, 1), (1,), device=self.device)
        return episode_ids, time_ids

    def tensor_at(self, key: str, episode_ids: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor | None:
        if any(key not in self.episodes[episode_id] for episode_id in episode_ids.tolist()):
            return None
        values = []
        for episode_id, time_id in zip(episode_ids.tolist(), time_ids.tolist(), strict=True):
            episode = self.episodes[episode_id]
            safe_time = min(time_id, episode[key].shape[0] - 1)
            values.append(episode[key][safe_time])
        return torch.stack(values, dim=0)


class HumEnvMotionCommand(CommandTerm):
    cfg: HumEnvMotionCommandCfg

    def __init__(self, cfg: "HumEnvMotionCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion_joint_indices = motion_joint_indices_for_robot(self.robot.joint_names, self.device)
        self.motion: HumEnvMotionLoader | None = None
        if cfg.motions:
            self.motion = HumEnvMotionLoader(cfg.motions, cfg.motions_root, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_time"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.stack((self.motion_ids.float(), self.time_steps.float()), dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0 or self.motion is None:
            return
        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        motion_ids, time_steps = self.motion.sample_starts(len(env_ids_tensor))
        self.motion_ids[env_ids_tensor] = motion_ids
        self.time_steps[env_ids_tensor] = time_steps
        if self.cfg.reset_robot_state:
            self._reset_robot_to_motion(env_ids_tensor, motion_ids, time_steps)

    def _reset_robot_to_motion(
        self,
        env_ids: torch.Tensor,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> None:
        if self.motion is None:
            return

        qpos = self.motion.tensor_at("qpos", motion_ids, time_steps)
        qvel = self.motion.tensor_at("qvel", motion_ids, time_steps)
        if qpos is None or qvel is None:
            return

        root_state = self.robot.data.default_root_state[env_ids].clone()
        if qpos.shape[1] >= 7:
            root_state[:, :3] = qpos[:, :3] + self._env.scene.env_origins[env_ids]
            root_state[:, 3:7] = qpos[:, 3:7]
        if qvel.shape[1] >= 6:
            root_state[:, 7:10] = qvel[:, :3]
            root_state[:, 10:13] = root_ang_vel_from_humenv_qpos_qvel(qpos, qvel)

        joint_pos_count = self.robot.data.joint_pos.shape[1]
        joint_vel_count = self.robot.data.joint_vel.shape[1]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        if qpos.shape[1] >= 7 + joint_pos_count:
            joint_pos[:] = qpos[:, 7 + self.motion_joint_indices]
        if qvel.shape[1] >= 6 + joint_vel_count:
            joint_vel[:] = qvel[:, 6 + self.motion_joint_indices]

        if self.cfg.joint_position_noise > 0.0:
            joint_pos += torch.empty_like(joint_pos).uniform_(-self.cfg.joint_position_noise, self.cfg.joint_position_noise)
        if self.cfg.fall_prob > 0.0:
            fall_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.fall_prob
            root_state[fall_mask, 2] = torch.minimum(
                root_state[fall_mask, 2],
                torch.full_like(root_state[fall_mask, 2], self.cfg.fall_height),
            )

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _update_command(self):
        if self.motion is None:
            return
        self.time_steps += 1
        for motion_id in torch.unique(self.motion_ids).tolist():
            mask = self.motion_ids == motion_id
            length = self.motion.episodes[motion_id]["observation"].shape[0]
            reset_ids = torch.where(mask & (self.time_steps >= length))[0]
            self._resample_command(reset_ids)
        self.metrics["motion_id"] = self.motion_ids.float()
        self.metrics["motion_time"] = self.time_steps.float()

    def _update_metrics(self):
        self.metrics["motion_id"] = self.motion_ids.float()
        self.metrics["motion_time"] = self.time_steps.float()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class HumEnvMotionCommandCfg(CommandTermCfg):
    class_type: type = HumEnvMotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_name: str = MISSING
    motions: str = ""
    motions_root: str = ""
    anchor_body_name: str = "Pelvis"
    body_names: list[str] = MISSING
    reset_robot_state: bool = True
    fall_prob: float = 0.2
    fall_height: float = 0.35
    joint_position_noise: float = 0.0
