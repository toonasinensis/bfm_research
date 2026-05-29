from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from bfm.tasks.common.body_observations import body_self_obs_from_tensors

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def resolve_lafan_motion_files(motions: str | Path) -> list[Path]:
    import glob

    motions_path = Path(motions).expanduser()
    if motions_path.is_dir():
        files = sorted(motions_path.glob("*.npz"))
    elif motions_path.is_file() and motions_path.suffix == ".txt":
        files = []
        for line in motions_path.read_text().splitlines():
            item = line.strip()
            if not item:
                continue
            path = Path(item).expanduser()
            if not path.is_file():
                path = motions_path.parent / path
            files.append(path)
    elif motions_path.is_file() and motions_path.suffix == ".npz":
        files = [motions_path]
    else:
        files = [Path(path) for path in sorted(glob.glob(str(motions_path)))]
    if not files:
        raise FileNotFoundError(f"No LAFAN npz files found for motions={motions!s}.")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LAFAN motion files: {missing[:8]}")
    return files


def _load_lafan_npz(path: str | Path, device: str | torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path)
    episode = {}
    for key in [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]:
        if key not in data:
            raise KeyError(f"{path!s} is missing required LAFAN key {key!r}.")
        episode[key] = torch.as_tensor(data[key], dtype=torch.float32, device=device)
    episode["fps"] = torch.as_tensor(data["fps"], dtype=torch.float32, device=device)
    return episode


class G1LafanMotionLoader:
    def __init__(self, motions: str | Path, body_indexes: Sequence[int], device: str | torch.device = "cpu"):
        self.device = device
        self.files = resolve_lafan_motion_files(motions)
        self.body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=device)
        self.episodes = [_load_lafan_npz(path, device) for path in self.files]
        if not self.episodes:
            raise ValueError(f"No LAFAN episodes loaded from {motions!s}.")

    def __len__(self) -> int:
        return len(self.episodes)

    def sample_starts(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode_ids = torch.randint(0, len(self.episodes), (count,), device=self.device)
        time_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        for idx, episode_id in enumerate(episode_ids.tolist()):
            length = self.episodes[episode_id]["joint_pos"].shape[0]
            time_ids[idx] = torch.randint(0, max(length - 1, 1), (1,), device=self.device)
        return episode_ids, time_ids

    def length(self, episode_id: int) -> int:
        return int(self.episodes[episode_id]["joint_pos"].shape[0])

    def tensor_at(self, key: str, episode_ids: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        values = []
        for episode_id, time_id in zip(episode_ids.tolist(), time_ids.tolist(), strict=True):
            episode = self.episodes[episode_id]
            safe_time = min(time_id, episode[key].shape[0] - 1)
            values.append(episode[key][safe_time])
        return torch.stack(values, dim=0)

    def ordered_body_at(self, key: str, episode_ids: torch.Tensor, time_ids: torch.Tensor) -> torch.Tensor:
        values = self.tensor_at(key, episode_ids, time_ids)
        return values[:, self.body_indexes]

    def observation_episode(self, episode_id: int) -> torch.Tensor:
        episode = self.episodes[episode_id]
        return body_self_obs_from_tensors(
            episode["body_pos_w"][:, self.body_indexes],
            episode["body_quat_w"][:, self.body_indexes],
            episode["body_lin_vel_w"][:, self.body_indexes],
            episode["body_ang_vel_w"][:, self.body_indexes],
        )


def load_expert_trajectories(
    motions: str | Path,
    body_indexes: Sequence[int],
    device: str,
    sequence_length: int,
):
    from tqdm import tqdm

    from agents.metamotivo.buffers.buffers import TrajectoryBuffer

    loader = G1LafanMotionLoader(motions, body_indexes=body_indexes, device=device)
    episodes = []
    for episode_id in tqdm(range(len(loader)), leave=False):
        obs = loader.observation_episode(episode_id).detach().cpu().numpy().astype(np.float32)
        episodes.append({"observation": obs})

    buffer = TrajectoryBuffer(capacity=len(episodes), seq_length=sequence_length, device=device)
    buffer.extend(episodes)
    return buffer


class G1LafanMotionCommand(CommandTerm):
    cfg: G1LafanMotionCommandCfg

    def __init__(self, cfg: "G1LafanMotionCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.root_body_index = cfg.body_names.index(cfg.root_body_name)
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )
        self.motion = G1LafanMotionLoader(cfg.motions, self.body_indexes, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_time"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.stack((self.motion_ids.float(), self.time_steps.float()), dim=-1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.ordered_body_at("body_pos_w", self.motion_ids, self.time_steps) + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.ordered_body_at("body_quat_w", self.motion_ids, self.time_steps)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.ordered_body_at("body_lin_vel_w", self.motion_ids, self.time_steps)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.ordered_body_at("body_ang_vel_w", self.motion_ids, self.time_steps)

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        anchor_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        return self.body_pos_w[:, anchor_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        anchor_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        return self.body_quat_w[:, anchor_index]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
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
        joint_pos = self.motion.tensor_at("joint_pos", motion_ids, time_steps)
        joint_vel = self.motion.tensor_at("joint_vel", motion_ids, time_steps)
        body_pos = self.motion.ordered_body_at("body_pos_w", motion_ids, time_steps)
        body_quat = self.motion.ordered_body_at("body_quat_w", motion_ids, time_steps)
        body_lin_vel = self.motion.ordered_body_at("body_lin_vel_w", motion_ids, time_steps)
        body_ang_vel = self.motion.ordered_body_at("body_ang_vel_w", motion_ids, time_steps)

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = body_pos[:, self.root_body_index] + self._env.scene.env_origins[env_ids]
        root_state[:, 3:7] = body_quat[:, self.root_body_index]
        root_state[:, 7:10] = body_lin_vel[:, self.root_body_index]
        root_state[:, 10:13] = body_ang_vel[:, self.root_body_index]

        joint_pos = joint_pos[:, : self.robot.data.joint_pos.shape[1]].clone()
        joint_vel = joint_vel[:, : self.robot.data.joint_vel.shape[1]].clone()
        if self.cfg.joint_position_noise > 0.0:
            joint_pos += torch.empty_like(joint_pos).uniform_(-self.cfg.joint_position_noise, self.cfg.joint_position_noise)

        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _update_command(self):
        self.time_steps += 1
        reset_ids = []
        for motion_id in torch.unique(self.motion_ids).tolist():
            mask = self.motion_ids == motion_id
            reset_ids.append(torch.where(mask & (self.time_steps >= self.motion.length(motion_id)))[0])
        if reset_ids:
            self._resample_command(torch.cat(reset_ids))
        self.metrics["motion_id"] = self.motion_ids.float()
        self.metrics["motion_time"] = self.time_steps.float()

    def _update_metrics(self):
        body_pos_error = torch.norm(self.body_pos_w - self.robot_body_pos_w, dim=-1)
        self.metrics["error_body_pos"] = body_pos_error.mean(dim=-1)
        self.metrics["max_error_body_pos"] = body_pos_error.max(dim=-1).values
        self.metrics["motion_id"] = self.motion_ids.float()
        self.metrics["motion_time"] = self.time_steps.float()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_body_visualizers"):
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/G1Lafan/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/G1Lafan/expert/" + name)
                        )
                    )
            for visualizer in self.current_body_visualizers + self.goal_body_visualizers:
                visualizer.set_visibility(True)
        elif hasattr(self, "current_body_visualizers"):
            for visualizer in self.current_body_visualizers + self.goal_body_visualizers:
                visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        del event
        if not self.robot.is_initialized:
            return
        for body_idx in range(len(self.cfg.body_names)):
            self.current_body_visualizers[body_idx].visualize(
                self.robot_body_pos_w[:, body_idx],
                self.robot_body_quat_w[:, body_idx],
            )
            self.goal_body_visualizers[body_idx].visualize(
                self.body_pos_w[:, body_idx],
                self.body_quat_w[:, body_idx],
            )


@configclass
class G1LafanMotionCommandCfg(CommandTermCfg):
    class_type: type = G1LafanMotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    asset_name: str = MISSING
    motions: str = MISSING
    root_body_name: str = "pelvis"
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    reset_robot_state: bool = True
    joint_position_noise: float = 0.0
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Lafan/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
