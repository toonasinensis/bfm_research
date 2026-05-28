from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.managers.manager_term_cfg import ActionTermCfg


class HumEnvAffineTorqueAction(ActionTerm):
    """MuJoCo-style affine actuator torque for HumEnv general actuators."""

    cfg: HumEnvAffineTorqueActionCfg
    _asset: Articulation

    def __init__(self, cfg: "HumEnvAffineTorqueActionCfg", env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names,
            preserve_order=cfg.preserve_order,
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == 0:
            raise ValueError("HumEnvAffineTorqueAction resolved no joints.")

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._gain = self._resolve_parameter(cfg.gain)
        self._bias0 = self._resolve_parameter(cfg.bias0)
        self._bias1 = self._resolve_parameter(cfg.bias1)
        self._bias2 = self._resolve_parameter(cfg.bias2)
        self._effort_limit = self._resolve_parameter(cfg.effort_limit)

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        effort = self._gain * self._raw_actions + self._bias0 + self._bias1 * joint_pos + self._bias2 * joint_vel
        self._processed_actions = torch.clamp(effort, -self._effort_limit, self._effort_limit)
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def _resolve_parameter(self, value: dict[str, float] | float) -> torch.Tensor:
        if isinstance(value, (float, int)):
            return torch.full((self.num_envs, self.action_dim), float(value), device=self.device)
        tensor = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(value, self._joint_names)
        tensor[:, index_list] = torch.tensor(value_list, dtype=torch.float32, device=self.device)
        return tensor


@configclass
class HumEnvAffineTorqueActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HumEnvAffineTorqueAction
    joint_names: list[str] = MISSING
    preserve_order: bool = False
    gain: dict[str, float] | float = MISSING
    bias0: dict[str, float] | float = MISSING
    bias1: dict[str, float] | float = MISSING
    bias2: dict[str, float] | float = MISSING
    effort_limit: dict[str, float] | float = MISSING
