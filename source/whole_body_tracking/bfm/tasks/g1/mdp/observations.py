from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from bfm.tasks.common.body_observations import body_self_obs_from_tensors

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def g1_lafan_self_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    return body_self_obs_from_tensors(
        robot.data.body_pos_w[:, body_ids],
        robot.data.body_quat_w[:, body_ids],
        robot.data.body_lin_vel_w[:, body_ids],
        robot.data.body_ang_vel_w[:, body_ids],
    )
