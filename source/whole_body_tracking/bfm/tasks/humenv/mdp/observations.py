from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_w = quat[..., :1]
    quat_vec = quat[..., 1:]
    return (
        vec * (2.0 * quat_w * quat_w - 1.0)
        + torch.cross(quat_vec, vec, dim=-1) * quat_w * 2.0
        + quat_vec * torch.sum(quat_vec * vec, dim=-1, keepdim=True) * 2.0
    )


def _quat_from_z_angle(angle: torch.Tensor) -> torch.Tensor:
    half_angle = 0.5 * angle
    quat = torch.zeros(angle.shape + (4,), device=angle.device, dtype=angle.dtype)
    quat[..., 0] = torch.cos(half_angle)
    quat[..., 3] = torch.sin(half_angle)
    return quat


def _calc_heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    ref_dir[..., 0] = 1.0
    rot_dir = _quat_rotate(quat, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return _quat_from_z_angle(-heading)


def _remove_smpl_base_rot(quat: torch.Tensor) -> torch.Tensor:
    base_rot = torch.tensor((0.5, -0.5, -0.5, -0.5), device=quat.device, dtype=quat.dtype)
    return _quat_mul(quat, base_rot.expand_as(quat))


def _quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
    ref_tan = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    ref_tan[..., 0] = 1.0
    tan = _quat_rotate(quat, ref_tan)

    ref_norm = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    ref_norm[..., -1] = 1.0
    norm = _quat_rotate(quat, ref_norm)

    return torch.cat((tan, norm), dim=-1)


def humanoid_self_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    upright_start: bool = False,
    root_height_obs: bool = True,
) -> torch.Tensor:
    """HumEnv proprioception observation ported from ``compute_humanoid_self_obs_v2``."""

    robot: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids

    body_pos = robot.data.body_pos_w[:, body_ids]
    body_rot = robot.data.body_quat_w[:, body_ids]
    body_vel = robot.data.body_lin_vel_w[:, body_ids]
    body_ang_vel = robot.data.body_ang_vel_w[:, body_ids]

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    if not upright_start:
        root_rot = _remove_smpl_base_rot(root_rot)

    heading_rot_inv = _calc_heading_quat_inv(root_rot)
    heading_rot_inv_expand = heading_rot_inv[:, None, :].expand(-1, body_pos.shape[1], -1).reshape(-1, 4)

    obs: list[torch.Tensor] = []
    if root_height_obs:
        obs.append(root_pos[:, 2:3])

    local_body_pos = body_pos - root_pos[:, None, :]
    flat_local_body_pos = local_body_pos.reshape(-1, 3)
    flat_local_body_pos = _quat_rotate(heading_rot_inv_expand, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1] * 3)
    obs.append(local_body_pos[:, 3:])

    flat_body_rot = body_rot.reshape(-1, 4)
    flat_local_body_rot = _quat_mul(heading_rot_inv_expand, flat_body_rot)
    flat_local_body_rot_obs = _quat_to_tan_norm(flat_local_body_rot)
    obs.append(flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * 6))

    flat_body_vel = body_vel.reshape(-1, 3)
    flat_local_body_vel = _quat_rotate(heading_rot_inv_expand, flat_body_vel)
    obs.append(flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * 3))

    flat_body_ang_vel = body_ang_vel.reshape(-1, 3)
    flat_local_body_ang_vel = _quat_rotate(heading_rot_inv_expand, flat_body_ang_vel)
    obs.append(flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * 3))

    return torch.cat(obs, dim=-1)
