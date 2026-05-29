from __future__ import annotations

import torch


def quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
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


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_w = quat[..., :1]
    quat_vec = quat[..., 1:]
    return (
        vec * (2.0 * quat_w * quat_w - 1.0)
        + torch.cross(quat_vec, vec, dim=-1) * quat_w * 2.0
        + quat_vec * torch.sum(quat_vec * vec, dim=-1, keepdim=True) * 2.0
    )


def quat_from_z_angle(angle: torch.Tensor) -> torch.Tensor:
    half_angle = 0.5 * angle
    quat = torch.zeros(angle.shape + (4,), device=angle.device, dtype=angle.dtype)
    quat[..., 0] = torch.cos(half_angle)
    quat[..., 3] = torch.sin(half_angle)
    return quat


def calc_heading_quat_inv(quat: torch.Tensor) -> torch.Tensor:
    ref_dir = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    ref_dir[..., 0] = 1.0
    rot_dir = quat_rotate(quat, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return quat_from_z_angle(-heading)


def quat_to_tan_norm(quat: torch.Tensor) -> torch.Tensor:
    ref_tan = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    ref_tan[..., 0] = 1.0
    tan = quat_rotate(quat, ref_tan)

    ref_norm = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    ref_norm[..., -1] = 1.0
    norm = quat_rotate(quat, ref_norm)

    return torch.cat((tan, norm), dim=-1)


def body_self_obs_from_tensors(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    *,
    root_height_obs: bool = True,
    root_quat_transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a heading-normalized body observation from ordered body tensors."""

    root_pos = body_pos_w[:, 0, :]
    root_rot = body_quat_w[:, 0, :]
    if root_quat_transform is not None:
        root_rot = quat_mul(root_rot, root_quat_transform.expand_as(root_rot))

    heading_rot_inv = calc_heading_quat_inv(root_rot)
    heading_rot_inv_expand = heading_rot_inv[:, None, :].expand(-1, body_pos_w.shape[1], -1).reshape(-1, 4)

    obs: list[torch.Tensor] = []
    if root_height_obs:
        obs.append(root_pos[:, 2:3])

    local_body_pos = body_pos_w - root_pos[:, None, :]
    flat_local_body_pos = local_body_pos.reshape(-1, 3)
    flat_local_body_pos = quat_rotate(heading_rot_inv_expand, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos_w.shape[0], body_pos_w.shape[1] * 3)
    obs.append(local_body_pos[:, 3:])

    flat_body_rot = body_quat_w.reshape(-1, 4)
    flat_local_body_rot = quat_mul(heading_rot_inv_expand, flat_body_rot)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
    obs.append(flat_local_body_rot_obs.reshape(body_quat_w.shape[0], body_quat_w.shape[1] * 6))

    flat_body_vel = body_lin_vel_w.reshape(-1, 3)
    flat_local_body_vel = quat_rotate(heading_rot_inv_expand, flat_body_vel)
    obs.append(flat_local_body_vel.reshape(body_lin_vel_w.shape[0], body_lin_vel_w.shape[1] * 3))

    flat_body_ang_vel = body_ang_vel_w.reshape(-1, 3)
    flat_local_body_ang_vel = quat_rotate(heading_rot_inv_expand, flat_body_ang_vel)
    obs.append(flat_local_body_ang_vel.reshape(body_ang_vel_w.shape[0], body_ang_vel_w.shape[1] * 3))

    return torch.cat(obs, dim=-1)
