from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _robot(env, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def penalty_torques(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    return torch.sum(torch.square(robot.data.applied_torque[:, joint_ids]), dim=1)


def penalty_action_rate(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def limits_dof_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_pos = robot.data.joint_pos[:, joint_ids]
    soft_limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    out_of_limits = -(joint_pos - soft_limits[..., 0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def limits_torque(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_torque_limit: float = 0.95,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    torque = torch.abs(robot.data.applied_torque[:, joint_ids])
    limits = torch.zeros_like(robot.data.applied_torque)
    for actuator in robot.actuators.values():
        limits[:, actuator.joint_indices] = actuator.effort_limit
    limits = limits[:, joint_ids]
    limits = torch.where(limits > 0, limits, torch.ones_like(limits))
    return torch.sum((torque - limits * soft_torque_limit).clip(min=0.0), dim=1)


def penalty_undesired_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.any(contact, dim=1).float()


def penalty_feet_ori(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)
    # For feet that should remain flat, the local z-axis should align with world z.
    quat = robot.data.body_quat_w[:, body_ids]
    qw, qx, qy, qz = quat.unbind(dim=-1)
    z_axis_x = 2.0 * (qx * qz + qw * qy)
    z_axis_y = 2.0 * (qy * qz - qw * qx)
    return torch.sqrt(torch.square(z_axis_x) + torch.square(z_axis_y) + 1e-8).sum(dim=1)


def penalty_ankle_roll(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    return torch.sum(torch.square(robot.data.joint_pos[:, joint_ids]), dim=1)


def penalty_slippage(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else slice(None)
    sensor_body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else body_ids
    contacts = torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_body_ids], dim=-1), dim=1)[0] > threshold
    foot_vel = torch.norm(robot.data.body_lin_vel_w[:, body_ids, :2], dim=-1)
    return torch.sum(foot_vel * contacts.float(), dim=1)
