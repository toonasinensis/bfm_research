"""Inspect HumEnv MJCF runtime physics parameters after IsaacLab import."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect HumEnv MJCF IsaacLab physics parameters.")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--limit", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import bfm.tasks  # noqa: F401
from bfm.robots.humenv_smpl import (
    HUMENV_ACTION_OFFSET,
    HUMENV_ACTION_SCALE,
    HUMENV_ACTUATOR_DAMPING,
    HUMENV_ACTUATOR_JOINT_NAMES,
    HUMENV_ACTUATOR_STIFFNESS,
    HUMENV_DAMPING,
    HUMENV_EFFORT_LIMIT,
    HUMENV_PASSIVE_DAMPING,
    HUMENV_PASSIVE_STIFFNESS,
    HUMENV_STIFFNESS,
)
from bfm.tasks.humenv.humenv_env_cfg import HumEnvMjcfEnvCfg


def main() -> None:
    env_cfg = HumEnvMjcfEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = gym.make("HumEnv-MJCF-Flat-v0", cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    action = env.unwrapped.action_manager.get_term("joint_pos")

    print(f"num_joints={robot.num_joints} action_dim={env.unwrapped.action_manager.total_action_dim}", flush=True)
    print(f"sim_dt={env.unwrapped.cfg.sim.dt} decimation={env.unwrapped.cfg.decimation}", flush=True)
    terrain_material = env.unwrapped.cfg.scene.terrain.physics_material
    print(
        "terrain_material="
        f"static_friction={terrain_material.static_friction} "
        f"dynamic_friction={terrain_material.dynamic_friction} "
        f"combine={terrain_material.friction_combine_mode}",
        flush=True,
    )
    print(f"action_joint_order_matches_xml={action._joint_names == HUMENV_ACTUATOR_JOINT_NAMES}", flush=True)
    print(f"robot_joint_order_matches_xml={robot.joint_names == HUMENV_ACTUATOR_JOINT_NAMES}", flush=True)
    print(f"active_affine_torque_action={action.__class__.__name__ == 'HumEnvAffineTorqueAction'}", flush=True)
    print(
        "name runtime_kp runtime_kd xml_act_kp xml_pass_kp xml_total_kp "
        "xml_act_kd xml_pass_kd xml_total_kd action_scale action_offset effort_limit "
        "passive_terms_present",
        flush=True,
    )
    for i, name in enumerate(action._joint_names[: args_cli.limit]):
        jid = action._joint_ids[i] if isinstance(action._joint_ids, list) else i
        passive_terms_present = (
            abs(HUMENV_PASSIVE_STIFFNESS[name]) > 0.0 or abs(HUMENV_PASSIVE_DAMPING[name]) > 0.0
        )
        print(
            name,
            round(float(robot.data.joint_stiffness[0, jid]), 6),
            round(float(robot.data.joint_damping[0, jid]), 6),
            round(float(HUMENV_ACTUATOR_STIFFNESS[name]), 6),
            round(float(HUMENV_PASSIVE_STIFFNESS[name]), 6),
            round(float(HUMENV_STIFFNESS[name]), 6),
            round(float(HUMENV_ACTUATOR_DAMPING[name]), 6),
            round(float(HUMENV_PASSIVE_DAMPING[name]), 6),
            round(float(HUMENV_DAMPING[name]), 6),
            round(float(HUMENV_ACTION_SCALE[name]), 6),
            round(float(HUMENV_ACTION_OFFSET[name]), 6),
            round(float(HUMENV_EFFORT_LIMIT[name]), 6),
            passive_terms_present,
            flush=True,
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
