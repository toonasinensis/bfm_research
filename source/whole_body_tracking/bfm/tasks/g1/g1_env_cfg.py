from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from bfm.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from bfm.tasks.g1.defaults import DEFAULT_LAFAN_MOTIONS, G1_LAFAN_BODY_NAMES
import bfm.tasks.g1.mdp as mdp


@configclass
class G1LafanSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    motion = mdp.G1LafanMotionCommandCfg(
        asset_name="robot",
        motions=DEFAULT_LAFAN_MOTIONS,
        root_body_name="pelvis",
        anchor_body_name="torso_link",
        body_names=G1_LAFAN_BODY_NAMES,
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        scale=G1_ACTION_SCALE,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.g1_lafan_self_obs,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=G1_LAFAN_BODY_NAMES,
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    zero = RewTerm(func=mdp.zero_reward, weight=1.0)
    penalty_torques = RewTerm(func=mdp.penalty_torques, weight=1.0)
    penalty_action_rate = RewTerm(func=mdp.penalty_action_rate, weight=1.0)
    limits_dof_pos = RewTerm(func=mdp.limits_dof_pos, weight=1.0)
    limits_torque = RewTerm(func=mdp.limits_torque, weight=1.0, params={"soft_torque_limit": 0.95})
    penalty_undesired_contact = RewTerm(
        func=mdp.penalty_undesired_contact,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "pelvis",
                    "torso_link",
                    ".*_hip_.*",
                    ".*_knee_link",
                    ".*_shoulder_.*",
                    ".*_elbow_link",
                    ".*_wrist_.*",
                ],
            ),
            "threshold": 1.0,
        },
    )
    penalty_feet_ori = RewTerm(
        func=mdp.penalty_feet_ori,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"])},
    )
    penalty_ankle_roll = RewTerm(
        func=mdp.penalty_ankle_roll,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
    )
    penalty_slippage = RewTerm(
        func=mdp.penalty_slippage,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class G1LafanEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1LafanSceneCfg = G1LafanSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # self.viewer.eye = (1.5, 1.5, 1.5)
        # self.viewer.lookat = (0.0, 0.0, 0.8)
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
