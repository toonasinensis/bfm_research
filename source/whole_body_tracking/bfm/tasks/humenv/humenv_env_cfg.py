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
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg

from bfm.robots.humenv_smpl import (
    HUMENV_ACTUATOR_BIAS0,
    HUMENV_ACTUATOR_BIAS1,
    HUMENV_ACTUATOR_BIAS2,
    HUMENV_ACTUATOR_GAIN,
    HUMENV_ACTUATOR_JOINT_NAMES,
    HUMENV_BODY_NAMES,
    HUMENV_EFFORT_LIMIT,
    HUMENV_SMPL_MJCF_CFG,
)
import bfm.tasks.humenv.mdp as mdp


@configclass
class HumEnvSceneCfg(InteractiveSceneCfg):
    """Scene for the minimal HumEnv MJCF task."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.7,
            dynamic_friction=0.7,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
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
    """HumEnv reference motion command."""

    motion = mdp.HumEnvMotionCommandCfg(
        asset_name="robot",
        body_names=HUMENV_BODY_NAMES,
        anchor_body_name="Pelvis",
    )


@configclass
class ActionsCfg:
    """Action terms matching HumEnv's XML affine actuators."""

    joint_pos = mdp.HumEnvAffineTorqueActionCfg(
        asset_name="robot",
        joint_names=HUMENV_ACTUATOR_JOINT_NAMES,
        preserve_order=True,
        gain=HUMENV_ACTUATOR_GAIN,
        bias0=HUMENV_ACTUATOR_BIAS0,
        bias1=HUMENV_ACTUATOR_BIAS1,
        bias2=HUMENV_ACTUATOR_BIAS2,
        effort_limit=HUMENV_EFFORT_LIMIT,
    )


@configclass
class ObservationsCfg:
    """Observation terms copied from HumEnv proprioception."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.humanoid_self_obs,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=HUMENV_BODY_NAMES,
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """No reset randomization for the minimal startup smoke task."""

    pass


@configclass
class RewardsCfg:
    """Zero reward placeholder. HumEnv task rewards are intentionally not ported yet."""

    zero = RewTerm(func=mdp.zero_reward, weight=1.0)


@configclass
class TerminationsCfg:
    """Only time-limit termination for the minimal task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """No curriculum for the minimal task."""

    pass


@configclass
class HumEnvMjcfEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal HumEnv MJCF environment using IsaacLab simulation."""

    scene: HumEnvSceneCfg = HumEnvSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 10.0
        self.scene.robot = HUMENV_SMPL_MJCF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.sim.dt = 1.0 / 450.0
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.viewer.eye = (2.5, -2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.9)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
