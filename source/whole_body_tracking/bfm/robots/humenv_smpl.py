from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from bfm.assets import ASSET_DIR


HUMENV_XML_PATH = Path(ASSET_DIR) / "humenv" / "robot.xml"
HumEnvActuatorParams = tuple[
    list[str],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]

enable_extension("isaacsim.asset.importer.mjcf")

HUMENV_BODY_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]


def _parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def _parse_humenv_actuators(xml_path: Path) -> HumEnvActuatorParams:
    """Parse MuJoCo affine actuators into IsaacLab implicit-PD parameters."""

    root = ET.parse(xml_path).getroot()
    actuator_root = root.find("actuator")
    if actuator_root is None:
        raise ValueError(f"Missing <actuator> section in {xml_path}")

    joint_names: list[str] = []
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    effort_limit: dict[str, float] = {}
    action_scale: dict[str, float] = {}
    action_offset: dict[str, float] = {}

    for actuator in actuator_root.findall("general"):
        joint_name = actuator.attrib["joint"]
        gainprm = _parse_float_list(actuator.attrib["gainprm"])
        biasprm = _parse_float_list(actuator.attrib["biasprm"])
        forcerange = _parse_float_list(actuator.attrib["forcerange"])

        if len(gainprm) < 1 or len(biasprm) < 3 or len(forcerange) != 2:
            raise ValueError(f"Unsupported actuator parameters for joint {joint_name!r} in {xml_path}")

        kp = -biasprm[1]
        kd = -biasprm[2]
        if kp <= 0.0 or kd < 0.0:
            raise ValueError(f"Invalid PD gains parsed for joint {joint_name!r}: kp={kp}, kd={kd}")

        joint_names.append(joint_name)
        stiffness[joint_name] = kp
        damping[joint_name] = kd
        effort_limit[joint_name] = max(abs(forcerange[0]), abs(forcerange[1]))
        action_scale[joint_name] = gainprm[0] / kp
        action_offset[joint_name] = biasprm[0] / kp

    return joint_names, stiffness, damping, effort_limit, action_scale, action_offset


(
    HUMENV_ACTUATOR_JOINT_NAMES,
    HUMENV_STIFFNESS,
    HUMENV_DAMPING,
    HUMENV_EFFORT_LIMIT,
    HUMENV_ACTION_SCALE,
    HUMENV_ACTION_OFFSET,
) = _parse_humenv_actuators(HUMENV_XML_PATH)


HUMENV_SMPL_MJCF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    articulation_root_prim_path="/Pelvis/Pelvis",
    spawn=sim_utils.MjcfFileCfg(
        asset_path=str(HUMENV_XML_PATH),
        fix_base=False,
        import_sites=True,
        import_inertia_tensor=True,
        self_collision=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.94),
        rot=(0.7071067811865476, 0.7071067811865475, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "humenv_pd": ImplicitActuatorCfg(
            joint_names_expr=HUMENV_ACTUATOR_JOINT_NAMES,
            effort_limit_sim=HUMENV_EFFORT_LIMIT,
            stiffness=HUMENV_STIFFNESS,
            damping=HUMENV_DAMPING,
            armature=0.01,
        ),
    },
)
