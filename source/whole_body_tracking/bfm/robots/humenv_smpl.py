from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from bfm.assets import ASSET_DIR
from bfm.utils.isaaclab_compat import MjcfFileCfg


HUMENV_XML_PATH = Path(ASSET_DIR) / "humenv" / "robot.xml"
HumEnvActuatorParams = tuple[
    list[str],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
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
    """Parse MuJoCo affine actuators into equivalent IsaacLab implicit-PD parameters."""

    root = ET.parse(xml_path).getroot()
    default_root = root.find("default")
    default_joint = default_root.find("joint").attrib if default_root is not None and default_root.find("joint") is not None else {}
    class_joint_defaults = {}
    if default_root is not None:
        for default in default_root.findall("default"):
            class_name = default.attrib.get("class")
            joint_default = default.find("joint")
            if class_name and joint_default is not None:
                class_joint_defaults[class_name] = joint_default.attrib

    def _effective_joint_float(joint_attr: dict[str, str], key: str, default: float) -> float:
        class_attr = class_joint_defaults.get(joint_attr.get("class", ""), {})
        if key in joint_attr:
            return float(joint_attr[key])
        if key in class_attr:
            return float(class_attr[key])
        if key in default_joint:
            return float(default_joint[key])
        return default

    passive_stiffness = {}
    passive_damping = {}
    passive_armature = {}
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if not name or joint.attrib.get("type") == "free":
            continue
        passive_stiffness[name] = _effective_joint_float(joint.attrib, "stiffness", default=0.0)
        passive_damping[name] = _effective_joint_float(joint.attrib, "damping", default=0.0)
        passive_armature[name] = _effective_joint_float(joint.attrib, "armature", default=0.0)

    actuator_root = root.find("actuator")
    if actuator_root is None:
        raise ValueError(f"Missing <actuator> section in {xml_path}")

    joint_names: list[str] = []
    actuator_stiffness: dict[str, float] = {}
    actuator_damping: dict[str, float] = {}
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    effort_limit: dict[str, float] = {}
    action_scale: dict[str, float] = {}
    action_offset: dict[str, float] = {}
    actuator_gain: dict[str, float] = {}
    actuator_bias0: dict[str, float] = {}
    actuator_bias1: dict[str, float] = {}
    actuator_bias2: dict[str, float] = {}

    for actuator in actuator_root.findall("general"):
        joint_name = actuator.attrib["joint"]
        gainprm = _parse_float_list(actuator.attrib["gainprm"])
        biasprm = _parse_float_list(actuator.attrib["biasprm"])
        forcerange = _parse_float_list(actuator.attrib["forcerange"])

        if len(gainprm) < 1 or len(biasprm) < 3 or len(forcerange) != 2:
            raise ValueError(f"Unsupported actuator parameters for joint {joint_name!r} in {xml_path}")

        kp_actuator = -biasprm[1]
        kd_actuator = -biasprm[2]
        if kp_actuator <= 0.0 or kd_actuator < 0.0:
            raise ValueError(f"Invalid PD gains parsed for joint {joint_name!r}: kp={kp_actuator}, kd={kd_actuator}")

        kp = kp_actuator + passive_stiffness.get(joint_name, 0.0)
        kd = kd_actuator + passive_damping.get(joint_name, 0.0)
        if kp <= 0.0 or kd < 0.0:
            raise ValueError(f"Invalid total PD gains parsed for joint {joint_name!r}: kp={kp}, kd={kd}")

        joint_names.append(joint_name)
        actuator_stiffness[joint_name] = kp_actuator
        actuator_damping[joint_name] = kd_actuator
        stiffness[joint_name] = kp
        damping[joint_name] = kd
        effort_limit[joint_name] = max(abs(forcerange[0]), abs(forcerange[1]))
        action_scale[joint_name] = gainprm[0] / kp
        action_offset[joint_name] = biasprm[0] / kp
        actuator_gain[joint_name] = gainprm[0]
        actuator_bias0[joint_name] = biasprm[0]
        actuator_bias1[joint_name] = biasprm[1]
        actuator_bias2[joint_name] = biasprm[2]

    return (
        joint_names,
        stiffness,
        damping,
        effort_limit,
        action_scale,
        action_offset,
        actuator_stiffness,
        actuator_damping,
        passive_stiffness,
        passive_damping,
        actuator_gain,
        actuator_bias0,
        actuator_bias1,
        actuator_bias2,
    )


(
    HUMENV_ACTUATOR_JOINT_NAMES,
    HUMENV_STIFFNESS,
    HUMENV_DAMPING,
    HUMENV_EFFORT_LIMIT,
    HUMENV_ACTION_SCALE,
    HUMENV_ACTION_OFFSET,
    HUMENV_ACTUATOR_STIFFNESS,
    HUMENV_ACTUATOR_DAMPING,
    HUMENV_PASSIVE_STIFFNESS,
    HUMENV_PASSIVE_DAMPING,
    HUMENV_ACTUATOR_GAIN,
    HUMENV_ACTUATOR_BIAS0,
    HUMENV_ACTUATOR_BIAS1,
    HUMENV_ACTUATOR_BIAS2,
) = _parse_humenv_actuators(HUMENV_XML_PATH)


HUMENV_SMPL_MJCF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=MjcfFileCfg(
        asset_path=str(HUMENV_XML_PATH),
        articulation_root_prim_path="/Pelvis/Pelvis",
        discard_worldbody=True,
        fix_base=False,
        import_sites=True,
        import_inertia_tensor=True,
        self_collision=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
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
            effort_limit_sim=1.0e9,
            velocity_limit_sim=50.0,
            stiffness=HUMENV_PASSIVE_STIFFNESS,
            damping=HUMENV_PASSIVE_DAMPING,
            armature=0.01,
        ),
    },
)
