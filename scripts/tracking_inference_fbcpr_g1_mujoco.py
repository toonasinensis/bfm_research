from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import safetensors.torch
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_SOURCE_DIR = PROJECT_DIR / "source" / "whole_body_tracking"
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_SOURCE_DIR))

DEFAULT_LAFAN_MOTIONS = PROJECT_SOURCE_DIR / "bfm" / "data" / "lafan"
DEFAULT_MUJOCO_XML = Path(
    "/home/chn/hajimi/BFM-Zero/humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml"
)

G1_LAFAN_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

LAFAN_BODY_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ISAAC_TO_MUJOCO = [ISAAC_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES]
ACTION_TO_MUJOCO = ISAAC_TO_MUJOCO

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_hip_pitch_joint": -0.312,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 0.6,
}

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
BFM_ZERO_ACTION_SCALE = 0.25
BFM_ZERO_NORMALIZE_ACTION_TO = 5.0
BFM_ZERO_WAIST_STIFFNESS = 300.0
BFM_ZERO_WAIST_DAMPING = 5.0

NATURAL_FREQ = 10 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FB-CPR G1 LAFAN tracking in MuJoCo.")
    parser.add_argument("--checkpoint", type=str, default=None, help="FB-CPR checkpoint folder or model folder.")
    parser.add_argument(
        "--trace-input",
        type=str,
        default=None,
        help="IsaacLab alignment trace written by scripts/tracking_inference_fbcpr_g1.py --trace-output.",
    )
    parser.add_argument("--motions", type=str, default=str(DEFAULT_LAFAN_MOTIONS))
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=600, help="0 means run until the motion ends.")
    parser.add_argument("--xml", type=str, default=str(DEFAULT_MUJOCO_XML), help="MuJoCo scene XML.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for FB-CPR inference.")
    parser.add_argument("--mean", action="store_true", help="Use mean action instead of sampling.")
    parser.add_argument("--loop-motion", action="store_true", help="Loop the selected motion forever.")
    parser.add_argument("--iterate-motions", action="store_true", help="Visit all motions, starting at --motion-index.")
    parser.add_argument("--loop-motions", action="store_true", help="Cycle motion index after each motion ends.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--debug-vis", action="store_true", help="Draw expert body markers in the MuJoCo viewer.")
    parser.add_argument("--print-every", type=int, default=60)
    parser.add_argument("--continue-on-nan", action="store_true")
    parser.add_argument("--no-realtime", action="store_true", help="Do not sleep to match 50 Hz when the viewer is open.")
    parser.add_argument("--mode", choices=["dynamic", "kinematic-target", "state-replay"], default="dynamic")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--torque-scale", type=float, default=1.0)
    parser.add_argument("--action-scale-mult", type=float, default=1.0)
    parser.add_argument("--root-z-offset", type=float, default=0.0)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--camera-azimuth", type=float, default=140.0)
    return parser.parse_args()


def resolve_lafan_motion_files(motions: str | Path) -> list[Path]:
    import glob

    motions_path = Path(motions).expanduser()
    if motions_path.is_dir():
        files = sorted(motions_path.glob("*.npz"))
    elif motions_path.is_file() and motions_path.suffix == ".txt":
        files = []
        for line in motions_path.read_text().splitlines():
            item = line.strip()
            if not item:
                continue
            path = Path(item).expanduser()
            if not path.is_file():
                path = motions_path.parent / path
            files.append(path)
    elif motions_path.is_file() and motions_path.suffix == ".npz":
        files = [motions_path]
    else:
        files = [Path(path) for path in sorted(glob.glob(str(motions_path)))]
    if not files:
        raise FileNotFoundError(f"No LAFAN npz files found for motions={motions!s}.")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LAFAN motion files: {missing[:8]}")
    return files


def load_lafan_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    episode: dict[str, np.ndarray] = {}
    for key in [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]:
        if key not in data:
            raise KeyError(f"{path!s} is missing required LAFAN key {key!r}.")
        episode[key] = np.asarray(data[key], dtype=np.float32)
    episode["fps"] = np.asarray(data["fps"], dtype=np.float32)
    return episode


def load_alignment_trace(path: str | Path) -> dict[str, np.ndarray | dict]:
    trace_path = Path(path).expanduser()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Cannot find alignment trace {trace_path!s}.")
    raw = np.load(trace_path, allow_pickle=False)
    trace: dict[str, np.ndarray | dict] = {key: raw[key] for key in raw.files}
    metadata_value = trace.get("metadata")
    if metadata_value is not None:
        trace["metadata"] = json.loads(str(np.asarray(metadata_value).item()))
    else:
        trace["metadata"] = {}
    return trace


def _trace_strings(trace: dict[str, np.ndarray | dict], key: str) -> list[str]:
    value = trace.get(key)
    if value is None:
        return []
    return [str(item) for item in np.asarray(value).tolist()]


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
) -> torch.Tensor:
    root_pos = body_pos_w[:, 0, :]
    root_rot = body_quat_w[:, 0, :]
    heading_rot_inv = calc_heading_quat_inv(root_rot)
    heading_rot_inv_expand = heading_rot_inv[:, None, :].expand(-1, body_pos_w.shape[1], -1).reshape(-1, 4)

    obs: list[torch.Tensor] = [root_pos[:, 2:3]]

    local_body_pos = body_pos_w - root_pos[:, None, :]
    flat_local_body_pos = quat_rotate(heading_rot_inv_expand, local_body_pos.reshape(-1, 3))
    local_body_pos = flat_local_body_pos.reshape(body_pos_w.shape[0], body_pos_w.shape[1] * 3)
    obs.append(local_body_pos[:, 3:])

    flat_body_rot = body_quat_w.reshape(-1, 4)
    flat_local_body_rot = quat_mul(heading_rot_inv_expand, flat_body_rot)
    obs.append(quat_to_tan_norm(flat_local_body_rot).reshape(body_quat_w.shape[0], body_quat_w.shape[1] * 6))

    flat_body_vel = body_lin_vel_w.reshape(-1, 3)
    obs.append(quat_rotate(heading_rot_inv_expand, flat_body_vel).reshape(body_lin_vel_w.shape[0], body_lin_vel_w.shape[1] * 3))

    flat_body_ang_vel = body_ang_vel_w.reshape(-1, 3)
    obs.append(
        quat_rotate(heading_rot_inv_expand, flat_body_ang_vel).reshape(body_ang_vel_w.shape[0], body_ang_vel_w.shape[1] * 3)
    )
    return torch.cat(obs, dim=-1)


def expert_observation_episode(episode: dict[str, np.ndarray], device: str | torch.device) -> torch.Tensor:
    body_indexes = np.asarray(LAFAN_BODY_INDEXES, dtype=np.int64)
    return body_self_obs_from_tensors(
        torch.as_tensor(episode["body_pos_w"][:, body_indexes], dtype=torch.float32, device=device),
        torch.as_tensor(episode["body_quat_w"][:, body_indexes], dtype=torch.float32, device=device),
        torch.as_tensor(episode["body_lin_vel_w"][:, body_indexes], dtype=torch.float32, device=device),
        torch.as_tensor(episode["body_ang_vel_w"][:, body_indexes], dtype=torch.float32, device=device),
    )


def _resolve_model_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint path {path!s}.")
    return path / "model" if (path / "model").is_dir() else path


def _load_fbcpr_model(model_dir: Path, device: str):
    from agents.metamotivo.fb_cpr.model import FBcprModel

    with (model_dir / "config.json").open() as f:
        model_cfg = json.load(f)
    model_cfg["device"] = device
    model = FBcprModel(**model_cfg)
    missing, unexpected = safetensors.torch.load_model(
        model,
        model_dir / "model.safetensors",
        strict=False,
        device=device,
    )
    if missing or unexpected:
        print(f"[INFO] loaded model with missing_keys={len(missing)} unexpected_keys={len(unexpected)}", flush=True)
    model.train(False)
    return model


def _finite(name: str, value: torch.Tensor | np.ndarray, step: int) -> bool:
    if isinstance(value, torch.Tensor):
        if not torch.is_floating_point(value):
            return True
        bad = ~torch.isfinite(value)
        if not bool(bad.any().item()):
            return True
        flat = value.detach().reshape(-1)
        bad_flat = bad.reshape(-1)
        first_values = flat[bad_flat][:5].detach().cpu().tolist()
        count = int(bad.sum().item())
    else:
        if not np.issubdtype(value.dtype, np.floating):
            return True
        bad = ~np.isfinite(value)
        if not bool(bad.any()):
            return True
        first_values = value.reshape(-1)[bad.reshape(-1)][:5].tolist()
        count = int(bad.sum())
    print(f"[WARN] non-finite {name} step={step} count={count} first_values={first_values}", flush=True)
    return False


def _quat_angle_error(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(lhs * rhs, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _effort_limits() -> np.ndarray:
    effort = []
    for name in MUJOCO_JOINT_NAMES:
        if "knee" in name:
            effort.append(139.0)
        elif "hip_roll" in name:
            effort.append(139.0)
        elif "hip" in name:
            effort.append(88.0)
        elif "ankle" in name:
            effort.append(50.0)
        elif "waist_yaw" in name:
            effort.append(88.0)
        elif "waist" in name:
            effort.append(50.0)
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            effort.append(5.0)
        else:
            effort.append(25.0)
    return np.asarray(effort, dtype=np.float64)


def _armatures() -> np.ndarray:
    armatures = []
    for name in MUJOCO_JOINT_NAMES:
        if "hip_yaw" in name:
            armatures.append(ARMATURE_7520_14)
        elif "hip" in name or "knee" in name:
            armatures.append(ARMATURE_7520_22)
        elif "ankle" in name:
            armatures.append(2.0 * ARMATURE_5020)
        elif "waist_yaw" in name:
            armatures.append(ARMATURE_7520_14)
        elif "waist" in name:
            armatures.append(2.0 * ARMATURE_5020)
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            armatures.append(ARMATURE_4010)
        else:
            armatures.append(ARMATURE_5020)
    return np.asarray(armatures, dtype=np.float64)


def _pd_gains() -> tuple[np.ndarray, np.ndarray]:
    stiffness_5020 = ARMATURE_5020 * NATURAL_FREQ**2
    stiffness_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
    stiffness_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
    stiffness_4010 = ARMATURE_4010 * NATURAL_FREQ**2

    damping_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
    damping_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
    damping_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
    damping_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

    kp = []
    kd = []
    for name in MUJOCO_JOINT_NAMES:
        if "hip_yaw" in name:
            kp.append(stiffness_7520_14)
            kd.append(damping_7520_14)
        elif "hip" in name or "knee" in name:
            kp.append(stiffness_7520_22)
            kd.append(damping_7520_22)
        elif "ankle" in name:
            kp.append(2.0 * stiffness_5020)
            kd.append(2.0 * damping_5020)
        elif "waist" in name:
            kp.append(BFM_ZERO_WAIST_STIFFNESS)
            kd.append(BFM_ZERO_WAIST_DAMPING)
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            kp.append(stiffness_4010)
            kd.append(damping_4010)
        else:
            kp.append(stiffness_5020)
            kd.append(damping_5020)
    return np.asarray(kp, dtype=np.float64), np.asarray(kd, dtype=np.float64)


def make_controller_params(model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model.nq != 36 or model.nv != 35 or model.nu != 29:
        raise ValueError(f"Expected 29-DoF free-base G1, got nq={model.nq} nv={model.nv} nu={model.nu}.")
    xml_joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    if xml_joint_names != MUJOCO_JOINT_NAMES:
        raise ValueError(f"Unexpected MuJoCo joint order:\n{xml_joint_names}")

    default_joint_pos = np.zeros(29, dtype=np.float64)
    for idx, name in enumerate(MUJOCO_JOINT_NAMES):
        default_joint_pos[idx] = DEFAULT_JOINT_POS.get(name, 0.0)

    effort = _effort_limits()
    kp, kd = _pd_gains()
    action_scale = BFM_ZERO_NORMALIZE_ACTION_TO * BFM_ZERO_ACTION_SCALE * effort / kp
    return default_joint_pos, action_scale, kp, kd


def configure_mujoco_physics(model, torque_limits: np.ndarray) -> None:
    """Patch the generic G1 XML so its joint dynamics match the IsaacLab G1 config."""
    model.dof_armature[6:] = _armatures()
    model.dof_damping[6:] = 0.0
    model.dof_frictionloss[6:] = 0.0
    for idx, name in enumerate(MUJOCO_JOINT_NAMES):
        joint_id = model.joint(name).id
        actuator_id = model.actuator(name.removesuffix("_joint")).id
        limit = float(torque_limits[idx])
        model.jnt_actfrcrange[joint_id] = (-limit, limit)
        model.actuator_ctrlrange[actuator_id] = (-limit, limit)
        model.actuator_forcerange[actuator_id] = (-limit, limit)


def set_motion_frame(model, data, episode: dict[str, np.ndarray], frame: int, root_z_offset: float = 0.0) -> None:
    import mujoco

    data.qpos[:] = 0.0
    data.qpos[:3] = episode["body_pos_w"][frame, 0].astype(np.float64)
    data.qpos[2] += root_z_offset
    data.qpos[3:7] = episode["body_quat_w"][frame, 0].astype(np.float64)
    data.qpos[7:] = episode["joint_pos"][frame, ISAAC_TO_MUJOCO].astype(np.float64)

    data.qvel[:] = 0.0
    data.qvel[:3] = episode["body_lin_vel_w"][frame, 0].astype(np.float64)
    data.qvel[3:6] = episode["body_ang_vel_w"][frame, 0].astype(np.float64)
    data.qvel[6:] = episode["joint_vel"][frame, ISAAC_TO_MUJOCO].astype(np.float64)
    mujoco.mj_forward(model, data)


def set_trace_state(
    model,
    data,
    trace: dict[str, np.ndarray | dict],
    step: int,
    prefix: str,
    *,
    root_z_offset: float = 0.0,
) -> None:
    import mujoco

    root_pos = np.asarray(trace[f"{prefix}_root_pos"][step], dtype=np.float64).copy()
    root_pos[2] += root_z_offset
    root_quat = np.asarray(trace[f"{prefix}_root_quat"][step], dtype=np.float64)
    root_lin_vel = np.asarray(trace[f"{prefix}_root_lin_vel"][step], dtype=np.float64)
    root_ang_vel = np.asarray(trace[f"{prefix}_root_ang_vel"][step], dtype=np.float64)
    joint_pos = np.asarray(trace[f"{prefix}_joint_pos"][step], dtype=np.float64)[ISAAC_TO_MUJOCO]
    joint_vel = np.asarray(trace[f"{prefix}_joint_vel"][step], dtype=np.float64)[ISAAC_TO_MUJOCO]

    data.qpos[:] = 0.0
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[7:] = joint_pos
    data.qvel[:] = 0.0
    data.qvel[:3] = root_lin_vel
    data.qvel[3:6] = root_ang_vel
    data.qvel[6:] = joint_vel
    mujoco.mj_forward(model, data)


def compute_mujoco_body_tensors(model, data, body_ids: list[int], device: str | torch.device):
    import mujoco

    body_pos = np.asarray(data.xpos[body_ids], dtype=np.float32)
    body_quat = np.asarray(data.xquat[body_ids], dtype=np.float32)
    body_lin_vel = np.zeros((len(body_ids), 3), dtype=np.float32)
    body_ang_vel = np.zeros((len(body_ids), 3), dtype=np.float32)
    velocity = np.zeros(6, dtype=np.float64)
    for local_idx, body_id in enumerate(body_ids):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        body_ang_vel[local_idx] = velocity[:3]
        body_lin_vel[local_idx] = velocity[3:]
    return (
        torch.as_tensor(body_pos[None], dtype=torch.float32, device=device),
        torch.as_tensor(body_quat[None], dtype=torch.float32, device=device),
        torch.as_tensor(body_lin_vel[None], dtype=torch.float32, device=device),
        torch.as_tensor(body_ang_vel[None], dtype=torch.float32, device=device),
    )


def compute_mujoco_obs(model, data, body_ids: list[int], device: str | torch.device) -> torch.Tensor:
    return body_self_obs_from_tensors(*compute_mujoco_body_tensors(model, data, body_ids, device))


def compute_torque(
    action: np.ndarray,
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    torque_limits: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    action_scale_mult: float,
    torque_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    action = np.clip(action, -1.0, 1.0)
    target = default_joint_pos + action * action_scale * action_scale_mult
    torque = compute_torque_to_target(target, kp, kd, torque_limits, qpos, qvel, torque_scale=torque_scale)
    return target, torque


def compute_torque_to_target(
    target: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    torque_limits: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    torque_scale: float,
) -> np.ndarray:
    torque = kp * (target - qpos[7:]) - kd * qvel[6:]
    torque = np.clip(torque, -torque_limits, torque_limits) * torque_scale
    return torque


def draw_expert_markers(handle, body_pos: np.ndarray, rgba=(0.05, 0.85, 1.0, 0.85)) -> None:
    import mujoco

    if handle is None or handle.user_scn is None:
        return
    scene = handle.user_scn
    with handle.lock():
        scene.ngeom = 0
        for pos in body_pos:
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.asarray([0.035, 0.035, 0.035], dtype=np.float64),
                pos.astype(np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                np.asarray(rgba, dtype=np.float32),
            )
            scene.ngeom += 1


def setup_camera(handle, data, args: argparse.Namespace) -> None:
    if handle is None:
        return
    handle.cam.type = 1
    handle.cam.trackbodyid = 1
    handle.cam.distance = args.camera_distance
    handle.cam.elevation = args.camera_elevation
    handle.cam.azimuth = args.camera_azimuth
    handle.cam.lookat[:] = data.xpos[1]


def run_motion(
    *,
    args: argparse.Namespace,
    model,
    data,
    handle,
    fbcpr_model,
    motion_path: Path,
    motion_index: int,
    body_ids: list[int],
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    torque_limits: np.ndarray,
    trace: dict[str, np.ndarray | dict] | None = None,
) -> int:
    import mujoco

    if trace is None:
        if fbcpr_model is None:
            raise ValueError("--checkpoint is required when --trace-input is not used.")
        episode = load_lafan_npz(motion_path)
        motion_length = int(episode["joint_pos"].shape[0])
        start_frame = max(0, min(args.start_frame, motion_length - 2))
        end_frame = motion_length if args.num_steps <= 0 else min(start_frame + args.num_steps + 1, motion_length)
        if end_frame <= start_frame + 1:
            raise ValueError(f"Motion {motion_index} is too short for start_frame={start_frame}.")

        reference_obs = expert_observation_episode(episode, fbcpr_model.cfg.device).to(dtype=torch.float32)
        if reference_obs.shape[1] != fbcpr_model.cfg.obs_dim:
            raise ValueError(f"Reference obs dim {reference_obs.shape[1]} does not match model obs dim {fbcpr_model.cfg.obs_dim}.")
        with torch.inference_mode():
            z_reference = fbcpr_model.tracking_inference(reference_obs[start_frame + 1 : end_frame])
        set_motion_frame(model, data, episode, start_frame, args.root_z_offset)
    else:
        metadata = trace.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        trace_steps = int(np.asarray(trace["action"]).shape[0])
        start_frame = int(metadata.get("start_frame", 0))
        end_frame = start_frame + trace_steps + 1
        motion_length = end_frame
        motion_path = Path(str(metadata.get("motion_file", motion_path)))
        device = fbcpr_model.cfg.device if fbcpr_model is not None else args.device
        reference_obs = torch.as_tensor(np.asarray(trace["reference_obs"]), dtype=torch.float32, device=device)
        z_reference = torch.as_tensor(np.asarray(trace["z"]), dtype=torch.float32, device=device)
        set_trace_state(model, data, trace, 0, "before", root_z_offset=args.root_z_offset)
    setup_camera(handle, data, args)
    obs_device = fbcpr_model.cfg.device if fbcpr_model is not None else args.device
    obs_tensor = compute_mujoco_obs(model, data, body_ids, obs_device)

    expert_now = (
        reference_obs[start_frame].reshape(1, -1)
        if trace is None
        else torch.as_tensor(np.asarray(trace["obs_before"])[0], dtype=torch.float32, device=obs_device).reshape(1, -1)
    )
    obs_mse0 = (obs_tensor - expert_now).square().mean().item()
    body_pos, body_quat, _, _ = compute_mujoco_body_tensors(model, data, body_ids, obs_device)
    if trace is None:
        expert_body_pos_np = episode["body_pos_w"][start_frame, LAFAN_BODY_INDEXES][None]
        expert_body_quat_np = episode["body_quat_w"][start_frame, LAFAN_BODY_INDEXES][None]
    else:
        expert_body_pos_np = np.asarray(trace["before_body_pos"])[0][None]
        expert_body_quat_np = np.asarray(trace["before_body_quat"])[0][None]
    expert_body_pos = torch.as_tensor(expert_body_pos_np, dtype=torch.float32, device=obs_device)
    expert_body_quat = torch.as_tensor(expert_body_quat_np, dtype=torch.float32, device=obs_device)
    pos_err0 = torch.norm(body_pos - expert_body_pos, dim=-1)
    rot_err0 = _quat_angle_error(body_quat, expert_body_quat)

    print(
        f"[INFO] motion={motion_path} index={motion_index} frames=[{start_frame}, {end_frame}) "
        f"obs={tuple(reference_obs.shape)} z={tuple(z_reference.shape)} mode={args.mode} mean={args.mean} "
        f"trace={trace is not None}",
        flush=True,
    )
    print(
        f"[INFO] reset parity obs_mse={obs_mse0:.8f} body_pos_mean={pos_err0.mean().item():.8f} "
        f"body_pos_max={pos_err0.max().item():.8f} body_rot_mean={rot_err0.mean().item():.8f}",
        flush=True,
    )

    step = 0
    while args.headless or (handle is not None and handle.is_running()):
        local = step % z_reference.shape[0] if args.loop_motion else step
        if local >= z_reference.shape[0]:
            break

        frame = min(start_frame + local + 1, motion_length - 1)
        if args.debug_vis and handle is not None and trace is None:
            draw_expert_markers(handle, episode["body_pos_w"][frame, LAFAN_BODY_INDEXES])
        elif args.debug_vis and handle is not None and trace is not None:
            draw_expert_markers(handle, np.asarray(trace["expert_body_pos"])[local])

        z = z_reference[local].reshape(1, -1)
        if not _finite("obs", obs_tensor, step) and not args.continue_on_nan:
            break
        if not _finite("z", z, step) and not args.continue_on_nan:
            break

        if trace is None:
            with torch.inference_mode():
                action = fbcpr_model.act(obs=obs_tensor, z=z, mean=args.mean)
            if not _finite("action", action, step) and not args.continue_on_nan:
                break
            action_np = action.detach().cpu().numpy().reshape(-1)
            target_mujoco = None
        else:
            action_np = np.asarray(trace["action"])[local].reshape(-1).astype(np.float64)
            target_mujoco = np.asarray(trace["processed_action"])[local].reshape(-1)[ISAAC_TO_MUJOCO].astype(np.float64)
        action_mujoco = action_np[ACTION_TO_MUJOCO]

        t0 = time.time()
        if target_mujoco is None:
            target, torque = compute_torque(
                action_mujoco,
                default_joint_pos,
                action_scale,
                kp,
                kd,
                torque_limits,
                data.qpos,
                data.qvel,
                action_scale_mult=args.action_scale_mult,
                torque_scale=args.torque_scale,
            )
        else:
            target = target_mujoco
            torque = compute_torque_to_target(
                target,
                kp,
                kd,
                torque_limits,
                data.qpos,
                data.qvel,
                torque_scale=args.torque_scale,
            )
        if args.mode == "state-replay":
            if trace is None:
                raise ValueError("--mode state-replay requires --trace-input.")
            set_trace_state(model, data, trace, local, "after", root_z_offset=args.root_z_offset)
        elif args.mode == "kinematic-target":
            data.qpos[7:] = target
            data.qvel[6:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            for _ in range(max(args.decimation, 1)):
                torque = compute_torque_to_target(
                    target,
                    kp,
                    kd,
                    torque_limits,
                    data.qpos,
                    data.qvel,
                    torque_scale=args.torque_scale,
                )
                data.ctrl[:] = torque
                mujoco.mj_step(model, data)

        obs_tensor = compute_mujoco_obs(model, data, body_ids, obs_device)
        if trace is None:
            obs_target = reference_obs[frame].reshape(1, -1)
            isaac_after = None
        else:
            obs_target = torch.as_tensor(
                np.asarray(trace["reference_obs"])[local],
                dtype=torch.float32,
                device=obs_device,
            ).reshape(1, -1)
            isaac_after = torch.as_tensor(
                np.asarray(trace["obs_after"])[local],
                dtype=torch.float32,
                device=obs_device,
            ).reshape(1, -1)
        obs_mse = (obs_tensor - obs_target).square().mean()
        isaac_obs_mse = (obs_tensor - isaac_after).square().mean() if isaac_after is not None else None

        body_pos, body_quat, _, _ = compute_mujoco_body_tensors(model, data, body_ids, obs_device)
        if trace is None:
            expert_body_pos_np = episode["body_pos_w"][frame, LAFAN_BODY_INDEXES][None]
            expert_body_quat_np = episode["body_quat_w"][frame, LAFAN_BODY_INDEXES][None]
        else:
            expert_body_pos_np = np.asarray(trace["expert_body_pos"])[local][None]
            expert_body_quat_np = np.asarray(trace["expert_body_quat"])[local][None]
        expert_body_pos = torch.as_tensor(expert_body_pos_np, dtype=torch.float32, device=obs_device)
        expert_body_quat = torch.as_tensor(expert_body_quat_np, dtype=torch.float32, device=obs_device)
        body_pos_error = torch.norm(body_pos - expert_body_pos, dim=-1)
        body_rot_error = _quat_angle_error(body_quat, expert_body_quat)

        if not _finite("obs_after_step", obs_tensor, step) and not args.continue_on_nan:
            break
        if not _finite("qpos", data.qpos, step) and not args.continue_on_nan:
            break

        if step == 0 or (args.print_every > 0 and (step + 1) % args.print_every == 0):
            isaac_obs_msg = f" isaac_obs_mse={isaac_obs_mse.item():.6f}" if isaac_obs_mse is not None else ""
            print(
                f"[INFO] step={step + 1} frame={frame} sim_time={data.time:.3f} "
                f"obs_mse={obs_mse.item():.6f} body_pos_mean={body_pos_error.mean().item():.6f} "
                f"body_pos_max={body_pos_error.max().item():.6f} body_rot_mean={body_rot_error.mean().item():.6f} "
                f"action_abs_max={float(np.max(np.abs(action_np))):.4f} torque_abs_max={float(np.max(np.abs(torque))):.2f}"
                f"{isaac_obs_msg}",
                flush=True,
            )

        if handle is not None:
            handle.cam.lookat[:] = data.xpos[1]
            handle.sync()
            if not args.no_realtime:
                elapsed = time.time() - t0
                time.sleep(max(0.0, 0.02 - elapsed))
        step += 1
    print(f"[INFO] completed motion={motion_index} steps={step}", flush=True)
    return step


def main() -> None:
    args = parse_args()

    import mujoco

    if args.checkpoint is None and args.trace_input is None:
        raise ValueError("Provide --checkpoint for policy rollout or --trace-input for IsaacLab trace replay.")

    fbcpr_model = None
    if args.checkpoint is not None:
        model_dir = _resolve_model_dir(args.checkpoint)
        print(f"[INFO] loading FB-CPR model from {model_dir}", flush=True)
        fbcpr_model = _load_fbcpr_model(model_dir, device=args.device)

    trace = load_alignment_trace(args.trace_input) if args.trace_input else None
    if trace is not None:
        metadata = trace.get("metadata", {})
        if isinstance(metadata, dict):
            args.dt = float(metadata.get("dt", args.dt))
            args.decimation = int(metadata.get("decimation", args.decimation))
        trace_joint_names = _trace_strings(trace, "joint_names")
        if trace_joint_names and trace_joint_names != ISAAC_JOINT_NAMES:
            raise ValueError(f"Unexpected Isaac joint names in trace:\n{trace_joint_names}")
        trace_action_joint_names = _trace_strings(trace, "action_joint_names")
        if trace_action_joint_names and trace_action_joint_names != ISAAC_JOINT_NAMES:
            raise ValueError(f"Unexpected Isaac action joint names in trace:\n{trace_action_joint_names}")
        trace_body_names = _trace_strings(trace, "body_names")
        if trace_body_names and trace_body_names != G1_LAFAN_BODY_NAMES:
            raise ValueError(f"Unexpected body names in trace:\n{trace_body_names}")
        print(f"[INFO] loaded IsaacLab alignment trace: {args.trace_input}", flush=True)

    xml_path = Path(args.xml).expanduser()
    if not xml_path.is_file():
        raise FileNotFoundError(f"Cannot find MuJoCo XML {xml_path!s}.")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    body_ids = [model.body(name).id for name in G1_LAFAN_BODY_NAMES]
    default_joint_pos, action_scale, kp, kd = make_controller_params(model)
    torque_limits = _effort_limits()
    configure_mujoco_physics(model, torque_limits)

    obs_dim = fbcpr_model.cfg.obs_dim if fbcpr_model is not None else int(np.asarray(trace["obs_before"]).shape[1])
    action_dim = fbcpr_model.cfg.action_dim if fbcpr_model is not None else int(np.asarray(trace["action"]).shape[1])
    print(
        f"[INFO] MuJoCo xml={xml_path} nq={model.nq} nv={model.nv} nu={model.nu} "
        f"obs_dim={obs_dim} action_dim={action_dim} dt={model.opt.timestep} decimation={args.decimation}",
        flush=True,
    )
    if action_dim != model.nu:
        raise ValueError(f"Action dim={action_dim} does not match MuJoCo nu={model.nu}.")

    if trace is None:
        motion_files = resolve_lafan_motion_files(args.motions)
        if args.motion_index < 0 or args.motion_index >= len(motion_files):
            raise IndexError(f"motion_index={args.motion_index} out of range for {len(motion_files)} motions.")
    else:
        metadata = trace.get("metadata", {})
        motion_files = [Path(str(metadata.get("motion_file", args.motions)) if isinstance(metadata, dict) else str(args.motions))]
        args.motion_index = 0

    handle = None
    if not args.headless:
        import mujoco.viewer

        handle = mujoco.viewer.launch_passive(model, data)

    try:
        motion_index = args.motion_index
        while True:
            run_motion(
                args=args,
                model=model,
                data=data,
                handle=handle,
                fbcpr_model=fbcpr_model,
                motion_path=motion_files[motion_index],
                motion_index=motion_index,
                body_ids=body_ids,
                default_joint_pos=default_joint_pos,
                action_scale=action_scale,
                kp=kp,
                kd=kd,
                torque_limits=torque_limits,
                trace=trace,
            )
            if args.loop_motion:
                continue
            if not args.iterate_motions and not args.loop_motions:
                break
            motion_index += 1
            if motion_index >= len(motion_files):
                if not args.loop_motions:
                    break
                motion_index = 0
            if handle is not None and not handle.is_running():
                break
    finally:
        if handle is not None:
            handle.close()


if __name__ == "__main__":
    main()
