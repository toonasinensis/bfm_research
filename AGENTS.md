# Codex Runbook

Always read this file before running IsaacLab/HumEnv commands in this repo.

Also read `docs/fbcpr_system_notes.md` before touching FB-CPR training, G1, tracking inference, MuJoCo rollout, or output-directory conventions. It is the long-lived context for the current port.

## Environment

- Use the `mimic` conda environment.
- Do not manually source `/home/chn/isaacsim450/setup_conda_env.sh` after activating `mimic`.
- `mimic` already has `/home/chn/miniforge3/envs/mimic/etc/conda/activate.d/setenv.sh`, which sets:
  - `ISAACLAB_PATH=/home/chn/IsaacLab`
  - `ISAAC_PATH=/home/chn/IsaacLab/_isaac_sim`
  - IsaacSim `PYTHONPATH` / `LD_LIBRARY_PATH`
- Correct command prefix:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
cd /home/chn/hajimi/bfm_research
```

## Quick Checks

Use this to check the environment without launching simulation:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python -c "import torch, gymnasium, h5py; print(torch.__version__, torch.cuda.is_available())"
```

IsaacSim extension imports like `isaacsim.core` may require launching through `AppLauncher`; do not spend time trying to validate every extension with a raw Python import.

For online W&B curves, the `mimic` environment must have a configured API key:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python -c "import wandb; print(bool(wandb.api.api_key))"
```

If it prints `False`, run `wandb login` or export `WANDB_API_KEY` before starting training. The training script will still run without it, but `--use-wandb` is disabled after a warning.

## HumEnv Data

Training motion list:

```text
/home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt
```

Motion root:

```text
/home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass
```

The txt file contains relative `.hdf5` names, so always pass both `--motions` and `--motions-root`.

G1 LAFAN training data:

```text
/home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan
```

This directory contains `.npz` episodes with `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`,
`body_lin_vel_w`, and `body_ang_vel_w`.

## Current FB-CPR Smoke Command

Run from `/home/chn/hajimi/bfm_research`:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/train_fbcpr_humenv.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --online-parallel-envs 1 \
  --num-env-steps 80 \
  --num-seed-steps 16 \
  --update-agent-every 16 \
  --num-agent-updates 1 \
  --log-every-updates 16 \
  --checkpoint-every-steps 80 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cpu \
  --use-wandb
```

Use larger values only after this smoke command runs.

## Vectorized Smoke

This has been verified with 16 environments:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/train_fbcpr_humenv.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --online-parallel-envs 16 \
  --num-env-steps 320 \
  --num-seed-steps 64 \
  --update-agent-every 64 \
  --num-agent-updates 2 \
  --log-every-updates 64 \
  --checkpoint-every-steps 320 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cpu
```

For an online run after W&B is logged in, add `--use-wandb` and scale `--online-parallel-envs`, `--num-env-steps`, and checkpoint/log intervals together.

## G1 FB-CPR

G1 uses the same FB-CPR runner/adapter as HumEnv. The task-specific pieces live under
`source/whole_body_tracking/bfm/tasks/g1`: `G1LafanEnvCfg`, `G1LafanMotionCommand`, the LAFAN loader, and the G1
self-observation. The expert observation and online observation are both built by the shared
`body_self_obs_from_tensors()` helper, so body-order bugs are easier to catch.

Before training G1 after touching body names, robot import, or the LAFAN loader, run the parity check:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/check_g1_lafan_obs_parity.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --frame 0 \
  --num-envs 1 \
  --device cuda \
  --output /tmp/g1_lafan_parity.npz
```

Expected signs from the current implementation:

```text
obs_shape=(1, 208)
obs mse=0.00000000
body_pos mean~=0 max~=0
robot_name=<name> cfg_name=<same name>
```

Use `--debug-vis` without `--headless` to show current robot body frames and expert body frames in the viewer. GUI
debug keeps the viewer open by default. Add `--play-motion` to animate the LAFAN frames, or `--hold` to keep a
single frame open explicitly:

```bash
python scripts/check_g1_lafan_obs_parity.py \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --frame 0 \
  --num-envs 1 \
  --device cuda \
  --debug-vis \
  --play-motion \
  --print-every 120
```

G1 smoke training command:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/train_fbcpr_g1.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --online-parallel-envs 1 \
  --num-env-steps 80 \
  --num-seed-steps 16 \
  --update-agent-every 16 \
  --num-agent-updates 1 \
  --log-every-updates 16 \
  --checkpoint-every-steps 80 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cpu
```

This smoke has been verified to load 40 LAFAN episodes, run FB-CPR updates, and write
`checkpoint/model/model.safetensors`.

## HumEnv Reset Parity

MetaMotivo trains with `state_init="MoCapAndFall"` and `fall_prob=0.2`. The IsaacLab port matches this through
`HumEnvMotionCommandCfg.reset_robot_state=True` and `fall_prob=0.2`, which reset the robot from motion `qpos/qvel`
at env reset. Keep this on for training unless you are deliberately running an ablation:

```bash
python scripts/train_fbcpr_humenv.py ... --no-motion-reset
```

A quick reset check previously showed the motion-reset observation much closer to the sampled reference
(`mse ~= 0.19`) than the no-reset path (`mse ~= 0.53`).

## HumEnv Physics Parity

The HumEnv MJCF uses MuJoCo affine actuators plus passive joint `stiffness`/`damping`/`armature` from the XML. The
IsaacLab port should keep these two parts separate:

```text
active_tau = clip(gain * action + bias0 + bias1 * q + bias2 * qd, -forcerange, forcerange)
passive_tau = -K_passive * q - D_passive * qd
```

Do not fold passive stiffness/damping into one implicit PD drive with `effort_limit_sim=forcerange`: that clips the
passive spring/damper together with the motor force, unlike MuJoCo. The current HumEnv task uses
`HumEnvAffineTorqueAction` for the active affine actuator and keeps only passive stiffness/damping in the implicit
actuator.

HumEnv/MuJoCo freejoint `qvel[3:6]` is in the root body frame. Convert it with `qpos[3:7]` before writing IsaacLab
root angular velocity:

```text
root_ang_vel_world = quat_rotate(qpos[3:7], qvel[3:6])
```

The original HumEnv XML includes a MuJoCo floor geom. The IsaacLab port uses an IsaacLab terrain plane; keep its
friction close to the XML floor (`0.7`) and avoid changing this while debugging policy behavior.

Use this inspect command after changing robot physics:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/inspect_humenv_physics.py --headless --device cuda --limit 20
```

Expected signs:

```text
action_joint_order_matches_xml=True
robot_joint_order_matches_xml=False
L_Hip_x runtime_kp=10 runtime_kd=15 xml_act_kp=180 xml_pass_kp=10 xml_total_kp=190
```

`robot_joint_order_matches_xml=False` is expected for IsaacLab's imported articulation. Motion `qpos/qvel` follows the
XML/MuJoCo order, so reset code must remap motion joints by name before writing IsaacLab joint state. Do not replace
that remap with a raw `qpos[7:]` / `qvel[6:]` slice.

For observation/reset parity diagnostics:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/check_humenv_obs_parity.py \
  --headless \
  --device cuda \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --motion-index 1 \
  --episode-index 0 \
  --num-frames 24 \
  --stride 20
```

The h5 reference observations were checked against the original MuJoCo HumEnv source in the `amp` conda environment
and reproduce to numerical precision, so persistent parity errors are on the IsaacLab side rather than in the h5.

## FB-CPR Tracking Inference Smoke

After a checkpoint exists, run tracking inference on a reference HumEnv h5:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/tracking_inference_fbcpr_humenv.py \
  --headless \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr/B13105BCD7/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --motion-index 0 \
  --episode-index 0 \
  --num-steps 8 \
  --num-envs 1 \
  --device cuda \
  --mean \
  --output /tmp/fbcpr_tracking_smoke.npz
```

The script loads only `FBcprModel` weights from `checkpoint/model/model.safetensors`, then calls
`model.tracking_inference(reference_observation)` like metamotivo's `TrackingWrapper` and rolls the resulting
per-frame `z` through the IsaacLab HumEnv task. Do not switch this back to `FBcprAgent.load(...)` for inference:
constructing the training agent creates optimizers and can touch `torch.compiler`/Dynamo, which is broken in the
IsaacSim GUI Python stack.

The tracking inference script checks for NaN/Inf by default in reference observations, `z`, observations, actions,
rewards, done flags, and tensor values inside `info`. It stops at the first non-finite value and prints a summary at
shutdown. Add `--continue-on-nan` only when you intentionally want to keep rolling after the first report.

For render mode, omit `--headless`. This has been smoke-tested with:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/tracking_inference_fbcpr_humenv.py \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr/B13105BCD7/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --motion-index 0 \
  --episode-index 0 \
  --num-steps 64 \
  --num-envs 1 \
  --device cuda \
  --mean \
  --output /tmp/fbcpr_tracking_render_64.npz
```

Expected output shape check:

```text
obs: (64, 1, 358)
action: (64, 1, 69)
z: (64, 1, 256)
done: (64, 1, 1)
reference_observation: (64, 358)
```

The GUI run prints many IsaacSim warnings about duplicated Gym registry entries and shutdown cleanup. Treat those as
noise when the process exits with code 0 and the output npz has the expected shapes.

To continuously browse through the motion list in the same viewer, add `--iterate-motions`. Use `--num-steps 0`
to play each episode to its natural end, and add `--loop-motions` to wrap back to the beginning after the last
motion:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/tracking_inference_fbcpr_humenv.py \
  --checkpoint /home/chn/hajimi/bfm_research/pretrained/metamotivo-S-1 \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --iterate-motions \
  --loop-motions \
  --num-steps 0 \
  --num-envs 1 \
  --device cuda \
  --mean
```

## MetaMotivo Reference Model

Official MetaMotivo checkpoints are on HuggingFace, for example `facebook/metamotivo-S-1`. The local cache currently
contains:

```text
/home/chn/hajimi/bfm_research/pretrained/metamotivo-S-1
```

Run the official S-1 reference on the IsaacLab HumEnv task:

```bash
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
python scripts/tracking_inference_fbcpr_humenv.py \
  --headless \
  --checkpoint facebook/metamotivo-S-1 \
  --hf-cache-dir /home/chn/hajimi/bfm_research/pretrained \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --motion-index 0 \
  --episode-index 0 \
  --num-steps 64 \
  --num-envs 1 \
  --device cuda \
  --mean \
  --output /tmp/fbcpr_tracking_official_s1_64.npz
```

Compare a local checkpoint rollout with the official model:

```bash
python scripts/compare_fbcpr_rollouts.py \
  --a /tmp/fbcpr_tracking_render_64.npz \
  --b /tmp/fbcpr_tracking_official_s1_64.npz \
  --label-a ours \
  --label-b official-S1
```

Current diagnostic result: dimensions match (`obs=358`, `action=69`, `z=256`), so the interface is probably not the
main issue. The short local checkpoint is far more conservative than S-1: local action std was about `0.28` with no
action saturation, while official S-1 action std was about `0.80` with roughly `45%` of action values at `[-1, 1]`.
