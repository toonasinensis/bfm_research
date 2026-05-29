# FB-CPR HumEnv / G1 Notes

This document is the long-lived memory for the current FB-CPR port in `bfm_research`.
Read it before changing the runner, training scripts, G1 env, tracking inference, or MuJoCo rollout.

## What Exists

The FB-CPR stack is split into three layers:

- Lifecycle: `scripts/train_fbcpr_humenv.py`, `scripts/train_fbcpr_g1.py`, `agents/runner/env_runner.py`.
- Plugin config/factories: `source/whole_body_tracking/bfm/config/fb_cpr.py` and task-local `config/fb_cpr.py` files.
- Task implementations: `source/whole_body_tracking/bfm/tasks/humenv` and `source/whole_body_tracking/bfm/tasks/g1`.

The runner is algorithm/task agnostic. Algorithm behavior goes through `FBcprAdapter`; logging, checkpointing, eval, and progress go through hooks. Task selection is config-driven through entry points, not task-specific `if task == ...` branches in the main lifecycle.

## BFM-Zero FB-CPR Aux Port

G1 defaults use `agents.metamotivo.fb_cpr_aux:FBcprAuxAgent`, ported from BFM-Zero. Compared with plain FB-CPR, the Aux variant adds:

- `_aux_critic` and `_target_aux_critic`,
- `_aux_reward_normalizer`,
- `aux_critic_optimizer`,
- `update_aux_critic()`,
- actor loss term `-Q_aux * reg_coeff_aux * weight`,
- train metrics `Q_aux`, `aux_critic_loss`, `mean_aux_reward`, and `aux_rew/...`.

The G1 env exposes BFM-Zero-style auxiliary reward terms through `extras["aux_rewards"]`. The default terms and scalings live in `bfm.config.fb_cpr:G1FBcprRunnerCfg`, not in the command line.

Default Aux terms:

```text
penalty_torques,
penalty_action_rate,
limits_dof_pos,
limits_torque,
penalty_undesired_contact,
penalty_feet_ori,
penalty_ankle_roll,
penalty_slippage
```

Default G1 entry points:

```text
agent_class_entry_point = agents.metamotivo.fb_cpr_aux:FBcprAuxAgent
agent_config_builder_entry_point = bfm.config.fb_cpr:build_fbcpr_aux_agent_config
```

Plain FB-CPR remains available by overriding those two entry points; do this only for ablations.

## Output Directories

Training outputs must stay under `logs/`:

- HumEnv and old HumEnv-compatible runs: `logs/tmp_fbcpr/<RUN_ID>/`
- G1 runs: `logs/tmp_fbcpr_g1/<RUN_ID>/`
- Service/nohup logs: `logs/*.log`

The default `make_work_dir()` now writes to those folders. Do not create new root-level `tmp_fbcpr*` folders. If a command passes `--work-dir`, keep it under `logs/` unless there is a specific reason not to.

Important local checkpoints currently used in examples:

- HumEnv local: `logs/tmp_fbcpr/B13105BCD7/checkpoint`
- G1 local: `logs/tmp_fbcpr_g1/BB963303D1/checkpoint`
- MetaMotivo official cache: `pretrained/metamotivo-S-1`

## Main Commands

G1 default training:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --device cuda \
  --agent-device cuda \
  --use-wandb
```

G1 plain FB-CPR ablation:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --device cuda \
  --agent-device cuda \
  --agent-class-entry-point agents.metamotivo.fb_cpr:FBcprAgent \
  --agent-config-builder-entry-point bfm.config.fb_cpr:build_fbcpr_agent_config
```

HumEnv default training still uses `scripts/train_fbcpr_humenv.py` with explicit HumEnv data paths unless defaults are configured:

```bash
python scripts/train_fbcpr_humenv.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --device cuda \
  --agent-device cuda \
  --use-wandb
```

G1 IsaacLab tracking inference:

```bash
python scripts/tracking_inference_fbcpr_g1.py \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr_g1/BB963303D1/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --num-steps 2000 \
  --device cuda \
  --mean \
  --debug-vis
```

G1 MuJoCo tracking inference:

```bash
python scripts/tracking_inference_fbcpr_g1_mujoco.py \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr_g1/BB963303D1/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --num-steps 2000 \
  --device cuda \
  --mean \
  --debug-vis
```

## G1 LAFAN Task

The G1 task uses LAFAN `.npz` episodes from:

```text
source/whole_body_tracking/bfm/data/lafan
```

The observation is 208 dims from 14 ordered bodies:

```text
pelvis,
left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link,
torso_link,
left_shoulder_roll_link, left_elbow_link, left_wrist_yaw_link,
right_shoulder_roll_link, right_elbow_link, right_wrist_yaw_link
```

The body indexes in the full G1 body list are:

```text
[0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
```

`scripts/check_g1_lafan_obs_parity.py` verifies that IsaacLab reset state and expert frame-0 obs match. This check should stay near zero before trusting training, eval, or MuJoCo comparison.

## MuJoCo Alignment

The MuJoCo script intentionally does not import IsaacLab. It uses:

```text
/home/chn/hajimi/BFM-Zero/humanoidverse/data/robots/g1/scene_29dof_freebase_noadditional_actuators.xml
```

The raw XML is not enough by itself. The script patches MuJoCo model parameters at runtime to match the IsaacLab G1 config:

- per-joint armature,
- zero passive damping/frictionloss,
- actuator and joint force ranges,
- BFM-Zero action-scale semantics,
- hard waist gains,
- per-substep PD torque recomputation.

Do not remove this patching unless the XML itself is regenerated to already match training.

### Joint Order

IsaacLab action/joint order differs from MuJoCo XML order. The MuJoCo script contains explicit `ISAAC_JOINT_NAMES`, `MUJOCO_JOINT_NAMES`, and reorder maps. Any new trace, torque, qpos, qvel, action, or processed target must use the correct order.

Isaac order starts:

```text
left_hip_pitch_joint, right_hip_pitch_joint, waist_yaw_joint,
left_hip_roll_joint, right_hip_roll_joint, waist_roll_joint, ...
```

MuJoCo order starts:

```text
left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint,
left_knee_joint, left_ankle_pitch_joint, ...
```

### Trace Workflow

When MuJoCo and IsaacLab disagree, export an IsaacLab trace and replay it in MuJoCo.

```bash
python scripts/tracking_inference_fbcpr_g1.py \
  --headless \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr_g1/BB963303D1/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --num-steps 120 \
  --num-envs 1 \
  --device cuda \
  --mean \
  --trace-output /tmp/g1_isaac_align_trace.npz

python scripts/tracking_inference_fbcpr_g1_mujoco.py \
  --headless \
  --trace-input /tmp/g1_isaac_align_trace.npz \
  --mode state-replay \
  --device cuda \
  --print-every 30
```

`state-replay` loads Isaac qpos/qvel/body state directly into MuJoCo and compares MuJoCo obs to Isaac obs. This is the first check. It should be around `isaac_obs_mse ~= 1e-5`.

After that, run dynamic trace replay:

```bash
python scripts/tracking_inference_fbcpr_g1_mujoco.py \
  --headless \
  --trace-input /tmp/g1_isaac_align_trace.npz \
  --mode dynamic \
  --device cuda \
  --print-every 30
```

After the current alignment fixes, a short 16-step trace had approximately:

```text
state-replay isaac_obs_mse ~= 1e-5
dynamic trace step16 isaac_obs_mse ~= 0.0028
direct checkpoint step120 obs_mse ~= 0.0034, body_pos_mean ~= 0.0405m
```

## Interpreting MuJoCo Logs

`obs_mse` compares local policy obs to reference obs. The G1 obs is heading-relative and mostly local; it does not strongly penalize global root XY drift.

`body_pos_mean` is world-frame distance from robot bodies to expert body markers. If this grows to meters while `obs_mse` is moderate, the rollout has global drift.

`action_abs_max=1.0` means at least one policy action component saturated. It does not mean every joint is saturated.

`torque_abs_max=50.0` often means ankle or waist roll/pitch hit their effort limit. Current effort limits:

```text
knee: 139
hip_roll: 139
hip_pitch / hip_yaw: 88
ankle_pitch / ankle_roll: 50
waist_yaw: 88
waist_roll / waist_pitch: 50
shoulder / elbow / wrist_roll: 25
wrist_pitch / wrist_yaw: 5
```

For long motions, open-loop tracking can drift globally over 100+ seconds. Use shorter windows and trace replay when debugging dynamics parity.

## Evaluation

`--evaluate` enables tracking eval through config entry points. `EvalHook` only triggers, prints, and logs metrics; the concrete evaluator lives under the task module.

G1 eval is IsaacLab-local and vectorized. It logs metrics such as:

```text
eval/tracking/obs_distance
eval/tracking/obs_emd
eval/tracking/body_pos_mean
eval/tracking/body_rot_mean
eval/tracking/done_rate
eval/tracking/nan_count
```

HumEnv eval has an IsaacLab-local path and an optional MetaMotivo bench path:

```bash
--tracking-eval-entry-point bfm.tasks.humenv.eval.metamotivo_tracking:make_tracking_evaluator
```

## Design Rules

Keep the plugin/factory design:

- Train scripts choose config entry points.
- Config entry points create env cfg, configure env cfg, build expert buffers, and build agent cfg.
- Runner knows lifecycle only.
- Hooks own side effects.
- New tasks should add task-local config/factory modules, not branches in runner.

Avoid:

- task-specific branches in the training lifecycle,
- PPO/AMP/FB-CPR branches in the runner,
- importing IsaacLab from pure MuJoCo tools,
- root-level training output folders.

## Common Smoke Tests

Compile:

```bash
python -m py_compile \
  scripts/train_fbcpr_humenv.py \
  scripts/train_fbcpr_g1.py \
  scripts/tracking_inference_fbcpr_g1.py \
  scripts/tracking_inference_fbcpr_g1_mujoco.py
```

G1 short train:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
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

MuJoCo short rollout:

```bash
python scripts/tracking_inference_fbcpr_g1_mujoco.py \
  --headless \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr_g1/BB963303D1/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --num-steps 120 \
  --device cuda \
  --mean \
  --print-every 30
```
