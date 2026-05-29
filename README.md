# BFM Research Run Commands

## Environment

Run commands from the repo root with the `mimic` conda environment:

```bash
cd /home/chn/hajimi/bfm_research
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
```

Install the local IsaacLab extension if needed:

```bash
python -m pip install -e source/whole_body_tracking
```

## Data Paths

HumEnv motion list:

```text
/home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt
```

HumEnv motion root:

```text
/home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass
```

G1 LAFAN data:

```text
/home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan
```

## HumEnv FB-CPR Training

Short smoke:

```bash
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
  --buffer-device cpu
```

Vectorized smoke:

```bash
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

Long W&B run:

```bash
python scripts/train_fbcpr_humenv.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --online-parallel-envs 50 \
  --num-env-steps 30000000 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cpu \
  --use-wandb
```

## HumEnv Tracking Rollout

Local checkpoint:

```bash
python scripts/tracking_inference_fbcpr_humenv.py \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr/B13105BCD7/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --motion-index 0 \
  --episode-index 0 \
  --num-steps 300 \
  --num-envs 1 \
  --device cuda \
  --mean
```

Loop through all motions with the official MetaMotivo S-1 checkpoint:

```bash
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

Headless trace output:

```bash
python scripts/tracking_inference_fbcpr_humenv.py \
  --headless \
  --checkpoint /home/chn/hajimi/bfm_research/pretrained/metamotivo-S-1 \
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

The rollout script checks NaN/Inf by default and prints a summary at shutdown. Add `--continue-on-nan` only when you want to keep running after the first report.

## G1 LAFAN Parity / Visualization

One-frame body-index and obs parity check:

```bash
python scripts/check_g1_lafan_obs_parity.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --frame 0 \
  --num-envs 1 \
  --device cuda \
  --output /tmp/g1_lafan_parity.npz
```

Hold one frame in the viewer:

```bash
python scripts/check_g1_lafan_obs_parity.py \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --frame 0 \
  --num-envs 1 \
  --device cuda \
  --debug-vis \
  --hold
```

Play the expert motion in the viewer:

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

Expected parity signs:

```text
obs_shape=(1, 208)
obs mse=0.00000000
body_pos mean~=0 max~=0
robot_name=<name> cfg_name=<same name>
```

## G1 FB-CPR MuJoCo Rollout

This viewer does not start IsaacLab. It loads the trained FB-CPR checkpoint, reads LAFAN `.npz` motions directly, and rolls the policy in MuJoCo with the BFM-Zero G1 XML.

Headless smoke:

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

Interactive MuJoCo viewer:

```bash
python scripts/tracking_inference_fbcpr_g1_mujoco.py \
  --checkpoint /home/chn/hajimi/bfm_research/logs/tmp_fbcpr_g1/BB963303D1/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --motion-index 0 \
  --num-steps 20000000000 \
  --device cuda \
  --mean \
  --debug-vis
```

IsaacLab-to-MuJoCo alignment trace:

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

Use `--iterate-motions` to play through the LAFAN folder once, or `--loop-motions` to cycle forever. `--debug-vis` draws expert body markers. `--mode state-replay` checks pure state/obs parity from an IsaacLab trace; `--mode kinematic-target` is useful for debugging policy targets without MuJoCo torque dynamics.

## G1 FB-CPR Aux Training

Config-default BFM-Zero-style run:

G1 defaults now use the BFM-Zero-style `FBcprAuxAgent`: normal FB-CPR plus an auxiliary critic trained from G1 safety/style reward terms. All main training hyperparameters live in `bfm.config.fb_cpr:G1FBcprRunnerCfg`. Use CLI flags only when overriding the config.
Training shows a total rollout tqdm bar by default; add `--no-progress` to disable it.
`--evaluate` enables config-driven tracking eval. Env construction, expert buffers, agent class, agent config, and tracking eval are selected from config entry points (`env_cfg_entry_point`, `configure_env_entry_point`, `expert_buffer_entry_point`, `agent_class_entry_point`, `agent_config_builder_entry_point`, `tracking_eval_entry_point`), so the train script does not need task-specific branches.

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --device cuda \
  --agent-device cuda \
  --use-wandb
```

The default Aux reward terms are `penalty_torques`, `penalty_action_rate`, `limits_dof_pos`, `limits_torque`, `penalty_undesired_contact`, `penalty_feet_ori`, `penalty_ankle_roll`, and `penalty_slippage`. Training logs include `train/Q_aux`, `train/aux_critic_loss`, `train/mean_aux_reward`, and `train/aux_rew/...`.

To force the old non-Aux FB-CPR agent:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --device cuda \
  --agent-device cuda \
  --agent-class-entry-point agents.metamotivo.fb_cpr:FBcprAgent \
  --agent-config-builder-entry-point bfm.config.fb_cpr:build_fbcpr_agent_config
```

Config-default run with tracking eval:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --device cuda \
  --agent-device cuda \
  --evaluate \
  --use-wandb
```

Small G1 training + tracking-eval smoke:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --online-parallel-envs 1 \
  --num-env-steps 32 \
  --num-seed-steps 8 \
  --update-agent-every 8 \
  --num-agent-updates 1 \
  --log-every-updates 8 \
  --checkpoint-every-steps 32 \
  --evaluate \
  --eval-every-steps 16 \
  --tracking-eval-max-motions 1 \
  --tracking-eval-max-steps 64 \
  --tracking-eval-num-envs 1 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cpu
```

Short smoke:

```bash
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

Vectorized smoke:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --online-parallel-envs 16 \
  --num-env-steps 160 \
  --num-seed-steps 32 \
  --update-agent-every 32 \
  --num-agent-updates 1 \
  --log-every-updates 32 \
  --checkpoint-every-steps 160 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cpu
```

Expanded G1 defaults reference:

```bash
python scripts/train_fbcpr_g1.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/lafan \
  --online-parallel-envs 1024 \
  --num-env-steps 384000000 \
  --num-seed-steps 10240 \
  --update-agent-every 1024 \
  --num-agent-updates 16 \
  --log-every-updates 384000 \
  --checkpoint-every-steps 9600000 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cuda \
  --model residual \
  --hidden-dim 2048 \
  --hidden-layers 6 \
  --actor-std 0.05 \
  --update-z-every-step 100 \
  --lr-f 3e-4 \
  --lr-b 1e-5 \
  --lr-actor 3e-4 \
  --lr-critic 3e-4 \
  --lr-discriminator 1e-5 \
  --fb-target-tau 0.01 \
  --critic-target-tau 0.005 \
  --ortho-coef 100.0 \
  --train-goal-ratio 0.2 \
  --fb-pessimism-penalty 0.0 \
  --actor-pessimism-penalty 0.5 \
  --critic-pessimism-penalty 0.5 \
  --stddev-clip 0.3 \
  --expert-asm-ratio 0.6 \
  --relabel-ratio 0.8 \
  --reg-coeff 0.05 \
  --q-loss-coef 0.0 \
  --grad-penalty-discriminator 10.0 \
  --z-buffer-size 8192 \
  --use-wandb
```

## HumEnv Training With Tracking Eval

Small HumEnv eval smoke:

```bash
python scripts/train_fbcpr_humenv.py \
  --headless \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --online-parallel-envs 1 \
  --num-env-steps 32 \
  --num-seed-steps 8 \
  --update-agent-every 8 \
  --num-agent-updates 1 \
  --log-every-updates 8 \
  --checkpoint-every-steps 32 \
  --evaluate \
  --eval-every-steps 16 \
  --tracking-eval-max-motions 1 \
  --tracking-eval-max-steps 64 \
  --tracking-eval-num-envs 1 \
  --device cuda \
  --agent-device cuda \
  --buffer-device cpu
```

Eval metrics are logged under `eval/tracking/*` in W&B, for example `eval/tracking/obs_emd`, `eval/tracking/obs_distance`, or G1-specific `eval/tracking/body_pos_mean`.
`--tracking-eval-max-steps` limits the IsaacLab tracking eval rollout length.
HumEnv defaults to an IsaacLab-local tracking evaluator that emits MetaMotivo-style names such as `eval/tracking/distance`, `eval/tracking/proximity`, `eval/tracking/emd`, and PHC metrics when the obs layout supports them.
The original MetaMotivo HumEnv bench path is also available with `--tracking-eval-entry-point bfm.tasks.humenv.eval.metamotivo_tracking:make_tracking_evaluator`; it requires the `humenv` package or `/home/chn/hajimi/humenv` on `PYTHONPATH`.
To add another task later, point `--runner-cfg-entry-point` at a config class that provides the same entry point fields instead of editing the training lifecycle.
