#!/usr/bin/env bash
set -euo pipefail

cd /home/chn/hajimi/bfm_research

set +u
source /home/chn/isaacsim450/setup_conda_env.sh
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic
set -u

export PYTHONPATH=/home/chn/hajimi/bfm_research:/home/chn/hajimi/bfm_research/source/whole_body_tracking:${PYTHONPATH:-}
export WANDB_API_KEY=${WANDB_API_KEY:-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}

mkdir -p logs tmp_fbcpr_g1

python scripts/train_fbcpr_g1.py \
  --headless \
  --device cuda \
  --agent-device cuda \
  --log-every-updates 10240 \
  --eval-every-steps 102400 \
  --tracking-eval-num-envs 1024 \
  --tracking-eval-max-motions 8 \
  --tracking-eval-max-steps 128 \
  --use-wandb \
  --evaluate
