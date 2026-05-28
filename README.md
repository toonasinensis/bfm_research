 
```bash
python -m pip install -e source/whole_body_tracking
```
python scripts/train_fbcpr_humenv.py   --headless   --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt   --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass   --online-parallel-envs 50   --num-env-steps 30000000   --device cuda   --agent-device cuda   --buffer-device cpu   --use-wandb


python scripts/tracking_inference_fbcpr_humenv.py \
  --checkpoint /home/chn/hajimi/bfm_research/tmp_fbcpr/B13105BCD7/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --motion-index 0 \
  --episode-index 0 \
  --num-steps 300 \
  --num-envs 1 \
  --device cuda \
  --mean


cd /home/chn/hajimi/bfm_research
source /home/chn/miniforge3/etc/profile.d/conda.sh
conda activate mimic

python scripts/tracking_inference_fbcpr_humenv.py \
  --checkpoint /home/chn/hajimi/bfm_research/tmp_fbcpr/B13105BCD7/checkpoint \
  --motions /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/accad_generated_train.txt \
  --motions-root /home/chn/hajimi/bfm_research/source/whole_body_tracking/bfm/data/humenv_amass \
  --motion-index 0 \
  --loop-motions \
  --episode-index 0 \
  --num-steps 30000 \
  --num-envs 10 \
  --device cuda \
  --mean \
  --output /tmp/fbcpr_official_s1_rollout.npz


  cd /home/chn/hajimi/bfm_research
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