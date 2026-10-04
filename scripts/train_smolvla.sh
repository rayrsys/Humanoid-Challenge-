#!/usr/bin/env bash
# Fine-tune SmolVLA on the retargeted Panda chess dataset (GPU; tested target: RTX 5070 Ti 16 GB).
#
#   1. Build the dataset (fast with EGL on a GPU machine):
#        MUJOCO_GL=egl python scripts/generate_demos.py --aug-per-demo 4
#        MUJOCO_GL=egl python scripts/export_lerobot.py --repo-id $HF_USER/panda-chess-from-quest --push
#   2. bash scripts/train_smolvla.sh $HF_USER/panda-chess-from-quest
#   3. MUJOCO_GL=egl python scripts/eval_smolvla.py --ckpt outputs/train/smolvla_panda_chess/checkpoints/last/pretrained_model
#
# Our two cameras map onto SmolVLA's pretrained camera slots; the third slot is left empty.
set -euo pipefail
REPO_ID=${1:?usage: train_smolvla.sh <hf-user>/<dataset>}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-32}

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$REPO_ID" \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --output_dir=outputs/train/smolvla_panda_chess \
  --job_name=smolvla_panda_chess \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false
