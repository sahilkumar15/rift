#!/usr/bin/env bash

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Main component ablation
# Needs training only for:
#   ROW=logit_h4
#   ROW=full_h4
# Rows Random / Grad-CAM / Grad-CAM+causal / CIFT-Δ+causal do not need training.
# ═══════════════════════════════════════════════════════════════════════════════

# ROW=logit_h4 GPUS=4,5 EPOCHS=30 BATCH=128 bash ablations/scripts/train_table123_row.sh

# ROW=full_h4 GPUS=6,7 EPOCHS=30 BATCH=128 bash ablations/scripts/train_table123_row.sh

# GPU=4 TABLES=table1_component MAX_ITEMS=512 bash ablations/scripts/eval_table123.sh
# ═══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Reward objective ablation
# Needs training for:
#   ROW=necessity_h4
#   ROW=sufficiency_h4
#   ROW=nosparsity_h4
#   ROW=full_h4
#
# NOTE:
#   full_h4 is shared with Table 1. If you already trained full_h4, skip that row.
# ═══════════════════════════════════════════════════════════════════════════════

# ROW=necessity_h4 GPUS=6,7 EPOCHS=30 BATCH=128 bash ablations/scripts/train_table123_row.sh

# ROW=sufficiency_h4 GPUS=6,7 EPOCHS=30 BATCH=128 bash ablations/scripts/train_table123_row.sh

# ROW=nosparsity_h4 GPUS=4,5,6,7 EPOCHS=30 BATCH=128 bash ablations/scripts/train_table123_row.sh
# ROW=full_h4 GPUS=4,5,6,7 EPOCHS=30 BATCH=128 bash ablations/scripts/train_table123_row.sh

# GPU=4 TABLES=table2_objective MAX_ITEMS=512 bash ablations/scripts/eval_table123.sh

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — RL horizon ablation
# Needs training for:
#   ROW=full_h1
#   ROW=full_h4
#   ROW=full_h8
#
# CIFT-Δ top-k fixed does not need training.
# NOTE:
#   full_h4 is shared with Table 1 and Table 2. If already trained, skip it.
# ═══════════════════════════════════════════════════════════════════════════════

# ROW=full_h1 GPUS=4,5,6,7 EPOCHS=30 BATCH=64 bash ablations/scripts/train_table123_row.sh

# ROW=full_h4 GPUS=4,5,6,7 EPOCHS=30 BATCH=64 bash ablations/scripts/train_table123_row.sh

# ROW=full_h8 GPUS=4,5,6,7 EPOCHS=30 BATCH=64 bash ablations/scripts/train_table123_row.sh

# GPU=4 TABLES=table3_horizon MAX_ITEMS=512 bash ablations/scripts/eval_table123.sh
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════


set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
ROW="${ROW:-}"

if [[ -z "$ROW" ]]; then
  echo "[ERROR] set ROW, e.g.: ROW=full_h4 bash ablations/scripts/train_table123_row.sh"
  exit 1
fi

eval "$(python -m ablations.lib.manifest --config "$ABLCFG" --row "$ROW" --emit-bash-train)"

GPUS="${GPUS:-$GPUS_DEFAULT}"
BATCH="${BATCH:-$BATCH_DEFAULT}"
EPOCHS="${EPOCHS:-$EPOCHS_DEFAULT}"
WORKERS="${WORKERS:-$WORKERS_DEFAULT}"
WANDB_MODE="${WANDB_MODE:-$WANDB_MODE_DEFAULT}"
RESUME="${RESUME:-$RESUME_DEFAULT}"

mkdir -p "$ROW_DIR/logs" "$ROW_DIR/ckpt"

echo "═══════════════════════════════════════════════════════════"
echo " RIFT Table123 row training"
echo " row        : $ROW"
echo " run_name   : $RUN_NAME"
echo " row_dir    : $ROW_DIR"
echo " ckpt       : $POLICY_CKPT"
echo " reward     : $REWARD_PRESET"
echo " horizon    : $HORIZON"
echo " gpus       : $GPUS"
echo " batch      : $BATCH"
echo " epochs     : $EPOCHS"
echo " wandb      : $WANDB_ENABLED / $WANDB_MODE"
echo "═══════════════════════════════════════════════════════════"

WANDB_FLAG="--no-wandb"
if [[ "$WANDB_ENABLED" == "true" ]]; then
  WANDB_FLAG="--wandb"
fi

bash scripts/run_rift.sh \
  --mode train \
  --gpus "$GPUS" \
  --config "$CONFIG" \
  --cift-root "$CIFT_ROOT" \
  --ckpt "$CIFT_CKPT" \
  --batch "$BATCH" \
  --epochs "$EPOCHS" \
  --workers "$WORKERS" \
  --horizon "$HORIZON" \
  $WANDB_FLAG \
  experiment.root_dir="$ROOT_DIR" \
  experiment.name="$RUN_NAME" \
  checkpoint.dir="$ROW_DIR/ckpt" \
  wandb.project="$WANDB_PROJECT" \
  wandb.group="$WANDB_GROUP" \
  wandb.name="$RUN_NAME" \
  wandb.mode="$WANDB_MODE" \
  rl.reward_preset="$REWARD_PRESET" \
  rl.val_every="$VAL_EVERY" \
  rl.val_max_batches="$VAL_MAX_BATCHES" \
  dataset.split_csv="$EVAL_CSV" \
  data.train_csv="$TRAIN_CSV" \
  data.val_csv="$VAL_CSV" \
  detector.cift_root="$CIFT_ROOT" \
  detector.cift_ckpt="$CIFT_CKPT" \
  resume="$RESUME"

echo "[done] trained row=$ROW"
