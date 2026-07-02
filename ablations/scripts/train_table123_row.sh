#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
ROW="${ROW:?ERROR: set ROW=<policy_key>, e.g. ROW=full_h4}"

python ablations/patch_for_ablations.py

eval "$(python -m ablations.lib.manifest --config "$ABLCFG" --row "$ROW" --emit-bash-train)"

GPUS="${GPUS:-$GPUS_DEFAULT}"
BATCH="${BATCH:-$BATCH_DEFAULT}"
EPOCHS="${EPOCHS:-$EPOCHS_DEFAULT}"
WORKERS="${WORKERS:-$WORKERS_DEFAULT}"
WANDB_MODE="${WANDB_MODE:-$WANDB_MODE_DEFAULT}"
RESUME="${RESUME:-$RESUME_DEFAULT}"

mkdir -p "$ROW_DIR/logs" "$ROW_DIR/ckpt"

WANDB_FLAG="--no-wandb"

if [[ "$WANDB_ENABLED" == "true" ]]; then
  WANDB_FLAG="--wandb"
fi

echo "═══════════════════════════════════════════════════════════"
echo " RIFT Table123 row training"
echo " row        : $ROW"
echo " run_name   : $RUN_NAME"
echo " reward     : $REWARD_PRESET"
echo " horizon    : $HORIZON"
echo " gpus       : $GPUS"
echo " batch      : $BATCH"
echo " epochs     : $EPOCHS"
echo " ckpt_dir   : $ROW_DIR/ckpt"
echo " wandb      : $WANDB_ENABLED / $WANDB_MODE"
echo "═══════════════════════════════════════════════════════════"

export PYTHONPATH="$(pwd):${CIFT_ROOT}:${PYTHONPATH:-}"

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
  data.train_csv="$TRAIN_CSV" \
  data.val_csv="$VAL_CSV" \
  dataset.split_csv="$EVAL_CSV" \
  detector.cift_root="$CIFT_ROOT" \
  detector.cift_ckpt="$CIFT_CKPT" \
  detector.cift_config="$CIFT_CONFIG" \
  resume="$RESUME" \
  2>&1 | tee "$ROW_DIR/logs/train.log"

echo "[done] row=$ROW"
