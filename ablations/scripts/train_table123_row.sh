#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
ROW="${ROW:?ERROR: set ROW=<policy_key>, e.g. ROW=full_h4}"

# Apply reward/objective patches once per run.
python ablations/patch_for_ablations.py

eval "$(python -m ablations.lib.manifest --config "$ABLCFG" --row "$ROW" --emit-bash-train)"

GPUS="${GPUS:-$GPUS_DEFAULT}"
BATCH="${BATCH:-$BATCH_DEFAULT}"
EPOCHS="${EPOCHS:-$EPOCHS_DEFAULT}"
WORKERS="${WORKERS:-$WORKERS_DEFAULT}"
WANDB_MODE="${WANDB_MODE:-$WANDB_MODE_DEFAULT}"
RESUME="${RESUME:-$RESUME_DEFAULT}"
FAST_REWARD="${FAST_REWARD:-true}"
SKIP_UNUSED_INTERVENTIONS="${SKIP_UNUSED_INTERVENTIONS:-true}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"

# Fast-ablation controls.
TRAIN_MAX_ITEMS="${TRAIN_MAX_ITEMS:-$TRAIN_MAX_ITEMS_DEFAULT}"
VAL_MAX_ITEMS="${VAL_MAX_ITEMS:-$VAL_MAX_ITEMS_DEFAULT}"
TRAIN_VAL_EVERY="${TRAIN_VAL_EVERY:-$VAL_EVERY}"
TRAIN_VAL_MAX_BATCHES="${TRAIN_VAL_MAX_BATCHES:-$VAL_MAX_BATCHES}"

mkdir -p "$ROW_DIR/logs" "$ROW_DIR/ckpt"

WANDB_FLAG="--no-wandb"
if [[ "$WANDB_ENABLED" == "true" ]]; then
  WANDB_FLAG="--wandb"
fi

echo "═══════════════════════════════════════════════════════════"
echo " RIFT Table123 row training"
echo " row              : $ROW"
echo " run_name         : $RUN_NAME"
echo " reward           : $REWARD_PRESET"
echo " horizon          : $HORIZON"
echo " gpus             : $GPUS"
IFS=',' read -ra _RIFT_ABL_GPU_ARR <<< "$GPUS"
_WORLD_SIZE="${#_RIFT_ABL_GPU_ARR[@]}"
_GLOBAL_BATCH=$(( _WORLD_SIZE * BATCH ))
_TRAIN_ROWS="$(python - "$TRAIN_CSV" <<'PYROWS'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print("")
else:
    print(max(0, sum(1 for _ in p.open("rb")) - 1))
PYROWS
)"
_EXPECTED_BATCHES="<unknown>"
if [[ "$_TRAIN_ROWS" =~ ^[0-9]+$ && "$_GLOBAL_BATCH" -gt 0 ]]; then
  _EXPECTED_BATCHES=$(( (_TRAIN_ROWS + _GLOBAL_BATCH - 1) / _GLOBAL_BATCH ))
fi
echo " batch/per-gpu    : $BATCH"
echo " world/global_bs  : $_WORLD_SIZE / $_GLOBAL_BATCH"
echo " expected batches : $_EXPECTED_BATCHES"
echo " epochs           : $EPOCHS"
echo " train_max_items  : ${TRAIN_MAX_ITEMS:-FULL}"
echo " val_max_items    : ${VAL_MAX_ITEMS:-FULL}"
echo " val_every        : $TRAIN_VAL_EVERY"
echo " val_max_batches  : $TRAIN_VAL_MAX_BATCHES"
echo " fast_reward      : $FAST_REWARD"
echo " skip_unused_intv : $SKIP_UNUSED_INTERVENTIONS"
echo " ppo_epochs       : $PPO_EPOCHS"
echo " ckpt_dir         : $ROW_DIR/ckpt"
echo " wandb            : $WANDB_ENABLED / $WANDB_MODE"
echo "═══════════════════════════════════════════════════════════"

export PYTHONPATH="$(pwd):${CIFT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RIFT_FAST_REWARD="$FAST_REWARD"
export RIFT_SKIP_UNUSED_INTERVENTIONS="$SKIP_UNUSED_INTERVENTIONS"

CMD=(
  bash scripts/run_rift.sh
  --mode train
  --gpus "$GPUS"
  --config "$CONFIG"
  --cift-root "$CIFT_ROOT"
  --ckpt "$CIFT_CKPT"
  --batch "$BATCH"
  --epochs "$EPOCHS"
  --workers "$WORKERS"
  --horizon "$HORIZON"
  $WANDB_FLAG
  experiment.root_dir="$ROOT_DIR"
  experiment.name="$RUN_NAME"
  checkpoint.dir="$ROW_DIR/ckpt"
  wandb.project="$WANDB_PROJECT"
  wandb.group="$WANDB_GROUP"
  wandb.name="$RUN_NAME"
  wandb.mode="$WANDB_MODE"
  rl.reward_preset="$REWARD_PRESET"
  rl.ppo_epochs="$PPO_EPOCHS"
  rl.val_every="$TRAIN_VAL_EVERY"
  rl.val_max_batches="$TRAIN_VAL_MAX_BATCHES"
  data.train_csv="$TRAIN_CSV"
  data.val_csv="$VAL_CSV"
  dataset.split_csv="$EVAL_CSV"
  detector.cift_root="$CIFT_ROOT"
  detector.cift_ckpt="$CIFT_CKPT"
  detector.cift_config="$CIFT_CONFIG"
  resume="$RESUME"
)

# TRAIN_MAX_ITEMS / VAL_MAX_ITEMS:
#   unset/default -> use YAML ablation default
#   FULL/all/none/null/None/0 -> use full CSV
#   integer -> use that many samples
_is_full_value () {
  local x="$1"
  [[ "$x" == "FULL" || "$x" == "full" || "$x" == "ALL" || "$x" == "all" || "$x" == "none" || "$x" == "None" || "$x" == "null" || "$x" == "0" ]]
}

if [[ -n "$TRAIN_MAX_ITEMS" ]]; then
  if _is_full_value "$TRAIN_MAX_ITEMS"; then
    CMD+=(data.max_items=null)
  else
    CMD+=(data.max_items="$TRAIN_MAX_ITEMS")
  fi
fi

if [[ -n "$VAL_MAX_ITEMS" ]]; then
  if _is_full_value "$VAL_MAX_ITEMS"; then
    CMD+=(data.val_max_items=null)
  else
    CMD+=(data.val_max_items="$VAL_MAX_ITEMS")
  fi
fi

printf "%q " "${CMD[@]}" > "$ROW_DIR/logs/command.sh"
echo "" >> "$ROW_DIR/logs/command.sh"

"${CMD[@]}" 2>&1 | tee "$ROW_DIR/logs/train.log"

echo "[done] row=$ROW"
