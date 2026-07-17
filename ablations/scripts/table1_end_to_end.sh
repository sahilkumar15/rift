#!/usr/bin/env bash
# Train both Gate-2 arms (FF++ only), then evaluate every dataset that exists.
# Both arms derive from ONE set of vars so they cannot diverge -- if logit_h4 and
# full_h4 differ in anything but reward_preset, row6-row5 measures optimisation
# effort, not Delta-grounding, and the paper's novelty claim silently dies.
set -euo pipefail

GPUS="${GPUS:-0,1,2,3}"
BATCH="${BATCH:-256}"
EPOCHS="${EPOCHS:-30}"
PPO_EPOCHS="${PPO_EPOCHS:-3}"
CFG="${CFG:-ablations/configs/table1.yaml}"
MODE="${MODE:-blur}"
MAX_ITEMS="${MAX_ITEMS:-full}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
ARMS=(logit_h4 full_h4)

ck_of() {
  python3 -c "
import yaml,sys; sys.path.insert(0,'.')
from ablations.lib.manifest import policy_ckpt
print(policy_ckpt(yaml.safe_load(open('ablations/configs/table123_rift.yaml')), '$1'))"
}

echo "TABLE1 E2E  gpus=$GPUS batch=$BATCH epochs=$EPOCHS ppo=$PPO_EPOCHS"

echo; echo "-- free VRAM --"
need=$(python3 -c "print(int($BATCH*0.16)+4)")
IFS=',' read -r -a GA <<< "$GPUS"
short=0
for g in "${GA[@]}"; do
  fr=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
  gib=$((fr/1024))
  [ "$gib" -lt "$need" ] && { echo "  GPU $g: ${gib} GiB free, need ~${need}  X"; short=1; } \
                         || echo "  GPU $g: ${gib} GiB free  ok"
done
if [ "$short" = "1" ] && [ "$SKIP_TRAIN" != "1" ]; then
  echo; echo "Not enough VRAM at BATCH=$BATCH. Free GPUs, or retry with BATCH=128."
  echo "Whatever BATCH you pick, BOTH arms must use it."
  exit 1
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  for arm in "${ARMS[@]}"; do
    ck="$(ck_of "$arm")"
    if [ -f "$ck" ]; then echo; echo "-- $arm already trained, skipping"; continue; fi
    echo; echo "===== TRAIN $arm ====="
    ROW="$arm" GPUS="$GPUS" BATCH="$BATCH" EPOCHS="$EPOCHS" PPO_EPOCHS="$PPO_EPOCHS" \
      bash ablations/scripts/train_table123_row.sh
    [ -f "$ck" ] || { echo "ERROR: $arm produced no checkpoint at $ck" >&2; exit 1; }
  done
fi

echo; echo "-- verifying both Gate-2 arms --"
miss=0
for arm in "${ARMS[@]}"; do
  ck="$(ck_of "$arm")"
  [ -f "$ck" ] && echo "  OK   $arm" || { echo "  MISS $arm"; miss=1; }
done
[ "$miss" = "1" ] && { echo "Gate 2 needs BOTH arms. Aborting."; exit 1; }

echo; echo "===== EVAL: one GPU per dataset ====="
GPUS="$GPUS" MODE="$MODE" MAX_ITEMS="$MAX_ITEMS" CFG="$CFG" FORCE=1 \
  bash ablations/scripts/run_table1_parallel.sh

echo; echo "table: experiments/ablations/table1/TABLE1_combined_${MODE}.md"
