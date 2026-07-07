#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
ROWS="${ROWS:-$(python -m ablations.status_table123 --config "$ABLCFG" --print-missing-train)}"

if [[ -z "$ROWS" ]]; then
  echo "[ok] no missing/incompatible policy checkpoints"
  exit 0
fi

for R in $ROWS; do
  echo ""
  echo ">>> train missing/stale row: $R"

  ROW="$R" \
  ABLCFG="$ABLCFG" \
  RESUME="${RESUME:-none}" \
  GPUS="${GPUS:-0,1,2,3}" \
  BATCH="${BATCH:-192}" \
  EPOCHS="${EPOCHS:-40}" \
  PPO_EPOCHS="${PPO_EPOCHS:-3}" \
  FAST_REWARD="${FAST_REWARD:-false}" \
  SKIP_UNUSED_INTERVENTIONS="${SKIP_UNUSED_INTERVENTIONS:-false}" \
  RIFT_EXTRA_OVERRIDES="${RIFT_EXTRA_OVERRIDES:-rl.lr=1e-4 rl.entropy_coef=0.03 early_stopping.patience=10 early_stopping.min_delta=0.001}" \
  bash ablations/scripts/train_table123_row.sh
done
