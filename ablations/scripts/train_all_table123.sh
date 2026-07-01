#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
FORCE="${FORCE:-0}"

for ROW in $(python -m ablations.lib.manifest --config "$ABLCFG" --list-train-rows); do
  CKPT="$(python -m ablations.lib.manifest --config "$ABLCFG" --row "$ROW" --get ckpt)"

  if [[ -f "$CKPT" && "$FORCE" != "1" ]]; then
    echo "[skip] $ROW already has checkpoint: $CKPT"
    continue
  fi

  ROW="$ROW" ABLCFG="$ABLCFG" bash ablations/scripts/train_table123_row.sh
done

echo "[done] all trainable Table123 policies are ready."
