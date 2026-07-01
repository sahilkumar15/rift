#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
GPU="${GPU:-4}"
TABLES="${TABLES:-table1_component,table2_objective,table3_horizon}"
MAX_ITEMS="${MAX_ITEMS:-}"

CIFT_ROOT="$(python - <<PY
import yaml
m=yaml.safe_load(open("$ABLCFG"))
print(m["cift"]["root"])
PY
)"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CMD=(
  python ablations/eval_table123.py
  --ablation-config "$ABLCFG"
  --tables "$TABLES"
)

if [[ -n "$MAX_ITEMS" ]]; then
  CMD+=(--max-items "$MAX_ITEMS")
fi

echo "═══════════════════════════════════════════════════════════"
echo " Evaluating RIFT Table123"
echo " gpu       : $GPU"
echo " tables    : $TABLES"
echo " max_items : ${MAX_ITEMS:-<yaml>}"
echo " config    : $ABLCFG"
echo "═══════════════════════════════════════════════════════════"

"${CMD[@]}"
