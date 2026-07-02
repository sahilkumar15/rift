#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
GPU="${GPU:-4}"
TABLES="${TABLES:-table1_component,table2_objective,table3_horizon}"
MAX_ITEMS="${MAX_ITEMS:-}"

python ablations/patch_for_ablations.py

CIFT_ROOT="$(python -c "import yaml; print(yaml.safe_load(open('$ABLCFG'))['cift']['root'])")"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CMD=(python -m ablations.eval_table123 --ablation-config "$ABLCFG" --tables "$TABLES" --device "cuda:0")

if [[ -n "$MAX_ITEMS" ]]; then
  CMD+=(--max-items "$MAX_ITEMS")
fi

echo "═══════════════════════════════════════════════════════════"
echo " RIFT Table123 evaluation"
echo " gpu       : $GPU"
echo " tables    : $TABLES"
echo " max_items : ${MAX_ITEMS:-<yaml>}"
echo " config    : $ABLCFG"
echo "═══════════════════════════════════════════════════════════"

"${CMD[@]}"
