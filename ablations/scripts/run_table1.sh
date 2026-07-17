#!/usr/bin/env bash
# Table 1 : component ablation x dataset generalization
#   smoke : GPU=4 MAX_ITEMS=256 bash ablations/scripts/run_table1.sh
#   full  : GPU=4 MAX_ITEMS=full bash ablations/scripts/run_table1.sh
#   one ds: GPU=4 DATASETS=ffpp_c23 bash ablations/scripts/run_table1.sh
#   sweep : GPU=4 MODE=zero bash ablations/scripts/run_table1.sh
set -euo pipefail
CFG="${CFG:-ablations/configs/table1.yaml}"
GPU="${GPU:-0}"; MAX_ITEMS="${MAX_ITEMS:-256}"
DATASETS="${DATASETS:-}"; VARIANTS="${VARIANTS:-}"; MODE="${MODE:-}"
BATCH="${BATCH:-8}"; FWD="${FWD:-64}"

python ablations/patch_for_ablations.py
CIFT_ROOT="$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['cift']['root'])")"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CMD=(python -m ablations.run_table1 --config "$CFG" --device cuda:0
     --max-items "$MAX_ITEMS" --batch-size "$BATCH" --forward-batch-size "$FWD")
[[ -n "$DATASETS" ]] && CMD+=(--datasets "$DATASETS")
[[ -n "$VARIANTS" ]] && CMD+=(--variants "$VARIANTS")
[[ -n "$MODE" ]] && CMD+=(--intervention-mode "$MODE")

echo "TABLE 1  gpu=$GPU  max_items=$MAX_ITEMS  datasets=${DATASETS:-<all>}  mode=${MODE:-<yaml>}"
"${CMD[@]}"
