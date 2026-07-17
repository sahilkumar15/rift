#!/usr/bin/env bash
# Gate 1 smoke test, then full run.
#   Smoke  (~2 min):  GPU=4 MAX_ITEMS=256 bash ablations/scripts/run_gate1.sh
#   Full:             GPU=4 MAX_ITEMS=full bash ablations/scripts/run_gate1.sh
set -euo pipefail

GPU="${GPU:-0}"
ROWS="${ROWS:-full_h1,full_h4,full_h8,full_h10,full_h12}"
MAX_ITEMS="${MAX_ITEMS:-256}"
N_RANDOM="${N_RANDOM:-4}"
BATCH="${BATCH:-8}"
FWD_BATCH="${FWD_BATCH:-64}"
OUT="${OUT:-experiments/ablations/gates}"
CONFIG="${CONFIG:-ablations/configs/table123_rift.yaml}"

echo "GATE 1  gpu=${GPU} rows=${ROWS} max_items=${MAX_ITEMS} n_random=${N_RANDOM}"

CUDA_VISIBLE_DEVICES="${GPU}" python -m ablations.gate1_validity \
  --ablation-config "${CONFIG}" \
  --device cuda:0 \
  --rows "${ROWS}" \
  --max-items "${MAX_ITEMS}" \
  --n-random "${N_RANDOM}" \
  --batch-size "${BATCH}" \
  --forward-batch-size "${FWD_BATCH}" \
  --output-dir "${OUT}"

status=$?
echo
echo "verdict file: ${OUT}/gate1_verdict.json"
exit "${status}"
