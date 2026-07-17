#!/usr/bin/env bash
# Full Table 1: every dataset, every row, one GPU per dataset.
#
#   bash ablations/scripts/run_table1_all.sh                 # blur, all datasets
#   MODE=zero bash ablations/scripts/run_table1_all.sh       # zero-fill sweep
#   GPUS=4,5 DATASETS=ffpp_c23,celebdf_v2 bash ablations/scripts/run_table1_all.sh
#
# Datasets are parallelised (each is self-contained: it carries its own row-0
# random anchor, so gap_vs_random stays correct within a process). Variants are
# NOT split across GPUs - a process without row 0 cannot compute gap_vs_random.
set -euo pipefail

CFG="${CFG:-ablations/configs/table1.yaml}"
GPUS="${GPUS:-4,5,6,7}"
MODE="${MODE:-blur}"
MAX_ITEMS="${MAX_ITEMS:-full}"
BATCH="${BATCH:-8}"
FWD="${FWD:-64}"
WORKERS="${WORKERS:-8}"
OUT_ROOT="${OUT_ROOT:-experiments/ablations/table1}"

python ablations/patch_for_ablations.py
CIFT_ROOT="$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['cift']['root'])")"
export PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "───────────────────────────────────────────────────────────────"
echo " PREFLIGHT"
echo "───────────────────────────────────────────────────────────────"
if ! python -m ablations.preflight_table1 --config "$CFG"; then
  echo
  echo "Preflight found blockers. Fix them or pass DATASETS= to run a subset."
  [[ "${FORCE:-0}" == "1" ]] || exit 1
  echo "FORCE=1 set - continuing anyway."
fi

if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a DS <<< "$DATASETS"
else
  IFS=' ' read -r -a DS <<< "$(python -c "
import yaml,sys
t=yaml.safe_load(open('$CFG'))
from pathlib import Path
print(' '.join(k for k,v in t['datasets'].items() if Path(v['eval_csv']).exists()))
")"
fi
IFS=',' read -r -a GA <<< "$GPUS"

if [[ ${#DS[@]} -eq 0 ]]; then
  echo "No dataset CSVs exist yet. Nothing to run."; exit 1
fi

echo
echo "───────────────────────────────────────────────────────────────"
echo " LAUNCH  datasets=${DS[*]}  gpus=${GA[*]}  mode=$MODE"
echo "───────────────────────────────────────────────────────────────"

pids=(); i=0
for ds in "${DS[@]}"; do
  gpu="${GA[$(( i % ${#GA[@]} ))]}"
  out="$OUT_ROOT/$ds"
  mkdir -p "$out"
  echo "  [gpu $gpu] $ds -> $out/$MODE/"
  CUDA_VISIBLE_DEVICES="$gpu" python -m ablations.run_table1 \
    --config "$CFG" --device cuda:0 --datasets "$ds" \
    --max-items "$MAX_ITEMS" --intervention-mode "$MODE" \
    --batch-size "$BATCH" --forward-batch-size "$FWD" \
    --output-dir "$out" > "$out/${MODE}.log" 2>&1 &
  pids+=($!); i=$((i+1))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

echo
echo "───────────────────────────────────────────────────────────────"
for ds in "${DS[@]}"; do
  f="$OUT_ROOT/$ds/$MODE/table1_component_x_dataset.csv"
  [[ -f "$f" ]] && echo "  wrote $f" || echo "  MISSING $f  (see $OUT_ROOT/$ds/${MODE}.log)"
done
echo "───────────────────────────────────────────────────────────────"
exit "$fail"
