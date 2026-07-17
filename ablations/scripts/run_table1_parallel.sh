#!/usr/bin/env bash
# =============================================================================
# Table 1, fully parallel: one GPU per dataset, all 7 rows each, auto-combined
# into ONE final ablation table (ticks + per-dataset metrics + Avg.) when every
# job finishes.
#
#   bash ablations/scripts/run_table1_parallel.sh
#   MAP="ffpp_c23=4,celebdf_v2=5,diffswap=6,ffpp_c23_retrieved=7" \
#     bash ablations/scripts/run_table1_parallel.sh
#   MODE=zero MAX_ITEMS=512 bash ablations/scripts/run_table1_parallel.sh   # smoke
#
# MAP is "dataset=gpu,dataset=gpu,...". Default assigns one of GPUS round-robin
# to every dataset whose eval_csv currently exists (skips the rest, they simply
# won't appear as columns - combine step tolerates partial coverage).
# =============================================================================
set -euo pipefail

CFG="${CFG:-ablations/configs/table1.yaml}"
GPUS="${GPUS:-4,5,6,7}"
MODE="${MODE:-blur}"
MAX_ITEMS="${MAX_ITEMS:-full}"
BATCH="${BATCH:-8}"
FWD="${FWD:-64}"
OUT_ROOT="${OUT_ROOT:-experiments/ablations/table1}"
METRICS="${METRICS:-faith_delta,rift_score}"

python ablations/patch_for_ablations.py
CIFT_ROOT="$(python -c "import yaml;print(yaml.safe_load(open('$CFG'))['cift']['root'])")"
export PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "───────────────────────────────────────────────────────────────"
echo " PREFLIGHT"
echo "───────────────────────────────────────────────────────────────"
if ! python -m ablations.preflight_table1 --config "$CFG"; then
  echo
  echo "Preflight found blockers. Fix them, or set FORCE=1 to run available datasets anyway."
  [[ "${FORCE:-0}" == "1" ]] || exit 1
fi

# ---- build the dataset:gpu map -------------------------------------------
if [[ -n "${MAP:-}" ]]; then
  IFS=',' read -r -a PAIRS <<< "$MAP"
else
  mapfile -t EXISTING < <(python -c "
import yaml
from pathlib import Path
t = yaml.safe_load(open('$CFG'))
for k, v in t['datasets'].items():
    if Path(v['eval_csv']).exists():
        print(k)
")
  IFS=',' read -r -a GA <<< "$GPUS"
  PAIRS=()
  i=0
  for ds in "${EXISTING[@]}"; do
    PAIRS+=("${ds}=${GA[$(( i % ${#GA[@]} ))]}")
    i=$((i + 1))
  done
fi

if [[ ${#PAIRS[@]} -eq 0 ]]; then
  echo "No dataset has an existing eval_csv. Nothing to run."
  exit 1
fi

echo
echo "───────────────────────────────────────────────────────────────"
echo " LAUNCH  (dataset=gpu)  mode=$MODE  max_items=$MAX_ITEMS"
for p in "${PAIRS[@]}"; do echo "   $p"; done
echo "───────────────────────────────────────────────────────────────"

pids=(); dss=()
for p in "${PAIRS[@]}"; do
  ds="${p%=*}"; gpu="${p#*=}"
  out="$OUT_ROOT/$ds"
  mkdir -p "$out"
  echo "  [gpu $gpu] $ds -> $out/$MODE/  (log: $out/${MODE}.log)"
  CUDA_VISIBLE_DEVICES="$gpu" python -m ablations.run_table1 \
    --config "$CFG" --device cuda:0 --datasets "$ds" \
    --max-items "$MAX_ITEMS" --intervention-mode "$MODE" \
    --batch-size "$BATCH" --forward-batch-size "$FWD" \
    --output-dir "$out" > "$out/${MODE}.log" 2>&1 &
  pids+=($!); dss+=("$ds")
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "  [done]   ${dss[$i]}"
  else
    echo "  [FAILED] ${dss[$i]}  -> see $OUT_ROOT/${dss[$i]}/${MODE}.log"
    fail=1
  fi
done

echo
echo "───────────────────────────────────────────────────────────────"
echo " COMBINE  -> one final ablation table across every finished dataset"
echo "───────────────────────────────────────────────────────────────"
ds_csv="$(IFS=,; echo "${dss[*]}")"
python -m ablations.format_table1_paper \
  --config "$CFG" --root "$OUT_ROOT" --mode "$MODE" \
  --datasets "$ds_csv" --metrics "$METRICS"

echo
echo "final table : $OUT_ROOT/TABLE1_combined_${MODE}.csv"
echo "              $OUT_ROOT/TABLE1_combined_${MODE}.md"
exit "$fail"
