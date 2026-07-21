#!/usr/bin/env bash
set -euo pipefail

cd /scratch/sahil/projects/img_deepfake/code/rift

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
GPU_LIST="${GPU:-${GPUS:-6,7}}"
MAX_ITEMS="${MAX_ITEMS:-2048}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-8}"
GRID="${GRID:-8}"
CELLS="${CELLS:-4}"
OPERATORS="${OPERATORS:-blur,mean,noise,zero}"
RANDOM_DRAWS="${RANDOM_DRAWS:-3}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-32}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
MIN_RATIO="${MIN_RATIO:-1.5}"
SEED="${SEED:-3407}"
EVAL_ROOT="${EVAL_ROOT:-experiments/ablations/rift_table6}"
EVAL_NAME="${EVAL_NAME:-table6_validity}"

IFS=',' read -r -a GPUS_ARR <<< "$GPU_LIST"
SHARDS="${#GPUS_ARR[@]}"
if [[ "$SHARDS" -lt 1 ]]; then
  echo "[error] No GPUs supplied through GPU/GPUS" >&2
  exit 1
fi
if [[ ! -f "$ABLCFG" ]]; then
  echo "[error] Missing ablation config: $ABLCFG" >&2
  exit 1
fi

OUT="$EVAL_ROOT/$EVAL_NAME"
SHARD_ROOT="$OUT/shards"
LOG_ROOT="$OUT/logs"
TABLE_ROOT="$OUT/tables"
mkdir -p "$SHARD_ROOT" "$LOG_ROOT" "$TABLE_ROOT"

cat <<EOF
========================================================================
 RIFT Table 6 — matched-area intervention validity
 gpus          : $GPU_LIST
 shards        : $SHARDS
 max_items     : $MAX_ITEMS global forged frames
 frames/video  : $FRAMES_PER_VIDEO (0 means all)
 grid/cells    : $GRID / $CELLS
 area          : $(python - <<PY
print(f"{int('$CELLS')/(int('$GRID')**2):.4f}")
PY
)
 operators     : $OPERATORS
 random draws  : $RANDOM_DRAWS per frame
 forward batch : $FORWARD_BATCH_SIZE
 bootstrap     : $BOOTSTRAP video-cluster replicates
 min ratio     : $MIN_RATIO
 config        : $ABLCFG
 output        : $OUT
========================================================================
EOF

pids=()
for shard_id in "${!GPUS_ARR[@]}"; do
  physical_gpu="${GPUS_ARR[$shard_id]}"
  shard_dir="$SHARD_ROOT/shard_${shard_id}"
  log_file="$LOG_ROOT/shard_${shard_id}.log"
  mkdir -p "$shard_dir"
  echo "[launch] shard $shard_id/$SHARDS on physical GPU $physical_gpu"
  (
    export CUDA_VISIBLE_DEVICES="$physical_gpu"
    export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
    python -u -m ablations.eval_table6_validity \
      --ablation-config "$ABLCFG" \
      --device cuda:0 \
      --max-items "$MAX_ITEMS" \
      --frames-per-video "$FRAMES_PER_VIDEO" \
      --shard-id "$shard_id" \
      --shard-count "$SHARDS" \
      --output-dir "$shard_dir" \
      --grid "$GRID" \
      --cells "$CELLS" \
      --operators "$OPERATORS" \
      --random-draws "$RANDOM_DRAWS" \
      --forward-batch-size "$FORWARD_BATCH_SIZE" \
      --seed "$SEED" \
      2>&1 | tee "$log_file"
  ) &
  pids+=("$!")
done

failed=0
for shard_id in "${!pids[@]}"; do
  if wait "${pids[$shard_id]}"; then
    echo "[ok] shard $shard_id finished"
  else
    echo "[error] shard $shard_id failed; inspect $LOG_ROOT/shard_${shard_id}.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 2
fi

for shard_id in "${!GPUS_ARR[@]}"; do
  file="$SHARD_ROOT/shard_${shard_id}/table6_samples.csv"
  if [[ ! -s "$file" ]]; then
    echo "[error] Missing shard result: $file" >&2
    exit 2
  fi
done

set +e
python -u -m ablations.eval_table6_validity \
  --summarize-only \
  --sample-glob "$SHARD_ROOT/shard_*/table6_samples.csv" \
  --output-dir "$TABLE_ROOT" \
  --bootstrap "$BOOTSTRAP" \
  --min-ratio "$MIN_RATIO" \
  --seed "$SEED"
summary_status=$?
set -e

cat <<EOF
========================================================================
 DONE — Table 6 outputs
 paper CSV : $TABLE_ROOT/table6_intervention_validity.csv
 markdown  : $TABLE_ROOT/table6_intervention_validity.md
 gate JSON : $TABLE_ROOT/table6_gate_status.json
 samples   : $TABLE_ROOT/table6_per_sample_merged.csv
 logs      : $LOG_ROOT
========================================================================
EOF

# Exit 3 means the experiment completed but Gate 1 did not pass.
exit "$summary_status"
