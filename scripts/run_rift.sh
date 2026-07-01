#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_rift.sh — RIFT generalized launcher (gates / audit / correlation / ablations / train)
#
# USAGE (always run from the RIFT root):
#   # Phase 0 gates first (precondition — single GPU, small slice):
#   bash scripts/run_rift.sh --mode gates  --csv data/slices/example_ffpp_forged.csv
#
#   # Dry-run the ablation plan (✓/✗ matrix, no model, no GPU):
#   bash scripts/run_rift.sh --mode ablations --dry-run
#
#   # Full ablation table (Blocks 1–4), 3 seeds:
#   bash scripts/run_rift.sh --mode ablations --seeds 0,1,2
#
#   # Just the decisive method cells:
#   bash scripts/run_rift.sh --mode ablations --block 1 --only generic_logit,delta_grounded
#
#   # Audit leaderboard / correlation only:
#   bash scripts/run_rift.sh --mode audit
#   bash scripts/run_rift.sh --mode correlation --corr-csv checkpoints_metrics.csv
#
#   # RL repair horizon sweep (Block 4 cells), one horizon per call:
#   bash scripts/run_rift.sh --gpus 0,1,2,3 --mode train --horizon 4
#
# COMMON FLAGS:
#   --gpus 4,5,6,7   --batch 8   --epochs 20   --workers 4   --no-wandb   --smoke
#   --cift-root /path/to/ImageDifussionFake   --ckpt /path/to/cift.ckpt
#   --config configs/rift_general.yaml   --csv <split.csv>   --seeds 0,1,2
#   --block {0|1|2|3|4}   --only id1,id2   --dry-run
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RIFT_PKG="${RIFT_ROOT}"
CIFT_ROOT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake"

CONFIG="${RIFT_ROOT}/configs/rift_general.yaml"
CSV="${RIFT_ROOT}/data/slices/rift_ffpp_rela.csv"

GPUS="0,1,2,3"
EPOCHS=20
BATCH=32
NUM_WORKERS=4
WANDB="true"
MODE="gates"
BLOCK=""
ONLY=""
SEEDS="0,1,2"
CKPT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake/experiments/Mix5_AllDomains_v5/ckpt/best-eer-epoch=34.ckpt"
HORIZON=""
CORR_CSV=""
DRY=0
SMOKE=0
EXTRA=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)      GPUS="$2"; shift 2 ;;
    --epochs)    EPOCHS="$2"; shift 2 ;;
    --batch)     BATCH="$2"; shift 2 ;;
    --workers)   NUM_WORKERS="$2"; shift 2 ;;
    --mode)      MODE="$2"; shift 2 ;;
    --block)     BLOCK="$2"; shift 2 ;;
    --only)      ONLY="$2"; shift 2 ;;
    --seeds)     SEEDS="$2"; shift 2 ;;
    --ckpt)      CKPT="$2"; shift 2 ;;
    --horizon)   HORIZON="$2"; shift 2 ;;
    --csv)       CSV="$2"; shift 2 ;;
    --corr-csv)  CORR_CSV="$2"; shift 2 ;;
    --config)    CONFIG="$2"; shift 2 ;;
    --cift-root) CIFT_ROOT="$2"; shift 2 ;;
    --no-wandb)  WANDB="false"; shift ;;
    --dry-run)   DRY=1; shift ;;
    --smoke)     SMOKE=1; shift ;;
    *)           EXTRA="$EXTRA $1"; shift ;;
  esac
done

if [[ $SMOKE -eq 1 ]]; then
  EPOCHS=1
  NUM_WORKERS=0
  WANDB="false"
  EXTRA="$EXTRA dataset.max_items=8"
  echo "[rift] SMOKE — tiny slice, wandb off"
fi

need_model=1
[[ "$MODE" == "correlation" || $DRY -eq 1 ]] && need_model=0

MISSING=0
for p in "$RIFT_PKG/src" "$CONFIG"; do
  [[ -e "$p" ]] || { echo "[ERROR] missing: $p"; MISSING=1; }
done

if [[ $need_model -eq 1 ]]; then
  [[ -e "$CIFT_ROOT" ]] || { echo "[ERROR] missing --cift-root: $CIFT_ROOT"; MISSING=1; }
  [[ -n "$CKPT" ]] || echo "[warn] --ckpt not set; set detector.cift_ckpt or pass --ckpt for real runs."
fi

[[ $MISSING -eq 1 ]] && { echo "Fix paths above, then re-run."; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="${RIFT_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "═══════════════════════════════════════════════════════════"
echo " RIFT  ·  mode=$MODE  block=${BLOCK:-all}  only=${ONLY:-all}"
echo " root      : $RIFT_ROOT"
echo " config    : $CONFIG"
echo " cift-root : $CIFT_ROOT"
echo " ckpt      : ${CKPT:-<unset>}"
echo " gpus      : $GPUS   seeds: $SEEDS   batch: $BATCH   epochs: $EPOCHS"
echo " wandb     : $WANDB   dry-run: $([[ $DRY -eq 1 ]] && echo yes || echo no)"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "${RIFT_ROOT}/outputs/cells" "${RIFT_ROOT}/outputs/logs"
cd "$RIFT_ROOT"

OV=( "device=cuda" "wandb.enabled=${WANDB}" "data.num_workers=${NUM_WORKERS}" )

[[ -n "$CKPT" ]] && OV+=( "detector.cift_ckpt=${CKPT}" )
[[ -n "$CSV"  ]] && OV+=( "dataset.split_csv=${CSV}" )
[[ -n "${EXTRA// /}" ]] && OV+=( $EXTRA )

case "$MODE" in
  gates)
    echo "─── Gate 1 (intervention validity) ───"
    python -m src.gates.gate1_validity \
      --csv "$CSV" --cift-root "$CIFT_ROOT" --ckpt "${CKPT}" \
      --device cuda --n 50 --min-sep 0.15 || echo "[gate1] FAIL/STOP — read the verdict above."

    echo "─── Gate 2 (novelty isolation) ───"
    python -m src.gates.gate2_separation \
      --csv "$CSV" --cift-root "$CIFT_ROOT" --ckpt "${CKPT}" --device cuda --margin 0.10 || true

    if [[ -n "$CORR_CSV" ]]; then
      echo "─── Gate 3 (correlation headline) ───"
      python -m src.gates.gate3_correlation --csv "$CORR_CSV" || true
    fi
    ;;

  ablations)
    CMD=( python "${RIFT_ROOT}/ablate_rift.py" -c "$CONFIG"
          --mode $([[ $DRY -eq 1 ]] && echo dry-run || echo ablations)
          --seeds "$SEEDS" --cift-root "$CIFT_ROOT" )
    [[ -n "$BLOCK" ]] && CMD+=( --block "$BLOCK" )
    [[ -n "$ONLY"  ]] && CMD+=( --only "$ONLY" )
    CMD+=( "${OV[@]}" )
    echo " ${CMD[*]}"
    "${CMD[@]}"
    ;;

  audit)
    python "${RIFT_ROOT}/ablate_rift.py" -c "$CONFIG" --mode ablations --block 2 \
      --cift-root "$CIFT_ROOT" "${OV[@]}"
    ;;

  correlation)
    [[ -n "$CORR_CSV" ]] || { echo "[ERROR] --corr-csv required for correlation mode"; exit 1; }
    python -m src.gates.gate3_correlation --csv "$CORR_CSV"
    ;;

  train)
    [[ -n "$HORIZON" ]] && OV+=( "rl.horizon=${HORIZON}" )
    OV+=( "rl.epochs=${EPOCHS}" "data.batch_size=${BATCH}" )
    [[ -n "$CSV" ]] && OV+=( "data.train_csv=${CSV}" "data.val_csv=${CSV}" )

    echo "─── RL repair policy (horizon=${HORIZON:-config}) ───"
    python "${RIFT_ROOT}/train_rift_rl.py" -c "$CONFIG" --cift-root "$CIFT_ROOT" "${OV[@]}"
    ;;

  *)
    echo "[ERROR] unknown --mode '$MODE' (gates|audit|correlation|ablations|train)"
    exit 1
    ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE ($MODE).  Outputs under ${RIFT_ROOT}/outputs/"
[[ -f "${RIFT_ROOT}/outputs/table_rift.csv" ]] && echo "  table : outputs/table_rift.csv"
echo "═══════════════════════════════════════════════════════════"

# chmod +x scripts/run_rift.sh
# bash scripts/run_rift.sh --mode ablations --dry-run
