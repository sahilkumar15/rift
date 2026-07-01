#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_rift.sh — generalized RIFT launcher
#
# Normal 4-GPU train:
#   bash scripts/run_rift.sh --gpus 0,1,2,3 --mode train --horizon 4
#
# Behavior:
#   - YAML owns train_csv, val_csv, batch, epochs, workers, wandb, resume.
#   - CLI overrides YAML only when explicitly passed.
#   - Multi-GPU train uses torchrun/DDP.
#   - Resume works because checkpoint dir is stable, not timestamped.
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RIFT_PKG="${RIFT_ROOT}"

CONFIG="${RIFT_ROOT}/configs/rift_general.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Fixed project paths
# These are your stable CIFT paths.
# CLI can still override them using --cift-root and --ckpt.
# ─────────────────────────────────────────────────────────────────────────────
CIFT_ROOT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake"
CKPT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake/experiments/Mix5_AllDomains_v5/ckpt/best-eer-epoch=34.ckpt"

# ─────────────────────────────────────────────────────────────────────────────
# CSVs
# Keep these empty by default.
# Empty means train/val CSVs come from configs/rift_general.yaml:
#   data.train_csv
#   data.val_csv
#
# Only set these when user explicitly passes:
#   --csv
#   --val-csv
# ─────────────────────────────────────────────────────────────────────────────
CSV=""
VAL_CSV=""

# ─────────────────────────────────────────────────────────────────────────────
# Runtime defaults
# Empty values mean YAML controls them.
# CLI can override using --epochs, --batch, --workers, --seeds, etc.
# ─────────────────────────────────────────────────────────────────────────────
GPUS="0,1,2,3"
EPOCHS="20"
BATCH="64"
NUM_WORKERS="4"
WANDB=""
MODE="gates"
BLOCK=""
ONLY=""
SEEDS=""
HORIZON=""
CORR_CSV=""
DRY=0
SMOKE=0
EXTRA=()

yaml_get() {
  local key="$1"
  python - "$CONFIG" "$key" <<'PY'
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
key = sys.argv[2]

try:
    import yaml
except Exception:
    print("")
    raise SystemExit(0)

if not cfg_path.exists():
    print("")
    raise SystemExit(0)

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

cur = cfg
for part in key.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        print("")
        raise SystemExit(0)

if cur is None:
    print("")
elif isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PY
}

abs_path() {
  local p="$1"

  if [[ -z "$p" ]]; then
    echo ""
  elif [[ "$p" = /* ]]; then
    echo "$p"
  else
    echo "${RIFT_ROOT}/${p}"
  fi
}

has_extra_prefix() {
  local prefix="$1"
  local x

  for x in "${EXTRA[@]}"; do
    [[ "$x" == "$prefix"* ]] && return 0
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --val-csv)   VAL_CSV="$2"; shift 2 ;;
    --corr-csv)  CORR_CSV="$2"; shift 2 ;;
    --config)    CONFIG="$2"; shift 2 ;;
    --cift-root) CIFT_ROOT="$2"; shift 2 ;;
    --wandb)     WANDB="true"; shift ;;
    --no-wandb)  WANDB="false"; shift ;;
    --dry-run)   DRY=1; shift ;;
    --smoke)     SMOKE=1; shift ;;
    *)           EXTRA+=("$1"); shift ;;
  esac
done

CONFIG="$(abs_path "$CONFIG")"

YAML_SEED="$(yaml_get seed)"
YAML_HORIZON="$(yaml_get rl.horizon)"
YAML_EPOCHS="$(yaml_get rl.epochs)"
YAML_BATCH="$(yaml_get data.batch_size)"
YAML_WORKERS="$(yaml_get data.num_workers)"
YAML_WANDB="$(yaml_get wandb.enabled)"
YAML_EXPERIMENT_ROOT="$(yaml_get experiment.root_dir)"
YAML_EXPERIMENT_NAME="$(yaml_get experiment.name)"
YAML_CIFT_ROOT="$(yaml_get detector.cift_root)"
YAML_CKPT="$(yaml_get detector.cift_ckpt)"
YAML_AUDIT_CSV="$(yaml_get dataset.split_csv)"
YAML_TRAIN_CSV="$(yaml_get data.train_csv)"
YAML_VAL_CSV="$(yaml_get data.val_csv)"
YAML_CKPT_DIR="$(yaml_get checkpoint.dir)"

[[ -z "$CIFT_ROOT" ]] && CIFT_ROOT="$YAML_CIFT_ROOT"
[[ -z "$CKPT" ]] && CKPT="$YAML_CKPT"

CIFT_ROOT="$(abs_path "$CIFT_ROOT")"
CKPT="$(abs_path "$CKPT")"

if [[ $SMOKE -eq 1 ]]; then
  EPOCHS="1"
  NUM_WORKERS="0"
  WANDB="false"
  EXTRA+=("dataset.max_items=8")
  EXTRA+=("data.max_items=8")
  EXTRA+=("data.val_max_items=8")
  EXTRA+=("rl.val_max_batches=1")
  EXTRA+=("resume=none")
  echo "[rift] SMOKE — tiny slice, wandb off, fresh resume=none"
fi

need_model=1
[[ "$MODE" == "correlation" || $DRY -eq 1 ]] && need_model=0

MISSING=0

for p in "$RIFT_PKG/src" "$CONFIG"; do
  [[ -e "$p" ]] || { echo "[ERROR] missing: $p"; MISSING=1; }
done

if [[ $need_model -eq 1 ]]; then
  [[ -n "$CIFT_ROOT" && -e "$CIFT_ROOT" ]] || { echo "[ERROR] missing cift_root: ${CIFT_ROOT:-<empty>}"; MISSING=1; }
  [[ -n "$CKPT" && -e "$CKPT" ]] || { echo "[ERROR] missing cift_ckpt: ${CKPT:-<empty>}"; MISSING=1; }
fi

[[ $MISSING -eq 1 ]] && { echo "Fix paths above, then re-run."; exit 1; }

EXPERIMENT_ROOT="${YAML_EXPERIMENT_ROOT:-experiments}"
EXPERIMENT_NAME="${YAML_EXPERIMENT_NAME:-RIFT}"
EXPERIMENT_DIR="$(abs_path "${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}")"

mkdir -p "${EXPERIMENT_DIR}/logs" "${EXPERIMENT_DIR}/ckpt"
cd "$RIFT_ROOT"

export PYTHONPATH="${RIFT_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

COMMON_OV=("device=cuda")

# Use YAML experiment.name as the deterministic W&B display name.
# This prevents W&B names like rift_ddp_h4_seed3407.
if ! has_extra_prefix "wandb.name="; then
  COMMON_OV+=("wandb.name=${EXPERIMENT_NAME}")
fi

# Keep legacy exp_name aligned too, because the train code may use exp_name fallback.
if ! has_extra_prefix "exp_name="; then
  COMMON_OV+=("exp_name=${EXPERIMENT_NAME}")
fi

[[ -n "$WANDB" ]] && COMMON_OV+=("wandb.enabled=${WANDB}")
[[ -n "$CKPT" ]] && COMMON_OV+=("detector.cift_ckpt=${CKPT}")
[[ "${#EXTRA[@]}" -gt 0 ]] && COMMON_OV+=("${EXTRA[@]}")

DISPLAY_HORIZON="${HORIZON:-${YAML_HORIZON:-<yaml>}}"
DISPLAY_EPOCHS="${EPOCHS:-${YAML_EPOCHS:-<yaml>}}"
DISPLAY_BATCH="${BATCH:-${YAML_BATCH:-<yaml>}}"
DISPLAY_WORKERS="${NUM_WORKERS:-${YAML_WORKERS:-<yaml>}}"
DISPLAY_SEEDS="${SEEDS:-${YAML_SEED:-<yaml>}}"
DISPLAY_WANDB="${WANDB:-${YAML_WANDB:-<yaml>}}"

echo "═══════════════════════════════════════════════════════════"
echo " RIFT  ·  mode=$MODE  block=${BLOCK:-all}  only=${ONLY:-all}"
echo " root      : $RIFT_ROOT"
echo " config    : $CONFIG"
echo " experiment: ${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}"
echo " cift-root : ${CIFT_ROOT:-<from yaml>}"
echo " ckpt      : ${CKPT:-<from yaml>}"
echo " csv       : ${CSV:-<from yaml>}"
echo " val-csv   : ${VAL_CSV:-<from yaml>}"
echo " gpus      : $GPUS   seeds: $DISPLAY_SEEDS   batch: $DISPLAY_BATCH   epochs: $DISPLAY_EPOCHS"
echo " workers   : $DISPLAY_WORKERS   horizon: $DISPLAY_HORIZON"
echo " wandb     : $DISPLAY_WANDB   dry-run: $([[ $DRY -eq 1 ]] && echo yes || echo no)"
echo "═══════════════════════════════════════════════════════════"

get_audit_csv() {
  local c="${CSV:-$YAML_AUDIT_CSV}"
  c="$(abs_path "$c")"

  if [[ -z "$c" ]]; then
    echo "[ERROR] no audit CSV found. Set dataset.split_csv in YAML or pass --csv." >&2
    exit 1
  fi

  echo "$c"
}

make_train_overrides() {
  TRAIN_OV=()

  [[ -n "$HORIZON" ]] && TRAIN_OV+=("rl.horizon=${HORIZON}")
  [[ -n "$EPOCHS" ]] && TRAIN_OV+=("rl.epochs=${EPOCHS}")
  [[ -n "$BATCH" ]] && TRAIN_OV+=("data.batch_size=${BATCH}")
  [[ -n "$NUM_WORKERS" ]] && TRAIN_OV+=("data.num_workers=${NUM_WORKERS}")

  # Only override YAML CSVs if explicitly passed.
  if [[ -n "$CSV" ]]; then
    TRAIN_OV+=("dataset.split_csv=${CSV}")
    TRAIN_OV+=("data.train_csv=${CSV}")
  fi

  if [[ -n "$VAL_CSV" ]]; then
    TRAIN_OV+=("data.val_csv=${VAL_CSV}")
  fi
}

stable_ckpt_dir() {
  local run_kind="$1"
  local seed="$2"
  local horizon="$3"

  if has_extra_prefix "checkpoint.dir="; then
    echo ""
    return
  fi

  local root="${YAML_EXPERIMENT_ROOT:-experiments}"
  local name="${YAML_EXPERIMENT_NAME:-RIFT}"
  local base="${YAML_CKPT_DIR:-${root}/${name}/ckpt}"

  base="$(abs_path "$base")"

  # CIFT-style default:
  #   experiments/<experiment.name>/ckpt
  #
  # No ddp/single/seed suffix here, because experiment.name already identifies
  # the run in YAML and W&B.
  echo "$base"
}

run_single_train() {
  export CUDA_VISIBLE_DEVICES="$GPUS"

  make_train_overrides

  local seed="${SEEDS%%,*}"
  [[ -z "$seed" ]] && seed="${YAML_SEED:-3407}"

  local h_label="${HORIZON:-${YAML_HORIZON:-4}}"

  local ckpt_dir
  ckpt_dir="$(stable_ckpt_dir single "$seed" "$h_label")"

  CKPT_OV=()
  [[ -n "$ckpt_dir" ]] && CKPT_OV+=("checkpoint.dir=${ckpt_dir}")

  [[ -n "$ckpt_dir" ]] && mkdir -p "$ckpt_dir"

  echo "─── RL repair policy single-GPU train ───"
  echo " train_csv : ${CSV:-${YAML_TRAIN_CSV:-<from yaml>}}"
  echo " val_csv   : ${VAL_CSV:-${YAML_VAL_CSV:-<from yaml>}}"
  echo " ckpt_dir  : ${ckpt_dir#$RIFT_ROOT/}"

  python "${RIFT_ROOT}/train_rift_rl.py" \
    -c "$CONFIG" \
    --cift-root "$CIFT_ROOT" \
    "${COMMON_OV[@]}" \
    "${TRAIN_OV[@]}" \
    "${CKPT_OV[@]}" \
    seed="$seed"
}

run_ddp_train() {
  IFS=',' read -ra GPU_ARR <<< "$GPUS"

  local nproc="${#GPU_ARR[@]}"
  local seed="${SEEDS%%,*}"

  [[ -z "$seed" ]] && seed="${YAML_SEED:-3407}"

  local h_label="${HORIZON:-${YAML_HORIZON:-4}}"
  local e_label="${EPOCHS:-${YAML_EPOCHS:-yaml}}"

  local ckpt_dir
  ckpt_dir="$(stable_ckpt_dir ddp "$seed" "$h_label")"

  local logroot="${EXPERIMENT_DIR}/logs/train_ddp_h${h_label}_seed${seed}_e${e_label}_$(date +%Y%m%d_%H%M%S)"
  local log="${logroot}/train_ddp.log"

  mkdir -p "$logroot"
  [[ -n "$ckpt_dir" ]] && mkdir -p "$ckpt_dir"

  export CUDA_VISIBLE_DEVICES="$GPUS"
  export PYTHONPATH="${RIFT_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

  make_train_overrides

  CKPT_OV=()
  [[ -n "$ckpt_dir" ]] && CKPT_OV+=("checkpoint.dir=${ckpt_dir}")

  echo "─── DDP multi-GPU training: one policy, one seed, dataset distributed ───"
  echo " gpus       : $GPUS"
  echo " nproc      : $nproc"
  echo " seed       : $seed"
  echo " train_csv  : ${CSV:-${YAML_TRAIN_CSV:-<from yaml>}}"
  echo " val_csv    : ${VAL_CSV:-${YAML_VAL_CSV:-<from yaml>}}"
  echo " batch      : ${BATCH:-${YAML_BATCH:-<from yaml>}}"
  echo " epochs     : ${EPOCHS:-${YAML_EPOCHS:-<from yaml>}}"
  echo " ckpt_dir   : ${ckpt_dir#$RIFT_ROOT/}"
  echo " log        : ${log#$RIFT_ROOT/}"

  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$nproc" \
    "${RIFT_ROOT}/train_rift_rl.py" \
      -c "$CONFIG" \
      --cift-root "$CIFT_ROOT" \
      "${COMMON_OV[@]}" \
      "${TRAIN_OV[@]}" \
      "${CKPT_OV[@]}" \
      seed="$seed" \
      2>&1 | tee "$log"
}

run_gates() {
  export CUDA_VISIBLE_DEVICES="$GPUS"

  local gate_csv
  gate_csv="$(get_audit_csv)"

  echo "─── Gate 1 intervention validity ───"
  python -m src.gates.gate1_validity \
    --csv "$gate_csv" \
    --cift-root "$CIFT_ROOT" \
    --ckpt "$CKPT" \
    --device cuda \
    --n 50 \
    --min-sep 0.15 || echo "[gate1] FAIL/STOP — read verdict above."

  echo "─── Gate 2 novelty isolation ───"
  python -m src.gates.gate2_separation \
    --csv "$gate_csv" \
    --cift-root "$CIFT_ROOT" \
    --ckpt "$CKPT" \
    --device cuda \
    --margin 0.10 || true
}

run_ablations() {
  export CUDA_VISIBLE_DEVICES="$GPUS"

  CMD=(
    python "${RIFT_ROOT}/ablate_rift.py"
    -c "$CONFIG"
    --mode "$([[ $DRY -eq 1 ]] && echo dry-run || echo ablations)"
    --cift-root "$CIFT_ROOT"
  )

  [[ -n "$SEEDS" ]] && CMD+=(--seeds "$SEEDS")
  [[ -n "$BLOCK" ]] && CMD+=(--block "$BLOCK")
  [[ -n "$ONLY" ]] && CMD+=(--only "$ONLY")

  CMD+=("${COMMON_OV[@]}")

  if [[ -n "$CSV" ]]; then
    CMD+=("dataset.split_csv=$CSV")
  fi

  echo " ${CMD[*]}"
  "${CMD[@]}"
}

run_audit() {
  export CUDA_VISIBLE_DEVICES="$GPUS"

  AUDIT_OV=("${COMMON_OV[@]}")

  if [[ -n "$CSV" ]]; then
    AUDIT_OV+=("dataset.split_csv=$CSV")
  fi

  python "${RIFT_ROOT}/ablate_rift.py" \
    -c "$CONFIG" \
    --mode ablations \
    --block 2 \
    --cift-root "$CIFT_ROOT" \
    "${AUDIT_OV[@]}"
}

case "$MODE" in
  gates)
    run_gates
    ;;

  ablations)
    run_ablations
    ;;

  audit)
    run_audit
    ;;

  correlation)
    [[ -n "$CORR_CSV" ]] || { echo "[ERROR] --corr-csv required for correlation mode"; exit 1; }
    python -m src.gates.gate3_correlation --csv "$CORR_CSV"
    ;;

  train)
    if [[ "$GPUS" == *,* ]]; then
      run_ddp_train
    else
      run_single_train
    fi
    ;;

  *)
    echo "[ERROR] unknown --mode '$MODE' expected gates|audit|correlation|ablations|train"
    exit 1
    ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE ($MODE). Outputs under ${EXPERIMENT_DIR#$RIFT_ROOT/}/"
[[ -f "${EXPERIMENT_DIR}/table_rift.csv" ]] && echo " table : ${EXPERIMENT_DIR#$RIFT_ROOT/}/table_rift.csv"
echo "═══════════════════════════════════════════════════════════"
