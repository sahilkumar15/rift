#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_rift.sh — Robust RIFT launcher
#
# Modes:
#   gates        : intervention validity checks
#   audit        : baseline/rift-policy faithfulness audit
#   train        : RL repair-policy training
#   ablations    : ablation plan/table
#   correlation  : correlation analysis
#
# Multi-GPU behavior:
#   audit + --gpus 0,1,2,3 : shards CSV across GPUs and merges table
#   train + --gpus 0,1,2,3 : launches one independent seed per GPU
#
# Examples:
#   bash scripts/run_rift.sh --mode audit --gpus 0,1,2,3 --csv data/slices/rift_ffpp_rela_c23.csv dataset.max_items=1000 --no-wandb
#
#   bash scripts/run_rift.sh --mode train --gpus 0,1,2,3 --horizon 4 --csv data/slices/rift_ffpp_train_c23.csv \
#     --val-csv data/slices/rift_ffpp_val_c23.csv --epochs 20 --batch 8 --seeds 0,1,2,3 --no-wandb data.max_items=1000
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIFT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RIFT_PKG="${RIFT_ROOT}"

CIFT_ROOT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake"
CKPT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake/experiments/Mix5_AllDomains_v5/ckpt/best-eer-epoch=34.ckpt"

CONFIG="${RIFT_ROOT}/configs/rift_general.yaml"
CSV="${RIFT_ROOT}/data/slices/rift_ffpp_rela_c23.csv"
VAL_CSV=""

GPUS="0"
EPOCHS=20
BATCH=8
NUM_WORKERS=4
WANDB="true"
MODE="gates"
BLOCK=""
ONLY=""
SEEDS="0,1,2"
HORIZON=""
CORR_CSV=""
DRY=0
SMOKE=0
EXTRA=""

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

  if [[ "$EXTRA" != *"dataset.max_items="* && "$EXTRA" != *"data.max_items="* ]]; then
    EXTRA="$EXTRA dataset.max_items=8 data.max_items=8"
  fi

  if [[ "$EXTRA" != *"resume="* ]]; then
    EXTRA="$EXTRA resume=none"
  fi

  echo "[rift] SMOKE — tiny slice, wandb off, fresh resume=none"
fi

need_model=1
[[ "$MODE" == "correlation" || $DRY -eq 1 ]] && need_model=0

MISSING=0

for p in "$RIFT_PKG/src" "$CONFIG"; do
  [[ -e "$p" ]] || { echo "[ERROR] missing: $p"; MISSING=1; }
done

if [[ $need_model -eq 1 ]]; then
  [[ -e "$CIFT_ROOT" ]] || { echo "[ERROR] missing --cift-root: $CIFT_ROOT"; MISSING=1; }
  [[ -n "$CKPT" && -e "$CKPT" ]] || { echo "[ERROR] missing --ckpt: $CKPT"; MISSING=1; }
fi

[[ $MISSING -eq 1 ]] && { echo "Fix paths above, then re-run."; exit 1; }

mkdir -p "${RIFT_ROOT}/outputs/cells" "${RIFT_ROOT}/outputs/logs"
cd "$RIFT_ROOT"

export PYTHONPATH="${RIFT_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

OV=( "device=cuda" "wandb.enabled=${WANDB}" "data.num_workers=${NUM_WORKERS}" )

[[ -n "$CKPT" ]] && OV+=( "detector.cift_ckpt=${CKPT}" )
[[ -n "$CSV"  ]] && OV+=( "dataset.split_csv=${CSV}" )
[[ -n "${EXTRA// /}" ]] && OV+=( $EXTRA )

echo "═══════════════════════════════════════════════════════════"
echo " RIFT  ·  mode=$MODE  block=${BLOCK:-all}  only=${ONLY:-all}"
echo " root      : $RIFT_ROOT"
echo " config    : $CONFIG"
echo " cift-root : $CIFT_ROOT"
echo " ckpt      : ${CKPT:-<unset>}"
echo " csv       : ${CSV:-<unset>}"
[[ -n "$VAL_CSV" ]] && echo " val-csv   : $VAL_CSV"
echo " gpus      : $GPUS   seeds: $SEEDS   batch: $BATCH   epochs: $EPOCHS"
echo " wandb     : $WANDB   dry-run: $([[ $DRY -eq 1 ]] && echo yes || echo no)"
echo "═══════════════════════════════════════════════════════════"


split_csv_for_audit() {
  local src_csv="$1"
  local max_items="$2"
  local n_shards="$3"
  local shard_dir="$4"

  mkdir -p "$shard_dir"

  python - "$src_csv" "$max_items" "$n_shards" "$shard_dir" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
max_items = sys.argv[2]
n = int(sys.argv[3])
outdir = Path(sys.argv[4])

if max_items in ("", "null", "None", "none"):
    max_items = 0
else:
    max_items = int(max_items)

if not src.exists():
    raise SystemExit(f"[ERROR] missing CSV: {src}")

with open(src, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = []
    for r in reader:
        label = str(r.get("label", "1")).strip()
        if label not in ("1", "fake", "forged", "True", "true"):
            continue
        rows.append(r)
        if max_items > 0 and len(rows) >= max_items:
            break

if not rows:
    raise SystemExit("[ERROR] no forged rows available for audit sharding")

for i in range(n):
    part = rows[i::n]
    out = outdir / f"shard_{i}.csv"
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(part)
    print(f"[OK] shard={i} rows={len(part)} -> {out}")
PY
}


merge_audit_tables() {
  python - <<'PY'
from pathlib import Path
from collections import OrderedDict, defaultdict
import csv
import math

outdir = Path("outputs/mgpu_audit")
files = sorted(outdir.glob("table_rank*_gpu*.csv"))

if not files:
    raise SystemExit("[ERROR] no shard tables found")

KEY_COLS = ["explainer", "block", "identity_gap_mode"]
BOOL_COLS = {"exposed"}
COUNT_COLS = {"n"}
FIRST_COLS = [
    "explainer",
    "identity_gap_mode",
    "faithfulness_delta",
    "faithfulness_logit",
    "necessity_delta",
    "sufficiency_delta",
    "plausibility_iou",
    "mask_area",
    "identity_preservation",
    "perceptual_distance",
    "n",
    "exposed",
    "block",
]

def fnum(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v

def fbool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

groups = OrderedDict()
all_cols = []

for file in files:
    if file.stat().st_size == 0:
        continue

    with open(file, newline="") as fp:
        reader = csv.DictReader(fp)

        for row in reader:
            if not row:
                continue

            for c in row:
                if c not in all_cols:
                    all_cols.append(c)

            key = tuple(row.get(c, "") for c in KEY_COLS)

            if key not in groups:
                groups[key] = {
                    "keys": {c: row.get(c, "") for c in KEY_COLS},
                    "n": 0.0,
                    "sums": defaultdict(float),
                    "weights": defaultdict(float),
                    "bools": defaultdict(bool),
                    "strings": {},
                }

            g = groups[key]
            n = fnum(row.get("n")) or 1.0
            g["n"] += n

            for c, v in row.items():
                if c in KEY_COLS or c in COUNT_COLS:
                    continue

                if c in BOOL_COLS:
                    g["bools"][c] = g["bools"][c] or fbool(v)
                    continue

                fv = fnum(v)
                if fv is not None:
                    g["sums"][c] += fv * n
                    g["weights"][c] += n
                elif v not in (None, ""):
                    g["strings"][c] = v

rows = []

for _, g in groups.items():
    out = {}

    for c in KEY_COLS:
        out[c] = g["keys"].get(c, "")

    for c in all_cols:
        if c in KEY_COLS:
            continue
        if c == "n":
            out[c] = str(int(round(g["n"])))
        elif c in BOOL_COLS:
            out[c] = "True" if g["bools"].get(c, False) else "False"
        elif c in g["weights"] and g["weights"][c] > 0:
            out[c] = f"{(g['sums'][c] / g['weights'][c]):.4f}"
        elif c in g["strings"]:
            out[c] = g["strings"][c]
        else:
            out[c] = ""

    rows.append(out)

cols = [c for c in FIRST_COLS if any(c in r for r in rows)]
cols += [c for c in all_cols if c not in cols]

for path in [Path("outputs/mgpu_audit/table_rift_merged.csv"), Path("outputs/table_rift.csv")]:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

print("[OK] merged -> outputs/mgpu_audit/table_rift_merged.csv")
print("[OK] copied -> outputs/table_rift.csv")
with open("outputs/table_rift.csv") as f:
    print(f.read())
PY
}


run_single_audit() {
  export CUDA_VISIBLE_DEVICES="$GPUS"
  python "${RIFT_ROOT}/ablate_rift.py" -c "$CONFIG" --mode ablations --block 2 \
    --cift-root "$CIFT_ROOT" "${OV[@]}"
}


run_multi_audit() {
  IFS=',' read -ra GPU_ARR <<< "$GPUS"
  local n="${#GPU_ARR[@]}"
  local outdir="${RIFT_ROOT}/outputs/mgpu_audit"
  local sharddir="${outdir}/shards"

  rm -rf "$outdir"
  mkdir -p "$outdir" "$sharddir"

  local max_items="null"
  for x in "${OV[@]}"; do
    if [[ "$x" == dataset.max_items=* ]]; then
      max_items="${x#dataset.max_items=}"
    fi
  done

  if [[ "$max_items" == "null" ]]; then
    # Read from config fallback by not limiting here.
    max_items="0"
  fi

  echo "─── Multi-GPU audit sharding ───"
  echo " shards    : $n"
  echo " gpus      : $GPUS"
  echo " max_items : $max_items"
  echo " outdir    : outputs/mgpu_audit"

  split_csv_for_audit "$CSV" "$max_items" "$n" "$sharddir"

  PIDS=()

  for RANK in "${!GPU_ARR[@]}"; do
    GPU="${GPU_ARR[$RANK]}"
    SHARD_CSV="${sharddir}/shard_${RANK}.csv"
    TABLE="${outdir}/table_rank${RANK}_gpu${GPU}.csv"
    LOG="${outdir}/audit_rank${RANK}_gpu${GPU}.log"

    echo "[launch] audit shard=$RANK/$n gpu=$GPU log=$LOG"

    (
      export CUDA_VISIBLE_DEVICES="$GPU"
      export PYTHONPATH="${RIFT_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
      export PYTORCH_ALLOC_CONF="expandable_segments:True"

      python "${RIFT_ROOT}/ablate_rift.py" -c "$CONFIG" --mode ablations --block 2 \
        --cift-root "$CIFT_ROOT" \
        device=cuda \
        wandb.enabled=false \
        data.num_workers=0 \
        detector.cift_ckpt="$CKPT" \
        dataset.split_csv="$SHARD_CSV" \
        dataset.max_items=0 \
        output.table_csv="$TABLE"
    ) > "$LOG" 2>&1 &

    PIDS+=("$!")
  done

  FAIL=0

  for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
  done

  if [[ "$FAIL" -ne 0 ]]; then
    echo "[ERROR] one or more audit shards failed."
    echo "Check:"
    echo "  tail -n 100 outputs/mgpu_audit/audit_rank*_gpu*.log"
    exit 1
  fi

  merge_audit_tables
}


run_single_train() {
  export CUDA_VISIBLE_DEVICES="$GPUS"

  [[ -n "$HORIZON" ]] && OV+=( "rl.horizon=${HORIZON}" )

  OV+=( "rl.epochs=${EPOCHS}" "data.batch_size=${BATCH}" )

  [[ -n "$CSV" ]] && OV+=( "data.train_csv=${CSV}" )

  if [[ -n "$VAL_CSV" ]]; then
    OV+=( "data.val_csv=${VAL_CSV}" )
  elif [[ -n "$CSV" ]]; then
    OV+=( "data.val_csv=${CSV}" )
  fi

  echo "─── RL repair policy single-run (horizon=${HORIZON:-config}) ───"
  python "${RIFT_ROOT}/train_rift_rl.py" -c "$CONFIG" --cift-root "$CIFT_ROOT" "${OV[@]}"
}


run_multi_train_seeds() {
  IFS=',' read -ra GPU_ARR <<< "$GPUS"
  IFS=',' read -ra SEED_ARR <<< "$SEEDS"

  local n_gpu="${#GPU_ARR[@]}"
  local outroot="${RIFT_ROOT}/outputs/train_mgpu_h${HORIZON:-4}_e${EPOCHS}_$(date +%Y%m%d_%H%M%S)"

  mkdir -p "$outroot/logs"

  echo "─── Multi-GPU training by independent seeds ───"
  echo " gpus    : $GPUS"
  echo " seeds   : $SEEDS"
  echo " outroot : ${outroot#$RIFT_ROOT/}"

  PIDS=()

  for IDX in "${!SEED_ARR[@]}"; do
    SEED="${SEED_ARR[$IDX]}"
    GPU="${GPU_ARR[$((IDX % n_gpu))]}"

    RUN_DIR="${outroot}/seed_${SEED}"
    LOG="${outroot}/logs/train_seed${SEED}_gpu${GPU}.log"

    mkdir -p "$RUN_DIR/ckpt"

    echo "[launch] train seed=$SEED gpu=$GPU log=$LOG"

    (
      export CUDA_VISIBLE_DEVICES="$GPU"
      export PYTHONPATH="${RIFT_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
      export PYTORCH_ALLOC_CONF="expandable_segments:True"

      python "${RIFT_ROOT}/train_rift_rl.py" -c "$CONFIG" --cift-root "$CIFT_ROOT" \
        device=cuda \
        seed="$SEED" \
        wandb.enabled="$WANDB" \
        wandb.name="rift_h${HORIZON:-4}_seed${SEED}" \
        detector.cift_ckpt="$CKPT" \
        dataset.split_csv="$CSV" \
        data.train_csv="$CSV" \
        data.val_csv="${VAL_CSV:-$CSV}" \
        data.num_workers="$NUM_WORKERS" \
        data.batch_size="$BATCH" \
        rl.epochs="$EPOCHS" \
        rl.horizon="${HORIZON:-4}" \
        checkpoint.dir="$RUN_DIR/ckpt" \
        output.name="rift_h${HORIZON:-4}_seed${SEED}" \
        resume=none \
        $EXTRA
    ) > "$LOG" 2>&1 &

    PIDS+=("$!")
  done

  FAIL=0

  for PID in "${PIDS[@]}"; do
    wait "$PID" || FAIL=1
  done

  if [[ "$FAIL" -ne 0 ]]; then
    echo "[ERROR] one or more training seeds failed."
    echo "Check:"
    echo "  tail -n 100 ${outroot#$RIFT_ROOT/}/logs/train_seed*_gpu*.log"
    exit 1
  fi

  python - "$outroot" <<'PY'
from pathlib import Path
import csv
import re
import sys

outroot = Path(sys.argv[1])
rows = []

for seed_dir in sorted(outroot.glob("seed_*")):
    seed = seed_dir.name.replace("seed_", "")
    ckpts = sorted((seed_dir / "ckpt").glob("best_e*.pth"))

    def score(p):
        m = re.search(r"_(-?\d+\.\d+)\.pth$", p.name)
        return float(m.group(1)) if m else -1e9

    if ckpts:
        best = max(ckpts, key=score)
        rows.append({"seed": seed, "best_ckpt": str(best), "score": f"{score(best):.4f}", "status": "ok"})
    else:
        rows.append({"seed": seed, "best_ckpt": "", "score": "", "status": "missing"})

summary = outroot / "summary_best_ckpts.csv"

with open(summary, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "best_ckpt", "score", "status"])
    w.writeheader()
    w.writerows(rows)

print(f"[OK] summary -> {summary}")
with open(summary) as f:
    print(f.read())
PY
}



run_ddp_train() {
  IFS=',' read -ra GPU_ARR <<< "$GPUS"
  local nproc="${#GPU_ARR[@]}"
  local seed="${SEEDS%%,*}"
  local outroot="${RIFT_ROOT}/outputs/train_ddp_h${HORIZON:-4}_seed${seed}_e${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
  local log="${outroot}/train_ddp.log"

  mkdir -p "$outroot/ckpt" "$outroot/logs"

  export CUDA_VISIBLE_DEVICES="$GPUS"
  export PYTHONPATH="${RIFT_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
  export PYTORCH_ALLOC_CONF="expandable_segments:True"

  echo "─── DDP multi-GPU training: one policy, one seed, dataset distributed ───"
  echo " gpus       : $GPUS"
  echo " nproc      : $nproc"
  echo " seed       : $seed"
  echo " train_csv  : $CSV"
  echo " val_csv    : ${VAL_CSV:-$CSV}"
  echo " outroot    : ${outroot#$RIFT_ROOT/}"
  echo " log        : ${log#$RIFT_ROOT/}"

  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$nproc" \
    "${RIFT_ROOT}/train_rift_rl.py" \
      -c "$CONFIG" \
      --cift-root "$CIFT_ROOT" \
      device=cuda \
      seed="$seed" \
      wandb.enabled="$WANDB" \
      wandb.name="rift_ddp_h${HORIZON:-4}_seed${seed}" \
      detector.cift_ckpt="$CKPT" \
      dataset.split_csv="$CSV" \
      data.train_csv="$CSV" \
      data.val_csv="${VAL_CSV:-$CSV}" \
      data.num_workers="$NUM_WORKERS" \
      data.batch_size="$BATCH" \
      rl.epochs="$EPOCHS" \
      rl.horizon="${HORIZON:-4}" \
      checkpoint.dir="$outroot/ckpt" \
      output.name="rift_ddp_h${HORIZON:-4}_seed${seed}" \
      resume=none \
      $EXTRA 2>&1 | tee "$log"
}


case "$MODE" in
  gates)
    export CUDA_VISIBLE_DEVICES="$GPUS"

    echo "─── Gate 1 (intervention validity) ───"
    python -m src.gates.gate1_validity \
      --csv "$CSV" --cift-root "$CIFT_ROOT" --ckpt "$CKPT" \
      --device cuda --n 50 --min-sep 0.15 || echo "[gate1] FAIL/STOP — read verdict above."

    echo "─── Gate 2 (novelty isolation) ───"
    python -m src.gates.gate2_separation \
      --csv "$CSV" --cift-root "$CIFT_ROOT" --ckpt "$CKPT" --device cuda --margin 0.10 || true
    ;;

  ablations)
    export CUDA_VISIBLE_DEVICES="$GPUS"

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
    if [[ "$GPUS" == *,* ]]; then
      run_multi_audit
    else
      run_single_audit
    fi
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
    echo "[ERROR] unknown --mode '$MODE' (gates|audit|correlation|ablations|train)"
    exit 1
    ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE ($MODE). Outputs under ${RIFT_ROOT}/outputs/"
[[ -f "${RIFT_ROOT}/outputs/table_rift.csv" ]] && echo "  table : outputs/table_rift.csv"
echo "═══════════════════════════════════════════════════════════"
