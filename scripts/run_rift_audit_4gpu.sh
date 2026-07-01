#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_rift_audit_4gpu.sh data/slices/rift_ffpp_rela.csv 200 0,1,2,3
#
# Args:
#   $1 = CSV path
#   $2 = max items
#   $3 = GPU list

CSV="${1:-data/slices/rift_ffpp_rela.csv}"
MAX_ITEMS="${2:-200}"
GPUS="${3:-0,1,2,3}"

RIFT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIFT_ROOT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake"
CONFIG="${RIFT_ROOT}/configs/rift_general.yaml"
CKPT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake/experiments/Mix5_AllDomains_v5/ckpt/best-eer-epoch=34.ckpt"

cd "$RIFT_ROOT"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N_SHARDS="${#GPU_ARR[@]}"

OUTDIR="${RIFT_ROOT}/outputs/mgpu_audit"
SHARDDIR="${OUTDIR}/shards"
mkdir -p "$OUTDIR" "$SHARDDIR"

echo "═══════════════════════════════════════════════════════════"
echo " RIFT MULTI-GPU AUDIT"
echo " root      : $RIFT_ROOT"
echo " csv       : $CSV"
echo " max_items : $MAX_ITEMS"
echo " gpus      : $GPUS"
echo " shards    : $N_SHARDS"
echo " outdir    : outputs/mgpu_audit"
echo "═══════════════════════════════════════════════════════════"

echo "[split] creating shard CSVs..."

python - "$CSV" "$MAX_ITEMS" "$N_SHARDS" "$SHARDDIR" <<'PY_SPLIT'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
max_items = int(sys.argv[2])
n_shards = int(sys.argv[3])
outdir = Path(sys.argv[4])

if not csv_path.exists():
    raise SystemExit(f"[ERROR] CSV not found: {csv_path}")

rows = []

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames

    for row in reader:
        label = str(row.get("label", "1")).strip()

        if label not in ("1", "fake", "forged", "True", "true"):
            continue

        rows.append(row)

        if len(rows) >= max_items:
            break

if not rows:
    raise SystemExit("[ERROR] no forged rows found for audit")

for shard_id in range(n_shards):
    shard_rows = rows[shard_id::n_shards]
    out = outdir / f"rift_audit_shard{shard_id}.csv"

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shard_rows)

    print(f"[OK] shard={shard_id} rows={len(shard_rows)} -> {out}")
PY_SPLIT

PIDS=()

for RANK in "${!GPU_ARR[@]}"; do
  GPU="${GPU_ARR[$RANK]}"
  SHARD_CSV="${SHARDDIR}/rift_audit_shard${RANK}.csv"
  LOG="${OUTDIR}/audit_rank${RANK}_gpu${GPU}.log"
  TABLE="${OUTDIR}/table_rank${RANK}_gpu${GPU}.csv"

  echo "[launch] rank=$RANK gpu=$GPU csv=$SHARD_CSV log=$LOG"

  (
    export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTHONPATH="${RIFT_ROOT}:${CIFT_ROOT}:${PYTHONPATH:-}"
    export PYTORCH_ALLOC_CONF="expandable_segments:True"

    python "${RIFT_ROOT}/ablate_rift.py" \
      -c "$CONFIG" \
      --mode ablations \
      --block 2 \
      --cift-root "$CIFT_ROOT" \
      device=cuda \
      wandb.enabled=false \
      data.num_workers=0 \
      detector.cift_ckpt="$CKPT" \
      dataset.split_csv="$SHARD_CSV" \
      dataset.max_items="$MAX_ITEMS" \
      output.table_csv="$TABLE"

  ) > "$LOG" 2>&1 &

  PIDS+=("$!")
done

echo "[wait] waiting for all GPU workers..."

FAIL=0

for PID in "${PIDS[@]}"; do
  wait "$PID" || FAIL=1
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[ERROR] one or more GPU audit workers failed."
  echo "Check logs:"
  echo "  tail -n 100 outputs/mgpu_audit/audit_rank*_gpu*.log"
  exit 1
fi

echo "[merge] merging shard outputs..."

python - <<'PY_MERGE'
from pathlib import Path
from collections import OrderedDict, defaultdict
import csv
import math
import shutil

outdir = Path("outputs/mgpu_audit")
files = sorted(outdir.glob("table_rank*_gpu*.csv"))

if not files:
    raise SystemExit("[ERROR] no shard table files found")

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

def parse_float(x):
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

def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

groups = OrderedDict()
all_cols = []

for f in files:
    if f.stat().st_size == 0:
        continue

    with open(f, newline="") as fp:
        reader = csv.DictReader(fp)

        for row in reader:
            if not row:
                continue

            for c in row.keys():
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
            n = parse_float(row.get("n")) or 1.0
            g["n"] += n

            for c, val in row.items():
                if c in KEY_COLS:
                    continue

                if c in COUNT_COLS:
                    continue

                if c in BOOL_COLS:
                    g["bools"][c] = g["bools"][c] or parse_bool(val)
                    continue

                fv = parse_float(val)

                if fv is not None:
                    g["sums"][c] += fv * n
                    g["weights"][c] += n
                else:
                    if val not in (None, ""):
                        g["strings"][c] = val

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

merged_path = Path("outputs/mgpu_audit/table_rift_4gpu_merged.csv")
standard_path = Path("outputs/table_rift.csv")

for path in [merged_path, standard_path]:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})

print(f"[OK] merged -> {merged_path}")
print(f"[OK] copied -> {standard_path}")

with open(standard_path) as f:
    print(f.read())
PY_MERGE

echo "═══════════════════════════════════════════════════════════"
echo " DONE MULTI-GPU AUDIT"
echo " table  : outputs/table_rift.csv"
echo " merged : outputs/mgpu_audit/table_rift_4gpu_merged.csv"
echo " logs   : outputs/mgpu_audit/audit_rank*_gpu*.log"
echo "═══════════════════════════════════════════════════════════"
