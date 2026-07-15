#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
GPU="${GPU:-6,7}"
TABLES="${TABLES:-table1_component}"
MAX_ITEMS="${MAX_ITEMS:-full}"
SKIP_OK_EXISTING="${SKIP_OK_EXISTING:-false}"

# Generalized eval saving.
# Change these when needed.
EVAL_ROOT="${EVAL_ROOT:-experiments/ablations/rift_table123/eval}"
SAFE_TABLES="${TABLES//,/__}"
SAFE_MAX="${MAX_ITEMS//\//_}"
EVAL_NAME="${EVAL_NAME:-${SAFE_TABLES}_${SAFE_MAX}}"

EVAL_DIR="${EVAL_ROOT}/${EVAL_NAME}"
SHARD_ROOT="${EVAL_DIR}/shards"
LOG_ROOT="${EVAL_DIR}/logs"
OUT_DIR="${EVAL_DIR}/tables"

python ablations/patch_for_ablations.py

CIFT_ROOT="$(python - "$ABLCFG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['cift']['root'])
PY
)"

IFS=',' read -r -a GPUS_ARR <<< "$GPU"
SHARD_COUNT="${#GPUS_ARR[@]}"

if [[ "$SHARD_COUNT" -lt 1 ]]; then
  echo "[error] No GPUs found in GPU='$GPU'" >&2
  exit 1
fi

mkdir -p "$SHARD_ROOT" "$LOG_ROOT" "$OUT_DIR"

export PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python -m py_compile ablations/eval_table123.py

echo "═══════════════════════════════════════════════════════════"
echo " RIFT Table123 PARALLEL evaluation"
echo " gpus        : $GPU"
echo " shards      : $SHARD_COUNT"
echo " tables      : $TABLES"
echo " max_items   : $MAX_ITEMS"
echo " config      : $ABLCFG"
echo " eval_root   : $EVAL_ROOT"
echo " eval_name   : $EVAL_NAME"
echo " shard_root  : $SHARD_ROOT"
echo " log_root    : $LOG_ROOT"
echo " out_dir     : $OUT_DIR"
echo " skip_ok     : $SKIP_OK_EXISTING"
echo "═══════════════════════════════════════════════════════════"

pids=()

for i in "${!GPUS_ARR[@]}"; do
  gpu_i="${GPUS_ARR[$i]}"
  shard_dir="$SHARD_ROOT/shard_${i}"
  mkdir -p "$shard_dir"

  echo "[launch] shard $i/$SHARD_COUNT on physical GPU $gpu_i"

  cmd="cd $(pwd) && \
CUDA_VISIBLE_DEVICES=$gpu_i \
SHARD_ID=$i \
SHARD_COUNT=$SHARD_COUNT \
PYTHONUNBUFFERED=1 \
PYTHONPATH=$(pwd):$CIFT_ROOT:${PYTHONPATH:-} \
python -u -m ablations.eval_table123 \
  --ablation-config $ABLCFG \
  --tables $TABLES \
  --device cuda:0 \
  --max-items $MAX_ITEMS \
  --shard-id $i \
  --shard-count $SHARD_COUNT \
  --output-dir $shard_dir"

  if [[ "$SKIP_OK_EXISTING" == "true" ]]; then
    cmd="$cmd --skip-ok-existing"
  fi

  # pseudo-terminal mode keeps tqdm as real progress bars
  script -q -f -c "$cmd" "$LOG_ROOT/shard_${i}.log" &

  pids+=("$!")
done

fail=0

for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[ok] shard $i finished"
  else
    echo "[FAILED] shard $i failed. See $LOG_ROOT/shard_${i}.log" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[error] one or more shards failed" >&2
  exit 1
fi

python - "$ABLCFG" "$TABLES" "$SHARD_ROOT" "$SHARD_COUNT" "$OUT_DIR" <<'PY'
import csv
import sys
from pathlib import Path
import yaml

ABLCFG, TABLES, SHARD_ROOT, SHARD_COUNT, OUT_DIR = (
    sys.argv[1],
    sys.argv[2],
    Path(sys.argv[3]),
    int(sys.argv[4]),
    Path(sys.argv[5]),
)

manifest = yaml.safe_load(open(ABLCFG))
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS = {
    "table", "ID", "Variant", "Mask source", "ΔG", "NS", "RP",
    "Necessity", "Sufficiency", "Sparsity", "Horizon",
    "n", "status", "error",
}

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def to_float(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(str(x).replace(",", ""))
    except Exception:
        return None

def to_int(x):
    try:
        return int(float(str(x).replace(",", "")))
    except Exception:
        return 0

def merge_rows(shard_rows, ordered_ids):
    by_id = {}
    for rows in shard_rows:
        for r in rows:
            by_id.setdefault(str(r.get("ID", "")), []).append(r)

    merged = []
    for rid in ordered_ids:
        parts = by_id.get(str(rid), [])
        if not parts:
            continue

        first = dict(parts[0])
        out = dict(first)
        n_sum = sum(to_int(p.get("n", 0)) for p in parts)
        out["n"] = n_sum

        failed = [p for p in parts if str(p.get("status", "ok")).lower() != "ok"]
        out["status"] = "FAILED" if failed else "ok"
        out["error"] = " | ".join(p.get("error", "") for p in failed if p.get("error"))

        for col in list(first.keys()):
            if col in META_COLS:
                continue

            vals, weights = [], []
            for p in parts:
                if str(p.get("status", "ok")).lower() != "ok":
                    continue
                v = to_float(p.get(col))
                if v is None:
                    vals = []
                    break
                vals.append(v)
                weights.append(to_int(p.get("n", 0)))

            if vals:
                denom = sum(weights) if sum(weights) > 0 else len(vals)
                num = sum(v * (w if sum(weights) > 0 else 1) for v, w in zip(vals, weights))
                out[col] = f"{num / max(1, denom):.4f}"

        merged.append(out)

    return merged

wanted = [x.strip() for x in TABLES.split(",") if x.strip()]
combined = []

for table_key in wanted:
    if table_key not in manifest["tables"]:
        raise SystemExit(f"Unknown table key {table_key}. Available: {list(manifest['tables'])}")

    spec = manifest["tables"][table_key]
    fname = spec["filename"]

    shard_rows = []
    for i in range(SHARD_COUNT):
        path = SHARD_ROOT / f"shard_{i}" / fname
        if not path.exists():
            raise SystemExit(f"Missing shard CSV: {path}")
        shard_rows.append(read_csv(path))

    ordered_ids = [r["id"] for r in spec["rows"]]
    merged = merge_rows(shard_rows, ordered_ids)

    final_path = OUT_DIR / fname
    write_csv(final_path, merged)
    print(f"[merged] {table_key}: {final_path}")

    for r in merged:
        combined.append({"table": table_key, **r})

combined_path = OUT_DIR / "combined_tables_1_2_3.csv"
write_csv(combined_path, combined)
print(f"[merged] combined: {combined_path}")
PY

echo "[done] evaluation complete"
echo "[logs]   $LOG_ROOT"
echo "[tables] $OUT_DIR"
