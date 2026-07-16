#!/usr/bin/env bash
set -euo pipefail

ABLCFG="${ABLCFG:-ablations/configs/table123_rift.yaml}"
GPU="${GPU:-6,7}"
TABLES="${TABLES:-table1_component}"
MAX_ITEMS="${MAX_ITEMS:-full}"
SKIP_OK_EXISTING="${SKIP_OK_EXISTING:-true}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FORWARD_BATCH_SIZE="${FORWARD_BATCH_SIZE:-32}"

EVAL_ROOT="${EVAL_ROOT:-experiments/ablations/rift_table123/eval}"
SAFE_TABLES="${TABLES//,/__}"
SAFE_MAX="${MAX_ITEMS//\//_}"
EVAL_NAME="${EVAL_NAME:-${SAFE_TABLES}_${SAFE_MAX}}"

EVAL_DIR="${EVAL_ROOT}/${EVAL_NAME}"
SHARD_ROOT="${EVAL_DIR}/shards"
LOG_ROOT="${EVAL_DIR}/logs"
OUT_DIR="${EVAL_DIR}/tables"
PROGRESS_ROOT="${EVAL_DIR}/progress"

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

mkdir -p "$SHARD_ROOT" "$LOG_ROOT" "$OUT_DIR" "$PROGRESS_ROOT"
rm -f "$PROGRESS_ROOT"/*.json "$PROGRESS_ROOT"/*.exit 2>/dev/null || true

export PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m py_compile ablations/eval_table123.py ablations/lib/explainers.py

echo "═══════════════════════════════════════════════════════════"
echo " RIFT Table123 FAST PARALLEL evaluation"
echo " gpus          : $GPU"
echo " shards        : $SHARD_COUNT"
echo " tables        : $TABLES"
echo " max_items     : $MAX_ITEMS"
echo " eval_batch    : $BATCH_SIZE images"
echo " forward_batch : $FORWARD_BATCH_SIZE CIFT inputs"
echo " config        : $ABLCFG"
echo " eval_name     : $EVAL_NAME"
echo " logs          : $LOG_ROOT"
echo " tables        : $OUT_DIR"
echo " skip_ok       : $SKIP_OK_EXISTING"
echo "═══════════════════════════════════════════════════════════"

pids=()

for i in "${!GPUS_ARR[@]}"; do
  gpu_i="${GPUS_ARR[$i]}"
  shard_dir="$SHARD_ROOT/shard_${i}"
  log_file="$LOG_ROOT/shard_${i}.log"
  progress_file="$PROGRESS_ROOT/shard_${i}.json"
  exit_file="$PROGRESS_ROOT/shard_${i}.exit"
  mkdir -p "$shard_dir"

  echo "[launch] shard $i/$SHARD_COUNT on physical GPU $gpu_i"

  cmd=(
    python -u -m ablations.eval_table123
    --ablation-config "$ABLCFG"
    --tables "$TABLES"
    --device cuda:0
    --max-items "$MAX_ITEMS"
    --batch-size "$BATCH_SIZE"
    --forward-batch-size "$FORWARD_BATCH_SIZE"
    --shard-id "$i"
    --shard-count "$SHARD_COUNT"
    --output-dir "$shard_dir"
    --progress-file "$progress_file"
    --no-tqdm
  )

  if [[ "$SKIP_OK_EXISTING" == "true" || "$SKIP_OK_EXISTING" == "1" ]]; then
    cmd+=(--skip-ok-existing)
  fi

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu_i" \
    SHARD_ID="$i" \
    SHARD_COUNT="$SHARD_COUNT" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$(pwd):$CIFT_ROOT:${PYTHONPATH:-}" \
      "${cmd[@]}" >"$log_file" 2>&1
    rc=$?
    printf '%s\n' "$rc" >"$exit_file"
    exit "$rc"
  ) &
  pids+=("$!")
done

# One clean aggregate tqdm bar. Worker output stays in per-GPU logs.
python - "$PROGRESS_ROOT" "$SHARD_COUNT" <<'PY'
import json
import sys
import time
from pathlib import Path
from tqdm import tqdm

root = Path(sys.argv[1])
count = int(sys.argv[2])
bar = tqdm(total=1, desc="Loading CIFT", unit="img", dynamic_ncols=True, mininterval=0.5)
last_done = 0

while True:
    states = []
    for i in range(count):
        path = root / f"shard_{i}.json"
        if not path.exists():
            continue
        try:
            states.append(json.loads(path.read_text()))
        except Exception:
            pass

    if states:
        total = sum(max(0, int(s.get("overall_total", 0))) for s in states)
        done = sum(max(0, int(s.get("overall_done", 0))) for s in states)
        if total > 0 and bar.total != total:
            bar.total = total
            bar.refresh()
        delta = max(0, done - last_done)
        if delta:
            bar.update(delta)
            last_done = done

        active = []
        for s in states:
            name = str(s.get("variant", "")).strip()
            if name and name not in ("Complete", "Loading CIFT") and name not in active:
                active.append(name)
        if active:
            bar.set_description(" | ".join(active[:2]))
        elif all(str(s.get("variant")) == "Complete" for s in states):
            bar.set_description("Complete")

    exits = [root / f"shard_{i}.exit" for i in range(count)]
    if all(path.exists() for path in exits):
        break
    time.sleep(1.0)

if bar.total > 0 and last_done < bar.total:
    bar.update(bar.total - last_done)
bar.close()
PY

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[ok] shard $i finished"
  else
    echo "[FAILED] shard $i failed. Last log lines:" >&2
    tail -80 "$LOG_ROOT/shard_${i}.log" >&2 || true
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
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def to_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def to_int(value):
    try:
        return int(float(str(value).replace(",", "")))
    except Exception:
        return 0


def merge_rows(shard_rows, ordered_ids):
    by_id = {}
    for rows in shard_rows:
        for row in rows:
            by_id.setdefault(str(row.get("ID", "")), []).append(row)

    merged = []
    for row_id in ordered_ids:
        parts = by_id.get(str(row_id), [])
        if not parts:
            continue

        output = dict(parts[0])
        n_sum = sum(to_int(part.get("n", 0)) for part in parts)
        output["n"] = n_sum

        failed = [
            part
            for part in parts
            if str(part.get("status", "ok")).lower() != "ok"
        ]
        output["status"] = "FAILED" if failed else "ok"
        output["error"] = " | ".join(
            part.get("error", "") for part in failed if part.get("error")
        )

        for column in list(output.keys()):
            if column in META_COLS:
                continue
            values = []
            weights = []
            for part in parts:
                if str(part.get("status", "ok")).lower() != "ok":
                    continue
                value = to_float(part.get(column))
                if value is None:
                    values = []
                    break
                values.append(value)
                weights.append(to_int(part.get("n", 0)))

            if values:
                weight_sum = sum(weights)
                if weight_sum > 0:
                    merged_value = sum(v * w for v, w in zip(values, weights)) / weight_sum
                else:
                    merged_value = sum(values) / len(values)
                output[column] = f"{merged_value:.4f}"

        merged.append(output)

    return merged


wanted = [item.strip() for item in TABLES.split(",") if item.strip()]
combined = []

for table_key in wanted:
    if table_key not in manifest["tables"]:
        raise SystemExit(
            f"Unknown table key {table_key}. Available: {list(manifest['tables'])}"
        )

    spec = manifest["tables"][table_key]
    filename = spec["filename"]
    shard_rows = []

    for i in range(SHARD_COUNT):
        path = SHARD_ROOT / f"shard_{i}" / filename
        if not path.exists():
            raise SystemExit(f"Missing shard CSV: {path}")
        shard_rows.append(read_csv(path))

    ordered_ids = [row["id"] for row in spec["rows"]]
    merged = merge_rows(shard_rows, ordered_ids)
    final_path = OUT_DIR / filename
    write_csv(final_path, merged)
    print(f"[merged] {table_key}: {final_path}")

    for row in merged:
        combined.append({"table": table_key, **row})

combined_path = OUT_DIR / "combined_tables_1_2_3.csv"
write_csv(combined_path, combined)
print(f"[merged] combined: {combined_path}")
PY

echo "[done] evaluation complete"
echo "[logs]   $LOG_ROOT"
echo "[tables] $OUT_DIR"
