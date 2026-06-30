# Path: iganer/rift/utils/io.py
# Status: NEW
"""CSV/JSON helpers used by audit + correlation runners."""
import csv, json, os
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=str)
    return path
def save_csv(rows, path):
    if not rows: 
        open(path, "w").close(); return path
    ensure_dir(os.path.dirname(path) or ".")
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    return path
def append_csv_row(row, path):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row))
        if not exists: w.writeheader()
        w.writerow(row)
    return path
