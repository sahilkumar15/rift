# Path: src/utils/io.py
"""CSV/JSON helpers used by audit + correlation runners."""

import csv
import json
import os


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _format_cell(v, ndigits=4):
    if v is None:
        return ""

    if isinstance(v, bool):
        return v

    if isinstance(v, float):
        if v != v:  # NaN
            return ""
        return f"{v:.{ndigits}f}"

    return v


def _format_row(row, ndigits=4):
    return {k: _format_cell(v, ndigits=ndigits) for k, v in row.items()}


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

    return path


def save_csv(rows, path):
    ensure_dir(os.path.dirname(path) or ".")

    if not rows:
        open(path, "w").close()
        return path

    keys = sorted({k for r in rows for k in r})

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()

        for r in rows:
            w.writerow(_format_row(r, ndigits=4))

    return path


def append_csv_row(row, path):
    exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row))

        if not exists:
            w.writeheader()

        w.writerow(_format_row(row, ndigits=4))

    return path
