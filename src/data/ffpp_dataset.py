"""
FF++/RIFT CSV dataset.

Required columns:
  image_path,label

Optional but important columns:
  donor_path or source_ref_path     -> actual donor/reference image used for true CIFT Δ
  source_id,target_id               -> metadata only; IDs are not enough to compute Δ
  manipulation_type,mask_path,metadata_json

Important:
  CIFT's donor-grounded identity gap Δ requires a donor/reference tensor.
  Merely having source_id/target_id strings is not enough.

Path handling:
  Many example/debug CSVs contain placeholder absolute paths such as /data/ffpp/...
  On Katz your real dataset usually lives under /scratch/....
  This dataset can rewrite a prefix at load time, e.g.
  /data/ffpp -> /scratch/sahil/projects/.../datasets/ffpp.
"""
from __future__ import annotations

import csv
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Sample:
    image_path: str
    label: int
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    manipulation_type: Optional[str] = None
    mask_path: Optional[str] = None
    donor_path: Optional[str] = None
    metadata: Optional[dict] = None
    row_index: int = -1

    @property
    def has_donor(self) -> bool:
        return bool(self.donor_path)


class FFPPDataset:
    def __init__(
        self,
        csv_path,
        image_loader=None,
        transform=None,
        strict_identity_gap=False,
        data_root: Optional[str] = None,
        path_prefix_from: Optional[str] = None,
        path_prefix_to: Optional[str] = None,
        path_rewrites: Optional[Dict[str, str]] = None,
        check_files: bool = True,
        check_limit: int = 0,
    ):
        self.samples: List[Sample] = []
        self.transform = transform
        self.image_loader = image_loader
        self.strict = bool(strict_identity_gap)
        self.csv_path = str(csv_path)
        self.data_root = self._norm_root(data_root)
        self.check_files = bool(check_files)
        self.check_limit = int(check_limit or 0)
        self.path_rewrites = self._build_rewrites(
            path_prefix_from,
            path_prefix_to,
            path_rewrites,
        )

        self._load(self.csv_path)

        n_donor = sum(s.has_donor for s in self.samples)

        if self.strict and n_donor < len(self.samples):
            missing = len(self.samples) - n_donor
            raise ValueError(
                f"strict_identity_gap=True but {missing}/{len(self.samples)} rows in "
                f"{self.csv_path} do not have donor_path/source_ref_path. "
                "Add a real donor reference image column, or run with "
                "detector.strict_identity_gap=false for proxy/logit-only debugging."
            )

        if n_donor == 0:
            warnings.warn(
                f"FFPPDataset: 0/{len(self.samples)} samples have donor_path/source_ref_path -> "
                "identity_gap_mode=PROXY for all. Donor-grounded Δ claims are invalid here.",
                RuntimeWarning,
            )

        self.identity_gap_mode = (
            "true"
            if n_donor == len(self.samples) and n_donor > 0
            else ("mixed" if n_donor > 0 else "proxy")
        )

        if self.check_files and self.image_loader is not None:
            self.validate_paths(max_items=self.check_limit)

    @staticmethod
    def _norm_root(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        return os.path.abspath(os.path.expandvars(os.path.expanduser(str(p))))

    def _build_rewrites(
        self,
        path_prefix_from: Optional[str],
        path_prefix_to: Optional[str],
        path_rewrites: Optional[Dict[str, str]],
    ) -> List[Tuple[str, str]]:
        rewrites: List[Tuple[str, str]] = []

        if path_rewrites:
            for src, dst in path_rewrites.items():
                if src and dst:
                    rewrites.append(
                        (
                            str(src).rstrip("/"),
                            self._norm_root(dst).rstrip("/"),
                        )
                    )

        if path_prefix_from and path_prefix_to:
            rewrites.append(
                (
                    str(path_prefix_from).rstrip("/"),
                    self._norm_root(path_prefix_to).rstrip("/"),
                )
            )

        if self.data_root:
            rewrites.append(("/data/ffpp", self.data_root.rstrip("/")))

        seen = set()
        unique = []

        for src, dst in rewrites:
            key = (src, dst)
            if key not in seen:
                seen.add(key)
                unique.append(key)

        return unique

    def _resolve_path(self, raw_path: Optional[str]) -> Optional[str]:
        if raw_path is None:
            return None

        p = os.path.expandvars(os.path.expanduser(str(raw_path).strip()))

        if not p:
            return None

        if os.path.isabs(p) and os.path.exists(p):
            return p

        if os.path.isabs(p):
            for src, dst in self.path_rewrites:
                if p == src or p.startswith(src + "/"):
                    candidate = dst + p[len(src):]
                    return candidate

            return p

        if self.data_root:
            return os.path.join(self.data_root, p)

        return os.path.abspath(os.path.join(os.path.dirname(self.csv_path), p))

    def _candidate_paths_for_message(self, raw_path: str) -> List[str]:
        candidates = []
        p = os.path.expandvars(os.path.expanduser(str(raw_path).strip()))
        candidates.append(p)

        if os.path.isabs(p):
            for src, dst in self.path_rewrites:
                if p == src or p.startswith(src + "/"):
                    candidates.append(dst + p[len(src):])
        else:
            if self.data_root:
                candidates.append(os.path.join(self.data_root, p))

            candidates.append(
                os.path.abspath(os.path.join(os.path.dirname(self.csv_path), p))
            )

        out = []

        for c in candidates:
            if c and c not in out:
                out.append(c)

        return out

    def _load(self, p):
        with open(p, newline="") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {p}")

            for row_index, row in enumerate(reader, start=2):
                if not row.get("image_path") or row.get("label") in (None, ""):
                    continue

                meta = row.get("metadata_json") or ""

                try:
                    meta = json.loads(meta) if meta else None
                except Exception:
                    meta = None

                image_path = self._resolve_path(row["image_path"])

                donor_raw = (
                    row.get("donor_path")
                    or row.get("source_ref_path")
                    or row.get("donor")
                    or None
                )

                mask_raw = row.get("mask_path") or None

                self.samples.append(
                    Sample(
                        image_path=image_path,
                        label=int(row["label"]),
                        source_id=row.get("source_id") or None,
                        target_id=row.get("target_id") or None,
                        manipulation_type=row.get("manipulation_type") or None,
                        mask_path=self._resolve_path(mask_raw) if mask_raw else None,
                        donor_path=self._resolve_path(donor_raw) if donor_raw else None,
                        metadata=meta,
                        row_index=row_index,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def validate_paths(self, max_items: int = 0) -> None:
        missing = []

        n = (
            len(self.samples)
            if not max_items or max_items < 0
            else min(len(self.samples), max_items)
        )

        for s in self.samples[:n]:
            required = [("image_path", s.image_path)]

            if self.strict or s.donor_path:
                required.append(("donor_path", s.donor_path))

            for col, path in required:
                if not path or not os.path.exists(path):
                    missing.append((s.row_index, col, path))

                    if len(missing) >= 8:
                        break

            if len(missing) >= 8:
                break

        if missing:
            msg = [
                f"CSV file paths are missing in {self.csv_path}.",
                "This is a dataset-path problem, not a CIFT/RL error.",
                "First missing paths:",
            ]

            for row_idx, col, path in missing:
                msg.append(f"  row={row_idx} col={col} path={path}")

            msg.extend(
                [
                    "",
                    "Fix options:",
                    "  1) Pass the real FF++ root:",
                    "     bash scripts/run_rift.sh --mode train data.data_root=/scratch/.../datasets/ffpp",
                    "",
                    "  2) Or explicitly rewrite the CSV prefix:",
                    "     bash scripts/run_rift.sh --mode train data.path_prefix_from=/data/ffpp data.path_prefix_to=/scratch/.../datasets/ffpp",
                    "",
                    "  3) Or edit the CSV so image_path and donor_path are real absolute paths.",
                ]
            )

            raise FileNotFoundError("\n".join(msg))

    def _load_image(self, path: str, *, kind: str):
        if self.image_loader is None:
            return None

        try:
            img = self.image_loader(path)
        except FileNotFoundError as e:
            candidates = self._candidate_paths_for_message(path)
            hint = "\n".join(f"    - {c}" for c in candidates)

            raise FileNotFoundError(
                f"Could not load {kind} image from CSV '{self.csv_path}'.\n"
                f"Resolved path: {path}\n"
                f"Tried/candidate paths:\n{hint}\n"
                "Use data.data_root, data.path_prefix_from/data.path_prefix_to, or fix the CSV."
            ) from e

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, i):
        s = self.samples[i]

        img = self._load_image(s.image_path, kind="target/analyzed")

        donor = None

        if s.donor_path:
            donor = self._load_image(s.donor_path, kind="donor/reference")
        elif self.strict:
            raise ValueError(
                f"Row {s.row_index} in {self.csv_path} has no donor_path/source_ref_path while "
                "strict_identity_gap=True."
            )

        return {
            "image": img,
            "donor": donor,
            "label": s.label,
            "sample": s,
        }
