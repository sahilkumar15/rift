# Path: src/data/ffpp_dataset.py
# Status: MODIFIED
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

  When strict_identity_gap=True, this dataset fails early if any row lacks donor_path.
"""
from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from typing import Optional, List


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

    @property
    def has_donor(self) -> bool:
        return bool(self.donor_path)


class FFPPDataset:
    def __init__(self, csv_path, image_loader=None, transform=None, strict_identity_gap=False):
        self.samples: List[Sample] = []
        self.transform = transform
        self.image_loader = image_loader
        self.strict = bool(strict_identity_gap)
        self.csv_path = csv_path

        self._load(csv_path)

        n_donor = sum(s.has_donor for s in self.samples)

        if self.strict and n_donor < len(self.samples):
            missing = len(self.samples) - n_donor
            raise ValueError(
                f"strict_identity_gap=True but {missing}/{len(self.samples)} rows in "
                f"{csv_path} do not have donor_path/source_ref_path. "
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
            "true" if n_donor == len(self.samples) and n_donor > 0
            else ("mixed" if n_donor > 0 else "proxy")
        )

    def _load(self, p):
        with open(p) as f:
            for row in csv.DictReader(f):
                if not row.get("image_path") or row.get("label") in (None, ""):
                    continue

                meta = row.get("metadata_json") or ""
                try:
                    meta = json.loads(meta) if meta else None
                except Exception:
                    meta = None

                self.samples.append(
                    Sample(
                        image_path=row["image_path"],
                        label=int(row["label"]),
                        source_id=row.get("source_id") or None,
                        target_id=row.get("target_id") or None,
                        manipulation_type=row.get("manipulation_type") or None,
                        mask_path=row.get("mask_path") or None,
                        donor_path=(
                            row.get("donor_path")
                            or row.get("source_ref_path")
                            or row.get("donor")
                            or None
                        ),
                        metadata=meta,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str, *, kind: str):
        if self.image_loader is None:
            return None

        try:
            img = self.image_loader(path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not load {kind} image '{path}' from CSV '{self.csv_path}'. "
                "Use absolute paths or paths relative to the current run directory."
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
                f"Row {i} in {self.csv_path} has no donor_path/source_ref_path while "
                "strict_identity_gap=True."
            )

        return {
            "image": img,
            "donor": donor,
            "label": s.label,
            "sample": s,
        }