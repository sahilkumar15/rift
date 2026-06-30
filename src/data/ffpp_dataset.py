# Path: src/data/ffpp_dataset.py
# Status: NEW
"""
FF++-style CSV dataset. Columns (header required):
  image_path,label,source_id,target_id,manipulation_type,mask_path,metadata_json
Only image_path,label required. Missing source_id/target_id -> sample.has_donor=False,
which downstream forces proxy mode + warning (never silent fake Δ).
"""
from __future__ import annotations
import csv, json, os, warnings
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Sample:
    image_path: str; label: int
    source_id: Optional[str]=None; target_id: Optional[str]=None
    manipulation_type: Optional[str]=None; mask_path: Optional[str]=None
    metadata: Optional[dict]=None
    @property
    def has_donor(self): return bool(self.source_id) and bool(self.target_id)

class FFPPDataset:
    def __init__(self, csv_path, image_loader=None, transform=None,
                 strict_identity_gap=False):
        self.samples: List[Sample]=[]
        self.transform=transform; self.image_loader=image_loader
        self.strict=strict_identity_gap
        self._load(csv_path)
        n_donor=sum(s.has_donor for s in self.samples)
        if n_donor==0:
            warnings.warn(
                f"FFPPDataset: 0/{len(self.samples)} samples have donor metadata -> "
                "identity_gap_mode=PROXY for all. Donor-grounded Δ claims invalid here.",
                RuntimeWarning)
        self.identity_gap_mode = "true" if n_donor==len(self.samples) and n_donor>0 else \
                                 ("mixed" if n_donor>0 else "proxy")
    def _load(self, p):
        with open(p) as f:
            for row in csv.DictReader(f):
                if not row.get("image_path") or row.get("label") in (None,""):
                    continue
                meta=row.get("metadata_json") or ""
                try: meta=json.loads(meta) if meta else None
                except Exception: meta=None
                self.samples.append(Sample(
                    image_path=row["image_path"], label=int(row["label"]),
                    source_id=row.get("source_id") or None,
                    target_id=row.get("target_id") or None,
                    manipulation_type=row.get("manipulation_type") or None,
                    mask_path=row.get("mask_path") or None, metadata=meta))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s=self.samples[i]
        img=None
        if self.image_loader is not None:
            img=self.image_loader(s.image_path)
            if self.transform is not None: img=self.transform(img)
        return {"image": img, "label": s.label, "sample": s}
