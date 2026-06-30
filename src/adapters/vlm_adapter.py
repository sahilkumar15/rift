# Path: src/adapters/vlm_adapter.py
# Status: NEW
"""Optional VLM explanation adapter: turns a VLM rationale + bbox into a mask.
Stub interface; wire your Qwen2.5-VL / external map here for the VLM leaderboard row."""
from __future__ import annotations
class VLMAdapter:
    def __init__(self, model=None, processor=None, device="cuda"):
        self.model=model; self.processor=processor; self.device=device
    def explain(self, image, question="Is this real or fake? Box the evidence."):
        """Return (rationale_text, bbox_list). NotImplemented until you wire a VLM."""
        raise NotImplementedError("Wire your VLM here (optional leaderboard row).")
    def bbox_to_mask(self, bbox_list, H, W):
        import torch
        m = torch.zeros(1,1,H,W)
        for (x0,y0,x1,y1) in bbox_list:
            m[..., int(y0):int(y1), int(x0):int(x1)] = 1.0
        return m
