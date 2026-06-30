# Path: iganer/rift/adapters/cift_adapter.py
# Status: NEW
"""
cift_adapter.py — THE seam between RIFT and YOUR CIFT model on Katz.

This is the only file whose body you must complete with your real codebase. I
cannot write its internals correctly without your CIFT class/checkpoint, so
instead of guessing (which would compile but return noise) every method that
needs your model raises NotImplementedError with the EXACT thing to wire in.

The honesty machinery around it is already done: identity_gap() returns an
IdentityGapResult tagged true/proxy, and resolve_mode() enforces the strict/
proxy/error policy from the contract. Fill the 5 marked spots; everything else
in RIFT consumes this interface and never touches your model directly.

Search for  # === WIRE ===  to find every spot you must complete.
"""
from __future__ import annotations

from typing import Optional, Any, Dict
import warnings

from .identity_gap_contract import (
    IdentityGapMode, IdentityGapResult, resolve_mode, MechanismValidityError,
)

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                       # torch absent in this sandbox; present on Katz
    _HAS_TORCH = False


class CIFTAdapter:
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        backbone: str = "convnextv2_base",
        strict_identity_gap: bool = False,
        cift_root: Optional[str] = None,
    ):
        self.ckpt_path = ckpt_path
        self.device = device
        self.backbone = backbone
        self.strict_identity_gap = strict_identity_gap
        self.cift_root = cift_root
        self.model = None
        self._proxy_warned = False

    # ------------------------------------------------------------------ load
    def load_detector(self) -> "CIFTAdapter":
        """
        # === WIRE 1 ===
        Construct your CIFT model and load self.ckpt_path. In your repo this is
        the ConvNeXt-V2-B + Global Head + SMC source-free inference graph (the
        donor/XID-Mamba branches are training-only and not needed for inference).
        Put `cift_root` on sys.path so `share`, `cldm`, `datasets` import, exactly
        as your run_iganer.sh does. Set self.model to an eval()'d, frozen module.
        """
        raise NotImplementedError(
            "WIRE 1: build + load your CIFT checkpoint here. See cift_adapter docstring."
        )

    # -------------------------------------------------------------- forward
    def predict_logits(self, x: "Any") -> "Any":
        """# === WIRE 2 === return raw detection logit(s) for image batch x (B,3,H,W)."""
        raise NotImplementedError("WIRE 2: return CIFT detection logits for x.")

    def extract_features(self, x: "Any") -> "Any":
        """# === WIRE 3 === return the spatial feature/token map used by SMC (for Grad-CAM
        and for the policy state). Shape e.g. (B, C, h, w)."""
        raise NotImplementedError("WIRE 3: return CIFT spatial features for x.")

    # ------------------------------------------------------- identity gap Δ
    def identity_gap(
        self,
        x: "Any",
        donor: "Optional[Any]" = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> IdentityGapResult:
        """
        Return Δ with an honest mode tag.

        TRUE mode requires a real donor stream (donor image or donor embedding via
        source_id/target_id). Then:
          # === WIRE 4 ===  Δ = || g_s - g_t ||_2  using your XID-Mamba/gap readout.
          IMPORTANT: your IGANER notes flag cift_adapter.identity_gap() as possibly
          returning a NORM PROXY marked `=== CONFIRM ===`. If your model exposes a
          dedicated gap_readout(feat) or returns delta, call THAT here. A norm of a
          single-stream feature is NOT donor-grounded Δ and must be tagged proxy.

        PROXY mode (no donor): a feature/embedding distance stand-in — tagged proxy,
        never reported as the mechanism.
        """
        has_donor = donor is not None or (source_id is not None and target_id is not None)
        mode = resolve_mode(has_donor, self.strict_identity_gap)

        if mode == IdentityGapMode.ERROR:
            raise MechanismValidityError(
                "strict_identity_gap=True but no donor metadata for this sample. "
                "Provide donor/source_id/target_id or set strict_identity_gap=False (proxy)."
            )

        if mode == IdentityGapMode.TRUE:
            # === WIRE 4 ===
            raise NotImplementedError(
                "WIRE 4: compute true donor-grounded Δ=||g_s-g_t||_2 via your gap readout. "
                "Confirm it is the gap readout, NOT a single-stream norm proxy."
            )

        # PROXY: weak fallback so the pipeline runs end-to-end on zero-shot data.
        # === WIRE 5 (optional) === replace with your preferred proxy if you have one.
        value = self._proxy_gap(x)
        return IdentityGapResult(
            value=float(value), mode=IdentityGapMode.PROXY, has_donor_metadata=False,
            detail="feature-norm proxy; donor metadata absent",
        )

    def identity_gap_map(self, x: "Any", **kw) -> "Optional[Any]":
        """Optional spatial Δ heatmap if your model exposes one; else None and RIFT
        falls back to explain_identity_gap()."""
        return None

    def explain_identity_gap(self, x: "Any") -> "Any":
        """Fallback Δ-attribution when no native map exists: gradient of Δ proxy w.r.t.
        input, |∂Δ/∂x| pooled over channels. Implement on Katz where torch is live."""
        raise NotImplementedError(
            "Optional: gradient-based Δ attribution map. Only needed for cift_gap_explainer."
        )

    # ---------------------------------------------------------------- proxy
    def _proxy_gap(self, x: "Any") -> float:
        if not self._proxy_warned:
            warnings.warn("CIFTAdapter using PROXY identity-gap (no donor).", RuntimeWarning)
            self._proxy_warned = True
        if not _HAS_TORCH:
            return 0.0
        with torch.no_grad():
            feat = self.extract_features(x) if self.model is not None else x
            # L2 norm of mean-pooled feature as a stand-in scalar
            if feat.dim() > 2:
                feat = feat.flatten(2).mean(-1)
            return feat.norm(dim=-1).mean().item()
