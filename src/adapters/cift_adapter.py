# Path: src/adapters/cift_adapter.py
# Status: MODIFIED (5 WIRE points completed against the real CIFT codebase)
"""
cift_adapter.py — THE seam between RIFT and the real CIFT model
(ImageDifussionFake / cldm.diffusionfake.DiffusionFake).

This version is WIRED against the actual uploaded CIFT source, not a guess.
Every WIRE point cites the exact CIFT symbol it relies on:

  load / inference contract  -> cift_eval_complete.load_cift_model + predict_probs_cift
  model class                -> cldm/diffusionfake.py::DiffusionFake (+ GuideNet control_model)
  TRUE delta readout         -> cldm/mamba_modules.py::MambaFakeHead.forward, which sets
                                gap = || normalize(g_s) - normalize(g_t) ||_2
                                and DiffusionFake stashes it at control_model._gap (diffusionfake.py)
  batch / conditioning keys  -> configs/diffusionfake_mixed.yaml:
                                first_stage_key="source" (DONOR face, raw [-1,1] BCHW)
                                target_stage_key="target", control_key="hint" (ANALYZED image),
                                label_key="label", cond_stage_key="txt"

HONESTY NOTES (read before trusting any delta number):
  * The deployed CIFT detector is SOURCE-FREE: its decision logit comes from the
    Global Head + Spatial-Mamba branches on the *target/hint* image alone. delta is
    NOT used by the deployed decision (CIFT paper section C.6). RIFT therefore audits
    two different evidence channels and MUST keep them separate:
        - logit channel : the actual deployed decision (always available)
        - delta channel : CIFT's analysis-time donor-grounded forensic signal,
                          available ONLY when a donor stream is supplied.
  * TRUE delta requires a donor stream. Per CIFT, on test benchmarks the donor is a
    *retrieved same-identity reference* (paper section C.6 / Table 10), NOT necessarily
    a ground-truth swap donor. So donor= may be a retrieved reference frame; that
    still yields TRUE mode here, because the dual-stream gap readout is genuinely
    computed. Absent any donor, we return a single-stream PROXY tagged proxy and
    w_delta is forced to 0 downstream (faithfulness_score.compute_rift_score).
  * The single-stream "||g_t||" proxy used in your earlier IGANER adapter is NOT
    the true delta. Confirmed against mamba_modules.py: true delta needs g_s and g_t.

What I could NOT do in the sandbox (no GPU / no 860M ckpt): actually execute CIFT.
The *interface* is verified against the uploaded code; do ONE smoke run on Katz
(scripts below) to confirm tensor conventions before reporting numbers.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Optional, Any, Dict, Tuple

from .identity_gap_contract import (
    IdentityGapMode, IdentityGapResult, resolve_mode, MechanismValidityError,
)

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:                       # torch absent in this sandbox; present on Katz
    _HAS_TORCH = False


# CIFT config keys, taken verbatim from configs/diffusionfake_mixed.yaml.
# If your eval yaml overrides these, pass overrides via key_overrides.
_CIFT_KEYS = dict(
    first_stage_key="source",   # DONOR face (raw [-1,1] BCHW), also VAE-encoded as source latent
    target_stage_key="target",  # target face
    control_key="hint",         # the ANALYZED image (candidate forgery we classify/mask)
    label_key="label",
)


def _no_grad(f):
    return f


class CIFTAdapter:
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        backbone: str = "convnextv2_base",
        strict_identity_gap: bool = False,
        cift_root: Optional[str] = None,
        config_path: Optional[str] = None,
        key_overrides: Optional[Dict[str, str]] = None,
    ):
        self.ckpt_path = ckpt_path
        self.device = device
        self.backbone = backbone
        self.strict_identity_gap = strict_identity_gap
        self.cift_root = cift_root
        # default points at the mixed-training config the model was trained with
        self.config_path = config_path or "configs/diffusionfake_mixed.yaml"
        self.keys = dict(_CIFT_KEYS)
        if key_overrides:
            self.keys.update(key_overrides)
        self.model = None
        self._proxy_warned = False
        self._spatial_feat = None         # filled by a forward hook (WIRE 3)
        self._hook_handle = None

    # ------------------------------------------------------------------ load
    def load_detector(self) -> "CIFTAdapter":
        """
        # === WIRE 1 ===  (verified against cift_eval_complete.load_cift_model)
        Build CIFT, load the checkpoint, eval+freeze. Mirrors the exact, working
        load path in the CIFT repo so we inherit its checkpoint-healing behaviour.
        """
        if not _HAS_TORCH:
            raise RuntimeError("WIRE 1 needs torch (run on Katz).")
        if self.cift_root is None:
            raise ValueError("cift_root must point at the ImageDifussionFake repo root "
                             "(so cldm, share, datasets import), exactly as run_iganer.sh sets it.")
        if self.cift_root not in sys.path:
            sys.path.insert(0, self.cift_root)

        from cldm.model import create_model   # CIFT loader

        cfg = self.config_path
        if not os.path.isabs(cfg):
            cfg = os.path.join(self.cift_root, cfg)

        model = create_model(cfg)                                   # instantiate DiffusionFake
        bb = (self.backbone or "convnextv2_base").split(".")[0]
        model.control_model.define_feature_filter(bb)               # build encoder for backbone

        raw = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)
        state.pop("cond_stage_model.transformer.text_model.embeddings.position_ids", None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(unexpected) > 50:
            warnings.warn(f"CIFT load: {len(missing)} missing / {len(unexpected)} unexpected keys "
                          f"- check backbone='{self.backbone}' matches the checkpoint.", RuntimeWarning)

        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        # WIRE 3 hook: capture the (B, 1792, h, w) spatial map that feeds SMC.
        # encoder_proj output is exactly the feature SMC/global-head consume (GuideNet.forward).
        cm = self.model.control_model
        target_module = getattr(cm, "encoder_proj", None)
        if target_module is not None and not isinstance(target_module, torch.nn.Identity):
            self._hook_handle = target_module.register_forward_hook(
                lambda m, i, o: setattr(self, "_spatial_feat", o.detach())
            )
        return self

    # -------------------------------------------------------- batch plumbing
    def _build_batch(
        self,
        x: "torch.Tensor",
        donor: "Optional[torch.Tensor]" = None,
        forgery_type: str = "swap",
    ) -> Dict[str, Any]:
        """
        Assemble the CIFT batch dict. x is the ANALYZED image (candidate forgery),
        BCHW float in [-1, 1]. It is placed on control_key (hint) AND target.
        donor, if given, is the source/donor (or retrieved same-id reference) and
        is placed on first_stage_key (source) -> activates the dual-identity gap.
        """
        x = x.to(self.device).float()
        B = x.shape[0]
        # Genuine-frame convention in CIFT: source == target when no donor (twin mode).
        src = donor.to(self.device).float() if donor is not None else x
        batch = {
            self.keys["control_key"]:      x,          # hint = analyzed image
            "hint_ori":                    x,
            self.keys["target_stage_key"]: x,          # target stream = analyzed image
            self.keys["first_stage_key"]:  src,        # source/donor stream
            self.keys["label_key"]:        torch.ones(B, device=self.device),
            "forgery_type":                forgery_type,
            "txt":                         [""] * B,    # cond_stage_key
        }
        return batch

    def _forward(self, x, donor=None, forgery_type="swap") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Run one CIFT forward. Returns (logits[B], gap[B]).
        gap is control_model._gap (TRUE delta when donor present, else zeros)."""
        batch = self._build_batch(x, donor=donor, forgery_type=forgery_type)
        with torch.no_grad():
            source, target, c, _ = self.model.get_input(batch, self.model.first_stage_key)
            out = self.model(source, target, c, batch[self.keys["label_key"]])
        loss_dict = out[1] if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict) else {}
        if "v/logits" in loss_dict:
            logits = loss_dict["v/logits"].detach().float().view(-1)
        elif "v/probs" in loss_dict:
            p = loss_dict["v/probs"].detach().float().clamp(1e-6, 1 - 1e-6).view(-1)
            logits = torch.log(p / (1 - p))
        else:
            raise RuntimeError(f"No logits/probs in CIFT output. Keys: {list(loss_dict.keys())}")
        gap = getattr(self.model.control_model, "_gap", None)
        if gap is None:
            gap = torch.zeros_like(logits)
        gap = gap.detach().float().view(-1)
        return logits, gap

    # -------------------------------------------------------------- forward
    def predict_logits(self, x: "Any") -> "Any":
        """# === WIRE 2 === raw deployed detection logit(s) for image batch x (B,3,H,W)."""
        if not _HAS_TORCH:
            raise RuntimeError("WIRE 2 needs torch (Katz).")
        logits, _ = self._forward(x, donor=None)
        return logits

    def extract_features(self, x: "Any") -> "Any":
        """# === WIRE 3 === (B,1792,h,w) spatial map used by SMC; for Grad-CAM + policy state.
        Captured via a forward hook on control_model.encoder_proj (set in load_detector)."""
        if not _HAS_TORCH:
            raise RuntimeError("WIRE 3 needs torch (Katz).")
        self._spatial_feat = None
        with torch.no_grad():
            self._forward(x, donor=None)
        if self._spatial_feat is None:
            raise RuntimeError("No spatial feature captured - encoder_proj is Identity for this "
                               "backbone; hook a different layer or use global_pool input.")
        return self._spatial_feat

    # ------------------------------------------------------- identity gap delta
    def identity_gap(
        self,
        x: "Any",
        donor: "Optional[Any]" = None,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> IdentityGapResult:
        """
        delta with an honest mode tag.

        TRUE  : a donor stream is supplied (donor image, OR a retrieved same-id ref).
                # === WIRE 4 ===  delta = || normalize(g_s) - normalize(g_t) ||_2 read
                from CIFT's own gap readout (control_model._gap), NOT a single-stream norm.
        PROXY : no donor -> single-stream feature-norm stand-in, tagged proxy.
        ERROR : no donor AND strict_identity_gap=True -> raise.
        """
        has_donor = donor is not None or (source_id is not None and target_id is not None)
        mode = resolve_mode(has_donor, self.strict_identity_gap)

        if mode == IdentityGapMode.ERROR:
            raise MechanismValidityError(
                "strict_identity_gap=True but no donor stream for this sample. Provide a donor "
                "image (or retrieved same-identity reference), or set strict_identity_gap=False."
            )

        if mode == IdentityGapMode.TRUE:
            if not _HAS_TORCH:
                raise RuntimeError("WIRE 4 needs torch (Katz).")
            if donor is None:
                raise MechanismValidityError(
                    "TRUE mode requested via source_id/target_id but no donor tensor was passed. "
                    "Resolve the retrieved same-identity reference to a (B,3,H,W) tensor and pass donor=."
                )
            _, gap = self._forward(x, donor=donor)
            val = float(gap.mean().item())
            # CIFT genuine delta~0, forged delta>1 (Table 10). A flat-zero gap here means the
            # dual path did not activate (e.g. use_dimf off) -> downgrade to proxy honestly.
            if abs(val) < 1e-6:
                warnings.warn("identity_gap: dual path returned ~0 delta despite a donor - DIMF likely "
                              "inactive; downgrading to PROXY.", RuntimeWarning)
                return self._proxy_result(x)
            return IdentityGapResult(
                value=val, mode=IdentityGapMode.TRUE, has_donor_metadata=True,
                detail="delta=||norm(g_s)-norm(g_t)||_2 from control_model._gap (dual-identity forward)",
            )

        return self._proxy_result(x)

    def identity_gap_map(self, x: "Any", **kw) -> "Optional[Any]":
        """No native spatial delta heatmap is exposed by CIFT -> return None so the
        cift_gap_explainer falls back to explain_identity_gap()."""
        return None

    def explain_identity_gap(self, x: "Any") -> "Any":
        """Gradient-based delta attribution: |d delta / d x| pooled over channels.
        Requires a donor for a TRUE-delta gradient; without one this attributes the
        proxy and the resulting map must NOT be reported as a delta-mechanism map."""
        if not _HAS_TORCH:
            raise RuntimeError("explain_identity_gap needs torch (Katz).")
        x = x.to(self.device).float().requires_grad_(True)
        feat = self.extract_features(x)
        delta = feat.flatten(2).mean(-1).norm(dim=-1).sum()
        grad, = torch.autograd.grad(delta, x, retain_graph=False, create_graph=False)
        return grad.abs().mean(dim=1, keepdim=True).detach()   # (B,1,H,W)

    # ---------------------------------------------------------------- proxy
    def _proxy_result(self, x) -> IdentityGapResult:
        value = self._proxy_gap(x)
        return IdentityGapResult(
            value=float(value), mode=IdentityGapMode.PROXY, has_donor_metadata=False,
            detail="single-stream feature-norm proxy; donor stream absent",
        )

    def _proxy_gap(self, x: "Any") -> float:
        # === WIRE 5 === single-stream stand-in so the pipeline runs on donor-free data.
        if not self._proxy_warned:
            warnings.warn("CIFTAdapter using PROXY identity-gap (no donor stream).", RuntimeWarning)
            self._proxy_warned = True
        if not _HAS_TORCH:
            return 0.0
        with torch.no_grad():
            feat = self.extract_features(x) if self.model is not None else x
            if feat.dim() > 2:
                feat = feat.flatten(2).mean(-1)
            return feat.norm(dim=-1).mean().item()

# =============================================================================
# RIFT PATCH: gradient-enabled CIFT calls for audit explainers
# =============================================================================
# The original adapter intentionally runs CIFT under torch.no_grad() for normal
# detection and identity-gap scoring. That is correct for evaluation, but Grad-CAM
# and CIFT-gap attribution need gradients w.r.t. the input image.
#
# These monkey-patched methods keep normal predict_logits()/identity_gap() intact
# and add gradient-safe paths used only by explainers.
# =============================================================================

def _rift_forward_grad(self, x, donor=None, forgery_type="swap"):
    """
    Gradient-enabled CIFT forward.

    Returns:
      logits: tensor with grad if CIFT graph exposes grad to input
      gap:    tensor with grad if dual identity branch exposes grad to input

    This is used by GradCAMExplainer and CIFTGapExplainer only.
    """
    if not _HAS_TORCH:
        raise RuntimeError("gradient CIFT forward needs torch.")

    if self.model is None:
        raise RuntimeError("CIFT model not loaded. Call load_detector() first.")

    x = x.to(self.device).float()

    if not x.requires_grad:
        x.requires_grad_(True)

    if donor is not None:
        donor = donor.to(self.device).float()

    batch = self._build_batch(x, donor=donor, forgery_type=forgery_type)

    with torch.enable_grad():
        source, target, c, _ = self.model.get_input(batch, self.model.first_stage_key)
        out = self.model(source, target, c, batch[self.keys["label_key"]])

    loss_dict = (
        out[1]
        if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict)
        else {}
    )

    if "v/logits" in loss_dict:
        logits = loss_dict["v/logits"].float().view(-1)
    elif "v/probs" in loss_dict:
        p = loss_dict["v/probs"].float().clamp(1e-6, 1 - 1e-6).view(-1)
        logits = torch.log(p / (1 - p))
    else:
        raise RuntimeError(f"No logits/probs in CIFT output. Keys: {list(loss_dict.keys())}")

    gap = getattr(self.model.control_model, "_gap", None)

    if gap is None:
        gap = torch.zeros_like(logits)

    gap = gap.float().view(-1)

    return logits, gap


def _rift_predict_logits_for_grad(self, x):
    """Gradient-enabled deployed logit for logit-saliency/Grad-CAM fallback."""
    logits, _ = self._forward_grad(x, donor=None)
    return logits


def _rift_explain_identity_gap(self, x, donor=None, source_id=None, target_id=None):
    """
    Gradient attribution for CIFT identity-gap.

    Priority:
      1. donor-grounded true gap gradient if donor is provided
      2. deployed logit input-gradient fallback
      3. deterministic image-energy fallback, so audit never crashes
    """
    if not _HAS_TORCH:
        raise RuntimeError("explain_identity_gap needs torch.")

    x = x.clone().detach().to(self.device).float().requires_grad_(True)

    if donor is not None:
        donor = donor.to(self.device).float()

    # 1) True donor-grounded identity-gap gradient.
    if donor is not None:
        try:
            _, gap = self._forward_grad(x, donor=donor)
            score = gap.sum()

            if getattr(score, "requires_grad", False):
                grad = torch.autograd.grad(
                    score,
                    x,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]

                if grad is not None:
                    return grad.abs().mean(dim=1, keepdim=True).detach()
        except Exception as e:
            warnings.warn(
                f"CIFT true-gap gradient failed; falling back to logit gradient. Error: {e}",
                RuntimeWarning,
            )

    # 2) Deployed logit gradient fallback.
    try:
        logits = self._forward_grad(x, donor=None)[0]
        score = logits.sum()

        if getattr(score, "requires_grad", False):
            grad = torch.autograd.grad(
                score,
                x,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is not None:
                return grad.abs().mean(dim=1, keepdim=True).detach()
    except Exception as e:
        warnings.warn(
            f"CIFT logit gradient failed; falling back to image-energy saliency. Error: {e}",
            RuntimeWarning,
        )

    # 3) Last-resort deterministic fallback, normalized image energy.
    sal = x.detach().abs().mean(dim=1, keepdim=True)
    sal = sal - sal.amin(dim=(2, 3), keepdim=True)
    sal = sal / (sal.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return sal


# Attach patched methods to CIFTAdapter.
CIFTAdapter._forward_grad = _rift_forward_grad
CIFTAdapter.predict_logits_for_grad = _rift_predict_logits_for_grad
CIFTAdapter.explain_identity_gap = _rift_explain_identity_gap

