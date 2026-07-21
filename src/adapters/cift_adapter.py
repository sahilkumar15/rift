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
        self._cached_feature_key = None
        self._cached_feature = None

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

    def _feature_cache_key(self, x):
        if not _HAS_TORCH or not torch.is_tensor(x):
            return None
        try:
            return (int(x.data_ptr()), tuple(x.shape), str(x.device), str(x.dtype))
        except Exception:
            return None

    def _crop_spatial_feature(self, feat, target_b: int):
        if not torch.is_tensor(feat):
            raise RuntimeError(f"Captured CIFT feature is not a tensor: {type(feat)}")

        if feat.shape[0] == target_b:
            return feat.detach()

        if feat.shape[0] > target_b:
            return feat[-target_b:].detach().contiguous()

        if feat.shape[0] == 1 and target_b > 1:
            return feat.repeat(target_b, 1, 1, 1).detach().contiguous()

        raise RuntimeError(
            f"CIFT feature batch mismatch: feature batch={feat.shape[0]}, "
            f"image batch={target_b}, feature shape={tuple(feat.shape)}"
        )

    def _cache_current_spatial_feature(self, x):
        if not _HAS_TORCH or not torch.is_tensor(x):
            return
        feat = getattr(self, "_spatial_feat", None)
        if feat is None or not torch.is_tensor(feat):
            return
        key = self._feature_cache_key(x)
        if key is None:
            return
        try:
            self._cached_feature_key = key
            self._cached_feature = self._crop_spatial_feature(feat, int(x.shape[0]))
        except Exception:
            self._cached_feature_key = None
            self._cached_feature = None

    def predict_logits(self, x: "Any") -> "Any":
        """Raw deployed detection logit(s) for image batch x.

        Also caches the spatial feature captured by the forward hook so
        extract_features(x) can avoid a second identical CIFT forward.
        """
        if not _HAS_TORCH:
            raise RuntimeError("WIRE 2 needs torch (Katz).")

        logits, _ = self._forward(x, donor=None)
        self._cache_current_spatial_feature(x)
        return logits

    def extract_features(self, x: "Any") -> "Any":
        """Spatial map used by RIFT policy state.

        Fast path:
          predict_logits(x) already ran CIFT and filled encoder_proj hook.
          If the same tensor is requested, return the cached feature.
        """
        if not _HAS_TORCH:
            raise RuntimeError("WIRE 3 needs torch (Katz).")

        x = x.to(self.device).float()
        key = self._feature_cache_key(x)

        if (
            key is not None
            and key == getattr(self, "_cached_feature_key", None)
            and torch.is_tensor(getattr(self, "_cached_feature", None))
        ):
            return self._cached_feature

        self._spatial_feat = None

        with torch.no_grad():
            self._forward(x, donor=None)

        if self._spatial_feat is None:
            raise RuntimeError(
                "No spatial feature captured - encoder_proj hook did not fire."
            )

        self._cache_current_spatial_feature(x)

        if (
            key is not None
            and key == getattr(self, "_cached_feature_key", None)
            and torch.is_tensor(getattr(self, "_cached_feature", None))
        ):
            return self._cached_feature

        return self._crop_spatial_feature(self._spatial_feat, int(x.shape[0]))

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



# =============================================================================
# RIFT FIX: torch.enable_grad() CANNOT override an @torch.no_grad() DECORATOR
# =============================================================================
# _rift_forward_grad() wrapped its CIFT call in `with torch.enable_grad():` and
# assumed that was enough. It is not.
#
# LDM decorates the input path with @torch.no_grad():
#     DiffusionFakeMixed.get_input        (cldm/diffusionfake.py)
#       -> LatentDiffusion.get_input      (ldm/models/diffusion/ddpm.py)
#         -> encode_first_stage           (also decorated)
#
# A decorator enters no_grad WHEN THE FUNCTION IS CALLED, i.e. INSIDE our
# enable_grad() block. Contexts nest and the innermost wins, so no_grad wins.
# Result: every gradient-based explainer (GradCAM, CIFT-gap) silently received a
# graph-less tensor, and torch.autograd.grad() then raised. Rows 1-4 of Table 1
# were all dead for this one reason.
#
# torch's decorate_context uses functools.wraps, so the ORIGINAL undecorated
# function survives at __wrapped__. Swap it in for the duration of the gradient
# call and restore afterwards.
#
# This mutates CLASS attributes, so restoration in `finally` is mandatory: a
# leaked unwrap would silently disable no_grad for ordinary evaluation and blow
# up memory on every subsequent forward.
# =============================================================================
import contextlib as _rift_contextlib


@_rift_contextlib.contextmanager
def _rift_enable_cift_grad(model):
    """Temporarily strip @torch.no_grad() from LDM's input/encode path."""
    targets = ("get_input", "encode_first_stage")
    patched = []
    try:
        for cls in type(model).__mro__:
            for name in targets:
                fn = cls.__dict__.get(name)
                if fn is not None and hasattr(fn, "__wrapped__"):
                    patched.append((cls, name, fn))
                    setattr(cls, name, fn.__wrapped__)
        if not patched:
            warnings.warn(
                "RIFT: found no @torch.no_grad()-decorated get_input/"
                "encode_first_stage to unwrap. Either the CIFT version changed "
                "or gradients already flow. If gradient explainers still fail, "
                "this is why.",
                RuntimeWarning,
            )
        yield len(patched)
    finally:
        for cls, name, fn in patched:
            setattr(cls, name, fn)


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

    # _rift_enable_cift_grad() is REQUIRED here: enable_grad() alone is a no-op
    # against LDM's @torch.no_grad()-decorated get_input(). See the comment block
    # above _rift_enable_cift_grad for the full explanation.
    with torch.enable_grad(), _rift_enable_cift_grad(self.model):
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
        # Substituting zeros here is what let explain_identity_gap fall through to
        # the image-energy fallback with no warning: zeros.sum() has
        # requires_grad=False, so the "is the gap differentiable?" check failed
        # silently. When a donor WAS supplied, a missing _gap is a real error.
        if donor is not None:
            raise RuntimeError(
                "CIFT control_model._gap is absent after a donor-grounded forward. "
                "The dual identity branch did not run or did not expose _gap, so no "
                "TRUE Delta is available for this batch."
            )
        gap = torch.zeros_like(logits)

    gap = gap.float().view(-1)

    return logits, gap


def _rift_predict_logits_for_grad(self, x):
    """Gradient-enabled deployed logit for logit-saliency/Grad-CAM fallback."""
    logits, _ = self._forward_grad(x, donor=None)
    return logits


def _rift_explain_identity_gap(self, x, donor=None, source_id=None, target_id=None,
                               strict=None):
    """Delta-grounded input attribution: |d(delta)/dx| pooled over channels.

    WHY THIS WAS REWRITTEN (_rift_explain_identity_gap_strict)
    ----------------------------------------------------------
    The previous version had a three-tier silent fallback:
        1. donor-grounded gap gradient
        2. deployed logit gradient        (NOT Delta-grounded)
        3. normalized image energy        (x.abs().mean() -- literally brightness)

    Tiers 1 and 2 only *returned* when `score.requires_grad` was True. When it was
    False there was NO exception, so the try/except never fired and NO warning was
    emitted -- execution simply fell through to tier 3 and returned image
    brightness. That map would then be reported in Table 1 under a "Delta-grounded:
    yes" tick, which is false.

    This is not hypothetical: tier 2 is known-broken on this checkpoint, because
    CIFT's logits are read out of LDM's `loss_dict`, whose entries are detached by
    convention. GradCAMExplainer raising "CIFT logit score does not require grad"
    is the same root cause.

    A wrong number that looks plausible is worse than a failed row: the failed row
    gets fixed, the wrong number gets published. So: raise, with a precise
    diagnosis of which tier failed and why.

    Escape hatch: strict=False, or RIFT_ALLOW_SALIENCY_FALLBACK=1. If you use it,
    self.last_saliency_source records what actually ran, and run_table1 surfaces it
    as a CSV column so the row cannot be reported as Delta-grounded by accident.
    """
    import os

    if not _HAS_TORCH:
        raise RuntimeError("explain_identity_gap needs torch.")

    if strict is None:
        strict = os.environ.get("RIFT_ALLOW_SALIENCY_FALLBACK", "0") != "1"

    x = x.clone().detach().to(self.device).float().requires_grad_(True)
    if donor is not None:
        donor = donor.to(self.device).float()

    why = []

    # ---- tier 1: donor-grounded TRUE gap gradient (the only Delta-valid path)
    if donor is None:
        why.append("no donor supplied -> true-gap gradient not attempted "
                   "(Delta is a DONOR identity gap; without a donor there is no Delta)")
    else:
        try:
            _, gap = self._forward_grad(x, donor=donor)
            score = gap.sum()
            if not getattr(score, "requires_grad", False):
                why.append(
                    "gap tensor has requires_grad=False -- control_model._gap is "
                    "either detached inside CIFT, or absent so _forward_grad "
                    "substituted torch.zeros_like(logits)"
                )
            else:
                grad = torch.autograd.grad(
                    score, x, retain_graph=False, create_graph=False, allow_unused=True
                )[0]
                if grad is None:
                    why.append("autograd.grad returned None for the gap path -- "
                               "the graph does not reach the input tensor")
                else:
                    self.last_saliency_source = "true_gap_gradient"
                    return grad.abs().mean(dim=1, keepdim=True).detach()
        except Exception as e:
            why.append(f"gap path raised {type(e).__name__}: {e}")

    # ---- tier 2: deployed logit gradient. NOT Delta-grounded. -------------
    try:
        logits = self._forward_grad(x, donor=None)[0]
        score = logits.sum()
        if not getattr(score, "requires_grad", False):
            why.append(
                "logit tensor has requires_grad=False -- CIFT logits are read from "
                "LDM's loss_dict['v/logits'], and loss_dict entries are detached by "
                "convention. Hook the classifier module instead of reading loss_dict."
            )
        else:
            grad = torch.autograd.grad(
                score, x, retain_graph=False, create_graph=False, allow_unused=True
            )[0]
            if grad is not None:
                if strict:
                    raise RuntimeError(
                        "explain_identity_gap fell through to the LOGIT gradient.\n"
                        + "\n".join(f"  - {w}" for w in why)
                        + "\n\nThat map is grounded in the deployed logit, NOT in the "
                          "donor identity gap. Returning it under a 'Delta-grounded' "
                          "tick would make Table 1's Delta column false, and would "
                          "collapse the row-4-vs-row-3 contrast the paper depends on."
                    )
                warnings.warn("explain_identity_gap using LOGIT gradient, not Delta.",
                              RuntimeWarning)
                self.last_saliency_source = "logit_gradient_fallback"
                return grad.abs().mean(dim=1, keepdim=True).detach()
            why.append("autograd.grad returned None for the logit path")
    except RuntimeError:
        raise
    except Exception as e:
        why.append(f"logit path raised {type(e).__name__}: {e}")

    # ---- tier 3: image energy. This is brightness, not attribution. -------
    if strict:
        raise RuntimeError(
            "explain_identity_gap could not produce a Delta-grounded map.\n"
            + "\n".join(f"  - {w}" for w in why)
            + "\n\nRefusing to return the image-energy fallback: it is "
              "x.abs().mean() -- the average brightness of the input -- and has "
              "nothing to do with CIFT's identity gap. Reporting it under a "
              "'Delta-grounded' tick would be a fabricated result.\n"
              "To inspect it anyway: RIFT_ALLOW_SALIENCY_FALLBACK=1 (the row is then "
              "tagged saliency_source=image_energy_fallback and MUST NOT be reported "
              "as Delta-grounded)."
        )

    warnings.warn("explain_identity_gap returning IMAGE-ENERGY fallback. This is NOT "
                  "a Delta map and must not be reported as one.", RuntimeWarning)
    self.last_saliency_source = "image_energy_fallback"
    sal = x.detach().abs().mean(dim=1, keepdim=True)
    sal = sal - sal.amin(dim=(2, 3), keepdim=True)
    sal = sal / (sal.amax(dim=(2, 3), keepdim=True) + 1e-8)
    return sal


_rift_explain_identity_gap_strict = _rift_explain_identity_gap


# Attach patched methods to CIFTAdapter.
CIFTAdapter._forward_grad = _rift_forward_grad
CIFTAdapter.predict_logits_for_grad = _rift_predict_logits_for_grad
CIFTAdapter.explain_identity_gap = _rift_explain_identity_gap


# =============================================================================
# RIFT FAST-DDP PATCH: per-sample identity-gap vector for batched PPO
# =============================================================================
def _rift_identity_gap_tensor(self, x, donor=None, source_id=None, target_id=None):
    """Return (gap[B], mode) without collapsing the batch to one scalar."""

    if not _HAS_TORCH:
        raise RuntimeError("identity_gap_tensor needs torch.")

    x = x.to(self.device).float()
    B = int(x.shape[0])

    has_donor = donor is not None or (source_id is not None and target_id is not None)
    mode = resolve_mode(has_donor, self.strict_identity_gap)

    if mode == IdentityGapMode.ERROR:
        raise MechanismValidityError(
            "strict_identity_gap=True but no donor tensor was passed. "
            "Fast batched RIFT requires donor_path/source_ref_path in every row."
        )

    if mode == IdentityGapMode.TRUE:
        if donor is None:
            raise MechanismValidityError(
                "TRUE identity-gap mode needs donor tensor, not only metadata IDs."
            )

        donor = donor.to(self.device).float()

        _, gap = self._forward(x, donor=donor)
        gap = gap.detach().float().view(-1)

        if gap.numel() == 1 and B > 1:
            gap = gap.repeat(B)
        elif gap.numel() > B:
            gap = gap[:B].contiguous()

        return gap, IdentityGapMode.TRUE.value

    if not self._proxy_warned:
        warnings.warn(
            "CIFTAdapter using PROXY identity-gap tensor (no donor stream).",
            RuntimeWarning,
        )
        self._proxy_warned = True

    feat = self.extract_features(x) if self.model is not None else x

    if feat.dim() > 2:
        feat = feat.flatten(2).mean(-1)

    gap = feat.detach().float().norm(dim=-1).view(-1)

    if gap.numel() == 1 and B > 1:
        gap = gap.repeat(B)
    elif gap.numel() > B:
        gap = gap[:B].contiguous()

    return gap, IdentityGapMode.PROXY.value


CIFTAdapter.identity_gap_tensor = _rift_identity_gap_tensor

# =============================================================================
# RIFT FAST PATCH: batch identity-gap tensor
# =============================================================================
# BatchedRIFTEnv calls adapter.identity_gap_tensor(...) so it can compute Δ for
# an entire mini-batch in one CIFT forward instead of looping image-by-image.
# This keeps TRUE donor-grounded Δ when donor tensors are available.
# =============================================================================

def _rift_proxy_gap_tensor(self, x):
    if not _HAS_TORCH:
        raise RuntimeError("proxy batch gap needs torch.")

    x = x.to(self.device).float()

    with torch.no_grad():
        feat = self.extract_features(x) if self.model is not None else x

        if feat.dim() > 2:
            feat = feat.flatten(2).mean(-1)

        gap = feat.norm(dim=-1).detach().float().view(-1)

    if gap.numel() == 1 and x.shape[0] > 1:
        gap = gap.repeat(x.shape[0])

    if gap.numel() > x.shape[0]:
        gap = gap[: x.shape[0]].contiguous()

    return gap


def _rift_identity_gap_tensor(self, x, donor=None, source_id=None, target_id=None):
    """Return per-sample identity gap tensor [B], plus mode string.

    TRUE mode:
      Uses CIFT dual donor-target stream and returns control_model._gap per sample.

    PROXY mode:
      Uses single-stream feature norm per sample.
    """
    if not _HAS_TORCH:
        raise RuntimeError("identity_gap_tensor needs torch.")

    x = x.to(self.device).float()
    B = int(x.shape[0])

    has_donor = donor is not None or (source_id is not None and target_id is not None)
    mode = resolve_mode(has_donor, self.strict_identity_gap)

    if mode == IdentityGapMode.ERROR:
        raise MechanismValidityError(
            "strict_identity_gap=True but no donor stream for this batch. "
            "Provide donor_path/source_ref_path in the CSV, or set strict_identity_gap=False."
        )

    if mode == IdentityGapMode.TRUE:
        if donor is None:
            raise MechanismValidityError(
                "TRUE identity-gap mode needs a donor tensor batch. "
                "source_id/target_id metadata alone is not enough."
            )

        donor = donor.to(self.device).float()

        with torch.no_grad():
            _, gap = self._forward(x, donor=donor)

        gap = gap.detach().float().view(-1)

        if gap.numel() == 1 and B > 1:
            gap = gap.repeat(B)

        if gap.numel() > B:
            gap = gap[:B].contiguous()

        if gap.numel() < B:
            reps = B // gap.numel() + 1
            gap = gap.repeat(reps)[:B].contiguous()

        # If dual path gives all-zero despite donor, downgrade honestly.
        if float(gap.abs().mean().item()) < 1e-6:
            return self._proxy_gap_tensor(x), IdentityGapMode.PROXY.value

        return gap, IdentityGapMode.TRUE.value

    return self._proxy_gap_tensor(x), IdentityGapMode.PROXY.value


CIFTAdapter._proxy_gap_tensor = _rift_proxy_gap_tensor
CIFTAdapter.identity_gap_tensor = _rift_identity_gap_tensor
