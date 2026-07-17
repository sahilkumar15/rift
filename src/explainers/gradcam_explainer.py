# Path: src/explainers/gradcam_explainer.py
# Status: REWRITTEN - the silent Delta fallback is now opt-in and reported.
"""Logit-gradient saliency explainer.

CORRECTNESS NOTE (this is why the file was rewritten)
-----------------------------------------------------
This explainer is the LOGIT-CHANNEL baseline. Its entire job in Table 1 is to
answer: "what does a standard gradient method, grounded in the deployed logit
and knowing nothing about Delta, cite?"

The previous version wrapped everything in `except Exception:` and fell back to
adapter.explain_identity_gap(), i.e. a DELTA-GROUNDED map. That fallback is
reachable in normal operation, because the deployed CIFT adapter runs inference
under torch.no_grad(). When it fired, the "Grad-CAM logit" row silently became a
Delta row, the paper's `Delta-grounding: no` tick became false, and nothing in
the CSV recorded it.

Default is now strict=True: if the logit gradient cannot be obtained, RAISE.
A failed row is recoverable. A silently mislabelled row is a retracted paper.
"""
from __future__ import annotations

import warnings

from .base_explainer import BaseExplainer


class GradCAMExplainer(BaseExplainer):
    name = "gradcam_logit"

    def __init__(self, target_class: int = 1, *, strict: bool = True):
        self.target_class = int(target_class)
        self.strict = bool(strict)
        # Inspect after a run; surfaced as a CSV column by run_table1.
        self.used_delta_fallback = False

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        x = image.clone().detach().to(adapter.device).float().requires_grad_(True)

        try:
            if hasattr(adapter, "predict_logits_for_grad"):
                logits = adapter.predict_logits_for_grad(x)
            else:
                logits = adapter.predict_logits(x)

            score = (
                logits[:, self.target_class].sum()
                if logits.dim() > 1
                else logits.sum()
            )

            if not getattr(score, "requires_grad", False):
                raise RuntimeError(
                    "CIFT logit score does not require grad. The adapter is "
                    "running under no_grad; expose predict_logits_for_grad()."
                )

            grad = torch.autograd.grad(
                score, x, retain_graph=False, create_graph=False, allow_unused=True
            )[0]

            if grad is None:
                raise RuntimeError("Input gradient is None (graph detached).")

            cam = grad.abs().mean(dim=1, keepdim=True)

        except Exception as exc:
            if self.strict:
                raise RuntimeError(
                    "GradCAMExplainer failed to obtain a LOGIT gradient: "
                    f"{type(exc).__name__}: {exc}\n"
                    "Refusing to fall back to the identity-gap map: that would "
                    "silently turn the logit-channel baseline into a "
                    "Delta-grounded row and invalidate Table 1. Pass "
                    "strict=False only if you will report the fallback."
                ) from exc

            warnings.warn(
                "GradCAMExplainer fell back to the DELTA map. This row is NOT a "
                "logit-channel baseline and must not be reported as one.",
                RuntimeWarning,
            )
            self.used_delta_fallback = True
            cam = adapter.explain_identity_gap(
                image,
                donor=kw.get("donor"),
                source_id=kw.get("source_id"),
                target_id=kw.get("target_id"),
            )

        cam = F.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return cam.detach()
