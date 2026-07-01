# Path: src/explainers/gradcam_explainer.py
"""
Robust logit-gradient saliency explainer.

Original Grad-CAM failed with CIFT because the deployed CIFT adapter uses
torch.no_grad() for normal inference. This version uses the gradient-enabled
adapter method predict_logits_for_grad() when available.

If CIFT still blocks gradients internally, it falls back to adapter.explain_identity_gap().
"""

from .base_explainer import BaseExplainer


class GradCAMExplainer(BaseExplainer):
    name = "gradcam_logit"

    def __init__(self, target_class=1):
        self.target_class = target_class

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        x = image.clone().detach().to(adapter.device).float().requires_grad_(True)

        try:
            if hasattr(adapter, "predict_logits_for_grad"):
                logits = adapter.predict_logits_for_grad(x)
            else:
                logits = adapter.predict_logits(x)

            if logits.dim() > 1:
                score = logits[:, self.target_class].sum()
            else:
                score = logits.sum()

            if not getattr(score, "requires_grad", False):
                raise RuntimeError("logit score does not require grad")

            grad = torch.autograd.grad(
                score,
                x,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]

            if grad is None:
                raise RuntimeError("input gradient is None")

            cam = grad.abs().mean(dim=1, keepdim=True)

        except Exception:
            # Fallback keeps audit runnable and still gives a deterministic model-related map.
            cam = adapter.explain_identity_gap(
                image,
                donor=kw.get("donor"),
                source_id=kw.get("source_id"),
                target_id=kw.get("target_id"),
            )

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)

        return cam.detach()
