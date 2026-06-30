# Path: src/explainers/gradcam_explainer.py
# Status: NEW
"""Grad-CAM saliency from detector features w.r.t. the fake logit."""
from .base_explainer import BaseExplainer
class GradCAMExplainer(BaseExplainer):
    name = "gradcam"
    def __init__(self, target_class=1): self.target_class=target_class
    def explain(self, image, adapter, **kw):
        import torch, torch.nn.functional as F
        image = image.clone().requires_grad_(True)
        feat = adapter.extract_features(image)
        logit = adapter.predict_logits(image)
        score = logit[:, self.target_class] if logit.dim()>1 else logit
        grads = torch.autograd.grad(score.sum(), feat, retain_graph=False,
                                    create_graph=False, allow_unused=True)[0]
        if grads is None:  # feature not in graph; fall back to input grad
            g = torch.autograd.grad(score.sum(), image)[0]
            cam = g.abs().mean(1, keepdim=True)
        else:
            w = grads.mean(dim=(2,3), keepdim=True)
            cam = F.relu((w*feat).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = (cam - cam.amin(dim=(2,3),keepdim=True)) / \
              (cam.amax(dim=(2,3),keepdim=True)-cam.amin(dim=(2,3),keepdim=True)+1e-8)
        return cam.detach()
