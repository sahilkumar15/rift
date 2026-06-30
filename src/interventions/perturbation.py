# Path: src/interventions/perturbation.py
# Status: NEW
"""Optional frequency-domain / patch-dropout perturbations. torch-guarded."""
from __future__ import annotations
try:
    import torch; import torch.fft as fft; _HAS=True
except Exception: _HAS=False
def freq_lowpass(image, cutoff=0.25):
    if not _HAS: raise RuntimeError("torch required (Katz).")
    F = fft.fftshift(fft.fft2(image), dim=(-2,-1))
    B,C,H,W = image.shape
    yy,xx = torch.meshgrid(torch.linspace(-1,1,H), torch.linspace(-1,1,W), indexing="ij")
    keep = ((yy**2+xx**2).sqrt() < cutoff).to(image.device).float()
    return fft.ifft2(fft.ifftshift(F*keep, dim=(-2,-1))).real
def patch_dropout(image, mask, drop=0.5):
    if not _HAS: raise RuntimeError("torch required (Katz).")
    keep = (torch.rand_like(mask) > drop).float()
    return image*(1-mask) + image*mask*keep
