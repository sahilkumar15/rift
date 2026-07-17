# Path: src/explainers/grid_topk.py
# Status: NEW
"""Force ANY soft saliency explainer onto the policy's mask budget + geometry.

WHY THIS EXISTS
---------------
GradCAMExplainer and CIFTGapExplainer return soft, pixel-resolution maps in
[0,1]. Downstream, _to_binary()/_mask_area_tensor() keep the top `topk_frac`
fraction of INDIVIDUAL PIXELS. With topk_frac=0.12 that yields mask_area=0.12.

RIFT policies return HARD 8x8 grid-cell masks. At horizon 4 that is
mask_area = 4/64 = 0.0625, laid out as 4 contiguous 32x32 blocks.

Comparing a 0.12-area pixel-scatter against a 0.0625-area cell mask confounds
area and geometry with faithfulness. rift_score is monotone in area (H1..H12
demonstrates this), so the row with the bigger budget wins for free.

This wrapper pools any soft map to the grid, keeps the top `cells` CELLS, and
emits a hard mask. Every row in Table 1 then has:
    mask_area  == cells / (grid*grid)      EXACTLY, not in expectation
    geometry   == contiguous grid blocks   identical to the policy
so the only thing that varies across rows is WHERE the method points.
"""
from __future__ import annotations

from .base_explainer import BaseExplainer


class GridTopKExplainer(BaseExplainer):
    """Wrap a soft explainer -> hard top-k grid-cell mask.

    Args:
        base: any BaseExplainer returning (B,1,H,W) or (B,H,W) soft maps.
        cells: cell budget. Set equal to the policy horizon being compared.
        grid: lattice size (RIFT uses 8).
        pool: 'avg' scores a cell by its mean saliency (default; standard
            practice for region scoring). 'max' scores by peak saliency, which
            favors methods with sharp isolated spikes.
    """

    def __init__(self, base, *, cells: int = 4, grid: int = 8, pool: str = "avg"):
        self.base = base
        self.grid = int(grid)
        self.n_cells = self.grid * self.grid
        self.cells = max(1, min(int(cells), self.n_cells))
        self.pool = str(pool).lower()
        self.name = f"gridtopk[{getattr(base, 'name', 'base')}]"

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        soft = self.base.explain(image, adapter, **kw)
        soft = soft.float()
        if soft.dim() == 3:
            soft = soft.unsqueeze(1)
        if soft.shape[1] > 1:
            soft = soft.mean(dim=1, keepdim=True)

        batch = int(soft.shape[0])

        with torch.inference_mode():
            if self.pool == "max":
                pooled = F.adaptive_max_pool2d(soft, (self.grid, self.grid))
            else:
                pooled = F.adaptive_avg_pool2d(soft, (self.grid, self.grid))

            scores = pooled.flatten(1)
            chosen = scores.topk(self.cells, dim=1).indices

            flat = torch.zeros(
                batch, self.n_cells, device=soft.device, dtype=image.dtype
            )
            flat.scatter_(1, chosen, 1.0)

            mask = F.interpolate(
                flat.view(batch, 1, self.grid, self.grid),
                size=image.shape[-2:],
                mode="nearest",
            )

        return mask.detach().clone()
