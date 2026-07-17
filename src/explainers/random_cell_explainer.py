# Path: src/explainers/random_cell_explainer.py
# Status: NEW
"""Area- AND geometry-matched random-cell control.

WHY THIS EXISTS
---------------
The legacy RandomExplainer returned torch.rand(B,1,H,W): a continuous field at
pixel resolution. _to_binary() then kept the top-k fraction of INDIVIDUAL,
SPATIALLY SCATTERED pixels. That is not a valid control for a RIFT policy mask:

  * The policy selects whole 8x8 grid cells (each cell = 1/64 of the image,
    a contiguous 32x32 block at 256x256).
  * apply_necessity(mode="blur") convolves with a 15x15 average kernel. Blurring
    an ISOLATED pixel replaces it with approximately its own local mean, i.e.
    a near-identity operation. Blurring a contiguous 32x32 block genuinely
    destroys the content.

So a scattered-pixel control is guaranteed a near-zero evidence move for reasons
of geometry, not faithfulness. It would make Gate 1 pass trivially and falsely.

This explainer samples exactly `cells` distinct cells uniformly without
replacement from the grid*grid lattice, so mask_area == cells/(grid*grid) and
the intervention geometry is identical to the policy's.
"""
from __future__ import annotations

from typing import Optional

from .base_explainer import BaseExplainer


class RandomCellExplainer(BaseExplainer):
    """Uniform random selection of `cells` distinct grid cells.

    Args:
        cells: number of grid cells to select. Match this to the horizon of the
            policy you are controlling against (H=4 -> cells=4 -> area=0.0625).
        grid: lattice resolution (RIFT uses 8).
        seed: base seed. Each explain() call advances the generator, so repeated
            calls give independent draws; construct with a fixed seed for a
            reproducible sequence.
    """

    name = "random_cell"

    def __init__(self, cells: int = 4, grid: int = 8, seed: Optional[int] = None):
        self.grid = int(grid)
        self.n_cells = self.grid * self.grid
        self.cells = max(1, min(int(cells), self.n_cells))
        self.seed = seed
        self._generators = {}

    def _generator(self, device):
        import torch

        key = str(device)
        gen = self._generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            if self.seed is not None:
                gen.manual_seed(int(self.seed))
            self._generators[key] = gen
        return gen

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        batch = int(image.shape[0])
        device = image.device
        gen = self._generator(device)

        with torch.inference_mode():
            # Random permutation per sample via argsort of uniform noise. This is
            # sampling WITHOUT replacement, so mask_area is exact, not expected.
            noise = torch.rand(
                batch,
                self.n_cells,
                device=device,
                generator=gen,
                dtype=torch.float32,
            )
            chosen = noise.topk(self.cells, dim=1).indices

            flat = torch.zeros(
                batch,
                self.n_cells,
                device=device,
                dtype=image.dtype,
            )
            flat.scatter_(1, chosen, 1.0)

            grid_mask = flat.view(batch, 1, self.grid, self.grid)
            mask = F.interpolate(
                grid_mask,
                size=image.shape[-2:],
                mode="nearest",
            )

        return mask.detach().clone()
