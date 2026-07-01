# Path: src/rl/policy.py
"""Grid policy network for RIFT.

DDP-safe version:
  - No nn.LazyLinear
  - No uninitialized parameters
  - Works with torchrun + DistributedDataParallel
  - Handles CIFT feature tensors robustly

State expected from RIFTEnv:
  state["feat"]         : optional tensor [B,C,H,W] or [B,C] or [C,H,W]
  state["current_mask"] : tensor [B,1,G,G] or [B,G,G] or [G,G]
  state["confidence"]   : scalar/tensor
  state["step_idx"]     : scalar
  state["last_action"]  : scalar
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


if _HAS_TORCH:

    class GridPolicy(nn.Module):
        def __init__(self, feat_dim=1024, grid=8, n_actions=None, hidden=256):
            super().__init__()

            self.feat_dim = int(feat_dim)
            self.grid = int(grid)
            self.hidden = int(hidden)
            self.n_actions = int(n_actions or self.grid * self.grid + 1)

            self.mask_enc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.grid * self.grid, self.hidden),
                nn.ReLU(inplace=True),
            )

            self.scalar_dim = self.hidden // 4

            self.scalar_enc = nn.Sequential(
                nn.Linear(3, self.scalar_dim),
                nn.ReLU(inplace=True),
            )

            self.feat_pool = nn.AdaptiveAvgPool2d(1)

            self.feat_enc = nn.Sequential(
                nn.Linear(self.feat_dim, self.hidden),
                nn.ReLU(inplace=True),
            )

            trunk_in = self.hidden + self.hidden + self.scalar_dim

            self.trunk = nn.Sequential(
                nn.Linear(trunk_in, self.hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(inplace=True),
            )

            self.pi = nn.Linear(self.hidden, self.n_actions)
            self.v = nn.Linear(self.hidden, 1)

        def _batch_size(self, state):
            m = state["current_mask"]

            if not torch.is_tensor(m):
                raise RuntimeError("state['current_mask'] must be a tensor")

            if m.dim() == 2:
                return 1

            return int(m.shape[0])

        def _match_batch(self, x, batch_size):
            if x.shape[0] == batch_size:
                return x

            if x.shape[0] == 1:
                reps = [batch_size] + [1] * (x.dim() - 1)
                return x.repeat(*reps)

            if x.shape[0] > batch_size:
                return x[:batch_size]

            reps = [batch_size // x.shape[0] + 1] + [1] * (x.dim() - 1)
            return x.repeat(*reps)[:batch_size]

        def _encode_mask(self, state):
            m = state["current_mask"]

            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:
                m = m.unsqueeze(1)

            m = m.float()

            if m.shape[-2:] != (self.grid, self.grid):
                m = torch.nn.functional.interpolate(
                    m,
                    size=(self.grid, self.grid),
                    mode="nearest",
                )

            return self.mask_enc(m)

        def _scalar_to_float(self, x):
            if torch.is_tensor(x):
                if x.numel() == 0:
                    return 0.0
                return float(x.detach().flatten()[0].item())

            try:
                return float(x)
            except Exception:
                return 0.0

        def _encode_scalars(self, state, batch_size, device):
            confidence = self._scalar_to_float(state.get("confidence", 0.0))
            step_idx = self._scalar_to_float(state.get("step_idx", 0.0))
            last_action = self._scalar_to_float(state.get("last_action", 0.0))

            sc = torch.tensor(
                [[confidence, step_idx, last_action]],
                device=device,
                dtype=torch.float32,
            )

            sc = sc.repeat(batch_size, 1)

            return self.scalar_enc(sc)

        def _prepare_feat_vector(self, feat, batch_size, device):
            if feat is None:
                return torch.zeros(
                    batch_size,
                    self.feat_dim,
                    device=device,
                    dtype=torch.float32,
                )

            if not torch.is_tensor(feat):
                return torch.zeros(
                    batch_size,
                    self.feat_dim,
                    device=device,
                    dtype=torch.float32,
                )

            feat = feat.to(device=device, dtype=torch.float32)

            # Possible shapes:
            #   [C,H,W]      -> [1,C,H,W]
            #   [B,C,H,W]    -> pool -> [B,C]
            #   [B,C]        -> keep
            #   [B,C,L]      -> mean over L -> [B,C]
            #   [C]          -> [1,C]
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)

            elif feat.dim() == 3:
                # If looks like [C,H,W], add batch.
                if feat.shape[0] != batch_size:
                    feat = feat.unsqueeze(0)

                if feat.dim() == 4:
                    feat = self.feat_pool(feat).flatten(1)
                else:
                    # [B,C,L] -> [B,C]
                    feat = feat.mean(dim=-1)

            elif feat.dim() == 4:
                feat = self.feat_pool(feat).flatten(1)

            elif feat.dim() > 4:
                feat = feat.flatten(2).mean(dim=-1)

            feat = self._match_batch(feat, batch_size)

            # Pad/truncate to configured feat_dim.
            c = feat.shape[1]

            if c < self.feat_dim:
                pad = torch.zeros(
                    feat.shape[0],
                    self.feat_dim - c,
                    device=feat.device,
                    dtype=feat.dtype,
                )
                feat = torch.cat([feat, pad], dim=1)

            elif c > self.feat_dim:
                feat = feat[:, : self.feat_dim]

            return feat

        def _encode_feat(self, state, batch_size, device):
            feat = state.get("feat", None)
            feat_vec = self._prepare_feat_vector(feat, batch_size, device)
            return self.feat_enc(feat_vec)

        def _encode(self, state):
            if "current_mask" not in state:
                raise RuntimeError("GridPolicy state missing key: current_mask")

            batch_size = self._batch_size(state)
            device = state["current_mask"].device

            mask_h = self._encode_mask(state)
            mask_h = self._match_batch(mask_h, batch_size)

            feat_h = self._encode_feat(state, batch_size, device)

            scalar_h = self._encode_scalars(state, batch_size, device)

            h = torch.cat([feat_h, mask_h, scalar_h], dim=-1)

            return self.trunk(h)

        def forward(self, state):
            h = self._encode(state)
            logits = self.pi(h)
            value = self.v(h)

            return logits, value

        @torch.no_grad()
        def act(self, state, deterministic=False):
            logits, _ = self.forward(state)

            if deterministic:
                return int(logits.argmax(dim=-1)[0].item())

            probs = torch.softmax(logits, dim=-1)

            return int(torch.multinomial(probs, 1)[0, 0].item())


else:

    class GridPolicy:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch required")
