# Path: src/rl/policy.py
"""Grid policy network for RIFT.

This policy consumes image features, the current grid mask, and normalized scalar
state. state_blind=True is a budget-matched control: it hides the Markov state
(mask/step/last action) while preserving image features and the same action
budget.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


if _HAS_TORCH:

    class GridPolicy(nn.Module):
        def __init__(
            self,
            feat_dim=1024,
            grid=8,
            n_actions=None,
            hidden=256,
            state_blind=False,
        ):
            super().__init__()
            self.state_blind = bool(state_blind)
            self.feat_dim = int(feat_dim)
            self.grid = int(grid)
            self.hidden = int(hidden)
            self.n_actions = int(n_actions or self.grid * self.grid + 1)

            self.mask_enc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.grid * self.grid, self.hidden),
                nn.ReLU(inplace=True),
            )

            self.scalar_dim = max(16, self.hidden // 4)
            # confidence, step_idx, last_action, selected_frac
            self.scalar_enc = nn.Sequential(
                nn.Linear(4, self.scalar_dim),
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
                return x.repeat(*([batch_size] + [1] * (x.dim() - 1)))
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
                m = F.interpolate(m, size=(self.grid, self.grid), mode="nearest")
            return self.mask_enc(m)

        def _scalar_vec(self, x, batch_size, device):
            if torch.is_tensor(x):
                t = x.detach().to(device=device, dtype=torch.float32).view(-1)
            else:
                try:
                    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
                except Exception:
                    t = torch.zeros(1, device=device, dtype=torch.float32)

            if t.numel() == batch_size:
                return t
            if t.numel() == 1:
                return t.repeat(batch_size)
            if t.numel() > batch_size:
                return t[:batch_size]
            reps = batch_size // t.numel() + 1
            return t.repeat(reps)[:batch_size]

        def _encode_scalars(self, state, batch_size, device):
            confidence = self._scalar_vec(state.get("confidence", 0.0), batch_size, device)
            step_idx = self._scalar_vec(state.get("step_idx", 0.0), batch_size, device)
            last_action = self._scalar_vec(state.get("last_action", -1.0), batch_size, device)
            selected_frac = self._scalar_vec(state.get("selected_frac", 0.0), batch_size, device)

            # Keep scalar magnitudes comparable. Raw last_action is 0..64 and
            # otherwise dominates the scalar MLP.
            confidence = torch.log1p(confidence.clamp_min(0.0))
            step_idx = step_idx / max(float(self.grid), 1.0)
            last_action = (last_action + 1.0) / max(float(self.n_actions), 1.0)
            selected_frac = selected_frac.clamp(0.0, 1.0)

            sc = torch.stack([confidence, step_idx, last_action, selected_frac], dim=1)
            return self.scalar_enc(sc)

        def _prepare_feat_vector(self, feat, batch_size, device):
            if feat is None or not torch.is_tensor(feat):
                return torch.zeros(batch_size, self.feat_dim, device=device, dtype=torch.float32)

            feat = feat.to(device=device, dtype=torch.float32)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            elif feat.dim() == 3:
                if feat.shape[0] != batch_size:
                    feat = feat.unsqueeze(0)
                    feat = self.feat_pool(feat).flatten(1)
                else:
                    feat = feat.mean(dim=-1)
            elif feat.dim() == 4:
                feat = self.feat_pool(feat).flatten(1)
            elif feat.dim() > 4:
                feat = feat.flatten(2).mean(dim=-1)

            feat = self._match_batch(feat, batch_size)
            c = int(feat.shape[1])
            if c < self.feat_dim:
                pad = torch.zeros(feat.shape[0], self.feat_dim - c, device=feat.device, dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=1)
            elif c > self.feat_dim:
                feat = feat[:, : self.feat_dim]
            return feat

        def _encode_feat(self, state, batch_size, device):
            feat_vec = self._prepare_feat_vector(state.get("feat"), batch_size, device)
            return self.feat_enc(feat_vec)

        def _encode(self, state):
            if "current_mask" not in state:
                raise RuntimeError("GridPolicy state missing key: current_mask")
            batch_size = self._batch_size(state)
            device = state["current_mask"].device

            if self.state_blind:
                state = dict(state)
                state["current_mask"] = torch.zeros_like(state["current_mask"])
                state["step_idx"] = 0.0
                state["last_action"] = -1.0
                state["selected_frac"] = 0.0

            mask_h = self._match_batch(self._encode_mask(state), batch_size)
            feat_h = self._encode_feat(state, batch_size, device)
            scalar_h = self._encode_scalars(state, batch_size, device)
            return self.trunk(torch.cat([feat_h, mask_h, scalar_h], dim=-1))

        def forward(self, state):
            h = self._encode(state)
            return self.pi(h), self.v(h)

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
