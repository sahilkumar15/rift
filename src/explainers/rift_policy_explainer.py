# Path: src/explainers/rift_policy_explainer.py
# Status: MODIFIED
"""Wrap a trained RIFT-RL policy as an explainer."""
from .base_explainer import BaseExplainer


class RIFTPolicyExplainer(BaseExplainer):
    name = "rift_policy"

    def __init__(self, policy, env_builder, horizon=4):
        self.policy = policy
        self.env_builder = env_builder
        self.horizon = horizon

    def explain(self, image, adapter, **kw):
        env = self.env_builder(image, adapter, **kw)
        state = env.reset()

        for _ in range(self.horizon):
            action = self.policy.act(state, deterministic=True)
            state, _, done, _ = env.step(action)

            if done:
                break

        return env.current_mask()