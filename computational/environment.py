"""Factor-structured bandit environment for DLN simulations.

Environment:
- K options
- F latent factors (shared structure)
- Each option has factor loadings L[k, f]
- Latent factor state s[f] is fixed within a run (can be extended to evolve over time)

Reward model (simple):
  reward = L[action] · s + noise

This environment is intentionally minimal: it supplies shared structure that the
Network agent can exploit and the Linear agent cannot (by design).

"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EnvConfig:
    K: int
    F: int
    noise_sd: float = 0.15
    factor_sd: float = 1.0
    seed: int = 0


class FactorBanditEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.K = int(cfg.K)
        self.F = int(cfg.F)

        # Option-factor structure (shared loadings)
        self.L = self.rng.normal(0.0, 1.0, size=(self.K, self.F))
        # Normalize loadings for stability
        self.L = self.L / (np.linalg.norm(self.L, axis=1, keepdims=True) + 1e-8)

        # Latent factor state
        self.s = self.rng.normal(0.0, cfg.factor_sd, size=(self.F,))

    def step(self, action: int) -> float:
        action = int(action)
        mean = float(self.L[action] @ self.s)
        reward = mean + float(self.rng.normal(0.0, self.cfg.noise_sd))
        return reward
