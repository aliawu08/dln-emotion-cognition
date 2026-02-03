"""DLN agent implementations.

These agents mirror the mapping used in the DLN computational framework (Wu, 2026):

- Dot: reactive policy with negligible persistent belief state.
- Linear: K independent option-value estimates (no information sharing).
- Network: shared latent structure (factorized representation) + a structural learning cycle.

This code is a simplified, runnable reference implementation intended for:
- generating qualitative regime diagrams,
- producing synthetic predictions that connect to the fusion model's emotion–cognition hypotheses,
- serving as a starting point for more faithful reproductions of the DLN computational model.

"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) / max(temperature, 1e-8)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


@dataclass
class StepResult:
    action: int
    reward: float
    cost: float
    prediction_error: float


class DotAgent:
    """Reactive agent (dot stage): minimal persistent state."""

    def __init__(self, K: int, temperature: float = 1.0, rng: np.random.Generator | None = None):
        self.K = int(K)
        self.temperature = float(temperature)
        self.rng = rng or np.random.default_rng()

        # minimal state: last reward per action (optional)
        self.last_reward = np.zeros(self.K, dtype=float)

    def choose(self) -> int:
        # weakly prefer actions that just paid off
        probs = softmax(self.last_reward, temperature=self.temperature)
        return int(self.rng.choice(self.K, p=probs))

    def update(self, action: int, reward: float) -> StepResult:
        self.last_reward[action] = reward
        # Dot: negligible cognitive cost
        return StepResult(action=action, reward=float(reward), cost=0.0, prediction_error=0.0)


class LinearAgent:
    """Linear agent (linear stage): K independent option-value estimates."""

    def __init__(self, K: int, lr: float = 0.15, temperature: float = 0.8, rng: np.random.Generator | None = None):
        self.K = int(K)
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.rng = rng or np.random.default_rng()

        self.Q = np.zeros(self.K, dtype=float)

    def choose(self) -> int:
        probs = softmax(self.Q, temperature=self.temperature)
        return int(self.rng.choice(self.K, p=probs))

    def update(self, action: int, reward: float) -> StepResult:
        pred = self.Q[action]
        pe = float(reward - pred)
        self.Q[action] = pred + self.lr * pe

        # Linear: O(K) cognitive cost proxy (evaluating K independent options)
        cost = float(self.K)
        return StepResult(action=action, reward=float(reward), cost=cost, prediction_error=pe)


class NetworkAgent:
    """Network agent (network stage): shared latent factors + structural learning cycle."""

    def __init__(
        self,
        K: int,
        F: int,
        option_loadings: np.ndarray,
        lr: float = 0.12,
        temperature: float = 0.8,
        structure_threshold: float = 0.35,
        rng: np.random.Generator | None = None,
    ):
        self.K = int(K)
        self.F = int(F)
        self.L = np.asarray(option_loadings, dtype=float)  # shape (K, F)
        assert self.L.shape == (self.K, self.F)

        self.lr = float(lr)
        self.temperature = float(temperature)
        self.structure_threshold = float(structure_threshold)
        self.rng = rng or np.random.default_rng()

        # Factor belief state
        self.f_hat = np.zeros(self.F, dtype=float)

        # Track structural updates as a proxy for explicit hypothesis->test->update cycles
        self.structural_updates = 0

    def values(self) -> np.ndarray:
        return self.L @ self.f_hat

    def choose(self) -> int:
        v = self.values()
        probs = softmax(v, temperature=self.temperature)
        return int(self.rng.choice(self.K, p=probs))

    def update(self, action: int, reward: float) -> StepResult:
        pred = float(self.values()[action])
        pe = float(reward - pred)

        # Gradient-like update on factor beliefs
        self.f_hat = self.f_hat + self.lr * pe * self.L[action]

        # Structural learning cycle (simplified): if error is large, pay an additional cost
        extra_cost = 0.0
        if abs(pe) > self.structure_threshold:
            self.structural_updates += 1
            extra_cost = 0.5 * self.F  # cost proxy for hypothesis/test/update

        # Network: O(F) cognitive cost proxy (operate in factor space) + structural update cost
        cost = float(self.F) + float(extra_cost)

        return StepResult(action=action, reward=float(reward), cost=cost, prediction_error=pe)
