"""Tests for DLN agent behavioral invariants.

These tests verify the theoretical properties that underpin the paper's claims:
- DotAgent: reactive, no cross-option learning
- LinearAgent: independent per-option learning, no information sharing
- NetworkAgent: factor-based generalization across options (given known structure)
- Ordering: Network > Linear > Dot under shared structure with large K
"""

import numpy as np
import pytest

from computational.agents import DotAgent, LinearAgent, NetworkAgent, StepResult, softmax
from computational.environment import EnvConfig, FactorBanditEnv


# ---------------------------------------------------------------------------
# DotAgent
# ---------------------------------------------------------------------------

class TestDotAgent:
    def test_returns_step_result(self):
        agent = DotAgent(K=4, rng=np.random.default_rng(0))
        a = agent.choose()
        result = agent.update(a, 1.0)
        assert isinstance(result, StepResult)

    def test_no_cross_option_contamination(self):
        """Updating action 0 must not change last_reward for other actions."""
        agent = DotAgent(K=4, rng=np.random.default_rng(0))
        agent.update(0, 5.0)
        assert agent.last_reward[0] == 5.0
        assert agent.last_reward[1] == 0.0
        assert agent.last_reward[2] == 0.0
        assert agent.last_reward[3] == 0.0

    def test_uniform_choice_under_equal_rewards(self):
        """With equal last_reward, choices should be approximately uniform."""
        agent = DotAgent(K=4, temperature=1.0, rng=np.random.default_rng(42))
        counts = np.zeros(4)
        for _ in range(4000):
            counts[agent.choose()] += 1
        proportions = counts / counts.sum()
        np.testing.assert_allclose(proportions, 0.25, atol=0.04)

    def test_zero_cognitive_cost(self):
        agent = DotAgent(K=3, rng=np.random.default_rng(0))
        result = agent.update(0, 1.0)
        assert result.cost == 0.0

    def test_zero_prediction_error(self):
        """Dot agent reports no prediction error (no model to err against)."""
        agent = DotAgent(K=3, rng=np.random.default_rng(0))
        result = agent.update(1, 3.0)
        assert result.prediction_error == 0.0

    def test_prefers_recently_rewarded_action(self):
        """After a large reward on action 0, Dot should prefer it."""
        agent = DotAgent(K=4, temperature=0.5, rng=np.random.default_rng(7))
        agent.update(0, 10.0)
        counts = np.zeros(4)
        for _ in range(1000):
            counts[agent.choose()] += 1
        assert counts[0] > counts[1]
        assert counts[0] > counts[2]
        assert counts[0] > counts[3]


# ---------------------------------------------------------------------------
# LinearAgent
# ---------------------------------------------------------------------------

class TestLinearAgent:
    def test_independent_learning(self):
        """Updating action 0 must not change Q-values for other actions."""
        agent = LinearAgent(K=4, rng=np.random.default_rng(0))
        Q_before = agent.Q.copy()
        agent.update(0, 5.0)
        # action 0 should have changed
        assert agent.Q[0] != Q_before[0]
        # others must be unchanged
        np.testing.assert_array_equal(agent.Q[1:], Q_before[1:])

    def test_q_moves_toward_reward(self):
        """After repeated reward=1.0 on action 0, Q[0] should approach 1.0."""
        agent = LinearAgent(K=2, lr=0.15, rng=np.random.default_rng(0))
        for _ in range(100):
            agent.update(0, 1.0)
        assert agent.Q[0] > 0.9
        assert agent.Q[1] == 0.0  # untouched

    def test_prediction_error_sign(self):
        """PE should be positive when reward > current Q."""
        agent = LinearAgent(K=2, rng=np.random.default_rng(0))
        result = agent.update(0, 1.0)
        assert result.prediction_error > 0.0
        # After convergence, PE should be near zero
        for _ in range(200):
            agent.update(0, 1.0)
        result = agent.update(0, 1.0)
        assert abs(result.prediction_error) < 0.05

    def test_cognitive_cost_scales_with_k(self):
        for K in [4, 8, 16]:
            agent = LinearAgent(K=K, rng=np.random.default_rng(0))
            result = agent.update(0, 1.0)
            assert result.cost == float(K)


# ---------------------------------------------------------------------------
# NetworkAgent
# ---------------------------------------------------------------------------

class TestNetworkAgent:
    @pytest.fixture
    def simple_env(self):
        """2 options, 2 factors — minimal testable setup."""
        K, F = 4, 2
        rng = np.random.default_rng(42)
        L = rng.normal(0, 1, size=(K, F))
        L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
        return K, F, L

    def test_cross_option_generalization(self, simple_env):
        """Updating one action should change predicted values for other actions
        with correlated factor loadings. This is the core Network property."""
        K, F, L = simple_env
        agent = NetworkAgent(K=K, F=F, option_loadings=L, rng=np.random.default_rng(0))

        values_before = agent.values().copy()
        agent.update(0, 5.0)
        values_after = agent.values()

        # action 0 should have changed
        assert values_after[0] != values_before[0]
        # at least one other action should also have changed (shared factors)
        other_changed = np.any(values_after[1:] != values_before[1:])
        assert other_changed, "Network agent must generalize across options via shared factors"

    def test_no_generalization_with_orthogonal_loadings(self):
        """With perfectly orthogonal loadings (identity matrix), updating
        action 0 should not affect action 1's predicted value."""
        K, F = 2, 2
        L = np.eye(2)  # action 0 loads on factor 0 only, action 1 on factor 1 only
        agent = NetworkAgent(K=K, F=F, option_loadings=L, rng=np.random.default_rng(0))

        agent.update(0, 5.0)
        # action 1's value should be unchanged (zero, since f_hat[1] untouched)
        assert agent.values()[1] == 0.0

    def test_cognitive_cost_scales_with_f(self):
        """Base cost should be O(F), not O(K)."""
        for F in [2, 4, 8]:
            K = 16
            L = np.random.default_rng(0).normal(0, 1, size=(K, F))
            agent = NetworkAgent(
                K=K, F=F, option_loadings=L,
                structure_threshold=999.0,  # prevent structural update cost
                rng=np.random.default_rng(0),
            )
            result = agent.update(0, 0.1)
            assert result.cost == float(F)

    def test_structural_update_on_large_error(self):
        """Large prediction errors should trigger structural updates."""
        K, F = 4, 2
        L = np.ones((K, F)) * 0.5
        agent = NetworkAgent(
            K=K, F=F, option_loadings=L,
            structure_threshold=0.1,
            rng=np.random.default_rng(0),
        )
        agent.update(0, 10.0)  # large error from zero prediction
        assert agent.structural_updates >= 1

    def test_no_structural_update_on_small_error(self):
        """Small prediction errors should not trigger structural updates."""
        K, F = 4, 2
        L = np.ones((K, F)) * 0.5
        agent = NetworkAgent(
            K=K, F=F, option_loadings=L,
            structure_threshold=100.0,  # very high threshold
            rng=np.random.default_rng(0),
        )
        agent.update(0, 0.01)
        assert agent.structural_updates == 0

    def test_loadings_shape_validation(self):
        """Constructor should reject mismatched loadings."""
        with pytest.raises(AssertionError):
            NetworkAgent(K=4, F=2, option_loadings=np.zeros((3, 2)),
                         rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Agent ordering under shared structure (core paper prediction)
# ---------------------------------------------------------------------------

class TestAgentOrdering:
    def _run_comparison(self, K, F, T, n_seeds):
        """Run each agent on its own fresh env (same seed) so the noise
        stream is controlled — all agents face identical noise for the
        same action sequence."""
        rewards = {"dot": [], "linear": [], "network": []}
        for seed in range(n_seeds):
            # Need loadings from the env to construct NetworkAgent
            ref_env = FactorBanditEnv(EnvConfig(K=K, F=F, seed=seed))
            L = ref_env.L

            agents = {
                "dot": DotAgent(K=K, rng=np.random.default_rng(seed + 1)),
                "linear": LinearAgent(K=K, rng=np.random.default_rng(seed + 2)),
                "network": NetworkAgent(K=K, F=F, option_loadings=L,
                                        rng=np.random.default_rng(seed + 3)),
            }
            for name, agent in agents.items():
                env = FactorBanditEnv(EnvConfig(K=K, F=F, seed=seed))
                total = 0.0
                for _ in range(T):
                    a = agent.choose()
                    r = env.step(a)
                    agent.update(a, r)
                    total += r
                rewards[name].append(total / T)
        return {k: np.mean(v) for k, v in rewards.items()}

    def test_network_beats_linear(self):
        """Core paper prediction: Network outperforms Linear when K >> F
        because factor-based generalization exploits shared structure."""
        mean_r = self._run_comparison(K=16, F=2, T=300, n_seeds=5)
        assert mean_r["network"] > mean_r["linear"], (
            f"Network ({mean_r['network']:.3f}) should outperform "
            f"Linear ({mean_r['linear']:.3f}) under shared structure"
        )

    def test_linear_beats_dot_given_sufficient_horizon(self):
        """With enough trials for Q-learning to converge, Linear should
        outperform Dot. Uses fewer options and longer horizon to ensure
        convergence — the advantage is systematic value learning vs.
        reactive last-reward tracking."""
        mean_r = self._run_comparison(K=4, F=2, T=800, n_seeds=10)
        assert mean_r["linear"] > mean_r["dot"], (
            f"Linear ({mean_r['linear']:.3f}) should outperform "
            f"Dot ({mean_r['dot']:.3f}) given sufficient horizon"
        )
