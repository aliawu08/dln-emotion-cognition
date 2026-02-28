"""Numerical robustness tests.

Checks edge cases that could cause silent failures or NaN propagation
in the computational and meta-analytic pipelines.
"""

import numpy as np
import pytest

from computational.agents import softmax, DotAgent, LinearAgent, NetworkAgent
from evidence_synthesis.analysis.meta_pipeline import fit_reml, design_matrix_stage


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        for x in [np.array([1, 2, 3]), np.array([0, 0, 0]), np.array([-1, 0, 1])]:
            np.testing.assert_allclose(softmax(x).sum(), 1.0, atol=1e-10)

    def test_all_non_negative(self):
        result = softmax(np.array([-100, 0, 100]))
        assert np.all(result >= 0.0)

    def test_large_positive_values_no_overflow(self):
        result = softmax(np.array([1000, 1001, 1002]))
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_large_negative_values(self):
        result = softmax(np.array([-1000, -999, -998]))
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_zero_temperature(self):
        """temperature=0 should produce valid probabilities (argmax-like)."""
        result = softmax(np.array([1, 2, 3]), temperature=0.0)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)
        assert result[2] > 0.99  # highest input dominates

    def test_high_temperature_approaches_uniform(self):
        result = softmax(np.array([1, 2, 3]), temperature=100.0)
        np.testing.assert_allclose(result, 1 / 3, atol=0.01)

    def test_single_element(self):
        result = softmax(np.array([42.0]))
        np.testing.assert_allclose(result, [1.0])


# ---------------------------------------------------------------------------
# REML edge cases
# ---------------------------------------------------------------------------

class TestRemlEdgeCases:
    def test_two_studies_minimum(self):
        """Minimum practical case: 2 studies should not crash."""
        y = np.array([0.1, 0.5])
        v = np.array([0.05, 0.05])
        X = np.ones((2, 1))
        result = fit_reml(y, v, X)
        assert np.isfinite(result.tau2)
        assert np.all(np.isfinite(result.beta))

    def test_single_study_does_not_crash(self):
        """k=1 is not identifiable but should not raise."""
        y = np.array([0.3])
        v = np.array([0.05])
        X = np.ones((1, 1))
        result = fit_reml(y, v, X)
        assert np.isfinite(result.beta[0])

    def test_very_small_sampling_variance(self):
        """Near-zero (but positive) variance should work."""
        y = np.array([0.1, 0.2, 0.3])
        v = np.array([1e-8, 1e-8, 1e-8])
        X = np.ones((3, 1))
        result = fit_reml(y, v, X)
        assert np.isfinite(result.tau2)
        assert np.all(np.isfinite(result.beta))

    def test_large_k(self):
        """50 studies should work without numerical issues."""
        rng = np.random.default_rng(0)
        k = 50
        y = rng.normal(0, 1, size=k)
        v = rng.uniform(0.01, 0.1, size=k)
        X = np.ones((k, 1))
        result = fit_reml(y, v, X)
        assert result.k == 50
        assert np.isfinite(result.tau2)

    def test_all_effects_identical_positive(self):
        """All identical positive effects."""
        y = np.full(10, 0.8)
        v = np.full(10, 0.02)
        X = np.ones((10, 1))
        result = fit_reml(y, v, X)
        assert abs(result.beta[0] - 0.8) < 0.01
        assert result.tau2 < 0.01

    def test_q_non_negative(self):
        """Cochran's Q should always be non-negative."""
        rng = np.random.default_rng(77)
        for _ in range(20):
            k = rng.integers(3, 15)
            y = rng.normal(0, 0.5, size=k)
            v = rng.uniform(0.01, 0.3, size=k)
            X = np.ones((k, 1))
            result = fit_reml(y, v, X)
            assert result.Q >= 0.0, f"Q={result.Q} should be non-negative"


# ---------------------------------------------------------------------------
# Agent numerical stability
# ---------------------------------------------------------------------------

class TestAgentNumerics:
    def test_dot_large_reward(self):
        agent = DotAgent(K=3, rng=np.random.default_rng(0))
        result = agent.update(0, 1e6)
        assert np.isfinite(result.reward)
        # Should still be able to choose without error
        a = agent.choose()
        assert 0 <= a < 3

    def test_linear_large_reward(self):
        agent = LinearAgent(K=3, rng=np.random.default_rng(0))
        result = agent.update(0, 1e6)
        assert np.isfinite(result.prediction_error)
        a = agent.choose()
        assert 0 <= a < 3

    def test_network_large_reward(self):
        L = np.eye(3)
        agent = NetworkAgent(K=3, F=3, option_loadings=L,
                             rng=np.random.default_rng(0))
        result = agent.update(0, 1e6)
        assert np.isfinite(result.prediction_error)
        assert np.all(np.isfinite(agent.f_hat))
        a = agent.choose()
        assert 0 <= a < 3

    def test_network_many_updates_stable(self):
        """1000 updates should not produce NaN or Inf in factor beliefs."""
        rng_env = np.random.default_rng(0)
        K, F = 8, 3
        L = rng_env.normal(0, 1, size=(K, F))
        L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
        agent = NetworkAgent(K=K, F=F, option_loadings=L,
                             rng=np.random.default_rng(1))
        rng_reward = np.random.default_rng(2)
        for _ in range(1000):
            a = agent.choose()
            r = rng_reward.normal(0, 1)
            agent.update(a, r)
        assert np.all(np.isfinite(agent.f_hat)), "Factor beliefs diverged after 1000 updates"
        assert np.all(np.isfinite(agent.values())), "Predicted values diverged after 1000 updates"
