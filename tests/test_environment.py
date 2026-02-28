"""Tests for the factor bandit environment.

Validates that the generative model produces rewards consistent with the
specified latent factor structure: reward = L[action] @ s + noise.
"""

import numpy as np
import pytest

from computational.environment import EnvConfig, FactorBanditEnv


class TestFactorBanditEnv:
    def test_reward_mean_matches_generative_model(self):
        """Empirical mean reward for an action should converge to L[action] @ s."""
        cfg = EnvConfig(K=4, F=2, noise_sd=0.1, seed=42)
        env = FactorBanditEnv(cfg)

        action = 0
        expected_mean = float(env.L[action] @ env.s)

        rewards = [env.step(action) for _ in range(5000)]
        empirical_mean = np.mean(rewards)

        np.testing.assert_allclose(empirical_mean, expected_mean, atol=0.02,
                                   err_msg="Empirical reward mean should match L[a] @ s")

    def test_reward_noise_calibration(self):
        """Empirical reward SD should match cfg.noise_sd."""
        cfg = EnvConfig(K=4, F=2, noise_sd=0.15, seed=7)
        env = FactorBanditEnv(cfg)

        rewards = [env.step(0) for _ in range(5000)]
        empirical_sd = np.std(rewards)

        np.testing.assert_allclose(empirical_sd, cfg.noise_sd, atol=0.02,
                                   err_msg="Reward SD should match noise_sd")

    def test_loadings_normalized(self):
        """Each row of L should have unit norm (loadings are normalized)."""
        cfg = EnvConfig(K=8, F=3, seed=0)
        env = FactorBanditEnv(cfg)

        norms = np.linalg.norm(env.L, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6,
                                   err_msg="Loading vectors should be unit-normalized")

    def test_dimensions(self):
        """L shape should be (K, F), s shape should be (F,)."""
        cfg = EnvConfig(K=6, F=3, seed=0)
        env = FactorBanditEnv(cfg)
        assert env.L.shape == (6, 3)
        assert env.s.shape == (3,)

    def test_different_seeds_different_environments(self):
        """Different seeds should produce different latent structures."""
        env1 = FactorBanditEnv(EnvConfig(K=4, F=2, seed=0))
        env2 = FactorBanditEnv(EnvConfig(K=4, F=2, seed=99))
        assert not np.allclose(env1.L, env2.L)
        assert not np.allclose(env1.s, env2.s)

    def test_same_seed_reproducible(self):
        """Same seed should produce identical environments."""
        env1 = FactorBanditEnv(EnvConfig(K=4, F=2, seed=42))
        env2 = FactorBanditEnv(EnvConfig(K=4, F=2, seed=42))
        np.testing.assert_array_equal(env1.L, env2.L)
        np.testing.assert_array_equal(env1.s, env2.s)

    def test_actions_produce_different_means(self):
        """Different actions should (generally) have different expected rewards."""
        cfg = EnvConfig(K=8, F=2, seed=0)
        env = FactorBanditEnv(cfg)
        means = env.L @ env.s
        # With 8 options and 2 factors, not all means should be identical
        assert np.std(means) > 0.01, "Actions should have distinguishable mean rewards"
