"""Tests for the LOO cross-validation analysis."""

import numpy as np
import pytest

from evidence_synthesis.analysis.loo_cross_validation import (
    loo_cross_validate,
    loo_r2,
    loo_mae,
    simulate_null_loo,
    run_one_dataset,
    DATASETS,
)
from evidence_synthesis.analysis.meta_pipeline import fit_reml


class TestLOOCrossValidate:
    def test_returns_correct_shape(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        v = np.full(5, 0.01)
        X = np.ones((5, 1))
        y_hat = loo_cross_validate(y, v, X)
        assert y_hat.shape == (5,)

    def test_predictions_are_finite(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        v = np.full(5, 0.01)
        X = np.ones((5, 1))
        y_hat = loo_cross_validate(y, v, X)
        assert np.all(np.isfinite(y_hat))

    def test_moderator_with_clear_groups(self):
        """LOO predictions for a clear 2-group structure should be close to
        within-group means."""
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        v = np.full(6, 0.001)
        group = np.array([0, 0, 0, 1, 1, 1])
        X = np.column_stack([np.ones(6), group.astype(float)])
        y_hat = loo_cross_validate(y, v, X)
        # Each held-out unit should be predicted near its group mean
        for i in range(3):
            assert abs(y_hat[i] - 0.0) < 0.05
        for i in range(3, 6):
            assert abs(y_hat[i] - 1.0) < 0.05

    def test_synthetic_recovers_known_r2(self):
        """For a perfect 3-group structure with negligible noise, LOO R2 should
        be close to 1.0."""
        y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        v = np.full(9, 0.001)
        group = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        X = np.column_stack([
            np.ones(9),
            (group == 1).astype(float),
            (group == 2).astype(float),
        ])
        y_hat = loo_cross_validate(y, v, X)
        r2 = loo_r2(y, y_hat)
        assert r2 > 0.90, f"LOO R2={r2:.3f} should be > 0.90 for clean data"


class TestLOOMetrics:
    def test_r2_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([1.0, 2.0, 3.0])
        assert abs(loo_r2(y, y_hat) - 1.0) < 1e-10

    def test_r2_mean_predictions(self):
        """Predicting the mean should give R2 = 0."""
        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.full(3, 2.0)
        assert abs(loo_r2(y, y_hat)) < 1e-10

    def test_r2_can_be_negative(self):
        """Worse-than-mean predictions give negative R2."""
        y = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([3.0, 1.0, 2.0])
        assert loo_r2(y, y_hat) < 0.0

    def test_mae_nonnegative(self):
        y = np.array([0.1, 0.2, 0.3])
        y_hat = np.array([0.15, 0.25, 0.25])
        assert loo_mae(y, y_hat) >= 0.0

    def test_mae_zero_for_perfect(self):
        y = np.array([1.0, 2.0])
        y_hat = np.array([1.0, 2.0])
        assert abs(loo_mae(y, y_hat)) < 1e-10


class TestRealDatasets:
    """Run LOO on all four real datasets and verify basic properties."""

    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_predictions_finite(self, name):
        result = run_one_dataset(name)
        assert np.all(np.isfinite(result.y_hat)), (
            f"{name}: LOO predictions contain non-finite values"
        )

    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_dln_beats_null(self, name):
        """DLN moderator LOO-R2 should exceed null LOO-R2 for all datasets."""
        result = run_one_dataset(name)
        assert result.loo_r2 > result.loo_r2_null, (
            f"{name}: DLN LOO-R2 ({result.loo_r2:.3f}) should exceed "
            f"null LOO-R2 ({result.loo_r2_null:.3f})"
        )

    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_dln_r2_positive(self, name):
        """DLN moderator LOO-R2 should be positive (better than mean)."""
        result = run_one_dataset(name)
        assert result.loo_r2 > 0.0, (
            f"{name}: DLN LOO-R2 ({result.loo_r2:.3f}) should be positive"
        )

    @pytest.mark.parametrize("name", list(DATASETS.keys()))
    def test_mae_improves_over_null(self, name):
        """DLN MAE should be lower than null MAE."""
        result = run_one_dataset(name)
        assert result.loo_mae < result.loo_mae_null, (
            f"{name}: DLN MAE ({result.loo_mae:.4f}) should be less than "
            f"null MAE ({result.loo_mae_null:.4f})"
        )


class TestSimulateNullLOO:
    """Tests for the null-model LOO simulation."""

    def test_returns_correct_shape(self):
        y = np.array([0.0, 0.1, 0.5, 0.6, 1.0, 1.1])
        v = np.full(6, 0.01)
        null_r2 = simulate_null_loo(y, v, n_groups=2, n_perms=50, seed=0)
        assert null_r2.shape == (50,)

    def test_all_values_finite(self):
        y = np.array([0.0, 0.1, 0.5, 0.6, 1.0, 1.1])
        v = np.full(6, 0.01)
        null_r2 = simulate_null_loo(y, v, n_groups=2, n_perms=50, seed=0)
        assert np.all(np.isfinite(null_r2))

    def test_null_median_below_signal(self):
        """For data with a clear 3-group structure, random partitions should
        produce lower LOO R² than the true partition on average."""
        y = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        v = np.full(9, 0.001)
        null_r2 = simulate_null_loo(y, v, n_groups=3, n_perms=200, seed=42)
        # True partition LOO R² should be near 1.0; null median should be much lower
        assert np.median(null_r2) < 0.8

    def test_reproducible_with_same_seed(self):
        y = np.array([0.1, 0.3, 0.5, 0.7])
        v = np.full(4, 0.01)
        r1 = simulate_null_loo(y, v, n_groups=2, n_perms=20, seed=99)
        r2 = simulate_null_loo(y, v, n_groups=2, n_perms=20, seed=99)
        np.testing.assert_array_equal(r1, r2)
