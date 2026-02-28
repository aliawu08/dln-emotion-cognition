"""Tests for the profile-likelihood CI output across all datasets."""

import numpy as np
import pandas as pd
import pytest

from evidence_synthesis.analysis.run_profile_ci_all import (
    DATASETS,
    _desmedt_mod_matrix,
)
from evidence_synthesis.analysis.meta_pipeline import (
    design_matrix_stage,
    profile_likelihood_ci,
)


class TestProfileCIAllDatasets:
    """Verify profile-likelihood CIs satisfy basic invariants for every dataset."""

    @pytest.fixture(params=list(DATASETS.keys()))
    def dataset(self, request):
        name = request.param
        y, v, stages = DATASETS[name]()
        return name, y, v, stages

    def _moderator_matrix(self, name, stages):
        if name == "Desmedt":
            return _desmedt_mod_matrix(stages)
        X, _ = design_matrix_stage(stages, reference="dot")
        return X

    def test_baseline_ci_contains_point(self, dataset):
        name, y, v, _ = dataset
        X_base = np.ones((len(y), 1))
        pl = profile_likelihood_ci(y, v, X_base)
        assert pl.ci_lo <= pl.tau2_hat <= pl.ci_hi, (
            f"{name} baseline: CI [{pl.ci_lo}, {pl.ci_hi}] does not contain tau2={pl.tau2_hat}"
        )

    def test_moderator_ci_contains_point(self, dataset):
        name, y, v, stages = dataset
        X_mod = self._moderator_matrix(name, stages)
        pl = profile_likelihood_ci(y, v, X_mod)
        assert pl.ci_lo <= pl.tau2_hat <= pl.ci_hi, (
            f"{name} moderator: CI [{pl.ci_lo}, {pl.ci_hi}] does not contain tau2={pl.tau2_hat}"
        )

    def test_ci_lower_nonnegative(self, dataset):
        name, y, v, stages = dataset
        X_base = np.ones((len(y), 1))
        X_mod = self._moderator_matrix(name, stages)
        for label, X in [("baseline", X_base), ("moderator", X_mod)]:
            pl = profile_likelihood_ci(y, v, X)
            assert pl.ci_lo >= 0.0, f"{name} {label}: CI lower bound is negative"

    def test_moderator_tau2_leq_baseline(self, dataset):
        """Moderator model should have tau2 <= baseline tau2."""
        name, y, v, stages = dataset
        X_base = np.ones((len(y), 1))
        X_mod = self._moderator_matrix(name, stages)
        pl_base = profile_likelihood_ci(y, v, X_base)
        pl_mod = profile_likelihood_ci(y, v, X_mod)
        assert pl_mod.tau2_hat <= pl_base.tau2_hat + 1e-8, (
            f"{name}: moderator tau2 ({pl_mod.tau2_hat}) > baseline ({pl_base.tau2_hat})"
        )

    def test_ci_width_positive(self, dataset):
        name, y, v, _ = dataset
        X_base = np.ones((len(y), 1))
        pl = profile_likelihood_ci(y, v, X_base)
        assert pl.ci_hi > pl.ci_lo, (
            f"{name}: CI has zero width [{pl.ci_lo}, {pl.ci_hi}]"
        )


class TestDesmedtModMatrix:
    def test_shape(self):
        stages = pd.Series(["dot", "dot", "dot", "linear", "linear"])
        X = _desmedt_mod_matrix(stages)
        assert X.shape == (5, 2)

    def test_intercept_column(self):
        stages = pd.Series(["dot", "linear", "dot"])
        X = _desmedt_mod_matrix(stages)
        np.testing.assert_array_equal(X[:, 0], [1, 1, 1])

    def test_dot_indicator(self):
        stages = pd.Series(["dot", "linear", "dot", "linear"])
        X = _desmedt_mod_matrix(stages)
        np.testing.assert_array_equal(X[:, 1], [1, 0, 1, 0])
