"""Tests for the REML meta-analysis pipeline.

Validates the hand-rolled REML implementation against known analytical
properties and expected statistical behavior.
"""

import numpy as np
import pandas as pd
import pytest

from evidence_synthesis.analysis.meta_pipeline import (
    MetaResult,
    fit_reml,
    design_matrix_stage,
    summarize_meta,
    summarize_meta_with_stage,
    fisher_z_to_r,
    results_to_frame,
    egger_test,
    EggerResult,
    leave_one_out,
    LeaveOneOutResult,
    profile_likelihood_ci,
    ProfileLikelihoodCI,
    compute_qm,
    QMResult,
    forest_plot,
)


# ---------------------------------------------------------------------------
# fit_reml: basic correctness
# ---------------------------------------------------------------------------

class TestFitReml:
    def test_homogeneous_data_tau2_near_zero(self):
        """Identical effect sizes => tau^2 should be ~0."""
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((5, 1))
        result = fit_reml(y, v, X)
        assert result.tau2 < 0.01, f"tau2={result.tau2} should be near zero for homogeneous data"

    def test_heterogeneous_data_tau2_positive(self):
        """Widely dispersed effect sizes => tau^2 should be substantially positive."""
        y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((5, 1))
        result = fit_reml(y, v, X)
        assert result.tau2 > 0.1, f"tau2={result.tau2} should be positive for heterogeneous data"

    def test_recovers_known_mean(self):
        """With low variance and tight clustering, beta should recover the mean."""
        true_mu = 0.3
        y = np.array([0.29, 0.30, 0.31, 0.30, 0.29])
        v = np.array([0.001, 0.001, 0.001, 0.001, 0.001])
        X = np.ones((5, 1))
        result = fit_reml(y, v, X)
        assert abs(result.beta[0] - true_mu) < 0.02

    def test_ci_contains_estimate(self):
        """The 95% CI should contain the point estimate."""
        y = np.array([0.1, 0.3, 0.2, 0.4, 0.25])
        v = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        X = np.ones((5, 1))
        result = fit_reml(y, v, X)
        assert result.ci95[0, 0] <= result.beta[0] <= result.ci95[0, 1]

    def test_i2_bounds(self):
        """I^2 must always be in [0, 1]."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            k = rng.integers(3, 15)
            y = rng.normal(0, 1, size=k)
            v = rng.uniform(0.01, 0.5, size=k)
            X = np.ones((k, 1))
            result = fit_reml(y, v, X)
            assert 0.0 <= result.I2 <= 1.0, f"I2={result.I2} out of bounds"

    def test_tau2_upper_bound_not_clipped(self):
        """Verify the REML bound is wide enough for high-heterogeneity data.
        This was a known bug (previously capped at 1.0)."""
        y = np.array([-3.0, -1.0, 0.0, 1.5, 3.5])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((5, 1))
        result = fit_reml(y, v, X)
        assert result.tau2 > 1.0, (
            f"tau2={result.tau2} should exceed 1.0 for highly heterogeneous data"
        )

    def test_k_and_p_reported_correctly(self):
        y = np.array([0.1, 0.2, 0.3])
        v = np.array([0.01, 0.01, 0.01])
        X = np.ones((3, 1))
        result = fit_reml(y, v, X)
        assert result.k == 3
        assert result.p == 1

    def test_ci_coverage_large_k(self):
        """With k=30 studies, Wald-type 95% CIs should achieve near-nominal coverage."""
        rng = np.random.default_rng(12345)
        true_mu, true_tau2, k = 0.3, 0.05, 30
        hits = 0
        n_sim = 400
        for _ in range(n_sim):
            theta_i = rng.normal(true_mu, np.sqrt(true_tau2), size=k)
            vi = rng.uniform(0.02, 0.08, size=k)
            yi = theta_i + rng.normal(0, np.sqrt(vi))
            res = fit_reml(yi, vi, np.ones((k, 1)))
            if res.ci95[0, 0] <= true_mu <= res.ci95[0, 1]:
                hits += 1
        coverage = hits / n_sim
        assert 0.92 <= coverage <= 0.98, (
            f"Coverage={coverage:.3f}; Wald CIs should be near-nominal with k={k}"
        )

    def test_ci_coverage_small_k(self):
        """With k=5 and the Knapp-Hartung correction (t-based CIs),
        coverage should be near-nominal even for small meta-analyses."""
        rng = np.random.default_rng(54321)
        true_mu, true_tau2, k = 0.3, 0.05, 5
        hits = 0
        n_sim = 400
        for _ in range(n_sim):
            theta_i = rng.normal(true_mu, np.sqrt(true_tau2), size=k)
            vi = rng.uniform(0.02, 0.08, size=k)
            yi = theta_i + rng.normal(0, np.sqrt(vi))
            res = fit_reml(yi, vi, np.ones((k, 1)))
            if res.ci95[0, 0] <= true_mu <= res.ci95[0, 1]:
                hits += 1
        coverage = hits / n_sim
        assert 0.92 <= coverage <= 0.99, (
            f"Coverage={coverage:.3f}; Knapp-Hartung should achieve near-nominal "
            f"coverage even with k={k}"
        )

    def test_with_moderator_reduces_heterogeneity(self):
        """When a moderator truly explains variance, moderated tau^2 < baseline tau^2."""
        rng = np.random.default_rng(10)
        # Two groups with distinct true means
        y_group1 = rng.normal(0.0, 0.05, size=10)
        y_group2 = rng.normal(1.0, 0.05, size=10)
        y = np.concatenate([y_group1, y_group2])
        v = np.full(20, 0.01)

        # Baseline: intercept only
        X_base = np.ones((20, 1))
        base_result = fit_reml(y, v, X_base)

        # Moderated: intercept + group dummy
        X_mod = np.hstack([np.ones((20, 1)),
                           np.concatenate([np.zeros(10), np.ones(10)]).reshape(-1, 1)])
        mod_result = fit_reml(y, v, X_mod)

        assert mod_result.tau2 < base_result.tau2, (
            f"Moderated tau2 ({mod_result.tau2:.4f}) should be less than "
            f"baseline tau2 ({base_result.tau2:.4f})"
        )


# ---------------------------------------------------------------------------
# design_matrix_stage
# ---------------------------------------------------------------------------

class TestDesignMatrix:
    def test_shape_and_names(self):
        stage = pd.Series(["dot", "linear", "network", "dot", "network"])
        X, names = design_matrix_stage(stage, reference="dot")
        assert X.shape == (5, 3)
        assert names == ["Intercept", "stage[linear]", "stage[network]"]

    def test_reference_level_is_zero(self):
        """Reference level should have zeros in all dummy columns."""
        stage = pd.Series(["dot", "linear", "network"])
        X, _ = design_matrix_stage(stage, reference="dot")
        # Row 0 = "dot" (reference): dummies should be [1, 0, 0]
        np.testing.assert_array_equal(X[0], [1.0, 0.0, 0.0])

    def test_dummy_coding_correct(self):
        stage = pd.Series(["dot", "linear", "network"])
        X, _ = design_matrix_stage(stage, reference="dot")
        np.testing.assert_array_equal(X[1], [1.0, 1.0, 0.0])  # linear
        np.testing.assert_array_equal(X[2], [1.0, 0.0, 1.0])  # network

    def test_intercept_column_all_ones(self):
        stage = pd.Series(["dot", "linear", "network", "dot"])
        X, _ = design_matrix_stage(stage, reference="dot")
        np.testing.assert_array_equal(X[:, 0], np.ones(4))

    def test_alternative_reference(self):
        stage = pd.Series(["dot", "linear", "network"])
        X, names = design_matrix_stage(stage, reference="linear")
        assert names == ["Intercept", "stage[dot]", "stage[network]"]
        # "linear" row should be [1, 0, 0]
        np.testing.assert_array_equal(X[1], [1.0, 0.0, 0.0])

    def test_invalid_reference_raises(self):
        with pytest.raises(ValueError, match="reference must be one of"):
            design_matrix_stage(pd.Series(["dot"]), reference="invalid")

    def test_missing_level_produces_zero_column(self):
        """If a level has no observations, its dummy column should be all zeros."""
        stage = pd.Series(["dot", "dot", "linear"])
        X, names = design_matrix_stage(stage, reference="dot")
        network_col = names.index("stage[network]")
        assert np.all(X[:, network_col] == 0.0)


# ---------------------------------------------------------------------------
# fisher_z_to_r
# ---------------------------------------------------------------------------

class TestFisherZ:
    def test_zero_maps_to_zero(self):
        assert fisher_z_to_r(np.array([0.0]))[0] == 0.0

    def test_output_bounded(self):
        z = np.array([-100, -1, 0, 1, 100])
        r = fisher_z_to_r(z)
        assert np.all(np.abs(r) <= 1.0)

    def test_moderate_values_match_tanh(self):
        z = np.array([0.1, 0.5, 1.0])
        np.testing.assert_allclose(fisher_z_to_r(z), np.tanh(z))


# ---------------------------------------------------------------------------
# summarize_meta / summarize_meta_with_stage / results_to_frame
# ---------------------------------------------------------------------------

class TestSummarize:
    @pytest.fixture
    def sample_df(self):
        """Minimal synthetic dataset with two paradigms and three stages."""
        rng = np.random.default_rng(55)
        rows = []
        for paradigm in ["paradigm_A", "paradigm_B"]:
            for stage in ["dot", "linear", "network"]:
                for _ in range(5):
                    rows.append({
                        "paradigm_family": paradigm,
                        "dln_stage_code": stage,
                        "yi": rng.normal(0.3 if stage == "network" else 0.0, 0.1),
                        "vi": 0.02,
                    })
        return pd.DataFrame(rows)

    def test_summarize_meta_returns_dict(self, sample_df):
        result = summarize_meta(sample_df)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"paradigm_A", "paradigm_B"}
        for v in result.values():
            assert isinstance(v, MetaResult)

    def test_summarize_meta_with_stage_returns_dict(self, sample_df):
        result = summarize_meta_with_stage(sample_df)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(v, tuple)
            assert isinstance(v[0], MetaResult)
            assert isinstance(v[1], list)

    def test_results_to_frame(self, sample_df):
        base = summarize_meta(sample_df)
        mod = summarize_meta_with_stage(sample_df)
        df = results_to_frame(base, mod)
        assert isinstance(df, pd.DataFrame)
        assert "paradigm_family" in df.columns
        assert "tau2_base" in df.columns
        assert "delta_tau2" in df.columns
        assert len(df) == 2  # two paradigms


# ---------------------------------------------------------------------------
# egger_test
# ---------------------------------------------------------------------------

class TestEggerTest:
    def test_returns_egger_result(self):
        y = np.array([0.1, 0.3, 0.2, 0.4, 0.25, 0.15, 0.35])
        v = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        result = egger_test(y, v)
        assert isinstance(result, EggerResult)
        assert result.k == 7

    def test_equal_variances_returns_nan(self):
        """When all variances are identical, Egger's test is undefined."""
        y = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        result = egger_test(y, v)
        assert np.isnan(result.p_value)

    def test_symmetric_data_nonsignificant(self):
        """Symmetric effects with no size-effect correlation should not flag bias."""
        # Large and small studies have similar mean effects (no asymmetry)
        y = np.array([0.20, 0.22, 0.18, 0.21, 0.19, 0.20])
        v = np.array([0.01, 0.05, 0.01, 0.05, 0.01, 0.05])
        result = egger_test(y, v)
        assert result.p_value > 0.05

    def test_asymmetric_data_lower_p(self):
        """Deliberately asymmetric: small studies have large positive effects."""
        y = np.array([0.1, 0.1, 0.1, 0.5, 0.8, 1.2])
        v = np.array([0.01, 0.01, 0.01, 0.10, 0.20, 0.50])
        result = egger_test(y, v)
        # Should at least have a positive intercept (small-study bias)
        assert result.intercept > 0


# ---------------------------------------------------------------------------
# leave_one_out
# ---------------------------------------------------------------------------

class TestLeaveOneOut:
    def test_returns_correct_shape(self):
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((5, 1))
        result = leave_one_out(y, v, X)
        assert isinstance(result, LeaveOneOutResult)
        assert len(result.tau2) == 5
        assert result.beta.shape == (5, 1)
        assert len(result.Q) == 5

    def test_dropping_outlier_reduces_tau2(self):
        """Dropping an extreme outlier should reduce heterogeneity."""
        y = np.array([0.1, 0.1, 0.1, 0.1, 2.0])  # study 4 is outlier
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((5, 1))
        full_res = fit_reml(y, v, X)
        loo = leave_one_out(y, v, X)
        # Dropping the outlier (index 4) should reduce tau2
        assert loo.tau2[4] < full_res.tau2

    def test_indices_complete(self):
        y = np.array([0.1, 0.2, 0.3])
        v = np.array([0.01, 0.01, 0.01])
        X = np.ones((3, 1))
        result = leave_one_out(y, v, X)
        assert result.indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# profile_likelihood_ci
# ---------------------------------------------------------------------------

class TestProfileLikelihoodCI:
    def test_returns_correct_type(self):
        y = np.array([0.1, 0.3, 0.2, 0.5, 0.15, 0.35])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((6, 1))
        result = profile_likelihood_ci(y, v, X)
        assert isinstance(result, ProfileLikelihoodCI)

    def test_ci_contains_point_estimate(self):
        y = np.array([0.1, 0.3, 0.2, 0.5, 0.15, 0.35])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((6, 1))
        result = profile_likelihood_ci(y, v, X)
        assert result.ci_lo <= result.tau2_hat <= result.ci_hi

    def test_ci_lower_bound_nonnegative(self):
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((5, 1))
        result = profile_likelihood_ci(y, v, X)
        assert result.ci_lo >= 0.0

    def test_homogeneous_data_ci_includes_zero(self):
        """For homogeneous data, tau2 CI should include zero."""
        y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        v = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        X = np.ones((5, 1))
        result = profile_likelihood_ci(y, v, X)
        assert result.ci_lo == 0.0


# ---------------------------------------------------------------------------
# compute_qm
# ---------------------------------------------------------------------------

class TestComputeQM:
    def test_returns_qm_result(self):
        rng = np.random.default_rng(42)
        y = np.concatenate([rng.normal(0.0, 0.1, 10), rng.normal(1.0, 0.1, 10)])
        v = np.full(20, 0.01)
        X_base = np.ones((20, 1))
        X_mod = np.hstack([np.ones((20, 1)),
                           np.concatenate([np.zeros(10), np.ones(10)]).reshape(-1, 1)])
        result = compute_qm(y, v, X_base, X_mod)
        assert isinstance(result, QMResult)

    def test_df_equals_parameter_difference(self):
        """df should equal the number of extra parameters in the moderator model."""
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        v = np.full(6, 0.01)
        X_base = np.ones((6, 1))
        # Two dummy columns (3 groups)
        X_mod = np.hstack([np.ones((6, 1)),
                           np.array([1, 1, 0, 0, 0, 0], dtype=float).reshape(-1, 1),
                           np.array([0, 0, 0, 0, 1, 1], dtype=float).reshape(-1, 1)])
        result = compute_qm(y, v, X_base, X_mod)
        assert result.df == 2

    def test_p_in_valid_range(self):
        rng = np.random.default_rng(99)
        y = rng.normal(0, 1, size=15)
        v = rng.uniform(0.01, 0.1, size=15)
        X_base = np.ones((15, 1))
        X_mod = np.hstack([np.ones((15, 1)),
                           rng.choice([0.0, 1.0], size=(15, 1))])
        result = compute_qm(y, v, X_base, X_mod)
        assert 0.0 <= result.p <= 1.0

    def test_significant_when_true_moderator(self):
        """When groups have clearly different means, QM should be significant."""
        rng = np.random.default_rng(10)
        y = np.concatenate([rng.normal(0.0, 0.05, 10), rng.normal(1.0, 0.05, 10)])
        v = np.full(20, 0.01)
        X_base = np.ones((20, 1))
        X_mod = np.hstack([np.ones((20, 1)),
                           np.concatenate([np.zeros(10), np.ones(10)]).reshape(-1, 1)])
        result = compute_qm(y, v, X_base, X_mod)
        assert result.QM > 0
        assert result.p < 0.05

    def test_nonsignificant_when_no_moderator(self):
        """When all effects come from the same distribution, QM should be small."""
        rng = np.random.default_rng(77)
        y = rng.normal(0.3, 0.1, size=20)
        v = np.full(20, 0.05)
        X_base = np.ones((20, 1))
        # Arbitrary grouping that should not explain variance
        group = np.array([0]*10 + [1]*10, dtype=float).reshape(-1, 1)
        X_mod = np.hstack([np.ones((20, 1)), group])
        result = compute_qm(y, v, X_base, X_mod)
        assert result.p > 0.05

    def test_qm_nonnegative(self):
        """QM (Q_base - Q_residual) should always be non-negative."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            k = rng.integers(6, 20)
            y = rng.normal(0, 0.5, size=k)
            v = rng.uniform(0.01, 0.1, size=k)
            X_base = np.ones((k, 1))
            group = rng.choice([0.0, 1.0], size=k)
            if len(set(group)) < 2:
                group[0] = 1.0 - group[0]
            X_mod = np.hstack([np.ones((k, 1)), group.reshape(-1, 1)])
            result = compute_qm(y, v, X_base, X_mod)
            assert result.QM >= -1e-10, f"QM={result.QM} should be non-negative"


# ---------------------------------------------------------------------------
# forest_plot
# ---------------------------------------------------------------------------

class TestForestPlot:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        labels = ["Study A", "Study B", "Study C"]
        y = np.array([0.1, 0.3, 0.2])
        v = np.array([0.01, 0.02, 0.015])
        fig = forest_plot(labels, y, v)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_stage_coloring(self):
        import matplotlib.pyplot as plt
        labels = ["S1", "S2", "S3"]
        y = np.array([0.1, 0.3, 0.2])
        v = np.array([0.01, 0.02, 0.015])
        stage = pd.Series(["dot", "linear", "network"])
        fig = forest_plot(labels, y, v, stage=stage)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_pooled_diamond(self):
        import matplotlib.pyplot as plt
        labels = ["S1", "S2", "S3"]
        y = np.array([0.1, 0.3, 0.2])
        v = np.array([0.01, 0.02, 0.015])
        fig = forest_plot(labels, y, v, pooled_beta=0.2, pooled_ci=(0.1, 0.3))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
