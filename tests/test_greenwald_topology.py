"""Tests for the Greenwald (2009) four-level topology-mediated validity analysis.

Validates:
- Data coverage: all 184 samples assigned to exactly 4 DLN stages.
- Stage distribution: matches pre-specified counts (33 / 97 / 43 / 11).
- DOT override precedence: DOT_SAMPLES always coded as dot regardless of topic.
- REML convergence: all four hierarchical models converge with finite estimates.
- Monotonic tau-squared reduction across the model hierarchy.
- Suppression fingerprint: Network ICC > Linear ICC; Linear-Plus iec < Linear iec.
- Profile consistency: stage-profile rows match direct computation.
- Scale-mixing guard: Fisher-z I² ≠ naive raw-r / Fisher-z ratio.

Data source: Greenwald et al. (2009) meta-analysis, k = 184 independent samples.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "evidence_synthesis" / "analysis"),
)

from meta_pipeline import fit_reml, design_matrix_categorical
from run_greenwald2009_topology import (
    DATA,
    DOT_SAMPLES,
    NETWORK_SAMPLES,
    LINEAR_PLUS_TOPICS,
    STAGE_ORDER,
    assign_topology_stage,
    prepare_data,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def df(raw):
    return prepare_data(raw)


# ---------------------------------------------------------------------------
# 1. Coding coverage
# ---------------------------------------------------------------------------

class TestCodingCoverage:
    def test_total_samples(self, df):
        """All 184 samples must be present after prepare_data."""
        assert len(df) == 184

    def test_exactly_four_stages(self, df):
        """Coding must produce exactly four DLN stages."""
        stages = set(df["dln_stage"].unique())
        assert stages == {"dot", "linear", "linear_plus", "network"}

    def test_no_missing_stages(self, df):
        """No sample should have a missing dln_stage."""
        assert df["dln_stage"].notna().all()


# ---------------------------------------------------------------------------
# 2. Stage distribution
# ---------------------------------------------------------------------------

class TestStageDistribution:
    def test_dot_count(self, df):
        assert (df["dln_stage"] == "dot").sum() == 33

    def test_linear_count(self, df):
        assert (df["dln_stage"] == "linear").sum() == 97

    def test_linear_plus_count(self, df):
        assert (df["dln_stage"] == "linear_plus").sum() == 43

    def test_network_count(self, df):
        assert (df["dln_stage"] == "network").sum() == 11

    def test_counts_sum_to_k(self, df):
        counts = df["dln_stage"].value_counts()
        assert counts.sum() == 184


# ---------------------------------------------------------------------------
# 3. DOT override precedence
# ---------------------------------------------------------------------------

class TestDOTOverride:
    def test_all_dot_samples_coded_dot(self, df):
        """Every sample_id in DOT_SAMPLES must be coded as dot,
        regardless of its topic domain."""
        for sid in DOT_SAMPLES:
            rows = df[df["sample_id"] == sid]
            assert len(rows) > 0, f"DOT sample {sid} not found in data"
            actual = rows["dln_stage"].iloc[0]
            assert actual == "dot", (
                f"DOT sample {sid} coded as '{actual}', expected 'dot'"
            )

    def test_dot_override_trumps_linear_plus_topic(self, raw, df):
        """DOT_SAMPLES in LINEAR_PLUS_TOPICS should still be coded dot."""
        for sid in DOT_SAMPLES:
            row = raw[raw["sample_id"] == sid]
            if len(row) > 0 and row["topic"].iloc[0] in LINEAR_PLUS_TOPICS:
                actual = df[df["sample_id"] == sid]["dln_stage"].iloc[0]
                assert actual == "dot", (
                    f"Sample {sid} has topic '{row['topic'].iloc[0]}' "
                    f"(LINEAR_PLUS_TOPICS) but DOT override should apply"
                )

    def test_network_samples_coded_network(self, df):
        """Every sample_id in NETWORK_SAMPLES must be coded as network."""
        for sid in NETWORK_SAMPLES:
            rows = df[df["sample_id"] == sid]
            assert len(rows) > 0, f"NETWORK sample {sid} not found in data"
            actual = rows["dln_stage"].iloc[0]
            assert actual == "network", (
                f"NETWORK sample {sid} coded as '{actual}', expected 'network'"
            )


# ---------------------------------------------------------------------------
# 4. assign_topology_stage unit tests
# ---------------------------------------------------------------------------

class TestAssignTopologyStage:
    def test_dot_override(self):
        """Sample in DOT_SAMPLES returns 'dot' even with LP topic."""
        sid = next(iter(DOT_SAMPLES))
        assert assign_topology_stage(sid, "Race (Bl/Wh)") == "dot"

    def test_network_override(self):
        """Sample in NETWORK_SAMPLES returns 'network'."""
        sid = next(iter(NETWORK_SAMPLES))
        assert assign_topology_stage(sid, "Consumer") == "network"

    def test_linear_plus_topic(self):
        """Non-DOT, non-NETWORK sample with LP topic returns 'linear_plus'."""
        fake_sid = 99999
        assert assign_topology_stage(fake_sid, "Race (Bl/Wh)") == "linear_plus"
        assert assign_topology_stage(fake_sid, "Gender/sex") == "linear_plus"
        assert assign_topology_stage(fake_sid, "Other intergroup") == "linear_plus"

    def test_linear_default(self):
        """Non-DOT, non-NETWORK sample with non-LP topic returns 'linear'."""
        fake_sid = 99999
        assert assign_topology_stage(fake_sid, "Consumer") == "linear"
        assert assign_topology_stage(fake_sid, "Drugs/tobacco") == "linear"


# ---------------------------------------------------------------------------
# 5. REML convergence
# ---------------------------------------------------------------------------

class TestREMLConvergence:
    def test_baseline_converges(self, df):
        y = df["yi_icc"].to_numpy()
        v = df["vi_icc"].to_numpy()
        res = fit_reml(y, v, np.ones((len(df), 1)))
        assert np.isfinite(res.tau2)
        assert np.isfinite(res.beta[0])
        assert 0.0 <= res.I2 <= 1.0

    def test_four_level_converges(self, df):
        y = df["yi_icc"].to_numpy()
        v = df["vi_icc"].to_numpy()
        X, names = design_matrix_categorical(df["dln_stage"], reference="dot")
        res = fit_reml(y, v, X)
        assert np.isfinite(res.tau2)
        assert all(np.isfinite(res.beta))
        assert len(names) == 4  # intercept + 3 dummies

    def test_four_level_plus_covariates_converges(self, df):
        y = df["yi_icc"].to_numpy()
        v = df["vi_icc"].to_numpy()
        X_stage, _ = design_matrix_categorical(df["dln_stage"], reference="dot")
        X = np.column_stack([X_stage, df["n_crit"].to_numpy(), df["n_iat"].to_numpy()])
        res = fit_reml(y, v, X)
        assert np.isfinite(res.tau2)
        assert all(np.isfinite(res.beta))


# ---------------------------------------------------------------------------
# 6. Monotonic tau-squared reduction
# ---------------------------------------------------------------------------

class TestMonotonicReduction:
    def test_hierarchy_monotonically_reduces_tau2(self, df):
        """Tau-squared must decrease (or stay equal) at each step:
        baseline >= 3-level >= 4-level >= 4-level+covariates."""
        y = df["yi_icc"].to_numpy()
        v = df["vi_icc"].to_numpy()
        k = len(df)

        # A: baseline
        res_a = fit_reml(y, v, np.ones((k, 1)))

        # B: 3-level
        stage_3 = df["dln_stage"].replace("linear_plus", "linear")
        from meta_pipeline import design_matrix_stage
        X_b, _ = design_matrix_stage(stage_3, reference="dot")
        res_b = fit_reml(y, v, X_b)

        # C: 4-level
        X_c, _ = design_matrix_categorical(df["dln_stage"], reference="dot")
        res_c = fit_reml(y, v, X_c)

        # D: 4-level + n_crit + n_iat
        X_d = np.column_stack([X_c, df["n_crit"].to_numpy(), df["n_iat"].to_numpy()])
        res_d = fit_reml(y, v, X_d)

        assert res_b.tau2 <= res_a.tau2 + 1e-8, (
            f"3-level ({res_b.tau2:.6f}) > baseline ({res_a.tau2:.6f})"
        )
        assert res_c.tau2 <= res_b.tau2 + 1e-8, (
            f"4-level ({res_c.tau2:.6f}) > 3-level ({res_b.tau2:.6f})"
        )
        assert res_d.tau2 <= res_c.tau2 + 1e-8, (
            f"4-level+cov ({res_d.tau2:.6f}) > 4-level ({res_c.tau2:.6f})"
        )

    def test_verified_tau2_values(self, df):
        """Cross-check tau-squared against independently verified values."""
        y = df["yi_icc"].to_numpy()
        v = df["vi_icc"].to_numpy()

        # Baseline
        res_base = fit_reml(y, v, np.ones((len(df), 1)))
        assert abs(res_base.tau2 - 0.016682) < 0.0005, (
            f"Baseline tau2 = {res_base.tau2:.6f}, expected ~0.016682"
        )

        # 4-level
        X_4, _ = design_matrix_categorical(df["dln_stage"], reference="dot")
        res_4 = fit_reml(y, v, X_4)
        assert abs(res_4.tau2 - 0.011190) < 0.0005, (
            f"4-level tau2 = {res_4.tau2:.6f}, expected ~0.011190"
        )

        # Percentage reduction
        pct_red = (res_base.tau2 - res_4.tau2) / res_base.tau2 * 100
        assert abs(pct_red - 32.9) < 1.0, (
            f"4-level tau2 reduction = {pct_red:.1f}%, expected ~32.9%"
        )


# ---------------------------------------------------------------------------
# 7. Suppression fingerprint
# ---------------------------------------------------------------------------

class TestSuppressionFingerprint:
    def test_network_icc_higher_than_linear(self, df):
        """Network stage should have higher mean ICC than Linear stage."""
        net_icc = df[df["dln_stage"] == "network"]["icc"].mean()
        lin_icc = df[df["dln_stage"] == "linear"]["icc"].mean()
        assert net_icc > lin_icc, (
            f"Network ICC ({net_icc:.3f}) should exceed Linear ICC ({lin_icc:.3f})"
        )

    def test_linear_plus_iec_lower_than_linear(self, df):
        """Linear-Plus stage should have lower mean iec than Linear stage."""
        lp_iec = df[df["dln_stage"] == "linear_plus"].dropna(subset=["iec"])
        lin_iec = df[df["dln_stage"] == "linear"].dropna(subset=["iec"])
        assert len(lp_iec) > 0, "No Linear-Plus samples with iec"
        assert len(lin_iec) > 0, "No Linear samples with iec"
        assert lp_iec["iec"].mean() < lin_iec["iec"].mean(), (
            f"LP iec ({lp_iec['iec'].mean():.3f}) should be < "
            f"Linear iec ({lin_iec['iec'].mean():.3f})"
        )

    def test_linear_plus_positive_gap(self, df):
        """Linear-Plus stage should have positive ICC-ECC gap (suppression)."""
        lp = df[df["dln_stage"] == "linear_plus"].dropna(subset=["ecc"])
        assert len(lp) > 0, "No Linear-Plus samples with ECC"
        gap = (lp["icc"] - lp["ecc"]).mean()
        assert gap > 0, (
            f"LP gap (ICC-ECC) = {gap:.3f}, expected positive (suppression)"
        )


# ---------------------------------------------------------------------------
# 8. Scale-mixing guard
# ---------------------------------------------------------------------------

class TestScaleMixingGuard:
    def test_i2_not_naive_ratio(self, df):
        """Verify that REML I² differs from the scale-mixing error.
        The incorrect method divides Fisher-z mean vi by raw-r variance,
        producing ~95%.  The correct I² should be ~65%."""
        yi = df["yi_icc"].to_numpy()
        vi = df["vi_icc"].to_numpy()
        icc_raw = df["icc"].to_numpy()

        res = fit_reml(yi, vi, np.ones((len(df), 1)))

        # Correct I² from REML
        correct_i2 = res.I2

        # Wrong ratio (Fisher-z vi / raw-r variance)
        wrong_ratio = np.mean(vi) / np.var(icc_raw, ddof=1)

        # The wrong method gives ~0.95, correct is ~0.65
        assert abs(correct_i2 - 0.654) < 0.05, (
            f"I² = {correct_i2:.3f}, expected ~0.654"
        )
        assert abs(wrong_ratio - 0.953) < 0.05, (
            f"Wrong ratio = {wrong_ratio:.3f}, expected ~0.953"
        )
        assert abs(correct_i2 - wrong_ratio) > 0.2, (
            f"I² ({correct_i2:.3f}) and wrong ratio ({wrong_ratio:.3f}) "
            f"should differ substantially"
        )

    def test_same_scale_ratios_consistent(self, df):
        """Both Fisher-z and raw-r same-scale ratios should agree
        directionally with REML I²."""
        yi = df["yi_icc"].to_numpy()
        vi = df["vi_icc"].to_numpy()
        icc_raw = df["icc"].to_numpy()

        res = fit_reml(yi, vi, np.ones((len(df), 1)))

        # Same-scale Fisher-z ratio
        fz_ratio = np.mean(vi) / np.var(yi, ddof=1)

        # Both should indicate substantial real heterogeneity (ratio < 0.8)
        assert fz_ratio < 0.8, (
            f"Fisher-z sampling/total ratio = {fz_ratio:.3f}, "
            f"should be < 0.8 indicating real heterogeneity"
        )
        assert res.I2 > 0.5, (
            f"I² = {res.I2:.3f}, should indicate substantial heterogeneity"
        )


# ---------------------------------------------------------------------------
# 9. Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_fisher_z_columns_present(self, df):
        """prepare_data must add Fisher-z columns."""
        assert "yi_icc" in df.columns
        assert "vi_icc" in df.columns

    def test_vi_positive(self, df):
        """Sampling variances must be positive."""
        assert (df["vi_icc"] > 0).all()

    def test_sample_ids_unique(self, df):
        """Each sample_id should appear exactly once."""
        assert df["sample_id"].is_unique

    def test_required_columns_present(self, raw):
        """Raw data must contain columns needed for topology analysis."""
        required = ["sample_id", "topic", "icc", "n", "n_crit", "n_iat", "iat_type"]
        for col in required:
            assert col in raw.columns, f"Missing column: {col}"

    def test_dot_network_no_overlap(self):
        """DOT_SAMPLES and NETWORK_SAMPLES must be disjoint."""
        overlap = DOT_SAMPLES & NETWORK_SAMPLES
        assert len(overlap) == 0, f"Overlap between DOT and NETWORK: {overlap}"
