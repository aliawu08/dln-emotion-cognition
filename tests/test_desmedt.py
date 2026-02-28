"""Tests for the Desmedt et al. (2022) DLN-stage moderator analysis.

Validates:
- Extraction CSV structure and DLN coding consistency.
- Analysis script produces valid outputs.
- DLN coding follows the pre-specified rubric.

Data source: Published results section (page 4) of Desmedt, Bhatt,
Bhatt & Bhatt (2022) Collabra: Psychology 8(1), 33271.
All seven r values verified from the text.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from evidence_synthesis.analysis.meta_pipeline import (
    fit_reml,
    egger_test,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence_synthesis" / "extraction" / "desmedt2022_criterion_extraction.csv"


def r_to_fisher_z(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


# ---------------------------------------------------------------------------
# Extraction CSV structure
# ---------------------------------------------------------------------------

class TestExtractionCSV:
    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA)

    def test_required_columns_present(self, df):
        required = [
            "criterion", "k", "r_pooled", "N_approx",
            "se_r", "vi", "dln_stage_code",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_seven_criteria(self, df):
        assert len(df) == 7

    def test_expected_criteria_present(self, df):
        expected = {
            "heart_rate", "bmi", "age", "sex",
            "trait_anxiety", "depression", "alexithymia",
        }
        actual = set(df["criterion"].unique())
        assert actual == expected, f"Mismatch: {actual.symmetric_difference(expected)}"

    def test_dln_stage_codes_valid(self, df):
        valid = {"dot", "linear"}
        actual = set(df["dln_stage_code"].unique())
        assert actual == valid, f"Expected {valid}, got {actual}"

    def test_k_positive(self, df):
        assert (df["k"] > 0).all()

    def test_r_in_valid_range(self, df):
        assert (df["r_pooled"] >= -1.0).all()
        assert (df["r_pooled"] <= 1.0).all()

    def test_se_positive(self, df):
        assert (df["se_r"] > 0).all()

    def test_vi_positive(self, df):
        assert (df["vi"] > 0).all()

    def test_vi_consistent_with_N_approx(self, df):
        """vi should approximate (1-r^2)^2 / (N_approx - 1)."""
        r = df["r_pooled"].to_numpy()
        N = df["N_approx"].to_numpy()
        expected_vi = (1 - r**2) ** 2 / (N - 1)
        np.testing.assert_allclose(df["vi"], expected_vi, rtol=0.02)

    def test_data_status_verified(self, df):
        """All criterion-level r values should be verified from published text."""
        assert df["estimate_status"].str.contains("verified").all()

    def test_verified_r_values_match_text(self, df):
        """Cross-check verified r values against Desmedt et al. (2022) text."""
        published = {
            "trait_anxiety": 0.03,
            "depression": -0.04,
            "alexithymia": -0.01,
            "heart_rate": -0.17,
            "bmi": -0.11,
            "age": -0.06,
            "sex": -0.14,
        }
        for criterion, expected_r in published.items():
            actual_r = df.loc[df["criterion"] == criterion, "r_pooled"].iloc[0]
            assert abs(actual_r - expected_r) < 0.005, (
                f"{criterion}: expected r={expected_r}, got {actual_r}"
            )

    def test_coding_rationale_nonempty(self, df):
        assert df["coding_rationale"].notna().all()
        assert (df["coding_rationale"].str.len() > 10).all()


# ---------------------------------------------------------------------------
# DLN coding consistency
# ---------------------------------------------------------------------------

class TestDLNCoding:
    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA)

    def test_dot_criteria_are_biological(self, df):
        """Dot-coded criteria should be biological/physiological measures."""
        dot_rows = df[df["dln_stage_code"] == "dot"]
        expected_dot = {"heart_rate", "bmi", "age", "sex"}
        actual_dot = set(dot_rows["criterion"])
        assert actual_dot == expected_dot

    def test_linear_criteria_are_psychological(self, df):
        """Linear-coded criteria should be psychological self-report measures."""
        linear_rows = df[df["dln_stage_code"] == "linear"]
        expected_linear = {"trait_anxiety", "depression", "alexithymia"}
        actual_linear = set(linear_rows["criterion"])
        assert actual_linear == expected_linear

    def test_dot_rationale_references_biological(self, df):
        """Dot-coded criteria rationales should reference biological processing."""
        dot_rows = df[df["dln_stage_code"] == "dot"]
        for _, row in dot_rows.iterrows():
            rationale = row["coding_rationale"].lower()
            has_dot_keywords = any(
                kw in rationale for kw in [
                    "biological", "physiological", "somatic",
                    "physical", "reflexive", "direct",
                ]
            )
            assert has_dot_keywords, (
                f"Dot-coded criterion '{row['criterion']}' rationale lacks "
                f"dot-stage keywords: {row['coding_rationale'][:80]}..."
            )

    def test_linear_rationale_references_self_report(self, df):
        """Linear-coded criteria rationales should reference psychological self-report."""
        linear_rows = df[df["dln_stage_code"] == "linear"]
        for _, row in linear_rows.iterrows():
            rationale = row["coding_rationale"].lower()
            has_linear_keywords = any(
                kw in rationale for kw in [
                    "self-report", "psychological", "questionnaire",
                    "cognitive", "emotional", "subjective",
                ]
            )
            assert has_linear_keywords, (
                f"Linear-coded criterion '{row['criterion']}' rationale lacks "
                f"linear-stage keywords: {row['coding_rationale'][:80]}..."
            )

    def test_stage_distribution(self, df):
        """DLN coding should have 4 dot, 3 linear."""
        counts = df["dln_stage_code"].value_counts()
        assert counts.get("dot", 0) == 4
        assert counts.get("linear", 0) == 3


# ---------------------------------------------------------------------------
# Analysis pipeline correctness
# ---------------------------------------------------------------------------

class TestAnalysisPipeline:
    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA)

    def test_baseline_model_runs(self, df):
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        z = r_to_fisher_z(df["abs_r"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)
        X = np.ones((len(df), 1))
        result = fit_reml(z, vi_z, X)
        assert result.k == 7
        assert result.tau2 >= 0
        assert 0.0 <= result.I2 <= 1.0

    def test_moderator_model_runs(self, df):
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        z = r_to_fisher_z(df["abs_r"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)
        is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
        X_mod = np.column_stack([np.ones(len(df)), is_dot])
        result = fit_reml(z, vi_z, X_mod)
        assert result.k == 7

    def test_moderator_reduces_heterogeneity(self, df):
        """DLN stage coding should reduce tau-squared relative to baseline."""
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        z = r_to_fisher_z(df["abs_r"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)

        X_base = np.ones((len(df), 1))
        res_base = fit_reml(z, vi_z, X_base)

        is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
        X_mod = np.column_stack([np.ones(len(df)), is_dot])
        res_mod = fit_reml(z, vi_z, X_mod)

        assert res_mod.tau2 <= res_base.tau2, (
            f"Moderator tau2 ({res_mod.tau2:.4f}) should be <= baseline "
            f"tau2 ({res_base.tau2:.4f})"
        )

    def test_reduction_substantial(self, df):
        """Tau-squared reduction should be at least 50%."""
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        z = r_to_fisher_z(df["abs_r"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)

        X_base = np.ones((len(df), 1))
        res_base = fit_reml(z, vi_z, X_base)

        is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
        X_mod = np.column_stack([np.ones(len(df)), is_dot])
        res_mod = fit_reml(z, vi_z, X_mod)

        if res_base.tau2 > 0:
            reduction = (res_base.tau2 - res_mod.tau2) / res_base.tau2
            assert reduction >= 0.50, (
                f"Expected >= 50% reduction, got {reduction * 100:.1f}%"
            )

    def test_egger_test_runs(self, df):
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        z = r_to_fisher_z(df["abs_r"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)
        result = egger_test(z, vi_z)
        assert result.k == 7

    def test_design_matrix_shape(self, df):
        is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
        X = np.column_stack([np.ones(len(df)), is_dot])
        assert X.shape == (7, 2)


# ---------------------------------------------------------------------------
# DLN theoretical predictions: same-level correspondence
# ---------------------------------------------------------------------------

class TestDLNPredictions:
    """Test that the verified data support the same-level correspondence prediction.

    DLN predicts: dot |r| > linear |r|
    Because HCT is a dot-stage task, dot-coded criteria (biological measures)
    should show stronger associations than cross-level criteria (psychological
    self-report = linear).
    """

    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA)

    def test_dot_abs_r_greater_than_linear(self, df):
        """Dot-stage criteria should have larger |r| than linear-stage criteria."""
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        dot_mean = df.loc[df["dln_stage_code"] == "dot", "abs_r"].mean()
        linear_mean = df.loc[df["dln_stage_code"] == "linear", "abs_r"].mean()
        assert dot_mean > linear_mean, (
            f"Dot mean |r| ({dot_mean:.3f}) should exceed "
            f"linear mean |r| ({linear_mean:.3f})"
        )

    def test_weighted_dot_abs_r_greater(self, df):
        """Weighted (by inverse variance) dot |r| should exceed linear |r|."""
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        df["z_abs"] = r_to_fisher_z(df["abs_r"].to_numpy())
        df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)

        dot = df[df["dln_stage_code"] == "dot"]
        linear = df[df["dln_stage_code"] == "linear"]

        dot_wm = np.average(dot["z_abs"], weights=1.0 / dot["vi_z"])
        linear_wm = np.average(linear["z_abs"], weights=1.0 / linear["vi_z"])

        assert dot_wm > linear_wm, (
            f"Weighted dot z ({dot_wm:.3f}) should exceed "
            f"linear z ({linear_wm:.3f})"
        )

    def test_dot_coefficient_positive(self, df):
        """The dot coefficient in the moderator model should be positive."""
        df = df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        z = r_to_fisher_z(df["abs_r"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)

        is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
        X_mod = np.column_stack([np.ones(len(df)), is_dot])
        result = fit_reml(z, vi_z, X_mod)

        dot_coef = result.beta[1]
        assert dot_coef > 0, (
            f"Dot coefficient ({dot_coef:.4f}) should be positive "
            f"(same-level correspondence)"
        )
