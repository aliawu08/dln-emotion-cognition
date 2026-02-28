"""Tests for the Hoyt et al. (2024) DLN-stage moderator analysis.

Validates:
- Extraction CSV structure and DLN coding consistency.
- Analysis script produces valid outputs.
- DLN coding follows the pre-specified rubric.

Data source: Table 3 and Figure 2 of Hoyt, Llave, Wang et al. (2024)
Health Psychology 43(6), 397-417.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from evidence_synthesis.analysis.meta_pipeline import (
    fit_reml,
    design_matrix_stage,
    egger_test,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence_synthesis" / "extraction" / "hoyt2024_domain_extraction.csv"


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
            "health_domain", "domain_desc", "k", "r_pooled", "N_approx",
            "se_r", "vi", "dln_stage_code", "coding_rationale", "source",
            "estimate_status",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_eight_health_domains(self, df):
        assert len(df) == 8

    def test_expected_domains_present(self, df):
        expected = {
            "biological_physiological", "physical_health", "behavioral",
            "mental_emotional_distress", "risk_related_adjustment",
            "positive_psychological_health", "social_functioning",
            "resilience_adjustment",
        }
        actual = set(df["health_domain"].unique())
        assert actual == expected, f"Mismatch: {actual.symmetric_difference(expected)}"

    def test_dln_stage_codes_valid(self, df):
        valid = {"dot", "linear", "network"}
        actual = set(df["dln_stage_code"].unique())
        assert actual.issubset(valid), f"Invalid stage codes: {actual - valid}"

    def test_all_three_stages_represented(self, df):
        stages = set(df["dln_stage_code"].unique())
        assert "dot" in stages
        assert "linear" in stages
        assert "network" in stages

    def test_k_positive(self, df):
        assert (df["k"] > 0).all()

    def test_r_in_valid_range(self, df):
        assert (df["r_pooled"] >= -1.0).all()
        assert (df["r_pooled"] <= 1.0).all()

    def test_se_positive(self, df):
        assert (df["se_r"] > 0).all()

    def test_vi_positive(self, df):
        assert (df["vi"] > 0).all()

    def test_vi_equals_se_squared(self, df):
        """vi should be se_r^2."""
        expected_vi = df["se_r"] ** 2
        np.testing.assert_allclose(df["vi"], expected_vi, atol=1e-6)

    def test_data_status_verified(self, df):
        """All domain-level r values should be verified from Table 3."""
        assert df["estimate_status"].str.contains("verified").all()

    def test_coding_rationale_nonempty(self, df):
        assert df["coding_rationale"].notna().all()
        assert (df["coding_rationale"].str.len() > 10).all()

    def test_verified_r_values_match_table3(self, df):
        """Cross-check verified r values against Table 3 of Hoyt et al. (2024)."""
        table3 = {
            "biological_physiological": -0.02,
            "physical_health": 0.02,
            "behavioral": 0.14,
            "mental_emotional_distress": -0.11,
            "risk_related_adjustment": -0.18,
            "positive_psychological_health": 0.29,
            "social_functioning": 0.28,
            "resilience_adjustment": 0.31,
        }
        for domain, expected_r in table3.items():
            actual_r = df.loc[df["health_domain"] == domain, "r_pooled"].iloc[0]
            assert abs(actual_r - expected_r) < 0.005, (
                f"{domain}: expected r={expected_r}, got {actual_r}"
            )


# ---------------------------------------------------------------------------
# DLN coding consistency
# ---------------------------------------------------------------------------

class TestDLNCoding:
    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA)

    def test_dot_domains_are_somatic_reactive(self, df):
        """Dot-coded domains should reference somatic/reactive/stimulus-driven processing."""
        dot_rows = df[df["dln_stage_code"] == "dot"]
        for _, row in dot_rows.iterrows():
            rationale = row["coding_rationale"].lower()
            has_dot_keywords = any(
                kw in rationale for kw in [
                    "somatic", "stimulus", "reactive", "action",
                    "body", "symptom", "habit",
                ]
            )
            assert has_dot_keywords, (
                f"Dot-coded domain '{row['health_domain']}' rationale lacks "
                f"dot-stage keywords: {row['coding_rationale'][:80]}..."
            )

    def test_linear_domains_reference_compartmentalization(self, df):
        """Linear-coded domains should reference unidimensional/isolated structure."""
        linear_rows = df[df["dln_stage_code"] == "linear"]
        for _, row in linear_rows.iterrows():
            rationale = row["coding_rationale"].lower()
            has_linear_keywords = any(
                kw in rationale for kw in [
                    "pathology", "interference", "distress", "compartment",
                    "amplif", "risk", "failure",
                    "unidimensional", "single-valence", "isolated",
                    "without cross-domain", "without relational",
                ]
            )
            assert has_linear_keywords, (
                f"Linear-coded domain '{row['health_domain']}' rationale lacks "
                f"linear-stage keywords: {row['coding_rationale'][:80]}..."
            )

    def test_network_domains_reference_integration(self, df):
        """Network-coded domains should reference integrative/meaning-making processing."""
        network_rows = df[df["dln_stage_code"] == "network"]
        for _, row in network_rows.iterrows():
            rationale = row["coding_rationale"].lower()
            has_network_keywords = any(
                kw in rationale for kw in [
                    "integrat", "meaning", "multi-dimensional",
                    "reappraisal", "flexible",
                ]
            )
            assert has_network_keywords, (
                f"Network-coded domain '{row['health_domain']}' rationale lacks "
                f"network-stage keywords: {row['coding_rationale'][:80]}..."
            )

    def test_stage_distribution(self, df):
        """DLN coding should have 3 dot, 2 linear, 3 network."""
        counts = df["dln_stage_code"].value_counts()
        assert counts.get("dot", 0) == 3
        assert counts.get("linear", 0) == 2
        assert counts.get("network", 0) == 3


# ---------------------------------------------------------------------------
# Analysis pipeline correctness
# ---------------------------------------------------------------------------

class TestAnalysisPipeline:
    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA)

    def test_baseline_model_runs(self, df):
        z = r_to_fisher_z(df["r_pooled"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)
        X = np.ones((len(df), 1))
        result = fit_reml(z, vi_z, X)
        assert result.k == 8
        assert result.tau2 >= 0
        assert 0.0 <= result.I2 <= 1.0

    def test_moderator_model_runs(self, df):
        z = r_to_fisher_z(df["r_pooled"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)
        X_mod, names = design_matrix_stage(df["dln_stage_code"], reference="dot")
        result = fit_reml(z, vi_z, X_mod)
        assert result.k == 8
        assert len(names) == 3  # Intercept + linear + network

    def test_moderator_reduces_heterogeneity(self, df):
        """DLN stage coding should reduce tau-squared relative to baseline."""
        z = r_to_fisher_z(df["r_pooled"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)

        X_base = np.ones((len(df), 1))
        res_base = fit_reml(z, vi_z, X_base)

        X_mod, _ = design_matrix_stage(df["dln_stage_code"], reference="dot")
        res_mod = fit_reml(z, vi_z, X_mod)

        assert res_mod.tau2 <= res_base.tau2, (
            f"Moderator tau2 ({res_mod.tau2:.4f}) should be <= baseline "
            f"tau2 ({res_base.tau2:.4f})"
        )

    def test_baseline_heterogeneity_substantial(self, df):
        """Baseline I-squared should be substantial (domains have different effects)."""
        z = r_to_fisher_z(df["r_pooled"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)
        X = np.ones((len(df), 1))
        result = fit_reml(z, vi_z, X)
        assert result.I2 > 0.5, (
            f"Expected substantial baseline heterogeneity, got I2={result.I2:.3f}"
        )

    def test_egger_test_runs(self, df):
        z = r_to_fisher_z(df["r_pooled"].to_numpy())
        vi_z = 1.0 / (df["N_approx"] - 3.0)
        result = egger_test(z, vi_z)
        assert result.k == 8

    def test_design_matrix_shape(self, df):
        X, names = design_matrix_stage(df["dln_stage_code"], reference="dot")
        assert X.shape[0] == 8
        assert X.shape[1] == 3
        assert names[0] == "Intercept"


# ---------------------------------------------------------------------------
# DLN theoretical predictions: dangerous-middle pattern
# ---------------------------------------------------------------------------

class TestDLNPredictions:
    """Test that the verified data support the dangerous-middle (V-shape) prediction.

    DLN predicts: dot (+) > linear (-) < network (+)
    - Dot domains: near-zero or small positive (basic somatic benefit)
    - Linear domains: negative (EAC amplifies distress without integration)
    - Network domains: positive (EAC facilitates integrative processing)
    """

    @pytest.fixture
    def df(self):
        return pd.read_csv(DATA)

    def test_linear_negative(self, df):
        """Linear-coded domains should have negative pooled EAC-health association."""
        linear = df[df["dln_stage_code"] == "linear"]
        linear_mean = np.average(
            r_to_fisher_z(linear["r_pooled"].to_numpy()),
            weights=1.0 / (linear["N_approx"] - 3.0),
        )
        assert linear_mean < 0, (
            f"Linear mean z={linear_mean:.3f} should be negative "
            f"(dangerous-middle prediction)"
        )

    def test_network_positive(self, df):
        """Network-coded domains should have positive pooled EAC-health association."""
        network = df[df["dln_stage_code"] == "network"]
        network_mean = np.average(
            r_to_fisher_z(network["r_pooled"].to_numpy()),
            weights=1.0 / (network["N_approx"] - 3.0),
        )
        assert network_mean > 0, (
            f"Network mean z={network_mean:.3f} should be positive"
        )

    def test_v_shape_pattern(self, df):
        """Linear < dot AND linear < network (V-shaped / dangerous-middle)."""
        stage_means = {}
        for stage in ["dot", "linear", "network"]:
            sub = df[df["dln_stage_code"] == stage]
            z = r_to_fisher_z(sub["r_pooled"].to_numpy())
            w = 1.0 / (sub["N_approx"] - 3.0)
            stage_means[stage] = np.average(z, weights=w)

        assert stage_means["linear"] < stage_means["dot"], (
            f"linear ({stage_means['linear']:.3f}) should be < "
            f"dot ({stage_means['dot']:.3f})"
        )
        assert stage_means["linear"] < stage_means["network"], (
            f"linear ({stage_means['linear']:.3f}) should be < "
            f"network ({stage_means['network']:.3f})"
        )

    def test_network_exceeds_dot(self, df):
        """Network domains should show stronger positive effects than dot domains."""
        stage_means = {}
        for stage in ["dot", "network"]:
            sub = df[df["dln_stage_code"] == stage]
            z = r_to_fisher_z(sub["r_pooled"].to_numpy())
            w = 1.0 / (sub["N_approx"] - 3.0)
            stage_means[stage] = np.average(z, weights=w)

        assert stage_means["network"] > stage_means["dot"], (
            f"Network mean ({stage_means['network']:.3f}) should exceed "
            f"Dot mean ({stage_means['dot']:.3f})"
        )
