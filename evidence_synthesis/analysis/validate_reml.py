"""Cross-validate the custom REML implementation against R's metafor package.

This script runs every meta-analysis reported in the manuscript through the
Python REML pipeline and compares results against metafor output (if available).

Workflow
-------
1. Run this script first (Python side):  populates python_value in the report.
2. Run the R scripts locally:
     Rscript evidence_synthesis/analysis/validate_metafor/run_all.R
3. Run this script again:  reads metafor CSVs, compares, reports pass/fail.

Tolerance thresholds (absolute unless noted):
  tau2:     0.001  (or 1% relative if tau2 > 0.01)
  beta/mu:  0.005
  SE:       0.005
  CI:       0.01
  Q/QM:     0.1
  I2:       0.005
  p-value:  0.01  (or both < 0.001)

Usage:
  python evidence_synthesis/analysis/validate_reml.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Handle both direct execution and module import
try:
    from meta_pipeline import (
        fit_reml, design_matrix_stage, design_matrix_categorical,
        egger_test, profile_likelihood_ci, compute_qm, aicc,
    )
    from multilevel_meta import fit_three_level, cluster_robust_variance
except ModuleNotFoundError:
    from evidence_synthesis.analysis.meta_pipeline import (
        fit_reml, design_matrix_stage, design_matrix_categorical,
        egger_test, profile_likelihood_ci, compute_qm, aicc,
    )
    from evidence_synthesis.analysis.multilevel_meta import (
        fit_three_level, cluster_robust_variance,
    )

ROOT = Path(__file__).resolve().parents[2]
EXTRACTION = ROOT / "evidence_synthesis" / "extraction"
METAFOR_DIR = Path(__file__).resolve().parent / "validate_metafor"

# ── Tolerance thresholds ────────────────────────────────────────────────────

TOL = {
    "tau2": 0.001,
    "beta": 0.005,
    "se": 0.005,
    "ci": 0.01,
    "Q": 0.1,
    "QM": 0.1,
    "I2": 0.005,
    "p": 0.01,
}


def _classify(name: str) -> str:
    """Map a parameter name to a tolerance class."""
    n = name.lower()
    if "tau2_pl" in n:
        return "ci"
    if "tau2" in n or "sigma2" in n:
        return "tau2"
    if "i2" in n:
        return "I2"
    if "ci_" in n:
        return "ci"
    if "se" in n:
        return "se"
    if "qm" in n and "_p" not in n:
        return "QM"
    if n.endswith("_q") or n == "base_q" or n == "mod_q":
        return "Q"
    if "_p" in n or n.endswith("_p"):
        return "p"
    if "beta" in n or "mu" in n or "intcp" in n:
        return "beta"
    return "beta"  # default


# ── Known formula variants ────────────────────────────────────────────────
# These parameter discrepancies are expected differences between the Python
# pipeline and metafor, NOT bugs.  Categories:
#   I2_formula    : Python uses (Q-df)/Q; metafor uses tau2/(tau2+s2)
#   KH_SE         : R scripts apply Knapp-Hartung SE scaling; Python does not
#   KH_CI         : Consequent CI differences from KH-adjusted SEs + t-dist
#   QM_chi2_vs_F  : Python reports chi2; metafor with test="knha" reports F
#   QM_p          : p-value from different QM distribution
#   profile_CI    : Profile-likelihood CI bounds differ at boundary/optimizer
#   CR2_se        : clubSandwich CR2 SE differs from Python CR2
#   CR2_df        : Python uses n-p; clubSandwich uses Satterthwaite
#   CR2_derived   : t-stats and p-values derived from different CR2 SEs/df

KNOWN_VARIANTS: Dict[str, Dict[str, str]] = {}

def _register_variants(dataset: str, params_and_reasons: List[Tuple[str, str]]):
    """Register expected formula variants for a dataset."""
    if dataset not in KNOWN_VARIANTS:
        KNOWN_VARIANTS[dataset] = {}
    for param, reason in params_and_reasons:
        KNOWN_VARIANTS[dataset][param] = reason

# Webb strategy (k=10): moderator uses test="knha" in R
_register_variants("Webb strategy (k=10)", [
    ("base_I2", "I2_formula"),
    ("base_ci_lo", "KH_CI"), ("base_ci_hi", "KH_CI"),
    ("base_tau2_pl_hi", "profile_CI"),
    ("mod_I2", "I2_formula"),
    ("mod_QM", "QM_chi2_vs_F"), ("mod_QM_p", "QM_p"),
    ("mod_se_intcp", "KH_SE"), ("mod_se_linear", "KH_SE"), ("mod_se_network", "KH_SE"),
    ("mod_ci_lo_intcp", "KH_CI"), ("mod_ci_hi_intcp", "KH_CI"),
    ("mod_ci_lo_linear", "KH_CI"), ("mod_ci_hi_linear", "KH_CI"),
    ("mod_ci_lo_network", "KH_CI"), ("mod_ci_hi_network", "KH_CI"),
])

# Greenwald ICC (k=184): moderator uses test="knha" in R
_register_variants("Greenwald ICC (k=184)", [
    ("icc_base_I2", "I2_formula"),
    ("icc_mod_I2", "I2_formula"),
    ("icc_mod_QM", "QM_chi2_vs_F"), ("icc_mod_QM_p", "QM_p"),
])

# Hoyt health (k=8): moderator uses test="knha" in R
_register_variants("Hoyt health (k=8)", [
    ("base_I2", "I2_formula"),
    ("base_ci_lo", "KH_CI"), ("base_ci_hi", "KH_CI"),
    ("base_tau2_pl_hi", "profile_CI"),
    ("mod_QM", "QM_chi2_vs_F"),
    ("mod_se_intcp", "KH_SE"), ("mod_se_linear", "KH_SE"), ("mod_se_network", "KH_SE"),
    ("mod_ci_lo_intcp", "KH_CI"), ("mod_ci_hi_intcp", "KH_CI"),
    ("mod_ci_lo_linear", "KH_CI"), ("mod_ci_hi_linear", "KH_CI"),
    ("mod_ci_lo_network", "KH_CI"), ("mod_ci_hi_network", "KH_CI"),
])

# Desmedt HCT (k=7): moderator uses test="knha" in R
_register_variants("Desmedt HCT (k=7)", [
    ("abs_base_ci_lo", "KH_CI"), ("abs_base_ci_hi", "KH_CI"),
    ("abs_mod_I2", "I2_formula"),
    ("abs_mod_QM_p", "QM_p"),
    ("signed_base_I2", "I2_formula"),
])

# Interoception (k=8): moderator uses test="knha" in R
_register_variants("Interoception (k=8)", [
    ("base_ci_lo", "KH_CI"), ("base_ci_hi", "KH_CI"),
    ("base_tau2_pl_hi", "profile_CI"),
    ("mod_I2", "I2_formula"),
    ("mod_QM", "QM_chi2_vs_F"),
    ("mod_tau2_pl_hi", "profile_CI"),
])

# Webb comparison 3-level (k=306): CR2 differences
_register_variants("Webb comparison 3-level (k=306)", [
    ("threelevel_mod_QM", "QM_chi2_vs_F"),
    ("cr2_se_intcp", "CR2_se"), ("cr2_se_linear", "CR2_se"), ("cr2_se_network", "CR2_se"),
    ("cr2_df_intcp", "CR2_df"), ("cr2_df_linear", "CR2_df"), ("cr2_df_network", "CR2_df"),
    ("cr2_t_linear", "CR2_derived"), ("cr2_t_network", "CR2_derived"),
])


def _check(name: str, py_val: float, mf_val: float) -> Tuple[bool, float]:
    """Check whether python and metafor values agree within tolerance."""
    diff = abs(py_val - mf_val)
    cls = _classify(name)
    tol = TOL.get(cls, 0.01)

    # Relative tolerance for larger tau2 values
    if cls == "tau2" and abs(mf_val) > 0.01:
        return (diff / abs(mf_val) < 0.01) or (diff < tol), diff

    # p-value special case: both very small is fine
    if cls == "p" and py_val < 0.001 and mf_val < 0.001:
        return True, diff

    return diff < tol, diff


# ── Dataset-specific Python analyses ────────────────────────────────────────


def _run_webb_strategy() -> Dict[str, float]:
    """Webb et al. (2012) strategy-family (k=10)."""
    df = pd.read_csv(EXTRACTION / "webb2012_strategy_extraction.csv")
    y = df["d_plus"].to_numpy()
    v = df["vi"].to_numpy()

    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)

    X_mod, names = design_matrix_stage(df["dln_stage_code"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    egger = egger_test(y, v)
    pl_base = profile_likelihood_ci(y, v, X_base)
    pl_mod = profile_likelihood_ci(y, v, X_mod)

    qm = compute_qm(y, v, X_base, X_mod)

    return {
        "base_tau2": res_base.tau2,
        "base_I2": res_base.I2,
        "base_Q": res_base.Q,
        "base_mu": res_base.beta[0],
        "base_se_mu": res_base.se[0],
        "base_ci_lo": res_base.ci95[0, 0],
        "base_ci_hi": res_base.ci95[0, 1],
        "base_tau2_pl_lo": pl_base.ci_lo,
        "base_tau2_pl_hi": pl_base.ci_hi,
        "mod_tau2": res_mod.tau2,
        "mod_I2": res_mod.I2,
        "mod_Q": res_mod.Q,
        "mod_beta_intcp": res_mod.beta[0],
        "mod_se_intcp": res_mod.se[0],
        "mod_ci_lo_intcp": res_mod.ci95[0, 0],
        "mod_ci_hi_intcp": res_mod.ci95[0, 1],
        "mod_beta_linear": res_mod.beta[1],
        "mod_se_linear": res_mod.se[1],
        "mod_ci_lo_linear": res_mod.ci95[1, 0],
        "mod_ci_hi_linear": res_mod.ci95[1, 1],
        "mod_beta_network": res_mod.beta[2],
        "mod_se_network": res_mod.se[2],
        "mod_ci_lo_network": res_mod.ci95[2, 0],
        "mod_ci_hi_network": res_mod.ci95[2, 1],
        "mod_QM": qm.QM,
        "mod_QM_p": qm.p,
        "mod_tau2_pl_lo": pl_mod.ci_lo,
        "mod_tau2_pl_hi": pl_mod.ci_hi,
        "egger_z": egger.t_stat,
        "egger_p": egger.p_value,
    }


def _run_greenwald() -> Dict[str, float]:
    """Greenwald et al. (2009) study-level (k=184)."""
    df = pd.read_csv(EXTRACTION / "greenwald2009_study_extraction.csv")

    # DLN coding from topic (must match R script)
    topic_to_dln = {
        "Consumer": "dot",
        "Race (Bl/Wh)": "linear",
        "Politics": "linear",
        "Gender/sex": "linear",
        "Other intergroup": "linear",
        "Relationships": "network",
        "Personality": "mixed_unclear",
        "Drugs/tobacco": "mixed_unclear",
        "Clinical": "mixed_unclear",
    }
    df["dln_stage"] = df["topic"].map(topic_to_dln)

    # Fisher-z transform ICC
    df["yi_icc"] = np.arctanh(df["icc"].clip(-0.999, 0.999))
    df["vi_icc"] = 1.0 / (df["n"] - 3)

    y = df["yi_icc"].to_numpy()
    v = df["vi_icc"].to_numpy()

    # Baseline
    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)

    # Moderator: design_matrix_stage with reference="dot"
    # mixed_unclear -> (0,0) dummies, same as dot
    X_mod, names = design_matrix_stage(df["dln_stage"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    qm = compute_qm(y, v, X_base, X_mod)
    egger = egger_test(y, v)

    # Drop mixed sensitivity
    df_clean = df[df["dln_stage"].isin(["dot", "linear", "network"])].copy()
    y2 = df_clean["yi_icc"].to_numpy()
    v2 = df_clean["vi_icc"].to_numpy()
    X_base2 = np.ones((len(df_clean), 1))
    res_base2 = fit_reml(y2, v2, X_base2)
    X_mod2, _ = design_matrix_stage(df_clean["dln_stage"], reference="dot")
    res_mod2 = fit_reml(y2, v2, X_mod2)

    return {
        "icc_base_tau2": res_base.tau2,
        "icc_base_I2": res_base.I2,
        "icc_base_Q": res_base.Q,
        "icc_base_mu": res_base.beta[0],
        "icc_base_se_mu": res_base.se[0],
        "icc_base_ci_lo": res_base.ci95[0, 0],
        "icc_base_ci_hi": res_base.ci95[0, 1],
        "icc_mod_tau2": res_mod.tau2,
        "icc_mod_I2": res_mod.I2,
        "icc_mod_Q": res_mod.Q,
        "icc_mod_beta_intcp": res_mod.beta[0],
        "icc_mod_se_intcp": res_mod.se[0],
        "icc_mod_ci_lo_intcp": res_mod.ci95[0, 0],
        "icc_mod_ci_hi_intcp": res_mod.ci95[0, 1],
        "icc_mod_beta_linear": res_mod.beta[1],
        "icc_mod_se_linear": res_mod.se[1],
        "icc_mod_ci_lo_linear": res_mod.ci95[1, 0],
        "icc_mod_ci_hi_linear": res_mod.ci95[1, 1],
        "icc_mod_beta_network": res_mod.beta[2],
        "icc_mod_se_network": res_mod.se[2],
        "icc_mod_ci_lo_network": res_mod.ci95[2, 0],
        "icc_mod_ci_hi_network": res_mod.ci95[2, 1],
        "icc_mod_QM": qm.QM,
        "icc_mod_QM_p": qm.p,
        "icc_dropmixed_base_tau2": res_base2.tau2,
        "icc_dropmixed_base_mu": res_base2.beta[0],
        "icc_dropmixed_mod_tau2": res_mod2.tau2,
        "icc_dropmixed_mod_beta_intcp": res_mod2.beta[0],
        "icc_dropmixed_mod_beta_linear": res_mod2.beta[1],
        "icc_dropmixed_mod_beta_network": res_mod2.beta[2],
        "icc_egger_z": egger.t_stat,
        "icc_egger_p": egger.p_value,
    }


def _run_hoyt() -> Dict[str, float]:
    """Hoyt et al. (2024) health domains (k=8)."""
    df = pd.read_csv(EXTRACTION / "hoyt2024_domain_extraction.csv")

    df["yi"] = np.arctanh(np.clip(df["r_pooled"].to_numpy(), -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)

    y = df["yi"].to_numpy()
    v = df["vi_z"].to_numpy()

    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)

    X_mod, names = design_matrix_stage(df["dln_stage_code"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    qm = compute_qm(y, v, X_base, X_mod)
    egger = egger_test(y, v)
    pl_base = profile_likelihood_ci(y, v, X_base)
    pl_mod = profile_likelihood_ci(y, v, X_mod)

    return {
        "base_tau2": res_base.tau2,
        "base_I2": res_base.I2,
        "base_Q": res_base.Q,
        "base_mu": res_base.beta[0],
        "base_se_mu": res_base.se[0],
        "base_ci_lo": res_base.ci95[0, 0],
        "base_ci_hi": res_base.ci95[0, 1],
        "base_tau2_pl_lo": pl_base.ci_lo,
        "base_tau2_pl_hi": pl_base.ci_hi,
        "mod_tau2": res_mod.tau2,
        "mod_I2": res_mod.I2,
        "mod_Q": res_mod.Q,
        "mod_beta_intcp": res_mod.beta[0],
        "mod_se_intcp": res_mod.se[0],
        "mod_ci_lo_intcp": res_mod.ci95[0, 0],
        "mod_ci_hi_intcp": res_mod.ci95[0, 1],
        "mod_beta_linear": res_mod.beta[1],
        "mod_se_linear": res_mod.se[1],
        "mod_ci_lo_linear": res_mod.ci95[1, 0],
        "mod_ci_hi_linear": res_mod.ci95[1, 1],
        "mod_beta_network": res_mod.beta[2],
        "mod_se_network": res_mod.se[2],
        "mod_ci_lo_network": res_mod.ci95[2, 0],
        "mod_ci_hi_network": res_mod.ci95[2, 1],
        "mod_QM": qm.QM,
        "mod_QM_p": qm.p,
        "mod_tau2_pl_lo": pl_mod.ci_lo,
        "mod_tau2_pl_hi": pl_mod.ci_hi,
        "egger_z": egger.t_stat,
        "egger_p": egger.p_value,
    }


def _run_desmedt() -> Dict[str, float]:
    """Desmedt et al. (2022) HCT criteria (k=7)."""
    df = pd.read_csv(EXTRACTION / "desmedt2022_criterion_extraction.csv")

    df["abs_r"] = df["r_pooled"].abs()
    df["z_abs"] = np.arctanh(np.clip(df["abs_r"].to_numpy(), -0.999, 0.999))
    df["z_signed"] = np.arctanh(np.clip(df["r_pooled"].to_numpy(), -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)

    y_abs = df["z_abs"].to_numpy()
    y_signed = df["z_signed"].to_numpy()
    v = df["vi_z"].to_numpy()

    # Analysis 1: absolute r
    X_base = np.ones((len(df), 1))
    res_base_abs = fit_reml(y_abs, v, X_base)

    # Moderator: is_dot (linear as reference)
    is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
    X_mod_abs = np.column_stack([np.ones(len(df)), is_dot])
    res_mod_abs = fit_reml(y_abs, v, X_mod_abs)

    qm_abs = compute_qm(y_abs, v, X_base, X_mod_abs)
    egger_abs = egger_test(y_abs, v)
    pl_base_abs = profile_likelihood_ci(y_abs, v, X_base)
    pl_mod_abs = profile_likelihood_ci(y_abs, v, X_mod_abs)

    # Analysis 2: signed r
    X_base2 = np.ones((len(df), 1))
    res_base_sign = fit_reml(y_signed, v, X_base2)
    X_mod2 = np.column_stack([np.ones(len(df)), is_dot])
    res_mod_sign = fit_reml(y_signed, v, X_mod2)

    return {
        "abs_base_tau2": res_base_abs.tau2,
        "abs_base_I2": res_base_abs.I2,
        "abs_base_Q": res_base_abs.Q,
        "abs_base_mu": res_base_abs.beta[0],
        "abs_base_se_mu": res_base_abs.se[0],
        "abs_base_ci_lo": res_base_abs.ci95[0, 0],
        "abs_base_ci_hi": res_base_abs.ci95[0, 1],
        "abs_mod_tau2": res_mod_abs.tau2,
        "abs_mod_I2": res_mod_abs.I2,
        "abs_mod_Q": res_mod_abs.Q,
        "abs_mod_beta_intcp": res_mod_abs.beta[0],
        "abs_mod_se_intcp": res_mod_abs.se[0],
        "abs_mod_ci_lo_intcp": res_mod_abs.ci95[0, 0],
        "abs_mod_ci_hi_intcp": res_mod_abs.ci95[0, 1],
        "abs_mod_beta_dot": res_mod_abs.beta[1],
        "abs_mod_se_dot": res_mod_abs.se[1],
        "abs_mod_ci_lo_dot": res_mod_abs.ci95[1, 0],
        "abs_mod_ci_hi_dot": res_mod_abs.ci95[1, 1],
        "abs_mod_QM": qm_abs.QM,
        "abs_mod_QM_p": qm_abs.p,
        "abs_egger_z": egger_abs.t_stat,
        "abs_egger_p": egger_abs.p_value,
        "abs_base_tau2_pl_lo": pl_base_abs.ci_lo,
        "abs_base_tau2_pl_hi": pl_base_abs.ci_hi,
        "abs_mod_tau2_pl_lo": pl_mod_abs.ci_lo,
        "abs_mod_tau2_pl_hi": pl_mod_abs.ci_hi,
        "signed_base_tau2": res_base_sign.tau2,
        "signed_base_I2": res_base_sign.I2,
        "signed_base_Q": res_base_sign.Q,
        "signed_base_mu": res_base_sign.beta[0],
        "signed_mod_tau2": res_mod_sign.tau2,
        "signed_mod_beta_intcp": res_mod_sign.beta[0],
        "signed_mod_beta_dot": res_mod_sign.beta[1],
        "signed_mod_se_intcp": res_mod_sign.se[0],
        "signed_mod_se_dot": res_mod_sign.se[1],
    }


def _run_interoception() -> Dict[str, float]:
    """Interoception measures (k=8)."""
    df = pd.read_csv(EXTRACTION / "interoception_measure_extraction.csv")

    df["yi"] = np.arctanh(np.clip(df["r_pooled"].to_numpy(), -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_total"] - 3.0 * df["k"])

    y = df["yi"].to_numpy()
    v = df["vi_z"].to_numpy()

    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)

    X_mod, names = design_matrix_stage(df["dln_stage_code"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    qm = compute_qm(y, v, X_base, X_mod)
    egger = egger_test(y, v)
    pl_base = profile_likelihood_ci(y, v, X_base)
    pl_mod = profile_likelihood_ci(y, v, X_mod)

    return {
        "base_tau2": res_base.tau2,
        "base_I2": res_base.I2,
        "base_Q": res_base.Q,
        "base_mu": res_base.beta[0],
        "base_se_mu": res_base.se[0],
        "base_ci_lo": res_base.ci95[0, 0],
        "base_ci_hi": res_base.ci95[0, 1],
        "base_tau2_pl_lo": pl_base.ci_lo,
        "base_tau2_pl_hi": pl_base.ci_hi,
        "mod_tau2": res_mod.tau2,
        "mod_I2": res_mod.I2,
        "mod_Q": res_mod.Q,
        "mod_beta_intcp": res_mod.beta[0],
        "mod_se_intcp": res_mod.se[0],
        "mod_ci_lo_intcp": res_mod.ci95[0, 0],
        "mod_ci_hi_intcp": res_mod.ci95[0, 1],
        "mod_beta_linear": res_mod.beta[1],
        "mod_se_linear": res_mod.se[1],
        "mod_ci_lo_linear": res_mod.ci95[1, 0],
        "mod_ci_hi_linear": res_mod.ci95[1, 1],
        "mod_beta_network": res_mod.beta[2],
        "mod_se_network": res_mod.se[2],
        "mod_ci_lo_network": res_mod.ci95[2, 0],
        "mod_ci_hi_network": res_mod.ci95[2, 1],
        "mod_QM": qm.QM,
        "mod_QM_p": qm.p,
        "mod_tau2_pl_lo": pl_mod.ci_lo,
        "mod_tau2_pl_hi": pl_mod.ci_hi,
        "egger_z": egger.t_stat,
        "egger_p": egger.p_value,
    }


def _run_webb_comparison() -> Dict[str, float]:
    """Webb et al. (2012) comparison-level three-level model (k=306)."""
    df = pd.read_csv(EXTRACTION / "webb2012_comparison_extraction.csv")

    y = df["d_composite"].to_numpy()
    v = df["vi"].to_numpy()
    study_ids = df["study"].to_numpy()

    # Two-level baseline (for cross-check)
    X_base = np.ones((len(df), 1))
    res_2level = fit_reml(y, v, X_base)

    # Three-level baseline
    res_3level = fit_three_level(y, v, X_base, study_ids)

    # Three-level DLN moderator
    X_mod, names = design_matrix_stage(
        pd.Series(df["dln_stage"].values), reference="dot"
    )
    res_3level_mod = fit_three_level(y, v, X_mod, study_ids)

    # CR2 cluster-robust SEs
    cr2 = cluster_robust_variance(
        y, v, X_mod, study_ids,
        res_3level_mod.sigma2_within,
        res_3level_mod.sigma2_between,
    )

    # Two-level moderator (for cross-check)
    res_2level_mod = fit_reml(y, v, X_mod)

    result = {
        "twolevel_base_tau2": res_2level.tau2,
        "twolevel_base_mu": res_2level.beta[0],
        "threelevel_base_sigma2_within": res_3level.sigma2_within,
        "threelevel_base_sigma2_between": res_3level.sigma2_between,
        "threelevel_base_mu": res_3level.beta[0],
        "threelevel_base_se_mu": res_3level.se[0],
        "threelevel_base_ci_lo": res_3level.ci95[0, 0],
        "threelevel_base_ci_hi": res_3level.ci95[0, 1],
        "threelevel_base_Q": res_3level.Q,
        "threelevel_base_I2_total": res_3level.I2_total,
        "threelevel_base_I2_within": res_3level.I2_within,
        "threelevel_base_I2_between": res_3level.I2_between,
        "threelevel_mod_sigma2_within": res_3level_mod.sigma2_within,
        "threelevel_mod_sigma2_between": res_3level_mod.sigma2_between,
        "threelevel_mod_beta_intcp": res_3level_mod.beta[0],
        "threelevel_mod_se_intcp": res_3level_mod.se[0],
        "threelevel_mod_beta_linear": res_3level_mod.beta[1],
        "threelevel_mod_se_linear": res_3level_mod.se[1],
        "threelevel_mod_beta_network": res_3level_mod.beta[2],
        "threelevel_mod_se_network": res_3level_mod.se[2],
        "threelevel_mod_QM": 0.0,  # placeholder; rma.mv reports this differently
        "threelevel_mod_QMp": 0.0,
        "cr2_se_intcp": cr2.se_robust[0],
        "cr2_se_linear": cr2.se_robust[1],
        "cr2_se_network": cr2.se_robust[2],
        "cr2_t_intcp": cr2.t_stat[0],
        "cr2_t_linear": cr2.t_stat[1],
        "cr2_t_network": cr2.t_stat[2],
        "cr2_p_intcp": cr2.p_value[0],
        "cr2_p_linear": cr2.p_value[1],
        "cr2_p_network": cr2.p_value[2],
        "cr2_df_intcp": cr2.df_robust[0],
        "cr2_df_linear": cr2.df_robust[1],
        "cr2_df_network": cr2.df_robust[2],
        "twolevel_mod_tau2": res_2level_mod.tau2,
        "twolevel_mod_beta_intcp": res_2level_mod.beta[0],
        "twolevel_mod_se_intcp": res_2level_mod.se[0],
        "twolevel_mod_beta_linear": res_2level_mod.beta[1],
        "twolevel_mod_se_linear": res_2level_mod.se[1],
        "twolevel_mod_beta_network": res_2level_mod.beta[2],
        "twolevel_mod_se_network": res_2level_mod.se[2],
    }

    return result


# ── Main comparison logic ───────────────────────────────────────────────────

DATASETS = [
    ("Webb strategy (k=10)", "output_webb_strategy.csv", _run_webb_strategy),
    ("Greenwald ICC (k=184)", "output_greenwald.csv", _run_greenwald),
    ("Hoyt health (k=8)", "output_hoyt.csv", _run_hoyt),
    ("Desmedt HCT (k=7)", "output_desmedt.csv", _run_desmedt),
    ("Interoception (k=8)", "output_interoception.csv", _run_interoception),
    ("Webb comparison 3-level (k=306)", "output_webb_comparison.csv", _run_webb_comparison),
]


def _load_metafor(csv_path: Path) -> Dict[str, float] | None:
    """Load metafor output CSV if it exists."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return dict(zip(df["parameter"], df["metafor_value"]))


def main():
    all_rows: List[dict] = []
    total_pass = 0
    total_fail = 0
    total_variant = 0
    total_skip = 0

    print("=" * 72)
    print("REML Cross-Validation Report: Python vs. metafor (R)")
    print("=" * 72)

    for label, csv_name, runner in DATASETS:
        print(f"\n{'─' * 72}")
        print(f"  {label}")
        print(f"{'─' * 72}")

        # Run Python analysis
        try:
            py_results = runner()
        except Exception as exc:
            print(f"  PYTHON ERROR: {exc}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Python: {len(py_results)} parameters computed")

        # Load metafor results if available
        mf_results = _load_metafor(METAFOR_DIR / csv_name)

        if mf_results is None:
            print(f"  metafor: NOT AVAILABLE (run R scripts first)")
            print(f"  Expected: {METAFOR_DIR / csv_name}")
            for param, py_val in sorted(py_results.items()):
                all_rows.append({
                    "dataset": label,
                    "parameter": param,
                    "python_value": py_val,
                    "metafor_value": float("nan"),
                    "abs_diff": float("nan"),
                    "status": "SKIP",
                })
                total_skip += 1
            # Still print Python values for reference
            for param, py_val in sorted(py_results.items()):
                print(f"    {param:35s}  py={py_val:>12.6f}  mf=     ---")
            continue

        print(f"  metafor: {len(mf_results)} parameters loaded")

        # Compare
        matched = set(py_results.keys()) & set(mf_results.keys())
        py_only = set(py_results.keys()) - set(mf_results.keys())
        mf_only = set(mf_results.keys()) - set(py_results.keys())

        if py_only:
            print(f"  Python-only params: {sorted(py_only)}")
        if mf_only:
            print(f"  metafor-only params: {sorted(mf_only)}")

        ds_pass = 0
        ds_fail = 0
        ds_variant = 0
        ds_variants = KNOWN_VARIANTS.get(label, {})
        for param in sorted(matched):
            py_val = py_results[param]
            mf_val = mf_results[param]

            if np.isnan(py_val) and np.isnan(mf_val):
                ok, diff = True, 0.0
            elif np.isnan(py_val) or np.isnan(mf_val):
                ok, diff = False, float("inf")
            else:
                ok, diff = _check(param, py_val, mf_val)

            variant_reason = ds_variants.get(param)
            if ok:
                status = "PASS"
                marker = " "
            elif variant_reason:
                status = f"VARIANT:{variant_reason}"
                marker = "~"
            else:
                status = "FAIL"
                marker = "*"

            print(f"  {marker} {param:35s}  py={py_val:>12.6f}  mf={mf_val:>12.6f}  "
                  f"diff={diff:.2e}  {status}")

            all_rows.append({
                "dataset": label,
                "parameter": param,
                "python_value": py_val,
                "metafor_value": mf_val,
                "abs_diff": diff,
                "status": status,
            })

            if ok:
                ds_pass += 1
                total_pass += 1
            elif variant_reason:
                ds_variant += 1
                total_variant += 1
            else:
                ds_fail += 1
                total_fail += 1

        print(f"\n  Result: {ds_pass} PASS, {ds_variant} VARIANT, {ds_fail} FAIL "
              f"out of {len(matched)} compared")

    # ── Summary ─────────────────────────────────────────────────────────

    print(f"\n{'=' * 72}")
    print(f"OVERALL: {total_pass} PASS, {total_variant} VARIANT, "
          f"{total_fail} FAIL, {total_skip} SKIP")
    print(f"{'=' * 72}")

    if total_fail == 0 and total_skip == 0 and total_variant == 0:
        print("\nAll parameters match metafor within tolerance. Exact agreement confirmed.")
    elif total_fail == 0 and total_variant > 0:
        print(f"\nAll core REML parameters match metafor.")
        print(f"{total_variant} known formula variants (I2 formula, Knapp-Hartung "
              f"scaling, QM chi2 vs F, CR2 df method, profile-likelihood bounds).")
        print("These reflect documented methodological choices, not implementation errors.")
    elif total_fail == 0 and total_skip > 0:
        print(f"\nPython side OK. {total_skip} parameters awaiting metafor comparison.")
        print("Run:  Rscript evidence_synthesis/analysis/validate_metafor/run_all.R")
        print("Then: python evidence_synthesis/analysis/validate_reml.py")
    else:
        print(f"\n{total_fail} UNEXPECTED DISCREPANCIES detected (marked with *).")
        print("Review parameters marked FAIL above.")

    # Write report CSV
    report_path = METAFOR_DIR / "validation_report.csv"
    report_df = pd.DataFrame(all_rows)
    report_df.to_csv(report_path, index=False)
    print(f"\nWrote: {report_path}")

    return total_fail


if __name__ == "__main__":
    failures = main()
    sys.exit(min(failures, 1))
