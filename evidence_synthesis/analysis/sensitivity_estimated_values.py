"""Bounded sensitivity analysis for estimated effect sizes.

For each analysis containing estimated (unverified) effect sizes, this script
perturbs the estimated values within plausible bounds (+/-10%, +/-20%, +/-30%)
and reports the proportion of perturbations that preserve the qualitative
pattern (sign ordering, tau-squared reduction direction, dangerous-middle
V-shape).

This addresses the reviewer concern that unverified effect sizes may not
support the reported qualitative conclusions.

Analyses tested:
  - Hoyt (0/8 estimated after Table 3 verification): dangerous-middle V-shape (linear negative)
  - Desmedt (0/7 estimated after text verification): dot |r| > linear |r|

Outputs:
  - evidence_synthesis/outputs/tables/sensitivity_estimated_values.csv

Usage:
  python evidence_synthesis/analysis/sensitivity_estimated_values.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from meta_pipeline import fit_reml, design_matrix_stage
except ModuleNotFoundError:
    from evidence_synthesis.analysis.meta_pipeline import fit_reml, design_matrix_stage


ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "sensitivity_estimated_values.csv"

N_SAMPLES = 1000
RNG = np.random.default_rng(42)
PERTURBATION_LEVELS = [0.10, 0.20, 0.30]


def r_to_fisher_z(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def _load_hoyt():
    path = ROOT / "evidence_synthesis" / "extraction" / "hoyt2024_domain_extraction.csv"
    return pd.read_csv(path)


def _load_desmedt():
    path = ROOT / "evidence_synthesis" / "extraction" / "desmedt2022_criterion_extraction.csv"
    return pd.read_csv(path)


def _test_hoyt_dangerous_middle(df):
    """Test: linear-coded domains negative, dot and network positive (V-shape)."""
    df = df.copy()
    df["z"] = r_to_fisher_z(df["r_pooled"].to_numpy())
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    dot = df[df["dln_stage_code"] == "dot"]
    linear = df[df["dln_stage_code"] == "linear"]
    network = df[df["dln_stage_code"] == "network"]
    if len(dot) == 0 or len(linear) == 0 or len(network) == 0:
        return False
    dot_mean = np.average(dot["z"], weights=1.0 / dot["vi_z"])
    lin_mean = np.average(linear["z"], weights=1.0 / linear["vi_z"])
    net_mean = np.average(network["z"], weights=1.0 / network["vi_z"])
    # V-shape: linear < dot AND linear < network
    return lin_mean < dot_mean and lin_mean < net_mean


def _test_hoyt_tau2_reduction(df):
    """Test: DLN stage reduces tau-squared."""
    df = df.copy()
    df["z"] = r_to_fisher_z(df["r_pooled"].to_numpy())
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    y = df["z"].to_numpy()
    v = df["vi_z"].to_numpy()
    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)
    X_mod, _ = design_matrix_stage(df["dln_stage_code"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)
    return res_mod.tau2 <= res_base.tau2


def _test_desmedt_same_level(df):
    """Test: dot-stage |r| > linear-stage |r| (same-level correspondence)."""
    df = df.copy()
    df["abs_r"] = df["r_pooled"].abs()
    df["z_abs"] = r_to_fisher_z(df["abs_r"].to_numpy())
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    dot = df[df["dln_stage_code"] == "dot"]
    linear = df[df["dln_stage_code"] == "linear"]
    if len(dot) == 0 or len(linear) == 0:
        return False
    dot_mean = np.average(dot["z_abs"], weights=1.0 / dot["vi_z"])
    lin_mean = np.average(linear["z_abs"], weights=1.0 / linear["vi_z"])
    return dot_mean > lin_mean


def _test_desmedt_tau2_reduction(df):
    """Test: DLN stage reduces tau-squared."""
    df = df.copy()
    df["abs_r"] = df["r_pooled"].abs()
    df["z_abs"] = r_to_fisher_z(df["abs_r"].to_numpy())
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    y = df["z_abs"].to_numpy()
    v = df["vi_z"].to_numpy()
    is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)
    X_mod = np.column_stack([np.ones(len(df)), is_dot])
    res_mod = fit_reml(y, v, X_mod)
    return res_mod.tau2 <= res_base.tau2


def _identify_estimated_mask(df, status_col):
    """Return boolean mask of rows with estimated (unverified) values."""
    return df[status_col].str.contains("estimate", case=False, na=False).to_numpy()


def run_sensitivity(analysis_name, df, effect_col, status_col,
                    pattern_test, tau2_test, perturbation_target="effect"):
    """Run bounded sensitivity analysis for one dataset.

    For each perturbation level, generate N_SAMPLES random perturbations
    of the estimated values and test whether the qualitative pattern holds.
    """
    est_mask = _identify_estimated_mask(df, status_col)
    n_estimated = int(est_mask.sum())
    n_total = len(df)

    print(f"\n=== {analysis_name} ===")
    print(f"  {n_estimated} of {n_total} values estimated")

    if n_estimated == 0:
        print("  No estimated values; skipping.")
        return []

    original_values = df[effect_col].to_numpy().copy()
    rows = []

    for pct in PERTURBATION_LEVELS:
        pattern_preserved = 0
        tau2_preserved = 0

        for _ in range(N_SAMPLES):
            df_pert = df.copy()
            perturbed = original_values.copy()
            # Apply random uniform perturbation to estimated values only
            noise = RNG.uniform(-pct, pct, size=n_total)
            perturbed[est_mask] = original_values[est_mask] * (1.0 + noise[est_mask])
            df_pert[effect_col] = perturbed

            try:
                if pattern_test(df_pert):
                    pattern_preserved += 1
                if tau2_test(df_pert):
                    tau2_preserved += 1
            except Exception:
                continue

        pct_pattern = pattern_preserved / N_SAMPLES * 100
        pct_tau2 = tau2_preserved / N_SAMPLES * 100

        print(f"  +/-{pct*100:.0f}%: pattern preserved {pct_pattern:.1f}%, "
              f"tau2 reduction preserved {pct_tau2:.1f}%")

        rows.append({
            "analysis": analysis_name,
            "k_total": n_total,
            "k_estimated": n_estimated,
            "pct_estimated": round(n_estimated / n_total * 100, 0),
            "perturbation_pct": int(pct * 100),
            "n_samples": N_SAMPLES,
            "pattern_preserved_pct": round(pct_pattern, 1),
            "tau2_reduction_preserved_pct": round(pct_tau2, 1),
        })

    return rows


def main():
    all_rows = []

    # --- Hoyt ---
    df_hy = _load_hoyt()
    rows_hy = run_sensitivity(
        "Hoyt (dangerous-middle V-shape)",
        df_hy, "r_pooled", "estimate_status",
        _test_hoyt_dangerous_middle, _test_hoyt_tau2_reduction,
    )
    all_rows.extend(rows_hy)

    # --- Desmedt ---
    df_dm = _load_desmedt()
    rows_dm = run_sensitivity(
        "Desmedt (dot |r| > linear |r|)",
        df_dm, "r_pooled", "estimate_status",
        _test_desmedt_same_level, _test_desmedt_tau2_reduction,
    )
    all_rows.extend(rows_dm)

    # --- Output ---
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote: {OUT_TABLE}")

    # --- Summary ---
    print("\n=== SUMMARY ===")
    if summary.empty:
        print("  All datasets fully verified; no estimated values to perturb.")
    else:
        for analysis in summary["analysis"].unique():
            sub = summary[summary["analysis"] == analysis]
            worst_pattern = sub["pattern_preserved_pct"].min()
            worst_tau2 = sub["tau2_reduction_preserved_pct"].min()
            print(f"  {analysis}:")
            print(f"    Pattern preserved (worst case +/-30%): {worst_pattern:.1f}%")
            print(f"    Tau2 reduction preserved (worst case +/-30%): {worst_tau2:.1f}%")


if __name__ == "__main__":
    main()
