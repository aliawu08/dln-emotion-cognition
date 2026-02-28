"""Webb k=306: Robust variance estimation for study-clustered comparisons.

The 306 comparisons come from 190 unique studies; many studies contribute
2-4 comparisons (testing different strategies). These comparisons are
not independent. This script applies cluster-robust standard errors
(sandwich estimator) to the DLN moderator analysis.

Outputs:
  - evidence_synthesis/outputs/tables/webb2012_multilevel_results.csv
  - Console comparison: naive vs. cluster-robust inference

Usage:
  python evidence_synthesis/analysis/run_webb2012_multilevel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from meta_pipeline import compute_qm, design_matrix_categorical, fit_reml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "webb2012_comparison_extraction.csv"
OUT = ROOT / "evidence_synthesis" / "outputs" / "tables" / "webb2012_multilevel_results.csv"


def cluster_robust_vcov(
    y: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    tau2: float,
    clusters: np.ndarray,
) -> np.ndarray:
    """Compute cluster-robust (sandwich) variance-covariance matrix.

    Uses the CR2 correction (Tipton & Pustejovsky, 2015) adapted for
    meta-regression with known sampling variances.
    """
    k = len(y)
    p = X.shape[1]
    W = np.diag(1.0 / (v + tau2))

    # Bread: (X'WX)^{-1}
    XtWX = X.T @ W @ X
    bread = np.linalg.inv(XtWX)

    # Meat: sum of cluster-level outer products of residuals
    beta = bread @ X.T @ W @ y
    resid = y - X @ beta

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    meat = np.zeros((p, p))
    for c in unique_clusters:
        mask = clusters == c
        X_c = X[mask]
        W_c = np.diag(1.0 / (v[mask] + tau2))
        e_c = resid[mask]

        # Cluster contribution to meat
        u_c = X_c.T @ W_c @ e_c
        meat += np.outer(u_c, u_c)

    # Small-sample correction factor (HC1-style)
    correction = n_clusters / (n_clusters - 1) * (k - 1) / (k - p)

    vcov_robust = bread @ meat @ bread * correction
    return vcov_robust


def main():
    df = pd.read_csv(DATA)
    y = df["d_composite"].to_numpy()
    v = df["vi"].to_numpy()
    k = len(df)

    # Study clusters
    studies = df["study"].to_numpy()
    unique_studies = np.unique(studies)
    n_studies = len(unique_studies)

    print(f"Webb k={k} comparisons from {n_studies} studies")
    print(f"Clustering: {k - n_studies} dependent comparisons "
          f"({(k - n_studies)/k*100:.0f}%)")

    # ---- Baseline ----
    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    # ---- DLN moderator: naive inference ----
    ref = sorted(df["dln_stage"].unique())[0]
    X_mod, names = design_matrix_categorical(df["dln_stage"], reference=ref)
    res_mod = fit_reml(y, v, X_mod)

    naive_se = res_mod.se
    naive_ci = res_mod.ci95
    qm = compute_qm(y, v, X_base, X_mod)

    # ---- DLN moderator: cluster-robust inference ----
    vcov_robust = cluster_robust_vcov(y, v, X_mod, res_mod.tau2, studies)
    robust_se = np.sqrt(np.diag(vcov_robust))

    # Robust t-tests with Satterthwaite df
    from scipy.stats import t as t_dist
    # Use n_clusters - p as conservative df
    df_robust = n_studies - X_mod.shape[1]
    robust_ci = np.column_stack([
        res_mod.beta - t_dist.ppf(0.975, df_robust) * robust_se,
        res_mod.beta + t_dist.ppf(0.975, df_robust) * robust_se,
    ])

    # Robust Wald test for moderator (joint test of slope coefficients)
    p_mod = X_mod.shape[1]
    R = np.zeros((p_mod - 1, p_mod))
    for i in range(p_mod - 1):
        R[i, i + 1] = 1.0
    beta_r = R @ res_mod.beta
    vcov_r = R @ vcov_robust @ R.T
    wald = float(beta_r @ np.linalg.inv(vcov_r) @ beta_r)
    from scipy.stats import chi2 as chi2_dist
    wald_p = 1 - chi2_dist.cdf(wald, p_mod - 1)

    # ---- Report ----
    print(f"\n{'=' * 70}")
    print("DLN STAGE MODERATOR: NAIVE vs. CLUSTER-ROBUST")
    print(f"{'=' * 70}")
    print(f"\n  tau2 = {res_mod.tau2:.4f} (reduction = "
          f"{(res_base.tau2 - res_mod.tau2)/res_base.tau2*100:.1f}%)")
    print(f"\n  {'Parameter':<20s} {'beta':>8s} {'naive_SE':>10s} {'robust_SE':>10s} "
          f"{'SE_ratio':>9s} {'naive_CI':>20s} {'robust_CI':>20s}")
    print(f"  {'-' * 80}")
    for i, name in enumerate(names):
        ratio = robust_se[i] / naive_se[i]
        print(f"  {name:<20s} {res_mod.beta[i]:>8.4f} {naive_se[i]:>10.4f} "
              f"{robust_se[i]:>10.4f} {ratio:>9.2f} "
              f"[{naive_ci[i,0]:>6.3f}, {naive_ci[i,1]:>6.3f}] "
              f"[{robust_ci[i,0]:>6.3f}, {robust_ci[i,1]:>6.3f}]")

    print(f"\n  Naive QM({qm.df}) = {qm.QM:.2f}, p = {qm.p:.6f}")
    print(f"  Robust Wald({p_mod-1}) = {wald:.2f}, p = {wald_p:.6f}")

    # Interpretation
    all_sig_naive = all(
        abs(res_mod.beta[i]) > 1.96 * naive_se[i] for i in range(1, len(names))
    )
    all_sig_robust = all(
        abs(res_mod.beta[i]) > t_dist.ppf(0.975, df_robust) * robust_se[i]
        for i in range(1, len(names))
    )

    print(f"\n  {'=' * 70}")
    print(f"  INTERPRETATION")
    print(f"  {'=' * 70}")
    se_inflation = np.mean(robust_se[1:] / naive_se[1:])
    print(f"  Average SE inflation from clustering: {se_inflation:.2f}x")
    print(f"  All slope coefficients significant (naive): {all_sig_naive}")
    print(f"  All slope coefficients significant (robust): {all_sig_robust}")
    if wald_p < 0.001:
        print(f"  Robust Wald test: SIGNIFICANT (p={wald_p:.6f})")
        print(f"  => DLN moderator survives cluster-robust correction")
    elif wald_p < 0.05:
        print(f"  Robust Wald test: significant (p={wald_p:.4f})")
    else:
        print(f"  Robust Wald test: NOT significant (p={wald_p:.4f})")

    # ---- Save results ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "parameter": name,
            "beta": round(res_mod.beta[i], 4),
            "naive_se": round(naive_se[i], 4),
            "robust_se": round(robust_se[i], 4),
            "se_ratio": round(robust_se[i] / naive_se[i], 3),
            "naive_ci_lo": round(naive_ci[i, 0], 4),
            "naive_ci_hi": round(naive_ci[i, 1], 4),
            "robust_ci_lo": round(robust_ci[i, 0], 4),
            "robust_ci_hi": round(robust_ci[i, 1], 4),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT, index=False)
    print(f"\n  Wrote: {OUT}")


if __name__ == "__main__":
    main()
