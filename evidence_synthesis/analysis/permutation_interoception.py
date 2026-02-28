"""Interoception--alexithymia permutation robustness test for DLN stage coding.

Demonstrates that the 85% tau-squared reduction from DLN coding is not an
artifact of any arbitrary 3-way partition of 8 interoceptive measure families.

Method:
  1. Enumerate all 3-category assignments of 8 items into 3 non-empty groups
     (Stirling numbers of the second kind: S(8,3) = 966 partitions).
  2. For each assignment, fit the same REML meta-regression and record residual
     tau-squared.
  3. Compare the DLN-coded tau-squared against the permutation distribution.
  4. Report the percentile rank (permutation p-value).

Outputs:
  - evidence_synthesis/outputs/figures/interoception_permutation_dist.png
  - evidence_synthesis/outputs/tables/interoception_permutation_results.csv

Usage:
  python evidence_synthesis/analysis/permutation_interoception.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from meta_pipeline import fit_reml, design_matrix_stage
except ModuleNotFoundError:
    from evidence_synthesis.analysis.meta_pipeline import fit_reml, design_matrix_stage

try:
    from permutation_webb2012 import (
        generate_surjective_assignments,
        canonicalize,
        build_design_matrix_from_assignment,
        run_permutation_test,
    )
except ModuleNotFoundError:
    from evidence_synthesis.analysis.permutation_webb2012 import (
        generate_surjective_assignments,
        canonicalize,
        build_design_matrix_from_assignment,
        run_permutation_test,
    )


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "interoception_measure_extraction.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "interoception_permutation_dist.png"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "interoception_permutation_results.csv"


def r_to_fisher_z(r):
    """Convert correlation to Fisher z."""
    return np.arctanh(np.clip(r, -0.999, 0.999))


def main():
    df = pd.read_csv(DATA)
    n_items = len(df)
    print(f"Loaded {n_items} interoceptive measure families")

    # Convert to Fisher z, matching run_interoception_moderator.py
    df["z"] = r_to_fisher_z(df["r_pooled"].to_numpy())
    df["vi_z"] = 1.0 / (df["N_total"] - 3.0 * df["k"])

    y = df["z"].to_numpy()
    v = df["vi_z"].to_numpy()

    # Get the DLN-coded tau-squared
    X_dln, _ = design_matrix_stage(df["dln_stage_code"], reference="dot")
    res_dln = fit_reml(y, v, X_dln)
    dln_tau2 = res_dln.tau2

    print(f"DLN-coded tau-squared: {dln_tau2:.6f}")
    print(f"Running exhaustive permutation test (S({n_items},3) partitions)...")

    perm_tau2, p_value = run_permutation_test(y, v, dln_tau2, n_groups=3)

    print(f"Total unique partitions tested: {len(perm_tau2)}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"DLN percentile rank: {(1 - p_value) * 100:.1f}th percentile")

    # --- Output table ---
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "analysis": "interoception",
        "k": n_items,
        "n_groups": 3,
        "stirling_number": len(perm_tau2),
        "dln_tau2": round(dln_tau2, 6),
        "median_perm_tau2": round(float(np.median(perm_tau2)), 6),
        "mean_perm_tau2": round(float(np.mean(perm_tau2)), 6),
        "min_perm_tau2": round(float(np.min(perm_tau2)), 6),
        "pct_5_perm_tau2": round(float(np.percentile(perm_tau2, 5)), 6),
        "p_value": round(p_value, 4),
        "percentile_rank": round((1 - p_value) * 100, 1),
    }])
    summary.to_csv(OUT_TABLE, index=False)
    print(f"Wrote: {OUT_TABLE}")

    # --- Distribution plot ---
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_tau2, bins=50, color="#95a5a6", edgecolor="white", alpha=0.85)
    ax.axvline(dln_tau2, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"DLN coding ($\\tau^2$ = {dln_tau2:.4f})")
    ax.set_xlabel("Residual $\\tau^2$ (random 3-way partition)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Interoception permutation test: {len(perm_tau2):,} partitions\n"
        f"DLN coding outperforms {(1 - p_value) * 100:.1f}% of random assignments "
        f"(p = {p_value:.3f})"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")


if __name__ == "__main__":
    main()
