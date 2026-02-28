"""Greenwald et al. (2009) permutation robustness test for four-level DLN coding.

Tests whether the tau-squared reduction from four-level DLN coding is
unlikely to arise from any arbitrary partition with the same group sizes.

Method:
  1. Record the DLN-coded tau-squared from the four-level model.
  2. Generate 10,000 random permutations that reassign the 184 samples to
     four groups with the same sizes as the DLN coding (33, 97, 43, 11).
  3. For each permutation, fit the same REML meta-regression and record
     the residual tau-squared.
  4. Report the permutation p-value (proportion of random partitions
     achieving tau-squared <= DLN tau-squared).

Outputs:
  - evidence_synthesis/outputs/tables/greenwald2009_topology_permutation.csv
  - evidence_synthesis/outputs/figures/greenwald2009_topology_permutation.png

Usage:
  python evidence_synthesis/analysis/permutation_greenwald2009.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_pipeline import fit_reml, design_matrix_categorical
from run_greenwald2009_topology import (
    DATA, DOT_SAMPLES, NETWORK_SAMPLES, LINEAR_PLUS_TOPICS,
    STAGE_ORDER, assign_topology_stage, prepare_data,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "greenwald2009_topology_permutation.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "greenwald2009_topology_permutation.png"

N_PERMUTATIONS = 10_000
SEED = 20260219


def run_permutation_test(y, v, dln_tau2, group_sizes, n_perm, seed):
    """Monte Carlo permutation test with fixed group sizes.

    Parameters
    ----------
    y : array, shape (k,)
        Effect sizes (Fisher z).
    v : array, shape (k,)
        Sampling variances.
    dln_tau2 : float
        Tau-squared from the actual DLN coding.
    group_sizes : list of int
        Sizes of the four groups in the DLN coding.
    n_perm : int
        Number of random permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    perm_tau2 : ndarray
        Tau-squared for each random permutation.
    p_value : float
        Fraction of permutations achieving tau2 <= dln_tau2.
    """
    k = len(y)
    rng = np.random.default_rng(seed)
    perm_tau2 = np.full(n_perm, np.nan)

    # Pre-compute group labels matching the sizes
    labels = []
    for i, size in enumerate(group_sizes):
        labels.extend([i] * size)
    labels = np.array(labels)
    assert len(labels) == k

    for t in range(n_perm):
        # Random permutation of sample indices
        perm_idx = rng.permutation(k)
        stage_assignment = labels[perm_idx]

        # Build design matrix (intercept + 3 dummies for 4 groups)
        X = np.ones((k, len(group_sizes)))
        for col in range(1, len(group_sizes)):
            X[:, col] = (stage_assignment == col).astype(float)

        try:
            res = fit_reml(y, v, X)
            perm_tau2[t] = res.tau2
        except Exception:
            continue

    perm_tau2 = perm_tau2[~np.isnan(perm_tau2)]
    p_value = float(np.mean(perm_tau2 <= dln_tau2))
    return perm_tau2, p_value


def main():
    raw = pd.read_csv(DATA)
    df = prepare_data(raw)

    y = df["yi_icc"].to_numpy()
    v = df["vi_icc"].to_numpy()

    # Get the DLN-coded tau-squared (four-level model)
    X_dln, _ = design_matrix_categorical(df["dln_stage"], reference="dot")
    res_dln = fit_reml(y, v, X_dln)
    dln_tau2 = res_dln.tau2

    # Group sizes from the actual coding
    group_sizes = [
        (df["dln_stage"] == stage).sum() for stage in STAGE_ORDER
    ]
    print(f"DLN four-level tau-squared: {dln_tau2:.6f}")
    print(f"Group sizes: {dict(zip(STAGE_ORDER, group_sizes))}")
    print(f"Running {N_PERMUTATIONS:,} random permutations (seed={SEED})...")

    perm_tau2, p_value = run_permutation_test(
        y, v, dln_tau2, group_sizes, N_PERMUTATIONS, SEED
    )

    print(f"Permutations completed: {len(perm_tau2):,}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"DLN percentile rank: {(1 - p_value) * 100:.1f}th percentile")
    print(f"Median permutation tau2: {np.median(perm_tau2):.6f}")
    print(f"Min permutation tau2: {np.min(perm_tau2):.6f}")

    # --- Output table ---
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "dln_tau2": round(dln_tau2, 6),
        "n_permutations": len(perm_tau2),
        "seed": SEED,
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
    ax.hist(perm_tau2, bins=80, color="#95a5a6", edgecolor="white", alpha=0.85)
    ax.axvline(dln_tau2, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"DLN coding ($\\tau^2$ = {dln_tau2:.4f})")
    ax.set_xlabel("Residual $\\tau^2$ (random four-way partition)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Greenwald (2009) permutation test: {len(perm_tau2):,} random partitions\n"
        f"DLN coding outperforms {(1 - p_value) * 100:.1f}% of random assignments "
        f"(p = {p_value:.4f})"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")


if __name__ == "__main__":
    main()
