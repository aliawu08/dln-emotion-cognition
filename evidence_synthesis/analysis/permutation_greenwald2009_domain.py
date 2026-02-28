"""Greenwald (2009) domain-level permutation test for four-level DLN coding.

Tests whether the tau-squared reduction from DLN coding is unlikely to arise
from any arbitrary assignment of the 9 topic domains to 4 non-empty groups.

Unlike the study-level permutation (permutation_greenwald2009.py) which shuffles
184 individual studies, this test respects the domain structure: all studies
within a domain are always assigned to the same group.  The effective sample
size is therefore 9 domains, not 184 studies.

Method:
  1. Enumerate all S(9,4) * 4! surjective functions from 9 domains to 4 groups,
     deduplicated to S(9,4) = 7,770 unique partitions via canonicalization.
  2. For each partition, assign every study to its domain's group, build a
     four-level design matrix, and fit REML meta-regression.
  3. Compare the DLN partition's residual tau-squared to the distribution.
  4. Report the permutation p-value.

Outputs:
  - evidence_synthesis/outputs/tables/greenwald2009_domain_permutation.csv
  - evidence_synthesis/outputs/figures/greenwald2009_domain_permutation.png

Usage:
  python evidence_synthesis/analysis/permutation_greenwald2009_domain.py
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_pipeline import fit_reml
from run_greenwald2009_topology import DATA, prepare_data

ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "greenwald2009_domain_permutation.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "greenwald2009_domain_permutation.png"


# ---------------------------------------------------------------------------
# Surjective assignment enumeration (reused from permutation_webb2012.py)
# ---------------------------------------------------------------------------

def generate_surjective_assignments(n: int, k: int):
    """Yield all surjective functions from {0,...,n-1} -> {0,...,k-1}.

    Each item is assigned to one of k groups, and every group is non-empty.
    """
    for assignment in product(range(k), repeat=n):
        if len(set(assignment)) == k:
            yield assignment


def canonicalize(assignment: tuple) -> tuple:
    """Map an assignment to its canonical form by relabelling groups
    in order of first appearance."""
    mapping = {}
    next_label = 0
    result = []
    for g in assignment:
        if g not in mapping:
            mapping[g] = next_label
            next_label += 1
        result.append(mapping[g])
    return tuple(result)


# ---------------------------------------------------------------------------
# Domain-level permutation test
# ---------------------------------------------------------------------------

def build_design_matrix_from_domain_assignment(
    domain_labels: list[str],
    domain_assignment: tuple,
    study_domains: np.ndarray,
) -> np.ndarray:
    """Build intercept + dummy design matrix from a domain-level group assignment.

    Parameters
    ----------
    domain_labels : list of str
        The 9 unique domain names in sorted order.
    domain_assignment : tuple of int
        Group assignment for each domain (length = n_domains).
    study_domains : array of str, shape (k,)
        Domain label for each study.

    Returns
    -------
    X : ndarray, shape (k, n_groups)
        Intercept + dummy-coded design matrix.  Group 0 is reference.
    """
    # Map each domain to its assigned group
    domain_to_group = {d: g for d, g in zip(domain_labels, domain_assignment)}

    # Map each study to its group
    k = len(study_domains)
    study_groups = np.array([domain_to_group[d] for d in study_domains])

    n_groups = len(set(domain_assignment))
    X = np.ones((k, n_groups))
    for col in range(1, n_groups):
        X[:, col] = (study_groups == col).astype(float)
    return X


def run_domain_permutation_test(
    y: np.ndarray,
    v: np.ndarray,
    study_domains: np.ndarray,
    dln_tau2: float,
    n_groups: int = 4,
):
    """Exhaustive domain-level permutation test.

    Parameters
    ----------
    y : array, shape (k,)
        Effect sizes (Fisher z).
    v : array, shape (k,)
        Sampling variances.
    study_domains : array of str, shape (k,)
        Domain label per study.
    dln_tau2 : float
        Tau-squared from the actual DLN coding.
    n_groups : int
        Number of groups (4 for four-level DLN).

    Returns
    -------
    perm_tau2 : ndarray
        Tau-squared for each unique partition.
    p_value : float
        Fraction of partitions achieving tau2 <= dln_tau2.
    n_partitions : int
        Number of unique partitions tested.
    """
    domain_labels = sorted(set(study_domains))
    n_domains = len(domain_labels)

    seen = set()
    perm_tau2 = []

    for assignment in generate_surjective_assignments(n_domains, n_groups):
        canon = canonicalize(assignment)
        if canon in seen:
            continue
        seen.add(canon)

        X = build_design_matrix_from_domain_assignment(
            domain_labels, assignment, study_domains
        )
        try:
            res = fit_reml(y, v, X)
            perm_tau2.append(res.tau2)
        except Exception:
            continue

    perm_tau2 = np.array(perm_tau2)
    p_value = float(np.mean(perm_tau2 <= dln_tau2))
    return perm_tau2, p_value, len(seen)


def get_dln_domain_assignment(df: pd.DataFrame) -> tuple:
    """Extract the actual DLN assignment of domains to stages.

    Returns the assignment as a tuple of group indices (sorted by domain name),
    matching the order used in the permutation test.
    """
    domain_labels = sorted(df["topic"].unique())

    # Get the DLN stage for each domain
    # (all studies within a domain have the same modal stage assignment)
    domain_stages = {}
    for domain in domain_labels:
        stages = df.loc[df["topic"] == domain, "dln_stage"].unique()
        # Domains may have mixed stages (e.g., Race has dot + linear_plus).
        # Use the modal assignment.
        stage_counts = df.loc[df["topic"] == domain, "dln_stage"].value_counts()
        domain_stages[domain] = stage_counts.index[0]

    # Map the 4 DLN stages to group indices 0-3
    stage_to_group = {}
    next_group = 0
    assignment = []
    for domain in domain_labels:
        stage = domain_stages[domain]
        if stage not in stage_to_group:
            stage_to_group[stage] = next_group
            next_group += 1
        assignment.append(stage_to_group[stage])

    return tuple(assignment), domain_labels, domain_stages


def main():
    raw = pd.read_csv(DATA)
    df = prepare_data(raw)

    y = df["yi_icc"].to_numpy()
    v = df["vi_icc"].to_numpy()
    study_domains = df["topic"].to_numpy()

    domain_labels = sorted(df["topic"].unique())
    n_domains = len(domain_labels)

    # Get DLN-coded tau-squared using the actual four-level design matrix
    from meta_pipeline import design_matrix_categorical
    X_dln, _ = design_matrix_categorical(df["dln_stage"], reference="dot")
    res_dln = fit_reml(y, v, X_dln)
    dln_tau2 = res_dln.tau2

    print(f"Greenwald (2009): k={len(df)} studies, {n_domains} domains")
    print(f"DLN four-level tau-squared: {dln_tau2:.6f}")
    print(f"\nDomains and their DLN stage assignments:")
    _, _, domain_stages = get_dln_domain_assignment(df)
    for domain in domain_labels:
        k_d = (df["topic"] == domain).sum()
        print(f"  {domain:<20s}: {domain_stages[domain]:<14s} (k={k_d})")

    print(f"\nEnumerating all S({n_domains},4) = 7,770 unique partitions...")

    perm_tau2, p_value, n_partitions = run_domain_permutation_test(
        y, v, study_domains, dln_tau2, n_groups=4
    )

    print(f"Partitions tested: {n_partitions:,}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"DLN percentile rank: {(1 - p_value) * 100:.1f}th percentile")
    print(f"Median permutation tau2: {np.median(perm_tau2):.6f}")
    print(f"Min permutation tau2: {np.min(perm_tau2):.6f}")

    # --- Output table ---
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{
        "analysis": "domain_level_permutation",
        "n_domains": n_domains,
        "n_groups": 4,
        "dln_tau2": round(dln_tau2, 6),
        "n_partitions": n_partitions,
        "median_perm_tau2": round(float(np.median(perm_tau2)), 6),
        "mean_perm_tau2": round(float(np.mean(perm_tau2)), 6),
        "min_perm_tau2": round(float(np.min(perm_tau2)), 6),
        "pct_5_perm_tau2": round(float(np.percentile(perm_tau2, 5)), 6),
        "p_value": round(p_value, 4),
        "percentile_rank": round((1 - p_value) * 100, 1),
    }])
    summary.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote: {OUT_TABLE}")

    # --- Distribution plot ---
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(perm_tau2, bins=80, color="#95a5a6", edgecolor="white", alpha=0.85)
    ax.axvline(dln_tau2, color="#e74c3c", linewidth=2, linestyle="--",
               label=f"DLN coding ($\\tau^2$ = {dln_tau2:.4f})")
    ax.set_xlabel("Residual $\\tau^2$ (random domain-to-group partition)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Greenwald (2009) domain-level permutation: {n_partitions:,} partitions\n"
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
