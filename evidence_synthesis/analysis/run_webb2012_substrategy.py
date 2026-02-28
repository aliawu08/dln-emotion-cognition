"""Webb et al. (2012) sub-strategy-level DLN moderator analysis (k=15).

Expands the original k=10 strategy-family analysis to k=15 sub-strategies
extracted directly from Webb Table 3 (Overall column).

This script:
  1. Fits baseline (intercept-only) and DLN-stage moderator meta-regressions.
  2. Compares DLN vs. Gross process model as competing moderators.
  3. Runs a Monte Carlo permutation test (10,000 iterations) for each moderator.
  4. Reports heterogeneity reduction, AICc, and permutation p-values.

Outputs:
  - evidence_synthesis/outputs/tables/webb2012_substrategy_results.csv
  - evidence_synthesis/outputs/tables/webb2012_substrategy_competing.csv
  - evidence_synthesis/outputs/figures/webb2012_substrategy_forest.png
  - evidence_synthesis/outputs/figures/webb2012_substrategy_permutation.png

Usage:
  python evidence_synthesis/analysis/run_webb2012_substrategy.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import (
    aicc,
    compute_qm,
    design_matrix_categorical,
    fit_reml,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "webb2012_substrategy_extraction.csv"
OUT_DIR_T = ROOT / "evidence_synthesis" / "outputs" / "tables"
OUT_DIR_F = ROOT / "evidence_synthesis" / "outputs" / "figures"


# -----------------------------------------------------------------------
# Permutation test (Monte Carlo for k=15)
# -----------------------------------------------------------------------

def monte_carlo_permutation(
    y: np.ndarray,
    v: np.ndarray,
    observed_tau2: float,
    n_groups: int,
    n_iter: int = 10_000,
    seed: int = 2026,
) -> tuple[np.ndarray, float]:
    """Monte Carlo permutation test for categorical moderator.

    Randomly assigns k items to n_groups (ensuring all groups non-empty)
    and fits REML meta-regression.  Returns distribution of tau-squared
    and permutation p-value.
    """
    rng = np.random.default_rng(seed)
    k = len(y)
    perm_tau2 = []

    for _ in range(n_iter):
        # Random assignment ensuring all groups are non-empty
        while True:
            assignment = rng.integers(0, n_groups, size=k)
            if len(set(assignment)) == n_groups:
                break

        # Build design matrix
        groups = sorted(set(assignment))
        X = np.ones((k, len(groups)))
        for col_idx, g in enumerate(groups[1:], start=1):
            X[:, col_idx] = (assignment == g).astype(float)

        try:
            res = fit_reml(y, v, X)
            perm_tau2.append(res.tau2)
        except Exception:
            continue

    perm_tau2 = np.array(perm_tau2)
    p_value = float(np.mean(perm_tau2 <= observed_tau2 + 1e-10))
    return perm_tau2, p_value


# -----------------------------------------------------------------------
# Moderator fitting helper
# -----------------------------------------------------------------------

def fit_moderator(
    y: np.ndarray,
    v: np.ndarray,
    codes: pd.Series,
    label: str,
    n_perm: int = 10_000,
) -> dict:
    """Fit baseline + moderator and return comparison metrics."""
    X_base = np.ones((len(y), 1))
    res_base = fit_reml(y, v, X_base)

    ref = sorted(codes.unique())[0]
    X_mod, names = design_matrix_categorical(codes, reference=ref)
    res_mod = fit_reml(y, v, X_mod)

    tau2_base = res_base.tau2
    tau2_mod = res_mod.tau2
    pct_red = (tau2_base - tau2_mod) / tau2_base * 100 if tau2_base > 0 else 0.0

    aicc_base = aicc(y, v, X_base, res_base.tau2)
    aicc_mod = aicc(y, v, X_mod, res_mod.tau2)

    qm = compute_qm(y, v, X_base, X_mod)

    # Permutation test
    n_levels = len(codes.unique())
    perm_tau2, perm_p = monte_carlo_permutation(
        y, v, tau2_mod, n_levels, n_iter=n_perm,
    )

    return {
        "moderator": label,
        "n_levels": n_levels,
        "tau2_base": tau2_base,
        "tau2_mod": tau2_mod,
        "pct_reduction": pct_red,
        "aicc_base": aicc_base,
        "aicc_mod": aicc_mod,
        "QM": qm.QM,
        "QM_df": qm.df,
        "QM_p": qm.p,
        "perm_p": perm_p,
        "perm_tau2": perm_tau2,
        "beta": res_mod.beta,
        "se": res_mod.se,
        "ci95": res_mod.ci95,
        "names": names,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    df = pd.read_csv(DATA)
    y = df["d_plus"].to_numpy()
    v = df["vi"].to_numpy()
    k = len(df)

    print(f"Webb sub-strategy analysis: k={k} sub-strategies")
    print(f"DLN partition:  dot={sum(df['dln_stage_code']=='dot')}, "
          f"linear={sum(df['dln_stage_code']=='linear')}, "
          f"network={sum(df['dln_stage_code']=='network')}")
    # ---- Fit competing moderators ----
    moderators = {
        "DLN stage": df["dln_stage_code"],
        "Gross process model": df["gross_process"],
    }

    results = []
    perm_distributions: Dict[str, np.ndarray] = {}

    for label, codes in moderators.items():
        print(f"\nFitting: {label} ...")
        res = fit_moderator(y, v, codes, label)
        results.append(res)
        perm_distributions[label] = res["perm_tau2"]

        flag = " ***" if label == "DLN stage" else ""
        print(f"  tau2={res['tau2_mod']:.4f}  "
              f"reduction={res['pct_reduction']:.1f}%  "
              f"QM({res['QM_df']})={res['QM']:.2f} p={res['QM_p']:.4f}  "
              f"AICc={res['aicc_mod']:.2f}  "
              f"perm_p={res['perm_p']:.4f}{flag}")
        for i, name in enumerate(res["names"]):
            print(f"    {name}: b={res['beta'][i]:.4f} "
                  f"[{res['ci95'][i,0]:.4f}, {res['ci95'][i,1]:.4f}]")

    # ---- Comparison table ----
    comp_rows = []
    for res in results:
        comp_rows.append({
            "moderator": res["moderator"],
            "n_levels": res["n_levels"],
            "tau2_base": round(res["tau2_base"], 6),
            "tau2_mod": round(res["tau2_mod"], 6),
            "pct_reduction": round(res["pct_reduction"], 1),
            "QM": round(res["QM"], 2),
            "QM_df": res["QM_df"],
            "QM_p": round(res["QM_p"], 4),
            "aicc": round(res["aicc_mod"], 2),
            "perm_p": round(res["perm_p"], 4),
        })
    comp_df = pd.DataFrame(comp_rows)

    OUT_DIR_T.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(OUT_DIR_T / "webb2012_substrategy_competing.csv", index=False)

    # ---- Detailed DLN results table ----
    dln_res = results[0]
    detail_rows = []
    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    detail_rows.append({
        "model": "baseline",
        "k": k,
        "tau2": round(res_base.tau2, 6),
        "I2": round(res_base.I2, 3),
        "Q": round(res_base.Q, 2),
        "parameter": "mu",
        "estimate": round(res_base.beta[0], 4),
        "se": round(res_base.se[0], 4),
        "ci_lo": round(res_base.ci95[0, 0], 4),
        "ci_hi": round(res_base.ci95[0, 1], 4),
    })
    for i, name in enumerate(dln_res["names"]):
        detail_rows.append({
            "model": "DLN-stage moderator",
            "k": k,
            "tau2": round(dln_res["tau2_mod"], 6),
            "I2": 0,
            "Q": 0,
            "parameter": name,
            "estimate": round(dln_res["beta"][i], 4),
            "se": round(dln_res["se"][i], 4),
            "ci_lo": round(dln_res["ci95"][i, 0], 4),
            "ci_hi": round(dln_res["ci95"][i, 1], 4),
        })
    detail_rows.append({
        "model": "heterogeneity_reduction",
        "k": k,
        "tau2": round(dln_res["tau2_base"] - dln_res["tau2_mod"], 6),
        "I2": round(dln_res["pct_reduction"], 1),
        "Q": 0,
        "parameter": "delta_tau2",
        "estimate": round(dln_res["tau2_base"] - dln_res["tau2_mod"], 6),
        "se": 0,
        "ci_lo": 0,
        "ci_hi": 0,
    })
    pd.DataFrame(detail_rows).to_csv(
        OUT_DIR_T / "webb2012_substrategy_results.csv", index=False
    )

    # ---- Forest plot ----
    OUT_DIR_F.mkdir(parents=True, exist_ok=True)
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (_, row) in enumerate(df.iterrows()):
        color = stage_colors[row["dln_stage_code"]]
        ci_lo = row["d_plus"] - 1.96 * row["se_d"]
        ci_hi = row["d_plus"] + 1.96 * row["se_d"]
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=2)
        ax.plot(row["d_plus"], i, "s", color=color, markersize=7)

    labels = [f"{row['webb_code']} {row['strategy_sub'].replace('_', ' ')}"
              for _, row in df.iterrows()]
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Effect size (d+)")
    ax.set_title(f"Webb (2012) sub-strategies by DLN stage (k={k})")

    # Legend
    for stage, color in stage_colors.items():
        ax.plot([], [], "s", color=color, label=stage, markersize=8)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR_F / "webb2012_substrategy_forest.png", dpi=200)
    plt.close(fig)

    # ---- Permutation distribution plot ----
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    for ax, res in zip(axes, results):
        tau2 = res["perm_tau2"]
        ax.hist(tau2, bins=60, color="#95a5a6", edgecolor="white", alpha=0.85)
        ax.axvline(res["tau2_mod"], color="#e74c3c", linewidth=2,
                   linestyle="--", label=f"Observed τ²={res['tau2_mod']:.4f}")
        ax.set_title(f"{res['moderator']}\nperm p={res['perm_p']:.4f}", fontsize=10)
        ax.set_xlabel("Residual τ²")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle(f"Webb (2012) k={k}: permutation tests (10,000 iterations)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR_F / "webb2012_substrategy_permutation.png", dpi=200)
    plt.close(fig)

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY: Webb sub-strategy competing moderators (k=15)")
    print(f"{'='*70}")
    best_red = max(results, key=lambda r: r["pct_reduction"])
    best_aicc = min(results, key=lambda r: r["aicc_mod"])
    print(f"  Best τ² reduction: {best_red['moderator']} ({best_red['pct_reduction']:.1f}%)")
    print(f"  Best AICc:         {best_aicc['moderator']} ({best_aicc['aicc_mod']:.2f})")
    for res in results:
        print(f"  {res['moderator']} perm p: {res['perm_p']:.4f}")

    if best_red["moderator"] == "DLN stage":
        print("  --> DLN achieves highest heterogeneity reduction")
    if best_aicc["moderator"] == "DLN stage":
        print("  --> DLN achieves lowest AICc")

    print(f"\nWrote outputs to:")
    print(f"  {OUT_DIR_T / 'webb2012_substrategy_results.csv'}")
    print(f"  {OUT_DIR_T / 'webb2012_substrategy_competing.csv'}")
    print(f"  {OUT_DIR_F / 'webb2012_substrategy_forest.png'}")
    print(f"  {OUT_DIR_F / 'webb2012_substrategy_permutation.png'}")


if __name__ == "__main__":
    main()
