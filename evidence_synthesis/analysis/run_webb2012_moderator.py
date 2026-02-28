"""Webb et al. (2012) DLN-stage moderator analysis.

Re-extracts strategy-family-level effect sizes from Webb, Miles, & Sheeran (2012)
Psychological Bulletin 138(4), codes each for DLN stage, and runs a random-effects
meta-regression testing whether DLN stage explains between-strategy heterogeneity.

Outputs:
- evidence_synthesis/outputs/tables/webb2012_moderator_summary.csv
- evidence_synthesis/outputs/figures/webb2012_stage_forest.png

Usage:
  python evidence_synthesis/analysis/run_webb2012_moderator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import fit_reml, design_matrix_stage, design_matrix_categorical, compute_qm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "webb2012_strategy_extraction.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "webb2012_moderator_summary.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "webb2012_stage_forest.png"


def main():
    df = pd.read_csv(DATA)

    y = df["d_plus"].to_numpy()
    v = df["vi"].to_numpy()

    # --- Baseline: intercept-only random-effects ---
    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)

    # --- Moderator: DLN stage (reference = first occupied stage) ---
    X_mod, names_mod = design_matrix_categorical(df["dln_stage_code"])
    res_mod = fit_reml(y, v, X_mod)

    # Heterogeneity reduction
    delta_tau2 = res_base.tau2 - res_mod.tau2
    pct_reduction = (delta_tau2 / res_base.tau2 * 100) if res_base.tau2 > 0 else 0.0

    # QM omnibus moderator test
    qm = compute_qm(y, v, X_base, X_mod)

    # Summary table
    rows = []
    rows.append({
        "model": "baseline (intercept-only)",
        "k": res_base.k,
        "tau2": round(res_base.tau2, 4),
        "I2": round(res_base.I2, 3),
        "Q": round(res_base.Q, 2),
        "parameter": "mu",
        "estimate": round(res_base.beta[0], 4),
        "se": round(res_base.se[0], 4),
        "ci_lo": round(res_base.ci95[0, 0], 4),
        "ci_hi": round(res_base.ci95[0, 1], 4),
    })
    for i, name in enumerate(names_mod):
        rows.append({
            "model": "DLN-stage moderator",
            "k": res_mod.k,
            "tau2": round(res_mod.tau2, 4),
            "I2": round(res_mod.I2, 3),
            "Q": round(res_mod.Q, 2),
            "parameter": name,
            "estimate": round(res_mod.beta[i], 4),
            "se": round(res_mod.se[i], 4),
            "ci_lo": round(res_mod.ci95[i, 0], 4),
            "ci_hi": round(res_mod.ci95[i, 1], 4),
        })
    rows.append({
        "model": "heterogeneity_reduction",
        "k": res_mod.k,
        "tau2": round(delta_tau2, 4),
        "I2": round(pct_reduction, 1),
        "Q": 0,
        "parameter": "delta_tau2",
        "estimate": round(delta_tau2, 4),
        "se": 0,
        "ci_lo": 0,
        "ci_hi": 0,
    })

    summary = pd.DataFrame(rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)

    # --- Forest-style figure: stage means ---
    stage_order = ["dot", "linear", "network"]
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = list(range(len(df)))

    for i, row in df.iterrows():
        color = stage_colors[row["dln_stage_code"]]
        ax.barh(i, row["d_plus"], xerr=1.96 * row["se_d"], color=color,
                alpha=0.8, capsize=3, height=0.7)
        ax.text(row["d_plus"] + 1.96 * row["se_d"] + 0.02, i,
                row["strategy_sub"].replace("_", " "),
                va="center", fontsize=8)

    # Stage mean lines
    for stage in stage_order:
        sub = df[df["dln_stage_code"] == stage]
        if len(sub) == 0:
            continue
        mean_d = np.average(sub["d_plus"], weights=1.0 / sub["vi"])
        ax.axvline(mean_d, color=stage_colors[stage], linestyle="--",
                   alpha=0.7, label=f"{stage} pooled d={mean_d:.2f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([""] * len(y_pos))
    ax.set_xlabel("Effect size (d)")
    ax.set_title("Webb et al. (2012) strategies coded by DLN stage")
    ax.legend(loc="lower right", fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.5)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    print(f"Wrote: {OUT_TABLE}")
    print(f"Wrote: {OUT_FIG}")
    print(f"\n--- Results ---")
    print(f"Baseline tau2: {res_base.tau2:.4f}, I2: {res_base.I2:.3f}")
    print(f"Moderator tau2: {res_mod.tau2:.4f}, I2: {res_mod.I2:.3f}")
    print(f"Heterogeneity reduction: {delta_tau2:.4f} ({pct_reduction:.1f}%)")
    print(f"QM({qm.df}) = {qm.QM:.2f}, p = {qm.p:.4f}")
    for i, name in enumerate(names_mod):
        print(f"  {name}: b={res_mod.beta[i]:.4f} [{res_mod.ci95[i,0]:.4f}, {res_mod.ci95[i,1]:.4f}]")


if __name__ == "__main__":
    main()
