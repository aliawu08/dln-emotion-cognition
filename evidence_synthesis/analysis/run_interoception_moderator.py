"""Interoception–alexithymia DLN-stage moderator analysis.

Combines measure-family-level pooled correlations from Trevisan et al. (2019)
and Van Bael et al. (2024), codes each interoceptive measure for DLN stage,
and runs a random-effects meta-regression testing whether DLN stage explains
between-measure heterogeneity in the interoception–alexithymia association.

Key prediction: DLN stage predicts a SIGN REVERSAL — linear measures show
positive associations (somatic amplification), network measures show negative
associations (integrative awareness), and dot measures show near-zero.

Outputs:
- evidence_synthesis/outputs/tables/interoception_moderator_summary.csv
- evidence_synthesis/outputs/figures/interoception_stage_forest.png

Usage:
  python evidence_synthesis/analysis/run_interoception_moderator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import fit_reml, design_matrix_stage, egger_test, compute_qm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "interoception_measure_extraction.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "interoception_moderator_summary.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "interoception_stage_forest.png"


def r_to_fisher_z(r):
    """Convert correlation to Fisher z."""
    return np.arctanh(np.clip(r, -0.999, 0.999))


def fisher_z_to_r(z):
    """Convert Fisher z back to correlation."""
    return np.tanh(z)


def main():
    df = pd.read_csv(DATA)

    # Convert pooled r to Fisher z for meta-regression
    df["z"] = r_to_fisher_z(df["r_pooled"].to_numpy())

    # Compute sampling variance for Fisher z
    # vi for z ≈ 1/(N - 3k) as approximate fixed-effects variance
    # This is conservative; random-effects SE would be larger
    df["vi_z"] = 1.0 / (df["N_total"] - 3.0 * df["k"])

    y = df["z"].to_numpy()
    v = df["vi_z"].to_numpy()

    # --- Baseline: intercept-only random-effects ---
    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)

    # --- Moderator: DLN stage (dot as reference) ---
    X_mod, names_mod = design_matrix_stage(df["dln_stage_code"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    # Heterogeneity reduction
    delta_tau2 = res_base.tau2 - res_mod.tau2
    pct_reduction = (delta_tau2 / res_base.tau2 * 100) if res_base.tau2 > 0 else 0.0

    # QM omnibus moderator test
    qm = compute_qm(y, v, X_base, X_mod)

    # Egger's test for publication bias
    egger = egger_test(y, v)

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
    rows.append({
        "model": "egger_test",
        "k": egger.k,
        "tau2": 0,
        "I2": 0,
        "Q": 0,
        "parameter": "intercept",
        "estimate": round(egger.intercept, 4) if not np.isnan(egger.intercept) else np.nan,
        "se": round(egger.se, 4) if not np.isnan(egger.se) else np.nan,
        "ci_lo": round(egger.t_stat, 4) if not np.isnan(egger.t_stat) else np.nan,
        "ci_hi": round(egger.p_value, 4) if not np.isnan(egger.p_value) else np.nan,
    })

    summary = pd.DataFrame(rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)

    # --- Forest-style figure ---
    stage_order = ["dot", "linear", "network"]
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, row in df.iterrows():
        color = stage_colors[row["dln_stage_code"]]
        se_z = np.sqrt(row["vi_z"])
        z_val = row["z"]
        ax.barh(i, z_val, xerr=1.96 * se_z, color=color,
                alpha=0.8, capsize=3, height=0.7)
        label = f"{row['measure_family']} (r={row['r_pooled']:.2f}, k={row['k']})"
        x_offset = max(z_val + 1.96 * se_z, z_val) + 0.02
        ax.text(x_offset, i, label, va="center", fontsize=7.5)

    # Stage mean lines
    for stage in stage_order:
        sub = df[df["dln_stage_code"] == stage]
        mean_z = np.average(sub["z"], weights=1.0 / sub["vi_z"])
        mean_r = fisher_z_to_r(mean_z)
        ax.axvline(mean_z, color=stage_colors[stage], linestyle="--",
                   alpha=0.7, label=f"{stage} pooled r={mean_r:.2f}")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([""] * len(df))
    ax.set_xlabel("Fisher z (interoception–alexithymia association)")
    ax.set_title("Interoceptive measures coded by DLN stage\n(Trevisan 2019 + Van Bael 2024)")
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
    print(f"\nStage coefficients (Fisher z scale, dot as reference):")
    for i, name in enumerate(names_mod):
        print(f"  {name}: b={res_mod.beta[i]:.4f} "
              f"[{res_mod.ci95[i,0]:.4f}, {res_mod.ci95[i,1]:.4f}]")

    # Stage means in r scale
    print(f"\nStage means (r scale):")
    for stage in stage_order:
        sub = df[df["dln_stage_code"] == stage]
        mean_z = np.average(sub["z"], weights=1.0 / sub["vi_z"])
        print(f"  {stage}: r = {fisher_z_to_r(mean_z):.3f} (k_families={len(sub)})")

    print(f"\nEgger's test: intercept={egger.intercept:.3f}, p={egger.p_value:.4f}")


if __name__ == "__main__":
    main()
