"""Hoyt et al. (2024) emotional approach coping DLN-stage moderator analysis.

Uses verified domain-level pooled correlations from Hoyt, Llave, Wang et al.
(2024) Health Psychology 43(6), 397-417 Table 3 and Figure 2.  Codes each
health outcome domain for DLN stage and runs a random-effects meta-regression
testing whether DLN stage explains the domain-specific sign changes in the
EAC-health association.

Key prediction: DLN stage predicts a SIGN REVERSAL — linear-stage domains
(distress, risk adjustment) show negative associations (emotional approach
coping amplifies distress without integration), while dot-stage domains
(biological, physical, behavioral) show near-zero effects and network-stage
domains (resilience, positive psychology, social) show the largest positive
effects.  This is the "dangerous-middle" pattern: partial emotional engagement
without metacognitive integration produces worse outcomes than not engaging
at all.

Data status: All 8 domain-level EAC r values verified from Table 3 of
Hoyt et al. (2024).  k values verified from Figure 2.  N_approx values are
effective sample sizes derived from the reported robust standard errors to
reproduce the correct Fisher's z sampling variance.  DLN coding is
pre-specified and independent of effect sizes.

Outputs:
- evidence_synthesis/outputs/tables/hoyt2024_moderator_summary.csv
- evidence_synthesis/outputs/figures/hoyt2024_stage_forest.png

Usage:
  python evidence_synthesis/analysis/run_hoyt2024_moderator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import fit_reml, design_matrix_stage, egger_test, compute_qm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "hoyt2024_domain_extraction.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "hoyt2024_moderator_summary.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "hoyt2024_stage_forest.png"


def r_to_fisher_z(r):
    """Convert correlation to Fisher z."""
    return np.arctanh(np.clip(r, -0.999, 0.999))


def fisher_z_to_r(z):
    """Convert Fisher z back to correlation."""
    return np.tanh(z)


def main():
    df = pd.read_csv(DATA)

    print(f"Loaded {len(df)} health outcome domains")
    for stage in ["dot", "linear", "network"]:
        n = (df["dln_stage_code"] == stage).sum()
        print(f"  {stage}: k={n}")

    # Convert pooled r to Fisher z for meta-regression
    df["z"] = r_to_fisher_z(df["r_pooled"].to_numpy())

    # Sampling variance for Fisher z: vi_z ≈ 1/(N - 3)
    # Use approximate N per domain from extraction
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)

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

    # --- Summary table ---
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
        "estimate": round(egger.intercept, 4) if not np.isnan(egger.intercept) else 0,
        "se": round(egger.se, 4) if not np.isnan(egger.se) else 0,
        "ci_lo": round(egger.t_stat, 4) if not np.isnan(egger.t_stat) else 0,
        "ci_hi": round(egger.p_value, 4) if not np.isnan(egger.p_value) else 1,
    })

    summary = pd.DataFrame(rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)

    # --- Forest-style figure ---
    stage_order = ["dot", "linear", "network"]
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}

    # Sort by stage then by effect size for visual clarity
    df["stage_rank"] = df["dln_stage_code"].map({"dot": 0, "linear": 1, "network": 2})
    df_sorted = df.sort_values(["stage_rank", "r_pooled"], ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, row in df_sorted.iterrows():
        color = stage_colors[row["dln_stage_code"]]
        se_z = np.sqrt(row["vi_z"])
        z_val = row["z"]
        ax.barh(i, z_val, xerr=1.96 * se_z, color=color,
                alpha=0.8, capsize=3, height=0.7)
        label = row["health_domain"].replace("_", " ")
        x_offset = max(z_val + 1.96 * se_z, abs(z_val)) + 0.01
        ax.text(x_offset, i, f"{label} (r={row['r_pooled']:.2f}, k={row['k']})",
                va="center", fontsize=7.5)

    # Stage mean lines
    for stage in stage_order:
        sub = df_sorted[df_sorted["dln_stage_code"] == stage]
        if len(sub) > 0:
            mean_z = np.average(sub["z"], weights=1.0 / sub["vi_z"])
            mean_r = fisher_z_to_r(mean_z)
            ax.axvline(mean_z, color=stage_colors[stage], linestyle="--",
                       alpha=0.7, label=f"{stage} pooled r={mean_r:.2f}")

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([""] * len(df_sorted))
    ax.set_xlabel("Fisher z (EAC–health association)")
    ax.set_title("Hoyt et al. (2024) health domains coded by DLN stage\n"
                 "(emotional approach coping × health outcome)")
    ax.legend(loc="lower right", fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.5)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    # --- Print results ---
    print(f"\nWrote: {OUT_TABLE}")
    print(f"Wrote: {OUT_FIG}")
    print(f"\n--- Results ---")
    print(f"Baseline tau2: {res_base.tau2:.4f}, I2: {res_base.I2:.3f}")
    print(f"Moderator tau2: {res_mod.tau2:.4f}, I2: {res_mod.I2:.3f}")
    print(f"Heterogeneity reduction: {delta_tau2:.4f} ({pct_reduction:.1f}%)")
    print(f"QM({qm.df}) = {qm.QM:.2f}, p = {qm.p:.4f}")
    print(f"\nStage coefficients (Fisher z scale, dot as reference):")
    for i, name in enumerate(names_mod):
        lo = res_mod.ci95[i, 0]
        hi = res_mod.ci95[i, 1]
        print(f"  {name}: b={res_mod.beta[i]:.4f} [{lo:.4f}, {hi:.4f}]")

    # Stage means in r scale
    print(f"\nStage means (r scale):")
    for stage in stage_order:
        sub = df_sorted[df_sorted["dln_stage_code"] == stage]
        if len(sub) > 0:
            mean_z = np.average(sub["z"], weights=1.0 / sub["vi_z"])
            print(f"  {stage}: r = {fisher_z_to_r(mean_z):.3f} (k_domains={len(sub)})")

    print(f"\nEgger's test: intercept={egger.intercept:.3f}, p={egger.p_value:.4f}")

    # Dangerous-middle check
    print(f"\n--- Dangerous-middle pattern check ---")
    for stage in stage_order:
        sub = df_sorted[df_sorted["dln_stage_code"] == stage]
        if len(sub) > 0:
            mean_r = fisher_z_to_r(np.average(sub["z"], weights=1.0 / sub["vi_z"]))
            sign = "+" if mean_r > 0 else "-" if mean_r < 0 else "0"
            print(f"  {stage}: r={mean_r:+.3f} ({sign})")
    print("  Expected: dot(+) > linear(-) < network(+)  [V-shaped / dangerous middle]")

    # Data status confirmation
    statuses = df["estimate_status"].unique()
    if all("verified" in s for s in statuses):
        print("\nAll domain-level r values verified from Table 3 of Hoyt et al. (2024).")
    else:
        n_est = sum("estimate" in s and "verified" not in s for s in df["estimate_status"])
        if n_est > 0:
            print(f"\n*** NOTE: {n_est} of {len(df)} domain effects are estimates. ***")


if __name__ == "__main__":
    main()
