"""Desmedt et al. (2022) HCT cross-level mismatch analysis.

Tests whether DLN stage of the criterion measure explains variation in the
HCT–outcome association across 7 outcome categories (133 studies, N=11,524).

Key prediction: HCT is a dot-stage task (passive signal detection).  Criterion
measures coded as dot-stage (biological: heart rate, BMI, age, sex) should show
stronger absolute associations than linear-stage criteria (psychological self-
report: trait anxiety, depression, alexithymia).  This is a cross-level
mismatch prediction: same-level pairings (dot-dot) yield stronger associations
than cross-level pairings (dot-linear).

Data status: All 7 criterion-level r values verified from the results
section (page 4) of Desmedt et al. (2022).  Standard errors derived from
reported 95% confidence intervals.  N_approx values are effective sample
sizes reproducing the correct Fisher's z sampling variance.  DLN coding
is pre-specified and independent of effect sizes.

Outputs:
- evidence_synthesis/outputs/tables/desmedt2022_moderator_summary.csv
- evidence_synthesis/outputs/figures/desmedt2022_stage_forest.png

Usage:
  python evidence_synthesis/analysis/run_desmedt2022_moderator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import fit_reml, egger_test, compute_qm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "desmedt2022_criterion_extraction.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "desmedt2022_moderator_summary.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "desmedt2022_stage_forest.png"


def r_to_fisher_z(r):
    """Convert correlation to Fisher z."""
    return np.arctanh(np.clip(r, -0.999, 0.999))


def fisher_z_to_r(z):
    """Convert Fisher z back to correlation."""
    return np.tanh(z)


def main():
    df = pd.read_csv(DATA)

    print(f"Loaded {len(df)} criterion measures")
    for stage in ["dot", "linear"]:
        n = (df["dln_stage_code"] == stage).sum()
        print(f"  {stage}: k={n}")

    # For the cross-level mismatch analysis, we use ABSOLUTE r values
    # because the sign direction varies (some positive, some negative)
    # and the prediction is about strength of association, not direction
    df["abs_r"] = df["r_pooled"].abs()
    df["z_abs"] = r_to_fisher_z(df["abs_r"].to_numpy())

    # Also run with signed r for completeness
    df["z_signed"] = r_to_fisher_z(df["r_pooled"].to_numpy())

    # Sampling variance for Fisher z
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)

    # ============================================================
    # Analysis 1: Absolute r — does dot-stage criterion yield
    #   stronger |r| than linear-stage criterion?
    # ============================================================
    print("\n=== Analysis 1: Absolute |r| by DLN stage ===")

    y_abs = df["z_abs"].to_numpy()
    v = df["vi_z"].to_numpy()

    # Baseline: intercept-only
    X_base = np.ones((len(df), 1))
    res_base_abs = fit_reml(y_abs, v, X_base)

    # Moderator: DLN stage (linear as reference)
    # Only two levels present (dot, linear) — build design matrix manually
    # to avoid singular matrix from unused network dummy
    is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
    X_mod_abs = np.column_stack([np.ones(len(df)), is_dot])
    names_abs = ["Intercept", "stage[dot]"]
    res_mod_abs = fit_reml(y_abs, v, X_mod_abs)

    delta_tau2_abs = res_base_abs.tau2 - res_mod_abs.tau2
    pct_abs = (delta_tau2_abs / res_base_abs.tau2 * 100) if res_base_abs.tau2 > 0 else 0.0

    # QM omnibus moderator test
    qm_abs = compute_qm(y_abs, v, X_base, X_mod_abs)

    print(f"Baseline tau2: {res_base_abs.tau2:.4f}, I2: {res_base_abs.I2:.3f}")
    print(f"Moderator tau2: {res_mod_abs.tau2:.4f}, I2: {res_mod_abs.I2:.3f}")
    print(f"Reduction: {delta_tau2_abs:.4f} ({pct_abs:.1f}%)")
    print(f"QM({qm_abs.df}) = {qm_abs.QM:.2f}, p = {qm_abs.p:.4f}")
    for i, name in enumerate(names_abs):
        print(f"  {name}: b={res_mod_abs.beta[i]:.4f} "
              f"[{res_mod_abs.ci95[i,0]:.4f}, {res_mod_abs.ci95[i,1]:.4f}]")

    # Stage means
    for stage in ["dot", "linear"]:
        sub = df[df["dln_stage_code"] == stage]
        mean_abs_z = np.average(sub["z_abs"], weights=1.0 / sub["vi_z"])
        mean_abs_r = fisher_z_to_r(mean_abs_z)
        print(f"  {stage} mean |r| = {mean_abs_r:.3f} (k={len(sub)})")

    # ============================================================
    # Analysis 2: Signed r for completeness
    # ============================================================
    print("\n=== Analysis 2: Signed r by DLN stage ===")

    y_signed = df["z_signed"].to_numpy()

    X_base2 = np.ones((len(df), 1))
    res_base_signed = fit_reml(y_signed, v, X_base2)

    X_mod2 = np.column_stack([np.ones(len(df)), is_dot])
    res_mod_signed = fit_reml(y_signed, v, X_mod2)

    delta_tau2_signed = res_base_signed.tau2 - res_mod_signed.tau2
    pct_signed = (delta_tau2_signed / res_base_signed.tau2 * 100) if res_base_signed.tau2 > 0 else 0.0

    print(f"Baseline tau2: {res_base_signed.tau2:.4f}")
    print(f"Moderator tau2: {res_mod_signed.tau2:.4f}")
    print(f"Reduction: {delta_tau2_signed:.4f} ({pct_signed:.1f}%)")

    # Egger's test
    egger = egger_test(y_signed, v)

    # ============================================================
    # Summary table (report both analyses)
    # ============================================================
    rows = []

    # Absolute analysis
    rows.append({
        "model": "absolute_baseline",
        "k": res_base_abs.k,
        "tau2": round(res_base_abs.tau2, 4),
        "I2": round(res_base_abs.I2, 3),
        "Q": round(res_base_abs.Q, 2),
        "parameter": "mu_abs",
        "estimate": round(res_base_abs.beta[0], 4),
        "se": round(res_base_abs.se[0], 4),
        "ci_lo": round(res_base_abs.ci95[0, 0], 4),
        "ci_hi": round(res_base_abs.ci95[0, 1], 4),
    })
    for i, name in enumerate(names_abs):
        rows.append({
            "model": "absolute_DLN_moderator",
            "k": res_mod_abs.k,
            "tau2": round(res_mod_abs.tau2, 4),
            "I2": round(res_mod_abs.I2, 3),
            "Q": round(res_mod_abs.Q, 2),
            "parameter": name,
            "estimate": round(res_mod_abs.beta[i], 4),
            "se": round(res_mod_abs.se[i], 4),
            "ci_lo": round(res_mod_abs.ci95[i, 0], 4),
            "ci_hi": round(res_mod_abs.ci95[i, 1], 4),
        })
    rows.append({
        "model": "absolute_heterogeneity_reduction",
        "k": res_mod_abs.k,
        "tau2": round(delta_tau2_abs, 4),
        "I2": round(pct_abs, 1),
        "Q": 0,
        "parameter": "delta_tau2_abs",
        "estimate": round(delta_tau2_abs, 4),
        "se": 0,
        "ci_lo": 0,
        "ci_hi": 0,
    })

    # Signed analysis
    rows.append({
        "model": "signed_baseline",
        "k": res_base_signed.k,
        "tau2": round(res_base_signed.tau2, 4),
        "I2": round(res_base_signed.I2, 3),
        "Q": round(res_base_signed.Q, 2),
        "parameter": "mu_signed",
        "estimate": round(res_base_signed.beta[0], 4),
        "se": round(res_base_signed.se[0], 4),
        "ci_lo": round(res_base_signed.ci95[0, 0], 4),
        "ci_hi": round(res_base_signed.ci95[0, 1], 4),
    })
    rows.append({
        "model": "signed_heterogeneity_reduction",
        "k": res_mod_signed.k,
        "tau2": round(delta_tau2_signed, 4),
        "I2": round(pct_signed, 1),
        "Q": 0,
        "parameter": "delta_tau2_signed",
        "estimate": round(delta_tau2_signed, 4),
        "se": 0,
        "ci_lo": 0,
        "ci_hi": 0,
    })

    # Egger's test
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

    # ============================================================
    # Forest figure: signed r, colour-coded by criterion DLN stage
    # ============================================================
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12"}

    # Sort: dot criteria first (by abs r descending), then linear
    df["stage_rank"] = df["dln_stage_code"].map({"dot": 0, "linear": 1})
    df_sorted = df.sort_values(
        ["stage_rank", "abs_r"], ascending=[True, False]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, row in df_sorted.iterrows():
        color = stage_colors[row["dln_stage_code"]]
        se_z = np.sqrt(row["vi_z"])
        z_val = row["z_signed"]
        ax.barh(i, z_val, xerr=1.96 * se_z, color=color,
                alpha=0.8, capsize=3, height=0.65)
        label = row["criterion"].replace("_", " ")
        # Position label to the right of the bar + error bar
        x_edge = z_val + 1.96 * se_z if z_val >= 0 else z_val - 1.96 * se_z
        x_offset = max(abs(x_edge), abs(z_val)) + 0.01
        ax.text(x_offset, i,
                f"{label} (r={row['r_pooled']:.2f}, k={row['k']})",
                va="center", fontsize=8)

    # Stage mean |r| annotations
    for stage in ["dot", "linear"]:
        sub = df_sorted[df_sorted["dln_stage_code"] == stage]
        mean_abs_z = np.average(sub["z_abs"], weights=1.0 / sub["vi_z"])
        mean_abs_r = fisher_z_to_r(mean_abs_z)
        color = stage_colors[stage]
        # Show positive side for |r| reference
        ax.axvline(mean_abs_z, color=color, linestyle="--", alpha=0.5)
        ax.axvline(-mean_abs_z, color=color, linestyle="--", alpha=0.5)
        ax.text(mean_abs_z, len(df_sorted) - 0.3,
                f"{stage} |r|={mean_abs_r:.2f}",
                color=color, fontsize=8, ha="left", va="bottom")

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([""] * len(df_sorted))
    ax.set_xlabel("Fisher z (HCT–criterion association)")
    ax.set_title("Desmedt et al. (2022): HCT criterion measures coded by DLN stage\n"
                 "(dot-stage task × dot/linear-stage criteria)")
    # Manual legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.8, label="Dot (biological)"),
        Patch(facecolor="#f39c12", alpha=0.8, label="Linear (psychological)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.5)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    # --- Final summary ---
    print(f"\nWrote: {OUT_TABLE}")
    print(f"Wrote: {OUT_FIG}")

    print(f"\n--- Cross-level mismatch summary ---")
    for stage in ["dot", "linear"]:
        sub = df_sorted[df_sorted["dln_stage_code"] == stage]
        mean_abs_r = fisher_z_to_r(
            np.average(sub["z_abs"], weights=1.0 / sub["vi_z"])
        )
        print(f"  {stage} criteria: mean |r| = {mean_abs_r:.3f} "
              f"(k_criteria={len(sub)}, "
              f"k_studies={sub['k'].sum()})")
    print("  Prediction: dot |r| > linear |r| (same-level > cross-level)")

    print(f"\nEgger's test: intercept={egger.intercept:.3f}, "
          f"p={egger.p_value:.4f}")

    # Data status confirmation
    statuses = df["estimate_status"].unique()
    if all("verified" in s for s in statuses):
        print("\nAll criterion-level r values verified from Desmedt et al. (2022).")
    else:
        n_est = sum("estimate" in s and "verified" not in s for s in df["estimate_status"])
        if n_est > 0:
            print(f"\n*** NOTE: {n_est} of {len(df)} criterion r values are estimates. ***")


if __name__ == "__main__":
    main()
