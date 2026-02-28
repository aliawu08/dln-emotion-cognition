"""Greenwald et al. (2009) study-level DLN moderator analysis.

Uses the full Appendix data (k=184 independent samples) from Greenwald,
Poehlman, Uhlmann, & Banaji (2009) to test whether DLN stage explains
between-study variation in IAT predictive validity (ICC).

DLN stage is coded at the topic-domain level per the pre-specified rubric
(protocol/greenwald2009_coding_rubric.md):
  - Consumer          -> dot   (snap judgments, approach-avoidance)
  - Race (Bl/Wh)      -> linear (intergroup, social desirability)
  - Politics           -> linear (social identity, norms)
  - Gender/sex         -> linear (intergroup, social desirability)
  - Other intergroup   -> linear (intergroup, social desirability)
  - Relationships      -> network (sustained relational integration)
  - Personality        -> mixed_unclear (Dot-Network ambiguous)
  - Drugs/tobacco      -> mixed_unclear (clinical/health varies)
  - Clinical           -> mixed_unclear (clinical/health varies)

Additionally tests the DLN prediction about explicit-implicit discrepancy:
  - Linear domains should show the LARGEST ICC-ECC gap (suppression)
  - Network domains should show the SMALLEST gap (integration aligns both)
  - Dot domains: intermediate (both tap reactive, but explicit misses)

Source:
  Greenwald, A. G., Poehlman, T. A., Uhlmann, E. L., & Banaji, M. R. (2009).
  Understanding and using the Implicit Association Test: III. Meta-analysis of
  predictive validity. Journal of Personality and Social Psychology, 97(1), 17-41.

Outputs:
- evidence_synthesis/outputs/tables/greenwald2009_moderator_summary.csv
- evidence_synthesis/outputs/figures/greenwald2009_stage_forest.png

Usage:
  python evidence_synthesis/analysis/run_greenwald2009_moderator.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import fit_reml, design_matrix_stage, egger_test

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "greenwald2009_study_extraction.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "greenwald2009_moderator_summary.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "greenwald2009_stage_forest.png"

# Pre-specified DLN coding by topic domain
TOPIC_TO_DLN = {
    "Consumer": "dot",
    "Race (Bl/Wh)": "linear",
    "Politics": "linear",
    "Gender/sex": "linear",
    "Other intergroup": "linear",
    "Relationships": "network",
    "Personality": "mixed_unclear",
    "Drugs/tobacco": "mixed_unclear",
    "Clinical": "mixed_unclear",
}


def prepare_data(df):
    """Add DLN coding, Fisher-z transforms, and sampling variances."""
    df = df.copy()
    df["dln_stage"] = df["topic"].map(TOPIC_TO_DLN)
    assert df["dln_stage"].notna().all(), f"Unmapped topics: {df[df['dln_stage'].isna()]['topic'].unique()}"

    # Fisher-z transform for ICC
    df["yi_icc"] = np.arctanh(df["icc"].clip(-0.999, 0.999))
    df["vi_icc"] = 1.0 / (df["n"] - 3)

    # Fisher-z for ECC (where available)
    has_ecc = df["ecc"].notna()
    df.loc[has_ecc, "yi_ecc"] = np.arctanh(df.loc[has_ecc, "ecc"].clip(-0.999, 0.999))
    df.loc[has_ecc, "vi_ecc"] = 1.0 / (df.loc[has_ecc, "n"] - 3)

    # ICC - ECC discrepancy (where both available)
    has_both = has_ecc
    df.loc[has_both, "yi_disc"] = df.loc[has_both, "yi_icc"] - df.loc[has_both, "yi_ecc"]
    # Approximate variance of difference (ignoring covariance, conservative)
    df.loc[has_both, "vi_disc"] = df.loc[has_both, "vi_icc"] + df.loc[has_both, "vi_ecc"]

    return df


def run_meta(df, y_col, v_col, label, exclude_mixed=False):
    """Run baseline + DLN moderator meta-regression."""
    if exclude_mixed:
        df = df[df["dln_stage"] != "mixed_unclear"].copy()

    sub = df.dropna(subset=[y_col, v_col]).copy()
    y = sub[y_col].to_numpy()
    v = sub[v_col].to_numpy()
    k = len(sub)

    # Baseline
    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    # Moderator
    stages_present = [s for s in sub["dln_stage"].unique() if s != "mixed_unclear"]
    if len(stages_present) < 2:
        return res_base, None, None, sub

    X_mod, names_mod = design_matrix_stage(sub["dln_stage"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    return res_base, res_mod, names_mod, sub


def sensitivity_recode(df, mixed_target):
    """Reclassify mixed_unclear to a target stage."""
    df = df.copy()
    df.loc[df["dln_stage"] == "mixed_unclear", "dln_stage"] = mixed_target
    return df


def format_results(res_base, res_mod, names_mod, label, k):
    """Format results into summary rows."""
    rows = []
    rows.append({
        "analysis": label,
        "model": "baseline",
        "k": k,
        "tau2": round(res_base.tau2, 6),
        "I2": round(res_base.I2, 3),
        "Q": round(res_base.Q, 4),
        "parameter": "mu",
        "estimate": round(res_base.beta[0], 4),
        "se": round(res_base.se[0], 4),
        "ci_lo": round(res_base.ci95[0, 0], 4),
        "ci_hi": round(res_base.ci95[0, 1], 4),
    })
    if res_mod is not None:
        delta_tau2 = res_base.tau2 - res_mod.tau2
        pct = (delta_tau2 / res_base.tau2 * 100) if res_base.tau2 > 0 else 0.0
        for i, name in enumerate(names_mod):
            rows.append({
                "analysis": label,
                "model": "DLN-stage",
                "k": k,
                "tau2": round(res_mod.tau2, 6),
                "I2": round(res_mod.I2, 3),
                "Q": round(res_mod.Q, 4),
                "parameter": name,
                "estimate": round(res_mod.beta[i], 4),
                "se": round(res_mod.se[i], 4),
                "ci_lo": round(res_mod.ci95[i, 0], 4),
                "ci_hi": round(res_mod.ci95[i, 1], 4),
            })
        rows.append({
            "analysis": label,
            "model": "reduction",
            "k": k,
            "tau2": round(delta_tau2, 6),
            "I2": round(pct, 1),
            "Q": 0,
            "parameter": "delta_tau2",
            "estimate": round(delta_tau2, 6),
            "se": 0,
            "ci_lo": 0,
            "ci_hi": 0,
        })
    return rows


def print_stage_means(df, y_col, v_col, outcome_label):
    """Print weighted mean effect sizes by DLN stage."""
    print(f"\n  Stage means ({outcome_label}, Fisher z -> r):")
    for stage in ["dot", "linear", "network", "mixed_unclear"]:
        sub = df[df["dln_stage"] == stage].dropna(subset=[y_col, v_col])
        if len(sub) > 0:
            mean_z = np.average(sub[y_col], weights=1.0 / sub[v_col])
            mean_r = np.tanh(mean_z)
            sd_r = sub[y_col.replace("yi_", "")].std() if y_col.replace("yi_", "") in sub.columns else 0
            print(f"    {stage:15s}: k={len(sub):3d}, weighted mean r={mean_r:.3f}, "
                  f"raw SD(r)={sub[y_col.replace('yi_', '')].std():.3f}" if y_col.replace("yi_", "") in sub.columns
                  else f"    {stage:15s}: k={len(sub):3d}, weighted mean z={mean_z:.4f}, r={mean_r:.3f}")


def main():
    raw = pd.read_csv(DATA)
    print(f"Loaded {len(raw)} independent samples from {DATA.name}")

    df = prepare_data(raw)

    print(f"\nDLN stage distribution:")
    print(df["dln_stage"].value_counts().sort_index().to_string())
    print(f"\nBy topic:")
    for topic, stage in sorted(TOPIC_TO_DLN.items(), key=lambda x: x[1]):
        k_topic = (df["topic"] == topic).sum()
        print(f"  {topic:20s} -> {stage:15s} (k={k_topic})")

    all_rows = []

    # ================================================================
    # 1. PRIMARY: ICC as outcome, all k=184
    # ================================================================
    print("\n" + "=" * 70)
    print("1. PRIMARY: DLN stage moderator on ICC (k=184)")
    print("=" * 70)
    res_base, res_mod, names_mod, df_used = run_meta(df, "yi_icc", "vi_icc", "icc_primary")
    all_rows.extend(format_results(res_base, res_mod, names_mod, "icc_primary_k184", len(df_used)))
    print(f"  Baseline: tau2={res_base.tau2:.6f}, I2={res_base.I2:.1%}, Q={res_base.Q:.1f}")
    if res_mod:
        delta = res_base.tau2 - res_mod.tau2
        pct = (delta / res_base.tau2 * 100) if res_base.tau2 > 0 else 0
        print(f"  DLN moderator: tau2={res_mod.tau2:.6f}, heterogeneity reduction={pct:.1f}%")
        for i, name in enumerate(names_mod):
            print(f"    {name}: b={res_mod.beta[i]:.4f} [{res_mod.ci95[i,0]:.4f}, {res_mod.ci95[i,1]:.4f}]")

    print_stage_means(df, "yi_icc", "vi_icc", "ICC")

    # Egger's test
    egger = egger_test(df["yi_icc"].to_numpy(), df["vi_icc"].to_numpy())
    print(f"\n  Egger's test: intercept={egger.intercept:.3f}, t={egger.t_stat:.3f}, p={egger.p_value:.4f}")

    # ================================================================
    # 2. DROP MIXED: ICC, k=125 (dot + linear + network only)
    # ================================================================
    print("\n" + "=" * 70)
    print("2. SENSITIVITY: Drop mixed_unclear (dot + linear + network only)")
    print("=" * 70)
    res_b2, res_m2, names2, df2 = run_meta(df, "yi_icc", "vi_icc", "icc_drop_mixed", exclude_mixed=True)
    all_rows.extend(format_results(res_b2, res_m2, names2, "icc_drop_mixed", len(df2)))
    print(f"  k={len(df2)}, Baseline tau2={res_b2.tau2:.6f}, I2={res_b2.I2:.1%}")
    if res_m2:
        delta2 = res_b2.tau2 - res_m2.tau2
        pct2 = (delta2 / res_b2.tau2 * 100) if res_b2.tau2 > 0 else 0
        print(f"  DLN moderator: tau2={res_m2.tau2:.6f}, reduction={pct2:.1f}%")
        for i, name in enumerate(names2):
            print(f"    {name}: b={res_m2.beta[i]:.4f} [{res_m2.ci95[i,0]:.4f}, {res_m2.ci95[i,1]:.4f}]")

    # ================================================================
    # 3. SENSITIVITY: Recode mixed -> dot
    # ================================================================
    print("\n" + "=" * 70)
    print("3. SENSITIVITY: Recode mixed_unclear -> dot (k=184)")
    print("=" * 70)
    df_s3 = sensitivity_recode(df, "dot")
    res_b3, res_m3, names3, df3 = run_meta(df_s3, "yi_icc", "vi_icc", "icc_mixed_dot")
    all_rows.extend(format_results(res_b3, res_m3, names3, "icc_mixed_as_dot", len(df3)))
    if res_m3:
        delta3 = res_b3.tau2 - res_m3.tau2
        pct3 = (delta3 / res_b3.tau2 * 100) if res_b3.tau2 > 0 else 0
        print(f"  DLN moderator: tau2={res_m3.tau2:.6f}, reduction={pct3:.1f}%")
        for i, name in enumerate(names3):
            print(f"    {name}: b={res_m3.beta[i]:.4f} [{res_m3.ci95[i,0]:.4f}, {res_m3.ci95[i,1]:.4f}]")

    # ================================================================
    # 4. SENSITIVITY: Recode mixed -> linear
    # ================================================================
    print("\n" + "=" * 70)
    print("4. SENSITIVITY: Recode mixed_unclear -> linear (k=184)")
    print("=" * 70)
    df_s4 = sensitivity_recode(df, "linear")
    res_b4, res_m4, names4, df4 = run_meta(df_s4, "yi_icc", "vi_icc", "icc_mixed_linear")
    all_rows.extend(format_results(res_b4, res_m4, names4, "icc_mixed_as_linear", len(df4)))
    if res_m4:
        delta4 = res_b4.tau2 - res_m4.tau2
        pct4 = (delta4 / res_b4.tau2 * 100) if res_b4.tau2 > 0 else 0
        print(f"  DLN moderator: tau2={res_m4.tau2:.6f}, reduction={pct4:.1f}%")
        for i, name in enumerate(names4):
            print(f"    {name}: b={res_m4.beta[i]:.4f} [{res_m4.ci95[i,0]:.4f}, {res_m4.ci95[i,1]:.4f}]")

    # ================================================================
    # 4b. SENSITIVITY: Recode mixed -> network
    # ================================================================
    print("\n" + "=" * 70)
    print("4b. SENSITIVITY: Recode mixed_unclear -> network (k=184)")
    print("=" * 70)
    df_s4b = sensitivity_recode(df, "network")
    res_b4b, res_m4b, names4b, df4b = run_meta(df_s4b, "yi_icc", "vi_icc", "icc_mixed_network")
    all_rows.extend(format_results(res_b4b, res_m4b, names4b, "icc_mixed_as_network", len(df4b)))
    if res_m4b:
        delta4b = res_b4b.tau2 - res_m4b.tau2
        pct4b = (delta4b / res_b4b.tau2 * 100) if res_b4b.tau2 > 0 else 0
        print(f"  DLN moderator: tau2={res_m4b.tau2:.6f}, reduction={pct4b:.1f}%")
        for i, name in enumerate(names4b):
            print(f"    {name}: b={res_m4b.beta[i]:.4f} [{res_m4b.ci95[i,0]:.4f}, {res_m4b.ci95[i,1]:.4f}]")

    # ================================================================
    # 5. DISCREPANCY ANALYSIS: ICC - ECC by DLN stage
    # ================================================================
    print("\n" + "=" * 70)
    print("5. DISCREPANCY: DLN stage moderator on ICC-ECC gap")
    print("=" * 70)
    df_disc = df.dropna(subset=["yi_disc", "vi_disc"]).copy()
    print(f"  Samples with both ICC and ECC: k={len(df_disc)}")
    res_b5, res_m5, names5, df5 = run_meta(df_disc, "yi_disc", "vi_disc", "disc_primary")
    all_rows.extend(format_results(res_b5, res_m5, names5, "disc_icc_minus_ecc", len(df5)))
    print(f"  Baseline: tau2={res_b5.tau2:.6f}, I2={res_b5.I2:.1%}")
    if res_m5:
        delta5 = res_b5.tau2 - res_m5.tau2
        pct5 = (delta5 / res_b5.tau2 * 100) if res_b5.tau2 > 0 else 0
        print(f"  DLN moderator: tau2={res_m5.tau2:.6f}, reduction={pct5:.1f}%")
        for i, name in enumerate(names5):
            print(f"    {name}: b={res_m5.beta[i]:.4f} [{res_m5.ci95[i,0]:.4f}, {res_m5.ci95[i,1]:.4f}]")

    print_stage_means(df_disc, "yi_disc", "vi_disc", "ICC-ECC discrepancy")

    # DLN prediction check for discrepancy
    print("\n  DLN prediction: Linear > Dot > Network for ICC-ECC gap")
    for stage in ["dot", "linear", "network", "mixed_unclear"]:
        sub = df_disc[df_disc["dln_stage"] == stage]
        if len(sub) > 0:
            raw_disc = sub["icc"] - sub["ecc"]
            print(f"    {stage:15s}: k={len(sub):3d}, mean(ICC-ECC)={raw_disc.mean():.3f}, "
                  f"median={raw_disc.median():.3f}")

    # ================================================================
    # 6. ECC as outcome: DLN stage moderator
    # ================================================================
    print("\n" + "=" * 70)
    print("6. ECC ANALYSIS: DLN stage moderator on explicit-criterion correlation")
    print("=" * 70)
    df_ecc = df.dropna(subset=["yi_ecc", "vi_ecc"]).copy()
    print(f"  Samples with ECC: k={len(df_ecc)}")
    res_b6, res_m6, names6, df6 = run_meta(df_ecc, "yi_ecc", "vi_ecc", "ecc_primary")
    all_rows.extend(format_results(res_b6, res_m6, names6, "ecc_primary", len(df6)))
    print(f"  Baseline: tau2={res_b6.tau2:.6f}, I2={res_b6.I2:.1%}")
    if res_m6:
        delta6 = res_b6.tau2 - res_m6.tau2
        pct6 = (delta6 / res_b6.tau2 * 100) if res_b6.tau2 > 0 else 0
        print(f"  DLN moderator: tau2={res_m6.tau2:.6f}, reduction={pct6:.1f}%")
        for i, name in enumerate(names6):
            print(f"    {name}: b={res_m6.beta[i]:.4f} [{res_m6.ci95[i,0]:.4f}, {res_m6.ci95[i,1]:.4f}]")

    print_stage_means(df_ecc, "yi_ecc", "vi_ecc", "ECC")

    # ================================================================
    # THEORETICAL SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("THEORETICAL PREDICTION SUMMARY")
    print("=" * 70)
    print("DLN predictions for IAT predictive validity (Greenwald 2009 data):")
    print("  1. ICC magnitude: Dot ~ Linear > Network (reactive alignment)")
    print("  2. ECC magnitude: Network > Dot > Linear (explicit-criterion)")
    print("  3. ICC-ECC gap: Linear > Dot > Network (suppression vs integration)")
    print("  4. IEC (IAT-explicit): Network > Dot > Linear (alignment)")

    # Save table
    summary = pd.DataFrame(all_rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote: {OUT_TABLE}")

    # ================================================================
    # FOREST PLOT (study-level, grouped by DLN stage)
    # ================================================================
    stage_colors = {
        "dot": "#e74c3c",
        "linear": "#f39c12",
        "network": "#27ae60",
        "mixed_unclear": "#95a5a6",
    }
    stage_order_map = {"dot": 0, "linear": 1, "network": 2, "mixed_unclear": 3}

    df_plot = df.copy()
    df_plot["stage_order"] = df_plot["dln_stage"].map(stage_order_map)
    df_plot = df_plot.sort_values(["stage_order", "icc"], ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 0.18 * len(df_plot) + 2))

    # Draw study-level points with CI
    for i, row in df_plot.iterrows():
        color = stage_colors[row["dln_stage"]]
        se = np.sqrt(row["vi_icc"])
        ci_lo = np.tanh(row["yi_icc"] - 1.96 * se)
        ci_hi = np.tanh(row["yi_icc"] + 1.96 * se)
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=0.8, alpha=0.6)
        ax.plot(row["icc"], i, "s", color=color, markersize=3, alpha=0.8)

    # Stage separators and labels
    prev_stage = None
    for i, row in df_plot.iterrows():
        if row["dln_stage"] != prev_stage and prev_stage is not None:
            ax.axhline(i - 0.5, color="grey", linewidth=0.5, linestyle=":")
        prev_stage = row["dln_stage"]

    # Stage pooled means
    for stage in ["dot", "linear", "network", "mixed_unclear"]:
        sub = df_plot[df_plot["dln_stage"] == stage]
        if len(sub) > 0:
            mean_z = np.average(sub["yi_icc"], weights=1.0 / sub["vi_icc"])
            mean_r = np.tanh(mean_z)
            ax.axvline(mean_r, color=stage_colors[stage], linestyle="--",
                       alpha=0.6, linewidth=1.2,
                       label=f"{stage} (k={len(sub)}) r={mean_r:.3f}")

    ax.set_yticks([])
    ax.set_xlabel("ICC (IAT-criterion correlation, r)")
    ax.set_title("Greenwald et al. (2009) — 184 samples coded by DLN stage")
    ax.legend(loc="lower right", fontsize=7)
    ax.axvline(0, color="grey", linewidth=0.5)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")


if __name__ == "__main__":
    main()
