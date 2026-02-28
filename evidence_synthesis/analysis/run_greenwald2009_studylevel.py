"""Greenwald et al. (2009) study-level DLN moderator analysis.

Recodes all 184 independent samples based on the FORMAL DLN definitions
from Wu (2026) compression model paper:

  - Dot:     No persistent belief state. O(1) memory. Criterion is a
             reflexive/physiological/automatic response (amygdala, startle,
             EMG, approach-avoidance motor, face perception, phobic avoidance).

  - Linear:  K independent option-value estimates. O(K) memory. Criterion
             involves evaluating targets/options independently with no cross-
             option information sharing (brand choice, hiring, discrimination,
             substance use, independent trait ratings).

  - Network: Shared latent factor structure. O(F) memory. Criterion requires
             cross-option integration through shared factors (political behavior
             connected through latent ideological factor structure).

This replaces the domain-level coding (Consumer=Dot, Relationships=Network)
with study-level coding based on the criterion behavior's representational
topology.

Key changes from domain-level coding:
  - Consumer (k=40): Dot -> Linear (independent brand evaluations, O(K))
  - Race (k=32): split ~16 Dot (physiological/nonverbal) + ~16 Linear (deliberative)
  - Politics (k=11): Linear -> Network (ideology = latent factor structure)
  - Relationships (k=12): Network -> mostly Linear (independent partner evaluations)
  - Clinical (k=19): Mixed -> split Dot (phobia/reactivity) + Linear (health decisions)
  - Drugs/tobacco (k=24): Mixed -> mostly Linear + 1 Dot (attentional bias)
  - Personality (k=16): Mixed -> Linear (independent trait estimates)

DLN predictions under formal definitions:
  - Dot: ICC moderate (IAT captures reactive), ECC LOW (no articulable state)
         -> ICC > ECC (positive gap)
  - Linear: ICC moderate, ECC moderate (K estimates are articulable)
         -> ICC ~ ECC (near-zero gap)
  - Network: both HIGH (factor structure accessible both ways)
         -> ICC ~ ECC, with higher absolute levels

Usage:
  python evidence_synthesis/analysis/run_greenwald2009_studylevel.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import fit_reml, design_matrix_stage, egger_test, compute_qm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "greenwald2009_study_extraction.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "greenwald2009_studylevel_summary.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "greenwald2009_studylevel_forest.png"

# ---------------------------------------------------------------------------
# Study-level DLN coding based on CRITERION BEHAVIOR representational topology
# ---------------------------------------------------------------------------
# Dot: criterion is reflexive/physiological/automatic (no persistent belief state)
DOT_SAMPLES = {
    # Race - physiological / nonverbal / perceptual
    2,   # Amodio & Devine (2006) s1: evaluative priming, startle blink
    3,   # Amodio & Devine (2006) s2: weapons identification (reflexive)
    18,  # Carney (2006) s1: nonverbal behavior (spontaneous leakage)
    19,  # Carney (2006) s2: nonverbal behavior
    20,  # Carney et al. (2006): nonverbal behavior
    24,  # Cunningham et al. (2004): fMRI amygdala activation
    58,  # Glaser & Knowles (2008): shooter bias RT task
    74,  # Hugenberg & Bodenhausen (2003) s1: face categorization
    75,  # Hugenberg & Bodenhausen (2003) s2: face categorization
    76,  # Hugenberg & Bodenhausen (2004) s1: face perception
    77,  # Hugenberg & Bodenhausen (2004) s2: face perception
    104, # McConnell & Leibold (2001): nonverbal interaction behavior
    122, # Phelps et al. (2000): amygdala + startle
    127, # Richeson et al. (2003) s1: executive depletion (resource measure)
    128, # Richeson et al. (2003) s2: executive depletion
    171, # Vanman et al. (2004) s1: EMG facial muscle
    172, # Vanman et al. (2004) s2: EMG facial muscle
    # Relationships - reactive
    29,  # DeSteno et al. (2006): jealousy reaction (reflexive emotional)
    # Clinical - phobic/anxiety reactivity
    30,  # Egloff & Schmukle (2002) s1: anxiety behavior in speech task
    31,  # Egloff & Schmukle (2002) s2: anxiety behavior
    32,  # Ellwart et al. (2006) s1: spider phobia avoidance
    33,  # Ellwart et al. (2006) s2: spider phobia avoidance
    103, # Mauss et al. (2006): emotional reactivity
    108, # Nock & Banaji (2007a): self-harm (impulsive)
    109, # Nock & Banaji (2007b): self-harm (impulsive)
    163, # Teachman (2005): phobic avoidance
    164, # Teachman (2007): phobic avoidance
    165, # Teachman et al. (2001): phobic avoidance
    166, # Teachman et al. (2007): phobic avoidance
    167, # Teachman & Woody (2003): phobic avoidance
    # Drugs/tobacco - attentional capture
    35,  # Field et al. (2004): attentional bias to smoking cues
    # Other intergroup - reflexive
    98,  # Maner et al. (2005): threat avoidance (reflexive)
    181, # Yabar et al. (2006): automatic mimicry
}

# Network: criterion requires shared latent factor structure (political ideology)
NETWORK_SAMPLES = {
    4,   # Arcuri et al. (2008) s1: voting (ideology = latent factor)
    5,   # Arcuri et al. (2008) s2: voting
    8,   # Bain et al. (2004): political behavior
    45,  # Friese et al. (2007): voting behavior
    84,  # Karpinski et al. (2005) s1: voting
    105, # McGraw & Mulligan (2003): political behavior
    106, # Mitchell et al. (2006): political behavior
    110, # Nosek & Hansen (2008) s1: voting
    112, # Nosek & Hansen (2008) s3: voting
    118, # M. A. Olson & Fazio (2004) s3: political attitude-behavior
    119, # M. A. Olson & Fazio (2004) s4: political attitude-behavior
}

# Everything else: Linear (K independent option evaluations)
# Includes: all Consumer, most Race (deliberative), Gender/sex, Other intergroup,
# most Relationships, all Personality, most Drugs/tobacco, some Clinical


def assign_studylevel_stage(sample_id):
    """Assign DLN stage based on study-level criterion behavior topology."""
    if sample_id in DOT_SAMPLES:
        return "dot"
    elif sample_id in NETWORK_SAMPLES:
        return "network"
    else:
        return "linear"


def prepare_data(df):
    """Add study-level DLN coding, Fisher-z transforms, and sampling variances."""
    df = df.copy()

    # Study-level coding
    df["dln_stage"] = df["sample_id"].apply(assign_studylevel_stage)

    # Also keep domain-level coding for comparison
    TOPIC_TO_DLN = {
        "Consumer": "dot", "Race (Bl/Wh)": "linear", "Politics": "linear",
        "Gender/sex": "linear", "Other intergroup": "linear",
        "Relationships": "network", "Personality": "mixed_unclear",
        "Drugs/tobacco": "mixed_unclear", "Clinical": "mixed_unclear",
    }
    df["dln_stage_domain"] = df["topic"].map(TOPIC_TO_DLN)

    # Fisher-z transform for ICC
    df["yi_icc"] = np.arctanh(df["icc"].clip(-0.999, 0.999))
    df["vi_icc"] = 1.0 / (df["n"] - 3)

    # Fisher-z for ECC (where available)
    has_ecc = df["ecc"].notna()
    df.loc[has_ecc, "yi_ecc"] = np.arctanh(df.loc[has_ecc, "ecc"].clip(-0.999, 0.999))
    df.loc[has_ecc, "vi_ecc"] = 1.0 / (df.loc[has_ecc, "n"] - 3)

    # ICC - ECC discrepancy
    has_both = has_ecc
    df.loc[has_both, "yi_disc"] = df.loc[has_both, "yi_icc"] - df.loc[has_both, "yi_ecc"]
    df.loc[has_both, "vi_disc"] = df.loc[has_both, "vi_icc"] + df.loc[has_both, "vi_ecc"]

    return df


def run_meta(df, y_col, v_col, label):
    """Run baseline + DLN moderator meta-regression."""
    sub = df.dropna(subset=[y_col, v_col]).copy()
    y = sub[y_col].to_numpy()
    v = sub[v_col].to_numpy()
    k = len(sub)

    # Baseline
    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    # Moderator
    X_mod, names_mod = design_matrix_stage(sub["dln_stage"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    return res_base, res_mod, names_mod, sub


def format_results(res_base, res_mod, names_mod, label, k):
    """Format results into summary rows."""
    rows = []
    rows.append({
        "analysis": label, "model": "baseline", "k": k,
        "tau2": round(res_base.tau2, 6), "I2": round(res_base.I2, 3),
        "Q": round(res_base.Q, 4), "parameter": "mu",
        "estimate": round(res_base.beta[0], 4), "se": round(res_base.se[0], 4),
        "ci_lo": round(res_base.ci95[0, 0], 4), "ci_hi": round(res_base.ci95[0, 1], 4),
    })
    if res_mod is not None:
        delta_tau2 = res_base.tau2 - res_mod.tau2
        pct = (delta_tau2 / res_base.tau2 * 100) if res_base.tau2 > 0 else 0.0
        for i, name in enumerate(names_mod):
            rows.append({
                "analysis": label, "model": "DLN-stage", "k": k,
                "tau2": round(res_mod.tau2, 6), "I2": round(res_mod.I2, 3),
                "Q": round(res_mod.Q, 4), "parameter": name,
                "estimate": round(res_mod.beta[i], 4), "se": round(res_mod.se[i], 4),
                "ci_lo": round(res_mod.ci95[i, 0], 4), "ci_hi": round(res_mod.ci95[i, 1], 4),
            })
        rows.append({
            "analysis": label, "model": "reduction", "k": k,
            "tau2": round(delta_tau2, 6), "I2": round(pct, 1), "Q": 0,
            "parameter": "delta_tau2", "estimate": round(delta_tau2, 6),
            "se": 0, "ci_lo": 0, "ci_hi": 0,
        })
    return rows


def print_stage_means(df, y_col, v_col, outcome_label):
    """Print weighted mean effect sizes by DLN stage."""
    print(f"\n  Stage means ({outcome_label}, Fisher z -> r):")
    for stage in ["dot", "linear", "network"]:
        sub = df[df["dln_stage"] == stage].dropna(subset=[y_col, v_col])
        if len(sub) > 0:
            mean_z = np.average(sub[y_col], weights=1.0 / sub[v_col])
            mean_r = np.tanh(mean_z)
            print(f"    {stage:10s}: k={len(sub):3d}, weighted mean r={mean_r:.3f}")


def main():
    raw = pd.read_csv(DATA)
    print(f"Loaded {len(raw)} independent samples from {DATA.name}")

    df = prepare_data(raw)

    # Distribution summary
    print(f"\n{'='*70}")
    print("STUDY-LEVEL DLN STAGE DISTRIBUTION")
    print(f"{'='*70}")
    print(f"\nOverall: {df['dln_stage'].value_counts().sort_index().to_string()}")

    print(f"\nStage assignment by original topic domain:")
    for topic in sorted(df["topic"].unique()):
        sub = df[df["topic"] == topic]
        counts = sub["dln_stage"].value_counts().sort_index()
        old_stage = sub["dln_stage_domain"].iloc[0]
        print(f"  {topic:20s} (was {old_stage:15s}): {dict(counts)}")

    # Show what moved
    changed = df[df["dln_stage"] != df["dln_stage_domain"].replace("mixed_unclear", None)]
    print(f"\n  Samples reclassified from domain-level coding: "
          f"{(df['dln_stage'] != df['dln_stage_domain']).sum()}")

    all_rows = []

    # ================================================================
    # 1. PRIMARY: ICC as outcome, all k=184
    # ================================================================
    print(f"\n{'='*70}")
    print("1. ICC ANALYSIS: Study-level DLN moderator (k=184)")
    print(f"{'='*70}")
    res_base, res_mod, names_mod, df_used = run_meta(df, "yi_icc", "vi_icc", "icc_studylevel")
    all_rows.extend(format_results(res_base, res_mod, names_mod, "icc_studylevel_k184", len(df_used)))

    delta = res_base.tau2 - res_mod.tau2
    pct = (delta / res_base.tau2 * 100) if res_base.tau2 > 0 else 0
    qm_icc = compute_qm(df_used["yi_icc"].to_numpy(), df_used["vi_icc"].to_numpy(),
                         np.ones((len(df_used), 1)),
                         design_matrix_stage(df_used["dln_stage"], reference="dot")[0])
    print(f"  Baseline: tau2={res_base.tau2:.6f}, I2={res_base.I2:.1%}, Q={res_base.Q:.1f}")
    print(f"  DLN moderator: tau2={res_mod.tau2:.6f}, heterogeneity reduction={pct:.1f}%")
    print(f"  QM({qm_icc.df}) = {qm_icc.QM:.2f}, p = {qm_icc.p:.6f}")
    for i, name in enumerate(names_mod):
        print(f"    {name}: b={res_mod.beta[i]:.4f} [{res_mod.ci95[i,0]:.4f}, {res_mod.ci95[i,1]:.4f}]")

    print_stage_means(df, "yi_icc", "vi_icc", "ICC")

    egger = egger_test(df["yi_icc"].to_numpy(), df["vi_icc"].to_numpy())
    print(f"\n  Egger's test: intercept={egger.intercept:.3f}, t={egger.t_stat:.3f}, p={egger.p_value:.4f}")

    # ================================================================
    # 2. ECC ANALYSIS: Study-level DLN moderator
    # ================================================================
    print(f"\n{'='*70}")
    print("2. ECC ANALYSIS: Study-level DLN moderator")
    print(f"{'='*70}")
    df_ecc = df.dropna(subset=["yi_ecc", "vi_ecc"]).copy()
    print(f"  Samples with ECC: k={len(df_ecc)}")
    res_b2, res_m2, names2, df2 = run_meta(df_ecc, "yi_ecc", "vi_ecc", "ecc_studylevel")
    all_rows.extend(format_results(res_b2, res_m2, names2, "ecc_studylevel", len(df2)))

    delta2 = res_b2.tau2 - res_m2.tau2
    pct2 = (delta2 / res_b2.tau2 * 100) if res_b2.tau2 > 0 else 0
    print(f"  Baseline: tau2={res_b2.tau2:.6f}, I2={res_b2.I2:.1%}")
    print(f"  DLN moderator: tau2={res_m2.tau2:.6f}, reduction={pct2:.1f}%")
    for i, name in enumerate(names2):
        print(f"    {name}: b={res_m2.beta[i]:.4f} [{res_m2.ci95[i,0]:.4f}, {res_m2.ci95[i,1]:.4f}]")

    print_stage_means(df_ecc, "yi_ecc", "vi_ecc", "ECC")

    # ================================================================
    # 3. DISCREPANCY: ICC - ECC by DLN stage
    # ================================================================
    print(f"\n{'='*70}")
    print("3. DISCREPANCY: Study-level DLN moderator on ICC-ECC gap")
    print(f"{'='*70}")
    df_disc = df.dropna(subset=["yi_disc", "vi_disc"]).copy()
    print(f"  Samples with both ICC and ECC: k={len(df_disc)}")
    res_b3, res_m3, names3, df3 = run_meta(df_disc, "yi_disc", "vi_disc", "disc_studylevel")
    all_rows.extend(format_results(res_b3, res_m3, names3, "disc_studylevel", len(df3)))

    delta3 = res_b3.tau2 - res_m3.tau2
    pct3 = (delta3 / res_b3.tau2 * 100) if res_b3.tau2 > 0 else 0
    print(f"  Baseline: tau2={res_b3.tau2:.6f}, I2={res_b3.I2:.1%}")
    print(f"  DLN moderator: tau2={res_m3.tau2:.6f}, reduction={pct3:.1f}%")
    for i, name in enumerate(names3):
        print(f"    {name}: b={res_m3.beta[i]:.4f} [{res_m3.ci95[i,0]:.4f}, {res_m3.ci95[i,1]:.4f}]")

    print_stage_means(df_disc, "yi_disc", "vi_disc", "ICC-ECC discrepancy")

    # Raw discrepancy by stage
    print(f"\n  DLN prediction (formal): Dot > 0 (ICC>ECC), Linear ~ 0, Network ~ 0 (both high)")
    for stage in ["dot", "linear", "network"]:
        sub = df_disc[df_disc["dln_stage"] == stage]
        if len(sub) > 0:
            raw_disc = sub["icc"] - sub["ecc"]
            print(f"    {stage:10s}: k={len(sub):3d}, mean(ICC-ECC)={raw_disc.mean():+.3f}, "
                  f"median={raw_disc.median():+.3f}")

    # ================================================================
    # 4. STAGE MEANS TABLE (raw r, for interpretation)
    # ================================================================
    print(f"\n{'='*70}")
    print("4. STAGE MEANS SUMMARY (raw r)")
    print(f"{'='*70}")
    for stage in ["dot", "linear", "network"]:
        sub = df[df["dln_stage"] == stage]
        sub_ecc = sub.dropna(subset=["ecc"])
        k_all = len(sub)
        k_ecc = len(sub_ecc)
        mean_icc = sub["icc"].mean()
        mean_ecc = sub_ecc["ecc"].mean() if k_ecc > 0 else float("nan")
        mean_gap = (sub_ecc["icc"] - sub_ecc["ecc"]).mean() if k_ecc > 0 else float("nan")
        print(f"  {stage:10s}: k={k_all:3d} (k_ecc={k_ecc:3d}), "
              f"mean ICC={mean_icc:.3f}, mean ECC={mean_ecc:.3f}, "
              f"mean gap={mean_gap:+.3f}")

    # ================================================================
    # 5. COMPARISON: Domain-level vs Study-level
    # ================================================================
    print(f"\n{'='*70}")
    print("5. COMPARISON: Domain-level vs Study-level coding")
    print(f"{'='*70}")

    # Run domain-level for comparison (original coding, clean 3-level only)
    df_domain = df.copy()
    df_domain_clean = df_domain[df_domain["dln_stage_domain"].isin(["dot", "linear", "network"])].copy()
    df_domain_clean["dln_stage"] = df_domain_clean["dln_stage_domain"]

    print(f"\n  Domain-level (k={len(df_domain_clean)}, excluding mixed):")
    res_dom_b, res_dom_m, names_dom, _ = run_meta(df_domain_clean, "yi_icc", "vi_icc", "domain")
    dom_delta = res_dom_b.tau2 - res_dom_m.tau2
    dom_pct = (dom_delta / res_dom_b.tau2 * 100) if res_dom_b.tau2 > 0 else 0
    print(f"    ICC reduction: {dom_pct:.1f}%")
    for i, name in enumerate(names_dom):
        print(f"    {name}: b={res_dom_m.beta[i]:.4f} [{res_dom_m.ci95[i,0]:.4f}, {res_dom_m.ci95[i,1]:.4f}]")

    # Domain-level discrepancy (for comparison)
    df_dom_disc = df_domain_clean.dropna(subset=["yi_disc", "vi_disc"]).copy()
    if len(df_dom_disc) > 5:
        res_dd_b, res_dd_m, names_dd, _ = run_meta(df_dom_disc, "yi_disc", "vi_disc", "domain_disc")
        dd_delta = res_dd_b.tau2 - res_dd_m.tau2
        dd_pct = (dd_delta / res_dd_b.tau2 * 100) if res_dd_b.tau2 > 0 else 0
        print(f"\n    Discrepancy reduction: {dd_pct:.1f}%")
        for i, name in enumerate(names_dd):
            print(f"    {name}: b={res_dd_m.beta[i]:.4f} [{res_dd_m.ci95[i,0]:.4f}, {res_dd_m.ci95[i,1]:.4f}]")

    print(f"\n  Study-level (k=184, all classified):")
    print(f"    ICC reduction: {pct:.1f}%")
    print(f"    ECC reduction: {pct2:.1f}%")
    print(f"    Discrepancy reduction: {pct3:.1f}%")

    # ================================================================
    # THEORETICAL SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("THEORETICAL PREDICTION CHECK (Formal DLN)")
    print(f"{'='*70}")
    print("Formal DLN predictions for criterion validity:")
    print("  1. Dot: ICC > ECC (reactive criterion, no articulable state)")
    print("  2. Linear: ICC ~ ECC (independent estimates, both articulable)")
    print("  3. Network: both HIGH, ICC ~ ECC (factor structure accessible both ways)")
    print("  4. ICC magnitude: Network > Linear (factor structure aids prediction)")
    print("  5. ECC magnitude: Network > Linear > Dot (articulability increases)")

    # Save table
    summary = pd.DataFrame(all_rows)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote: {OUT_TABLE}")

    # ================================================================
    # FOREST PLOT
    # ================================================================
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}
    stage_order_map = {"dot": 0, "linear": 1, "network": 2}

    df_plot = df.copy()
    df_plot["stage_order"] = df_plot["dln_stage"].map(stage_order_map)
    df_plot = df_plot.sort_values(["stage_order", "icc"], ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 0.18 * len(df_plot) + 2))

    for i, row in df_plot.iterrows():
        color = stage_colors[row["dln_stage"]]
        se = np.sqrt(row["vi_icc"])
        ci_lo = np.tanh(row["yi_icc"] - 1.96 * se)
        ci_hi = np.tanh(row["yi_icc"] + 1.96 * se)
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=0.8, alpha=0.6)
        ax.plot(row["icc"], i, "s", color=color, markersize=3, alpha=0.8)

    prev_stage = None
    for i, row in df_plot.iterrows():
        if row["dln_stage"] != prev_stage and prev_stage is not None:
            ax.axhline(i - 0.5, color="grey", linewidth=0.5, linestyle=":")
        prev_stage = row["dln_stage"]

    for stage in ["dot", "linear", "network"]:
        sub = df_plot[df_plot["dln_stage"] == stage]
        if len(sub) > 0:
            mean_z = np.average(sub["yi_icc"], weights=1.0 / sub["vi_icc"])
            mean_r = np.tanh(mean_z)
            ax.axvline(mean_r, color=stage_colors[stage], linestyle="--",
                       alpha=0.6, linewidth=1.2,
                       label=f"{stage} (k={len(sub)}) r={mean_r:.3f}")

    ax.set_yticks([])
    ax.set_xlabel("ICC (IAT-criterion correlation, r)")
    ax.set_title("Greenwald (2009) — 184 samples, study-level DLN coding (formal definitions)")
    ax.legend(loc="lower right", fontsize=7)
    ax.axvline(0, color="grey", linewidth=0.5)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")


if __name__ == "__main__":
    main()
