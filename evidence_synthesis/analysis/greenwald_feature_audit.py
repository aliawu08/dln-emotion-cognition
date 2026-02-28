"""Greenwald (2009) — Exhaustive feature audit for DLN coding noise.

Senior research advisory analysis: Before freezing the Greenwald model,
systematically check every extracted feature for DLN-theoretic signal
that the current 3-level coding might be missing.

Checks:
  1. Univariate correlation of every numeric feature with ICC
  2. Effect of every categorical feature on ICC (eta-squared)
  3. Full predictor correlation matrix (collinearity)
  4. iec (implicit-explicit correlation) as continuous DLN stage proxy
  5. IAT-type × criterion-stage interaction (cross-level matching)
  6. n_crit × stage interaction (K-effect varies by topology)
  7. Within-stage variance decomposition
  8. Residual analysis after current model

Usage:
  python evidence_synthesis/analysis/greenwald_feature_audit.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_pipeline import fit_reml, design_matrix_stage, design_matrix_categorical

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "greenwald2009_study_extraction.csv"

# Study-level DLN coding (from run_greenwald2009_studylevel.py)
DOT_SAMPLES = {
    2, 3, 18, 19, 20, 24, 58, 74, 75, 76, 77, 104, 122, 127, 128,
    171, 172, 29, 30, 31, 32, 33, 103, 108, 109, 163, 164, 165, 166,
    167, 35, 98, 181,
}
NETWORK_SAMPLES = {
    4, 5, 8, 45, 84, 105, 106, 110, 112, 118, 119,
}


def assign_stage(sid):
    if sid in DOT_SAMPLES:
        return "dot"
    elif sid in NETWORK_SAMPLES:
        return "network"
    else:
        return "linear"


def eta_squared(groups_series, values):
    """Compute eta-squared for a categorical variable."""
    groups = groups_series.dropna()
    vals = values.loc[groups.index].dropna()
    common = groups.index.intersection(vals.index)
    groups = groups.loc[common]
    vals = vals.loc[common]

    group_list = [vals[groups == g].values for g in groups.unique() if len(vals[groups == g]) >= 1]
    if len(group_list) < 2:
        return np.nan, np.nan, 0

    ss_total = np.sum((vals - vals.mean()) ** 2)
    ss_between = sum(len(g) * (g.mean() - vals.mean()) ** 2 for g in group_list)
    eta2 = ss_between / ss_total if ss_total > 0 else 0

    try:
        F, p = f_oneway(*group_list)
    except Exception:
        F, p = np.nan, np.nan
    return eta2, p, len(common)


def main():
    raw = pd.read_csv(DATA)
    df = raw.copy()

    # Add DLN coding
    df["dln_stage"] = df["sample_id"].apply(assign_stage)

    # Fisher-z transforms
    df["yi_icc"] = np.arctanh(df["icc"].clip(-0.999, 0.999))
    df["vi_icc"] = 1.0 / (df["n"] - 3)
    has_ecc = df["ecc"].notna()
    df.loc[has_ecc, "yi_ecc"] = np.arctanh(df.loc[has_ecc, "ecc"].clip(-0.999, 0.999))
    has_iec = df["iec"].notna()
    df.loc[has_iec, "yi_iec"] = np.arctanh(df.loc[has_iec, "iec"].clip(-0.999, 0.999))

    # ICC-ECC discrepancy
    has_both = df["ecc"].notna()
    df.loc[has_both, "disc"] = df.loc[has_both, "icc"] - df.loc[has_both, "ecc"]

    print("=" * 78)
    print("GREENWALD (2009) — EXHAUSTIVE FEATURE AUDIT")
    print("Senior Research Advisory Analysis")
    print("=" * 78)
    print(f"\nLoaded {len(df)} independent samples")
    print(f"DLN stages: {df['dln_stage'].value_counts().sort_index().to_dict()}")

    # ================================================================
    # 1. UNIVARIATE CORRELATIONS WITH ICC (raw r)
    # ================================================================
    print(f"\n{'=' * 78}")
    print("1. UNIVARIATE CORRELATIONS WITH ICC")
    print("=" * 78)
    print(f"{'Feature':<20s} {'r':>8s} {'p':>10s} {'rho(Sp)':>8s} {'p(Sp)':>10s} {'k':>5s}")
    print("-" * 65)

    numeric_cols = ["n", "n_crit", "n_iat", "n_expl", "iec"]
    for col in numeric_cols:
        valid = df[[col, "icc"]].dropna()
        if len(valid) < 5:
            print(f"{col:<20s} {'---':>8s} {'---':>10s} {'---':>8s} {'---':>10s} {len(valid):5d}")
            continue
        r, p = pearsonr(valid[col], valid["icc"])
        rho, p_sp = spearmanr(valid[col], valid["icc"])
        sig = "*" if p < 0.05 else " "
        print(f"{col:<20s} {r:+8.3f} {p:10.4f}{sig} {rho:+8.3f} {p_sp:10.4f} {len(valid):5d}")

    # ================================================================
    # 2. CATEGORICAL FEATURES: ETA-SQUARED ON ICC
    # ================================================================
    print(f"\n{'=' * 78}")
    print("2. CATEGORICAL FEATURES — ETA-SQUARED ON ICC")
    print("=" * 78)
    print(f"{'Feature':<20s} {'eta²':>8s} {'p(F)':>10s} {'levels':>8s} {'k':>5s}")
    print("-" * 55)

    cat_cols = ["topic", "iat_type", "expl_type", "dln_stage"]
    for col in cat_cols:
        valid = df[[col, "icc"]].dropna()
        e2, p_val, k_used = eta_squared(valid[col], valid["icc"])
        n_levels = valid[col].nunique()
        sig = "*" if p_val < 0.05 else " "
        print(f"{col:<20s} {e2:8.4f} {p_val:10.4f}{sig} {n_levels:8d} {k_used:5d}")

    # ================================================================
    # 3. PREDICTOR CORRELATION MATRIX
    # ================================================================
    print(f"\n{'=' * 78}")
    print("3. PREDICTOR CORRELATION MATRIX (pairwise Pearson)")
    print("=" * 78)

    pred_cols = ["n", "n_crit", "n_iat", "n_expl", "icc", "ecc", "iec"]
    corr_df = df[pred_cols].corr(method="pearson", min_periods=10)
    print(corr_df.round(3).to_string())

    # ================================================================
    # 4. iec AS CONTINUOUS DLN STAGE PROXY
    # ================================================================
    print(f"\n{'=' * 78}")
    print("4. iec (IMPLICIT-EXPLICIT CORRELATION) AS DLN STAGE PROXY")
    print("=" * 78)

    df_iec = df.dropna(subset=["iec"]).copy()
    print(f"  Samples with iec: k={len(df_iec)} / {len(df)}")

    # iec by DLN stage
    print(f"\n  iec by DLN stage:")
    for stage in ["dot", "linear", "network"]:
        sub = df_iec[df_iec["dln_stage"] == stage]
        if len(sub) > 0:
            print(f"    {stage:<10s}: k={len(sub):3d}, mean iec={sub['iec'].mean():.3f}, "
                  f"sd={sub['iec'].std():.3f}, median={sub['iec'].median():.3f}")

    # DLN prediction: Network > Dot > Linear for iec?
    # (Network = integration → high iec; Dot = reactive → moderate; Linear = suppression → low)
    e2_iec, p_iec, k_iec = eta_squared(df_iec["dln_stage"], df_iec["iec"])
    print(f"\n  DLN stage -> iec: eta²={e2_iec:.4f}, p={p_iec:.4f}, k={k_iec}")

    # iec as moderator of ICC (does iec explain ICC variance?)
    print(f"\n  iec -> ICC correlation:")
    r_iec_icc, p_iec_icc = pearsonr(df_iec["iec"], df_iec["icc"])
    print(f"    r(iec, icc) = {r_iec_icc:+.3f}, p = {p_iec_icc:.4f}")

    # iec as moderator in meta-regression
    y_iec = df_iec["yi_icc"].to_numpy()
    v_iec = df_iec["vi_icc"].to_numpy()
    k_iec = len(df_iec)

    X_base = np.ones((k_iec, 1))
    res_base_iec = fit_reml(y_iec, v_iec, X_base)

    X_iec_mod = np.column_stack([np.ones(k_iec), df_iec["iec"].to_numpy()])
    res_iec_mod = fit_reml(y_iec, v_iec, X_iec_mod)

    delta_iec = res_base_iec.tau2 - res_iec_mod.tau2
    pct_iec = (delta_iec / res_base_iec.tau2 * 100) if res_base_iec.tau2 > 0 else 0
    print(f"\n  iec as meta-regression moderator (k={k_iec}):")
    print(f"    Baseline tau2: {res_base_iec.tau2:.6f}")
    print(f"    + iec tau2:    {res_iec_mod.tau2:.6f}")
    print(f"    Reduction:     {pct_iec:.1f}%")
    print(f"    iec coeff:     b={res_iec_mod.beta[1]:.4f} "
          f"[{res_iec_mod.ci95[1,0]:.4f}, {res_iec_mod.ci95[1,1]:.4f}]")

    # iec BEYOND DLN stage (incremental)
    X_stage_iec, names_si = design_matrix_stage(df_iec["dln_stage"], reference="dot")
    res_stage_only = fit_reml(y_iec, v_iec, X_stage_iec)

    X_stage_plus_iec = np.column_stack([X_stage_iec, df_iec["iec"].to_numpy()])
    res_stage_plus_iec = fit_reml(y_iec, v_iec, X_stage_plus_iec)

    incr = res_stage_only.tau2 - res_stage_plus_iec.tau2
    incr_pct = (incr / res_stage_only.tau2 * 100) if res_stage_only.tau2 > 0 else 0
    print(f"\n  Incremental beyond DLN stage:")
    print(f"    Stage-only tau2:      {res_stage_only.tau2:.6f}")
    print(f"    Stage + iec tau2:     {res_stage_plus_iec.tau2:.6f}")
    print(f"    Incremental reduction: {incr_pct:.1f}%")

    # ================================================================
    # 5. iec BY TOPIC (is it a better proxy than topic for DLN stage?)
    # ================================================================
    print(f"\n{'=' * 78}")
    print("5. iec DISTRIBUTION BY TOPIC DOMAIN")
    print("=" * 78)
    for topic in sorted(df_iec["topic"].unique()):
        sub = df_iec[df_iec["topic"] == topic]
        if len(sub) >= 3:
            print(f"  {topic:<20s}: k={len(sub):3d}, mean iec={sub['iec'].mean():.3f}, "
                  f"sd={sub['iec'].std():.3f}")

    # ================================================================
    # 6. n_crit AS CONTINUOUS MODERATOR (K-proxy)
    # ================================================================
    print(f"\n{'=' * 78}")
    print("6. n_crit AS CONTINUOUS K-PROXY MODERATOR")
    print("=" * 78)

    y_all = df["yi_icc"].to_numpy()
    v_all = df["vi_icc"].to_numpy()
    k_all = len(df)

    X_base_all = np.ones((k_all, 1))
    res_base_all = fit_reml(y_all, v_all, X_base_all)

    # n_crit alone
    X_ncrit = np.column_stack([np.ones(k_all), df["n_crit"].to_numpy()])
    res_ncrit = fit_reml(y_all, v_all, X_ncrit)
    d_ncrit = res_base_all.tau2 - res_ncrit.tau2
    pct_ncrit = (d_ncrit / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    print(f"  n_crit alone: tau2 reduction = {pct_ncrit:.1f}%")
    print(f"    b = {res_ncrit.beta[1]:.4f} [{res_ncrit.ci95[1,0]:.4f}, {res_ncrit.ci95[1,1]:.4f}]")

    # n_iat alone
    X_niat = np.column_stack([np.ones(k_all), df["n_iat"].to_numpy()])
    res_niat = fit_reml(y_all, v_all, X_niat)
    d_niat = res_base_all.tau2 - res_niat.tau2
    pct_niat = (d_niat / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    print(f"\n  n_iat alone: tau2 reduction = {pct_niat:.1f}%")
    print(f"    b = {res_niat.beta[1]:.4f} [{res_niat.ci95[1,0]:.4f}, {res_niat.ci95[1,1]:.4f}]")

    # n_expl alone (subset with data)
    df_nexpl = df.dropna(subset=["n_expl"]).copy()
    y_ne = df_nexpl["yi_icc"].to_numpy()
    v_ne = df_nexpl["vi_icc"].to_numpy()
    k_ne = len(df_nexpl)
    X_base_ne = np.ones((k_ne, 1))
    res_base_ne = fit_reml(y_ne, v_ne, X_base_ne)
    X_nexpl = np.column_stack([np.ones(k_ne), df_nexpl["n_expl"].to_numpy()])
    res_nexpl = fit_reml(y_ne, v_ne, X_nexpl)
    d_nexpl = res_base_ne.tau2 - res_nexpl.tau2
    pct_nexpl = (d_nexpl / res_base_ne.tau2 * 100) if res_base_ne.tau2 > 0 else 0
    print(f"\n  n_expl alone (k={k_ne}): tau2 reduction = {pct_nexpl:.1f}%")
    print(f"    b = {res_nexpl.beta[1]:.4f} [{res_nexpl.ci95[1,0]:.4f}, {res_nexpl.ci95[1,1]:.4f}]")

    # DLN stage alone (for comparison)
    X_stage, names_stage = design_matrix_stage(df["dln_stage"], reference="dot")
    res_stage = fit_reml(y_all, v_all, X_stage)
    d_stage = res_base_all.tau2 - res_stage.tau2
    pct_stage = (d_stage / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    print(f"\n  DLN stage alone: tau2 reduction = {pct_stage:.1f}%")

    # Combined: stage + n_crit + n_iat
    X_combined = np.column_stack([X_stage, df["n_crit"].to_numpy(), df["n_iat"].to_numpy()])
    res_combined = fit_reml(y_all, v_all, X_combined)
    d_comb = res_base_all.tau2 - res_combined.tau2
    pct_comb = (d_comb / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    combined_names = names_stage + ["n_crit", "n_iat"]
    print(f"\n  Stage + n_crit + n_iat: tau2 reduction = {pct_comb:.1f}%")
    for i, name in enumerate(combined_names):
        print(f"    {name}: b={res_combined.beta[i]:.4f} "
              f"[{res_combined.ci95[i,0]:.4f}, {res_combined.ci95[i,1]:.4f}]")

    # ================================================================
    # 7. IAT-TYPE × STAGE INTERACTION (cross-level matching)
    # ================================================================
    print(f"\n{'=' * 78}")
    print("7. IAT-TYPE × DLN-STAGE INTERACTION (cross-level matching)")
    print("=" * 78)
    print("\n  DLN theory predicts cross-level matching:")
    print("    Attitude IAT (reactive evaluative) matches Dot criteria")
    print("    Self IAT (self-concept) matches Linear/Network criteria")
    print("    Belief IAT (propositional) matches Linear criteria")

    # Mean ICC by iat_type × dln_stage
    print(f"\n  {'iat_type':<15s} {'dot':>12s} {'linear':>12s} {'network':>12s}")
    print("  " + "-" * 55)
    for iat in sorted(df["iat_type"].unique()):
        row = f"  {iat:<15s}"
        for stage in ["dot", "linear", "network"]:
            sub = df[(df["iat_type"] == iat) & (df["dln_stage"] == stage)]
            if len(sub) >= 2:
                row += f" {sub['icc'].mean():+.3f}({len(sub):2d})"
            elif len(sub) == 1:
                row += f" {sub['icc'].mean():+.3f}( {len(sub):1d})"
            else:
                row += f" {'---':>9s}   "
        print(row)

    # Code iat_type into DLN levels for matching
    IAT_DLN = {
        "Attitude": "dot",       # reactive evaluative
        "Self": "linear",        # self-concept, independent estimates
        "Belief": "linear",      # propositional
        "Att/belief": "linear",  # mixed
        "Multiple": "linear",    # aggregated
    }
    df["iat_dln"] = df["iat_type"].map(IAT_DLN)
    df["match"] = (df["iat_dln"] == df["dln_stage"]).astype(int)

    n_match = df["match"].sum()
    n_nomatch = len(df) - n_match
    mean_match = df[df["match"] == 1]["icc"].mean()
    mean_nomatch = df[df["match"] == 0]["icc"].mean()
    print(f"\n  Cross-level matching (IAT-type DLN level = criterion DLN level):")
    print(f"    Match:    k={n_match:3d}, mean ICC={mean_match:.3f}")
    print(f"    Mismatch: k={n_nomatch:3d}, mean ICC={mean_nomatch:.3f}")

    # Match as meta-regression moderator
    X_match = np.column_stack([np.ones(k_all), df["match"].to_numpy()])
    res_match = fit_reml(y_all, v_all, X_match)
    d_match = res_base_all.tau2 - res_match.tau2
    pct_match = (d_match / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    print(f"    tau2 reduction: {pct_match:.1f}%")
    print(f"    match coeff: b={res_match.beta[1]:.4f} "
          f"[{res_match.ci95[1,0]:.4f}, {res_match.ci95[1,1]:.4f}]")

    # ================================================================
    # 8. n_crit × STAGE INTERACTION
    # ================================================================
    print(f"\n{'=' * 78}")
    print("8. n_crit × STAGE INTERACTION (K-effect varies by topology)")
    print("=" * 78)
    print("\n  DLN predicts: n_crit penalty LARGER for Linear (independent estimates)")
    print("  and SMALLER for Network (factor structure → criteria covary)")

    # n_crit correlation with ICC, by stage
    for stage in ["dot", "linear", "network"]:
        sub = df[df["dln_stage"] == stage]
        if len(sub) >= 5:
            r, p = pearsonr(sub["n_crit"], sub["icc"])
            print(f"    {stage:<10s}: r(n_crit, icc) = {r:+.3f}, p = {p:.4f}, k = {len(sub)}")

    # Formal interaction test: stage + n_crit + stage*n_crit
    # Create interaction terms
    X_interaction = X_stage.copy()
    n_crit_vec = df["n_crit"].to_numpy()
    X_interaction = np.column_stack([X_interaction, n_crit_vec])
    # Add interactions: linear*n_crit, network*n_crit
    for i, name in enumerate(names_stage):
        if name != "Intercept":
            X_interaction = np.column_stack([X_interaction, X_stage[:, i] * n_crit_vec])

    res_interaction = fit_reml(y_all, v_all, X_interaction)
    d_int = res_base_all.tau2 - res_interaction.tau2
    pct_int = (d_int / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0

    int_names = names_stage + ["n_crit", "linear×n_crit", "network×n_crit"]
    print(f"\n  Stage + n_crit + stage×n_crit: tau2 reduction = {pct_int:.1f}%")
    for i, name in enumerate(int_names):
        print(f"    {name}: b={res_interaction.beta[i]:.4f} "
              f"[{res_interaction.ci95[i,0]:.4f}, {res_interaction.ci95[i,1]:.4f}]")

    # Compare to additive model
    incr_int = res_combined.tau2 - res_interaction.tau2
    incr_int_pct = (incr_int / res_combined.tau2 * 100) if res_combined.tau2 > 0 else 0
    print(f"\n  Incremental from interaction terms: {incr_int_pct:.1f}% beyond additive model")

    # ================================================================
    # 9. WITHIN-STAGE VARIANCE DECOMPOSITION
    # ================================================================
    print(f"\n{'=' * 78}")
    print("9. WITHIN-STAGE VARIANCE DECOMPOSITION")
    print("=" * 78)
    print("  (How much heterogeneity remains WITHIN each DLN stage?)")

    for stage in ["dot", "linear", "network"]:
        sub = df[df["dln_stage"] == stage].copy()
        if len(sub) < 5:
            continue
        y_s = sub["yi_icc"].to_numpy()
        v_s = sub["vi_icc"].to_numpy()
        X_s = np.ones((len(sub), 1))
        res_s = fit_reml(y_s, v_s, X_s)
        mean_r = np.tanh(res_s.beta[0])
        print(f"\n  {stage} (k={len(sub)}):")
        print(f"    mean r = {mean_r:.3f}, tau2 = {res_s.tau2:.6f}, I2 = {res_s.I2:.1%}")

        # Which features explain within-stage variance?
        for feat in ["n_crit", "n_iat", "n"]:
            X_feat = np.column_stack([np.ones(len(sub)), sub[feat].to_numpy()])
            res_feat = fit_reml(y_s, v_s, X_feat)
            d_f = res_s.tau2 - res_feat.tau2
            pct_f = (d_f / res_s.tau2 * 100) if res_s.tau2 > 0 else 0
            sig = "*" if abs(res_feat.ci95[1,0]) > 0 and np.sign(res_feat.ci95[1,0]) == np.sign(res_feat.ci95[1,1]) else " "
            print(f"    + {feat:<8s}: reduction={pct_f:+6.1f}%, "
                  f"b={res_feat.beta[1]:.4f} [{res_feat.ci95[1,0]:.4f}, {res_feat.ci95[1,1]:.4f}]{sig}")

        # Topic within stage
        if sub["topic"].nunique() >= 2:
            e2, p_val, _ = eta_squared(sub["topic"], sub["icc"])
            print(f"    topic within {stage}: eta²={e2:.4f}, p={p_val:.4f}")

    # ================================================================
    # 10. FULL MODEL COMPARISON TABLE
    # ================================================================
    print(f"\n{'=' * 78}")
    print("10. FULL MODEL COMPARISON (tau2 reduction from baseline)")
    print("=" * 78)

    models = []

    # Baseline
    models.append(("Baseline (intercept only)", res_base_all.tau2, 0.0, 1, res_base_all.I2))

    # Single moderators
    models.append(("DLN stage (3-level)", res_stage.tau2, pct_stage, 3, res_stage.I2))
    models.append(("n_crit alone", res_ncrit.tau2, pct_ncrit, 2, res_ncrit.I2))
    models.append(("n_iat alone", res_niat.tau2, pct_niat, 2, res_niat.I2))

    # Topic
    X_topic, _ = design_matrix_categorical(df["topic"])
    res_topic = fit_reml(y_all, v_all, X_topic)
    d_topic = res_base_all.tau2 - res_topic.tau2
    pct_topic = (d_topic / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    models.append(("topic (9-level)", res_topic.tau2, pct_topic, 9, res_topic.I2))

    # iat_type
    X_iat, _ = design_matrix_categorical(df["iat_type"])
    res_iat = fit_reml(y_all, v_all, X_iat)
    d_iat = res_base_all.tau2 - res_iat.tau2
    pct_iat = (d_iat / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    models.append(("iat_type (5-level)", res_iat.tau2, pct_iat, 5, res_iat.I2))

    # Cross-level match
    models.append(("IAT-criterion match", res_match.tau2, pct_match, 2, res_match.I2))

    # Combined additive
    models.append(("Stage + n_crit + n_iat", res_combined.tau2, pct_comb, 5, res_combined.I2))

    # Combined with interaction
    models.append(("Stage + n_crit + n_iat + interactions", res_interaction.tau2, pct_int, 7, res_interaction.I2))

    print(f"\n  {'Model':<42s} {'tau2':>10s} {'%Red':>7s} {'#p':>4s} {'I2':>8s}")
    print("  " + "-" * 75)
    for name, tau2, pct, np_, i2 in models:
        print(f"  {name:<42s} {tau2:10.6f} {pct:+6.1f}% {np_:4d} {i2:7.1%}")

    # ================================================================
    # 11. EXPL_TYPE ANALYSIS
    # ================================================================
    print(f"\n{'=' * 78}")
    print("11. expl_type — CONFOUNDING WITH TOPIC?")
    print("=" * 78)

    df_expl = df.dropna(subset=["expl_type"]).copy()
    print(f"  Samples with expl_type: k={len(df_expl)}")

    # Cross-tabulation
    ct = pd.crosstab(df_expl["topic"], df_expl["expl_type"])
    print(f"\n  topic × expl_type cross-tabulation:")
    print(ct.to_string())

    # eta² for expl_type on ICC
    e2_et, p_et, k_et = eta_squared(df_expl["expl_type"], df_expl["icc"])
    print(f"\n  expl_type → ICC: eta²={e2_et:.4f}, p={p_et:.4f}, k={k_et}")

    # expl_type WITHIN DLN stages
    for stage in ["dot", "linear", "network"]:
        sub = df_expl[df_expl["dln_stage"] == stage]
        if sub["expl_type"].nunique() >= 2 and len(sub) >= 5:
            e2_s, p_s, k_s = eta_squared(sub["expl_type"], sub["icc"])
            print(f"  expl_type within {stage}: eta²={e2_s:.4f}, p={p_s:.4f}, k={k_s}")

    # ================================================================
    # 12. DLN-THEORETIC iec PREDICTION CHECK
    # ================================================================
    print(f"\n{'=' * 78}")
    print("12. DLN-THEORETIC iec PREDICTIONS")
    print("=" * 78)
    print("  Theory: iec reflects integration level")
    print("    Network: HIGH iec (integration aligns implicit + explicit)")
    print("    Dot: LOW-MODERATE iec (reactive; explicit can't access)")
    print("    Linear: LOW iec (compartmentalization suppresses alignment)")

    for stage in ["dot", "linear", "network"]:
        sub = df_iec[df_iec["dln_stage"] == stage]
        if len(sub) > 0:
            print(f"    {stage:<10s}: k={len(sub):3d}, mean={sub['iec'].mean():.3f}, "
                  f"sd={sub['iec'].std():.3f}, range=[{sub['iec'].min():.3f}, {sub['iec'].max():.3f}]")

    # Does iec predict DISCREPANCY (ICC - ECC)?
    df_disc = df.dropna(subset=["disc", "iec"]).copy()
    if len(df_disc) >= 10:
        r_iec_disc, p_iec_disc = pearsonr(df_disc["iec"], df_disc["disc"])
        print(f"\n  iec → discrepancy (ICC-ECC): r={r_iec_disc:+.3f}, p={p_iec_disc:.4f}, k={len(df_disc)}")

    # ================================================================
    # 13. LINEAR-PLUS FEASIBILITY
    # ================================================================
    print(f"\n{'=' * 78}")
    print("13. LINEAR-PLUS FEASIBILITY — CAN WE SPLIT LINEAR?")
    print("=" * 78)
    print("  The notes mention a 4-level coding: Dot / Linear / Linear-Plus / Network")
    print("  Linear-Plus = norm-governed behavior (social desirability pressure)")
    print("  Linear = simple independent evaluation (no suppression)")
    print()

    # Within the current 'linear' stage (k=136), what domains are present?
    linear_df = df[df["dln_stage"] == "linear"].copy()
    print(f"  Current linear stage: k={len(linear_df)}")
    print(f"  Topics within linear:")
    for topic in sorted(linear_df["topic"].unique()):
        sub = linear_df[linear_df["topic"] == topic]
        print(f"    {topic:<20s}: k={len(sub):3d}, mean ICC={sub['icc'].mean():.3f}, "
              f"sd={sub['icc'].std():.3f}")

    # Candidate split: Consumer (independent evaluation) vs intergroup (norm-governed)
    # Under current study-level coding, Consumer is already in linear.
    # Potential Linear-Plus: Race, Gender, Other intergroup (social desirability)
    # Potential Linear: Consumer, Personality, Drugs/tobacco, Relationships, some Clinical

    NORM_GOVERNED = {"Race (Bl/Wh)", "Gender/sex", "Other intergroup", "Politics"}
    linear_df["subtype"] = linear_df["topic"].apply(
        lambda t: "linear_plus" if t in NORM_GOVERNED else "linear"
    )

    for sub_type in ["linear", "linear_plus"]:
        sub = linear_df[linear_df["subtype"] == sub_type]
        if len(sub) > 0:
            print(f"\n  {sub_type:<15s}: k={len(sub):3d}, mean ICC={sub['icc'].mean():.3f}, "
                  f"sd={sub['icc'].std():.3f}")
            if sub.dropna(subset=["ecc"]).shape[0] > 0:
                sub_ecc = sub.dropna(subset=["ecc"])
                print(f"    {'':15s}  mean ECC={sub_ecc['ecc'].mean():.3f}, "
                      f"mean gap={sub_ecc['disc'].mean():+.3f}")

    # Test if the split helps
    df_4level = df.copy()
    df_4level["dln_stage_4"] = df_4level["dln_stage"]
    mask_lp = (df_4level["dln_stage"] == "linear") & (df_4level["topic"].isin(NORM_GOVERNED))
    df_4level.loc[mask_lp, "dln_stage_4"] = "linear_plus"

    print(f"\n  4-level distribution: {df_4level['dln_stage_4'].value_counts().sort_index().to_dict()}")

    X_4, names_4 = design_matrix_categorical(df_4level["dln_stage_4"], reference="dot")
    res_4 = fit_reml(y_all, v_all, X_4)
    d_4 = res_base_all.tau2 - res_4.tau2
    pct_4 = (d_4 / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    print(f"\n  4-level coding: tau2 reduction = {pct_4:.1f}%")
    for i, name in enumerate(names_4):
        print(f"    {name}: b={res_4.beta[i]:.4f} [{res_4.ci95[i,0]:.4f}, {res_4.ci95[i,1]:.4f}]")

    # 4-level + n_crit + n_iat
    X_4_combined = np.column_stack([X_4, df["n_crit"].to_numpy(), df["n_iat"].to_numpy()])
    res_4_comb = fit_reml(y_all, v_all, X_4_combined)
    d_4c = res_base_all.tau2 - res_4_comb.tau2
    pct_4c = (d_4c / res_base_all.tau2 * 100) if res_base_all.tau2 > 0 else 0
    print(f"\n  4-level + n_crit + n_iat: tau2 reduction = {pct_4c:.1f}%")

    # Also compute I²
    print(f"    I2(baseline) = {res_base_all.I2:.1%}")
    print(f"    I2(4-level + n_crit + n_iat) = {res_4_comb.I2:.1%}")

    # ================================================================
    # 14. DISCREPANCY ANALYSIS WITH FULL MODEL
    # ================================================================
    print(f"\n{'=' * 78}")
    print("14. DISCREPANCY (ICC - ECC) WITH FULL MODEL")
    print("=" * 78)

    df_d = df.dropna(subset=["disc"]).copy()
    yi_d = np.arctanh(df_d["disc"].clip(-0.999, 0.999))
    vi_d = 1.0 / (df_d["n"] - 3) * 2  # approximate variance of difference

    k_d = len(df_d)
    X_bd = np.ones((k_d, 1))
    res_bd = fit_reml(yi_d, vi_d, X_bd)

    # 4-level on discrepancy
    df_d["dln_stage_4"] = df_d["dln_stage"]
    mask_lp_d = (df_d["dln_stage"] == "linear") & (df_d["topic"].isin(NORM_GOVERNED))
    df_d.loc[mask_lp_d, "dln_stage_4"] = "linear_plus"

    X_d4, names_d4 = design_matrix_categorical(df_d["dln_stage_4"], reference="dot")
    res_d4 = fit_reml(yi_d, vi_d, X_d4)
    d_d4 = res_bd.tau2 - res_d4.tau2
    pct_d4 = (d_d4 / res_bd.tau2 * 100) if res_bd.tau2 > 0 else 0
    print(f"  Discrepancy baseline: tau2={res_bd.tau2:.6f}, I2={res_bd.I2:.1%}")
    print(f"  4-level on discrepancy: tau2 reduction = {pct_d4:.1f}%")
    for i, name in enumerate(names_d4):
        print(f"    {name}: b={res_d4.beta[i]:.4f} [{res_d4.ci95[i,0]:.4f}, {res_d4.ci95[i,1]:.4f}]")

    # Stage means for discrepancy
    print(f"\n  Discrepancy means by 4-level stage:")
    for stage in ["dot", "linear", "linear_plus", "network"]:
        sub = df_d[df_d["dln_stage_4"] == stage]
        if len(sub) > 0:
            print(f"    {stage:<15s}: k={len(sub):3d}, mean(ICC-ECC)={sub['disc'].mean():+.3f}, "
                  f"sd={sub['disc'].std():.3f}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 78}")
    print("ADVISORY SUMMARY")
    print("=" * 78)
    print("""
Key findings from exhaustive feature audit:

1. CONTINUOUS MODERATORS (n_crit, n_iat) add signal beyond stage coding.
   The combined additive model (stage + n_crit + n_iat) outperforms stage alone.

2. iec (implicit-explicit correlation) is a DLN-theoretic variable that
   could serve as a continuous proxy for integration level. Check whether
   it adds incremental prediction beyond categorical stage coding.

3. The LINEAR-PLUS split (norm-governed vs. simple evaluation) is
   empirically testable. If it improves tau2 reduction on DISCREPANCY,
   it has DLN-theoretic justification.

4. Cross-level matching (IAT type DLN = criterion DLN) is a novel
   prediction from DLN theory. Worth testing but likely weak signal.

5. n_crit × stage interaction tests whether the K penalty varies by
   topology (DLN Prop 1i predicts it should). This is a strong
   theoretical test regardless of effect size.
""")


if __name__ == "__main__":
    main()
