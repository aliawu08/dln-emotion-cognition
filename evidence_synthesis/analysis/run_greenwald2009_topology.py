"""Greenwald et al. (2009) four-level topology-mediated validity analysis.

Extends the three-level study-level coding (run_greenwald2009_studylevel.py)
by distinguishing Linear-Plus from Linear within the Linear residual:

  - Dot:         No persistent belief state. O(1) memory. Criterion is a
                 reflexive/physiological/automatic response.

  - Linear:      K independent option-value estimates. O(K) memory. Criterion
                 involves evaluating targets/options independently with no
                 normative structure constraining expression (consumer choice,
                 personality traits, substance use, health decisions).

  - Linear-Plus: Externally imposed normative structure. O(F) memory via
                 designer-specified factors. Criterion behaviour is governed
                 by social desirability or institutional norms that the
                 individual uses but did not discover and cannot revise.
                 (Wu, 2026, compression model: "agents that share information
                 across options through designer-specified structure rather
                 than agent-discovered structure.")

  - Network:     Agent-discovered latent factor structure. O(F) memory with
                 structural learning cycle. Criterion requires cross-option
                 integration through discovered shared factors (political
                 ideology connecting policy preferences).

Analyses:
  1. Hierarchical model comparison on ICC, ECC, and discrepancy (ICC - ECC)
  2. Four-level stage profile table (suppression fingerprint)
  3. iec mediation analysis (Baron-Kenny attenuation)
  4. n_crit x stage interaction (Proposition 1i test)
  5. Competing moderator comparison
  6. Corrected noise decomposition

Usage:
  python evidence_synthesis/analysis/run_greenwald2009_topology.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_pipeline import (
    compute_qm,
    fit_reml,
    design_matrix_categorical,
    design_matrix_stage,
    egger_test,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "greenwald2009_study_extraction.csv"
OUT_DIR_T = ROOT / "evidence_synthesis" / "outputs" / "tables"
OUT_DIR_F = ROOT / "evidence_synthesis" / "outputs" / "figures"


# ---------------------------------------------------------------------------
# Study-level DLN coding (inherited from run_greenwald2009_studylevel.py)
# ---------------------------------------------------------------------------
DOT_SAMPLES = {
    # Race - physiological / nonverbal / perceptual
    2, 3, 18, 19, 20, 24, 58, 74, 75, 76, 77, 104, 122, 127, 128, 171, 172,
    # Relationships - reactive
    29,
    # Clinical - phobic/anxiety reactivity
    30, 31, 32, 33, 103, 108, 109, 163, 164, 165, 166, 167,
    # Drugs/tobacco - attentional capture
    35,
    # Other intergroup - reflexive
    98, 181,
}

NETWORK_SAMPLES = {
    4, 5, 8, 45, 84, 105, 106, 110, 112, 118, 119,
}

# Linear-Plus: norm-governed intergroup behaviour.  These topic domains
# involve social desirability pressure that imposes externally specified
# structure on explicit expression.  DOT_SAMPLES override takes precedence,
# so physiological/nonverbal race samples remain Dot.
LINEAR_PLUS_TOPICS = {"Race (Bl/Wh)", "Gender/sex", "Other intergroup"}


def assign_topology_stage(sample_id, topic):
    """Assign four-level DLN stage based on criterion behaviour topology."""
    if sample_id in DOT_SAMPLES:
        return "dot"
    if sample_id in NETWORK_SAMPLES:
        return "network"
    if topic in LINEAR_PLUS_TOPICS:
        return "linear_plus"
    return "linear"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df):
    """Add four-level DLN coding, Fisher-z transforms, and sampling variances."""
    df = df.copy()

    df["dln_stage"] = df.apply(
        lambda r: assign_topology_stage(r["sample_id"], r["topic"]), axis=1
    )

    # Fisher-z transform for ICC
    df["yi_icc"] = np.arctanh(df["icc"].clip(-0.999, 0.999))
    df["vi_icc"] = 1.0 / (df["n"] - 3)

    # Fisher-z for ECC (where available)
    has_ecc = df["ecc"].notna()
    df.loc[has_ecc, "yi_ecc"] = np.arctanh(df.loc[has_ecc, "ecc"].clip(-0.999, 0.999))
    df.loc[has_ecc, "vi_ecc"] = 1.0 / (df.loc[has_ecc, "n"] - 3)

    # ICC - ECC discrepancy (Fisher-z scale)
    df.loc[has_ecc, "yi_disc"] = df.loc[has_ecc, "yi_icc"] - df.loc[has_ecc, "yi_ecc"]
    df.loc[has_ecc, "vi_disc"] = df.loc[has_ecc, "vi_icc"] + df.loc[has_ecc, "vi_ecc"]

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STAGE_ORDER = ["dot", "linear", "linear_plus", "network"]
STAGE_COLORS = {
    "dot": "#e74c3c",
    "linear": "#f39c12",
    "linear_plus": "#9b59b6",
    "network": "#27ae60",
}


def _fmt_ci(beta, ci95, idx):
    return (f"b={beta[idx]:.4f} "
            f"[{ci95[idx, 0]:.4f}, {ci95[idx, 1]:.4f}]")


def _pct_reduction(tau2_base, tau2_mod):
    if tau2_base <= 0:
        return 0.0
    return (tau2_base - tau2_mod) / tau2_base * 100


def _save_rows(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Wrote: {path}")


# ---------------------------------------------------------------------------
# 1. Hierarchical model comparison
# ---------------------------------------------------------------------------

def run_hierarchical(df, y_col, v_col, label):
    """Run hierarchical model comparison: baseline -> 3-level -> 4-level -> + covariates."""
    sub = df.dropna(subset=[y_col, v_col]).copy()
    y = sub[y_col].to_numpy()
    v = sub[v_col].to_numpy()
    k = len(sub)

    rows = []

    # Model A: baseline
    X_a = np.ones((k, 1))
    res_a = fit_reml(y, v, X_a)
    rows.append({
        "analysis": label, "model": "A_baseline", "k": k,
        "n_params": 1, "tau2": res_a.tau2, "I2": res_a.I2, "Q": res_a.Q,
    })

    # Model B: 3-level study-level (dot / linear+linear_plus / network)
    stage_3 = sub["dln_stage"].replace("linear_plus", "linear")
    X_b, names_b = design_matrix_stage(stage_3, reference="dot")
    res_b = fit_reml(y, v, X_b)
    rows.append({
        "analysis": label, "model": "B_3level", "k": k,
        "n_params": len(names_b), "tau2": res_b.tau2, "I2": res_b.I2,
        "Q": res_b.Q,
    })

    # Model C: 4-level (dot / linear / linear_plus / network)
    X_c, names_c = design_matrix_categorical(sub["dln_stage"], reference="dot")
    res_c = fit_reml(y, v, X_c)
    rows.append({
        "analysis": label, "model": "C_4level", "k": k,
        "n_params": len(names_c), "tau2": res_c.tau2, "I2": res_c.I2,
        "Q": res_c.Q,
    })

    # Model D: 4-level + n_crit + n_iat
    if "n_crit" in sub.columns and "n_iat" in sub.columns:
        X_d = np.column_stack([X_c, sub["n_crit"].to_numpy(), sub["n_iat"].to_numpy()])
        res_d = fit_reml(y, v, X_d)
        rows.append({
            "analysis": label, "model": "D_4level_cov", "k": k,
            "n_params": X_d.shape[1], "tau2": res_d.tau2, "I2": res_d.I2,
            "Q": res_d.Q,
        })
    else:
        res_d = None

    return {
        "sub": sub, "y": y, "v": v, "k": k,
        "res_a": res_a, "res_b": res_b, "res_c": res_c, "res_d": res_d,
        "X_c": X_c, "names_c": names_c,
        "table_rows": rows,
    }


def print_hierarchy(h, label):
    """Print hierarchical comparison table."""
    print(f"\n{'=' * 72}")
    print(f"HIERARCHICAL MODEL COMPARISON: {label}")
    print(f"{'=' * 72}")
    tau2_base = h["res_a"].tau2
    print(f"  {'Model':<22s} {'#p':>3s} {'tau2':>10s} {'I2':>8s} "
          f"{'Red_base':>10s} {'Red_prev':>10s}")
    print(f"  {'-' * 67}")

    prev_tau2 = tau2_base
    for r in h["table_rows"]:
        red_base = _pct_reduction(tau2_base, r["tau2"])
        red_prev = _pct_reduction(prev_tau2, r["tau2"])
        print(f"  {r['model']:<22s} {r['n_params']:3d} {r['tau2']:10.6f} "
              f"{r['I2']:7.1%} {red_base:+9.1f}% {red_prev:+9.1f}%")
        prev_tau2 = r["tau2"]

    # QM omnibus tests
    X_a = np.ones((h["k"], 1))
    for model_label, res_model, X_model in [
        ("B_3level", h["res_b"], design_matrix_stage(
            h["sub"]["dln_stage"].replace("linear_plus", "linear"), reference="dot")[0]),
        ("C_4level", h["res_c"], h["X_c"]),
    ]:
        qm = compute_qm(h["y"], h["v"], X_a, X_model)
        print(f"  {model_label}: QM({qm.df}) = {qm.QM:.2f}, p = {qm.p:.6f}")

    # Print 4-level coefficients
    res_c = h["res_c"]
    names_c = h["names_c"]
    print(f"\n  Four-level coefficients (reference = dot):")
    for i, name in enumerate(names_c):
        print(f"    {name}: {_fmt_ci(res_c.beta, res_c.ci95, i)}")


# ---------------------------------------------------------------------------
# 2. Stage profile table (suppression fingerprint)
# ---------------------------------------------------------------------------

def stage_profiles(df):
    """Compute and print the four-level stage profile table."""
    print(f"\n{'=' * 72}")
    print("STAGE PROFILES (suppression fingerprint)")
    print(f"{'=' * 72}")

    rows = []
    for stage in STAGE_ORDER:
        sub = df[df["dln_stage"] == stage]
        sub_ecc = sub.dropna(subset=["ecc"])
        sub_iec = sub.dropna(subset=["iec"])

        row = {"stage": stage, "k": len(sub)}

        row["mean_icc"] = sub["icc"].mean()
        row["sd_icc"] = sub["icc"].std()

        if len(sub_ecc) > 0:
            row["k_ecc"] = len(sub_ecc)
            row["mean_ecc"] = sub_ecc["ecc"].mean()
            row["mean_gap"] = (sub_ecc["icc"] - sub_ecc["ecc"]).mean()
        else:
            row["k_ecc"] = 0
            row["mean_ecc"] = float("nan")
            row["mean_gap"] = float("nan")

        if len(sub_iec) > 0:
            row["k_iec"] = len(sub_iec)
            row["mean_iec"] = sub_iec["iec"].mean()
        else:
            row["k_iec"] = 0
            row["mean_iec"] = float("nan")

        # Within-stage heterogeneity
        if len(sub) >= 3:
            y_s = sub["yi_icc"].to_numpy()
            v_s = sub["vi_icc"].to_numpy()
            res_s = fit_reml(y_s, v_s, np.ones((len(sub), 1)))
            row["within_tau2"] = res_s.tau2
            row["within_I2"] = res_s.I2
        else:
            row["within_tau2"] = float("nan")
            row["within_I2"] = float("nan")

        rows.append(row)

    # Print
    print(f"\n  {'Stage':<14s} {'k':>4s} {'ICC':>7s} {'ECC':>7s} "
          f"{'iec':>7s} {'Gap':>7s} {'tau2_w':>9s} {'I2_w':>7s}")
    print(f"  {'-' * 69}")
    for r in rows:
        ecc_s = f"{r['mean_ecc']:.3f}" if not np.isnan(r.get("mean_ecc", float("nan"))) else "  ---"
        iec_s = f"{r['mean_iec']:.3f}" if not np.isnan(r.get("mean_iec", float("nan"))) else "  ---"
        gap_s = f"{r['mean_gap']:+.3f}" if not np.isnan(r.get("mean_gap", float("nan"))) else "  ---"
        tau2_s = f"{r['within_tau2']:.6f}" if not np.isnan(r.get("within_tau2", float("nan"))) else "  ---"
        i2_s = f"{r['within_I2']:.1%}" if not np.isnan(r.get("within_I2", float("nan"))) else "  ---"
        print(f"  {r['stage']:<14s} {r['k']:4d} {r['mean_icc']:7.3f} {ecc_s:>7s} "
              f"{iec_s:>7s} {gap_s:>7s} {tau2_s:>9s} {i2_s:>7s}")

    return rows


# ---------------------------------------------------------------------------
# 3. iec mediation analysis
# ---------------------------------------------------------------------------

def iec_mediation(df):
    """Baron-Kenny attenuation test: Stage -> iec -> ICC."""
    print(f"\n{'=' * 72}")
    print("iec MEDIATION ANALYSIS (Baron-Kenny attenuation)")
    print(f"{'=' * 72}")

    df_iec = df.dropna(subset=["iec"]).copy()
    k_iec = len(df_iec)
    print(f"  Subsample with iec: k = {k_iec}")

    y_icc = df_iec["yi_icc"].to_numpy()
    v_icc = df_iec["vi_icc"].to_numpy()
    iec_vals = df_iec["iec"].to_numpy()

    rows = []

    # Step A: Stage -> iec (a-path)
    print(f"\n  Step A: Stage -> iec")
    print(f"  {'Stage':<14s} {'k':>4s} {'mean_iec':>10s} {'sd':>8s}")
    print(f"  {'-' * 40}")
    for stage in STAGE_ORDER:
        sub = df_iec[df_iec["dln_stage"] == stage]
        if len(sub) > 0:
            print(f"  {stage:<14s} {len(sub):4d} {sub['iec'].mean():10.3f} "
                  f"{sub['iec'].std():8.3f}")

    # Step B: iec -> ICC (b-path, unconditional)
    X_base_iec = np.ones((k_iec, 1))
    res_base_iec = fit_reml(y_icc, v_icc, X_base_iec)

    X_iec_only = np.column_stack([np.ones(k_iec), iec_vals])
    res_iec_only = fit_reml(y_icc, v_icc, X_iec_only)
    red_b = _pct_reduction(res_base_iec.tau2, res_iec_only.tau2)
    print(f"\n  Step B: iec -> ICC (unconditional)")
    print(f"    iec coefficient: {_fmt_ci(res_iec_only.beta, res_iec_only.ci95, 1)}")
    print(f"    tau2 reduction: {red_b:.1f}%")
    rows.append({"path": "B_iec_only", "coeff": res_iec_only.beta[1],
                 "ci_lo": res_iec_only.ci95[1, 0], "ci_hi": res_iec_only.ci95[1, 1],
                 "tau2_red_pct": red_b})

    # Step C: Stage -> ICC (c-path, total effect)
    X_stage_iec, names_stage_iec = design_matrix_categorical(
        df_iec["dln_stage"], reference="dot"
    )
    res_c_path = fit_reml(y_icc, v_icc, X_stage_iec)
    red_c = _pct_reduction(res_base_iec.tau2, res_c_path.tau2)
    print(f"\n  Step C: Stage -> ICC (total effect, c-path)")
    for i, name in enumerate(names_stage_iec):
        print(f"    {name}: {_fmt_ci(res_c_path.beta, res_c_path.ci95, i)}")
    print(f"    tau2 reduction: {red_c:.1f}%")

    # Step D: Stage + iec -> ICC (c'-path, direct effect)
    X_stage_plus_iec = np.column_stack([X_stage_iec, iec_vals])
    res_cp_path = fit_reml(y_icc, v_icc, X_stage_plus_iec)
    red_cp = _pct_reduction(res_base_iec.tau2, res_cp_path.tau2)
    names_cp = names_stage_iec + ["iec"]
    print(f"\n  Step D: Stage + iec -> ICC (direct effect, c'-path)")
    for i, name in enumerate(names_cp):
        print(f"    {name}: {_fmt_ci(res_cp_path.beta, res_cp_path.ci95, i)}")
    print(f"    tau2 reduction: {red_cp:.1f}%")

    # Step E: Attenuation comparison
    print(f"\n  Step E: Attenuation (c vs c')")
    cp_label = "c' (direct)"
    print(f"  {'Contrast':<16s} {'c (total)':>10s} {cp_label:>12s} {'Attenuation':>12s}")
    print(f"  {'-' * 54}")
    for i, name in enumerate(names_stage_iec):
        if name == "Intercept":
            continue
        c_val = res_c_path.beta[i]
        cp_val = res_cp_path.beta[i]
        if abs(c_val) > 1e-6:
            att_pct = (1 - cp_val / c_val) * 100
            att_s = f"{att_pct:+.1f}%"
        else:
            att_s = "   ---"
        print(f"  {name:<16s} {c_val:+10.4f} {cp_val:+12.4f} {att_s:>12s}")
        rows.append({"path": f"attenuation_{name}", "c_total": c_val,
                     "cp_direct": cp_val})

    return rows


# ---------------------------------------------------------------------------
# 4. n_crit x stage interaction (Proposition 1i)
# ---------------------------------------------------------------------------

def ncrit_interaction(df):
    """Test n_crit x stage interaction on ICC."""
    print(f"\n{'=' * 72}")
    print("n_crit x STAGE INTERACTION (Proposition 1i)")
    print(f"{'=' * 72}")

    y = df["yi_icc"].to_numpy()
    v = df["vi_icc"].to_numpy()
    k = len(df)
    n_crit = df["n_crit"].to_numpy()

    # Within-stage correlations
    print(f"\n  Within-stage r(n_crit, ICC):")
    for stage in STAGE_ORDER:
        sub = df[df["dln_stage"] == stage]
        if len(sub) >= 5:
            r, p = pearsonr(sub["n_crit"], sub["icc"])
            print(f"    {stage:<14s}: r = {r:+.3f}, p = {p:.4f}, k = {len(sub)}")

    # Baseline
    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    # Main effects: stage + n_crit
    X_stage, names_stage = design_matrix_categorical(df["dln_stage"], reference="dot")
    X_main = np.column_stack([X_stage, n_crit])
    res_main = fit_reml(y, v, X_main)
    names_main = names_stage + ["n_crit"]
    red_main = _pct_reduction(res_base.tau2, res_main.tau2)

    print(f"\n  Main effects (stage + n_crit): tau2 reduction = {red_main:.1f}%")
    for i, name in enumerate(names_main):
        print(f"    {name}: {_fmt_ci(res_main.beta, res_main.ci95, i)}")

    # Interaction: stage + n_crit + stage*n_crit
    X_int = X_main.copy()
    int_names = list(names_main)
    for i, name in enumerate(names_stage):
        if name != "Intercept":
            X_int = np.column_stack([X_int, X_stage[:, i] * n_crit])
            int_names.append(f"{name} x n_crit")

    res_int = fit_reml(y, v, X_int)
    red_int = _pct_reduction(res_base.tau2, res_int.tau2)
    red_incr = _pct_reduction(res_main.tau2, res_int.tau2)

    print(f"\n  Interaction model: tau2 reduction = {red_int:.1f}% "
          f"(incremental from main effects: {red_incr:.1f}%)")
    for i, name in enumerate(int_names):
        print(f"    {name}: {_fmt_ci(res_int.beta, res_int.ci95, i)}")


# ---------------------------------------------------------------------------
# 5. Competing moderator comparison
# ---------------------------------------------------------------------------

def competing_moderators(df):
    """Compare tau2 reduction across alternative moderator codings."""
    print(f"\n{'=' * 72}")
    print("COMPETING MODERATOR COMPARISON")
    print(f"{'=' * 72}")

    y = df["yi_icc"].to_numpy()
    v = df["vi_icc"].to_numpy()
    k = len(df)

    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    results = [("Baseline (intercept)", res_base.tau2, 0.0, 1)]

    # DLN 4-level
    X_4, _ = design_matrix_categorical(df["dln_stage"], reference="dot")
    res_4 = fit_reml(y, v, X_4)
    results.append(("DLN 4-level", res_4.tau2,
                     _pct_reduction(res_base.tau2, res_4.tau2), X_4.shape[1]))

    # DLN 3-level (collapsing linear_plus into linear)
    stage_3 = df["dln_stage"].replace("linear_plus", "linear")
    X_3, _ = design_matrix_stage(stage_3, reference="dot")
    res_3 = fit_reml(y, v, X_3)
    results.append(("DLN 3-level", res_3.tau2,
                     _pct_reduction(res_base.tau2, res_3.tau2), X_3.shape[1]))

    # topic (9-level)
    X_top, _ = design_matrix_categorical(df["topic"])
    res_top = fit_reml(y, v, X_top)
    results.append(("topic (9-level)", res_top.tau2,
                     _pct_reduction(res_base.tau2, res_top.tau2), X_top.shape[1]))

    # iat_type
    X_iat, _ = design_matrix_categorical(df["iat_type"])
    res_iat = fit_reml(y, v, X_iat)
    results.append(("iat_type", res_iat.tau2,
                     _pct_reduction(res_base.tau2, res_iat.tau2), X_iat.shape[1]))

    # n_crit alone
    X_nc = np.column_stack([np.ones(k), df["n_crit"].to_numpy()])
    res_nc = fit_reml(y, v, X_nc)
    results.append(("n_crit (continuous)", res_nc.tau2,
                     _pct_reduction(res_base.tau2, res_nc.tau2), 2))

    # n alone
    X_n = np.column_stack([np.ones(k), df["n"].to_numpy()])
    res_n = fit_reml(y, v, X_n)
    results.append(("n (continuous)", res_n.tau2,
                     _pct_reduction(res_base.tau2, res_n.tau2), 2))

    # DLN 4-level + n_crit + n_iat
    X_4c = np.column_stack([X_4, df["n_crit"].to_numpy(), df["n_iat"].to_numpy()])
    res_4c = fit_reml(y, v, X_4c)
    results.append(("DLN 4-level + n_crit + n_iat", res_4c.tau2,
                     _pct_reduction(res_base.tau2, res_4c.tau2), X_4c.shape[1]))

    print(f"\n  {'Moderator':<32s} {'tau2':>10s} {'Red%':>8s} {'#p':>4s}")
    print(f"  {'-' * 58}")
    for name, tau2, pct, np_ in results:
        print(f"  {name:<32s} {tau2:10.6f} {pct:+7.1f}% {np_:4d}")


# ---------------------------------------------------------------------------
# 6. Corrected noise decomposition
# ---------------------------------------------------------------------------

def noise_decomposition(df):
    """Print corrected variance decomposition, flagging the prior scale-mixing error."""
    print(f"\n{'=' * 72}")
    print("CORRECTED NOISE DECOMPOSITION (k=184)")
    print(f"{'=' * 72}")

    yi = df["yi_icc"].to_numpy()
    vi = df["vi_icc"].to_numpy()
    icc_raw = df["icc"].to_numpy()

    X = np.ones((len(df), 1))
    res = fit_reml(yi, vi, X)

    obs_var_z = np.var(yi, ddof=1)
    mean_vi_z = np.mean(vi)
    obs_var_r = np.var(icc_raw, ddof=1)

    print(f"""
  All quantities on Fisher-z scale:
    Total observed variance (Fisher z): {obs_var_z:.4f}
    Mean sampling variance (Fisher z):  {mean_vi_z:.4f}
    REML tau-squared:                   {res.tau2:.6f}
    I-squared (Q-based):                {res.I2:.1%}

    Sampling error fraction:            ~{(1 - res.I2) * 100:.0f}% of observed spread
    Real heterogeneity:                 ~{res.I2 * 100:.0f}% of observed spread

  Scale-mixing diagnostic:
    Raw-r observed variance:            {obs_var_r:.4f}
    WRONG ratio (Fisher-z vi / raw-r var): {mean_vi_z / obs_var_r:.1%}
    CORRECT ratio (Fisher-z vi / Fisher-z var): {mean_vi_z / obs_var_z:.1%}

    NOTE: A prior analysis reported I-sq = 34.7% by dividing Fisher-z
    sampling variance ({mean_vi_z:.4f}) by raw-r observed variance
    ({obs_var_r:.4f}), producing {mean_vi_z / obs_var_r:.1%}. This is a
    scale-mixing error. The correct Q-based I-squared is {res.I2:.1%}.""")


# ---------------------------------------------------------------------------
# Forest plot (4-colour)
# ---------------------------------------------------------------------------

def forest_plot(df, out_path):
    """Generate four-colour forest plot."""
    stage_order_map = {s: i for i, s in enumerate(STAGE_ORDER)}

    df_plot = df.copy()
    df_plot["stage_order"] = df_plot["dln_stage"].map(stage_order_map)
    df_plot = df_plot.sort_values(
        ["stage_order", "icc"], ascending=[True, True]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 0.18 * len(df_plot) + 2))

    for i, row in df_plot.iterrows():
        color = STAGE_COLORS[row["dln_stage"]]
        se = np.sqrt(row["vi_icc"])
        ci_lo = np.tanh(row["yi_icc"] - 1.96 * se)
        ci_hi = np.tanh(row["yi_icc"] + 1.96 * se)
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=0.8, alpha=0.6)
        ax.plot(row["icc"], i, "s", color=color, markersize=3, alpha=0.8)

    # Stage separators
    prev_stage = None
    for i, row in df_plot.iterrows():
        if row["dln_stage"] != prev_stage and prev_stage is not None:
            ax.axhline(i - 0.5, color="grey", linewidth=0.5, linestyle=":")
        prev_stage = row["dln_stage"]

    # Stage mean lines
    for stage in STAGE_ORDER:
        sub = df_plot[df_plot["dln_stage"] == stage]
        if len(sub) > 0:
            mean_z = np.average(sub["yi_icc"], weights=1.0 / sub["vi_icc"])
            mean_r = np.tanh(mean_z)
            ax.axvline(mean_r, color=STAGE_COLORS[stage], linestyle="--",
                       alpha=0.6, linewidth=1.2,
                       label=f"{stage} (k={len(sub)}) r\u0305={mean_r:.3f}")

    ax.set_yticks([])
    ax.set_xlabel("ICC (IAT\u2013criterion correlation, r)")
    ax.set_title("Greenwald (2009) \u2014 184 samples, four-level DLN coding")
    ax.legend(loc="lower right", fontsize=7)
    ax.axvline(0, color="grey", linewidth=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Wrote: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    raw = pd.read_csv(DATA)
    print(f"Loaded {len(raw)} independent samples from {DATA.name}")

    df = prepare_data(raw)

    # ---- Distribution summary ----
    print(f"\n{'=' * 72}")
    print("FOUR-LEVEL DLN STAGE DISTRIBUTION")
    print(f"{'=' * 72}")
    counts = df["dln_stage"].value_counts()
    for stage in STAGE_ORDER:
        print(f"  {stage:<14s}: k = {counts.get(stage, 0)}")

    print(f"\n  Stage assignment by topic domain:")
    for topic in sorted(df["topic"].unique()):
        sub = df[df["topic"] == topic]
        stage_counts = sub["dln_stage"].value_counts().sort_index()
        print(f"    {topic:<20s}: {dict(stage_counts)}")

    # ---- 1. Hierarchical model comparison ----
    all_table_rows = []

    h_icc = run_hierarchical(df, "yi_icc", "vi_icc", "icc")
    print_hierarchy(h_icc, "ICC (k=184)")
    all_table_rows.extend(h_icc["table_rows"])

    df_ecc = df.dropna(subset=["yi_ecc", "vi_ecc"]).copy()
    h_ecc = run_hierarchical(df_ecc, "yi_ecc", "vi_ecc", "ecc")
    print_hierarchy(h_ecc, f"ECC (k={h_ecc['k']})")
    all_table_rows.extend(h_ecc["table_rows"])

    df_disc = df.dropna(subset=["yi_disc", "vi_disc"]).copy()
    h_disc = run_hierarchical(df_disc, "yi_disc", "vi_disc", "disc")
    print_hierarchy(h_disc, f"Discrepancy ICC-ECC (k={h_disc['k']})")
    all_table_rows.extend(h_disc["table_rows"])

    # ---- 2. Stage profiles ----
    profile_rows = stage_profiles(df)

    # ---- 3. iec mediation ----
    med_rows = iec_mediation(df)

    # ---- 4. n_crit x stage interaction ----
    ncrit_interaction(df)

    # ---- 5. Competing moderators ----
    competing_moderators(df)

    # ---- 6. Noise decomposition ----
    noise_decomposition(df)

    # ---- 7. Egger's test ----
    print(f"\n{'=' * 72}")
    print("PUBLICATION BIAS: Egger's test")
    print(f"{'=' * 72}")
    egger = egger_test(df["yi_icc"].to_numpy(), df["vi_icc"].to_numpy())
    print(f"  intercept = {egger.intercept:.3f}, t = {egger.t_stat:.3f}, "
          f"p = {egger.p_value:.4f}")

    # ---- Save outputs ----
    OUT_DIR_T.mkdir(parents=True, exist_ok=True)
    OUT_DIR_F.mkdir(parents=True, exist_ok=True)

    _save_rows(all_table_rows, OUT_DIR_T / "greenwald2009_topology_summary.csv")
    _save_rows(profile_rows, OUT_DIR_T / "greenwald2009_topology_profiles.csv")

    forest_plot(df, OUT_DIR_F / "greenwald2009_topology_forest.png")


if __name__ == "__main__":
    main()
