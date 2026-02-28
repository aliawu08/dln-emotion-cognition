"""Webb et al. (2012) outcome-type decomposition analysis.

DLN predicts channel-specific regulation patterns: linear-stage strategies
should suppress behavioural expression while leaving experiential processing
intact, whereas network-stage strategies should regulate the experiential
channel directly.  This script tests those predictions by running separate
DLN-stage moderator analyses for experiential, behavioural, and physiological
outcomes, and by analysing the experience–behaviour gap for comparisons that
report both outcome types.

Outputs:
  - evidence_synthesis/outputs/tables/webb2012_outcome_type_summary.csv
  - evidence_synthesis/outputs/tables/webb2012_outcome_gap_results.csv

Usage:
  python evidence_synthesis/analysis/run_webb2012_outcome_type.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist

from meta_pipeline import (
    compute_qm,
    design_matrix_categorical,
    design_matrix_stage,
    fit_reml,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "webb2012_comparison_extraction.csv"
OUT_DIR_T = ROOT / "evidence_synthesis" / "outputs" / "tables"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _reml_gap(
    y: np.ndarray, v: np.ndarray, X: np.ndarray
) -> dict:
    """Direct REML fit for the gap analysis.

    Uses explicit scipy minimisation to avoid convergence edge-cases
    in the main fit_reml when the number of reference-group observations
    is small (dot k=4).
    """
    k = len(y)
    p = X.shape[1]

    def negll(tau2):
        V = v + tau2
        if np.any(V <= 0):
            return np.inf
        W = np.diag(1.0 / V)
        XtWX = X.T @ W @ X
        try:
            sign, logdet = np.linalg.slogdet(XtWX)
            if sign <= 0:
                return np.inf
            beta = np.linalg.solve(XtWX, X.T @ W @ y)
        except np.linalg.LinAlgError:
            return np.inf
        e = y - X @ beta
        return 0.5 * (np.sum(np.log(V)) + logdet + float(e.T @ W @ e))

    tau2_upper = max(10.0, float(np.max(v)) * 100.0)
    res = minimize_scalar(negll, bounds=(0.0, tau2_upper), method="bounded",
                          options={"xatol": 1e-8})
    tau2 = max(float(res.x), 0.0)

    V = v + tau2
    W = np.diag(1.0 / V)
    XtWX = X.T @ W @ X
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ X.T @ W @ y

    # Knapp–Hartung adjustment
    e = y - X @ beta
    qe = max(float(e.T @ W @ e) / max(k - p, 1), 1.0)
    vcov = qe * XtWX_inv
    se = np.sqrt(np.diag(vcov))

    df_kh = max(k - p, 1)
    t_crit = float(t_dist.ppf(0.975, df_kh))
    ci95 = np.vstack([beta - t_crit * se, beta + t_crit * se]).T

    return {"tau2": tau2, "beta": beta, "se": se, "ci95": ci95, "k": k, "p": p}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    df = pd.read_csv(DATA)
    k_total = len(df)
    print(f"Webb outcome-type decomposition: k={k_total}")

    # Use the composite vi from the extraction CSV for all analyses.
    # Some rows lack n_ctrl (73 / 306 are within-subjects designs reporting
    # only n_total), so recomputing vi from n_exp / n_ctrl would introduce
    # NaNs.  The composite vi is pre-computed for every comparison and
    # provides a conservative (if anything slightly inflated) variance
    # estimate when used for a single outcome channel.
    vi_all = df["vi"].to_numpy()

    # ==================================================================
    # Analysis 1: Separate DLN moderator regressions per outcome type
    # ==================================================================
    outcome_cols = {
        "experiential": "d_experiential",
        "behavioral": "d_behavioral",
        "physiological": "d_physiological",
    }

    summary_rows: list[dict] = []

    for otype, col in outcome_cols.items():
        mask = df[col].notna()
        k_ot = int(mask.sum())

        sub = df.loc[mask].copy()
        n_stages = sub["dln_stage"].nunique()

        if k_ot < 10 or n_stages < 2:
            print(f"\n--- {otype}: k={k_ot}, stages={n_stages} "
                  f"(too few; skipping regression) ---")
            continue

        y = sub[col].to_numpy()
        v = vi_all[mask.to_numpy()]

        # Baseline
        X_base = np.ones((k_ot, 1))
        res_base = fit_reml(y, v, X_base)

        # DLN stage moderator (use dot as reference when all three stages
        # are present; fall back to design_matrix_categorical otherwise)
        stages_present = sorted(sub["dln_stage"].unique())
        if set(stages_present) == {"dot", "linear", "network"}:
            X_mod, names = design_matrix_stage(sub["dln_stage"], reference="dot")
        else:
            ref = stages_present[0]
            X_mod, names = design_matrix_categorical(sub["dln_stage"], reference=ref)
        res_mod = fit_reml(y, v, X_mod)
        qm = compute_qm(y, v, X_base, X_mod)

        delta_tau2 = res_base.tau2 - res_mod.tau2
        pct_red = (delta_tau2 / res_base.tau2 * 100) if res_base.tau2 > 0 else 0.0

        # Baseline row
        summary_rows.append({
            "outcome_type": otype, "model": "baseline",
            "k": res_base.k,
            "tau2": round(res_base.tau2, 6), "I2": round(res_base.I2, 3),
            "parameter": "mu",
            "estimate": round(res_base.beta[0], 4),
            "se": round(res_base.se[0], 4),
            "ci_lo": round(res_base.ci95[0, 0], 4),
            "ci_hi": round(res_base.ci95[0, 1], 4),
        })
        # Moderator rows
        for i, name in enumerate(names):
            summary_rows.append({
                "outcome_type": otype, "model": "DLN-stage",
                "k": res_mod.k,
                "tau2": round(res_mod.tau2, 6), "I2": round(res_mod.I2, 3),
                "parameter": name,
                "estimate": round(res_mod.beta[i], 4),
                "se": round(res_mod.se[i], 4),
                "ci_lo": round(res_mod.ci95[i, 0], 4),
                "ci_hi": round(res_mod.ci95[i, 1], 4),
            })
        # Reduction row
        summary_rows.append({
            "outcome_type": otype, "model": "reduction",
            "k": res_mod.k,
            "tau2": round(delta_tau2, 6), "I2": round(pct_red, 1),
            "parameter": "delta_tau2",
            "estimate": round(delta_tau2, 6),
            "se": 0, "ci_lo": 0, "ci_hi": 0,
        })
        # QM row
        summary_rows.append({
            "outcome_type": otype, "model": "QM_test",
            "k": res_mod.k,
            "tau2": 0, "I2": 0,
            "parameter": f"QM({qm.df})",
            "estimate": round(qm.QM, 4),
            "se": 0, "ci_lo": 0,
            "ci_hi": round(qm.p, 6),
        })

        # Console output
        print(f"\n{'='*60}")
        print(f"OUTCOME TYPE: {otype} (k={k_ot})")
        print(f"{'='*60}")
        print(f"  Baseline: mu={res_base.beta[0]:.4f}, tau2={res_base.tau2:.4f}")
        print(f"  DLN mod:  tau2={res_mod.tau2:.4f}, reduction={pct_red:.1f}%")
        print(f"  QM({qm.df}) = {qm.QM:.2f}, p = {qm.p:.6f}")
        for i, name in enumerate(names):
            print(f"    {name}: b={res_mod.beta[i]:.4f} "
                  f"[{res_mod.ci95[i,0]:.4f}, {res_mod.ci95[i,1]:.4f}]")

        # Stage means (implied from dummy coding)
        print(f"  Stage means:")
        for stage in ["dot", "linear", "network"]:
            sub_stage = sub[sub["dln_stage"] == stage]
            if len(sub_stage) > 0:
                raw_mean = sub_stage[col].mean()
                print(f"    {stage}: raw mean={raw_mean:.3f} (k={len(sub_stage)})")

    # Save outcome-type summary
    OUT_DIR_T.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR_T / "webb2012_outcome_type_summary.csv", index=False)

    # ==================================================================
    # Analysis 2: Experience–behaviour gap
    # ==================================================================
    print(f"\n{'='*60}")
    print("EXPERIENCE-BEHAVIOUR GAP ANALYSIS")
    print(f"{'='*60}")

    both = df[df["d_behavioral"].notna() & df["d_experiential"].notna()].copy()
    k_gap = len(both)
    print(f"  Comparisons with both outcomes: k={k_gap}")

    both["gap"] = both["d_experiential"] - both["d_behavioral"]

    # Sampling variance for the gap depends on the within-comparison
    # correlation between experiential and behavioural outcomes (r_w).
    # We use the composite vi as the base SE for each channel (conservative)
    # and report a primary analysis at r_w = 0.5 (conventional default)
    # with sensitivity analyses across r_w = {0.0, 0.3, 0.7}.
    se_each = np.sqrt(both["vi"].to_numpy())

    gap_rows: list[dict] = []

    for r_w in [0.0, 0.3, 0.5, 0.7]:
        # For a difference of two correlated outcomes:
        # Var(gap) = Var(exp) + Var(beh) - 2*r_w*SE(exp)*SE(beh)
        # Using composite SE for both channels (conservative):
        se_gap = se_each * np.sqrt(2 * (1 - r_w))
        vi_gap = se_gap**2

        y_gap = both["gap"].to_numpy()
        X_mod, names = design_matrix_stage(both["dln_stage"], reference="dot")

        res = _reml_gap(y_gap, vi_gap, X_mod)

        # Implied stage means
        mean_dot = res["beta"][0]
        mean_lin = res["beta"][0] + res["beta"][1]
        mean_net = res["beta"][0] + res["beta"][2]

        label = "primary" if r_w == 0.5 else "sensitivity"

        for i, name in enumerate(names):
            gap_rows.append({
                "analysis": label, "r_within": r_w,
                "k": k_gap, "tau2": round(res["tau2"], 4),
                "parameter": name,
                "estimate": round(res["beta"][i], 4),
                "se": round(res["se"][i], 4),
                "ci_lo": round(res["ci95"][i, 0], 4),
                "ci_hi": round(res["ci95"][i, 1], 4),
            })
        # Stage means
        for stage, mean_val in [("dot", mean_dot), ("linear", mean_lin),
                                ("network", mean_net)]:
            gap_rows.append({
                "analysis": label, "r_within": r_w,
                "k": k_gap, "tau2": round(res["tau2"], 4),
                "parameter": f"mean[{stage}]",
                "estimate": round(mean_val, 4),
                "se": 0, "ci_lo": 0, "ci_hi": 0,
            })

        if r_w == 0.5:
            print(f"\n  Primary analysis (r_within = 0.5):")
            print(f"    tau2 = {res['tau2']:.4f}")
            for i, name in enumerate(names):
                print(f"    {name}: b={res['beta'][i]:+.4f} "
                      f"[{res['ci95'][i,0]:+.4f}, {res['ci95'][i,1]:+.4f}]")
            print(f"    Stage means (exp − beh gap):")
            print(f"      dot:     {mean_dot:+.3f}  (positive = experience > behaviour)")
            print(f"      linear:  {mean_lin:+.3f}")
            print(f"      network: {mean_net:+.3f}")
        else:
            print(f"  Sensitivity (r_within = {r_w}): "
                  f"dot={mean_dot:+.3f}, lin={mean_lin:+.3f}, net={mean_net:+.3f}, "
                  f"tau2={res['tau2']:.4f}")

    pd.DataFrame(gap_rows).to_csv(
        OUT_DIR_T / "webb2012_outcome_gap_results.csv", index=False)

    # ==================================================================
    # Analysis 3: Within-suppression dissociation (S1 paired test)
    # ==================================================================
    print(f"\n{'='*60}")
    print("WITHIN-SUPPRESSION DISSOCIATION (S1)")
    print(f"{'='*60}")

    s1 = df[(df["strategy_code"] == "S1") & df["d_behavioral"].notna()].copy()
    k_s1 = len(s1)
    gap_s1 = s1["d_experiential"] - s1["d_behavioral"]
    se_s1 = gap_s1.std() / np.sqrt(k_s1)
    t_s1 = gap_s1.mean() / se_s1
    p_s1 = 2 * float(t_dist.sf(abs(t_s1), k_s1 - 1))

    print(f"  Expressive suppression (S1): k={k_s1}")
    print(f"  d_experiential mean = {s1['d_experiential'].mean():.3f}")
    print(f"  d_behavioral mean   = {s1['d_behavioral'].mean():.3f}")
    print(f"  gap (exp − beh)     = {gap_s1.mean():+.3f}")
    print(f"  t({k_s1-1}) = {t_s1:.2f}, p = {p_s1:.6f}")

    # ==================================================================
    # Summary: DLN stage × outcome type matrix
    # ==================================================================
    print(f"\n{'='*60}")
    print("SUMMARY: DLN STAGE × OUTCOME TYPE (raw means)")
    print(f"{'='*60}")
    print(f"{'':20s} {'dot':>10s} {'linear':>10s} {'network':>10s}")
    for otype, col in outcome_cols.items():
        vals = {}
        for stage in ["dot", "linear", "network"]:
            sub = df[(df["dln_stage"] == stage) & df[col].notna()]
            if len(sub) > 0:
                vals[stage] = f"{sub[col].mean():.2f} ({len(sub)})"
            else:
                vals[stage] = "---"
        print(f"{otype:20s} {vals['dot']:>10s} {vals['linear']:>10s} "
              f"{vals['network']:>10s}")

    print(f"\nWrote outputs to:")
    print(f"  {OUT_DIR_T / 'webb2012_outcome_type_summary.csv'}")
    print(f"  {OUT_DIR_T / 'webb2012_outcome_gap_results.csv'}")


if __name__ == "__main__":
    main()
