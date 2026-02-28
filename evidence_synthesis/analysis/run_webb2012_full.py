"""Webb et al. (2012) comparison-level DLN moderator analysis (k=306).

Expands the original k=10 strategy-family analysis to the full k=306
comparison-level dataset extracted from Webb Table 2. Each comparison's
DLN stage is determined by its experimental strategy code.

This script:
  1. Fits baseline (intercept-only) and DLN-stage moderator meta-regressions.
  2. Fits sub-strategy moderator to assess within-stage heterogeneity.
  3. Compares DLN vs. Gross process model vs. cognitive effort as moderators.
  4. Runs Monte Carlo permutation tests (10,000 iterations).
  5. Runs leave-one-out cross-validation.

Outputs:
  - evidence_synthesis/outputs/tables/webb2012_full_results.csv
  - evidence_synthesis/outputs/tables/webb2012_full_competing.csv
  - evidence_synthesis/outputs/figures/webb2012_full_forest_by_stage.png
  - evidence_synthesis/outputs/figures/webb2012_full_permutation.png

Usage:
  python evidence_synthesis/analysis/run_webb2012_full.py
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
DATA = ROOT / "evidence_synthesis" / "extraction" / "webb2012_comparison_extraction.csv"
OUT_DIR_T = ROOT / "evidence_synthesis" / "outputs" / "tables"
OUT_DIR_F = ROOT / "evidence_synthesis" / "outputs" / "figures"


# -----------------------------------------------------------------------
# Cognitive effort coding
# -----------------------------------------------------------------------

COGNITIVE_EFFORT = {
    'C1': 'low', 'C2': 'low', 'C3': 'low',
    'D2': 'low', 'D4': 'low', 'S2': 'low',
    'D1': 'high', 'D3': 'high',
    'R1': 'high', 'R2': 'high', 'R3': 'high', 'R4': 'high',
    'S1': 'high', 'S3': 'high', 'S4': 'high',
}

# Antecedent vs response-focused (Gross binary split)
GROSS_BINARY = {
    'D1': 'antecedent', 'D2': 'antecedent', 'D3': 'antecedent', 'D4': 'antecedent',
    'C1': 'antecedent', 'C2': 'antecedent', 'C3': 'antecedent',
    'R1': 'antecedent', 'R2': 'antecedent', 'R3': 'antecedent', 'R4': 'antecedent',
    'S1': 'response', 'S2': 'response', 'S3': 'response', 'S4': 'response',
}


# -----------------------------------------------------------------------
# Permutation test (Monte Carlo for k=306)
# -----------------------------------------------------------------------

def monte_carlo_permutation(
    y: np.ndarray,
    v: np.ndarray,
    observed_tau2: float,
    n_groups: int,
    group_sizes: list[int],
    n_iter: int = 10_000,
    seed: int = 2026,
) -> tuple[np.ndarray, float]:
    """Monte Carlo permutation test preserving group sizes."""
    rng = np.random.default_rng(seed)
    k = len(y)
    perm_tau2 = []

    for _ in range(n_iter):
        # Shuffle assignments preserving group sizes
        assignment = np.zeros(k, dtype=int)
        idx = rng.permutation(k)
        start = 0
        for g, size in enumerate(group_sizes):
            assignment[idx[start:start + size]] = g
            start += size

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
    res_base=None,
) -> dict:
    """Fit moderator and return comparison metrics."""
    X_base = np.ones((len(y), 1))
    if res_base is None:
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

    return {
        "moderator": label,
        "n_levels": len(codes.unique()),
        "tau2_base": tau2_base,
        "tau2_mod": tau2_mod,
        "pct_reduction": pct_red,
        "aicc_base": aicc_base,
        "aicc_mod": aicc_mod,
        "QM": qm.QM,
        "QM_df": qm.df,
        "QM_p": qm.p,
        "beta": res_mod.beta,
        "se": res_mod.se,
        "ci95": res_mod.ci95,
        "names": names,
        "res_mod": res_mod,
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    df = pd.read_csv(DATA)
    y = df["d_composite"].to_numpy()
    v = df["vi"].to_numpy()
    k = len(df)

    print(f"Webb full comparison-level analysis: k={k}")
    print(f"DLN partition: dot={sum(df['dln_stage']=='dot')}, "
          f"linear={sum(df['dln_stage']=='linear')}, "
          f"network={sum(df['dln_stage']=='network')}")

    # ---- Baseline ----
    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)
    print(f"\nBaseline: mu={res_base.beta[0]:.4f}, tau2={res_base.tau2:.4f}, "
          f"I2={res_base.I2:.1f}%, Q={res_base.Q:.1f}")

    # ---- Analysis 3a: DLN stage moderator ----
    print(f"\n{'=' * 70}")
    print("ANALYSIS 3a: DLN stage moderator")
    print(f"{'=' * 70}")
    dln_res = fit_moderator(y, v, df["dln_stage"], "DLN stage", res_base)
    print(f"  tau2={dln_res['tau2_mod']:.4f}  "
          f"reduction={dln_res['pct_reduction']:.1f}%  "
          f"QM({dln_res['QM_df']})={dln_res['QM']:.2f} p={dln_res['QM_p']:.6f}  "
          f"AICc={dln_res['aicc_mod']:.2f}")
    print("\n  Stage means:")
    for i, name in enumerate(dln_res["names"]):
        print(f"    {name}: b={dln_res['beta'][i]:.4f} "
              f"[{dln_res['ci95'][i,0]:.4f}, {dln_res['ci95'][i,1]:.4f}]")

    # ---- Analysis 3b: Sub-strategy moderator ----
    print(f"\n{'=' * 70}")
    print("ANALYSIS 3b: Sub-strategy moderator")
    print(f"{'=' * 70}")
    substrat_res = fit_moderator(y, v, df["strategy_code"], "Sub-strategy", res_base)
    print(f"  tau2={substrat_res['tau2_mod']:.4f}  "
          f"reduction={substrat_res['pct_reduction']:.1f}%  "
          f"QM({substrat_res['QM_df']})={substrat_res['QM']:.2f} p={substrat_res['QM_p']:.6f}")

    incremental = dln_res['pct_reduction'] - 0  # DLN beyond baseline
    substrat_beyond_dln = substrat_res['pct_reduction'] - dln_res['pct_reduction']
    print(f"\n  Hierarchical decomposition:")
    print(f"    Baseline → DLN stage:    {dln_res['pct_reduction']:.1f}% reduction")
    print(f"    DLN stage → Sub-strategy: {substrat_beyond_dln:.1f}% additional")
    print(f"    Total (sub-strategy):     {substrat_res['pct_reduction']:.1f}% reduction")

    # ---- Competing moderators ----
    print(f"\n{'=' * 70}")
    print("ANALYSIS 3c: Competing moderators")
    print(f"{'=' * 70}")

    # Add coded columns
    df["cognitive_effort"] = df["strategy_code"].map(COGNITIVE_EFFORT)
    df["gross_binary"] = df["strategy_code"].map(GROSS_BINARY)

    moderators = {
        "DLN stage": df["dln_stage"],
        "Gross process model": df["gross_family"],
        "Cognitive effort": df["cognitive_effort"],
        "Gross binary (ant/resp)": df["gross_binary"],
        "Sub-strategy": df["strategy_code"],
    }

    all_results = []
    for label, codes in moderators.items():
        res = fit_moderator(y, v, codes, label, res_base)
        all_results.append(res)
        flag = " ***" if label == "DLN stage" else ""
        print(f"  {label:30s}  levels={res['n_levels']:>2d}  "
              f"tau2={res['tau2_mod']:.4f}  "
              f"red={res['pct_reduction']:+6.1f}%  "
              f"QM({res['QM_df']})={res['QM']:.2f} p={res['QM_p']:.6f}  "
              f"AICc={res['aicc_mod']:.2f}{flag}")

    # ---- Permutation test for DLN ----
    print(f"\n{'=' * 70}")
    print("PERMUTATION TEST (10,000 iterations)")
    print(f"{'=' * 70}")
    dln_sizes = [
        sum(df['dln_stage'] == 'dot'),
        sum(df['dln_stage'] == 'linear'),
        sum(df['dln_stage'] == 'network'),
    ]
    perm_tau2, perm_p = monte_carlo_permutation(
        y, v, dln_res['tau2_mod'], 3, dln_sizes,
    )
    print(f"  Observed tau2 = {dln_res['tau2_mod']:.4f}")
    print(f"  Permutation p = {perm_p:.4f}")
    print(f"  Permutation tau2: mean={perm_tau2.mean():.4f}, "
          f"5th={np.percentile(perm_tau2, 5):.4f}, "
          f"95th={np.percentile(perm_tau2, 95):.4f}")

    # ---- Leave-one-out cross-validation ----
    print(f"\n{'=' * 70}")
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print(f"{'=' * 70}")
    loo_errors = []
    for i in range(k):
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        y_train, v_train = y[mask], v[mask]
        codes_train = df["dln_stage"].iloc[mask]

        ref = sorted(codes_train.unique())[0]
        X_train, _ = design_matrix_categorical(codes_train, reference=ref)
        res_train = fit_reml(y_train, v_train, X_train)

        # Predict: which stage is observation i?
        stage_i = df["dln_stage"].iloc[i]
        stage_names = sorted(codes_train.unique())
        stage_idx = stage_names.index(stage_i)
        if stage_idx == 0:
            pred = res_train.beta[0]  # reference group mean
        else:
            pred = res_train.beta[0] + res_train.beta[stage_idx]
        loo_errors.append(y[i] - pred)

    loo_errors = np.array(loo_errors)
    ss_res = np.sum(loo_errors**2)
    ss_tot = np.sum((y - y.mean())**2)
    loo_r2 = 1 - ss_res / ss_tot
    loo_rmse = np.sqrt(np.mean(loo_errors**2))
    print(f"  LOO R² = {loo_r2:.4f}")
    print(f"  LOO RMSE = {loo_rmse:.4f}")
    print(f"  (R² > 0 confirms predictive value beyond the grand mean)")

    # ---- Save results tables ----
    OUT_DIR_T.mkdir(parents=True, exist_ok=True)
    OUT_DIR_F.mkdir(parents=True, exist_ok=True)

    # Detailed results table
    detail_rows = [
        {"model": "baseline", "k": k, "tau2": round(res_base.tau2, 6),
         "I2": round(res_base.I2, 1), "Q": round(res_base.Q, 2),
         "pct_reduction": 0, "QM": 0, "QM_p": 1.0,
         "aicc": round(aicc(y, v, X_base, res_base.tau2), 2),
         "perm_p": None, "loo_r2": None},
        {"model": "DLN stage", "k": k,
         "tau2": round(dln_res['tau2_mod'], 6),
         "I2": 0, "Q": 0,
         "pct_reduction": round(dln_res['pct_reduction'], 1),
         "QM": round(dln_res['QM'], 2),
         "QM_p": round(dln_res['QM_p'], 6),
         "aicc": round(dln_res['aicc_mod'], 2),
         "perm_p": round(perm_p, 4),
         "loo_r2": round(loo_r2, 4)},
    ]
    pd.DataFrame(detail_rows).to_csv(
        OUT_DIR_T / "webb2012_full_results.csv", index=False)

    # Competing moderators table
    comp_rows = []
    for res in all_results:
        comp_rows.append({
            "moderator": res["moderator"],
            "n_levels": res["n_levels"],
            "tau2_base": round(res["tau2_base"], 6),
            "tau2_mod": round(res["tau2_mod"], 6),
            "pct_reduction": round(res["pct_reduction"], 1),
            "QM": round(res["QM"], 2),
            "QM_df": res["QM_df"],
            "QM_p": round(res["QM_p"], 6),
            "aicc": round(res["aicc_mod"], 2),
        })
    pd.DataFrame(comp_rows).to_csv(
        OUT_DIR_T / "webb2012_full_competing.csv", index=False)

    # ---- Forest plot by DLN stage ----
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}
    stage_order = {"dot": 0, "linear": 1, "network": 2}

    fig, ax = plt.subplots(figsize=(10, 6))
    for stage, color in stage_colors.items():
        mask = df['dln_stage'] == stage
        subset = df[mask]
        d_vals = subset['d_composite'].values
        jitter = np.random.default_rng(42).normal(0, 0.05, len(d_vals))
        y_pos = stage_order[stage] + jitter
        sizes = 20 / subset['vi'].values  # Size proportional to precision
        sizes = np.clip(sizes, 5, 200)
        ax.scatter(d_vals, y_pos, c=color, s=sizes, alpha=0.4, label=f"{stage} (k={len(d_vals)})")

    # Stage means
    for i, name in enumerate(dln_res['names']):
        if 'dot' in name or i == 0:
            stage = 'dot'
            idx = 0
        elif 'linear' in name:
            stage = 'linear'
            idx = 1
        elif 'network' in name:
            stage = 'network'
            idx = 2
        else:
            continue

    # Compute stage means directly
    for stage, idx in stage_order.items():
        mask = df['dln_stage'] == stage
        stage_mean = df.loc[mask, 'd_composite'].mean()
        ax.axvline(stage_mean, ymin=(idx)/3, ymax=(idx+1)/3,
                   color=stage_colors[stage], linewidth=2, linestyle='--', alpha=0.7)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['dot', 'linear', 'network'])
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Effect size (d composite)")
    ax.set_title(f"Webb (2012) k={k} comparisons by DLN stage")
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR_F / "webb2012_full_forest_by_stage.png", dpi=200)
    plt.close(fig)

    # ---- Permutation distribution plot ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(perm_tau2, bins=60, color="#95a5a6", edgecolor="white", alpha=0.85)
    ax.axvline(dln_res['tau2_mod'], color="#e74c3c", linewidth=2,
               linestyle="--", label=f"Observed τ²={dln_res['tau2_mod']:.4f}")
    ax.set_title(f"DLN stage (k={k}): permutation test (10,000 iter), p={perm_p:.4f}")
    ax.set_xlabel("Residual τ²")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR_F / "webb2012_full_permutation.png", dpi=200)
    plt.close(fig)

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: Webb k={k} full comparison-level analysis")
    print(f"{'=' * 70}")
    print(f"  Baseline: tau2={res_base.tau2:.4f}, I2={res_base.I2:.1f}%")
    print(f"  DLN stage: tau2 reduction={dln_res['pct_reduction']:.1f}%, "
          f"QM({dln_res['QM_df']})={dln_res['QM']:.2f}, p={dln_res['QM_p']:.6f}")
    print(f"  Permutation p={perm_p:.4f}")
    print(f"  LOO R²={loo_r2:.4f}")
    print(f"\n  Stage means (d_composite):")
    for stage in ['dot', 'linear', 'network']:
        mask = df['dln_stage'] == stage
        mean_d = df.loc[mask, 'd_composite'].mean()
        print(f"    {stage}: mean={mean_d:.3f} (k={mask.sum()})")

    best_red = max(all_results, key=lambda r: r["pct_reduction"])
    best_aicc = min(all_results, key=lambda r: r["aicc_mod"])
    print(f"\n  Best tau2 reduction: {best_red['moderator']} ({best_red['pct_reduction']:.1f}%)")
    print(f"  Best AICc: {best_aicc['moderator']} ({best_aicc['aicc_mod']:.2f})")

    print(f"\nWrote outputs to:")
    print(f"  {OUT_DIR_T / 'webb2012_full_results.csv'}")
    print(f"  {OUT_DIR_T / 'webb2012_full_competing.csv'}")
    print(f"  {OUT_DIR_F / 'webb2012_full_forest_by_stage.png'}")
    print(f"  {OUT_DIR_F / 'webb2012_full_permutation.png'}")


if __name__ == "__main__":
    main()
