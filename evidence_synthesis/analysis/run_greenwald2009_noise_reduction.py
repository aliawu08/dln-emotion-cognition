"""Greenwald et al. (2009) — noise-reduction sensitivity analysis.

Tests whether DLN-stage heterogeneity reduction improves when we remove
sources of within-stage noise that DLN is not designed to explain:

Filter levels (cumulative):
  0. Full sample                                  (k=184)
  1. Drop mixed_unclear (only dot/linear/network)  (cleanly codeable)
  2. + Single/focused criterion (n_crit <= 2)      (specific behavior, not composite)
  3. + Adequate sample size (n >= 30)              (precise ICC estimates)
  4. + Single IAT formulation (n_iat == 1)         (clean measurement)

Rationale: DLN is a structural theory about representational topology.
It should explain ~100% of *true* heterogeneity. When it explains only ~30%,
the gap is noise from measurement composites, small samples, and mixed
operationalizations. Progressive filtering isolates the DLN signal.

Usage:
  python evidence_synthesis/analysis/run_greenwald2009_noise_reduction.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import fit_reml, design_matrix_stage

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "greenwald2009_study_extraction.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "greenwald2009_noise_reduction.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "greenwald2009_noise_reduction.png"

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
    """Add DLN coding and Fisher-z transforms."""
    df = df.copy()
    df["dln_stage"] = df["topic"].map(TOPIC_TO_DLN)
    df["yi_icc"] = np.arctanh(df["icc"].clip(-0.999, 0.999))
    df["vi_icc"] = 1.0 / (df["n"] - 3)
    return df


def run_moderator(df):
    """Run baseline + DLN moderator, return tau2 values and stage means."""
    y = df["yi_icc"].to_numpy()
    v = df["vi_icc"].to_numpy()
    k = len(df)

    # Baseline
    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    # Moderator
    X_mod, names_mod = design_matrix_stage(df["dln_stage"], reference="dot")
    res_mod = fit_reml(y, v, X_mod)

    delta_tau2 = res_base.tau2 - res_mod.tau2
    pct = (delta_tau2 / res_base.tau2 * 100) if res_base.tau2 > 0 else 0.0

    # Stage means (weighted)
    stage_means = {}
    for stage in ["dot", "linear", "network"]:
        sub = df[df["dln_stage"] == stage]
        if len(sub) > 0:
            mean_z = np.average(sub["yi_icc"], weights=1.0 / sub["vi_icc"])
            stage_means[stage] = {"k": len(sub), "mean_r": float(np.tanh(mean_z))}

    return {
        "k": k,
        "tau2_base": res_base.tau2,
        "I2_base": res_base.I2,
        "tau2_mod": res_mod.tau2,
        "I2_mod": res_mod.I2,
        "delta_tau2": delta_tau2,
        "pct_reduction": pct,
        "Q_base": res_base.Q,
        "Q_mod": res_mod.Q,
        "stage_means": stage_means,
        "beta": res_mod.beta,
        "se": res_mod.se,
        "ci95": res_mod.ci95,
        "names": names_mod,
    }


def main():
    raw = pd.read_csv(DATA)
    df = prepare_data(raw)

    print(f"Loaded {len(raw)} samples\n")

    # ================================================================
    # Define progressive filter levels
    # ================================================================
    filters = [
        {
            "label": "0. Full sample (k=184)",
            "desc": "All samples, all DLN stages incl. mixed",
            "include_mixed": True,
            "n_crit_max": None,
            "n_min": None,
            "n_iat_max": None,
        },
        {
            "label": "1. Drop mixed_unclear",
            "desc": "Only dot/linear/network (cleanly codeable domains)",
            "include_mixed": False,
            "n_crit_max": None,
            "n_min": None,
            "n_iat_max": None,
        },
        {
            "label": "2. + Focused criterion (n_crit<=2)",
            "desc": "Single/dual criterion behavior (not composites)",
            "include_mixed": False,
            "n_crit_max": 2,
            "n_min": None,
            "n_iat_max": None,
        },
        {
            "label": "3. + Adequate N (n>=30)",
            "desc": "Remove very small imprecise samples",
            "include_mixed": False,
            "n_crit_max": 2,
            "n_min": 30,
            "n_iat_max": None,
        },
        {
            "label": "4. + Single IAT (n_iat==1)",
            "desc": "Clean single-IAT measurement",
            "include_mixed": False,
            "n_crit_max": 2,
            "n_min": 30,
            "n_iat_max": 1,
        },
    ]

    results = []

    for filt in filters:
        sub = df.copy()

        # Apply filters
        if not filt["include_mixed"]:
            sub = sub[sub["dln_stage"] != "mixed_unclear"]

        if filt["n_crit_max"] is not None:
            sub = sub[sub["n_crit"] <= filt["n_crit_max"]]

        if filt["n_min"] is not None:
            sub = sub[sub["n"] >= filt["n_min"]]

        if filt["n_iat_max"] is not None:
            sub = sub[sub["n_iat"] <= filt["n_iat_max"]]

        # Need at least 2 stages with data
        stages_present = [s for s in sub["dln_stage"].unique() if s in ("dot", "linear", "network")]
        if len(stages_present) < 2:
            print(f"\n{filt['label']}: SKIPPED (only {stages_present} present, k={len(sub)})")
            continue

        # For moderator analysis, keep only clean 3-stage data
        sub_clean = sub[sub["dln_stage"].isin(["dot", "linear", "network"])]
        if len(sub_clean) < 6:
            print(f"\n{filt['label']}: SKIPPED (k={len(sub_clean)} too small)")
            continue

        res = run_moderator(sub_clean)

        print(f"\n{'='*70}")
        print(f"{filt['label']}")
        print(f"  {filt['desc']}")
        print(f"{'='*70}")
        print(f"  k = {res['k']}")
        for stage, info in res["stage_means"].items():
            print(f"    {stage:10s}: k={info['k']:3d}, weighted mean r={info['mean_r']:.3f}")
        print(f"  Baseline:  tau2={res['tau2_base']:.6f}, I2={res['I2_base']:.1%}, Q={res['Q_base']:.1f}")
        print(f"  DLN mod:   tau2={res['tau2_mod']:.6f}, I2={res['I2_mod']:.1%}, Q={res['Q_mod']:.1f}")
        print(f"  REDUCTION: {res['pct_reduction']:.1f}% of heterogeneity explained by DLN stage")
        print(f"  Coefficients (dot = reference):")
        for i, name in enumerate(res["names"]):
            print(f"    {name}: b={res['beta'][i]:.4f} [{res['ci95'][i,0]:.4f}, {res['ci95'][i,1]:.4f}]")

        results.append({
            "filter_level": filt["label"],
            "description": filt["desc"],
            "k": res["k"],
            "k_dot": res["stage_means"].get("dot", {}).get("k", 0),
            "k_linear": res["stage_means"].get("linear", {}).get("k", 0),
            "k_network": res["stage_means"].get("network", {}).get("k", 0),
            "r_dot": round(res["stage_means"].get("dot", {}).get("mean_r", 0), 3),
            "r_linear": round(res["stage_means"].get("linear", {}).get("mean_r", 0), 3),
            "r_network": round(res["stage_means"].get("network", {}).get("mean_r", 0), 3),
            "tau2_base": round(res["tau2_base"], 6),
            "tau2_mod": round(res["tau2_mod"], 6),
            "I2_base": round(res["I2_base"], 3),
            "I2_mod": round(res["I2_mod"], 3),
            "pct_reduction": round(res["pct_reduction"], 1),
        })

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("PROGRESSIVE NOISE REDUCTION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Filter':<40s} {'k':>4s} {'tau2_base':>10s} {'tau2_mod':>10s} {'%Reduced':>10s}")
    print("-" * 78)
    for r in results:
        print(f"{r['filter_level']:<40s} {r['k']:4d} {r['tau2_base']:10.6f} {r['tau2_mod']:10.6f} {r['pct_reduction']:9.1f}%")

    # Save table
    summary_df = pd.DataFrame(results)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote: {OUT_TABLE}")

    # ================================================================
    # Bar chart of progressive reduction
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: heterogeneity reduction by filter level
    labels = [r["filter_level"].split(". ", 1)[1] if ". " in r["filter_level"] else r["filter_level"]
              for r in results]
    pct_vals = [r["pct_reduction"] for r in results]
    k_vals = [r["k"] for r in results]

    colors = ["#95a5a6", "#3498db", "#2ecc71", "#f39c12", "#e74c3c"][:len(results)]
    bars = ax1.barh(range(len(results)), pct_vals, color=colors, alpha=0.85, height=0.6)

    for i, (bar, k, pct) in enumerate(zip(bars, k_vals, pct_vals)):
        ax1.text(bar.get_width() + 1, i, f"{pct:.1f}% (k={k})", va="center", fontsize=9)

    ax1.set_yticks(range(len(results)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("% Heterogeneity Explained by DLN Stage")
    ax1.set_title("Progressive Noise Reduction")
    ax1.set_xlim(0, max(pct_vals) * 1.3 if pct_vals else 100)
    ax1.invert_yaxis()

    # Right: stage mean ICCs at each filter level
    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}
    x_pos = range(len(results))

    for stage, color in stage_colors.items():
        r_vals = [r[f"r_{stage}"] for r in results]
        ax2.plot(x_pos, r_vals, "o-", color=color, label=stage, markersize=8, linewidth=2)

    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels([f"Filter {i}" for i in range(len(results))], fontsize=8)
    ax2.set_ylabel("Weighted Mean ICC (r)")
    ax2.set_title("Stage Mean ICCs Across Filter Levels")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)
    print(f"Wrote: {OUT_FIG}")


if __name__ == "__main__":
    main()
