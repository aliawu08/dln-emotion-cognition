"""Run a worked example of the DLN-stage moderator meta-analysis.

This script uses `effect_sizes_synthetic_example.csv` (synthetic demo data) to:
1) Fit baseline random-effects models per paradigm family.
2) Fit random-effects meta-regressions with DLN stage as moderator.
3) Write summary tables + a simple stage-by-paradigm plot.

Outputs are written to:
- evidence_synthesis/outputs/tables/meta_summary_example.csv
- evidence_synthesis/outputs/figures/stage_effects_example.png

Usage:
  python evidence_synthesis/analysis/run_meta_example.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meta_pipeline import summarize_meta, summarize_meta_with_stage, results_to_frame, fisher_z_to_r

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "effect_sizes_synthetic_example.csv"
OUT_TABLE = ROOT / "evidence_synthesis" / "outputs" / "tables" / "meta_summary_example.csv"
OUT_FIG = ROOT / "evidence_synthesis" / "outputs" / "figures" / "stage_effects_example.png"

def main():
    df = pd.read_csv(DATA)

    base = summarize_meta(df)
    mod = summarize_meta_with_stage(df)

    summary = results_to_frame(base, mod)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)

    # Simple visualization: stage means (raw, by paradigm) in r scale (approx)
    # Note: For a publication-quality plot, compute pooled estimates by stage within each paradigm.
    stage_order = ["dot", "linear", "network"]
    paradigms = sorted(df["paradigm_family"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(paradigms))
    width = 0.25

    for i, stage in enumerate(stage_order):
        ys = []
        for p in paradigms:
            sub = df[(df["paradigm_family"] == p) & (df["dln_stage_code"] == stage)]
            ys.append(sub["r_approx"].mean())
        ax.bar([xx + (i - 1) * width for xx in x], ys, width=width, label=stage)

    ax.set_xticks(list(x))
    ax.set_xticklabels(paradigms, rotation=15, ha="right")
    ax.set_ylabel("Mean r (synthetic demo)")
    ax.set_title("Illustrative stage differences (synthetic data)")
    ax.legend()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    print(f"Wrote: {OUT_TABLE}\nWrote: {OUT_FIG}")

if __name__ == "__main__":
    main()
