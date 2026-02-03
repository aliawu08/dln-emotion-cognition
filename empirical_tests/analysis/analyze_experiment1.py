"""Analyze Experiment 1 (stage × affective load interaction).

This script is provided as a runnable template for the empirical layer.

Inputs:
- empirical_tests/data/synthetic_experiment1_demo.csv  (synthetic demo data)

Outputs:
- empirical_tests/outputs/experiment1_summary.txt
- empirical_tests/outputs/experiment1_plot.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "empirical_tests" / "data" / "synthetic_experiment1_demo.csv"
OUT_TXT = ROOT / "empirical_tests" / "outputs" / "experiment1_summary.txt"
OUT_FIG = ROOT / "empirical_tests" / "outputs" / "experiment1_plot.png"

def main():
    df = pd.read_csv(DATA)

    # Encode categoricals
    df["dln_stage"] = pd.Categorical(df["dln_stage"], categories=["dot","linear","network"], ordered=False)
    df["affective_load"] = pd.Categorical(df["affective_load"], categories=["low","high"], ordered=False)

    # OLS with interaction (for bounded outcomes you may prefer beta regression; OLS is a transparent starter)
    model = smf.ols("decision_quality ~ affective_load * dln_stage", data=df).fit()

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    # Plot cell means
    means = df.groupby(["dln_stage","affective_load"])["decision_quality"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7,4))
    for stage in ["dot","linear","network"]:
        sub = means[means["dln_stage"] == stage]
        ax.plot(sub["affective_load"], sub["decision_quality"], marker="o", label=stage)

    ax.set_xlabel("Affective load")
    ax.set_ylabel("Decision quality (mean)")
    ax.set_title("Stage × load pattern (demo)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    print(f"Wrote: {OUT_TXT}\nWrote: {OUT_FIG}")

if __name__ == "__main__":
    main()
