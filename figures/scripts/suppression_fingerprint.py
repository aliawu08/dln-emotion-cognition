"""Suppression fingerprint heatmap and model comparison waterfall.

Generates two publication figures for the Greenwald four-level topology analysis:

  Figure A: Suppression fingerprint heatmap
    4 rows (Dot, Linear, Linear-Plus, Network) x 4 columns (ICC, ECC, iec, Gap)
    Each cell shows the mean value.  Colour-coded by magnitude.

  Figure B: Model comparison waterfall
    Horizontal bars showing incremental tau-squared reduction from each step
    in the hierarchical model comparison.

Outputs:
  - figures/export/suppression_fingerprint.pdf / .png
  - figures/export/model_waterfall.pdf / .png

Usage:
  python figures/scripts/suppression_fingerprint.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "evidence_synthesis" / "analysis"))
from run_greenwald2009_topology import DATA, STAGE_ORDER, prepare_data

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "export"


def fingerprint_heatmap(df, out_stem):
    """Generate the suppression fingerprint heatmap."""
    metrics = ["ICC", "ECC", "iec", "Gap"]
    data = np.full((len(STAGE_ORDER), len(metrics)), np.nan)
    labels = np.full((len(STAGE_ORDER), len(metrics)), "", dtype=object)

    for i, stage in enumerate(STAGE_ORDER):
        sub = df[df["dln_stage"] == stage]
        sub_ecc = sub.dropna(subset=["ecc"])
        sub_iec = sub.dropna(subset=["iec"])

        icc_val = sub["icc"].mean()
        data[i, 0] = icc_val
        labels[i, 0] = f"{icc_val:.3f}\n(k={len(sub)})"

        if len(sub_ecc) > 0:
            ecc_val = sub_ecc["ecc"].mean()
            data[i, 1] = ecc_val
            labels[i, 1] = f"{ecc_val:.3f}\n(k={len(sub_ecc)})"

            gap_val = (sub_ecc["icc"] - sub_ecc["ecc"]).mean()
            data[i, 3] = gap_val
            labels[i, 3] = f"{gap_val:+.3f}\n(k={len(sub_ecc)})"

        if len(sub_iec) > 0:
            iec_val = sub_iec["iec"].mean()
            data[i, 2] = iec_val
            labels[i, 2] = f"{iec_val:.3f}\n(k={len(sub_iec)})"

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Diverging norm centred at 0.3 for ICC/ECC/iec, special handling for gap
    masked_data = np.ma.masked_invalid(data)
    im = ax.imshow(masked_data, cmap="RdYlGn", aspect="auto",
                   vmin=-0.2, vmax=0.7)

    # Cell annotations
    for i in range(len(STAGE_ORDER)):
        for j in range(len(metrics)):
            if labels[i, j]:
                ax.text(j, i, labels[i, j], ha="center", va="center",
                        fontsize=8, fontweight="bold" if not np.isnan(data[i, j]) else "normal")

    stage_labels = ["Dot", "Linear", "Linear-Plus", "Network"]
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(stage_labels, fontsize=10)
    ax.set_title("Suppression fingerprint: validity profile by DLN stage", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean correlation (r)", fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Wrote: {out_stem}.pdf / .png")


def model_waterfall(out_stem):
    """Generate model comparison waterfall chart."""
    # Verified numbers from Phase 1
    steps = [
        ("Baseline", 0.016682, 0.0),
        ("+ 3-level DLN", 0.011744, 29.6),
        ("+ Linear-Plus (4-level)", 0.011190, 32.9),
        ("+ n_crit + n_iat", 0.008522, 48.9),
    ]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    labels = [s[0] for s in steps]
    total_reds = [s[2] for s in steps]

    # Incremental reductions
    increments = [0.0]
    for i in range(1, len(steps)):
        increments.append(total_reds[i] - total_reds[i - 1])

    colors = ["#95a5a6", "#3498db", "#9b59b6", "#2ecc71"]
    left = 0.0
    for i in range(len(steps)):
        width = increments[i]
        bar = ax.barh(0, width, left=left, color=colors[i], edgecolor="white",
                      height=0.5, label=f"{labels[i]} (+{width:.1f}%)")
        if width > 2:
            ax.text(left + width / 2, 0, f"+{width:.1f}%",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if width > 5 else "black")
        left += width

    ax.set_xlim(0, 60)
    ax.set_xlabel("Cumulative heterogeneity reduction (%)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("Greenwald (2009): incremental model building", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Wrote: {out_stem}.pdf / .png")


def main():
    raw = pd.read_csv(DATA)
    df = prepare_data(raw)

    fingerprint_heatmap(df, OUT_DIR / "suppression_fingerprint")
    model_waterfall(OUT_DIR / "model_waterfall")


if __name__ == "__main__":
    main()
