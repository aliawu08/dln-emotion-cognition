"""Generate Figure 1 (Contradiction Map) for Paper 2.

This script produces a diagram that visualizes the core claim of the paper:
contradictory emotion–cognition findings persist because studies aggregate across
different cognitive architectures; DLN stage acts as a hidden moderator that
predicts when emotion functions primarily as noise vs signal.

Outputs:
  figures/export/contradiction_map.png
  figures/export/contradiction_map.pdf
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "export"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _wrap(s: str, width_chars: int) -> str:
    # Normalize whitespace and wrap to reduce overflow/overlap.
    s = " ".join(s.split())
    return textwrap.fill(s, width=width_chars, break_long_words=False, break_on_hyphens=False)

def main():
    fig_path_png = OUT_DIR / "contradiction_map.png"
    fig_path_pdf = OUT_DIR / "contradiction_map.pdf"

    # Slightly larger canvas to prevent text collisions.
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_axis_off()

    stages = [
        ("Network stage\n(Integrative fusion)", 0.66),
        ("Linear stage\n(Suppressive compartmentalization)", 0.33),
        ("Dot stage\n(Reactive separation)", 0.0),
    ]
    band_height = 0.30

    for label, y0 in stages:
        rect = Rectangle((0.02, y0 + 0.02), 0.96, band_height, fill=False, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(0.03, y0 + band_height - 0.02, label, va="top", ha="left", fontsize=11)

    # Boxes are sized and wrapped to avoid overlap.
    box_h = 0.18
    boxes = [
        ("Somatic marker (Damasio/Bechara). Emotion as signal. "
         "Real-world decision failures when affect is unavailable.",
         (0.58, 0.78), (0.38, box_h)),
        ("Affective neuroscience (Pessoa; constructionist). "
         "Neural integration and bidirectional coupling.",
         (0.12, 0.78), (0.40, box_h)),
        ("Dual-process / bias (Kahneman/Tversky; affect heuristic). "
         "Emotion as noise/bias via context-inappropriate signals.",
         (0.58, 0.45), (0.38, box_h)),
        ("Emotion regulation (Gross). "
         "Suppression vs reappraisal: strategy costs and benefits.",
         (0.12, 0.45), (0.40, box_h)),
        ("Reactive affect / impulsivity. "
         "Stimulus-driven responding and post-hoc rationalization.",
         (0.12, 0.12), (0.40, box_h)),
        ("Implicit–explicit gaps (IAT vs self-report). "
         "Influence without awareness.",
         (0.58, 0.12), (0.38, box_h)),
    ]

    for text, (x, y), (w, h) in boxes:
        rect = Rectangle((x, y), w, h, fill=False, linewidth=1.0)
        ax.add_patch(rect)
        width_chars = max(22, int(w * 78))  # heuristic mapping from box width to chars
        ax.text(
            x + 0.012, y + h - 0.012,
            _wrap(text, width_chars=width_chars),
            va="top", ha="left",
            fontsize=8.6,
            clip_on=True,
        )

    # Moderator arrows
    ax.add_patch(FancyArrowPatch((0.77, 0.64), (0.77, 0.78), arrowstyle="->", mutation_scale=12, linewidth=1.0))
    ax.add_patch(FancyArrowPatch((0.77, 0.30), (0.77, 0.45), arrowstyle="->", mutation_scale=12, linewidth=1.0))
    ax.text(0.79, 0.71, "DLN stage\nmoderates", fontsize=9, va="center", ha="left")

    ax.text(
        0.5, 0.02,
        "Key claim: contradictions persist because studies aggregate across architectures.\n"
        "DLN stage predicts when emotion functions primarily as noise vs signal.",
        ha="center", va="bottom", fontsize=10
    )

    fig.tight_layout()
    fig.savefig(fig_path_png, dpi=300)
    fig.savefig(fig_path_pdf)
    plt.close(fig)
    print(f"Wrote: {fig_path_png}\nWrote: {fig_path_pdf}")

if __name__ == "__main__":
    main()
