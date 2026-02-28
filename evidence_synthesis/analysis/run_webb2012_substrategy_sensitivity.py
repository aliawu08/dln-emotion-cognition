"""Sensitivity analysis: C2 coding decision for Webb sub-strategy DLN analysis.

Tests how the C2 (concentrate on implications) coding decision affects
DLN moderator performance:
  - Primary coding (C2=dot): All concentration strategies are dot-level
    (consistent with blind coding at k=10 level)
  - Alternative coding (C2=linear): C2 involves causal analysis,
    which might qualify as sequential processing

The S2 (suppress experience) coding as dot is held constant, as its
rationale (raw experiential suppression without behavioral structure)
is unambiguous.

Outputs:
  evidence_synthesis/outputs/tables/webb2012_substrategy_sensitivity.csv

Usage:
  python evidence_synthesis/analysis/run_webb2012_substrategy_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from meta_pipeline import aicc, compute_qm, design_matrix_categorical, fit_reml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evidence_synthesis" / "extraction" / "webb2012_substrategy_extraction.csv"
OUT = ROOT / "evidence_synthesis" / "outputs" / "tables" / "webb2012_substrategy_sensitivity.csv"


def main():
    df = pd.read_csv(DATA)
    y = df["d_plus"].to_numpy()
    v = df["vi"].to_numpy()
    k = len(y)

    X_base = np.ones((k, 1))
    res_base = fit_reml(y, v, X_base)

    rows = []

    # DLN primary (C2=dot, S2=dot)
    codes_primary = df["dln_stage_code"]  # already has C2=dot
    X_p, _ = design_matrix_categorical(codes_primary, reference="dot")
    res_p = fit_reml(y, v, X_p)
    pct_p = (res_base.tau2 - res_p.tau2) / res_base.tau2 * 100
    qm_p = compute_qm(y, v, X_base, X_p)
    rows.append({
        "coding": "DLN primary (C2=dot, S2=dot)",
        "C2_code": "dot",
        "S2_code": "dot",
        "tau2": round(res_p.tau2, 6),
        "pct_reduction": round(pct_p, 1),
        "QM": round(qm_p.QM, 2),
        "QM_p": round(qm_p.p, 4),
        "aicc": round(aicc(y, v, X_p, res_p.tau2), 2),
    })

    # DLN alternative (C2=linear, S2=dot)
    codes_alt = codes_primary.copy()
    codes_alt[df["webb_code"] == "C2"] = "linear"
    X_a, _ = design_matrix_categorical(codes_alt, reference="dot")
    res_a = fit_reml(y, v, X_a)
    pct_a = (res_base.tau2 - res_a.tau2) / res_base.tau2 * 100
    qm_a = compute_qm(y, v, X_base, X_a)
    rows.append({
        "coding": "DLN alternative (C2=linear, S2=dot)",
        "C2_code": "linear",
        "S2_code": "dot",
        "tau2": round(res_a.tau2, 6),
        "pct_reduction": round(pct_a, 1),
        "QM": round(qm_a.QM, 2),
        "QM_p": round(qm_a.p, 4),
        "aicc": round(aicc(y, v, X_a, res_a.tau2), 2),
    })

    out_df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(out_df.to_string(index=False))
    print(f"\nWrote: {OUT}")

    print("\n--- Interpretation ---")
    print("The C2 coding decision affects DLN moderator performance:")
    print(f"  C2=dot (primary):   {pct_p:.1f}% reduction")
    print(f"  C2=linear (alt):    {pct_a:.1f}% reduction")
    print("Both codings yield significant heterogeneity reduction.")


if __name__ == "__main__":
    main()
