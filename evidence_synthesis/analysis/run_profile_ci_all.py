"""Compute profile-likelihood CIs for tau-squared across all primary datasets.

For each of the four small-k datasets (Webb, interoception, Desmedt, Hoyt),
computes the REML point estimate and 95 % profile-likelihood CI for tau^2
under both the intercept-only baseline and the DLN-stage moderator model.

Outputs:
- evidence_synthesis/outputs/tables/profile_ci_all_datasets.csv

Usage:
  python evidence_synthesis/analysis/run_profile_ci_all.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta_pipeline import (
    fit_reml,
    design_matrix_stage,
    profile_likelihood_ci,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evidence_synthesis" / "outputs" / "tables" / "profile_ci_all_datasets.csv"


def _r_to_z(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def _load_webb():
    df = pd.read_csv(ROOT / "evidence_synthesis" / "extraction" / "webb2012_strategy_extraction.csv")
    return df["d_plus"].to_numpy(), df["vi"].to_numpy(), df["dln_stage_code"]


def _load_interoception():
    df = pd.read_csv(ROOT / "evidence_synthesis" / "extraction" / "interoception_measure_extraction.csv")
    df["z"] = _r_to_z(df["r_pooled"].to_numpy())
    df["vi_z"] = 1.0 / (df["N_total"] - 3.0 * df["k"])
    return df["z"].to_numpy(), df["vi_z"].to_numpy(), df["dln_stage_code"]


def _load_desmedt():
    df = pd.read_csv(ROOT / "evidence_synthesis" / "extraction" / "desmedt2022_criterion_extraction.csv")
    df["z_abs"] = _r_to_z(df["r_pooled"].abs().to_numpy())
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    # Only dot + linear present; return a sentinel so main() builds the
    # design matrix manually (same approach as run_desmedt2022_moderator.py).
    return df["z_abs"].to_numpy(), df["vi_z"].to_numpy(), df["dln_stage_code"]


def _desmedt_mod_matrix(stages):
    """Two-level design matrix for Desmedt (dot vs linear only)."""
    is_dot = (stages == "dot").astype(float).to_numpy()
    return np.column_stack([np.ones(len(stages)), is_dot])


def _load_hoyt():
    df = pd.read_csv(ROOT / "evidence_synthesis" / "extraction" / "hoyt2024_domain_extraction.csv")
    df["z"] = _r_to_z(df["r_pooled"].to_numpy())
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    return df["z"].to_numpy(), df["vi_z"].to_numpy(), df["dln_stage_code"]


DATASETS = {
    "Webb": _load_webb,
    "Interoception": _load_interoception,
    "Desmedt": _load_desmedt,
    "Hoyt": _load_hoyt,
}


def main():
    rows = []
    for name, loader in DATASETS.items():
        y, v, stages = loader()
        k = len(y)

        # Baseline
        X_base = np.ones((k, 1))
        pl_base = profile_likelihood_ci(y, v, X_base)

        # Moderator
        if name == "Desmedt":
            X_mod = _desmedt_mod_matrix(stages)
        else:
            X_mod, _ = design_matrix_stage(stages, reference="dot")
        pl_mod = profile_likelihood_ci(y, v, X_mod)

        rows.append({
            "dataset": name,
            "model": "baseline",
            "k": k,
            "tau2_hat": round(pl_base.tau2_hat, 6),
            "tau2_ci_lo": round(pl_base.ci_lo, 6),
            "tau2_ci_hi": round(pl_base.ci_hi, 6),
        })
        rows.append({
            "dataset": name,
            "model": "DLN_moderator",
            "k": k,
            "tau2_hat": round(pl_mod.tau2_hat, 6),
            "tau2_ci_lo": round(pl_mod.ci_lo, 6),
            "tau2_ci_hi": round(pl_mod.ci_hi, 6),
        })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote: {OUT}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
