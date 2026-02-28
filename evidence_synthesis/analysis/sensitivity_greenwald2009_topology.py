"""Greenwald (2009) coding sensitivity for the Linear / Linear-Plus boundary.

Identifies the most debatable sample assignments at the boundary between
Linear and Linear-Plus, toggles each assignment, and reruns the four-level
REML meta-regression under all 2^N alternative codings.

Reports:
  - Range of ICC tau-squared reduction (min, median, max)
  - Whether Linear-Plus shows lowest iec in each scenario
  - Whether the suppression fingerprint (positive ICC-ECC gap) holds

Outputs:
  - evidence_synthesis/outputs/tables/greenwald2009_topology_sensitivity.csv

Usage:
  python evidence_synthesis/analysis/sensitivity_greenwald2009_topology.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_pipeline import fit_reml, design_matrix_categorical
from run_greenwald2009_topology import (
    DATA, DOT_SAMPLES, NETWORK_SAMPLES, LINEAR_PLUS_TOPICS,
    STAGE_ORDER, prepare_data,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = (
    ROOT / "evidence_synthesis" / "outputs" / "tables"
    / "greenwald2009_topology_sensitivity.csv"
)


# ---------------------------------------------------------------------------
# Debatable boundary cases
# ---------------------------------------------------------------------------
# Each entry: (description, set of sample_ids, current assignment, alternative)
# These are samples where the Linear vs Linear-Plus boundary is arguable.
DEBATABLE = [
    (
        "Other intergroup non-reflexive: lower social desirability than race/gender",
        # Other intergroup samples currently coded linear_plus (excluding DOT overrides)
        # These could be argued as simple independent evaluation (linear)
        "other_intergroup_to_linear",
    ),
    (
        "Gender/sex: some studies measure explicit gender attitudes without strong norms",
        "gender_to_linear",
    ),
    (
        "Race deliberative: some discrimination measures may not invoke norms",
        "race_delib_to_linear",
    ),
    (
        "Clinical stigma samples: mental health stigma is norm-governed",
        "clinical_stigma_to_lplus",
    ),
    (
        "Drugs/tobacco social use: substance use in social contexts is norm-governed",
        "drugs_social_to_lplus",
    ),
]


def _get_debatable_ids(df):
    """Return sample_id sets for each debatable boundary case."""
    ids = {}

    # Other intergroup currently in linear_plus (not in DOT_SAMPLES)
    mask = (df["topic"] == "Other intergroup") & (~df["sample_id"].isin(DOT_SAMPLES))
    ids["other_intergroup_to_linear"] = set(df.loc[mask, "sample_id"])

    # Gender/sex currently in linear_plus
    mask = (df["topic"] == "Gender/sex") & (~df["sample_id"].isin(DOT_SAMPLES))
    ids["gender_to_linear"] = set(df.loc[mask, "sample_id"])

    # Race deliberative currently in linear_plus (Race not in DOT_SAMPLES)
    mask = (df["topic"] == "Race (Bl/Wh)") & (~df["sample_id"].isin(DOT_SAMPLES))
    ids["race_delib_to_linear"] = set(df.loc[mask, "sample_id"])

    # Clinical non-phobic currently in linear — could be linear_plus (stigma)
    mask = (df["topic"] == "Clinical") & (~df["sample_id"].isin(DOT_SAMPLES))
    ids["clinical_stigma_to_lplus"] = set(df.loc[mask, "sample_id"])

    # Drugs/tobacco non-attentional currently in linear — could be linear_plus
    mask = (df["topic"] == "Drugs/tobacco") & (~df["sample_id"].isin(DOT_SAMPLES))
    ids["drugs_social_to_lplus"] = set(df.loc[mask, "sample_id"])

    return ids


def _recode(df, flips, debatable_ids):
    """Apply a set of flips to produce an alternative four-level coding."""
    df = df.copy()
    for key, do_flip in zip(
        ["other_intergroup_to_linear", "gender_to_linear", "race_delib_to_linear",
         "clinical_stigma_to_lplus", "drugs_social_to_lplus"],
        flips,
    ):
        if not do_flip:
            continue
        sample_ids = debatable_ids[key]
        if key.endswith("_to_linear"):
            # Move from linear_plus to linear
            mask = df["sample_id"].isin(sample_ids) & (df["dln_stage"] == "linear_plus")
            df.loc[mask, "dln_stage"] = "linear"
        elif key.endswith("_to_lplus"):
            # Move from linear to linear_plus
            mask = df["sample_id"].isin(sample_ids) & (df["dln_stage"] == "linear")
            df.loc[mask, "dln_stage"] = "linear_plus"
    return df


def main():
    raw = pd.read_csv(DATA)
    df = prepare_data(raw)

    debatable_ids = _get_debatable_ids(raw)

    print(f"Loaded {len(df)} samples")
    print(f"\nDebatable boundary cases:")
    for desc, key in DEBATABLE:
        ids = debatable_ids[key]
        print(f"  {key:<35s}: {len(ids):3d} samples  ({desc})")

    n_cases = len(DEBATABLE)
    n_scenarios = 2 ** n_cases
    print(f"\nRunning {n_scenarios} alternative codings...")

    y = df["yi_icc"].to_numpy()
    v = df["vi_icc"].to_numpy()

    # Baseline
    X_base = np.ones((len(df), 1))
    res_base = fit_reml(y, v, X_base)
    tau2_base = res_base.tau2

    results = []

    for scenario_idx, flips in enumerate(product([False, True], repeat=n_cases)):
        df_alt = _recode(df, flips, debatable_ids)

        # Check that all 4 stages are present (need >=1 per stage for moderator)
        stage_counts = df_alt["dln_stage"].value_counts()
        if any(stage_counts.get(s, 0) == 0 for s in STAGE_ORDER):
            continue

        # Fit 4-level model
        X_mod, _ = design_matrix_categorical(df_alt["dln_stage"], reference="dot")
        res_mod = fit_reml(y, v, X_mod)
        pct_red = (tau2_base - res_mod.tau2) / tau2_base * 100 if tau2_base > 0 else 0

        # Stage distribution
        dist = {s: int(stage_counts.get(s, 0)) for s in STAGE_ORDER}

        # Suppression fingerprint checks
        lp_sub = df_alt[df_alt["dln_stage"] == "linear_plus"]
        lin_sub = df_alt[df_alt["dln_stage"] == "linear"]
        lp_iec = lp_sub.dropna(subset=["iec"])
        lin_iec = lin_sub.dropna(subset=["iec"])
        lp_ecc = lp_sub.dropna(subset=["ecc"])

        lp_iec_lowest = (
            lp_iec["iec"].mean() < lin_iec["iec"].mean()
            if len(lp_iec) > 0 and len(lin_iec) > 0 else False
        )
        lp_gap_positive = (
            (lp_ecc["icc"] - lp_ecc["ecc"]).mean() > 0
            if len(lp_ecc) > 0 else False
        )

        flip_labels = [DEBATABLE[i][1] for i in range(n_cases) if flips[i]]

        results.append({
            "scenario": scenario_idx,
            "flips": "|".join(flip_labels) if flip_labels else "baseline_coding",
            "n_flips": sum(flips),
            "k_dot": dist["dot"],
            "k_linear": dist["linear"],
            "k_linear_plus": dist["linear_plus"],
            "k_network": dist["network"],
            "tau2_mod": round(res_mod.tau2, 6),
            "pct_reduction": round(pct_red, 1),
            "lp_iec_lowest": lp_iec_lowest,
            "lp_gap_positive": lp_gap_positive,
        })

    results_df = pd.DataFrame(results)

    # Summary
    print(f"\n{'=' * 72}")
    print("CODING SENSITIVITY SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total scenarios tested: {len(results_df)}")
    print(f"  tau2 reduction range: "
          f"{results_df['pct_reduction'].min():.1f}% - "
          f"{results_df['pct_reduction'].max():.1f}%")
    print(f"  Median tau2 reduction: {results_df['pct_reduction'].median():.1f}%")
    print(f"  Linear-Plus lowest iec: "
          f"{results_df['lp_iec_lowest'].sum()}/{len(results_df)} scenarios")
    print(f"  Linear-Plus positive gap: "
          f"{results_df['lp_gap_positive'].sum()}/{len(results_df)} scenarios")

    # Baseline coding row
    baseline_row = results_df[results_df["flips"] == "baseline_coding"]
    if len(baseline_row) > 0:
        print(f"\n  Baseline coding: tau2 reduction = "
              f"{baseline_row['pct_reduction'].iloc[0]:.1f}%")

    # Save
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_TABLE, index=False)
    print(f"\n  Wrote: {OUT_TABLE}")


if __name__ == "__main__":
    main()
