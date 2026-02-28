"""Competing moderator analysis: DLN stage vs. alternative codings.

For each primary dataset (Webb, Hoyt, Interoception, Desmedt), this script
fits the same REML meta-regression with 2--3 alternative moderator codings
and compares tau-squared reduction, AICc, and permutation p-values against
the DLN coding.

This analysis addresses the editorial concern that DLN coding may not
provide explanatory value beyond simpler, atheoretical alternatives.
Decision rules are stated in advance:
  - DLN highest reduction AND lowest AICc in >=3/4 datasets: strong support.
  - DLN ties with a simpler moderator: acknowledge parsimony, note
    cross-domain generality.
  - DLN loses to a simpler moderator: report honestly.

Outputs
-------
- evidence_synthesis/outputs/tables/competing_moderators_comparison.csv
- Console summary with head-to-head rankings per dataset

Usage
-----
  python evidence_synthesis/analysis/competing_moderators.py
"""

from __future__ import annotations

from itertools import product as iter_product
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from meta_pipeline import (
    aicc,
    compute_qm,
    design_matrix_categorical,
    design_matrix_stage,
    fit_reml,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_TABLE = (
    ROOT / "evidence_synthesis" / "outputs" / "tables"
    / "competing_moderators_comparison.csv"
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _fit_moderator(
    y: np.ndarray,
    v: np.ndarray,
    codes: pd.Series,
    reference: str | None = None,
) -> dict:
    """Fit baseline + moderator and return comparison metrics."""
    X_base = np.ones((len(y), 1))
    res_base = fit_reml(y, v, X_base)

    X_mod, names = design_matrix_categorical(codes, reference=reference)
    res_mod = fit_reml(y, v, X_mod)

    tau2_base = res_base.tau2
    tau2_mod = res_mod.tau2
    pct_reduction = (
        (tau2_base - tau2_mod) / tau2_base * 100 if tau2_base > 0 else 0.0
    )
    aicc_val = aicc(y, v, X_mod, res_mod.tau2)
    aicc_base = aicc(y, v, X_base, res_base.tau2)

    qm = compute_qm(y, v, X_base, X_mod)

    return {
        "tau2_base": tau2_base,
        "tau2_mod": tau2_mod,
        "pct_reduction": pct_reduction,
        "aicc": aicc_val,
        "aicc_base": aicc_base,
        "n_levels": len(codes.unique()),
        "n_params": X_mod.shape[1],
        "beta": res_mod.beta,
        "names": names,
        "QM": qm.QM,
        "QM_df": qm.df,
        "QM_p": qm.p,
    }


def _exhaustive_permutation_p(
    y: np.ndarray,
    v: np.ndarray,
    observed_tau2: float,
    n_groups: int,
) -> float:
    """Exhaustive permutation p-value over all unique partitions."""
    k = len(y)
    seen: set = set()
    count_leq = 0
    count_total = 0

    for assignment in iter_product(range(n_groups), repeat=k):
        if len(set(assignment)) < n_groups:
            continue
        # Canonical relabelling
        mapping: Dict[int, int] = {}
        canon = []
        next_label = 0
        for a in assignment:
            if a not in mapping:
                mapping[a] = next_label
                next_label += 1
            canon.append(mapping[a])
        canon_key = tuple(canon)
        if canon_key in seen:
            continue
        seen.add(canon_key)
        count_total += 1

        # Build design matrix
        n_grp = len(set(canon))
        X = np.ones((k, n_grp))
        groups = sorted(set(canon))
        for col_idx, g in enumerate(groups[1:], start=1):
            X[:, col_idx] = np.array(
                [1.0 if canon[i] == g else 0.0 for i in range(k)]
            )
        try:
            res = fit_reml(y, v, X)
            if res.tau2 <= observed_tau2 + 1e-10:
                count_leq += 1
        except (ValueError, np.linalg.LinAlgError):
            continue

    return count_leq / count_total if count_total > 0 else 1.0


# ═══════════════════════════════════════════════════════════════════════
# Dataset definitions: DLN + alternative moderator codings
# ═══════════════════════════════════════════════════════════════════════


def _load_webb():
    """Load Webb data and return (y, v, item_ids, moderator_dict)."""
    path = ROOT / "evidence_synthesis" / "extraction" / "webb2012_strategy_extraction.csv"
    df = pd.read_csv(path)
    y = df["d_plus"].to_numpy()
    v = df["vi"].to_numpy()
    items = df["strategy_sub"].to_numpy()

    moderators: Dict[str, Dict[str, str]] = {
        "DLN stage": dict(zip(items, df["dln_stage_code"])),
        "Cognitive effort": {
            "situation_selection": "low", "concentration": "low",
            "distraction": "low", "expressive_suppression": "low",
            "other_modulation": "low",
            "situation_modification": "high", "reappraisal": "high",
            "perspective_taking": "high", "other_cognitive": "high",
            "acceptance": "high",
        },
        "Gross process model": {
            "situation_selection": "antecedent",
            "situation_modification": "antecedent",
            "distraction": "antecedent", "concentration": "antecedent",
            "reappraisal": "antecedent", "perspective_taking": "antecedent",
            "other_cognitive": "antecedent", "acceptance": "antecedent",
            "expressive_suppression": "response",
            "other_modulation": "response",
        },
    }
    return y, v, items, moderators


def _load_hoyt():
    """Load Hoyt data and return (y, v, item_ids, moderator_dict)."""
    path = ROOT / "evidence_synthesis" / "extraction" / "hoyt2024_domain_extraction.csv"
    df = pd.read_csv(path)
    # Fisher-z transform
    df["z"] = np.arctanh(np.clip(df["r_pooled"], -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    y = df["z"].to_numpy()
    v = df["vi_z"].to_numpy()
    items = df["health_domain"].to_numpy()

    moderators: Dict[str, Dict[str, str]] = {
        "DLN stage": dict(zip(items, df["dln_stage_code"])),
        "Construct valence": {
            "mental_emotional_distress": "negative",
            "risk_related_adjustment": "negative",
            "positive_psychological_health": "positive",
            "social_functioning": "positive",
            "resilience_adjustment": "positive",
            "biological_physiological": "positive",
            "physical_health": "positive",
            "behavioral": "positive",
        },
        "Measurement dimensionality": {
            "mental_emotional_distress": "unidimensional",
            "risk_related_adjustment": "unidimensional",
            "behavioral": "unidimensional",
            "positive_psychological_health": "multidimensional",
            "social_functioning": "multidimensional",
            "resilience_adjustment": "multidimensional",
            "biological_physiological": "multidimensional",
            "physical_health": "multidimensional",
        },
        "Temporal frame": {
            "mental_emotional_distress": "acute",
            "risk_related_adjustment": "acute",
            "biological_physiological": "acute",
            "positive_psychological_health": "adaptive",
            "resilience_adjustment": "adaptive",
            "social_functioning": "adaptive",
            "physical_health": "adaptive",
            "behavioral": "adaptive",
        },
    }
    return y, v, items, moderators


def _load_interoception():
    """Load interoception data and return (y, v, item_ids, moderator_dict)."""
    path = ROOT / "evidence_synthesis" / "extraction" / "interoception_measure_extraction.csv"
    df = pd.read_csv(path)
    # Fisher-z transform
    df["z"] = np.arctanh(np.clip(df["r_pooled"], -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_total"] - 3.0 * df["k"])
    y = df["z"].to_numpy()
    v = df["vi_z"].to_numpy()
    items = df["measure_family"].to_numpy()

    moderators: Dict[str, Dict[str, str]] = {
        "DLN stage": dict(zip(items, df["dln_stage_code"])),
        "Garfinkel trichotomy": {
            "heartbeat_tasks": "accuracy", "ias": "accuracy",
            "bpq_ba": "sensibility", "bpq_r": "sensibility",
            "icq": "awareness", "edi_iaw": "awareness",
            "maia_total": "awareness", "baq": "awareness",
        },
        "Measurement method": {
            "heartbeat_tasks": "objective",
            "bpq_ba": "subjective", "icq": "subjective",
            "bpq_r": "subjective", "edi_iaw": "subjective",
            "ias": "subjective", "maia_total": "subjective",
            "baq": "subjective",
        },
        "Emotion-body link": {
            "heartbeat_tasks": "pure_signal", "bpq_ba": "pure_signal",
            "icq": "pure_signal", "bpq_r": "pure_signal",
            "edi_iaw": "emotion_integrated", "ias": "emotion_integrated",
            "maia_total": "emotion_integrated", "baq": "emotion_integrated",
        },
    }
    return y, v, items, moderators


def _load_desmedt():
    """Load Desmedt data and return (y, v, item_ids, moderator_dict)."""
    path = ROOT / "evidence_synthesis" / "extraction" / "desmedt2022_criterion_extraction.csv"
    df = pd.read_csv(path)
    # Fisher-z of absolute r as DV (matching run_desmedt2022_moderator.py)
    df["abs_r"] = np.abs(df["r_pooled"])
    df["z_abs"] = np.arctanh(np.clip(df["abs_r"], -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    y = df["z_abs"].to_numpy()
    v = df["vi_z"].to_numpy()
    items = df["criterion"].to_numpy()

    moderators: Dict[str, Dict[str, str]] = {
        "DLN stage": dict(zip(items, df["dln_stage_code"])),
        "Measurement objectivity": {
            "heart_rate": "objective", "bmi": "objective",
            "age": "objective", "sex": "objective",
            "trait_anxiety": "subjective", "depression": "subjective",
            "alexithymia": "subjective",
        },
        "Clinical relevance": {
            "trait_anxiety": "clinical", "depression": "clinical",
            "alexithymia": "clinical", "heart_rate": "clinical",
            "bmi": "demographic", "age": "demographic",
            "sex": "demographic",
        },
    }
    return y, v, items, moderators


# ═══════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════

DATASETS = [
    ("Webb (k=10)", _load_webb),
    ("Interoception (k=8)", _load_interoception),
    ("Hoyt (k=8)", _load_hoyt),
    ("Desmedt (k=7)", _load_desmedt),
]


def run_all() -> pd.DataFrame:
    """Run competing moderator analysis for all datasets."""
    all_rows: List[dict] = []

    for dataset_name, loader in DATASETS:
        y, v, items, moderators = loader()
        k = len(y)

        print(f"\n{'=' * 70}")
        print(f"  {dataset_name}")
        print(f"{'=' * 70}")

        for mod_name, coding_dict in moderators.items():
            codes = pd.Series([coding_dict[item] for item in items])
            n_levels = len(codes.unique())
            ref = sorted(codes.unique())[0]

            result = _fit_moderator(y, v, codes, reference=ref)

            # Permutation test (exhaustive)
            perm_p = _exhaustive_permutation_p(
                y, v, result["tau2_mod"], n_levels,
            )

            is_dln = mod_name == "DLN stage"

            row = {
                "dataset": dataset_name,
                "moderator": mod_name,
                "is_dln": is_dln,
                "k": k,
                "n_levels": n_levels,
                "n_params": result["n_params"],
                "tau2_base": round(result["tau2_base"], 6),
                "tau2_mod": round(result["tau2_mod"], 6),
                "pct_reduction": round(result["pct_reduction"], 1),
                "QM": round(result["QM"], 2),
                "QM_df": result["QM_df"],
                "QM_p": round(result["QM_p"], 4),
                "aicc": round(result["aicc"], 3),
                "aicc_base": round(result["aicc_base"], 3),
                "perm_p": round(perm_p, 4),
            }
            all_rows.append(row)

            flag = " *** DLN ***" if is_dln else ""
            print(
                f"  {mod_name:30s}  levels={n_levels}  "
                f"tau2={result['tau2_mod']:.4f}  "
                f"red={result['pct_reduction']:+5.1f}%  "
                f"QM({result['QM_df']})={result['QM']:.2f} p={result['QM_p']:.4f}  "
                f"AICc={result['aicc']:.2f}  "
                f"perm_p={perm_p:.4f}{flag}"
            )

        # Rank moderators within dataset
        dataset_rows = [r for r in all_rows if r["dataset"] == dataset_name]
        best_red = max(dataset_rows, key=lambda r: r["pct_reduction"])
        best_aicc = min(dataset_rows, key=lambda r: r["aicc"])
        dln_row = [r for r in dataset_rows if r["is_dln"]][0]

        print(f"\n  Best tau2 reduction:  {best_red['moderator']} "
              f"({best_red['pct_reduction']:.1f}%)")
        print(f"  Best AICc:           {best_aicc['moderator']} "
              f"({best_aicc['aicc']:.2f})")
        if dln_row["moderator"] == best_red["moderator"]:
            print("  --> DLN achieves highest heterogeneity reduction")
        if dln_row["moderator"] == best_aicc["moderator"]:
            print("  --> DLN achieves lowest AICc")

    return pd.DataFrame(all_rows)


def main():
    print("=" * 70)
    print("COMPETING MODERATOR ANALYSIS")
    print("DLN stage vs. alternative codings — head-to-head comparison")
    print("=" * 70)

    results = run_all()

    # Save output
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_TABLE, index=False)
    print(f"\nWrote: {OUT_TABLE}")

    # Summary across datasets
    print(f"\n{'=' * 70}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'=' * 70}")

    n_datasets = len(DATASETS)
    n_dln_best_red = 0
    n_dln_best_aicc = 0

    for dataset_name, _ in DATASETS:
        sub = results[results["dataset"] == dataset_name]
        best_red = sub.loc[sub["pct_reduction"].idxmax()]
        best_aicc_row = sub.loc[sub["aicc"].idxmin()]
        if best_red["is_dln"]:
            n_dln_best_red += 1
        if best_aicc_row["is_dln"]:
            n_dln_best_aicc += 1

    print(f"  DLN highest tau2 reduction: {n_dln_best_red}/{n_datasets}")
    print(f"  DLN lowest AICc:            {n_dln_best_aicc}/{n_datasets}")

    if n_dln_best_red >= 3 and n_dln_best_aicc >= 3:
        print("  ==> Strong support: DLN outperforms alternatives "
              "in majority of datasets")
    elif n_dln_best_red >= 3 or n_dln_best_aicc >= 3:
        print("  ==> Moderate support: DLN outperforms alternatives "
              "on one criterion in majority of datasets")
    else:
        print("  ==> Mixed support: DLN does not consistently outperform "
              "alternatives across datasets")


if __name__ == "__main__":
    main()
