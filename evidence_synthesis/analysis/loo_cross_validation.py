"""Leave-one-out cross-validation for four small-k DLN moderator analyses.

For each unit i in a dataset, drops it from the sample, fits the DLN moderator
model on the remaining k-1 units, and predicts the held-out effect size from
its stage assignment.  Reports out-of-sample R-squared and MAE.

Datasets:
  1. Webb et al. (2012)       — k=10 strategies, 3 stages, y=d_plus
  2. Interoception            — k=8 measure families, 3 stages, y=Fisher-z(r)
  3. Hoyt et al. (2024)       — k=8 health domains, 3 stages, y=Fisher-z(r)
  4. Desmedt et al. (2022)    — k=7 criteria, 2 stages, y=Fisher-z(|r|)

Outputs:
  - evidence_synthesis/outputs/tables/loo_cross_validation_summary.csv
  - evidence_synthesis/outputs/tables/loo_cross_validation_predictions.csv

Usage:
  python evidence_synthesis/analysis/loo_cross_validation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_pipeline import fit_reml, design_matrix_stage


ROOT = Path(__file__).resolve().parents[2]
OUT_SUMMARY = ROOT / "evidence_synthesis" / "outputs" / "tables" / "loo_cross_validation_summary.csv"
OUT_PREDICTIONS = ROOT / "evidence_synthesis" / "outputs" / "tables" / "loo_cross_validation_predictions.csv"


@dataclass
class LOOResult:
    dataset: str
    k: int
    y: np.ndarray
    y_hat: np.ndarray
    stages: List[str]
    loo_r2: float
    loo_r2_null: float
    loo_mae: float
    loo_mae_null: float


def loo_cross_validate(
    y: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Compute LOO predicted values.

    For each unit i, fits REML on the remaining k-1 units and predicts
    y_hat_i = X[i] @ beta_loo.

    Parameters
    ----------
    y : array, shape (k,)
    v : array, shape (k,)
    X : array, shape (k, p)

    Returns
    -------
    y_hat : array, shape (k,)
        Out-of-sample predicted values.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    k = len(y)
    y_hat = np.full(k, np.nan)

    for i in range(k):
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        res_i = fit_reml(y[mask], v[mask], X[mask])
        y_hat[i] = float(X[i] @ res_i.beta)

    return y_hat


def loo_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Out-of-sample R-squared: 1 - SS_res / SS_tot."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-15:
        return 0.0
    return 1.0 - ss_res / ss_tot


def loo_mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean absolute error of LOO predictions."""
    return float(np.mean(np.abs(y - y_hat)))


# ---------------------------------------------------------------------------
# Dataset loaders (mirroring the existing run_* scripts exactly)
# ---------------------------------------------------------------------------

def _load_webb():
    """Webb et al. (2012): k=10 strategies, y=d_plus, 3 stages."""
    data_path = ROOT / "evidence_synthesis" / "extraction" / "webb2012_strategy_extraction.csv"
    df = pd.read_csv(data_path)
    y = df["d_plus"].to_numpy()
    v = df["vi"].to_numpy()
    X_mod, _ = design_matrix_stage(df["dln_stage_code"], reference="dot")
    X_null = np.ones((len(df), 1))
    stages = df["dln_stage_code"].tolist()
    return y, v, X_mod, X_null, stages


def _load_interoception():
    """Interoception: k=8 measure families, y=Fisher-z(r), 3 stages."""
    data_path = ROOT / "evidence_synthesis" / "extraction" / "interoception_measure_extraction.csv"
    df = pd.read_csv(data_path)
    df["z"] = np.arctanh(np.clip(df["r_pooled"].to_numpy(), -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_total"] - 3.0 * df["k"])
    y = df["z"].to_numpy()
    v = df["vi_z"].to_numpy()
    X_mod, _ = design_matrix_stage(df["dln_stage_code"], reference="dot")
    X_null = np.ones((len(df), 1))
    stages = df["dln_stage_code"].tolist()
    return y, v, X_mod, X_null, stages


def _load_hoyt():
    """Hoyt et al. (2024): k=8 health domains, y=Fisher-z(r), 3 stages."""
    data_path = ROOT / "evidence_synthesis" / "extraction" / "hoyt2024_domain_extraction.csv"
    df = pd.read_csv(data_path)
    df["z"] = np.arctanh(np.clip(df["r_pooled"].to_numpy(), -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    y = df["z"].to_numpy()
    v = df["vi_z"].to_numpy()
    X_mod, _ = design_matrix_stage(df["dln_stage_code"], reference="dot")
    X_null = np.ones((len(df), 1))
    stages = df["dln_stage_code"].tolist()
    return y, v, X_mod, X_null, stages


def _load_desmedt():
    """Desmedt et al. (2022): k=7 criteria, y=Fisher-z(|r|), 2 stages."""
    data_path = ROOT / "evidence_synthesis" / "extraction" / "desmedt2022_criterion_extraction.csv"
    df = pd.read_csv(data_path)
    df["abs_r"] = df["r_pooled"].abs()
    df["z_abs"] = np.arctanh(np.clip(df["abs_r"].to_numpy(), -0.999, 0.999))
    df["vi_z"] = 1.0 / (df["N_approx"] - 3.0)
    y = df["z_abs"].to_numpy()
    v = df["vi_z"].to_numpy()
    # Two-level design matrix (only dot and linear present)
    is_dot = (df["dln_stage_code"] == "dot").astype(float).to_numpy()
    X_mod = np.column_stack([np.ones(len(df)), is_dot])
    X_null = np.ones((len(df), 1))
    stages = df["dln_stage_code"].tolist()
    return y, v, X_mod, X_null, stages


DATASETS = {
    "Webb2012": _load_webb,
    "Interoception": _load_interoception,
    "Hoyt2024": _load_hoyt,
    "Desmedt2022": _load_desmedt,
}


def run_one_dataset(name: str) -> LOOResult:
    """Run LOO cross-validation for a single dataset."""
    loader = DATASETS[name]
    y, v, X_mod, X_null, stages = loader()
    k = len(y)

    # DLN moderator LOO predictions
    y_hat_mod = loo_cross_validate(y, v, X_mod)

    # Null (intercept-only) LOO predictions
    y_hat_null = loo_cross_validate(y, v, X_null)

    return LOOResult(
        dataset=name,
        k=k,
        y=y,
        y_hat=y_hat_mod,
        stages=stages,
        loo_r2=loo_r2(y, y_hat_mod),
        loo_r2_null=loo_r2(y, y_hat_null),
        loo_mae=loo_mae(y, y_hat_mod),
        loo_mae_null=loo_mae(y, y_hat_null),
    )


@dataclass
class NullLOOResult:
    dataset: str
    observed_r2: float
    null_r2: np.ndarray
    null_median: float
    null_p95: float
    p_value: float
    n_perms: int


def simulate_null_loo(
    y: np.ndarray,
    v: np.ndarray,
    n_groups: int,
    n_perms: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate LOO R-squared under random group assignments.

    For each permutation, randomly assigns k units to n_groups non-empty
    groups, builds a dummy-coded design matrix, runs LOO cross-validation,
    and records the resulting R-squared.

    Parameters
    ----------
    y : array, shape (k,)
    v : array, shape (k,)
    n_groups : int
        Number of groups (must match the real analysis).
    n_perms : int
        Number of random permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    null_r2 : array, shape (n_perms,)
        LOO R-squared for each random partition.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    null_r2_arr = np.full(n_perms, np.nan)

    for perm_i in range(n_perms):
        # Random assignment ensuring each group has at least 2 members
        # (so that dropping one in LOO never empties a group)
        while True:
            labels = rng.integers(0, n_groups, size=k)
            counts = np.bincount(labels, minlength=n_groups)
            if np.all(counts >= 2):
                break
        # Build design matrix (intercept + n_groups-1 dummies)
        X_perm = np.ones((k, n_groups))
        for g in range(1, n_groups):
            X_perm[:, g] = (labels == g).astype(float)
        y_hat = loo_cross_validate(y, v, X_perm)
        null_r2_arr[perm_i] = loo_r2(y, y_hat)

    return null_r2_arr


def run_null_simulation(name: str, n_perms: int = 10000) -> NullLOOResult:
    """Run null-model LOO simulation for a single dataset."""
    loader = DATASETS[name]
    y, v, X_mod, X_null, stages = loader()
    n_groups = X_mod.shape[1]  # intercept + dummies = number of groups

    # Observed LOO R²
    y_hat_mod = loo_cross_validate(y, v, X_mod)
    observed = loo_r2(y, y_hat_mod)

    # Null distribution
    null_r2_arr = simulate_null_loo(y, v, n_groups, n_perms=n_perms)
    p_value = float(np.mean(null_r2_arr >= observed))
    return NullLOOResult(
        dataset=name,
        observed_r2=observed,
        null_r2=null_r2_arr,
        null_median=float(np.median(null_r2_arr)),
        null_p95=float(np.percentile(null_r2_arr, 95)),
        p_value=p_value,
        n_perms=n_perms,
    )


OUT_NULL = ROOT / "evidence_synthesis" / "outputs" / "tables" / "loo_null_simulation.csv"


def main():
    print("=" * 60)
    print("LOO CROSS-VALIDATION: DLN MODERATOR MODEL")
    print("=" * 60)

    summary_rows = []
    prediction_rows = []

    for name in DATASETS:
        result = run_one_dataset(name)

        print(f"\n--- {name} (k={result.k}) ---")
        print(f"  DLN moderator LOO-R2:  {result.loo_r2:+.3f}")
        print(f"  Null (intercept) LOO-R2: {result.loo_r2_null:+.3f}")
        print(f"  DLN moderator LOO-MAE: {result.loo_mae:.4f}")
        print(f"  Null (intercept) LOO-MAE: {result.loo_mae_null:.4f}")
        print(f"  MAE improvement: {(1 - result.loo_mae / result.loo_mae_null) * 100:.1f}%"
              if result.loo_mae_null > 0 else "  MAE improvement: N/A")

        summary_rows.append({
            "dataset": result.dataset,
            "k": result.k,
            "loo_r2_dln": round(result.loo_r2, 4),
            "loo_r2_null": round(result.loo_r2_null, 4),
            "loo_mae_dln": round(result.loo_mae, 4),
            "loo_mae_null": round(result.loo_mae_null, 4),
        })

        for i in range(result.k):
            prediction_rows.append({
                "dataset": result.dataset,
                "unit_index": i,
                "stage": result.stages[i],
                "y_obs": round(float(result.y[i]), 6),
                "y_hat_dln": round(float(result.y_hat[i]), 6),
                "residual": round(float(result.y[i] - result.y_hat[i]), 6),
            })

    # Save outputs
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    print(f"\nWrote: {OUT_SUMMARY}")

    pd.DataFrame(prediction_rows).to_csv(OUT_PREDICTIONS, index=False)
    print(f"Wrote: {OUT_PREDICTIONS}")

    # --- Null-model LOO simulation ---
    print("\n" + "=" * 60)
    print("NULL-MODEL LOO SIMULATION (10,000 random partitions)")
    print("=" * 60)

    null_rows = []
    for name in DATASETS:
        null_result = run_null_simulation(name, n_perms=10000)
        print(f"\n--- {name} ---")
        print(f"  Observed LOO-R2:     {null_result.observed_r2:.4f}")
        print(f"  Null median LOO-R2:  {null_result.null_median:.4f}")
        print(f"  Null 95th pctile:    {null_result.null_p95:.4f}")
        print(f"  p-value:             {null_result.p_value:.4f}")

        null_rows.append({
            "dataset": null_result.dataset,
            "observed_loo_r2": round(null_result.observed_r2, 4),
            "null_median": round(null_result.null_median, 4),
            "null_p95": round(null_result.null_p95, 4),
            "p_value": round(null_result.p_value, 4),
            "n_perms": null_result.n_perms,
        })

    OUT_NULL.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(null_rows).to_csv(OUT_NULL, index=False)
    print(f"\nWrote: {OUT_NULL}")


if __name__ == "__main__":
    main()
