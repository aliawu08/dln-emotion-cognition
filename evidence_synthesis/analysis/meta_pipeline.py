"""Meta-analysis utilities for the DLN Emotion–Cognition evidence synthesis.

This module implements:
- Random-effects meta-analysis (intercept-only) via REML.
- Random-effects meta-regression with categorical moderators (e.g., DLN stage) via REML.

The implementation is intentionally lightweight (numpy/scipy) so the repo is runnable
without requiring specialized meta-analysis packages.

Notes
-----
- `yi` should be on a scale appropriate for meta-analysis (e.g., Fisher z for correlations).
- `vi` is the sampling variance of `yi`.

For more advanced workflows (three-level models, correlated effects, robust variance),
consider extending this code or using dedicated tooling (e.g., metafor in R).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import t as t_dist


@dataclass
class MetaResult:
    k: int
    p: int
    tau2: float
    beta: np.ndarray
    vcov: np.ndarray
    se: np.ndarray
    ci95: np.ndarray
    Q: float
    I2: float


def _reml_objective(tau2: float, y: np.ndarray, v: np.ndarray, X: np.ndarray) -> float:
    """Restricted-likelihood objective for tau^2 (constants dropped)."""
    tau2 = max(float(tau2), 0.0)
    y = y.ravel()
    v = v.ravel()

    V = v + tau2
    if np.any(V <= 0):
        return np.inf

    W = np.diag(1.0 / V)

    XtW = X.T @ W
    XtWX = XtW @ X

    try:
        sign, logdet = np.linalg.slogdet(XtWX)
        if sign <= 0:
            return np.inf
        beta = np.linalg.solve(XtWX, XtW @ y)
    except np.linalg.LinAlgError:
        return np.inf

    e = y - X @ beta
    obj = 0.5 * (np.sum(np.log(V)) + logdet + float(e.T @ W @ e))
    return obj


def fit_reml(y: np.ndarray, v: np.ndarray, X: np.ndarray) -> MetaResult:
    """Fit random-effects meta-(re)gression via REML for tau^2."""
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)

    k = int(y.shape[0])
    p = int(X.shape[1])

    res = minimize_scalar(
        lambda t: _reml_objective(t, y, v, X),
        bounds=(0.0, 10.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    tau2 = float(max(res.x, 0.0))

    V = v + tau2
    W = np.diag(1.0 / V)
    XtW = X.T @ W
    XtWX = XtW @ X

    beta = np.linalg.solve(XtWX, XtW @ y)
    vcov_naive = np.linalg.inv(XtWX)

    # Knapp-Hartung correction: scale vcov by residual heterogeneity
    e_re = y - X @ beta
    qe = max(float(e_re.T @ W @ e_re) / max(k - p, 1), 1.0)
    vcov = qe * vcov_naive
    se = np.sqrt(np.diag(vcov))

    # t-based CI with k-p degrees of freedom (Knapp-Hartung)
    df_kh = max(k - p, 1)
    t_crit = float(t_dist.ppf(0.975, df_kh))
    ci95 = np.vstack([beta - t_crit * se, beta + t_crit * se]).T

    # Cochran's Q on the fixed-effects weights (classic definition)
    e_fe = y - X @ np.linalg.solve(X.T @ np.diag(1.0 / v) @ X, X.T @ np.diag(1.0 / v) @ y)
    Q = float(e_fe.T @ np.diag(1.0 / v) @ e_fe)

    df = max(k - p, 1)
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0

    return MetaResult(k=k, p=p, tau2=tau2, beta=beta, vcov=vcov, se=se, ci95=ci95, Q=Q, I2=I2)


def design_matrix_stage(stage: pd.Series, reference: str = "dot") -> Tuple[np.ndarray, List[str]]:
    """Build an intercept + dummy-coded design matrix for DLN stage."""
    stage = stage.astype(str)
    levels = ["dot", "linear", "network"]
    if reference not in levels:
        raise ValueError(f"reference must be one of {levels}")

    stage_cat = pd.Categorical(stage, categories=levels, ordered=False)

    X_parts = [np.ones((len(stage_cat), 1))]
    names = ["Intercept"]

    for lvl in levels:
        if lvl == reference:
            continue
        X_parts.append(np.asarray(stage_cat == lvl, dtype=float).reshape(-1, 1))
        names.append(f"stage[{lvl}]")
    X = np.hstack(X_parts)
    return X, names


def summarize_meta(df: pd.DataFrame, y_col: str = "yi", v_col: str = "vi") -> Dict[str, MetaResult]:
    """Fit intercept-only random-effects models per paradigm family."""
    out: Dict[str, MetaResult] = {}
    for paradigm, sub in df.groupby("paradigm_family"):
        X = np.ones((len(sub), 1))
        out[paradigm] = fit_reml(sub[y_col].to_numpy(), sub[v_col].to_numpy(), X)
    return out


def summarize_meta_with_stage(
    df: pd.DataFrame,
    y_col: str = "yi",
    v_col: str = "vi",
    stage_col: str = "dln_stage_code",
) -> Dict[str, Tuple[MetaResult, List[str]]]:
    """Fit random-effects meta-regressions per paradigm family with DLN stage as moderator."""
    out: Dict[str, Tuple[MetaResult, List[str]]] = {}
    for paradigm, sub in df.groupby("paradigm_family"):
        X, names = design_matrix_stage(sub[stage_col], reference="dot")
        out[paradigm] = (fit_reml(sub[y_col].to_numpy(), sub[v_col].to_numpy(), X), names)
    return out


def fisher_z_to_r(z: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(z, dtype=float))


def results_to_frame(
    base: Dict[str, MetaResult],
    mod: Dict[str, Tuple[MetaResult, List[str]]],
) -> pd.DataFrame:
    rows = []
    for paradigm, base_res in base.items():
        mod_res, names = mod[paradigm]
        rows.append({
            "paradigm_family": paradigm,
            "k": base_res.k,
            "tau2_base": base_res.tau2,
            "I2_base": base_res.I2,
            "mu_base": base_res.beta[0],
            "mu_base_r": float(fisher_z_to_r(np.array([base_res.beta[0]]))[0]),
            "tau2_mod": mod_res.tau2,
            "I2_mod": mod_res.I2,
            "delta_tau2": base_res.tau2 - mod_res.tau2,
            "beta_names": "|".join(names),
            "beta_mod": "|".join([f"{b:.4f}" for b in mod_res.beta]),
        })
    return pd.DataFrame(rows)
