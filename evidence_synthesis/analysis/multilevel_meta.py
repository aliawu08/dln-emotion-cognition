"""Three-level random-effects meta-analysis and robust variance estimation.

Implements:
- Three-level REML meta-regression: effects nested within studies.
  Equivalent to metafor::rma.mv(yi, vi, random = ~ 1 | study/effect, mods = X).
- Cluster-robust variance estimation (CR2 sandwich estimator) for standard
  errors that are valid even when the variance structure is misspecified.

The three-level model decomposes residual heterogeneity into:
  sigma2_within  (level 2): variation among effects within the same study
  sigma2_between (level 3): variation among studies

This is the correct specification when multiple effects are extracted from
the same primary study.

References
----------
- Assink, M., & Wink, C. H. M. (2016). Fitting three-level meta-analytic
  models in R: A step-by-step tutorial. The Quantitative Methods for
  Psychology, 12(3), 154-174.
- Cheung, M. W.-L. (2014). Modeling dependent effect sizes with three-level
  meta-analyses. Research Synthesis Methods, 5(3), 186-205.
- Tipton, E. (2015). Small sample adjustments for robust variance estimation
  with meta-regression. Psychological Methods, 20(3), 375-393.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as t_dist


@dataclass
class ThreeLevelResult:
    """Result container for a three-level meta-regression."""

    k: int  # total number of effects
    n_studies: int  # number of clusters (studies)
    p: int  # number of fixed-effects parameters
    sigma2_within: float  # level-2 variance (within-study)
    sigma2_between: float  # level-3 variance (between-study)
    beta: np.ndarray  # fixed-effects coefficients
    vcov: np.ndarray  # variance-covariance matrix of beta
    se: np.ndarray  # standard errors of beta
    ci95: np.ndarray  # 95% confidence intervals (k x 2)
    Q: float  # Cochran's Q
    I2_total: float  # total I-squared
    I2_within: float  # within-study I-squared
    I2_between: float  # between-study I-squared


def _build_V_matrix(
    vi: np.ndarray,
    study_ids: np.ndarray,
    sigma2_within: float,
    sigma2_between: float,
) -> np.ndarray:
    """Build the marginal variance-covariance matrix V.

    V_ij = vi*I + sigma2_within*I(same_study) + sigma2_between*J(same_study)

    For effects i,j in the same study:
      V[i,i] = vi[i] + sigma2_within + sigma2_between
      V[i,j] = sigma2_between  (i != j, same study)
    For effects in different studies:
      V[i,j] = 0
    """
    k = len(vi)
    V = np.zeros((k, k))
    for i in range(k):
        V[i, i] = vi[i] + sigma2_within + sigma2_between
        for j in range(i + 1, k):
            if study_ids[i] == study_ids[j]:
                V[i, j] = sigma2_between
                V[j, i] = sigma2_between
    return V


def _reml_objective_3level(
    params: np.ndarray,
    y: np.ndarray,
    vi: np.ndarray,
    X: np.ndarray,
    study_ids: np.ndarray,
) -> float:
    """REML negative log-likelihood for the three-level model."""
    sigma2_within = max(float(params[0]), 0.0)
    sigma2_between = max(float(params[1]), 0.0)

    k = len(y)
    p = X.shape[1]

    V = _build_V_matrix(vi, study_ids, sigma2_within, sigma2_between)

    try:
        L = np.linalg.cholesky(V)
    except np.linalg.LinAlgError:
        return 1e12

    # log|V| via Cholesky
    log_det_V = 2.0 * np.sum(np.log(np.diag(L)))

    # V^{-1} via Cholesky
    V_inv = np.linalg.solve(V, np.eye(k))

    # X'V^{-1}X
    XtVi = X.T @ V_inv
    XtViX = XtVi @ X

    try:
        sign, log_det_XtViX = np.linalg.slogdet(XtViX)
        if sign <= 0:
            return 1e12
        beta = np.linalg.solve(XtViX, XtVi @ y)
    except np.linalg.LinAlgError:
        return 1e12

    e = y - X @ beta
    obj = 0.5 * (log_det_V + log_det_XtViX + float(e.T @ V_inv @ e))
    return obj


def fit_three_level(
    y: np.ndarray,
    vi: np.ndarray,
    X: np.ndarray,
    study_ids: np.ndarray,
) -> ThreeLevelResult:
    """Fit a three-level random-effects meta-regression via REML.

    Parameters
    ----------
    y : array, shape (k,)
        Effect sizes.
    vi : array, shape (k,)
        Known sampling variances (level 1).
    X : array, shape (k, p)
        Design matrix for fixed effects.
    study_ids : array, shape (k,)
        Study identifiers (used for clustering).

    Returns
    -------
    ThreeLevelResult
    """
    y = np.asarray(y, dtype=float).ravel()
    vi = np.asarray(vi, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    study_ids = np.asarray(study_ids)

    k = len(y)
    p = X.shape[1]
    n_studies = len(np.unique(study_ids))

    # Starting values: DerSimonian-Laird total tau2, split evenly
    w_fe = 1.0 / vi
    mu_fe = np.sum(w_fe * y) / np.sum(w_fe)
    Q_total = float(np.sum(w_fe * (y - mu_fe) ** 2))
    c = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
    tau2_dl = max((Q_total - (k - 1)) / c, 0.01)
    init = np.array([tau2_dl / 2.0, tau2_dl / 2.0])

    # Optimize
    res = minimize(
        _reml_objective_3level,
        init,
        args=(y, vi, X, study_ids),
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
        options={"maxiter": 5000, "ftol": 1e-10},
    )

    sigma2_within = max(float(res.x[0]), 0.0)
    sigma2_between = max(float(res.x[1]), 0.0)

    # Final estimates
    V = _build_V_matrix(vi, study_ids, sigma2_within, sigma2_between)
    V_inv = np.linalg.inv(V)

    XtVi = X.T @ V_inv
    XtViX = XtVi @ X
    beta = np.linalg.solve(XtViX, XtVi @ y)
    vcov_model = np.linalg.inv(XtViX)

    # Knapp-Hartung adjustment
    e = y - X @ beta
    qe = float(e.T @ V_inv @ e)
    kh_scale = max(qe / max(k - p, 1), 1.0)
    vcov = kh_scale * vcov_model
    se = np.sqrt(np.diag(vcov))

    # t-based CIs (Satterthwaite df approximation: n_studies - p for
    # between-study moderators; conservative lower bound)
    df_kh = max(n_studies - p, 1)
    t_crit = float(t_dist.ppf(0.975, df_kh))
    ci95 = np.vstack([beta - t_crit * se, beta + t_crit * se]).T

    # Cochran's Q (on fixed-effects weights)
    Q = float(np.sum((1.0 / vi) * (y - mu_fe) ** 2))

    # I-squared decomposition
    v_bar = float(np.mean(vi))
    total_var = sigma2_within + sigma2_between + v_bar
    I2_total = (sigma2_within + sigma2_between) / total_var if total_var > 0 else 0.0
    I2_within = sigma2_within / total_var if total_var > 0 else 0.0
    I2_between = sigma2_between / total_var if total_var > 0 else 0.0

    return ThreeLevelResult(
        k=k,
        n_studies=n_studies,
        p=p,
        sigma2_within=sigma2_within,
        sigma2_between=sigma2_between,
        beta=beta,
        vcov=vcov,
        se=se,
        ci95=ci95,
        Q=Q,
        I2_total=I2_total,
        I2_within=I2_within,
        I2_between=I2_between,
    )


# ---------------------------------------------------------------------------
# Cluster-robust variance estimation (CR2)
# ---------------------------------------------------------------------------


@dataclass
class RobustResult:
    """Result container for cluster-robust variance estimation."""

    beta: np.ndarray
    vcov_robust: np.ndarray
    se_robust: np.ndarray
    ci95_robust: np.ndarray
    df_robust: np.ndarray  # Satterthwaite degrees of freedom per coefficient
    t_stat: np.ndarray
    p_value: np.ndarray
    n_clusters: int


def cluster_robust_variance(
    y: np.ndarray,
    vi: np.ndarray,
    X: np.ndarray,
    study_ids: np.ndarray,
    sigma2_within: float,
    sigma2_between: float,
) -> RobustResult:
    """Compute CR2 cluster-robust standard errors.

    Uses the bias-reduced linearization (BRL / CR2) adjustment of
    Tipton (2015) and Tipton & Pustejovsky (2015).

    Parameters
    ----------
    y, vi, X, study_ids : as in fit_three_level
    sigma2_within, sigma2_between : variance component estimates from the
        three-level model (used to build the working covariance).
    """
    y = np.asarray(y, dtype=float).ravel()
    vi = np.asarray(vi, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    study_ids = np.asarray(study_ids)

    k = len(y)
    p = X.shape[1]

    V = _build_V_matrix(vi, study_ids, sigma2_within, sigma2_between)
    V_inv = np.linalg.inv(V)

    XtVi = X.T @ V_inv
    XtViX = XtVi @ X
    XtViX_inv = np.linalg.inv(XtViX)
    beta = np.linalg.solve(XtViX, XtVi @ y)
    e = y - X @ beta

    # Hat matrix: H = X (X'V^{-1}X)^{-1} X'V^{-1}
    H = X @ XtViX_inv @ XtVi

    # Cluster-level meat
    unique_studies = np.unique(study_ids)
    n_clusters = len(unique_studies)
    meat = np.zeros((p, p))

    for s in unique_studies:
        idx = np.where(study_ids == s)[0]
        e_s = e[idx]
        X_s = X[idx]

        # CR2 adjustment: A_s = (I - H_ss)^{-1/2} per cluster
        H_ss = H[np.ix_(idx, idx)]
        I_s = np.eye(len(idx))
        M_s = I_s - H_ss

        # Regularize if near-singular
        eigvals = np.linalg.eigvalsh(M_s)
        if np.min(eigvals) < 1e-6:
            M_s += 1e-6 * I_s

        # Matrix square root of inverse
        eigvals_m, eigvecs_m = np.linalg.eigh(M_s)
        eigvals_m = np.maximum(eigvals_m, 1e-8)
        A_s = eigvecs_m @ np.diag(1.0 / np.sqrt(eigvals_m)) @ eigvecs_m.T

        e_adj = A_s @ e_s
        u_s = X_s.T @ np.diag(1.0 / (vi[idx] + sigma2_within + sigma2_between)) @ e_adj
        meat += np.outer(u_s, u_s)

    # Sandwich: (X'V^{-1}X)^{-1} M (X'V^{-1}X)^{-1}
    vcov_robust = XtViX_inv @ meat @ XtViX_inv

    se_robust = np.sqrt(np.diag(vcov_robust))

    # Satterthwaite df (simplified: use n_clusters - p as conservative bound)
    df_robust = np.full(p, max(n_clusters - p, 1), dtype=float)

    t_stat = beta / se_robust
    p_value = np.array(
        [2.0 * float(t_dist.sf(abs(t), df)) for t, df in zip(t_stat, df_robust)]
    )
    t_crit = np.array([float(t_dist.ppf(0.975, df)) for df in df_robust])
    ci95_robust = np.vstack([beta - t_crit * se_robust, beta + t_crit * se_robust]).T

    return RobustResult(
        beta=beta,
        vcov_robust=vcov_robust,
        se_robust=se_robust,
        ci95_robust=ci95_robust,
        df_robust=df_robust,
        t_stat=t_stat,
        p_value=p_value,
        n_clusters=n_clusters,
    )
