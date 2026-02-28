"""Meta-analysis utilities for the DLN Emotion–Cognition evidence synthesis.

This module implements:
- Random-effects meta-analysis (intercept-only) via REML.
- Random-effects meta-regression with categorical moderators (e.g., DLN stage) via REML.
- Egger's regression test for funnel-plot asymmetry (publication bias).
- Leave-one-out influence diagnostics.
- Profile-likelihood confidence intervals for tau-squared.
- Forest-plot generation.

The implementation is intentionally lightweight (numpy/scipy) so the repo is runnable
without requiring specialized meta-analysis packages.

Notes
-----
- `yi` should be on a scale appropriate for meta-analysis (e.g., Fisher z for correlations).
- `vi` is the sampling variance of `yi`.
- When effects are nested within studies, three-level or robust-variance models
  are recommended. This pipeline currently assumes independent effects; see
  metafor::rma.mv() in R for multi-level extensions.

For more advanced workflows (three-level models, correlated effects, robust variance),
consider extending this code or using dedicated tooling (e.g., metafor in R).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.stats import chi2 as chi2_dist
from scipy.stats import t as t_dist
from scipy.stats import linregress


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

    # Data-driven upper bound: max sampling variance * 100, floored at 10.
    tau2_upper = max(10.0, float(np.max(v)) * 100.0)
    res = minimize_scalar(
        lambda t: _reml_objective(t, y, v, X),
        bounds=(0.0, tau2_upper),
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


def design_matrix_categorical(
    codes: pd.Series, reference: str | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build an intercept + dummy-coded design matrix for an arbitrary
    categorical moderator.

    Parameters
    ----------
    codes : pd.Series
        Categorical codes (strings) for each study / effect.
    reference : str, optional
        Reference level.  If *None*, the alphabetically first level is used.

    Returns
    -------
    X : ndarray, shape (k, 1 + n_levels - 1)
    names : list of str
    """
    codes = codes.astype(str)
    levels = sorted(codes.unique())
    if reference is None:
        reference = levels[0]
    if reference not in levels:
        raise ValueError(f"reference '{reference}' not in levels {levels}")

    X_parts = [np.ones((len(codes), 1))]
    names: List[str] = ["Intercept"]

    for lvl in levels:
        if lvl == reference:
            continue
        X_parts.append(np.asarray(codes == lvl, dtype=float).reshape(-1, 1))
        names.append(f"mod[{lvl}]")
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


# ---------------------------------------------------------------------------
# Egger's regression test for funnel-plot asymmetry
# ---------------------------------------------------------------------------

@dataclass
class EggerResult:
    intercept: float
    se: float
    t_stat: float
    p_value: float
    k: int


def egger_test(y: np.ndarray, v: np.ndarray) -> EggerResult:
    """Egger's regression test for funnel-plot asymmetry.

    Regresses standardized effect sizes (y / sqrt(v)) on precision (1 / sqrt(v)).
    A significant intercept indicates asymmetry suggestive of publication bias.

    References
    ----------
    Egger, M., et al. (1997). BMJ, 315(7109), 629-634.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    k = len(y)
    se_i = np.sqrt(v)
    precision = 1.0 / se_i
    z_i = y / se_i

    # linregress requires variation in x; if all precisions are identical
    # (equal sampling variances), the test is undefined.
    if np.ptp(precision) < 1e-12 or k < 3:
        return EggerResult(intercept=np.nan, se=np.nan, t_stat=np.nan,
                           p_value=np.nan, k=k)

    result = linregress(precision, z_i)
    intercept_se = float(result.intercept_stderr)
    if intercept_se > 0:
        t_val = float(result.intercept / intercept_se)
        p_val = float(2.0 * t_dist.sf(abs(t_val), max(k - 2, 1)))
    else:
        t_val = 0.0
        p_val = 1.0

    return EggerResult(
        intercept=float(result.intercept),
        se=intercept_se,
        t_stat=t_val,
        p_value=p_val,
        k=k,
    )


# ---------------------------------------------------------------------------
# Leave-one-out influence diagnostics
# ---------------------------------------------------------------------------

@dataclass
class LeaveOneOutResult:
    indices: List[int]
    tau2: np.ndarray
    beta: np.ndarray
    Q: np.ndarray


def leave_one_out(y: np.ndarray, v: np.ndarray, X: np.ndarray) -> LeaveOneOutResult:
    """Refit the model k times, each time dropping one study.

    Returns arrays of tau-squared, beta, and Q values, one per dropped study.
    Useful for identifying influential outliers.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    k = len(y)

    indices = list(range(k))
    tau2_arr = np.zeros(k)
    beta_arr = np.zeros((k, X.shape[1]))
    Q_arr = np.zeros(k)

    for i in range(k):
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        res_i = fit_reml(y[mask], v[mask], X[mask])
        tau2_arr[i] = res_i.tau2
        beta_arr[i] = res_i.beta
        Q_arr[i] = res_i.Q

    return LeaveOneOutResult(indices=indices, tau2=tau2_arr, beta=beta_arr, Q=Q_arr)


# ---------------------------------------------------------------------------
# Profile-likelihood confidence interval for tau-squared
# ---------------------------------------------------------------------------

@dataclass
class ProfileLikelihoodCI:
    tau2_hat: float
    ci_lo: float
    ci_hi: float
    alpha: float


def profile_likelihood_ci(
    y: np.ndarray,
    v: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05,
) -> ProfileLikelihoodCI:
    """Compute a profile-likelihood confidence interval for tau-squared.

    Finds the region where the REML log-likelihood is within chi2(1, 1-alpha)/2
    of the maximum, following Viechtbauer (2007).
    """
    from scipy.stats import chi2

    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)

    # Fit to get point estimate
    full_res = fit_reml(y, v, X)
    tau2_hat = full_res.tau2
    ll_max = -_reml_objective(tau2_hat, y, v, X)
    cutoff = ll_max - 0.5 * float(chi2.ppf(1.0 - alpha, 1))

    tau2_upper = max(10.0, float(np.max(v)) * 100.0)

    # Find lower bound via bisection
    lo = 0.0
    hi = tau2_hat
    if -_reml_objective(0.0, y, v, X) >= cutoff:
        ci_lo = 0.0
    else:
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if -_reml_objective(mid, y, v, X) >= cutoff:
                hi = mid
            else:
                lo = mid
        ci_lo = (lo + hi) / 2.0

    # Find upper bound via bisection
    lo = tau2_hat
    hi = tau2_upper
    # Expand if needed
    while -_reml_objective(hi, y, v, X) >= cutoff and hi < 1e6:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if -_reml_objective(mid, y, v, X) >= cutoff:
            lo = mid
        else:
            hi = mid
    ci_hi = (lo + hi) / 2.0

    return ProfileLikelihoodCI(
        tau2_hat=tau2_hat, ci_lo=ci_lo, ci_hi=ci_hi, alpha=alpha,
    )


# ---------------------------------------------------------------------------
# AICc (small-sample corrected Akaike Information Criterion)
# ---------------------------------------------------------------------------


def aicc(y: np.ndarray, v: np.ndarray, X: np.ndarray, tau2: float) -> float:
    """Compute small-sample corrected AIC for a meta-regression model.

    AICc = -2 * REML_LL + 2p + 2p(p+1)/(k - p - 1)

    where p is the number of fixed-effects parameters plus one (for tau^2),
    and k is the number of studies.  The correction term penalises complexity
    more heavily when k is small, following Hurvich and Tsai (1989).

    Parameters
    ----------
    y : array, shape (k,)
        Effect sizes.
    v : array, shape (k,)
        Sampling variances.
    X : array, shape (k, p)
        Design matrix.
    tau2 : float
        Estimated between-study variance.

    Returns
    -------
    float
        AICc value (lower is better).
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X = np.asarray(X, dtype=float)

    k = len(y)
    # p_total includes fixed-effects parameters + 1 for tau^2
    p_total = X.shape[1] + 1
    ll = -_reml_objective(tau2, y, v, X)
    aic = -2.0 * ll + 2.0 * p_total
    # Small-sample correction (undefined when k - p_total - 1 <= 0)
    denom = k - p_total - 1
    if denom > 0:
        correction = 2.0 * p_total * (p_total + 1) / denom
    else:
        correction = np.inf
    return aic + correction


# ---------------------------------------------------------------------------
# QM omnibus moderator test
# ---------------------------------------------------------------------------

@dataclass
class QMResult:
    QM: float
    df: int
    p: float


def compute_qm(
    y: np.ndarray,
    v: np.ndarray,
    X_base: np.ndarray,
    X_mod: np.ndarray,
) -> QMResult:
    """Compute the omnibus moderator test QM.

    QM equals Cochran's Q under the baseline (intercept-only) model minus Q
    under the moderator model, evaluated against a chi-squared distribution
    with degrees of freedom equal to the difference in model parameters.
    Both Q values use fixed-effects (inverse-variance) weights, following
    the standard meta-analytic definition.

    Parameters
    ----------
    y : array, shape (k,)
        Effect sizes.
    v : array, shape (k,)
        Sampling variances.
    X_base : array, shape (k, p_base)
        Baseline design matrix (typically intercept-only).
    X_mod : array, shape (k, p_mod)
        Moderator design matrix (intercept + moderator dummies).

    Returns
    -------
    QMResult
        Named tuple with QM statistic, degrees of freedom, and p-value.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    X_base = np.asarray(X_base, dtype=float)
    X_mod = np.asarray(X_mod, dtype=float)

    W_fe = np.diag(1.0 / v)

    # Q under baseline model
    beta_base = np.linalg.solve(X_base.T @ W_fe @ X_base, X_base.T @ W_fe @ y)
    e_base = y - X_base @ beta_base
    Q_base = float(e_base.T @ W_fe @ e_base)

    # Q under moderator model
    beta_mod = np.linalg.solve(X_mod.T @ W_fe @ X_mod, X_mod.T @ W_fe @ y)
    e_mod = y - X_mod @ beta_mod
    Q_res = float(e_mod.T @ W_fe @ e_mod)

    QM = Q_base - Q_res
    df = X_mod.shape[1] - X_base.shape[1]
    p = float(chi2_dist.sf(max(QM, 0.0), max(df, 1)))

    return QMResult(QM=QM, df=df, p=p)


# ---------------------------------------------------------------------------
# Forest plot
# ---------------------------------------------------------------------------

def forest_plot(
    labels: List[str],
    y: np.ndarray,
    v: np.ndarray,
    stage: pd.Series | None = None,
    pooled_beta: float | None = None,
    pooled_ci: tuple | None = None,
    title: str = "Forest plot",
    xlabel: str = "Effect size",
    outpath: str | None = None,
) -> plt.Figure:
    """Produce a standard forest plot with study-level CIs and optional pooled diamond.

    Parameters
    ----------
    labels : list of str
        Study or strategy labels (one per effect).
    y : array-like
        Point estimates.
    v : array-like
        Sampling variances.
    stage : pd.Series, optional
        DLN stage codes for color-coding.
    pooled_beta : float, optional
        Pooled effect estimate (drawn as diamond).
    pooled_ci : (lo, hi), optional
        95% CI for the pooled estimate.
    title : str
    xlabel : str
    outpath : str, optional
        If provided, save figure to this path.
    """
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    se = np.sqrt(v)
    k = len(y)

    stage_colors = {"dot": "#e74c3c", "linear": "#f39c12", "network": "#27ae60"}

    fig, ax = plt.subplots(figsize=(7, max(3, k * 0.4 + 1)))
    positions = list(range(k))

    for i in range(k):
        ci_lo = y[i] - 1.96 * se[i]
        ci_hi = y[i] + 1.96 * se[i]
        color = "#333333"
        if stage is not None:
            s = str(stage.iloc[i]) if hasattr(stage, "iloc") else str(stage[i])
            color = stage_colors.get(s, "#333333")
        ax.plot([ci_lo, ci_hi], [i, i], color=color, linewidth=1.5)
        ax.plot(y[i], i, "s", color=color, markersize=6)

    if pooled_beta is not None and pooled_ci is not None:
        diamond_y = -1
        dh = 0.3
        diamond_x = [pooled_ci[0], pooled_beta, pooled_ci[1], pooled_beta]
        diamond_ys = [diamond_y, diamond_y - dh, diamond_y, diamond_y + dh]
        ax.fill(diamond_x, diamond_ys, color="#2c3e50", alpha=0.7)
        positions.append(diamond_y)
        labels = list(labels) + ["Pooled"]

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=200)

    return fig
