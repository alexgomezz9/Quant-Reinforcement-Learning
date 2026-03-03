"""
Tail-risk metrics: empirical VaR and CVaR (Expected Shortfall).

Both metrics operate on a Series of returns (negative = loss).
All values are returned as negative floats representing losses.

Key distinction
---------------
VaR(alpha)  = quantile threshold: P(r < VaR) = alpha
CVaR(alpha) = mean of returns *below* the VaR threshold (the tail average)

CVaR is always <= VaR (worse or equal), and is the metric used in the
RL reward function because it captures the severity of tail losses,
not just the boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Compute empirical (historical) Value-at-Risk.

    Parameters
    ----------
    returns:
        Daily return series. Negative values represent losses.
    alpha:
        Tail probability level. alpha=0.05 means the 5% worst days.

    Returns
    -------
    float
        VaR as a negative number (e.g. -0.02 means a loss of 2 %).
        Interpretation: on the worst alpha% of days, losses exceed this value.

    Raises
    ------
    ValueError
        If returns is empty or alpha is not in (0, 1).
    """
    _validate_inputs(returns, alpha)
    return float(np.quantile(returns, alpha))


def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """Compute empirical (historical) CVaR / Expected Shortfall.

    CVaR is the mean of all returns that fall below the VaR threshold.
    It answers: "given that we are in the worst alpha% of days,
    what is the average loss?"

    Parameters
    ----------
    returns:
        Daily return series. Negative values represent losses.
    alpha:
        Tail probability level. alpha=0.05 means the worst 5% of days.

    Returns
    -------
    float
        CVaR as a negative number. Always <= VaR (same or worse).

    Raises
    ------
    ValueError
        If returns is empty or alpha is not in (0, 1).
    """
    _validate_inputs(returns, alpha)
    var = var_historical(returns, alpha)
    tail = returns[returns <= var]
    if tail.empty:
        # Edge case: no return falls below VaR threshold (very short series)
        return var
    return float(tail.mean())


def rolling_cvar(
    returns: pd.Series,
    window: int,
    alpha: float = 0.05,
) -> pd.Series:
    """Compute a rolling CVaR estimate over a sliding window.

    This is the estimator used inside the RL reward function:
        reward_t = net_return_t - lambda * rolling_cvar_t

    Parameters
    ----------
    returns:
        Daily return series.
    window:
        Number of past days to include in each CVaR estimate (K in the spec).
        Must be >= 2 / alpha to have at least 2 observations in the tail.
    alpha:
        Tail probability level (default 0.05).

    Returns
    -------
    pd.Series
        Rolling CVaR series, same index as `returns`.
        First (window - 1) values are NaN (insufficient history).

    Notes
    -----
    With window=60 and alpha=0.05, each estimate uses the worst ~3 returns,
    which is noisy but workable. With alpha=0.01 you would need window>=200
    for at least 2 tail observations — see docs/project_spec.md.
    """
    _validate_inputs(returns, alpha)

    def _cvar_window(r: np.ndarray) -> float:
        var = np.quantile(r, alpha)
        tail = r[r <= var]
        return float(tail.mean()) if len(tail) > 0 else float(var)

    return returns.rolling(window=window, min_periods=window).apply(
        _cvar_window, raw=True
    )


# ── Internal helpers ──────────────────────────────────────────────────


def _validate_inputs(returns: pd.Series, alpha: float) -> None:
    if returns.empty:
        raise ValueError("returns series is empty.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
