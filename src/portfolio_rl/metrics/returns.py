"""
Performance metrics: returns, volatility, Sharpe ratio, max drawdown.

All functions operate on pandas Series of *daily* returns (not prices).
Use `log_returns()` to convert a price series to log returns first.

Annualisation convention: 252 trading days per year throughout.

Returns type convention
-----------------------
Several functions accept a ``returns_type`` parameter:
- "log"    : logarithmic returns  r_t = ln(P_t / P_{t-1})
- "simple" : simple (arithmetic) returns  r_t = P_t/P_{t-1} - 1

When in doubt: use "log" for statistical analysis and feature construction;
use "simple" when computing actual wealth / P&L accumulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Standard number of trading days used to annualise daily statistics.
TRADING_DAYS_PER_YEAR: int = 252

# Tolerance for near-zero volatility checks (avoids division by tiny float).
EPS: float = 1e-12


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Compute daily log returns from a price series or matrix.

    Parameters
    ----------
    prices:
        Series or DataFrame of adjusted closing prices.
        Index must be a DatetimeIndex sorted in ascending order.

    Returns
    -------
    Series or DataFrame of the same shape minus the first row (NaN dropped).

    Notes
    -----
    log return at t = ln(P_t / P_{t-1})
    Equivalent to np.log(prices).diff().dropna()
    """
    return np.log(prices / prices.shift(1)).dropna()


def annualised_return(
    returns: pd.Series,
    returns_type: str = "log",
) -> float:
    """Compute the annualised return.

    Parameters
    ----------
    returns:
        Daily return series.
    returns_type:
        "log" (default) or "simple".
        - log    → arithmetic mean × 252 (exact for log returns).
        - simple → geometric mean: (∏(1+r))^(252/n) − 1 (compound growth).

    Returns
    -------
    float
        Annualised return (e.g. 0.08 means 8 % per year).
    """
    r = returns.dropna()
    if returns_type == "log":
        return float(r.mean() * TRADING_DAYS_PER_YEAR)
    elif returns_type == "simple":
        n = len(r)
        if n == 0:
            return 0.0
        total_growth = (1 + r).prod()
        return float(total_growth ** (TRADING_DAYS_PER_YEAR / n) - 1)
    else:
        raise ValueError("returns_type must be 'log' or 'simple'.")


def annualised_volatility(returns: pd.Series) -> float:
    """Compute the annualised return volatility (standard deviation).

    Parameters
    ----------
    returns:
        Daily return series (log or simple — std is the same either way
        for small daily moves, which is the standard industry approximation).

    Returns
    -------
    float
        Annualised volatility (e.g. 0.15 means 15 % per year).

    Notes
    -----
    Annualisation: σ_annual = σ_daily × √252.
    This assumes i.i.d. returns (standard simplification in practice).
    """
    return float(returns.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Daily return series.
    risk_free_rate:
        Annualised risk-free rate (default 0.0).
        When non-zero, it is converted to a daily rate internally.

    Returns
    -------
    float
        Sharpe ratio. Returns 0.0 if volatility is zero.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns.dropna() - daily_rf
    vol = excess.std(ddof=1)
    # Use EPS tolerance instead of exact == 0 to handle floating-point
    # precision issues when all returns are identical.
    if vol < EPS:
        return 0.0
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: pd.Series, returns_type: str = "simple") -> float:
    """Compute the maximum drawdown from a daily return series.

    Maximum drawdown is the largest peak-to-trough decline in cumulative
    portfolio wealth over the full history. It is always <= 0.

    Parameters
    ----------
    returns:
        Daily return series.
    returns_type:
        "simple" (default) or "log".
        Must match how the returns were computed:
        - simple → wealth = ∏(1 + r_t)   [multiplicative compounding]
        - log    → wealth = exp(∑ r_t)    [equivalent, but different formula]
        Mixing up the two gives a silently wrong result.

    Returns
    -------
    float
        Max drawdown as a negative fraction (e.g. -0.34 means −34 %).
    """
    r = returns.dropna()
    if r.empty:
        return 0.0

    if returns_type == "log":
        # exp(cumulative sum) recovers the wealth ratio relative to start
        wealth = np.exp(r.cumsum())
    elif returns_type == "simple":
        wealth = (1 + r).cumprod()
    else:
        raise ValueError("returns_type must be 'simple' or 'log'.")

    # Prepend W_0 = 1.0 (capital before any return) with a proper index.
    # For DatetimeIndex: one step back. For integer index (tests): index - 1.
    if isinstance(wealth.index, pd.DatetimeIndex) and len(wealth) > 1:
        freq = wealth.index[1] - wealth.index[0]
        start_idx: pd.Index = pd.DatetimeIndex([wealth.index[0] - freq])
    else:
        start_idx = pd.Index([wealth.index[0] - 1])

    wealth_with_start = pd.concat(
        [
            pd.Series([1.0], index=start_idx),
            wealth,
        ]
    )

    running_max = wealth_with_start.cummax()
    drawdown = (wealth_with_start / running_max) - 1.0
    return float(drawdown.min())


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute the annualised Sortino ratio.

    Like Sharpe but penalises only downside volatility (negative returns),
    which makes it more aligned with tail-risk thinking.

    Parameters
    ----------
    returns:
        Daily return series.
    risk_free_rate:
        Annualised risk-free rate (default 0.0).

    Returns
    -------
    float
        Sortino ratio. Returns 0.0 if downside deviation is zero.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns.dropna() - daily_rf
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside_std < EPS or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))
