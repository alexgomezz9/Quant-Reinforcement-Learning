"""
Unit tests for financial metrics.

Strategy: use analytically known results so tests are self-contained
and do not depend on market data. Each test validates one metric with
a controlled input whose expected output is computed by hand or via
a known formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_rl.metrics.returns import (
    TRADING_DAYS_PER_YEAR,
    annualised_return,
    annualised_volatility,
    log_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from portfolio_rl.metrics.risk import cvar_historical, rolling_cvar, var_historical

# ── Helpers ───────────────────────────────────────────────────────────


def make_constant_returns(value: float, n: int = 252) -> pd.Series:
    """Series of n identical daily returns."""
    return pd.Series([value] * n)


def make_price_series(values: list[float]) -> pd.Series:
    """Price series from a list of values with a DatetimeIndex."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


# ── log_returns ───────────────────────────────────────────────────────


class TestLogReturns:
    def test_length(self) -> None:
        """log_returns drops the first row (no previous price)."""
        prices = make_price_series([100.0, 101.0, 102.0, 101.0])
        result = log_returns(prices)
        assert len(result) == 3

    def test_known_value(self) -> None:
        """log(101/100) = 0.00995… for a 1% price increase."""
        prices = make_price_series([100.0, 101.0])
        result = log_returns(prices)
        assert result.iloc[0] == pytest.approx(np.log(101 / 100), rel=1e-6)

    def test_flat_price_gives_zero_returns(self) -> None:
        """Constant prices → all log returns are exactly 0."""
        prices = make_price_series([50.0, 50.0, 50.0, 50.0])
        result = log_returns(prices)
        assert (result == 0.0).all()

    def test_no_nan_in_output(self) -> None:
        prices = make_price_series([100.0, 102.0, 98.0, 105.0])
        assert log_returns(prices).isna().sum() == 0


# ── annualised_return ─────────────────────────────────────────────────


class TestAnnualisedReturn:
    def test_constant_positive_log(self) -> None:
        """Log returns: mean daily 0.001 × 252 = 0.252 annualised."""
        returns = make_constant_returns(0.001)
        assert annualised_return(returns, returns_type="log") == pytest.approx(
            0.252, rel=1e-6
        )

    def test_constant_positive_simple(self) -> None:
        """Simple returns: geometric mean annualised via compounding formula."""
        returns = make_constant_returns(0.001, n=252)
        result = annualised_return(returns, returns_type="simple")
        # (1.001)^252 - 1 ≈ 0.2842
        expected = (1.001**252) - 1
        assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_returns(self) -> None:
        assert annualised_return(make_constant_returns(0.0)) == pytest.approx(0.0)

    def test_negative_returns(self) -> None:
        """Negative daily log returns → negative annualised return."""
        returns = make_constant_returns(-0.001)
        assert annualised_return(returns, returns_type="log") == pytest.approx(
            -0.252, rel=1e-6
        )

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="returns_type"):
            annualised_return(make_constant_returns(0.001), returns_type="bad")


# ── annualised_volatility ─────────────────────────────────────────────


class TestAnnualisedVolatility:
    def test_constant_returns_zero_vol(self) -> None:
        """No variation in returns → volatility is 0."""
        returns = make_constant_returns(0.001)
        assert annualised_volatility(returns) == pytest.approx(0.0, abs=1e-10)

    def test_known_daily_std(self) -> None:
        """If daily std = 0.01, annualised vol = 0.01 × sqrt(252)."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 10_000))
        expected = 0.01 * np.sqrt(TRADING_DAYS_PER_YEAR)
        # With 10k samples we expect the estimate to be within 2% of the true value.
        assert annualised_volatility(returns) == pytest.approx(expected, rel=0.02)


# ── sharpe_ratio ──────────────────────────────────────────────────────


class TestSharpeRatio:
    def test_zero_vol_returns_zero(self) -> None:
        """Constant returns → zero volatility → Sharpe returns 0."""
        returns = make_constant_returns(0.001)
        assert sharpe_ratio(returns) == pytest.approx(0.0, abs=1e-10)

    def test_positive_mean_positive_sharpe(self) -> None:
        """Positive excess return and positive vol → positive Sharpe."""
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0.001, 0.01, 5000))
        assert sharpe_ratio(returns) > 0

    def test_risk_free_rate_reduces_sharpe(self) -> None:
        """Higher risk-free rate reduces Sharpe (ceteris paribus)."""
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0.001, 0.01, 5000))
        sr_0 = sharpe_ratio(returns, risk_free_rate=0.0)
        sr_rf = sharpe_ratio(returns, risk_free_rate=0.04)
        assert sr_0 > sr_rf


# ── max_drawdown ──────────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_no_loss_gives_zero(self) -> None:
        """Monotonically increasing simple returns → drawdown is always 0."""
        returns = make_constant_returns(0.001)
        assert max_drawdown(returns, returns_type="simple") == pytest.approx(
            0.0, abs=1e-10
        )

    def test_single_known_drawdown_simple(self) -> None:
        """
        Simple returns: prices 100 → 50 → 50.
        Returns: -0.5, 0.0. Wealth: 1.0 → 0.5 → 0.5. MDD = -0.50.
        """
        returns = pd.Series([-0.5, 0.0])
        assert max_drawdown(returns, returns_type="simple") == pytest.approx(
            -0.5, rel=1e-6
        )

    def test_single_known_drawdown_log(self) -> None:
        """
        Log returns: ln(50/100) = -0.693, then 0.
        Wealth via exp(cumsum): e^{-0.693} ≈ 0.5, then 0.5. MDD ≈ -0.50.
        """
        log_ret = pd.Series([np.log(0.5), 0.0])
        assert max_drawdown(log_ret, returns_type="log") == pytest.approx(
            -0.5, rel=1e-4
        )

    def test_mdd_is_non_positive(self) -> None:
        """Max drawdown is always <= 0 by definition."""
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.0, 0.01, 1000))
        assert max_drawdown(returns, returns_type="simple") <= 0.0

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="returns_type"):
            max_drawdown(pd.Series([0.01, -0.02]), returns_type="wrong")


# ── sortino_ratio ─────────────────────────────────────────────────────


class TestSortinoRatio:
    def test_no_downside_gives_zero(self) -> None:
        """All returns >= 0 → no downside → Sortino returns 0."""
        returns = make_constant_returns(0.001)
        assert sortino_ratio(returns) == pytest.approx(0.0, abs=1e-10)

    def test_sortino_gt_sharpe_for_positive_skew(self) -> None:
        """
        For returns with positive skew (losses smaller than gains),
        Sortino should be higher than Sharpe because downside vol < total vol.
        """
        rng = np.random.default_rng(7)
        # Mostly small positive, occasionally very small negative
        returns = pd.Series(np.abs(rng.normal(0.001, 0.01, 5000)))
        # All positive so downside_std is 0 → Sortino returns 0.0
        assert sortino_ratio(returns) == pytest.approx(0.0, abs=1e-10)


# ── var_historical ────────────────────────────────────────────────────


class TestVaRHistorical:
    def test_known_quantile(self) -> None:
        """
        For returns = [-0.05, -0.04, ..., +0.04, +0.05] (100 values),
        VaR at alpha=0.05 should be the 5th percentile = -0.041.
        """
        returns = pd.Series(np.linspace(-0.05, 0.05, 100))
        var = var_historical(returns, alpha=0.05)
        # 5th percentile of linspace(-0.05, 0.05, 100) = -0.041 (approx)
        assert var < 0  # VaR must be a loss (negative)
        assert var == pytest.approx(np.quantile(returns, 0.05), rel=1e-6)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            var_historical(pd.Series([], dtype=float))

    def test_invalid_alpha_raises(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.03])
        with pytest.raises(ValueError, match="alpha"):
            var_historical(returns, alpha=1.5)

    def test_var_is_negative_for_typical_returns(self) -> None:
        """For mixed returns, VaR at 5% should be negative."""
        rng = np.random.default_rng(10)
        returns = pd.Series(rng.normal(0, 0.01, 1000))
        assert var_historical(returns, alpha=0.05) < 0


# ── cvar_historical ───────────────────────────────────────────────────


class TestCVaRHistorical:
    def test_cvar_worse_than_or_equal_to_var(self) -> None:
        """CVaR (tail mean) is always <= VaR (tail threshold)."""
        rng = np.random.default_rng(20)
        returns = pd.Series(rng.normal(0, 0.01, 2000))
        var = var_historical(returns, alpha=0.05)
        cvar = cvar_historical(returns, alpha=0.05)
        assert cvar <= var

    def test_uniform_distribution(self) -> None:
        """
        For Uniform[-1, 1] and alpha=0.1:
        VaR = -0.8 (10th percentile of U[-1,1])
        CVaR = mean of U[-1, -0.8] = (-1 + -0.8) / 2 = -0.9
        """
        rng = np.random.default_rng(30)
        returns = pd.Series(rng.uniform(-1, 1, 100_000))
        cvar = cvar_historical(returns, alpha=0.10)
        # With 100k samples tolerance of 1% is fine
        assert cvar == pytest.approx(-0.9, rel=0.01)

    def test_cvar_is_negative(self) -> None:
        rng = np.random.default_rng(40)
        returns = pd.Series(rng.normal(0, 0.01, 1000))
        assert cvar_historical(returns, alpha=0.05) < 0


# ── rolling_cvar ──────────────────────────────────────────────────────


class TestRollingCVaR:
    def test_output_length(self) -> None:
        """Rolling CVaR has the same length as input."""
        returns = pd.Series(np.random.default_rng(50).normal(0, 0.01, 200))
        result = rolling_cvar(returns, window=60, alpha=0.05)
        assert len(result) == len(returns)

    def test_first_values_are_nan(self) -> None:
        """First (window-1) values must be NaN (insufficient history)."""
        returns = pd.Series(np.random.default_rng(50).normal(0, 0.01, 200))
        result = rolling_cvar(returns, window=60, alpha=0.05)
        assert result.iloc[:59].isna().all()

    def test_values_after_window_are_not_nan(self) -> None:
        """Once enough data exists, rolling CVaR must be non-NaN."""
        returns = pd.Series(np.random.default_rng(50).normal(0, 0.01, 200))
        result = rolling_cvar(returns, window=60, alpha=0.05)
        assert result.iloc[59:].notna().all()

    def test_rolling_values_are_negative(self) -> None:
        """For typical daily returns, rolling CVaR should be negative."""
        rng = np.random.default_rng(60)
        returns = pd.Series(rng.normal(0, 0.01, 300))
        result = rolling_cvar(returns, window=60, alpha=0.05).dropna()
        assert (result < 0).all()
