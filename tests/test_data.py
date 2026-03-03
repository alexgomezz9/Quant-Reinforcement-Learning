"""
Sanity tests for the data pipeline.

These tests validate:
- download_ticker() stores the expected file with correct structure.
- build_prices() produces an aligned DataFrame with no NaNs.
- The processed prices.parquet passes data quality checks.

Tests that require network access are marked with @pytest.mark.network and
can be skipped in CI with: pytest -m "not network"
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ── Paths ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PRICES_PATH = PROCESSED_DIR / "prices.parquet"

EXPECTED_TICKERS = ["SPY", "EFA", "TLT", "IEF", "GLD", "VNQ", "DBC", "SHV"]
EXPECTED_START = pd.Timestamp("2007-01-01")  # SHV limits start; must be after this


class TestRawData:
    """Tests on individual raw parquet files."""

    @pytest.mark.parametrize("ticker", EXPECTED_TICKERS)
    def test_raw_file_exists(self, ticker: str) -> None:
        """Each ticker must have a parquet file in data/raw/."""
        path = RAW_DIR / f"{ticker}.parquet"
        assert (
            path.exists()
        ), f"Missing raw file for {ticker}. Run: python scripts/download_data.py"

    @pytest.mark.parametrize("ticker", EXPECTED_TICKERS)
    def test_raw_has_adj_close(self, ticker: str) -> None:
        """Raw parquet must contain an 'Adj Close' column."""
        df = pd.read_parquet(RAW_DIR / f"{ticker}.parquet")
        assert "Adj Close" in df.columns, f"{ticker}: 'Adj Close' column missing."

    @pytest.mark.parametrize("ticker", EXPECTED_TICKERS)
    def test_raw_no_nan_in_adj_close(self, ticker: str) -> None:
        """Adj Close must not have NaN values."""
        df = pd.read_parquet(RAW_DIR / f"{ticker}.parquet")
        n_nan = df["Adj Close"].isna().sum()
        assert n_nan == 0, f"{ticker}: {n_nan} NaN values in 'Adj Close'."

    @pytest.mark.parametrize("ticker", EXPECTED_TICKERS)
    def test_raw_prices_positive(self, ticker: str) -> None:
        """All adjusted close prices must be strictly positive."""
        df = pd.read_parquet(RAW_DIR / f"{ticker}.parquet")
        n_bad = (df["Adj Close"] <= 0).sum()
        assert n_bad == 0, f"{ticker}: {n_bad} non-positive prices in 'Adj Close'."

    @pytest.mark.parametrize("ticker", EXPECTED_TICKERS)
    def test_raw_index_is_datetime(self, ticker: str) -> None:
        """Index must be a DatetimeIndex (not object/string)."""
        df = pd.read_parquet(RAW_DIR / f"{ticker}.parquet")
        assert isinstance(
            df.index, pd.DatetimeIndex
        ), f"{ticker}: index is {type(df.index)}, expected DatetimeIndex."


class TestProcessedPrices:
    """Tests on the aligned prices.parquet."""

    @pytest.fixture(scope="class")
    def prices(self) -> pd.DataFrame:
        """Load prices.parquet once and share across tests in this class."""
        assert (
            PRICES_PATH.exists()
        ), "prices.parquet not found. Run: python scripts/build_processed_prices.py"
        return pd.read_parquet(PRICES_PATH)

    def test_all_tickers_present(self, prices: pd.DataFrame) -> None:
        """All 8 tickers must appear as columns."""
        missing = set(EXPECTED_TICKERS) - set(prices.columns)
        assert not missing, f"Missing tickers in prices.parquet: {missing}"

    def test_no_nan(self, prices: pd.DataFrame) -> None:
        """After inner join, the matrix must have zero NaN values."""
        n_nan = prices.isna().sum().sum()
        assert n_nan == 0, f"prices.parquet has {n_nan} NaN values."

    def test_all_prices_positive(self, prices: pd.DataFrame) -> None:
        """All prices must be strictly positive."""
        n_bad = (prices <= 0).sum().sum()
        assert n_bad == 0, f"{n_bad} non-positive values found in prices.parquet."

    def test_index_is_datetime(self, prices: pd.DataFrame) -> None:
        """Index must be a DatetimeIndex."""
        assert isinstance(prices.index, pd.DatetimeIndex)

    def test_index_is_sorted(self, prices: pd.DataFrame) -> None:
        """Dates must be in ascending order."""
        assert (
            prices.index.is_monotonic_increasing
        ), "prices.parquet index is not sorted."

    def test_start_date_after_shv_launch(self, prices: pd.DataFrame) -> None:
        """Start date must be after SHV's first available date (~2007-01)."""
        assert prices.index[0] >= EXPECTED_START, (
            f"Start date {prices.index[0].date()} is earlier than expected. "
            "SHV was not available before 2007."
        )

    def test_minimum_rows(self, prices: pd.DataFrame) -> None:
        """Must have at least 4000 rows (roughly 16 years of trading days)."""
        assert (
            len(prices) >= 4000
        ), f"Only {len(prices)} rows in prices.parquet — expected at least 4000."

    def test_no_duplicate_dates(self, prices: pd.DataFrame) -> None:
        """Each date must appear exactly once."""
        n_dupes = prices.index.duplicated().sum()
        assert n_dupes == 0, f"{n_dupes} duplicate dates found in prices.parquet."
