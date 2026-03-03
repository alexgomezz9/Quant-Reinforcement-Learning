"""
Build the aligned price matrix from individual raw parquet files.

Reads data/raw/<TICKER>.parquet for each ticker, extracts the Adj Close column,
joins them on dates that are common to all tickers (inner join), and saves the
result to data/processed/prices.parquet.

Why inner join?
    In a real portfolio you cannot rebalance on a day where any asset lacks a
    price. Inner join ensures every row is fully populated with no imputation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Column in the raw parquet that holds the adjusted closing price.
ADJ_CLOSE_COL = "Adj Close"


def build_prices(
    tickers: list[str],
    raw_dir: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Align Adj Close prices for all tickers and save to parquet.

    Parameters
    ----------
    tickers:
        List of ticker symbols to include, e.g. ["SPY", "GLD", "SHV"].
    raw_dir:
        Directory containing one <TICKER>.parquet file per ticker.
    output_path:
        Destination path for the processed prices.parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_dates, n_tickers), DatetimeIndex, no NaNs.

    Raises
    ------
    FileNotFoundError
        If a raw parquet file is missing for any ticker.
    ValueError
        If the resulting aligned DataFrame is empty or contains NaNs.
    """
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.Series] = []

    for ticker in tickers:
        parquet_path = raw_dir / f"{ticker}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Raw data not found for '{ticker}' at {parquet_path}. "
                "Run scripts/download_data.py first."
            )

        df = pd.read_parquet(parquet_path, columns=[ADJ_CLOSE_COL])

        # Flatten MultiIndex columns if present (yfinance artifact)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        series = df[ADJ_CLOSE_COL].rename(ticker)
        series.index = pd.to_datetime(series.index).tz_localize(None)
        frames.append(series)

        logger.info(
            "Loaded %-4s  %d rows, from %s to %s",
            ticker,
            len(series),
            series.index[0].date(),
            series.index[-1].date(),
        )

    # Inner join: keep only dates present in ALL tickers.
    prices = pd.concat(frames, axis=1, join="inner")
    prices.index.name = "Date"
    prices.sort_index(inplace=True)

    # Validate
    if prices.empty:
        raise ValueError("Aligned price DataFrame is empty. Check date ranges.")

    n_nan = prices.isna().sum().sum()
    if n_nan > 0:
        raise ValueError(
            f"Aligned prices contain {n_nan} NaN values after inner join. "
            "This should not happen — investigate raw data."
        )

    prices.to_parquet(output_path)

    logger.info(
        "Saved prices.parquet → %s | shape: %s | %s → %s",
        output_path,
        prices.shape,
        prices.index[0].date(),
        prices.index[-1].date(),
    )
    return prices
