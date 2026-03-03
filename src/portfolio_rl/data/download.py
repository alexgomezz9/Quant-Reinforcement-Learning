"""
Download raw OHLCV data from Yahoo Finance and persist to parquet.

Each ticker is saved independently to data/raw/<TICKER>.parquet.
This keeps downloads atomic: if one ticker fails, others are unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Download OHLCV data for a single ticker and save it as parquet.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol, e.g. "SPY".
    start:
        Start date (inclusive) in "YYYY-MM-DD" format.
    end:
        End date (inclusive) in "YYYY-MM-DD" format.
    output_dir:
        Directory where the parquet file will be written.
    overwrite:
        If False (default), skip download if the file already exists.
        Set to True to force a fresh download.

    Returns
    -------
    Path
        Path to the saved parquet file.

    Raises
    ------
    ValueError
        If yfinance returns an empty DataFrame (ticker unavailable or bad dates).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{ticker}.parquet"

    if output_path.exists() and not overwrite:
        logger.info("Skipping %s — file already exists at %s", ticker, output_path)
        return output_path

    logger.info("Downloading %s from %s to %s …", ticker, start, end)
    raw: pd.DataFrame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,  # keep Adj Close as explicit column
        progress=False,  # suppress yfinance progress bar
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start} and {end}. "
            "Check the ticker symbol and date range."
        )

    # yfinance may return MultiIndex columns when auto_adjust=False.
    # Flatten them to simple strings: ("Adj Close", "SPY") -> "Adj Close"
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Ensure the index is a proper DatetimeIndex with no timezone info
    # (yfinance sometimes returns tz-aware index; we normalise to naive UTC dates)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "Date"

    raw.to_parquet(output_path)
    logger.info("Saved %s → %s (%d rows)", ticker, output_path, len(raw))
    return output_path
