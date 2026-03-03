"""
CLI script: Download raw OHLCV data for all tickers in configs/default.yaml.

Usage
-----
    python scripts/download_data.py
    python scripts/download_data.py --overwrite       # force fresh download
    python scripts/download_data.py --ticker SPY GLD  # single tickers only

Run from the repository root with the venv activated.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Make sure the src/ package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio_rl.data.download import download_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw market data.")
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="TICKER",
        help="Download only these tickers (default: all tickers in config).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if the parquet file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(CONFIG_PATH)

    tickers: list[str] = args.ticker or cfg["universe"]["tickers"]
    start: str = cfg["data"]["start_date"]
    end: str = cfg["data"]["end_date"]
    output_dir = REPO_ROOT / cfg["data"]["raw_dir"]

    logger.info("Tickers : %s", tickers)
    logger.info("Period  : %s → %s", start, end)
    logger.info("Output  : %s", output_dir)

    failed: list[str] = []
    for ticker in tickers:
        try:
            download_ticker(ticker, start, end, output_dir, overwrite=args.overwrite)
        except Exception as exc:
            logger.error("FAILED %s: %s", ticker, exc)
            failed.append(ticker)

    if failed:
        logger.error("The following tickers failed: %s", failed)
        sys.exit(1)

    logger.info(
        "Done. %d/%d tickers downloaded successfully.",
        len(tickers) - len(failed),
        len(tickers),
    )


if __name__ == "__main__":
    main()
