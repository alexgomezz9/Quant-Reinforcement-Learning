"""
CLI script: Build data/processed/prices.parquet from raw parquet files.

Usage
-----
    python scripts/build_processed_prices.py

Run from the repository root with the venv activated.
Requires data/raw/<TICKER>.parquet to exist for all tickers in default.yaml.
Run scripts/download_data.py first if needed.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio_rl.data.process import build_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def main() -> None:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    tickers: list[str] = cfg["universe"]["tickers"]
    raw_dir = REPO_ROOT / cfg["data"]["raw_dir"]
    output_path = REPO_ROOT / cfg["data"]["processed_dir"] / "prices.parquet"

    build_prices(tickers, raw_dir, output_path)


if __name__ == "__main__":
    main()
