# Risk-Aware RL Portfolio Optimization (CVaR)

> **Master Thesis (TFM)** — MSc in Artificial Intelligence
> Author: Alejandro Gomez Ruiz

Dynamic multi-asset portfolio allocation using Reinforcement Learning with explicit tail-risk control via Conditional Value-at-Risk (CVaR).

## Objective

Study whether RL agents (PPO, SAC) can learn allocation policies that improve downside protection — measured by CVaR and maximum drawdown — compared to classical benchmarks, under strict out-of-sample evaluation with transaction costs.

## Key Features

- **8-asset universe**: SPY, EFA, TLT, IEF, GLD, VNQ, DBC, SHV (cash proxy)
- **CVaR-penalized reward**: explicit tail-risk control in the RL objective
- **Walk-forward validation**: no lookahead bias, strict OOS evaluation
- **Transaction costs**: included in all strategies (5–10 bps)
- **Classical benchmarks**: Equal-Weight, Buy & Hold, Static allocation
- **MLflow tracking**: full experiment reproducibility

## Project Structure

```
├── configs/              # YAML configuration (parameters, tickers, etc.)
├── data/
│   ├── raw/              # Per-ticker OHLCV parquet files
│   └── processed/        # Aligned price matrix
├── docs/                 # Project specification
├── outputs/
│   ├── figures/          # Equity curves, plots
│   └── reports/          # Metrics tables (CSV)
├── scripts/              # Runnable CLI scripts
├── src/portfolio_rl/     # Installable Python package
│   ├── data/             # Download & processing
│   ├── metrics/          # Returns, Sharpe, CVaR, drawdown
│   ├── benchmarks/       # Classical strategies
│   ├── env/              # Gymnasium RL environment
│   ├── agents/           # PPO, SAC wrappers
│   └── evaluation/       # Walk-forward, reporting
└── tests/                # Unit & integration tests
```

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-user>/tfm.git
cd tfm
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Install pre-commit hooks
pre-commit install

# 3. Download market data
python scripts/download_data.py

# 4. Build processed prices
python scripts/build_processed_prices.py

# 5. Run benchmarks
python scripts/run_benchmarks.py
```

## Configuration

All parameters are centralized in [`configs/default.yaml`](configs/default.yaml):
tickers, date range, CVaR levels, transaction costs, walk-forward splits, etc.

## Development

```bash
# Run tests
pytest

# Format code
black src/ scripts/ tests/

# Lint
ruff check src/ scripts/ tests/
```

## Roadmap

- [x] PR0: Project scaffolding & configuration
- [ ] PR1: Data pipeline (download + processing)
- [ ] PR2: Financial metrics (Sharpe, CVaR, drawdown)
- [ ] PR3: Benchmark strategies
- [ ] PR4: Walk-forward evaluation framework
- [ ] PR5: RL environment (Gymnasium MDP)
- [ ] PR6: PPO agent
- [ ] PR7: SAC agent + comparison
- [ ] PR8: Sensitivity & robustness analysis
- [ ] PR9: Final polish & documentation

## License

MIT

---

*This is a research prototype for academic purposes. Not financial advice.*
