# TFM Project Specification — Risk-Aware RL Portfolio Optimization (CVaR)

Author: Alejandro Gomez Ruiz
Master: MSc in Artificial Intelligence
Topic: Dynamic portfolio allocation under non-stationarity with explicit tail-risk control (CVaR) using Reinforcement Learning

---

## 1) Problem Statement

Real markets are non-stationary and expose portfolios to rare but severe downside events. Classical portfolio optimization approaches often rely on stationarity assumptions and risk proxies such as volatility, which may underestimate tail risk.

This TFM studies whether Reinforcement Learning (RL) can learn dynamic portfolio allocation policies that improve risk-adjusted performance while explicitly controlling tail risk using Conditional Value-at-Risk (CVaR).

**Scope note:** this is a research prototype and evaluation framework. It is not a production trading system.

---

## 2) Core Objectives

1. Implement a reproducible pipeline for multi-asset daily portfolio allocation.
2. Evaluate RL policies under strict out-of-sample testing with walk-forward validation.
3. Incorporate explicit tail-risk control using CVaR at alpha = 0.05 and 0.01.
4. Compare RL agents (PPO and SAC) to classical benchmarks (Equal-Weight, Static, Markowitz/Risk-Parity if time).
5. Analyze robustness under regime shifts(2008 crisis. covid etc) and sensitivity to transaction costs.

---

## 3) Key Research Questions

- Does RL improve downside protection (CVaR / max drawdown) without sacrificing too much return?
- How stable are learned policies under regime changes?
- How does explicit CVaR control shift allocations and turnover?
- How do results change when transaction costs increase (5 bps vs 10 bps)?

---

## 4) Asset Universe & Data

**Assets (8 ETFs, final list):**

| Ticker | Description                     | Role in portfolio          |
|--------|---------------------------------|----------------------------|
| SPY    | S&P 500 (US large-cap equity)   | Core equity exposure       |
| EFA    | International developed equity  | Geographic diversification |
| TLT    | US long-term bonds (20+ yr)     | Duration / flight-to-safety|
| IEF    | US intermediate bonds (7-10 yr) | Lower-vol bond exposure    |
| GLD    | Gold                            | Inflation / crisis hedge   |
| VNQ    | US REITs (real estate)          | Real-asset diversification |
| DBC    | Commodities (diversified)       | Commodity cycle exposure   |
| SHV    | Short-term Treasury (cash proxy)| Risk-off / cash allocation |

**SHV rationale:** Including a cash-like asset allows the agent to learn to "take shelter" during stress periods, which directly supports the CVaR-control objective.

**Data source:** Yahoo Finance via `yfinance`.
**Frequency:** Daily.
**Period:** 2005–2025 (subject to ticker availability).
**Price field:** adjusted close (corporate actions handled).

**Data storage plan**
- `data/raw/`: one parquet per ticker (OHLCV + adjusted close)
- `data/processed/`: aligned prices/features shared across all tickers

**Configuration:** All parameters (tickers, dates, costs, etc.) are centralized in `configs/default.yaml`. Scripts read from this file — no hardcoded values in code.

---

## 5) Non-Negotiable Constraints (Reproducibility & Validity)

- **No leakage:** features at time t can only use information ≤ t.
- **Strict OOS evaluation:** walk-forward / rolling evaluation only.
- **Transaction costs included:** fixed bps per rebalance proportional to turnover.
- **Long-only weights:** initial phase without leverage/shorting.
- **Seeds fixed:** reproducible runs where possible.
- **Benchmarks first:** RL is evaluated only after benchmarks are validated.

---

## 6) Benchmarks (must exist before RL)

Minimum set:
- Buy & Hold (per-asset)
- Equal-Weight (periodic rebalance)
- Static allocation (manual fixed weights)

Optional (if time):
- Risk parity
- Mean-variance (Markowitz) with lookback + constraints

All benchmarks must be evaluated under the same walk-forward protocol and costs.

---

## 7) MDP Formulation (Environment)

### State (Observation)
A flattened vector built from:
- Window of features over W days (e.g., returns, vol, momentum)
- Current portfolio weights
- Portfolio value normalization (optional)

Defaults:
- Lookback window `W = 60` (tunable)
- Features baseline: log-returns, rolling volatility (20d), rolling Sharpe proxy (20d)
- Optional: regime indicator / correlation summary

### Action
Continuous vector in R^N that is transformed to valid weights:
- long-only, weights sum to 1 (simplex)
- implemented via softmax or clipping + renormalization

### Transition
At each step:
1. action -> target weights
2. apply transaction cost proportional to turnover
3. realize next-day asset returns
4. compute portfolio return net of costs
5. update portfolio value + weights after market drift

### Reward
Base reward starts as **net daily portfolio return**.

CVaR control options:
1) **Soft penalty (baseline)**
   `reward_t = net_return_t - λ * CVaR_hat(alpha, K)`
2) **Lagrangian / constrained (advanced)**
   dual update to enforce `CVaR_hat(alpha, K) <= δ`

We will implement (1) first, then (2) if time permits.

**Normalization note:** `net_return_t` is O(1e-3) daily, while `CVaR_hat` depends on the window and may have a different scale. Care must be taken to normalize or scale λ so that neither term dominates training. This will be addressed during reward tuning in the RL phase.

---

## 8) Tail-Risk Estimation (CVaR)

We use historical empirical CVaR on a rolling window of portfolio returns.

- CVaR levels: alpha = 0.05 (primary) and alpha = 0.01 (sensitivity only)
- Rolling window length: `K = 60` (baseline), sensitivity K ∈ {20, 120} if time

**Statistical note:** With K=60 and alpha=0.01, only ~0.6 observations fall in the tail, making the estimate unstable. Therefore:
- alpha=0.05 with K=60 is the **primary** configuration (≈3 tail obs — still noisy but usable).
- alpha=0.01 is reported only as **sensitivity analysis** and should use K≥120.

---

## 9) Validation Protocol (Walk-Forward)

We evaluate out-of-sample performance using walk-forward splits.

Baseline plan:
- Train window: 5 years
- Test window: 1 year
- Slide: 1 year
- Aggregate OOS returns across windows

**Note on early folds:** With data starting in 2005, the first training window (2005-2009) happens to include the 2008 crisis. Later folds trained on calmer periods may behave differently. We will document per-fold regime context in the analysis.

We report:
- per-window metrics
- aggregated OOS metrics
- stability and regime discussion

---

## 10) Metrics (Evaluation)

Core metrics:
- Annualized return
- Annualized volatility
- Sharpe ratio
- Sortino ratio (optional)
- Max drawdown
- VaR and CVaR (95% and 99%)
- Turnover (proxy for trading intensity)
- Cost impact: gross vs net (optional)

---

## 11) Tracking & Experiment Management

- Use MLflow for:
  - params (W, K, alpha, cost_bps, algo)
  - metrics (per-episode, per-window, aggregated OOS)
  - artifacts (equity curves, tables, configs, model checkpoints)

---

## 12) Implementation Principles (Coding Rules)

- Modular design: data / env / agents / evaluation / reporting
- Small increments: implement the smallest runnable slice first (PR-based)
- Tests: minimal unit tests for core math and sanity-checks
- No over-engineering early: only add complexity when baseline is stable
- Configuration via YAML: all tunable parameters in `configs/default.yaml`
- Package structure: installable via `pip install -e .` from `src/portfolio_rl/`
- Code quality: pre-commit hooks (black + ruff) enforce consistent style
- Reproducibility: fixed seeds, deterministic data pipeline, all outputs regenerable from scripts

---
