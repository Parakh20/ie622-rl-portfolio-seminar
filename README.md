# RL Portfolio Management (IE 622 Seminar)

A reinforcement-learning approach to single-asset signal trading using a recurrent
PPO agent (LSTM policy), benchmarked against classical portfolio baselines
(equal-weight, Markowitz minimum-variance, risk parity). Per-asset RL signals
are combined into an equal-weighted portfolio and compared against the
baselines on the same asset universe and held-out test period. The agent is
trained independently per ticker on a 20-day rolling window of technical
features and a 3-action (short/flat/long) policy, with a reward that blends a
rolling Sharpe ratio, a drawdown penalty, a directional bonus, and a
transaction-cost penalty.

## Architecture

**Environment** (`envs/portfolio_env.py`): `PortfolioEnv` is a `gymnasium.Env`
wrapping a single asset's price series. The state is a 20-day window of 9
engineered features plus the current position (181-dim observation). The
action space is discrete: short (-1), flat (0), long (+1). At each step the
reward is:

```
reward = SHARPE_COEF * rolling_sharpe(30d)
       + DD_COEF * min(drawdown, 0)
       + direction_bonus
       - TRANSACTION_COST * |Δposition|
```

**Features** (`envs/features.py`): log-return, momentum (5/15/30-day, clipped),
RSI-14, rolling volatility (10/30-day), Bollinger %B, absolute 5-day momentum.

**Agent** (`agents/recurrent_ppo.py`): `RecurrentPPO` (`sb3_contrib`) with an
LSTM policy (`MlpLstmPolicy`, hidden size 128), trained for 300,000 timesteps
per ticker with `n_steps=252` (one trading year of truncated BPTT), entropy
coefficient linearly decayed 0.05 → 0.01 over training, and observation
normalization (`VecNormalize`, stats frozen at evaluation time). Full
hyperparameters and the rationale for each tuning change vs. the original
report are in [`results/tuning_notes.md`](results/tuning_notes.md).

**Baselines** (`baselines/baselines.py`): equal-weight, Markowitz
minimum-variance (SLSQP, 60-day lookback, rebalanced every 20 days), and
risk-parity (equal risk contribution, same lookback/rebalance schedule).

**Asset universe and backtest period** (`data/universe.py`): 5 equities —
AAPL, MSFT, GOOGL, AMD, INTC (Period A). Train: 2018-01-01 to 2021-12-31.
Test: 2022-01-01 to 2023-12-31. Two additional equity periods (Period B) and
a crypto universe/period are defined for extension but not run in the
results below.

## Install & Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train one ticker (saves model + VecNormalize stats to results/models/)
python scripts/train.py --ticker AAPL --period A

# Train all 5 tickers used in the results below
for t in AAPL MSFT GOOGL AMD INTC; do
    python scripts/train.py --ticker $t --period A
done

# Evaluate a single trained model on its test period
python scripts/evaluate.py --ticker AAPL --period A

# Full RL-vs-baselines comparison (table + plot)
python scripts/backtest.py --tickers AAPL MSFT GOOGL AMD INTC --period A
```

Data is downloaded automatically via `yfinance` on first use and cached in
`data/raw/`.

## Results

Test period: 2022-01-01 to 2023-12-31, equal-weighted across AAPL, MSFT,
GOOGL, AMD, INTC.

| Strategy | Ann. Return | Cumulative Return | Sharpe | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| **RL Agent (RecurrentPPO)** | **24.9%** | **48.7%** | **1.06** | **-16.1%** | **1.54** |
| Equal Weight | 4.1% | 8.3% | 0.22 | -41.0% | 0.10 |
| Markowitz (Min-Var) | -3.7% | -7.2% | -0.06 | -41.2% | -0.09 |
| Risk Parity | 2.4% | 4.9% | 0.16 | -40.3% | 0.06 |

The RL agent outperforms every baseline on every metric over this period —
notably with roughly a third of the drawdown of the buy-and-hold-style
baselines, driven by the policy's ability to flatten or short rather than
hold a static allocation through the 2022 drawdown.

![RL Agent vs Baselines](results/backtest_comparison.png)
