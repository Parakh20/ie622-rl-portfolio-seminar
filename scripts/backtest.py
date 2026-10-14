"""Backtest the RL agent portfolio against all baselines on a shared test period.

The "RL portfolio" equally weights the per-asset RecurrentPPO signals
(trained independently per ticker) so it is directly comparable to the
equal-weight / Markowitz / risk-parity baselines, which allocate the same
asset universe.

Usage:
    python scripts/backtest.py --tickers AAPL MSFT GOOGL AMD INTC --period A
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import RecurrentPPO

from agents.metrics import summarize
from baselines.baselines import (
    equal_weight_returns,
    markowitz_returns,
    risk_parity_returns,
)
from data.download import load_close
from data.universe import PERIODS
from scripts.evaluate import evaluate_agent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def build_rl_portfolio(tickers, period_key: str) -> pd.Series:
    period = PERIODS[period_key]
    model_dir = os.path.join(RESULTS_DIR, "models")

    per_asset_returns = {}
    for ticker in tickers:
        model_path = os.path.join(model_dir, f"{ticker}_period{period_key}.zip")
        vecnorm_path = os.path.join(model_dir, f"{ticker}_period{period_key}_vecnormalize.pkl")
        if not os.path.exists(model_path):
            print(f"Skipping {ticker}: no trained model at {model_path}")
            continue
        model = RecurrentPPO.load(model_path)
        test_close = load_close(ticker, period["test_start"], period["test_end"])
        per_asset_returns[ticker] = evaluate_agent(model, test_close, vecnorm_path)

    if not per_asset_returns:
        raise RuntimeError("No trained models found; run scripts/train.py first.")

    aligned = pd.DataFrame(per_asset_returns).dropna(how="all").fillna(0.0)
    return aligned.mean(axis=1)


def build_baseline_returns(tickers, period_key: str) -> pd.DataFrame:
    period = PERIODS[period_key]
    closes = {t: load_close(t, period["test_start"], period["test_end"]) for t in tickers}
    prices = pd.DataFrame(closes).dropna()
    returns = prices.pct_change().dropna()
    return returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "AMD", "INTC"])
    parser.add_argument("--period", default="A", choices=list(PERIODS))
    args = parser.parse_args()

    rl_returns = build_rl_portfolio(args.tickers, args.period)
    asset_returns = build_baseline_returns(args.tickers, args.period)

    strategies = {
        "RL Agent (RecurrentPPO)": rl_returns,
        "Equal Weight": equal_weight_returns(asset_returns),
        "Markowitz (Min-Var)": markowitz_returns(asset_returns),
        "Risk Parity": risk_parity_returns(asset_returns),
    }

    rows = []
    for name, rets in strategies.items():
        m = summarize(rets)
        rows.append({"strategy": name, **m})
    summary = pd.DataFrame(rows).set_index("strategy")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "backtest_summary.csv")
    summary.to_csv(csv_path)
    print(summary.to_string())
    print(f"\nSaved table -> {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, rets in strategies.items():
        (1 + rets).cumprod().plot(ax=ax, label=name, lw=1.8)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"RL Agent vs Baselines — Cumulative Returns (Period {args.period})")
    ax.set_ylabel("Portfolio Value (start = 1)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "backtest_comparison.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
