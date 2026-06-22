"""Evaluate a trained RecurrentPPO agent on a held-out test period.

Usage:
    python scripts/evaluate.py --ticker AAPL --period A
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.metrics import summarize
from data.download import download_ticker
from data.universe import CRYPTO_PERIOD, PERIOD_A, PERIOD_B
from envs.portfolio_env import PortfolioEnv

PERIODS = {"A": PERIOD_A, "B": PERIOD_B, "crypto": CRYPTO_PERIOD}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_vecnormalize(test_close: pd.Series, vecnorm_path: str | None):
    """Wrap the test env in the training-time VecNormalize stats (frozen, no updates)."""
    venv = DummyVecEnv([lambda: PortfolioEnv(test_close)])
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=False)
    return venv


def evaluate_agent(model: RecurrentPPO, test_close: pd.Series, vecnorm_path: str | None = None) -> pd.Series:
    """Roll the trained agent deterministically over the test period.

    Returns a pd.Series of daily strategy returns indexed by date.
    """
    venv = load_vecnormalize(test_close, vecnorm_path)
    raw_env = venv.venv.envs[0]

    obs = venv.reset()
    lstm_states = None
    episode_start = np.array([True])

    returns, dates = [], []
    done = False
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_start, deterministic=True
        )
        obs, reward, done_arr, infos = venv.step(action)
        info = infos[0]
        returns.append(info["strategy_return"])
        dates.append(info["date"])
        episode_start = np.array([False])
        done = bool(done_arr[0])

    return pd.Series(returns, index=pd.to_datetime(dates), name="strategy_return")


def load_close(ticker: str, start: str, end: str) -> pd.Series:
    df = download_ticker(ticker, start=start)
    close = df["Close"].astype(float)
    close.index = pd.to_datetime(close.index)
    return close.loc[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="A", choices=list(PERIODS))
    args = parser.parse_args()

    period = PERIODS[args.period]
    model_path = os.path.join(RESULTS_DIR, "models", f"{args.ticker}_period{args.period}.zip")
    vecnorm_path = os.path.join(RESULTS_DIR, "models", f"{args.ticker}_period{args.period}_vecnormalize.pkl")
    model = RecurrentPPO.load(model_path)

    test_close = load_close(args.ticker, period["test_start"], period["test_end"])
    strategy_returns = evaluate_agent(model, test_close, vecnorm_path)

    metrics = summarize(strategy_returns)
    print(f"{args.ticker} (period {args.period}) test metrics: "
          f"Sharpe={metrics['sharpe']:.3f}  CumRet={100*metrics['cumulative_return']:.1f}%  "
          f"MaxDD={100*metrics['max_drawdown']:.1f}%")


if __name__ == "__main__":
    main()
