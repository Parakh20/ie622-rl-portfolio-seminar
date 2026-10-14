"""Train a RecurrentPPO agent on one asset's signal-trading task.

Usage:
    python scripts/train.py --ticker AAPL --period A
    python scripts/train.py --ticker AAPL --period A --timesteps 20000   # quick smoke test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.recurrent_ppo import DEFAULT_TOTAL_TIMESTEPS, EntCoefDecayCallback, build_agent
from data.download import load_close
from data.universe import PERIODS
from envs.portfolio_env import PortfolioEnv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="A", choices=list(PERIODS))
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    period = PERIODS[args.period]
    train_close = load_close(args.ticker, period["train_start"], period["train_end"])
    print(f"Training {args.ticker} on {len(train_close)} days "
          f"({period['train_start']} to {period['train_end']})")

    env = DummyVecEnv([lambda: PortfolioEnv(train_close)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = build_agent(env, seed=args.seed)
    callback = EntCoefDecayCallback(start=model.ent_coef, end=0.01)
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)

    model_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.ticker}_period{args.period}.zip")
    vecnorm_path = os.path.join(model_dir, f"{args.ticker}_period{args.period}_vecnormalize.pkl")
    model.save(model_path)
    env.save(vecnorm_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved normalization stats -> {vecnorm_path}")


if __name__ == "__main__":
    main()
