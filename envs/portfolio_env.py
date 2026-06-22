"""Single-asset signal-trading environment, per the seminar report's MDP.

State:  20-day window of 9 features, flattened, plus current position -> dim 181
Action: {0, 1, 2} -> {-1 (short), 0 (flat), +1 (long)}
Reward: 0.5 * rolling_30d_Sharpe + 1.0 * min(drawdown, 0) + direction_bonus - transaction_cost

Tuning notes (see results/tuning_notes.md):
- SHARPE_COEF raised 0.1->0.5 and DD_COEF lowered 2.0->1.0 so the Sharpe term
  is a real competing signal instead of being dominated ~20x by the drawdown
  penalty (the report's own failure mode: a degenerate always-flat policy).
- TRANSACTION_COST penalizes position flips so the agent can't churn
  positions for free; the original report reward had no cost of trading.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.features import FEATURE_NAMES, compute_features

WINDOW = 20
N_FEATURES = len(FEATURE_NAMES)
SHARPE_WINDOW = 30
DD_COEF = 1.0
SHARPE_COEF = 0.5
LONG_BONUS = 0.8
SHORT_BONUS = 1.0
TRANSACTION_COST = 0.0005


class PortfolioEnv(gym.Env):
    """Gymnasium environment trading a single asset long/flat/short."""

    metadata = {"render_modes": []}

    def __init__(self, close: pd.Series, window: int = WINDOW):
        super().__init__()
        self.window = window
        features = compute_features(close)
        # log-return of the *next* day relative to each feature row is the
        # asset return realised when an action taken at that row plays out.
        log_returns = np.log(close / close.shift(1)).reindex(features.index)

        self.features = features.values.astype(np.float32)
        self.asset_log_returns = log_returns.values.astype(np.float32)
        self.dates = features.index
        self.n_steps = len(self.features) - 1  # last row has no "next return"

        if self.n_steps <= self.window:
            raise ValueError("Not enough data after feature warm-up to form an episode.")

        obs_dim = self.window * N_FEATURES + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=short, 1=flat, 2=long

        self._t = self.window
        self._position = 0.0
        self._equity_curve = []
        self._strategy_returns = []

    def _get_obs(self) -> np.ndarray:
        window_feats = self.features[self._t - self.window:self._t].flatten()
        return np.concatenate([window_feats, [self._position]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.window
        self._position = 0.0
        self._equity_curve = [1.0]
        self._strategy_returns = []
        return self._get_obs(), {}

    def step(self, action: int):
        new_position = {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]

        asset_return = float(self.asset_log_returns[self._t])
        strategy_return = new_position * asset_return
        self._strategy_returns.append(strategy_return)
        self._equity_curve.append(self._equity_curve[-1] * np.exp(strategy_return))

        recent = np.array(self._strategy_returns[-SHARPE_WINDOW:])
        if len(recent) >= 2 and recent.std() > 1e-8:
            rolling_sharpe = np.sqrt(252) * recent.mean() / recent.std()
        else:
            rolling_sharpe = 0.0

        peak = max(self._equity_curve)
        drawdown = (self._equity_curve[-1] - peak) / peak

        direction_bonus = 0.0
        if new_position < 0 and asset_return < 0:
            direction_bonus = SHORT_BONUS * abs(asset_return)
        elif new_position > 0 and asset_return > 0:
            direction_bonus = LONG_BONUS * abs(asset_return)

        transaction_cost = TRANSACTION_COST * abs(new_position - self._position)

        reward = (SHARPE_COEF * rolling_sharpe + DD_COEF * min(drawdown, 0.0)
                  + direction_bonus - transaction_cost)

        self._position = new_position
        self._t += 1
        terminated = self._t >= self.n_steps
        truncated = False

        info = {
            "strategy_return": strategy_return,
            "asset_return": asset_return,
            "position": new_position,
            "equity": self._equity_curve[-1],
            "date": self.dates[self._t - 1],
        }
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info
