"""Portfolio-construction baselines compared against the RL agent.

All baselines take a DataFrame of daily simple returns (columns = tickers)
and return a single portfolio return series. Weights are estimated on a
rolling lookback window and rebalanced periodically, so the test period is
never used to choose weights in hindsight.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    """1/n allocation, no rebalancing needed since weights never drift the metric here."""
    return returns.mean(axis=1)


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    # Daily covariance entries are ~1e-4; SLSQP's default ftol (1e-6) sees
    # near-zero objective deltas at that scale and reports false convergence
    # at x0. Scaling the objective up fixes this without changing the optimum.
    scale = 1.0 / max(np.abs(cov).mean(), 1e-12)
    result = minimize(lambda w: scale * (w @ cov @ w), x0, bounds=bounds,
                       constraints=constraints, method="SLSQP")
    return result.x if result.success else x0


def markowitz_returns(returns: pd.DataFrame, lookback: int = 60, rebalance: int = 20) -> pd.Series:
    """Rolling minimum-variance (mean-variance) portfolio, rebalanced periodically."""
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    out = []
    for i, (_, row) in enumerate(returns.iterrows()):
        if i >= lookback and i % rebalance == 0:
            window = returns.iloc[i - lookback:i]
            cov = window.cov().values
            weights = _min_variance_weights(cov)
        out.append(float(row.values @ weights))
    return pd.Series(out, index=returns.index)


def _risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]

    scale = 1.0 / max(np.abs(cov).mean(), 1e-12)

    def objective(w):
        portfolio_var = w @ cov @ w
        marginal_contrib = cov @ w
        risk_contrib = w * marginal_contrib
        target = portfolio_var / n
        return scale ** 2 * np.sum((risk_contrib - target) ** 2)

    x0 = np.ones(n) / n
    bounds = [(1e-6, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = result.x if result.success else x0
    return weights / weights.sum()


def risk_parity_returns(returns: pd.DataFrame, lookback: int = 60, rebalance: int = 20) -> pd.Series:
    """Rolling equal-risk-contribution portfolio, rebalanced periodically."""
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets
    out = []
    for i, (_, row) in enumerate(returns.iterrows()):
        if i >= lookback and i % rebalance == 0:
            window = returns.iloc[i - lookback:i]
            cov = window.cov().values
            weights = _risk_parity_weights(cov)
        out.append(float(row.values @ weights))
    return pd.Series(out, index=returns.index)


def buy_and_hold_returns(close: pd.Series) -> pd.Series:
    """Daily simple returns of holding a single asset throughout the period."""
    return close.pct_change().dropna()
