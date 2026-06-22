"""Shared performance metrics for the RL agent and all baselines."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_DAILY = 0.02 / TRADING_DAYS


def sharpe(rets: pd.Series, rf: float = RISK_FREE_DAILY) -> float:
    """Annualised Sharpe ratio from daily simple/log returns."""
    rets = pd.Series(rets) if not isinstance(rets, pd.Series) else rets
    excess = rets - rf
    std = excess.std()
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / std)


def max_drawdown(rets: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    rets = pd.Series(rets) if not isinstance(rets, pd.Series) else rets
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min())


def ann_return(rets: pd.Series) -> float:
    """Annualised geometric return."""
    rets = pd.Series(rets) if not isinstance(rets, pd.Series) else rets
    n = len(rets)
    if n == 0:
        return 0.0
    return float((1 + rets).prod() ** (TRADING_DAYS / n) - 1)


def cumulative_return(rets: pd.Series) -> float:
    """Total compounded return over the period."""
    rets = pd.Series(rets) if not isinstance(rets, pd.Series) else rets
    return float((1 + rets).prod() - 1)


def calmar(rets: pd.Series) -> float:
    """Calmar ratio = Annualised Return / |Max Drawdown|."""
    mdd = abs(max_drawdown(rets))
    return ann_return(rets) / mdd if mdd > 1e-8 else float("nan")


def summarize(rets: pd.Series) -> dict:
    return {
        "ann_return": ann_return(rets),
        "cumulative_return": cumulative_return(rets),
        "sharpe": sharpe(rets),
        "max_drawdown": max_drawdown(rets),
        "calmar": calmar(rets),
    }
